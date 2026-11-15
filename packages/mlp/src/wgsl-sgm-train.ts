/**
 * Subgroup-matrix forward-backward training arm (`forward: 'sgm'`, research track N step N3b).
 *
 * Same entry point (`nnForwardBackward`), group-0 bindings, `activations` / `deltas` layout and delta scaling (not
 * divided by N) as the tiled arm (`wgsl-forward.ts`), so the tiled weight-gradient pass and Adam run unchanged after
 * it. Differences: binding 0 is the PADDED sgm weight buffer (`packSgmWgsl` over the train weights, dispatched by the
 * engine at the start of every step), and one workgroup is `subgroups` subgroups of 32 lanes handling R records.
 *
 *  forward:  nnA (R × S, operand type T) holds a_l; Z = a_l · W_lᵀ per layer (right operand = column-major view of the
 *            row-major [out][in] weights, `loadRight`) → nnS (result type); barrier; a ReLU/convert pass through
 *            workgroup memory writes a_{l+1} to nnA and to `activations` (gpuweb #11077: no element-wise ops).
 *  loss:     per record, y = nnS[0..2]: relative-L2 gradient exactly as the other arms, δ_hidden written to nnA
 *            (columns 3..padded-out zero) and `deltas`; targets kept in workgroup memory.
 *  backward: δ_{l-1} = (δ_l · W_l) ⊙ [a_l > 0]: left operand nnA (δ_l), right operand the weights as a ROW-major view
 *            (K = layer outputs, N = layer inputs; `loadRightN`, i.e. Wᵀ needs no copy because the forward load already
 *            used the transposed flag the other way round); the mask pass reads a_l back from `activations` (the same
 *            lane wrote it in the forward pass, so no storage barrier is needed) and writes nnA and `deltas`.
 *
 * Records at or past the GPU count load nothing, contribute zero deltas and write nothing. The `emulate` form uses the
 * plain-WGSL stand-ins of the inference arm (32 lanes share each tile, f32) and runs on SwiftShader.
 */

import { layerCount, layerIn, layerOut } from './config.ts';
import { encodingWgsl } from './encoding.ts';
import { header, rowStride } from './wgsl.ts';
import {
  directives,
  emulatedOps,
  mlpOptions,
  nativeOps,
  sgmLanes,
  sgmLayerOffset,
  sgmLayerOut,
  sgmWorkgroupSize,
  validateSgm,
  type SgmOptions,
} from './wgsl-sgm.ts';

/** Workgroup memory the kernel needs (bytes): nnA (T) + nnS (result type) + targets (vec4f per record). */
export function sgmTrainWorkgroupBytes(o: SgmOptions): number {
  const s = rowStride(o.shape);
  const t = o.emulate ? 4 : o.config.componentType === 'f16' ? 2 : 4;
  const tr = o.emulate ? 4 : o.config.resultComponentType === 'f16' ? 2 : 4;
  return o.records * s * (t + tr) + o.records * 16;
}

export function validateSgmTrain(o: SgmOptions): void {
  validateSgm(o);
  const { shape, config: c } = o;
  if (shape.outputs !== 3) throw new Error('sgm train: the relative-L2 luminance loss needs 3 outputs');
  const paddedOut = sgmLayerOut(shape, c, shape.hidden);
  if (paddedOut % c.K) throw new Error(`sgm train: padded output width ${paddedOut} is not a multiple of K=${c.K}`);
  for (let l = 1; l < layerCount(shape); l++) {
    if (layerIn(shape, l) % c.N) throw new Error(`sgm train: layer ${l} input ${layerIn(shape, l)} is not a multiple of N=${c.N}`);
  }
  if (shape.inputs % c.K) throw new Error(`sgm train: inputs ${shape.inputs} is not a multiple of K=${c.K}`);
}

/** Training kernel `nnForwardBackward`. Hooks: the caller's group-1 `nnTrainLoad(r, &x) -> target`. */
export function trainSgmWgsl(o: SgmOptions, hooks: string): string {
  validateSgmTrain(o);
  const { shape, config: c } = o;
  const ops = o.emulate ? emulatedOps(c) : nativeOps(c);
  const S = rowStride(shape);
  const R = o.records;
  const lanes = sgmLanes(o);
  const wg = sgmWorkgroupSize(o);
  const rowBlocksPerSubgroup = R / c.M / o.subgroups;
  const TR = o.emulate ? 'f32' : c.resultComponentType;
  const sgIndex = o.subgroups === 1 ? '0u' : o.emulate || o.offsetsFromLocalIndex ? `lid / ${lanes}u` : 'sgid';
  const hidden = shape.hidden;
  const paddedOut = sgmLayerOut(shape, c, hidden);

  /** One layer product Z = A · B over all output column blocks; `right(kb, nb)` builds the right tile load. */
  const product = (kDim: number, nDim: number, right: (kb: number, nb: string) => string, mma: Ops2['mma']): string => {
    const kBlocks = kDim / c.K;
    const rights = Array.from({ length: kBlocks }, (_, kb) => `      let r${kb} = ${right(kb, 'nb')};`);
    const mmas = Array.from({ length: kBlocks }, (_, kb) => `        acc = ${mma(ops.loadLeft(`rowBase + ${kb * c.K}u`, `${S}u`), `r${kb}`, 'acc')};`);
    return `  for (var nb = 0u; nb < ${nDim / c.N}u; nb++) {
${rights.join('\n')}
    for (var mbi = 0u; mbi < ${rowBlocksPerSubgroup}u; mbi++) {
      let rowBase = (sg * ${rowBlocksPerSubgroup}u + mbi) * ${c.M * S}u;
      var acc = ${ops.zero()};
${mmas.join('\n')}
      ${ops.store(`rowBase + nb * ${c.N}u`, 'acc', `${S}u`)};
    }
  }
  workgroupBarrier();`;
  };

  const body: string[] = [];

  // ---- forward ----
  for (let l = 0; l < layerCount(shape); l++) {
    const nIn = layerIn(shape, l);
    const nOut = layerOut(shape, l);
    const nOutPadded = sgmLayerOut(shape, c, l);
    const off = sgmLayerOffset(shape, c, l);
    body.push(`  // forward layer ${l}: ${nIn} → ${nOut}${nOutPadded !== nOut ? ` (padded to ${nOutPadded})` : ''}
${product(nIn, nOutPadded, (kb, nb) => ops.loadRight(`${off}u + ${nb} * ${c.N * nIn}u + ${kb * c.K}u`, `${nIn}u`), ops.mma)}`);
    if (l < hidden) {
      body.push(`  // a_${l + 1} = relu(z_${l}) in the operand type, to workgroup memory and \`activations\`
  for (var i = lid; i < ${R * nOut}u; i += ${wg}u) {
    let rl = i / ${nOut}u;
    let col = i % ${nOut}u;
    let v = T(max(f32(nnS[rl * ${S}u + col]), 0.0));
    nnA[rl * ${S}u + col] = v;
    if (base + rl < count) { nnActs[(${l + 1}u * params.capacity + base + rl) * ${S}u + col] = v; }
  }
  workgroupBarrier();`);
    }
  }

  // ---- loss gradient (relative L2 on luminance, denominator constant; not divided by the batch) ----
  body.push(`  // loss gradient: δ_${hidden} = 2 (y - t) / (lum(y)² + ε), zero beyond the 3 outputs and for records past the count
  for (var rl = lid; rl < ${R}u; rl += ${wg}u) {
    let r = base + rl;
    var d = vec3f(0.0);
    if (r < count) {
      let y = vec3f(f32(nnS[rl * ${S}u]), f32(nnS[rl * ${S}u + 1u]), f32(nnS[rl * ${S}u + 2u]));
      let e = y - nnTarget[rl].xyz;
      let lum = dot(vec3f(0.299, 0.587, 0.114), max(y, vec3f(0.0)));
      let denom = lum * lum + params.lossEpsilon;
      d = 2.0 * e / denom;
      nnLoss[r] = dot(e, e) / denom;
      nnDeltas[(${hidden}u * params.capacity + r) * ${S}u + 3u] = T(0.0);
      for (var j = 0u; j < 3u; j++) { nnDeltas[(${hidden}u * params.capacity + r) * ${S}u + j] = T(d[j]); }
    }
    for (var j = 0u; j < ${paddedOut}u; j++) { nnA[rl * ${S}u + j] = T(select(0.0, d[min(j, 2u)], j < 3u)); }
  }
  workgroupBarrier();`);

  // ---- backward ----
  for (let l = hidden; l >= 1; l--) {
    const nIn = layerIn(shape, l);
    const nOutPadded = sgmLayerOut(shape, c, l);
    const off = sgmLayerOffset(shape, c, l);
    body.push(`  // backward layer ${l}: δ_${l - 1} = (δ_${l} · W_${l}) ⊙ [a_${l} > 0]   (K = ${nOutPadded} layer outputs, N = ${nIn} inputs)
${product(nOutPadded, nIn, (kb, nb) => ops.loadRightN(`${off}u + ${kb * c.K * nIn}u + ${nb} * ${c.N}u`, `${nIn}u`), ops.mmaN)}
  for (var i = lid; i < ${R * nIn}u; i += ${wg}u) {
    let rl = i / ${nIn}u;
    let col = i % ${nIn}u;
    let r = base + rl;
    var v = T(0.0);
    if (r < count) {
      let a = nnActs[(${l}u * params.capacity + r) * ${S}u + col];
      v = T(select(0.0, f32(nnS[rl * ${S}u + col]), a > T(0.0)));
      nnDeltas[(${l - 1}u * params.capacity + r) * ${S}u + col] = v;
    }${l > 1 ? `
    nnA[rl * ${S}u + col] = v;` : ''}
  }${l > 1 ? `
  workgroupBarrier();` : ''}`);
  }

  const builtins = [
    '@builtin(workgroup_id) wid: vec3u',
    '@builtin(num_workgroups) nwg: vec3u',
    '@builtin(local_invocation_index) lid: u32',
    ...(sgIndex === 'sgid' ? ['@builtin(subgroup_id) sgid: u32'] : []),
  ];

  return `${header(mlpOptions(o), directives(o))}
${encodingWgsl()}
alias TR = ${TR};
@group(0) @binding(0) var<storage, read> nnSgmW: array<T>;
@group(0) @binding(1) var<storage, read> nnState: NnState;
@group(0) @binding(2) var<uniform> params: NnParams;
@group(0) @binding(3) var<storage, read_write> nnActs: array<T>;
@group(0) @binding(4) var<storage, read_write> nnDeltas: array<T>;
@group(0) @binding(5) var<storage, read_write> nnLoss: array<f32>;

var<workgroup> nnA: array<T, ${R * S}>;   // a_l, then δ_l (left operand)
var<workgroup> nnS: array<TR, ${R * S}>;  // matrix results
var<workgroup> nnTarget: array<vec4f, ${R}>;
${ops.prelude}
${hooks}

// ${o.emulate ? 'EMULATED ' : ''}subgroup matrix ${c.M}×${c.N}×${c.K} ${c.componentType} → ${c.resultComponentType}; ${R} records, ${o.subgroups} subgroup(s) of ${lanes} per workgroup.
@compute @workgroup_size(${wg})
fn nnForwardBackward(${builtins.join(', ')}) {
  let base = (wid.x + wid.y * nwg.x) * ${R}u;
  let count = nnState.count;
  let sg = ${sgIndex};${o.emulate ? `
  nnEmuLane = lid % ${32}u;` : ''}
  for (var rl = lid; rl < ${R}u; rl += ${wg}u) {
    let r = base + rl;
    var x: array<f32, NN_IN>;
    var t = vec3f(0.0);
    if (r < count) { t = nnTrainLoad(r, &x); }
    nnTarget[rl] = vec4f(t, 0.0);
    for (var k = 0u; k < ${S}u; k++) {
      let v = T(select(0.0, x[min(k, NN_IN - 1u)], k < NN_IN));
      nnA[rl * ${S}u + k] = v;
      if (r < count && k < NN_IN) { nnActs[r * ${S}u + k] = v; }
    }
  }
  workgroupBarrier();
${body.join('\n')}
}
`;
}

type Ops2 = { mma: (l: string, r: string, acc: string) => string };
