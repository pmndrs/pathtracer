/**
 * Tiled forward-backward arm (N1.2b), the alternative to `trainWgsl`'s one-thread-per-record `nnForwardBackward`.
 * Same entry point name, bindings (group 0 bindings 0-5, caller hooks in group 1), `activations` / `deltas` layout and
 * delta scaling (not divided by N), so the gradient pass and the engine are unchanged.
 *
 * A workgroup of 256 threads (16 × 16) handles `records` records (R, a multiple of 16). The activation tile
 * (R × row stride, f32 vec4 rows, values rounded to `T` first on the f16 arm) and the layer's weights (`T4`, staged
 * one layer at a time) live in workgroup memory. Thread (tx, ty) owns outputs 4tx..4tx+3 of records
 * ty·R/16 .. (ty+1)·R/16 − 1, a (R/16) × 4 register block of z = W·a: per k it loads 4 weight vec4s and R/16
 * activation vec4s for 4·R/16 dot products.
 *
 *  forward:  stage W_l → z = W_l a_l in registers → barrier → ReLU, round to T, write tile and `activations`.
 *  loss:     last layer's tx = 0 threads hold y (xyz): relative-L2 gradient exactly as the per-record kernel,
 *            written to the tile (as δ_hidden) and `deltas`.
 *  backward: stage W_l → δ_{l-1} = (W_lᵀ δ_l) ⊙ [a_l > 0] (a_l re-read from `activations`) → barrier → write tile and
 *            `deltas`. Thread (tx, ty) owns inputs 4tx..4tx+3 of the same records.
 *
 * Records at or past the GPU count load nothing (zero tile), contribute zero deltas and write nothing. The caller's
 * hook `nnTrainLoad` is called once per record by thread `lid < R`. All barriers are in uniform control flow (the
 * loops have constant bounds). Limits: every layer's input and output width ≤ 64; R ≤ 256.
 */

import { layerCount, layerIn, layerOffset, layerOut } from './config.ts';
import { encodingWgsl } from './encoding.ts';
import { header, rowStride, type MlpWgslOptions } from './wgsl.ts';

export type ForwardKind = 'perRecord' | 'tiled' | 'sgm';

export interface ForwardOptions {
  /** Records per workgroup (multiple of 16, at most 256). */
  records: number;
}

export const DEFAULT_FORWARD: ForwardOptions = { records: 64 };

/** Records per workgroup by precision (N1.2b sweep, M5 Pro): 64 for f16; 48 for f32, where 64 needs 33.8 KB of workgroup memory (> 32 KiB). */
export function defaultForwardRecords(precision: 'f32' | 'f16'): number {
  return precision === 'f16' ? 64 : 48;
}

export const FORWARD_THREADS = 256;
const TX = 16;
const MAX_WIDTH = 64;

function forwardWeightRows(nIn4: number, nOut: number): string {
  const rows = nOut % 4 || 4;
  return Array.from({ length: 4 }, (_, j) => {
    const value = j < rows ? `vec4f(nnLayer[(4u * tx + ${j}u) * ${nIn4}u + k])` : 'vec4f(0.0)';
    return `      let w${j} = ${value};`;
  }).join('\n');
}

export function validateForwardOptions(o: MlpWgslOptions, f: ForwardOptions): void {
  const { shape } = o;
  for (let l = 0; l < layerCount(shape); l++) {
    if (layerIn(shape, l) > MAX_WIDTH || layerOut(shape, l) > MAX_WIDTH) throw new Error(`tiled forward: layer ${l} is ${layerOut(shape, l)}×${layerIn(shape, l)}, the tile is ${MAX_WIDTH}×${MAX_WIDTH}`);
  }
  if (!Number.isInteger(f.records) || f.records < 16 || f.records % 16 || f.records > 256) throw new Error(`tiled forward: records must be a multiple of 16 in 16..256, got ${f.records}`);
}

/** Workgroup memory the kernel needs (bytes): f32 activation/delta tile, staged layer in T, targets. */
export function forwardWorkgroupBytes(o: MlpWgslOptions, f: ForwardOptions): number {
  const { shape } = o;
  const maxLayer = Math.max(...Array.from({ length: layerCount(shape) }, (_, l) => layerIn(shape, l) * layerOut(shape, l)));
  const t = o.precision === 'f16' ? 2 : 4;
  return f.records * rowStride(shape) * 4 + maxLayer * t + f.records * 16;
}

export function trainTiledWgsl(o: MlpWgslOptions, hooks: string, f: ForwardOptions = DEFAULT_FORWARD): string {
  validateForwardOptions(o, f);
  const { shape } = o;
  if (shape.outputs !== 3) throw new Error('trainTiledWgsl: the relative-L2 luminance loss needs 3 outputs');
  const R = f.records;
  const RB = R / TX;
  const S4 = rowStride(shape) / 4;
  const layers = layerCount(shape);
  const maxLayer = Math.max(...Array.from({ length: layers }, (_, l) => layerIn(shape, l) * layerOut(shape, l)));
  const each = (fn: (rb: number) => string, sep = '\n') => Array.from({ length: RB }, (_, rb) => fn(rb)).join(sep);

  const body: string[] = [];

  // ---- forward ----
  for (let l = 0; l < layers; l++) {
    const nIn4 = layerIn(shape, l) / 4;
    const nOut = layerOut(shape, l);
    const nOut4 = Math.ceil(nOut / 4);
    const last = l === shape.hidden;
    body.push(`  // forward layer ${l}: ${nOut} × ${layerIn(shape, l)}
  nnStageTiled(${layerOffset(shape, l) / 4}u, ${(layerIn(shape, l) * nOut) / 4}u, lid);
${each((rb) => `  var acc${rb} = vec4f(0.0);`)}
  if (tx < ${nOut4}u) {
    for (var k = 0u; k < ${nIn4}u; k++) {
${forwardWeightRows(nIn4, nOut)}
${each((rb) => `      { let av = nnTile[(ty * ${RB}u + ${rb}u) * NN_S4 + k]; acc${rb} += vec4f(dot(av, w0), dot(av, w1), dot(av, w2), dot(av, w3)); }`)}
    }
  }
  workgroupBarrier();`);
    if (!last) {
      body[body.length - 1] += `\n  if (tx < ${nOut4}u) {
${each(
  (rb) => `    {
      let rl = ty * ${RB}u + ${rb}u;
      let v = vec4f(T4(max(acc${rb}, vec4f(0.0))));
      nnTile[rl * NN_S4 + tx] = v;
      if (r0 + rl < count) { nnActs[(${l + 1}u * params.capacity + r0 + rl) * NN_S4 + tx] = T4(v); }
    }`,
)}
  }`;
    } else {
      body[body.length - 1] += `\n  // Relative L2 on luminance, denominator held constant; not divided by the batch (nnGradientTile divides by count).
  if (tx == 0u) {
${each(
  (rb) => `    {
      let rl = ty * ${RB}u + ${rb}u;
      let r = r0 + rl;
      var d = vec4f(0.0);
      if (r < count) {
        let y = acc${rb}.xyz;
        let e = y - nnTarget[rl].xyz;
        let lum = dot(vec3f(0.299, 0.587, 0.114), max(y, vec3f(0.0)));
        let denom = lum * lum + params.lossEpsilon;
        d = vec4f(2.0 * e / denom, 0.0);
        nnLoss[r] = dot(e, e) / denom;
        nnDeltas[(${l}u * params.capacity + r) * NN_S4] = T4(d);
      }
      nnTile[rl * NN_S4] = d;
    }`,
)}
  }`;
    }
  }

  // ---- backward ----
  for (let l = shape.hidden; l >= 1; l--) {
    const nIn = layerIn(shape, l);
    const nIn4 = nIn / 4;
    const nOut = layerOut(shape, l);
    const nOut4 = Math.ceil(nOut / 4);
    // The only ragged layer has three outputs, so its fourth row is never staged.
    const rows = Array.from({ length: nOut % 4 || 4 }, (_, j) => j);
    body.push(`  // backward layer ${l}: δ_${l - 1} = (W_${l}ᵀ δ_${l}) ⊙ [a_${l} > 0]
  nnStageTiled(${layerOffset(shape, l) / 4}u, ${(nIn * nOut) / 4}u, lid);
${each((rb) => `  var acc${rb} = vec4f(0.0);`)}
  if (tx < ${nIn4}u) {
    for (var o4 = 0u; o4 < ${nOut4}u; o4++) {
${rows.map((j) => `      let w${j} = vec4f(nnLayer[(4u * o4 + ${j}u) * ${nIn4}u + tx]);`).join('\n')}
${each((rb) => `      { let dv${rb} = nnTile[(ty * ${RB}u + ${rb}u) * NN_S4 + o4]; acc${rb} += ${rows.map((j) => `dv${rb}[${j}] * w${j}`).join(' + ')}; }`)}
    }
${each(
  (rb) => `    {
      let r = r0 + ty * ${RB}u + ${rb}u;
      var act = T4(0.0);
      if (r < count) { act = nnActs[(${l}u * params.capacity + r) * NN_S4 + tx]; }
      acc${rb} = select(vec4f(0.0), acc${rb}, act > T4(0.0));
    }`,
)}
  }
  workgroupBarrier();
  if (tx < ${nIn4}u) {
${each(
  (rb) => `    {
      let rl = ty * ${RB}u + ${rb}u;
      nnTile[rl * NN_S4 + tx] = acc${rb};
      if (r0 + rl < count) { nnDeltas[(${l - 1}u * params.capacity + r0 + rl) * NN_S4 + tx] = T4(acc${rb}); }
    }`,
)}
  }`);
  }

  return `${header(o)}
${encodingWgsl()}
const NN_R = ${R}u;
@group(0) @binding(0) var<storage, read> nnWeights: array<T4>;
@group(0) @binding(1) var<storage, read> nnState: NnState;
@group(0) @binding(2) var<uniform> params: NnParams;
@group(0) @binding(3) var<storage, read_write> nnActs: array<T4>;
@group(0) @binding(4) var<storage, read_write> nnDeltas: array<T4>;
@group(0) @binding(5) var<storage, read_write> nnLoss: array<f32>;

var<workgroup> nnLayer: array<T4, ${maxLayer / 4}>;
var<workgroup> nnTile: array<vec4f, ${R * S4}>;   // R rows of NN_S4 vec4s: a_l, then δ_l
var<workgroup> nnTarget: array<vec4f, ${R}>;

// Copies \`n4\` vec4s of weights starting at vec4 \`off4\` into workgroup memory. Uniform control flow only.
fn nnStageTiled(off4: u32, n4: u32, lid: u32) {
  workgroupBarrier();
  for (var k = lid; k < n4; k += ${FORWARD_THREADS}u) { nnLayer[k] = nnWeights[off4 + k]; }
  workgroupBarrier();
}
${hooks}

@compute @workgroup_size(${FORWARD_THREADS})
fn nnForwardBackward(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let r0 = (wid.x + wid.y * nwg.x) * NN_R;
  let count = nnState.count;
  let tx = lid % ${TX}u;
  let ty = lid / ${TX}u;
  for (var rl = lid; rl < NN_R; rl += ${FORWARD_THREADS}u) {
    let r = r0 + rl;
    var x: array<f32, NN_IN>;
    var t = vec3f(0.0);
    if (r < count) { t = nnTrainLoad(r, &x); }
    for (var k = 0u; k < NN_IN / 4u; k++) { nnTile[rl * NN_S4 + k] = vec4f(T4(vec4f(x[4u * k], x[4u * k + 1u], x[4u * k + 2u], x[4u * k + 3u]))); }
    nnTarget[rl] = vec4f(t, 0.0);
  }
  workgroupBarrier();
  for (var e = lid; e < NN_R * (NN_IN / 4u); e += ${FORWARD_THREADS}u) {
    let rl = e / (NN_IN / 4u);
    let k = e % (NN_IN / 4u);
    if (r0 + rl < count) { nnActs[(r0 + rl) * NN_S4 + k] = T4(nnTile[rl * NN_S4 + k]); }
  }
${body.map((b) => `  {\n${b}\n  }`).join('\n')}
}
`;
}

/**
 * Tiled portable INFERENCE (N1.4): the forward half of `trainTiledWgsl` without activation stores. Same entry point name
 * (`nnInfer`) and bindings as `inferWgsl` (group 0: weights, state; caller hooks `nnInferLoad` / `nnInferStore` in group 1),
 * so the engine only swaps the generator. A workgroup of 256 threads handles R records; thread (tx, ty) owns outputs 4tx..4tx+3
 * of R/16 records. Workgroup memory: the f32 activation tile and one staged layer.
 */
export function inferTiledWgsl(o: MlpWgslOptions, hooks: string, f: ForwardOptions = DEFAULT_FORWARD): string {
  validateForwardOptions(o, f);
  const { shape } = o;
  const R = f.records;
  const RB = R / TX;
  const S4 = rowStride(shape) / 4;
  const layers = layerCount(shape);
  const maxLayer = Math.max(...Array.from({ length: layers }, (_, l) => layerIn(shape, l) * layerOut(shape, l)));
  const each = (fn: (rb: number) => string, sep = '\n') => Array.from({ length: RB }, (_, rb) => fn(rb)).join(sep);
  const body: string[] = [];
  for (let l = 0; l < layers; l++) {
    const nIn4 = layerIn(shape, l) / 4;
    const nOut = layerOut(shape, l);
    const nOut4 = Math.ceil(nOut / 4);
    const last = l === shape.hidden;
    let b = `  // layer ${l}: ${nOut} × ${layerIn(shape, l)}
  nnStageTiled(${layerOffset(shape, l) / 4}u, ${(layerIn(shape, l) * nOut) / 4}u, lid);
${each((rb) => `  var acc${rb} = vec4f(0.0);`)}
  if (tx < ${nOut4}u) {
    for (var k = 0u; k < ${nIn4}u; k++) {
${forwardWeightRows(nIn4, nOut)}
${each((rb) => `      { let av = nnTile[(ty * ${RB}u + ${rb}u) * NN_S4 + k]; acc${rb} += vec4f(dot(av, w0), dot(av, w1), dot(av, w2), dot(av, w3)); }`)}
    }
  }
  workgroupBarrier();`;
    if (!last) {
      b += `\n  if (tx < ${nOut4}u) {
${each((rb) => `    nnTile[(ty * ${RB}u + ${rb}u) * NN_S4 + tx] = vec4f(T4(max(acc${rb}, vec4f(0.0))));`)}
  }`;
    } else {
      if (nOut > 4) throw new Error('inferTiledWgsl: the output layer must have at most 4 outputs');
      b += `\n  if (tx == 0u) {
${each((rb) => `    { let r = r0 + ty * ${RB}u + ${rb}u; if (r < count) { nnInferStore(r, acc${rb}.xyz); } }`)}
  }`;
    }
    body.push(b);
  }
  return `${header(o)}
${encodingWgsl()}
const NN_R = ${R}u;
@group(0) @binding(0) var<storage, read> nnWeights: array<T4>;
@group(0) @binding(1) var<storage, read> nnState: NnState;

var<workgroup> nnLayer: array<T4, ${maxLayer / 4}>;
var<workgroup> nnTile: array<vec4f, ${R * S4}>;

// Copies \`n4\` vec4s of weights starting at vec4 \`off4\` into workgroup memory. Uniform control flow only.
fn nnStageTiled(off4: u32, n4: u32, lid: u32) {
  workgroupBarrier();
  for (var k = lid; k < n4; k += ${FORWARD_THREADS}u) { nnLayer[k] = nnWeights[off4 + k]; }
  workgroupBarrier();
}
${hooks}

@compute @workgroup_size(${FORWARD_THREADS})
fn nnInfer(@builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_index) lid: u32) {
  let r0 = (wid.x + wid.y * nwg.x) * NN_R;
  let count = nnState.count;
  let tx = lid % ${TX}u;
  let ty = lid / ${TX}u;
  for (var rl = lid; rl < NN_R; rl += ${FORWARD_THREADS}u) {
    let r = r0 + rl;
    var x: array<f32, NN_IN>;
    if (r < count) { x = nnInferLoad(r); }
    for (var k = 0u; k < NN_IN / 4u; k++) { nnTile[rl * NN_S4 + k] = vec4f(T4(vec4f(x[4u * k], x[4u * k + 1u], x[4u * k + 2u], x[4u * k + 3u]))); }
  }
  workgroupBarrier();
${body.map((b) => `  {\n${b}\n  }`).join('\n')}
}
`;
}

/** Workgroup memory of the tiled inference kernel (bytes): f32 activation tile and one staged layer in T. */
export function inferTiledWorkgroupBytes(o: MlpWgslOptions, f: ForwardOptions): number {
  const { shape } = o;
  const maxLayer = Math.max(...Array.from({ length: layerCount(shape) }, (_, l) => layerIn(shape, l) * layerOut(shape, l)));
  return f.records * rowStride(shape) * 4 + maxLayer * (o.precision === 'f16' ? 2 : 4);
}
