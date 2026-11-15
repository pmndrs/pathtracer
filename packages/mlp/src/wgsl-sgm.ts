/**
 * Subgroup-matrix inference arm (`wgsl-sgm`, research track N step N3a; docs/design/nrc.md, nrc-diverges.md §B).
 *
 * Layer l computes Z = A · W_lᵀ for a tile of R records: A (R × in, record-major in workgroup memory) is the left
 * operand, W_lᵀ the right operand, loaded straight from storage as a column-major view of row-major [out][in] weights
 * (stride = in), so no transpose copy is needed. Each output column block holds its in/K right tiles in registers while
 * the subgroup walks its row blocks.
 *
 * WGSL gaps and the switches that will delete them (subgroup-matrix-feedback.md §2):
 *  - no element access (gpuweb #11077): results go to workgroup memory, a per-lane pass applies ReLU and converts to
 *    the operand type, the next layer reloads (`relu: 'roundtrip'`, the only form today);
 *  - no result → left conversion (#11079): same round trip (`chain: 'roundtrip'`);
 *  - several subgroups per workgroup: row blocks split by `@builtin(subgroup_id)` (needs `subgroup_uniformity`,
 *    Chrome 145+). Whether matrix built-ins accept those offsets is what the probe page tests (TASKS N0.2).
 *
 * `emulate: true` emits the same kernel with plain-WGSL stand-ins (every lane holds whole tiles, f32). It runs
 * anywhere, including SwiftShader, so tiling, offsets, the transposed weight view, output padding and the
 * multi-subgroup row split are checked against the CPU oracle before a device with subgroup matrices is available.
 * Not for timing.
 *
 * The output layer is padded to a multiple of N rows with zeros (`nnPackSgm` repacks the inference weights).
 */

import { layerCount, layerIn, layerOut, type MlpShape } from './config.ts';
import { encodingWgsl } from './encoding.ts';
import { header, rowStride, type MlpWgslOptions } from './wgsl.ts';

export type SgmScalar = 'f16' | 'f32';

/** One entry of `GPUAdapterInfo.subgroupMatrixConfigs`, reduced to what we use. */
export interface SgmConfig {
  componentType: SgmScalar;
  resultComponentType: SgmScalar;
  M: number;
  N: number;
  K: number;
  minSubgroupSize: number;
  maxSubgroupSize: number;
}

export interface SgmOptions {
  shape: MlpShape;
  config: SgmConfig;
  /** Records per workgroup (multiple of M × subgroups). */
  records: number;
  /** Subgroups per workgroup; > 1 splits row blocks by `subgroup_id`. */
  subgroups: number;
  /** WGSL enable name: `chromium_experimental_subgroup_matrix` (today) or `subgroup_matrix`. */
  enableName?: string;
  /** Emit `diagnostic(off, <name>);` for the matrix uniformity diagnostic (probe only). */
  uniformityDiagnosticOff?: string;
  /** Per-subgroup offsets from `local_invocation_index / size` instead of `subgroup_id` (probe control case). */
  offsetsFromLocalIndex?: boolean;
  /** Plain-WGSL stand-ins for the matrix built-ins (correctness on any device; f32 only). */
  emulate?: boolean;
}

/** Lanes per subgroup the kernel is laid out for (the config's max; the proposal requires workgroup x to be a multiple). */
export function sgmLanes(o: SgmOptions): number {
  return o.emulate ? 32 : o.config.maxSubgroupSize;
}

export function sgmWorkgroupSize(o: SgmOptions): number {
  return sgmLanes(o) * o.subgroups;
}

/** Output rows of layer l, padded to N. */
export function sgmLayerOut(shape: MlpShape, config: Pick<SgmConfig, 'N'>, l: number): number {
  return Math.ceil(layerOut(shape, l) / config.N) * config.N;
}

export function sgmLayerOffset(shape: MlpShape, config: Pick<SgmConfig, 'N'>, l: number): number {
  let off = 0;
  for (let i = 0; i < l; i++) off += sgmLayerOut(shape, config, i) * layerIn(shape, i);
  return off;
}

export function sgmWeightCount(shape: MlpShape, config: Pick<SgmConfig, 'N'>): number {
  return sgmLayerOffset(shape, config, layerCount(shape));
}

/** Workgroup memory the kernel needs (bytes): activations in T plus results in TR. */
export function sgmWorkgroupBytes(o: SgmOptions): number {
  const s = rowStride(o.shape);
  const t = o.emulate ? 4 : o.config.componentType === 'f16' ? 2 : 4;
  const tr = o.emulate ? 4 : o.config.resultComponentType === 'f16' ? 2 : 4;
  return o.records * s * (t + tr);
}

export function validateSgm(o: SgmOptions): void {
  const { shape, config } = o;
  for (let l = 0; l < layerCount(shape); l++) {
    if (layerIn(shape, l) % config.K) throw new Error(`sgm: layer ${l} input ${layerIn(shape, l)} is not a multiple of K=${config.K}`);
    if (l < shape.hidden && layerOut(shape, l) % config.N) throw new Error(`sgm: layer ${l} width is not a multiple of N=${config.N}`);
  }
  if (o.records % (config.M * o.subgroups)) throw new Error(`sgm: records ${o.records} must be a multiple of M × subgroups = ${config.M * o.subgroups}`);
  if (o.emulate && (config.componentType !== 'f32' || config.resultComponentType !== 'f32')) throw new Error('sgm emulation is f32 only');
}

/**
 * Picks a usable config from `adapter.info.subgroupMatrixConfigs`: square-friendly sizes that divide the network,
 * operand type `f16` when `preferF16` (and the device has shader-f16), else `f32`; f32 results preferred.
 */
export function pickSubgroupMatrixConfig(info: GPUAdapterInfo | undefined, shape: MlpShape, preferF16: boolean): SgmConfig | null {
  const raw = (info as unknown as { subgroupMatrixConfigs?: ArrayLike<Record<string, unknown>> } | undefined)?.subgroupMatrixConfigs;
  if (!raw) return null;
  const read = (c: Record<string, unknown>): SgmConfig => ({
    componentType: String(c.componentType) as SgmScalar,
    resultComponentType: String(c.resultComponentType) as SgmScalar,
    M: Number(c.M),
    N: Number(c.N),
    K: Number(c.K),
    minSubgroupSize: Number(c.minSubgroupSize ?? (info as unknown as { subgroupMinSize?: number }).subgroupMinSize ?? 32),
    maxSubgroupSize: Number(c.maxSubgroupSize ?? (info as unknown as { subgroupMaxSize?: number }).subgroupMaxSize ?? 32),
  });
  const fits = (c: SgmConfig) =>
    (c.componentType === 'f16' || c.componentType === 'f32') &&
    (c.resultComponentType === 'f16' || c.resultComponentType === 'f32') &&
    shape.width % c.N === 0 && shape.width % c.K === 0 && shape.inputs % c.K === 0;
  const all = Array.from(raw, read).filter(fits);
  const want: SgmScalar = preferF16 ? 'f16' : 'f32';
  const score = (c: SgmConfig) => (c.componentType === want ? 2 : 0) + (c.resultComponentType === 'f32' ? 1 : 0);
  all.sort((a, b) => score(b) - score(a));
  return all[0] ?? null;
}

/** Matrix built-ins, native or emulated. */
interface Ops {
  prelude: string;
  left: string;
  right: string;
  result: string;
  /** Load a left tile from workgroup activations (row-major). */
  loadLeft(offset: string, stride: string): string;
  /** Load a right tile from the weight buffer as a column-major view. */
  loadRight(offset: string, stride: string): string;
  zero(): string;
  mma(l: string, r: string, acc: string): string;
  /** Store a result tile to workgroup results (row-major). */
  store(offset: string, value: string, stride: string): string;
}

function nativeOps(c: SgmConfig): Ops {
  const left = `subgroup_matrix_left<${c.componentType}, ${c.K}, ${c.M}>`;
  const right = `subgroup_matrix_right<${c.componentType}, ${c.N}, ${c.K}>`;
  const result = `subgroup_matrix_result<${c.resultComponentType}, ${c.N}, ${c.M}>`;
  return {
    prelude: '',
    left,
    right,
    result,
    loadLeft: (off, stride) => `subgroupMatrixLoad<${left}>(&nnA, ${off}, false, ${stride})`,
    loadRight: (off, stride) => `subgroupMatrixLoad<${right}>(&nnSgmW, ${off}, true, ${stride})`,
    zero: () => `${result}()`,
    mma: (l, r, acc) => `subgroupMatrixMultiplyAccumulate(${l}, ${r}, ${acc})`,
    store: (off, value, stride) => `subgroupMatrixStore(&nnS, ${off}, ${value}, false, ${stride})`,
  };
}

function emulatedOps(c: SgmConfig): Ops {
  const { M, N, K } = c;
  return {
    prelude: /* wgsl */ `
// Emulated subgroup matrices (correctness only). Left/right "tiles" are lazy views (offset, stride) read at multiply
// time, which matches the native load semantics here because nnA and the weights do not change in between. Every
// lane computes the whole result tile (f32).
alias EmuLeft = vec2u;  // M rows × K cols, row-major view of nnA
alias EmuRight = vec2u; // K rows × N cols, column-major view of the weights: element (k, n) at off + n * stride + k
alias EmuResult = array<f32, ${M * N}>; // M rows × N cols, row-major
fn emuMma(l: EmuLeft, r: EmuRight, acc: EmuResult) -> EmuResult {
  var o = acc;
  for (var i = 0u; i < ${M}u; i++) {
    for (var n = 0u; n < ${N}u; n++) {
      var s = o[i * ${N}u + n];
      for (var k = 0u; k < ${K}u; k++) { s += f32(nnA[l.x + i * l.y + k]) * f32(nnSgmW[r.x + n * r.y + k]); }
      o[i * ${N}u + n] = s;
    }
  }
  return o;
}
// Every lane writes the same values: a benign race, and only in this emulation.
fn emuStore(off: u32, v: EmuResult, stride: u32) {
  for (var i = 0u; i < ${M}u; i++) { for (var n = 0u; n < ${N}u; n++) { nnS[off + i * stride + n] = TR(v[i * ${N}u + n]); } }
}
`,
    left: 'EmuLeft',
    right: 'EmuRight',
    result: 'EmuResult',
    loadLeft: (off, stride) => `vec2u(${off}, ${stride})`,
    loadRight: (off, stride) => `vec2u(${off}, ${stride})`,
    zero: () => 'EmuResult()',
    mma: (l, r, acc) => `emuMma(${l}, ${r}, ${acc})`,
    store: (off, value, stride) => `emuStore(${off}, ${value}, ${stride})`,
  };
}

function directives(o: SgmOptions): string[] {
  const d: string[] = [];
  const usesSubgroupId = o.subgroups > 1 && !o.offsetsFromLocalIndex && !o.emulate;
  if (!o.emulate) d.push(`enable ${o.enableName ?? 'chromium_experimental_subgroup_matrix'};`);
  if (usesSubgroupId) d.push('enable subgroups;');
  if (!o.emulate && o.config.resultComponentType === 'f16' && o.config.componentType !== 'f16') d.push('enable f16;');
  if (o.uniformityDiagnosticOff && !o.emulate) d.push(`diagnostic(off, ${o.uniformityDiagnosticOff});`);
  return d;
}

function mlpOptions(o: SgmOptions): MlpWgslOptions {
  // T is the operand type; emulation runs in f32.
  return { shape: o.shape, precision: o.emulate ? 'f32' : o.config.componentType };
}

/** Inference kernel `nnInferSgm`. Hooks as for the portable arm (`nnInferLoad`, `nnInferStore`, group 1). */
export function inferSgmWgsl(o: SgmOptions, hooks: string): string {
  validateSgm(o);
  const { shape, config: c } = o;
  const ops = o.emulate ? emulatedOps(c) : nativeOps(c);
  const S = rowStride(shape);
  const R = o.records;
  const lanes = sgmLanes(o);
  const wg = sgmWorkgroupSize(o);
  const rowBlocksPerSubgroup = R / c.M / o.subgroups;
  const TR = o.emulate ? 'f32' : c.resultComponentType;
  const sgIndex = o.subgroups === 1 ? '0u' : o.emulate || o.offsetsFromLocalIndex ? `lid / ${lanes}u` : 'sgid';

  const layers: string[] = [];
  for (let l = 0; l < layerCount(shape); l++) {
    const nIn = layerIn(shape, l);
    const nOut = sgmLayerOut(shape, c, l);
    const off = sgmLayerOffset(shape, c, l);
    const kBlocks = nIn / c.K;
    const rights = Array.from({ length: kBlocks }, (_, kb) => `      let r${kb} = ${ops.loadRight(`${off}u + nb * ${c.N * nIn}u + ${kb * c.K}u`, `${nIn}u`)};`);
    const mmas = Array.from({ length: kBlocks }, (_, kb) => `        acc = ${ops.mma(ops.loadLeft(`rowBase + ${kb * c.K}u`, `${S}u`), `r${kb}`, 'acc')};`);
    layers.push(`  // Layer ${l}: ${nIn} → ${layerOut(shape, l)}${nOut !== layerOut(shape, l) ? ` (padded to ${nOut})` : ''}
  for (var nb = 0u; nb < ${nOut / c.N}u; nb++) {
${rights.join('\n')}
    for (var mbi = 0u; mbi < ${rowBlocksPerSubgroup}u; mbi++) {
      let rowBase = (sg * ${rowBlocksPerSubgroup}u + mbi) * ${c.M * S}u;
      var acc = ${ops.zero()};
${mmas.join('\n')}
      ${ops.store(`rowBase + nb * ${c.N}u`, 'acc', `${S}u`)};
    }
  }
  workgroupBarrier();`);
    if (l < shape.hidden) {
      // Round trip (gpuweb #11077 / #11079): ReLU + convert to the operand type through workgroup memory.
      layers.push(`  for (var i = lid; i < ${R * layerOut(shape, l)}u; i += ${wg}u) {
    let at = (i / ${layerOut(shape, l)}u) * ${S}u + i % ${layerOut(shape, l)}u;
    nnA[at] = T(max(f32(nnS[at]), 0.0));
  }
  workgroupBarrier();`);
    }
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

var<workgroup> nnA: array<T, ${R * S}>;
var<workgroup> nnS: array<TR, ${R * S}>;
${ops.prelude}
${hooks}

// ${o.emulate ? 'EMULATED ' : ''}subgroup matrix ${c.M}×${c.N}×${c.K} ${c.componentType} → ${c.resultComponentType}; ${R} records, ${o.subgroups} subgroup(s) of ${lanes} per workgroup.
@compute @workgroup_size(${wg})
fn nnInferSgm(${builtins.join(', ')}) {
  let base = (wid.x + wid.y * nwg.x) * ${R}u;
  let sg = ${sgIndex};
  for (var i = lid; i < ${R}u; i += ${wg}u) {
    let r = base + i;
    var x: array<f32, NN_IN>;
    if (r < nnState.count) { x = nnInferLoad(r); }
    for (var k = 0u; k < NN_IN; k++) { nnA[i * ${S}u + k] = T(x[k]); }
  }
  workgroupBarrier();
${layers.join('\n')}
  for (var i = lid; i < ${R}u; i += ${wg}u) {
    let r = base + i;
    if (r < nnState.count) { nnInferStore(r, vec3f(f32(nnS[i * ${S}u]), f32(nnS[i * ${S}u + 1u]), f32(nnS[i * ${S}u + 2u]))); }
  }
}
`;
}

/** `nnPackSgm`: inference weights (`T`, row-major per layer) → the padded sgm layout. One thread per padded weight. */
export function packSgmWgsl(o: SgmOptions): string {
  const { shape, config: c } = o;
  const cases: string[] = [];
  let src = 0;
  for (let l = 0; l < layerCount(shape); l++) {
    const nIn = layerIn(shape, l);
    const end = sgmLayerOffset(shape, c, l + 1);
    cases.push(`  ${l ? 'else ' : ''}if (w < ${end}u) { let i = w - ${sgmLayerOffset(shape, c, l)}u; let row = i / ${nIn}u; v = select(T(0.0), nnUse[${src}u + i], row < ${layerOut(shape, l)}u); }`);
    src += nIn * layerOut(shape, l);
  }
  return `${header(mlpOptions(o))}
@group(0) @binding(0) var<storage, read> nnUse: array<T>;
@group(0) @binding(1) var<storage, read_write> nnPacked: array<T>;

@compute @workgroup_size(64)
fn nnPackSgm(@builtin(global_invocation_id) gid: vec3u) {
  let w = gid.x;
  if (w >= ${sgmWeightCount(shape, c)}u) { return; }
  var v = T(0.0);
${cases.join('\n')}
  nnPacked[w] = v;
}
`;
}
