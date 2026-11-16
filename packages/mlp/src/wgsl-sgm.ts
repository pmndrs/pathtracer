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
 * `emulate: true` emits the same kernel with plain-WGSL stand-ins (32 lanes share each tile, f32). It runs
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
  /**
   * TIMING PROBE ONLY (N3): the ReLU/convert pass between layers. 'relu' (default, correct), 'skip' (no pass: the next
   * layer reads stale inputs, wrong results), 'double' (the pass twice). relu − skip ≈ double − relu = its cost.
   */
  roundTrip?: 'relu' | 'skip' | 'double';
  /**
   * Inference data layout (N2r-b). 'row' (default): per output column block, its right tiles are loaded and one
   * accumulator walks the subgroup's row blocks, a left load per multiply. 'blocked': the layer's right tiles are all
   * held in registers, each row block loads its left tiles once and feeds one accumulator per column block; with
   * f16 results the result tile is stored straight into a second activation buffer (ping-pong) and ReLU runs in place,
   * so there is no convert pass. Native only (no emulated form). The roofline that motivated it: TASKS N2r.
   */
  layout?: 'row' | 'blocked';
  /** Row stride padding in elements for 'blocked' (default 8: a 64-half stride bank-conflicts on NVIDIA, N2r). */
  pad?: number;
  /** TIMING PROBE ONLY ('blocked'): 'skip' writes a constant instead of calling `nnInferLoad` (wrong results). */
  inputProbe?: 'skip' | 'none';
  /** 'blocked' f16 ReLU: adjacent halves per lane item (1, 2, 4 or 8; default 4). N2r-c: 1 → 4.63, 2 → 3.98 ms/M on the 3060. */
  reluRun?: number;
  /**
   * TIMING PROBE ONLY ('blocked' training, N2r-d; wrong gradients): 'noMaskRead' tests the ReLU mask on the delta itself
   * instead of reading a_l from `activations`; 'noStores' skips the activation and delta stores (and so reads stale
   * activations); 'forwardOnly' stops after the loss gradient.
   */
  trainProbe?: 'noMaskRead' | 'noStores' | 'forwardOnly';
  /**
   * 'blocked' training: where the backward pass gets the ReLU mask [a_l > 0]. 'global' reads a_l back from `activations`
   * (N2r-d probe: ~0.13 of 0.42 ms on the 3060); 'bits' keeps one bit per hidden activation in workgroup memory
   * (`atomicOr` in the forward ReLU pass, R × width × hidden / 8 bytes).
   */
  trainMask?: 'global' | 'bits';
}

/** Lanes per subgroup the kernel is laid out for (the config's max; the proposal requires workgroup x to be a multiple). */
export function sgmLanes(o: SgmOptions): number {
  return o.emulate ? EMU_LANES : o.config.maxSubgroupSize;
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
  const blocked = o.layout === 'blocked';
  const s = rowStride(o.shape) + (blocked ? sgmPad(o) : 0);
  const t = o.emulate ? 4 : o.config.componentType === 'f16' ? 2 : 4;
  const tr = o.emulate ? 4 : o.config.resultComponentType === 'f16' ? 2 : 4;
  // 'blocked' with equal operand and result types ping-pongs two activation buffers instead of activations + results.
  return o.records * s * (blocked && o.config.resultComponentType === o.config.componentType ? 2 * t : t + tr);
}

export function sgmPad(o: Pick<SgmOptions, 'pad'>): number {
  return o.pad ?? 8;
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
 * operand type `f16` when `preferF16` (and the device has shader-f16), else `f32`; f32 results preferred unless
 * `accumulate` asks for a result type (f16 sums run at twice the f32-sum rate on GeForce tensor cores; VkNRC uses them).
 */
export function pickSubgroupMatrixConfig(info: GPUAdapterInfo | undefined, shape: MlpShape, preferF16: boolean, accumulate?: SgmScalar): SgmConfig | null {
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
  const score = (c: SgmConfig) => (c.componentType === want ? 2 : 0) + (c.resultComponentType === (accumulate ?? 'f32') ? 1 : 0);
  // Ties go to the biggest tile (NVIDIA lists 16x8x8, 16x8x16 and 16x16x16 per type pair).
  all.sort((a, b) => score(b) - score(a) || b.M * b.N * b.K - a.M * a.N * a.K);
  return all[0] ?? null;
}

/** Matrix built-ins, native or emulated. */
export interface Ops {
  prelude: string;
  left: string;
  right: string;
  result: string;
  /** Load a left tile from workgroup activations (row-major). */
  loadLeft(offset: string, stride: string): string;
  /** Load a right tile from the weight buffer as a column-major view. */
  loadRight(offset: string, stride: string): string;
  /** Load a right tile as a row-major view of the weight buffer (K rows × N columns, element (k, n) at offset + k * stride + n): Wᵀ-free products δ·W of the backward pass. */
  loadRightN(offset: string, stride: string): string;
  zero(): string;
  mma(l: string, r: string, acc: string): string;
  /** Multiply-accumulate with a right tile from `loadRightN`. */
  mmaN(l: string, r: string, acc: string): string;
  /** Store a result tile to workgroup results (row-major). */
  store(offset: string, value: string, stride: string): string;
}

export function nativeOps(c: SgmConfig): Ops {
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
    loadRightN: (off, stride) => `subgroupMatrixLoad<${right}>(&nnSgmW, ${off}, false, ${stride})`,
    zero: () => `${result}()`,
    mma: (l, r, acc) => `subgroupMatrixMultiplyAccumulate(${l}, ${r}, ${acc})`,
    mmaN: (l, r, acc) => `subgroupMatrixMultiplyAccumulate(${l}, ${r}, ${acc})`,
    store: (off, value, stride) => `subgroupMatrixStore(&nnS, ${off}, ${value}, false, ${stride})`,
  };
}

/** Result elements each emulated lane owns: element e of the M × N tile belongs to lane e % 32. */
export function sgmEmuElementsPerLane(c: Pick<SgmConfig, 'M' | 'N'>): number {
  return Math.ceil((c.M * c.N) / EMU_LANES);
}

const EMU_LANES = 32;

export function emulatedOps(c: SgmConfig): Ops {
  const { M, N, K } = c;
  const E = sgmEmuElementsPerLane(c);
  const guard = M * N % EMU_LANES ? `if (e >= ${M * N}u) { continue; }\n    ` : '';
  return {
    prelude: /* wgsl */ `
// Emulated subgroup matrices (correctness only). Left/right "tiles" are lazy views (offset, stride) read at multiply
// time, which matches the native load semantics here because nnA and the weights do not change in between. Like a real
// subgroup, the ${EMU_LANES} lanes share the result tile: lane j owns elements j, j + ${EMU_LANES}, ... (${E} each, f32).
// An earlier form had every lane compute and store the whole tile in a by-value private array: on Apple (Chrome 157)
// that ran for minutes per dispatch and once took the machine down (CLAUDE.md Traps).
alias EmuLeft = vec2u;  // M rows × K cols, row-major view of nnA
alias EmuRight = vec2u; // K rows × N cols, column-major view of the weights: element (k, n) at off + n * stride + k
alias EmuResult = array<f32, ${E}>; // this lane's elements of the M × N row-major tile
var<private> nnEmuLane: u32;
fn emuMma(l: EmuLeft, r: EmuRight, acc: EmuResult) -> EmuResult {
  var o = acc;
  for (var j = 0u; j < ${E}u; j++) {
    let e = nnEmuLane + j * ${EMU_LANES}u;
    ${guard}let i = e / ${N}u;
    let n = e % ${N}u;
    var s = o[j];
    for (var k = 0u; k < ${K}u; k++) { s += f32(nnA[l.x + i * l.y + k]) * f32(nnSgmW[r.x + n * r.y + k]); }
    o[j] = s;
  }
  return o;
}
fn emuMmaN(l: EmuLeft, r: EmuRight, acc: EmuResult) -> EmuResult {
  var o = acc;
  for (var j = 0u; j < ${E}u; j++) {
    let e = nnEmuLane + j * ${EMU_LANES}u;
    ${guard}let i = e / ${N}u;
    let n = e % ${N}u;
    var s = o[j];
    for (var k = 0u; k < ${K}u; k++) { s += f32(nnA[l.x + i * l.y + k]) * f32(nnSgmW[r.x + k * r.y + n]); }
    o[j] = s;
  }
  return o;
}
fn emuStore(off: u32, v: EmuResult, stride: u32) {
  for (var j = 0u; j < ${E}u; j++) {
    let e = nnEmuLane + j * ${EMU_LANES}u;
    ${guard}nnS[off + (e / ${N}u) * stride + e % ${N}u] = TR(v[j]);
  }
}
`,
    left: 'EmuLeft',
    right: 'EmuRight',
    result: 'EmuResult',
    loadLeft: (off, stride) => `vec2u(${off}, ${stride})`,
    loadRight: (off, stride) => `vec2u(${off}, ${stride})`,
    loadRightN: (off, stride) => `vec2u(${off}, ${stride})`,
    zero: () => 'EmuResult()',
    mma: (l, r, acc) => `emuMma(${l}, ${r}, ${acc})`,
    mmaN: (l, r, acc) => `emuMmaN(${l}, ${r}, ${acc})`,
    store: (off, value, stride) => `emuStore(${off}, ${value}, ${stride})`,
  };
}

export function directives(o: SgmOptions): string[] {
  const d: string[] = [];
  const usesSubgroupId = o.subgroups > 1 && !o.offsetsFromLocalIndex && !o.emulate;
  if (!o.emulate) d.push(`enable ${o.enableName ?? 'chromium_experimental_subgroup_matrix'};`);
  if (usesSubgroupId) d.push('enable subgroups;');
  if (!o.emulate && o.config.resultComponentType === 'f16' && o.config.componentType !== 'f16') d.push('enable f16;');
  if (o.uniformityDiagnosticOff && !o.emulate) d.push(`diagnostic(off, ${o.uniformityDiagnosticOff});`);
  return d;
}

export function mlpOptions(o: SgmOptions): MlpWgslOptions {
  // T is the operand type; emulation runs in f32.
  return { shape: o.shape, precision: o.emulate ? 'f32' : o.config.componentType };
}

/** Inference kernel `nnInferSgm`. Hooks as for the portable arm (`nnInferLoad`, `nnInferStore`, group 1). */
export function inferSgmWgsl(o: SgmOptions, hooks: string): string {
  validateSgm(o);
  if (o.layout === 'blocked') return inferSgmBlockedWgsl(o, hooks);
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
      const pass = `  for (var i = lid; i < ${R * layerOut(shape, l)}u; i += ${wg}u) {
    let at = (i / ${layerOut(shape, l)}u) * ${S}u + i % ${layerOut(shape, l)}u;
    nnA[at] = T(max(f32(nnS[at]), 0.0));
  }
  workgroupBarrier();`;
      const times = { relu: 1, skip: 0, double: 2 }[o.roundTrip ?? 'relu'];
      for (let t = 0; t < times; t++) layers.push(pass);
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
  let sg = ${sgIndex};${o.emulate ? `
  nnEmuLane = lid % ${EMU_LANES}u;` : ''}
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

/**
 * `count` work items over the workgroup's lanes with a CONSTANT trip count (`i` = item): a `for (var i = lid; ...)` loop
 * gets Tint's 64-bit loop counter, which cost the blocked kernel's ReLU passes ~4x at 64 records on NVIDIA (N2r-c, as P-W1).
 */
export function laneLoop(count: number, wg: number, body: string): string {
  const trips = Math.ceil(count / wg);
  const guarded = count % wg ? `if (i < ${count}u) {
    ${body}
    }` : body;
  return `  for (var j = 0u; j < ${trips}u; j++) {
    let i = lid + j * ${wg}u;
    ${guarded}
  }`;
}

/** Inference kernel `nnInferSgm` in the 'blocked' layout (see `SgmOptions.layout`). Same bindings and hooks as 'row'. */
function inferSgmBlockedWgsl(o: SgmOptions, hooks: string): string {
  if (o.emulate) throw new Error("sgm layout 'blocked' has no emulated form");
  if (o.roundTrip === 'double') throw new Error("sgm layout 'blocked' has no roundTrip 'double' probe");
  const { shape, config: c } = o;
  const ops = nativeOps(c);
  const S = rowStride(shape) + sgmPad(o);
  const R = o.records;
  const wg = sgmWorkgroupSize(o);
  const rowBlocks = R / c.M / o.subgroups;
  const TR = c.resultComponentType;
  const direct = TR === c.componentType;
  const sgIndex = o.subgroups === 1 ? '0u' : o.offsetsFromLocalIndex ? `lid / ${sgmLanes(o)}u` : 'sgid';
  const count = layerCount(shape);
  const act = (l: number) => (direct ? `nnA${l % 2}` : 'nnA');
  const reluRun = o.reluRun ?? 4;
  if (![1, 2, 4, 8].includes(reluRun)) throw new Error(`sgm: reluRun must be 1, 2, 4 or 8, got ${reluRun}`);

  const layers: string[] = [];
  for (let l = 0; l < count; l++) {
    const nIn = layerIn(shape, l);
    const nOut = sgmLayerOut(shape, c, l);
    const off = sgmLayerOffset(shape, c, l);
    const kBlocks = nIn / c.K;
    const nBlocks = nOut / c.N;
    const src = act(l);
    const dst = direct ? act(l + 1) : 'nnS';
    const rights: string[] = [];
    for (let nb = 0; nb < nBlocks; nb++)
      for (let kb = 0; kb < kBlocks; kb++)
        rights.push(`    let w${nb}_${kb} = subgroupMatrixLoad<${ops.right}>(&nnSgmW, ${off + nb * c.N * nIn + kb * c.K}u, true, ${nIn}u);`);
    const lefts = Array.from({ length: kBlocks }, (_, kb) => `      let a${kb} = subgroupMatrixLoad<${ops.left}>(&${src}, rowBase + ${kb * c.K}u, false, ${S}u);`);
    const body: string[] = [];
    for (let nb = 0; nb < nBlocks; nb++) {
      body.push(`      var z${nb} = ${ops.zero()};`);
      for (let kb = 0; kb < kBlocks; kb++) body.push(`      z${nb} = subgroupMatrixMultiplyAccumulate(a${kb}, w${nb}_${kb}, z${nb});`);
      body.push(`      subgroupMatrixStore(&${dst}, rowBase + ${nb * c.N}u, z${nb}, false, ${S}u);`);
    }
    const width = layerOut(shape, l);
    const relu = l >= shape.hidden || o.roundTrip === 'skip'
      ? ''
      : direct
        ? `
${laneLoop((R * width) / reluRun, wg, `let at = (i / ${width / reluRun}u) * ${S}u + (i % ${width / reluRun}u) * ${reluRun}u;
${Array.from({ length: reluRun }, (_, e) => `    ${dst}[at + ${e}u] = max(${dst}[at + ${e}u], T(0.0));`).join('\n')}`)}
  workgroupBarrier();`
        : `
${laneLoop(R * width, wg, `let at = (i / ${width}u) * ${S}u + i % ${width}u;
    nnA[at] = T(max(f32(nnS[at]), 0.0));`)}
  workgroupBarrier();`;
    layers.push(`  // Layer ${l}: ${nIn} → ${width}${nOut !== width ? ` (padded to ${nOut})` : ''}; ${nBlocks * kBlocks} right tiles held.
  {
${rights.join('\n')}
    for (var mbi = 0u; mbi < ${rowBlocks}u; mbi++) {
      let rowBase = (sg * ${rowBlocks}u + mbi) * ${c.M * S}u;
${lefts.join('\n')}
${body.join('\n')}
    }
  }
  workgroupBarrier();${relu}`);
  }
  const out = direct ? act(count) : 'nnS';
  const builtins = [
    '@builtin(workgroup_id) wid: vec3u',
    '@builtin(num_workgroups) nwg: vec3u',
    '@builtin(local_invocation_index) lid: u32',
    ...(sgIndex === 'sgid' ? ['@builtin(subgroup_id) sgid: u32'] : []),
  ];
  const buffers = direct
    ? `var<workgroup> nnA0: array<T, ${R * S}>;
var<workgroup> nnA1: array<T, ${R * S}>;`
    : `var<workgroup> nnA: array<T, ${R * S}>;
var<workgroup> nnS: array<TR, ${R * S}>;`;

  return `${header(mlpOptions(o), directives(o))}
${encodingWgsl()}
alias TR = ${TR};
@group(0) @binding(0) var<storage, read> nnSgmW: array<T>;
@group(0) @binding(1) var<storage, read> nnState: NnState;

${buffers}
${hooks}

// Subgroup matrix ${c.M}×${c.N}×${c.K} ${c.componentType} → ${TR}, layout 'blocked'; ${R} records, ${o.subgroups} subgroup(s) per workgroup, row stride ${S}.
@compute @workgroup_size(${wg})
fn nnInferSgm(${builtins.join(', ')}) {
  let base = (wid.x + wid.y * nwg.x) * ${R}u;
  let sg = ${sgIndex};
  for (var j = 0u; j < ${Math.ceil(R / wg)}u; j++) {
    let i = lid + j * ${wg}u;${R % wg ? `
    if (i >= ${R}u) { break; }` : ''}
    let r = base + i;
    var x: array<f32, NN_IN>;
    ${o.inputProbe ? 'if (r > 0xfffffff0u) { x = nnInferLoad(r); } x[r % NN_IN] = 1.0; // keeps the hook bindings alive' : 'if (r < nnState.count) { x = nnInferLoad(r); }'}
    ${o.inputProbe === 'none' ? `if (r > 0xfffffff0u) { ${act(0)}[i] = T(x[0]); } // probe: no input stores` : `for (var k = 0u; k < NN_IN; k++) { ${act(0)}[i * ${S}u + k] = T(x[k]); }`}
  }
  workgroupBarrier();
${layers.join('\n')}
  for (var j = 0u; j < ${Math.ceil(R / wg)}u; j++) {
    let i = lid + j * ${wg}u;${R % wg ? `
    if (i >= ${R}u) { break; }` : ''}
    let r = base + i;
    if (r < nnState.count) { nnInferStore(r, vec3f(f32(${out}[i * ${S}u]), f32(${out}[i * ${S}u + 1u]), f32(${out}[i * ${S}u + 2u]))); }
  }
}
`;
}
