/**
 * WGSL generator for the portable MLP arms (`wgsl-f32`, `wgsl-f16`), docs/design/nrc.md §3.
 *
 * Kernels (all @workgroup_size(64), WGSL prefix `nn`):
 *  - `nnPrepare`        one thread: clamps the GPU record count to capacity, writes indirect args, and (train only)
 *                       advances the Adam step state. No CPU in the loop.
 *  - `nnInfer`          one thread per record: load (caller hook) → layers → store (caller hook).
 *  - `nnForwardBackward` one thread per record: forward, storing every activation a_l, loss gradient, backward,
 *                       storing every delta (dL/dz of each layer's output).
 *  - `nnWeightGradient` one thread per weight: dW = Σ_r delta_l[r] ⊗ a_l[r] / count. Deterministic, no atomics
 *                       (WGSL has no float atomics; the partial-sum and fixed-point-atomic forms are N2 A/B arms).
 *  - `nnAdam`           one thread per weight: Adam on f32 master weights, EMA, writes the `T` copies.
 *  - `nnSync`           one thread per weight: master → `T` copies and EMA (after uploading weights).
 *
 * Layer weights are staged in workgroup memory one layer at a time as vec4 rows (f32: 16 KiB for 64×64, the WebGPU
 * default limit; f16: 8 KiB). Activations are vec4<T>; sums accumulate in f32, and the f16 arm sums each vec4 dot
 * product in f16 first (the Denoiser's "f16 sums per chunk" trick, docs/specs/runtimes.md there).
 *
 * Caller hooks (group 1, the caller's own bindings):
 *  - inference: `fn nnInferLoad(r: u32) -> array<f32, NN_IN>` and `fn nnInferStore(r: u32, y: vec3f)`
 *  - training:  `fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f` (fills x, returns target)
 */

import { layerCount, layerIn, layerOffset, layerOut, validateShape, weightCount, type MlpShape } from './config.ts';
import { encodingWgsl } from './encoding.ts';

export type Precision = 'f32' | 'f16';

export const NN_WORKGROUP = 64;

export interface MlpWgslOptions {
  shape: MlpShape;
  precision: Precision;
}

/** Row stride (in scalars) of the activation/delta buffers: the widest layer input. */
export function rowStride(shape: MlpShape): number {
  return Math.max(shape.inputs, shape.width);
}

function layerHelpers(o: MlpWgslOptions): string {
  const { shape } = o;
  const maxLayer = Math.max(...Array.from({ length: layerCount(shape) }, (_, l) => layerIn(shape, l) * layerOut(shape, l)));
  return /* wgsl */ `
var<workgroup> nnLayer: array<T4, ${maxLayer / 4}>;

// Copies \`n4\` vec4s of weights starting at vec4 \`off4\` into workgroup memory. Uniform control flow only.
fn nnStage(off4: u32, n4: u32, lid: u32) {
  workgroupBarrier();
  for (var k = lid; k < n4; k += NN_WG) { nnLayer[k] = nnWeights[off4 + k]; }
  workgroupBarrier();
}

// Row \`o\` of the staged layer (nIn4 vec4s per row) times \`a\`.
fn nnDotRow(o: u32, nIn4: u32, a: ptr<function, array<T4, NN_S4>>) -> f32 {
  var s = 0.0;
  for (var k = 0u; k < nIn4; k++) { s += f32(dot(nnLayer[o * nIn4 + k], (*a)[k])); }
  return s;
}
`;
}

/** Shared module header. `directives` (enable / diagnostic lines) must come first in WGSL, so callers pass them here. */
export function header(o: MlpWgslOptions, directives: string[] = []): string {
  const { shape, precision } = o;
  validateShape(shape);
  const T = precision;
  const S = rowStride(shape);
  const lines = [...(precision === 'f16' ? ['enable f16;'] : []), ...directives];
  return /* wgsl */ `${lines.join('\n')}${lines.length ? '\n' : ''}
// @pathtracer/mlp: ${shape.inputs} → ${shape.hidden}×${shape.width} → ${shape.outputs}, ${precision}
alias T = ${T};
alias T4 = vec4<${T}>;
const NN_IN = ${shape.inputs}u;
const NN_W = ${shape.width}u;
const NN_OUT = ${shape.outputs}u;
const NN_HIDDEN = ${shape.hidden}u;
const NN_WEIGHTS = ${weightCount(shape)}u;
const NN_S4 = ${S / 4}u; // vec4s per activation/delta row
const NN_WG = ${NN_WORKGROUP}u;

struct NnState {
  count: u32,     // records this pass (clamped to capacity)
  step: u32,      // Adam steps taken
  beta1t: f32,    // beta1^step
  beta2t: f32,    // beta2^step
}

struct NnParams {
  capacity: u32,
  countWord: u32,
  train: u32,     // 1: advance the Adam step in nnPrepare
  useEma: u32,    // 1: inference weights are the EMA
  learningRate: f32,
  beta1: f32,
  beta2: f32,
  epsilon: f32,
  emaDecay: f32,
  lossEpsilon: f32,
  groupRecords: u32, // records per workgroup of the kernel nnPrepare sizes
  _pad1: f32,
}

`;
}

/** Forward layers into `a` (input already in `a`), optionally storing each a_l row. Returns y. */
function forwardBody(shape: MlpShape, store: boolean): string {
  const lines: string[] = [];
  for (let l = 0; l < layerCount(shape); l++) {
    const nIn4 = layerIn(shape, l) / 4;
    const nOut = layerOut(shape, l);
    const off4 = layerOffset(shape, l) / 4;
    if (store) lines.push(`  if (valid) { for (var k = 0u; k < ${nIn4}u; k++) { nnActs[(${l}u * params.capacity + r) * NN_S4 + k] = a[k]; } }`);
    lines.push(`  nnStage(${off4}u, ${(layerIn(shape, l) * nOut) / 4}u, lid);`);
    if (l < shape.hidden) {
      lines.push(`  {
    var b: array<T4, NN_S4>;
    for (var o = 0u; o < ${nOut}u; o++) { b[o / 4u][o % 4u] = T(max(nnDotRow(o, ${nIn4}u, &a), 0.0)); }
    a = b;
  }`);
    } else {
      lines.push(`  var y = vec3f(0.0);
  for (var o = 0u; o < ${Math.min(nOut, 3)}u; o++) { y[o] = nnDotRow(o, ${nIn4}u, &a); }`);
    }
  }
  return lines.join('\n');
}

function loadInput(): string {
  return `  var a: array<T4, NN_S4>;
  for (var k = 0u; k < NN_IN / 4u; k++) { a[k] = T4(vec4f(x[4u * k], x[4u * k + 1u], x[4u * k + 2u], x[4u * k + 3u])); }`;
}

export function inferWgsl(o: MlpWgslOptions, hooks: string): string {
  const { shape } = o;
  return `${header(o)}
${encodingWgsl()}
@group(0) @binding(0) var<storage, read> nnWeights: array<T4>;
@group(0) @binding(1) var<storage, read> nnState: NnState;
@group(0) @binding(2) var<uniform> params: NnParams;
${layerHelpers(o)}
${hooks}

@compute @workgroup_size(${NN_WORKGROUP})
fn nnInfer(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(num_workgroups) nwg: vec3u) {
  let r = gid.x + gid.y * nwg.x * NN_WG;
  let valid = r < nnState.count;
  var x: array<f32, NN_IN>;
  if (valid) { x = nnInferLoad(r); }
${loadInput()}
${forwardBody(shape, false)}
  if (valid) { nnInferStore(r, y); }
}
`;
}

export function trainWgsl(o: MlpWgslOptions, hooks: string): string {
  const { shape } = o;
  if (shape.outputs !== 3) throw new Error('trainWgsl: the relative-L2 luminance loss needs 3 outputs');
  const back: string[] = [];
  // Backward: d holds dL/dz of layer l's output (length layerOut(l)); store it, then push through W_l and the ReLU of a_l.
  for (let l = shape.hidden; l >= 0; l--) {
    const nIn = layerIn(shape, l);
    const nOut = layerOut(shape, l);
    back.push(`  if (valid) { for (var k = 0u; k < ${Math.ceil(nOut / 4)}u; k++) { nnDeltas[(${l}u * params.capacity + r) * NN_S4 + k] = T4(d[k]); } }`);
    if (l === 0) break;
    back.push(`  nnStage(${layerOffset(shape, l) / 4}u, ${(nIn * nOut) / 4}u, lid);
  {
    var nd: array<vec4f, NN_S4>;
    for (var i = 0u; i < ${nIn}u; i++) {
      let act = nnActs[(${l}u * params.capacity + r) * NN_S4 + i / 4u][i % 4u];
      var s = 0.0;
      for (var oo = 0u; oo < ${nOut}u; oo++) { s += f32(nnLayer[(oo * ${nIn}u + i) / 4u][i % 4u]) * d[oo / 4u][oo % 4u]; }
      nd[i / 4u][i % 4u] = select(0.0, s, act > T(0.0));
    }
    d = nd;
  }`);
  }
  return `${header(o)}
${encodingWgsl()}
@group(0) @binding(0) var<storage, read> nnWeights: array<T4>;
@group(0) @binding(1) var<storage, read> nnState: NnState;
@group(0) @binding(2) var<uniform> params: NnParams;
@group(0) @binding(3) var<storage, read_write> nnActs: array<T4>;
@group(0) @binding(4) var<storage, read_write> nnDeltas: array<T4>;
@group(0) @binding(5) var<storage, read_write> nnLoss: array<f32>;
${layerHelpers(o)}
${hooks}

@compute @workgroup_size(${NN_WORKGROUP})
fn nnForwardBackward(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(num_workgroups) nwg: vec3u) {
  let r = gid.x + gid.y * nwg.x * NN_WG;
  let valid = r < nnState.count;
  var x: array<f32, NN_IN>;
  var t = vec3f(0.0);
  if (valid) { t = nnTrainLoad(r, &x); }
${loadInput()}
${forwardBody(shape, true)}
  // Relative L2 on luminance, denominator held constant. Not divided by the batch: nnWeightGradient divides by count,
  // which keeps f16 deltas out of the subnormal range.
  let lum = dot(vec3f(0.299, 0.587, 0.114), max(y, vec3f(0.0)));
  let denom = lum * lum + params.lossEpsilon;
  var d: array<vec4f, NN_S4>;
  if (valid) {
    let e = y - t;
    d[0] = vec4f(2.0 * e / denom, 0.0);
    nnLoss[r] = dot(e, e) / denom;
  }
${back.join('\n')}
}
`;
}

function optimizerHeader(o: MlpWgslOptions): string {
  return `${header(o)}
@group(0) @binding(0) var<storage, read_write> nnState: NnState;
@group(0) @binding(1) var<uniform> params: NnParams;
`;
}

export function prepareWgsl(o: MlpWgslOptions): string {
  return `${optimizerHeader(o)}
@group(0) @binding(2) var<storage, read> nnCounts: array<u32>;
@group(0) @binding(3) var<storage, read_write> nnArgs: array<u32>;

@compute @workgroup_size(1)
fn nnPrepare() {
  let count = min(nnCounts[params.countWord], params.capacity);
  nnState.count = count;
  // 2D grid past 65535 workgroups.
  let per = max(params.groupRecords, 1u);
  let groups = (count + per - 1u) / per;
  let gx = min(groups, 65535u);
  nnArgs[0] = gx;
  nnArgs[1] = select(1u, (groups + gx - 1u) / max(gx, 1u), gx > 0u);
  nnArgs[2] = 1u;
  if (params.train == 1u && count > 0u) {
    nnState.step += 1u;
    nnState.beta1t *= params.beta1;
    nnState.beta2t *= params.beta2;
  }
}
`;
}

export function weightGradientWgsl(o: MlpWgslOptions): string {
  const { shape } = o;
  const cases: string[] = [];
  for (let l = 0; l < layerCount(shape); l++) {
    cases.push(`  ${l ? 'else ' : ''}if (w < ${layerOffset(shape, l + 1)}u) { l = ${l}u; nIn = ${layerIn(shape, l)}u; idx = w - ${layerOffset(shape, l)}u; }`);
  }
  return `${optimizerHeader(o)}
@group(0) @binding(2) var<storage, read> nnActs: array<T4>;
@group(0) @binding(3) var<storage, read> nnDeltas: array<T4>;
@group(0) @binding(4) var<storage, read_write> nnGrad: array<f32>;

@compute @workgroup_size(${NN_WORKGROUP})
fn nnWeightGradient(@builtin(global_invocation_id) gid: vec3u) {
  let w = gid.x;
  if (w >= NN_WEIGHTS) { return; }
  var l = 0u; var nIn = 0u; var idx = 0u;
${cases.join('\n')}
  let o = idx / nIn;
  let i = idx % nIn;
  let count = nnState.count;
  var s = 0.0;
  for (var r = 0u; r < count; r++) {
    let row = (l * params.capacity + r) * NN_S4;
    s += f32(nnDeltas[row + o / 4u][o % 4u]) * f32(nnActs[row + i / 4u][i % 4u]);
  }
  nnGrad[w] = select(0.0, s / f32(count), count > 0u);
}
`;
}

export function adamWgsl(o: MlpWgslOptions): string {
  return `${optimizerHeader(o)}
@group(0) @binding(2) var<storage, read> nnGrad: array<f32>;
@group(0) @binding(3) var<storage, read_write> nnMaster: array<f32>;
@group(0) @binding(4) var<storage, read_write> nnMoments: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> nnEma: array<f32>;
@group(0) @binding(6) var<storage, read_write> nnTrainWeights: array<T>;
@group(0) @binding(7) var<storage, read_write> nnUseWeights: array<T>;

fn nnFinite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }

@compute @workgroup_size(${NN_WORKGROUP})
fn nnAdam(@builtin(global_invocation_id) gid: vec3u) {
  let w = gid.x;
  if (w >= NN_WEIGHTS || nnState.count == 0u) { return; }
  var g = nnGrad[w];
  if (!nnFinite(g)) { g = 0.0; }
  var mv = nnMoments[w];
  mv = vec2f(params.beta1, params.beta2) * mv + (vec2f(1.0) - vec2f(params.beta1, params.beta2)) * vec2f(g, g * g);
  nnMoments[w] = mv;
  let mHat = mv.x / (1.0 - nnState.beta1t);
  let vHat = mv.y / (1.0 - nnState.beta2t);
  let weight = nnMaster[w] - params.learningRate * mHat / (sqrt(vHat) + params.epsilon);
  nnMaster[w] = weight;
  let ema = params.emaDecay * nnEma[w] + (1.0 - params.emaDecay) * weight;
  nnEma[w] = ema;
  nnTrainWeights[w] = T(weight);
  nnUseWeights[w] = T(select(weight, ema, params.useEma == 1u));
}

@compute @workgroup_size(${NN_WORKGROUP})
fn nnSync(@builtin(global_invocation_id) gid: vec3u) {
  let w = gid.x;
  if (w >= NN_WEIGHTS) { return; }
  let weight = nnMaster[w];
  nnEma[w] = weight;
  nnMoments[w] = vec2f(0.0);
  nnTrainWeights[w] = T(weight);
  nnUseWeights[w] = T(weight);
}
`;
}

/**
 * Default hooks over plain buffers (tests, microbenchmarks): group 1 binding 0 = inputs (N × NN_IN f32),
 * 1 = targets (N × 3 f32), 2 = outputs (N × 3 f32).
 */
export const BUFFER_HOOKS_INFER = /* wgsl */ `
@group(1) @binding(0) var<storage, read> hookInputs: array<f32>;
@group(1) @binding(2) var<storage, read_write> hookOutputs: array<f32>;
fn nnInferLoad(r: u32) -> array<f32, NN_IN> {
  var x: array<f32, NN_IN>;
  for (var k = 0u; k < NN_IN; k++) { x[k] = hookInputs[r * NN_IN + k]; }
  return x;
}
fn nnInferStore(r: u32, y: vec3f) {
  hookOutputs[r * 3u] = y.x; hookOutputs[r * 3u + 1u] = y.y; hookOutputs[r * 3u + 2u] = y.z;
}
`;

export const BUFFER_HOOKS_TRAIN = /* wgsl */ `
@group(1) @binding(0) var<storage, read> hookInputs: array<f32>;
@group(1) @binding(1) var<storage, read> hookTargets: array<f32>;
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
  for (var k = 0u; k < NN_IN; k++) { (*x)[k] = hookInputs[r * NN_IN + k]; }
  return vec3f(hookTargets[r * 3u], hookTargets[r * 3u + 1u], hookTargets[r * 3u + 2u]);
}
`;
