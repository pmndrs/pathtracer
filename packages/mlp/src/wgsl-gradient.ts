/**
 * Tiled weight-gradient arm (N1.2), the alternative to `nnWeightGradient` (one thread per weight, serial over all
 * records). dW_l[o][i] = Σ_r delta_l[r][o] · a_l[r][i] / count is a batched outer product, so:
 *
 *  - `nnGradientTile`: workgroup (split, layer), 256 threads as a 16×16 grid, each owning a 4×4 register block of
 *    the layer's (≤ 64 × 64) dW tile. The workgroup sums its split of the records (`ceil(count / splits)` each) in
 *    chunks of `chunk` records staged in workgroup memory (deltas and activations as f32 vec4 rows), accumulating in
 *    f32, and writes a partial dW per split.
 *  - `nnGradientReduce`: one thread per weight sums the `splits` partials in a fixed order and divides by count.
 *
 * Deterministic (fixed split and chunk order, no atomics; WGSL has no float atomics and no "last workgroup" handoff,
 * hence the second dispatch). The record loop bound is made workgroup-uniform with `workgroupUniformLoad`.
 * Limits: layer input and output widths must be ≤ 64 (one tile per layer); `chunk` is a multiple of 16.
 */

import { layerCount, layerIn, layerOffset, layerOut, weightCount } from './config.ts';
import { header, type MlpWgslOptions } from './wgsl.ts';

export type GradientKind = 'perWeight' | 'tiled' | 'sgm';

export interface GradientOptions {
  /** Record ranges (workgroups per layer) the batch is split into. */
  splits: number;
  /** Records staged in workgroup memory per iteration. */
  chunk: number;
}

export const DEFAULT_GRADIENT: GradientOptions = { splits: 64, chunk: 32 };

const TILE_THREADS = 256;
const TILE = 64;

export function validateGradientOptions(o: MlpWgslOptions, g: GradientOptions): void {
  const { shape } = o;
  for (let l = 0; l < layerCount(shape); l++) {
    if (layerIn(shape, l) > TILE || layerOut(shape, l) > TILE) throw new Error(`tiled gradient: layer ${l} is ${layerOut(shape, l)}×${layerIn(shape, l)}, the tile is ${TILE}×${TILE}`);
  }
  if (!Number.isInteger(g.splits) || g.splits < 1) throw new Error(`tiled gradient: splits must be a positive integer, got ${g.splits}`);
  if (!Number.isInteger(g.chunk) || g.chunk < 16 || g.chunk % 16) throw new Error(`tiled gradient: chunk must be a positive multiple of 16, got ${g.chunk}`);
}

/** Bytes of the partial-gradient buffer (f32 × splits × weights). */
export function gradientPartialBytes(o: MlpWgslOptions, g: GradientOptions): number {
  return g.splits * weightCount(o.shape) * 4;
}

export function gradientTileWgsl(o: MlpWgslOptions, g: GradientOptions): string {
  validateGradientOptions(o, g);
  const { shape } = o;
  const layers = layerCount(shape);
  const offsets = Array.from({ length: layers }, (_, l) => `${layerOffset(shape, l)}u`).join(', ');
  const rowVec4 = TILE / 4;
  return `${header(o)}
const NN_SPLITS = ${g.splits}u;
const NN_CHUNK = ${g.chunk}u;
const NN_LAYER_OFFSET = array<u32, ${layers}>(${offsets});

@group(0) @binding(0) var<storage, read> nnState: NnState;
@group(0) @binding(1) var<uniform> params: NnParams;
@group(0) @binding(2) var<storage, read> nnActs: array<T4>;
@group(0) @binding(3) var<storage, read> nnDeltas: array<T4>;
@group(0) @binding(4) var<storage, read_write> nnPartials: array<vec4f>;

var<workgroup> nnCountShared: u32;
var<workgroup> nnSDelta: array<vec4f, ${g.chunk * rowVec4}>;
var<workgroup> nnSAct: array<vec4f, ${g.chunk * rowVec4}>;

@compute @workgroup_size(${TILE_THREADS})
fn nnGradientTile(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) lid: u32) {
  let split = wid.x;
  let l = wid.y;
  if (lid == 0u) { nnCountShared = nnState.count; }
  let count = workgroupUniformLoad(&nnCountShared);
  let per = (count + NN_SPLITS - 1u) / NN_SPLITS;
  let begin = min(split * per, count);
  let end = min(begin + per, count);
  let chunks = (end - begin + NN_CHUNK - 1u) / NN_CHUNK;
  let nIn = select(NN_W, NN_IN, l == 0u);
  let nOut = select(NN_W, NN_OUT, l == NN_HIDDEN);
  let tx = lid % ${rowVec4}u;  // columns (inputs) 4·tx … 4·tx+3
  let ty = lid / ${rowVec4}u;  // rows (outputs) 4·ty … 4·ty+3
  var acc0 = vec4f(0.0);
  var acc1 = vec4f(0.0);
  var acc2 = vec4f(0.0);
  var acc3 = vec4f(0.0);
  for (var c = 0u; c < chunks; c++) {
    let base = begin + c * NN_CHUNK;
    workgroupBarrier();
    for (var e = lid; e < NN_CHUNK * ${rowVec4}u; e += ${TILE_THREADS}u) {
      let rr = e / ${rowVec4}u;
      let k = e % ${rowVec4}u;
      let r = base + rr;
      var dv = vec4f(0.0);
      var av = vec4f(0.0);
      if (r < end && k < NN_S4) {
        let row = (l * params.capacity + r) * NN_S4 + k;
        dv = vec4f(nnDeltas[row]);
        av = vec4f(nnActs[row]);
      }
      nnSDelta[e] = dv;
      nnSAct[e] = av;
    }
    workgroupBarrier();
    for (var rr = 0u; rr < NN_CHUNK; rr++) {
      let dv = nnSDelta[rr * ${rowVec4}u + ty];
      let av = nnSAct[rr * ${rowVec4}u + tx];
      acc0 += dv.x * av;
      acc1 += dv.y * av;
      acc2 += dv.z * av;
      acc3 += dv.w * av;
    }
  }
  // Partial dW of this split: layer-major, row-major [out][in], vec4 along the input.
  if (4u * tx < nIn) {
    let base4 = (split * NN_WEIGHTS + NN_LAYER_OFFSET[l]) / 4u + tx;
    let o = 4u * ty;
    if (o + 0u < nOut) { nnPartials[base4 + (o + 0u) * nIn / 4u] = acc0; }
    if (o + 1u < nOut) { nnPartials[base4 + (o + 1u) * nIn / 4u] = acc1; }
    if (o + 2u < nOut) { nnPartials[base4 + (o + 2u) * nIn / 4u] = acc2; }
    if (o + 3u < nOut) { nnPartials[base4 + (o + 3u) * nIn / 4u] = acc3; }
  }
}
`;
}

export function gradientReduceWgsl(o: MlpWgslOptions, g: GradientOptions): string {
  validateGradientOptions(o, g);
  return `${header(o)}
const NN_SPLITS = ${g.splits}u;

@group(0) @binding(0) var<storage, read> nnState: NnState;
@group(0) @binding(2) var<storage, read> nnPartials: array<f32>;
@group(0) @binding(3) var<storage, read_write> nnGrad: array<f32>;

@compute @workgroup_size(64)
fn nnGradientReduce(@builtin(global_invocation_id) gid: vec3u) {
  let w = gid.x;
  if (w >= NN_WEIGHTS) { return; }
  var s = 0.0;
  for (var k = 0u; k < NN_SPLITS; k++) { s += nnPartials[k * NN_WEIGHTS + w]; }
  let count = nnState.count;
  nnGrad[w] = select(0.0, s / f32(count), count > 0u);
}
`;
}
