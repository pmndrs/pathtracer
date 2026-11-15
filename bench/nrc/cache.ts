/**
 * Neural radiance cache on main's megakernel path tracer (research track N, TASKS N6a, docs/design/nrc-plan.md §5).
 *
 * Two hooked kernels share one hook text (`NRC_HOOK_WGSL`), both built from the plain kernel's own options so the only
 * difference from plain path tracing is the hook:
 *
 *  - **render** (the image): every path terminates into the cache at its vertex `queryDepth` (after that vertex's
 *    emission and NEE), appends a query record and adds 0. After the pass, inference over the queries and a resolve pass
 *    add `throughput × max(prediction, 0)` into the render kernel's accumulator (one query per pixel per pass, so the
 *    add is a plain store, no atomics).
 *  - **trainer** (training paths): a small film over the same camera. Targets come from deterministic replays instead
 *    of a path-end callback (the hooks expose neither the NEE term of a vertex nor the final escape, V84): with the same
 *    seed and sample index, run k terminates every path at vertex k with 0, so its accumulator holds the radiance up to
 *    and including vertex k's emission and NEE (R_k) and writes the vertex record; the tail run terminates at vertex M
 *    and writes a tail query. Then target_k = (R_M − R_k + β_M · cache(x_M)) / β_k is exactly what the render hook
 *    must return at vertex k (MIS included), with the cache's own prediction as the tail (self-training, N-I5).
 *    Cost: M + 1 trainer passes per frame (DIVERGES N-I9 in nrc-diverges.md).
 *
 * Per frame, everything is encoded into one command encoder with no readback: trainer runs → tail inference → targets
 * → `stepsPerFrame` training steps → render pass → inference → resolve.
 */

import { MlpEngine, nrcEncodingWgsl, initWeights, DEFAULT_SHAPE, DEFAULT_TRAIN, type Precision, type MlpEngineOptions } from '@pathtracer/mlp';
import { PtMegakernel, type PtHooksOptions, type PtLookAtCamera, type PtMegakernelOptions } from '@pmndrs/pathtracer';
import type { Caps } from '@pmndrs/raytracer';

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;
const COPY_SRC = GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM;

/** vec4 per record: (position, pixel as an exact f32), (direction θ/π φ/2π, normal θ/π φ/2π), (albedo, roughness), (throughput, valid). */
export const NRC_RECORD_VEC4 = 4;
const RECORD_BYTES = NRC_RECORD_VEC4 * 16;
const HOOK_SLOT = 256;

/** The hook both NRC kernels compile. mode 0: render query at depth >= stopDepth; mode 1: trainer record at stopDepth. */
export const NRC_HOOK_WGSL = /* wgsl */ `
struct NrcHookParams { mode: u32, stopDepth: u32, layerOffset: u32, capacity: u32 }
@group(3) @binding(0) var<uniform> nrcHook: NrcHookParams;
@group(3) @binding(1) var<storage, read_write> nrcRecords: array<vec4f>;
@group(3) @binding(2) var<storage, read_write> nrcCount: array<atomic<u32>>;

fn nrcSpherical(d: vec3f) -> vec2f {
    return vec2f(acos(clamp(d.y, -1.0, 1.0)) * 0.3183098861837907, atan2(d.z, d.x) * 0.15915494309189535 + 0.5);
}

fn nrcWrite(slot: u32, v: PtPathVertex) {
    let b = slot * ${NRC_RECORD_VEC4}u;
    nrcRecords[b] = vec4f(v.position, f32(v.pixel));
    nrcRecords[b + 1u] = vec4f(nrcSpherical(v.wo), nrcSpherical(v.normal));
    nrcRecords[b + 2u] = vec4f(v.albedo, 1.0);
    nrcRecords[b + 3u] = vec4f(v.throughput, 1.0);
}

fn ptOnVertex(v: PtPathVertex) -> u32 {
    if (v.depth < nrcHook.stopDepth) {
        return PT_HOOK_CONTINUE;
    }
    if (nrcHook.mode == 0u) {
        let slot = atomicAdd(&nrcCount[0], 1u);
        if (slot < nrcHook.capacity) {
            nrcWrite(slot, v);
        }
    } else {
        nrcWrite(nrcHook.layerOffset + v.pixel, v);
    }
    return PT_HOOK_TERMINATE;
}

fn ptTerminate(v: PtPathVertex) -> vec3f {
    return vec3f(0.0);
}
`;

/**
 * One-pass training (`training: 'inline'`, TASKS N6d): some of the render pass's own paths are the training paths (`trainSelect`:
 * whole 8 × 8 tiles or single pixels, `nrcTrainSlot`). Their vertices 0..M−1 are recorded per block slot, `ptOnRadiance` sums what they gather per
 * vertex index, and at vertex M they end in an ordinary query (its prediction goes into the image and into the targets).
 * target_k = (Σ_{j>k} gathered_j + β_M · cache(x_M)) / β_k: the same quantity as the replays, with no extra pass.
 */
export const NRC_HOOK_INLINE_WGSL = /* wgsl */ `
struct NrcHookParams { mode: u32, stopDepth: u32, layerOffset: u32, capacity: u32, trainDepth: u32, width: u32, block: u32, slotsX: u32, slots: u32, tileEvery: u32, tilesX: u32, _c: u32 }
@group(3) @binding(0) var<uniform> nrcHook: NrcHookParams;
@group(3) @binding(1) var<storage, read_write> nrcRecords: array<vec4f>;
@group(3) @binding(2) var<storage, read_write> nrcCount: array<atomic<u32>>;
@group(3) @binding(3) var<storage, read_write> nrcTrain: array<vec4f>;
@group(3) @binding(4) var<storage, read_write> nrcSums: array<vec4f>;

fn nrcSpherical(d: vec3f) -> vec2f {
    return vec2f(acos(clamp(d.y, -1.0, 1.0)) * 0.3183098861837907, atan2(d.z, d.x) * 0.15915494309189535 + 0.5);
}

// Training-path slot + 1 for this pixel and sample, 0 when the path is not a training path. Tiles (tileEvery > 0): every
// tileEvery-th 8 x 8 tile (the megakernel's workgroup) trains, rotating with the sample, so a workgroup is all training paths or
// none (no lane waits on a long training path; VkNRC picks whole subgroups for the same reason). Pixels: one pixel per
// block x block, rotating (main's updateBlock rule).
fn nrcTrainSlot(pixel: u32, sampleIndex: u32) -> u32 {
    // Rotate with the cache's own frame counter (nrcCount[1], bumped by the resolve pass), not the sample index: clearing the
    // image restarts the sample index, which would pin training to the same tiles while the camera moves.
    let frame = atomicLoad(&nrcCount[1]);
    let x = pixel % nrcHook.width;
    let y = pixel / nrcHook.width;
    if (nrcHook.tileEvery > 0u) {
        // Diagonal pattern: tile (tx, ty) trains when (tx + block * ty) % tileEvery == frame % tileEvery, so each row shifts by
        // block tiles (training tiles scatter instead of lining up in columns when tilesX is a multiple of tileEvery) and
        // every tile trains once per tileEvery frames. One selected tile per tileEvery-wide group of a row: a unique slot.
        let tx = x / 8u;
        let ty = y / 8u;
        if ((tx + nrcHook.block * ty) % nrcHook.tileEvery != frame % nrcHook.tileEvery) {
            return 0u;
        }
        let groupsX = (nrcHook.tilesX + nrcHook.tileEvery - 1u) / nrcHook.tileEvery;
        return (ty * groupsX + tx / nrcHook.tileEvery) * 64u + (y % 8u) * 8u + x % 8u + 1u;
    }
    let b = nrcHook.block;
    if ((x % b) + (y % b) * b != frame % (b * b)) {
        return 0u;
    }
    return (y / b) * nrcHook.slotsX + x / b + 1u;
}

// Appends a query; returns its index + 1 (0 when full). flag 1: the resolve pass adds it to the image; 2: training only.
fn nrcQuery(v: PtPathVertex, flag: f32) -> u32 {
    let q = atomicAdd(&nrcCount[0], 1u);
    if (q >= nrcHook.capacity) {
        return 0u;
    }
    let b = q * ${NRC_RECORD_VEC4}u;
    nrcRecords[b] = vec4f(v.position, f32(v.pixel));
    nrcRecords[b + 1u] = vec4f(nrcSpherical(v.wo), nrcSpherical(v.normal));
    nrcRecords[b + 2u] = vec4f(v.albedo, 1.0);
    nrcRecords[b + 3u] = vec4f(v.throughput, flag);
    return q + 1u;
}

fn ptOnVertex(v: PtPathVertex) -> u32 {
    let train = nrcTrainSlot(v.pixel, v.sampleIndex);
    if (train != 0u) {
        let slot = train - 1u;
        // The image gets the same estimate as every other pixel: a display query at the cache vertex (resolved into the
        // image), and the fix-up pass later removes what the path gathers beyond it. The path itself trains to vertex M.
        let display = nrcHook.stopDepth < nrcHook.trainDepth;
        if (v.depth < nrcHook.trainDepth) {
            let b = (v.depth * nrcHook.slots + slot) * ${NRC_RECORD_VEC4}u;
            nrcTrain[b] = vec4f(v.position, f32(v.pixel));
            nrcTrain[b + 1u] = vec4f(nrcSpherical(v.wo), nrcSpherical(v.normal));
            nrcTrain[b + 2u] = vec4f(v.albedo, 1.0);
            nrcTrain[b + 3u] = vec4f(v.throughput, 1.0);
            if (display && v.depth == nrcHook.stopDepth) {
                _ = nrcQuery(v, 1.0);
            }
            return PT_HOOK_CONTINUE;
        }
        let s0 = slot * (nrcHook.trainDepth + 1u);
        nrcSums[s0] = vec4f(nrcSums[s0].xyz, f32(nrcQuery(v, select(1.0, 2.0, display))));
        return PT_HOOK_TERMINATE;
    }
    if (v.depth < nrcHook.stopDepth) {
        return PT_HOOK_CONTINUE;
    }
    _ = nrcQuery(v, 1.0);
    return PT_HOOK_TERMINATE;
}

fn ptTerminate(v: PtPathVertex) -> vec3f {
    return vec3f(0.0);
}

fn ptOnRadiance(pixel: u32, sampleIndex: u32, index: u32, flags: u32, value: vec3f) {
    let train = nrcTrainSlot(pixel, sampleIndex);
    if (train == 0u || index > nrcHook.trainDepth) {
        return;
    }
    let i = (train - 1u) * (nrcHook.trainDepth + 1u) + index;
    nrcSums[i] = vec4f(nrcSums[i].xyz + value, nrcSums[i].w);
}
`;

const NRC_HOOK_INLINE_BINDINGS: GPUBindGroupLayoutEntry[] = [
  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ...[1, 2, 3, 4].map((binding): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } })),
];

const TARGETS_INLINE_WGSL = /* wgsl */ `
struct P { slots: u32, layers: u32, batch: u32, capacity: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;
@group(0) @binding(2) var<storage, read> sums: array<vec4f>;
@group(0) @binding(3) var<storage, read> queries: array<vec4f>;
@group(0) @binding(4) var<storage, read> preds: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outSet: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> count: array<atomic<u32>>;

fn finite3(v: vec3f) -> bool {
  let e = (bitcast<vec3u>(v) >> vec3u(23u)) & vec3u(0xffu);
  return all(e != vec3u(0xffu));
}

@compute @workgroup_size(64)
fn nrcTargets(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.slots * p.layers) { return; }
  let k = i / p.slots;
  let slot = i % p.slots;
  let rb = (k * p.slots + slot) * ${NRC_RECORD_VEC4}u;
  let beta = verts[rb + 3u];
  if (beta.w == 0.0) { return; }
  let s0 = slot * (p.layers + 1u);
  var after = vec3f(0.0);
  for (var j = k + 1u; j <= p.layers; j++) { after += sums[s0 + j].xyz; }
  let tq = u32(sums[s0].w);
  if (tq != 0u) {
    let qb = (tq - 1u) * ${NRC_RECORD_VEC4}u;
    after += queries[qb + 3u].xyz * max(preds[tq - 1u].xyz, vec3f(0.0));
  }
  let t = max(select(vec3f(0.0), after / beta.xyz, beta.xyz > vec3f(1.0e-8)), vec3f(0.0));
  if (!finite3(t)) { return; }
  let slotOut = atomicAdd(&count[1], 1u);
  if (slotOut >= p.capacity) { return; }
  let o = slotOut * ${NRC_RECORD_VEC4}u;
  outSet[o] = verts[rb];
  outSet[o + 1u] = verts[rb + 1u];
  outSet[o + 2u] = verts[rb + 2u];
  outSet[o + 3u] = vec4f(t, 1.0);
}

@compute @workgroup_size(1)
fn nrcFinish() {
  let n = min(atomicLoad(&count[1]), p.capacity);
  atomicStore(&count[1], n);
  atomicStore(&count[0], min(n, p.batch));
}
`;

const NRC_HOOK_BINDINGS: GPUBindGroupLayoutEntry[] = [
  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
];

/** Shared by the MLP hooks: records → the 64 NRC inputs (VkNRC layout), position normalised to the scene box. */
const NRC_INPUTS_WGSL = /* wgsl */ `
${nrcEncodingWgsl()}
struct NrcBox { lo: vec4f, scale: vec4f }
fn nrcInputsOf(r0: vec4f, r1: vec4f, r2: vec4f) -> array<f32, NN_IN> {
  let p = clamp((r0.xyz - nrcBox.lo.xyz) * nrcBox.scale.xyz - 1.0, vec3f(-1.0), vec3f(1.0));
  return nnEncodeNrc(p, r1.xy, r1.zw, r2.w, r2.xyz, vec3f(0.0));
}
`;

const NRC_INFER_HOOKS = /* wgsl */ `
@group(1) @binding(0) var<storage, read> nrcIn: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> nrcOut: array<vec4f>;
@group(1) @binding(2) var<uniform> nrcBox: NrcBox;
${NRC_INPUTS_WGSL}
fn nnInferLoad(r: u32) -> array<f32, NN_IN> {
  let b = r * ${NRC_RECORD_VEC4}u;
  return nrcInputsOf(nrcIn[b], nrcIn[b + 1u], nrcIn[b + 2u]);
}
fn nnInferStore(r: u32, y: vec3f) { nrcOut[r] = vec4f(y, 0.0); }
`;

const NRC_TRAIN_HOOKS = /* wgsl */ `
@group(1) @binding(0) var<storage, read> nrcSet: array<vec4f>;
@group(1) @binding(1) var<storage, read> nrcSetCount: array<u32>;
@group(1) @binding(2) var<uniform> nrcBox: NrcBox;
${NRC_INPUTS_WGSL}
fn nrcHash(x: u32) -> u32 {
  let s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
  let n = max(nrcSetCount[1], 1u);
  let i = nrcHash(r ^ nrcHash(nnState.step * 0x9e3779b9u + 17u)) % n;
  let b = i * ${NRC_RECORD_VEC4}u;
  *x = nrcInputsOf(nrcSet[b], nrcSet[b + 1u], nrcSet[b + 2u]);
  return nrcSet[b + 3u].xyz;
}
`;

const TARGETS_WGSL = /* wgsl */ `
struct P { npix: u32, layers: u32, batch: u32, capacity: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;
@group(0) @binding(2) var<storage, read> radiance: array<vec4f>;
@group(0) @binding(3) var<storage, read> tailPred: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outSet: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> count: array<atomic<u32>>;

fn finite3(v: vec3f) -> bool {
  let e = (bitcast<vec3u>(v) >> vec3u(23u)) & vec3u(0xffu);
  return all(e != vec3u(0xffu));
}

@compute @workgroup_size(64)
fn nrcTargets(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.npix * p.layers) { return; }
  let k = i / p.npix;
  let px = i % p.npix;
  let rb = (k * p.npix + px) * ${NRC_RECORD_VEC4}u;
  let beta = verts[rb + 3u];
  if (beta.w == 0.0) { return; }
  let tail = verts[(p.layers * p.npix + px) * ${NRC_RECORD_VEC4}u + 3u];
  var after = radiance[p.layers * p.npix + px].xyz - radiance[k * p.npix + px].xyz;
  if (tail.w != 0.0) { after += tail.xyz * max(tailPred[px].xyz, vec3f(0.0)); }
  let t = max(select(vec3f(0.0), after / beta.xyz, beta.xyz > vec3f(1.0e-8)), vec3f(0.0));
  if (!finite3(t)) { return; }
  let slot = atomicAdd(&count[1], 1u);
  if (slot >= p.capacity) { return; }
  let o = slot * ${NRC_RECORD_VEC4}u;
  outSet[o] = verts[rb];
  outSet[o + 1u] = verts[rb + 1u];
  outSet[o + 2u] = verts[rb + 2u];
  outSet[o + 3u] = vec4f(t, 1.0);
}

@compute @workgroup_size(1)
fn nrcFinish() {
  let n = min(atomicLoad(&count[1]), p.capacity);
  atomicStore(&count[1], n);
  atomicStore(&count[0], min(n, p.batch));
}
`;

const RESOLVE_WGSL = /* wgsl */ `
struct P { capacity: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> queries: array<vec4f>;
@group(0) @binding(2) var<storage, read> preds: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> count: array<u32>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4f>;

@compute @workgroup_size(64)
fn nrcResolve(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r == 0u) {
    count[1] = count[1] + 1u; // the frame counter the hook rotates training tiles with
  }
  if (r >= min(count[0], p.capacity)) { return; }
  let b = r * ${NRC_RECORD_VEC4}u;
  if (queries[b + 3u].w != 1.0) { return; } // a training path's tail: training only
  let px = u32(queries[b].w);
  let beta = queries[b + 3u].xyz;
  accum[px] = accum[px] + vec4f(beta * max(preds[r].xyz, vec3f(0.0)), 0.0);
}
`;

/**
 * Inline training, after the resolve: a training path's pixel already holds everything the path gathered (the integrator added it);
 * remove what it gathered beyond the cache vertex K (reported values with index > K, summed per path by ptOnRadiance), so the pixel
 * keeps the same estimate as every other pixel (its display query at K was resolved into the image). Only when K < M.
 */
const FIXUP_WGSL = /* wgsl */ `
struct P { slots: u32, queryDepth: u32, trainDepth: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;
@group(0) @binding(2) var<storage, read> sums: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4f>;

@compute @workgroup_size(64)
fn nrcFixup(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= p.slots || p.queryDepth >= p.trainDepth) { return; }
  let v0 = slot * ${NRC_RECORD_VEC4}u; // layer 0: the primary vertex (no record: the camera ray escaped, nothing to remove)
  if (verts[v0 + 3u].w == 0.0) { return; }
  let s0 = slot * (p.trainDepth + 1u);
  var beyond = vec3f(0.0);
  for (var j = p.queryDepth + 1u; j <= p.trainDepth; j++) { beyond += sums[s0 + j].xyz; }
  let px = u32(verts[v0].w);
  accum[px] = accum[px] - vec4f(beyond, 0.0);
}
`;

export interface NrcSettings {
  /**
   * Where training paths come from: 'inline' (default) the render pass's own update paths, gathered through `ptOnRadiance` (no extra
   * pass); 'replay' a separate trainer film replayed trainDepth + 1 times per frame (the first version, kept for the A/B).
   */
  training: 'inline' | 'replay';
  /** Inline training: 1 / updateBlock² of the pixels train each frame (default 4: 6.25%). */
  updateBlock: number;
  /**
   * Inline training: which pixels. 'tile' (default) every updateBlock²-th 8 × 8 tile, so a workgroup is all training paths or none;
   * 'pixel' one pixel per updateBlock × updateBlock block (every workgroup waits on its longest, training, path).
   */
  trainSelect: 'tile' | 'pixel';
  precision: Precision;
  /** Inference backend options from `inferArmOptions` (portable when absent). */
  sgm?: MlpEngineOptions['sgm'];
  /** Vertex the render paths terminate into the cache at (0 = the primary hit: the "cache only" view). Default 1. */
  queryDepth: number;
  /** Training paths are terminated at vertex M with the cache as the tail; targets for vertices 0..M−1. Default 4. */
  trainDepth: number;
  /** Replay training: the trainer film (same camera, same aspect). Default 128 × 72. */
  trainWidth: number;
  trainHeight: number;
  stepsPerFrame: number;
  batch: number;
  learningRate: number;
  useEma: boolean;
  seed: number;
}

export const DEFAULT_NRC_SETTINGS: NrcSettings = {
  training: 'inline',
  updateBlock: 4,
  trainSelect: 'tile',
  precision: 'f16',
  queryDepth: 1,
  trainDepth: 4,
  trainWidth: 128,
  trainHeight: 72,
  stepsPerFrame: 4,
  batch: 16384,
  learningRate: DEFAULT_TRAIN.learningRate,
  useEma: false,
  seed: 1,
};

type Stage = 'trainingPaths' | 'trainSteps' | 'renderPass' | 'inference';

export class NrcCache {
  readonly device!: GPUDevice;
  readonly render!: PtMegakernel;
  /** Replay training only. */
  readonly trainer!: PtMegakernel | null;
  readonly mlp!: MlpEngine;
  readonly settings!: NrcSettings;
  readonly width!: number;
  readonly height!: number;
  /** Training paths per frame: trainer pixels (replay) or update-path slots (inline). */
  readonly trainPaths!: number;
  private readonly hookParams!: GPUBuffer;
  private readonly trainGroups!: GPUBindGroup[];
  private readonly trainVerts!: GPUBuffer;
  private readonly trainSums!: GPUBuffer | null;
  private readonly trainRadiance!: GPUBuffer | null;
  private readonly tailPred!: GPUBuffer | null;
  private readonly trainSet!: GPUBuffer;
  private readonly trainCount!: GPUBuffer;
  private readonly tailCount!: GPUBuffer | null;
  private readonly queries!: GPUBuffer;
  private readonly preds!: GPUBuffer;
  private readonly queryCount!: GPUBuffer;
  private readonly box!: GPUBuffer;
  private readonly targetsPipeline!: GPUComputePipeline;
  private readonly finishPipeline!: GPUComputePipeline;
  private readonly resolvePipeline!: GPUComputePipeline;
  private readonly targetsGroup!: GPUBindGroup;
  private readonly finishGroup!: GPUBindGroup;
  private readonly resolveGroup!: GPUBindGroup;
  private readonly tailInferGroup!: GPUBindGroup | null;
  private readonly renderInferGroup!: GPUBindGroup;
  private readonly trainHookGroup!: GPUBindGroup;
  private readonly targetParams!: GPUBuffer;
  private readonly resolveParams!: GPUBuffer;
  private readonly queryCapacity!: number;
  private readonly fixupPipeline!: GPUComputePipeline | null;
  private readonly fixupGroup!: GPUBindGroup | null;
  private readonly fixupParams!: GPUBuffer | null;
  private frame = 0;

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  /**
   * Builds the hooked kernel(s) from the plain kernel's options, the MLP and the buffers.
   * `bounds`: the scene box (min, max) for the position encoding.
   */
  static async create(device: GPUDevice, caps: Caps, plain: PtMegakernel, sceneBuffers: Parameters<PtMegakernel['setScene']>, film: { width: number; height: number; camera: PtLookAtCamera; maxDepth: number; rrStartDepth?: number }, bounds: { min: [number, number, number]; max: [number, number, number] }, settings: Partial<NrcSettings> = {}): Promise<NrcCache> {
    const s: NrcSettings = { ...DEFAULT_NRC_SETTINGS, ...settings };
    const inline = s.training === 'inline';
    const M = s.trainDepth;
    const pixels = film.width * film.height;
    const B = Math.max(1, Math.floor(s.updateBlock));
    const slotsX = Math.ceil(film.width / B);
    const tileEvery = s.trainSelect === 'tile' ? B * B : 0;
    const tilesX = Math.ceil(film.width / 8);
    const trainPaths = !inline ? s.trainWidth * s.trainHeight : tileEvery ? Math.ceil(film.height / 8) * Math.ceil(tilesX / tileEvery) * 64 : slotsX * Math.ceil(film.height / B);
    const hooks: PtHooksOptions = inline ? { wgsl: NRC_HOOK_INLINE_WGSL, bindings: NRC_HOOK_INLINE_BINDINGS } : { wgsl: NRC_HOOK_WGSL, bindings: NRC_HOOK_BINDINGS };
    const hooked = (label: string): PtMegakernelOptions => ({ ...plain.options, integrator: { ...plain.options.integrator, hooks }, label });
    const [render, trainer] = await Promise.all([PtMegakernel.create(device, caps, hooked('nrc render')), inline ? Promise.resolve(null) : PtMegakernel.create(device, caps, hooked('nrc trainer'))]);
    render.setScene(...sceneBuffers);
    render.setFilm(film.width, film.height);
    render.setCamera(film.camera);
    render.setLimits({ maxDepth: film.maxDepth, ...(film.rrStartDepth ? { rrStartDepth: film.rrStartDepth } : {}) });
    render.setSeed(s.seed);
    if (trainer) {
      trainer.setScene(...sceneBuffers);
      trainer.setFilm(s.trainWidth, s.trainHeight);
      trainer.setCamera(film.camera);
      // Vertex M must get a hook call (hooks run at vertices < maxDepth); no Russian roulette on training paths.
      trainer.setLimits({ maxDepth: M + 1 });
      if ((M * trainPaths * RECORD_BYTES) % device.limits.minStorageBufferOffsetAlignment) throw new Error('NrcCache: trainDepth × trainWidth × trainHeight must be a multiple of 4 (storage offset alignment)');
    }

    const buf = (label: string, size: number, usage = STORAGE | COPY_DST | COPY_SRC) => device.createBuffer({ label, size: Math.max(16, size), usage });
    const trainVerts = buf('nrc train vertices', (inline ? M : M + 1) * trainPaths * RECORD_BYTES);
    const trainSums = inline ? buf('nrc train sums', trainPaths * (M + 1) * 16) : null;
    const trainRadiance = inline ? null : buf('nrc train radiance', (M + 1) * trainPaths * 16);
    const tailPred = inline ? null : buf('nrc tail predictions', trainPaths * 16);
    const capacity = M * trainPaths;
    const trainSet = buf('nrc training set', capacity * RECORD_BYTES);
    const trainCount = buf('nrc training count', 16);
    const tailCount = inline ? null : buf('nrc tail count', 16);
    if (tailCount) device.queue.writeBuffer(tailCount, 0, new Uint32Array([trainPaths, 0, 0, 0]));
    // One query per pixel, plus (inline) a training path's display query and tail: up to two per training path.
    const queryCapacity = pixels + (inline ? trainPaths : 0);
    const queries = buf('nrc queries', queryCapacity * RECORD_BYTES);
    const preds = buf('nrc predictions', queryCapacity * 16);
    const queryCount = buf('nrc query count', 16);
    const box = buf('nrc box', 32, UNIFORM | COPY_DST);
    const lo = bounds.min;
    const scale = bounds.max.map((h, i) => 2 / Math.max(h - (lo[i] as number), 1e-6));
    device.queue.writeBuffer(box, 0, new Float32Array([...lo, 0, ...scale, 0]));

    // Hook params: replay slots 0..M (trainer run k), slot M+1 the render kernel; inline uses slot M+1 only.
    const hookParams = buf('nrc hook params', (M + 2) * HOOK_SLOT, UNIFORM | COPY_DST);
    for (let k = 0; k <= M; k++) device.queue.writeBuffer(hookParams, k * HOOK_SLOT, new Uint32Array([1, k, k * trainPaths, 0]));
    const renderParams = new Uint32Array([0, s.queryDepth, 0, queryCapacity, M, film.width, B, slotsX, trainPaths, tileEvery, tilesX, 0]);
    device.queue.writeBuffer(hookParams, (M + 1) * HOOK_SLOT, renderParams);
    const paramsEntry = (k: number) => ({ binding: 0, resource: { buffer: hookParams, offset: k * HOOK_SLOT, size: inline ? 48 : 16 } });
    let trainGroups: GPUBindGroup[] = [];
    if (inline) {
      render.setHookBindGroup(device.createBindGroup({
        layout: render.hookBindGroupLayout as GPUBindGroupLayout,
        entries: [paramsEntry(M + 1), ...[queries, queryCount, trainVerts, trainSums as GPUBuffer].map((buffer, i) => ({ binding: i + 1, resource: { buffer } }))],
      }));
    } else {
      const dummyCount = buf('nrc trainer hook count', 16);
      const group = (k: number, records: GPUBuffer, count: GPUBuffer, layout: GPUBindGroupLayout) => device.createBindGroup({ layout, entries: [paramsEntry(k), { binding: 1, resource: { buffer: records } }, { binding: 2, resource: { buffer: count } }] });
      trainGroups = Array.from({ length: M + 1 }, (_, k) => group(k, trainVerts, dummyCount, (trainer as PtMegakernel).hookBindGroupLayout as GPUBindGroupLayout));
      render.setHookBindGroup(group(M + 1, queries, queryCount, render.hookBindGroupLayout as GPUBindGroupLayout));
    }

    const mlp = await MlpEngine.create(device, {
      precision: s.precision,
      capacity: s.batch,
      inferCapacity: Math.max(queryCapacity, trainPaths),
      useEma: s.useEma,
      train: { ...DEFAULT_TRAIN, learningRate: s.learningRate },
      trainHooks: NRC_TRAIN_HOOKS,
      inferHooks: NRC_INFER_HOOKS,
      ...(s.sgm ? { sgm: s.sgm } : {}),
    });
    mlp.uploadWeights(initWeights(DEFAULT_SHAPE, s.seed));
    const tailInferGroup = inline
      ? null
      : device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 0, resource: { buffer: trainVerts, offset: M * trainPaths * RECORD_BYTES, size: trainPaths * RECORD_BYTES } }, { binding: 1, resource: { buffer: tailPred as GPUBuffer } }, { binding: 2, resource: { buffer: box } }] });
    const renderInferGroup = device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 0, resource: { buffer: queries } }, { binding: 1, resource: { buffer: preds } }, { binding: 2, resource: { buffer: box } }] });
    const trainHookGroup = device.createBindGroup({ layout: mlp.trainLayout, entries: [{ binding: 0, resource: { buffer: trainSet } }, { binding: 1, resource: { buffer: trainCount } }, { binding: 2, resource: { buffer: box } }] });

    const targetsModule = device.createShaderModule({ label: 'nrc targets', code: inline ? TARGETS_INLINE_WGSL : TARGETS_WGSL });
    const [targetsPipeline, finishPipeline, resolvePipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: 'nrc targets', layout: 'auto', compute: { module: targetsModule, entryPoint: 'nrcTargets' } }),
      device.createComputePipelineAsync({ label: 'nrc finish', layout: 'auto', compute: { module: targetsModule, entryPoint: 'nrcFinish' } }),
      device.createComputePipelineAsync({ label: 'nrc resolve', layout: 'auto', compute: { module: device.createShaderModule({ label: 'nrc resolve', code: RESOLVE_WGSL }), entryPoint: 'nrcResolve' } }),
    ]);
    const targetParams = buf('nrc targets params', 16, UNIFORM | COPY_DST);
    device.queue.writeBuffer(targetParams, 0, new Uint32Array([trainPaths, M, s.batch, capacity]));
    const targetBuffers = inline ? [targetParams, trainVerts, trainSums as GPUBuffer, queries, preds, trainSet, trainCount] : [targetParams, trainVerts, trainRadiance as GPUBuffer, tailPred as GPUBuffer, trainSet, trainCount];
    const targetsGroup = device.createBindGroup({ layout: targetsPipeline.getBindGroupLayout(0), entries: targetBuffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    // nrcFinish uses the params and the count only (auto layout drops the rest).
    const finishGroup = device.createBindGroup({ layout: finishPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: targetParams } }, { binding: targetBuffers.length - 1, resource: { buffer: trainCount } }] });
    const resolveParams = buf('nrc resolve params', 16, UNIFORM | COPY_DST);
    device.queue.writeBuffer(resolveParams, 0, new Uint32Array([queryCapacity, 0, 0, 0]));
    const resolveGroup = device.createBindGroup({
      layout: resolvePipeline.getBindGroupLayout(0),
      entries: [resolveParams, queries, preds, queryCount, render.accumulator as GPUBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    // Inline training: the fix-up that gives training pixels the same image estimate as the rest.
    let fixupPipeline: GPUComputePipeline | null = null;
    let fixupGroup: GPUBindGroup | null = null;
    let fixupParams: GPUBuffer | null = null;
    if (inline) {
      fixupPipeline = await device.createComputePipelineAsync({ label: 'nrc fixup', layout: 'auto', compute: { module: device.createShaderModule({ label: 'nrc fixup', code: FIXUP_WGSL }), entryPoint: 'nrcFixup' } });
      fixupParams = buf('nrc fixup params', 16, UNIFORM | COPY_DST);
      device.queue.writeBuffer(fixupParams, 0, new Uint32Array([trainPaths, s.queryDepth, M, 0]));
      fixupGroup = device.createBindGroup({ layout: fixupPipeline.getBindGroupLayout(0), entries: [fixupParams, trainVerts, trainSums as GPUBuffer, render.accumulator as GPUBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    }

    return new NrcCache({
      device, render, trainer, mlp, settings: s, width: film.width, height: film.height, trainPaths, hookParams, trainGroups, trainVerts, trainSums, trainRadiance, tailPred, trainSet, trainCount, tailCount, queries, preds, queryCount, box,
      targetsPipeline, finishPipeline, resolvePipeline, targetsGroup, finishGroup, resolveGroup, tailInferGroup, renderInferGroup, trainHookGroup, targetParams, resolveParams, fixupPipeline, fixupGroup, fixupParams, queryCapacity,
    });
  }

  get accumulator(): GPUBuffer {
    return this.render.accumulator as GPUBuffer;
  }

  get samples(): number {
    return this.render.samples;
  }

  /** Clears the image (keeps what the network learned). */
  reset(encoder: GPUCommandEncoder): void {
    this.render.reset(encoder);
  }

  /** Forgets everything the network learned (fresh weights, Adam state). */
  resetNetwork(): void {
    this.mlp.uploadWeights(initWeights(DEFAULT_SHAPE, this.settings.seed + this.frame));
  }

  setCamera(camera: PtLookAtCamera): void {
    this.render.setCamera(camera);
    this.trainer?.setCamera(camera);
  }

  /** Vertex the render paths terminate into the cache at (0 = primary hit). Clear the image after a change. */
  setQueryDepth(depth: number): void {
    const M = this.settings.trainDepth;
    this.device.queue.writeBuffer(this.hookParams, (M + 1) * HOOK_SLOT + 4, new Uint32Array([depth]));
    if (this.fixupParams) this.device.queue.writeBuffer(this.fixupParams, 4, new Uint32Array([depth]));
  }

  /**
   * A frame's stages in the order `encodeFrame` runs them. Replay: training paths, steps, render, inference. Inline: render (it
   * carries the training paths), inference (+ targets), steps (VkNRC's order: inference before training).
   */
  get stages(): readonly Stage[] {
    return this.settings.training === 'inline' ? ['renderPass', 'inference', 'trainSteps'] : ['trainingPaths', 'trainSteps', 'renderPass', 'inference'];
  }

  /**
   * One frame: one render sample pass with the cache, and training unless `train` is false. No readback.
   * (No timestamp writes: about 20 timestamped passes per frame lost the Metal device with "Insufficient Memory" about half
   * the time; time stages by submitting them one at a time instead, `encodeStage`.)
   */
  encodeFrame(encoder: GPUCommandEncoder, options: { train?: boolean } = {}): void {
    this.frame++;
    const train = options.train !== false;
    for (const stage of this.stages) if (train || stage === 'renderPass' || stage === 'inference') this.encodeStage(encoder, stage, train);
  }

  /** One stage of a frame, in `stages` order (`nextFrame` first when calling stages yourself). */
  encodeStage(encoder: GPUCommandEncoder, stage: Stage, train = true): void {
    const { settings: s } = this;
    const M = s.trainDepth;
    const inline = s.training === 'inline';
    if (stage === 'trainingPaths') {
      if (inline || !this.trainer) return;
      const n = this.trainPaths;
      encoder.clearBuffer(this.trainVerts);
      encoder.clearBuffer(this.trainCount);
      this.trainer.setSeed(s.seed * 7919 + this.frame);
      for (let k = 0; k <= M; k++) {
        this.trainer.reset(encoder);
        this.trainer.setHookBindGroup(this.trainGroups[k] as GPUBindGroup);
        this.trainer.encode(encoder, 1);
        encoder.copyBufferToBuffer(this.trainer.accumulator as GPUBuffer, 0, this.trainRadiance as GPUBuffer, k * n * 16, n * 16);
      }
      this.mlp.encodeInfer(encoder, this.tailCount as GPUBuffer, 0, this.tailInferGroup as GPUBindGroup);
      this.encodeTargets(encoder);
    } else if (stage === 'trainSteps') {
      for (let i = 0; i < s.stepsPerFrame; i++) this.mlp.encodeTrainStep(encoder, this.trainCount, 0, this.trainHookGroup);
    } else if (stage === 'renderPass') {
      encoder.clearBuffer(this.queryCount, 0, 4); // word 1 is the frame counter
      if (inline) {
        encoder.clearBuffer(this.trainVerts);
        encoder.clearBuffer(this.trainSums as GPUBuffer);
      }
      this.render.encode(encoder, 1);
    } else {
      this.mlp.encodeInfer(encoder, this.queryCount, 0, this.renderInferGroup);
      if (inline && train) {
        encoder.clearBuffer(this.trainCount);
        this.encodeTargets(encoder);
      }
      const pass = encoder.beginComputePass({ label: 'nrc resolve' });
      pass.setPipeline(this.resolvePipeline);
      pass.setBindGroup(0, this.resolveGroup);
      pass.dispatchWorkgroups(Math.ceil(this.queryCapacity / 64));
      if (this.fixupPipeline && this.fixupGroup) {
        pass.setPipeline(this.fixupPipeline);
        pass.setBindGroup(0, this.fixupGroup);
        pass.dispatchWorkgroups(Math.ceil(this.trainPaths / 64));
      }
      pass.end();
    }
  }

  private encodeTargets(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: 'nrc targets' });
    pass.setPipeline(this.targetsPipeline);
    pass.setBindGroup(0, this.targetsGroup);
    pass.dispatchWorkgroups(Math.ceil((this.trainPaths * this.settings.trainDepth) / 64));
    pass.setPipeline(this.finishPipeline);
    pass.setBindGroup(0, this.finishGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  /** Starts a frame for `encodeStage` callers (the trainer's seed advances per frame). */
  nextFrame(): void {
    this.frame++;
  }

  /** Debug only (never in a frame loop): training records of the last frame, and render queries. */
  async readTrainCount(): Promise<{ batch: number; records: number }> {
    const v = new Uint32Array(await readU32s(this.device, this.trainCount, 2));
    return { batch: v[0] as number, records: v[1] as number };
  }

  async readQueryCount(): Promise<number> {
    return new Uint32Array(await readU32s(this.device, this.queryCount, 1))[0] as number;
  }

  destroy(): void {
    this.render.destroy();
    this.trainer?.destroy();
    this.mlp.destroy();
    for (const b of [this.hookParams, this.trainVerts, this.trainSums, this.trainRadiance, this.tailPred, this.trainSet, this.trainCount, this.tailCount, this.queries, this.preds, this.queryCount, this.box, this.targetParams, this.resolveParams, this.fixupParams]) b?.destroy();
  }
}

async function readU32s(device: GPUDevice, buffer: GPUBuffer, n: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, 16);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0, n * 4);
  staging.unmap();
  staging.destroy();
  return out;
}
