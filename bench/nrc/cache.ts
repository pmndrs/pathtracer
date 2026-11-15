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
import { PtMegakernel, type PtLookAtCamera, type PtMegakernelOptions } from '@pmndrs/pathtracer';
import type { Caps } from '@pmndrs/raytracer';

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;
const COPY_SRC = GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM;

/** vec4 per record: (position, pixel bits), (direction θ/π φ/2π, normal θ/π φ/2π), (albedo, roughness), (throughput, valid). */
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
    nrcRecords[b] = vec4f(v.position, bitcast<f32>(v.pixel));
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
@group(0) @binding(3) var<storage, read> count: array<u32>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4f>;

@compute @workgroup_size(64)
fn nrcResolve(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= min(count[0], p.capacity)) { return; }
  let b = r * ${NRC_RECORD_VEC4}u;
  let px = bitcast<u32>(queries[b].w);
  let beta = queries[b + 3u].xyz;
  accum[px] = accum[px] + vec4f(beta * max(preds[r].xyz, vec3f(0.0)), 0.0);
}
`;

export interface NrcSettings {
  precision: Precision;
  /** Inference backend options from `inferArmOptions` (portable when absent). */
  sgm?: MlpEngineOptions['sgm'];
  /** Vertex the render paths terminate into the cache at (0 = the primary hit: the "cache only" view). Default 1. */
  queryDepth: number;
  /** Training paths are terminated at vertex M with the cache as the tail; targets for vertices 0..M−1. Default 4. */
  trainDepth: number;
  /** Training film (same camera, same aspect). Default about 128 × 72. */
  trainWidth: number;
  trainHeight: number;
  stepsPerFrame: number;
  batch: number;
  learningRate: number;
  useEma: boolean;
  seed: number;
}

export const DEFAULT_NRC_SETTINGS: NrcSettings = {
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

export interface NrcStageTimes {
  querySet: GPUQuerySet;
  /** 8 slots from `first`: trainer (runs + tail inference + targets), training, render pass, inference + resolve. */
  first: number;
}

export class NrcCache {
  readonly device!: GPUDevice;
  readonly render!: PtMegakernel;
  readonly trainer!: PtMegakernel;
  readonly mlp!: MlpEngine;
  readonly settings!: NrcSettings;
  readonly width!: number;
  readonly height!: number;
  private readonly npix!: number;
  private readonly hookParams!: GPUBuffer;
  private readonly trainGroups!: GPUBindGroup[];
  private readonly renderGroup!: GPUBindGroup;
  private readonly trainVerts!: GPUBuffer;
  private readonly trainRadiance!: GPUBuffer;
  private readonly tailPred!: GPUBuffer;
  private readonly trainSet!: GPUBuffer;
  private readonly trainCount!: GPUBuffer;
  private readonly tailCount!: GPUBuffer;
  private readonly queries!: GPUBuffer;
  private readonly preds!: GPUBuffer;
  private readonly queryCount!: GPUBuffer;
  private readonly box!: GPUBuffer;
  private readonly targetsPipeline!: GPUComputePipeline;
  private readonly finishPipeline!: GPUComputePipeline;
  private readonly resolvePipeline!: GPUComputePipeline;
  private readonly targetsGroup!: GPUBindGroup;
  private readonly resolveGroup!: GPUBindGroup;
  private readonly tailInferGroup!: GPUBindGroup;
  private readonly renderInferGroup!: GPUBindGroup;
  private readonly trainHookGroup!: GPUBindGroup;
  private readonly resolveParams!: GPUBuffer;
  private readonly finishGroup!: GPUBindGroup;
  private frame = 0;

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  /**
   * Builds the two hooked kernels from the plain kernel's options, the MLP and the buffers.
   * `bounds`: the scene box (min, max) for the position encoding.
   */
  static async create(device: GPUDevice, caps: Caps, plain: PtMegakernel, sceneBuffers: Parameters<PtMegakernel['setScene']>, film: { width: number; height: number; camera: PtLookAtCamera; maxDepth: number; rrStartDepth?: number }, bounds: { min: [number, number, number]; max: [number, number, number] }, settings: Partial<NrcSettings> = {}): Promise<NrcCache> {
    const s: NrcSettings = { ...DEFAULT_NRC_SETTINGS, ...settings };
    const hooked = (label: string): PtMegakernelOptions => ({ ...plain.options, integrator: { ...plain.options.integrator, hooks: { wgsl: NRC_HOOK_WGSL, bindings: NRC_HOOK_BINDINGS } }, label });
    const [render, trainer] = await Promise.all([PtMegakernel.create(device, caps, hooked('nrc render')), PtMegakernel.create(device, caps, hooked('nrc trainer'))]);
    for (const k of [render, trainer]) k.setScene(...sceneBuffers);
    render.setFilm(film.width, film.height);
    render.setCamera(film.camera);
    render.setLimits({ maxDepth: film.maxDepth, ...(film.rrStartDepth ? { rrStartDepth: film.rrStartDepth } : {}) });
    render.setSeed(s.seed);
    trainer.setFilm(s.trainWidth, s.trainHeight);
    trainer.setCamera(film.camera);
    // Vertex M must get a hook call (hooks run at vertices < maxDepth); no Russian roulette on training paths.
    trainer.setLimits({ maxDepth: s.trainDepth + 1 });

    const npix = s.trainWidth * s.trainHeight;
    if ((s.trainDepth * npix * RECORD_BYTES) % device.limits.minStorageBufferOffsetAlignment) throw new Error('NrcCache: trainDepth × trainWidth × trainHeight must be a multiple of 4 (storage offset alignment)');
    const pixels = film.width * film.height;
    const M = s.trainDepth;
    const buf = (label: string, size: number, usage = STORAGE | COPY_DST | COPY_SRC) => device.createBuffer({ label, size: Math.max(16, size), usage });
    const trainVerts = buf('nrc train vertices', (M + 1) * npix * RECORD_BYTES);
    const trainRadiance = buf('nrc train radiance', (M + 1) * npix * 16);
    const tailPred = buf('nrc tail predictions', npix * 16);
    const capacity = M * npix;
    const trainSet = buf('nrc training set', capacity * RECORD_BYTES);
    const trainCount = buf('nrc training count', 16);
    const tailCount = buf('nrc tail count', 16);
    device.queue.writeBuffer(tailCount, 0, new Uint32Array([npix, 0, 0, 0]));
    const queries = buf('nrc queries', pixels * RECORD_BYTES);
    const preds = buf('nrc predictions', pixels * 16);
    const queryCount = buf('nrc query count', 16);
    const box = buf('nrc box', 32, UNIFORM | COPY_DST);
    const lo = bounds.min;
    const scale = bounds.max.map((h, i) => 2 / Math.max(h - (lo[i] as number), 1e-6));
    device.queue.writeBuffer(box, 0, new Float32Array([...lo, 0, ...scale, 0]));

    // Hook params: slot k (trainer run k, k = 0..M) and slot M+1 (render).
    const hookParams = buf('nrc hook params', (M + 2) * HOOK_SLOT, UNIFORM | COPY_DST);
    for (let k = 0; k <= M; k++) device.queue.writeBuffer(hookParams, k * HOOK_SLOT, new Uint32Array([1, k, k * npix, 0]));
    device.queue.writeBuffer(hookParams, (M + 1) * HOOK_SLOT, new Uint32Array([0, s.queryDepth, 0, pixels]));
    const dummyCount = buf('nrc trainer hook count', 16);
    const hookGroup = (k: number, records: GPUBuffer, count: GPUBuffer, layout: GPUBindGroupLayout) => device.createBindGroup({
      layout,
      entries: [{ binding: 0, resource: { buffer: hookParams, offset: k * HOOK_SLOT, size: 16 } }, { binding: 1, resource: { buffer: records } }, { binding: 2, resource: { buffer: count } }],
    });
    const trainGroups = Array.from({ length: M + 1 }, (_, k) => hookGroup(k, trainVerts, dummyCount, trainer.hookBindGroupLayout as GPUBindGroupLayout));
    const renderGroup = hookGroup(M + 1, queries, queryCount, render.hookBindGroupLayout as GPUBindGroupLayout);
    render.setHookBindGroup(renderGroup);

    const mlp = await MlpEngine.create(device, {
      precision: s.precision,
      capacity: s.batch,
      inferCapacity: Math.max(pixels, npix),
      useEma: s.useEma,
      train: { ...DEFAULT_TRAIN, learningRate: s.learningRate },
      trainHooks: NRC_TRAIN_HOOKS,
      inferHooks: NRC_INFER_HOOKS,
      ...(s.sgm ? { sgm: s.sgm } : {}),
    });
    mlp.uploadWeights(initWeights(DEFAULT_SHAPE, s.seed));
    const tailSlice = { buffer: trainVerts, offset: M * npix * RECORD_BYTES, size: npix * RECORD_BYTES };
    const tailInferGroup = device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 0, resource: tailSlice }, { binding: 1, resource: { buffer: tailPred } }, { binding: 2, resource: { buffer: box } }] });
    const renderInferGroup = device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 0, resource: { buffer: queries } }, { binding: 1, resource: { buffer: preds } }, { binding: 2, resource: { buffer: box } }] });
    const trainHookGroup = device.createBindGroup({ layout: mlp.trainLayout, entries: [{ binding: 0, resource: { buffer: trainSet } }, { binding: 1, resource: { buffer: trainCount } }, { binding: 2, resource: { buffer: box } }] });

    const targetsModule = device.createShaderModule({ label: 'nrc targets', code: TARGETS_WGSL });
    const [targetsPipeline, finishPipeline, resolvePipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: 'nrc targets', layout: 'auto', compute: { module: targetsModule, entryPoint: 'nrcTargets' } }),
      device.createComputePipelineAsync({ label: 'nrc finish', layout: 'auto', compute: { module: targetsModule, entryPoint: 'nrcFinish' } }),
      device.createComputePipelineAsync({ label: 'nrc resolve', layout: 'auto', compute: { module: device.createShaderModule({ label: 'nrc resolve', code: RESOLVE_WGSL }), entryPoint: 'nrcResolve' } }),
    ]);
    const targetParams = buf('nrc targets params', 16, UNIFORM | COPY_DST);
    device.queue.writeBuffer(targetParams, 0, new Uint32Array([npix, M, s.batch, capacity]));
    const targetsGroup = device.createBindGroup({
      layout: targetsPipeline.getBindGroupLayout(0),
      entries: [targetParams, trainVerts, trainRadiance, tailPred, trainSet, trainCount].map((b, binding) => ({ binding, resource: { buffer: b } })),
    });
    const resolveParams = buf('nrc resolve params', 16, UNIFORM | COPY_DST);
    device.queue.writeBuffer(resolveParams, 0, new Uint32Array([pixels, 0, 0, 0]));
    const resolveGroup = device.createBindGroup({
      layout: resolvePipeline.getBindGroupLayout(0),
      entries: [resolveParams, queries, preds, queryCount, render.accumulator as GPUBuffer].map((b, binding) => ({ binding, resource: { buffer: b } })),
    });
    // nrcFinish uses bindings 0 and 5 only: its own bind group (auto layout drops the rest).
    const finishGroup = device.createBindGroup({ layout: finishPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: targetParams } }, { binding: 5, resource: { buffer: trainCount } }] });

    const cache = new NrcCache({
      device, render, trainer, mlp, settings: s, width: film.width, height: film.height, npix, hookParams, trainGroups, renderGroup, trainVerts, trainRadiance, tailPred, trainSet, trainCount, tailCount, queries, preds, queryCount, box,
      targetsPipeline, finishPipeline, resolvePipeline, targetsGroup, finishGroup, resolveGroup, tailInferGroup, renderInferGroup, trainHookGroup, resolveParams,
    });
    return cache;
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
    this.trainer.setCamera(camera);
  }

  /** Vertex the render paths terminate into the cache at (0 = primary hit). Clear the image after a change. */
  setQueryDepth(depth: number): void {
    const M = this.settings.trainDepth;
    this.device.queue.writeBuffer(this.hookParams, (M + 1) * HOOK_SLOT, new Uint32Array([0, depth, 0, this.width * this.height]));
  }

  /**
   * One frame: train (unless `train` is false), then one render sample pass with the cache. No readback.
   * `times`: 8 timestamp slots (trainer, training, render pass, inference + resolve).
   */
  encodeFrame(encoder: GPUCommandEncoder, options: { train?: boolean; times?: NrcStageTimes } = {}): void {
    const { device, settings: s, npix } = this;
    const M = s.trainDepth;
    const t = options.times;
    const stamp = (i: number): GPUComputePassTimestampWrites | undefined => (t ? { querySet: t.querySet, beginningOfPassWriteIndex: t.first + i, endOfPassWriteIndex: t.first + i + 1 } : undefined);
    this.frame++;
    if (options.train !== false) {
      encoder.clearBuffer(this.trainVerts);
      encoder.clearBuffer(this.trainCount);
      this.trainer.setSeed(s.seed * 7919 + this.frame);
      for (let k = 0; k <= M; k++) {
        this.trainer.reset(encoder);
        this.trainer.setHookBindGroup(this.trainGroups[k] as GPUBindGroup);
        // The start stamp rides on the first trainer pass (Dawn drops an empty pass's timestamps).
        this.trainer.encode(encoder, 1, t && k === 0 ? { querySet: t.querySet, beginningOfPassWriteIndex: t.first } : undefined);
        encoder.copyBufferToBuffer(this.trainer.accumulator as GPUBuffer, 0, this.trainRadiance, k * npix * 16, npix * 16);
      }
      this.mlp.encodeInfer(encoder, this.tailCount, 0, this.tailInferGroup);
      const pass = encoder.beginComputePass({ label: 'nrc targets', ...(t ? { timestampWrites: { querySet: t.querySet, endOfPassWriteIndex: t.first + 1 } } : {}) });
      pass.setPipeline(this.targetsPipeline);
      pass.setBindGroup(0, this.targetsGroup);
      pass.dispatchWorkgroups(Math.ceil((npix * M) / 64));
      pass.setPipeline(this.finishPipeline);
      pass.setBindGroup(0, this.finishGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      for (let i = 0; i < s.stepsPerFrame; i++) {
        const ts = t && (i === 0 || i === s.stepsPerFrame - 1) ? { querySet: t.querySet, ...(i === 0 ? { beginningOfPassWriteIndex: t.first + 2 } : {}), ...(i === s.stepsPerFrame - 1 ? { endOfPassWriteIndex: t.first + 3 } : {}) } : undefined;
        this.mlp.encodeTrainStep(encoder, this.trainCount, 0, this.trainHookGroup, ts);
      }
    }
    encoder.clearBuffer(this.queryCount);
    this.render.encode(encoder, 1, stamp(4));
    this.mlp.encodeInfer(encoder, this.queryCount, 0, this.renderInferGroup, t ? { querySet: t.querySet, beginningOfPassWriteIndex: t.first + 6 } : undefined);
    const pass = encoder.beginComputePass({ label: 'nrc resolve', ...(t ? { timestampWrites: { querySet: t.querySet, endOfPassWriteIndex: t.first + 7 } } : {}) });
    pass.setPipeline(this.resolvePipeline);
    pass.setBindGroup(0, this.resolveGroup);
    pass.dispatchWorkgroups(Math.ceil((this.width * this.height) / 64));
    pass.end();
    void device;
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
    this.trainer.destroy();
    this.mlp.destroy();
    for (const b of [this.hookParams, this.trainVerts, this.trainRadiance, this.tailPred, this.trainSet, this.trainCount, this.tailCount, this.queries, this.preds, this.queryCount, this.box, this.resolveParams]) b.destroy();
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
