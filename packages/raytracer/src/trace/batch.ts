/**
 * Batch dispatcher, queue ABI v0: a GPU ray buffer plus a count (direct, or a GPU-side count for indirect dispatch)
 * goes in, one result per ray comes out (closest-hit record or visibility word). Everything is recorded into the
 * caller's encoder or pass; nothing is read back and nothing is written from the CPU once a batch shape has been seen
 * (params uniforms and bind groups are cached per shape).
 *
 * Also here: the GPU helpers the inline path needs (scene bindings), so a caller writing their own kernel
 * with `rtModuleWgsl` binds exactly what the batch kernel binds.
 */
import { traceSourceFormats, type TraceSceneBuffers } from '@pmndrs/bvh';
import type { Caps } from '../caps/detect.ts';
import { RAY_BYTES, RAY_CONE_BYTES } from './constants.ts';
import {
  rtBatchArgsKernelWgsl,
  rtBatchEntries,
  rtBatchKernelWgsl,
  rtWalkEntries,
  resolveBatchOptions,
  sceneFormatOf,
  type BatchKernelOptions,
  type ModuleBindingEntry,
  type ResolvedBatchOptions,
  type SceneFormat,
} from './modules.ts';
import { BATCH_PARAMS_BYTES, dispatchGrid, packBatchParams, resultBytes } from './records.ts';
import { rtCreateComputePipelineAsync } from './wgsl-source-map.ts';

// GPUBufferUsage bits, spelled out so this module loads in Node (tests) where the WebGPU globals do not exist.
const STORAGE = 0x80; // GPUBufferUsage.STORAGE
const COPY_DST = 0x8; // GPUBufferUsage.COPY_DST
const UNIFORM = 0x40; // GPUBufferUsage.UNIFORM
const INDIRECT = 0x100; // GPUBufferUsage.INDIRECT

/** Puts binding numbers on module entries, starting at `firstBinding`. */
export function withBindings(entries: readonly ModuleBindingEntry[], firstBinding = 0): GPUBindGroupLayoutEntry[] {
  return entries.map((entry, i) => ({ ...entry, binding: firstBinding + i }) as GPUBindGroupLayoutEntry);
}

/** The walk's bind group entries for a bvh scene (nodes, triangles, top level, and positions for indexed leaves). */
export function rtSceneBindGroupEntries(scene: TraceSceneBuffers, firstBinding = 0): GPUBindGroupEntry[] {
  const buffers = [scene.nodes, scene.triangles, scene.topLevel];
  if (scene.triangleLeaves === 'indexed') {
    if (!scene.positions) throw new RangeError('rt: an indexed scene has no positions buffer');
    buffers.push(scene.positions);
  }
  return buffers.map((buffer, i) => ({ binding: firstBinding + i, resource: { buffer } }));
}

/** A GPU-side ray count (indirect dispatch). The batch traces `min(buffer[word], capacity)` rays. */
export interface RtIndirectCount {
  readonly buffer: GPUBuffer;
  /** u32 index of the count inside `buffer`. Default 0. */
  readonly word?: number;
  /** Upper bound: the size the ray and result buffers were made for. */
  readonly capacity: number;
}

/** One batch. Buffers are the caller's; the tracer never reallocates them. */
export interface RtBatch {
  /** AoS: one buffer of 32 B `Ray`s. SoA tracer: `[originTMin, directionTMax]`, 16 B per ray each. */
  readonly rays: GPUBuffer | readonly [GPUBuffer, GPUBuffer];
  /** Closest: `resultBytes('closest', hitRecord)` per ray. Visible: one u32 per ray, 1 = visible. */
  readonly results: GPUBuffer;
  /** Direct: the ray count. Indirect tracer: a GPU-side count. */
  readonly count: number | RtIndirectCount;
  /** Ray mask for the whole batch (bvh instance masks). Default all ones. */
  readonly mask?: number;
  /** Cone tracer only: per-ray cones in (8 B each), cone width at the hit out (one f32 each, 0 on a miss). */
  readonly cones?: { readonly cones: GPUBuffer; readonly widths: GPUBuffer };
  /** For `encode` only: timestamps around the batch's pass. */
  readonly timestampWrites?: GPUComputePassTimestampWrites;
}

export interface RtBatchTracerOptions extends BatchKernelOptions {
  label?: string;
}

let nextId = 1;
const ids = new WeakMap<object, number>();
function idOf(object: object): number {
  let id = ids.get(object);
  if (id === undefined) {
    id = nextId++;
    ids.set(object, id);
  }
  return id;
}

interface BatchShape {
  params: GPUBuffer;
  io: GPUBindGroup;
  /** Indirect: args buffer and the args kernel's bind group. */
  args?: { buffer: GPUBuffer; group: GPUBindGroup };
}

/**
 * Traces batches with one compiled kernel (mode, layout, workgroup size, scene formats are compile-time). Group 0 is
 * the scene (bvh's walk bindings); group 1 the batch IO (`rtBatchEntries`).
 */
export class RtBatchTracer {
  readonly options: ResolvedBatchOptions;
  readonly format: Required<SceneFormat>;
  readonly label: string;
  /** Bytes per result. */
  readonly resultBytes: number;
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly sceneLayout: GPUBindGroupLayout;
  private readonly ioLayout: GPUBindGroupLayout;
  private readonly argsPipeline: GPUComputePipeline | null;
  private readonly sceneGroups = new WeakMap<TraceSceneBuffers, GPUBindGroup>();
  private readonly shapes = new Map<string, BatchShape>();
  private readonly owned: GPUBuffer[] = [];

  private constructor(
    device: GPUDevice,
    options: ResolvedBatchOptions,
    format: Required<SceneFormat>,
    label: string,
    pipeline: GPUComputePipeline,
    sceneLayout: GPUBindGroupLayout,
    ioLayout: GPUBindGroupLayout,
    argsPipeline: GPUComputePipeline | null,
  ) {
    this.device = device;
    this.options = options;
    this.format = format;
    this.label = label;
    this.pipeline = pipeline;
    this.sceneLayout = sceneLayout;
    this.ioLayout = ioLayout;
    this.argsPipeline = argsPipeline;
    this.resultBytes = resultBytes(options.mode, options.hitRecord);
  }

  static async create(device: GPUDevice, caps: Caps, options: RtBatchTracerOptions): Promise<RtBatchTracer> {
    const resolved = resolveBatchOptions({ ...options, maxWorkgroupsPerDimension: options.maxWorkgroupsPerDimension ?? device.limits.maxComputeWorkgroupsPerDimension });
    const format = sceneFormatOf(options);
    const label = options.label ?? `rt batch ${resolved.mode}`;
    const sceneLayout = device.createBindGroupLayout({ label: `${label} scene`, entries: withBindings(rtWalkEntries(format)) });
    const ioLayout = device.createBindGroupLayout({ label: `${label} io`, entries: rtBatchEntries(resolved) });
    const code = rtBatchKernelWgsl(caps, { ...options, ...resolved });
    const pipeline = await rtCreateComputePipelineAsync(device, { label, layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }), code, entryPoint: 'rtBatch' });
    let argsPipeline: GPUComputePipeline | null = null;
    if (resolved.indirect) {
      const argsCode = rtBatchArgsKernelWgsl(caps, resolved);
      argsPipeline = await rtCreateComputePipelineAsync(device, { label: `${label} args`, layout: 'auto', code: argsCode, entryPoint: 'rtBatchArgs' });
    }
    return new RtBatchTracer(device, resolved, format, label, pipeline, sceneLayout, ioLayout, argsPipeline);
  }

  /** Opens a compute pass on `encoder` and records the batch (and, indirect, its args dispatch) into it. */
  encode(encoder: GPUCommandEncoder, scene: TraceSceneBuffers, batch: RtBatch): void {
    const pass = encoder.beginComputePass({ label: this.label, timestampWrites: batch.timestampWrites });
    this.dispatch(pass, scene, batch);
    pass.end();
  }

  /**
   * Records the batch into an open pass. Several batches may share one pass. Indirect batches dispatch the one-thread
   * args kernel first; WebGPU orders the two dispatches (storage write, then indirect read).
   */
  dispatch(pass: GPUComputePassEncoder, scene: TraceSceneBuffers, batch: RtBatch): void {
    const shape = this.shapeFor(batch);
    if (shape === null) return; // direct, zero rays
    if (shape.args) {
      pass.setPipeline(this.argsPipeline as GPUComputePipeline);
      pass.setBindGroup(0, shape.args.group);
      pass.dispatchWorkgroups(1);
    }
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.sceneGroup(scene));
    pass.setBindGroup(1, shape.io);
    if (shape.args) {
      pass.dispatchWorkgroupsIndirect(shape.args.buffer, 0);
    } else {
      const [x, y] = dispatchGrid(batch.count as number, this.options.workgroupSize, this.options.maxWorkgroupsPerDimension);
      pass.dispatchWorkgroups(x, y);
    }
  }

  /** Frees the uniforms and args buffers the tracer made. The caller's buffers are untouched. */
  destroy(): void {
    for (const buffer of this.owned) buffer.destroy();
    this.owned.length = 0;
    this.shapes.clear();
  }

  private sceneGroup(scene: TraceSceneBuffers): GPUBindGroup {
    let group = this.sceneGroups.get(scene);
    if (group) return group;
    const formats = traceSourceFormats(scene, 'RtBatchTracer');
    const want = this.format;
    if (formats.nodeFormat !== want.nodeFormat || formats.triangleLeaves !== want.triangleLeaves || (want.triangleLeaves === 'indexed' && formats.positionLayout !== want.positionLayout)) {
      throw new RangeError(
        `${this.label}: compiled for ${want.nodeFormat}/${want.triangleLeaves}/${want.positionLayout}, scene is ${formats.nodeFormat}/${formats.triangleLeaves}/${formats.positionLayout}`,
      );
    }
    group = this.device.createBindGroup({ label: `${this.label} scene`, layout: this.sceneLayout, entries: rtSceneBindGroupEntries(scene) });
    this.sceneGroups.set(scene, group);
    return group;
  }

  private shapeFor(batch: RtBatch): BatchShape | null {
    const o = this.options;
    const indirect = typeof batch.count !== 'number';
    if (indirect !== o.indirect) throw new RangeError(`${this.label}: tracer is ${o.indirect ? 'indirect' : 'direct'}; batch.count must be ${o.indirect ? 'an RtIndirectCount' : 'a number'}`);
    const count = indirect ? 0 : (batch.count as number);
    const capacity = indirect ? (batch.count as RtIndirectCount).capacity : count;
    const countWord = indirect ? ((batch.count as RtIndirectCount).word ?? 0) : 0;
    if (!Number.isInteger(capacity) || capacity < 0) throw new RangeError(`${this.label}: bad ray count/capacity ${capacity}`);
    if (!indirect && count === 0) return null;
    const mask = (batch.mask ?? 0xffffffff) >>> 0;

    // Sizes: every buffer must hold `capacity` entries.
    const soa = o.rayLayout === 'soa';
    const rays = batch.rays;
    if (soa !== Array.isArray(rays)) throw new RangeError(`${this.label}: tracer takes ${soa ? 'SoA rays [a, b]' : 'one AoS ray buffer'}`);
    const rayBuffers: GPUBuffer[] = soa ? [...(rays as readonly [GPUBuffer, GPUBuffer])] : [rays as GPUBuffer];
    for (const buffer of rayBuffers) need(buffer, capacity * (soa ? 16 : RAY_BYTES), 'rays', this.label);
    need(batch.results, capacity * this.resultBytes, 'results', this.label);
    if (o.cones !== !!batch.cones) throw new RangeError(`${this.label}: tracer ${o.cones ? 'needs' : 'takes no'} cones`);
    if (batch.cones) {
      need(batch.cones.cones, capacity * RAY_CONE_BYTES, 'cones', this.label);
      need(batch.cones.widths, capacity * 4, 'cone widths', this.label);
    }

    const counted = indirect ? (batch.count as RtIndirectCount).buffer : null;
    const key = [count, mask, capacity, countWord, batch.results, ...rayBuffers, counted, batch.cones?.cones, batch.cones?.widths]
      .map((part) => (part === null || part === undefined ? '-' : typeof part === 'object' ? `#${idOf(part)}` : String(part)))
      .join('|');
    const cached = this.shapes.get(key);
    if (cached) {
      // Most recently used last, so eviction drops the oldest shape.
      this.shapes.delete(key);
      this.shapes.set(key, cached);
      return cached;
    }
    if (this.shapes.size >= MAX_CACHED_SHAPES) this.evictOldest();

    const params = this.device.createBuffer({ label: `${this.label} params`, size: BATCH_PARAMS_BYTES, usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(params, 0, packBatchParams({ count, mask, capacity, countWord }));
    this.owned.push(params);
    const entries: GPUBindGroupEntry[] = [];
    if (soa) {
      entries.push({ binding: 0, resource: { buffer: rayBuffers[0] as GPUBuffer } }, { binding: 1, resource: { buffer: rayBuffers[1] as GPUBuffer } });
    } else {
      entries.push({ binding: 0, resource: { buffer: rayBuffers[0] as GPUBuffer } });
    }
    entries.push({ binding: 2, resource: { buffer: batch.results } }, { binding: 3, resource: { buffer: params } });
    if (counted) entries.push({ binding: 4, resource: { buffer: counted } });
    if (batch.cones) entries.push({ binding: 5, resource: { buffer: batch.cones.cones } }, { binding: 6, resource: { buffer: batch.cones.widths } });
    const shape: BatchShape = { params, io: this.device.createBindGroup({ label: `${this.label} io`, layout: this.ioLayout, entries }) };
    if (counted) {
      const args = this.device.createBuffer({ label: `${this.label} indirect args`, size: 16, usage: STORAGE | INDIRECT });
      this.owned.push(args);
      const argsPipeline = this.argsPipeline as GPUComputePipeline;
      shape.args = {
        buffer: args,
        group: this.device.createBindGroup({
          label: `${this.label} args`,
          layout: argsPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: counted } },
            { binding: 1, resource: { buffer: args } },
            { binding: 2, resource: { buffer: params } },
          ],
        }),
      };
    }
    this.shapes.set(key, shape);
    return shape;
  }

  private evictOldest(): void {
    const [key, shape] = this.shapes.entries().next().value as [string, BatchShape];
    this.shapes.delete(key);
    for (const buffer of [shape.params, shape.args?.buffer]) {
      if (!buffer) continue;
      buffer.destroy();
      const at = this.owned.indexOf(buffer);
      if (at >= 0) this.owned.splice(at, 1);
    }
  }
}

/** Batch shapes (buffers, count, mask) whose uniforms and bind groups stay cached; the oldest beyond this are freed. */
export const MAX_CACHED_SHAPES = 256;

function need(buffer: GPUBuffer, bytes: number, what: string, label: string): void {
  if (buffer.size < bytes) throw new RangeError(`${label}: ${what} buffer is ${buffer.size} B, needs ${bytes} B`);
}
