/**
 * Batched interval tracer (P5.rc1): the dispatcher gi's Split Radiance Cascades calls. One compiled kernel, many dispatches per frame
 * (one per cascade, or one per probe batch), each with its own interval, jitter and seed (a dynamic-offset uniform ring, no CPU work
 * beyond one small `writeBuffer` per dispatch, nothing read back).
 *
 * Two ray sources:
 *  - `rays`: a GPU buffer of 32 B `RtRay`s whose `tMin` / `tMax` ARE the interval (Split RC spawns rays from visible surfaces and
 *    splits them; gi writes those rays itself).
 *  - `probes`: a buffer of `vec4f` probe positions (xyz, w = interval scale: 0 skips the probe, 1 = nominal; Split RC's LOD doubles it)
 *    and a direction bundle of `directions x directions` equal-area octahedral cells per probe. The ray `i` is probe `i / D`, cell
 *    `i % D` (D = directions^2, cell = (c % directions, c / directions)): a workgroup of 64 is one probe's directions when D >= 64
 *    (one origin, neighbouring directions: coherent rays, no ray buffer traffic) and the results come out in (probe, cell) order,
 *    ready for a merge that reads a probe's cells contiguously. The interval is `(t0, t1] * scale` from the dispatch.
 *
 * Outputs (`output`): `hit` (the 40 / 24 B closest-hit record, as `RtBatchTracer`), `distance` (one f32 per ray, `RT_MISS_T` = no hit
 * in the interval), `visible` (one u32, 1 = nothing in the interval; an any-hit walk) or `shaded` (one `vec4f` per ray: rtShade's
 * radiance at the hit, `w` = hit distance, `w = -1` and black when the interval is empty of geometry; one next-event light sample
 * per ray from a PCG hash of ray index and seed). Semantics of the interval: see `interval.ts`.
 *
 * Group 0: the scene (bvh walk bindings, as `RtBatchTracer`). Group 1: this batch's IO (`rtIntervalEntries`). Group 2: the scene copy
 * (`shaded` only, `rtSceneCopyLayoutEntries`). Kernel text size: the shaded kernel is built with the module minified (the shade module
 * with the scene copy is over the ~50 KB DXC class otherwise).
 */
import { traceSourceFormats, type TraceSceneBuffers } from '@pmndrs/bvh';
import type { Caps } from '../caps/detect.ts';
import { buildWgsl } from '../caps/wgsl.ts';
import { DEFAULT_SCENE_PACKING, type SceneCopyPacking } from '../scene/constants.ts';
import { rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, type SceneCopyBuffers } from '../scene/gpu.ts';
import { rtPackSceneAccess } from '../scene/scene-wgsl.ts';
import type { SurfaceLayout } from './surface-table.ts';
import type { TextureLayout } from '../scene/texture.ts';
import type { RtShadeMode, RtShadeOptions } from '../shade/shade.ts';
import { RAY_BYTES, HIT_BYTES, HIT_COMPACT_BYTES, DEFAULT_BATCH_WORKGROUP_SIZE, type BatchWorkgroupSize } from './constants.ts';
import { rtSceneBindGroupEntries, withBindings, type RtIndirectCount } from './batch.ts';
import type { IntervalConvention } from './interval.ts';
import { rtModuleWgsl, rtWalkEntries, sceneFormatOf, type ResolvedSceneFormat, type RtModuleOptions, type SceneFormat } from './modules.ts';
import { dispatchGrid } from './records.ts';
import { rtCreateComputePipelineAsync } from './wgsl-source-map.ts';

const STORAGE = 0x80;
const COPY_DST = 0x8;
const UNIFORM = 0x40;
const INDIRECT = 0x100;
const COMPUTE = 0x4;

export type IntervalSource = 'rays' | 'probes';
export type IntervalOutput = 'hit' | 'distance' | 'visible' | 'shaded';

/** Bytes of the `RtIntervalParams` uniform, and the stride of one slot of the dynamic-offset ring (min uniform offset alignment). */
export const INTERVAL_PARAMS_BYTES = 48;
export const INTERVAL_RING_STRIDE = 256;
/** Dispatches per frame before the ring wraps (`beginFrame` resets the cursor; a wrap inside one submit would overwrite a live slot, so it throws). */
export const INTERVAL_RING_SLOTS = 256;

export interface IntervalKernelOptions extends SceneFormat {
  source: IntervalSource;
  output: IntervalOutput;
  /** `open-low` (t0, t1] (default, the Split RC convention) or `closed-low` [t0, t1). */
  convention?: IntervalConvention;
  workgroupSize?: BatchWorkgroupSize;
  /** The item count (rays, or probes) comes from a GPU buffer (`RtIndirectCount`). Default false. */
  indirect?: boolean;
  /** `hit` output: 40 B record (default) or the 24 B compact one. */
  hitRecord?: 'full' | 'compact';
  rayMasks?: boolean;
  triangleFilter?: string | null;
  /** Device limit, used by the indirect-args kernel. Default 65535. */
  maxWorkgroupsPerDimension?: number;
  /** `shaded` output: the scene copy layout (`SceneCopy.layout`), and its textures / packing. Required for `shaded`. */
  scene?: { layout: SurfaceLayout; textures?: TextureLayout | null; packing?: SceneCopyPacking };
  /** `shaded`: rtShade's depth (default 'lambert') and options (the indirect hook: `{ cache: 'external' }` plus `hooksWgsl`). */
  shadeMode?: RtShadeMode;
  shade?: RtShadeOptions;
  /** `shaded`: WGSL appended before the kernel, for the shade hooks (`fn rtShadeIndirect(p: RtShadePoint) -> vec3f { ... }`). */
  hooksWgsl?: string;
  /** Minify the module text. Default: true for `shaded`, false otherwise. */
  minify?: boolean;
}

export interface ResolvedIntervalOptions {
  source: IntervalSource;
  output: IntervalOutput;
  convention: IntervalConvention;
  workgroupSize: BatchWorkgroupSize;
  indirect: boolean;
  hitRecord: 'full' | 'compact';
  rayMasks: boolean;
  maxWorkgroupsPerDimension: number;
  shadeMode: RtShadeMode;
}

export function resolveIntervalOptions(o: IntervalKernelOptions): ResolvedIntervalOptions {
  if (o.output === 'shaded' && !o.scene) throw new RangeError("rt interval: output 'shaded' needs the scene copy layout (options.scene)");
  if (o.output !== 'shaded' && (o.scene || o.shade || o.hooksWgsl)) throw new RangeError("rt interval: scene / shade / hooksWgsl apply to output 'shaded' only");
  return {
    source: o.source,
    output: o.output,
    convention: o.convention ?? 'open-low',
    workgroupSize: o.workgroupSize ?? DEFAULT_BATCH_WORKGROUP_SIZE,
    indirect: o.indirect ?? false,
    hitRecord: o.hitRecord ?? 'full',
    rayMasks: o.rayMasks ?? true,
    maxWorkgroupsPerDimension: o.maxWorkgroupsPerDimension ?? 65535,
    shadeMode: o.shadeMode ?? 'lambert',
  };
}

/** Bytes per ray in the result buffer. */
export function intervalResultBytes(output: IntervalOutput, hitRecord: 'full' | 'compact' = 'full'): number {
  switch (output) {
    case 'hit':
      return hitRecord === 'full' ? HIT_BYTES : HIT_COMPACT_BYTES;
    case 'shaded':
      return 16;
    default:
      return 4;
  }
}

/** Rays a job produces: `count` for the ray source, `count * directions^2` for probes. */
export function intervalRayCount(source: IntervalSource, count: number, directions: number): number {
  return source === 'probes' ? count * directions * directions : count;
}

/** One `RtIntervalParams` record (48 B). `t0` / `t1` / `jitter` apply to the probe source (a ray carries its own interval). */
export function packIntervalParams(p: { count: number; mask: number; capacity: number; countWord: number; t0: number; t1: number; directions: number; seed: number; jitter: readonly [number, number] }): ArrayBuffer {
  const buffer = new ArrayBuffer(INTERVAL_PARAMS_BYTES);
  const u = new Uint32Array(buffer);
  const f = new Float32Array(buffer);
  u[0] = p.count >>> 0;
  u[1] = p.mask >>> 0;
  u[2] = p.capacity >>> 0;
  u[3] = p.countWord >>> 0;
  f[4] = p.t0;
  f[5] = p.t1;
  u[6] = p.directions >>> 0;
  u[7] = p.seed >>> 0;
  f[8] = p.jitter[0];
  f[9] = p.jitter[1];
  return buffer;
}

function sourceDeclaration(o: ResolvedIntervalOptions): string {
  return o.source === 'rays' ? 'array<RtRay>' : 'array<vec4f>';
}

function resultDeclaration(o: ResolvedIntervalOptions): string {
  switch (o.output) {
    case 'hit':
      return 'array<RtIntervalHit>';
    case 'distance':
      return 'array<f32>';
    case 'visible':
      return 'array<u32>';
    default:
      return 'array<vec4f>';
  }
}

const HIT_STRUCT_FULL = `struct RtIntervalHit {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,
    bary: vec2f,
    material: u32,
    normalX: f32,
    normalY: f32,
    normalZ: f32,
}`;
const HIT_STRUCT_COMPACT = `struct RtIntervalHit {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,
    bary: vec2f,
}`;

/** The kernel text appended to the module. No preprocessor directives: the caller must not run it through `buildWgsl` again. */
export function rtIntervalEntryWgsl(o: ResolvedIntervalOptions): string {
  const wg = o.workgroupSize;
  const lo = o.convention === 'closed-low' ? 'Lo' : '';
  const probes = o.source === 'probes';
  const shaded = o.output === 'shaded';
  const modeConst = `RT_SHADE_${o.shadeMode.toUpperCase()}`;
  const items = o.indirect ? 'min(rtIvCounts[rtIv.countWord], rtIv.capacity)' : 'min(rtIv.count, rtIv.capacity)';
  const total = probes ? `${items} * rtIv.dirs * rtIv.dirs` : items;
  const query = o.output === 'visible'
    ? `    var visible = true;
    if (live) {
        visible = rtVisibleInterval${lo}(origin, direction, t0, t1, rtIv.mask);
    }
`
    : `    var hit = rtIntervalMiss();
    if (live) {
        hit = rtClosestInterval${lo}(origin, direction, t0, t1, rtIv.mask);
    }
`;
  let write: string;
  switch (o.output) {
    case 'hit':
      write = o.hitRecord === 'full'
        ? '    rtIvOut[i] = RtIntervalHit(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.bary, hit.material, hit.normal.x, hit.normal.y, hit.normal.z);\n'
        : '    rtIvOut[i] = RtIntervalHit(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.bary);\n';
      break;
    case 'distance':
      write = '    rtIvOut[i] = hit.t;\n';
      break;
    case 'visible':
      write = '    rtIvOut[i] = select(0u, 1u, visible);\n';
      break;
    default:
      write = `    var out = vec4f(0.0, 0.0, 0.0, RT_MISS_T);
    if (hit.t >= 0.0) {
        let shaded = rtShade(hit, RtRay(origin, 0.0, direction, hit.t), ${modeConst}, rtIvRandom(i), rtIv.mask);
        let c = rtShadeTotal(shaded);
        out = vec4f(select(vec3f(0.0), c, all(abs(c) < vec3f(3.0e38))), hit.t);
    }
    rtIvOut[i] = out;
`;
  }
  const rayFromSource = probes
    ? `    let d = rtIv.dirs * rtIv.dirs;
    let p = i / d;
    let c = i - p * d;
    let probe = rtIvSource[p];
    let origin = probe.xyz;
    let direction = rtOctEqualAreaDecode((vec2f(f32(c % rtIv.dirs), f32(c / rtIv.dirs)) + rtIv.jitter) / f32(rtIv.dirs));
    let t0 = rtIv.t0 * probe.w;
    let t1 = rtIv.t1 * probe.w;
    let live = probe.w > 0.0;
`
    : `    let ray = rtIvSource[i];
    let origin = ray.origin;
    let direction = ray.direction;
    let t0 = ray.tMin;
    let t1 = ray.tMax;
    let live = true;
`;
  return `
// ---- interval kernel (P5.rc1) ------------------------------------------------------------------------------------------

struct RtIntervalParams {
    count: u32,
    mask: u32,
    capacity: u32,
    countWord: u32,
    t0: f32,
    t1: f32,
    dirs: u32,
    seed: u32,
    jitter: vec2f,
    pad: vec2f,
}
${o.output === 'hit' ? (o.hitRecord === 'full' ? HIT_STRUCT_FULL : HIT_STRUCT_COMPACT) + '\n' : ''}
@group(1) @binding(0) var<uniform> rtIv: RtIntervalParams;
@group(1) @binding(1) var<storage, read> rtIvSource: ${sourceDeclaration(o)};
@group(1) @binding(2) var<storage, read_write> rtIvOut: ${resultDeclaration(o)};
${o.indirect ? '@group(1) @binding(3) var<storage, read> rtIvCounts: array<u32>;\n' : ''}${shaded ? `
fn rtIvPcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rtIvRandom(i: u32) -> vec3f {
    let a = rtIvPcg(i + rtIvPcg(rtIv.seed));
    let b = rtIvPcg(a);
    let c = rtIvPcg(b);
    return vec3f(f32(a >> 8u), f32(b >> 8u), f32(c >> 8u)) * (1.0 / 16777216.0);
}
` : ''}
@compute @workgroup_size(${wg})
fn rtInterval(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let i = (workgroup.y * workgroupCount.x + workgroup.x) * ${wg}u + localIndex;
    if (i >= ${total}) {
        return;
    }
${rayFromSource}${query}${write}}
`;
}

/** Everything the module generator needs for an interval kernel (the inline module's options). */
function moduleOptionsOf(options: IntervalKernelOptions, resolved: ResolvedIntervalOptions): RtModuleOptions {
  const shaded = resolved.output === 'shaded';
  const scene = shaded ? options.scene : undefined;
  return {
    ...(options.nodeFormat ? { nodeFormat: options.nodeFormat } : {}),
    ...(options.triangleLeaves ? { triangleLeaves: options.triangleLeaves } : {}),
    ...(options.positionLayout ? { positionLayout: options.positionLayout } : {}),
    traceSlot: { group: 0, firstBinding: 0 },
    rayMasks: resolved.rayMasks,
    triangleFilter: options.triangleFilter,
    intervals: true,
    ...(scene
      ? {
          scene: { slot: { group: 2, firstBinding: 0 }, layout: scene.layout, textures: scene.textures ?? null, packing: scene.packing ?? DEFAULT_SCENE_PACKING },
          shade: options.shade ?? true,
          minify: options.minify ?? true,
        }
      : { minify: options.minify ?? false }),
  };
}

/** The interval kernel (entry point `rtInterval`): scene in group 0, this batch's IO in group 1 (`rtIntervalEntries`), the scene copy in group 2 (shaded). */
export function rtIntervalKernelWgsl(caps: Caps, options: IntervalKernelOptions): string {
  const resolved = resolveIntervalOptions(options);
  const module = rtModuleWgsl(caps, moduleOptionsOf(options, resolved));
  let hooks = options.hooksWgsl ?? '';
  const packed = options.scene && (options.scene.packing ?? DEFAULT_SCENE_PACKING) === 'packed';
  if (hooks && packed) hooks = rtPackSceneAccess(hooks);
  const entry = rtIntervalEntryWgsl(resolved);
  return `${module}\n${hooks}${entry}`;
}

/** The one-thread kernel that turns a GPU-side item count into DispatchIndirect args (entry point `rtIntervalArgs`). Bindings: counts, args out, params (dynamic offset). */
export function rtIntervalArgsKernelWgsl(caps: Caps, o: Pick<ResolvedIntervalOptions, 'source' | 'workgroupSize' | 'maxWorkgroupsPerDimension'>): string {
  const perItem = o.source === 'probes' ? 'rtArgsParams.dirs * rtArgsParams.dirs' : '1u';
  return buildWgsl(
    /* wgsl */ `
struct RtIntervalParams {
    count: u32,
    mask: u32,
    capacity: u32,
    countWord: u32,
    t0: f32,
    t1: f32,
    dirs: u32,
    seed: u32,
    jitter: vec2f,
    pad: vec2f,
}
@group(0) @binding(0) var<storage, read> rtArgsCounts: array<u32>;
@group(0) @binding(1) var<storage, read_write> rtArgsOut: array<u32>;
@group(0) @binding(2) var<uniform> rtArgsParams: RtIntervalParams;

@compute @workgroup_size(1)
fn rtIntervalArgs() {
    let count = min(rtArgsCounts[rtArgsParams.countWord], rtArgsParams.capacity) * ${perItem};
    let groups = (count + ${o.workgroupSize - 1}u) / ${o.workgroupSize}u;
    let x = min(groups, ${o.maxWorkgroupsPerDimension}u);
    rtArgsOut[0] = x;
    rtArgsOut[1] = select(1u, (groups + x - 1u) / x, groups > 0u);
    rtArgsOut[2] = 1u;
}
`,
    caps,
  );
}

/** Group 1 layout entries of an interval kernel. */
export function rtIntervalEntries(o: ResolvedIntervalOptions): GPUBindGroupLayoutEntry[] {
  const entries: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: INTERVAL_PARAMS_BYTES } },
    { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
  ];
  if (o.indirect) entries.push({ binding: 3, visibility: COMPUTE, buffer: { type: 'read-only-storage' } });
  return entries;
}

/** One dispatch. Buffers are the caller's. */
export interface RtIntervalJob {
  /** `rays`: `RtRay` AoS (32 B each; tMin / tMax are the interval). `probes`: `vec4f` per probe (xyz position, w interval scale, 0 = skip). */
  readonly source: GPUBuffer;
  /** `intervalResultBytes(output)` per ray (`intervalRayCount` of them). */
  readonly results: GPUBuffer;
  /** Rays (source `rays`) or probes (source `probes`): a number, or a GPU-side count for an indirect tracer. */
  readonly count: number | RtIndirectCount;
  /** Ray mask (bvh instance masks). Default all ones. */
  readonly mask?: number;
  /** `probes`: the cascade's interval `(t0, t1]` before the per-probe scale, and `directions` cells per axis (D = directions^2 rays per probe). */
  readonly interval?: { readonly t0: number; readonly t1: number };
  readonly directions?: number;
  /** `probes`: position inside each cell in [0, 1)^2, 0.5 = centre. Default [0.5, 0.5]. Change it per frame for temporal coverage. */
  readonly jitter?: readonly [number, number];
  /** `shaded`: the random seed of the light samples. Default 0. */
  readonly seed?: number;
  /**
   * Use the previous dispatch's parameters (same slot of the ring, nothing written): for repeating one job back to back (timing, the same
   * cascade traced for several frames of a batch) without spending a slot each time. The buffers, mask and count may differ only if the
   * previous job's parameters still fit them; keep the job identical.
   */
  readonly reuseParams?: boolean;
  /** `shaded`: the scene copy buffers. */
  readonly sceneCopy?: SceneCopyBuffers;
  readonly timestampWrites?: GPUComputePassTimestampWrites;
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

interface IntervalShape {
  io: GPUBindGroup;
  args?: { buffer: GPUBuffer; group: GPUBindGroup };
}

const MAX_CACHED_SHAPES = 256;

/**
 * Traces interval jobs with one compiled kernel. Per frame: `beginFrame()`, then `encode` / `dispatch` as many jobs as needed (each
 * takes the next slot of the uniform ring; up to `INTERVAL_RING_SLOTS` per frame), one submit.
 */
export class RtIntervalTracer {
  readonly options: ResolvedIntervalOptions;
  readonly format: ResolvedSceneFormat;
  readonly label: string;
  readonly resultBytes: number;
  /** Kernel text length (the size cap is ~50 KB). */
  readonly kernelBytes: number;
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly sceneLayout: GPUBindGroupLayout;
  private readonly ioLayout: GPUBindGroupLayout;
  private readonly copyLayout: GPUBindGroupLayout | null;
  private readonly argsPipeline: GPUComputePipeline | null;
  private readonly argsLayout: GPUBindGroupLayout | null;
  private readonly ring: GPUBuffer;
  private cursor = 0;
  private readonly sceneGroups = new WeakMap<TraceSceneBuffers, GPUBindGroup>();
  private readonly copyGroups = new WeakMap<SceneCopyBuffers, GPUBindGroup>();
  private readonly shapes = new Map<string, IntervalShape>();
  private readonly owned: GPUBuffer[] = [];

  private constructor(device: GPUDevice, options: ResolvedIntervalOptions, format: ResolvedSceneFormat, label: string, bytes: number, parts: { pipeline: GPUComputePipeline; sceneLayout: GPUBindGroupLayout; ioLayout: GPUBindGroupLayout; copyLayout: GPUBindGroupLayout | null; argsPipeline: GPUComputePipeline | null; argsLayout: GPUBindGroupLayout | null }) {
    this.device = device;
    this.options = options;
    this.format = format;
    this.label = label;
    this.kernelBytes = bytes;
    this.resultBytes = intervalResultBytes(options.output, options.hitRecord);
    this.pipeline = parts.pipeline;
    this.sceneLayout = parts.sceneLayout;
    this.ioLayout = parts.ioLayout;
    this.copyLayout = parts.copyLayout;
    this.argsPipeline = parts.argsPipeline;
    this.argsLayout = parts.argsLayout;
    this.ring = device.createBuffer({ label: `${label} params ring`, size: INTERVAL_RING_STRIDE * INTERVAL_RING_SLOTS, usage: UNIFORM | COPY_DST });
  }

  static async create(device: GPUDevice, caps: Caps, options: IntervalKernelOptions & { label?: string }): Promise<RtIntervalTracer> {
    const resolved = resolveIntervalOptions({ ...options, maxWorkgroupsPerDimension: options.maxWorkgroupsPerDimension ?? device.limits.maxComputeWorkgroupsPerDimension });
    const format = sceneFormatOf(options);
    const label = options.label ?? `rt interval ${resolved.source} ${resolved.output}`;
    const sceneLayout = device.createBindGroupLayout({ label: `${label} scene`, entries: withBindings(rtWalkEntries(format)) });
    const ioLayout = device.createBindGroupLayout({ label: `${label} io`, entries: rtIntervalEntries(resolved) });
    const shaded = resolved.output === 'shaded';
    const copyLayout = shaded ? device.createBindGroupLayout({ label: `${label} scene copy`, entries: rtSceneCopyLayoutEntries(0, options.scene?.textures ?? undefined, options.scene?.packing ?? DEFAULT_SCENE_PACKING) }) : null;
    const code = rtIntervalKernelWgsl(caps, { ...options, ...resolved });
    const layouts = copyLayout ? [sceneLayout, ioLayout, copyLayout] : [sceneLayout, ioLayout];
    const pipeline = await rtCreateComputePipelineAsync(device, { label, layout: device.createPipelineLayout({ bindGroupLayouts: layouts }), code, entryPoint: 'rtInterval' });
    let argsPipeline: GPUComputePipeline | null = null;
    let argsLayout: GPUBindGroupLayout | null = null;
    if (resolved.indirect) {
      argsLayout = device.createBindGroupLayout({
        label: `${label} args`,
        entries: [
          { binding: 0, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
          { binding: 2, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: INTERVAL_PARAMS_BYTES } },
        ],
      });
      argsPipeline = await rtCreateComputePipelineAsync(device, { label: `${label} args`, layout: device.createPipelineLayout({ bindGroupLayouts: [argsLayout] }), code: rtIntervalArgsKernelWgsl(caps, resolved), entryPoint: 'rtIntervalArgs' });
    }
    return new RtIntervalTracer(device, resolved, format, label, code.length, { pipeline, sceneLayout, ioLayout, copyLayout, argsPipeline, argsLayout });
  }

  /** Resets the uniform ring: call once per frame, before the first job of the frame's submit. */
  beginFrame(): void {
    this.cursor = 0;
  }

  /** Opens a compute pass on `encoder` and records the job into it. */
  encode(encoder: GPUCommandEncoder, scene: TraceSceneBuffers, job: RtIntervalJob): void {
    const pass = encoder.beginComputePass({ label: this.label, timestampWrites: job.timestampWrites });
    this.dispatch(pass, scene, job);
    pass.end();
  }

  /** Records the job into an open pass (several jobs may share one pass). */
  dispatch(pass: GPUComputePassEncoder, scene: TraceSceneBuffers, job: RtIntervalJob): void {
    const o = this.options;
    const probes = o.source === 'probes';
    const indirect = typeof job.count !== 'number';
    if (indirect !== o.indirect) throw new RangeError(`${this.label}: tracer is ${o.indirect ? 'indirect' : 'direct'}; job.count must be ${o.indirect ? 'an RtIndirectCount' : 'a number'}`);
    const items = indirect ? 0 : (job.count as number);
    const capacity = indirect ? (job.count as RtIndirectCount).capacity : items;
    const countWord = indirect ? ((job.count as RtIndirectCount).word ?? 0) : 0;
    if (!Number.isInteger(capacity) || capacity < 0) throw new RangeError(`${this.label}: bad count/capacity ${capacity}`);
    let directions = 1;
    if (probes) {
      directions = job.directions ?? 0;
      if (!Number.isInteger(directions) || directions < 1) throw new RangeError(`${this.label}: probes need job.directions (cells per axis, >= 1)`);
      if (!job.interval) throw new RangeError(`${this.label}: probes need job.interval { t0, t1 }`);
    } else if (job.interval || job.directions) {
      throw new RangeError(`${this.label}: the rays source takes its interval from each ray; job.interval / job.directions apply to probes`);
    }
    if (!indirect && items === 0) return;
    if (job.reuseParams) {
      if (this.cursor === 0) throw new RangeError(`${this.label}: reuseParams needs an earlier dispatch since beginFrame()`);
    } else if (this.cursor >= INTERVAL_RING_SLOTS) {
      throw new RangeError(`${this.label}: more than ${INTERVAL_RING_SLOTS} jobs since beginFrame()`);
    }
    const shaded = o.output === 'shaded';
    if (shaded !== !!job.sceneCopy) throw new RangeError(`${this.label}: ${shaded ? 'the shaded output needs job.sceneCopy' : 'job.sceneCopy applies to the shaded output only'}`);

    const rayCapacity = intervalRayCount(o.source, capacity, directions);
    need(job.source, capacity * (probes ? 16 : RAY_BYTES), 'source', this.label);
    need(job.results, rayCapacity * this.resultBytes, 'results', this.label);

    const slot = job.reuseParams ? this.cursor - 1 : this.cursor++;
    const offset = slot * INTERVAL_RING_STRIDE;
    if (!job.reuseParams) {
      this.device.queue.writeBuffer(
        this.ring,
        offset,
        packIntervalParams({ count: items, mask: (job.mask ?? 0xffffffff) >>> 0, capacity, countWord, t0: job.interval?.t0 ?? 0, t1: job.interval?.t1 ?? 0, directions, seed: job.seed ?? 0, jitter: job.jitter ?? [0.5, 0.5] }),
      );
    }

    const shape = this.shapeFor(job, indirect);
    if (shape.args) {
      pass.setPipeline(this.argsPipeline as GPUComputePipeline);
      pass.setBindGroup(0, shape.args.group, [offset]);
      pass.dispatchWorkgroups(1);
    }
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.sceneGroup(scene));
    pass.setBindGroup(1, shape.io, [offset]);
    if (shaded) pass.setBindGroup(2, this.copyGroup(job.sceneCopy as SceneCopyBuffers));
    if (shape.args) {
      pass.dispatchWorkgroupsIndirect(shape.args.buffer, 0);
    } else {
      const [x, y] = dispatchGrid(intervalRayCount(o.source, items, directions), o.workgroupSize, o.maxWorkgroupsPerDimension);
      pass.dispatchWorkgroups(x, y);
    }
  }

  /** Frees the ring and the args buffers. The caller's buffers are untouched. */
  destroy(): void {
    this.ring.destroy();
    for (const buffer of this.owned) buffer.destroy();
    this.owned.length = 0;
    this.shapes.clear();
  }

  private sceneGroup(scene: TraceSceneBuffers): GPUBindGroup {
    let group = this.sceneGroups.get(scene);
    if (group) return group;
    const formats = traceSourceFormats(scene, 'RtIntervalTracer');
    const want = this.format;
    if (formats.nodeFormat !== want.nodeFormat || formats.triangleLeaves !== want.triangleLeaves || (want.triangleLeaves === 'indexed' && formats.positionLayout !== want.positionLayout)) {
      throw new RangeError(`${this.label}: compiled for ${want.nodeFormat}/${want.triangleLeaves}/${want.positionLayout}, scene is ${formats.nodeFormat}/${formats.triangleLeaves}/${formats.positionLayout}`);
    }
    group = this.device.createBindGroup({ label: `${this.label} scene`, layout: this.sceneLayout, entries: rtSceneBindGroupEntries(scene) });
    this.sceneGroups.set(scene, group);
    return group;
  }

  private copyGroup(buffers: SceneCopyBuffers): GPUBindGroup {
    let group = this.copyGroups.get(buffers);
    if (group) return group;
    group = this.device.createBindGroup({ label: `${this.label} scene copy`, layout: this.copyLayout as GPUBindGroupLayout, entries: rtSceneCopyBindGroupEntries(buffers) });
    this.copyGroups.set(buffers, group);
    return group;
  }

  private shapeFor(job: RtIntervalJob, indirect: boolean): IntervalShape {
    const counted = indirect ? (job.count as RtIndirectCount).buffer : null;
    const key = [job.source, job.results, counted].map((part) => (part ? `#${idOf(part)}` : '-')).join('|');
    const cached = this.shapes.get(key);
    if (cached) {
      this.shapes.delete(key);
      this.shapes.set(key, cached);
      return cached;
    }
    if (this.shapes.size >= MAX_CACHED_SHAPES) this.evictOldest();
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.ring, size: INTERVAL_PARAMS_BYTES } },
      { binding: 1, resource: { buffer: job.source } },
      { binding: 2, resource: { buffer: job.results } },
    ];
    if (counted) entries.push({ binding: 3, resource: { buffer: counted } });
    const shape: IntervalShape = { io: this.device.createBindGroup({ label: `${this.label} io`, layout: this.ioLayout, entries }) };
    if (counted) {
      const args = this.device.createBuffer({ label: `${this.label} indirect args`, size: 16, usage: STORAGE | INDIRECT });
      this.owned.push(args);
      shape.args = {
        buffer: args,
        group: this.device.createBindGroup({
          label: `${this.label} args`,
          layout: this.argsLayout as GPUBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: counted } },
            { binding: 1, resource: { buffer: args } },
            { binding: 2, resource: { buffer: this.ring, size: INTERVAL_PARAMS_BYTES } },
          ],
        }),
      };
    }
    this.shapes.set(key, shape);
    return shape;
  }

  private evictOldest(): void {
    const [key, shape] = this.shapes.entries().next().value as [string, IntervalShape];
    this.shapes.delete(key);
    if (shape.args) {
      shape.args.buffer.destroy();
      const at = this.owned.indexOf(shape.args.buffer);
      if (at >= 0) this.owned.splice(at, 1);
    }
  }
}

function need(buffer: GPUBuffer, bytes: number, what: string, label: string): void {
  if (buffer.size < bytes) throw new RangeError(`${label}: ${what} buffer is ${buffer.size} B, needs ${bytes} B`);
}

