/**
 * Generated WGSL modules: the inline module (bvh walk + L0 + L1 + optional rtSurface) and the batch kernels (queue ABI
 * v0). All of it goes through the caps preprocessor (`buildWgsl`): `#if` blocks in wgsl-source.ts, caps and layer defines.
 *
 * Kernel size is capped (~50 KB, Windows DXC compile cliff, spec section 5); the tests check every combination.
 */
import { DEFAULT_NODE_FORMAT, TRACER_GPU_LAYOUT_VERSION, traceModuleEntries, traceModuleWgsl } from '@pmndrs/bvh';
import { BVH_LAYOUT_VERSION } from '../contract/ray.ts';
import type { AnyHitOrder, BindingSlot, ModuleBindingEntry } from '@pmndrs/bvh';
import { buildWgsl } from '../caps/wgsl.ts';
import type { Caps } from '../caps/detect.ts';
import type { Defines } from '../caps/wgsl.ts';
import { DEFAULT_BATCH_WORKGROUP_SIZE, STORAGE_BUFFER_BUDGET, type BatchWorkgroupSize } from './constants.ts';
import { surfaceVertexOffsets, type SurfaceLayout } from './surface-table.ts';
import { RT_CORE_WGSL, rtBatchArgsWgsl, rtBatchWgsl, rtSurfaceWgsl } from './wgsl-source.ts';

export type { AnyHitOrder, BindingSlot, ModuleBindingEntry };

/** The scene's formats, compiled into the walk. Read them off the buffers with `sceneFormatOf`. */
export interface SceneFormat {
  nodeFormat?: 'f32' | 'quantized';
  triangleLeaves?: 'records' | 'indexed';
  positionLayout?: 'vec4' | 'packed';
}

/**
 * The formats of bvh's `TraceSceneBuffers` (`SceneTrees.traceBuffers`, `uploadTraceScene`). Absent fields mean bvh's
 * defaults: quantized nodes (`DEFAULT_NODE_FORMAT`), record leaves, vec4 positions.
 */
export function sceneFormatOf(buffers: SceneFormat): Required<SceneFormat> {
  return {
    nodeFormat: buffers.nodeFormat ?? DEFAULT_NODE_FORMAT,
    triangleLeaves: buffers.triangleLeaves ?? 'records',
    positionLayout: buffers.positionLayout ?? 'vec4',
  };
}

export interface RtModuleOptions extends SceneFormat {
  /** Where bvh's walk bindings land. Default group 0, binding 0. */
  traceSlot?: BindingSlot;
  /** Include `rtSurface`. Its three bindings land at `slot`; `layout` is `SurfaceTable.layout`. Default: no surface. */
  surface?: { slot: BindingSlot; layout: SurfaceLayout } | false;
  /** Instance masks against ray masks (bvh's `rayMasks`). Default true. */
  rayMasks?: boolean;
  /** bvh's `bvhTrianglePasses`; default every triangle blocks, `null` to declare it yourself. */
  triangleFilter?: string | null;
  /** Any-hit child order (bvh's `anyHitOrder`). Default 'far'. */
  anyHitOrder?: AnyHitOrder;
}

/** Spec section 3: check the bvh layout version before generating any walk. Throws on a bvh build we were not written for. */
export function assertBvhLayout(version: number = TRACER_GPU_LAYOUT_VERSION): void {
  if (version !== BVH_LAYOUT_VERSION) {
    throw new Error(`@pmndrs/raytracer expects bvh GPU layout v${BVH_LAYOUT_VERSION}, the installed @pmndrs/bvh walk is v${version}`);
  }
}

function bvhWalk(options: RtModuleOptions & { traceSlot: BindingSlot }): string {
  assertBvhLayout();
  const format = sceneFormatOf(options);
  return traceModuleWgsl({
    traceSlot: options.traceSlot,
    rayMasks: options.rayMasks ?? true,
    triangleFilter: options.triangleFilter,
    anyHitOrder: options.anyHitOrder,
    triangleLeaves: format.triangleLeaves,
    positionLayout: format.positionLayout,
    nodeFormat: format.nodeFormat,
  });
}

function coreDefines(options: RtModuleOptions): Defines {
  const layout = options.surface ? options.surface.layout : null;
  return {
    RT_RAY_MASKS: options.rayMasks ?? true,
    RT_SURFACE: !!options.surface,
    RT_SURFACE_NORMALS: !!layout?.normals,
    RT_SURFACE_UVS: !!layout?.uvs,
    RT_SURFACE_POSITIONS: !!layout?.positions,
  };
}

function inlineSource(options: RtModuleOptions & { traceSlot: BindingSlot }): string {
  let source = `${bvhWalk(options)}\n${RT_CORE_WGSL}`;
  if (options.surface) {
    const offsets = surfaceVertexOffsets(options.surface.layout);
    source += rtSurfaceWgsl({
      group: options.surface.slot.group,
      firstBinding: options.surface.slot.firstBinding,
      vertexStride: Math.max(offsets.stride, 1),
      normalOffset: offsets.normal,
      uvOffset: offsets.uv,
      positionOffset: offsets.position,
    });
  }
  return source;
}

/**
 * The inline module: paste it into your own compute shader and call `rtVisible`, `rtClosest`, `rtClosestCone` and (with
 * `surface`) `rtSurface`. Bindings: bvh's walk at `traceSlot` (3, or 4 for indexed leaves), then the surface's 3.
 */
export function rtModuleWgsl(caps: Caps, options: RtModuleOptions = {}): string {
  const traceSlot = options.traceSlot ?? { group: 0, firstBinding: 0 };
  return buildWgsl(inlineSource({ ...options, traceSlot }), caps, coreDefines(options));
}

/** Bind group layout entries (without binding numbers) for the walk, as bvh's `traceModuleEntries`. */
export function rtWalkEntries(options: SceneFormat = {}): readonly ModuleBindingEntry[] {
  return traceModuleEntries({ triangleLeaves: sceneFormatOf(options).triangleLeaves });
}

/** The three `rtSurface` bindings, in order: table, indices, vertices. */
export function rtSurfaceEntries(): readonly ModuleBindingEntry[] {
  const entry: ModuleBindingEntry = { visibility: 0x4 /* GPUShaderStage.COMPUTE */, buffer: { type: 'read-only-storage' } };
  return [entry, entry, entry];
}

/** Storage buffers the module adds to a kernel (spec section 3 binding budget). */
export function rtModuleStorageBuffers(options: RtModuleOptions = {}): number {
  const walk = sceneFormatOf(options).triangleLeaves === 'indexed' ? STORAGE_BUFFER_BUDGET.walkIndexed : STORAGE_BUFFER_BUDGET.walk;
  return walk + (options.surface ? STORAGE_BUFFER_BUDGET.surface : 0);
}

// ---------------------------------------------------------------------------------------------------------------
// Batch kernels (queue ABI v0)
// ---------------------------------------------------------------------------------------------------------------

export type BatchMode = 'closest' | 'visible';
export type RayLayout = 'aos' | 'soa';

export interface BatchKernelOptions extends SceneFormat {
  mode: BatchMode;
  /** Workgroup size setting (D11). Default 64. */
  workgroupSize?: BatchWorkgroupSize;
  /** Ray buffer layout setting (D11): 'aos' one 32 B `Ray` array (contract; default) or 'soa' two vec4 arrays. */
  rayLayout?: RayLayout;
  /** GPU-side ray count (`rtBatchCounts`). Default false: the count is a uniform. */
  indirect?: boolean;
  /** Per-ray cones in, cone width at the hit out (closest mode only). Default false. */
  cones?: boolean;
  /** 40-byte records (default) or the 24-byte compact record without material and normal. */
  hitRecord?: 'full' | 'compact';
  rayMasks?: boolean;
  triangleFilter?: string | null;
  anyHitOrder?: AnyHitOrder;
  /** Device limit `maxComputeWorkgroupsPerDimension`, used by the indirect-args kernel. Default 65535. */
  maxWorkgroupsPerDimension?: number;
}

export interface ResolvedBatchOptions {
  mode: BatchMode;
  workgroupSize: BatchWorkgroupSize;
  rayLayout: RayLayout;
  indirect: boolean;
  cones: boolean;
  hitRecord: 'full' | 'compact';
  rayMasks: boolean;
  maxWorkgroupsPerDimension: number;
}

export function resolveBatchOptions(options: BatchKernelOptions): ResolvedBatchOptions {
  if (options.cones && options.mode === 'visible') throw new RangeError('rt batch: cones apply to closest mode only');
  const workgroupSize = options.workgroupSize ?? DEFAULT_BATCH_WORKGROUP_SIZE;
  return {
    mode: options.mode,
    workgroupSize,
    rayLayout: options.rayLayout ?? 'aos',
    indirect: options.indirect ?? false,
    cones: options.cones ?? false,
    hitRecord: options.hitRecord ?? 'full',
    rayMasks: options.rayMasks ?? true,
    maxWorkgroupsPerDimension: options.maxWorkgroupsPerDimension ?? 65535,
  };
}

/** The batch kernel (entry point `rtBatch`): scene in group 0 at binding 0, batch IO in group 1. */
export function rtBatchKernelWgsl(caps: Caps, options: BatchKernelOptions): string {
  const resolved = resolveBatchOptions(options);
  const moduleOptions = { ...options, traceSlot: { group: 0, firstBinding: 0 }, rayMasks: resolved.rayMasks, surface: false as const };
  const source = `${inlineSource(moduleOptions)}\n${rtBatchWgsl({ workgroupSize: resolved.workgroupSize, maxWorkgroupsPerDimension: resolved.maxWorkgroupsPerDimension })}`;
  return buildWgsl(source, caps, {
    ...coreDefines(moduleOptions),
    RT_MODE_CLOSEST: resolved.mode === 'closest',
    RT_MODE_VISIBLE: resolved.mode === 'visible',
    RT_INDIRECT: resolved.indirect,
    RT_CONES: resolved.cones,
    RT_RAYS_SOA: resolved.rayLayout === 'soa',
    RT_HIT_FULL: resolved.hitRecord === 'full',
  });
}

/** The one-thread kernel that writes DispatchIndirect args (entry point `rtBatchArgs`) from the GPU-side count. */
export function rtBatchArgsKernelWgsl(caps: Caps, options: Pick<BatchKernelOptions, 'workgroupSize' | 'maxWorkgroupsPerDimension'> = {}): string {
  return buildWgsl(
    rtBatchArgsWgsl({ workgroupSize: options.workgroupSize ?? DEFAULT_BATCH_WORKGROUP_SIZE, maxWorkgroupsPerDimension: options.maxWorkgroupsPerDimension ?? 65535 }),
    caps,
  );
}

/** Group 1 layout entries of a batch kernel, by binding number. Explicit so pipelines and bind groups never disagree. */
export function rtBatchEntries(options: ResolvedBatchOptions): GPUBindGroupLayoutEntry[] {
  const COMPUTE = 0x4;
  const storage = (binding: number, type: 'read-only-storage' | 'storage'): GPUBindGroupLayoutEntry => ({ binding, visibility: COMPUTE, buffer: { type } });
  const entries: GPUBindGroupLayoutEntry[] = [storage(0, 'read-only-storage')];
  if (options.rayLayout === 'soa') entries.push(storage(1, 'read-only-storage'));
  entries.push(storage(2, 'storage'));
  entries.push({ binding: 3, visibility: COMPUTE, buffer: { type: 'uniform' } });
  if (options.indirect) entries.push(storage(4, 'read-only-storage'));
  if (options.cones) {
    entries.push(storage(5, 'read-only-storage'));
    entries.push(storage(6, 'storage'));
  }
  return entries;
}
