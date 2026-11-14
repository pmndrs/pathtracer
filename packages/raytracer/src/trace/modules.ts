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
import { SCENE_COPY_STORAGE_BUFFERS, SCENE_HEADER_VEC4S } from '../scene/constants.ts';
import { rtSceneWgsl } from '../scene/scene-wgsl.ts';
import { rtShadeWgsl } from '../shade/shade-wgsl.ts';
import type { RtShadeOptions } from '../shade/shade.ts';

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
  /**
   * Include the scene copy (D4): `rtSurface` (unless `surface: false`), `rtMaterial` and the light functions, over the scene
   * copy's two bindings at `slot` (`rtSceneEntries`). `layout` is `SceneCopy.layout`. Default: no scene, no `rtSurface`.
   */
  scene?: { slot: BindingSlot; layout: SurfaceLayout; surface?: boolean; materials?: boolean; lights?: boolean; sampling?: boolean } | false;
  /**
   * Add L2 `rtShade` (P1.3, `shade/`): one-bounce shading with a hook for the indirect term. Needs `scene` with surface, materials and lights;
   * it adds the light point sampling (`rtSampleLightLi`, as `scene.sampling`), so do not append `rtLightSampleWgsl()` yourself. `true` = defaults.
   */
  shade?: RtShadeOptions | boolean;
  /**
   * Remove `//` and block comments from the generated text (line breaks stay, so compiler line numbers still match). The bvh
   * walk is 40 KB of mostly comments; with the scene copy the module would pass the ~50 KB kernel cap without this.
   * Default: true.
   */
  stripComments?: boolean;
  /**
   * Also drop the indentation and the spaces around `{ } ( ) , ;` (line breaks stay). About 17% of the text, applied last, to the
   * finished module only (append your own code through `rtMinifyWgsl` if you want it small too). Default false.
   * With `shade` and indexed leaves the module is 50.2 KB, over the ~50 KB kernel cap: turn this on (about 42.2 KB).
   */
  minify?: boolean;
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

/** Removes `//` line comments and block comments, keeping every line break (WGSL has no string literals, so no `//` is ever text). */
export function stripWgslComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, (block) => block.replace(/[^\n]/g, '')).replace(/[ \t]*\/\/[^\n]*/g, '');
}

/** Indentation and the spaces around `{ } ( ) , ;` removed; line breaks kept. Only safe on comment-free, preprocessed source (no `#` lines). */
export function rtMinifyWgsl(code: string): string {
  return code.replace(/^[ \t]+/gm, '').replace(/[ \t]*([{}(),;])[ \t]*/g, '$1');
}

function surfaceLayoutOf(options: RtModuleOptions): SurfaceLayout | null {
  if (options.scene && (options.scene.surface ?? true)) return options.scene.layout;
  return null;
}

function coreDefines(options: RtModuleOptions): Defines {
  const layout = surfaceLayoutOf(options);
  return {
    RT_RAY_MASKS: options.rayMasks ?? true,
    RT_SURFACE: !!layout,
    RT_SURFACE_NORMALS: !!layout?.normals,
    RT_SURFACE_UVS: !!layout?.uvs,
    RT_SURFACE_POSITIONS: !!layout?.positions,
  };
}

function inlineSource(options: RtModuleOptions & { traceSlot: BindingSlot }): string {
  let source = `${bvhWalk(options)}\n${RT_CORE_WGSL}`;
  const layout = surfaceLayoutOf(options);
  if (options.scene) {
    const { slot } = options.scene;
    const shade = !!options.shade;
    if (shade && (options.scene.surface === false || options.scene.materials === false || options.scene.lights === false)) {
      throw new Error('rtModuleWgsl: shade needs scene surface, materials and lights');
    }
    source += rtSceneWgsl({ group: slot.group, firstBinding: slot.firstBinding, materials: options.scene.materials, lights: options.scene.lights, sampling: options.scene.sampling || shade });
  } else if (options.shade) {
    throw new Error('rtModuleWgsl: shade needs the scene copy (scene option)');
  }
  if (layout) {
    const offsets = surfaceVertexOffsets(layout);
    const base = { vertexStride: Math.max(offsets.stride, 1), normalOffset: offsets.normal, uvOffset: offsets.uv, positionOffset: offsets.position };
    source += rtSurfaceWgsl({ ...base, headerVec4s: SCENE_HEADER_VEC4S });
  }
  if (options.shade) source += rtShadeWgsl(options.shade === true ? {} : options.shade);
  return source;
}

/**
 * The inline module: paste it into your own compute shader and call `rtVisible`, `rtClosest`, `rtClosestCone` and (with
 * `scene`) `rtSurface`, the materials and the lights, and (with `shade`) `rtShade`. Bindings: bvh's walk at `traceSlot` (3, or 4 for indexed leaves), then the scene copy's 2.
 */
export function rtModuleWgsl(caps: Caps, options: RtModuleOptions = {}): string {
  const traceSlot = options.traceSlot ?? { group: 0, firstBinding: 0 };
  const source = inlineSource({ ...options, traceSlot });
  const code = buildWgsl((options.stripComments ?? true) ? stripWgslComments(source) : source, caps, coreDefines(options));
  return options.minify ? rtMinifyWgsl(code) : code;
}

/** Bind group layout entries (without binding numbers) for the walk, as bvh's `traceModuleEntries`. */
export function rtWalkEntries(options: SceneFormat = {}): readonly ModuleBindingEntry[] {
  return traceModuleEntries({ triangleLeaves: sceneFormatOf(options).triangleLeaves });
}

/** The scene copy's two bindings, in order: table, data. */
export function rtSceneEntries(): readonly ModuleBindingEntry[] {
  const entry: ModuleBindingEntry = { visibility: 0x4 /* GPUShaderStage.COMPUTE */, buffer: { type: 'read-only-storage' } };
  return [entry, entry];
}

/** Storage buffers the module adds to a kernel (spec section 3 binding budget). */
export function rtModuleStorageBuffers(options: RtModuleOptions = {}): number {
  const walk = sceneFormatOf(options).triangleLeaves === 'indexed' ? STORAGE_BUFFER_BUDGET.walkIndexed : STORAGE_BUFFER_BUDGET.walk;
  return walk + (options.scene ? SCENE_COPY_STORAGE_BUFFERS : 0);
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
  const moduleOptions = { ...options, traceSlot: { group: 0, firstBinding: 0 }, rayMasks: resolved.rayMasks };
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
