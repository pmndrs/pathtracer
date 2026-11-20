/**
 * `buildGpuScene`: a `SceneCopyInput` (from `loadGltf` or your own meshes) -> everything a kernel binds: the bvh trace buffers and the
 * uploaded scene copy, with the formats the kernel must be compiled for. The two-step form (bvh `buildTraceScene` + `uploadTraceScene`,
 * `buildSceneCopy` + `uploadSceneCopy`) stays available for callers that rebuild one half alone.
 *
 * Every instance is static and merged into one tree (bvh `mergeStaticInstances`), the scene copy stores vertex positions (`rtSurface`
 * computes hit positions from barycentrics, the path tracer's default), and the trees are built on the CPU.
 */
import { buildTraceScene, createMeshBvhBuilder, uploadTraceScene } from '@pmndrs/bvh';
import type { NodeFormat, TraceSceneBuffers } from '@pmndrs/bvh';
import { sceneFormatOf } from '../trace/modules.ts';
import type { ResolvedSceneFormat } from '../trace/modules.ts';
import { uploadSceneCopy } from './gpu.ts';
import type { RtMipGenerator, SceneCopyBuffers } from './gpu.ts';
import { buildSceneCopy } from './scene-copy.ts';
import type { SceneCopy, SceneCopyInput } from './scene-copy.ts';
import type { SceneCopyPacking } from './constants.ts';

export interface GpuSceneOptions {
  /** Buffer labels. Default `'scene'`. */
  label?: string;
  /** bvh node format: `quantized` (default), `f32`, or the experimental wide `bvh4` / `cwbvh8`. Kernels must be compiled for it (`format`). */
  nodeFormat?: NodeFormat;
  /** Scene copy binding (`DEFAULT_SCENE_PACKING` when omitted). Kernels must be compiled with the same `scenePacking`. */
  packing?: SceneCopyPacking;
  /** Hit positions from the triangle's vertices (default true) or from `origin + t * direction` (false: a smaller scene copy). */
  positions?: boolean;
  /** Fills texture mip levels 1+ (default: a per-device `@pmndrs/downsampler`). */
  mipGenerator?: RtMipGenerator;
}

/** What a kernel binds for one scene. Call `destroy()` when the scene is no longer drawn. */
export interface GpuScene {
  readonly input: SceneCopyInput;
  /** The CPU scene copy (layout, texture layout, bounds, light table). */
  readonly copy: SceneCopy;
  /** bvh trace buffers (`setScene(trace, buffers)` on a path tracer kernel; `rtSceneBindGroupEntries(trace)` for your own). */
  readonly trace: TraceSceneBuffers;
  /** The uploaded scene copy. */
  readonly buffers: SceneCopyBuffers;
  /** The formats a kernel walking this scene is compiled for (`format` of `PtMegakernel`, `RtBatchTracer`, `rtModuleWgsl`). */
  readonly format: ResolvedSceneFormat;
  readonly packing: SceneCopyPacking;
  destroy(): void;
}

/** Builds the trees and the scene copy of `input` on the CPU and uploads both. */
export async function buildGpuScene(device: GPUDevice, input: SceneCopyInput, options: GpuSceneOptions = {}): Promise<GpuScene> {
  const label = options.label ?? 'scene';
  const builder = createMeshBvhBuilder();
  let data;
  try {
    data = await buildTraceScene(
      input.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : null })),
      input.instances.map((instance) => ({ ...instance, static: true })),
      { build: (geometry) => builder.build(geometry), mergeStaticInstances: true, ...(options.nodeFormat ? { nodeFormat: options.nodeFormat } : {}) },
    );
  } finally {
    builder.dispose();
  }
  const trace = uploadTraceScene(device, data, label);
  const copy = buildSceneCopy({ ...input, options: { ...input.options, surface: { ...input.options?.surface, positions: options.positions ?? true } } });
  const buffers = uploadSceneCopy(device, copy, { label: `${label} copy`, ...(options.packing ? { packing: options.packing } : {}), ...(options.mipGenerator ? { mipGenerator: options.mipGenerator } : {}) });
  return {
    input, copy, trace, buffers, packing: buffers.packing,
    format: sceneFormatOf({ nodeFormat: trace.nodeFormat, triangleLeaves: trace.triangleLeaves, positionLayout: trace.positionLayout }),
    destroy() {
      trace.destroy();
      buffers.destroy();
    },
  };
}
