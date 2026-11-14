/**
 * The one interface every renderer sits behind in the bench: ours, web competitors (mutracer,
 * three-gpu-pathtracer, Rayzee, …) and offline references (pbrt-v4, Mitsuba 3, Cycles, via a Node-side adapter that
 * runs the exporter and reads the PFM back). Results are always linear HDR float32, row 0 = top, so the metrics and
 * the A/B viewer never care where an image came from.
 */
import type { MaterialDesc, SceneDesc } from '../scenes/schema.ts';

export type Backend = 'webgpu' | 'webgl2' | 'cpu-native' | 'gpu-native';

export interface RendererInfo {
  /** Stable id used in result rows, e.g. "ours-mega", "mutracer", "tgpt-wavefront", "pbrt-v4-cpu". */
  id: string;
  name: string;
  version: string;
  /** Git commit or package version of the renderer under test. */
  commit?: string;
  backend: Backend;
  /** Free-form configuration that changes output or speed; hashed into the result row. */
  config: Record<string, unknown>;
}

/** Which parts of the scene schema the renderer can represent exactly. Unsupported scenes are skipped, not faked. */
export interface FeatureSupport {
  materials: ReadonlyArray<'diffuse' | 'conductor' | 'dielectric' | 'black' | 'pbr' | 'openpbr'>;
  /** Renders `SceneDesc.textures` (image-textured materials). Absent = false: textured scenes are skipped. */
  textures?: boolean;
  emissiveMeshes: boolean;
  directionalLight: boolean;
  environment: ReadonlyArray<'uniform' | 'hdri'>;
  /**
   * Per-material check beyond the `materials` type list (e.g. an `openpbr` material that uses a lobe the renderer does not implement):
   * a reason string skips the scene, null accepts it.
   */
  materialUnsupported?: (material: MaterialDesc) => string | null;
  /** Can load `file` mesh geometry (Sponza). Absent = true. */
  fileGeometry?: boolean;
  /** Can the renderer be told to disable its denoiser / tonemapper / own jitter? Required for raw comparisons. */
  rawOutput: boolean;
  notes?: string;
}

export interface RenderBudget {
  /** Stop after this many samples per pixel. */
  spp?: number;
  /** Stop after this much wall-clock time. */
  ms?: number;
  /** Return snapshots at these spp counts (for convergence curves). Must be ascending. */
  snapshotsAtSpp?: number[];
  /** Fixed seed for deterministic runs. */
  seed?: number;
}

export interface Image {
  width: number;
  height: number;
  /** 3 (RGB) or 4 (RGBA; alpha ignored by metrics). */
  channels: 3 | 4;
  /** Linear HDR float32, row-major, row 0 = top. */
  data: Float32Array;
}

export interface Timings {
  /** Wall-clock from first render call to the result being ready on the GPU (no readback inside). */
  wallMs: number;
  /** Sum of GPU timestamp intervals, if the renderer exposes them. */
  gpuMs?: number;
  /** Named GPU passes, if available. */
  passes?: Record<string, number>;
  /** Whether Chrome's timestamp quantization was active (100 µs granularity). */
  timestampsQuantized?: boolean;
}

export interface RenderResult {
  image: Image;
  spp: number;
  timings: Timings;
  snapshots?: Array<{ spp: number; wallMs: number; image: Image }>;
  /** Anything else worth logging (ray counts, memory, compile time). */
  stats?: Record<string, number>;
}

export interface RendererAdapter {
  readonly info: RendererInfo;
  readonly features: FeatureSupport;
  /** Build acceleration structures, compile pipelines, upload textures. Timed separately as load time. */
  load(scene: SceneDesc, options?: { device?: GPUDevice; baseUrl?: string }): Promise<{ loadMs: number; compileMs?: number }>;
  render(budget: RenderBudget): Promise<RenderResult>;
  dispose(): void;
}
