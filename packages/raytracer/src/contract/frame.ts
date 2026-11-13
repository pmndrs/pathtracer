/**
 * Camera and AOV contract. `FrameCamera` is the superset already used by @pmndrs/reflections
 * (`packages/core/src/frame/contract.ts`) and the Denoiser's `temporal.ts`, so all siblings can consume it.
 */

export type Mat4 = Float32Array | ArrayLike<number>;

export interface FrameCamera {
  /** World → view, column-major. */
  view: Mat4;
  /** Unjittered projection. */
  projection: Mat4;
  /** Projection with the current jitter applied; omit when jitter is zero. */
  projectionJittered?: Mat4;
  prevView: Mat4;
  prevProjection: Mat4;
  prevViewProjection: Mat4;
  /** Pixels, x right / y down, in [−0.5, 0.5]. See conventions.JITTER. */
  jitter: readonly [number, number];
  near: number;
  far: number;
  position: readonly [number, number, number];
  reversedDepth: boolean;
}

/**
 * AOVs the tracer can write. Names are stable identifiers used by the bench, the Denoiser and the upscaler adapters.
 * Formats are defaults; precision rules: accumulate in f32, store history/AOVs in f16 where the caps allow.
 */
export const AOV = {
  /** Linear HDR radiance (beauty). rgba16float display, rgba32float accumulation. */
  radiance: 'radiance',
  /** Demodulated diffuse radiance (radiance / max(albedo, 1e-3)). */
  diffuse: 'diffuse',
  /** Demodulated specular radiance (divided by the split-sum E term). */
  specular: 'specular',
  /** First-scatter diffuse albedo, [0, 1]. Accumulated by the tracer, not from a raster G-buffer. */
  albedo: 'albedo',
  /** First-scatter specular albedo (split-sum E). */
  specularAlbedo: 'specularAlbedo',
  /** World-space normal at first scatter; (0,0,0) on miss. */
  normal: 'normal',
  /** View-space Z, negative in front; −far on miss. */
  viewZ: 'viewZ',
  /** NDC delta current − previous, jitter-free. */
  motion: 'motion',
  /** Perceptual roughness at first scatter. */
  roughness: 'roughness',
  /** Hit distance of the first diffuse / specular bounce (denoiser guides). */
  hitDistanceDiffuse: 'hitDistanceDiffuse',
  hitDistanceSpecular: 'hitDistanceSpecular',
  /** Instance id at first hit (u32), for disocclusion and debugging. */
  objectId: 'objectId',
} as const;
export type AovName = (typeof AOV)[keyof typeof AOV];
