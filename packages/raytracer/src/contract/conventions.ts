/**
 * Shared conventions for @pmndrs/raytracer, @pmndrs/pathtracer and the bench.
 *
 * This file is the single source of truth for spaces, signs and units. Sibling packages (Denoiser, upscaler,
 * reflections, gi) disagree on some of these; adapters convert at the edges and this core never invents a third
 * convention. See spec/SPEC-draft.md §3 and research/notes/01-local-ecosystem.md §6–7.
 */

/** World space: right-handed, +Y up, metres. View space looks down −Z. Matrices are column-major (WGSL mat4x4f). */
export const WORLD = { handedness: 'right', up: [0, 1, 0], units: 'metres', forwardView: [0, 0, -1] } as const;

/** Clip/NDC: x,y in [−1, 1] with +y up; depth z in [0, 1] (reversed-Z is a per-camera flag, never implied). */
export const NDC = { yUp: true, depthRange: [0, 1] } as const;

/**
 * Linear view depth: `viewZ` is negative in front of the camera (three.js / Denoiser convention).
 * Misses (sky) store `-far`. Consumers wanting positive distance negate in their adapter.
 */
export const VIEW_Z_SIGN = -1 as const;

/**
 * Motion vectors: NDC delta `current − previous`, +y up, with jitter removed.
 * To get a UV-space offset (prevUV = uv − motionUV), multiply by MOTION_TO_UV.
 */
export const MOTION = { space: 'ndc', direction: 'current-minus-previous', jitterFree: true } as const;
export const MOTION_TO_UV: readonly [number, number] = [0.5, -0.5];

/** Normals in AOVs are world-space unit vectors in [−1, 1]; misses write (0, 0, 0). OIDN adapters rotate to view. */
export const NORMAL_SPACE = 'world' as const;

/**
 * Sub-pixel jitter in pixels: x right, y down, each in [−0.5, 0.5]; the sample position is (i + 0.5 + jx, j + 0.5 + jy).
 * NDC offset is (2·jx / width, −2·jy / height). Exactly one jitter owner per pipeline: when @pmndrs/upscaler owns it
 * (Halton(2,3), 8·ratio² phases) the tracer's own AA offset must be zero.
 */
export const JITTER = { units: 'pixels', xRight: true, yDown: true, range: [-0.5, 0.5] } as const;

/** Images: row 0 is the top row in memory. Linear scene-referred HDR; no pre-exposure unless explicitly given. */
export const IMAGE = { rowOrder: 'top-down', colour: 'linear-scene-referred', accumulation: 'f32' } as const;

/**
 * Roughness: material `roughness` is perceptual (glTF / OpenPBR). GGX alpha = roughness².
 * Exporters to pbrt-v4 must write alpha with `remaproughness false` (pbrt's remap is alpha = sqrt(r), not r²).
 */
export function roughnessToAlpha(roughness: number): number {
  return roughness * roughness;
}

/** WGSL identifier prefixes reserved per package. WGSL has no namespaces, so never declare another package's prefix. */
export const WGSL_PREFIX = { bvh: 'bvh', gi: 'gi', raytracer: 'rt', pathtracer: 'pt' } as const;
