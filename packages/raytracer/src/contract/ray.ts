/**
 * Ray and hit records. These mirror @pmndrs/bvh GPU layout v2 (`~/Developer/bvh/src/trace/layout.ts`,
 * `TRACER_GPU_LAYOUT_VERSION = 2`) so a ray buffer can go straight to `bvhTraceClosest/Any` without repacking.
 * Anything past barycentrics (UV, shading normal, tangent, material params) is decoded by this package, not by bvh.
 */

/** bvh GPU layout version this contract matches. Check it at start-up against the bvh build in use. */
export const BVH_LAYOUT_VERSION = 2;

/** `Ray { origin: vec3f, tMin: f32, direction: vec3f (normalized), tMax: f32 }`, 32 bytes. */
export interface Ray {
  origin: readonly [number, number, number];
  tMin: number;
  /** Unit length. */
  direction: readonly [number, number, number];
  tMax: number;
}
export const RAY_BYTES = 32;

/**
 * Closest-hit record (bvh batch layout, 40 bytes):
 * `t, instance, triangle, frontFace (u32), bary.xy, material, normal.xyz`.
 * `t < 0` means miss. `barycentrics` are the weights of v1 and v2 (v0 = 1 − u − v).
 * `triangle` is the original (source-order) triangle index within the instance's mesh.
 * `material` is bvh's 16-bit per-triangle/instance id: treat it as an index into our own material table.
 */
export interface Hit {
  t: number;
  instance: number;
  triangle: number;
  frontFace: boolean;
  barycentrics: readonly [number, number];
  material: number;
  /** World-space geometric normal, unit length. */
  normal: readonly [number, number, number];
}
export const HIT_BYTES = 40;
export const HIT_COMPACT_BYTES = 24;
export const MISS_T = -1;

/** Largest material id bvh can carry (16 bits, 0xffff reserved). */
export const BVH_MAX_MATERIAL = 65534;
