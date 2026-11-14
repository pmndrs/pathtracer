/**
 * Sizes, offsets and flag values shared by the WGSL generators, the TypeScript packers and the tests. WGSL text is
 * generated from these numbers, so a shader can never disagree with a packer about an offset.
 */
import { HIT_BYTES, HIT_COMPACT_BYTES, RAY_BYTES, RAY_CONE_BYTES } from '../contract/ray.ts';

export { HIT_BYTES, HIT_COMPACT_BYTES, RAY_BYTES, RAY_CONE_BYTES };

/** `0xffffffff`: what `RtHit.material` holds when neither the triangle nor the instance has a material. */
export const RT_NO_MATERIAL = 0xffffffff;
/** All 32 ray-mask bits set: every instance whose 8-bit mask is non-zero is visible. */
export const RT_MASK_ALL = 0xffffffff;

/** Word positions inside a 40-byte closest-hit record in a batch result buffer (`HIT_BYTES`; bvh layout v2). */
export const HIT_WORD = { t: 0, instance: 1, triangle: 2, frontFace: 3, barycentrics: 4, material: 6, normal: 7 } as const;

/** `SurfaceMask` bits of `rtSurface(hit, ray, want)`. Geometric normal, front-face flag and material id are free (they come with the hit). */
export const SURFACE = {
  /** World position. From the ray (`origin + t * direction`) unless the table was built with `positions: true`. */
  POSITION: 1,
  /** Geometric normal and front-face flag (already in the hit; the bit exists so callers state what they need). */
  GEOMETRIC_NORMAL: 2,
  /** Interpolated shading normal, world space, on the geometric normal's side. Falls back to the geometric normal. */
  SHADING_NORMAL: 4,
  /** Interpolated UV. Zero when the mesh has none. */
  UV: 8,
  /** Material id (index into our own table). */
  MATERIAL: 16,
  /**
   * Texture frame (P1.4c): `uvDensity` (sqrt(uv area / world area), for ray-cone level of detail) and `tangent` (world dp/du, xyz unit, w =
   * bitangent handedness). Opt-in: not part of ALL, and only compiled when the scene has textures (needs positions and uvs in the surface data).
   */
  TEXTURE_FRAME: 32,
  ALL: 31,
} as const;

/** Surface table (`SurfaceTable.table`, read as `array<vec4u>`): header, then instance records, then mesh records. */
export const SURFACE_TABLE = {
  /** u32 words per vec4. */
  VEC4_WORDS: 4,
  /** Header vec4u: (instanceSlots, meshBase in vec4u units, vertexStride in floats, reserved). */
  HEADER_VEC4S: 1,
  /** Instance record, in vec4u units: 3 rows of object-to-world (xyz + translation), 3 rows of the normal matrix (xyz), meta, spare. */
  INSTANCE_VEC4S: 8,
  /** Words inside an instance record, in vec4u units. */
  INSTANCE_O2W: 0,
  INSTANCE_NORMAL: 3,
  INSTANCE_META: 6,
  /** Mesh record, in vec4u units: (indexOffset, vertexOffset, flags, triangleCount). */
  MESH_VEC4S: 1,
  /** Bytes per instance slot. */
  INSTANCE_BYTES: 128,
} as const;

/** Mesh-record flag bits. */
export const MESH_FLAG = { HAS_NORMALS: 1, HAS_UVS: 2 } as const;

/** Batch kernel workgroup sizes offered as a setting (D11). Default 64; 128 is the other candidate. */
export const BATCH_WORKGROUP_SIZES = [32, 64, 128, 256] as const;
export type BatchWorkgroupSize = (typeof BATCH_WORKGROUP_SIZES)[number];
export const DEFAULT_BATCH_WORKGROUP_SIZE: BatchWorkgroupSize = 64;

/** Budget published per module (spec section 3): storage buffers each part adds to a kernel's bindings. */
export const STORAGE_BUFFER_BUDGET = {
  /** bvh walk: nodes, triangles, top level (+1 positions for indexed leaves). */
  walk: 3,
  walkIndexed: 4,
  /** Batch kernel group 1: rays (+1 SoA), results (+1 counts when indirect, +2 cones). */
  batchMin: 2,
} as const;
