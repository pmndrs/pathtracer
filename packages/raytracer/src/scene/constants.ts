/**
 * Layout of the scene copy (D4): one `array<vec4u>` table and one `array<u32>` data buffer. Every offset the WGSL
 * generators and the TypeScript packers use lives here, so a shader cannot disagree with a packer.
 *
 * Table (`SceneCopy.table`, bound as `rtSceneTable`), in vec4u units:
 *
 *     header        SCENE_HEADER_VEC4S (4)           see SCENE_HEADER
 *     instances     instanceSlots x 8 vec4u (128 B)  the surface-table instance record (trace/constants.ts), plus the light map words
 *     meshes        meshCount x 1 vec4u              (indexOffset, vertexOffset, flags, triangleCount)
 *     materials     (materialCount + 1) x 8 vec4u    128 B each; the last record is the default material
 *     lights        lightCount x 6 vec4u             96 B each; emissive triangles, then the sun, then the environment
 *     textures      textureCount x 1 vec4u           (bucket, layer, width | height << 16, mip levels); absent without textures (texture.ts)
 *
 * Data (`SceneCopy.data`, bound as `rtSceneData`), in u32 words: triangle vertex indices (mesh-local, original triangle
 * order), then vertex attributes (f32 bit patterns, interleaved as `SurfaceTable.vertices`), then the light maps of
 * instances with mixed emissive and plain triangles.
 *
 * Floats inside the table and the data are stored as bit patterns and read with `bitcast`, so no integer passes
 * through a float register and nothing is canonicalised.
 */

/** "No such entry": a light index, a texture slot in a u32 field, a missing environment map. */
export const SCENE_NONE = 0xffffffff;
/** A texture slot in a material's u16 field (base colour, roughness/metal, normal, emission): no texture. */
export const TEXTURE_SLOT_NONE = 0xffff;

/** Bump when any layout below changes. Part of the fingerprint. */
export const SCENE_COPY_VERSION = 1;

/** Storage buffers a kernel spends on the whole scene copy with `packing: 'split'` (spec section 3 budget). */
export const SCENE_COPY_STORAGE_BUFFERS = 2;
/** With `packing: 'packed'` the table and the data share one storage buffer. */
export const SCENE_COPY_PACKED_STORAGE_BUFFERS = 1;

/**
 * How the scene copy is bound (P3.11, `sceneCopy.packing`):
 *  - `split`: two storage buffers, `rtSceneTable: array<vec4u>` and `rtSceneData: array<u32>`.
 *  - `packed`: one storage buffer `rtScene: array<vec4u>` = [ header vec4 | table | data ], frees one binding in every kernel. Header vec4 `(dataBase, tableBase, dataWords, totalVec4s)`
 *    in vec4 units from the start of the buffer (`PACKED_HEADER`). The table words are byte-identical to `SceneCopy.table` (so every table offset in the
 *    scene header stays valid, only shifted by `PACKED_TABLE_BASE_VEC4S`, a compile-time constant); the data region starts at vec4 `dataBase` (16-byte
 *    aligned: the table is whole vec4s, the data is zero-padded to whole vec4s) and a data word `i` is `rtScene[dataBase + (i >> 2)][i & 3]`.
 *    `dataBase` is the buffer's first vec4 (one uniform-address load, L1/constant-cache hot), not a uniform buffer or a pipeline constant: a uniform would
 *    cost a binding (the thing this setting frees) and an `override` would recompile the kernel per scene.
 */
export type SceneCopyPacking = 'split' | 'packed';
/**
 * The library default. Stays `split` (the byte-identical pre-P3.11 kernels) until the quiet-pass A/B (docs/QUIET-PASS.md, `scene-packing` rows) is in; the
 * suggestion is `packed` when timing is a wash, because it frees a binding in every kernel.
 */
export const DEFAULT_SCENE_PACKING: SceneCopyPacking = 'split';
export const PACKED_HEADER_VEC4S = 1;
/** vec4 index of the table inside a packed buffer (a compile-time constant in the generated WGSL). */
export const PACKED_TABLE_BASE_VEC4S = 1;
/** Words of the packed header vec4. */
export const PACKED_HEADER = { dataBase: 0, tableBase: 1, dataWords: 2, totalVec4s: 3 } as const;

/** Storage buffers the scene copy adds to a kernel for this packing. */
export function sceneCopyStorageBuffers(packing: SceneCopyPacking = DEFAULT_SCENE_PACKING): number {
  return packing === 'packed' ? SCENE_COPY_PACKED_STORAGE_BUFFERS : SCENE_COPY_STORAGE_BUFFERS;
}

export const SCENE_HEADER_VEC4S = 4;
/** Word positions inside the header (4 vec4u = 16 words). */
export const SCENE_HEADER = {
  /** Instance slots (largest instance index + 1). */
  instanceSlots: 0,
  /** First mesh record, in vec4u units from the start of the table. */
  meshBase: 1,
  /** Floats per vertex in the vertex attributes. */
  vertexStride: 2,
  /** Word index in `data` where the vertex attributes start (= the index count). */
  vertexWordBase: 3,
  /** Materials the caller gave; record `materialCount` is the default material. */
  materialCount: 4,
  /** First material record, in vec4u units. */
  materialBase: 5,
  lightCount: 6,
  /** First light record, in vec4u units. */
  lightBase: 7,
  triangleLightCount: 8,
  /** Light index of the sun, or SCENE_NONE. */
  sunLight: 9,
  /** Light index of the environment, or SCENE_NONE. */
  envLight: 10,
  /** SCENE_FLAG bits. */
  flags: 11,
  /** f32: sum of all light powers (the alias table's normaliser), watts of luminance-weighted power. */
  totalPower: 12,
  /** f32: radius of the scene's bounding sphere, used for the power of the sun and the environment. */
  sceneRadius: 13,
  /** First texture slot record, in vec4u units (0 and unused without textures; texture.ts). */
  textureBase: 14,
  /** Texture slots in the table. */
  textureCount: 15,
} as const;

export const SCENE_FLAG = {
  /** Vertex positions are in the vertex attributes (barycentric `rtSurface` positions). */
  positions: 1,
} as const;

/** Extra words the scene copy writes into the 8-vec4u surface instance record (vec4u units, see trace/constants.ts SURFACE_TABLE). */
export const INSTANCE_LIGHT = {
  /** vec4u index inside the record that holds (meshIndex, valid, lightMode, lightBase): `SURFACE_TABLE.INSTANCE_META`. */
  metaLightMode: 2,
  metaLightBase: 3,
  /** vec4u index of the spare vec4u: word 0 = light map base (word index in `data`). */
  spareVec4: 7,
  mapBaseWord: 0,
} as const;

/** How an instance's triangles map to lights. */
export const LIGHT_MAP_MODE = {
  /** No emissive triangle. */
  none: 0,
  /** Every triangle is a light: light index = lightBase + triangle. */
  dense: 1,
  /** Mixed: light index = data[mapBase + triangle], SCENE_NONE for a plain triangle. */
  map: 2,
} as const;

export const LIGHT_KIND = { triangle: 0, sun: 1, environment: 2 } as const;
export const LIGHT_FLAG = { twoSided: 1 } as const;

export const LIGHT_RECORD_VEC4S = 6;
export const LIGHT_RECORD_BYTES = 96;
/**
 * Light record, 6 vec4u. Words (vec4 index . component):
 *
 *     0 xyz  triangle: world p0          sun: unit vector toward the sun        environment: 0
 *     0 w    kind (LIGHT_KIND)
 *     1 xyz  triangle: world edge1 = p1 - p0   sun: (angular radius, cos radius, solid angle)   environment: (scale, rotation about Y, 0)
 *     1 w    triangle: area (m^2)   sun, environment: 0
 *     2 xyz  triangle: world edge2 = p2 - p0   sun: 0   environment: (map base word in `data`, width | height << 16, pdf normaliser f32); 0 without a map
 *     2 w    selection probability (f32), power / totalPower
 *     3 xyz  triangle: unit normal of the emitting (authored front) side   sun, environment: 0
 *     3 w    triangle: material id   sun: SCENE_NONE   environment: environment map slot or SCENE_NONE
 *     4 xyz  triangle: emitted radiance (W/sr/m^2)   sun: irradiance on a surface facing it (W/m^2)   environment: radiance
 *     4 w    triangle: instance index   others: SCENE_NONE
 *     5 x    triangle: triangle index   others: SCENE_NONE
 *     5 y    flags (LIGHT_FLAG)
 *     5 z    alias table: acceptance probability (f32)
 *     5 w    alias table: alias light index
 */
export const LIGHT_WORD = {
  position: 0,
  kind: 3,
  edge1: 4,
  area: 7,
  sunAngularRadius: 4,
  sunCosRadius: 5,
  sunSolidAngle: 6,
  envScale: 4,
  envRotation: 5,
  edge2: 8,
  pmf: 11,
  normal: 12,
  material: 15,
  envMapSlot: 15,
  /** Environment map (see env-map.ts): word index in `data` of its first word, width | height << 16, and the f32 pdf normaliser. */
  envMapBase: 8,
  envMapSize: 9,
  envPdfNorm: 10,
  radiance: 16,
  instance: 19,
  triangle: 20,
  flags: 21,
  aliasProb: 22,
  alias: 23,
} as const;

/** Rec.709 luminance weights: light power, emission luminance and the sampling weights use them. */
export const LUMA = [0.2126, 0.7152, 0.0722] as const;
