/**
 * Neutral scene description for the bench. Every renderer (ours, web competitors, pbrt-v4, Mitsuba 3, Cycles)
 * renders from this one description via an exporter or adapter, so differences come from the renderers, not the
 * scenes.
 *
 * Rules that keep comparisons exact:
 * - **Everything is triangles.** Spheres/boxes/quads are tessellated by `bench/scenes/geometry.ts` and every
 *   renderer receives the same triangles (no analytic spheres in pbrt/Mitsuba), so BVH-vs-analytic differences can't
 *   leak into error numbers.
 * - **Bench material subset only** in quantitative scenes: each type below has an exact equivalent in pbrt-v4 and
 *   Mitsuba 3. Anything richer (OpenPBR/Principled) is compared separately and marked as such.
 * - Units: metres, right-handed, +Y up (see packages/raytracer/src/contract/conventions.ts). Exporters to
 *   left-handed pbrt handle the flip.
 * - Pixel filter is a box of radius 0.5 (one sample = one pixel footprint) for every renderer.
 * - Radiometric units: emission/env values are radiance in W·sr⁻¹·m⁻² (linear RGB, Rec.709 primaries).
 */

export type Vec3 = readonly [number, number, number];
export type Rgb = readonly [number, number, number];
/** Column-major 4×4 object-to-world transform. */
export type Mat4 = readonly number[];

export interface SceneDesc {
  version: 1;
  name: string;
  /** One line: what this scene tests (e.g. "white furnace: energy conservation of rough dielectric"). */
  purpose: string;
  /** Tier: 0 analytic/correctness, 1 material coverage, 2 quality+speed, 3 speed under load. */
  tier: 0 | 1 | 2 | 3;
  camera: CameraDesc;
  film: FilmDesc;
  integrator: IntegratorDesc;
  materials: Record<string, MaterialDesc>;
  /** Procedural textures by name (generated in code by `bench/scenes/textures.ts`; nothing is downloaded). Materials reference them by name. */
  textures?: Record<string, TextureDesc>;
  meshes: MeshDesc[];
  lights?: LightDesc[];
  environment?: EnvironmentDesc;
  /** Analytic expectation, when one exists (furnace tests): checked by `bench/metrics`. */
  expected?: ExpectedDesc;
}

export interface CameraDesc {
  type: 'perspective';
  position: Vec3;
  target: Vec3;
  up: Vec3;
  /** Vertical field of view in degrees (exporters convert for pbrt's shorter-axis rule and Mitsuba's fov_axis). */
  yFovDeg: number;
}

export interface FilmDesc {
  width: number;
  height: number;
  /** Always 'box' radius 0.5 for quantitative comparisons. */
  filter: 'box';
}

export interface IntegratorDesc {
  /** Maximum path length counted in scattering events (pbrt `maxdepth` semantics: 1 = direct lighting only). */
  maxDepth: number;
  /**
   * Russian roulette starts at this path vertex (1 = first surface hit; RR may terminate the path after scattering at
   * vertex `rrStartDepth` and later ones). Omit to disable RR (reference default: disabled for Tier 0). pbrt-v4:
   * `rrdepth = rrStartDepth - 1` (patched); Mitsuba: `rr_depth = rrStartDepth`.
   */
  rrStartDepth?: number;
}

/**
 * A procedural RGBA8 texture, row 0 = top, uv (0, 0) = its top-left texel (glTF convention). `srgb` textures hold colours (base colour,
 * emission); `linear` ones hold data (metallic-roughness, normal). All sizes are powers of two in the reference scenes so every renderer
 * reads the same texels (ours would resample others, see raytracer `scene/texture.ts`).
 */
export type TextureDesc = (
  /** `cells` x `cells` checker of colours `a` and `b` (0-255 sRGB codes); `marker` paints the top-left cell of each tile `markerColor` so the orientation is visible. */
  | { generator: 'checker'; cells: number; a: readonly [number, number, number]; b: readonly [number, number, number]; marker?: readonly [number, number, number]; /** Alpha (0-255) of the `a` and `b` cells (default 255; the marker cell is always 255). For alpha-mask scenes. */ alphaA?: number; alphaB?: number }
  /** One colour everywhere (0-255 codes; a `linear` texture holds raw bytes). */
  | { generator: 'solid'; color: readonly [number, number, number, number?] }
  /** Tangent-space normal map: a sine bump pattern `periods` times across, slope `amplitude` (tan of the maximum tilt). */
  | { generator: 'bumps'; periods: number; amplitude: number }
  /** Metallic-roughness (glTF: G = roughness, B = metalness): vertical stripes, `periods` of them, alternating (roughness, metalness) `a` and `b` (0-1). */
  | { generator: 'stripes-mr'; periods: number; a: readonly [number, number]; b: readonly [number, number] }
  /**
   * An image of a glTF file (`path` relative to bench/assets, `image` = index in the file's `images`; PNG or JPEG), decoded by the Node
   * loader (`gltf-images.ts`, which registers itself with `textures.ts` when imported). `width`/`height` are the decoded size. The loader
   * has already applied `KHR_texture_transform` to the uvs; it never changes the texels.
   */
  | { generator: 'image'; path: string; image: number; sha256?: string }
) & {
  width: number;
  height: number;
  colorSpace: 'srgb' | 'linear';
  /** Default `repeat`. glTF has separate S and T wraps: the loader uses S and records a note when T differs. */
  wrap?: 'repeat' | 'clamp' | 'mirror';
  /** Magnification filter the source asked for (glTF sampler). Default `linear`. Informational: the scene copy always samples bilinearly with mips. */
  filter?: 'linear' | 'nearest';
};

/**
 * Parameters of the `openpbr` material. Names are OpenPBR 1.1.2's (`base_color` -> `baseColor`, `specular_roughness_anisotropy` ->
 * `specularRoughnessAnisotropy`, ...); every field is optional and absent ones take the OpenPBR default (listed per field, applied by
 * `resolveOpenPbr`). Colours are linear Rec.709. Units: lengths are **metres** (our scene unit; OpenPBR's own default is
 * centimetres, so a literal OpenPBR/MaterialX export multiplies by 100), thin-film thickness is **micrometres** (OpenPBR's unit),
 * emission is radiance `emissionLuminance * emissionColor` in W/sr/m^2 (OpenPBR: nits).
 */
export interface OpenPbrParams {
  /** Default 1. */ baseWeight?: number;
  /** Default 0.8 grey. */ baseColor?: Rgb;
  /** Default 0. */ baseMetalness?: number;
  /** Default 0 (Lambert). */ baseDiffuseRoughness?: number;
  /** Default 1. Scales the dielectric specular lobe (F at both ends, unlike glTF's F0-only scale). */ specularWeight?: number;
  /** Default white. */ specularColor?: Rgb;
  /** Perceptual; GGX alpha = roughness^2 (before anisotropy). Default 0.3. */ specularRoughness?: number;
  /** Default 0. 1 gives alpha_y = 0 (OpenPBR: alpha_x = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_y = (1 - a) alpha_x). */ specularRoughnessAnisotropy?: number;
  /** Default 1.5. */ specularIor?: number;
  /** Default 0. */ transmissionWeight?: number;
  /** Default white. At `transmissionDepth` 0 a non-physical tint of the refraction; otherwise the Beer-Lambert colour at that depth. */ transmissionColor?: Rgb;
  /** Metres. Default 0 = no medium. */ transmissionDepth?: number;
  /** Default 0 (no dispersion). */ transmissionDispersionScale?: number;
  /** Default 20. */ transmissionDispersionAbbeNumber?: number;
  /** Default 0. glTF `KHR_materials_diffuse_transmission` maps here (thin-walled). */ subsurfaceWeight?: number;
  /** Default white. */ subsurfaceColor?: Rgb;
  /** Default 0. */ coatWeight?: number;
  /** Default white. */ coatColor?: Rgb;
  /** Default 0. */ coatRoughness?: number;
  /** OpenPBR's default is 1.6; the glTF mapping sets 1.5 (`KHR_materials_clearcoat` is fixed at 1.5). */ coatIor?: number;
  /** Default 0. */ fuzzWeight?: number;
  /** Default white. */ fuzzColor?: Rgb;
  /** Default 0.5. */ fuzzRoughness?: number;
  /** Default 0. */ thinFilmWeight?: number;
  /** Micrometres. Default 0.5. */ thinFilmThickness?: number;
  /** Default 1.4. */ thinFilmIor?: number;
  /** Default 0. */ emissionLuminance?: number;
  /** Default white. */ emissionColor?: Rgb;
  /** Default false. */ geometryThinWalled?: boolean;
  /** Default 1. */ geometryOpacity?: number;
  // --- bench extras (not OpenPBR parameters; the scene copy has fields for them) ---
  /** Radians, rotation of the anisotropy axis about the normal (glTF `anisotropyRotation`); OpenPBR takes the axis from the tangent field. Default 0. */ anisotropyRotation?: number;
  /** Default 'opaque'. 'mask' tests `geometryOpacity` (times the base colour alpha texture) against `alphaCutoff`; 'blend' is stochastic. */ alphaMode?: 'opaque' | 'mask' | 'blend';
  /** Default 0.5. */ alphaCutoff?: number;
  /** Default false. */ doubleSided?: boolean;
  /** Emit from both faces (default false: authored front face). */ emissionTwoSided?: boolean;
  /** `textures` keys (as for `pbr`); the factors above multiply them. */
  baseColorTexture?: string;
  metallicRoughnessTexture?: string;
  normalTexture?: string;
  emissiveTexture?: string;
}

export type MaterialDesc =
  /** Lambertian. `reflectanceTexture` (a `textures` key, sRGB) multiplies `reflectance`. */
  | { type: 'diffuse'; reflectance: Rgb; reflectanceTexture?: string }
  /**
   * OpenPBR 1.1.2 subset (docs/design/tier1.md section 2): the Tier 1 material. Fields and defaults are in `OpenPbrParams`
   * (`bench/scenes/openpbr.ts` resolves the defaults). Ours renders what the scene copy and BSDF implement today and ignores the
   * rest; the exporters (pbrt, Mitsuba) do not read it yet (P2.20). `gltfMaterialToOpenPbr` (`gltf-openpbr.ts`) produces it from glTF.
   */
  | ({ type: 'openpbr' } & OpenPbrParams)
  /**
   * glTF-style metallic-roughness (ours only: pbrt-v4 and Mitsuba cannot express it exactly; exporters throw). Textures multiply the factors
   * (base colour rgb, roughness from G and metalness from B of `metallicRoughnessTexture`, `emissive` rgb); `normalTexture` is tangent space.
   *
   * Kept for the procedural-texture scenes (`tier1/textured.ts`, `texture-check`). It is the `openpbr` type restricted to
   * base colour / metalness / roughness / emission plus the same four texture slots, so new scenes should use `openpbr`; `pbr`
   * can be removed once those two scene files move over (TASKS P2.17 follow-up).
   */
  | {
      type: 'pbr';
      baseColor: Rgb;
      roughness: number;
      metalness: number;
      emissive?: Rgb;
      baseColorTexture?: string;
      metallicRoughnessTexture?: string;
      normalTexture?: string;
      emissiveTexture?: string;
    }
  /** GGX conductor. `roughness` is perceptual; alpha = roughness² (conventions.roughnessToAlpha). */
  | { type: 'conductor'; eta: Rgb; k: Rgb; roughness: number }
  /** GGX dielectric (smooth when roughness = 0). Interior is vacuum-free: no absorption. */
  | { type: 'dielectric'; ior: number; roughness: number }
  /** Perfectly black, non-scattering (light sources' backing, absorbers). */
  | { type: 'black' };

export type GeometryDesc =
  | { type: 'quad'; /** Size in local X and Z; normal is local +Y. */ size: readonly [number, number] }
  | { type: 'box'; size: Vec3 }
  | { type: 'sphere'; radius: number; /** UV-sphere tessellation; same triangles go to every renderer. */ segments: number; rings: number }
  | { type: 'triangles'; positions: readonly number[]; indices: readonly number[]; normals?: readonly number[] }
  /** External file, path relative to bench/assets (downloaded, not in git). Formats: glb/gltf, ply, obj. */
  | {
      type: 'file'; path: string; format: 'glb' | 'gltf' | 'ply' | 'obj'; sha256?: string;
      /** glTF only: only the triangles of this glTF material index (-1 = primitives without a material), merged over all nodes. Default: everything. */
      select?: { material: number };
      /** glTF only (`gltf-scene.ts` options): `normals` default `geometric`; `uvs` = the material's texture coordinates (with `KHR_texture_transform` applied); `tangents`, `colors` (COLOR_0 as rgba) are carried on the `TriangleMesh` for consumers that read them. */
      attributes?: { normals?: 'geometric' | 'shading'; uvs?: boolean; tangents?: boolean; colors?: boolean };
    };

export interface MeshDesc {
  name: string;
  geometry: GeometryDesc;
  material: string;
  transform?: Mat4;
  /** Makes this mesh a diffuse area emitter with this radiance on its front face. */
  emission?: Rgb;
  /** Emit from both faces (default false: front face only, as pbrt `diffuse` area light with twosided false). */
  twoSided?: boolean;
  /**
   * Give the tessellation uvs (`quad`: u along local +X, v along +Z, both 0..1 across the quad; `sphere`: u around +Y from +X toward +Z, v from the
   * north pole down; seams are un-welded so no texel smears), times `scale`. Needed by any material that uses a texture.
   */
  uv?: { scale?: readonly [number, number] };
}

export type LightDesc =
  /** Directional light with optional angular radius (radians); radiance-equivalent irradiance in W·m⁻². */
  { type: 'directional'; direction: Vec3; irradiance: Rgb; angularRadius?: number };

export type EnvironmentDesc =
  | { type: 'uniform'; radiance: Rgb }
  /** Equirectangular HDR (PFM or EXR), +Y up, u=0 at −Z then rotating toward +X (documented per exporter). */
  | { type: 'hdri'; path: string; scale: number; rotationYDeg: number; sha256?: string };

export type ExpectedDesc =
  /** Every pixel should equal this value (white furnace with an energy-conserving material). */
  | { type: 'uniform'; value: Rgb; tolerance: number }
  /** Mean over the image should equal this value. */
  | { type: 'mean'; value: Rgb; tolerance: number };
