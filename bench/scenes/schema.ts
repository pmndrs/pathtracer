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
  | { generator: 'checker'; cells: number; a: readonly [number, number, number]; b: readonly [number, number, number]; marker?: readonly [number, number, number] }
  /** One colour everywhere (0-255 codes; a `linear` texture holds raw bytes). */
  | { generator: 'solid'; color: readonly [number, number, number, number?] }
  /** Tangent-space normal map: a sine bump pattern `periods` times across, slope `amplitude` (tan of the maximum tilt). */
  | { generator: 'bumps'; periods: number; amplitude: number }
  /** Metallic-roughness (glTF: G = roughness, B = metalness): vertical stripes, `periods` of them, alternating (roughness, metalness) `a` and `b` (0-1). */
  | { generator: 'stripes-mr'; periods: number; a: readonly [number, number]; b: readonly [number, number] }
) & {
  width: number;
  height: number;
  colorSpace: 'srgb' | 'linear';
  /** Default `repeat`. */
  wrap?: 'repeat' | 'clamp' | 'mirror';
};

export type MaterialDesc =
  /** Lambertian. `reflectanceTexture` (a `textures` key, sRGB) multiplies `reflectance`. */
  | { type: 'diffuse'; reflectance: Rgb; reflectanceTexture?: string }
  /**
   * glTF-style metallic-roughness (ours only: pbrt-v4 and Mitsuba cannot express it exactly; exporters throw). Textures multiply the factors
   * (base colour rgb, roughness from G and metalness from B of `metallicRoughnessTexture`, `emissive` rgb); `normalTexture` is tangent space.
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
  | { type: 'file'; path: string; format: 'glb' | 'gltf' | 'ply' | 'obj'; sha256?: string };

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
