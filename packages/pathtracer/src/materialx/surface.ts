/**
 * MaterialX surface shaders to OpenPBR parameters (P5): `open_pbr_surface` (MaterialX 1.39, OpenPBR 1.1) one to one and
 * `standard_surface` (Autodesk Standard Surface 1.0.1) through the mapping below. Each parameter is a `MtlxSlot`: a lowered expression
 * (`MtlxIr`, constant or connected) plus a `scale` the value is multiplied with (unit conversions) and a `reduce` rule.
 *
 * The parameter names (`OpenPbrKey`) are the camelCase names of the bench's `OpenPbrParams`, so a baked or folded material is
 * assignable to it without a copy.
 */
import type { MtlxDocument, MtlxNode } from './document.ts';
import { MtlxLowering, type MtlxIr } from './eval.ts';
import { MtlxUnsupportedError } from './nodes.ts';

export type OpenPbrKey =
  | 'baseWeight' | 'baseColor' | 'baseDiffuseRoughness' | 'baseMetalness'
  | 'specularWeight' | 'specularColor' | 'specularRoughness' | 'specularRoughnessAnisotropy' | 'specularIor'
  | 'transmissionWeight' | 'transmissionColor' | 'transmissionDepth' | 'transmissionDispersionScale' | 'transmissionDispersionAbbeNumber'
  | 'subsurfaceWeight' | 'subsurfaceColor'
  | 'coatWeight' | 'coatColor' | 'coatRoughness' | 'coatIor'
  | 'fuzzWeight' | 'fuzzColor' | 'fuzzRoughness'
  | 'thinFilmWeight' | 'thinFilmThickness' | 'thinFilmIor'
  | 'emissionLuminance' | 'emissionColor'
  | 'geometryThinWalled' | 'geometryOpacity'
  | 'anisotropyRotation';

export interface MtlxSlot {
  key: OpenPbrKey;
  /** The expression (usually a constant). */
  ir: MtlxIr;
  /** The type the document wrote (`float`, `color3`, ...). */
  type: string;
  /** The final value is `scale * value`. */
  scale: number;
  /** `avg`: a colour feeds a float parameter (standard_surface `opacity`): the mean of its channels. */
  reduce?: 'avg';
}

export interface MtlxSurface {
  /** `open_pbr_surface` or `standard_surface`. */
  category: string;
  /** The shader node's name. */
  name: string;
  /** Only the parameters the document sets (everything else keeps its default). */
  slots: Partial<Record<OpenPbrKey, MtlxSlot>>;
  /** The `geometry_normal` / `normal` input (usually a `normalmap` node), if connected. */
  normal?: { ir: MtlxIr; type: string };
  /** Authored inputs the renderer does not read (named as in the document), for the support report. */
  ignored: string[];
}

interface Mapping {
  key: OpenPbrKey | null;
  scale?: number;
  reduce?: 'avg';
}

/** open_pbr_surface input -> parameter (`null` = read but not rendered: listed in `ignored` when authored). */
const OPEN_PBR_INPUTS: Record<string, Mapping> = {
  base_weight: { key: 'baseWeight' }, base_color: { key: 'baseColor' }, base_diffuse_roughness: { key: 'baseDiffuseRoughness' }, base_metalness: { key: 'baseMetalness' },
  specular_weight: { key: 'specularWeight' }, specular_color: { key: 'specularColor' }, specular_roughness: { key: 'specularRoughness' }, specular_ior: { key: 'specularIor' }, specular_roughness_anisotropy: { key: 'specularRoughnessAnisotropy' },
  transmission_weight: { key: 'transmissionWeight' }, transmission_color: { key: 'transmissionColor' }, transmission_depth: { key: 'transmissionDepth' }, transmission_dispersion_scale: { key: 'transmissionDispersionScale' }, transmission_dispersion_abbe_number: { key: 'transmissionDispersionAbbeNumber' },
  subsurface_weight: { key: 'subsurfaceWeight' }, subsurface_color: { key: 'subsurfaceColor' },
  coat_weight: { key: 'coatWeight' }, coat_color: { key: 'coatColor' }, coat_roughness: { key: 'coatRoughness' }, coat_ior: { key: 'coatIor' },
  fuzz_weight: { key: 'fuzzWeight' }, fuzz_color: { key: 'fuzzColor' }, fuzz_roughness: { key: 'fuzzRoughness' },
  thin_film_weight: { key: 'thinFilmWeight' }, thin_film_thickness: { key: 'thinFilmThickness' }, thin_film_ior: { key: 'thinFilmIor' },
  emission_luminance: { key: 'emissionLuminance' }, emission_color: { key: 'emissionColor' },
  geometry_opacity: { key: 'geometryOpacity' }, geometry_thin_walled: { key: 'geometryThinWalled' },
  // Read by the spec but without a field here (the renderer has no such lobe parameter):
  transmission_scatter: { key: null }, transmission_scatter_anisotropy: { key: null }, subsurface_radius: { key: null }, subsurface_radius_scale: { key: null }, subsurface_scatter_anisotropy: { key: null },
  coat_roughness_anisotropy: { key: null }, coat_darkening: { key: null }, geometry_coat_normal: { key: null }, geometry_tangent: { key: null }, geometry_coat_tangent: { key: null },
  specular_roughness_anisotropy_rotation: { key: null },
};

/**
 * standard_surface 1.0.1 -> OpenPBR 1.1. Same-meaning parameters map one to one; `emission` is a luminance scale (nits in OpenPBR, the same
 * radiance multiplier here); `opacity` is a colour in standard_surface (mean of its channels); `thin_film_thickness` is nanometres there and
 * micrometres here, and its weight is 1 when the thickness is positive; `specular_rotation` is turns. See DIVERGES V261 for what does not map.
 */
const STANDARD_SURFACE_INPUTS: Record<string, Mapping> = {
  base: { key: 'baseWeight' }, base_color: { key: 'baseColor' }, diffuse_roughness: { key: 'baseDiffuseRoughness' }, metalness: { key: 'baseMetalness' },
  specular: { key: 'specularWeight' }, specular_color: { key: 'specularColor' }, specular_roughness: { key: 'specularRoughness' }, specular_IOR: { key: 'specularIor' },
  specular_anisotropy: { key: 'specularRoughnessAnisotropy' }, specular_rotation: { key: 'anisotropyRotation', scale: 2 * Math.PI },
  transmission: { key: 'transmissionWeight' }, transmission_color: { key: 'transmissionColor' }, transmission_depth: { key: 'transmissionDepth' },
  subsurface: { key: 'subsurfaceWeight' }, subsurface_color: { key: 'subsurfaceColor' },
  sheen: { key: 'fuzzWeight' }, sheen_color: { key: 'fuzzColor' }, sheen_roughness: { key: 'fuzzRoughness' },
  coat: { key: 'coatWeight' }, coat_color: { key: 'coatColor' }, coat_roughness: { key: 'coatRoughness' }, coat_IOR: { key: 'coatIor' },
  thin_film_thickness: { key: 'thinFilmThickness', scale: 1e-3 }, thin_film_IOR: { key: 'thinFilmIor' },
  emission: { key: 'emissionLuminance' }, emission_color: { key: 'emissionColor' },
  opacity: { key: 'geometryOpacity', reduce: 'avg' }, thin_walled: { key: 'geometryThinWalled' },
  transmission_scatter: { key: null }, transmission_scatter_anisotropy: { key: null }, transmission_dispersion: { key: null }, transmission_extra_roughness: { key: null },
  subsurface_radius: { key: null }, subsurface_scale: { key: null }, subsurface_anisotropy: { key: null },
  coat_anisotropy: { key: null }, coat_rotation: { key: null }, coat_affect_color: { key: null }, coat_affect_roughness: { key: null }, tangent: { key: null }, internal_reflections: { key: null }, caustics: { key: null }, exit_to_background: { key: null },
};

/** standard_surface defaults that differ from OpenPBR's (applied by `resolveSurface` when the document does not set the input). */
export const STANDARD_SURFACE_DEFAULTS: Readonly<Partial<Record<OpenPbrKey, number | number[]>>> = {
  baseWeight: 0.8, baseColor: [1, 1, 1], specularRoughness: 0.2, coatRoughness: 0.1, fuzzRoughness: 0.3, thinFilmIor: 1.5, subsurfaceColor: [1, 1, 1],
};

/**
 * gltf_pbr (MaterialX 1.39 / glTF 2.0 metallic-roughness with the common extensions) -> OpenPBR. glTF `specular` scales F0 only, OpenPBR's
 * `specular_weight` scales F at both ends (V74), and iridescence, thickness / attenuation, anisotropy and dispersion are listed as ignored.
 */
const GLTF_PBR_INPUTS: Record<string, Mapping> = {
  base_color: { key: 'baseColor' }, metallic: { key: 'baseMetalness' }, roughness: { key: 'specularRoughness' }, transmission: { key: 'transmissionWeight' },
  specular: { key: 'specularWeight' }, specular_color: { key: 'specularColor' }, ior: { key: 'specularIor' }, alpha: { key: 'geometryOpacity' },
  clearcoat: { key: 'coatWeight' }, clearcoat_roughness: { key: 'coatRoughness' }, sheen_color: { key: 'fuzzColor' }, sheen_roughness: { key: 'fuzzRoughness' },
  emissive: { key: 'emissionColor' }, emissive_strength: { key: 'emissionLuminance' },
  occlusion: { key: null }, alpha_mode: { key: null }, alpha_cutoff: { key: null }, tangent: { key: null }, clearcoat_normal: { key: null }, thickness: { key: null }, attenuation_distance: { key: null }, attenuation_color: { key: null },
  iridescence: { key: null }, iridescence_ior: { key: null }, iridescence_thickness: { key: null }, dispersion: { key: null }, anisotropy_strength: { key: null }, anisotropy_rotation: { key: null },
};

/** glTF defaults that differ from OpenPBR's. */
export const GLTF_PBR_DEFAULTS: Readonly<Partial<Record<OpenPbrKey, number | number[]>>> = {
  baseColor: [1, 1, 1], baseMetalness: 1, specularRoughness: 1, specularIor: 1.5,
};

export const SURFACE_CATEGORIES: readonly string[] = ['open_pbr_surface', 'standard_surface', 'gltf_pbr'];

/** Finds the surface shader node a `surfacematerial` (or a shader node name) refers to. */
export function findShaderNode(doc: MtlxDocument, name?: string): MtlxNode {
  const pick = name ?? doc.materials[0] ?? [...doc.nodes.values()].find((n) => SURFACE_CATEGORIES.includes(n.category))?.name;
  if (!pick) throw new Error('MaterialX: the document has no material or surface shader');
  const node = doc.nodes.get(pick);
  if (!node) throw new Error(`MaterialX: no node '${pick}'`);
  if (node.category === 'surfacematerial') {
    const input = node.inputs.surfaceshader;
    if (!input?.nodename) throw new Error(`MaterialX: material '${pick}' has no connected surfaceshader`);
    const shader = doc.nodes.get(input.nodename);
    if (!shader) throw new Error(`MaterialX: material '${pick}': shader '${input.nodename}' not found`);
    return shader;
  }
  return node;
}

/** Lowers a shader node into parameter slots. Throws `MtlxUnsupportedError` for a shader that is not open_pbr_surface / standard_surface. */
export function lowerSurface(lowering: MtlxLowering, shader: MtlxNode): MtlxSurface {
  const table = shader.category === 'open_pbr_surface' ? OPEN_PBR_INPUTS : shader.category === 'standard_surface' ? STANDARD_SURFACE_INPUTS : shader.category === 'gltf_pbr' ? GLTF_PBR_INPUTS : null;
  if (!table) throw new MtlxUnsupportedError(shader.category, `MaterialX shader <${shader.category}> '${shader.name}' is not supported (open_pbr_surface, standard_surface and gltf_pbr are)`);
  const surface: MtlxSurface = { category: shader.category, name: shader.name, slots: {}, ignored: [] };
  for (const [name, input] of Object.entries(shader.inputs)) {
    if (name === 'normal' || name === 'geometry_normal') {
      const src = lowering.inputIr(shader, name);
      if (src) surface.normal = src;
      continue;
    }
    const map = table[name];
    if (!map) {
      surface.ignored.push(`${name} (unknown input)`);
      continue;
    }
    const src = lowering.inputIr(shader, name);
    if (!src) continue;
    if (map.key === null) {
      surface.ignored.push(name);
      continue;
    }
    const slot: MtlxSlot = { key: map.key, ir: src.ir, type: src.type, scale: map.scale ?? 1 };
    if (map.reduce) slot.reduce = map.reduce;
    surface.slots[map.key] = slot;
    void input;
  }
  if (shader.category === 'gltf_pbr') {
    // glTF: a sheen colour is the whole sheen lobe (weight 1), and `emissive` is scaled by `emissive_strength` (default 1).
    const constant = (key: OpenPbrKey, value: number): MtlxSlot => ({ key, ir: { kind: 'const', type: 'float', value: [value, 0, 0, 0] }, type: 'float', scale: 1 });
    if (surface.slots.fuzzColor && !surface.slots.fuzzWeight) surface.slots.fuzzWeight = constant('fuzzWeight', 1);
    if (surface.slots.emissionColor && !surface.slots.emissionLuminance) surface.slots.emissionLuminance = constant('emissionLuminance', 1);
  }
  return surface;
}
