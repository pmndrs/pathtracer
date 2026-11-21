/**
 * OpenPBR 1.1.2 parameters as the glTF loader produces them: the parameter set, its defaults, default resolution and the glTF <-> OpenPBR
 * anisotropy conversion. Pure functions, no I/O. `openPbrToRtMaterial` (material.ts) turns a resolved set into a scene-copy material.
 */

/** A linear Rec.709 colour (read-only). */
export type OpenPbrRgb = readonly [number, number, number];
type Rgb = OpenPbrRgb;

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
  /**
   * glTF `KHR_materials_scatter` (draft, volumetric mode only: a transmissive material with `KHR_materials_volume` thickness > 0 and a finite attenuation distance). `strength` = scatterStrengthFactor,
   * `multiscatterColor` = the MULTI-scatter albedo of the volume, `anisotropy` = HG g. Present only when the extension is. Not read by `openPbrToRtMaterial`; `gltfVolumesAsMedia` (bench) turns it into sigma_s / g.
   */ volumeScatter?: { strength: number; multiscatterColor: Rgb; anisotropy: number };
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
  // --- extras (not OpenPBR parameters; the scene copy has fields for them) ---
  /** Radians, rotation of the anisotropy axis about the normal (glTF `anisotropyRotation`); OpenPBR takes the axis from the tangent field. Default 0. */ anisotropyRotation?: number;
  /** Default 'opaque'. 'mask' tests `geometryOpacity` (times the base colour alpha texture) against `alphaCutoff`; 'blend' is stochastic. */ alphaMode?: 'opaque' | 'mask' | 'blend';
  /** Default 0.5. */ alphaCutoff?: number;
  /** Default false. */ doubleSided?: boolean;
  /** Emit from both faces (default false: authored front face). */ emissionTwoSided?: boolean;
  /** Texture keys (the loader's texture table); the factors above multiply them. */
  baseColorTexture?: string;
  metallicRoughnessTexture?: string;
  normalTexture?: string;
  emissiveTexture?: string;
}

/** Every parameter with its OpenPBR 1.1.2 default (`coatIor` is OpenPBR's own 1.6; the glTF mapping overrides it with 1.5). */
export const OPENPBR_DEFAULTS = {
  baseWeight: 1,
  baseColor: [0.8, 0.8, 0.8] as Rgb,
  baseMetalness: 0,
  baseDiffuseRoughness: 0,
  specularWeight: 1,
  specularColor: [1, 1, 1] as Rgb,
  specularRoughness: 0.3,
  specularRoughnessAnisotropy: 0,
  specularIor: 1.5,
  transmissionWeight: 0,
  transmissionColor: [1, 1, 1] as Rgb,
  transmissionDepth: 0,
  transmissionDispersionScale: 0,
  transmissionDispersionAbbeNumber: 20,
  subsurfaceWeight: 0,
  subsurfaceColor: [1, 1, 1] as Rgb,
  coatWeight: 0,
  coatColor: [1, 1, 1] as Rgb,
  coatRoughness: 0,
  coatIor: 1.6,
  fuzzWeight: 0,
  fuzzColor: [1, 1, 1] as Rgb,
  fuzzRoughness: 0.5,
  thinFilmWeight: 0,
  thinFilmThickness: 0.5,
  thinFilmIor: 1.4,
  emissionLuminance: 0,
  emissionColor: [1, 1, 1] as Rgb,
  geometryThinWalled: false,
  geometryOpacity: 1,
  anisotropyRotation: 0,
  alphaMode: 'opaque' as 'opaque' | 'mask' | 'blend',
  alphaCutoff: 0.5,
  doubleSided: false,
  emissionTwoSided: false,
} as const;

/** The numeric / colour / boolean fields that have defaults (the texture names have none). */
export type OpenPbrResolved = { [K in keyof typeof OPENPBR_DEFAULTS]: (typeof OPENPBR_DEFAULTS)[K] extends readonly number[] ? Rgb : K extends 'alphaMode' ? 'opaque' | 'mask' | 'blend' : (typeof OPENPBR_DEFAULTS)[K] extends boolean ? boolean : number } & Pick<OpenPbrParams, 'baseColorTexture' | 'metallicRoughnessTexture' | 'normalTexture' | 'emissiveTexture'>;

/** Fills every missing field with its OpenPBR default (colours are copied). */
export function resolveOpenPbr(p: OpenPbrParams): OpenPbrResolved {
  const out: Record<string, unknown> = { ...OPENPBR_DEFAULTS };
  for (const [k, v] of Object.entries(p)) {
    if (v === undefined || k === 'type') continue;
    out[k] = Array.isArray(v) ? [...v] : v;
  }
  return out as unknown as OpenPbrResolved;
}

// ---------------------------------------------------------------------------------------------------------------
// Anisotropy: glTF KHR_materials_anisotropy <-> OpenPBR specular_roughness_anisotropy
// ---------------------------------------------------------------------------------------------------------------

/** GGX alphas along the tangent (t) and bitangent (b) as glTF defines them: alpha = roughness^2, alpha_t = mix(alpha, 1, strength^2), alpha_b = alpha. */
export function gltfAnisotropyAlphas(roughness: number, strength: number): { alphaT: number; alphaB: number } {
  const alpha = roughness * roughness;
  return { alphaT: alpha + (1 - alpha) * strength * strength, alphaB: alpha };
}

/** GGX alphas as OpenPBR defines them: alpha_x = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_y = (1 - a) alpha_x. */
export function openPbrAnisotropyAlphas(roughness: number, anisotropy: number): { alphaT: number; alphaB: number } {
  const alphaT = roughness * roughness * Math.sqrt(2 / (1 + (1 - anisotropy) * (1 - anisotropy)));
  return { alphaT, alphaB: (1 - anisotropy) * alphaT };
}

/**
 * glTF (perceptual roughness, anisotropy strength) -> OpenPBR (roughness r, anisotropy a) with identical (alpha_t, alpha_b):
 * a = 1 - alpha_b / alpha_t and r = sqrt(alpha_t sqrt((1 + (1 - a)^2) / 2)). Strength 0 gives r = roughness, a = 0; roughness 0
 * with strength > 0 gives alpha_b = 0, a = 1.
 */
export function gltfAnisotropyToOpenPbr(roughness: number, strength: number): { roughness: number; anisotropy: number } {
  const { alphaT, alphaB } = gltfAnisotropyAlphas(roughness, strength);
  if (alphaT <= 0) return { roughness: 0, anisotropy: 0 };
  const anisotropy = 1 - alphaB / alphaT;
  const r2 = alphaT * Math.sqrt((1 + (1 - anisotropy) * (1 - anisotropy)) / 2);
  return { roughness: Math.sqrt(r2), anisotropy };
}

/** Inverse of `gltfAnisotropyToOpenPbr`: strength = sqrt((alpha_t - alpha_b) / (1 - alpha_b)), roughness = sqrt(alpha_b). */
export function openPbrAnisotropyToGltf(roughness: number, anisotropy: number): { roughness: number; strength: number } {
  const { alphaT, alphaB } = openPbrAnisotropyAlphas(roughness, anisotropy);
  const strength = alphaB >= 1 ? 0 : Math.sqrt(Math.max(0, (alphaT - alphaB) / (1 - alphaB)));
  return { roughness: Math.sqrt(alphaB), strength };
}

// ---------------------------------------------------------------------------------------------------------------
// Lengths
// ---------------------------------------------------------------------------------------------------------------

/** OpenPBR's default length unit is the centimetre; our scenes (and glTF) are in metres. */
export const OPENPBR_CM_PER_METRE = 100;
export const metresToOpenPbrCm = (metres: number): number => metres * OPENPBR_CM_PER_METRE;
export const openPbrCmToMetres = (cm: number): number => cm / OPENPBR_CM_PER_METRE;

/** Thin-film thickness: glTF iridescence is in nanometres, OpenPBR in micrometres. */
export const nmToMicrometres = (nm: number): number => nm / 1000;
