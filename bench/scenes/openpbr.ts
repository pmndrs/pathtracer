/**
 * The `openpbr` bench material: OpenPBR 1.1.2 defaults, default resolution, anisotropy conversion and the reference-class
 * helper of docs/design/tier1.md sections 0 and 2. Pure functions, no I/O.
 */
import type { OpenPbrParams, Rgb } from './schema.ts';

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

// ---------------------------------------------------------------------------------------------------------------
// Reference classes (docs/design/tier1.md sections 0 and 2)
// ---------------------------------------------------------------------------------------------------------------

/**
 * A: exact in both pbrt-v4 and Mitsuba 3 (gate candidates). B: approximate in both, the references disagree. B': one reference only
 * (pbrt). C: no reference.
 */
export type ReferenceClass = 'A' | 'B' | "B'" | 'C';

export interface ClassContext {
  /** The source had a base-colour tint on a transmissive material that the OpenPBR parameters cannot hold next to a volume (`transmissionColor` is the attenuation colour). */
  lostTransmissionTint?: boolean;
}

export interface ClassResult { class: ReferenceClass; reasons: string[] }

const RANK: Record<ReferenceClass, number> = { A: 0, B: 1, "B'": 2, C: 3 };
const isWhite = (c: readonly number[]): boolean => c.every((v) => v === 1);
const isBlack = (c: readonly number[]): boolean => c.every((v) => v === 0);

/**
 * Which reference class a (resolved or partial) OpenPBR material falls in, with the reasons for anything above A. The worst feature
 * wins. A metalness between 0 and 1 is a linear blend (pbrt `mix`, Mitsuba `blendbsdf`), so it is as exact as its worse end.
 */
export function classifyOpenPbr(params: OpenPbrParams, ctx: ClassContext = {}): ClassResult {
  const m = resolveOpenPbr(params);
  let cls: ReferenceClass = 'A';
  const reasons: string[] = [];
  const raise = (c: ReferenceClass, why: string): void => {
    if (RANK[c] > RANK[cls]) cls = c;
    reasons.push(`${c}: ${why}`);
  };

  const metal = m.baseMetalness;
  const transmitting = m.transmissionWeight > 0;
  const dielectricPart = metal < 1;
  const baseBlack = m.baseWeight === 0 || isBlack(m.baseColor);

  if (m.fuzzWeight > 0) raise('C', 'fuzz (sheen) has no reference');
  if (m.thinFilmWeight > 0) raise('C', 'thin film (iridescence) has no reference');
  if (m.alphaMode === 'blend' && m.geometryOpacity < 1) raise('C', 'alpha blend has no reference');
  else if (m.alphaMode === 'mask') raise('B', 'alpha mask: pbrt `alpha` texture is exact (gain-amplified bilinear alpha), Mitsuba `mask` reads a thresholded opacity bitmap (contour to 1/16 texel, volpath for shadow rays); the Tier 1 models using it are kitchen sinks anyway');
  if (m.transmissionDispersionScale > 0) raise("B'", 'dispersion: pbrt only (spectral eta); Mitsuba has a scalar IOR');
  if (m.subsurfaceWeight > 0) raise("B'", 'diffuse transmission: pbrt `diffusetransmission` only');

  if (m.coatWeight > 0) raise('B', metal >= 1 ? 'clearcoat on a metal: pbrt coatedconductor is close, Mitsuba principled clearcoat is Disney' : 'clearcoat on a dielectric: coateddiffuse without the base specular lobe, Mitsuba principled is Disney');

  if (dielectricPart && m.specularWeight > 0) {
    if (m.specularWeight < 1 || !isWhite(m.specularColor)) raise('C', 'specular weight < 1 or specular colour != 1: glTF scales F0 only, OpenPBR scales F at both ends, no reference expresses either');
    if (transmitting) {
      if (m.transmissionWeight < 1) raise('B', 'partial transmission mixes a transmissive and an opaque lobe');
      if (m.geometryThinWalled && m.specularRoughness > 0) raise('C', 'thin-walled rough transmission has no reference');
      if (!isWhite(m.transmissionColor) && (m.transmissionDepth === 0 || ctx.lostTransmissionTint)) raise('B', 'coloured transmission tint: neither reference tints a dielectric (compare white-base only)');
      else if (ctx.lostTransmissionTint) raise('B', 'base-colour tint next to a volume cannot be held in OpenPBR');
    } else if (!baseBlack) {
      raise('B', 'dielectric over a diffuse base: pbrt coateddiffuse and Mitsuba roughplastic differ by a few percent');
    }
  }
  if (metal > 0 && m.specularRoughnessAnisotropy > 0 && metal < 1) raise('B', 'anisotropic metal blended with a dielectric');
  if (dielectricPart && m.specularRoughnessAnisotropy > 0 && metal === 0) raise('B', 'anisotropic dielectric: no exact reference (pbrt coateddiffuse uroughness only)');
  return { class: cls, reasons };
}
