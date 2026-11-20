/**
 * The `openpbr` bench material: the reference-class helper of docs/internal/design/tier1.md sections 0 and 2. The OpenPBR defaults, default
 * resolution and anisotropy conversion moved to `@pmndrs/raytracer` (gltf/openpbr.ts) and are re-exported here. Pure functions, no I/O.
 */
import type { OpenPbrParams } from './schema.ts';

export { OPENPBR_DEFAULTS, resolveOpenPbr, gltfAnisotropyAlphas, openPbrAnisotropyAlphas, gltfAnisotropyToOpenPbr, openPbrAnisotropyToGltf, OPENPBR_CM_PER_METRE, metresToOpenPbrCm, openPbrCmToMetres, nmToMicrometres } from '@pmndrs/raytracer';
export type { OpenPbrResolved } from '@pmndrs/raytracer';
import { resolveOpenPbr } from '@pmndrs/raytracer';

// ---------------------------------------------------------------------------------------------------------------
// Reference classes (docs/internal/design/tier1.md sections 0 and 2)
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
