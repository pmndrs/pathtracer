/**
 * P2.2: the adapter from the scene copy's material record (`RtMaterial`, `rtMaterial(id)` in WGSL) to the BSDF's plain
 * parameter set (`PtBsdfParams`). One function in TypeScript and one in WGSL, line for line.
 *
 * Rules, for the bench subset and the corners of OpenPBR the scene copy can already say (anything richer is a later lobe):
 *
 *  1. `transmissionWeight > 0.5`  -> smooth or rough dielectric: `ior`, alpha = specularRoughness^2. (The bench `dielectric`;
 *     the bench maps it to transmissionWeight 1.) Absorption (`transmissionDepth`) and thin walls are not applied.
 *  2. `metalness > 0.5` and `conductorEtaK` -> GGX conductor with the record's complex IOR (the bench `conductor`).
 *  3. `metalness > 0.5` without eta/k -> GGX conductor whose normal-incidence reflectance is `baseColor * baseWeight`
 *     (eta = 1, k = 2 sqrt(r) / sqrt(1 - r): Gulbrandsen's artist-friendly inverse). Approximation, not a Tier 0 case.
 *  4. `baseWeight * max(baseColor) > 0` -> Lambert with reflectance `baseColor * baseWeight`. The dielectric specular lobe
 *     over it (`specularWeight`, roughness) is NOT evaluated yet: the bench `diffuse` has `specularWeight` 0, a general
 *     OpenPBR dielectric on top of a diffuse base needs the layered lobe (a later task).
 *  5. otherwise black (scatters nothing; the bench `black`, emitters' backing).
 *
 * The 0.5 thresholds make the blend weights binary; the bench only produces 0 and 1.
 *
 * OpenPBR mode (P2.21/P2.22/P2.30; `PtMaterialOptions`, a scene with an `openpbr` material): same function with one more argument,
 * a uniform `u` of the vertex for the stochastic metal/dielectric choice, and these rules instead (metal first):
 *
 *  1. metal: with `features.mix`, `u < metalness` (pbrt `mix`: the metal end with probability `metalness`, so the expectation is the
 *     exact linear blend; metalness 0 and 1 stay exact); without it `metalness > 0.5`. Complex-Fresnel conductor as in 2./3. above.
 *  2. `transmissionWeight > 0.5`: thin-walled (and `features.thin`) -> `PT_BSDF_THIN` with `ior` and the transmission colour as tint
 *     (`transmissionTint`); otherwise `PT_BSDF_DIELECTRIC`. The interior medium (`transmissionDepth`) is path state, not BSDF.
 *  3. `features.layered` and a dielectric specular lobe (`specularWeight > 0` and `ior != 1`, or a coat with `coatWeight > 0.5`):
 *     `PT_BSDF_LAYERED`, an interface of (`ior`, `specularRoughness`, Fresnel scale `specularColor * specularWeight`) over a Lambert
 *     base. A coat replaces the interface with (`coatIor`, `coatRoughness`, scale 1) and drops the base's own specular lobe, which is
 *     what the pbrt exporter writes (`coateddiffuse` with the coat as interface; DIVERGES V69). `coatWeight` is thresholded at 0.5.
 *  4. otherwise Lambert / black as above.
 *
 *  P2.23 (`features.coatedMetal`): a metal with `coatWeight > 0.5` and `coatIor != 1` becomes `PT_BSDF_LAYERED` with a conductor bottom layer
 *  (pbrt `coatedconductor`): interface (`coatIor`, `coatRoughness`), conductor eta and k divided by `coatIor`, alpha `specularRoughness^2`.
 *  Packing: `ior`/`alpha` = interface, `reflectance` = conductor eta', `eta` = conductor k', `k` = (thickness, conductor alpha, 1 = conductor).
 */
import type { RtMaterial } from '@pmndrs/raytracer';
import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, PT_BSDF_LAYERED, PT_BSDF_THIN, ptFrComplex, ptFrDielectric, ptIsSmooth, type PtBsdfParams, type PtVec3 } from './bsdf.ts';
import { ptAlbedoBaseInternal, type PtLayeredOptions, type PtLayeredParams } from './layered.ts';
import { ptDielectricF0, ptLayeredAlbedo } from './layered-lut.ts';
import { ptRedFuzzAlbedo, ptRedFuzzWgsl } from './reduced-fuzz.ts';
import { ptLayerLutWgsl } from './layered-wgsl.ts';
import type { PtThinFilm } from './thinfilm.ts';

/** k that gives a conductor with eta = 1 the normal-incidence reflectance `r` (clamped just below 1). */
export function conductorKFromReflectance(r: number): number {
  const c = Math.min(Math.max(r, 0), 0.9999);
  return (2 * Math.sqrt(c)) / Math.sqrt(1 - c);
}

/** GGX alpha from a perceptual roughness, as `rtMaterialAlpha` (`roughness^2`). */
const alphaOf = (m: RtMaterial): number => m.specularRoughness * m.specularRoughness;

export function ptBsdfFromMaterial(m: RtMaterial, options?: PtMaterialOptions, u = 0): PtBsdfParams {
  if (options) return ptBsdfFromOpenPbr(m, options, u);
  const a = alphaOf(m);
  if (m.transmissionWeight > 0.5) {
    return { kind: PT_BSDF_DIELECTRIC, reflectance: [0, 0, 0], eta: [0, 0, 0], k: [0, 0, 0], alpha: [a, a], ior: m.ior };
  }
  if (m.metalness > 0.5) {
    if (m.conductorEtaK) {
      return { kind: PT_BSDF_CONDUCTOR, reflectance: [0, 0, 0], eta: [...m.conductorEta] as PtVec3, k: [...m.conductorK] as PtVec3, alpha: [a, a], ior: 1 };
    }
    const k: PtVec3 = [0, 1, 2].map((i) => conductorKFromReflectance((m.baseColor[i] as number) * m.baseWeight)) as unknown as PtVec3;
    return { kind: PT_BSDF_CONDUCTOR, reflectance: [0, 0, 0], eta: [1, 1, 1], k, alpha: [a, a], ior: 1 };
  }
  const reflectance: PtVec3 = [m.baseColor[0] * m.baseWeight, m.baseColor[1] * m.baseWeight, m.baseColor[2] * m.baseWeight];
  if (Math.max(...reflectance) > 0) {
    return { kind: PT_BSDF_DIFFUSE, reflectance, eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior: 1 };
  }
  return { kind: PT_BSDF_BLACK, reflectance: [0, 0, 0], eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior: 1 };
}

/** The WGSL twin. Needs `ptBsdfWgsl` (PtBsdfParams, PT_BSDF_*) and the scene module (`RtMaterial`) before it. */
export function ptBsdfAdapterWgsl(options?: PtMaterialOptions): string {
  if (options) return ptBsdfOpenPbrWgsl(options);
  return /* wgsl */ `
fn ptConductorKFromReflectance(r: vec3f) -> vec3f {
    let c = clamp(r, vec3f(0.0), vec3f(0.9999));
    return 2.0 * sqrt(c) / sqrt(1.0 - c);
}

fn ptBsdfFromMaterial(m: RtMaterial) -> PtBsdfParams {
    var p: PtBsdfParams;
    p.kind = PT_BSDF_BLACK;
    p.ior = 1.0;
    p.alpha = vec2f(0.0);
    p.reflectance = vec3f(0.0);
    p.eta = vec3f(0.0);
    p.k = vec3f(0.0);
    let a = m.specularRoughness * m.specularRoughness;
    if (m.transmissionWeight > 0.5) {
        p.kind = PT_BSDF_DIELECTRIC;
        p.alpha = vec2f(a);
        p.ior = m.ior;
        return p;
    }
    if (m.metalness > 0.5) {
        p.kind = PT_BSDF_CONDUCTOR;
        p.alpha = vec2f(a);
        if (m.conductorEtaK) {
            p.eta = m.conductorEta;
            p.k = m.conductorK;
        } else {
            p.eta = vec3f(1.0);
            p.k = ptConductorKFromReflectance(m.baseColor * m.baseWeight);
        }
        return p;
    }
    let r = m.baseColor * m.baseWeight;
    if (max(r.x, max(r.y, r.z)) > 0.0) {
        p.kind = PT_BSDF_DIFFUSE;
        p.reflectance = r;
    }
    return p;
}
`;
}

// ---------------------------------------------------------------------------------------------------------------
// OpenPBR mode

/** Which lobes a scene needs; each `false` keeps its code out of the kernel (`ptMaterialFeatures` derives them from the material table). */
export interface PtMaterialFeatures {
  /** Some material has `0 < metalness < 1`: metal or dielectric chosen stochastically per vertex. */
  mix: boolean;
  /** Some transmissive material is thin-walled: `PT_BSDF_THIN`. */
  thin: boolean;
  /** Some opaque dielectric has a specular lobe or a coat: `PT_BSDF_LAYERED` (`ptLayeredWgsl` is appended to the kernel). */
  layered: boolean;
  /** Some thick transmissive material has `transmissionDepth > 0`: Beer's law along the path inside the mesh. */
  medium: boolean;
  /** P2.23: some metal has a coat (`coatWeight > 0.5`, `coatIor != 1`): the layered BSDF gets its conductor bottom layer (implies `layered`). Optional so older literals still type. */
  coatedMetal?: boolean;
  /** P2.26: some metal has `specularAnisotropy > 0`: its GGX alphas split along the tangent (`ptOpenPbrAlpha`) and the integrator builds the frame from the vertex tangent (`ptFrameFromTangent`); needs a surface layout with tangents. Optional like `coatedMetal`. */
  aniso?: boolean;
  /** P2.25: some material has fuzz (`sheenWeight > 0` and a non-black `sheenColor`): the OpenPBR fuzz layer wraps the base BSDF (`ptFuzzWgsl`; the LTC table is bound at group 2, binding 2). Optional like `coatedMetal`. */
  fuzz?: boolean;
  /** P2.27: some metal (without a coat) or opaque dielectric (without a coat) has a thin film (`thinFilmWeight > 0`, thickness > 0): the film Fresnel is compiled (`ptThinFilmWgsl`) and the kernel needs albedo layering. Optional like `coatedMetal`. */
  thinFilm?: boolean;
}

export interface PtMaterialOptions {
  features: PtMaterialFeatures;
  /** Layered BSDF settings (the `ptLayeredWgsl` options; the kernel compiles one variant). Default `layering: 'albedo'`. */
  layered?: PtLayeredOptions;
  /** Layer thickness in pbrt's `coateddiffuse` sense (default 1e-6: transparent, what the reference exporters write; pbrt's own default is 0.01). Stochastic only. */
  layerThickness?: number;
  /** Tint thin-walled transmission by `transmissionColor` (default true). pbrt and Mitsuba `thindielectric` cannot tint. */
  transmissionTint?: boolean;
  /**
   * R1 (ray-class material LOD): the features of the FULL material table, set on a kernel that only ever shades reduced vertices (the wavefront's
   * secondary shade kernel: its own `features` have layered / fuzz / film / aniso / coatedMetal stripped, so none of that code is compiled, while
   * `ptBsdfReduced` still has to recognise those materials and reduce them). Omitted: `features` itself is the full table (megakernel).
   */
  reducedFrom?: PtMaterialFeatures;
}

export const PT_NO_FEATURES: PtMaterialFeatures = { mix: false, thin: false, layered: false, medium: false };

/**
 * The features a material table needs in OpenPBR mode (union over the records; a record the scene never hits still counts, which only
 * costs kernel size). Mirrors the branches of `ptBsdfFromMaterial` with options.
 */
export function ptMaterialFeatures(materials: readonly RtMaterial[]): PtMaterialFeatures {
  const f: PtMaterialFeatures = { mix: false, thin: false, layered: false, medium: false };
  for (const m of materials) {
    if (m.metalness > 0 && m.metalness < 1) f.mix = true;
    if (m.metalness > 0 && m.specularAnisotropy > 0) f.aniso = true; // the key exists only when true, as coatedMetal
    if (m.thinFilmWeight > 0 && m.thinFilmThickness > 0 && m.coatWeight <= 0.5 && m.transmissionWeight <= 0.5) f.thinFilm = true;
    if (m.sheenWeight > 0 && Math.max(m.sheenColor[0], m.sheenColor[1], m.sheenColor[2]) > 0) f.fuzz = true;
    if (m.metalness > 0 && m.coatWeight > 0.5 && m.coatIor !== 1) {
      f.coatedMetal = true; // the key exists only when true: scenes without coated metals keep their feature object
      f.layered = true;
    }
    const dielectricEnd = m.metalness < 1;
    if (!dielectricEnd) continue;
    if (m.transmissionWeight > 0.5) {
      if (m.thinWalled) f.thin = true;
      else if (m.transmissionDepth > 0) f.medium = true;
    } else if ((m.coatWeight > 0.5 || m.specularWeight > 0) && (m.coatWeight > 0.5 ? m.coatIor : m.ior) !== 1) f.layered = true;
  }
  return f;
}

/**
 * Cosine-weighted average Fresnel reflectance of a dielectric interface seen from inside (relative IOR `ior`, total internal reflection
 * included): Egan and Hilgeman's polynomial for the diffuse Fresnel reflectance, -1.440/n^2 + 0.710/n + 0.668 + 0.0636 n. The numeric value is
 * `ptInternalReflectance` (layered.ts); the fit is within 0.01 for ior 1.05-2.5 (tested). Only `albedoBase: 'internal'` reads it.
 */
export function ptInternalReflectanceFit(ior: number): number {
  if (ior <= 1) return 0;
  return Math.min(1, Math.max(0, -1.44 / (ior * ior) + 0.71 / ior + 0.668 + 0.0636 * ior));
}

/**
 * OpenPBR 1.1.2 GGX alphas of the specular lobe along the tangent (x) and bitangent (y): `alpha_x = r^2 sqrt(2 / (1 + (1 - a)^2))`,
 * `alpha_y = (1 - a) alpha_x` for roughness `r`, anisotropy `a` (a = 0 gives r^2 on both axes exactly). The same pair the pbrt and Mitsuba exporters write.
 */
export function ptOpenPbrAlpha(roughness: number, anisotropy: number): readonly [number, number] {
  if (!(anisotropy > 0)) return [roughness * roughness, roughness * roughness];
  const ax = roughness * roughness * Math.sqrt(2 / (1 + (1 - anisotropy) * (1 - anisotropy)));
  return [ax, (1 - anisotropy) * ax];
}

/**
 * The thin film a vertex of this material applies (P2.27), or undefined: the film sits between the coat-free dielectric or metal and the ambient
 * medium, so a coat (`coatWeight > 0.5`) or a transmissive base drops it (DIVERGES V82).
 */
export function ptFilmOfMaterial(m: RtMaterial): PtThinFilm | undefined {
  if (!(m.thinFilmWeight > 0 && m.thinFilmThickness > 0) || m.coatWeight > 0.5 || m.transmissionWeight > 0.5) return undefined;
  return { weight: m.thinFilmWeight, thickness: m.thinFilmThickness, ior: m.thinFilmIor };
}

/** Default layer thickness of the OpenPBR adapter (transparent layer, see `PtMaterialOptions.layerThickness`). */
export const PT_DEFAULT_LAYER_THICKNESS = 1e-6;

/** Absorption coefficient (per metre, linear RGB) of a thick transmissive material's interior: Beer-Lambert, `-ln(transmissionColor) / transmissionDepth`; 0 = none. */
export function ptMediumSigma(m: RtMaterial): PtVec3 {
  if (m.thinWalled || !(m.transmissionDepth > 0)) return [0, 0, 0];
  return [0, 1, 2].map((i) => -Math.log(Math.max(m.transmissionColor[i] as number, 1e-6)) / m.transmissionDepth) as unknown as PtVec3;
}

/** The interior medium after a transmission event: the material's own absorber when the ray enters (`woZ > 0`, it came from outside), none when it exits. Twin of the integrator's update (no nesting: a ray inside one medium that enters another replaces it). */
export function ptMediumAfterTransmission(m: RtMaterial, woZ: number): PtVec3 {
  return woZ > 0 ? ptMediumSigma(m) : [0, 0, 0];
}

/** TS twin of the WGSL `ptLayeredOf`: the layered parameters packed in a `PT_BSDF_LAYERED` `PtBsdfParams`. */
export function ptLayeredFromBsdf(p: PtBsdfParams, film?: PtThinFilm): PtLayeredParams {
  if (p.k[2] > 0.5) {
    return { baseColor: [0, 0, 0], ior: p.ior, alpha: p.alpha, specularWeight: 1, specularColor: [1, 1, 1], thickness: p.k[0], internalReflectance: 0, baseConductor: { eta: p.reflectance, k: p.eta, alpha: p.k[1] } };
  }
  return { baseColor: p.reflectance, ior: p.ior, alpha: p.alpha, specularWeight: 1, specularColor: p.eta, thickness: p.k[0], internalReflectance: p.k[1], ...(film ? { film } : {}) };
}

const BLACK_PARAMS = (): PtBsdfParams => ({ kind: PT_BSDF_BLACK, reflectance: [0, 0, 0], eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior: 1 });

/** OpenPBR-mode adapter (rules in the file comment). `u` in [0, 1) is the vertex's metal/dielectric choice (used with `features.mix`). */
export function ptBsdfFromOpenPbr(m: RtMaterial, options: PtMaterialOptions, u: number): PtBsdfParams {
  const f = options.features;
  const p = BLACK_PARAMS();
  const a = alphaOf(m);
  const metal = f.mix ? u < m.metalness : m.metalness > 0.5;
  if (metal) {
    p.kind = PT_BSDF_CONDUCTOR;
    p.alpha = f.aniso ? ptOpenPbrAlpha(m.specularRoughness, m.specularAnisotropy) : [a, a];
    if (m.conductorEtaK) {
      p.eta = [...m.conductorEta] as PtVec3;
      p.k = [...m.conductorK] as PtVec3;
    } else {
      p.eta = [1, 1, 1];
      p.k = [0, 1, 2].map((i) => conductorKFromReflectance((m.baseColor[i] as number) * m.baseWeight)) as unknown as PtVec3;
    }
    if (f.coatedMetal && m.coatWeight > 0.5 && m.coatIor !== 1) {
      const ci = m.coatIor;
      const cr = m.coatRoughness * m.coatRoughness;
      p.kind = PT_BSDF_LAYERED;
      p.reflectance = [p.eta[0] / ci, p.eta[1] / ci, p.eta[2] / ci];
      p.eta = [p.k[0] / ci, p.k[1] / ci, p.k[2] / ci];
      p.ior = ci;
      p.k = [options.layerThickness ?? PT_DEFAULT_LAYER_THICKNESS, a, 1];
      p.alpha = [cr, cr];
    }
    return p;
  }
  if (m.transmissionWeight > 0.5) {
    if (f.thin && m.thinWalled) {
      p.kind = PT_BSDF_THIN;
      p.ior = m.ior;
      p.reflectance = (options.transmissionTint ?? true) ? ([...m.transmissionColor] as PtVec3) : [1, 1, 1];
      return p;
    }
    p.kind = PT_BSDF_DIELECTRIC;
    p.alpha = [a, a];
    p.ior = m.ior;
    return p;
  }
  const base: PtVec3 = [m.baseColor[0] * m.baseWeight, m.baseColor[1] * m.baseWeight, m.baseColor[2] * m.baseWeight];
  if (f.layered) {
    const coat = m.coatWeight > 0.5;
    const ior = coat ? m.coatIor : m.ior;
    if ((coat || m.specularWeight > 0) && ior !== 1) {
      const r = coat ? m.coatRoughness : m.specularRoughness;
      const s: PtVec3 = coat ? [1, 1, 1] : [m.specularColor[0] * m.specularWeight, m.specularColor[1] * m.specularWeight, m.specularColor[2] * m.specularWeight];
      p.kind = PT_BSDF_LAYERED;
      p.reflectance = base;
      p.ior = ior;
      p.alpha = [r * r, r * r];
      p.eta = s;
      p.k = [options.layerThickness ?? PT_DEFAULT_LAYER_THICKNESS, ptAlbedoBaseInternal(options.layered) ? ptInternalReflectanceFit(ior) : 0, 0];
      return p;
    }
  }
  if (Math.max(...base) > 0) {
    p.kind = PT_BSDF_DIFFUSE;
    p.reflectance = base;
  }
  return p;
}

/**
 * R1: the features a kernel that shades only reduced vertices keeps (the wavefront's secondary shade kernel): no layered walk or albedo
 * layering, no fuzz, no thin film, no anisotropy, no coated metal. `mix` stays (the reduced adapter's lobe choice is a stochastic mix, like the metal / dielectric one).
 */
export function ptReducedKernelFeatures(f: PtMaterialFeatures): PtMaterialFeatures {
  return { mix: f.mix, thin: f.thin, layered: false, medium: f.medium };
}

/** Egan-Hilgeman style cosine-average internal reflectance used by the reduced base: the same fit as `ptInternalReflectanceFit`. */
const redInternal = ptInternalReflectanceFit;

/** Isotropic GGX alpha of a metal for roughness `r` and anisotropy `a`: the mean of the two axes of `ptOpenPbrAlpha`. */
export function ptReducedAlpha(roughness: number, anisotropy: number): number {
  const [ax, ay] = ptOpenPbrAlpha(roughness, anisotropy);
  return 0.5 * (ax + ay);
}

/** E(mu): directional albedo of the dielectric interface's reflection lobe (the layered kernel's table; exact Fresnel when smooth), roughness `r` perceptual. */
function redTop(ior: number, r: number, mu: number): number {
  if (ptIsSmooth([r * r, r * r])) return ptFrDielectric(Math.max(mu, 1e-6), ior);
  return ptLayeredAlbedo(ptDielectricF0(ior), mu, r);
}

/** Gauss-Legendre nodes and weights on [0, 1] (6 points): the cosine-weighted mean of E, 2 integral of mu E(mu) dmu, is taken with them. */
const GL6_X = [0.033765242898423986, 0.16939530676686776, 0.3806904069584016, 0.6193095930415985, 0.8306046932331322, 0.9662347571015760] as const;
const GL6_W = [0.08566224618958517, 0.18038078652406930, 0.23395696728634552, 0.23395696728634552, 0.18038078652406930, 0.08566224618958517] as const;

/** Cosine-weighted hemispherical mean of `redTop`: the share of light the interface sends back when it arrives diffusely. */
function redTopMean(ior: number, r: number): number {
  let sum = 0;
  for (let i = 0; i < 6; i++) sum += (GL6_W[i] as number) * 2 * (GL6_X[i] as number) * redTop(ior, r, GL6_X[i] as number);
  return sum;
}

/** k of the conductor with eta = 1 whose Fresnel reflectance at cosine `mu` is `target` (bisection on the normal-incidence reflectance, 10 steps; WGSL `ptRedFitK` is the twin). */
function redFitK(target: number, mu: number): number {
  let lo = 0;
  let hi = 0.9999;
  for (let i = 0; i < 10; i++) {
    const mid = 0.5 * (lo + hi);
    if (ptFrComplex(mu, 1, conductorKFromReflectance(mid)) < target) lo = mid;
    else hi = mid;
  }
  return conductorKFromReflectance(0.5 * (lo + hi));
}

/**
 * R1 reduced material (OpenPBR mode; `PtIntegratorOptions.materialLod: 'secondary'`): what a vertex that an indirect ray hits evaluates. Same metal /
 * transmission decisions as `ptBsdfFromOpenPbr`; the lobes that need more than one plain BSDF are folded into ONE lobe chosen at random per vertex
 * (the expectation over the choice is the full material's directional albedo, so the average neither brightens nor darkens; the price is variance):
 *
 *  - dielectric specular lobe or coat over a diffuse base (`PT_BSDF_LAYERED` in the full adapter): a GGX lobe with Fresnel 1 (probability `q`, chosen so that
 *    `q A` is the interface's directional albedo `s E(muO)` of the layered kernel's table, `A` the albedo of the Fresnel-1 lobe) or a Lambert lobe of reflectance
 *    `D / (1 - q)`, `D = rho keep(muO) (1 - s Em) / (eta^2 (1 - rho Ri))` the albedo layering's diffuse albedo with an internal base (`Em` the cosine mean of `E`).
 *    The coat's interface replaces the specular one, as in the full adapter.
 *  - coated metal: a plain conductor (eta 1) whose Fresnel at `muO` is the stack's `E + (1 - E)^2 r_in` (`r_in` the metal seen from inside the coat), alpha = max(metal, coat).
 *  - fuzz (sheen): the layer's albedo `E_f` (a 9 x 9 copy of the LTC table's R) is folded in: with probability `w max(F) E_f` the vertex is a Lambert lobe of colour
 *    `F / max(F)`, and the base lobes' weights are scaled by `1 - w E_f` (`f = w f_fuzz + (1 - w E_f) f_base`).
 *  - anisotropy: the mean of the two alphas. Thin film: not evaluated.
 *
 * `muO` is the cosine between the shading normal and the view direction (the kernel has it before the frame exists). Transmissive materials (thin-walled, rough
 * dielectric) and the absorbing medium are unchanged.
 */
export function ptBsdfReduced(m: RtMaterial, options: PtMaterialOptions, u: number, muO: number): PtBsdfParams {
  const f = options.reducedFrom ?? options.features;
  const k = options.features;
  const p = BLACK_PARAMS();
  const a = alphaOf(m);
  const metal = f.mix ? u < m.metalness : m.metalness > 0.5;
  // the chosen end's own uniform number in [0, 1)
  const uc = f.mix ? (metal ? u / Math.max(m.metalness, 1e-6) : (u - m.metalness) / Math.max(1 - m.metalness, 1e-6)) : u;
  const fuzzOn = f.fuzz === true && m.sheenWeight > 0 && Math.max(m.sheenColor[0], m.sheenColor[1], m.sheenColor[2]) > 0 && (metal || !(m.transmissionWeight > 0.5));
  const fMax = Math.max(m.sheenColor[0], m.sheenColor[1], m.sheenColor[2]);
  const we = fuzzOn ? m.sheenWeight * ptRedFuzzAlbedo(muO, m.sheenRoughness) : 0; // w E_f: the share of the base the fuzz hides
  const qf = fuzzOn ? we * fMax : 0; // probability of the fuzz outcome
  const fuzzLobe = (): PtBsdfParams => {
    const q = BLACK_PARAMS();
    q.kind = PT_BSDF_DIFFUSE;
    q.reflectance = [m.sheenColor[0] / fMax, m.sheenColor[1] / fMax, m.sheenColor[2] / fMax];
    return q;
  };
  if (metal) {
    p.kind = PT_BSDF_CONDUCTOR;
    const aIso = ptReducedAlpha(m.specularRoughness, m.specularAnisotropy);
    p.alpha = [aIso, aIso];
    let eta: PtVec3;
    let kk: PtVec3;
    if (m.conductorEtaK) {
      eta = [...m.conductorEta] as PtVec3;
      kk = [...m.conductorK] as PtVec3;
    } else {
      eta = [1, 1, 1];
      kk = [0, 1, 2].map((i) => conductorKFromReflectance((m.baseColor[i] as number) * m.baseWeight)) as unknown as PtVec3;
    }
    if (f.coatedMetal && m.coatWeight > 0.5 && m.coatIor !== 1) {
      const ci = m.coatIor;
      const ec = redTop(ci, m.coatRoughness, muO);
      const cosIn = Math.sqrt(Math.max(0, 1 - (1 - muO * muO) / (ci * ci)));
      const r = [0, 1, 2].map((i) => {
        const rIn = ptFrComplex(cosIn, (eta[i] as number) / ci, (kk[i] as number) / ci);
        return Math.min(ec + (1 - ec) * (1 - ec) * rIn, 0.9999);
      });
      // a conductor (eta 1) whose Fresnel at THIS angle is the stack's: its own angular rise is already in the stack's value
      kk = r.map((t) => redFitK(t, muO)) as unknown as PtVec3;
      eta = [1, 1, 1];
      const cr = m.coatRoughness * m.coatRoughness;
      p.alpha = [Math.max(aIso, cr), Math.max(aIso, cr)];
    }
    p.eta = eta;
    p.k = kk;
    if (fuzzOn) {
      if (uc < qf) return fuzzLobe();
      if (uc >= qf + (1 - we)) return BLACK_PARAMS(); // what the fuzz layer absorbs
    }
    return p;
  }
  if (m.transmissionWeight > 0.5) {
    if (k.thin && m.thinWalled) {
      p.kind = PT_BSDF_THIN;
      p.ior = m.ior;
      p.reflectance = (options.transmissionTint ?? true) ? ([...m.transmissionColor] as PtVec3) : [1, 1, 1];
      return p;
    }
    p.kind = PT_BSDF_DIELECTRIC;
    p.alpha = [a, a];
    p.ior = m.ior;
    return p;
  }
  const base: PtVec3 = [m.baseColor[0] * m.baseWeight, m.baseColor[1] * m.baseWeight, m.baseColor[2] * m.baseWeight];
  let q = 0; // probability of the GGX lobe
  let d: PtVec3 = base; // albedo of the diffuse part
  let spec: PtBsdfParams | null = null;
  if (f.layered) {
    const coat = m.coatWeight > 0.5;
    const ior = coat ? m.coatIor : m.ior;
    if ((coat || m.specularWeight > 0) && ior !== 1) {
      const r = coat ? m.coatRoughness : m.specularRoughness;
      const s: PtVec3 = coat ? [1, 1, 1] : [m.specularColor[0] * m.specularWeight, m.specularColor[1] * m.specularWeight, m.specularColor[2] * m.specularWeight];
      // E: directional albedo of the interface's reflection lobe (the layered kernel's table, exact Fresnel when smooth); A: the albedo of the same GGX lobe with Fresnel 1 (the lobe the reduced vertex picks)
      const smooth = ptIsSmooth([r * r, r * r]);
      const e = redTop(ior, r, muO);
      const aCond = smooth ? 1 : Math.max(ptLayeredAlbedo(1, muO, r), 1e-3);
      const em = redTopMean(ior, r);
      const ri = redInternal(ior);
      const sf: PtVec3 = [Math.min(s[0] * e, 1), Math.min(s[1] * e, 1), Math.min(s[2] * e, 1)];
      q = Math.min((sf[0] + sf[1] + sf[2]) / 3 / aCond, 1);
      const sAvg = (s[0] + s[1] + s[2]) / 3;
      spec = BLACK_PARAMS();
      spec.kind = PT_BSDF_CONDUCTOR;
      spec.alpha = [r * r, r * r];
      spec.eta = [1, 1, 1];
      spec.k = [0, 1, 2].map((i) => conductorKFromReflectance(Math.min((s[i] as number) / Math.max(sAvg, 1e-6), 1))) as unknown as PtVec3;
      // rho keep(wo) [1 / (eta^2 (1 - rho Ri))] int keep(wi) cos/pi dwi: the albedo layering's diffuse albedo with the internal base, the interface's own (rough) table inside the integral
      d = [0, 1, 2].map((i) => {
        const rho = Math.min(base[i] as number, 0.9999);
        return (rho * (1 - (sf[i] as number)) * (1 - Math.min((s[i] as number) * em, 1))) / (ior * ior * (1 - rho * ri));
      }) as unknown as PtVec3;
    }
  }
  // outcomes: fuzz [0, qf), GGX lobe next (probability (1 - we) q: its weight stays A), Lambert for the rest (weight (1 - we) D / its probability)
  let ur = uc;
  if (fuzzOn) {
    if (uc < qf) return fuzzLobe();
    ur = uc - qf;
  }
  const qs = (1 - we) * q;
  if (spec && ur < qs) return spec;
  const qd = Math.max(1 - qf - qs, 1e-6);
  const rd: PtVec3 = [Math.min(((1 - we) * (d[0] as number)) / qd, 1), Math.min(((1 - we) * (d[1] as number)) / qd, 1), Math.min(((1 - we) * (d[2] as number)) / qd, 1)];
  if (Math.max(...rd) > 0) {
    p.kind = PT_BSDF_DIFFUSE;
    p.reflectance = rd;
  }
  return p;
}

/** The OpenPBR-mode WGSL twin: `ptBsdfFromMaterial(m, u)` and, with `features.medium`, `ptMediumSigma(m)`. */
export function ptBsdfOpenPbrWgsl(options: PtMaterialOptions): string {
  const f = options.features;
  const tint = options.transmissionTint ?? true;
  const thickness = options.layerThickness ?? PT_DEFAULT_LAYER_THICKNESS;
  const internal = ptAlbedoBaseInternal(options.layered);
  const num = (v: number): string => (Number.isInteger(v) ? `${v}.0` : String(v));
  return /* wgsl */ `
fn ptConductorKFromReflectance(r: vec3f) -> vec3f {
    let c = clamp(r, vec3f(0.0), vec3f(0.9999));
    return 2.0 * sqrt(c) / sqrt(1.0 - c);
}
${f.layered && internal ? `
fn ptInternalReflectanceFit(ior: f32) -> f32 {
    return clamp(-1.44 / (ior * ior) + 0.71 / ior + 0.668 + 0.0636 * ior, 0.0, 1.0);
}
` : ''}${f.medium ? `
fn ptMediumSigma(m: RtMaterial) -> vec3f {
    if (m.thinWalled || !(m.transmissionDepth > 0.0)) {
        return vec3f(0.0);
    }
    return -log(max(m.transmissionColor, vec3f(1.0e-6))) / m.transmissionDepth;
}
` : ''}${f.aniso ? `
fn ptOpenPbrAlpha(r: f32, a: f32) -> vec2f {
    if (!(a > 0.0)) {
        return vec2f(r * r);
    }
    let ax = r * r * sqrt(2.0 / (1.0 + (1.0 - a) * (1.0 - a)));
    return vec2f(ax, (1.0 - a) * ax);
}
` : ''}
fn ptBsdfFromMaterial(m: RtMaterial, u: f32) -> PtBsdfParams {
    var p: PtBsdfParams;
    p.kind = PT_BSDF_BLACK;
    p.ior = 1.0;
    p.alpha = vec2f(0.0);
    p.reflectance = vec3f(0.0);
    p.eta = vec3f(0.0);
    p.k = vec3f(0.0);
    let a = m.specularRoughness * m.specularRoughness;
    if (${f.mix ? 'u < m.metalness' : 'm.metalness > 0.5'}) {
        p.kind = PT_BSDF_CONDUCTOR;
        ${f.aniso ? 'p.alpha = ptOpenPbrAlpha(m.specularRoughness, m.specularAnisotropy);' : 'p.alpha = vec2f(a);'}
        if (m.conductorEtaK) {
            p.eta = m.conductorEta;
            p.k = m.conductorK;
        } else {
            p.eta = vec3f(1.0);
            p.k = ptConductorKFromReflectance(m.baseColor * m.baseWeight);
        }
${f.coatedMetal ? `        if (m.coatWeight > 0.5 && m.coatIor != 1.0) {
            let cr = m.coatRoughness * m.coatRoughness;
            p.kind = PT_BSDF_LAYERED;
            p.reflectance = p.eta / m.coatIor;
            p.eta = p.k / m.coatIor;
            p.ior = m.coatIor;
            p.k = vec3f(${num(thickness)}, a, 1.0);
            p.alpha = vec2f(cr);
        }
` : ''}        return p;
    }
    if (m.transmissionWeight > 0.5) {
${f.thin ? `        if (m.thinWalled) {
            p.kind = PT_BSDF_THIN;
            p.ior = m.ior;
            p.reflectance = ${tint ? 'm.transmissionColor' : 'vec3f(1.0)'};
            return p;
        }
` : ''}        p.kind = PT_BSDF_DIELECTRIC;
        p.alpha = vec2f(a);
        p.ior = m.ior;
        return p;
    }
    let r = m.baseColor * m.baseWeight;
${f.layered ? `    let coat = m.coatWeight > 0.5;
    let li = select(m.ior, m.coatIor, coat);
    if ((coat || m.specularWeight > 0.0) && li != 1.0) {
        let lr = select(m.specularRoughness, m.coatRoughness, coat);
        p.kind = PT_BSDF_LAYERED;
        p.reflectance = r;
        p.ior = li;
        p.alpha = vec2f(lr * lr);
        p.eta = select(m.specularColor * m.specularWeight, vec3f(1.0), coat);
        p.k = vec3f(${num(thickness)}, ${internal ? 'ptInternalReflectanceFit(li)' : '0.0'}, 0.0);
        return p;
    }
` : ''}    if (max(r.x, max(r.y, r.z)) > 0.0) {
        p.kind = PT_BSDF_DIFFUSE;
        p.reflectance = r;
    }
    return p;
}
`;
}

/**
 * The WGSL twin of `ptBsdfReduced` (R1): `ptBsdfReduced(m, u, muO)`. Append after `ptBsdfAdapterWgsl(options)` (it uses `ptConductorKFromReflectance` and
 * `ptFrDielectric`). What it compiles follows `options.reducedFrom ?? options.features` (which reductions exist) and `options.features` (thin).
 */
export function ptBsdfReducedWgsl(options: PtMaterialOptions): string {
  const f = options.reducedFrom ?? options.features;
  const k = options.features;
  const tint = options.transmissionTint ?? true;
  // The albedo table lives in the layered module (albedo layering only); a kernel without it (secondary wavefront kernel, stochastic megakernel) carries its own copy.
  const ownLut = f.layered && !(k.layered && (options.layered?.layering ?? 'albedo') === 'albedo');
  const fuzz = f.fuzz === true;
  return /* wgsl */ `${ownLut ? ptLayerLutWgsl() : ''}${fuzz ? ptRedFuzzWgsl() : ''}
fn ptRedInternal(ior: f32) -> f32 {
    return clamp(-1.44 / (ior * ior) + 0.71 / ior + 0.668 + 0.0636 * ior, 0.0, 1.0);
}
${f.layered ? `fn ptRedTop(ior: f32, r: f32, mu: f32) -> f32 {
    if (ptIsSmooth(vec2f(r * r))) {
        return ptFrDielectric(max(mu, 1.0e-6), ior);
    }
    let t = (ior - 1.0) / (ior + 1.0);
    return ptLayerAlbedoLut(t * t, mu, r);
}
fn ptRedTopMean(ior: f32, r: f32) -> f32 {
    return ${GL6_X.map((x, i) => `${((GL6_W[i] as number) * 2 * x).toFixed(8)} * ptRedTop(ior, r, ${x.toFixed(8)})`).join('\n        + ')};
}
` : ''}${f.coatedMetal ? `
fn ptRedFitK(tgt: vec3f, mu: f32) -> vec3f {
    var lo = vec3f(0.0);
    var hi = vec3f(0.9999);
    for (var i = 0; i < 10; i++) {
        let mid = 0.5 * (lo + hi);
        let below = ptFrComplex(mu, vec3f(1.0), ptConductorKFromReflectance(mid)) < tgt;
        lo = select(lo, mid, below);
        hi = select(mid, hi, below);
    }
    return ptConductorKFromReflectance(0.5 * (lo + hi));
}` : ''}
fn ptRedBlack() -> PtBsdfParams {
    var p: PtBsdfParams;
    p.kind = PT_BSDF_BLACK;
    p.ior = 1.0;
    p.alpha = vec2f(0.0);
    p.reflectance = vec3f(0.0);
    p.eta = vec3f(0.0);
    p.k = vec3f(0.0);
    return p;
}

fn ptBsdfReduced(m: RtMaterial, u: f32, muO: f32) -> PtBsdfParams {
    var p = ptRedBlack();
    let a = m.specularRoughness * m.specularRoughness;
    let metal = ${f.mix ? 'u < m.metalness' : 'm.metalness > 0.5'};
    let uc = ${f.mix ? 'select((u - m.metalness) / max(1.0 - m.metalness, 1.0e-6), u / max(m.metalness, 1.0e-6), metal)' : 'u'};
${fuzz ? `    let fMax = max(m.sheenColor.x, max(m.sheenColor.y, m.sheenColor.z));
    let fuzzOn = m.sheenWeight > 0.0 && fMax > 0.0 && (metal || !(m.transmissionWeight > 0.5));
    var we = 0.0;
    if (fuzzOn) {
        we = m.sheenWeight * ptRedFuzzAlbedo(muO, m.sheenRoughness);
    }
    let qf = we * fMax;
    var fuzzLobe = ptRedBlack();
    fuzzLobe.kind = PT_BSDF_DIFFUSE;
    fuzzLobe.reflectance = m.sheenColor / max(fMax, 1.0e-6);
` : `    let we = 0.0;
    let qf = 0.0;
`}    if (metal) {
        p.kind = PT_BSDF_CONDUCTOR;
        var aIso = a;
        if (m.specularAnisotropy > 0.0) {
            let ax = a * sqrt(2.0 / (1.0 + (1.0 - m.specularAnisotropy) * (1.0 - m.specularAnisotropy)));
            aIso = 0.5 * (ax + (1.0 - m.specularAnisotropy) * ax);
        }
        p.alpha = vec2f(aIso);
        var eta = vec3f(1.0);
        var kk = ptConductorKFromReflectance(m.baseColor * m.baseWeight);
        if (m.conductorEtaK) {
            eta = m.conductorEta;
            kk = m.conductorK;
        }
${f.coatedMetal ? `        if (m.coatWeight > 0.5 && m.coatIor != 1.0) {
            let ci = m.coatIor;
            let ec = ptRedTop(ci, m.coatRoughness, muO);
            let cosIn = sqrt(max(0.0, 1.0 - (1.0 - muO * muO) / (ci * ci)));
            let rIn = ptFrComplex(cosIn, eta / ci, kk / ci);
            let rr = min(vec3f(ec) + ((1.0 - ec) * (1.0 - ec)) * rIn, vec3f(0.9999));
            kk = ptRedFitK(rr, muO);
            eta = vec3f(1.0);
            let cr = m.coatRoughness * m.coatRoughness;
            p.alpha = vec2f(max(aIso, cr));
        }
` : ''}        p.eta = eta;
        p.k = kk;
${fuzz ? `        if (fuzzOn) {
            if (uc < qf) {
                return fuzzLobe;
            }
            if (uc >= qf + (1.0 - we)) {
                return ptRedBlack();
            }
        }
` : ''}        return p;
    }
    if (m.transmissionWeight > 0.5) {
${k.thin ? `        if (m.thinWalled) {
            p.kind = PT_BSDF_THIN;
            p.ior = m.ior;
            p.reflectance = ${tint ? 'm.transmissionColor' : 'vec3f(1.0)'};
            return p;
        }
` : ''}        p.kind = PT_BSDF_DIELECTRIC;
        p.alpha = vec2f(a);
        p.ior = m.ior;
        return p;
    }
    let base = m.baseColor * m.baseWeight;
    var q = 0.0;
    var d = base;
    var spec = ptRedBlack();
${f.layered ? `    let coat = m.coatWeight > 0.5;
    let li = select(m.ior, m.coatIor, coat);
    if ((coat || m.specularWeight > 0.0) && li != 1.0) {
        let lr = select(m.specularRoughness, m.coatRoughness, coat);
        let s = select(m.specularColor * m.specularWeight, vec3f(1.0), coat);
        let isSm = ptIsSmooth(vec2f(lr * lr));
        let e = ptRedTop(li, lr, muO);
        let aCond = select(max(ptLayerAlbedoLut(1.0, muO, lr), 1.0e-3), 1.0, isSm);
        let em = ptRedTopMean(li, lr);
        let ri = ptRedInternal(li);
        let sf = min(s * e, vec3f(1.0));
        q = min((sf.x + sf.y + sf.z) / 3.0 / aCond, 1.0);
        let sAvg = (s.x + s.y + s.z) / 3.0;
        spec.kind = PT_BSDF_CONDUCTOR;
        spec.alpha = vec2f(lr * lr);
        spec.eta = vec3f(1.0);
        spec.k = ptConductorKFromReflectance(min(s / max(sAvg, 1.0e-6), vec3f(1.0)));
        let rho = min(base, vec3f(0.9999));
        d = (rho * (vec3f(1.0) - sf) * (vec3f(1.0) - min(s * em, vec3f(1.0)))) / (li * li * (vec3f(1.0) - rho * ri));
    }
` : ''}    var ur = uc;
${fuzz ? `    if (fuzzOn) {
        if (uc < qf) {
            return fuzzLobe;
        }
        ur = uc - qf;
    }
` : ''}    let qs = (1.0 - we) * q;
    if (ur < qs) {
        return spec;
    }
    let qd = max(1.0 - qf - qs, 1.0e-6);
    let rd = min((1.0 - we) * d / qd, vec3f(1.0));
    if (max(rd.x, max(rd.y, rd.z)) > 0.0) {
        p.kind = PT_BSDF_DIFFUSE;
        p.reflectance = rd;
    }
    return p;
}
`;
}

/**
 * Kernel glue for `PT_BSDF_LAYERED` (append after `ptLayeredWgsl`): the layered kind is evaluated by `ptLayeredEval/Sample`, every other kind
 * by `ptBsdfEval/Sample`. The stochastic walk needs a seed: the integrator sets the private `ptLayerSeed` once per vertex (a hash of pixel,
 * sample and vertex); the walk mixes the directions in itself. `PtBsdfParams` is reused for the layered parameters (see `PT_BSDF_LAYERED`).
 */
export function ptLayeredDispatchWgsl(coatedMetal = false): string {
  return /* wgsl */ `
const PT_BSDF_LAYERED: u32 = 5u;
var<private> ptLayerSeed: u32;
${coatedMetal ? `fn ptLayeredOf(p: PtBsdfParams) -> PtLayeredParams {
    if (p.k.z > 0.5) {
        return PtLayeredParams(vec3f(0.0), p.ior, p.alpha, 1.0, vec3f(1.0), p.k.x, 0.0, p.reflectance, p.eta, p.k.y, true);
    }
    return PtLayeredParams(p.reflectance, p.ior, p.alpha, 1.0, p.eta, p.k.x, p.k.y, vec3f(0.0), vec3f(0.0), 0.0, false);
}` : `fn ptLayeredOf(p: PtBsdfParams) -> PtLayeredParams {
    return PtLayeredParams(p.reflectance, p.ior, p.alpha, 1.0, p.eta, p.k.x, p.k.y);
}`}
fn ptEval(p: PtBsdfParams, wo: vec3f, wi: vec3f) -> PtBsdfEval {
    if (p.kind == PT_BSDF_LAYERED) {
        return ptLayeredEval(ptLayeredOf(p), wo, wi, ptLayerSeed);
    }
    return ptBsdfEval(p, wo, wi);
}
fn ptSample(p: PtBsdfParams, wo: vec3f, u: vec3f) -> PtBsdfSample {
    if (p.kind == PT_BSDF_LAYERED) {
        return ptLayeredSample(ptLayeredOf(p), wo, u, ptLayerSeed);
    }
    return ptBsdfSample(p, wo, u);
}
`;
}
