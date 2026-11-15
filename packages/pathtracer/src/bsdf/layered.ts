/**
 * P2.21: dielectric-over-diffuse layered BSDF (the glTF default non-metal, OpenPBR's dielectric base), TS twin of
 * `layered-wgsl.ts` (prefix `pt`). Two ways to layer, one setting (`PtLayeredOptions.layering`):
 *
 *  - `'albedo'`: OpenPBR Surface's albedo scaling (specification, section "Layering": `layer(top, base) = top + base *
 *    (1 - E_top)`, E_top the directional albedo of the slab above, taken at the view direction). The top is the GGX
 *    dielectric reflection lobe; the Lambert base is scaled by `1 - E(mu_o)` from a small table (`layered-lut.ts`).
 *    Deterministic, cheap, energy conserving by construction, NOT physical layering (no refraction into the layer, no
 *    internal reflection, no base-roughening); DIVERGES V57.
 *  - `'stochastic'`: pbrt-v4's `LayeredBxDF` random walk between the dielectric interface and the base, specialised to
 *    `CoatedDiffuseBxDF` (top = `DielectricBxDF`, bottom = `DiffuseBxDF`, two-sided, no scattering medium between the
 *    layers: `albedo = 0`, `g` unused). Reference mode: `f` and `pdf` are stochastic estimates (the sampler returns
 *    `PT_FLAG_PROPORTIONAL`: use the sample's `f * cos / pdf` as the weight, and `ptLayeredPdf` for MIS).
 *
 * Ported from pbrt-v4 (Apache-2.0, Copyright 1998-2020 Matt Pharr, Wenzel Jakob, Greg Humphreys): `LayeredBxDF::f`,
 * `Sample_f`, `PDF` and `CoatedDiffuseBxDF` (bxdfs.h), `DielectricBxDF` sampling with `BxDFReflTransFlags` (bxdfs.cpp),
 * `PowerHeuristic`. Departures (DIVERGES V58): a counter-based PCG hash instead of pbrt's `RNG` + `Hash` (so the WGSL
 * twin draws the same numbers); the top interface carries OpenPBR's `specular_weight * specular_color` as a per-channel
 * scale of its Fresnel reflectance (pbrt's is scalar); the medium (`albedo`, `g`) is not ported.
 *
 * `specular_weight` / `specular_color` (OpenPBR, "Specular"): s = weight * color scales the dielectric reflectance F at
 * every angle (F0 and F90 alike, unlike glTF `KHR_materials_specular`, which scales F0 only and keeps F90 = 1).
 * Reflection R' = s F; transmission into the layer T' = 1 - R'. Total internal reflection stays total (R' = 1: the
 * light cannot leave, scaling it would destroy energy). The albedo method scales the specular lobe by s and the base by
 * 1 - s E.
 *
 * Frame, conventions, `wo.z < 0` as in `bsdf.ts`: two-sided, so a ray arriving from below the surface sees the same
 * layers (wo and wi are negated, the sampled wi is negated back).
 */

import { ptFilmActive, ptFilmFresnel, type PtThinFilm } from './thinfilm.ts';
import {
  PT_BSDF_CONDUCTOR,
  PT_BSDF_DIFFUSE,
  PT_FLAG_DELTA,
  PT_FLAG_REFLECTION,
  PT_FLAG_TRANSMISSION,
  dielectricHalf,
  ptBsdfEval,
  ptBsdfSample,
  ptFrComplex,
  ptFrDielectric,
  ptGgxD,
  ptGgxG,
  ptGgxSampleWm,
  ptGgxVisiblePdf,
  ptIsSmooth,
  ptRefract,
  PT_MIN_ALPHA,
  type PtBsdfEval,
  type PtBsdfOptions,
  type PtBsdfParams,
  type PtBsdfSample,
  type PtVec3,
} from './bsdf.ts';
import { ptDielectricF0, ptLayeredAlbedo } from './layered-lut.ts';

/** Set on the sample of a stochastic layered BSDF: `pdf` is only proportional to the true density (pbrt `pdfIsProportional`). */
export const PT_FLAG_PROPORTIONAL = 8;

/** `BxDFReflTransFlags` for the interface helpers. */
export const PT_LAYER_REFLECTION = 1;
export const PT_LAYER_TRANSMISSION = 2;
export const PT_LAYER_ALL = 3;

/** pbrt clamps the layer thickness to at least the smallest float, so a zero thickness still alternates the interfaces. */
export const PT_LAYER_MIN_THICKNESS = 1e-30;

/**
 * Plain parameters. `alpha` is the GGX alpha of the dielectric interface (roughness^2). `thickness` is pbrt's
 * `coateddiffuse` thickness (default 0.01: it attenuates the walk by exp(-thickness / |cos|) per crossing, with a unit
 * extinction coefficient); 0 means no attenuation (white-furnace tests). Used by the stochastic method only.
 */
export interface PtLayeredParams {
  baseColor: PtVec3;
  ior: number;
  alpha: readonly [number, number];
  specularWeight: number;
  specularColor: PtVec3;
  thickness: number;
  /** Cosine-weighted average Fresnel reflectance of the interface seen from inside (`ptInternalReflectance(ior)`); `albedoBase: 'internal'` only. */
  internalReflectance: number;
  /**
   * P2.23: a GGX conductor as the bottom layer instead of the Lambert base (pbrt `coatedconductor`). `eta`/`k` are already relative to the
   * coat medium (pbrt divides the conductor's by the interface eta; `ptLayeredCoatedConductor` does), `alpha` is the conductor's isotropic
   * GGX alpha. `baseColor` is then unused.
   */
  baseConductor?: { eta: PtVec3; k: PtVec3; alpha: number };
  /** P2.27: thin film on the dielectric interface (the film sits between the ambient medium and the dielectric of IOR `ior`). Albedo layering only in the kernel. */
  film?: PtThinFilm;
}

/** D24: albedo layering defaults to the `'internal'` base; an explicit `albedoBase` wins; the stochastic walk has no albedo base (default `'openpbr'`, so its kernel text is unchanged). */
export function ptAlbedoBaseInternal(o: { layering?: 'albedo' | 'stochastic'; albedoBase?: 'openpbr' | 'internal' } | undefined): boolean {
  return (o?.albedoBase ?? ((o?.layering ?? 'albedo') === 'albedo' ? 'internal' : 'openpbr')) === 'internal';
}

export interface PtLayeredOptions extends PtBsdfOptions {
  /** Default `'albedo'` (the fast default; `'stochastic'` is the reference mode). */
  layering?: 'albedo' | 'stochastic';
  /** Stochastic: walk length (pbrt `maxdepth`, default 10). */
  maxDepth?: number;
  /** Stochastic: independent walks per evaluation (pbrt `nsamples`, default 1). */
  nSamples?: number;
  /**
   * Albedo layering only. Default (D24): `'internal'` with `layering: 'albedo'`; `'openpbr'` is the setting (see `ptAlbedoBaseInternal`). `'openpbr'`: base scaled by `1 - E(mu_o)` and nothing else (the specification's
   * albedo scaling). `'internal'`: the closed-form of a diffuse base under a dielectric (Weidlich and Wilkie 2007; the
   * form Mitsuba's `roughplastic` uses with `nonlinear: true`, written from the formula, not its code): base *
   * (1 - E(mu_o)) (1 - E(mu_i)) / (eta^2 (1 - rho R_i)), which adds the light the interface sends back down (the
   * darkening of a bright base) and is reciprocal. Exact for a smooth interface with s = 1; approximate otherwise.
   * DIVERGES V57.
   */
  albedoBase?: 'openpbr' | 'internal';
  /**
   * Stochastic, rough interface only. `'pbrt'` (default) weights the two exit estimators with pbrt's power heuristic, which
   * compares a density over inside directions with one over outside directions (a Jacobian apart): pbrt's own `f` then reads
   * +3.5% (roughness 0.3) to +7% (0.5) above its own `Sample_f` for a white base. `'none'` keeps only the presampled
   * next-event estimate (weight 1): `f` and the sampler agree. DIVERGES V58.
   */
  exitMis?: 'pbrt' | 'none';
  /**
   * WGSL only (the CPU twin always has it): compile the conductor bottom layer (`baseConductor`, P2.23 coat over a metal) into the kernel.
   * Default false: a scene without coated metals keeps the kernel text and size.
   */
  conductorBase?: boolean;
  /** Stochastic: 32-bit seed mixed into the hash that drives every walk (the caller varies it per pixel/frame). */
  seed?: number;
}

/** Constructor with OpenPBR defaults. `roughness` is perceptual (alpha = roughness^2). */
export function ptLayered(baseColor: PtVec3, ior: number, roughness: number, extra: Partial<Pick<PtLayeredParams, 'specularWeight' | 'specularColor' | 'thickness'>> = {}): PtLayeredParams {
  const a = roughness * roughness;
  return {
    baseColor,
    ior,
    alpha: [a, a],
    specularWeight: extra.specularWeight ?? 1,
    specularColor: extra.specularColor ?? [1, 1, 1],
    thickness: extra.thickness ?? 0.01,
    internalReflectance: ptInternalReflectance(ior),
  };
}

/**
 * P2.23: coat over a conductor (OpenPBR `coat_weight` over `base_metalness` 1; pbrt `coatedconductor`). `eta`/`k` are the conductor's complex
 * IOR in air; like pbrt they are divided by the coat IOR (the conductor sits under the coat, not under air). `baseRoughness` and
 * `coatRoughness` are perceptual (alpha = roughness^2). The specular weight/colour scale is 1 (OpenPBR's applies to the dielectric base only).
 */
export function ptLayeredCoatedConductor(eta: PtVec3, k: PtVec3, baseRoughness: number, coatIor: number, coatRoughness: number, extra: Partial<Pick<PtLayeredParams, 'thickness'>> = {}): PtLayeredParams {
  const d = (v: PtVec3): PtVec3 => [v[0] / coatIor, v[1] / coatIor, v[2] / coatIor];
  const a = coatRoughness * coatRoughness;
  return {
    baseColor: ZERO3,
    ior: coatIor,
    alpha: [a, a],
    specularWeight: 1,
    specularColor: [1, 1, 1],
    thickness: extra.thickness ?? 0.01,
    internalReflectance: 0,
    baseConductor: { eta: d(eta), k: d(k), alpha: baseRoughness * baseRoughness },
  };
}

/** Cosine-weighted average of the Fresnel reflectance seen from inside a dielectric of relative IOR `ior` (total internal reflection included). */
export function ptInternalReflectance(ior: number): number {
  if (ior <= 1) return 0;
  const n = 2048;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const c = (i + 0.5) / n;
    sum += ptFrDielectric(-c, ior) * c;
  }
  return Math.min(1, (2 * sum) / n);
}

const PI = Math.PI;
const ZERO3: PtVec3 = [0, 0, 0];
const neg = (a: PtVec3): PtVec3 => [-a[0], -a[1], -a[2]];
const maxc = (a: PtVec3): number => Math.max(a[0], a[1], a[2]);
const avg = (a: PtVec3): number => (a[0] + a[1] + a[2]) / 3;
const isBlack = (a: PtVec3): boolean => a[0] === 0 && a[1] === 0 && a[2] === 0;
const mul3 = (a: PtVec3, b: PtVec3): PtVec3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const scale3 = (a: PtVec3, s: number): PtVec3 => [a[0] * s, a[1] * s, a[2] * s];

function powerHeuristic(f: number, g: number): number {
  if (!Number.isFinite(f * f)) return 1;
  return (f * f) / (f * f + g * g);
}

function specScale(p: PtLayeredParams): PtVec3 {
  return scale3(p.specularColor, p.specularWeight);
}

function baseParams(p: PtLayeredParams): PtBsdfParams {
  const c = p.baseConductor;
  if (c) return { kind: PT_BSDF_CONDUCTOR, reflectance: ZERO3, eta: c.eta, k: c.k, alpha: [c.alpha, c.alpha], ior: 1 };
  return { kind: PT_BSDF_DIFFUSE, reflectance: p.baseColor, eta: ZERO3, k: ZERO3, alpha: [0, 0], ior: 1 };
}

/** True when the bottom layer is a delta lobe (a smooth conductor; the Lambert base never is). */
export function ptLayeredSmoothBase(p: PtLayeredParams): boolean {
  return !!p.baseConductor && ptIsSmooth([p.baseConductor.alpha, p.baseConductor.alpha]);
}

/** True when the dielectric interface is a delta lobe (smooth or eta = 1): the specular reflection is sampled, never evaluated. */
export function ptLayeredSmoothTop(p: PtLayeredParams): boolean {
  return p.ior === 1 || ptIsSmooth(p.alpha);
}

// ---------------------------------------------------------------------------------------------------------------
// Counter-based PCG hash RNG (Jarzynski and Olano 2020, `pcg_hash`), identical in WGSL. Every walk is a pure function of
// (seed, wo, wi) or (seed, wo, u), as pbrt's hashed `RNG` is.

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
function bits(x: number): number {
  f32buf[0] = x;
  return u32buf[0] as number;
}
export function ptPcg(v: number): number {
  const state = (Math.imul(v, 747796405) + 2891336453) >>> 0;
  const word = Math.imul((state >>> ((state >>> 28) + 4)) ^ state, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}
function hashSeed(seed: number, vals: readonly number[]): number {
  let h = ptPcg(seed >>> 0);
  for (const v of vals) h = ptPcg((h ^ bits(v)) >>> 0);
  return h;
}
class Rng {
  state: number;
  constructor(state: number) {
    this.state = state;
  }
  next(): number {
    this.state = ptPcg(this.state);
    return (this.state >>> 8) * 5.9604644775390625e-8;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The dielectric interface with restricted sampling (pbrt `DielectricBxDF` + `BxDFReflTransFlags`), chromatic Fresnel scale.

interface Top {
  ior: number;
  alpha: readonly [number, number];
  s: PtVec3;
  film?: PtThinFilm;
}

function topOf(p: PtLayeredParams): Top {
  return { ior: p.ior, alpha: p.alpha, s: specScale(p), ...(ptFilmActive(p.film) ? { film: p.film } : {}) };
}

const dielectricF0 = (ior: number): PtVec3 => {
  const f = ((ior - 1) / (ior + 1)) ** 2;
  return [f, f, f];
};

/** The interface's Fresnel per channel at the cosine `cos` (the half-vector angle for a rough lobe): the plain dielectric, or the thin film blended in. */
function topFresnel(top: Top, cos: number): PtVec3 {
  const F = ptFrDielectric(cos, top.ior);
  return top.film ? ptFilmFresnel(top.film, cos, [F, F, F], dielectricF0(top.ior)) : [F, F, F];
}

/** R' = s F, but total internal reflection stays total. */
function reflectance(top: Top, F: number, cos: number): PtVec3 {
  if (F >= 1) return [1, 1, 1];
  const Fv = top.film ? topFresnel(top, cos) : [F, F, F];
  return [Math.min(1, top.s[0] * (Fv[0] as number)), Math.min(1, top.s[1] * (Fv[1] as number)), Math.min(1, top.s[2] * (Fv[2] as number))];
}

function roughA(alpha: readonly [number, number]): readonly [number, number] {
  return [Math.max(alpha[0], PT_MIN_ALPHA), Math.max(alpha[1], PT_MIN_ALPHA)];
}

function topIsSpecular(top: Top): boolean {
  return top.ior === 1 || ptIsSmooth(top.alpha);
}

const FAIL: PtBsdfSample = { wi: ZERO3, f: ZERO3, pdf: 0, flags: 0, etap: 1 };

function topSample(top: Top, wo: PtVec3, uc: number, u0: number, u1: number, radiance: boolean, flags: number, vndf: 'standard' | 'cap'): PtBsdfSample {
  const eta = top.ior;
  if (topIsSpecular(top)) {
    const F = ptFrDielectric(wo[2], eta);
    const Rv = reflectance(top, F, wo[2]);
    const Tv: PtVec3 = [1 - Rv[0], 1 - Rv[1], 1 - Rv[2]];
    const pr = flags & PT_LAYER_REFLECTION ? avg(Rv) : 0;
    const pt = flags & PT_LAYER_TRANSMISSION ? avg(Tv) : 0;
    if (pr + pt === 0) return FAIL;
    if (uc < pr / (pr + pt)) {
      const c = Math.abs(wo[2]);
      return { wi: [-wo[0], -wo[1], wo[2]], f: scale3(Rv, 1 / c), pdf: pr / (pr + pt), flags: PT_FLAG_REFLECTION | PT_FLAG_DELTA, etap: 1 };
    }
    const r = ptRefract(wo, [0, 0, 1], eta);
    if (!r) return FAIL;
    const c = Math.abs(r.wt[2]);
    let k = 1 / c;
    if (radiance) k /= r.etap * r.etap;
    return { wi: r.wt, f: scale3(Tv, k), pdf: pt / (pr + pt), flags: PT_FLAG_TRANSMISSION | PT_FLAG_DELTA, etap: r.etap };
  }
  const a = roughA(top.alpha);
  if (wo[2] === 0) return FAIL;
  const wm = ptGgxSampleWm(a, wo, u0, u1, vndf);
  const dwo = wo[0] * wm[0] + wo[1] * wm[1] + wo[2] * wm[2];
  const F = ptFrDielectric(dwo, eta);
  const Rv = reflectance(top, F, dwo);
  const Tv: PtVec3 = [1 - Rv[0], 1 - Rv[1], 1 - Rv[2]];
  const pr = flags & PT_LAYER_REFLECTION ? avg(Rv) : 0;
  const pt = flags & PT_LAYER_TRANSMISSION ? avg(Tv) : 0;
  if (pr + pt === 0) return FAIL;
  const vis = ptGgxVisiblePdf(a, wo, wm);
  if (uc < pr / (pr + pt)) {
    const d2 = 2 * dwo;
    const wi: PtVec3 = [-wo[0] + d2 * wm[0], -wo[1] + d2 * wm[1], -wo[2] + d2 * wm[2]];
    if (wi[2] * wo[2] <= 0) return FAIL;
    const pdf = (vis / (4 * Math.abs(dwo))) * (pr / (pr + pt));
    const k = (ptGgxD(a, wm) * ptGgxG(a, wo, wi)) / (4 * wi[2] * wo[2]);
    return { wi, f: scale3(Rv, k), pdf, flags: PT_FLAG_REFLECTION, etap: 1 };
  }
  const r = ptRefract(wo, wm, eta);
  if (!r) return FAIL;
  const wi = r.wt;
  if (wi[2] * wo[2] > 0 || wi[2] === 0) return FAIL;
  const dwi = wi[0] * wm[0] + wi[1] * wm[1] + wi[2] * wm[2];
  const denomBase = dwi + dwo / r.etap;
  const denom = denomBase * denomBase;
  const pdf = vis * (Math.abs(dwi) / denom) * (pt / (pr + pt));
  let k = ptGgxD(a, wm) * ptGgxG(a, wo, wi) * Math.abs((dwi * dwo) / (wi[2] * wo[2] * denom));
  if (radiance) k /= r.etap * r.etap;
  return { wi, f: scale3(Tv, k), pdf, flags: PT_FLAG_TRANSMISSION, etap: r.etap };
}

/** f over both lobes, pdf under the `flags` restriction (pbrt `DielectricBxDF::f` / `PDF`). Delta interface: nothing. */
function topEval(top: Top, wo: PtVec3, wi: PtVec3, radiance: boolean, flags: number): PtBsdfEval {
  const none: PtBsdfEval = { f: ZERO3, pdf: 0 };
  if (topIsSpecular(top)) return none;
  const a = roughA(top.alpha);
  const h = dielectricHalf(wo, wi, top.ior);
  if (!h) return none;
  const { wm, etap } = h;
  const cosO = wo[2];
  const cosI = wi[2];
  const dwo = wo[0] * wm[0] + wo[1] * wm[1] + wo[2] * wm[2];
  const dwi = wi[0] * wm[0] + wi[1] * wm[1] + wi[2] * wm[2];
  const F = ptFrDielectric(dwo, top.ior);
  const Rv = reflectance(top, F, dwo);
  const Tv: PtVec3 = [1 - Rv[0], 1 - Rv[1], 1 - Rv[2]];
  const pr = flags & PT_LAYER_REFLECTION ? avg(Rv) : 0;
  const pt = flags & PT_LAYER_TRANSMISSION ? avg(Tv) : 0;
  const D = ptGgxD(a, wm);
  const G = ptGgxG(a, wo, wi);
  const vis = ptGgxVisiblePdf(a, wo, wm);
  if (h.reflect) {
    const k = (D * G) / Math.abs(4 * cosI * cosO);
    return { f: scale3(Rv, k), pdf: pr + pt === 0 ? 0 : (vis / (4 * Math.abs(dwo))) * (pr / (pr + pt)) };
  }
  const denomBase = dwi + dwo / etap;
  const denom = denomBase * denomBase * cosI * cosO;
  let k = D * G * Math.abs((dwi * dwo) / denom);
  if (radiance) k /= etap * etap;
  return { f: scale3(Tv, k), pdf: pr + pt === 0 ? 0 : vis * (Math.abs(dwi) / (denomBase * denomBase)) * (pt / (pr + pt)) };
}

const lerp = (t: number, a: number, b: number): number => (1 - t) * a + t * b;
const ok = (s: PtBsdfSample): boolean => s.flags !== 0 && !isBlack(s.f) && s.pdf > 0 && s.wi[2] !== 0;

function tr(dz: number, w: PtVec3): number {
  if (Math.abs(dz) <= PT_LAYER_MIN_THICKNESS) return 1;
  return Math.exp(-Math.abs(dz / w[2]));
}

// ---------------------------------------------------------------------------------------------------------------
// layering: 'stochastic' (pbrt LayeredBxDF specialised to CoatedDiffuse, albedo 0, twoSided)

function stochasticF(p: PtLayeredParams, wo0: PtVec3, wi0: PtVec3, o: PtLayeredOptions): PtVec3 {
  let wo = wo0;
  let wi = wi0;
  if (wo[2] < 0) {
    wo = neg(wo);
    wi = neg(wi);
  }
  // The opaque Lambert base transmits nothing: wo and wi must share the top hemisphere.
  if (wo[2] === 0 || wi[2] * wo[2] <= 0) return ZERO3;
  const top = topOf(p);
  const radiance = (o.transport ?? 'radiance') === 'radiance';
  const vndf = o.vndf ?? 'standard';
  const maxDepth = o.maxDepth ?? 10;
  const nSamples = o.nSamples ?? 1;
  const thickness = Math.max(p.thickness, PT_LAYER_MIN_THICKNESS);
  const base = baseParams(p);
  const topSpec = topIsSpecular(top);
  const baseSpec = ptLayeredSmoothBase(p);
  const pbrtMis = (o.exitMis ?? 'pbrt') === 'pbrt';
  const exitZ = thickness; // reflection: the walk leaves through the top interface
  let f: PtVec3 = scale3(topEval(top, wo, wi, radiance, PT_LAYER_ALL).f, nSamples);
  const rng = new Rng(hashSeed(o.seed ?? 0, [...wo, ...wi]));
  for (let s = 0; s < nSamples; s++) {
    const wos = topSample(top, wo, rng.next(), rng.next(), rng.next(), radiance, PT_LAYER_TRANSMISSION, vndf);
    if (!ok(wos)) continue;
    const wis = topSample(top, wi, rng.next(), rng.next(), rng.next(), !radiance, PT_LAYER_TRANSMISSION, vndf);
    if (!ok(wis)) continue;
    let beta = scale3(wos.f, Math.abs(wos.wi[2]) / wos.pdf);
    let z = thickness;
    let w = wos.wi;
    for (let depth = 0; depth < maxDepth; depth++) {
      if (depth > 3 && maxc(beta) < 0.25) {
        const q = Math.max(0, 1 - maxc(beta));
        if (rng.next() < q) break;
        beta = scale3(beta, 1 / (1 - q));
      }
      z = z === thickness ? 0 : thickness;
      beta = scale3(beta, tr(thickness, w));
      if (z === exitZ) {
        // reflect off the underside of the top interface
        const bs = topSample(top, neg(w), rng.next(), rng.next(), rng.next(), radiance, PT_LAYER_REFLECTION, vndf);
        if (!ok(bs)) break;
        beta = mul3(beta, scale3(bs.f, Math.abs(bs.wi[2]) / bs.pdf));
        w = bs.wi;
      } else {
        // at the base: next-event estimate through the presampled wis direction (a delta base has none: pbrt skips it)
        if (!baseSpec) {
          const nwi = neg(wis.wi);
          const fb = ptBsdfEval(base, neg(w), nwi);
          let wt = 1;
          if (!topSpec) wt = pbrtMis ? powerHeuristic(wis.pdf, fb.pdf) : 1;
          f = [0, 1, 2].map((c) => (f[c] as number) + (beta[c] as number) * (fb.f[c] as number) * Math.abs(wis.wi[2]) * wt * tr(thickness, wis.wi) * ((wis.f[c] as number) / wis.pdf)) as unknown as PtVec3;
        }
        const bs = ptBsdfSample(base, neg(w), [rng.next(), rng.next(), rng.next()]);
        if (bs.flags === 0 || isBlack(bs.f) || bs.pdf === 0 || bs.wi[2] === 0) break;
        beta = mul3(beta, scale3(bs.f, Math.abs(bs.wi[2]) / bs.pdf));
        w = bs.wi;
        if (!topSpec && (pbrtMis || baseSpec)) {
          // next-event estimate along the base's own sample, through the top interface (the only strategy for a delta base)
          const fe = topEval(top, neg(w), wi, radiance, PT_LAYER_ALL).f;
          if (!isBlack(fe)) {
            const exitPdf = topEval(top, neg(w), wi, radiance, PT_LAYER_TRANSMISSION).pdf;
            const wt2 = baseSpec ? 1 : powerHeuristic(bs.pdf, exitPdf);
            const k = tr(thickness, bs.wi) * wt2;
            f = [0, 1, 2].map((c) => (f[c] as number) + (beta[c] as number) * k * (fe[c] as number)) as unknown as PtVec3;
          }
        }
      }
    }
  }
  return scale3(f, 1 / nSamples);
}

function stochasticSample(p: PtLayeredParams, wo0: PtVec3, u: PtVec3, o: PtLayeredOptions): PtBsdfSample {
  let wo = wo0;
  let flipWi = false;
  if (wo[2] < 0) {
    wo = neg(wo);
    flipWi = true;
  }
  const top = topOf(p);
  const radiance = (o.transport ?? 'radiance') === 'radiance';
  const vndf = o.vndf ?? 'standard';
  const maxDepth = o.maxDepth ?? 10;
  const thickness = Math.max(p.thickness, PT_LAYER_MIN_THICKNESS);
  const base = baseParams(p);
  const bs0 = topSample(top, wo, u[0], u[1], u[2], radiance, PT_LAYER_ALL, vndf);
  if (!ok(bs0)) return FAIL;
  if (bs0.flags & PT_FLAG_REFLECTION) {
    return { ...bs0, wi: flipWi ? neg(bs0.wi) : bs0.wi, flags: bs0.flags | PT_FLAG_PROPORTIONAL };
  }
  let w = bs0.wi;
  let specularPath = (bs0.flags & PT_FLAG_DELTA) !== 0;
  const rng = new Rng(hashSeed(o.seed ?? 0, [...wo, ...u]));
  let f = scale3(bs0.f, Math.abs(bs0.wi[2]));
  let pdf = bs0.pdf;
  let z = thickness;
  for (let depth = 0; depth < maxDepth; depth++) {
    const rrBeta = maxc(f) / pdf;
    if (depth > 3 && rrBeta < 0.25) {
      const q = Math.max(0, 1 - rrBeta);
      if (rng.next() < q) return FAIL;
      pdf *= 1 - q;
    }
    if (w[2] === 0) return FAIL;
    z = z === thickness ? 0 : thickness;
    f = scale3(f, tr(thickness, w));
    const bs = z === 0 ? ptBsdfSample(base, neg(w), [rng.next(), rng.next(), rng.next()]) : topSample(top, neg(w), rng.next(), rng.next(), rng.next(), radiance, PT_LAYER_ALL, vndf);
    if (!ok(bs)) return FAIL;
    f = mul3(f, bs.f);
    pdf *= bs.pdf;
    // Only f / pdf is meaningful (PROPORTIONAL): keep both inside f32 range. A near-smooth rough interface multiplies each by ~1e4 per bounce (the GPU overflowed to inf / inf = NaN).
    if (pdf > 1e18 || pdf < 1e-18) {
      f = scale3(f, 1 / pdf);
      pdf = 1;
    }
    specularPath = specularPath && (bs.flags & PT_FLAG_DELTA) !== 0;
    w = bs.wi;
    if (bs.flags & PT_FLAG_TRANSMISSION) {
      const kind = w[2] * wo[2] > 0 ? PT_FLAG_REFLECTION : PT_FLAG_TRANSMISSION;
      return { wi: flipWi ? neg(w) : w, f, pdf, flags: kind | (specularPath ? PT_FLAG_DELTA : 0) | PT_FLAG_PROPORTIONAL, etap: 1 };
    }
    f = scale3(f, Math.abs(bs.wi[2]));
  }
  return FAIL;
}

function stochasticPdf(p: PtLayeredParams, wo0: PtVec3, wi0: PtVec3, o: PtLayeredOptions): number {
  let wo = wo0;
  let wi = wi0;
  if (wo[2] < 0) {
    wo = neg(wo);
    wi = neg(wi);
  }
  const top = topOf(p);
  const radiance = (o.transport ?? 'radiance') === 'radiance';
  const vndf = o.vndf ?? 'standard';
  const nSamples = o.nSamples ?? 1;
  const base = baseParams(p);
  const same = wi[2] * wo[2] > 0;
  const rng = new Rng(hashSeed(o.seed ?? 0, [...wi, ...wo]));
  let pdfSum = 0;
  if (same) pdfSum += nSamples * topEval(top, wo, wi, radiance, PT_LAYER_REFLECTION).pdf;
  // The opaque base transmits nothing, so the transmission (TT) term of pbrt's estimate is always 0: only TRT remains.
  if (same) {
    const topSpec = topIsSpecular(top);
    const baseSpec = ptLayeredSmoothBase(p);
    for (let s = 0; s < nSamples; s++) {
      const wos = topSample(top, wo, rng.next(), rng.next(), rng.next(), radiance, PT_LAYER_TRANSMISSION, vndf);
      const wis = topSample(top, wi, rng.next(), rng.next(), rng.next(), !radiance, PT_LAYER_TRANSMISSION, vndf);
      if (!(ok(wos) && ok(wis))) continue;
      if (topSpec) {
        pdfSum += ptBsdfEval(base, neg(wos.wi), neg(wis.wi)).pdf;
      } else {
        const rs = ptBsdfSample(base, neg(wos.wi), [rng.next(), rng.next(), rng.next()]);
        if (ok(rs)) {
          if (!baseSpec) {
            const rPdf = ptBsdfEval(base, neg(wos.wi), neg(wis.wi)).pdf;
            pdfSum += powerHeuristic(wis.pdf, rPdf) * rPdf;
          }
          const tPdf = topEval(top, neg(rs.wi), wi, radiance, PT_LAYER_ALL).pdf;
          pdfSum += (baseSpec ? 1 : powerHeuristic(rs.pdf, tPdf)) * tPdf;
        }
      }
    }
  }
  return lerp(0.9, 1 / (4 * PI), pdfSum / nSamples);
}

// ---------------------------------------------------------------------------------------------------------------
// layering: 'albedo' (OpenPBR)

/** E(mu): directional albedo of the dielectric reflection lobe (table; exact Fresnel when the interface is smooth). */
export function ptLayeredTopAlbedo(p: PtLayeredParams, mu: number): number {
  if (p.ior === 1) return 0;
  if (ptLayeredSmoothTop(p)) return ptFrDielectric(Math.max(mu, 1e-6), p.ior);
  const r = Math.sqrt(0.5 * (p.alpha[0] + p.alpha[1]));
  return ptLayeredAlbedo(ptDielectricF0(p.ior), mu, r);
}

interface Mix {
  /** base scale 1 - s E(mu), per channel */
  keep: PtVec3;
  /** probability of choosing the specular lobe */
  pS: number;
}

/** `ptLayeredTopAlbedo` per channel: with a thin film the plain albedo is scaled by the film's Fresnel over the plain one at this angle (exact for a smooth interface). */
function topAlbedo3(p: PtLayeredParams, mu: number): PtVec3 {
  const E = ptLayeredTopAlbedo(p, mu);
  if (!ptFilmActive(p.film) || p.ior === 1) return [E, E, E];
  const c = Math.max(mu, 1e-6);
  const F = ptFrDielectric(c, p.ior);
  const Fc = ptFilmFresnel(p.film, c, [F, F, F], dielectricF0(p.ior));
  const k = 1 / Math.max(F, 1e-6);
  return [Math.min(1, E * (Fc[0] as number) * k), Math.min(1, E * (Fc[1] as number) * k), Math.min(1, E * (Fc[2] as number) * k)];
}

function albedoKeep(p: PtLayeredParams, mu: number): PtVec3 {
  const s = specScale(p);
  const E = topAlbedo3(p, mu);
  return [1 - Math.min(1, s[0] * E[0]), 1 - Math.min(1, s[1] * E[1]), 1 - Math.min(1, s[2] * E[2])];
}

/** `'internal'` base: 1 / (eta^2 (1 - rho R_i)) per channel (the two keep factors are applied by the caller). */
function internalGain(p: PtLayeredParams): PtVec3 {
  const k = (c: number): number => 1 / (p.ior * p.ior * (1 - Math.min(0.9999, c * p.internalReflectance)));
  return [k(p.baseColor[0]), k(p.baseColor[1]), k(p.baseColor[2])];
}

function albedoMix(p: PtLayeredParams, mu: number, internal: boolean): Mix {
  const keep = albedoKeep(p, mu);
  const eS = avg([1 - keep[0], 1 - keep[1], 1 - keep[2]]);
  let eB = avg(mul3(p.baseColor, keep));
  // hemispherical albedo of the 'internal' base: rho keep(mu_o) (1 - R_i) / (1 - rho R_i)
  if (internal) eB = avg(mul3(mul3(p.baseColor, keep), internalGain(p))) * (1 - p.internalReflectance) * p.ior * p.ior;
  return { keep, pS: eS + eB > 0 ? eS / (eS + eB) : 0 };
}

/**
 * Coat over a conductor, albedo scaling (OpenPBR: `f = f_coat + (1 - E_coat(mu_o)) f_base`, base = the conductor with eta, k relative to the coat
 * medium). Directional albedo of the bottom lobe for the lobe choice: its Fresnel reflectance at mu (the GGX loss is ignored; it is a
 * probability, not a weight). `albedoBase: 'internal'` takes the spec's reciprocal alternative, `(1 - E(mu_o))(1 - E(mu_i))` (the exit transmission of the light leaving the metal). No refraction into the coat, no internal reflection, no roughening of the conductor by the coat (the spec's
 * r'_B is not applied; pbrt has none); DIVERGES V73.
 */
/** Cosine of the angle inside the coat for the outside cosine `d` (Snell; the spec's mu_c). */
function insideCos(ior: number, d: number): number {
  return Math.sqrt(Math.max(0, 1 - (1 - d * d) / (ior * ior)));
}

/** The conductor's Fresnel at the angle INSIDE the coat (its eta and k are relative to the coat, so the outside angle would be wrong), per channel. */
function condFresnel(p: PtLayeredParams, d: number): PtVec3 {
  const c = p.baseConductor as NonNullable<PtLayeredParams['baseConductor']>;
  const t = insideCos(p.ior, d);
  return [ptFrComplex(t, c.eta[0], c.k[0]), ptFrComplex(t, c.eta[1], c.k[1]), ptFrComplex(t, c.eta[2], c.k[2])];
}

/** `ptBsdfEval`'s conductor uses F(|wo.wm|) with the outside angle; the albedo layering swaps it for the inside one: ratio per channel. */
function condFresnelRatio(p: PtLayeredParams, d: number): PtVec3 {
  const c = p.baseConductor as NonNullable<PtLayeredParams['baseConductor']>;
  const now = condFresnel(p, d);
  const was: PtVec3 = [ptFrComplex(d, c.eta[0], c.k[0]), ptFrComplex(d, c.eta[1], c.k[1]), ptFrComplex(d, c.eta[2], c.k[2])];
  return [was[0] > 0 ? now[0] / was[0] : 1, was[1] > 0 ? now[1] / was[1] : 1, was[2] > 0 ? now[2] / was[2] : 1];
}

function condMix(p: PtLayeredParams, mu: number): Mix {
  const keep = albedoKeep(p, mu);
  const eS = avg([1 - keep[0], 1 - keep[1], 1 - keep[2]]);
  const F = condFresnel(p, mu);
  const eB = avg(mul3(F, keep));
  return { keep, pS: eS + eB > 0 ? eS / (eS + eB) : 0 };
}

function condEval(p: PtLayeredParams, wo: PtVec3, wi: PtVec3, o: PtLayeredOptions): PtBsdfEval {
  if (wo[2] === 0 || wi[2] <= 0) return { f: ZERO3, pdf: 0 };
  const mix = condMix(p, wo[2]);
  const m = ptBsdfEval(baseParams(p), wo, wi, o);
  const spec = topEval(topOf(p), wo, wi, (o.transport ?? 'radiance') === 'radiance', PT_LAYER_REFLECTION);
  let fm = mul3(m.f, mix.keep);
  if (m.pdf > 0) {
    const wx = wo[0] + wi[0];
    const wy = wo[1] + wi[1];
    const wz = wo[2] + wi[2];
    fm = mul3(fm, condFresnelRatio(p, Math.abs((wo[0] * wx + wo[1] * wy + wo[2] * wz) / Math.hypot(wx, wy, wz))));
    if (ptAlbedoBaseInternal(o)) fm = mul3(fm, albedoKeep(p, wi[2]));
  }
  return { f: [fm[0] + spec.f[0], fm[1] + spec.f[1], fm[2] + spec.f[2]], pdf: mix.pS * spec.pdf + (1 - mix.pS) * m.pdf };
}

function condSample(p: PtLayeredParams, wo: PtVec3, u: PtVec3, o: PtLayeredOptions): PtBsdfSample {
  const mix = condMix(p, wo[2]);
  const top = topOf(p);
  if (u[0] < mix.pS) {
    if (topIsSpecular(top)) {
      const F = ptFrDielectric(wo[2], p.ior);
      return { wi: [-wo[0], -wo[1], wo[2]], f: [F / wo[2], F / wo[2], F / wo[2]], pdf: mix.pS, flags: PT_FLAG_REFLECTION | PT_FLAG_DELTA, etap: 1 };
    }
    const s = topSample(top, wo, 0, u[1], u[2], (o.transport ?? 'radiance') === 'radiance', PT_LAYER_REFLECTION, o.vndf ?? 'standard');
    if (s.flags === 0) return FAIL;
    const e = condEval(p, wo, s.wi, o);
    return e.pdf === 0 ? FAIL : { wi: s.wi, f: e.f, pdf: e.pdf, flags: PT_FLAG_REFLECTION, etap: 1 };
  }
  const bp = baseParams(p);
  const d = ptBsdfSample(bp, wo, [0, u[1], u[2]], o);
  if (d.flags === 0) return FAIL;
  if (d.flags & PT_FLAG_DELTA) {
    const f = mul3(mul3(d.f, mix.keep), condFresnelRatio(p, wo[2]));
    return { wi: d.wi, f: ptAlbedoBaseInternal(o) ? mul3(f, albedoKeep(p, d.wi[2])) : f, pdf: 1 - mix.pS, flags: d.flags, etap: 1 };
  }
  const e = condEval(p, wo, d.wi, o);
  return e.pdf === 0 || isBlack(e.f) ? FAIL : { wi: d.wi, f: e.f, pdf: e.pdf, flags: PT_FLAG_REFLECTION, etap: 1 };
}

function albedoEval(p: PtLayeredParams, wo0: PtVec3, wi0: PtVec3, o: PtLayeredOptions): PtBsdfEval {
  if (p.baseConductor) {
    return wo0[2] < 0 ? condEval(p, neg(wo0), neg(wi0), o) : condEval(p, wo0, wi0, o);
  }
  let wo = wo0;
  let wi = wi0;
  if (wo[2] < 0) {
    wo = neg(wo);
    wi = neg(wi);
  }
  if (wo[2] === 0 || wi[2] <= 0) return { f: ZERO3, pdf: 0 };
  const internal = ptAlbedoBaseInternal(o);
  const mix = albedoMix(p, wo[2], internal);
  let fd = mul3(scale3(p.baseColor, 1 / PI), mix.keep);
  if (internal) fd = mul3(mul3(fd, albedoKeep(p, wi[2])), internalGain(p));
  const spec = topEval(topOf(p), wo, wi, (o.transport ?? 'radiance') === 'radiance', PT_LAYER_REFLECTION);
  // topEval's pdf is already restricted to reflection (pr/(pr+pt) = 1)
  return { f: [fd[0] + spec.f[0], fd[1] + spec.f[1], fd[2] + spec.f[2]], pdf: mix.pS * spec.pdf + (1 - mix.pS) * (wi[2] / PI) };
}

function albedoSample(p: PtLayeredParams, wo0: PtVec3, u: PtVec3, o: PtLayeredOptions): PtBsdfSample {
  let wo = wo0;
  let flip = false;
  if (wo[2] < 0) {
    wo = neg(wo);
    flip = true;
  }
  if (wo[2] === 0) return FAIL;
  if (p.baseConductor) {
    const s = condSample(p, wo, u, o);
    return flip && s.flags !== 0 ? { ...s, wi: neg(s.wi) } : s;
  }
  const mix = albedoMix(p, wo[2], ptAlbedoBaseInternal(o));
  const top = topOf(p);
  const fix = (s: PtBsdfSample): PtBsdfSample => (flip && s.flags !== 0 ? { ...s, wi: neg(s.wi) } : s);
  // One uniform picks the lobe, then is rescaled to a fresh uniform for the direction's first coordinate.
  if (u[0] < mix.pS) {
    if (topIsSpecular(top)) {
      const Fv = topFresnel(top, wo[2]);
      const sc = specScale(p);
      const c = wo[2];
      return fix({ wi: [-wo[0], -wo[1], wo[2]], f: scale3([sc[0] * Fv[0], sc[1] * Fv[1], sc[2] * Fv[2]], 1 / c), pdf: mix.pS, flags: PT_FLAG_REFLECTION | PT_FLAG_DELTA, etap: 1 });
    }
    const s = topSample(top, wo, 0, u[1], u[2], (o.transport ?? 'radiance') === 'radiance', PT_LAYER_REFLECTION, o.vndf ?? 'standard');
    if (s.flags === 0) return FAIL;
    const e = albedoEval(p, wo, s.wi, o);
    if (e.pdf === 0) return FAIL;
    return fix({ wi: s.wi, f: e.f, pdf: e.pdf, flags: PT_FLAG_REFLECTION, etap: 1 });
  }
  const d = ptBsdfSample(baseParams(p), wo, [0, u[1], u[2]]);
  if (d.flags === 0) return FAIL;
  const e = albedoEval(p, wo, d.wi, o);
  if (e.pdf === 0 || isBlack(e.f)) return FAIL;
  return fix({ wi: d.wi, f: e.f, pdf: e.pdf, flags: PT_FLAG_REFLECTION, etap: 1 });
}

// ---------------------------------------------------------------------------------------------------------------
// Public entry points (same shapes as `ptBsdfEval` / `ptBsdfSample` / `ptBsdfPdf`).

/**
 * f and pdf. Albedo: deterministic. Stochastic: `f` is one random-walk estimate (`nSamples` walks, driven by
 * `options.seed` and the directions), `pdf` pbrt's blended estimate; a delta top interface contributes only through sampling.
 */
export function ptLayeredEval(p: PtLayeredParams, wo: PtVec3, wi: PtVec3, options: PtLayeredOptions = {}): PtBsdfEval {
  if ((options.layering ?? 'albedo') === 'albedo') return albedoEval(p, wo, wi, options);
  return { f: stochasticF(p, wo, wi, options), pdf: stochasticPdf(p, wo, wi, options) };
}

export function ptLayeredPdf(p: PtLayeredParams, wo: PtVec3, wi: PtVec3, options: PtLayeredOptions = {}): number {
  if ((options.layering ?? 'albedo') === 'albedo') return albedoEval(p, wo, wi, options).pdf;
  return stochasticPdf(p, wo, wi, options);
}

/** `u = [uc, u0, u1]`. A stochastic sample has `PT_FLAG_PROPORTIONAL`: weight `f |wi.z| / pdf`, MIS with `ptLayeredPdf`. */
export function ptLayeredSample(p: PtLayeredParams, wo: PtVec3, u: PtVec3, options: PtLayeredOptions = {}): PtBsdfSample {
  const s = (options.layering ?? 'albedo') === 'albedo' ? albedoSample(p, wo, u, options) : stochasticSample(p, wo, u, options);
  if (s.flags === 0 || s.pdf === 0 || s.wi[2] === 0 || isBlack(s.f)) return FAIL;
  return s;
}
