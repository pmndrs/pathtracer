/**
 * Bench-material-subset BSDFs, TS twin of `bsdf-wgsl.ts` (prefix `pt`).
 *
 * Lobes: Lambert diffuse, GGX conductor (complex Fresnel from RGB eta/k), GGX dielectric (smooth and rough, no
 * absorption), black. Everything is in a LOCAL shading frame: +Z is the shading normal, wo and wi are unit vectors
 * pointing away from the surface, and `wo.z < 0` means the ray arrives from the inside of a dielectric.
 *
 * Ported from pbrt-v4 (Apache-2.0, Copyright 1998-2020 Matt Pharr, Wenzel Jakob, Greg Humphreys): `DiffuseBxDF`,
 * `ConductorBxDF`, `DielectricBxDF` (bxdfs.h / bxdfs.cpp), `TrowbridgeReitzDistribution`, `FrDielectric`,
 * `FrComplex`, `Refract` (util/scattering.h). Same conventions: a smooth lobe is a delta whose `f` already contains
 * the 1/|cos wi| (throughput update is always `f * |cos wi| / pdf`) and whose `pdf` is the discrete probability of
 * choosing it; transmission carries the 1/etap^2 radiance scale (`transport: 'radiance'`). Departures are listed in
 * DIVERGES.md (V28). The spherical-cap visible-normal sampler is Dupuy and Benyoub, "Sampling Visible GGX Normals
 * with Spherical Caps", HPG 2023 (formulas from the paper, original code).
 *
 * Not here on purpose: multiple-scattering compensation (the references do not have it; a Turquin LUT is a later
 * option), anisotropic rotation (alphaX/alphaY are kept in the struct, the frame has no tangent rotation yet),
 * absorption (the interior medium is path state, P2.22). The thin-walled dielectric (`PT_BSDF_THIN`) is pbrt's
 * `ThinDielectricBxDF`.
 */

export type PtVec3 = readonly [number, number, number];

/** `PtBsdfParams.kind`. */
export const PT_BSDF_DIFFUSE = 0;
export const PT_BSDF_CONDUCTOR = 1;
export const PT_BSDF_DIELECTRIC = 2;
export const PT_BSDF_BLACK = 3;
/**
 * Thin-walled smooth dielectric (P2.22; pbrt `ThinDielectricBxDF`, Mitsuba `thindielectric`): a sheet with two parallel interfaces and
 * the inner reflections summed, so R' = R + T^2 R / (1 - R^2), T' = 1 - R'. Always a delta lobe; transmission goes straight through
 * (`wi = -wo`, etap 1). `ior` is the relative IOR, `reflectance` the tint of the transmitted light (white = none; non-physical, as
 * glTF's baseColor on transmission). Not in the default kernel: `PtBsdfOptions.thin`.
 */
export const PT_BSDF_THIN = 4;
/**
 * Dielectric interface over a Lambertian base (P2.21), evaluated by `ptLayered*` (layered.ts, not by `ptBsdfEval`). Packed into
 * `PtBsdfParams` without growing it: `reflectance` = base colour, `ior`, `alpha` = interface GGX alpha, `eta` = interface Fresnel scale
 * (`specular_weight * specular_color`), `k` = (layer thickness, internal reflectance of the interface, unused).
 */
export const PT_BSDF_LAYERED = 5;

/** `PtBsdfSample.flags` bits. 0 means the sample failed (no direction, treat as absorbed). */
export const PT_FLAG_DELTA = 1;
export const PT_FLAG_REFLECTION = 2;
export const PT_FLAG_TRANSMISSION = 4;

/**
 * Plain unpacked parameters. Only the fields of `kind` are read:
 * diffuse `reflectance`; conductor `eta`, `k`, `alpha`; dielectric `ior` (inside over outside), `alpha`.
 * `alpha` is the GGX alpha (= perceptual roughness squared, `roughnessToAlpha`), x and y.
 */
export interface PtBsdfParams {
  kind: number;
  reflectance: PtVec3;
  eta: PtVec3;
  k: PtVec3;
  alpha: readonly [number, number];
  ior: number;
}

export interface PtBsdfOptions {
  /** Visible-normal sampler: Heitz 2018 / pbrt-v4 (default) or Dupuy-Benyoub spherical caps. Same pdf, different warp. */
  vndf?: 'standard' | 'cap';
  /** Radiance transport divides transmission by etap^2 (pbrt `TransportMode::Radiance`, default); importance does not. */
  transport?: 'radiance' | 'importance';
  /**
   * WGSL only (the CPU twin always has it): compile the thin-walled dielectric (`PT_BSDF_THIN`) into the kernel. Default false: a
   * scene without thin-walled materials keeps the kernel text and size.
   */
  thin?: boolean;
}

export interface PtBsdfEval {
  f: PtVec3;
  pdf: number;
}

export interface PtBsdfSample {
  wi: PtVec3;
  f: PtVec3;
  pdf: number;
  /** PT_FLAG_* bits; 0 = failed sample. */
  flags: number;
  /** Relative IOR along the path for transmission (pbrt `etap`), 1 otherwise. */
  etap: number;
}

const PI = Math.PI;
/** pbrt `EffectivelySmooth`: max(alphaX, alphaY) below this is a delta lobe. */
export const PT_SMOOTH_ALPHA = 1e-3;
/** pbrt clamps a rough lobe's alphas to at least this. */
export const PT_MIN_ALPHA = 1e-4;

// ---------------------------------------------------------------------------------------------------------------
// Small vector helpers (tuples; the twin favours clarity over speed).

function dot(a: PtVec3, b: PtVec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function neg(a: PtVec3): PtVec3 {
  return [-a[0], -a[1], -a[2]];
}
function normalize(a: PtVec3): PtVec3 {
  const l = Math.sqrt(dot(a, a));
  return [a[0] / l, a[1] / l, a[2] / l];
}
function reflect(wo: PtVec3, n: PtVec3): PtVec3 {
  const d = 2 * dot(wo, n);
  return [-wo[0] + d * n[0], -wo[1] + d * n[1], -wo[2] + d * n[2]];
}
const ZERO3: PtVec3 = [0, 0, 0];

/** pbrt `Refract`. Returns null on total internal reflection. `etap` is the relative IOR along the ray. */
export function ptRefract(wi: PtVec3, n0: PtVec3, eta0: number): { wt: PtVec3; etap: number } | null {
  let n = n0;
  let eta = eta0;
  let cosI = dot(n, wi);
  if (cosI < 0) {
    eta = 1 / eta;
    cosI = -cosI;
    n = neg(n);
  }
  const sin2I = Math.max(0, 1 - cosI * cosI);
  const sin2T = sin2I / (eta * eta);
  if (sin2T >= 1) return null;
  const cosT = Math.sqrt(1 - sin2T);
  const s = cosI / eta - cosT;
  return { wt: [-wi[0] / eta + s * n[0], -wi[1] / eta + s * n[1], -wi[2] / eta + s * n[2]], etap: eta };
}

/** pbrt `FrDielectric`: unpolarised Fresnel reflectance; negative cos means the ray is inside (eta inverted). */
export function ptFrDielectric(cosI0: number, eta0: number): number {
  let cosI = Math.min(1, Math.max(-1, cosI0));
  let eta = eta0;
  if (cosI < 0) {
    eta = 1 / eta;
    cosI = -cosI;
  }
  const sin2T = (1 - cosI * cosI) / (eta * eta);
  if (sin2T >= 1) return 1;
  const cosT = Math.sqrt(Math.max(0, 1 - sin2T));
  const rParl = (eta * cosI - cosT) / (eta * cosI + cosT);
  const rPerp = (cosI - eta * cosT) / (cosI + eta * cosT);
  return (rParl * rParl + rPerp * rPerp) / 2;
}

type Complex = readonly [number, number];
function cmul(a: Complex, b: Complex): Complex {
  return [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
}
function cdiv(a: Complex, b: Complex): Complex {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
}
function csqrt(z: Complex): Complex {
  const n = Math.hypot(z[0], z[1]);
  if (n === 0) return [0, 0];
  const t1 = Math.sqrt(0.5 * (n + Math.abs(z[0])));
  const t2 = (0.5 * z[1]) / t1;
  if (z[0] >= 0) return [t1, t2];
  return [Math.abs(t2), z[1] >= 0 ? t1 : -t1];
}

/** pbrt `FrComplex` for one channel: unpolarised Fresnel reflectance of a conductor with complex IOR eta + i k. */
export function ptFrComplex(cosI0: number, eta: number, k: number): number {
  const c = Math.min(1, Math.max(0, cosI0));
  const e: Complex = [eta, k];
  const sin2I = 1 - c * c;
  const sin2T = cdiv([sin2I, 0], cmul(e, e));
  const cosT = csqrt([1 - sin2T[0], -sin2T[1]]);
  const ec: Complex = [eta * c, k * c];
  const rParl = cdiv([ec[0] - cosT[0], ec[1] - cosT[1]], [ec[0] + cosT[0], ec[1] + cosT[1]]);
  const ecT = cmul(e, cosT);
  const rPerp = cdiv([c - ecT[0], -ecT[1]], [c + ecT[0], ecT[1]]);
  return (rParl[0] * rParl[0] + rParl[1] * rParl[1] + rPerp[0] * rPerp[0] + rPerp[1] * rPerp[1]) / 2;
}

// ---------------------------------------------------------------------------------------------------------------
// Trowbridge-Reitz (GGX). With unit w: tan^2 * cos^2(phi) = x^2 / z^2, so the pbrt formulas reduce to the forms below
// (no atan, no inf).

export function ptIsSmooth(alpha: readonly [number, number]): boolean {
  return Math.max(alpha[0], alpha[1]) < PT_SMOOTH_ALPHA;
}

/** pbrt `D(wm)`. */
export function ptGgxD(alpha: readonly [number, number], wm: PtVec3): number {
  const z2 = wm[2] * wm[2];
  const cos4 = z2 * z2;
  if (cos4 < 1e-16) return 0;
  const e = ((wm[0] * wm[0]) / (alpha[0] * alpha[0]) + (wm[1] * wm[1]) / (alpha[1] * alpha[1])) / z2;
  return 1 / (PI * alpha[0] * alpha[1] * cos4 * (1 + e) * (1 + e));
}

/** pbrt `Lambda(w)`. */
export function ptGgxLambda(alpha: readonly [number, number], w: PtVec3): number {
  const z2 = w[2] * w[2];
  if (z2 === 0) return 0;
  const a2t2 = (w[0] * w[0] * alpha[0] * alpha[0] + w[1] * w[1] * alpha[1] * alpha[1]) / z2;
  return (Math.sqrt(1 + a2t2) - 1) / 2;
}

export function ptGgxG1(alpha: readonly [number, number], w: PtVec3): number {
  return 1 / (1 + ptGgxLambda(alpha, w));
}

export function ptGgxG(alpha: readonly [number, number], wo: PtVec3, wi: PtVec3): number {
  return 1 / (1 + ptGgxLambda(alpha, wo) + ptGgxLambda(alpha, wi));
}

/** Visible-normal density D_w(wm) = G1(w) / |cos w| * D(wm) * |w . wm| (pbrt `PDF(w, wm)`). */
export function ptGgxVisiblePdf(alpha: readonly [number, number], w: PtVec3, wm: PtVec3): number {
  return ((ptGgxG1(alpha, w) / Math.abs(w[2])) * ptGgxD(alpha, wm) * Math.abs(dot(w, wm)));
}

/** Samples a visible normal (upper hemisphere) for direction `w` (either hemisphere; mirrored if below). */
export function ptGgxSampleWm(alpha: readonly [number, number], w: PtVec3, u0: number, u1: number, vndf: 'standard' | 'cap'): PtVec3 {
  let wh = normalize([alpha[0] * w[0], alpha[1] * w[1], w[2]]);
  if (wh[2] < 0) wh = neg(wh);
  if (vndf === 'cap') {
    // Dupuy and Benyoub 2023: uniform point on the spherical cap of directions visible from wh, then reproject.
    const phi = 2 * PI * u0;
    const z = (1 - u1) * (1 + wh[2]) - wh[2];
    const sinT = Math.sqrt(Math.min(1, Math.max(0, 1 - z * z)));
    const nh: PtVec3 = [sinT * Math.cos(phi) + wh[0], sinT * Math.sin(phi) + wh[1], z + wh[2]];
    return normalize([alpha[0] * nh[0], alpha[1] * nh[1], Math.max(1e-6, nh[2])]);
  }
  // Heitz 2018 as in pbrt-v4: projected disk warped to the visible half.
  const t1: PtVec3 = wh[2] < 0.99999 ? normalize([-wh[1], wh[0], 0]) : [1, 0, 0];
  const t2: PtVec3 = [wh[1] * t1[2] - wh[2] * t1[1], wh[2] * t1[0] - wh[0] * t1[2], wh[0] * t1[1] - wh[1] * t1[0]];
  const r = Math.sqrt(u0);
  const theta = 2 * PI * u1;
  const px = r * Math.cos(theta);
  let py = r * Math.sin(theta);
  const h = Math.sqrt(1 - px * px);
  const s = (1 + wh[2]) / 2;
  py = (1 - s) * h + s * py;
  const pz = Math.sqrt(Math.max(0, 1 - px * px - py * py));
  const nh: PtVec3 = [px * t1[0] + py * t2[0] + pz * wh[0], px * t1[1] + py * t2[1] + pz * wh[1], px * t1[2] + py * t2[2] + pz * wh[2]];
  return normalize([alpha[0] * nh[0], alpha[1] * nh[1], Math.max(1e-6, nh[2])]);
}

function roughAlpha(alpha: readonly [number, number]): readonly [number, number] {
  return [Math.max(alpha[0], PT_MIN_ALPHA), Math.max(alpha[1], PT_MIN_ALPHA)];
}

function sameHemisphere(a: PtVec3, b: PtVec3): boolean {
  return a[2] * b[2] > 0;
}

function frConductor(p: PtBsdfParams, cos: number): PtVec3 {
  return [ptFrComplex(cos, p.eta[0], p.k[0]), ptFrComplex(cos, p.eta[1], p.k[1]), ptFrComplex(cos, p.eta[2], p.k[2])];
}

function scale3(a: PtVec3, s: number): PtVec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

/** True when the BSDF is a delta lobe (conductor/dielectric below the smooth threshold, or eta = 1): eval and pdf are 0. */
export function ptBsdfIsDelta(p: PtBsdfParams): boolean {
  if (p.kind === PT_BSDF_CONDUCTOR) return ptIsSmooth(p.alpha);
  if (p.kind === PT_BSDF_DIELECTRIC) return p.ior === 1 || ptIsSmooth(p.alpha);
  if (p.kind === PT_BSDF_THIN) return true;
  return false;
}

/** pbrt `ThinDielectricBxDF`: total reflectance R' of a sheet at |cos| = c (two interfaces, inner reflections summed); T' = 1 - R'. */
export function ptThinReflectance(cos: number, eta: number): number {
  let R = ptFrDielectric(cos, eta);
  const T = 1 - R;
  if (R < 1) R += (T * T * R) / (1 - R * R);
  return R;
}

/** Orthonormal frame from a unit normal (Duff et al. 2017); local +Z = n. */
export function ptFrameFromNormal(n: PtVec3): { t: PtVec3; b: PtVec3; n: PtVec3 } {
  const s = n[2] >= 0 ? 1 : -1;
  const a = -1 / (s + n[2]);
  const b = n[0] * n[1] * a;
  return { t: [1 + s * n[0] * n[0] * a, s * b, -s * n[0]], b: [b, s + n[1] * n[1] * a, -n[1]], n };
}

// ---------------------------------------------------------------------------------------------------------------
// Eval and pdf. Generalised half vector for dielectrics as in pbrt.

export interface HalfVector {
  wm: PtVec3;
  reflect: boolean;
  etap: number;
}

export function dielectricHalf(wo: PtVec3, wi: PtVec3, eta: number): HalfVector | null {
  const cosO = wo[2];
  const cosI = wi[2];
  const refl = cosI * cosO > 0;
  let etap = 1;
  if (!refl) etap = cosO > 0 ? eta : 1 / eta;
  let wm: PtVec3 = [wi[0] * etap + wo[0], wi[1] * etap + wo[1], wi[2] * etap + wo[2]];
  if (cosI === 0 || cosO === 0 || dot(wm, wm) === 0) return null;
  wm = normalize(wm);
  if (wm[2] < 0) wm = neg(wm);
  // Discard back-facing microfacets.
  if (dot(wm, wi) * cosI < 0 || dot(wm, wo) * cosO < 0) return null;
  return { wm, reflect: refl, etap };
}

export function ptBsdfEval(p: PtBsdfParams, wo: PtVec3, wi: PtVec3, options: PtBsdfOptions = {}): PtBsdfEval {
  const none: PtBsdfEval = { f: ZERO3, pdf: 0 };
  switch (p.kind) {
    case PT_BSDF_DIFFUSE: {
      if (!sameHemisphere(wo, wi)) return none;
      return { f: scale3(p.reflectance, 1 / PI), pdf: Math.abs(wi[2]) / PI };
    }
    case PT_BSDF_CONDUCTOR: {
      if (!sameHemisphere(wo, wi) || ptIsSmooth(p.alpha)) return none;
      const a = roughAlpha(p.alpha);
      const cosO = Math.abs(wo[2]);
      const cosI = Math.abs(wi[2]);
      let wm: PtVec3 = [wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]];
      if (dot(wm, wm) === 0) return none;
      wm = normalize(wm);
      if (wm[2] < 0) wm = neg(wm);
      const dwo = Math.abs(dot(wo, wm));
      const F = frConductor(p, dwo);
      const s = (ptGgxD(a, wm) * ptGgxG(a, wo, wi)) / (4 * cosI * cosO);
      return { f: scale3(F, s), pdf: ptGgxVisiblePdf(a, wo, wm) / (4 * dwo) };
    }
    case PT_BSDF_DIELECTRIC: {
      if (p.ior === 1 || ptIsSmooth(p.alpha)) return none;
      const a = roughAlpha(p.alpha);
      const h = dielectricHalf(wo, wi, p.ior);
      if (!h) return none;
      const { wm, etap } = h;
      const cosO = wo[2];
      const cosI = wi[2];
      const dwo = dot(wo, wm);
      const dwi = dot(wi, wm);
      const F = ptFrDielectric(dwo, p.ior);
      const R = F;
      const T = 1 - F;
      const D = ptGgxD(a, wm);
      const G = ptGgxG(a, wo, wi);
      if (h.reflect) {
        const fr = (D * G * F) / Math.abs(4 * cosI * cosO);
        return { f: [fr, fr, fr], pdf: (ptGgxVisiblePdf(a, wo, wm) / (4 * Math.abs(dwo))) * (R / (R + T)) };
      }
      const denomBase = dwi + dwo / etap;
      const denom = denomBase * denomBase * cosI * cosO;
      let ft = (D * (1 - F) * G * Math.abs((dwi * dwo) / denom));
      if ((options.transport ?? 'radiance') === 'radiance') ft /= etap * etap;
      const dwmDwi = Math.abs(dwi) / (denomBase * denomBase);
      return { f: [ft, ft, ft], pdf: ptGgxVisiblePdf(a, wo, wm) * dwmDwi * (T / (R + T)) };
    }
    default:
      return none;
  }
}

/** Sampling density of `ptBsdfSample` for the direction wi (0 for delta lobes and black). Same as `ptBsdfEval(...).pdf`. */
export function ptBsdfPdf(p: PtBsdfParams, wo: PtVec3, wi: PtVec3): number {
  return ptBsdfEval(p, wo, wi).pdf;
}

/**
 * Samples a direction. `u = [uc, u0, u1]`: `uc` picks reflection or transmission, `u0, u1` shape the direction.
 * `flags === 0` means failure. Throughput update: `beta *= f * |wi.z| / pdf` (delta lobes included).
 */
export function ptBsdfSample(p: PtBsdfParams, wo: PtVec3, u: PtVec3, options: PtBsdfOptions = {}): PtBsdfSample {
  const fail: PtBsdfSample = { wi: ZERO3, f: ZERO3, pdf: 0, flags: 0, etap: 1 };
  const vndf = options.vndf ?? 'standard';
  const radiance = (options.transport ?? 'radiance') === 'radiance';
  let s: PtBsdfSample = fail;
  switch (p.kind) {
    case PT_BSDF_DIFFUSE: {
      // Concentric disk mapping (pbrt SampleUniformDiskConcentric), cosine-weighted hemisphere.
      const ox = 2 * u[1] - 1;
      const oy = 2 * u[2] - 1;
      let dx = 0;
      let dy = 0;
      if (ox !== 0 || oy !== 0) {
        let r: number;
        let th: number;
        if (Math.abs(ox) > Math.abs(oy)) {
          r = ox;
          th = (PI / 4) * (oy / ox);
        } else {
          r = oy;
          th = PI / 2 - (PI / 4) * (ox / oy);
        }
        dx = r * Math.cos(th);
        dy = r * Math.sin(th);
      }
      let z = Math.sqrt(Math.max(0, 1 - dx * dx - dy * dy));
      if (wo[2] < 0) z = -z;
      s = { wi: [dx, dy, z], f: scale3(p.reflectance, 1 / PI), pdf: Math.abs(z) / PI, flags: PT_FLAG_REFLECTION, etap: 1 };
      break;
    }
    case PT_BSDF_CONDUCTOR: {
      if (ptIsSmooth(p.alpha)) {
        const wi: PtVec3 = [-wo[0], -wo[1], wo[2]];
        const c = Math.abs(wi[2]);
        s = { wi, f: scale3(frConductor(p, c), 1 / c), pdf: 1, flags: PT_FLAG_REFLECTION | PT_FLAG_DELTA, etap: 1 };
        break;
      }
      if (wo[2] === 0) return fail;
      const a = roughAlpha(p.alpha);
      const wm = ptGgxSampleWm(a, wo, u[1], u[2], vndf);
      const wi = reflect(wo, wm);
      if (!sameHemisphere(wo, wi)) return fail;
      const dwo = Math.abs(dot(wo, wm));
      const pdf = ptGgxVisiblePdf(a, wo, wm) / (4 * dwo);
      const cosO = Math.abs(wo[2]);
      const cosI = Math.abs(wi[2]);
      if (cosI === 0 || cosO === 0) return fail;
      const f = scale3(frConductor(p, dwo), (ptGgxD(a, wm) * ptGgxG(a, wo, wi)) / (4 * cosI * cosO));
      s = { wi, f, pdf, flags: PT_FLAG_REFLECTION, etap: 1 };
      break;
    }
    case PT_BSDF_DIELECTRIC: {
      const eta = p.ior;
      if (eta === 1 || ptIsSmooth(p.alpha)) {
        const R = ptFrDielectric(wo[2], eta);
        const T = 1 - R;
        if (u[0] < R) {
          const wi: PtVec3 = [-wo[0], -wo[1], wo[2]];
          const c = Math.abs(wi[2]);
          s = { wi, f: [R / c, R / c, R / c], pdf: R, flags: PT_FLAG_REFLECTION | PT_FLAG_DELTA, etap: 1 };
        } else {
          const r = ptRefract(wo, [0, 0, 1], eta);
          if (!r) return fail;
          let ft = T / Math.abs(r.wt[2]);
          if (radiance) ft /= r.etap * r.etap;
          s = { wi: r.wt, f: [ft, ft, ft], pdf: T, flags: PT_FLAG_TRANSMISSION | PT_FLAG_DELTA, etap: r.etap };
        }
        break;
      }
      const a = roughAlpha(p.alpha);
      const wm = ptGgxSampleWm(a, wo, u[1], u[2], vndf);
      const dwo = dot(wo, wm);
      const R = ptFrDielectric(dwo, eta);
      const T = 1 - R;
      if (u[0] < R) {
        const wi = reflect(wo, wm);
        if (!sameHemisphere(wo, wi)) return fail;
        const pdf = (ptGgxVisiblePdf(a, wo, wm) / (4 * Math.abs(dwo))) * R;
        const fv = (ptGgxD(a, wm) * ptGgxG(a, wo, wi) * R) / (4 * wi[2] * wo[2]);
        s = { wi, f: [fv, fv, fv], pdf, flags: PT_FLAG_REFLECTION, etap: 1 };
      } else {
        const r = ptRefract(wo, wm, eta);
        if (!r) return fail;
        const wi = r.wt;
        if (sameHemisphere(wo, wi) || wi[2] === 0) return fail;
        const dwi = dot(wi, wm);
        const denomBase = dwi + dwo / r.etap;
        const denom = denomBase * denomBase;
        const dwmDwi = Math.abs(dwi) / denom;
        const pdf = ptGgxVisiblePdf(a, wo, wm) * dwmDwi * T;
        let ft = T * ptGgxD(a, wm) * ptGgxG(a, wo, wi) * Math.abs((dwi * dwo) / (wi[2] * wo[2] * denom));
        if (radiance) ft /= r.etap * r.etap;
        s = { wi, f: [ft, ft, ft], pdf, flags: PT_FLAG_TRANSMISSION, etap: r.etap };
      }
      break;
    }
    case PT_BSDF_THIN: {
      const c = Math.abs(wo[2]);
      const R = ptThinReflectance(c, p.ior);
      const T = 1 - R;
      if (u[0] < R) {
        s = { wi: [-wo[0], -wo[1], wo[2]], f: [R / c, R / c, R / c], pdf: R, flags: PT_FLAG_REFLECTION | PT_FLAG_DELTA, etap: 1 };
      } else {
        const t = T / c;
        s = { wi: neg(wo), f: [t * p.reflectance[0], t * p.reflectance[1], t * p.reflectance[2]], pdf: T, flags: PT_FLAG_TRANSMISSION | PT_FLAG_DELTA, etap: 1 };
      }
      break;
    }
    default:
      return fail;
  }
  // pbrt BSDF::Sample_f wrapper: reject empty samples.
  if (s.pdf === 0 || s.wi[2] === 0 || (s.f[0] === 0 && s.f[1] === 0 && s.f[2] === 0)) return fail;
  return s;
}
