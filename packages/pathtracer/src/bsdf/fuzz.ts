/**
 * OpenPBR 1.1.2 fuzz layer (P2.25), TS twin of `fuzz-wgsl.ts`. Spec, section "Fuzz" (index.html#model/fuzz) of
 * https://academysoftwarefoundation.github.io/OpenPBR/, which recommends the model of Zeltner, Burley and Chiang 2022
 * (LTC sheen, fitted to a volume of SGGX fibres; tables and reference code Apache-2.0, THIRD_PARTY 1.5).
 *
 * Spec formulas used here:
 *  - lobe: `mu_i f_fuzz(wi, wo) = F E_fuzz(mu_o, alpha) D(mu_i | mu_o, alpha)` with F = `fuzz_color`, alpha = `fuzz_roughness`, D a linearly
 *    transformed cosine (LTC) lobe tabulated on a (mu_o, alpha) grid; its albedo is `F E_fuzz`. Here E_fuzz is the table's R, D is
 *    `evalLTC` of the reference (a normalised density over the hemisphere), so `f_fuzz = F R D / mu_i`.
 *  - layering (albedo scaling, equation "fuzz-layering-approx"): `f = w f_fuzz + (1 - w E_fuzz) f_coated-base`, w = `fuzz_weight`.
 *    The scale `1 - w E_fuzz` is grey: the fuzz colour never tints the base. It scales everything below the fuzz, reflection and
 *    transmission alike. The fuzz normal is the base shading normal (no separate normal here).
 *
 * The LTC is a cosine-weighted lobe, so it is only approximately reciprocal (the repository's own note); the spec claims no reciprocity.
 *
 * Sampling: a one-sample mixture. The fuzz lobe is chosen with probability `pf`, the base with `1 - pf`; `pf` only shapes variance
 * (any value in (0, 1) is unbiased because every returned `pdf` is the mixture density). A delta lobe of the base keeps its delta flag.
 * The fuzz lobe lives on the +z side only: a ray inside a transmissive base (`wo.z <= 0`) sees no fuzz.
 */
import { PT_BSDF_BLACK, PT_FLAG_DELTA, PT_FLAG_REFLECTION, type PtBsdfEval, type PtBsdfParams, type PtBsdfSample, type PtVec3 } from './bsdf.ts';
import { PT_FUZZ_RES, PT_FUZZ_TABLE } from './fuzz-table.ts';

export { PT_FUZZ_RES, PT_FUZZ_TABLE };

/** `fuzz_weight`, `fuzz_color`, `fuzz_roughness` (all in [0, 1]). */
export interface PtFuzzParams {
  weight: number;
  color: PtVec3;
  roughness: number;
}

/** The table as a vec4 per cell (aInv, bInv, R, 0), row-major (alpha, cos): what the kernel binds (`ptFuzzTable`). */
export function ptFuzzTableF32(): Float32Array<ArrayBuffer> {
  const out = new Float32Array(PT_FUZZ_RES * PT_FUZZ_RES * 4);
  for (let i = 0; i < PT_FUZZ_RES * PT_FUZZ_RES; i++) {
    out[i * 4] = PT_FUZZ_TABLE[i * 3] as number;
    out[i * 4 + 1] = PT_FUZZ_TABLE[i * 3 + 1] as number;
    out[i * 4 + 2] = PT_FUZZ_TABLE[i * 3 + 2] as number;
  }
  return out;
}

const PI = Math.PI;
const ONE_MINUS_EPS = 1 - 2 ** -24;
/** Below this `aInv` a table cell is an empty (all zero) one: the lobe is absent there. */
const MIN_A = 1e-4;

/** Bilinear lookup of (aInv, bInv, R) at (cos theta_o, alpha), clamped as the reference (`fetchCoeffs`). */
export function ptFuzzCoeffs(cosO: number, alpha: number): [number, number, number] {
  const row = Math.max(0, Math.min(alpha, ONE_MINUS_EPS)) * (PT_FUZZ_RES - 1);
  const col = Math.max(0, Math.min(cosO, ONE_MINUS_EPS)) * (PT_FUZZ_RES - 1);
  const r = Math.floor(row);
  const c = Math.floor(col);
  const rf = row - r;
  const cf = col - c;
  const out: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const t = (rr: number, cc: number): number => PT_FUZZ_TABLE[(rr * PT_FUZZ_RES + cc) * 3 + k] as number;
    out[k] = (t(r, c) * (1 - cf) + t(r, c + 1) * cf) * (1 - rf) + (t(r + 1, c) * (1 - cf) + t(r + 1, c + 1) * cf) * rf;
  }
  return out;
}

/** E_fuzz(mu_o, alpha): the albedo of the fuzz lobe with a white `fuzz_color`. */
export function ptFuzzAlbedo(cosO: number, alpha: number): number {
  return ptFuzzCoeffs(cosO, alpha)[2];
}

/** The fuzz lobe is active for this view: weight and colour positive, outside, table has a lobe here. */
export function ptFuzzActive(f: PtFuzzParams, woZ: number): boolean {
  if (!(f.weight > 0) || !(Math.max(f.color[0], f.color[1], f.color[2]) > 0) || !(woZ > 0)) return false;
  const c = ptFuzzCoeffs(woZ, f.roughness);
  return c[0] > MIN_A && c[2] > 0;
}

/** Rotates v about +z so that wo's azimuth becomes 0 (`sign` = +1 undoes it). ρ = 0 (wo along the normal): identity. */
function rotZ(v: PtVec3, woX: number, woY: number, sign: 1 | -1): PtVec3 {
  const rho = Math.hypot(woX, woY);
  if (rho === 0) return v;
  const c = woX / rho;
  const s = woY / rho; // c = cos phi, s = sin phi; sign -1 rotates by -phi: x' = x c + y s, y' = -x s + y c
  return sign === -1 ? [v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2]] : [v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]];
}

/** Density of the LTC lobe (per solid angle) at wi for the view wo: the reference `evalLTC`, a normalised pdf over the hemisphere. */
export function ptFuzzLobePdf(coeffs: readonly number[], wo: PtVec3, wi: PtVec3): number {
  if (!(wi[2] > 0)) return 0;
  const a = coeffs[0] as number;
  const b = coeffs[1] as number;
  if (!(a > MIN_A)) return 0;
  const w = rotZ(wi, wo[0], wo[1], -1);
  const ox = a * w[0] + b * w[2];
  const oy = a * w[1];
  const len = Math.hypot(ox, oy, w[2]);
  // cos-hemisphere pdf (oz / len) / pi times the Jacobian a^2 / len^3
  return (w[2] * a * a) / (PI * len ** 4);
}

/** Cosine-weighted LTC sample (the reference `sampleLTC`): `u` in [0, 1)^2. */
export function ptFuzzLobeSample(coeffs: readonly number[], wo: PtVec3, u0: number, u1: number): PtVec3 {
  const a = coeffs[0] as number;
  const b = coeffs[1] as number;
  const r = Math.sqrt(u0);
  const phi = 2 * PI * u1;
  const ox = r * Math.cos(phi);
  const oy = r * Math.sin(phi);
  const oz = Math.sqrt(Math.max(0, 1 - u0));
  const x = ox / a - (oz * b) / a;
  const y = oy / a;
  const l = Math.hypot(x, y, oz);
  return rotZ([x / l, y / l, oz / l], wo[0], wo[1], 1);
}

/** The fuzz lobe alone: `f = F R D / mu_i` (RGB, F the fuzz colour, no weight) and the density D. */
export function ptFuzzLobeEval(f: PtFuzzParams, wo: PtVec3, wi: PtVec3): PtBsdfEval {
  if (!ptFuzzActive(f, wo[2]) || !(wi[2] > 0)) return { f: [0, 0, 0], pdf: 0 };
  const c = ptFuzzCoeffs(wo[2], f.roughness);
  const pdf = ptFuzzLobePdf(c, wo, wi);
  const k = (c[2] * pdf) / wi[2];
  return { f: [f.color[0] * k, f.color[1] * k, f.color[2] * k], pdf };
}

/** Base lobes the fuzz sits on (anything with the `ptBsdfEval` / `ptBsdfSample` shapes: a `PtBsdfParams`, a layered BSDF, ...). */
export interface PtFuzzBase {
  eval(wo: PtVec3, wi: PtVec3): PtBsdfEval;
  sample(wo: PtVec3, u: PtVec3): PtBsdfSample;
  /** True for `PT_BSDF_BLACK` (nothing to scatter below the fuzz). */
  black: boolean;
}

/** `pf`, the chance of choosing the fuzz lobe: its share of the albedo (the base counted as 1), clamped; 1 over a black base. */
export function ptFuzzChoice(f: PtFuzzParams, woZ: number, baseBlack: boolean): number {
  if (baseBlack) return 1;
  const e = ptFuzzAlbedo(woZ, f.roughness);
  const wf = f.weight * e * ((f.color[0] + f.color[1] + f.color[2]) / 3);
  const pf = wf / (wf + (1 - f.weight * e));
  return Math.min(0.95, Math.max(0.05, pf));
}

/** `w f_fuzz + (1 - w E_fuzz) f_base`, pdf the mixture. Inactive fuzz returns the base unchanged. */
export function ptFuzzedEval(f: PtFuzzParams, base: PtFuzzBase, wo: PtVec3, wi: PtVec3): PtBsdfEval {
  if (!ptFuzzActive(f, wo[2])) return base.eval(wo, wi);
  const pf = ptFuzzChoice(f, wo[2], base.black);
  const keep = 1 - f.weight * ptFuzzAlbedo(wo[2], f.roughness);
  const b = base.black ? { f: [0, 0, 0] as PtVec3, pdf: 0 } : base.eval(wo, wi);
  const z = ptFuzzLobeEval(f, wo, wi);
  return {
    f: [f.weight * z.f[0] + keep * b.f[0], f.weight * z.f[1] + keep * b.f[1], f.weight * z.f[2] + keep * b.f[2]],
    pdf: pf * z.pdf + (1 - pf) * b.pdf,
  };
}

const FAIL: PtBsdfSample = { wi: [0, 0, 0], f: [0, 0, 0], pdf: 0, flags: 0, etap: 1 };

/** `u = [uc, u0, u1]`. A sample of the fuzz lobe or of a non-delta base lobe returns the total f and the mixture pdf (flags without `PT_FLAG_PROPORTIONAL`); a delta base lobe returns `(1 - w E) f` and `(1 - pf) pdf`. */
export function ptFuzzedSample(f: PtFuzzParams, base: PtFuzzBase, wo: PtVec3, u: PtVec3): PtBsdfSample {
  if (!ptFuzzActive(f, wo[2])) return base.sample(wo, u);
  const pf = ptFuzzChoice(f, wo[2], base.black);
  if (u[0] < pf) {
    const c = ptFuzzCoeffs(wo[2], f.roughness);
    const wi = ptFuzzLobeSample(c, wo, u[1], u[2]);
    const e = ptFuzzedEval(f, base, wo, wi);
    return e.pdf > 0 ? { wi, f: e.f, pdf: e.pdf, flags: PT_FLAG_REFLECTION, etap: 1 } : FAIL;
  }
  const s = base.sample(wo, [(u[0] - pf) / (1 - pf), u[1], u[2]]);
  if (s.flags === 0 || !(s.pdf > 0)) return FAIL;
  if (s.flags & PT_FLAG_DELTA) {
    const keep = 1 - f.weight * ptFuzzAlbedo(wo[2], f.roughness);
    return { ...s, f: [keep * s.f[0], keep * s.f[1], keep * s.f[2]], pdf: (1 - pf) * s.pdf };
  }
  const e = ptFuzzedEval(f, base, wo, s.wi);
  return e.pdf > 0 ? { wi: s.wi, f: e.f, pdf: e.pdf, flags: s.flags & (PT_FLAG_REFLECTION | 4), etap: s.etap } : FAIL;
}

/** `PtFuzzBase` over a plain `PtBsdfParams` (anything but `PT_BSDF_LAYERED`), the shapes of `ptBsdfEval` / `ptBsdfSample`. */
export function ptFuzzBaseOf(p: PtBsdfParams, evalFn: (p: PtBsdfParams, wo: PtVec3, wi: PtVec3) => PtBsdfEval, sampleFn: (p: PtBsdfParams, wo: PtVec3, u: PtVec3) => PtBsdfSample): PtFuzzBase {
  return { eval: (wo, wi) => evalFn(p, wo, wi), sample: (wo, u) => sampleFn(p, wo, u), black: p.kind === PT_BSDF_BLACK };
}
