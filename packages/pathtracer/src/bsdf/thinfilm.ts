/**
 * OpenPBR 1.1.2 thin-film iridescence (P2.27), TS twin of `thinfilm-wgsl.ts`. Spec section "Thin-film iridescence"
 * (index.html#model/thinfilmiridescence) of https://academysoftwarefoundation.github.io/OpenPBR/:
 *  - parameters `thin_film_weight` (coverage: "a blend between the BSDF with and without the presence of the film"), `thin_film_thickness`
 *    (micrometres) and `thin_film_ior`;
 *  - the film is "incorporated directly into the Fresnel factor of the microfacet BSDFs" of the metal and the dielectric base (no explicit slab);
 *  - recommended model: Belcour and Barla 2017, an Airy sum over the film (their Equation 3) integrated against the CIE curves in Fourier
 *    space, which gives RGB without spectral sampling; the spec allows a spectral Airy evaluation instead and asks for validation against it;
 *  - for a metal the spec suggests mapping the Fresnel factor to an effective IOR (Gulbrandsen 2014): `eta = (1 + sqrt(F0)) / (1 - sqrt(F0))`.
 *
 * Implementation: the Fourier-space RGB evaluation as the Khronos glTF sample renderer writes it for `KHR_materials_iridescence`
 * (`iridescence.glsl`, Apache-2.0; THIRD_PARTY 1.5): exact (unpolarised) Fresnel reflectances at the two interfaces (Khronos uses Schlick), phase shifts 0 or pi from the IOR order, the
 * DC term as RGB and two harmonics through the Gaussian fits of the CIE curves. Changes from it (DIVERGES V82): thickness in micrometres (the
 * spec; glTF uses nm), `outside IOR` 1, a per-channel white balance on the harmonics (`PT_FILM_WHITE`), an exact Fresnel (not Schlick) for the
 * no-film term so weight 0 and thickness 0 are the plain material bit for bit, and the blend by `thin_film_weight`.
 *
 * Out of scope (the reflection lobe only): the film's colour fringes in the transmission lobe of a dielectric, and a film under a coat.
 */
import type { PtVec3 } from './bsdf.ts';

export interface PtThinFilm {
  /** `thin_film_weight` in [0, 1]; 0 = no film. */
  weight: number;
  /** `thin_film_thickness`, micrometres; 0 = no film. */
  thickness: number;
  /** `thin_film_ior`. */
  ior: number;
}

/** True when the film changes anything. */
export const ptFilmActive = (f: PtThinFilm | undefined): f is PtThinFilm => !!f && f.weight > 0 && f.thickness > 0;

/** XYZ to linear Rec.709 (Khronos' constants), row-major. */
const XYZ_TO_REC709 = [3.2404542, -1.5371385, -0.4985314, -0.969266, 1.8760108, 0.041556, 0.0556434, -0.2040259, 1.0572252] as const;
/** The Rec.709 image of XYZ = (1, 1, 1): the white of the Fourier curves (they are normalised to unit area, so an equal-energy spectrum is (1, 1, 1) in XYZ). */
export const PT_FILM_WHITE: PtVec3 = [3.2404542 - 1.5371385 - 0.4985314, -0.969266 + 1.8760108 + 0.041556, 0.0556434 - 0.2040259 + 1.0572252];

const sq = (x: number): number => x * x;
const smoothstep = (a: number, b: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** CIE XYZ curves in Fourier space (Gaussian fits), for the optical path difference `opdNm` (nm) and phase `shift`, as linear RGB. `balance` divides by `PT_FILM_WHITE`. */
export function ptFilmSensitivity(opdNm: number, shift: PtVec3, balance = true): PtVec3 {
  const phase = 2 * Math.PI * opdNm * 1e-9;
  const val = [5.4856e-13, 4.4201e-13, 5.2481e-13];
  const pos = [1.681e6, 1.7953e6, 2.2084e6];
  const vr = [4.3278e9, 9.3046e9, 6.6121e9];
  const xyz = [0, 1, 2].map((i) => (val[i] as number) * Math.sqrt(2 * Math.PI * (vr[i] as number)) * Math.cos((pos[i] as number) * phase + (shift[i] as number)) * Math.exp(-sq(phase) * (vr[i] as number)));
  xyz[0] = (xyz[0] as number) + 9.747e-14 * Math.sqrt(2 * Math.PI * 4.5282e9) * Math.cos(2.2399e6 * phase + shift[0]) * Math.exp(-4.5282e9 * sq(phase));
  const n = xyz.map((v) => v / 1.0685e-7) as [number, number, number];
  const rgb: PtVec3 = [0, 1, 2].map((r) => (XYZ_TO_REC709[r * 3] as number) * n[0] + (XYZ_TO_REC709[r * 3 + 1] as number) * n[1] + (XYZ_TO_REC709[r * 3 + 2] as number) * n[2]) as unknown as PtVec3;
  return balance ? [rgb[0] / PT_FILM_WHITE[0], rgb[1] / PT_FILM_WHITE[1], rgb[2] / PT_FILM_WHITE[2]] : rgb;
}

/** Unpolarised Fresnel reflectance of an interface from a medium of IOR `na` into one of IOR `nb`, at the cosine `c` in the first (Snell; 1 past the critical angle). */
function fresnel(c: number, na: number, nb: number): number {
  const eta = nb / na;
  const s2 = (1 - c * c) / (eta * eta);
  if (s2 >= 1) return 1;
  const ct = Math.sqrt(1 - s2);
  const rp = (eta * c - ct) / (eta * c + ct);
  const rs = (c - eta * ct) / (c + eta * ct);
  return 0.5 * (rp * rp + rs * rs);
}

/**
 * Reflectance of a film (`filmIor`, `thicknessUm`) on a base of normal-incidence reflectance `baseF0` (per channel), seen from a medium of IOR
 * `outsideIor` at `cosTheta1`: Airy sum, Fourier-space RGB. This is the film alone (weight not applied).
 */
export function ptThinFilmAiry(cosTheta1: number, filmIor: number, thicknessUm: number, baseF0: PtVec3, outsideIor = 1, balance = true): PtVec3 {
  const thicknessNm = thicknessUm * 1000;
  const eta2 = outsideIor + (filmIor - outsideIor) * smoothstep(0, 30, thicknessNm);
  const cos1 = Math.min(1, Math.max(0, cosTheta1));
  const sin2Sq = sq(outsideIor / eta2) * (1 - cos1 * cos1);
  const cos2Sq = 1 - sin2Sq;
  if (cos2Sq < 0) return [1, 1, 1];
  const cos2 = Math.sqrt(cos2Sq);
  // first interface
  const R12 = fresnel(cos1, outsideIor, eta2);
  const T121 = 1 - R12;
  const phi12 = eta2 < outsideIor ? Math.PI : 0;
  const phi21 = Math.PI - phi12;
  const opd = 2 * eta2 * thicknessNm * cos2;
  const out: [number, number, number] = [0, 0, 0];
  const cm: [number, number, number] = [0, 0, 0];
  const r123: [number, number, number] = [0, 0, 0];
  const phi: [number, number, number] = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const f0 = Math.min(0.9999, Math.max(0, baseF0[c] as number));
    const s = Math.sqrt(f0);
    const baseIor = (1 + s) / (1 - s);
    const R23 = fresnel(cos2, eta2, baseIor);
    phi[c] = phi21 + (baseIor < eta2 ? Math.PI : 0);
    const R123 = Math.min(0.9999, Math.max(1e-12, R12 * R23));
    r123[c] = Math.sqrt(R123);
    const Rs = (sq(T121) * R23) / (1 - R123);
    out[c] = R12 + Rs;
    cm[c] = Rs - T121;
  }
  for (let m = 1; m <= 2; m++) {
    const s = ptFilmSensitivity(m * opd, [m * (phi[0] as number), m * (phi[1] as number), m * (phi[2] as number)], balance);
    for (let c = 0; c < 3; c++) {
      cm[c] = (cm[c] as number) * (r123[c] as number);
      out[c] = (out[c] as number) + (cm[c] as number) * 2 * (s[c] as number);
    }
  }
  // out-of-gamut or Fourier-truncation overshoot: a reflectance stays in [0, 1] (the film is non-absorbing)
  return [Math.min(1, Math.max(0, out[0])), Math.min(1, Math.max(0, out[1])), Math.min(1, Math.max(0, out[2]))];
}

/**
 * The Fresnel reflectance with the film at coverage `weight`: `(1 - w) noFilm + w airy`. `noFilm` is the plain material's Fresnel at the same
 * angle (dielectric or conductor), `baseF0` its normal-incidence reflectance. A weight or thickness of 0 returns `noFilm` untouched.
 */
export function ptFilmFresnel(film: PtThinFilm | undefined, cosTheta: number, noFilm: PtVec3, baseF0: PtVec3): PtVec3 {
  if (!ptFilmActive(film)) return noFilm;
  const a = ptThinFilmAiry(cosTheta, film.ior, film.thickness, baseF0);
  const w = film.weight;
  return [noFilm[0] + w * (a[0] - noFilm[0]), noFilm[1] + w * (a[1] - noFilm[1]), noFilm[2] + w * (a[2] - noFilm[2])];
}

/** Normal-incidence reflectance of a conductor with complex IOR (eta, k) in air, per channel. */
export function ptConductorF0(eta: PtVec3, k: PtVec3): PtVec3 {
  return [0, 1, 2].map((i) => (sq((eta[i] as number) - 1) + sq(k[i] as number)) / (sq((eta[i] as number) + 1) + sq(k[i] as number))) as unknown as PtVec3;
}
