/**
 * Spectral Airy-sum reference for a thin film (P2.27 oracle): a non-absorbing film of IOR `filmIor` and thickness `thicknessUm` on a base
 * (real or complex IOR), in air, unpolarised light. Per wavelength the exact two-interface reflection coefficient (Belcour and Barla 2017,
 * Equation 3, summed in closed form: r = (r12 + r23 e^(i delta)) / (1 + r12 r23 e^(i delta)), delta = 4 pi n_film d cos(theta_film) / lambda),
 * Fresnel amplitudes for s and p polarisation with complex IORs, averaged as R = (|r_s|^2 + |r_p|^2) / 2; then the spectrum is integrated to
 * linear Rec.709 RGB under D65 (`cie1931.ts`): RGB = M XYZ / M XYZ(constant spectrum), so a constant spectrum c gives (c, c, c).
 * No code from the sources it is validated against.
 */
import { CIE_1931_D65 } from './cie1931.ts';

type C = readonly [number, number];
const add = (a: C, b: C): C => [a[0] + b[0], a[1] + b[1]];
const sub = (a: C, b: C): C => [a[0] - b[0], a[1] - b[1]];
const mul = (a: C, b: C): C => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
const div = (a: C, b: C): C => {
  const d = b[0] * b[0] + b[1] * b[1];
  return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d];
};
const csqrt = (a: C): C => {
  const m = Math.hypot(a[0], a[1]);
  const re = Math.sqrt(Math.max(0, (m + a[0]) / 2));
  const im = Math.sqrt(Math.max(0, (m - a[0]) / 2));
  return [re, a[1] < 0 ? -im : im];
};
const abs2 = (a: C): number => a[0] * a[0] + a[1] * a[1];

const XYZ_TO_REC709 = [3.2404542, -1.5371385, -0.4985314, -0.969266, 1.8760108, 0.041556, 0.0556434, -0.2040259, 1.0572252];

export interface FilmStack {
  filmIor: number;
  /** micrometres */
  thicknessUm: number;
  /** base IOR n + i k (k 0 for a dielectric) */
  baseN: number;
  baseK: number;
  /** IOR of the medium above the film (default 1) */
  outsideIor?: number;
}

/** Unpolarised reflectance of the stack at one wavelength (nm). */
export function airyReflectance(s: FilmStack, cosTheta: number, lambdaNm: number): number {
  const n1: C = [s.outsideIor ?? 1, 0];
  const n2: C = [s.filmIor, 0];
  const n3: C = [s.baseN, s.baseK];
  const sin1 = Math.sqrt(Math.max(0, 1 - cosTheta * cosTheta));
  const cosOf = (n: C): C => csqrt(sub([1, 0], mul(div([n1[0] * sin1, 0], n), div([n1[0] * sin1, 0], n))));
  const c1: C = [cosTheta, 0];
  const c2 = cosOf(n2);
  const c3 = cosOf(n3);
  const delta: C = [(4 * Math.PI * s.filmIor * (s.thicknessUm * 1000) * c2[0]) / lambdaNm, (4 * Math.PI * s.filmIor * (s.thicknessUm * 1000) * c2[1]) / lambdaNm];
  // e^(i delta) = e^(-Im) (cos Re + i sin Re)
  const ph: C = [Math.exp(-delta[1]) * Math.cos(delta[0]), Math.exp(-delta[1]) * Math.sin(delta[0])];
  const rs = (na: C, ca: C, nb: C, cb: C): C => div(sub(mul(na, ca), mul(nb, cb)), add(mul(na, ca), mul(nb, cb)));
  const rp = (na: C, ca: C, nb: C, cb: C): C => div(sub(mul(nb, ca), mul(na, cb)), add(mul(nb, ca), mul(na, cb)));
  const total = (r: (na: C, ca: C, nb: C, cb: C) => C): number => {
    const r12 = r(n1, c1, n2, c2);
    const r23 = r(n2, c2, n3, c3);
    const num = add(r12, mul(r23, ph));
    const den = add([1, 0], mul(mul(r12, r23), ph));
    return abs2(div(num, den));
  };
  return 0.5 * (total(rs) + total(rp));
}

/** Linear Rec.709 RGB of the stack's reflectance spectrum under D65 (constant spectrum c -> (c, c, c)). */
export function airyRgb(s: FilmStack, cosTheta: number): [number, number, number] {
  const xyz = [0, 0, 0];
  const white = [0, 0, 0];
  for (const [l, x, y, z, d] of CIE_1931_D65) {
    const R = airyReflectance(s, cosTheta, l);
    xyz[0]! += R * d * x; xyz[1]! += R * d * y; xyz[2]! += R * d * z;
    white[0]! += d * x; white[1]! += d * y; white[2]! += d * z;
  }
  const m = (v: number[], r: number): number => (XYZ_TO_REC709[r * 3] as number) * (v[0] as number) + (XYZ_TO_REC709[r * 3 + 1] as number) * (v[1] as number) + (XYZ_TO_REC709[r * 3 + 2] as number) * (v[2] as number);
  return [0, 1, 2].map((r) => m(xyz, r) / m(white, r)) as [number, number, number];
}
