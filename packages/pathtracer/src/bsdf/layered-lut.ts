/**
 * Directional albedo of the GGX dielectric reflection lobe (P2.21, `layering: 'albedo'`), split-sum form.
 *
 * E(mu, r, ior) = integral of f_spec(wo, wi) cos(wi) dwi, with f_spec the reflection lobe of `ptBsdfEval` (Fresnel
 * included). The dielectric Fresnel is written F = F0 (1 - w) + w with the shape w(wo.wm) of `ptFresnelShape` (Schlick: (1 - c)^5), so
 *     E = F0 * A(mu, r) + B(mu, r),   A = integral rho0 (1 - w),   B = integral rho0 w,   rho0 = D G / (4 mu_o)
 * and the table needs only two ior-independent 2D slices (A, B). F0 = ((ior - 1) / (ior + 1))^2. The exact-Fresnel
 * albedo (`ptDielectricAlbedoExact`) is the CPU oracle: the table's error against it is a test.
 *
 * Axes: sqrt(mu) uniform in [0, 1] (N points, the first at mu = 0; E climbs steeply at grazing), perceptual roughness r: sqrt(r) uniform in [0, 1]
 * (N points; alpha = r^2). Bilinear lookup. Built on the CPU on first use (a few ms) and embedded in the WGSL as a const.
 *
 * Reference for the structure: OpenPBR Surface specification, "Layering" (slab stacking by directional albedo,
 * `layer(top, base) = top + base * (1 - E_top)`), and Karis 2013 / Kulla-Conty 2017 for the split-sum table.
 */

import { ptFrDielectric, ptGgxD, ptGgxG, ptFrameFromNormal, type PtVec3 } from './bsdf.ts';

export const PT_LAYERED_LUT_N = 16;

/** pbrt/OpenPBR F0 of a dielectric of relative IOR `ior` against air. */
export function ptDielectricF0(ior: number): number {
  const t = (ior - 1) / (ior + 1);
  return t * t;
}

const SHAPE_IOR = 1.5;
/**
 * Grazing shape of the dielectric Fresnel, w(c) = (F(c) - F0) / (1 - F0), taken from the exact Fresnel at ior 1.5 instead
 * of Schlick's (1 - c)^5 (which is 3.5% off at 1.5 in the middle angles). F = F0 + (1 - F0) w holds at every ior; w is
 * only weakly ior dependent, so the table is exact at 1.5 (the glTF default) and close elsewhere.
 */
export function ptFresnelShape(c: number): number {
  const f0 = ptDielectricF0(SHAPE_IOR);
  return (ptFrDielectric(Math.max(c, 0), SHAPE_IOR) - f0) / (1 - f0);
}

/** integrand weights: mode 0 -> exact Fresnel, 1 -> A (rho0 (1-w)), 2 -> B (rho0 w) */
function integrate(mu: number, r: number, mode: 0 | 1 | 2, ior = SHAPE_IOR, nr = 96, nphi = 72): number {
  const alpha = Math.max(r * r, 1e-4);
  const a: readonly [number, number] = [alpha, alpha];
  const s = Math.sqrt(Math.max(0, 1 - mu * mu));
  const wo: PtVec3 = [s, 0, Math.max(mu, 1e-4)];
  const wl = Math.hypot(wo[0], wo[2]);
  const o: PtVec3 = [wo[0] / wl, 0, wo[2] / wl];
  const mirror: PtVec3 = [-o[0], 0, o[2]];
  const fr = ptFrameFromNormal(mirror);
  let sum = 0;
  for (let i = 0; i < nr; i++) {
    const t = (i + 0.5) / nr;
    const psi = Math.PI * t * t * t;
    const dpsi = (3 * Math.PI * t * t) / nr;
    const sp = Math.sin(psi);
    const cp = Math.cos(psi);
    const wgt = sp * dpsi * ((2 * Math.PI) / nphi);
    for (let j = 0; j < nphi; j++) {
      const chi = ((j + 0.5) / nphi) * 2 * Math.PI;
      const cc = Math.cos(chi) * sp;
      const ss = Math.sin(chi) * sp;
      const wi: PtVec3 = [
        mirror[0] * cp + fr.t[0] * cc + fr.b[0] * ss,
        mirror[1] * cp + fr.t[1] * cc + fr.b[1] * ss,
        mirror[2] * cp + fr.t[2] * cc + fr.b[2] * ss,
      ];
      if (wi[2] <= 0) continue;
      let wm: PtVec3 = [wi[0] + o[0], wi[1] + o[1], wi[2] + o[2]];
      const l = Math.hypot(wm[0], wm[1], wm[2]);
      if (l === 0) continue;
      wm = [wm[0] / l, wm[1] / l, wm[2] / l];
      const d = Math.abs(o[0] * wm[0] + o[1] * wm[1] + o[2] * wm[2]);
      const rho0 = (ptGgxD(a, wm) * ptGgxG(a, o, wi)) / (4 * o[2]);
      const w5 = ptFresnelShape(d);
      const g = mode === 0 ? ptFrDielectric(d, ior) : mode === 1 ? 1 - w5 : w5;
      sum += rho0 * g * wgt;
    }
  }
  return sum;
}

/** Exact directional albedo of the GGX dielectric reflection lobe (exact Fresnel), alpha = r^2. Smooth: F(mu). */
export function ptDielectricAlbedoExact(ior: number, mu: number, roughness: number): number {
  if (roughness * roughness < 1e-3) return ptFrDielectric(Math.max(mu, 1e-6), ior);
  return integrate(mu, roughness, 0, ior);
}

let cached: Float32Array | null = null;

/** The (A, B) table, interleaved: index ((ir * N + im) * 2 + c). Row ir = roughness, column im = mu. */
export function ptLayeredLut(): Float32Array {
  if (cached) return cached;
  const n = PT_LAYERED_LUT_N;
  const t = new Float32Array(n * n * 2);
  for (let ir = 0; ir < n; ir++) {
    const r = (ir / (n - 1)) ** 2;
    for (let im = 0; im < n; im++) {
      const mu = (im / (n - 1)) ** 2;
      let A: number;
      let B: number;
      if (r * r < 1e-3) {
        // smooth lobe: E = F, so A = 1 - w, B = w exactly
        const w = ptFresnelShape(mu);
        A = 1 - w;
        B = w;
      } else {
        A = integrate(Math.max(mu, 1e-3), r, 1);
        B = integrate(Math.max(mu, 1e-3), r, 2);
      }
      t[(ir * n + im) * 2] = A;
      t[(ir * n + im) * 2 + 1] = B;
    }
  }
  // Round to what the WGSL const carries (4 decimals), so CPU and GPU read the same numbers.
  for (let i = 0; i < t.length; i++) t[i] = Math.fround(Math.round((t[i] as number) * 1e4) / 1e4);
  cached = t;
  return t;
}

/** Bilinear lookup of E(mu, r) for F0 (the CPU twin of `ptLayeredAlbedo` in WGSL). */
export function ptLayeredAlbedo(f0: number, mu: number, roughness: number): number {
  const n = PT_LAYERED_LUT_N;
  const t = ptLayeredLut();
  const x = Math.sqrt(Math.min(Math.max(mu, 0), 1)) * (n - 1);
  const y = Math.sqrt(Math.min(Math.max(roughness, 0), 1)) * (n - 1);
  const x0 = Math.min(Math.floor(x), n - 2);
  const y0 = Math.min(Math.floor(y), n - 2);
  const fx = x - x0;
  const fy = y - y0;
  const at = (iy: number, ix: number): number => f0 * (t[(iy * n + ix) * 2] as number) + (t[(iy * n + ix) * 2 + 1] as number);
  const e = (at(y0, x0) * (1 - fx) + at(y0, x0 + 1) * fx) * (1 - fy) + (at(y0 + 1, x0) * (1 - fx) + at(y0 + 1, x0 + 1) * fx) * fy;
  return Math.min(1, Math.max(0, e));
}
