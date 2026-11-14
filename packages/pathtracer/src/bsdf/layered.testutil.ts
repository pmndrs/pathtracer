/** Test and bench helpers for the layered BSDF twin (P2.21): quadrature and Monte Carlo estimates of directional albedo. */

import { ptFrameFromNormal, type PtVec3 } from './bsdf.ts';
import { ptLayeredEval, ptLayeredSample, type PtLayeredOptions, type PtLayeredParams } from './layered.ts';
import { rng } from './bsdf.testutil.ts';

export interface Estimate {
  mean: [number, number, number];
  /** standard error of the channel means */
  se: [number, number, number];
}

/**
 * Deterministic quadrature of `ptLayeredEval` over the upper hemisphere (the layering must be `albedo`; a stochastic eval
 * is noisy). Polar coordinates around the mirror direction with a radial warp (see the bruteForceAlbedo trap): resolves a
 * narrow specular lobe and the broad diffuse one in the same pass. Returns the albedo per channel and the integral of the pdf.
 */
export function layeredQuadrature(p: PtLayeredParams, wo: PtVec3, options: PtLayeredOptions = {}, nr = 400, nphi = 360): { albedo: [number, number, number]; pdf: number } {
  const mirror: PtVec3 = [-wo[0], -wo[1], wo[2]];
  const fr = ptFrameFromNormal(mirror);
  const albedo: [number, number, number] = [0, 0, 0];
  let pdf = 0;
  for (let i = 0; i < nr; i++) {
    const t = (i + 0.5) / nr;
    const psi = Math.PI * t * t * t;
    const dpsi = (3 * Math.PI * t * t) / nr;
    const sp = Math.sin(psi);
    const cp = Math.cos(psi);
    const w = sp * dpsi * ((2 * Math.PI) / nphi);
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
      const e = ptLayeredEval(p, wo, wi, options);
      const k = wi[2] * w;
      for (let c = 0; c < 3; c++) albedo[c] = (albedo[c] as number) + (e.f[c] as number) * k;
      pdf += e.pdf * w;
    }
  }
  return { albedo, pdf };
}

function finish(sum: number[], sumSq: number[], n: number): Estimate {
  const mean = sum.map((s) => s / n) as [number, number, number];
  const se = sum.map((s, c) => Math.sqrt(Math.max(0, (sumSq[c] as number) / n - (s / n) ** 2) / n)) as [number, number, number];
  return { mean, se };
}

/** Mean of `f |cos| / pdf` over `ptLayeredSample` draws (failures count as 0): the sampler's own albedo estimate. */
export function sampleAlbedo(p: PtLayeredParams, wo: PtVec3, n: number, seed: number, options: PtLayeredOptions = {}): Estimate {
  const r = rng(seed);
  const sum = [0, 0, 0];
  const sq = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const s = ptLayeredSample(p, wo, [r(), r(), r()], { ...options, seed: (seed * 7919 + i) >>> 0 });
    if (s.flags === 0) continue;
    const k = Math.abs(s.wi[2]) / s.pdf;
    for (let c = 0; c < 3; c++) {
      const v = (s.f[c] as number) * k;
      sum[c] = (sum[c] as number) + v;
      sq[c] = (sq[c] as number) + v * v;
    }
  }
  return finish(sum, sq, n);
}

/** Mean of `f |cos| / (cos / pi)` for cosine-distributed wi over `ptLayeredEval` draws (a fresh seed each): the eval's own albedo estimate. */
export function evalAlbedo(p: PtLayeredParams, wo: PtVec3, n: number, seed: number, options: PtLayeredOptions = {}): Estimate {
  const r = rng(seed);
  const sum = [0, 0, 0];
  const sq = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const u0 = r();
    const u1 = r();
    const rr = Math.sqrt(u0);
    const ph = 2 * Math.PI * u1;
    const wi: PtVec3 = [rr * Math.cos(ph), rr * Math.sin(ph), Math.sqrt(1 - u0)];
    const e = ptLayeredEval(p, wo, wi, { ...options, seed: (seed * 7919 + i) >>> 0 });
    for (let c = 0; c < 3; c++) {
      const v = (e.f[c] as number) * Math.PI;
      sum[c] = (sum[c] as number) + v;
      sq[c] = (sq[c] as number) + v * v;
    }
  }
  return finish(sum, sq, n);
}

/** Mean of f(wo, wi) over `n` seeds (the stochastic estimate's expectation at one direction pair). */
export function meanF(p: PtLayeredParams, wo: PtVec3, wi: PtVec3, n: number, options: PtLayeredOptions = {}): Estimate {
  const sum = [0, 0, 0];
  const sq = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const e = ptLayeredEval(p, wo, wi, { ...options, seed: i + 1 });
    for (let c = 0; c < 3; c++) {
      sum[c] = (sum[c] as number) + (e.f[c] as number);
      sq[c] = (sq[c] as number) + (e.f[c] as number) ** 2;
    }
  }
  return finish(sum, sq, n);
}
