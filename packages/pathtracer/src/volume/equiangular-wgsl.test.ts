import { describe, expect, it } from 'vitest';
import { equiangularDistance, equiangularMixSample, equiangularPdf, equiangularSegment, expPdf, ptEquiangularWgsl } from './equiangular-wgsl.ts';

// One homogeneous grey segment [0, L) along +x, a point lamp q at distance h from the ray (isotropic phase): the single-scatter integrand toward q is
//   f(t) = exp(-s t) s a * (1/4pi) * E * exp(-s r(t)) / r(t)^2,  r(t)^2 = h^2 + (t - t0)^2   (E = lamp intensity; unoccluded, the MIS weight is 1).
// E[F] over the sampling density must equal integral_0^L f(t) dt (a 1D quadrature).
function run(h: number, t0: number, L: number, s: number, a: number, mode: 'exp' | 'eq' | 'mix') {
  const f = (t: number): number => {
    const r2 = h * h + (t - t0) * (t - t0);
    return (Math.exp(-s * t) * s * a * Math.exp(-s * Math.sqrt(r2))) / (4 * Math.PI * r2);
  };
  // reference: composite Gauss-Legendre-free brute force, midpoint on a graded grid (h can be 0.02 so 2e6 points)
  let exact = 0;
  const M = 2_000_000;
  for (let i = 0; i < M; i++) exact += f(((i + 0.5) / M) * L);
  exact *= L / M;
  const seg = equiangularSegment([t0, h, 0], [1, 0, 0], L)!;
  const N = 400_000;
  let sum = 0;
  let sum2 = 0;
  let state = 12345;
  const rnd = (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let z = Math.imul(state ^ (state >>> 15), 1 | state);
    z = (z + Math.imul(z ^ (z >>> 7), 61 | z)) ^ z;
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = 0; i < N; i++) {
    const u = rnd();
    let est: number;
    if (mode === 'exp') {
      const m = equiangularMixSample(u, s, L, null);
      est = f(m.t) / m.pdf;
    } else if (mode === 'eq') {
      const t = equiangularDistance(u, seg, L);
      est = f(t) / equiangularPdf(t, seg);
    } else {
      const m = equiangularMixSample(u, s, L, seg);
      est = f(m.t) / m.pdf;
    }
    sum += est;
    sum2 += est * est;
  }
  const mean = sum / N;
  const variance = sum2 / N - mean * mean;
  return { mean, exact, se: Math.sqrt(variance / N), variance };
}

describe('equiangular distance sampling (V4.1)', () => {
  const cases = [
    { h: 0.05, t0: 1.2, L: 3, s: 0.1, a: 0.9 },
    { h: 0.3, t0: -0.5, L: 3, s: 0.5, a: 0.9 }, // foot of the perpendicular outside the segment
    { h: 0.02, t0: 2.9, L: 3, s: 0.05, a: 0.99 },
  ];
  for (const c of cases) {
    it(`mixture and equiangular means equal the quadrature within 4 SE (h ${c.h}, t0 ${c.t0})`, () => {
      for (const mode of ['eq', 'mix'] as const) {
        const r = run(c.h, c.t0, c.L, c.s, c.a, mode);
        expect(Math.abs(r.mean - r.exact)).toBeLessThan(4 * r.se + 1e-4 * r.exact);
      }
    });
  }
  it('has lower variance than the exponential distance alone when the light is close to the ray', () => {
    const c = cases[0]!;
    const e = run(c.h, c.t0, c.L, c.s, c.a, 'exp');
    const m = run(c.h, c.t0, c.L, c.s, c.a, 'mix');
    expect(m.variance).toBeLessThan(e.variance / 3);
  });
  it('densities integrate to one and the degenerate segment is null', () => {
    const seg = equiangularSegment([1, 0.1, 0], [1, 0, 0], 3)!;
    let ie = 0;
    let iq = 0;
    const M = 400000;
    for (let i = 0; i < M; i++) {
      const t = ((i + 0.5) / M) * 3;
      ie += expPdf(t, 0.2, 3);
      iq += equiangularPdf(t, seg);
    }
    expect((ie * 3) / M).toBeCloseTo(1, 4);
    expect((iq * 3) / M).toBeCloseTo(1, 4);
    expect(equiangularSegment([1, 0, 0], [1, 0, 0], 3)).toBeNull();
  });
  it('emits a function that calls the forced vertex NEE and uses atan2 / tan', () => {
    const w = ptEquiangularWgsl();
    expect(w).toContain('ptSampleLdMedium');
    expect(w).toContain('atan2');
    expect(w).toContain('RT_LIGHT_TRIANGLE');
  });
});
