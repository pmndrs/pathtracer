import { describe, expect, it } from 'vitest';
import { forcedDistance, forcedNeeWeight, ptForcedNeeWgsl } from './forced-wgsl.ts';

describe('forced in-scatter NEE (V4.2)', () => {
  // One segment, light attenuation h(t) = exp(-k t) (stands for Ld(t) / beta): E = integral_0^L exp(-s t) s a h(t) dt, per channel.
  const cases: { s: number[]; a: number[]; mu: number; L: number; k: number }[] = [
    { s: [0.1, 0.1, 0.1], a: [0.9, 0.9, 0.9], mu: 0.1, L: 3, k: 0.7 },
    { s: [2, 2, 2], a: [1, 0.5, 0.2], mu: 2, L: 0.5, k: 0.1 },
    { s: [0.5, 1.0, 2.0], a: [0.8, 0.8, 0.8], mu: 2.0, L: 4, k: 0.3 },
  ];
  for (const c of cases) {
    it(`mean of the forced estimate equals the first-collision integral (mu ${c.mu}, L ${c.L})`, () => {
      const N = 200000;
      const sum = [0, 0, 0];
      for (let i = 0; i < N; i++) {
        const u = (i + 0.5) / N; // stratified
        const t = forcedDistance(u, c.mu, c.L);
        const w = forcedNeeWeight(c.s, c.a, c.mu, t, c.L);
        for (let j = 0; j < 3; j++) sum[j]! += w[j]! * Math.exp(-c.k * t);
      }
      for (let j = 0; j < 3; j++) {
        const s = c.s[j]!;
        const r = s + c.k;
        const exact = (s * c.a[j]! * (1 - Math.exp(-r * c.L))) / r;
        expect(sum[j]! / N / exact).toBeCloseTo(1, 3);
      }
    });
  }
  it('emits a function that uses the forced vertex slots', () => {
    expect(ptForcedNeeWgsl()).toContain('ptSampleLdMedium');
  });
});

import { PT_FORCED_VERTEX_BASE, PT_MARCH_MAX, PT_MARCH_STRIDE, forcedMarchVertex, marchEstimate, ptMarchCount } from './forced-wgsl.ts';
import { ptEquiangularWgsl } from './equiangular-wgsl.ts';
import { PT_PAIR } from '../integrator/dimensions.ts';

describe('V7.1 march strata', () => {
  it('dimension map: every (stratum, depth, slot) pair index is unique and stays in u32', () => {
    const seen = new Set<number>();
    let max = 0;
    for (let i = 0; i < PT_MARCH_MAX; i++) {
      for (const depth of [0, 1, 2, 7, 64, 1000]) {
        for (let slot = 0; slot < PT_PAIR.pairsPerVertex; slot++) {
          const pair = PT_PAIR.vertexBase + forcedMarchVertex(i, depth) * PT_PAIR.pairsPerVertex + slot;
          expect(pair).toBeLessThan(2 ** 32);
          expect(seen.has(pair)).toBe(false);
          seen.add(pair);
          max = Math.max(max, pair);
        }
      }
    }
    // stratum 0 is the V4.2 forced vertex; no stratum reaches a real (surface / scatter) vertex index below the base
    expect(forcedMarchVertex(0, 5)).toBe(PT_FORCED_VERTEX_BASE + 5);
    expect(forcedMarchVertex(1, 0)).toBe(PT_FORCED_VERTEX_BASE + PT_MARCH_STRIDE);
    expect(max).toBeLessThan(2 ** 32);
  });
  it('option resolution', () => {
    expect(ptMarchCount(undefined)).toBe(1);
    expect(ptMarchCount({ marchSamples: 1 })).toBe(1);
    expect(ptMarchCount({ marchSamples: 8 })).toBe(8);
    expect(() => ptMarchCount({ marchSamples: 2.5 })).toThrow();
    expect(() => ptMarchCount({ marchSamples: 65 })).toThrow();
  });
  it('WGSL: march text only for N > 1, default text unchanged', () => {
    expect(ptForcedNeeWgsl(1)).toBe(ptForcedNeeWgsl());
    expect(ptEquiangularWgsl(1)).toBe(ptEquiangularWgsl());
    expect(ptForcedNeeWgsl(8)).toContain('8u');
    expect(ptForcedNeeWgsl(8)).not.toBe(ptForcedNeeWgsl());
  });
  // Line integral E = integral_0^L exp(-s t) s a h(t) dt with h = exp(-k t); the estimator's mean must equal it at every N, its variance fall with N.
  const mu = 0.8, L = 3, s = 0.8, a = 0.9, k = 0.5;
  const exact = (s * a * (1 - Math.exp(-(s + k) * L))) / (s + k);
  const f = (u: number) => forcedNeeWeight([s], [a], mu, forcedDistance(u, mu, L), L)[0]! * Math.exp(-k * forcedDistance(u, mu, L));
  let seed = 12345;
  const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const stats = (n: number) => {
    const R = 40000;
    let m = 0, m2 = 0;
    for (let r = 0; r < R; r++) {
      const v = marchEstimate(f, n, rnd);
      m += v;
      m2 += v * v;
    }
    m /= R;
    return { mean: m, variance: m2 / R - m * m };
  };
  const res = [1, 4, 16].map((n) => ({ n, ...stats(n) }));
  for (const r of res) {
    it(`stratified mean equals the line integral at N = ${r.n}`, () => {
      expect(r.mean / exact).toBeCloseTo(1, 2);
    });
  }
  it('variance falls with N', () => {
    expect(res[1]!.variance).toBeLessThan(res[0]!.variance / 3);
    expect(res[2]!.variance).toBeLessThan(res[1]!.variance / 3);
  });
});
