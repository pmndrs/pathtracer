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
