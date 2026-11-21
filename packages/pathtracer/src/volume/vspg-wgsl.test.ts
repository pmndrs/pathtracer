import { describe, expect, it } from 'vitest';
import { PT_VSPG_STEP_WGSL, vspgDecide, vspgGuide, vspgNatural, vspgProbability } from './vspg-wgsl.ts';
import { ptIntegratorWgsl } from '../integrator/integrator-wgsl.ts';

// A homogeneous grey segment [0, L) in a medium of rate mu; f(t) is a smooth integrand of the first collision and f0 the value of a path that passes through.
// Natural expectation: a E[f(t)] + (1 - a) f0, with t the truncated exponential. The guided estimator must give the same for ANY guide Pg.
function pcg(seed: number) {
  let s = seed >>> 0;
  return (): number => {
    s = (s + 0x6d2b79f5) | 0;
    let z = Math.imul(s ^ (s >>> 15), 1 | s);
    z = (z + Math.imul(z ^ (z >>> 7), 61 | z)) ^ z;
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
}

describe('VSPG decision (V4.3)', () => {
  const f = (t: number): number => 1 + Math.sin(3 * t);
  const f0 = 2.5;
  for (const [mu, L] of [[0.05, 1], [1.2, 2], [3, 5], [0.01, 100]] as const) {
    const a = vspgNatural(mu, L);
    // exact: integral of mu exp(-mu t) f(t) dt on [0, L) + exp(-mu L) f0
    let exact = Math.exp(-mu * L) * f0;
    const M = 20000;
    for (let i = 0; i < M; i++) {
      const t = ((i + 0.5) / M) * L;
      exact += mu * Math.exp(-mu * t) * f(t) * (L / M);
    }
    for (const pg of [0.02, 0.3, 0.5, 0.95]) {
      it(`unbiased at mu ${mu}, L ${L}, Pg ${pg}`, () => {
        const rnd = pcg(7 + Math.floor(mu * 100));
        const N = 200000;
        let sum = 0;
        let sumSq = 0;
        for (let i = 0; i < N; i++) {
          const d = vspgDecide(mu, L, pg, rnd(), rnd());
          const x = d.weight * (d.t === null ? f0 : f(d.t));
          sum += x;
          sumSq += x * x;
          expect(d.weight).toBeLessThanOrEqual(4.0000001);
        }
        const mean = sum / N;
        const se = Math.sqrt(Math.max(sumSq / N - mean * mean, 0) / N);
        expect(Math.abs(mean - exact)).toBeLessThan(5 * se + 1e-9);
      });
    }
    it(`probability mixes natural and guide (a ${a.toFixed(4)})`, () => {
      expect(vspgProbability(a, a)).toBeCloseTo(a, 12);
      expect(vspgProbability(a, 1)).toBeCloseTo(0.25 * a + 0.75, 12);
    });
  }
  it('guide falls back to the natural probability without information', () => {
    expect(vspgGuide(0, 0, 0.3)).toBe(0.3);
    expect(vspgGuide(1, 3, 0.3)).toBe(0.25);
  });
  it('step text is only compiled with the option', () => {
    expect(PT_VSPG_STEP_WGSL).toContain('ptVolGuide');
  });
  it('integrator refuses vspg with outputs / non-accumulating kernels', () => {
    const media = [{ sigmaT: [0.1, 0.1, 0.1], albedo: [1, 1, 1], g: 0 }] as never;
    expect(() => ptIntegratorWgsl({ volumes: { media, materials: [], outputs: true, guiding: 'vspg' } as never })).toThrow(/outputs/);
    expect(() => ptIntegratorWgsl({ accumulate: false, volumes: { media, materials: [], guiding: 'vspg' } as never })).toThrow(/accumulate/);
  });
});
