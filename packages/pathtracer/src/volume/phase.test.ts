import { describe, expect, it } from 'vitest';
import { draine, draineMeanCosine, draineSampleU, hgSampleOneMinusU, phaseEval, phaseMeanCosine, phaseSample, ptMieFit, ptPhaseLobes, rayleighLikeU, type PtPhaseLobes } from './phase.ts';
import { hgPhase } from './volume.ts';

/** integral over the sphere of f(u) (u = cosine with the forward direction), in x = 1 - u on a log grid so a sub-degree forward peak is resolved */
function sphereIntegral(f: (u: number) => number, n = 40000): number {
  // u in [0, 1): log grid in x = 1 - u from 1e-14 to 1; u in [-1, 0]: uniform (back lobes are smooth)
  const lo = Math.log(1e-14);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const x = Math.exp(lo * (1 - (i + 0.5) / n));
    acc += f(1 - x) * x * (-lo / n);
  }
  for (let i = 0; i < n; i++) acc += f(-1 + (i + 0.5) / n) / n;
  return acc * 2 * Math.PI;
}

/** a deterministic low-discrepancy-free uniform stream (xorshift), enough for histograms */
function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

const CASES: Array<[string, PtPhaseLobes]> = [
  ['hg 0.85', ptPhaseLobes({ type: 'hg', g: 0.85 })],
  ['hg 0.99', ptPhaseLobes({ type: 'hg', g: 0.99 })],
  ['hg2 0.2/0.85 40 %', ptPhaseLobes({ type: 'hg2', g1: 0.2, g2: 0.85, w2: 0.4 })],
  ['draine only', { gA: 0, gB: 0.6, w: 1, alpha: 27 }],
  ['draine g 0, alpha 250', { gA: 0, gB: 0, w: 1, alpha: 250 }],
  ['mie 5 um', ptPhaseLobes({ type: 'mie', diameter: 5 })],
  ['mie 20 um', ptPhaseLobes({ type: 'mie', diameter: 20 })],
  ['mie 50 um', ptPhaseLobes({ type: 'mie', diameter: 50 })],
];

describe('the Mie fit (Jendersie and d\'Eon 2023, supplemental Eqs. 7-10)', () => {
  it('has the constants inside the exponents', () => {
    const f = ptMieFit(20);
    expect(f.gHG).toBeCloseTo(Math.exp(-0.0990567 / 18.32846), 12);
    expect(f.gD).toBeCloseTo(0.59379, 4); // exp(-2.20679 / 23.91029 - 0.428934); the misplaced form gives 0.483
    expect(f.wD).toBeCloseTo(0.49816, 4); // exp(-0.599085 / 19.358417 - 0.665888); the misplaced form gives 0.304
    expect(f.alpha).toBeCloseTo(Math.exp(3.62489 - 8.29288 / 25.52825), 12);
    expect(phaseMeanCosine(ptPhaseLobes({ type: 'mie', diameter: 20 }))).toBeCloseTo(0.879, 3);
  });
  it('refuses diameters outside the large-particle range', () => {
    expect(() => ptMieFit(2)).toThrow(RangeError);
    expect(() => ptMieFit(60)).toThrow(RangeError);
  });
});

describe('lobe densities', () => {
  it('Draine with alpha 0 is HG, in our convention too', () => {
    for (const c of [-1, -0.7, 0, 0.4, 1]) expect(draine(-c, 0.8, 0)).toBeCloseTo(hgPhase(c, 0.8), 12);
  });
  for (const [name, l] of CASES) {
    it(`${name}: integrates to 1, mean cosine matches the closed form`, () => {
      expect(sphereIntegral((u) => phaseEval(l, -u))).toBeCloseTo(1, 6);
      expect(sphereIntegral((u) => u * phaseEval(l, -u))).toBeCloseTo(phaseMeanCosine(l), 6);
    });
  }
  it('Draine mean cosine closed form against quadrature', () => {
    for (const [g, a] of [[0.6, 27], [0.3, 1], [-0.4, 5]] as const) expect(sphereIntegral((u) => u * draine(u, g, a))).toBeCloseTo(draineMeanCosine(g, a), 6);
  });
});

describe('sampling', () => {
  it('the HG 1 - u form equals 1 - u of the textbook inversion away from the peak, and stays accurate at it', () => {
    for (const g of [0.3, 0.85, -0.5]) for (const xi of [0.1, 0.5, 0.9]) {
      const t = (1 - g * g) / (1 - g + 2 * g * xi);
      expect(hgSampleOneMinusU(g, xi)).toBeCloseTo(1 - (1 + g * g - t * t) / (2 * g), 12);
    }
    // g 0.995, xi -> 1: 1 - u -> 0 with relative accuracy (the textbook form would cancel to 0 or noise)
    const x = hgSampleOneMinusU(0.995, 1 - 1e-6);
    expect(x).toBeGreaterThan(0);
    expect(x).toBeLessThan(1e-9);
  });
  it('the Draine inversion and the g = 0 Cardano form invert the CDF', () => {
    for (const [g, a] of [[0.6, 27], [0.59, 8], [0.3, 1], [-0.4, 5]] as const) {
      for (const xi of [0.01, 0.2, 0.5, 0.8, 0.99]) {
        const u = draineSampleU(g, a, xi);
        // CDF at u by quadrature on the uniform u grid (the Draine lobe is not sharply peaked)
        let cdf = 0;
        const n = 20000;
        for (let i = 0; i < n; i++) { const v = -1 + ((i + 0.5) / n) * (u + 1); cdf += draine(v, g, a); }
        cdf *= (2 * Math.PI * (u + 1)) / n;
        expect(cdf).toBeCloseTo(xi, 5);
      }
    }
    for (const xi of [0.05, 0.5, 0.95]) {
      const u = rayleighLikeU(250, xi);
      let cdf = 0;
      const n = 20000;
      for (let i = 0; i < n; i++) { const v = -1 + ((i + 0.5) / n) * (u + 1); cdf += draine(v, 0, 250); }
      expect((cdf * 2 * Math.PI * (u + 1)) / n).toBeCloseTo(xi, 5);
    }
  });
  for (const [name, l] of CASES) {
    it(`${name}: the sampled histogram of log(1 - u) matches the density`, () => {
      const bins = 24, lo = Math.log(1e-9), hi = Math.log(2);
      const counts = new Float64Array(bins);
      const r = rng(0x9e3779b9 ^ name.length);
      const n = 400000;
      for (let i = 0; i < n; i++) {
        const { oneMinusU } = phaseSample(l, [r(), r()]);
        const t = Math.log(Math.max(oneMinusU, 1e-300));
        const b = Math.min(bins - 1, Math.max(0, Math.floor(((t - lo) / (hi - lo)) * bins)));
        counts[b]! += 1;
      }
      for (let b = 0; b < bins; b++) {
        // expected mass in the bin (the first bin also takes everything below 1e-9)
        const t0 = b === 0 ? Math.log(1e-30) : lo + (b / bins) * (hi - lo), t1 = lo + ((b + 1) / bins) * (hi - lo);
        let m = 0;
        const k = 400;
        for (let i = 0; i < k; i++) { const x = Math.exp(t0 + ((i + 0.5) / k) * (t1 - t0)); m += phaseEval(l, -(1 - x)) * x; }
        m *= ((t1 - t0) / k) * 2 * Math.PI;
        const expected = m * n;
        // 5 sigma of a Poisson count, plus 1e-3 of the mass for the bin quadrature
        expect(Math.abs((counts[b] as number) - expected)).toBeLessThan(5 * Math.sqrt(expected + 1) + 1e-3 * n * m + 2);
      }
    });
  }
});
