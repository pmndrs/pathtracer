import { describe, expect, it } from 'vitest';
import { absDiffMap, biasMap, compareImages, percentile, relMseMap } from './metrics.ts';
import { makeImage } from './pfm.ts';
import { srgbEncode } from './tonemap.ts';

const grey = (...v: number[]) => makeImage(v.length, 1, (x) => [v[x]!, v[x]!, v[x]!]);

describe('compareImages (hand-computed)', () => {
  const ref = grey(1, 1);
  const test = grey(1, 2);

  it('mse / bias / relBias', () => {
    const m = compareImages(test, ref);
    expect(m.mse).toBeCloseTo(0.5, 12); // (0 + 1) / 2
    expect(m.bias).toBeCloseTo(0.5, 12);
    expect(m.biasRGB).toEqual([0.5, 0.5, 0.5]);
    expect(m.relBias).toBeCloseTo(0.5, 12); // (1+2)/(1+1) - 1
  });

  it('relMSE uses eps = 1e-2 on the reference', () => {
    expect(compareImages(test, ref).relMse).toBeCloseTo(0.5 * (1 / 1.01), 12);
  });

  it('SMAPE = |d| / (|t| + |r| + eps)', () => {
    expect(compareImages(test, ref).smape).toBeCloseTo(0.5 * (1 / 3.01), 12);
    expect(compareImages(grey(0), grey(0)).smape).toBe(0); // 0/0 -> 0
  });

  it('PSNR is on the stated tonemap + sRGB OETF; Infinity when identical', () => {
    expect(compareImages(ref, ref).psnr).toBe(Infinity);
    const m = compareImages(grey(0.5), grey(0), { psnrTonemap: 'clamp' });
    expect(srgbEncode(0.5)).toBeCloseTo(0.7353569830524495, 12);
    expect(m.psnr).toBeCloseTo(-10 * Math.log10(0.7353569830524495 ** 2), 9);
    expect(m.psnrTonemap).toBe('clamp');
  });

  it('firefly % counts pixels above 10x the reference (with a floor on the reference)', () => {
    expect(compareImages(grey(1, 20), grey(1, 1)).fireflyPct).toBeCloseTo(50, 12);
    expect(compareImages(grey(1, 10), grey(1, 1)).fireflyPct).toBe(0); // strictly greater
    // reference 0 -> floor 1e-2 -> threshold 0.1
    expect(compareImages(grey(0.05, 0.2), grey(0, 0)).fireflyPct).toBeCloseTo(50, 12);
  });

  it('ignores alpha and excludes non-finite pixels while counting them', () => {
    const t4 = { width: 2, height: 1, channels: 4 as const, data: new Float32Array([1, 1, 1, 99, NaN, 1, 1, 0]) };
    const m = compareImages(t4, ref);
    expect(m.nonFiniteTest).toBe(1);
    expect(m.pixels).toBe(1);
    expect(m.mse).toBe(0);
  });

  it('rejects mismatched sizes', () => {
    expect(() => compareImages(grey(1), grey(1, 1))).toThrow(/size mismatch/);
  });
});

describe('maps', () => {
  const ref = grey(1, 1);
  const test = makeImage(2, 1, (x) => (x === 0 ? [1, 0.5, 2] : [1, 1, 1]));
  it('absDiff and bias maps', () => {
    expect(Array.from(absDiffMap(test, ref).data)).toEqual([0, 0.5, 1, 0, 0, 0]);
    expect(Array.from(biasMap(test, ref).data)).toEqual([0, -0.5, 1, 0, 0, 0]);
  });
  it('relMse map mean equals relMse', () => {
    const map = relMseMap(test, ref);
    const mean = (map[0]! + map[1]!) / 2;
    expect(mean).toBeCloseTo(compareImages(test, ref).relMse, 6);
    expect(map[0]!).toBeCloseTo((0 + 0.25 / 1.01 + 1 / 1.01) / 3, 6);
  });
  it('percentile (nearest rank)', () => {
    expect(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90)).toBe(9);
    expect(percentile([5, 1, 3], 50)).toBe(3);
    expect(percentile([], 50)).toBeNaN();
  });
});
