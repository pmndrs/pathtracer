import { describe, expect, it } from 'vitest';
import { checkMean, checkUniform } from './analytic.ts';
import { makeImage } from './pfm.ts';

describe('analytic checks', () => {
  const img = makeImage(2, 2, (x, y) => (x === 1 && y === 1 ? [1.02, 1, 1] : [1, 1, 1]));
  it('checkUniform finds the worst pixel', () => {
    const r = checkUniform(img, 1, 0.01);
    expect(r.ok).toBe(false);
    expect(r.maxDeviation).toBeCloseTo(0.02, 6);
    expect(r.worst).toMatchObject({ x: 1, y: 1, channel: 0 });
    expect(checkUniform(img, 1, 0.05).ok).toBe(true);
    expect(checkUniform(img, [1, 1, 1], 0.05, { relative: true }).ok).toBe(true);
    expect(checkUniform(img, 1, 0.01, { relative: true }).ok).toBe(false);
  });
  it('checkMean averages per channel', () => {
    const r = checkMean(img, [1.005, 1, 1], 1e-6);
    expect(r.mean[0]).toBeCloseTo(1.005, 6);
    expect(r.ok).toBe(true);
    expect(checkMean(img, 1, 0.001).ok).toBe(false);
  });
  it('non-finite pixels fail both checks', () => {
    const bad = makeImage(1, 1, () => [NaN, 1, 1]);
    expect(checkUniform(bad, 1, 10).ok).toBe(false);
    expect(checkMean(bad, 1, 10).ok).toBe(false);
  });
});
