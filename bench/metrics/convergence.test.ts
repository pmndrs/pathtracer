import { describe, expect, it } from 'vitest';
import { convergenceSeries, sppToReach, timeToReach, xToReach } from './convergence.ts';
import { makeImage } from './pfm.ts';

describe('convergence', () => {
  const series = [
    { spp: 1, wallMs: 10, error: 1 },
    { spp: 4, wallMs: 40, error: 0.25 },
    { spp: 16, wallMs: 160, error: 0.0625 },
  ];
  it('log-log interpolation: error ~ 1/spp hits 0.125 at spp 8, 80 ms', () => {
    expect(sppToReach(series, 0.125)).toBeCloseTo(8, 9);
    expect(timeToReach(series, 0.125)).toBeCloseTo(80, 9);
  });
  it('exact hit, first point already below, and never reached', () => {
    expect(timeToReach(series, 0.25)).toBeCloseTo(40, 9);
    expect(timeToReach(series, 5)).toBe(10);
    expect(timeToReach(series, 0.001)).toBeNull();
    expect(xToReach([], 1)).toBeNull();
  });
  it('linear fallback when an error is 0', () => {
    const s = [{ spp: 1, wallMs: 10, error: 1 }, { spp: 2, wallMs: 20, error: 0 }];
    expect(timeToReach(s, 0.5)).toBeCloseTo(15, 9);
  });
  it('convergenceSeries computes the metric per snapshot', () => {
    const ref = makeImage(2, 2, () => [1, 1, 1]);
    const snap = (d: number) => makeImage(2, 2, () => [1 + d, 1 + d, 1 + d]);
    const s = convergenceSeries(
      [{ spp: 1, wallMs: 5, image: snap(0.5) }, { spp: 2, wallMs: 9, image: snap(0.25) }], ref, 'mse');
    expect(s.map((p) => p.error)).toEqual([0.25, 0.0625]);
    expect(s.map((p) => p.spp)).toEqual([1, 2]);
    expect(convergenceSeries([{ spp: 1, wallMs: 5, image: snap(-0.5) }], ref, 'absBias')[0]!.error).toBeCloseTo(0.5, 6);
  });
});
