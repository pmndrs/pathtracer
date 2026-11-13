import { describe, expect, it } from 'vitest';
import type { Image } from '../src/adapter.ts';
import { blockMeanDiff, fitPowerLaw, fmtMs, isPlateau, timeToTarget } from './baseline.ts';

describe('fitPowerLaw', () => {
  it('recovers a slope of -1 exactly', () => {
    const fit = fitPowerLaw([100, 400, 1600, 6400].map((t) => ({ t, e: 5 / t })))!;
    expect(fit.k).toBeCloseTo(1, 10);
    expect(fit.A).toBeCloseTo(5, 8);
    expect(fit.r2).toBeCloseTo(1, 10);
  });
  it('needs two usable points', () => {
    expect(fitPowerLaw([{ t: 1, e: 1 }])).toBeNull();
    expect(fitPowerLaw([{ t: 1, e: 0 }, { t: 2, e: 1 }])).toBeNull();
  });
});

describe('timeToTarget', () => {
  const series = [100, 400, 1600].map((wallMs, i) => ({ spp: 4 ** (i + 1), wallMs, error: 1 / wallMs }));
  const fit = fitPowerLaw(series.map((p) => ({ t: p.wallMs, e: p.error })));
  it('interpolates inside the series (log-log)', () => {
    const r = timeToTarget(series, 1 / 800, 0, fit);
    expect(r.kind).toBe('interp');
    expect(r.ms).toBeCloseTo(800, 3);
  });
  it('extrapolates from the fit past the last snapshot and says so', () => {
    const r = timeToTarget(series, 1 / 8000, 0, fit);
    expect(r.kind).toBe('extrap');
    expect(r.ms).toBeCloseTo(8000, 3);
  });
  it('refuses targets under twice the reference floor', () => {
    expect(timeToTarget(series, 1e-3, 1e-3, fit).kind).toBe('not-measurable');
    expect(timeToTarget(series, 2.1e-3, 1e-3, fit).kind).not.toBe('not-measurable');
  });
  it('reports not reached when the curve is flat', () => {
    const flat = [{ spp: 1, wallMs: 1, error: 0.5 }, { spp: 2, wallMs: 2, error: 0.49 }];
    expect(timeToTarget(flat, 0.1, 0, fitPowerLaw(flat.map((p) => ({ t: p.wallMs, e: p.error }))))).toEqual({ target: 0.1, kind: 'not-reached' });
  });
});

describe('blockMeanDiff and fmtMs', () => {
  const img = (v: number): Image => ({ width: 4, height: 4, channels: 4, data: new Float32Array(64).fill(v) });
  it('is zero for equal images and the relative gap otherwise', () => {
    expect(blockMeanDiff(img(1), img(1), 2).maxRel).toBe(0);
    expect(blockMeanDiff(img(1.1), img(1), 2).maxRel).toBeCloseTo(0.1, 5);
  });
  it('formats durations', () => {
    expect([fmtMs(12), fmtMs(1500), fmtMs(90000)]).toEqual(['12 ms', '1.5 s', '1.5 min']);
  });
});

describe('isPlateau', () => {
  it('flags a curve that stops following 1/spp and passes one that does', () => {
    expect(isPlateau([{ spp: 256, error: 1.07e-3 }, { spp: 1024, error: 6.8e-4 }])).toBe(true);
    expect(isPlateau([{ spp: 256, error: 4.5e-4 }, { spp: 1024, error: 1.5e-4 }])).toBe(false);
    expect(isPlateau([{ spp: 1, error: 1 }])).toBe(false);
  });
});
