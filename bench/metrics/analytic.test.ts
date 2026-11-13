import { describe, expect, it } from 'vitest';
import { checkMaskedImage, checkMean, checkUniform, quadLightAnalyticImage, quadLightCentreIrradianceClosedForm, quadLightIrradiance } from './analytic.ts';
import { quadLight } from '../scenes/tier0/quad-light.ts';
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

describe('quad-light analytic', () => {
  it('numeric irradiance matches the corner-rectangle closed form under the centre', () => {
    expect(quadLightIrradiance(0, 0, 400)).toBeCloseTo(quadLightCentreIrradianceClosedForm(), 4);
  });
  it('falls off away from the light', () => {
    expect(quadLightIrradiance(3, 0)).toBeLessThan(quadLightIrradiance(0, 0) * 0.05);
  });
  it('analytic image masks the light and sky, and an exact copy passes checkMaskedImage', () => {
    const small = { ...quadLight, film: { width: 24, height: 24, filter: 'box' as const } }; // keeps the test fast
    const exp = quadLightAnalyticImage(small);
    const { width: w, height: h } = small.film;
    const nan = exp.reduce((a, v, i) => a + (i % 3 === 0 && Number.isNaN(v) ? 1 : 0), 0);
    expect(nan).toBeGreaterThan(0);
    expect(nan).toBeLessThan(w * h);
    const img = { width: w, height: h, channels: 3 as const, data: Float32Array.from(exp, (v) => (Number.isNaN(v) ? 0 : v)) };
    const ok = checkMaskedImage(img, exp, 1e-6);
    expect(ok.ok).toBe(true);
    expect(ok.n).toBe(w * h - nan);
    // 1% too bright on every plane pixel: mean deviation +0.01 fails a 0.005 gate, passes a 0.02 gate
    const bright = { ...img, data: img.data.map((v) => v * 1.01) };
    expect(checkMaskedImage(bright, exp, 0.005).ok).toBe(false);
    expect(checkMaskedImage(bright, exp, 0.02).ok).toBe(true);
    expect(checkMaskedImage(bright, exp, 0.02).meanRelDev).toBeCloseTo(0.01, 5);
  });
  it('checkMaskedImage fails on non-finite test pixels', () => {
    const exp = new Float32Array([1, 1, 1]);
    expect(checkMaskedImage({ width: 1, height: 1, channels: 3, data: new Float32Array([1, NaN, 1]) }, exp, 0.1).ok).toBe(false);
  });
});

