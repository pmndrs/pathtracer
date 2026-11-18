import { describe, expect, it } from 'vitest';
import { BoilMeter, tonemappedLuma, type BoilFrame } from './boil.ts';

/** A smooth pattern that translates by `shift` NDC-x per frame; a normally distributed noise `sigma` on top (deterministic LCG). */
function sequence(frames: number, w: number, h: number, shiftPx: number, sigma: number, seed = 1): BoilFrame[] {
  let s = seed >>> 0;
  const rnd = () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(Math.max(rnd(), 1e-12))) * Math.cos(2 * Math.PI * rnd());
  const out: BoilFrame[] = [];
  for (let t = 0; t < frames; t++) {
    const luma = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const u = x + 0.5 - t * shiftPx; // the pattern moves +x by shiftPx per frame
        luma[y * w + x] = 0.5 + 0.3 * Math.sin((2 * Math.PI * u) / 40) * Math.cos((2 * Math.PI * y) / 31) + sigma * gauss();
      }
    // motion = current - previous in NDC: the content moved +x, so a pixel's previous position is to the left: previous = current - motion => motion = +shift
    const motion = new Float32Array(w * h * 2);
    for (let i = 0; i < w * h; i++) motion[i * 2] = (shiftPx / w) * 2;
    out.push({ luma, width: w, height: h, motion, viewZ: new Float32Array(w * h).fill(-5), motionZ: new Float32Array(w * h), mw: w, mh: h });
  }
  return out;
}

function boil(frames: BoilFrame[], stride: number): number {
  const m = new BoilMeter(stride, 1);
  for (const f of frames) m.push(f);
  return m.result().boil;
}

describe('BoilMeter', () => {
  it('is ~0 for a noise-free image under exact reprojection (integer shift)', () => {
    expect(boil(sequence(6, 96, 64, 2, 0), 1)).toBeLessThan(1e-4);
    expect(boil(sequence(12, 96, 64, 2, 0), 4)).toBeLessThan(1e-4);
  });
  it('reads the noise level for independent noise', () => {
    for (const sigma of [0.02, 0.05]) {
      const b = boil(sequence(10, 128, 96, 2, sigma, 7), 1);
      expect(b).toBeGreaterThan(sigma * 0.9);
      expect(b).toBeLessThan(sigma * 1.1);
    }
  });
  it('does not count a depth discontinuity as boil', () => {
    const frames = sequence(5, 64, 48, 2, 0);
    for (const f of frames) for (let i = 0; i < f.viewZ.length; i++) f.motionZ[i] = 1; // previous depth disagrees with the expectation everywhere
    const m = new BoilMeter(1, 1);
    for (const f of frames) m.push(f);
    expect(m.result().samples).toBe(0);
  });
  it('tonemaps luma', () => {
    const l = tonemappedLuma(new Float32Array([1, 1, 1, 1, 0, 0, 0, 1]), 2, 1);
    expect(l[0]).toBeCloseTo(0.5, 6);
    expect(l[1]).toBe(0);
  });
});
