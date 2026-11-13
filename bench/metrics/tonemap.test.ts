import { describe, expect, it } from 'vitest';
import { srgbEncode, tonemapImage, tonemapPixel, type TonemapName } from './tonemap.ts';

const tm = (n: TonemapName, v: number) => { const o = [0, 0, 0]; tonemapPixel(n, v, v, v, o); return o[0]!; };

describe('tonemaps', () => {
  it('clamp / reinhard exact values', () => {
    expect(tm('clamp', 0.25)).toBe(0.25);
    expect(tm('clamp', 7)).toBe(1);
    expect(tm('reinhard', 1)).toBe(0.5);
    expect(tm('reinhard', 3)).toBe(0.75);
  });
  it('aces: 0 -> 0, bright -> 1, 1.0 -> ~0.80 (Narkowicz)', () => {
    expect(tm('aces', 0)).toBe(0);
    expect(tm('aces', 1000)).toBe(1);
    expect(tm('aces', 1)).toBeCloseTo((2.51 + 0.03) / (2.43 + 0.59 + 0.14), 12);
  });
  it('all are monotone, bounded to [0,1], and grey stays grey', () => {
    for (const n of ['clamp', 'reinhard', 'aces', 'agx'] as const) {
      let prev = -1;
      for (const v of [0, 1e-4, 0.01, 0.18, 0.5, 1, 4, 16, 100]) {
        const y = tm(n, v);
        expect(y).toBeGreaterThanOrEqual(prev - 1e-6);
        expect(y).toBeLessThanOrEqual(1);
        prev = y;
      }
    }
    const o = [0, 0, 0];
    tonemapPixel('agx', 0.18, 0.18, 0.18, o);
    expect(Math.abs(o[0]! - o[1]!)).toBeLessThan(0.01);
    expect(Math.abs(o[0]! - o[2]!)).toBeLessThan(0.01);
    // AgX puts mid grey roughly at the sRGB-encoded middle of the range
    expect(srgbEncode(o[0]!)).toBeGreaterThan(0.35);
    expect(srgbEncode(o[0]!)).toBeLessThan(0.6);
  });
  it('EV gain and non-finite handling in tonemapImage', () => {
    const img = { width: 2, height: 1, channels: 3 as const, data: new Float32Array([0.25, 0.25, 0.25, NaN, Infinity, -Infinity]) };
    const out = tonemapImage(img, 'clamp', 1);
    expect(Array.from(out.subarray(0, 3))).toEqual([0.5, 0.5, 0.5]);
    expect(Array.from(out.subarray(3))).toEqual([0, 0, 0]);
  });
});
