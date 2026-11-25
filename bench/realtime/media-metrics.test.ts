import { describe, expect, it } from 'vitest';
import { edgeMask, maskedRelMse, temporalExcess, type MImg } from './media-metrics.ts';

const img = (w: number, h: number, f: (x: number, y: number) => number, channels = 3): MImg => {
  const data = new Float32Array(w * h * channels);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < channels; c++) data[(y * w + x) * channels + c] = f(x, y);
  return { width: w, height: h, channels, data };
};

describe('media metrics (V7.4)', () => {
  it('edge mask: a step edge marks a band of 2 * EDGE_R + 2 columns, a flat image none', () => {
    const step = img(32, 8, (x) => (x < 16 ? 0.1 : 5), 4);
    const { mask, fraction } = edgeMask(step);
    expect(mask[4 * 32 + 16]).toBe(1);
    expect(mask[4 * 32 + 2]).toBe(0);
    expect(fraction).toBeGreaterThan(0.2);
    expect(edgeMask(img(16, 16, () => 1)).fraction).toBe(0);
  });
  it('masked relMSE: inside / outside split, identical images are 0', () => {
    const a = img(4, 1, (x) => x);
    const b = img(4, 1, (x) => (x === 3 ? 4 : x));
    const m = new Uint8Array([0, 0, 0, 1]);
    expect(maskedRelMse(a, a, m)).toBe(0);
    expect(maskedRelMse(b, a, m, true)).toBeCloseTo(1 / (9 + 0.01), 6);
    expect(maskedRelMse(b, a, m, false)).toBe(0);
  });
  it('temporal excess is zero when the output changes like the reference', () => {
    const r0 = img(4, 4, () => 1);
    const r1 = img(4, 4, () => 2);
    expect(temporalExcess(r1, r0, r1, r0)).toBe(0);
    expect(temporalExcess(r0, r0, r1, r0)).toBeGreaterThan(0.3);
  });
});
