import { describe, expect, it } from 'vitest';
import { decodePfm, encodePfm } from './pfm.ts';

describe('pfm', () => {
  it('round-trips and stores scanlines bottom-to-top on disk', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]); // 2x2 RGB, row 0 = top
    const buf = encodePfm({ width: 2, height: 2, channels: 3, data });
    const header = '-1.0\n';
    const bodyStart = buf.indexOf(header) + header.length;
    expect(buf.readFloatLE(bodyStart)).toBe(7); // first stored scanline is the bottom row
    const back = decodePfm(buf);
    expect(Array.from(back.data)).toEqual(Array.from(data));
    expect(back.width).toBe(2);
  });
});
