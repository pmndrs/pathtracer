import { describe, expect, it } from 'vitest';
import { makeImage, readPFM, writePFM } from './pfm.ts';

describe('PFM', () => {
  it('writes bottom-to-top little-endian with the documented header', () => {
    // 1 wide, 2 tall: top = (1,2,3), bottom = (4,5,6)
    const img = makeImage(1, 2, (_x, y) => (y === 0 ? [1, 2, 3] : [4, 5, 6]));
    const bytes = writePFM(img);
    const header = 'PF\n1 2\n-1.0\n';
    expect(new TextDecoder().decode(bytes.subarray(0, header.length))).toBe(header);
    const dv = new DataView(bytes.buffer, bytes.byteOffset + header.length);
    expect([0, 1, 2].map((i) => dv.getFloat32(i * 4, true))).toEqual([4, 5, 6]); // first stored row = bottom
    expect([3, 4, 5].map((i) => dv.getFloat32(i * 4, true))).toEqual([1, 2, 3]);
  });

  it('round-trips with row 0 = top, including non-finite and large values', () => {
    const img = makeImage(3, 2, (x, y) => [x + 10 * y, -x, x === 2 && y === 1 ? Infinity : 1e30]);
    const back = readPFM(writePFM(img));
    expect(back.width).toBe(3);
    expect(back.height).toBe(2);
    expect(back.channels).toBe(3);
    expect(Array.from(back.data)).toEqual(Array.from(img.data));
  });

  it('drops alpha when writing a 4-channel image', () => {
    const img = { width: 1, height: 1, channels: 4 as const, data: new Float32Array([1, 2, 3, 0.5]) };
    expect(Array.from(readPFM(writePFM(img)).data)).toEqual([1, 2, 3]);
  });

  it('reads big-endian (positive scale) and grey "Pf"', () => {
    const header = new TextEncoder().encode('Pf\n2 1\n1.0\n');
    const body = new Uint8Array(8);
    const dv = new DataView(body.buffer);
    dv.setFloat32(0, 0.25, false);
    dv.setFloat32(4, 8, false);
    const file = new Uint8Array(header.length + 8);
    file.set(header);
    file.set(body, header.length);
    const img = readPFM(file);
    expect(Array.from(img.data)).toEqual([0.25, 0.25, 0.25, 8, 8, 8]);
  });

  it('rejects bad magic and truncated data', () => {
    expect(() => readPFM(new TextEncoder().encode('P6\n1 1\n255\n'))).toThrow(/not a PFM/);
    expect(() => readPFM(new TextEncoder().encode('PF\n2 2\n-1.0\n\0\0\0\0'))).toThrow(/truncated/);
  });
});
