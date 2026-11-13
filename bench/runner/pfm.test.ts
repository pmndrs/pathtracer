import { describe, expect, it } from 'vitest';
import { bytesToFloats, countNonFinite, encodePfm, pfmHeader } from './pfm.ts';

/** Minimal PFM reader for the round trip: returns rows in file order (bottom-to-top) as stored. */
function readPfm(file: Buffer): { channels: number; width: number; height: number; scale: number; floats: Float32Array } {
  let pos = 0;
  const line = (): string => {
    const end = file.indexOf(0x0a, pos);
    const text = file.subarray(pos, end).toString('latin1');
    pos = end + 1;
    return text;
  };
  const magic = line();
  const [width, height] = line().split(' ').map(Number) as [number, number];
  const scale = Number(line());
  const channels = magic === 'PF' ? 3 : 1;
  const body = file.subarray(pos);
  const floats = new Float32Array(body.length / 4);
  for (let i = 0; i < floats.length; i++) floats[i] = body.readFloatLE(i * 4);
  return { channels, width, height, scale, floats };
}

describe('pfm', () => {
  it('writes the exact header', () => {
    expect(pfmHeader(4, 2, 3)).toBe('PF\n4 2\n-1.0\n');
    expect(pfmHeader(4, 2, 1)).toBe('Pf\n4 2\n-1.0\n');
  });

  it('flips rows: row 0 (top) is stored last', () => {
    // 2x3 image, 3 channels, pixel value encodes (row, col): r = row, g = col, b = 7.
    const width = 2;
    const height = 3;
    const data = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) data.set([y, x, 7], (y * width + x) * 3);
    const file = readPfm(encodePfm({ width, height, channels: 3, data }));
    expect(file).toMatchObject({ channels: 3, width, height, scale: -1 });
    // First stored pixel is bottom-left (row 2, col 0); last stored is top-right (row 0, col 1).
    expect([...file.floats.slice(0, 3)]).toEqual([2, 0, 7]);
    expect([...file.floats.slice(-3)]).toEqual([0, 1, 7]);
    expect(file.floats).toHaveLength(width * height * 3);
  });

  it('drops alpha from RGBA', () => {
    const data = new Float32Array([1, 2, 3, 99, 4, 5, 6, 98]); // 2x1 RGBA
    const file = readPfm(encodePfm({ width: 2, height: 1, channels: 4, data }));
    expect([...file.floats]).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('writes one channel as Pf', () => {
    const file = readPfm(encodePfm({ width: 2, height: 2, channels: 1, data: new Float32Array([1, 2, 3, 4]) }));
    expect(file.channels).toBe(1);
    expect([...file.floats]).toEqual([3, 4, 1, 2]);
  });

  it('is little-endian regardless of values (1.0 = 00 00 80 3f)', () => {
    const out = encodePfm({ width: 1, height: 1, channels: 1, data: new Float32Array([1]) });
    expect([...out.subarray(out.length - 4)]).toEqual([0x00, 0x00, 0x80, 0x3f]);
  });

  it('rejects a wrong data length or size', () => {
    expect(() => encodePfm({ width: 2, height: 2, channels: 3, data: new Float32Array(5) })).toThrow(/expected 12/);
    expect(() => encodePfm({ width: 0, height: 2, channels: 3, data: new Float32Array(0) })).toThrow();
  });

  it('counts non-finite values and converts bytes to floats even when unaligned', () => {
    expect(countNonFinite(new Float32Array([1, NaN, Infinity, -Infinity, 2]))).toBe(3);
    const raw = Buffer.alloc(9);
    raw.writeFloatLE(1.5, 1);
    expect([...bytesToFloats(raw.subarray(1, 5))]).toEqual([1.5]);
    expect(() => bytesToFloats(Buffer.alloc(3))).toThrow();
  });
});
