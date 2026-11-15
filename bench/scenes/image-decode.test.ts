import { deflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { decodeImage, decodeJpeg, decodePng, imageSize } from './image-decode.ts';
import { JPEG_QUADRANTS_B64, JPEG_QUADRANT_COLOURS } from './image-decode.testdata.ts';

// --- a tiny PNG writer: any colour type / bit depth, filter per row, optional Adam7 ---------------------------------------------------

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = (b: Uint8Array): number => { let c = 0xffffffff; for (const v of b) c = CRC[(c ^ v) & 0xff]! ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
const chunk = (type: string, data: Uint8Array): Buffer => {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0); out.write(type, 4, 'latin1'); out.set(data, 8); out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
};
const paeth = (a: number, b: number, c: number): number => { const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c); return pa <= pb && pa <= pc ? a : pb <= pc ? b : c; };

/** `rows`: one array of raw sample bytes per scanline (already packed to the bit depth). */
function filtered(rows: Uint8Array[], bpp: number, filterOf: (y: number) => number): Buffer {
  const parts: number[] = [];
  rows.forEach((row, y) => {
    const f = filterOf(y), prev = y > 0 ? rows[y - 1]! : new Uint8Array(row.length);
    parts.push(f);
    for (let i = 0; i < row.length; i++) {
      const a = i >= bpp ? row[i - bpp]! : 0, b = prev[i]!, c = i >= bpp ? prev[i - bpp]! : 0;
      parts.push((row[i]! - (f === 0 ? 0 : f === 1 ? a : f === 2 ? b : f === 3 ? (a + b) >> 1 : paeth(a, b, c))) & 0xff);
    }
  });
  return Buffer.from(parts);
}

interface PngSpec { width: number; height: number; ctype: number; depth: number; /** sample at (x, y, channel), integer of `depth` bits */ sample: (x: number, y: number, ch: number) => number; plte?: number[]; trns?: number[]; interlace?: boolean; filterOf?: (y: number) => number }
const CH: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

function pack(spec: PngSpec, xs: number[], y: number): Uint8Array {
  const ch = CH[spec.ctype]!, bits = ch * spec.depth, out = new Uint8Array(Math.ceil((xs.length * bits) / 8));
  xs.forEach((x, i) => {
    for (let c = 0; c < ch; c++) {
      const v = spec.sample(x, y, c), bit = (i * ch + c) * spec.depth;
      if (spec.depth === 16) { out[bit >> 3] = v >> 8; out[(bit >> 3) + 1] = v & 0xff; }
      else if (spec.depth === 8) out[bit >> 3] = v;
      else out[bit >> 3]! |= v << (8 - spec.depth - (bit & 7));
    }
  });
  return out;
}

function makePng(spec: PngSpec): Buffer {
  const bpp = Math.max(1, (CH[spec.ctype]! * spec.depth) >> 3);
  const filterOf = spec.filterOf ?? (() => 0);
  let raw: Buffer;
  if (!spec.interlace) {
    raw = filtered(Array.from({ length: spec.height }, (_, y) => pack(spec, Array.from({ length: spec.width }, (_, x) => x), y)), bpp, filterOf);
  } else {
    const passes = [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]] as const;
    const bufs: Buffer[] = [];
    for (const [x0, y0, dx, dy] of passes) {
      const xs: number[] = [], ys: number[] = [];
      for (let x = x0; x < spec.width; x += dx) xs.push(x);
      for (let y = y0; y < spec.height; y += dy) ys.push(y);
      if (!xs.length || !ys.length) continue;
      bufs.push(filtered(ys.map((y) => pack(spec, xs, y)), bpp, filterOf));
    }
    raw = Buffer.concat(bufs);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(spec.width, 0); ihdr.writeUInt32BE(spec.height, 4); ihdr[8] = spec.depth; ihdr[9] = spec.ctype; ihdr[12] = spec.interlace ? 1 : 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr),
    ...(spec.plte ? [chunk('PLTE', Uint8Array.from(spec.plte))] : []), ...(spec.trns ? [chunk('tRNS', Uint8Array.from(spec.trns))] : []),
    chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0)),
  ]);
}

const px = (img: { width: number; data: Uint8Array }, x: number, y: number): number[] => Array.from(img.data.subarray((y * img.width + x) * 4, (y * img.width + x) * 4 + 4));

describe('PNG decoder', () => {
  const rgba = (x: number, y: number, c: number): number => [(x * 37 + y * 11) & 255, (x * 5 + y * 71) & 255, (x * 101 + y * 3) & 255, (x * 13 + y * 29 + 40) & 255][c]!;
  for (const filter of [0, 1, 2, 3, 4]) {
    it(`RGBA 8-bit, filter type ${filter}`, () => {
      const img = decodePng(makePng({ width: 13, height: 9, ctype: 6, depth: 8, sample: rgba, filterOf: () => filter }));
      expect([img.width, img.height]).toEqual([13, 9]);
      for (let y = 0; y < 9; y++) for (let x = 0; x < 13; x++) expect(px(img, x, y)).toEqual([0, 1, 2, 3].map((c) => rgba(x, y, c)));
    });
  }
  it('mixed filters per row, RGB 8-bit gets alpha 255', () => {
    const img = decodePng(makePng({ width: 6, height: 10, ctype: 2, depth: 8, sample: rgba, filterOf: (y) => y % 5 }));
    for (let y = 0; y < 10; y++) for (let x = 0; x < 6; x++) expect(px(img, x, y)).toEqual([rgba(x, y, 0), rgba(x, y, 1), rgba(x, y, 2), 255]);
  });
  it('grey 1, 2, 4, 8 bit scale to 0-255; grey+alpha 8-bit', () => {
    for (const depth of [1, 2, 4, 8]) {
      const max = (1 << depth) - 1;
      const img = decodePng(makePng({ width: 9, height: 3, ctype: 0, depth, sample: (x, y) => (x + y * 3) % (max + 1) }));
      for (let y = 0; y < 3; y++) for (let x = 0; x < 9; x++) { const v = Math.round((((x + y * 3) % (max + 1)) * 255) / max); expect(px(img, x, y)).toEqual([v, v, v, 255]); }
    }
    const ga = decodePng(makePng({ width: 4, height: 2, ctype: 4, depth: 8, sample: (x, y, c) => (c === 0 ? x * 60 : 200 - y * 50) }));
    expect(px(ga, 3, 1)).toEqual([180, 180, 180, 150]);
  });
  it('16-bit samples reduce to 8 bits (v / 257)', () => {
    const img = decodePng(makePng({ width: 3, height: 1, ctype: 2, depth: 16, sample: (x, _y, c) => (x === 0 ? 0 : x === 1 ? 65535 : 257 * (c + 10)) }));
    expect(px(img, 0, 0)).toEqual([0, 0, 0, 255]);
    expect(px(img, 1, 0)).toEqual([255, 255, 255, 255]);
    expect(px(img, 2, 0)).toEqual([10, 11, 12, 255]);
  });
  it('palette (4-bit) with tRNS alpha; indices past tRNS are opaque', () => {
    const img = decodePng(makePng({ width: 4, height: 1, ctype: 3, depth: 4, sample: (x) => x, plte: [255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9], trns: [0, 128] }));
    expect(px(img, 0, 0)).toEqual([255, 0, 0, 0]);
    expect(px(img, 1, 0)).toEqual([0, 255, 0, 128]);
    expect(px(img, 2, 0)).toEqual([0, 0, 255, 255]);
    expect(px(img, 3, 0)).toEqual([9, 9, 9, 255]);
  });
  it('grey and RGB tRNS key colours become transparent', () => {
    const g = decodePng(makePng({ width: 2, height: 1, ctype: 0, depth: 8, sample: (x) => 10 + x, trns: [0, 10] }));
    expect([px(g, 0, 0)[3], px(g, 1, 0)[3]]).toEqual([0, 255]);
    const c = decodePng(makePng({ width: 2, height: 1, ctype: 2, depth: 8, sample: (x, _y, ch) => (x === 0 ? [1, 2, 3][ch]! : 9), trns: [0, 1, 0, 2, 0, 3] }));
    expect([px(c, 0, 0)[3], px(c, 1, 0)[3]]).toEqual([0, 255]);
  });
  it('Adam7 interlace (7 x 5 and 3 x 2: some passes are empty)', () => {
    for (const [w, h] of [[7, 5], [3, 2], [17, 11]] as const) {
      const img = decodePng(makePng({ width: w, height: h, ctype: 6, depth: 8, sample: rgba, interlace: true, filterOf: (y) => y % 5 }));
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) expect(px(img, x, y), `${w}x${h} (${x},${y})`).toEqual([0, 1, 2, 3].map((c) => rgba(x, y, c)));
    }
  });
  it('imageSize reads PNG and JPEG headers; unknown data is rejected', () => {
    expect(imageSize(makePng({ width: 21, height: 5, ctype: 2, depth: 8, sample: () => 0 }))).toEqual({ width: 21, height: 5 });
    expect(imageSize(Buffer.from(JPEG_QUADRANTS_B64, 'base64'))).toEqual({ width: 16, height: 16 });
    expect(() => decodeImage(Buffer.from('GIF89a......'))).toThrow(/only PNG and JPEG/);
  });
});

describe('JPEG decoder', () => {
  it('decodes a baseline JPEG to RGBA8 (lossy: quadrant colours within 6 codes)', () => {
    const img = decodeJpeg(Buffer.from(JPEG_QUADRANTS_B64, 'base64'));
    expect([img.width, img.height, img.data.length]).toEqual([16, 16, 16 * 16 * 4]);
    [[3, 3], [12, 3], [3, 12], [12, 12]].forEach(([x, y], q) => {
      const p = px(img, x!, y!), want = JPEG_QUADRANT_COLOURS[q]!;
      for (let c = 0; c < 3; c++) expect(Math.abs(p[c]! - want[c]!)).toBeLessThanOrEqual(6);
      expect(p[3]).toBe(255);
    });
    expect(decodeImage(Buffer.from(JPEG_QUADRANTS_B64, 'base64')).width).toBe(16);
  });
});
