/**
 * PNG and JPEG decoding for glTF images (Node only). Output is RGBA8, row 0 = top, exactly the bytes the file stores:
 * no colour management, no gamma chunk, no ICC (glTF says images are sRGB-encoded or linear data by use, and pbrt-v4 /
 * Mitsuba read an untagged 8-bit PNG the same way).
 *
 * PNG: every colour type (grey, RGB, palette, grey+alpha, RGBA), bit depths 1/2/4/8/16 (16-bit is reduced to 8 bits by v / 257, so 8-bit
 * data stays exact), `tRNS` for palette / grey / RGB, Adam7 interlace. JPEG: `jpeg-js` (BSD-3-Clause,
 * THIRD_PARTY.md), baseline and progressive; a JPEG has no alpha.
 */
import { inflateSync } from 'node:zlib';
import jpeg from 'jpeg-js';

export interface DecodedImage { width: number; height: number; /** RGBA8, `width * height * 4`. */ data: Uint8Array }

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export function isPng(b: Uint8Array): boolean {
  return b.length >= 8 && PNG_SIGNATURE.every((v, i) => b[i] === v);
}
export function isJpeg(b: Uint8Array): boolean {
  return b.length >= 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff;
}

/** Decode a PNG or JPEG by its magic bytes (`mimeType` is only used in the error message). */
export function decodeImage(bytes: Uint8Array, mimeType?: string): DecodedImage {
  if (isPng(bytes)) return decodePng(bytes);
  if (isJpeg(bytes)) return decodeJpeg(bytes);
  throw new Error(`unsupported image format${mimeType ? ` (${mimeType})` : ''}: only PNG and JPEG are decoded`);
}

export function decodeJpeg(bytes: Uint8Array): DecodedImage {
  const img = jpeg.decode(bytes, { useTArray: true, formatAsRGBA: true, maxMemoryUsageInMB: 1024 });
  return { width: img.width, height: img.height, data: new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength) };
}

const SAMPLES: Record<number, number> = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Undo the per-row filters of one (sub)image in place; returns the unfiltered rows without their filter bytes. */
function unfilter(raw: Uint8Array, offset: number, width: number, height: number, bitsPerPixel: number): { rows: Uint8Array; end: number } {
  const bpp = Math.max(1, bitsPerPixel >> 3);
  const rowBytes = Math.ceil((width * bitsPerPixel) / 8);
  const out = new Uint8Array(rowBytes * height);
  let o = offset;
  for (let y = 0; y < height; y++) {
    const filter = raw[o++]!;
    const row = y * rowBytes, prev = row - rowBytes;
    for (let i = 0; i < rowBytes; i++) {
      const x = raw[o + i]!;
      const a = i >= bpp ? out[row + i - bpp]! : 0;
      const b = y > 0 ? out[prev + i]! : 0;
      const c = y > 0 && i >= bpp ? out[prev + i - bpp]! : 0;
      let v: number;
      switch (filter) {
        case 0: v = x; break;
        case 1: v = x + a; break;
        case 2: v = x + b; break;
        case 3: v = x + ((a + b) >> 1); break;
        case 4: v = x + paeth(a, b, c); break;
        default: throw new Error(`PNG: bad filter type ${filter}`);
      }
      out[row + i] = v & 0xff;
    }
    o += rowBytes;
  }
  return { rows: out, end: o };
}

const ADAM7 = [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]] as const;

export function decodePng(buf: Uint8Array): DecodedImage {
  if (!isPng(buf)) throw new Error('not a PNG');
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let width = 0, height = 0, depth = 0, ctype = 0, interlace = 0;
  let palette: Uint8Array | undefined, trns: Uint8Array | undefined;
  const idat: Uint8Array[] = [];
  for (let o = 8; o + 8 <= buf.length;) {
    const len = dv.getUint32(o), type = String.fromCharCode(buf[o + 4]!, buf[o + 5]!, buf[o + 6]!, buf[o + 7]!);
    const data = buf.subarray(o + 8, o + 8 + len);
    if (type === 'IHDR') { width = dv.getUint32(o + 8); height = dv.getUint32(o + 12); depth = buf[o + 16]!; ctype = buf[o + 17]!; interlace = buf[o + 20]!; }
    else if (type === 'PLTE') palette = data;
    else if (type === 'tRNS') trns = data;
    else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    o += 12 + len;
  }
  const samples = SAMPLES[ctype];
  if (!samples || !width || !height) throw new Error(`PNG: bad header (colour type ${ctype}, ${width}x${height})`);
  if (![1, 2, 4, 8, 16].includes(depth)) throw new Error(`PNG: bad bit depth ${depth}`);
  if (ctype === 3 && !palette) throw new Error('PNG: palette image without PLTE');
  const raw = new Uint8Array(inflateSync(Buffer.concat(idat)));
  const bitsPerPixel = samples * depth;
  const out = new Uint8Array(width * height * 4);
  const maxv = (1 << depth) - 1;

  // Reads sample `s` of pixel `x` in an unfiltered row starting at `row` (as its raw integer, 16-bit kept whole).
  const sample = (rows: Uint8Array, row: number, x: number, s: number): number => {
    const idx = x * samples + s;
    if (depth === 8) return rows[row + idx]!;
    if (depth === 16) return (rows[row + 2 * idx]! << 8) | rows[row + 2 * idx + 1]!;
    const bit = idx * depth;
    return (rows[row + (bit >> 3)]! >> (8 - depth - (bit & 7))) & maxv;
  };
  const to8 = (v: number): number => (depth === 8 ? v : depth === 16 ? Math.round(v / 257) : Math.round((v * 255) / maxv));
  const put = (rows: Uint8Array, row: number, x: number, dst: number): void => {
    switch (ctype) {
      case 0: {
        const g = sample(rows, row, x, 0), v = to8(g);
        out[dst] = out[dst + 1] = out[dst + 2] = v;
        out[dst + 3] = trns && trns.length >= 2 && g === ((trns[0]! << 8) | trns[1]!) ? 0 : 255;
        break;
      }
      case 2: {
        const r = sample(rows, row, x, 0), g = sample(rows, row, x, 1), b = sample(rows, row, x, 2);
        out[dst] = to8(r); out[dst + 1] = to8(g); out[dst + 2] = to8(b);
        out[dst + 3] = trns && trns.length >= 6 && r === ((trns[0]! << 8) | trns[1]!) && g === ((trns[2]! << 8) | trns[3]!) && b === ((trns[4]! << 8) | trns[5]!) ? 0 : 255;
        break;
      }
      case 3: {
        const i = sample(rows, row, x, 0);
        out[dst] = palette![i * 3] ?? 0; out[dst + 1] = palette![i * 3 + 1] ?? 0; out[dst + 2] = palette![i * 3 + 2] ?? 0;
        out[dst + 3] = trns && i < trns.length ? trns[i]! : 255;
        break;
      }
      case 4: {
        const v = to8(sample(rows, row, x, 0));
        out[dst] = out[dst + 1] = out[dst + 2] = v;
        out[dst + 3] = to8(sample(rows, row, x, 1));
        break;
      }
      default:
        out[dst] = to8(sample(rows, row, x, 0)); out[dst + 1] = to8(sample(rows, row, x, 1)); out[dst + 2] = to8(sample(rows, row, x, 2)); out[dst + 3] = to8(sample(rows, row, x, 3));
    }
  };

  if (!interlace) {
    const { rows } = unfilter(raw, 0, width, height, bitsPerPixel);
    const rowBytes = Math.ceil((width * bitsPerPixel) / 8);
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) put(rows, y * rowBytes, x, (y * width + x) * 4);
  } else {
    let off = 0;
    for (const [x0, y0, dx, dy] of ADAM7) {
      const pw = Math.ceil((width - x0) / dx), ph = Math.ceil((height - y0) / dy);
      if (pw <= 0 || ph <= 0) continue;
      const { rows, end } = unfilter(raw, off, pw, ph, bitsPerPixel);
      off = end;
      const rowBytes = Math.ceil((pw * bitsPerPixel) / 8);
      for (let y = 0; y < ph; y++) for (let x = 0; x < pw; x++) put(rows, y * rowBytes, x, ((y0 + y * dy) * width + (x0 + x * dx)) * 4);
    }
  }
  return { width, height, data: out };
}

/** Width and height of a PNG (IHDR) or JPEG (first SOF marker) without decoding it. */
export function imageSize(b: Uint8Array): { width: number; height: number } {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  if (isPng(b)) return { width: dv.getUint32(16), height: dv.getUint32(20) };
  if (isJpeg(b)) {
    for (let o = 2; o + 9 < b.length;) {
      if (b[o] !== 0xff) { o++; continue; }
      const m = b[o + 1]!;
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return { width: dv.getUint16(o + 7), height: dv.getUint16(o + 5) };
      if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { o += 2; continue; }
      o += 2 + dv.getUint16(o + 2);
    }
  }
  throw new Error('imageSize: not a PNG or JPEG');
}
