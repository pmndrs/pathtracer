/**
 * PFM (Portable Float Map) read/write.
 *
 * File: "PF" (RGB) or "Pf" (grey), whitespace, width, height, scale (sign = endianness: < 0 little-endian),
 * a single whitespace byte, then float32 data. Rows are stored bottom-to-top on disk; our in-memory `Image` is
 * row 0 = top, so both directions flip rows. |scale| is ignored (every tool we use writes 1).
 * Grey "Pf" is expanded to 3 identical channels on read. Writing a 4-channel image drops alpha.
 */
import type { Image } from '../src/adapter.ts';

const isSpace = (c: number) => c === 0x20 || c === 0x0a || c === 0x0d || c === 0x09;

export function readPFM(input: ArrayBuffer | Uint8Array): Image {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  let p = 0;
  const token = (): string => {
    while (p < bytes.length && isSpace(bytes[p]!)) p++;
    const s = p;
    while (p < bytes.length && !isSpace(bytes[p]!)) p++;
    return String.fromCharCode(...bytes.subarray(s, p));
  };
  const magic = token();
  if (magic !== 'PF' && magic !== 'Pf') throw new Error(`not a PFM file (magic "${magic.slice(0, 8)}")`);
  const width = Number(token());
  const height = Number(token());
  const scale = Number(token());
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0 || !Number.isFinite(scale) || scale === 0) {
    throw new Error('bad PFM header');
  }
  p++; // exactly one whitespace byte after the scale
  const srcCh = magic === 'PF' ? 3 : 1;
  const count = width * height * srcCh;
  if (bytes.length - p < count * 4) throw new Error(`PFM truncated: need ${count * 4} data bytes, have ${bytes.length - p}`);
  const little = scale < 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset + p, count * 4);
  const raw = new Float32Array(count);
  for (let i = 0; i < count; i++) raw[i] = view.getFloat32(i * 4, little);
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const srcRow = (height - 1 - y) * width;
    const dstRow = y * width;
    if (srcCh === 3) {
      data.set(raw.subarray(srcRow * 3, (srcRow + width) * 3), dstRow * 3);
    } else {
      for (let x = 0; x < width; x++) {
        const v = raw[srcRow + x]!;
        const o = (dstRow + x) * 3;
        data[o] = v; data[o + 1] = v; data[o + 2] = v;
      }
    }
  }
  return { width, height, channels: 3, data };
}

/** Always writes little-endian "PF" (RGB). */
export function writePFM(image: Image): Uint8Array {
  const { width, height, channels, data } = image;
  const header = new TextEncoder().encode(`PF\n${width} ${height}\n-1.0\n`);
  const out = new Uint8Array(header.length + width * height * 12);
  out.set(header, 0);
  const view = new DataView(out.buffer, header.length);
  for (let y = 0; y < height; y++) {
    const srcRow = y * width;
    const dstRow = (height - 1 - y) * width;
    for (let x = 0; x < width; x++) {
      const s = (srcRow + x) * channels;
      const o = (dstRow + x) * 12;
      view.setFloat32(o, data[s]!, true);
      view.setFloat32(o + 4, data[s + 1]!, true);
      view.setFloat32(o + 8, data[s + 2]!, true);
    }
  }
  return out;
}

/** Build an RGB image from a per-pixel function (handy for tests and synthetic scenes). */
export function makeImage(width: number, height: number, fn: (x: number, y: number) => readonly [number, number, number]): Image {
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = fn(x, y);
      const o = (y * width + x) * 3;
      data[o] = r; data[o + 1] = g; data[o + 2] = b;
    }
  }
  return { width, height, channels: 3, data };
}

/** Scalar field (1 float per pixel) as a grey RGB image, e.g. for writing a FLIP map to PFM. */
export function scalarToImage(width: number, height: number, v: Float32Array): Image {
  const data = new Float32Array(width * height * 3);
  for (let i = 0; i < width * height; i++) { const x = v[i]!; data[3 * i] = x; data[3 * i + 1] = x; data[3 * i + 2] = x; }
  return { width, height, channels: 3, data };
}
