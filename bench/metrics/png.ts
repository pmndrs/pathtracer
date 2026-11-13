/** Minimal 8-bit RGB PNG encoder for CLI previews. NODE ONLY (uses node:zlib); the viewer does not import this. */
import { deflateSync } from 'node:zlib';
import type { Image } from '../src/adapter.ts';
import { colormap, relErrToT, type ColormapName } from './colormap.ts';
import { srgbEncode, tonemapImage, type TonemapName } from './tonemap.ts';

let table: Uint32Array | undefined;
function crc32(buf: Uint8Array): number {
  if (!table) {
    table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; table[n] = c >>> 0; }
  }
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = table[(c ^ buf[i]!) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** rgb8: width*height*3 bytes. */
export function encodePNG(width: number, height: number, rgb8: Uint8Array): Uint8Array {
  const raw = new Uint8Array((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0;
    raw.set(rgb8.subarray(y * width * 3, (y + 1) * width * 3), y * (width * 3 + 1) + 1);
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width); dv.setUint32(4, height);
  ihdr[8] = 8; ihdr[9] = 2;
  const parts = [
    new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0)),
  ];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

const q = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255);

/** Tonemapped sRGB preview of a linear HDR image. */
export function imageToPNG(img: Image, tonemap: TonemapName = 'aces', ev = 0): Uint8Array {
  const lin = tonemapImage(img, tonemap, ev);
  const out = new Uint8Array(img.width * img.height * 3);
  for (let i = 0; i < out.length; i++) out[i] = q(srgbEncode(lin[i]!));
  return encodePNG(img.width, img.height, out);
}

/** False-colour preview of a scalar field: value -> t via `toT`, then a colour ramp. */
export function scalarToPNG(w: number, h: number, v: Float32Array, map: ColormapName, toT: (x: number) => number): Uint8Array {
  const out = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) {
    const c = colormap(map, toT(v[i]!));
    out[i * 3] = q(c[0]); out[i * 3 + 1] = q(c[1]); out[i * 3 + 2] = q(c[2]);
  }
  return encodePNG(w, h, out);
}

export { relErrToT };
