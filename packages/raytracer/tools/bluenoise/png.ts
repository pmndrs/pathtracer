/**
 * Minimal PNG codec for 8/16-bit grey and RGB images (no interlace). The encoder writes one filter type for every row
 * (0 = none by default: blue noise does not predict, so filters only cost time); the decoder handles all five.
 */
import { deflateSync, inflateSync } from 'node:zlib';

export interface PngImage {
  width: number;
  height: number;
  channels: 1 | 3;
  /** 8 or 16. */
  bitDepth: 8 | 16;
  /** Samples as integers (0..255 or 0..65535), row-major, interleaved channels. */
  data: Uint16Array;
}

const SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]!) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Buffer {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'latin1');
  out.set(data, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** `filter` (0..4) is applied to every row; non-zero values exist to test decoders. */
export function encodePng(img: PngImage, filter: 0 | 1 | 2 | 3 | 4 = 0): Buffer {
  const { width, height, channels, bitDepth, data } = img;
  const bytesPerSample = bitDepth / 8;
  const bpp = channels * bytesPerSample;
  const rowBytes = width * bpp;
  const raw = Buffer.alloc((rowBytes + 1) * height);
  const cur = new Uint8Array(rowBytes);
  const prev = new Uint8Array(rowBytes);
  for (let y = 0; y < height; y++) {
    for (let i = 0; i < width * channels; i++) {
      const v = data[y * width * channels + i]!;
      if (bitDepth === 16) {
        cur[2 * i] = v >> 8;
        cur[2 * i + 1] = v & 0xff;
      } else {
        cur[i] = v;
      }
    }
    const o = y * (rowBytes + 1);
    raw[o] = filter;
    for (let i = 0; i < rowBytes; i++) {
      const a = i >= bpp ? cur[i - bpp]! : 0;
      const b = prev[i]!;
      const c = i >= bpp ? prev[i - bpp]! : 0;
      const pred = filter === 0 ? 0 : filter === 1 ? a : filter === 2 ? b : filter === 3 ? (a + b) >> 1 : paeth(a, b, c);
      raw[o + 1 + i] = (cur[i]! - pred) & 0xff;
    }
    prev.set(cur);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = bitDepth;
  ihdr[9] = channels === 1 ? 0 : 2;
  return Buffer.concat([SIGNATURE, chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
}

export function decodePng(file: Uint8Array): PngImage {
  const buf = Buffer.from(file.buffer, file.byteOffset, file.byteLength);
  if (!buf.subarray(0, 8).equals(SIGNATURE)) throw new Error('not a PNG');
  let pos = 8;
  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colorType = -1;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('latin1', pos + 4, pos + 8);
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = body.readUInt32BE(0);
      height = body.readUInt32BE(4);
      bitDepth = body[8]!;
      colorType = body[9]!;
      if (body[12] !== 0) throw new Error('interlaced PNG not supported');
    } else if (type === 'IDAT') idat.push(body);
    else if (type === 'IEND') break;
    pos += 12 + len;
  }
  if ((bitDepth !== 8 && bitDepth !== 16) || (colorType !== 0 && colorType !== 2)) throw new Error('unsupported PNG format');
  const channels = colorType === 0 ? 1 : 3;
  const bps = bitDepth / 8;
  const bpp = channels * bps;
  const rowBytes = width * bpp;
  const raw = inflateSync(Buffer.concat(idat));
  const cur = Buffer.alloc(rowBytes);
  let prev: Buffer = Buffer.alloc(rowBytes);
  const data = new Uint16Array(width * height * channels);
  for (let y = 0; y < height; y++) {
    const o = y * (rowBytes + 1);
    const ft = raw[o]!;
    for (let i = 0; i < rowBytes; i++) {
      const x = raw[o + 1 + i]!;
      const a = i >= bpp ? cur[i - bpp]! : 0;
      const b = prev[i]!;
      const c = i >= bpp ? prev[i - bpp]! : 0;
      let v: number;
      if (ft === 0) v = x;
      else if (ft === 1) v = x + a;
      else if (ft === 2) v = x + b;
      else if (ft === 3) v = x + ((a + b) >> 1);
      else v = x + paeth(a, b, c);
      cur[i] = v & 0xff;
    }
    for (let i = 0; i < width * channels; i++) {
      data[y * width * channels + i] = bps === 2 ? (cur[2 * i]! << 8) | cur[2 * i + 1]! : cur[i]!;
    }
    prev = Buffer.from(cur);
  }
  return { width, height, channels: channels as 1 | 3, bitDepth: bitDepth as 8 | 16, data };
}
