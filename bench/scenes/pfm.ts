/**
 * Minimal PFM (Portable Float Map) reader/writer for the interchange format used across the bench.
 *
 * On disk we follow the PFM spec exactly so pbrt-v4, Mitsuba and generic tools read our files: little-endian float32
 * (scale -1), RGB ('PF') or grey ('Pf'), scanlines stored **bottom-to-top**. In memory (`PfmImage.data`) rows are
 * **top-down** (row 0 = top), the bench convention, so the reader/writer flip rows.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

export interface PfmImage {
  width: number;
  height: number;
  /** 1 (grey) or 3 (RGB). */
  channels: 1 | 3;
  /** Row-major, row 0 = top, interleaved channels. */
  data: Float32Array;
}

export function encodePfm(img: PfmImage): Buffer {
  const header = Buffer.from(`${img.channels === 3 ? 'PF' : 'Pf'}\n${img.width} ${img.height}\n-1.0\n`, 'latin1');
  const rowLen = img.width * img.channels;
  const body = Buffer.alloc(img.data.length * 4);
  for (let y = 0; y < img.height; y++) {
    const srcRow = img.height - 1 - y; // file row y (bottom-up) = image row height-1-y
    for (let i = 0; i < rowLen; i++) body.writeFloatLE(img.data[srcRow * rowLen + i]!, (y * rowLen + i) * 4);
  }
  return Buffer.concat([header, body]);
}

export function decodePfm(buf: Buffer): PfmImage {
  let pos = 0;
  const token = (): string => {
    while (pos < buf.length && /\s/.test(String.fromCharCode(buf[pos]!))) pos++;
    const start = pos;
    while (pos < buf.length && !/\s/.test(String.fromCharCode(buf[pos]!))) pos++;
    return buf.toString('latin1', start, pos);
  };
  const magic = token();
  if (magic !== 'PF' && magic !== 'Pf') throw new Error(`not a PFM file (magic '${magic}')`);
  const channels = magic === 'PF' ? 3 : 1;
  const width = Number(token());
  const height = Number(token());
  const scale = Number(token());
  pos++; // single whitespace after the scale
  const little = scale < 0;
  const rowLen = width * channels;
  const data = new Float32Array(width * height * channels);
  for (let y = 0; y < height; y++) {
    const dstRow = height - 1 - y;
    for (let i = 0; i < rowLen; i++) {
      const off = pos + (y * rowLen + i) * 4;
      data[dstRow * rowLen + i] = little ? buf.readFloatLE(off) : buf.readFloatBE(off);
    }
  }
  return { width, height, channels: channels as 1 | 3, data };
}

export function writePfm(path: string, img: PfmImage): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, encodePfm(img));
}

export function readPfm(path: string): PfmImage {
  return decodePfm(readFileSync(path));
}
