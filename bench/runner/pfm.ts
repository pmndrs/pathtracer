/**
 * PFM writer for the upload endpoint. Interchange images are PFM (float32, linear, little-endian).
 *
 * File layout: header "PF\n<w> <h>\n-1.0\n" (3 channels; "Pf" for 1), then rows of little-endian float32 stored
 * BOTTOM-TO-TOP. Our in-memory convention is row 0 = TOP (packages/raytracer/src/contract, bench/src/adapter.ts), so
 * the writer flips. A negative scale means little-endian. The reader lives in bench/metrics (P0.5); the round-trip
 * test here carries a minimal reader of its own.
 */

export interface PfmSource {
  width: number;
  height: number;
  /** 1, 3 or 4. Alpha (4) is dropped: PFM has no alpha channel. */
  channels: 1 | 3 | 4;
  /** Row-major, row 0 = top. Little-endian host byte order is assumed (every browser we target). */
  data: Float32Array;
}

export function pfmHeader(width: number, height: number, channels: 1 | 3): string {
  return `${channels === 1 ? 'Pf' : 'PF'}\n${width} ${height}\n-1.0\n`;
}

/** Number of floats a source must hold. */
export function expectedFloats(width: number, height: number, channels: number): number {
  return width * height * channels;
}

/** Encodes one image as a PFM file (flipped to bottom-to-top, alpha dropped). */
export function encodePfm(image: PfmSource): Buffer {
  const { width, height, channels, data } = image;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`PFM size must be positive integers, got ${width}x${height}.`);
  }
  if (channels !== 1 && channels !== 3 && channels !== 4) throw new Error(`PFM channels must be 1, 3 or 4, got ${channels}.`);
  if (data.length !== expectedFloats(width, height, channels)) {
    throw new Error(`PFM data has ${data.length} floats, expected ${expectedFloats(width, height, channels)} for ${width}x${height}x${channels}.`);
  }
  const outChannels = channels === 1 ? 1 : 3;
  const header = Buffer.from(pfmHeader(width, height, outChannels), 'latin1');
  const out = Buffer.allocUnsafe(header.length + width * height * outChannels * 4);
  header.copy(out, 0);
  const view = new DataView(out.buffer, out.byteOffset + header.length);
  let offset = 0;
  for (let y = height - 1; y >= 0; y--) {
    for (let x = 0; x < width; x++) {
      const base = (y * width + x) * channels;
      for (let c = 0; c < outChannels; c++) {
        view.setFloat32(offset, data[base + c] as number, true);
        offset += 4;
      }
    }
  }
  return out;
}

/** Count of NaN and infinite values (alpha included), reported next to every upload. */
export function countNonFinite(data: Float32Array): number {
  let bad = 0;
  for (let i = 0; i < data.length; i++) if (!Number.isFinite(data[i] as number)) bad++;
  return bad;
}

/** Reinterprets raw upload bytes as float32 (copying when the buffer is not 4-byte aligned). */
export function bytesToFloats(bytes: Buffer): Float32Array {
  if (bytes.length % 4 !== 0) throw new Error(`Body is ${bytes.length} bytes, not a multiple of 4.`);
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new Float32Array(copy.buffer);
}
