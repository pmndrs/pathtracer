import { readPfm } from '../scenes/pfm.ts';
import type { PfmImage } from '../scenes/pfm.ts';
import { resolve } from 'node:path';

/** Load an `hdri` environment image (PFM only for now). */
export function loadEnvImage(assetsDir: string, path: string): PfmImage {
  if (!path.toLowerCase().endsWith('.pfm')) throw new Error(`hdri '${path}': only PFM is supported by the exporters`);
  return readPfm(resolve(assetsDir, path));
}

/**
 * Bilinear lookup of an equirect image along world direction d. Convention (schema `hdri`): row 0 = +Y, u = 0 at −Z
 * increasing toward +X. `rotationYDeg` rotates the environment about +Y (counter-clockwise seen from above).
 */
export function sampleEquirect(img: PfmImage, d: readonly [number, number, number], rotationYDeg: number, scale: number): [number, number, number] {
  const a = (-rotationYDeg * Math.PI) / 180;
  const c = Math.cos(a), s = Math.sin(a);
  // rotationY(a) applied to d: x' = c·x + s·z, z' = −s·x + c·z
  const x = c * d[0] + s * d[2];
  const z = -s * d[0] + c * d[2];
  const y = Math.min(1, Math.max(-1, d[1]));
  let u = Math.atan2(x, -z) / (2 * Math.PI);
  u -= Math.floor(u);
  const v = Math.acos(y) / Math.PI;
  const fx = u * img.width - 0.5;
  const fy = v * img.height - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const tx = fx - x0, ty = fy - y0;
  const out: [number, number, number] = [0, 0, 0];
  for (let dy = 0; dy < 2; dy++) {
    for (let dx = 0; dx < 2; dx++) {
      const w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty);
      const px = (((x0 + dx) % img.width) + img.width) % img.width;
      const py = Math.min(img.height - 1, Math.max(0, y0 + dy));
      const o = (py * img.width + px) * img.channels;
      for (let k = 0; k < 3; k++) out[k]! += w * img.data[o + (img.channels === 3 ? k : 0)]!;
    }
  }
  return [out[0] * scale, out[1] * scale, out[2] * scale];
}

/** pbrt-v4 EqualAreaSquareToSphere (Clarberg), port of src/pbrt/util/math.cpp. */
export function equalAreaSquareToSphere(px: number, py: number): [number, number, number] {
  const u = 2 * px - 1, v = 2 * py - 1;
  const up = Math.abs(u), vp = Math.abs(v);
  const signedDistance = 1 - (up + vp);
  const d = Math.abs(signedDistance);
  const r = 1 - d;
  const phi = ((r === 0 ? 1 : (vp - up) / r + 1) * Math.PI) / 4;
  const z = Math.sign(signedDistance || 1) * (1 - r * r);
  const cosPhi = Math.abs(Math.cos(phi)) * (u < 0 ? -1 : 1);
  const sinPhi = Math.abs(Math.sin(phi)) * (v < 0 ? -1 : 1);
  const k = r * Math.sqrt(Math.max(0, 2 - r * r));
  return [cosPhi * k, sinPhi * k, z];
}

/**
 * Resample an equirect environment into pbrt-v4's mandatory square equal-area octahedral layout, in WORLD
 * coordinates (the pbrt light gets an identity transform, so pbrt's "z up" light space is simply our world axes).
 * Pixel (i, j) holds the radiance along EqualAreaSquareToSphere((i+.5)/N, (j+.5)/N); row 0 = top of the image file.
 */
export function equirectToOctahedral(img: PfmImage, rotationYDeg: number, scale: number, size: number): PfmImage {
  const data = new Float32Array(size * size * 3);
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const d = equalAreaSquareToSphere((i + 0.5) / size, (j + 0.5) / size);
      data.set(sampleEquirect(img, d, rotationYDeg, scale), (j * size + i) * 3);
    }
  }
  return { width: size, height: size, channels: 3, data };
}
