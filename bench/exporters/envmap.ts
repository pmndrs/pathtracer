import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readPFM } from '../metrics/pfm.ts';
import type { Image } from '../src/adapter.ts';

/** Load an `hdri` environment image (PFM only for now). */
export function loadEnvImage(assetsDir: string, path: string): Image {
  if (!path.toLowerCase().endsWith('.pfm')) throw new Error(`hdri '${path}': only PFM is supported by the exporters`);
  return readPFM(readFileSync(resolve(assetsDir, path)));
}

/**
 * Bilinear lookup of an equirect image along world direction d. Convention (schema `hdri`): row 0 = +Y, u = 0 at −Z
 * increasing toward +X. `rotationYDeg` rotates the environment about +Y (counter-clockwise seen from above).
 */
export function sampleEquirect(img: Image, d: readonly [number, number, number], rotationYDeg: number, scale: number): [number, number, number] {
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

/**
 * Mitsuba 3's `envmap` maps v in [0, 1] onto texel coordinate v * (H - 1) (rows sit at v = j / (H - 1), poles included),
 * while u keeps the usual (u * W - 0.5). Fed an equirect as is, its lookup is shifted by (0.5 - v) texels vertically
 * (+0.5 at the zenith, 0 at the equator, -0.5 at the nadir): on env-sphere that moves the sun 0.15 degrees and reads
 * +0.39% / +0.28% / +0.12% (R/G/B) of the true irradiance on a flat plane (V50). Resample the rows to v = j / (H - 1),
 * through our own bilinear lookup (clamped at the poles, texel-centre columns are already right), so Mitsuba's lookup
 * returns the schema's `hdri` values; the residual is the linear re-interpolation between the new rows.
 */
export function equirectForMitsuba(img: Image): Image {
  const { width: w, height: h, channels: c } = img;
  const data = new Float32Array(w * h * c);
  for (let j = 0; j < h; j++) {
    const fy = h > 1 ? (j * h) / (h - 1) - 0.5 : 0; // our row coordinate for v = j / (H - 1)
    const y0 = Math.floor(fy);
    const t = fy - y0;
    const ya = Math.min(h - 1, Math.max(0, y0)), yb = Math.min(h - 1, Math.max(0, y0 + 1));
    for (let i = 0; i < w * c; i++) data[j * w * c + i] = (1 - t) * img.data[ya * w * c + i]! + t * img.data[yb * w * c + i]!;
  }
  return { width: w, height: h, channels: c, data };
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
export function equirectToOctahedral(img: Image, rotationYDeg: number, scale: number, size: number): Image {
  const data = new Float32Array(size * size * 3);
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const d = equalAreaSquareToSphere((i + 0.5) / size, (j + 0.5) / size);
      data.set(sampleEquirect(img, d, rotationYDeg, scale), (j * size + i) * 3);
    }
  }
  return { width: size, height: size, channels: 3, data };
}
