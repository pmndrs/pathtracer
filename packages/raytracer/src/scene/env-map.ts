/**
 * Environment map (P1.4d): an equirectangular HDR image in the scene copy's `data` buffer, with a luminance-weighted 2D
 * sampling distribution (marginal CDF over rows, conditional CDF inside a row) and the TypeScript (f64) twin of the WGSL
 * lookup / pdf / sample in light-sample.ts (`rtEnvMapWgsl`).
 *
 * Convention (the bench schema's `hdri`, also Mitsuba's `envmap` with an identity transform): row 0 = +Y, `u = atan2(x, -z) / 2pi`
 * (u = 0 at -Z, 0.25 at +X), `v = acos(y) / pi`; pixel (i, j) covers [i/W, (i+1)/W] x [j/H, (j+1)/H]. `rotationY` (radians, about
 * +Y, counter-clockwise seen from above) turns the environment; it is the bench's `sampleEquirect` rotation.
 *
 * Radiance lookup: bilinear between the four nearest texel centres (u wraps, v clamps), as Mitsuba and the bench's own
 * `sampleEquirect`. The sampling density is piecewise constant per pixel: `p_ij ~ Y(L_ij) * sin(theta_j)` (Y = Rec.709 luminance
 * of the texel, theta_j the row centre), so in solid angle
 *
 *     pdf(w) = Y(L_ij) * sin(theta_j) * norm / sin(theta),   norm = W * H / (S * 2 pi^2),   S = sum_ij Y(L_ij) sin(theta_j)
 *
 * (`dw = 2 pi^2 sin(theta) du dv`). Any positive density is unbiased with MIS; it is only worth having where the radiance is.
 *
 * Layout inside `data` from the map base word: `H` f32 marginal CDF values (cumulative row weight / S, last = 1), then `W * H`
 * texels of `ENV_TEXEL_WORDS` words: r, g, b (f32 bit patterns, the file's radiance, not multiplied by `scale`) and the f32
 * conditional CDF of the pixel inside its row (last = 1). A search finds the first entry with `cdf > u`, so zero-weight rows and
 * pixels are never picked. Black or non-finite texels count as zero weight (negative and NaN are clamped to 0).
 */
import { luminance, f32Bits, bitsF32, type Rgb3 } from './material.ts';
import type { Vec3Tuple } from './lights.ts';

export const ENV_TEXEL_WORDS = 4;

export interface EnvMapInput {
  width: number;
  height: number;
  /** 1 (grey), 3 or 4 (alpha ignored). */
  channels: number;
  data: ArrayLike<number>;
}

export interface EnvMapTable {
  width: number;
  height: number;
  /** `[marginal CDF: height words][texels: width * height * ENV_TEXEL_WORDS words]`. */
  words: Uint32Array;
  /** `W * H / (S * 2 pi^2)`, rounded to f32; 0 for an all-black map. */
  pdfNorm: number;
  /** `S`. */
  weightSum: number;
}

const clampRadiance = (v: number): number => (v > 0 && Number.isFinite(v) ? v : 0);

function texelOf(img: EnvMapInput, i: number, j: number): Rgb3 {
  const o = (j * img.width + i) * img.channels;
  const c = (k: number) => clampRadiance(img.data[o + (img.channels >= 3 ? k : 0)] as number);
  return [c(0), c(1), c(2)];
}

const sinRow = (j: number, height: number): number => Math.sin((Math.PI * (j + 0.5)) / height);

function checkInput(img: EnvMapInput): void {
  if (!(Number.isInteger(img.width) && Number.isInteger(img.height) && img.width > 0 && img.height > 0)) throw new RangeError('environment map: bad size');
  if (img.width > 0xffff || img.height > 0xffff) throw new RangeError('environment map: size must fit 16 bits');
  if (img.channels !== 1 && img.channels !== 3 && img.channels !== 4) throw new RangeError(`environment map: ${img.channels} channels`);
  if (img.data.length < img.width * img.height * img.channels) throw new RangeError('environment map: data too short');
}

/** Solid-angle-weighted mean radiance (sin theta weights): the constant radiance that carries the same power. */
export function envMapMean(img: EnvMapInput): Rgb3 {
  checkInput(img);
  const sum = [0, 0, 0];
  let weight = 0;
  for (let j = 0; j < img.height; j++) {
    const s = sinRow(j, img.height);
    for (let i = 0; i < img.width; i++) {
      const t = texelOf(img, i, j);
      for (let k = 0; k < 3; k++) sum[k] = (sum[k] as number) + s * (t[k] as number);
      weight += s;
    }
  }
  return [(sum[0] as number) / weight, (sum[1] as number) / weight, (sum[2] as number) / weight];
}

export function buildEnvMapTable(img: EnvMapInput): EnvMapTable {
  checkInput(img);
  const { width: w, height: h } = img;
  const words = new Uint32Array(h + w * h * ENV_TEXEL_WORDS);
  const rowSum = new Float64Array(h);
  const weights = new Float64Array(w * h);
  let total = 0;
  for (let j = 0; j < h; j++) {
    const s = sinRow(j, h);
    let r = 0;
    for (let i = 0; i < w; i++) {
      const wt = luminance(texelOf(img, i, j)) * s;
      weights[j * w + i] = wt;
      r += wt;
    }
    rowSum[j] = r;
    total += r;
  }
  const uniform = !(total > 0);
  let cum = 0;
  for (let j = 0; j < h; j++) {
    cum += uniform ? sinRow(j, h) : (rowSum[j] as number);
    words[j] = f32Bits(j === h - 1 ? 1 : Math.min(1, cum / (uniform ? sinRowSum(h) : total)));
  }
  for (let j = 0; j < h; j++) {
    let c = 0;
    const r = rowSum[j] as number;
    for (let i = 0; i < w; i++) {
      const t = texelOf(img, i, j);
      const at = h + (j * w + i) * ENV_TEXEL_WORDS;
      words[at] = f32Bits(t[0]);
      words[at + 1] = f32Bits(t[1]);
      words[at + 2] = f32Bits(t[2]);
      c += r > 0 ? (weights[j * w + i] as number) : 1;
      words[at + 3] = f32Bits(i === w - 1 ? 1 : Math.min(1, c / (r > 0 ? r : w)));
    }
  }
  const norm = uniform ? 0 : (w * h) / (total * 2 * Math.PI * Math.PI);
  return { width: w, height: h, words, pdfNorm: Math.fround(norm), weightSum: uniform ? 0 : total };
}

function sinRowSum(h: number): number {
  let s = 0;
  for (let j = 0; j < h; j++) s += sinRow(j, h);
  return s;
}

// ---------------------------------------------------------------------------------------------------------------
// TypeScript twins of the WGSL (reading the packed words)
// ---------------------------------------------------------------------------------------------------------------

export interface EnvMapView {
  words: ArrayLike<number>;
  /** Word index of the first marginal CDF value. */
  base: number;
  width: number;
  height: number;
  rotationY: number;
  pdfNorm: number;
}

/** A view over a standalone table (tests). */
export function envMapViewOfTable(table: EnvMapTable, rotationY = 0): EnvMapView {
  return { words: table.words, base: 0, width: table.width, height: table.height, rotationY, pdfNorm: table.pdfNorm };
}

/** World direction -> map direction (the bench's `sampleEquirect` rotation). */
export function envWorldToMap(rotationY: number, d: Vec3Tuple): Vec3Tuple {
  const c = Math.cos(rotationY);
  const s = Math.sin(rotationY);
  return [c * d[0] - s * d[2], d[1], s * d[0] + c * d[2]];
}

export function envMapToWorld(rotationY: number, m: Vec3Tuple): Vec3Tuple {
  const c = Math.cos(rotationY);
  const s = Math.sin(rotationY);
  return [c * m[0] + s * m[2], m[1], -s * m[0] + c * m[2]];
}

/** Equirect coordinates of a map-space unit direction: u in [0, 1), v in [0, 1]. */
export function envDirToUv(m: Vec3Tuple): [number, number] {
  let u = Math.atan2(m[0], -m[2]) / (2 * Math.PI);
  u -= Math.floor(u);
  return [u, Math.acos(Math.min(1, Math.max(-1, m[1]))) / Math.PI];
}

/** Map-space unit direction of equirect coordinates (the inverse of `envDirToUv`). */
export function envUvToDir(u: number, v: number): Vec3Tuple {
  const phi = 2 * Math.PI * u;
  const theta = Math.PI * v;
  const st = Math.sin(theta);
  return [st * Math.sin(phi), Math.cos(theta), -st * Math.cos(phi)];
}

function texelAt(view: EnvMapView, i: number, j: number): Rgb3 {
  const at = view.base + view.height + (j * view.width + i) * ENV_TEXEL_WORDS;
  return [bitsF32(view.words[at] as number), bitsF32(view.words[at + 1] as number), bitsF32(view.words[at + 2] as number)];
}

/** Twin of `rtEnvMapLookup`: bilinear radiance along world direction `d`, not multiplied by the light's `scale`. */
export function envMapLookup(view: EnvMapView, d: Vec3Tuple): Rgb3 {
  const [u, v] = envDirToUv(envWorldToMap(view.rotationY, d));
  const fx = u * view.width - 0.5;
  const fy = v * view.height - 0.5;
  const x0 = Math.floor(fx);
  const y0 = Math.floor(fy);
  const tx = fx - x0;
  const ty = fy - y0;
  const out: Rgb3 = [0, 0, 0];
  for (let dy = 0; dy < 2; dy++) {
    for (let dx = 0; dx < 2; dx++) {
      const wgt = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty);
      const px = (((x0 + dx) % view.width) + view.width) % view.width;
      const py = Math.min(view.height - 1, Math.max(0, y0 + dy));
      const t = texelAt(view, px, py);
      for (let k = 0; k < 3; k++) out[k] = (out[k] as number) + wgt * (t[k] as number);
    }
  }
  return out;
}

function pixelPdf(view: EnvMapView, i: number, j: number, sinTheta: number): number {
  if (!(sinTheta > 1e-12)) return 0;
  return (luminance(texelAt(view, i, j)) * sinRow(j, view.height) * view.pdfNorm) / sinTheta;
}

/** Twin of `rtEnvMapPdf`: the solid-angle density with which `sampleEnvMap` produces world direction `d` (before light selection). */
export function envMapPdf(view: EnvMapView, d: Vec3Tuple): number {
  const m = envWorldToMap(view.rotationY, d);
  const [u, v] = envDirToUv(m);
  const i = Math.min(Math.floor(u * view.width), view.width - 1);
  const j = Math.min(Math.floor(v * view.height), view.height - 1);
  return pixelPdf(view, i, j, Math.sqrt(Math.max(0, 1 - m[1] * m[1])));
}

/** First index in `[0, n)` whose f32 CDF value (word `base + k * stride`) is greater than `u`. */
function searchCdf(words: ArrayLike<number>, base: number, stride: number, n: number, u: number): number {
  let lo = 0;
  let hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (bitsF32(words[base + mid * stride] as number) > u) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

export interface EnvMapSample {
  /** World direction. */
  wi: Vec3Tuple;
  /** Solid-angle density (before light selection). */
  pdf: number;
  /** Bilinear radiance along `wi`, not multiplied by `scale`. */
  radiance: Rgb3;
}

/**
 * Twin of `rtSampleEnvMap`: `u1` picks the row (marginal CDF), `u0` the column (that row's conditional CDF); the leftover of each
 * places the point inside the pixel. Returns `pdf = 0` for an all-black map.
 */
export function sampleEnvMap(view: EnvMapView, u0: number, u1: number): EnvMapSample {
  const { width: w, height: h, words, base } = view;
  const j = searchCdf(words, base, 1, h, u1);
  const rowLo = j > 0 ? bitsF32(words[base + j - 1] as number) : 0;
  const rowHi = bitsF32(words[base + j] as number);
  const dv = rowHi > rowLo ? Math.min(Math.max((u1 - rowLo) / (rowHi - rowLo), 0), 1 - 2 ** -24) : 0.5;
  const rowBase = base + h + j * w * ENV_TEXEL_WORDS + 3;
  const i = searchCdf(words, rowBase, ENV_TEXEL_WORDS, w, u0);
  const colLo = i > 0 ? bitsF32(words[rowBase + (i - 1) * ENV_TEXEL_WORDS] as number) : 0;
  const colHi = bitsF32(words[rowBase + i * ENV_TEXEL_WORDS] as number);
  const du = colHi > colLo ? Math.min(Math.max((u0 - colLo) / (colHi - colLo), 0), 1 - 2 ** -24) : 0.5;
  const m = envUvToDir((i + du) / w, (j + dv) / h);
  const wi = envMapToWorld(view.rotationY, m);
  return { wi, pdf: pixelPdf(view, i, j, Math.sqrt(Math.max(0, 1 - m[1] * m[1]))), radiance: envMapLookup(view, wi) };
}
