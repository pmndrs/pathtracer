/**
 * RGB -> spectrum uplift (P5, spectral mode): the sigmoid-polynomial spectra of Jakob and Hanika, "A Low-Dimensional Function Space for Efficient
 * Spectral Upsampling" (EGSR 2019), as pbrt-v4 uses them (`RGBToSpectrumTable`, Apache-2.0, THIRD_PARTY): a reflectance spectrum is
 * `s(t) = sigmoid(c0 t^2 + c1 t + c2)` with `t = (lambda - 360) / 470` and `sigmoid(x) = 1/2 + x / (2 sqrt(1 + x^2))`; a table of `(c0, c1, c2)` over the
 * linear Rec.709 cube is fitted by Gauss-Newton so that the spectrum lit by D65 reproduces the RGB (CIELAB residual, as the paper). This file is our own
 * implementation of the method from the paper (no code of the reference optimiser or of pbrt is copied); the table layout and lookup follow pbrt's so a
 * colour maps to the same spectrum up to the table resolution.
 *
 * Conventions (one place, shared by the kernel, `wavelengths.ts` and every check):
 *  - XYZ of a spectrum S lit by D65: `XYZ = integral S(l) D65(l) cmf(l) dl / Yd`, `Yd = integral D65 ybar dl`, so S = 1 gives the D65 white (Y = 1).
 *  - `RGB = MNORM XYZ`, the Rec.709 matrix with each row scaled so the table's D65 white is exactly (1, 1, 1): a constant spectrum c is the grey (c, c, c).
 *  - The table is indexed `[max component m][z node k][y][x]` with `rgb[m] = z`, `rgb[m+1] = x z`, `rgb[m+2] = y z` (indices mod 3), x and y on `i / (res-1)`,
 *    z on the nodes `smoothstep(smoothstep(k / (res-1)))` (they crowd toward 0 and 1, where the coefficients change fastest).
 */
import { CIE_1931_D65 } from './cie1931.ts';

export const SPEC_LAMBDA_MIN = 360;
export const SPEC_LAMBDA_MAX = 830;
const SPAN = SPEC_LAMBDA_MAX - SPEC_LAMBDA_MIN;
export type Vec3 = [number, number, number];

/** XYZ -> linear Rec.709 (sRGB primaries, D65). */
const XYZ_TO_REC709 = [3.2404542, -1.5371385, -0.4985314, -0.969266, 1.8760108, 0.041556, 0.0556434, -0.2040259, 1.0572252] as const;

const N = CIE_1931_D65.length;
const STEP = 5;

/** Linear interpolation of the 5 nm table at `lambda` nm; zero outside [360, 780]. Returns [xbar, ybar, zbar, D65]. */
export function cieAt(lambda: number): [number, number, number, number] {
  const f = (lambda - 360) / STEP;
  if (!(f >= 0) || f > N - 1) return [0, 0, 0, 0];
  const i = Math.min(Math.floor(f), N - 2);
  const t = f - i;
  const a = CIE_1931_D65[i] as readonly number[];
  const b = CIE_1931_D65[i + 1] as readonly number[];
  return [0, 1, 2, 3].map((k) => (a[k + 1] as number) * (1 - t) + (b[k + 1] as number) * t) as [number, number, number, number];
}

/** `Yd` and the D65 white XYZ (Y = 1) of the linearly interpolated tables, integrated at 0.05 nm. */
function whiteAndNorm(): { yd: number; white: Vec3; mnorm: number[] } {
  let yd = 0;
  const w: Vec3 = [0, 0, 0];
  const dl = 0.05;
  for (let l = 360 + dl / 2; l < 780; l += dl) {
    const [x, y, z, d] = cieAt(l);
    yd += d * y * dl;
    w[0] += d * x * dl;
    w[1] += d * y * dl;
    w[2] += d * z * dl;
  }
  const white: Vec3 = [w[0] / yd, 1, w[2] / yd];
  const mnorm: number[] = [];
  for (let r = 0; r < 3; r++) {
    const s = (XYZ_TO_REC709[r * 3] as number) * white[0] + (XYZ_TO_REC709[r * 3 + 1] as number) * white[1] + (XYZ_TO_REC709[r * 3 + 2] as number) * white[2];
    for (let c = 0; c < 3; c++) mnorm.push((XYZ_TO_REC709[r * 3 + c] as number) / s);
  }
  return { yd, white, mnorm };
}

const BASE = whiteAndNorm();
/** `Yd`: integral of D65 * ybar (the D65 table is divided by it so a white emitter has Y = 1). */
export const SPEC_YD = BASE.yd;
/** D65 white as XYZ with Y = 1. */
export const SPEC_WHITE_XYZ = BASE.white;
/** Row-major 3x3, XYZ -> RGB with the D65 white mapped to exactly (1, 1, 1). */
export const SPEC_XYZ_TO_RGB: readonly number[] = BASE.mnorm;

function invert3(m: readonly number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m as [number, number, number, number, number, number, number, number, number];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((v) => v / det);
}
/** Row-major 3x3, RGB -> XYZ (inverse of `SPEC_XYZ_TO_RGB`). */
export const SPEC_RGB_TO_XYZ: readonly number[] = invert3(BASE.mnorm);

export function xyzToRgb(xyz: readonly number[]): Vec3 {
  const m = SPEC_XYZ_TO_RGB;
  return [0, 1, 2].map((r) => (m[r * 3] as number) * (xyz[0] as number) + (m[r * 3 + 1] as number) * (xyz[1] as number) + (m[r * 3 + 2] as number) * (xyz[2] as number)) as Vec3;
}

/** `sigmoid(c0 t^2 + c1 t + c2)` at `lambda` nm. */
export function sigmoidSpectrum(c: readonly number[], lambda: number): number {
  const t = (lambda - SPEC_LAMBDA_MIN) / SPAN;
  const x = ((c[0] as number) * t + (c[1] as number)) * t + (c[2] as number);
  if (!Number.isFinite(x)) return x > 0 ? 1 : 0;
  return 0.5 + x / (2 * Math.sqrt(1 + x * x));
}

/** Linear Rec.709 RGB of a reflectance spectrum under D65 (integrated at 1 nm). */
export function reflectanceToRgb(s: (lambda: number) => number): Vec3 {
  const xyz: Vec3 = [0, 0, 0];
  for (let l = 360.5; l < 780; l += 1) {
    const [x, y, z, d] = cieAt(l);
    const v = s(l) * d;
    xyz[0] += v * x;
    xyz[1] += v * y;
    xyz[2] += v * z;
  }
  return xyzToRgb([xyz[0] / SPEC_YD, xyz[1] / SPEC_YD, xyz[2] / SPEC_YD]);
}

const smooth = (x: number): number => x * x * (3 - 2 * x);
/** The z node values of a table with `res` nodes per axis. */
export function specZNodes(res: number): Float64Array {
  const z = new Float64Array(res);
  for (let k = 0; k < res; k++) z[k] = smooth(smooth(k / (res - 1)));
  return z;
}

export interface Rgb2SpecTable {
  res: number;
  /** `[3][res][res][res][3]` f32: max component, z node, y, x, coefficient. */
  data: Float32Array;
}

// ---- the optimiser ----

/** Per-sample weights of the 5 nm quadrature: xyz weight (cmf * D65 * 5 / Yd) and the normalised wavelength t. */
const WX = new Float64Array(N), WY = new Float64Array(N), WZ = new Float64Array(N), TT = new Float64Array(N);
for (let i = 0; i < N; i++) {
  const row = CIE_1931_D65[i] as readonly number[];
  const k = ((row[4] as number) * STEP) / SPEC_YD;
  WX[i] = (row[1] as number) * k;
  WY[i] = (row[2] as number) * k;
  WZ[i] = (row[3] as number) * k;
  TT[i] = ((row[0] as number) - SPEC_LAMBDA_MIN) / SPAN;
}
// The 5 nm sum of a constant spectrum is not exactly the white of the 0.05 nm integral: scale the weights so S = 1 gives SPEC_WHITE_XYZ.
{
  let sx = 0, sy = 0, sz = 0;
  for (let i = 0; i < N; i++) { sx += WX[i] as number; sy += WY[i] as number; sz += WZ[i] as number; }
  for (let i = 0; i < N; i++) { WX[i] = (WX[i] as number) * (SPEC_WHITE_XYZ[0] / sx); WY[i] = (WY[i] as number) / sy; WZ[i] = (WZ[i] as number) * (SPEC_WHITE_XYZ[2] / sz); }
}

const labF = (t: number): number => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
function lab(x: number, y: number, z: number, out: Float64Array, o: number): void {
  const fx = labF(x / SPEC_WHITE_XYZ[0]), fy = labF(y), fz = labF(z / SPEC_WHITE_XYZ[2]);
  out[o] = 116 * fy - 16;
  out[o + 1] = 500 * (fx - fy);
  out[o + 2] = 200 * (fy - fz);
}

function specLab(c0: number, c1: number, c2: number, out: Float64Array, o: number): void {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < N; i++) {
    const t = TT[i] as number;
    const v = (c0 * t + c1) * t + c2;
    const s = 0.5 + v / (2 * Math.sqrt(1 + v * v));
    x += s * (WX[i] as number);
    y += s * (WY[i] as number);
    z += s * (WZ[i] as number);
  }
  lab(x, y, z, out, o);
}

/** Gauss-Newton on one RGB target; `c` is the start and the result. Returns the final Lab residual norm. */
function optimise(target: Vec3, c: Float64Array): number {
  const m = SPEC_RGB_TO_XYZ;
  const xt = (m[0] as number) * target[0] + (m[1] as number) * target[1] + (m[2] as number) * target[2];
  const yt = (m[3] as number) * target[0] + (m[4] as number) * target[1] + (m[5] as number) * target[2];
  const zt = (m[6] as number) * target[0] + (m[7] as number) * target[1] + (m[8] as number) * target[2];
  const tl = new Float64Array(3);
  lab(xt, yt, zt, tl, 0);
  const r = new Float64Array(3), l0 = new Float64Array(3), l1 = new Float64Array(3);
  const J = new Float64Array(9);
  let rn = 1e30;
  for (let it = 0; it < 40; it++) {
    specLab(c[0] as number, c[1] as number, c[2] as number, l0, 0);
    for (let k = 0; k < 3; k++) r[k] = (l0[k] as number) - (tl[k] as number);
    rn = Math.hypot(r[0] as number, r[1] as number, r[2] as number);
    if (rn < 1e-6) break;
    const eps = 1e-4;
    for (let j = 0; j < 3; j++) {
      const t = [c[0] as number, c[1] as number, c[2] as number];
      t[j] = (t[j] as number) + eps;
      specLab(t[0] as number, t[1] as number, t[2] as number, l1, 0);
      for (let k = 0; k < 3; k++) J[k * 3 + j] = ((l1[k] as number) - (l0[k] as number)) / eps;
    }
    // solve J d = r by Gaussian elimination with partial pivoting
    const a = [[J[0], J[1], J[2], r[0]], [J[3], J[4], J[5], r[1]], [J[6], J[7], J[8], r[2]]] as number[][];
    let singular = false;
    for (let col = 0; col < 3; col++) {
      let p = col;
      for (let row = col + 1; row < 3; row++) if (Math.abs((a[row] as number[])[col] as number) > Math.abs((a[p] as number[])[col] as number)) p = row;
      const tmp = a[col] as number[];
      a[col] = a[p] as number[];
      a[p] = tmp;
      const piv = (a[col] as number[])[col] as number;
      if (Math.abs(piv) < 1e-15) { singular = true; break; }
      for (let row = col + 1; row < 3; row++) {
        const f = ((a[row] as number[])[col] as number) / piv;
        for (let k = col; k < 4; k++) (a[row] as number[])[k] = ((a[row] as number[])[k] as number) - f * ((a[col] as number[])[k] as number);
      }
    }
    if (singular) break;
    const d = [0, 0, 0];
    for (let row = 2; row >= 0; row--) {
      let s = (a[row] as number[])[3] as number;
      for (let k = row + 1; k < 3; k++) s -= ((a[row] as number[])[k] as number) * (d[k] as number);
      d[row] = s / ((a[row] as number[])[row] as number);
    }
    // damped step: halve it until the residual no longer grows
    const c0 = c[0] as number, c1 = c[1] as number, c2 = c[2] as number;
    let step = 1;
    for (let tries = 0; tries < 8; tries++, step *= 0.5) {
      c[0] = c0 - step * (d[0] as number);
      c[1] = c1 - step * (d[1] as number);
      c[2] = c2 - step * (d[2] as number);
      const big = Math.max(Math.abs(c[0] as number), Math.abs(c[1] as number), Math.abs(c[2] as number));
      if (big > 200) for (let k = 0; k < 3; k++) c[k] = (c[k] as number) * (200 / big);
      specLab(c[0] as number, c[1] as number, c[2] as number, l1, 0);
      if (Math.hypot((l1[0] as number) - (tl[0] as number), (l1[1] as number) - (tl[1] as number), (l1[2] as number) - (tl[2] as number)) < rn) break;
    }
  }
  return rn;
}

/**
 * Fits the table (about 3 x res^3 Gauss-Newton solves: res 16 takes ~1 s, 32 ~8 s in Node). The fit sweeps the z axis from `res / 5` up and down,
 * starting each cell from its neighbour's solution. `onProgress(done in [0, 1])` is called once per max-component slice.
 */
export function buildRgb2SpecTable(res = 32, onProgress?: (done: number) => void): Rgb2SpecTable {
  if (!(res >= 10) || res > 128) throw new RangeError(`rgb2spec: res must be 10..128 (the CMF row needs res^2 >= ${N}), got ${res}`);
  const data = new Float32Array(3 * res * res * res * 3);
  const nodes = specZNodes(res);
  const c = new Float64Array(3);
  const idx = (l: number, k: number, j: number, i: number): number => (((l * res + k) * res + j) * res + i) * 3;
  const start = Math.floor(res / 5);
  for (let l = 0; l < 3; l++) {
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const x = i / (res - 1), y = j / (res - 1);
        const solve = (k: number): void => {
          const z = nodes[k] as number;
          const rgb: Vec3 = [0, 0, 0];
          rgb[l] = z;
          rgb[(l + 1) % 3] = x * z;
          rgb[(l + 2) % 3] = y * z;
          optimise(rgb, c);
          const o = idx(l, k, j, i);
          data[o] = c[0] as number;
          data[o + 1] = c[1] as number;
          data[o + 2] = c[2] as number;
        };
        if (i > 0) { const o = idx(l, start, j, i - 1); c[0] = data[o] as number; c[1] = data[o + 1] as number; c[2] = data[o + 2] as number; }
        else if (j > 0) { const o = idx(l, start, j - 1, 0); c[0] = data[o] as number; c[1] = data[o + 1] as number; c[2] = data[o + 2] as number; }
        else c[0] = c[1] = c[2] = 0;
        for (let k = start; k < res; k++) solve(k);
        // re-seed from the first solved node going down
        const o0 = idx(l, start, j, i);
        c[0] = data[o0] as number; c[1] = data[o0 + 1] as number; c[2] = data[o0 + 2] as number;
        for (let k = start - 1; k >= 1; k--) solve(k);
        // node 0 is black: a spectrum that is 0 everywhere
        const ob = idx(l, 0, j, i);
        data[ob] = 0; data[ob + 1] = 0; data[ob + 2] = -50;
      }
    }
    onProgress?.((l + 1) / 3);
  }
  return { res, data };
}

const smoothInv = (y: number): number => 0.5 - Math.sin(Math.asin(1 - 2 * y) / 3);

/** pbrt's `RGBToSpectrumTable::operator()`: sigmoid coefficients of a reflectance RGB in [0, 1]. TS twin of the kernel's `ptSpecCoeffs`. */
export function rgbToCoeffs(table: Rgb2SpecTable, rgbIn: readonly number[]): Vec3 {
  const rgb = rgbIn.map((v) => Math.min(1, Math.max(0, v))) as Vec3;
  if (rgb[0] === rgb[1] && rgb[1] === rgb[2]) return [0, 0, (rgb[0] - 0.5) / Math.sqrt(Math.max(rgb[0] * (1 - rgb[0]), 2.5e-9))];
  const m = rgb[0] >= rgb[1] && rgb[0] >= rgb[2] ? 0 : rgb[1] >= rgb[2] ? 1 : 2;
  const z = rgb[m] as number;
  const res = table.res;
  const x = ((rgb[(m + 1) % 3] as number) * (res - 1)) / z;
  const y = ((rgb[(m + 2) % 3] as number) * (res - 1)) / z;
  const kf = (res - 1) * smoothInv(smoothInv(z));
  const zi = Math.min(Math.max(Math.floor(kf), 0), res - 2);
  const z0 = smooth(smooth(zi / (res - 1))), z1 = smooth(smooth((zi + 1) / (res - 1)));
  const dz = Math.min(1, Math.max(0, (z - z0) / (z1 - z0)));
  const xi = Math.min(Math.floor(x), res - 2), yi = Math.min(Math.floor(y), res - 2);
  const dx = x - xi, dy = y - yi;
  const at = (k: number, j: number, i: number, ch: number): number => table.data[(((m * res + k) * res + j) * res + i) * 3 + ch] as number;
  const out: Vec3 = [0, 0, 0];
  for (let ch = 0; ch < 3; ch++) {
    const lerp = (a: number, b: number, t: number): number => a * (1 - t) + b * t;
    const plane = (k: number): number => lerp(lerp(at(k, yi, xi, ch), at(k, yi, xi + 1, ch), dx), lerp(at(k, yi + 1, xi, ch), at(k, yi + 1, xi + 1, ch), dx), dy);
    out[ch] = lerp(plane(zi), plane(zi + 1), dz);
  }
  return out;
}

const cached = new Map<number, Rgb2SpecTable>();
/** The table of resolution `res` (default 64, pbrt's), fitted on first use and kept: res 32 takes ~0.5 s in Node, res 64 ~4 s. A page should fit it once and pass it on. */
export function defaultRgb2SpecTable(res = 64): Rgb2SpecTable {
  let t = cached.get(res);
  if (!t) {
    t = buildRgb2SpecTable(res);
    cached.set(res, t);
  }
  return t;
}

/**
 * The kernel's texture (`rgba32float`, `width = res^2`, `height = 3 res + 1`): texel `(x + y res, m res + k)` = (c0, c1, c2, 0); the last row holds the 5 nm table:
 * texel i = (xbar, ybar, zbar, D65 / Yd) at 360 + 5 i nm.
 */
export function rgb2SpecTexture(table: Rgb2SpecTable): { width: number; height: number; data: Float32Array } {
  const res = table.res;
  const width = res * res, height = 3 * res + 1;
  const data = new Float32Array(width * height * 4);
  for (let m = 0; m < 3; m++) for (let k = 0; k < res; k++) for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const src = (((m * res + k) * res + j) * res + i) * 3;
    const dst = ((m * res + k) * width + i + j * res) * 4;
    data[dst] = table.data[src] as number;
    data[dst + 1] = table.data[src + 1] as number;
    data[dst + 2] = table.data[src + 2] as number;
  }
  for (let i = 0; i < N; i++) {
    const row = CIE_1931_D65[i] as readonly number[];
    const dst = (3 * res * width + i) * 4;
    data[dst] = row[1] as number;
    data[dst + 1] = row[2] as number;
    data[dst + 2] = row[3] as number;
    data[dst + 3] = (row[4] as number) / SPEC_YD;
  }
  return { width, height, data };
}
