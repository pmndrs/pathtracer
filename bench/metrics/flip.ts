/**
 * LDR-FLIP (Andersson, Nilsson, Akenine-Moller, Oskarsson, Astrom, Fairchild, "FLIP: A Difference Evaluator for
 * Alternating Images", HPG 2020).
 *
 * Ported from our own implementation in ~/Developer/Denoiser/examples/realtime-denoiser/src/flip.ts, which was
 * written after NVIDIA's reference (github.com/NVlabs/flip, BSD-3-Clause, Copyright (c) 2020-2024 NVIDIA CORPORATION
 * & AFFILIATES; attribution in THIRD_PARTY.md, pinned commit d3ab02a). Pipeline, as in the reference:
 *   linear RGB -> YCxCz; per-channel CSF filtering (sums of Gaussians: A, RG, BY); back to linear RGB clamped to
 *   [0,1]; -> L*a*b*, Hunt adjustment; HyAB distance ^ qc (0.7) redistributed with pc = 0.4, pt = 0.95 against
 *   cmax = HyAB(green, blue)^qc; feature (edge / point detector) difference ^ qf (0.5) on normalised Y;
 *   FLIP = dE_c ^ (1 - dE_f) per pixel.
 *
 * What we feed it: linear radiance -> EV gain -> a FIXED tonemap (default ACES fit) -> display-linear [0,1]. The
 * reference decodes sRGB to linear first; our tonemap output is already display-linear, so that step is skipped.
 * Quoted numbers must name the tonemap, EV and ppd. Pixels per degree default to FLIP's standard viewing condition
 * (0.7 m from a 0.7 m wide 3840 px monitor: 67.02 ppd). Borders clamp-to-edge; kernels are separable.
 *
 * Validation (P0.14, see DIVERGES V6): NVlabs flip 1.7 (the pip `flip-evaluator`) also uses separable kernels with
 * clamped borders, so those two are NOT departures from the current reference. The sRGB decode is: `srgbDecode` turns
 * it back on for validation only (input is then 8-bit-style encoded display values, not tonemapped radiance).
 *
 * HDR-FLIP (Andersson et al., "Visualizing Errors in Rendered High Dynamic Range Images", Eurographics 2021; code
 * NVlabs/flip, BSD-3-Clause, same attribution): LDR-FLIP of exposure-scaled, FLIP-tonemapped, clamped images, per-pixel
 * MAX over `numExposures` exposures evenly spaced between start = log2(xMax / Ymax) and stop = log2(xMax / Ymedian),
 * where xMax solves tonemap(x) = 0.85 and Y is the reference's luminance. numExposures = max(2, ceil(stop - start)).
 * Its tonemap is its own (ACES with the 0.6 pre-exposure cancelled, Reinhard on luminance, Hable); it is NOT our
 * display tonemap and there is no EV: the exposure range replaces it.
 */
import type { Image } from '../src/adapter.ts';
import { assertSameSize } from './metrics.ts';
import { tonemapImage, type TonemapName } from './tonemap.ts';

export const DEFAULT_PPD = ((0.7 * 3840) / 0.7) * Math.PI / 180;

// sRGB primaries / D65 (linear RGB -> XYZ), row-major, as in the FLIP reference
const D = 24577794, D2 = 12288897, D3 = 73733382;
const M = new Float64Array([
  10135552 / D, 8788810 / D, 4435075 / D,
  2613072 / D2, 8788810 / D2, 887015 / D2,
  1425312 / D3, 8788810 / D3, 70074185 / D3,
]);
const WX = M[0]! + M[1]! + M[2]!, WY = M[3]! + M[4]! + M[5]!, WZ = M[6]! + M[7]! + M[8]!;
const MI = invert3(M);

function invert3(m: Float64Array): Float64Array {
  const a = m[0]!, b = m[1]!, c = m[2]!, d = m[3]!, e = m[4]!, f = m[5]!, g = m[6]!, h = m[7]!, i = m[8]!;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return new Float64Array([
    A / det, -(b * i - c * h) / det, (b * f - c * e) / det,
    B / det, (a * i - c * g) / det, -(a * f - c * d) / det,
    C / det, -(a * h - b * g) / det, (a * e - b * d) / det,
  ]);
}

const labF = (t: number) => { const d = 6 / 29; return t > d * d * d ? Math.cbrt(t) : t / (3 * d * d) + 4 / 29; };

function linrgbToHuntLab(r: number, g: number, b: number, out: Float32Array, o: number) {
  const x = (M[0]! * r + M[1]! * g + M[2]! * b) / WX;
  const y = (M[3]! * r + M[4]! * g + M[5]! * b) / WY;
  const z = (M[6]! * r + M[7]! * g + M[8]! * b) / WZ;
  const fx = labF(x), fy = labF(y), fz = labF(z);
  const L = 116 * fy - 16;
  out[o] = L;
  out[o + 1] = 0.01 * L * 500 * (fx - fy); // Hunt adjustment
  out[o + 2] = 0.01 * L * 200 * (fy - fz);
}

function csfTaps(b: number, r: number, ppd: number) {
  const t = new Float32Array(2 * r + 1);
  for (let i = -r; i <= r; i++) { const x = i / ppd; t[i + r] = Math.exp(-Math.PI * Math.PI * x * x / b); }
  return t;
}

/**
 * Separable convolution, clamp-to-edge borders. Interior columns skip the clamp; the vertical pass accumulates whole
 * rows (sequential memory access), which is several times faster than the naive per-pixel gather on 4K images.
 */
function convolveSep(src: Float32Array, W: number, H: number, kx: Float32Array, ky: Float32Array, tmp: Float32Array, dst: Float32Array) {
  const rx = (kx.length - 1) >> 1, ry = (ky.length - 1) >> 1;
  const nx = kx.length;
  for (let y = 0; y < H; y++) {
    const row = y * W;
    const lo = Math.min(rx, W), hi = Math.max(W - rx, lo);
    for (let x = 0; x < lo; x++) {
      let s = 0;
      for (let k = -rx; k <= rx; k++) { let xx = x + k; xx = xx < 0 ? 0 : xx >= W ? W - 1 : xx; s += kx[k + rx]! * src[row + xx]!; }
      tmp[row + x] = s;
    }
    for (let x = lo; x < hi; x++) {
      let s = 0;
      const o = row + x - rx;
      for (let k = 0; k < nx; k++) s += kx[k]! * src[o + k]!;
      tmp[row + x] = s;
    }
    for (let x = hi; x < W; x++) {
      let s = 0;
      for (let k = -rx; k <= rx; k++) { let xx = x + k; xx = xx < 0 ? 0 : xx >= W ? W - 1 : xx; s += kx[k + rx]! * src[row + xx]!; }
      tmp[row + x] = s;
    }
  }
  for (let y = 0; y < H; y++) {
    const out = y * W;
    dst.fill(0, out, out + W);
    for (let k = -ry; k <= ry; k++) {
      let yy = y + k; yy = yy < 0 ? 0 : yy >= H ? H - 1 : yy;
      const w = ky[k + ry]!, inp = yy * W;
      for (let x = 0; x < W; x++) dst[out + x] = dst[out + x]! + w * tmp[inp + x]!;
    }
  }
}

interface Kernels {
  a: Float32Array; rg: Float32Array; by1: Float32Array; by2: Float32Array; wBY1: number; wBY2: number;
  gauss: Float32Array; edge: Float32Array; point: Float32Array;
}
const kernelCache = new Map<number, Kernels>();

function kernels(ppd: number): Kernels {
  const hit = kernelCache.get(ppd);
  if (hit) return hit;
  const maxB = 0.04;
  const r = Math.ceil(3 * Math.sqrt(maxB / (2 * Math.PI * Math.PI)) * ppd);
  const term = (a: number, b: number) => {
    const t = csfTaps(b, r, ppd);
    const s = t.reduce((p, v) => p + v, 0);
    return { t, w2d: a * Math.sqrt(Math.PI / b) * s * s, c: a * Math.sqrt(Math.PI / b) };
  };
  const tA = term(1, 0.0047), tRG = term(1, 0.0053);
  const tB1 = term(34.1, 0.04), tB2 = term(13.5, 0.025);
  const sumBY = tB1.w2d + tB2.w2d;
  const normalize1 = (t: Float32Array) => { const s = t.reduce((p, v) => p + v, 0); return t.map((v) => v / s); };

  const w = 0.082;
  const sd = 0.5 * w * ppd;
  const rf = Math.ceil(3 * sd);
  const gauss = new Float32Array(2 * rf + 1), edge = new Float32Array(2 * rf + 1), point = new Float32Array(2 * rf + 1);
  for (let i = -rf; i <= rf; i++) {
    const g = Math.exp(-(i * i) / (2 * sd * sd));
    gauss[i + rf] = g;
    edge[i + rf] = -i * g;
    point[i + rf] = (i * i / (sd * sd) - 1) * g;
  }
  const gs = gauss.reduce((p, v) => p + v, 0);
  const pos = (k: Float32Array) => k.reduce((p, v) => p + (v > 0 ? v : 0), 0) * gs;
  const neg = (k: Float32Array) => -k.reduce((p, v) => p + (v < 0 ? v : 0), 0) * gs;
  const norm2d = (k: Float32Array) => { const P = pos(k), Nn = neg(k); return k.map((v) => (v > 0 ? v / P : v / Nn)); };
  const K: Kernels = {
    a: normalize1(tA.t), rg: normalize1(tRG.t), by1: tB1.t, by2: tB2.t,
    wBY1: tB1.c / sumBY, wBY2: tB2.c / sumBY,
    gauss, edge: norm2d(edge), point: norm2d(point),
  };
  kernelCache.set(ppd, K);
  return K;
}

export interface FlipPrepared {
  W: number; H: number;
  lab: Float32Array;   // Hunt-adjusted L*a*b* of the CSF-filtered image, 3 floats / px
  edge: Float32Array;  // |edge| feature magnitude
  point: Float32Array; // |point| feature magnitude
}

/** Per-image FLIP work (cache it when one image is compared against many). `rgb` is display-linear [0,1], 3 floats/px. */
export function prepareFlip(rgb: Float32Array, W: number, H: number, ppd = DEFAULT_PPD): FlipPrepared {
  const K = kernels(ppd);
  const N = W * H;
  const Y = new Float32Array(N), Cx = new Float32Array(N), Cz = new Float32Array(N), yN = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const r = rgb[3 * i]!, g = rgb[3 * i + 1]!, b = rgb[3 * i + 2]!;
    const x = (M[0]! * r + M[1]! * g + M[2]! * b) / WX;
    const y = (M[3]! * r + M[4]! * g + M[5]! * b) / WY;
    const z = (M[6]! * r + M[7]! * g + M[8]! * b) / WZ;
    Y[i] = 116 * y - 16; Cx[i] = 500 * (x - y); Cz[i] = 200 * (y - z);
    yN[i] = y;
  }
  const tmp = new Float32Array(N);
  const fY = new Float32Array(N), fCx = new Float32Array(N), fCz = new Float32Array(N), t2 = new Float32Array(N);
  convolveSep(Y, W, H, K.a, K.a, tmp, fY);
  convolveSep(Cx, W, H, K.rg, K.rg, tmp, fCx);
  convolveSep(Cz, W, H, K.by1, K.by1, tmp, fCz);
  convolveSep(Cz, W, H, K.by2, K.by2, tmp, t2);
  const lab = new Float32Array(N * 3);
  const cl = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);
  for (let i = 0; i < N; i++) {
    const cz = K.wBY1 * fCz[i]! + K.wBY2 * t2[i]!;
    const y = (fY[i]! + 16) / 116;
    const x = fCx[i]! / 500 + y;
    const z = y - cz / 200;
    const X = x * WX, Yv = y * WY, Z = z * WZ;
    linrgbToHuntLab(
      cl(MI[0]! * X + MI[1]! * Yv + MI[2]! * Z),
      cl(MI[3]! * X + MI[4]! * Yv + MI[5]! * Z),
      cl(MI[6]! * X + MI[7]! * Yv + MI[8]! * Z),
      lab, 3 * i,
    );
  }
  const ex = new Float32Array(N), ey = new Float32Array(N), px = new Float32Array(N), py = new Float32Array(N);
  convolveSep(yN, W, H, K.edge, K.gauss, tmp, ex);
  convolveSep(yN, W, H, K.gauss, K.edge, tmp, ey);
  convolveSep(yN, W, H, K.point, K.gauss, tmp, px);
  convolveSep(yN, W, H, K.gauss, K.point, tmp, py);
  const edge = new Float32Array(N), point = new Float32Array(N);
  for (let i = 0; i < N; i++) { edge[i] = Math.hypot(ex[i]!, ey[i]!); point[i] = Math.hypot(px[i]!, py[i]!); }
  return { W, H, lab, edge, point };
}

const QC = 0.7, QF = 0.5, PC = 0.4, PT = 0.95;
let CMAX = -1;
function cmax() {
  if (CMAX < 0) {
    const g = new Float32Array(3), b = new Float32Array(3);
    linrgbToHuntLab(0, 1, 0, g, 0);
    linrgbToHuntLab(0, 0, 1, b, 0);
    CMAX = Math.pow(Math.abs(g[0]! - b[0]!) + Math.hypot(g[1]! - b[1]!, g[2]! - b[2]!), QC);
  }
  return CMAX;
}

/** Per-pixel FLIP in [0,1] (0 = identical) and its mean. */
export function flipMapPrepared(ref: FlipPrepared, test: FlipPrepared): { map: Float32Array; mean: number } {
  const N = ref.W * ref.H;
  const cm = cmax(), pccmax = PC * cm;
  const map = new Float32Array(N);
  let sum = 0;
  for (let i = 0; i < N; i++) {
    const dL = ref.lab[3 * i]! - test.lab[3 * i]!;
    const da = ref.lab[3 * i + 1]! - test.lab[3 * i + 1]!;
    const db = ref.lab[3 * i + 2]! - test.lab[3 * i + 2]!;
    const hy = Math.pow(Math.abs(dL) + Math.sqrt(da * da + db * db), QC);
    const dc = hy < pccmax ? (PT / pccmax) * hy : PT + ((hy - pccmax) / (cm - pccmax)) * (1 - PT);
    const df = Math.pow(Math.max(Math.abs(ref.edge[i]! - test.edge[i]!), Math.abs(ref.point[i]! - test.point[i]!)) / Math.SQRT2, QF);
    const v = Math.pow(dc, 1 - df);
    map[i] = v;
    sum += v;
  }
  return { map, mean: sum / N };
}

export interface FlipOptions {
  tonemap?: TonemapName;
  ev?: number;
  ppd?: number;
  /** VALIDATION ONLY (V6): treat the tonemapped values as sRGB-encoded and decode them to linear, like the reference does for PNG input. */
  srgbDecode?: boolean;
}

/** sRGB electro-optical transfer (encoded [0,1] -> linear). */
export const srgbDecode = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

/** LDR-FLIP of `test` against `ref` over a fixed tonemap (default ACES, 0 EV, 67 ppd). */
export function flipLdr(test: Image, ref: Image, opts: FlipOptions = {}): { map: Float32Array; mean: number; tonemap: TonemapName; ev: number; ppd: number } {
  assertSameSize(test, ref);
  const tonemap = opts.tonemap ?? 'aces';
  const ev = opts.ev ?? 0;
  const ppd = opts.ppd ?? DEFAULT_PPD;
  const rr = tonemapImage(ref, tonemap, ev), tt = tonemapImage(test, tonemap, ev);
  if (opts.srgbDecode) {
    for (let i = 0; i < rr.length; i++) { rr[i] = srgbDecode(rr[i]!); tt[i] = srgbDecode(tt[i]!); }
  }
  const R = prepareFlip(rr, ref.width, ref.height, ppd);
  const T = prepareFlip(tt, test.width, test.height, ppd);
  const { map, mean } = flipMapPrepared(R, T);
  return { map, mean, tonemap, ev, ppd };
}

// ---- HDR-FLIP -----------------------------------------------------------------------------------------------------

export type HdrTonemapName = 'aces' | 'reinhard' | 'hable';
export const HDR_TONEMAPS: readonly HdrTonemapName[] = ['aces', 'reinhard', 'hable'];
// (a x^2 + b x + c) / (d x^2 + e x + f), as in the reference's ToneMappingCoefficients (Reinhard is luminance-based).
const HDR_TC: Record<HdrTonemapName, readonly number[]> = {
  reinhard: [0, 1, 0, 0, 1, 1],
  aces: [0.6 * 0.6 * 2.51, 0.6 * 0.03, 0, 0.6 * 0.6 * 2.43, 0.6 * 0.59, 0.14], // 0.6 cancels the usual pre-exposure
  hable: [0.231683, 0.013791, 0, 0.18, 0.3, 0.018],
};

const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** Start / stop exposure (stops) from the reference image, as the reference implementation computes them. */
export function hdrFlipExposures(ref: Image, tonemap: HdrTonemapName = 'aces'): { start: number; stop: number; count: number } {
  const tc = HDR_TC[tonemap];
  const t = 0.85;
  const a = tc[0]! - t * tc[3]!, b = tc[1]! - t * tc[4]!, c = tc[2]! - t * tc[5]!;
  let xMax: number;
  if (a === 0) xMax = -c / b;
  else { const d1 = -0.5 * (b / a); xMax = d1 + Math.sqrt(d1 * d1 - c / a); }
  const n = ref.width * ref.height, ch = ref.channels, d = ref.data;
  const Y = new Float32Array(n);
  let ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    let r = d[i * ch]!, g = d[i * ch + 1]!, bl = d[i * ch + 2]!;
    if (!Number.isFinite(r) || r < 0) r = 0;
    if (!Number.isFinite(g) || g < 0) g = 0;
    if (!Number.isFinite(bl) || bl < 0) bl = 0;
    const y = lum(r, g, bl);
    Y[i] = y;
    if (y > ymax) ymax = y;
  }
  Y.sort();
  const median = Math.max(Y[n >> 1]!, 1.1920929e-7); // FLT_EPSILON: more than half the image black
  const start = Math.log2(xMax / Math.max(ymax, 1e-30));
  const stop = Math.log2(xMax / median);
  return { start, stop, count: Math.max(2, Math.ceil(stop - start)) };
}

/** Exposure gain, FLIP tonemap, clamp to [0,1]: display-linear RGB for one exposure (non-finite and negative -> 0). */
export function hdrFlipLdr(img: Image, exposureStops: number, tonemap: HdrTonemapName = 'aces'): Float32Array {
  const n = img.width * img.height, ch = img.channels, d = img.data;
  const out = new Float32Array(n * 3);
  const gain = Math.pow(2, exposureStops);
  const tc = HDR_TC[tonemap];
  const [a, b, c, dd, e, f] = tc as [number, number, number, number, number, number];
  const sat = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
  for (let i = 0; i < n; i++) {
    let r = d[i * ch]!, g = d[i * ch + 1]!, bl = d[i * ch + 2]!;
    r = Number.isFinite(r) && r > 0 ? r * gain : 0;
    g = Number.isFinite(g) && g > 0 ? g * gain : 0;
    bl = Number.isFinite(bl) && bl > 0 ? bl * gain : 0;
    if (tonemap === 'reinhard') {
      const k = 1 / (1 + lum(r, g, bl));
      out[3 * i] = sat(r * k); out[3 * i + 1] = sat(g * k); out[3 * i + 2] = sat(bl * k);
    } else {
      out[3 * i] = sat((r * r * a + r * b + c) / (r * r * dd + r * e + f));
      out[3 * i + 1] = sat((g * g * a + g * b + c) / (g * g * dd + g * e + f));
      out[3 * i + 2] = sat((bl * bl * a + bl * b + c) / (bl * bl * dd + bl * e + f));
    }
  }
  return out;
}

export interface HdrFlipOptions {
  tonemap?: HdrTonemapName;
  ppd?: number;
  /** Override the reference-derived range (stops) and/or count. */
  startExposure?: number;
  stopExposure?: number;
  numExposures?: number;
  /** Called after each exposure (progress / parallel drivers). */
  onExposure?: (index: number, count: number) => void;
}

export interface HdrFlipResult {
  map: Float32Array; mean: number;
  /** Per pixel, the exposure index (0..count-1) that gave the maximum. */
  exposureIndex: Uint8Array;
  start: number; stop: number; count: number; tonemap: HdrTonemapName; ppd: number;
}

/** Plan = the exposure list HDR-FLIP will evaluate (exposed so a parallel driver can split it). */
export function hdrFlipPlan(ref: Image, opts: HdrFlipOptions = {}): { start: number; stop: number; count: number; exposures: number[] } {
  const auto = hdrFlipExposures(ref, opts.tonemap ?? 'aces');
  const start = opts.startExposure ?? auto.start, stop = opts.stopExposure ?? auto.stop;
  if (start > stop) throw new Error('HDR-FLIP: start exposure must not exceed stop exposure');
  const count = opts.numExposures ?? Math.max(2, Math.ceil(stop - start));
  const step = (stop - start) / (count - 1);
  return { start, stop, count, exposures: Array.from({ length: count }, (_, i) => start + i * step) };
}

/** LDR-FLIP map (not the mean) of one exposure; HDR-FLIP is the per-pixel max of these. */
export function hdrFlipExposureMap(test: Image, ref: Image, exposure: number, tonemap: HdrTonemapName, ppd: number): Float32Array {
  const R = prepareFlip(hdrFlipLdr(ref, exposure, tonemap), ref.width, ref.height, ppd);
  const T = prepareFlip(hdrFlipLdr(test, exposure, tonemap), test.width, test.height, ppd);
  return flipMapPrepared(R, T).map;
}

/** Fold per-exposure maps (in exposure order) into the HDR-FLIP result. */
export function foldHdrFlip(maps: Float32Array[]): { map: Float32Array; mean: number; exposureIndex: Uint8Array } {
  const N = maps[0]!.length;
  const map = new Float32Array(N), exposureIndex = new Uint8Array(N);
  for (let k = 0; k < maps.length; k++) {
    const m = maps[k]!;
    for (let i = 0; i < N; i++) if (m[i]! > map[i]!) { map[i] = m[i]!; exposureIndex[i] = k; }
  }
  let sum = 0;
  for (let i = 0; i < N; i++) sum += map[i]!;
  return { map, mean: sum / N, exposureIndex };
}

/** HDR-FLIP of `test` against `ref` (linear radiance, no tonemap/EV choice: the exposure range is derived from `ref`). */
export function flipHdr(test: Image, ref: Image, opts: HdrFlipOptions = {}): HdrFlipResult {
  assertSameSize(test, ref);
  const tonemap = opts.tonemap ?? 'aces';
  const ppd = opts.ppd ?? DEFAULT_PPD;
  const plan = hdrFlipPlan(ref, opts);
  const N = ref.width * ref.height;
  const map = new Float32Array(N), exposureIndex = new Uint8Array(N);
  for (let k = 0; k < plan.count; k++) {
    const m = hdrFlipExposureMap(test, ref, plan.exposures[k]!, tonemap, ppd);
    for (let i = 0; i < N; i++) if (m[i]! > map[i]!) { map[i] = m[i]!; exposureIndex[i] = k; }
    opts.onExposure?.(k, plan.count);
  }
  let sum = 0;
  for (let i = 0; i < N; i++) sum += map[i]!;
  return { map, mean: sum / N, exposureIndex, start: plan.start, stop: plan.stop, count: plan.count, tonemap, ppd };
}
