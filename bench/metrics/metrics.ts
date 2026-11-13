/**
 * Image-vs-reference error metrics on linear HDR float images (row 0 = top, 3 or 4 channels, alpha ignored).
 *
 * Conventions (state these next to any number you quote):
 *   mse      mean over pixels and the 3 channels of (t - r)^2, linear radiance.
 *   relMse   mean over pixels and channels of (t - r)^2 / (r^2 + eps), eps = 1e-2 (Rousselle / Vogels style).
 *   smape    mean of |t - r| / (|t| + |r| + eps), eps = 1e-2; range [0, 1) (0/0 counts as 0).
 *   psnr     10 log10(1 / mse_d) where mse_d is the MSE of display values: tonemap(EV) then sRGB OETF, peak 1.
 *            Default tonemap 'aces', 0 EV. Infinity for identical images.
 *   bias     mean signed (t - r) per channel and averaged over channels; relBias = mean(t) / mean(r) - 1.
 *   firefly  % of pixels whose test luminance > factor (10) x max(reference luminance, floor 1e-2).
 * Pixels where the test or the reference has a non-finite channel are EXCLUDED from every mean and counted in
 * nonFiniteTest / nonFiniteRef (a NaN must not silently vanish: check those counts).
 * Metrics are asymmetric: pass the reference second. Argument order is (test, reference).
 */
import type { Image } from '../src/adapter.ts';
import { srgbEncode, tonemapPixel, type TonemapName } from './tonemap.ts';

export const REL_MSE_EPS = 1e-2;
export const SMAPE_EPS = 1e-2;
export const FIREFLY_FACTOR = 10;
export const FIREFLY_FLOOR = 1e-2;

export interface MetricOptions {
  eps?: number;
  smapeEps?: number;
  psnrTonemap?: TonemapName;
  psnrEv?: number;
  fireflyFactor?: number;
  fireflyFloor?: number;
}

export interface PairMetrics {
  width: number;
  height: number;
  /** Pixels that entered the means. */
  pixels: number;
  nonFiniteTest: number;
  nonFiniteRef: number;
  mse: number;
  relMse: number;
  smape: number;
  psnr: number;
  psnrTonemap: TonemapName;
  psnrEv: number;
  /** Mean signed error per channel (test - ref). */
  biasRGB: [number, number, number];
  bias: number;
  relBias: number;
  fireflyPct: number;
  /** Mean LDR-FLIP, filled in by flip.ts (compareImages leaves it undefined). */
  flip?: number;
}

export function assertSameSize(a: Image, b: Image): void {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(`image size mismatch: ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  }
}

const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

export function compareImages(test: Image, ref: Image, opts: MetricOptions = {}): PairMetrics {
  assertSameSize(test, ref);
  const eps = opts.eps ?? REL_MSE_EPS;
  const seps = opts.smapeEps ?? SMAPE_EPS;
  const tm = opts.psnrTonemap ?? 'aces';
  const ev = opts.psnrEv ?? 0;
  const gain = Math.pow(2, ev);
  const fFactor = opts.fireflyFactor ?? FIREFLY_FACTOR;
  const fFloor = opts.fireflyFloor ?? FIREFLY_FLOOR;
  const n = test.width * test.height;
  const tc = test.channels, rc = ref.channels;
  const t = test.data, r = ref.data;
  let mse = 0, rel = 0, smape = 0, dmse = 0, fire = 0, used = 0, nfT = 0, nfR = 0;
  let sumT = 0, sumR = 0;
  const bias = [0, 0, 0];
  const dt = [0, 0, 0], dr = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const to = i * tc, ro = i * rc;
    const t0 = t[to]!, t1 = t[to + 1]!, t2 = t[to + 2]!;
    const r0 = r[ro]!, r1 = r[ro + 1]!, r2 = r[ro + 2]!;
    const tOk = Number.isFinite(t0 + t1 + t2);
    const rOk = Number.isFinite(r0 + r1 + r2);
    if (!tOk) nfT++;
    if (!rOk) nfR++;
    if (!tOk || !rOk) continue;
    used++;
    for (let c = 0; c < 3; c++) {
      const a = c === 0 ? t0 : c === 1 ? t1 : t2;
      const b = c === 0 ? r0 : c === 1 ? r1 : r2;
      const d = a - b;
      mse += d * d;
      rel += (d * d) / (b * b + eps);
      const den = Math.abs(a) + Math.abs(b) + seps;
      smape += Math.abs(d) / den;
      bias[c]! += d;
      sumT += a; sumR += b;
    }
    tonemapPixel(tm, t0 * gain, t1 * gain, t2 * gain, dt);
    tonemapPixel(tm, r0 * gain, r1 * gain, r2 * gain, dr);
    for (let c = 0; c < 3; c++) {
      const d = srgbEncode(dt[c]!) - srgbEncode(dr[c]!);
      dmse += d * d;
    }
    if (lum(t0, t1, t2) > fFactor * Math.max(lum(r0, r1, r2), fFloor)) fire++;
  }
  const nn = Math.max(used, 1);
  const m3 = nn * 3;
  const dm = dmse / m3;
  const biasRGB: [number, number, number] = [bias[0]! / nn, bias[1]! / nn, bias[2]! / nn];
  return {
    width: test.width, height: test.height, pixels: used, nonFiniteTest: nfT, nonFiniteRef: nfR,
    mse: mse / m3, relMse: rel / m3, smape: smape / m3,
    psnr: dm === 0 ? Infinity : -10 * Math.log10(dm), psnrTonemap: tm, psnrEv: ev,
    biasRGB, bias: (biasRGB[0] + biasRGB[1] + biasRGB[2]) / 3,
    relBias: sumR === 0 ? NaN : sumT / sumR - 1,
    fireflyPct: (100 * fire) / nn,
  };
}

// ---- per-pixel maps (row 0 = top, same size as the inputs) ---------------------------------------------------

/** |t - r| per channel, RGB. */
export function absDiffMap(test: Image, ref: Image): Image {
  assertSameSize(test, ref);
  const n = test.width * test.height;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) out[i * 3 + c] = Math.abs(test.data[i * test.channels + c]! - ref.data[i * ref.channels + c]!);
  }
  return { width: test.width, height: test.height, channels: 3, data: out };
}

/** Signed t - r per channel, RGB: the per-pixel bias map. */
export function biasMap(test: Image, ref: Image): Image {
  assertSameSize(test, ref);
  const n = test.width * test.height;
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) out[i * 3 + c] = test.data[i * test.channels + c]! - ref.data[i * ref.channels + c]!;
  }
  return { width: test.width, height: test.height, channels: 3, data: out };
}

/** Per-pixel relMSE term (mean over channels); its mean equals `relMse`. 1 float / pixel. */
export function relMseMap(test: Image, ref: Image, eps = REL_MSE_EPS): Float32Array {
  assertSameSize(test, ref);
  const n = test.width * test.height;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < 3; c++) {
      const a = test.data[i * test.channels + c]!, b = ref.data[i * ref.channels + c]!;
      s += ((a - b) * (a - b)) / (b * b + eps);
    }
    out[i] = s / 3;
  }
  return out;
}

/** Percentile (0..100) of a numeric array, nearest-rank on a sorted copy. NaNs are dropped. */
export function percentile(values: ArrayLike<number>, p: number): number {
  const a = Float64Array.from(values as ArrayLike<number>).filter((v) => !Number.isNaN(v)).sort();
  if (a.length === 0) return NaN;
  const k = Math.min(a.length - 1, Math.max(0, Math.ceil((p / 100) * a.length) - 1));
  return a[k]!;
}
