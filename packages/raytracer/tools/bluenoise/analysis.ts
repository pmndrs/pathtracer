/** Spectral and histogram analysis used by the unit tests and the report. Plain JS, no dependencies. */

/** In-place radix-2 complex FFT (length must be a power of two). */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]!; re[i] = re[j]!; re[j] = tr;
      const ti = im[i]!; im[i] = im[j]!; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const xr = re[b]! * cr - im[b]! * ci;
        const xi = re[b]! * ci + im[b]! * cr;
        re[b] = re[a]! - xr;
        im[b] = im[a]! - xi;
        re[a] = re[a]! + xr;
        im[a] = im[a]! + xi;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

/** 2D FFT power spectrum |F|^2 / (W*H) of a real image, DC at index 0 (unshifted). */
export function powerSpectrum2D(img: ArrayLike<number>, w: number, h: number, subtractMean = true): Float64Array {
  let mean = 0;
  if (subtractMean) {
    for (let i = 0; i < w * h; i++) mean += img[i]!;
    mean /= w * h;
  }
  const re = new Float64Array(w * h);
  const im = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) re[i] = img[i]! - mean;
  const rr = new Float64Array(w);
  const ri = new Float64Array(w);
  for (let y = 0; y < h; y++) {
    rr.set(re.subarray(y * w, (y + 1) * w));
    ri.fill(0);
    fft(rr, ri);
    re.set(rr, y * w);
    im.set(ri, y * w);
  }
  const cr = new Float64Array(h);
  const ci = new Float64Array(h);
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      cr[y] = re[y * w + x]!;
      ci[y] = im[y * w + x]!;
    }
    fft(cr, ci);
    for (let y = 0; y < h; y++) {
      re[y * w + x] = cr[y]!;
      im[y * w + x] = ci[y]!;
    }
  }
  const p = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) p[i] = (re[i]! * re[i]! + im[i]! * im[i]!) / (w * h);
  return p;
}

/**
 * Radially averaged power spectrum. Bin b covers radial frequency [b, b+1) cycles per image (so 1/size cycles per
 * pixel per bin), b = 1 .. size/2 - 1 (DC excluded, corners beyond Nyquist radius dropped). Returns mean power per bin.
 */
export function radialAverage(power: Float64Array, w: number, h: number): Float64Array {
  const nb = Math.min(w, h) / 2;
  const sum = new Float64Array(nb);
  const cnt = new Float64Array(nb);
  for (let y = 0; y < h; y++) {
    const fy = y <= h / 2 ? y : y - h;
    for (let x = 0; x < w; x++) {
      const fx = x <= w / 2 ? x : x - w;
      const r = Math.hypot(fx * (nb * 2 / w), fy * (nb * 2 / h));
      const b = Math.floor(r);
      if (b >= 1 && b < nb) {
        sum[b]! += power[y * w + x]!;
        cnt[b]! += 1;
      }
    }
  }
  const out = new Float64Array(nb);
  for (let b = 1; b < nb; b++) out[b] = cnt[b]! > 0 ? sum[b]! / cnt[b]! : 0;
  return out;
}

export function variance(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m += a[i]!;
  m /= a.length;
  let v = 0;
  for (let i = 0; i < a.length; i++) v += (a[i]! - m) ** 2;
  return v / a.length;
}

/** Mean of the radial spectrum (normalised by `norm`) over bins [lo, hi). */
export function bandMean(raps: Float64Array, lo: number, hi: number, norm = 1): number {
  let s = 0;
  let c = 0;
  for (let b = Math.max(1, lo); b < Math.min(hi, raps.length); b++) { s += raps[b]!; c++; }
  return c ? s / c / norm : 0;
}

/** Average RAPS over the z-slices of a volume (values layout z-major), normalised so white noise reads 1. */
export function sliceSpectrum(values: Float32Array, w: number, h: number, d: number): Float64Array {
  let acc: Float64Array | null = null;
  const v = variance(values);
  for (let z = 0; z < d; z++) {
    const r = radialAverage(powerSpectrum2D(values.subarray(z * w * h, (z + 1) * w * h), w, h), w, h);
    if (!acc) acc = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) acc[i]! += r[i]! / d / v;
  }
  return acc!;
}

/** Per-pixel temporal power spectrum, averaged over all pixels, normalised so white noise reads 1. Index 0 = DC (dropped to 0). */
export function temporalSpectrum(values: Float32Array, w: number, h: number, d: number): Float64Array {
  const out = new Float64Array(d / 2 + 1);
  const re = new Float64Array(d);
  const im = new Float64Array(d);
  const v = variance(values);
  const pix = w * h;
  for (let p = 0; p < pix; p++) {
    let m = 0;
    for (let z = 0; z < d; z++) m += values[z * pix + p]!;
    m /= d;
    for (let z = 0; z < d; z++) { re[z] = values[z * pix + p]! - m; im[z] = 0; }
    fft(re, im);
    for (let k = 1; k <= d / 2; k++) out[k]! += (re[k]! ** 2 + im[k]! ** 2) / d / pix / v;
  }
  return out;
}

/** Kolmogorov-Smirnov distance between the empirical distribution of `a` and U[0,1]. */
export function ksUniform(a: ArrayLike<number>): number {
  const s = Float64Array.from(a as ArrayLike<number>).sort();
  let d = 0;
  for (let i = 0; i < s.length; i++) d = Math.max(d, Math.abs((i + 1) / s.length - s[i]!), Math.abs(i / s.length - s[i]!));
  return d;
}

/** True when `ranks` is a permutation of 0..N-1. */
export function isPermutation(ranks: ArrayLike<number>): boolean {
  const seen = new Uint8Array(ranks.length);
  for (let i = 0; i < ranks.length; i++) {
    const r = ranks[i]!;
    if (r < 0 || r >= ranks.length || seen[r]) return false;
    seen[r] = 1;
  }
  return true;
}
