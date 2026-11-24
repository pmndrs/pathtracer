/**
 * V7.4 metrics that are not in `metrics/metrics.ts` (pure TS, unit-tested): the edge mask of V7.3 (`bench/split-denoise.ts`) on images of any channel count, a masked
 * relMSE, and the temporal-excess stability of a moving sequence.
 */
export interface MImg {
  width: number;
  height: number;
  channels: number;
  data: Float32Array;
}

export const EDGE_GRAD = 0.25;
export const EDGE_R = 3;

const lumAt = (img: MImg, p: number): number => 0.2126 * (img.data[p * img.channels] as number) + 0.7152 * (img.data[p * img.channels + 1] as number) + 0.0722 * (img.data[p * img.channels + 2] as number);

/** Reference pixels within `EDGE_R` px (Chebyshev) of a strong |grad log(L + 0.05)| (> `EDGE_GRAD`). */
export function edgeMask(ref: MImg): { mask: Uint8Array; fraction: number } {
  const { width: W, height: H } = ref;
  const L = new Float32Array(W * H);
  for (let p = 0; p < W * H; p++) L[p] = Math.log(Math.max(lumAt(ref, p), 0) + 0.05);
  const strong = new Uint8Array(W * H);
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) {
      const g = Math.hypot((L[y * W + x + 1] as number) - (L[y * W + x - 1] as number), (L[(y + 1) * W + x] as number) - (L[(y - 1) * W + x] as number)) / 2;
      if (g > EDGE_GRAD) strong[y * W + x] = 1;
    }
  const mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      if (!strong[y * W + x]) continue;
      for (let j = -EDGE_R; j <= EDGE_R; j++)
        for (let i = -EDGE_R; i <= EDGE_R; i++) {
          const xx = x + i;
          const yy = y + j;
          if (xx >= 0 && yy >= 0 && xx < W && yy < H) mask[yy * W + xx] = 1;
        }
    }
  let n = 0;
  for (const m of mask) n += m;
  return { mask, fraction: n / (W * H) };
}

/** relMSE (eps 1e-2, as `compareImages`) over the mask pixels (`inside`) or the others; non-finite pixels skipped. */
export function maskedRelMse(test: MImg, ref: MImg, mask: Uint8Array, inside = true): number {
  let s = 0;
  let n = 0;
  for (let p = 0; p < mask.length; p++) {
    if ((mask[p] === 1) !== inside) continue;
    let e = 0;
    let ok = true;
    for (let c = 0; c < 3; c++) {
      const t = test.data[p * test.channels + c] as number;
      const r = ref.data[p * ref.channels + c] as number;
      if (!Number.isFinite(t) || !Number.isFinite(r)) {
        ok = false;
        break;
      }
      e += ((t - r) * (t - r)) / (r * r + 1e-2);
    }
    if (ok) {
      s += e / 3;
      n++;
    }
  }
  return n ? s / n : NaN;
}

/**
 * Temporal excess of a moving sequence: the frame-to-frame change of the output that the reference does not have,
 * mean over pixels and channels of |(O_t - O_(t-1)) - (R_t - R_(t-1))| / (R_t + 0.05). Zero for a perfect output; noise of standard deviation s gives about
 * 0.8 s sqrt(2) / R; a lag (history that does not follow the reference's own change) adds the missed change.
 */
export function temporalExcess(o1: MImg, o0: MImg, r1: MImg, r0: MImg): number {
  let s = 0;
  let n = 0;
  const px = o1.width * o1.height;
  for (let p = 0; p < px; p++)
    for (let c = 0; c < 3; c++) {
      const a = (o1.data[p * o1.channels + c] as number) - (o0.data[p * o0.channels + c] as number);
      const b = (r1.data[p * r1.channels + c] as number) - (r0.data[p * r0.channels + c] as number);
      const w = (r1.data[p * r1.channels + c] as number) + 0.05;
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      s += Math.abs(a - b) / w;
      n++;
    }
  return n ? s / n : NaN;
}
