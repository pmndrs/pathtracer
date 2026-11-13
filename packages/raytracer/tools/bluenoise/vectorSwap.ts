/**
 * Vector-valued (vec2) blue noise by pixel swapping, after Georgiev & Fajardo, "Blue-noise Dithered Sampling"
 * (SIGGRAPH 2016 Talks). Original code, written from the paper's energy:
 *
 *   E = sum over pixel pairs p != q of exp(-|p - q|^2 / sigmaI^2 - |s_p - s_q|^(d/2) / sigmaS^2),  d = 2
 *
 * Starting point: each channel is an independently shuffled stratified ramp k/(N-1), so both marginals are exact
 * permutations and stay so (swaps only move whole vec2 values between pixels). Greedy: a random pixel pair is
 * swapped when that lowers E. Pairs are drawn over the whole torus (local pairs converged worse in the P0.17 sweep).
 *
 * Why it exists: two independent scalar void-and-cluster channels are each blue, but a 2D integrand thresholds a
 * combination of both channels, and that pattern is only partly blue (REPORT.md, "vec2").
 */
import { mulberry32 } from './rng.ts';

export interface VectorSwapOptions {
  width: number;
  height: number;
  /** Image-space sigma in pixels (paper: 2.1). */
  sigmaImage?: number;
  /** Sample-space sigma (paper: 1). */
  sigmaSample?: number;
  /** Swap proposals per pixel (default 200). */
  iterationsPerPixel?: number;
  seed?: number;
}

export interface VectorSwapResult {
  /** Interleaved (x, y) per pixel, row-major, in [0, 1]. */
  values: Float32Array;
  /** Accepted swaps. */
  accepted: number;
  ms: number;
}

export function vectorSwapVec2(opts: VectorSwapOptions): VectorSwapResult {
  const t0 = performance.now();
  const W = opts.width;
  const H = opts.height;
  const N = W * H;
  const sigmaI = opts.sigmaImage ?? 2.1;
  const invSigmaS2 = 1 / (opts.sigmaSample ?? 1) ** 2;
  const R = Math.round(3 * sigmaI);
  if (W < 2 * R + 1 || H < 2 * R + 1) throw new Error('texture too small for the image kernel');
  const rand = mulberry32(opts.seed ?? 1);

  const sx = new Float64Array(N);
  const sy = new Float64Array(N);
  for (const arr of [sx, sy]) {
    for (let i = 0; i < N; i++) arr[i] = i / (N - 1);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = t;
    }
  }

  // Neighbour window (centre excluded): offsets and image-space weights.
  const offX: number[] = [];
  const offY: number[] = [];
  const wts: number[] = [];
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      if (dx === 0 && dy === 0) continue;
      offX.push(dx);
      offY.push(dy);
      wts.push(Math.exp(-(dx * dx + dy * dy) / (sigmaI * sigmaI)));
    }
  }
  const K = wts.length;
  const ox = Int32Array.from(offX);
  const oy = Int32Array.from(offY);
  const kw = Float64Array.from(wts);
  // Wrap tables avoid a modulo per term.
  const wrapX = new Int32Array(W + 2 * R);
  const wrapY = new Int32Array(H + 2 * R);
  for (let i = 0; i < W + 2 * R; i++) wrapX[i] = (i - R + W) % W;
  for (let i = 0; i < H + 2 * R; i++) wrapY[i] = ((i - R + H) % H) * W;

  /** Energy between pixel p holding (vx, vy) and its window, skipping pixel `skip`. */
  const local = (p: number, vx: number, vy: number, skip: number): number => {
    const x = p % W;
    const y = (p - x) / W;
    let e = 0;
    for (let k = 0; k < K; k++) {
      const q = wrapY[y + oy[k]! + R]! + wrapX[x + ox[k]! + R]!;
      if (q === skip) continue;
      const dx = vx - sx[q]!;
      const dy = vy - sy[q]!;
      e += kw[k]! * Math.exp(-Math.sqrt(dx * dx + dy * dy) * invSigmaS2);
    }
    return e;
  };

  let accepted = 0;
  const iters = Math.round((opts.iterationsPerPixel ?? 200) * N);
  for (let it = 0; it < iters; it++) {
    const p = Math.floor(rand() * N);
    const q = Math.floor(rand() * N);
    if (p === q) continue;
    const px = sx[p]!;
    const py = sy[p]!;
    const qx = sx[q]!;
    const qy = sy[q]!;
    const before = local(p, px, py, q) + local(q, qx, qy, p);
    const after = local(p, qx, qy, q) + local(q, px, py, p);
    if (after < before) {
      sx[p] = qx;
      sy[p] = qy;
      sx[q] = px;
      sy[q] = py;
      accepted++;
    }
  }

  const values = new Float32Array(2 * N);
  for (let i = 0; i < N; i++) {
    values[2 * i] = sx[i]!;
    values[2 * i + 1] = sy[i]!;
  }
  return { values, accepted, ms: performance.now() - t0 };
}
