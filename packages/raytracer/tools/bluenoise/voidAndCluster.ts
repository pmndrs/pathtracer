/**
 * Void-and-cluster blue-noise texture generator (Ulichney 1993), on a 2D or 3D torus.
 *
 * Original code, written from the paper. Two uses of one core:
 *
 *  - Classic 2D scalar: `depth = 1`, Gaussian energy sigma 1.5, toroidal. Output is a dither-array rank
 *    texture: every value k/(N-1) appears exactly once.
 *  - Spatiotemporal (this repo's "ST" variant): `depth = D > 1`. The energy of a filled cell is a SUM of two
 *    separate Gaussians, `g(dx,dy,dt) = [dt==0] gs(dx,dy) + w [dx==dy==0, dt != 0] gt(dt)`. Cells in the
 *    same slice repel with the spatial sigma, cells in the same pixel column (other slices) repel with the
 *    temporal sigma. Ranks are assigned over the whole volume, so each slice is blue at every threshold and
 *    each pixel's sequence over time is blue (high-pass in time). The sum form (not the product) is the
 *    "separate" filter combination of Wolfe et al. 2022 / Donnelly & Wolfe 2024.
 *
 * The three Ulichney phases are: (0) relax a random ~10% binary pattern until its tightest cluster is also
 * its largest void; (1) remove the tightest cluster repeatedly, ranks K-1 .. 0; (2) from the relaxed pattern
 * add the largest void repeatedly, ranks K .. N-1. Ulichney's third phase (cluster of zeros past 50%) is the
 * same cell as the largest void of the ones, since the zero-energy is a constant minus the one-energy; one
 * loop covers both.
 */
import { DualExtrema } from './extrema.ts';
import { mulberry32, sampleIndices } from './rng.ts';

export interface VoidAndClusterOptions {
  width: number;
  height: number;
  /** Number of slices; 1 gives the classic 2D texture. */
  depth?: number;
  /** Spatial Gaussian sigma in pixels (default 1.5). */
  sigmaSpatial?: number;
  /** Temporal Gaussian sigma in slices (default 1.5); ignored when depth is 1. */
  sigmaTemporal?: number;
  /** Weight of the temporal term relative to the spatial one (default 1). */
  temporalWeight?: number;
  /** Fraction of ones in the initial pattern (default 0.1). */
  initialFraction?: number;
  seed?: number;
}

export interface VoidAndClusterResult {
  /** A permutation of 0..N-1; index = (z * height + y) * width + x. */
  ranks: Uint32Array;
  /** rank / (N-1) as float32, in [0, 1]. */
  values: Float32Array;
  /** Wall-clock milliseconds spent. */
  ms: number;
}

export function voidAndCluster(opts: VoidAndClusterOptions): VoidAndClusterResult {
  const t0 = performance.now();
  const W = opts.width;
  const H = opts.height;
  const D = opts.depth ?? 1;
  const sigmaS = opts.sigmaSpatial ?? 1.5;
  const sigmaT = opts.sigmaTemporal ?? 1.5;
  const wT = opts.temporalWeight ?? 1;
  const N = W * H * D;
  const R = Math.ceil(4 * sigmaS);
  const RT = D > 1 ? Math.ceil(4 * sigmaT) : 0;
  if (W < 2 * R + 1 || H < 2 * R + 1) throw new Error('texture too small for the spatial kernel');
  if (D > 1 && D < 2 * RT + 1) throw new Error('depth too small for the temporal kernel');

  const side = 2 * R + 1;
  const kernelS = new Float64Array(side * side);
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      kernelS[(dy + R) * side + dx + R] = Math.exp(-(dx * dx + dy * dy) / (2 * sigmaS * sigmaS));
    }
  }
  const kernelT = new Float64Array(2 * RT + 1);
  for (let dt = -RT; dt <= RT; dt++) kernelT[dt + RT] = wT * Math.exp(-(dt * dt) / (2 * sigmaT * sigmaT));

  const plane = W * H;
  const state = new Uint8Array(N);
  const energy = new Float64Array(N);
  const ex = new DualExtrema(N);
  const xs = new Int32Array(side);

  const refresh = (i: number): void => {
    const e = energy[i]!;
    if (state[i]) {
      ex.hi[i] = e;
      ex.lo[i] = -Infinity;
    } else {
      ex.hi[i] = -Infinity;
      ex.lo[i] = -e;
    }
    ex.touch(i);
  };

  /** Add (sign = +1) or remove (-1) a filled cell: update energies in its torus window, then the trees. */
  const apply = (p: number, on: boolean): void => {
    state[p] = on ? 1 : 0;
    const sign = on ? 1 : -1;
    const z = (p / plane) | 0;
    const rem = p - z * plane;
    const y = (rem / W) | 0;
    const x = rem - y * W;
    for (let k = 0; k < side; k++) xs[k] = (x + k - R + W) % W;
    const zBase = z * plane;
    for (let dy = -R; dy <= R; dy++) {
      const rowBase = zBase + ((y + dy + H) % H) * W;
      const kRow = (dy + R) * side;
      for (let k = 0; k < side; k++) {
        const idx = rowBase + xs[k]!;
        energy[idx] = energy[idx]! + sign * kernelS[kRow + k]!;
        refresh(idx);
      }
    }
    if (D > 1) {
      for (let dt = -RT; dt <= RT; dt++) {
        if (dt === 0) continue;
        const idx = ((z + dt + D) % D) * plane + rem;
        energy[idx] = energy[idx]! + sign * kernelT[dt + RT]!;
        refresh(idx);
      }
    }
    ex.flush();
  };

  // Phase 0: random initial pattern, relaxed.
  const rand = mulberry32(opts.seed ?? 1);
  const K = Math.max(1, Math.round((opts.initialFraction ?? 0.1) * N));
  for (const p of sampleIndices(N, K, rand)) {
    state[p] = 1;
    // energies only: trees are rebuilt below
    addEnergy(p);
  }
  function addEnergy(p: number): void {
    const z = (p / plane) | 0;
    const rem = p - z * plane;
    const y = (rem / W) | 0;
    const x = rem - y * W;
    const zBase = z * plane;
    for (let dy = -R; dy <= R; dy++) {
      const rowBase = zBase + ((y + dy + H) % H) * W;
      for (let dx = -R; dx <= R; dx++) {
        energy[rowBase + ((x + dx + W) % W)]! += kernelS[(dy + R) * side + dx + R]!;
      }
    }
    for (let dt = -RT; dt <= RT; dt++) {
      if (dt !== 0) energy[((z + dt + D) % D) * plane + rem]! += kernelT[dt + RT]!;
    }
  }
  const rebuildKeys = (): void => {
    for (let i = 0; i < N; i++) {
      const e = energy[i]!;
      ex.hi[i] = state[i] ? e : -Infinity;
      ex.lo[i] = state[i] ? -Infinity : -e;
    }
    ex.rebuild();
  };
  rebuildKeys();

  const iterLimit = 20 * K;
  for (let it = 0; it < iterLimit; it++) {
    const c = ex.argMaxFilled();
    apply(c, false);
    const v = ex.argMinEmpty();
    if (v === c) {
      apply(c, true);
      break;
    }
    apply(v, true);
  }

  const initState = state.slice();
  const initEnergy = energy.slice();
  const ranks = new Uint32Array(N);

  // Phase 1: peel off the tightest cluster, ranks K-1 .. 0.
  for (let r = K - 1; r >= 0; r--) {
    const c = ex.argMaxFilled();
    ranks[c] = r;
    apply(c, false);
  }

  // Phase 2: restore the relaxed pattern, then fill the largest void, ranks K .. N-1.
  state.set(initState);
  energy.set(initEnergy);
  rebuildKeys();
  for (let r = K; r < N; r++) {
    const v = ex.argMinEmpty();
    ranks[v] = r;
    apply(v, true);
  }

  const values = new Float32Array(N);
  const inv = 1 / Math.max(1, N - 1);
  for (let i = 0; i < N; i++) values[i] = ranks[i]! * inv;
  return { ranks, values, ms: performance.now() - t0 };
}
