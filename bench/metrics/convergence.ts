/**
 * Convergence helpers: error-vs-spp and error-vs-time series from snapshots, and "time (or spp) to reach error E".
 */
import type { Image } from '../src/adapter.ts';
import { flipLdr, type FlipOptions } from './flip.ts';
import { compareImages, type MetricOptions } from './metrics.ts';

export interface Snapshot { spp: number; wallMs: number; image: Image }
export type ErrorMetric = 'mse' | 'relMse' | 'smape' | 'flip' | 'absBias' | 'fireflyPct';
export interface SeriesPoint { spp: number; wallMs: number; error: number }

/** Error of each snapshot against `ref`, in snapshot order. 'absBias' is |mean signed error|. */
export function convergenceSeries(
  snapshots: readonly Snapshot[], ref: Image, metric: ErrorMetric = 'relMse',
  opts: MetricOptions & FlipOptions = {},
): SeriesPoint[] {
  return snapshots.map((s) => {
    let error: number;
    if (metric === 'flip') error = flipLdr(s.image, ref, opts).mean;
    else {
      const m = compareImages(s.image, ref, opts);
      error = metric === 'absBias' ? Math.abs(m.bias) : m[metric];
    }
    return { spp: s.spp, wallMs: s.wallMs, error };
  });
}

/**
 * First x (spp or wallMs) at which the series drops to <= `target`, interpolated between the bracketing points
 * (log-log when all four values are positive, else linear). Returns null if it never gets there. If the very first
 * point is already <= target, returns that point's x (no extrapolation to the left). The series need not be
 * monotone: the FIRST crossing is reported.
 */
export function xToReach(series: readonly SeriesPoint[], target: number, axis: 'wallMs' | 'spp' = 'wallMs'): number | null {
  if (series.length === 0) return null;
  const first = series[0]!;
  if (first.error <= target) return first[axis];
  for (let i = 1; i < series.length; i++) {
    const a = series[i - 1]!, b = series[i]!;
    if (b.error <= target) {
      const xa = a[axis], xb = b[axis], ea = a.error, eb = b.error;
      if (xa > 0 && xb > 0 && ea > 0 && eb > 0 && target > 0) {
        const t = (Math.log(target) - Math.log(ea)) / (Math.log(eb) - Math.log(ea));
        return Math.exp(Math.log(xa) + t * (Math.log(xb) - Math.log(xa)));
      }
      const t = ea === eb ? 1 : (target - ea) / (eb - ea);
      return xa + t * (xb - xa);
    }
  }
  return null;
}

export const timeToReach = (series: readonly SeriesPoint[], target: number) => xToReach(series, target, 'wallMs');
export const sppToReach = (series: readonly SeriesPoint[], target: number) => xToReach(series, target, 'spp');
