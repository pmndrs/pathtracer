/** Analytic checks for furnace-style scenes whose exact answer is known. Tolerances are absolute unless `relative`. */
import type { Image } from '../src/adapter.ts';

export interface AnalyticResult {
  ok: boolean;
  /** Mean of the image per RGB channel. */
  mean: [number, number, number];
  expected: [number, number, number];
  /** Largest |pixel - expected| (checkUniform) or |mean - expected| (checkMean), after normalising if relative. */
  maxDeviation: number;
  /** Pixel of the largest deviation (checkUniform only). */
  worst?: { x: number; y: number; channel: number; value: number };
  nonFinite: number;
  tol: number;
}

type Value = number | readonly [number, number, number];
const vec3 = (v: Value): [number, number, number] => (typeof v === 'number' ? [v, v, v] : [v[0], v[1], v[2]]);

/** Every pixel and channel within `tol` of `value` (number or RGB). Non-finite pixels fail. */
export function checkUniform(image: Image, value: Value, tol: number, opts: { relative?: boolean } = {}): AnalyticResult {
  const expected = vec3(value);
  const n = image.width * image.height, ch = image.channels;
  const sum = [0, 0, 0];
  let worst = -1, worstAt = { x: 0, y: 0, channel: 0, value: 0 }, nonFinite = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = image.data[i * ch + c]!;
      if (!Number.isFinite(v)) {
        nonFinite++;
        if (worst < Infinity) { worst = Infinity; worstAt = { x: i % image.width, y: Math.floor(i / image.width), channel: c, value: v }; }
        continue;
      }
      sum[c]! += v;
      const e = expected[c]!;
      const dev = Math.abs(v - e) / (opts.relative ? Math.max(Math.abs(e), 1e-12) : 1);
      if (dev > worst) { worst = dev; worstAt = { x: i % image.width, y: Math.floor(i / image.width), channel: c, value: v }; }
    }
  }
  const mean: [number, number, number] = [sum[0]! / n, sum[1]! / n, sum[2]! / n];
  return { ok: nonFinite === 0 && worst <= tol, mean, expected, maxDeviation: worst, worst: worstAt, nonFinite, tol };
}

/** Image mean (per channel) within `tol` of `expected` (number or RGB). */
export function checkMean(image: Image, expected: Value, tol: number, opts: { relative?: boolean } = {}): AnalyticResult {
  const exp = vec3(expected);
  const n = image.width * image.height, ch = image.channels;
  const sum = [0, 0, 0];
  let nonFinite = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = image.data[i * ch + c]!;
      if (Number.isFinite(v)) sum[c]! += v; else nonFinite++;
    }
  }
  const mean: [number, number, number] = [sum[0]! / n, sum[1]! / n, sum[2]! / n];
  let dev = 0;
  for (let c = 0; c < 3; c++) {
    const d = Math.abs(mean[c]! - exp[c]!) / (opts.relative ? Math.max(Math.abs(exp[c]!), 1e-12) : 1);
    if (d > dev) dev = d;
  }
  return { ok: nonFinite === 0 && dev <= tol, mean, expected: exp, maxDeviation: dev, nonFinite, tol };
}
