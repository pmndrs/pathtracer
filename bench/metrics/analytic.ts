/** Analytic checks for furnace-style scenes whose exact answer is known. Tolerances are absolute unless `relative`. */
import type { Image } from '../src/adapter.ts';
import { lookAtCameraToWorld, transformDirection } from '../scenes/geometry.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { QUAD_LIGHT } from '../scenes/tier0/quad-light.ts';

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

export interface MaskedImageResult {
  /** Pixels compared (expected value finite). */
  n: number;
  meanTest: number;
  meanExpected: number;
  /** Mean of (test - expected) / expected over the compared pixels. */
  meanRelDev: number;
  maxRelDev: number;
  /** Test pixels that were non-finite (counted as failures). */
  nonFinite: number;
  /** |meanRelDev| < tol and no non-finite test pixel. */
  ok: boolean;
  tol: number;
}

/**
 * Compare one channel of `image` against a per-pixel expectation (`expected` = RGB float triples, NaN = masked out,
 * e.g. pixels that are not plain lit surface). `tol` bounds the |mean relative deviation|; the max per-pixel deviation
 * is reported but not gated (a noisy render has outlier pixels). Used for the quad-light closed form.
 */
export function checkMaskedImage(image: Image, expected: Float32Array, tol: number, opts: { channel?: number } = {}): MaskedImageResult {
  const c = opts.channel ?? 1;
  const n = image.width * image.height;
  let count = 0, sumT = 0, sumE = 0, sumRel = 0, maxRel = 0, nonFinite = 0;
  for (let i = 0; i < n; i++) {
    const e = expected[i * 3 + c]!;
    if (Number.isNaN(e)) continue;
    const v = image.data[i * image.channels + c]!;
    if (!Number.isFinite(v)) { nonFinite++; continue; }
    count++; sumT += v; sumE += e;
    const rel = (v - e) / e;
    sumRel += rel; maxRel = Math.max(maxRel, Math.abs(rel));
  }
  const meanRelDev = count ? sumRel / count : NaN;
  return {
    n: count, meanTest: sumT / count, meanExpected: sumE / count, meanRelDev, maxRelDev: maxRel, nonFinite, tol,
    ok: count > 0 && nonFinite === 0 && Math.abs(meanRelDev) < tol,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// quad-light scene: closed-form direct lighting (moved here from scenes/tier0 and references/compare.ts)
// ---------------------------------------------------------------------------------------------------------------

/**
 * Irradiance E (W/m²) on the plane point (x, 0, z) from the one-sided quad light facing down: both cosines equal h/r,
 * so E = L·h²·∫ dA / r⁴ over the light. Midpoint rule on an n×n grid (error ~1e-6 for n = 200).
 */
export function quadLightIrradiance(x: number, z: number, n = 200): number {
  const { size, height, radiance } = QUAD_LIGHT;
  const dA = (size / n) ** 2;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const lx = -size / 2 + ((i + 0.5) * size) / n;
    for (let j = 0; j < n; j++) {
      const lz = -size / 2 + ((j + 0.5) * size) / n;
      const r2 = (lx - x) ** 2 + (lz - z) ** 2 + height * height;
      sum += dA / (r2 * r2);
    }
  }
  return radiance * height * height * sum;
}

/** Closed form for the point directly under the centre of the light (4 corner rectangles of a/2 × b/2). */
export function quadLightCentreIrradianceClosedForm(): number {
  const { size, height, radiance } = QUAD_LIGHT;
  const A = size / 2 / height, B = size / 2 / height;
  const f = (1 / (2 * Math.PI)) * ((A / Math.sqrt(1 + A * A)) * Math.atan(B / Math.sqrt(1 + A * A)) + (B / Math.sqrt(1 + B * B)) * Math.atan(A / Math.sqrt(1 + B * B)));
  return radiance * Math.PI * 4 * f;
}

/**
 * Expected radiance image (albedo/π · E) for the quad-light scene, direct lighting only. Pixels that are not plain
 * lit plane (miss, light quad, plane edge) in ANY of 4×4 sub-samples are NaN (masked out of the comparison).
 */
export function quadLightAnalyticImage(scene: SceneDesc): Float32Array {
  const { width: W, height: H } = scene.film;
  const cam = scene.camera;
  const m = lookAtCameraToWorld(cam.position, cam.target, cam.up);
  const t = Math.tan((cam.yFovDeg * Math.PI) / 360);
  const aspect = W / H;
  const out = new Float32Array(W * H * 3);
  const sub = 4;
  const half = QUAD_LIGHT.planeSize / 2;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      let acc = 0;
      let ok = true;
      for (let sj = 0; sj < sub && ok; sj++) {
        for (let si = 0; si < sub; si++) {
          const px = (i + (si + 0.5) / sub) / W, py = (j + (sj + 0.5) / sub) / H;
          const d = transformDirection(m, [(2 * px - 1) * t * aspect, (1 - 2 * py) * t, -1]);
          if (d[1] >= 0) { ok = false; break; }
          const s = -cam.position[1] / d[1];
          const x = cam.position[0] + s * d[0], z = cam.position[2] + s * d[2];
          // light quad at y = height: does the ray cross it?
          const sl = (QUAD_LIGHT.height - cam.position[1]) / d[1];
          const lx = cam.position[0] + sl * d[0], lz = cam.position[2] + sl * d[2];
          if (sl > 0 && Math.abs(lx) <= QUAD_LIGHT.size / 2 && Math.abs(lz) <= QUAD_LIGHT.size / 2) { ok = false; break; }
          if (Math.abs(x) > half || Math.abs(z) > half) { ok = false; break; }
          acc += (QUAD_LIGHT.albedo / Math.PI) * quadLightIrradiance(x, z, 40);
        }
      }
      const v = ok ? acc / (sub * sub) : NaN;
      out.set([v, v, v], (j * W + i) * 3);
    }
  }
  return out;
}
