/**
 * Interval queries (P5.rc1, for gi's Split Radiance Cascades): WGSL `rtClosestInterval` and friends, their TypeScript twins, the cascade
 * interval arithmetic and the equal-area octahedral direction map that per-probe direction bundles use.
 *
 * A radiance cascade traces SEGMENTS of a ray: cascade n owns `(t_{n-1}, t_n]` with `t_{-1} = 0` (Split RC, section 3.2: an occluder
 * "between t_{n-1} and t_n", `t_{n-1} < t <= t_n`). The walk underneath (bvh) is inclusive at both ends (`tMin <= t <= tMax`), so a hit
 * exactly on a shared boundary would count in two neighbouring cascades, or in none if each side rounds its own bound. We turn the
 * inclusive walk into HALF-OPEN intervals by moving one bound by one ulp, in f32, on the value the walk itself compares against:
 *
 *   `open-low`   (t0, t1]  the paper's convention (default): a hit at exactly t0 belongs to the interval BELOW. A ray that starts on a
 *                surface and uses t0 = 0 never reports that surface at t = 0.
 *   `closed-low` [t0, t1)  the usual half-open range: a hit at exactly t0 belongs to this interval, at t1 to the one above.
 *
 * "Exactly" means the f32 `t` the walk computes for the hit (it is a function of the ray and the triangle only, not of the interval),
 * compared with the f32 `t0` / `t1` you pass. Consecutive intervals `(a, b], (b, c]` therefore partition `(a, c]`: every hit lands in
 * exactly one, and `closest(a, c)` equals the nearer of the answers of the parts (tested on the GPU on every boundary, `rc-interval-check`).
 *
 * Limits: `t0 >= 0`; t at or below the smallest normal f32 (1.18e-38, no geometry is that close) is never reported by `open-low`;
 * `t1` is clamped to `RT_T_INF` (3.4028235e38, finite: bvh's box test multiplies by a padding); an empty interval (`t1 <= t0`, or
 * `t1` below the smallest normal for `closed-low`) is answered without walking. NaN bounds are not supported.
 *
 * Why the closest-hit query widens the walk (found by `rc-interval-check`): bvh's box test is only exact up to the rounding of the slab
 * entry, so a triangle hit at exactly `tMax` could be culled with its box (1207 of 2.7k exact-boundary queries failed on the plain walk).
 * The query therefore walks a range widened by 2^-19 of the coordinates' magnitude and cuts the answer back to the exact bounds on the
 * computed t. The any-hit forms (`rtVisibleInterval*`) cannot cut back (they return no t): they use the exact bounds, so a blocker within
 * ~1e-6 relative of an interval end can be missed (they are exact everywhere else). Use the closest forms where a boundary hit matters.
 */

/** The largest finite f32: `t1` for "to infinity". */
export const RT_T_INF = 3.4028234663852886e38;
/** The smallest normal f32 (GPUs may flush denormals, so a bound is never moved into the denormal range). */
export const RT_T_NORMAL_MIN = 1.1754943508222875e-38;

export type IntervalConvention = 'open-low' | 'closed-low';
export const INTERVAL_CONVENTIONS: readonly IntervalConvention[] = ['open-low', 'closed-low'];

const scratchF = new Float32Array(1);
const scratchU = new Uint32Array(scratchF.buffer);

/** The next f32 above `x` (x finite, at least `RT_T_NORMAL_MIN`): twin of WGSL `rtNextUp`. */
export function nextUpF32(x: number): number {
  scratchF[0] = x;
  scratchU[0] = (scratchU[0] as number) + 1;
  return scratchF[0] as number;
}

/** The next f32 below `x` (x at least `RT_T_NORMAL_MIN`): twin of WGSL `rtNextDown`. */
export function nextDownF32(x: number): number {
  scratchF[0] = x;
  scratchU[0] = (scratchU[0] as number) - 1;
  return scratchF[0] as number;
}

export interface IntervalBounds {
  /** The walk's inclusive `tMin` (f32). */
  tMin: number;
  /** The walk's inclusive `tMax` (f32). */
  tMax: number;
  /** No `t` can satisfy the interval; the walk is skipped. */
  empty: boolean;
}

/** Twin of WGSL `rtIntervalBounds` (`open-low`) and `rtIntervalBoundsLo` (`closed-low`): the inclusive bounds the interval is defined by. */
export function intervalBounds(t0: number, t1: number, convention: IntervalConvention = 'open-low'): IntervalBounds {
  const f = Math.fround;
  const hi = Math.min(f(t1), RT_T_INF);
  const lo0 = Math.max(f(t0), 0);
  if (convention === 'open-low') {
    const tMin = lo0 < RT_T_NORMAL_MIN ? RT_T_NORMAL_MIN : nextUpF32(lo0);
    return { tMin, tMax: hi, empty: !(hi >= tMin) };
  }
  const tMax = hi >= RT_T_NORMAL_MIN ? nextDownF32(hi) : 0;
  return { tMin: lo0, tMax, empty: !(hi >= RT_T_NORMAL_MIN) || !(tMax >= lo0) };
}

/** Does the f32 distance `t` lie in the interval? The membership test the GPU's answers obey. */
export function inInterval(t: number, t0: number, t1: number, convention: IntervalConvention = 'open-low'): boolean {
  const b = intervalBounds(t0, t1, convention);
  return !b.empty && Math.fround(t) >= b.tMin && Math.fround(t) <= b.tMax;
}

// ---------------------------------------------------------------------------------------------------------------
// Cascade intervals
// ---------------------------------------------------------------------------------------------------------------

/** End of cascade `n`'s interval: `t0 * growth^n` by repeated f32 multiplies (the order the WGSL multiplies in). */
export function cascadeEnd(n: number, t0: number, growth: number): number {
  let t = Math.fround(t0);
  const g = Math.fround(growth);
  for (let i = 0; i < n; i++) t = Math.fround(t * g);
  return t;
}

/** Start of cascade `n`'s interval: 0 for n = 0, else the end of cascade n - 1. */
export function cascadeStart(n: number, t0: number, growth: number): number {
  return n > 0 ? cascadeEnd(n - 1, t0, growth) : 0;
}

/**
 * The cascade whose `(start, end]` contains `t` (the `open-low` partition), or `count` when `t` lies beyond the last cascade's end.
 * Twin of WGSL `rtCascadeIndex`. Total over every finite t >= 0, exact at every boundary (the same f32 comparison the walk bounds use).
 */
export function cascadeIndex(t: number, t0: number, growth: number, count: number): number {
  let end = Math.fround(t0);
  const g = Math.fround(growth);
  for (let n = 0; n < count; n++) {
    if (Math.fround(t) <= end) return n;
    end = Math.fround(end * g);
  }
  return count;
}

// ---------------------------------------------------------------------------------------------------------------
// Equal-area octahedral map (direction bundles)
// ---------------------------------------------------------------------------------------------------------------

/**
 * Unit square to unit sphere, equal area (Clarberg 2008 "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"; the form below is
 * pbrt-v4's `EqualAreaSquareToSphere`, Apache-2.0, THIRD_PARTY.md). Every cell of an N x N grid covers 4 pi / N^2 steradians, and a
 * 2 x 2 split of a cell stays inside it, so cascade n + 1 (twice the cells per axis) refines cascade n's directions exactly.
 * Twin of WGSL `rtOctEqualAreaDecode`.
 */
export function octEqualAreaDecode(u01: number, v01: number): [number, number, number] {
  const u = 2 * u01 - 1;
  const v = 2 * v01 - 1;
  const up = Math.abs(u);
  const vp = Math.abs(v);
  const signedDistance = 1 - (up + vp);
  const d = Math.abs(signedDistance);
  const r = 1 - d;
  const phi = (r === 0 ? 1 : (vp - up) / r + 1) * (Math.PI / 4);
  const z = (signedDistance >= 0 ? 1 : -1) * (1 - r * r);
  const cosPhi = (u >= 0 ? 1 : -1) * Math.cos(phi);
  const sinPhi = (v >= 0 ? 1 : -1) * Math.sin(phi);
  const s = r * Math.sqrt(Math.max(0, 2 - r * r));
  return [cosPhi * s, sinPhi * s, z];
}

/** Direction of cell `(x, y)` of an `n` x `n` bundle, `jitter` in [0, 1)^2 inside the cell (0.5 = centre). Twin of the probe kernel. */
export function bundleDirection(x: number, y: number, n: number, jitter: readonly [number, number] = [0.5, 0.5]): [number, number, number] {
  return octEqualAreaDecode((x + jitter[0]) / n, (y + jitter[1]) / n);
}

// ---------------------------------------------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------------------------------------------

/**
 * The interval functions, appended to the inline module with `rtModuleWgsl(caps, { intervals: true })` (nothing is added without it, so
 * every other kernel keeps its bytes). Needs the core (`RtRay`, `RtHit`, `rtClosest`, `rtVisible`).
 *
 *     fn rtIntervalBounds(t0, t1) -> vec3f        // (t0, t1]: x = tMin, y = tMax, z = 1 when empty (inclusive bounds the interval means)
 *     fn rtIntervalBoundsLo(t0, t1) -> vec3f      // [t0, t1)
 *     fn rtClosestInterval(origin, direction, t0, t1, mask) -> RtHit     // (t0, t1]; t < 0 = nothing in the interval
 *     fn rtClosestIntervalLo(...) -> RtHit                                 // [t0, t1)
 *     fn rtVisibleInterval(origin, direction, t0, t1, mask) -> bool      // (t0, t1], true = nothing blocks (see the any-hit note above)
 *     fn rtVisibleIntervalLo(...) -> bool
 *     fn rtCascadeEnd(n, t0, growth) -> f32 / rtCascadeStart / rtCascadeIndex(t, t0, growth, count) -> u32
 *     fn rtOctEqualAreaDecode(uv) -> vec3f
 */
export const RT_INTERVAL_WGSL = /* wgsl */ `
// ---- interval queries (P5.rc1): half-open intervals over the walk's inclusive [tMin, tMax] --------------------------------

const RT_T_INF: f32 = 3.4028234663852886e38;
const RT_T_NORMAL_MIN: f32 = 1.1754943508222875e-38;

fn rtNextUp(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) + 1u);
}

fn rtNextDown(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) - 1u);
}

// (t0, t1] as inclusive bounds: x = tMin, y = tMax, z = 1.0 when no t can satisfy it.
fn rtIntervalBounds(t0: f32, t1: f32) -> vec3f {
    let hi = min(t1, RT_T_INF);
    let lo0 = max(t0, 0.0);
    let lo = select(rtNextUp(lo0), RT_T_NORMAL_MIN, lo0 < RT_T_NORMAL_MIN);
    return vec3f(lo, hi, select(1.0, 0.0, hi >= lo));
}

// [t0, t1) as inclusive bounds.
fn rtIntervalBoundsLo(t0: f32, t1: f32) -> vec3f {
    let hi = min(t1, RT_T_INF);
    let lo = max(t0, 0.0);
    let ok = hi >= RT_T_NORMAL_MIN;
    let top = select(0.0, rtNextDown(max(hi, RT_T_NORMAL_MIN)), ok);
    return vec3f(lo, top, select(1.0, 0.0, ok && top >= lo));
}

fn rtIntervalMiss() -> RtHit {
    var hit: RtHit;
    hit.t = RT_MISS_T;
    hit.frontFace = true;
    hit.material = RT_NO_MATERIAL;
    return hit;
}

// Closest hit over the exact inclusive bounds b = (tMin, tMax, empty). bvh's box test is exact only up to the rounding of the slab
// entry ((box - origin) * inverseDirection): a triangle hit at EXACTLY tMax can have its box culled (entry a few ulp above tMax), and one
// just above tMin can lose its box to the far side's relative padding. So the walk runs on a range widened by a slack of 2^-19 of the
// coordinates' magnitude, and the answer is cut back to the exact bounds on the t the triangle test computed: above tMax it is a
// miss (nothing nearer exists), below tMin the walk is re-run from the next float (at most 3 times; it needs a hit inside the slack
// zone, a one in ten thousand case).
const RT_INTERVAL_SLACK: f32 = 1.9073486328125e-6;
const RT_INTERVAL_REQUERY: u32 = 4u;

fn rtClosestIntervalBounded(origin: vec3f, direction: vec3f, b: vec3f, mask: u32) -> RtHit {
    if (b.z != 0.0) {
        return rtIntervalMiss();
    }
    let o = max(abs(origin.x), max(abs(origin.y), abs(origin.z)));
    var tMin = max(b.x - RT_INTERVAL_SLACK * (o + b.x), 0.0);
    let tMax = min(b.y + RT_INTERVAL_SLACK * (o + min(b.y, 1.0e9)), RT_T_INF);
    var hit = rtIntervalMiss();
    for (var k = 0u; k < RT_INTERVAL_REQUERY; k++) {
        hit = rtClosest(RtRay(origin, tMin, direction, tMax), mask);
        if (hit.t < 0.0 || hit.t >= b.x) {
            break;
        }
        tMin = max(rtNextUp(hit.t), RT_T_NORMAL_MIN);
        hit = rtIntervalMiss();
    }
    if (hit.t > b.y) {
        return rtIntervalMiss();
    }
    return hit;
}

fn rtClosestInterval(origin: vec3f, direction: vec3f, t0: f32, t1: f32, mask: u32) -> RtHit {
    return rtClosestIntervalBounded(origin, direction, rtIntervalBounds(t0, t1), mask);
}

fn rtClosestIntervalLo(origin: vec3f, direction: vec3f, t0: f32, t1: f32, mask: u32) -> RtHit {
    return rtClosestIntervalBounded(origin, direction, rtIntervalBoundsLo(t0, t1), mask);
}

fn rtVisibleIntervalBounded(origin: vec3f, direction: vec3f, b: vec3f, mask: u32) -> bool {
    if (b.z != 0.0) {
        return true;
    }
    return rtVisible(RtRay(origin, b.x, direction, b.y), mask);
}

fn rtVisibleInterval(origin: vec3f, direction: vec3f, t0: f32, t1: f32, mask: u32) -> bool {
    return rtVisibleIntervalBounded(origin, direction, rtIntervalBounds(t0, t1), mask);
}

fn rtVisibleIntervalLo(origin: vec3f, direction: vec3f, t0: f32, t1: f32, mask: u32) -> bool {
    return rtVisibleIntervalBounded(origin, direction, rtIntervalBoundsLo(t0, t1), mask);
}

// Cascade n's interval is (rtCascadeStart, rtCascadeEnd]: t0 * growth^n by repeated multiplies, 0 below cascade 0.
fn rtCascadeEnd(n: u32, t0: f32, growth: f32) -> f32 {
    var t = t0;
    for (var i = 0u; i < n; i++) {
        t = t * growth;
    }
    return t;
}

fn rtCascadeStart(n: u32, t0: f32, growth: f32) -> f32 {
    // Not select(): WGSL evaluates both of its arguments, and rtCascadeEnd(0u - 1u) is a loop of 2^32 multiplies (a GPU hang at n = 0).
    if (n == 0u) {
        return 0.0;
    }
    return rtCascadeEnd(n - 1u, t0, growth);
}

// The cascade whose (start, end] holds t, or count when t is beyond the last one.
fn rtCascadeIndex(t: f32, t0: f32, growth: f32, count: u32) -> u32 {
    var end = t0;
    for (var n = 0u; n < count; n++) {
        if (t <= end) {
            return n;
        }
        end = end * growth;
    }
    return count;
}

// Equal-area square -> sphere (Clarberg; pbrt-v4 EqualAreaSquareToSphere, Apache-2.0). uv in [0, 1]^2. Normalised: GPU sin / cos are only good to ~3e-5.
fn rtOctEqualAreaDecode(uv: vec2f) -> vec3f {
    let u = 2.0 * uv.x - 1.0;
    let v = 2.0 * uv.y - 1.0;
    let up = abs(u);
    let vp = abs(v);
    let signedDistance = 1.0 - (up + vp);
    let r = 1.0 - abs(signedDistance);
    let phi = select(1.0, (vp - up) / r + 1.0, r != 0.0) * 0.78539816339744830962;
    let z = select(-1.0, 1.0, signedDistance >= 0.0) * (1.0 - r * r);
    let cosPhi = select(-1.0, 1.0, u >= 0.0) * cos(phi);
    let sinPhi = select(-1.0, 1.0, v >= 0.0) * sin(phi);
    let s = r * sqrt(max(2.0 - r * r, 0.0));
    return normalize(vec3f(cosPhi * s, sinPhi * s, z));
}
`;
