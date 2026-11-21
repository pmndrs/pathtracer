/**
 * Participating media (P5): the data model, the TypeScript twins of the WGSL estimators (`volume-wgsl.ts`) and the closed forms the checks compare with.
 * docs/internal/design/volumes.md has the derivations; DIVERGES V240+ the departures from pbrt-v4's `VolPathIntegrator`.
 *
 * A medium has `sigma_t(x) = density(x) * sigmaT` (RGB, per metre; density is 1 for a homogeneous medium), a single-scattering albedo
 * `albedo` (per channel, `sigma_s = albedo * sigma_t`) and a Henyey-Greenstein phase function with mean cosine `g` (pbrt's sign: g > 0 scatters forward).
 *
 * Estimators (all unbiased; the WGSL does the same arithmetic):
 *  - Free flight and collisions: delta tracking against the scalar majorant `mu = max density * max channel of sigmaT`. A tentative collision at x is
 *    one of {scatter, absorb, null} with per-channel shares (sigma_s / mu, sigma_a / mu, 1 - sigma_t / mu). Absorption contributes nothing, so it is never
 *    sampled: scatter is chosen with the scalar probability `ps = max(sigma_s) / mu` (weight sigma_s / max(sigma_s), at most 1 per channel), null otherwise
 *    (weight (1 - sigma_t / mu) / (1 - ps)). Reaching the end of the segment has weight 1. A grey medium gets weights 1 and 0 (classic delta tracking
 *    with absorption as a kill), a chromatic one gets the per-channel ratios, so one path serves all three channels.
 *  - Shadow transmittance: closed form `exp(-sigma_t d)` for a homogeneous medium, ratio tracking (a product of `1 - sigma_t / mu` over the tentative collisions) for a grid.
 *  - Phase: the HG density is sampled exactly, so the throughput weight of a phase-sampled bounce is 1.
 */

import type { PtPhaseLobes } from './phase.ts';

export type VolRgb = readonly [number, number, number];

/** A dense density grid in the medium's own space: `worldToIndex` is the row-major 3x4 matrix taking a world point to index space (voxel centres at integer coordinates). */
export interface PtMediumGrid {
  resolution: readonly [number, number, number];
  worldToIndex: readonly number[];
  /** Largest density value in the grid (the majorant is this times the largest sigmaT channel). */
  densityMax: number;
}

export interface PtMedium {
  /** Per metre, already multiplied by the medium's `scale`. */
  sigmaT: VolRgb;
  /** sigma_s / sigma_t per channel (0 where sigma_t is 0). */
  albedo: VolRgb;
  /** Henyey-Greenstein mean cosine in (-1, 1) (lobe A when `lobes` is set: then `g` must equal `lobes.gA`). */
  g: number;
  grid?: PtMediumGrid;
  /** A two-lobe phase function (P5.v14, `phase.ts`): `(1 - w) HG(gA) + w Draine(gB, alpha)`. Absent or `w` 0 = one HG lobe of mean cosine `g` (the kernel text is then unchanged). */
  lobes?: PtPhaseLobes;
}

/** What one material id means to the integrator (0 = no medium, otherwise `medium index + 1`). */
export interface PtMaterialVolume {
  /** Medium on the side the geometric normal points away from (inside a closed mesh with outward winding). */
  interior: number;
  exterior: number;
  /** The surface is only a medium boundary: rays pass through it (pbrt `interface`, Mitsuba `null`). */
  boundary: boolean;
}

export interface PtVolumeOptions {
  media: readonly PtMedium[];
  /** Indexed by material id; ids past the end have no medium. */
  materials: readonly PtMaterialVolume[];
  /** Medium id (index + 1) the camera starts in; 0 / undefined = vacuum. */
  cameraMedium?: number;
  /**
   * Transmittance of a shadow ray through a HOMOGENEOUS medium: `'analytic'` (default, `exp(-sigma_t d)`, no noise) or `'ratio'` (ratio tracking against the majorant, what pbrt does:
   * same mean, more variance; the A/B of V241). A grid medium always uses ratio tracking. V2.2 `'residual'` (grid media with `majorant: 'brick'` and `volumeGrid.majorant`
   * built with `brickMajorants(.., .., { minima: true })`): residual ratio tracking (Novak et al. 2014), per DDA cell the analytic transmittance of the cell's minorant
   * times a ratio-tracking estimate of the residual (majorant - minorant); homogeneous media stay analytic. Same mean, fewer lookups in a smooth dense grid.
   */
  shadowTracking?: 'analytic' | 'ratio' | 'residual';
  /**
   * Majorant of a grid medium: `'global'` (default; densityMax x max sigma_t everywhere, V243) or `'brick'` (the largest density per 8^3 voxels, walked by a DDA; needs
   * `volumeGrid.majorant` = `brickMajorants(...)`, group 2 binding `PT_VOLUME_MAJORANT_BINDING`). Same estimator, fewer null collisions in a sparse grid; the A/B of V243.
   */
  majorant?: 'global' | 'brick';
  /**
   * V1.1: Russian roulette in ratio tracking (pbrt-v4's rule): when the largest channel of the running transmittance estimate is below 0.05, the ray ends with
   * probability 0.75 (estimate 0) and otherwise continues with the estimate times 4. Unbiased; the random number comes from the same stream. Default false (kernels unchanged).
   */
  trRoulette?: boolean;
  /**
   * V1.3: medium scatter events get their own cap `maxVolumeDepth` (`PtLimits`, read from `ptParams.forward.w`; unset = `maxDepth`). `depth` stays the TOTAL vertex
   * count (emission, MIS and sampler dimensions keep using it), a second counter counts the scatters; a surface vertex ends the path at `depth - scatters >= maxDepth`,
   * a medium scatter at `scatters >= maxVolumeDepth`, so scatters no longer eat the surface budget (Cycles / three-gpu-pathtracer style; pbrt and Mitsuba share one
   * depth: references use it off or with equal caps). Default off: the kernel text is unchanged.
   */
  separateDepth?: boolean;
  /**
   * V1.2: how a grid medium's density is read. `'manual'` (default): eight `textureLoad`s blended in the shader, exact, zero outside the grid. `'hardware'`: one
   * `textureSampleLevel` with a linear clamp-to-edge sampler (group 2 binding `PT_VOLUME_SAMPLER_BINDING`) times a per-axis boundary factor that restores the zero
   * outside (`gridBoundaryFactor`); the grid texture is then bound as filterable, so `r32float` needs the `float32-filterable` device feature (`r16float` does not).
   * Hardware filter weights are 8-bit, so the image differs slightly from `'manual'` (not bit-equal). Default `'manual'`: kernel text unchanged.
   */
  densityFilter?: 'manual' | 'hardware';
  /**
   * P5.v15 / P5.v16 (megakernel only): extra per-pixel sums in the accumulator tail (`PT_VOLUME_OUTPUT_VEC4` vec4 per pixel after the radiance; layout in
   * `ptDecodeVolumeOutputs`): the radiance of scattering orders 0 and 1 (the light of paths with at most one scatter or bounce vertex, the expectation of a
   * `maxDepth` 1 render), the primary ray's transmittance through the media up to the first solid surface (ratio tracking on its own stream, so the beauty image
   * is bit-identical to the kernel without outputs), the per-sample second moment of the radiance (a standard error from one render) and the distance to the
   * camera ray's first scatter event. Default off (the kernel text is unchanged).
   */
  outputs?: boolean;
  /**
   * V4.2 (megakernel only): forced in-scatter next-event estimation in HOMOGENEOUS media (grid media are left alone). Every free-flight segment in a homogeneous medium adds one light
   * sample at a distance drawn from the truncated exponential on the segment, and a real collision of that segment no longer samples a light itself (the estimate is replaced, not
   * added): unbiased, and in thin media (most rays never scatter) it turns god rays from noise into a smooth estimate. Derivation and twin: `volume/forced-wgsl.ts`. Needs `nee`;
   * refused with `outputs`. Default false: the kernel text is unchanged.
   */
  forcedNee?: boolean;
}

/** vec4 per pixel the volume outputs add to the accumulator (after the `width * height` radiance sums). */
export const PT_VOLUME_OUTPUT_VEC4 = 3;

/** The volume outputs averaged over `samples` (per pixel, row 0 = top, like `readImage`). */
export interface PtVolumeOutputs {
  width: number;
  height: number;
  samples: number;
  /** Mean radiance (rgb per pixel), the beauty image. */
  radiance: Float32Array;
  /** Orders 0 and 1 (rgb); `radiance - order01` is orders 2 and above. */
  order01: Float32Array;
  /** Transmittance of the primary ray through the media (1 - alpha), one value per pixel. */
  transmittance: Float32Array;
  /** Per-pixel standard error of the mean radiance (rgb): sqrt((E[L^2] - E[L]^2) / (N - 1)). */
  stdError: Float32Array;
  /** Mean distance from the camera to the camera ray's first scatter event, over the samples that had one (0 where none did). */
  firstScatterDistance: Float32Array;
  /** Fraction of samples whose camera ray scattered in a medium before reaching a surface or escaping. */
  scatterFraction: Float32Array;
}

/**
 * Decodes the accumulator of a kernel built with `volumes.outputs` (`sums` = the whole buffer: n radiance vec4, then plane p of n vec4 at offset (1 + p) n):
 * plane 0 = (orders 0-1 rgb, transmittance), plane 1 = (radiance^2 rgb, first-scatter distance), plane 2 = (first scatter happened 0/1, 0, 0, 0).
 */
export function ptDecodeVolumeOutputs(sums: Float32Array, width: number, height: number, samples: number): PtVolumeOutputs {
  const n = width * height;
  if (sums.length < n * 4 * (1 + PT_VOLUME_OUTPUT_VEC4)) throw new RangeError(`ptDecodeVolumeOutputs: ${sums.length} floats for ${width} x ${height} pixels`);
  const N = Math.max(1, samples);
  const radiance = new Float32Array(n * 3), order01 = new Float32Array(n * 3), stdError = new Float32Array(n * 3);
  const transmittance = new Float32Array(n), firstScatterDistance = new Float32Array(n), scatterFraction = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = i * 4, p0 = (n + i) * 4, p1 = (2 * n + i) * 4, p2 = (3 * n + i) * 4;
    for (let c = 0; c < 3; c++) {
      const mean = (sums[r + c] as number) / N;
      radiance[i * 3 + c] = mean;
      order01[i * 3 + c] = (sums[p0 + c] as number) / N;
      const m2 = (sums[p1 + c] as number) / N;
      stdError[i * 3 + c] = N > 1 ? Math.sqrt(Math.max(0, m2 - mean * mean) / (N - 1)) : 0;
    }
    transmittance[i] = (sums[p0 + 3] as number) / N;
    const hits = sums[p2] as number;
    scatterFraction[i] = hits / N;
    firstScatterDistance[i] = hits > 0 ? (sums[p1 + 3] as number) / hits : 0;
  }
  return { width, height, samples, radiance, order01, transmittance, stdError, firstScatterDistance, scatterFraction };
}

/** Medium from the bench's coefficients. */
export function ptMediumFrom(sigmaA: VolRgb, sigmaS: VolRgb, g = 0, scale = 1, grid?: PtMediumGrid): PtMedium {
  const sigmaT = [0, 1, 2].map((i) => ((sigmaA[i] as number) + (sigmaS[i] as number)) * scale) as unknown as [number, number, number];
  const albedo = [0, 1, 2].map((i) => ((sigmaT[i] as number) > 0 ? ((sigmaS[i] as number) * scale) / (sigmaT[i] as number) : 0)) as unknown as [number, number, number];
  return { sigmaT, albedo, g, ...(grid ? { grid } : {}) };
}

/** The scalar majorant of delta tracking. */
export function ptMediumMajorant(m: PtMedium): number {
  return (m.grid ? m.grid.densityMax : 1) * Math.max(m.sigmaT[0], m.sigmaT[1], m.sigmaT[2]);
}

/** HG density of the angle between wo (towards the previous vertex) and wi: pbrt's convention (`+ 2 g cos`). Integrates to 1 over the sphere. */
export function hgPhase(cosTheta: number, g: number): number {
  const d = 1 + g * g + 2 * g * cosTheta;
  return ((1 - g * g) / (d * Math.sqrt(Math.max(d, 1e-12)))) / (4 * Math.PI);
}

/** cos(theta) between wo and the sampled wi, theta ~ HG (pbrt-v4 `SampleHenyeyGreenstein`). */
export function hgSampleCos(g: number, u: number): number {
  if (Math.abs(g) < 1e-3) return 1 - 2 * u;
  const s = (1 - g * g) / (1 + g - 2 * g * u);
  return -(1 + g * g - s * s) / (2 * g);
}

export type VolEventKind = 'pass' | 'scatter' | 'killed';
export interface VolEvent {
  kind: VolEventKind;
  /** Distance of the scatter (or `tmax` for a pass). */
  t: number;
  /** Throughput factor per channel. */
  weight: [number, number, number];
}

/** Density at a point; a homogeneous medium is 1 everywhere. `density` is the grid lookup (the TS twin of the texture fetch). */
export type DensityFn = (x: number, y: number, z: number) => number;

/** One free-flight segment of a path (`ptVolSample` in WGSL). `rand` returns uniform numbers in [0, 1). */
export function deltaTrack(m: PtMedium, o: VolRgb, d: VolRgb, tmax: number, rand: () => number, density: DensityFn = () => 1): VolEvent {
  const mu = ptMediumMajorant(m);
  const w: [number, number, number] = [1, 1, 1];
  if (!(mu > 0)) return { kind: 'pass', t: tmax, weight: w };
  let t = 0;
  for (let i = 0; i < 4096; i++) {
    t += -Math.log(1 - rand()) / mu;
    if (t >= tmax) return { kind: 'pass', t: tmax, weight: w };
    const dens = m.grid ? density(o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t) : 1;
    const st = m.sigmaT.map((s) => s * dens);
    const ss = st.map((s, c) => s * (m.albedo[c] as number));
    const ssMax = Math.max(ss[0] as number, ss[1] as number, ss[2] as number);
    const ps = ssMax / mu;
    if (rand() < ps) return { kind: 'scatter', t, weight: [w[0] * (ss[0] as number) / ssMax, w[1] * (ss[1] as number) / ssMax, w[2] * (ss[2] as number) / ssMax] };
    for (let c = 0; c < 3; c++) w[c] = (w[c] as number) * ((1 - (st[c] as number) / mu) / (1 - ps));
    if (!(Math.max(w[0], w[1], w[2]) > 0)) return { kind: 'killed', t, weight: w };
  }
  return { kind: 'killed', t: tmax, weight: w };
}

/** Transmittance estimate over a segment (`ptVolTrSegment`): closed form for a homogeneous medium, ratio tracking for a grid. */
export function transmittance(m: PtMedium, o: VolRgb, d: VolRgb, len: number, rand: () => number, density: DensityFn = () => 1, roulette = false): [number, number, number] {
  if (!m.grid) return m.sigmaT.map((s) => Math.exp(-s * len)) as [number, number, number];
  const mu = ptMediumMajorant(m);
  const tr: [number, number, number] = [1, 1, 1];
  if (!(mu > 0)) return tr;
  let t = 0;
  for (let i = 0; i < 4096; i++) {
    t += -Math.log(1 - rand()) / mu;
    if (t >= len) break;
    const dens = density(o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t);
    for (let c = 0; c < 3; c++) tr[c] = (tr[c] as number) * (1 - ((m.sigmaT[c] as number) * dens) / mu);
    if (!(Math.max(tr[0], tr[1], tr[2]) > 0)) break;
    if (roulette && Math.max(tr[0], tr[1], tr[2]) < 0.05) {
      if (rand() < 0.75) return [0, 0, 0];
      for (let c = 0; c < 3; c++) tr[c] = (tr[c] as number) * 4;
    }
  }
  return tr;
}

/** Trilinear density of a dense grid in index space (centres at integers, zero outside): the twin of `ptVolDensity`. */
export function gridDensity(grid: PtMediumGrid, data: ArrayLike<number>): DensityFn {
  const [nx, ny, nz] = grid.resolution;
  const m = grid.worldToIndex;
  const at = (i: number, j: number, k: number) => (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz ? 0 : (data[i + nx * (j + ny * k)] as number));
  return (x, y, z) => {
    const gx = (m[0] as number) * x + (m[1] as number) * y + (m[2] as number) * z + (m[3] as number);
    const gy = (m[4] as number) * x + (m[5] as number) * y + (m[6] as number) * z + (m[7] as number);
    const gz = (m[8] as number) * x + (m[9] as number) * y + (m[10] as number) * z + (m[11] as number);
    const i0 = Math.floor(gx), j0 = Math.floor(gy), k0 = Math.floor(gz);
    const fx = gx - i0, fy = gy - j0, fz = gz - k0;
    let acc = 0;
    for (let c = 0; c < 8; c++) {
      const di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
      acc += at(i0 + di, j0 + dj, k0 + dk) * (di ? fx : 1 - fx) * (dj ? fy : 1 - fy) * (dk ? fz : 1 - fz);
    }
    return acc;
  };
}

/**
 * V1.2: the boundary factor of the hardware density lookup for ONE axis: a clamp-to-edge trilinear filter returns the edge texel outside the grid, the manual lookup
 * blends with zero there. The filter is separable, so `manual = clampedTrilinear * fx * fy * fz` with `f = clamp(g + 1, 0, 1) * clamp(n - g, 0, 1)` per axis
 * (g = index-space coordinate, n = resolution on that axis): 1 inside [0, n - 1], linear to 0 over [-1, 0] and [n - 1, n], 0 beyond.
 */
export const gridBoundaryFactor = (g: number, n: number): number => Math.min(Math.max(g + 1, 0), 1) * Math.min(Math.max(n - g, 0), 1);

/** World-to-index matrix (row-major 3x4) of a grid with `bounds` in medium space under the medium-to-world matrix `toWorld` (column-major 4x4; identity when omitted). */
export function gridWorldToIndex(resolution: readonly [number, number, number], bounds: readonly [VolRgb, VolRgb], toWorld?: readonly number[]): number[] {
  // medium space -> index space is diagonal: i = (p - p0) / (p1 - p0) * n - 0.5 = s p + o; then compose with the inverse of the medium-to-world matrix.
  const [p0, p1] = bounds;
  const sc = [0, 1, 2].map((a) => (resolution[a] as number) / ((p1[a] as number) - (p0[a] as number)));
  const off = [0, 1, 2].map((a) => -(p0[a] as number) * (sc[a] as number) - 0.5);
  const inv = toWorld ? invertAffine(toWorld) : [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];
  const out: number[] = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) out.push((sc[r] as number) * (inv[r * 4 + c] as number) + (c === 3 ? (off[r] as number) : 0));
  return out;
}

/** Inverse of an affine column-major 4x4 as a row-major 3x4 (rows of [R | t]). */
export function invertAffine(m: readonly number[]): number[] {
  // column-major: m[col * 4 + row]
  const a = m[0] as number, b = m[4] as number, c = m[8] as number, tx = m[12] as number;
  const d = m[1] as number, e = m[5] as number, f = m[9] as number, ty = m[13] as number;
  const g = m[2] as number, h = m[6] as number, i = m[10] as number, tz = m[14] as number;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const id = 1 / det;
  const r00 = (e * i - f * h) * id, r01 = (c * h - b * i) * id, r02 = (b * f - c * e) * id;
  const r10 = (f * g - d * i) * id, r11 = (a * i - c * g) * id, r12 = (c * d - a * f) * id;
  const r20 = (d * h - e * g) * id, r21 = (b * g - a * h) * id, r22 = (a * e - b * d) * id;
  return [r00, r01, r02, -(r00 * tx + r01 * ty + r02 * tz), r10, r11, r12, -(r10 * tx + r11 * ty + r12 * tz), r20, r21, r22, -(r20 * tx + r21 * ty + r22 * tz)];
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Brick majorants (`majorant: 'brick'`): a coarse grid of the largest density per 8^3 voxels, walked by a 3D DDA
// ---------------------------------------------------------------------------------------------------------------------------------

export const BRICK = 8;

/** Coarse grid dimensions of a density grid: coarse coordinate u = (index + 1) / 8, so cell c holds index space [8c - 1, 8c + 7) (the support of the trilinear lookup starts at -1). */
export function brickDims(resolution: readonly [number, number, number]): [number, number, number] {
  return [0, 1, 2].map((a) => Math.floor(((resolution[a] as number) + BRICK) / BRICK)) as [number, number, number];
}

/**
 * The majorant density of every coarse cell: the maximum over the (up to 9^3) voxels its trilinear lookups can read, x fastest.
 * V2.1 `minima: true` also returns the MINORANT (`minima`): the minimum over the same voxels, where a voxel outside the grid counts as 0 (the lookup is zero there), so every
 * trilinear density sample inside the cell lies in [minima, values].
 */
export function brickMajorants(resolution: readonly [number, number, number], data: ArrayLike<number>, options: { minima?: boolean } = {}): { resolution: [number, number, number]; values: Float32Array; minima?: Float32Array } {
  const [nx, ny, nz] = resolution;
  const dims = brickDims(resolution);
  const out = new Float32Array(dims[0] * dims[1] * dims[2]);
  const mins = options.minima ? new Float32Array(out.length) : null;
  for (let ck = 0; ck < dims[2]; ck++) for (let cj = 0; cj < dims[1]; cj++) for (let ci = 0; ci < dims[0]; ci++) {
    let m = 0;
    let lo = Infinity;
    if (mins && (ci * BRICK - 1 < 0 || ci * BRICK + BRICK > nx - 1 || cj * BRICK - 1 < 0 || cj * BRICK + BRICK > ny - 1 || ck * BRICK - 1 < 0 || ck * BRICK + BRICK > nz - 1)) lo = 0;
    for (let k = Math.max(0, ck * BRICK - 1); k <= Math.min(nz - 1, ck * BRICK + BRICK - 1); k++)
      for (let j = Math.max(0, cj * BRICK - 1); j <= Math.min(ny - 1, cj * BRICK + BRICK - 1); j++)
        for (let i = Math.max(0, ci * BRICK - 1); i <= Math.min(nx - 1, ci * BRICK + BRICK - 1); i++) {
          const v = data[i + nx * (j + ny * k)] as number;
          m = Math.max(m, v);
          if (v < lo) lo = v;
        }
    out[ci + dims[0] * (cj + dims[1] * ck)] = m;
    if (mins) mins[ci + dims[0] * (cj + dims[1] * ck)] = Math.min(lo, m);
  }
  return { resolution: dims, values: out, ...(mins ? { minima: mins } : {}) };
}

/**
 * V2.2 twin of the WGSL residual ratio tracker over the DDA cells of one ray: `density(t)` is the (trilinear) density along it, `cells` come from `ddaSegments` with
 * `max` / `min` the cell's majorant / minorant density, `sigmaT` the scalar extinction scale (max channel; the twin is grey), `rand()` uniform in [0, 1).
 * Per cell: exp(-sigmaT min d) (the control, analytic) x ratio tracking of the residual with majorant sigmaT (max - min) and weights 1 - sigmaT (density - min) / mu.
 * `lookups` counts the density evaluations. `trRoulette` is the V1.1 rule on the running estimate.
 */
export function residualTransmittance(
  cells: readonly { t0: number; t1: number; max: number; min: number }[], density: (t: number) => number, sigmaT: number, rand: () => number, trRoulette = false,
): { tr: number; lookups: number } {
  let tr = 1, lookups = 0;
  for (const c of cells) {
    tr *= Math.exp(-sigmaT * c.min * (c.t1 - c.t0));
    const mu = sigmaT * (c.max - c.min);
    if (!(mu > 0)) continue;
    let t = c.t0;
    for (let i = 0; i < 4096; i++) {
      t += -Math.log(1 - rand()) / mu;
      if (t >= c.t1) break;
      lookups++;
      tr *= 1 - (sigmaT * (density(t) - c.min)) / mu;
      if (!(tr > 0)) return { tr: 0, lookups };
      if (trRoulette && tr < 0.05) {
        if (rand() < 0.75) return { tr: 0, lookups };
        tr *= 4;
      }
    }
  }
  return { tr, lookups };
}

export interface DdaSegment { t0: number; t1: number; cell: [number, number, number] }

/** The coarse cells a ray crosses inside the grid's support box, in order, with the parametric range of each (twin of `ptDdaInit` / `ptDdaNext`). `a`, `b`: the ray in index space (point and direction per unit t). */
export function ddaSegments(dims: readonly [number, number, number], a: VolRgb, b: VolRgb, tmax: number): DdaSegment[] {
  const ua = a.map((v) => (v + 1) / BRICK), ub = b.map((v) => v / BRICK);
  let tEnter = 0, tExit = tmax;
  for (let k = 0; k < 3; k++) {
    if (Math.abs(ub[k] as number) < 1e-20) {
      if ((ua[k] as number) < 0 || (ua[k] as number) >= (dims[k] as number)) return [];
    } else {
      const t0 = (0 - (ua[k] as number)) / (ub[k] as number), t1 = ((dims[k] as number) - (ua[k] as number)) / (ub[k] as number);
      tEnter = Math.max(tEnter, Math.min(t0, t1));
      tExit = Math.min(tExit, Math.max(t0, t1));
    }
  }
  if (!(tEnter < tExit)) return [];
  const cell = [0, 1, 2].map((k) => Math.min(dims[k]! - 1, Math.max(0, Math.floor((ua[k] as number) + (ub[k] as number) * tEnter))));
  const step = ub.map((v) => (v > 0 ? 1 : -1));
  const tNext = [0, 1, 2].map((k) => (Math.abs(ub[k] as number) < 1e-20 ? 1e30 : ((cell[k] as number) + ((ub[k] as number) > 0 ? 1 : 0) - (ua[k] as number)) / (ub[k] as number)));
  const dt = ub.map((v) => (Math.abs(v) < 1e-20 ? 1e30 : Math.abs(1 / v)));
  const out: DdaSegment[] = [];
  let t = tEnter;
  for (let guard = 0; guard < 4096; guard++) {
    let axis = 0;
    if ((tNext[1] as number) < (tNext[axis] as number)) axis = 1;
    if ((tNext[2] as number) < (tNext[axis] as number)) axis = 2;
    const t1 = Math.min(tNext[axis] as number, tExit);
    out.push({ t0: t, t1, cell: [cell[0] as number, cell[1] as number, cell[2] as number] });
    t = t1;
    if (t1 >= tExit) break;
    cell[axis] = (cell[axis] as number) + (step[axis] as number);
    tNext[axis] = (tNext[axis] as number) + (dt[axis] as number);
    if ((cell[axis] as number) < 0 || (cell[axis] as number) >= (dims[axis] as number)) break;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Closed forms (the oracle of `bench/pages/volume-check.ts` and the unit tests)
// ---------------------------------------------------------------------------------------------------------------------------------

/** Unscattered transmittance of a slab of thickness `thickness` seen at cosine `mu` of the normal: exp(-sigma_t d / mu). */
export function slabTransmittance(sigmaT: number, thickness: number, mu = 1): number {
  return Math.exp((-sigmaT * thickness) / mu);
}

/**
 * Radiance leaving a plane-parallel slab of homogeneous medium (the camera ray enters at cosine `mu` with the normal) under a uniform sky of radiance `L`
 * (no other light), counting the unscattered light and at most ONE scattering event (pbrt `maxdepth` 1 + the free medium vertex; see volume-check's depth
 * note). Isotropic or HG phase; the slab is bounded by interface surfaces (no reflection). Quadrature over the scatter depth and the exit direction.
 */
export function slabSingleScatter(sigmaT: number, albedo: number, g: number, thickness: number, mu: number, L = 1, steps = 96, nmu = 120, nphi = 32): number {
  const unscattered = L * Math.exp((-sigmaT * thickness) / mu);
  // scatter at slab depth z (along the ray: s = z / mu); then the sky light arrives from direction w' (cos mu' with the normal, both hemispheres: light comes
  // from above through the thickness z (up) or from below through thickness - z (down)), phase p(wo, wi) with wo = -ray direction.
  const wo = [0, Math.sqrt(1 - mu * mu), mu] as const; // pointing back towards the camera (up = +z is the camera side)
  let acc = 0;
  const nz = steps;
  for (let iz = 0; iz < nz; iz++) {
    const z = ((iz + 0.5) / nz) * thickness; // depth below the top face
    const s = z / mu;
    const wDepth = sigmaT * albedo * Math.exp(-sigmaT * s) * (thickness / nz / mu);
    // integrate the phase over the sphere of incoming directions
    let inner = 0;
    for (let a = 0; a < nmu; a++) {
      const c = -1 + ((a + 0.5) / nmu) * 2; // cos of wi with +z
      for (let b = 0; b < nphi; b++) {
        const phi = ((b + 0.5) / nphi) * 2 * Math.PI;
        const sn = Math.sqrt(Math.max(0, 1 - c * c));
        const wi = [sn * Math.cos(phi), sn * Math.sin(phi), c] as const;
        const cosT = wo[0] * wi[0] + wo[1] * wi[1] + wo[2] * wi[2];
        // distance to the slab face that wi exits through
        const dist = c > 0 ? z / c : (thickness - z) / -c;
        inner += hgPhase(cosT, g) * Math.exp(-sigmaT * dist);
      }
    }
    inner *= (2 / nmu) * ((2 * Math.PI) / nphi);
    acc += wDepth * inner;
  }
  return unscattered + L * acc;
}

/**
 * The same single-scattering slab as `slabSingleScatter`, for any phase function and peaked ones in particular (P5.v13: HG g 0.85 and up, droplet lobes).
 * `phase(cosTheta)` is the density of the angle between wo (towards the camera) and wi (towards the light), pbrt's convention: forward scattering is cosTheta = -1.
 * The depth integral is analytic (the scatter depth z enters only through two exponentials), so only the sphere of incident directions is a quadrature, in polar
 * coordinates around the FORWARD direction -wo with a quadratic warp alpha = pi t^2 (dense at the peak: g 0.99 has a half width of about 0.6 degrees).
 */
export function slabSingleScatterPhase(sigmaT: number, albedo: number, phase: (cosTheta: number) => number, thickness: number, mu: number, L = 1, nAlpha = 4096, nPhi = 256): number {
  const unscattered = L * Math.exp((-sigmaT * thickness) / mu);
  const s = Math.sqrt(Math.max(0, 1 - mu * mu));
  const f = [0, -s, -mu] as const; // forward = -wo, wo = (0, s, mu) points back to the camera
  // an orthonormal frame around f
  const e1 = Math.abs(f[2]) < 0.9 ? normalize3(cross3(f, [0, 0, 1])) : normalize3(cross3(f, [1, 0, 0]));
  const e2 = cross3(f, e1);
  const T = thickness, st = sigmaT;
  // integral over the scatter depth z in [0, T] of (sigma_t albedo / mu) exp(-st z / mu) exp(-st dist(z, c)) dz
  const depthFactor = (c: number): number => {
    if (Math.abs(c) < 1e-12) return 0; // grazing incidence: the light travels an infinite distance inside an infinite slab
    if (c > 0) {
      const k = 1 / mu + 1 / c;
      return (-Math.expm1(-st * T * k)) / (st * k * mu);
    }
    const ac = -c;
    const k = 1 / mu - 1 / ac;
    // exp(-st T / ac) (1 - exp(-st T k)) / (st k) = (exp(-st T / ac) - exp(-st T / mu)) / (st k), written so that neither factor overflows near grazing
    const x = st * T * k;
    if (Math.abs(x) < 1e-6) return (Math.exp((-st * T) / ac) * T * (1 - x / 2)) / mu;
    return (Math.exp((-st * T) / ac) - Math.exp((-st * T) / mu)) / (st * k * mu);
  };
  let acc = 0;
  for (let a = 0; a < nAlpha; a++) {
    const t = (a + 0.5) / nAlpha;
    const alpha = Math.PI * t * t;
    const dAlpha = 2 * Math.PI * t / nAlpha;
    const ca = Math.cos(alpha), sa = Math.sin(alpha);
    // cosTheta between wo and wi = -cos(alpha) (wi at angle alpha from forward)
    const p = phase(-ca);
    let ring = 0;
    for (let b = 0; b < nPhi; b++) {
      const phi = ((b + 0.5) / nPhi) * 2 * Math.PI;
      const cp = Math.cos(phi), sp = Math.sin(phi);
      const wiz = ca * f[2] + sa * (cp * (e1[2] as number) + sp * (e2[2] as number));
      ring += depthFactor(wiz);
    }
    acc += p * ring * ((2 * Math.PI) / nPhi) * sa * dAlpha;
  }
  return unscattered + L * st * albedo * acc;
}

function cross3(a: readonly number[], b: readonly number[]): [number, number, number] {
  return [(a[1] as number) * (b[2] as number) - (a[2] as number) * (b[1] as number), (a[2] as number) * (b[0] as number) - (a[0] as number) * (b[2] as number), (a[0] as number) * (b[1] as number) - (a[1] as number) * (b[0] as number)];
}
function normalize3(v: readonly number[]): [number, number, number] {
  const l = Math.hypot(v[0] as number, v[1] as number, v[2] as number);
  return [(v[0] as number) / l, (v[1] as number) / l, (v[2] as number) / l];
}

/** Fraction of a sphere of radius R (interface boundary, uniform sky, albedo 1, any phase) that reaches the camera: 1 (energy conservation); the radiance equals the sky. */
export const FURNACE_RADIANCE = 1;

// ---------------------------------------------------------------------------------------------------------------------------------
// r16float grids (P5.v18)
// ---------------------------------------------------------------------------------------------------------------------------------

const F16_VIEW = new DataView(new ArrayBuffer(4));
/** IEEE half bits of an f32 value, round to nearest even (subnormals and infinities included). */
export function f32ToF16Bits(v: number): number {
  F16_VIEW.setFloat32(0, v);
  const x = F16_VIEW.getUint32(0);
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  let mant = x & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  let e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    let m = mant >>> shift;
    const rest = mant & ((1 << shift) - 1);
    if (rest > half || (rest === half && (m & 1))) m++;
    return sign | m;
  }
  let m = mant >>> 13;
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (m & 1))) {
    m++;
    if (m === 0x400) { m = 0; e++; if (e >= 0x1f) return sign | 0x7c00; }
  }
  return sign | (e << 10) | m;
}

export function f16BitsToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return sign * m * 2 ** -24;
  if (e === 0x1f) return m ? NaN : sign * Infinity;
  return sign * (1 + m / 1024) * 2 ** (e - 15);
}

/** The grid with every value rounded to the nearest f16 (as f32): compute `densityMax` and `brickMajorants` from THIS array before uploading it as `r16float`. */
export function ptRoundGridF16(density: ArrayLike<number>): Float32Array {
  const out = new Float32Array(density.length);
  for (let i = 0; i < density.length; i++) out[i] = f16BitsToF32(f32ToF16Bits(density[i] as number));
  return out;
}

/** The half bits of a grid; with `exact`, throws on the first value f16 cannot hold exactly (round with `ptRoundGridF16` first). */
export function ptF16Bits(density: ArrayLike<number>, exact = false): Uint16Array {
  const out = new Uint16Array(density.length);
  for (let i = 0; i < density.length; i++) {
    const v = density[i] as number;
    const h = f32ToF16Bits(v);
    if (exact && f16BitsToF32(h) !== Math.fround(v)) throw new RangeError(`volumeGrid r16float: value ${v} at ${i} is not an f16 value; round the grid with ptRoundGridF16 and compute densityMax / the brick majorants from the rounded grid`);
    out[i] = h;
  }
  return out;
}
