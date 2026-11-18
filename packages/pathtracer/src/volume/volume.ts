/**
 * Participating media (P5): the data model, the TypeScript twins of the WGSL estimators (`volume-wgsl.ts`) and the closed forms the checks compare with.
 * docs/design/volumes.md has the derivations; DIVERGES V240+ the departures from pbrt-v4's `VolPathIntegrator`.
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
  /** Henyey-Greenstein mean cosine in (-1, 1). */
  g: number;
  grid?: PtMediumGrid;
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
export function transmittance(m: PtMedium, o: VolRgb, d: VolRgb, len: number, rand: () => number, density: DensityFn = () => 1): [number, number, number] {
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
export function slabSingleScatter(sigmaT: number, albedo: number, g: number, thickness: number, mu: number, L = 1, steps = 400): number {
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
    const nmu = 200, nphi = 64;
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

/** Fraction of a sphere of radius R (interface boundary, uniform sky, albedo 1, any phase) that reaches the camera: 1 (energy conservation); the radiance equals the sky. */
export const FURNACE_RADIANCE = 1;
