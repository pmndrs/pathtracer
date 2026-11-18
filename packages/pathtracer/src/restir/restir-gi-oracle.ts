/**
 * CPU oracle of the ReSTIR GI arithmetic (`restir-gi-wgsl.ts`: `ptRestirGiInit`'s candidate, `ptGiPhat`, `ptGiMerge`) on a toy world with a closed-form answer.
 *
 * World: the plane z = 0 over the box |a|, |b| <= L (the reconnection vertices x2 = (a, b, 0), normal +z, outgoing radiance `lo(a, b)`), and pixels (domains) x1 = (px, py, h)
 * with a Lambert BSDF (f = 1/pi), shading normal -z. The target of domain d for the sample y is the WGSL one, `F_d(y) = f cos1 Lo G V`, `G = cos(phi2) / r^2`, and the exact answer
 * for domain d is a quadrature of `F_d` over the box. A candidate is a cosine-weighted direction (pdf cos / pi) that lands in the box, with `W = r^2 / (pdf cos(phi2))`.
 * The region `a > diffuseLimit` stands for a non-Lambert x2: its samples are not shiftable (the WGSL `diffuse` flag).
 */
import { oracleRng } from './restir-oracle.ts';

export interface GiPixel {
  x: number;
  y: number;
  h: number;
}

export interface GiWorld {
  half: number;
  diffuseLimit: number;
  rmin: number;
  lo: (a: number, b: number) => number;
  /** Visibility of the plane point (a, b) from a pixel: any function with values in [0, 1]. */
  vis: (p: GiPixel, a: number, b: number) => number;
}

export interface GiSample {
  a: number;
  b: number;
  w: number;
  m: number;
  /** Domain the sample was generated in. */
  own: number;
}

const distance = (p: GiPixel, a: number, b: number): number => Math.hypot(a - p.x, b - p.y, p.h);

/** `ptGiTarget` (F, a scalar: Lambert f = 1/pi, luminance = the value): f cos1 Lo G V. */
export function giTarget(world: GiWorld, p: GiPixel, a: number, b: number, withVis: boolean): number {
  const r = distance(p, a, b);
  const c = p.h / r; // cos1 = cos(phi2): both normals are +-z
  const v = withVis ? world.vis(p, a, b) : 1;
  return ((1 / Math.PI) * c * world.lo(a, b) * (c / (r * r))) * v;
}

/** `ptGiPhat`: the sample of domain `s.own` evaluated at domain `at`. */
export function giPhat(world: GiWorld, pixels: readonly GiPixel[], s: GiSample, at: number): number {
  if (!(s.w > 0)) return 0;
  if (at === s.own) return giTarget(world, pixels[at] as GiPixel, s.a, s.b, false);
  const diffuse = s.a <= world.diffuseLimit;
  if (!diffuse || distance(pixels[s.own] as GiPixel, s.a, s.b) < world.rmin || distance(pixels[at] as GiPixel, s.a, s.b) < world.rmin) return 0;
  return giTarget(world, pixels[at] as GiPixel, s.a, s.b, true);
}

/** The initial candidate of `ptRestirGiInit`: one cosine-sampled direction from the pixel; empty (w = 0, m = 1) when it leaves the box or is occluded (the traced ray would hit something else). */
export function giInit(world: GiWorld, pixels: readonly GiPixel[], own: number, rand: () => number): GiSample {
  const p = pixels[own] as GiPixel;
  const u1 = rand();
  const u2 = rand();
  const sinT = Math.sqrt(u1);
  const cosT = Math.sqrt(1 - u1);
  const phi = 2 * Math.PI * u2;
  const t = p.h / cosT;
  const a = p.x + t * sinT * Math.cos(phi);
  const b = p.y + t * sinT * Math.sin(phi);
  const empty: GiSample = { a: 0, b: 0, w: 0, m: 1, own };
  if (Math.abs(a) > world.half || Math.abs(b) > world.half) return empty;
  if (!(world.vis(p, a, b) > 0)) return empty;
  const r = Math.hypot(a - p.x, b - p.y, p.h);
  const c2 = p.h / r;
  const pdf = cosT / Math.PI;
  return { a, b, w: (r * r) / (pdf * c2), m: 1, own };
}

/** `ptGiMerge`: domain 0 is the destination. Returns the selected sample with its contribution weight (w = 0 when nothing survived). */
export function giMerge(world: GiWorld, pixels: readonly GiPixel[], ys: readonly GiSample[], unbiased: boolean, rand: () => number, cap = Infinity): GiSample {
  const n = ys.length;
  // `ys[i].own` indexes `pixels`; the domain list of the merge is `ys` order, so remap own -> merge index through the pixel identity.
  const at = (i: number): number => i;
  const ph: number[][] = ys.map((y, i) => Array.from({ length: n }, (_, j) => (unbiased || j === 0 ? giPhat(world, pixels, { ...y, own: i }, at(j)) : 0)));
  let wsum = 0;
  let mtot = 0;
  let out: GiSample = { a: 0, b: 0, w: 0, m: 0, own: 0 };
  let selPh = 0;
  for (let i = 0; i < n; i++) {
    const y = ys[i] as GiSample;
    mtot += y.m;
    const p0 = (ph[i] as number[])[0] as number;
    if (!(p0 > 0)) continue;
    let w = p0 * y.w * y.m;
    if (unbiased) {
      let den = 0;
      for (let j = 0; j < n; j++) if ((ys[j] as GiSample).m > 0) den += (ys[j] as GiSample).m * ((ph[i] as number[])[j] as number);
      w = den > 0 ? ((y.m * ((ph[i] as number[])[i] as number)) / den) * p0 * y.w : 0;
    }
    wsum += w;
    if (rand() * wsum < w) {
      out = y;
      selPh = p0;
    }
  }
  const w = selPh > 0 && wsum > 0 ? (unbiased ? wsum / selPh : wsum / (mtot * selPh)) : 0;
  return { ...out, w, m: Math.min(mtot, cap) };
}

/** Quadrature of the exact answer for pixel `d`: the integral of F_d over the box. */
export function giTruth(world: GiWorld, p: GiPixel, grid = 600): number {
  let sum = 0;
  const cell = (2 * world.half) / grid;
  for (let i = 0; i < grid; i++) {
    for (let j = 0; j < grid; j++) {
      const a = -world.half + (i + 0.5) * cell;
      const b = -world.half + (j + 0.5) * cell;
      sum += giTarget(world, p, a, b, true) * cell * cell;
    }
  }
  return sum;
}

export { oracleRng };
