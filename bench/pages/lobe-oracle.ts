/**
 * lobe-oracle (P2.25 fuzz, P2.27 thin film): our GPU megakernel against a brute-force CPU integral of the same lobe, on a sphere under a
 * uniform unit environment with one scattering vertex (maxDepth 1, no other light). Then pixel = the directional albedo of the BSDF at the
 * view angle, `integral f(wo, wi) cos(wi) dwi`, which the page integrates on a (theta, phi) grid with the TypeScript twin and compares per
 * view-angle bin (pixels binned by mu_o = n . v, analytic for the sphere). Correctness only, no timing.
 *
 * Thin film (P2.27): the smooth cases are the Fresnel itself (a black base, one delta reflection), against the exact spectral Airy sum of
 * `bench/metrics/thinfilm-airy.ts` under D65; the rough case integrates GGX x the exact film Fresnel at the half-vector angle.
 *
 * Args: spp=256 (per case)  bins=24  tol=0.02 (fuzz relative deviation per bin, mu_o > 0.2)  filmtol=0.012 (film absolute error; metal 5x)  fuzz=1  film=1
 */
import { createMaterial, type RtMaterialInput } from '@pmndrs/raytracer';
import {
  PT_BSDF_LAYERED, ptBsdfEval, ptBsdfFromOpenPbr, ptLayeredEval, ptLayeredFromBsdf, ptFuzzBaseOf, ptFuzzedEval, ptMaterialFeatures,
  type PtBsdfEval, type PtBsdfParams, type PtFuzzBase, type PtFuzzParams, type PtMaterialOptions, type PtVec3,
} from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult } from '../runner/page-client.ts';
import type { OpenPbrParams, SceneDesc } from '../scenes/schema.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { airyRgb } from '../metrics/thinfilm-airy.ts';
import { ptGgxD, ptGgxG } from '@pmndrs/pathtracer';

const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string) => {
  if (problems.length < 30) problems.push(m);
};
let adapterSummary: AdapterSummary | null = null;

/** A sphere (radius 1) 4 m from the camera under a unit sky; yFov 30 degrees fills the frame with the sphere. */
export function sphereScene(name: string, material: OpenPbrParams, size = 96): SceneDesc {
  return {
    version: 1,
    name,
    purpose: 'Lobe oracle sphere: pixel = directional albedo at the view angle.',
    tier: 1,
    camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: size, height: size, filter: 'box' },
    integrator: { maxDepth: 1 },
    materials: { m: { type: 'openpbr' as const, ...material } },
    meshes: [{ name: 'ball', geometry: { type: 'sphere', radius: 1, segments: 128, rings: 64 }, material: 'm' }],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
  };
}

/** Per pixel mu_o = n . v of the analytic unit sphere seen from (0, 0, 4) with yFov 30; NaN for a miss. */
export function pixelMu(px: number, py: number, size: number): number {
  const t = Math.tan((15 * Math.PI) / 180);
  const x = ((px + 0.5) / size) * 2 - 1;
  const y = 1 - ((py + 0.5) / size) * 2;
  let dx = x * t;
  let dy = y * t;
  let dz = -1;
  const l = Math.hypot(dx, dy, dz);
  dx /= l; dy /= l; dz /= l;
  // |o + s d|^2 = 1 with o = (0, 0, 4)
  const b = 4 * dz;
  const disc = b * b - 15;
  if (disc < 0) return NaN;
  const s = -b - Math.sqrt(disc);
  const nx = s * dx;
  const ny = s * dy;
  const nz = 4 + s * dz;
  return -(nx * dx + ny * dy + nz * dz);
}

/** Integral over the +z hemisphere of f(wi) cos by a midpoint rule in (theta, phi); `refine` crowds samples toward the pole-aligned mirror lobe for narrow lobes. */
export function albedoRgb(evalF: (wi: PtVec3) => PtBsdfEval, nt = 360, np = 360): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < nt; i++) {
    const th = ((i + 0.5) / nt) * (Math.PI / 2);
    const st = Math.sin(th);
    const ct = Math.cos(th);
    for (let j = 0; j < np; j++) {
      const ph = ((j + 0.5) / np) * 2 * Math.PI;
      const e = evalF([st * Math.cos(ph), st * Math.sin(ph), ct]);
      const w = ct * st;
      out[0] += e.f[0] * w; out[1] += e.f[1] * w; out[2] += e.f[2] * w;
    }
  }
  const k = ((Math.PI / 2) * 2 * Math.PI) / (nt * np);
  return [out[0] * k, out[1] * k, out[2] * k];
}

interface Bins { mu: number[]; rgb: [number, number, number][]; count: number[] }

/** Mean GPU pixel per mu_o bin (bins uniform in mu from 0.05 to 1). */
export function binImage(data: Float32Array, size: number, bins: number): Bins {
  const sum: [number, number, number][] = Array.from({ length: bins }, () => [0, 0, 0]);
  const muSum = new Array<number>(bins).fill(0);
  const count = new Array<number>(bins).fill(0);
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const mu = pixelMu(px, py, size);
      if (!(mu > 0.05)) continue;
      // keep away from the silhouette's pixel-edge mixing: require the four corners to hit as well
      const corners = [pixelMu(px - 0.5, py, size), pixelMu(px + 0.5, py, size), pixelMu(px, py - 0.5, size), pixelMu(px, py + 0.5, size)];
      if (corners.some((c) => !(c > 0.03))) continue;
      const b = Math.min(bins - 1, Math.floor(((mu - 0.05) / 0.95) * bins));
      const i = (py * size + px) * 4;
      sum[b]![0] += data[i] as number; sum[b]![1] += data[i + 1] as number; sum[b]![2] += data[i + 2] as number;
      muSum[b]! += mu;
      count[b]!++;
    }
  }
  return { mu: muSum.map((m, i) => m / Math.max(1, count[i] as number)), rgb: sum.map((s, i) => s.map((v) => v / Math.max(1, count[i] as number)) as [number, number, number]), count };
}

// ---- fuzz ---------------------------------------------------------------------------------------------------------------

interface FuzzCase { name: string; material: OpenPbrParams; rt: RtMaterialInput }

const FUZZ_CASES: FuzzCase[] = [
  { name: 'white-over-grey-lambert', material: { baseColor: [0.5, 0.5, 0.5], specularWeight: 0, fuzzWeight: 1, fuzzColor: [1, 1, 1], fuzzRoughness: 0.5 }, rt: { baseColor: [0.5, 0.5, 0.5], specularWeight: 0, sheenWeight: 1, sheenColor: [1, 1, 1], sheenRoughness: 0.5 } },
  { name: 'coloured-w0.7-over-dark', material: { baseColor: [0.2, 0.2, 0.2], specularWeight: 0, fuzzWeight: 0.7, fuzzColor: [0.9, 0.4, 0.1], fuzzRoughness: 0.25 }, rt: { baseColor: [0.2, 0.2, 0.2], specularWeight: 0, sheenWeight: 0.7, sheenColor: [0.9, 0.4, 0.1], sheenRoughness: 0.25 } },
  { name: 'over-black-base', material: { baseColor: [0, 0, 0], specularWeight: 0, fuzzWeight: 1, fuzzColor: [0.3, 0.6, 0.9], fuzzRoughness: 0.8 }, rt: { baseColor: [0, 0, 0], specularWeight: 0, sheenWeight: 1, sheenColor: [0.3, 0.6, 0.9], sheenRoughness: 0.8 } },
  { name: 'over-layered-dielectric', material: { baseColor: [0.6, 0.3, 0.2], specularRoughness: 0.35, fuzzWeight: 0.8, fuzzColor: [0.8, 0.8, 1], fuzzRoughness: 0.4 }, rt: { baseColor: [0.6, 0.3, 0.2], specularRoughness: 0.35, sheenWeight: 0.8, sheenColor: [0.8, 0.8, 1], sheenRoughness: 0.4 } },
  { name: 'over-rough-metal', material: { baseColor: [0.9, 0.6, 0.3], baseMetalness: 1, specularRoughness: 0.5, fuzzWeight: 0.6, fuzzColor: [1, 1, 1], fuzzRoughness: 0.6 }, rt: { baseColor: [0.9, 0.6, 0.3], metalness: 1, specularRoughness: 0.5, sheenWeight: 0.6, sheenColor: [1, 1, 1], sheenRoughness: 0.6 } },
];

function cpuBase(rt: RtMaterialInput, options: PtMaterialOptions): PtFuzzBase {
  const m = createMaterial(rt);
  const p: PtBsdfParams = ptBsdfFromOpenPbr(m, options, 0);
  if (p.kind === PT_BSDF_LAYERED) {
    const lp = ptLayeredFromBsdf(p);
    return { eval: (wo, wi) => ptLayeredEval(lp, wo, wi, { layering: 'albedo' }), sample: () => { throw new Error('unused'); }, black: false };
  }
  return ptFuzzBaseOf(p, ptBsdfEval, () => { throw new Error('unused'); });
}

async function checkFuzz(spp: number, bins: number, tol: number): Promise<void> {
  const size = 96;
  for (const c of FUZZ_CASES) {
    const adapter = new PtMegakernelAdapter({ layering: 'albedo' });
    await adapter.load(sphereScene(`fuzz-${c.name}`, c.material, size));
    const result = await adapter.render({ spp });
    const features = ptMaterialFeatures([createMaterial(c.rt)]);
    const options: PtMaterialOptions = { features, layered: { layering: 'albedo' } };
    const base = cpuBase(c.rt, options);
    const fz: PtFuzzParams = { weight: c.rt.sheenWeight as number, color: c.rt.sheenColor as PtVec3, roughness: c.rt.sheenRoughness as number };
    const g = binImage(result.image.data, size, bins);
    let worst = 0;
    let sumDev = 0;
    let n = 0;
    for (let b = 0; b < bins; b++) {
      if ((g.count[b] as number) < 15 || (g.mu[b] as number) < 0.2) continue;
      const mu = g.mu[b] as number;
      const wo: PtVec3 = [Math.sqrt(1 - mu * mu), 0, mu];
      const ref = albedoRgb((wi) => ptFuzzedEval(fz, base, wo, wi));
      for (let k = 0; k < 3; k++) {
        const want = ref[k] as number;
        const got = (g.rgb[b] as number[])[k] as number;
        const dev = (got - want) / Math.max(want, 0.02);
        worst = Math.max(worst, Math.abs(dev));
        sumDev += dev;
        n++;
      }
    }
    metrics[`fuzz_${c.name}_max_rel_dev`] = worst;
    metrics[`fuzz_${c.name}_mean_rel_dev`] = sumDev / Math.max(1, n);
    if (worst > tol) {
      metrics[`fuzz_${c.name}_mismatch`] = 1;
      note(`fuzz ${c.name}: max rel dev ${worst.toFixed(4)} > ${tol}`);
    } else metrics[`fuzz_${c.name}_mismatch`] = 0;
    adapter.dispose();
  }
}

// ---- thin film ----------------------------------------------------------------------------------------------------------

interface FilmCase {
  name: string;
  material: OpenPbrParams;
  /** The base under the film for the exact spectral Airy sum: real IOR n (dielectric) or complex n + i k (metal). */
  baseN: number;
  baseK: number;
  filmIor: number;
  thickness: number;
  weight: number;
  /** GGX alpha of the lobe (0 = smooth: the pixel is the Fresnel itself). */
  alpha: number;
}

const filmCase = (name: string, d: number, extra: Partial<FilmCase> = {}, material: OpenPbrParams = {}): FilmCase => ({
  name,
  material: { baseWeight: 0, baseColor: [0, 0, 0], specularRoughness: 0, specularIor: 1.5, thinFilmWeight: extra.weight ?? 1, thinFilmThickness: d, thinFilmIor: extra.filmIor ?? 1.4, ...material },
  baseN: 1.5, baseK: 0, filmIor: 1.4, thickness: d, weight: 1, alpha: 0, ...extra,
});

const FILM_CASES: FilmCase[] = [
  ...[0.15, 0.3, 0.6, 1.0].map((d) => filmCase(`smooth-dielectric-d${d}`, d)),
  filmCase('smooth-dielectric-d0.3-n1.8', 0.3, { filmIor: 1.8 }),
  filmCase('smooth-dielectric-d0.3-w0.5', 0.3, { weight: 0.5 }),
  ...[0.2, 0.5].map((d) => filmCase(`smooth-grey-metal-d${d}`, d, { baseN: 1, baseK: 6 }, { baseWeight: 1, baseColor: [0.9, 0.9, 0.9], baseMetalness: 1, specularRoughness: 0 })),
  filmCase('rough-dielectric-d0.3-r0.3', 0.3, { alpha: 0.09 }, { specularRoughness: 0.3 }),
];

/** CPU brute force for a rough lobe: f = F_exact(h . wo) D G / (4 cos_i cos_o) with the exact spectral Airy Fresnel tabulated on cos. */
function roughFilmAlbedo(c: FilmCase, mu: number, table: [number, number, number][]): [number, number, number] {
  const wo: PtVec3 = [Math.sqrt(1 - mu * mu), 0, mu];
  const a: [number, number] = [c.alpha, c.alpha];
  const F = (cos: number): [number, number, number] => {
    const x = Math.min(1, Math.max(0, cos)) * (table.length - 1);
    const i = Math.min(table.length - 2, Math.floor(x));
    const t = x - i;
    const lo = table[i] as number[];
    const hi = table[i + 1] as number[];
    const f = (k: number) => (lo[k] as number) * (1 - t) + (hi[k] as number) * t;
    const w = c.weight;
    const plain = fresnelPlain(cos, 1.5);
    return [plain + w * (f(0) - plain), plain + w * (f(1) - plain), plain + w * (f(2) - plain)];
  };
  return albedoRgb((wi) => {
    const hx = wi[0] + wo[0];
    const hy = wi[1] + wo[1];
    const hz = wi[2] + wo[2];
    const l = Math.hypot(hx, hy, hz);
    const wm: PtVec3 = [hx / l, hy / l, hz / l];
    const d = ptGgxD(a, wm);
    const g = ptGgxG(a, wo, wi);
    const k = (d * g) / (4 * wi[2] * wo[2]);
    const f = F(wo[0] * wm[0] + wo[1] * wm[1] + wo[2] * wm[2]);
    return { f: [f[0] * k, f[1] * k, f[2] * k], pdf: 0 };
  }, 720, 360);
}

function fresnelPlain(c: number, n: number): number {
  const s2 = (1 - c * c) / (n * n);
  const ct = Math.sqrt(1 - s2);
  const rp = (n * c - ct) / (n * c + ct);
  const rs = (c - n * ct) / (c + n * ct);
  return 0.5 * (rp * rp + rs * rs);
}

async function checkFilm(spp: number, tol: number): Promise<void> {
  const size = 96;
  const bins = 48;
  for (const c of FILM_CASES) {
    const adapter = new PtMegakernelAdapter({ layering: 'albedo' });
    await adapter.load(sphereScene(`film-${c.name}`, c.material, size));
    const result = await adapter.render({ spp: c.alpha > 0 ? spp : Math.max(16, spp / 8) });
    const g = binImage(result.image.data, size, bins);
    let table: [number, number, number][] = [];
    if (c.alpha > 0) table = Array.from({ length: 201 }, (_, i) => airyRgb({ filmIor: c.filmIor, thicknessUm: c.thickness, baseN: c.baseN, baseK: c.baseK }, i / 200));
    let worstAbs = 0;
    let worstNear = 0;
    let sumAbs = 0;
    let n = 0;
    let worstRel = 0;
    for (let b = 0; b < bins; b++) {
      if ((g.count[b] as number) < 10 || (g.mu[b] as number) < 0.2) continue;
      const mu = g.mu[b] as number;
      let want: [number, number, number];
      if (c.alpha > 0) want = roughFilmAlbedo(c, mu, table);
      else {
        const film = airyRgb({ filmIor: c.filmIor, thicknessUm: c.thickness, baseN: c.baseN, baseK: c.baseK }, mu);
        const plain = c.baseK > 0 ? [0, 1, 2].map(() => (c.baseK * c.baseK) / (4 + c.baseK * c.baseK)) : [fresnelPlain(mu, 1.5), fresnelPlain(mu, 1.5), fresnelPlain(mu, 1.5)];
        want = [0, 1, 2].map((k) => (plain[k] as number) + c.weight * ((film[k] as number) - (plain[k] as number))) as [number, number, number];
      }
      for (let k = 0; k < 3; k++) {
        const got = (g.rgb[b] as number[])[k] as number;
        const err = Math.abs(got - (want[k] as number));
        worstAbs = Math.max(worstAbs, err);
        if (mu > 0.55) worstNear = Math.max(worstNear, err);
        worstRel = Math.max(worstRel, err / Math.max(want[k] as number, 0.02));
        sumAbs += err;
        n++;
      }
    }
    metrics[`film_${c.name}_max_abs`] = worstAbs;
    metrics[`film_${c.name}_max_abs_mu55`] = worstNear;
    metrics[`film_${c.name}_mean_abs`] = sumAbs / Math.max(1, n);
    metrics[`film_${c.name}_max_rel`] = worstRel;
    // Gates: mean error over mu_o > 0.2 and the worst error where mu_o > 0.55. The Fourier model uses the unpolarised Fresnel inside the Airy sum
    // and, for a metal, a real effective IOR, so its error grows toward grazing (reported as max_abs, not gated) and is larger on metals.
    const limit = c.baseK > 0 ? tol * 5 : tol;
    const meanAbs = sumAbs / Math.max(1, n);
    if (meanAbs > limit || worstNear > limit) {
      metrics[`film_${c.name}_mismatch`] = 1;
      note(`film ${c.name}: mean abs err ${meanAbs.toFixed(4)}, near-normal max ${worstNear.toFixed(4)} > ${limit}`);
    } else metrics[`film_${c.name}_mismatch`] = 0;
    adapter.dispose();
  }
}

try {
  const args = getArgs();
  const spp = numberArg(args, 'spp', 256);
  const bins = numberArg(args, 'bins', 24);
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const gpu = await navigator.gpu.requestAdapter();
  if (!gpu) throw new Error('requestAdapter() returned null: no usable GPU.');
  const { summarizeAdapter } = await import('../runner/page-client.ts');
  adapterSummary = summarizeAdapter(gpu);
  if (numberArg(args, 'fuzz', 1)) await checkFuzz(spp, bins, numberArg(args, 'tol', 0.02));
  if (numberArg(args, 'film', 1)) await checkFilm(spp, numberArg(args, 'filmtol', 0.012));
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult({ scene: 'lobe-oracle', renderer: 'ours', config: { spp, bins }, valid: total === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } }, { adapter: adapterSummary, deviceErrors: [] });
  if (total > 0) throw new Error(`lobe-oracle: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors: [] });
}
