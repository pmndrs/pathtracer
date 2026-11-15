/**
 * lod-check (R1, `integrator.materialLod: 'secondary'`): the reduced material against the full one, on the white furnace, and the wavefront's second
 * shade kernel against the megakernel's run-time branch.
 *
 * Furnace: a sphere under a uniform unit sky with one scattering vertex (maxDepth 1, as `lobe-oracle`), so a pixel is the directional albedo of the
 * BSDF at the view angle. With `materialLodDepth: 0` every vertex is reduced, so the same scene rendered `full` and `secondary` compares the two
 * materials' albedo per view angle (pixels binned by mu_o, analytic for the sphere). The full material is the reference mode (stochastic layered walk;
 * the thin-film case falls back to the albedo layering as the adapter does). A reduced material that neither brightens nor darkens reads a mean
 * relative deviation near 0 and a small per-bin deviation (the reduced BSDF is a stochastic lobe pick, so it is noisier: use spp >= 512).
 *
 * Scheduler agreement: the reduced render by the wavefront (a second shade kernel from depth 0 on) against the megakernel (a run-time branch), same seed:
 * per-pixel relMSE must be float-noise only.
 *
 * Args: spp=1024  bins=16  tol=0.04 (max per-bin relative deviation, mu_o > 0.25)  meantol=0.02 (|mean rel deviation|)  wf=1 (also run the wavefront)
 */
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult } from '../runner/page-client.ts';
import type { OpenPbrParams, SceneDesc } from '../scenes/schema.ts';
import { PtMegakernelAdapter, type PtAdapterSettings } from '../adapters/pt-megakernel/index.ts';

const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string) => {
  if (problems.length < 30) problems.push(m);
};
let adapterSummary: AdapterSummary | null = null;


/** A sphere (radius 1) 4 m from the camera under a unit sky; yFov 30 degrees fills the frame (as `lobe-oracle`, which is not imported: it runs on load). */
function sphereScene(name: string, material: OpenPbrParams, size: number): SceneDesc {
  return {
    version: 1,
    name,
    purpose: 'R1 furnace sphere: pixel = directional albedo at the view angle.',
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
function pixelMu(px: number, py: number, size: number): number {
  const t = Math.tan((15 * Math.PI) / 180);
  const x = ((px + 0.5) / size) * 2 - 1;
  const y = 1 - ((py + 0.5) / size) * 2;
  let dx = x * t;
  let dy = y * t;
  let dz = -1;
  const l = Math.hypot(dx, dy, dz);
  dx /= l;
  dy /= l;
  dz /= l;
  const b = 4 * dz;
  const disc = b * b - 15;
  if (disc < 0) return NaN;
  const s = -b - Math.sqrt(disc);
  const nx = s * dx;
  const ny = s * dy;
  const nz = 4 + s * dz;
  return -(nx * dx + ny * dy + nz * dz);
}

interface Case {
  name: string;
  material: OpenPbrParams;
}

const CASES: Case[] = [
  { name: 'grey-r0.1', material: { baseColor: [0.8, 0.8, 0.8], specularRoughness: 0.1 } },
  { name: 'grey-r0.3', material: { baseColor: [0.8, 0.8, 0.8], specularRoughness: 0.3 } },
  { name: 'grey-r0.6', material: { baseColor: [0.8, 0.8, 0.8], specularRoughness: 0.6 } },
  { name: 'red-r0.3', material: { baseColor: [0.8, 0.2, 0.1], specularRoughness: 0.3 } },
  { name: 'dark-r0.3', material: { baseColor: [0.1, 0.1, 0.1], specularRoughness: 0.3 } },
  { name: 'coat-r0.05', material: { baseColor: [0.6, 0.3, 0.2], specularRoughness: 0.8, coatWeight: 1, coatRoughness: 0.05, coatIor: 1.6 } },
  { name: 'coat-r0.3', material: { baseColor: [0.6, 0.3, 0.2], specularRoughness: 0.8, coatWeight: 1, coatRoughness: 0.3, coatIor: 1.6 } },
  { name: 'coated-gold', material: { baseColor: [1, 0.78, 0.34], baseMetalness: 1, specularRoughness: 0.2, coatWeight: 1, coatRoughness: 0.05, coatIor: 1.5 } },
  { name: 'coated-gold-rough', material: { baseColor: [1, 0.78, 0.34], baseMetalness: 1, specularRoughness: 0.4, coatWeight: 1, coatRoughness: 0.3, coatIor: 1.5 } },
  { name: 'fuzz-white-over-grey', material: { baseColor: [0.5, 0.5, 0.5], specularWeight: 0, fuzzWeight: 1, fuzzColor: [1, 1, 1], fuzzRoughness: 0.5 } },
  { name: 'fuzz-over-layered', material: { baseColor: [0.6, 0.3, 0.2], specularRoughness: 0.35, fuzzWeight: 0.8, fuzzColor: [0.8, 0.8, 1], fuzzRoughness: 0.4 } },
  { name: 'film-dielectric', material: { baseColor: [0.9, 0.9, 0.9], specularRoughness: 0.2, thinFilmWeight: 1, thinFilmThickness: 0.4, thinFilmIor: 1.4 } },
  { name: 'metal-mix-0.3', material: { baseColor: [0.9, 0.6, 0.2], baseMetalness: 0.3, specularRoughness: 0.4 } },
];

interface Binned {
  mu: number[];
  rgb: [number, number, number][];
  count: number[];
  mean: [number, number, number];
}

function bin(data: Float32Array, size: number, bins: number): Binned {
  const sum = Array.from({ length: bins }, () => [0, 0, 0] as [number, number, number]);
  const muSum = new Array<number>(bins).fill(0);
  const count = new Array<number>(bins).fill(0);
  const mean: [number, number, number] = [0, 0, 0];
  let n = 0;
  for (let py = 0; py < size; py++) {
    for (let px = 0; px < size; px++) {
      const mu = pixelMu(px, py, size);
      if (!(mu > 0.05)) continue;
      const b = Math.min(bins - 1, Math.floor(((mu - 0.05) / 0.95) * bins));
      const i = (py * size + px) * 4;
      for (let c = 0; c < 3; c++) {
        (sum[b] as number[])[c] = ((sum[b] as number[])[c] as number) + (data[i + c] as number);
        mean[c] = (mean[c] as number) + (data[i + c] as number);
      }
      muSum[b] = (muSum[b] as number) + mu;
      count[b] = (count[b] as number) + 1;
      n++;
    }
  }
  return { mu: muSum.map((m, i) => m / Math.max(1, count[i] as number)), rgb: sum.map((s, i) => s.map((v) => v / Math.max(1, count[i] as number)) as [number, number, number]), count, mean: mean.map((v) => v / Math.max(1, n)) as [number, number, number] };
}

async function render(settings: PtAdapterSettings, name: string, material: OpenPbrParams, size: number, spp: number): Promise<{ data: Float32Array; stats: Record<string, number> }> {
  const adapter = new PtMegakernelAdapter(settings);
  await adapter.load(sphereScene(name, material, size));
  const result = await adapter.render({ spp });
  adapter.dispose();
  return { data: result.image.data, stats: (result.stats ?? {}) as Record<string, number> };
}

try {
  const args = getArgs();
  const spp = numberArg(args, 'spp', 1024);
  const bins = numberArg(args, 'bins', 16);
  const tol = numberArg(args, 'tol', 0.04);
  const meanTol = numberArg(args, 'meantol', 0.02);
  const wavefront = numberArg(args, 'wf', 1) !== 0;
  const size = 96;
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const gpu = await navigator.gpu.requestAdapter();
  if (!gpu) throw new Error('requestAdapter() returned null: no usable GPU.');
  const { summarizeAdapter } = await import('../runner/page-client.ts');
  adapterSummary = summarizeAdapter(gpu);
  const base: PtAdapterSettings = { seed: 7 };
  // The reduced material is independent of the kernel's layering mode. It models the albedo layering with an internal base (the D24 library default), so
  // that is the gated comparison; the stochastic walk (reference mode) is reported next to it: the gap between the two is P2.32b's, not the reduction's.
  const compare = (c: string, tag: string, a: Binned, b: Binned, gate: boolean): void => {
    let meanDev = 0;
    let worstMean = 0;
    for (let k = 0; k < 3; k++) {
      const dev = ((b.mean[k] as number) - (a.mean[k] as number)) / Math.max(a.mean[k] as number, 1e-3);
      meanDev += dev / 3;
      worstMean = Math.max(worstMean, Math.abs(dev));
      metrics[`${c}_${tag}_mean_rel_dev_${'rgb'[k]}`] = dev;
    }
    let worstBin = 0;
    for (let i = 0; i < bins; i++) {
      if ((a.count[i] as number) < 15 || (a.mu[i] as number) < 0.25) continue;
      for (let k = 0; k < 3; k++) {
        const want = (a.rgb[i] as number[])[k] as number;
        const got = (b.rgb[i] as number[])[k] as number;
        worstBin = Math.max(worstBin, Math.abs(got - want) / Math.max(want, 0.03));
      }
    }
    metrics[`${c}_${tag}_mean_rel_dev`] = meanDev;
    metrics[`${c}_${tag}_worst_channel_mean_dev`] = worstMean;
    metrics[`${c}_${tag}_bin_max_dev`] = worstBin;
    if (!gate) return;
    const bad = worstMean > meanTol || worstBin > tol;
    metrics[`${c}_mismatch`] = (metrics[`${c}_mismatch`] ?? 0) || bad ? 1 : 0;
    if (bad) note(`${c}: vs ${tag}: worst channel mean dev ${worstMean.toFixed(4)} (tol ${meanTol}), worst bin dev ${worstBin.toFixed(4)} (tol ${tol})`);
  };
  for (const c of CASES) {
    const fullAlbedo = await render({ ...base, layering: 'albedo', materialLod: 'full' }, `lod-full-a-${c.name}`, c.material, size, spp);
    const fullWalk = await render({ ...base, layering: 'stochastic', materialLod: 'full' }, `lod-full-s-${c.name}`, c.material, size, spp);
    const reduced = await render({ ...base, layering: 'albedo', materialLod: 'secondary', materialLodDepth: 0 }, `lod-red-${c.name}`, c.material, size, spp);
    const r = bin(reduced.data, size, bins);
    compare(c.name, 'vs_albedo', bin(fullAlbedo.data, size, bins), r, true);
    compare(c.name, 'vs_walk', bin(fullWalk.data, size, bins), r, false);
    compare(c.name, 'albedo_vs_walk', bin(fullWalk.data, size, bins), bin(fullAlbedo.data, size, bins), false);
    metrics[`${c.name}_mismatch`] ??= 0;

    if (wavefront) {
      const wf = await render({ ...base, layering: 'albedo', materialLod: 'secondary', materialLodDepth: 0, scheduler: 'wavefront' }, `lod-wf-${c.name}`, c.material, size, spp);
      let se = 0;
      let ss = 0;
      for (let i = 0; i < reduced.data.length; i += 4) {
        for (let k = 0; k < 3; k++) {
          const d = (wf.data[i + k] as number) - (reduced.data[i + k] as number);
          se += d * d;
          ss += (reduced.data[i + k] as number) ** 2;
        }
      }
      const relMse = se / Math.max(ss, 1e-12);
      metrics[`${c.name}_wf_vs_mk_relmse`] = relMse;
      metrics[`${c.name}_wf_shade_bytes`] = wf.stats.kernel_bytes_shade ?? 0;
      metrics[`${c.name}_wf_shade_secondary_bytes`] = wf.stats.kernel_bytes_shadeSecondary ?? 0;
      if (relMse > 1e-9) {
        metrics[`${c.name}_mismatch`] = 1;
        note(`${c.name}: wavefront vs megakernel (reduced) relMSE ${relMse.toExponential(2)}`);
      }
    }
  }
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult({ scene: 'lod-check', renderer: 'ours', config: { spp, bins, tol, meanTol }, valid: true, metrics: { ...metrics, mismatches: total }, notes: { problems } }, { adapter: adapterSummary, deviceErrors: [] });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors: [] });
}
