/**
 * vol-forced-grid-check (V4.4, `integrator.volumes.forcedGrid`): unbiasedness and noise of the forced / equiangular estimate on GRID media, megakernel, in ONE page.
 * Each case = scene + estimator; K seeds x spp each. Per case: image mean (mean of the K seed means) with its standard error over the seeds, and the per-pixel variance of a single
 * seed's image (mean over pixels, optionally outside a mask of pixels brighter than `maskX` x the median: the lamp and its silhouette). Pairs (A, B): mean difference in SEs, and
 * chi = mean over pixels of (mean_A - mean_B)^2 / (se_A^2 + se_B^2) (about 1 when the images agree; heavy tails read below 1).
 * Args: cases=<scene>:<est>,...  (est: plain | forced | equi | forced-g | equi-g)  seeds=8  spp=128  pairs=0-1,2-3  maskX=20  indep=1 (0: every case uses the same seeds)
 * V7.6 density filters (est: stoch | brick | brick-stoch | r16 | r16-stoch | r16-hw | brick16 | brick16-stoch | brick16-hw): the cases are rendered INTERLEAVED, one seed of
 * every case per round with the order reversed every round (A B, B A, ...), so a pair's time ratio (`_time_ratio` = B / A, median of the per-round ratios, with min / max) is
 * paired. The time is the GPU work of one seed's render (submit to `onSubmittedWorkDone`; the read-back is outside it). Best effort unless the machine is quiet.
 * Seeds are HASHED before `setSeed` (`hashseeds=0`: the raw 1000 + 1000 case + s of before V7.6): `ptPixelSeed` is hash(x ^ hash(y ^ seed)), so two seeds whose XOR is below the
 * image height give the SAME per-pixel streams on permuted rows, their images are correlated and the standard error over such seeds is too small (a case against itself read z -5.5).
 * A case listed twice gets its own seeds and tags `..__<index>`: pair it with the first as a control (|z| ~ 1).
 */
import type { PtMegakernel } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
const metrics: Record<string, number> = {};
const problems: string[] = [];

async function render(device: GPUDevice, kernel: PtMegakernel, spp: number, seed: number): Promise<{ data: Float32Array; ms: number }> {
  kernel.setSeed(seed);
  const t0 = performance.now();
  const reset = device.createCommandEncoder();
  kernel.reset(reset);
  device.queue.submit([reset.finish()]);
  for (let done = 0; done < spp; done += 8) {
    const e = device.createCommandEncoder();
    kernel.encode(e, Math.min(8, spp - done));
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const ms = performance.now() - t0;
  return { data: (await kernel.readImage()).data, ms };
}

const SETTINGS: Record<string, object> = {
  plain: {},
  forced: { volumeForcedNee: true },
  equi: { volumeEquiangular: true },
  'forced-g': { volumeForcedNee: true, volumeForcedGrid: true },
  'equi-g': { volumeEquiangular: true, volumeForcedGrid: true },
  // V7.6: density filters. `r16*` share the r16float grid the hardware filter can use on every device; `brick*` add the brick majorant + ratio-tracking roulette.
  stoch: { volumeDensityFilter: 'stochastic' },
  brick: { volumeMajorant: 'brick', volumeTrRoulette: true },
  'brick-stoch': { volumeMajorant: 'brick', volumeTrRoulette: true, volumeDensityFilter: 'stochastic' },
  r16: { volumeGridFormat: 'r16float' },
  'r16-stoch': { volumeGridFormat: 'r16float', volumeDensityFilter: 'stochastic' },
  'r16-hw': { volumeGridFormat: 'r16float', volumeDensityFilter: 'hardware' },
  brick16: { volumeMajorant: 'brick', volumeTrRoulette: true, volumeGridFormat: 'r16float' },
  'brick16-stoch': { volumeMajorant: 'brick', volumeTrRoulette: true, volumeGridFormat: 'r16float', volumeDensityFilter: 'stochastic' },
  'brick16-hw': { volumeMajorant: 'brick', volumeTrRoulette: true, volumeGridFormat: 'r16float', volumeDensityFilter: 'hardware' },
};

interface CaseStats { tag: string; px: number; mean: Float64Array; se2: Float64Array; imgMean: number; imgSe: number; pixVar: number; pixVarMasked: number; ms: number; times: number[] }
/** lowbias32 (`rtHash32`): spreads consecutive seeds over all 32 bits. */
const hashSeed = (v: number): number => {
  let x = v >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  return (x ^ (x >>> 16)) >>> 0;
};
const median = (a: readonly number[]): number => { const b = [...a].sort((x, y) => x - y); return b.length % 2 ? b[(b.length - 1) / 2]! : (b[b.length / 2 - 1]! + b[b.length / 2]!) / 2; };

try {
  const args = getArgs();
  const gpu = await navigator.gpu?.requestAdapter();
  if (!gpu) throw new Error('no WebGPU adapter');
  adapterSummary = summarizeAdapter(gpu);
  const spp = numberArg(args, 'spp', 128);
  const seeds = numberArg(args, 'seeds', 8);
  const maskX = numberArg(args, 'maskX', 20);
  const cases = (args.cases ?? 'vol-cornell-smoke:equi,vol-cornell-smoke-grid:equi-g').split(',');
  const stats: CaseStats[] = [];
  const hashSeeds = args.hashseeds !== '0';
  const indep = args.indep !== '0'; // independent seeds per case (default): same seeds correlate the cases and the z / chi of a pair are no longer honest
  const live: Array<{ ci: number; name: string; est: string; adapter: PtMegakernelAdapter; imgs: Float32Array[]; times: number[] }> = [];
  for (const [ci, c] of cases.entries()) {
    const [name, est] = c.split(':') as [string, string];
    const settings = SETTINGS[est];
    if (!settings) throw new Error(`unknown estimator ${est}`);
    const adapter = new PtMegakernelAdapter(settings as never);
    await adapter.load(getScene(name));
    live.push({ ci, name, est, adapter, imgs: [], times: [] });
  }
  for (let s = 0; s < seeds; s++) {
    for (const c of s % 2 === 0 ? live : [...live].reverse()) {
      const r = await render(c.adapter.gpuContext!.device, c.adapter.renderer as PtMegakernel, spp, hashSeeds ? hashSeed(1000 + (indep ? 1000 * c.ci : 0) + s) : 1000 + (indep ? 1000 * c.ci : 0) + s);
      c.imgs.push(r.data);
      c.times.push(r.ms);
    }
  }
  for (const { ci, name, est, adapter, imgs, times } of live) {
    const ms = times.reduce((a, b) => a + b, 0) / seeds;
    const px = imgs[0]!.length / 4;
    const lum = (img: Float32Array, i: number): number => ((img[i * 4] as number) + (img[i * 4 + 1] as number) + (img[i * 4 + 2] as number)) / 3;
    const mean = new Float64Array(px), se2 = new Float64Array(px), pv = new Float64Array(px);
    const imgMeans: number[] = [];
    for (const img of imgs) {
      let m = 0;
      for (let i = 0; i < px; i++) m += lum(img, i) / px;
      imgMeans.push(m);
    }
    for (let i = 0; i < px; i++) {
      let s1 = 0, s2 = 0;
      for (const img of imgs) { const v = lum(img, i); s1 += v; s2 += v * v; }
      const m = s1 / seeds;
      const v = Math.max(0, (s2 - seeds * m * m) / (seeds - 1));
      mean[i] = m; pv[i] = v; se2[i] = v / seeds;
    }
    const sorted = Array.from(mean).sort((a, b) => a - b);
    const medianLum = sorted[Math.floor(px / 2)]!;
    let pixVar = 0, pixVarMasked = 0, nMasked = 0;
    for (let i = 0; i < px; i++) {
      pixVar += pv[i]! / px;
      if (mean[i]! <= maskX * medianLum) { pixVarMasked += pv[i]!; nMasked++; }
    }
    pixVarMasked /= Math.max(1, nMasked);
    const imgMean = imgMeans.reduce((a, b) => a + b, 0) / seeds;
    const imgSe = Math.sqrt(imgMeans.reduce((a, b) => a + (b - imgMean) ** 2, 0) / (seeds - 1) / seeds);
    const tag = cases.indexOf(cases[ci]!) === ci ? `${name}__${est}` : `${name}__${est}__${ci}`; // a case listed twice (its own seeds): a control pair of one estimator against itself
    stats.push({ tag, px, mean, se2, imgMean, imgSe, pixVar, pixVarMasked, ms, times });
    metrics[`${tag}_mean`] = imgMean; metrics[`${tag}_se`] = imgSe; metrics[`${tag}_pixvar`] = pixVar; metrics[`${tag}_pixvar_masked`] = pixVarMasked;
    metrics[`${tag}_masked_frac`] = nMasked / px; metrics[`${tag}_ms_per_seed`] = ms;
    metrics[`${tag}_ms_median`] = median(times); metrics[`${tag}_ms_min`] = Math.min(...times); metrics[`${tag}_ms_max`] = Math.max(...times);
    let nonFinite = 0;
    for (const img of imgs) for (let i = 0; i < img.length; i++) if (!Number.isFinite(img[i]!)) nonFinite++;
    metrics[`${tag}_non_finite`] = nonFinite;
    if (nonFinite > 0) problems.push(`${tag}: ${nonFinite} non-finite values`);
    adapter.dispose();
  }
  const pairs = (args.pairs ?? '0-1').split(',').filter(Boolean).map((p) => p.split('-').map(Number) as [number, number]);
  for (const [a, b] of pairs) {
    const A = stats[a]!, B = stats[b]!;
    const z = (A.imgMean - B.imgMean) / Math.hypot(A.imgSe, B.imgSe);
    let chi = 0;
    for (let i = 0; i < A.px; i++) { const d = A.mean[i]! - B.mean[i]!; const s = A.se2[i]! + B.se2[i]!; if (s > 0) chi += (d * d) / s / A.px; }
    const key = `${A.tag}_vs_${B.tag}`;
    metrics[`${key}_z`] = z; metrics[`${key}_rel`] = (B.imgMean - A.imgMean) / A.imgMean; metrics[`${key}_chi`] = chi;
    metrics[`${key}_pixvar_ratio`] = A.pixVar / B.pixVar; metrics[`${key}_pixvar_masked_ratio`] = A.pixVarMasked / B.pixVarMasked;
    const ratios = A.times.map((t, i) => B.times[i]! / t);
    metrics[`${key}_time_ratio`] = median(ratios); metrics[`${key}_time_ratio_min`] = Math.min(...ratios); metrics[`${key}_time_ratio_max`] = Math.max(...ratios);
  }
  reportResult({ scene: 'vol-forced-grid-check', renderer: 'ours', config: { spp, seeds, cases: cases.join(','), maskX }, valid: problems.length === 0, metrics, notes: { problems: problems.join(' | ') } });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors: [] });
}
