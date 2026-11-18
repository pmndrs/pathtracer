/**
 * wf-p5-check (V270-V272): participating media and spectral transport on the wavefront scheduler against the megakernel, PER PIXEL, in ONE page (one hold of the shared GPU lock).
 *
 * For every case (a scene + adapter settings) the adapter's own kernel is the megakernel; each variant (`variants=`) is a `PtWavefront` made by `createWavefrontKernel` from the same
 * kernel options (same scene copy, density grid and spectral table, film, limits, seed). Both render `spp` passes (default 16) and are read back with `readImage`; the images are
 * compared pixel by pixel: the estimators draw the same random numbers (V270: the volume stream, the shadow ray's forked stream, the sampler dimensions per vertex), so the only
 * differences are float summation order and the compiler's choices. Gate `tol` (default 1e-5): a pixel's largest |difference| over its largest |component| (floor 1e-3 of the image mean;
 * `wf-frame-check`'s metric). Also reported, not gated: the per-channel relative difference with the same floor (the definition of `bench/adapters/pt-wavefront-agreement.ts`), which a
 * spectral pixel with one channel near zero amplifies (the lane -> RGB matrix cancels; the megakernel converts the summed lanes once, the wavefront converts each addend).
 * Reported per case and variant: max, mean, pixels over the tolerance and over 1e-3, bit-identical fraction, the
 * image means of both, non-finite pixels. A pixel where a path flipped a discrete decision (a delta-tracking accept, a roulette) because of a 1-ulp difference would show as a
 * large max with a tiny `over` count: the page lists `over` so that is visible.
 *
 * `timing=1` (BEST EFFORT, the quiet pass owns numbers): the same kernels timed ABBA, `rounds` rounds of `spp` passes per sample (one at a time, `onSubmittedWorkDone` between), megakernel and variants interleaved,
 * paired wavefront / megakernel ratios per round (min..max).
 *
 * Args: cases=vol-absorb-slab,...  variants=plain,inline,batch,ordered  spp=16  tol=1e-5  timing=0|1  rounds=5  packing=packed|split
 */
import type { PtMegakernel, PtWavefront, PtWavefrontOptions } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import { PtMegakernelAdapter, type PtAdapterSettings } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
const deviceErrors: string[] = [];
const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string): void => {
  if (problems.length < 60) problems.push(m);
};

interface Case {
  scene: string;
  settings?: PtAdapterSettings;
}

const SPECTRAL: PtAdapterSettings = { spectral: {} };
const CASES: Record<string, Case> = {
  // media: closed forms and scenes (docs/design/volumes.md)
  'vol-absorb-slab': { scene: 'vol-absorb-slab' },
  'vol-camera-medium': { scene: 'vol-camera-medium' },
  'vol-furnace-sphere': { scene: 'vol-furnace-sphere' },
  'vol-slab-single': { scene: 'vol-slab-single' },
  'vol-grid-slab': { scene: 'vol-grid-slab' },
  'vol-grid-slab-brick': { scene: 'vol-grid-slab', settings: { volumeMajorant: 'brick' } },
  'vol-cornell-smoke': { scene: 'vol-cornell-smoke' },
  'vol-cornell-smoke-ratio': { scene: 'vol-cornell-smoke', settings: { volumeShadow: 'ratio' } },
  'vol-milk-sphere': { scene: 'vol-milk-sphere' },
  'vol-grid-cloud': { scene: 'vol-grid-cloud' },
  'vol-vdb-smoke': { scene: 'vol-vdb-smoke' },
  'vol-vdb-smoke-brick': { scene: 'vol-vdb-smoke', settings: { volumeMajorant: 'brick' } },
  // spectral transport
  cornell: { scene: 'cornell', settings: SPECTRAL },
  'swatches-refl': { scene: 'spectral-swatches-refl', settings: SPECTRAL },
  'swatches-emit': { scene: 'spectral-swatches-emit', settings: SPECTRAL },
  'swatches-env': { scene: 'spectral-swatches-env', settings: SPECTRAL },
  'dispersion-prism': { scene: 'dispersion-prism', settings: SPECTRAL },
  'dispersion-test': { scene: 'DispersionTest-notex', settings: SPECTRAL },
  'metal-rough': { scene: 'MetalRoughSpheresNoTextures', settings: SPECTRAL },
  // the same scene in RGB (no spectral) and with spectral but no dispersion: separates a deep-path chaos of the scene from the spectral port (a 1-ulp difference flips a discrete choice)
  'dispersion-test-rgb': { scene: 'DispersionTest-notex' },
  'dispersion-test-nodisp': { scene: 'DispersionTest-notex', settings: { spectral: { dispersion: false } } },
};

const VARIANTS: Record<string, Partial<PtWavefrontOptions>> = {
  plain: {},
  inline: { refill: true },
  batch: { refill: 'batch' },
  ordered: { refill: 'ordered' },
  layers2: { refill: 'ordered', layers: 2 },
  subgroup: { appends: 'subgroup' },
  sorted: { sort: 'both' },
  refillsg: { refill: true, appends: 'subgroup' },
};

interface Agreement {
  /** Per-channel relative difference (floor 1e-3 of the image mean): the strict view, amplified where one channel is near zero (spectral: the XYZ -> RGB matrix cancels). */
  maxRel: number;
  /** Largest |difference| of a pixel over the largest |component| of the pixel (the metric of `wf-frame-check`): what the gate uses. */
  maxRelPx: number;
  overPx: number;
  meanRel: number;
  over: number;
  over1e3: number;
  same: number;
  total: number;
  nonFinite: number;
  meanA: number[];
  meanB: number[];
}

function compare(a: Float32Array, b: Float32Array, tol: number): Agreement {
  const px = a.length / 4;
  const meanA: number[] = [0, 0, 0];
  const meanB: number[] = [0, 0, 0];
  for (let i = 0; i < px; i++) {
    for (let c = 0; c < 3; c++) {
      const va = a[i * 4 + c] as number;
      const vb = b[i * 4 + c] as number;
      if (Number.isFinite(va) && Number.isFinite(vb)) {
        meanA[c] = (meanA[c] as number) + va / px;
        meanB[c] = (meanB[c] as number) + vb / px;
      }
    }
  }
  const floor = 1e-3 * Math.max(1e-6, ((meanA[0] as number) + (meanA[1] as number) + (meanA[2] as number)) / 3);
  const r: Agreement = { maxRel: 0, maxRelPx: 0, overPx: 0, meanRel: 0, over: 0, over1e3: 0, same: 0, total: 0, nonFinite: 0, meanA, meanB };
  let sum = 0;
  for (let i = 0; i < px; i++) {
    let worst = 0;
    let diffPx = 0;
    let sizePx = 0;
    let identical = true;
    for (let c = 0; c < 3; c++) {
      const va = a[i * 4 + c] as number;
      const vb = b[i * 4 + c] as number;
      if (!Number.isFinite(va) || !Number.isFinite(vb)) {
        r.nonFinite++;
        identical = false;
        continue;
      }
      if (!Object.is(va, vb)) identical = false;
      worst = Math.max(worst, Math.abs(vb - va) / Math.max(Math.abs(va), floor));
      diffPx = Math.max(diffPx, Math.abs(vb - va));
      sizePx = Math.max(sizePx, Math.abs(va), Math.abs(vb));
    }
    const relPx = diffPx / Math.max(sizePx, floor);
    r.maxRelPx = Math.max(r.maxRelPx, relPx);
    if (relPx > tol) r.overPx++;
    r.maxRel = Math.max(r.maxRel, worst);
    sum += worst;
    if (worst > tol) r.over++;
    if (worst > 1e-3) r.over1e3++;
    if (identical) r.same++;
    r.total++;
  }
  r.meanRel = sum / Math.max(1, px);
  return r;
}

/** `spp` passes of a kernel from a fresh accumulator, in submits of at most `perSubmit` passes; refill kernels drain with `finish()`. */
async function render(device: GPUDevice, kernel: PtMegakernel | PtWavefront, spp: number): Promise<void> {
  const perSubmit = Math.min(8, 'maxPassesPerEncode' in kernel ? kernel.maxPassesPerEncode() : Infinity);
  const reset = device.createCommandEncoder({ label: 'p5 reset' });
  kernel.reset(reset);
  device.queue.submit([reset.finish()]);
  let done = 0;
  while (done < spp) {
    const n = Math.min(perSubmit, spp - done);
    const e = device.createCommandEncoder({ label: 'p5 samples' });
    kernel.encode(e, n);
    device.queue.submit([e.finish()]);
    done += n;
  }
  if ('finish' in kernel) await kernel.finish();
  await device.queue.onSubmittedWorkDone();
}

const med = (v: number[]): number => [...v].sort((x, y) => x - y)[Math.floor(v.length / 2)] as number;

async function runCase(name: string, c: Case, variants: string[], o: { spp: number; tol: number; timing: boolean; rounds: number; packing: 'packed' | 'split' }): Promise<void> {
  const scene = getScene(c.scene);
  const adapter = new PtMegakernelAdapter({ scenePacking: o.packing, ...(c.settings ?? {}) });
  await adapter.load(scene);
  const mk = adapter.renderer as PtMegakernel;
  const device = adapter.gpuContext!.device;
  metrics[`${name}_megakernel_bytes`] = mk.wgslBytes;
  await render(device, mk, o.spp);
  const ref = (await mk.readImage()).data;
  for (const v of variants) {
    const opt = VARIANTS[v];
    if (!opt) throw new Error(`unknown variant '${v}' (have ${Object.keys(VARIANTS).join(', ')})`);
    const tag = `${name}_${v}`;
    const wf = await adapter.createWavefrontKernel({ wavefront: opt });
    for (const [k, bytes] of Object.entries(wf.kernelBytes)) metrics[`${tag}_bytes_${k}`] = bytes as number;
    metrics[`${tag}_path_state_bytes`] = wf.memory().stateBytesPerPath;
    await render(device, wf, o.spp);
    const img = (await wf.readImage()).data;
    const a = compare(ref, img, o.tol);
    metrics[`${tag}_rel_max`] = a.maxRel;
    metrics[`${tag}_relpx_max`] = a.maxRelPx;
    metrics[`${tag}_relpx_over_tol`] = a.overPx;
    metrics[`${tag}_rel_mean`] = a.meanRel;
    metrics[`${tag}_pixels_over_tol`] = a.over;
    metrics[`${tag}_pixels_over_1e-3`] = a.over1e3;
    metrics[`${tag}_bit_identical_frac`] = a.same / Math.max(1, a.total);
    metrics[`${tag}_non_finite`] = a.nonFinite;
    metrics[`${tag}_pixels`] = a.total;
    for (let k = 0; k < 3; k++) {
      metrics[`${tag}_mean_mk_${'rgb'[k]}`] = a.meanA[k] as number;
      metrics[`${tag}_mean_wf_${'rgb'[k]}`] = a.meanB[k] as number;
    }
    const bad = a.overPx > 0 || a.nonFinite > 0;
    metrics[`${tag}_mismatch`] = bad ? 1 : 0;
    if (bad) note(`${tag}: ${a.overPx} of ${a.total} pixels differ by more than ${o.tol} of their size (max ${a.maxRelPx}; per channel ${a.over} over, max ${a.maxRel}, over 1e-3: ${a.over1e3}), non-finite ${a.nonFinite}`);
    if (o.timing) {
      // ABBA: the megakernel, then the variant, then the variant, then the megakernel; `spp` passes per sample, one sample in flight
      const a1: number[] = [];
      const b1: number[] = [];
      const time = async (k: PtMegakernel | PtWavefront): Promise<number> => {
        const t0 = performance.now();
        await render(device, k, o.spp);
        return performance.now() - t0;
      };
      await time(mk);
      await time(wf);
      for (let r = 0; r < o.rounds; r++) {
        if (r % 2 === 0) {
          a1.push(await time(mk));
          b1.push(await time(wf));
        } else {
          b1.push(await time(wf));
          a1.push(await time(mk));
        }
      }
      const ratios = b1.map((x, i) => x / (a1[i] as number));
      metrics[`${tag}_time_mk_ms_per_spp_median`] = med(a1) / o.spp;
      metrics[`${tag}_time_wf_ms_per_spp_median`] = med(b1) / o.spp;
      metrics[`${tag}_time_ratio_wf_over_mk_min`] = Math.min(...ratios);
      metrics[`${tag}_time_ratio_wf_over_mk_median`] = med(ratios);
      metrics[`${tag}_time_ratio_wf_over_mk_max`] = Math.max(...ratios);
    }
    wf.destroy();
  }
  adapter.dispose();
}

try {
  const args = getArgs();
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const gpu = await navigator.gpu.requestAdapter();
  if (!gpu) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(gpu);
  const names = (args.cases ?? 'vol-absorb-slab,vol-furnace-sphere,vol-grid-slab,vol-cornell-smoke,vol-vdb-smoke,cornell,swatches-refl,dispersion-prism,dispersion-test').split(',');
  const variants = (args.variants ?? 'plain,inline,batch').split(',');
  const o = { spp: numberArg(args, 'spp', 16), tol: numberArg(args, 'tol', 1e-5), timing: numberArg(args, 'timing', 0) === 1, rounds: numberArg(args, 'rounds', 5), packing: (args.packing ?? 'packed') as 'packed' | 'split' };
  if (o.timing) metrics.timing_best_effort = 1;
  for (const n of names) {
    const c = CASES[n];
    if (!c) throw new Error(`unknown case '${n}' (have ${Object.keys(CASES).join(', ')})`);
    try {
      await runCase(n, c, variants, o);
    } catch (error) {
      metrics[`${n}_mismatch`] = 1;
      note(`${n}: ${(error as Error).message}`);
    }
  }
  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult({ scene: 'wf-p5-check', renderer: 'ours', config: { cases: names.join(','), variants: variants.join(','), spp: o.spp, tol: o.tol, packing: o.packing, timing: o.timing }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } }, { adapter: adapterSummary, deviceErrors });
  if (total > 0) throw new Error(`wf-p5-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
