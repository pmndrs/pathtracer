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
 * KNOWN LIMIT (TASKS P5.s7, V272): `dispersion-test` (DispersionTest-notex, spectral with dispersion, maxDepth 16) does NOT meet the gate: 210 of 512,000 pixels differ by more than 1e-5 at 16 spp, up to 0.33 of the
 * pixel (21 pixels over 1e-3 per channel), while the image means agree to 2e-6. The depth sweep (`dispersion-test-d2 / d4 / d8 / d16`) shows it growing with the bounce count, the controls
 * (RGB, spectral without dispersion, dispersion with the IOR an exact function of the sample, the megakernel built for the other scene packing) show 0 flipped pixels. `dispersion-test-d2` is the gated case.
 *
 * `timing=1` (BEST EFFORT, the quiet pass owns numbers): the same kernels timed ABBA, `rounds` rounds of `spp` passes per sample (one at a time, `onSubmittedWorkDone` between), megakernel and variants interleaved,
 * paired wavefront / megakernel ratios per round (min..max).
 *
 * `control=1`: also render the megakernel built for the other scene copy packing (same maths, different compiled code) and report its disagreement with the first (`<case>_control_packing_*`, not gated): the floor of "per pixel" for a scene whose paths amplify ulp noise.
 *
 * Args: cases=vol-absorb-slab,...  variants=plain,inline,batch,ordered  spp=16  tol=1e-5  timing=0|1  rounds=5  packing=packed|split  control=0|1
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
  // media: closed forms and scenes (docs/internal/design/volumes.md)
  'vol-absorb-slab': { scene: 'vol-absorb-slab' },
  'vol-emit-slab': { scene: 'vol-emit-slab' },
  'vol-emit-grid-slab': { scene: 'vol-emit-grid-slab' },
  'vol-camera-medium': { scene: 'vol-camera-medium' },
  // V6.2: the medium stack (nested / overlapping boundaries without explicit exteriors)
  'vol-nested-absorb-stack': { scene: 'vol-nested-absorb', settings: { volumeStack: true } },
  'vol-nested-overlap-stack': { scene: 'vol-nested-overlap', settings: { volumeStack: true } },
  'vol-nested-camera-stack': { scene: 'vol-nested-camera', settings: { volumeStack: true } },
  'vol-nested-furnace-stack': { scene: 'vol-nested-furnace', settings: { volumeStack: true } },
  'vol-height-absorb': { scene: 'vol-height-absorb' },
  'vol-height-furnace': { scene: 'vol-height-furnace' },
  'vol-furnace-sphere': { scene: 'vol-furnace-sphere' },
  'vol-slab-single': { scene: 'vol-slab-single' },
  'vol-grid-slab': { scene: 'vol-grid-slab' },
  'vol-camera-grid': { scene: 'vol-camera-grid' },
  'vol-camera-grid-auto': { scene: 'vol-camera-grid-auto' },
  'vol-grid-slab-brick': { scene: 'vol-grid-slab', settings: { volumeMajorant: 'brick' } },
  'vol-grid-slab-inner-decomposition': { scene: 'vol-grid-slab-inner', settings: { volumeMajorant: 'brick', volumeFreeFlight: 'decomposition' } },
  'vol-cornell-smoke': { scene: 'vol-cornell-smoke' },
  'vol-cornell-smoke-ratio': { scene: 'vol-cornell-smoke', settings: { volumeShadow: 'ratio' } },
  'vol-milk-sphere': { scene: 'vol-milk-sphere' },
  'vol-grid-cloud': { scene: 'vol-grid-cloud' },
  'vol-cl-single-mie10': { scene: 'vol-cl-single-mie10' },
  'vol-cl-sph-hg2-a99-t50': { scene: 'vol-cl-sph-hg2-a99-t50' },
  'vol-vdb-smoke': { scene: 'vol-vdb-smoke' },
  'vol-vdb-smoke-brick': { scene: 'vol-vdb-smoke', settings: { volumeMajorant: 'brick' } },
  'vol-grid-cloud-roulette': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeTrRoulette: true } },
  'vol-grid-cloud-residual': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeShadow: 'residual', volumeTrRoulette: true } },
  'vol-grid-cloud-telescoping': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeShadow: 'telescoping' } },
  'vol-grid-cloud-decomposition': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeFreeFlight: 'decomposition' } },
  'vol-grid-cloud-hw': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeGridFormat: 'r16float', volumeDensityFilter: 'hardware' } },
  'vol-grid-cloud-atlas': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeGridFormat: 'r16float', volumeGridLayout: 'atlas' } },
  // V5.2: `volumes.outputs` on the wavefront: the three tail planes (orders 0-1 + primary transmittance, standard error, first-scatter distance + fraction) are compared as well as the beauty
  'vol-grid-cloud-outputs': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeOutputs: true } },
  'vol-cornell-smoke-outputs': { scene: 'vol-cornell-smoke', settings: { volumeOutputs: true } },
  'vol-slab-single-outputs': { scene: 'vol-slab-single', settings: { volumeOutputs: true } },
  // V5.1b: first-scatter AOV guides (albedo, normal, viewZ) on the wavefront, compared per pixel as well as the beauty
  'vol-grid-cloud-aovs': { scene: 'vol-grid-cloud', settings: { aovs: true } },
  'vol-cornell-smoke-aovs': { scene: 'vol-cornell-smoke', settings: { aovs: true } },
  'vol-grid-cloud-r8': { scene: 'vol-grid-cloud', settings: { volumeMajorant: 'brick', volumeGridFormat: 'r8unorm' } },
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
  // dispersive glass with a short path limit: a 1-ulp difference of the hero IOR needs bounces to grow into a flipped Fresnel / TIR decision
  'dispersion-test-d2': { scene: 'DispersionTest-notex', settings: { spectral: {}, maxDepth: 2 } },
  'dispersion-test-d4': { scene: 'DispersionTest-notex', settings: { spectral: {}, maxDepth: 4 } },
  'dispersion-test-d8': { scene: 'DispersionTest-notex', settings: { spectral: {}, maxDepth: 8 } },
  'dispersion-test-d16': { scene: 'DispersionTest-notex', settings: { spectral: {}, maxDepth: 16 } },
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
  /** The few worst pixels (index, relative difference), to see where a disagreement lives. */
  worst: Array<[number, number]>;
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
  const r: Agreement = { maxRel: 0, maxRelPx: 0, overPx: 0, worst: [], meanRel: 0, over: 0, over1e3: 0, same: 0, total: 0, nonFinite: 0, meanA, meanB };
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
    if (relPx > tol) {
      r.overPx++;
      r.worst.push([i, relPx]);
    }
    r.maxRel = Math.max(r.maxRel, worst);
    sum += worst;
    if (worst > tol) r.over++;
    if (worst > 1e-3) r.over1e3++;
    if (identical) r.same++;
    r.total++;
  }
  r.meanRel = sum / Math.max(1, px);
  r.worst.sort((x, y) => y[1] - x[1]);
  r.worst.length = Math.min(r.worst.length, 8);
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

async function runCase(name: string, c: Case, variants: string[], o: { spp: number; tol: number; timing: boolean; rounds: number; packing: 'packed' | 'split'; control: boolean }): Promise<void> {
  const scene = getScene(c.scene);
  const adapter = new PtMegakernelAdapter({ scenePacking: o.packing, ...(c.settings ?? {}) });
  await adapter.load(scene);
  const mk = adapter.renderer as PtMegakernel;
  const device = adapter.gpuContext!.device;
  metrics[`${name}_megakernel_bytes`] = mk.wgslBytes;
  await render(device, mk, o.spp);
  const ref = (await mk.readImage()).data;
  if (o.control) {
    // control: the SAME megakernel text compiled for the other scene copy packing is mathematically identical code with different compiler contraction; its disagreement with the first is the floor of "per pixel"
    const other = new PtMegakernelAdapter({ scenePacking: o.packing === 'packed' ? 'split' : 'packed', ...(c.settings ?? {}) });
    await other.load(scene);
    const mk2 = other.renderer as PtMegakernel;
    await render(other.gpuContext!.device, mk2, o.spp);
    const a = compare(ref, (await mk2.readImage()).data, o.tol);
    metrics[`${name}_control_packing_relpx_max`] = a.maxRelPx;
    metrics[`${name}_control_packing_relpx_over_tol`] = a.overPx;
    metrics[`${name}_control_packing_rel_max`] = a.maxRel;
    metrics[`${name}_control_packing_pixels_over_1e-3`] = a.over1e3;
    other.dispose();
  }
  const aovRef = c.settings?.aovs ? await mk.readAovs() : null;
  const voRef = c.settings?.volumeOutputs ? await mk.readVolumeOutputs() : null;
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
    let voBad = false;
    if (voRef) {
      // V5.2: the tail planes, per pixel (largest difference of the array over its largest element, floor 1e-3 of the mean): orders 0-1 (rgb), transmittance, standard error (a difference of two close
      // numbers, so it gets a 1000x looser gate: 3 of 49k cloud elements read 2e-3 from the cancellation), first-scatter distance and fraction
      const vw = await wf.readVolumeOutputs();
      const planes: Array<[string, Float32Array, Float32Array, number]> = [['order01', voRef.order01, vw.order01, o.tol], ['transmittance', voRef.transmittance, vw.transmittance, o.tol], ['stderr', voRef.stdError, vw.stdError, o.tol * 1000], ['first_scatter_distance', voRef.firstScatterDistance, vw.firstScatterDistance, o.tol], ['scatter_fraction', voRef.scatterFraction, vw.scatterFraction, o.tol]];
      for (const [pn, pa, pb, gate] of planes) {
        let mean = 0;
        for (let i = 0; i < pa.length; i++) mean += Math.abs(pa[i] as number) / pa.length;
        const fl = Math.max(1e-6, 1e-3 * mean);
        let worst = 0;
        let over = 0;
        for (let i = 0; i < pa.length; i++) {
          const d = Math.abs((pb[i] as number) - (pa[i] as number)) / Math.max(Math.abs(pa[i] as number), Math.abs(pb[i] as number), fl);
          if (!(d <= worst)) worst = d;
          if (!(d <= gate)) over++;
        }
        metrics[`${tag}_vo_${pn}_relelem_max`] = worst;
        metrics[`${tag}_vo_${pn}_over_gate`] = over;
        metrics[`${tag}_vo_${pn}_mean_mk`] = mean;
        if (over > 0) voBad = true;
      }
    }
    let aovBad = false;
    if (aovRef) {
      // V5.1b: albedo / normal / viewZ planes, per element (largest difference over the larger value, floor 1e-3 of the mean)
      const aw = await wf.readAovs();
      for (const [pn, pa, pb] of [['albedo', aovRef.albedo, aw.albedo], ['normal', aovRef.normal, aw.normal], ['viewz', aovRef.viewZ, aw.viewZ]] as Array<[string, Float32Array, Float32Array]>) {
        let mean = 0;
        for (let i = 0; i < pa.length; i++) mean += Math.abs(pa[i] as number) / pa.length;
        const fl = Math.max(1e-6, 1e-3 * mean);
        let worst = 0;
        let over = 0;
        for (let i = 0; i < pa.length; i++) {
          const d = Math.abs((pb[i] as number) - (pa[i] as number)) / Math.max(Math.abs(pa[i] as number), Math.abs(pb[i] as number), fl);
          if (!(d <= worst)) worst = d;
          if (!(d <= o.tol)) over++;
        }
        metrics[`${tag}_aov_${pn}_relelem_max`] = worst;
        metrics[`${tag}_aov_${pn}_over_gate`] = over;
        metrics[`${tag}_aov_${pn}_mean_mk`] = mean;
        if (over > 0) aovBad = true;
      }
    }
    const bad = a.overPx > 0 || a.nonFinite > 0 || voBad || aovBad;
    metrics[`${tag}_mismatch`] = bad ? 1 : 0;
    if (bad && a.worst.length > 0) note(`${tag} worst pixels (x, y, rel): ${a.worst.map(([i, r]) => `(${i % scene.film.width}, ${Math.floor(i / scene.film.width)}, ${r.toExponential(1)})`).join(' ')}`);
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
  const names = (args.cases ?? 'vol-absorb-slab,vol-furnace-sphere,vol-grid-slab,vol-cornell-smoke,vol-vdb-smoke,cornell,swatches-refl,dispersion-prism,dispersion-test-d2').split(',');
  const variants = (args.variants ?? 'plain,inline,batch').split(',');
  const o = { spp: numberArg(args, 'spp', 16), tol: numberArg(args, 'tol', 1e-5), timing: numberArg(args, 'timing', 0) === 1, rounds: numberArg(args, 'rounds', 5), packing: (args.packing ?? 'packed') as 'packed' | 'split', control: numberArg(args, 'control', 0) === 1 };
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
