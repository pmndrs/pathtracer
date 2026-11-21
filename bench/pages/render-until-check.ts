/**
 * render-until-check (V2.5a): `PtMegakernel.renderUntil` on a cloud scene with `integrator.volumes.outputs`.
 * Renders until the relative standard error of the image mean is below `se` (default 1e-3) or `max` samples, and reports the samples used, the achieved SE, the image mean and the
 * history (the SE after each batch times sqrt(samples) should be flat: 1/sqrt(N) scaling).
 * V2.5b: `adaptive=1` renders the same target twice, uniform then with per-pixel adaptive sampling (`volumes.adaptive`, minSamples `min`=64), and reports samples traced,
 * wall time, the luminance mean of both and the relMSE between the two images; `upload=1` uploads the samples-per-pixel map and both radiance images as PFMs.
 * V2.5c: `bias=1` measures the stopping bias of the adaptive variants: one uniform reference (`refspp`, default 8192, seed 1000), then every variant of `variants=` (default
 * uniform,min64,min256,min512,alloc,robust) over `seeds` (default 6) independent seeds on ONE adaptive kernel; per variant: total samples (pilot included), wall ms, the image mean luminance
 * over the seeds and relBias = (mean - reference) / reference with its standard error (sd over seeds / sqrt(seeds); the reference's own SE is reported once). `pixel=<target>` is required,
 * `uni` (default 256) = spp of the `uniform` control, `abatch` (default 64) = batch of the min* / robust variants, `pilot` (64), `budgetscale` (1).
 * Args: scene=vol-cl-sph-g85-a99-t50  se=0.001  pixel=0 (per-pixel quantile target, 0 = off)  pixelq=0.99  max=16384  first=64  w= h= (film override)
 */
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { ptRenderAllocated, type PtStopOptions } from '../../packages/pathtracer/src/integrator/render-until.ts';

let adapterSummary: AdapterSummary | undefined;
try {
  const args = getArgs();
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const gpu = await navigator.gpu.requestAdapter();
  if (!gpu) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(gpu);
  const name = args.scene ?? 'vol-cl-sph-g85-a99-t50';
  let scene = getScene(name);
  if (args.w && args.h) scene = { ...scene, film: { ...scene.film, width: numberArg(args, 'w', 0), height: numberArg(args, 'h', 0) } };
  const se = numberArg(args, 'se', 1e-3);
  const pixel = numberArg(args, 'pixel', 0);
  const adapter = new PtMegakernelAdapter({ passesPerSubmit: 8, volumeOutputs: true });
  await adapter.load(scene);
  const deviceErrors = watchDevice(adapter.gpuContext!.device);
  const target = {
    ...(se > 0 ? { meanRelSe: se } : {}),
    ...(pixel > 0 ? { pixelRelSe: { below: pixel, quantile: numberArg(args, 'pixelq', 0.99) } } : {}),
    maxSamples: args.bias === '1' ? 64 : numberArg(args, 'max', 16384),
    firstBatch: numberArg(args, 'first', 64),
  };
  const kernel = await adapter.createKernel();
  const t0 = performance.now();
  const r = await kernel.renderUntil(target);
  const wallMs = performance.now() - t0;
  const o = r.outputs;
  let mean = 0;
  for (let i = 0; i < o.width * o.height; i++) mean += (0.2126 * (o.radiance[i * 3] as number) + 0.7152 * (o.radiance[i * 3 + 1] as number) + 0.0722 * (o.radiance[i * 3 + 2] as number)) / (o.width * o.height);
  const metrics: Record<string, number> = {
    samples: r.samples,
    met: r.met ? 1 : 0,
    achieved_mean_rel_se: r.achieved.meanRelSe,
    achieved_pixel_rel_se: r.achieved.pixelRelSe,
    image_mean_luminance: mean,
    wall_ms: wallMs,
  };
  const lum = (o: { radiance: Float32Array }, i: number) => 0.2126 * (o.radiance[i * 3] as number) + 0.7152 * (o.radiance[i * 3 + 1] as number) + 0.0722 * (o.radiance[i * 3 + 2] as number);
  if (args.adaptive === '1') {
    if (pixel <= 0) throw new Error('adaptive=1 needs pixel=<target>');
    const minSamples = numberArg(args, 'min', 64);
    const base = (adapter as unknown as { lastKernelOptions: { integrator: { volumes: object } } }).lastKernelOptions.integrator.volumes;
    const ak = await adapter.createKernel({ integrator: { volumes: { ...base, adaptive: true } } as never });
    const t1 = performance.now();
    const ar = await ak.renderUntil({ ...target, adaptive: true, minSamples, firstBatch: numberArg(args, 'abatch', minSamples) });
    const adaptiveMs = performance.now() - t1;
    const ao = ar.outputs, pp = ar.perPixel!;
    const px = o.width * o.height;
    let sumU = 0, sumA = 0, mse = 0, ref2 = 0;
    for (let i = 0; i < px; i++) { const u = lum(o, i), a = lum(ao, i); sumU += u; sumA += a; mse += (a - u) * (a - u); ref2 += u * u; }
    // expected relMSE of two independent renders at the same SE: sum of both per-pixel variances / sum of mean^2
    let var2 = 0;
    for (let i = 0; i < px; i++) { const su = (0.2126 * (o.stdError[i * 3] as number) + 0.7152 * (o.stdError[i * 3 + 1] as number) + 0.0722 * (o.stdError[i * 3 + 2] as number)); const sa = (0.2126 * (ao.stdError[i * 3] as number) + 0.7152 * (ao.stdError[i * 3 + 1] as number) + 0.0722 * (ao.stdError[i * 3 + 2] as number)); var2 += su * su + sa * sa; }
    const uniformTotal = r.samples * px;
    Object.assign(metrics, {
      uniform_total_samples: uniformTotal, adaptive_total_samples: pp.total, adaptive_saving: 1 - pp.total / uniformTotal,
      adaptive_spp_min: pp.min, adaptive_spp_mean: pp.mean, adaptive_spp_max: pp.max, adaptive_passes: ar.samples, adaptive_met: ar.met ? 1 : 0,
      adaptive_done_fraction: pp.doneFraction, adaptive_wall_ms: adaptiveMs,
      adaptive_image_mean_luminance: sumA / px, uniform_image_mean_luminance: sumU / px, mean_rel_diff: (sumA - sumU) / sumU,
      rel_mse_adaptive_vs_uniform: mse / ref2, rel_mse_expected_from_se: var2 / ref2,
      adaptive_achieved_pixel_rel_se: ar.achieved.pixelRelSe,
    });
    if (args.upload === '1') {
      await uploadImage('spp-map', { width: ao.width, height: ao.height, channels: 3, data: Float32Array.from({ length: px * 3 }, (_, k) => (ao.sampleCount[Math.floor(k / 3)] as number)) });
      await uploadImage('uniform', { width: o.width, height: o.height, channels: 3, data: o.radiance });
      await uploadImage('adaptive', { width: ao.width, height: ao.height, channels: 3, data: ao.radiance });
    }
  }
  if (args.bias === '1') {
    if (pixel <= 0) throw new Error('bias=1 needs pixel=<target>');
    const base = (adapter as unknown as { lastKernelOptions: { integrator: { volumes: object } } }).lastKernelOptions.integrator.volumes;
    const ak = await adapter.createKernel({ integrator: { volumes: { ...base, adaptive: true } } as never });
    const device = adapter.gpuContext!.device;
    const restart = async (seed: number) => { const e = device.createCommandEncoder(); ak.reset(e); device.queue.submit([e.finish()]); await device.queue.onSubmittedWorkDone(); ak.setSeed(seed); };
    const kernelLike = { renderUntil: (o: PtStopOptions) => ak.renderUntil(o), restart };
    const px = o.width * o.height;
    const imageMean = (x: { radiance: Float32Array }) => { let t = 0; for (let i = 0; i < px; i++) t += lum(x, i); return t / px; };
    const refSpp = numberArg(args, 'refspp', 8192);
    await restart(1000);
    const rr = await ak.renderUntil({ pixelRelSe: { below: 0 }, firstBatch: refSpp, maxSamples: refSpp });
    const refMean = imageMean(rr.outputs);
    let se2 = 0;
    for (let i = 0; i < px; i++) { const s1 = 0.2126 * (rr.outputs.stdError[i * 3] as number) + 0.7152 * (rr.outputs.stdError[i * 3 + 1] as number) + 0.0722 * (rr.outputs.stdError[i * 3 + 2] as number); se2 += s1 * s1; }
    const refSe = Math.sqrt(se2) / px;
    metrics.ref_mean_luminance = refMean; metrics.ref_rel_se = refSe / refMean; metrics.ref_spp = refSpp;
    const seeds = numberArg(args, 'seeds', 6), uni = numberArg(args, 'uni', 256), abatch = numberArg(args, 'abatch', 64), pilot = numberArg(args, 'pilot', 64);
    const q = { below: pixel, quantile: numberArg(args, 'pixelq', 0.99) };
    const maxS = numberArg(args, 'max', 16384);  // per-pixel cap of the variants (the initial uniform run is capped at 64 in bias mode)
    const variants = (args.variants ?? 'uniform,min64,min256,min512,alloc,robust').split(',');
    const table: string[] = [];
    for (const v of variants) {
      const means: number[] = [], totals: number[] = [], walls: number[] = [];
      for (let k = 0; k < seeds; k++) {
        await restart(1 + k);
        const t = performance.now();
        let img: { radiance: Float32Array }, total: number;
        if (v === 'uniform') { const r1 = await ak.renderUntil({ pixelRelSe: { below: 0 }, firstBatch: uni, maxSamples: uni }); img = r1.outputs; total = uni * px; }
        else if (v === 'alloc') {
          const r1 = await ptRenderAllocated(kernelLike, { pixelRelSe: q, pilotSamples: pilot, minSamples: numberArg(args, 'amin', 16), maxSamples: maxS, firstBatch: numberArg(args, 'agrid', 16), seed: 5000 + k, budgetScale: numberArg(args, 'budgetscale', 1), dilate: args.dilate !== '0' });
          img = r1.outputs; total = r1.totalWithPilot;
        } else {
          const min = v.startsWith('min') ? Number(v.slice(3)) : numberArg(args, 'rmin', 64);
          const extra: Partial<PtStopOptions> = v === 'robust' ? { consecutive: numberArg(args, 'consec', 2), maxBatchRatio: numberArg(args, 'ratio', 3) } : {};
          const r1 = await ak.renderUntil({ pixelRelSe: q, adaptive: true, minSamples: min, firstBatch: abatch, maxSamples: maxS, ...extra });
          img = r1.outputs; total = r1.perPixel!.total;
        }
        walls.push(performance.now() - t); totals.push(total); means.push(imageMean(img));
      }
      const mu = means.reduce((a, b) => a + b, 0) / seeds;
      const sd = Math.sqrt(means.reduce((a, b) => a + (b - mu) * (b - mu), 0) / Math.max(1, seeds - 1));
      const relBias = (mu - refMean) / refMean, relBiasSe = sd / Math.sqrt(seeds) / refMean;
      const tot = totals.reduce((a, b) => a + b, 0) / seeds, wall = walls.reduce((a, b) => a + b, 0) / seeds;
      Object.assign(metrics, { [`${v}_total_samples`]: tot, [`${v}_spp_mean`]: tot / px, [`${v}_wall_ms`]: wall, [`${v}_rel_bias`]: relBias, [`${v}_rel_bias_se`]: relBiasSe, [`${v}_seed_sd_rel`]: sd / refMean });
      table.push(`${v}: spp=${(tot / px).toFixed(1)} wall=${wall.toFixed(0)}ms relBias=${(relBias * 100).toFixed(3)}% +- ${(relBiasSe * 100).toFixed(3)}% (z=${(relBias / Math.max(1e-12, Math.hypot(relBiasSe, refSe / refMean))).toFixed(1)})`);
    }
    (globalThis as Record<string, unknown>).__biasTable = table;
    console.log(table.join('\n'));
  }
  const scale = r.history.map((h) => h.meanRelSe * Math.sqrt(h.samples));
  reportResult({
    scene: name, renderer: 'ours', config: { se, pixel, max: numberArg(args, 'max', 16384), first: numberArg(args, 'first', 64), width: o.width, height: o.height },
    valid: deviceErrors.length === 0, metrics,
    notes: { history: r.history.map((h) => `${h.samples}: se=${h.meanRelSe.toExponential(3)} px=${h.pixelRelSe.toExponential(2)} batch=${h.batchMs.toFixed(0)}ms`).join(' | '), se_times_sqrt_n: scale.map((x) => x.toExponential(3)).join(' '), reason: r.reason, ...((globalThis as Record<string, unknown>).__biasTable ? { bias_table: ((globalThis as Record<string, unknown>).__biasTable as string[]).join(' || ') } : {}) },
  }, { adapter: adapterSummary, deviceErrors });
  adapter.dispose();
} catch (e) {
  reportError(e);
}
