/**
 * render-until-check (V2.5a): `PtMegakernel.renderUntil` on a cloud scene with `integrator.volumes.outputs`.
 * Renders until the relative standard error of the image mean is below `se` (default 1e-3) or `max` samples, and reports the samples used, the achieved SE, the image mean and the
 * history (the SE after each batch times sqrt(samples) should be flat: 1/sqrt(N) scaling).
 * V2.5b: `adaptive=1` renders the same target twice, uniform then with per-pixel adaptive sampling (`volumes.adaptive`, minSamples `min`=64), and reports samples traced,
 * wall time, the luminance mean of both and the relMSE between the two images; `upload=1` uploads the samples-per-pixel map and both radiance images as PFMs.
 * Args: scene=vol-cl-sph-g85-a99-t50  se=0.001  pixel=0 (per-pixel quantile target, 0 = off)  pixelq=0.99  max=16384  first=64  w= h= (film override)
 */
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';

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
    maxSamples: numberArg(args, 'max', 16384),
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
  const scale = r.history.map((h) => h.meanRelSe * Math.sqrt(h.samples));
  reportResult({
    scene: name, renderer: 'ours', config: { se, pixel, max: numberArg(args, 'max', 16384), first: numberArg(args, 'first', 64), width: o.width, height: o.height },
    valid: deviceErrors.length === 0, metrics,
    notes: { history: r.history.map((h) => `${h.samples}: se=${h.meanRelSe.toExponential(3)} px=${h.pixelRelSe.toExponential(2)} batch=${h.batchMs.toFixed(0)}ms`).join(' | '), se_times_sqrt_n: scale.map((x) => x.toExponential(3)).join(' '), reason: r.reason },
  }, { adapter: adapterSummary, deviceErrors });
  adapter.dispose();
} catch (e) {
  reportError(e);
}
