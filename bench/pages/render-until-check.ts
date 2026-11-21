/**
 * render-until-check (V2.5a): `PtMegakernel.renderUntil` on a cloud scene with `integrator.volumes.outputs`.
 * Renders until the relative standard error of the image mean is below `se` (default 1e-3) or `max` samples, and reports the samples used, the achieved SE, the image mean and the
 * history (the SE after each batch times sqrt(samples) should be flat: 1/sqrt(N) scaling).
 * Args: scene=vol-cl-sph-g85-a99-t50  se=0.001  pixel=0 (per-pixel quantile target, 0 = off)  pixelq=0.99  max=16384  first=64  w= h= (film override)
 */
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
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
  const kernel = await adapter.createKernel();
  const t0 = performance.now();
  const r = await kernel.renderUntil({
    ...(se > 0 ? { meanRelSe: se } : {}),
    ...(pixel > 0 ? { pixelRelSe: { below: pixel, quantile: numberArg(args, 'pixelq', 0.99) } } : {}),
    maxSamples: numberArg(args, 'max', 16384),
    firstBatch: numberArg(args, 'first', 64),
  });
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
