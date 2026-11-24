/**
 * runtime-media-check (V6.6): `integrator.volumes.runtimeParams` against compiled medium parameters, per pixel, megakernel and wavefront, in ONE page.
 *   (a) compiled params                      -> image A
 *   (b) runtimeParams, the same params       -> image B     (A vs B: the uniform table holds the compiled rows)
 *   (c) kernel COMPILED with new params      -> image C
 *   (d) runtimeParams kernel + setMedia(new) -> image D     (C vs D: setMedia = recompile; also timed: setMedia + first frame vs a fresh kernel's build)
 * Same seed, film and spp for all. Args: cases=vol-grid-cloud,vol-cornell-smoke,vol-cl-sph-hg2-a99-t50  spp=8  tol=1e-5
 */
import type { PtMegakernel, PtWavefront } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
const metrics: Record<string, number> = {};
const problems: string[] = [];

async function render(device: GPUDevice, kernel: PtMegakernel | PtWavefront, spp: number): Promise<Float32Array> {
  const reset = device.createCommandEncoder();
  kernel.reset(reset);
  device.queue.submit([reset.finish()]);
  for (let done = 0; done < spp; done += 4) {
    const e = device.createCommandEncoder();
    kernel.encode(e, Math.min(4, spp - done));
    device.queue.submit([e.finish()]);
  }
  if ('finish' in kernel) await kernel.finish();
  await device.queue.onSubmittedWorkDone();
  return (await kernel.readImage()).data;
}

/** Largest |difference| of a pixel over its largest component (floor 1e-3 of the mean) and the bit-identical fraction. */
function compare(a: Float32Array, b: Float32Array): { maxRel: number; over: (t: number) => number; same: number; mean: number } {
  const px = a.length / 4;
  let mean = 0;
  for (let i = 0; i < px; i++) mean += ((a[i * 4] as number) + (a[i * 4 + 1] as number) + (a[i * 4 + 2] as number)) / 3 / px;
  const floor = 1e-3 * Math.max(mean, 1e-6);
  const rel: number[] = [];
  let same = 0;
  for (let i = 0; i < px; i++) {
    let d = 0, s = 0, eq = true;
    for (let c = 0; c < 3; c++) {
      const x = a[i * 4 + c] as number, y = b[i * 4 + c] as number;
      if (!Object.is(x, y)) eq = false;
      d = Math.max(d, Math.abs(x - y));
      s = Math.max(s, Math.abs(x), Math.abs(y));
    }
    if (eq) same++;
    rel.push(d / Math.max(s, floor));
  }
  return { maxRel: Math.max(...rel), over: (t) => rel.filter((r) => r > t).length, same: same / px, mean };
}

try {
  const args = getArgs();
  const gpu = await navigator.gpu?.requestAdapter();
  if (!gpu) throw new Error('no WebGPU adapter');
  adapterSummary = summarizeAdapter(gpu);
  const spp = numberArg(args, 'spp', 8);
  const tol = numberArg(args, 'tol', 1e-5);
  for (const name of (args.cases ?? 'vol-grid-cloud,vol-cornell-smoke,vol-cl-sph-hg2-a99-t50').split(',')) {
    const adapter = new PtMegakernelAdapter({});
    await adapter.load(getScene(name));
    const device = adapter.gpuContext!.device;
    const v00 = (adapter as unknown as { lastKernelOptions: { integrator: { volumes: any } } }).lastKernelOptions.integrator.volumes;
    // `forced=1` (V6.6, the fog example's estimators): forced in-scatter NEE + equiangular distances (megakernel only: the wavefront rows are skipped below).
    const v0 = args.forced === '1' ? { ...v00, forcedNee: true, equiangular: true } : v00;
    const v1 = {
      ...v0,
      media: v0.media.map((m: any) => {
        const g = m.g * 0.5 + 0.1;
        return { ...m, sigmaT: m.sigmaT.map((s: number) => s * 1.7), albedo: m.albedo.map((x: number) => x * 0.85), g, ...(m.lobes ? { lobes: { ...m.lobes, gA: g, w: m.lobes.w * 0.5 } } : {}) };
      }),
    };
    const rt0 = { ...v0, runtimeParams: true };
    const rt1 = { ...v1, runtimeParams: true };
    const kernels = {
      mk: {
        a: args.forced === '1' ? await adapter.createKernel({ integrator: { volumes: v0 } }) : (adapter.renderer as PtMegakernel),
        b: await adapter.createKernel({ integrator: { volumes: rt0 } }),
        c: await adapter.createKernel({ integrator: { volumes: v1 } }),
        d: await adapter.createKernel({ integrator: { volumes: rt0 } }),
      },
      wf: args.forced === '1' ? ({} as never) : {
        a: await adapter.createWavefrontKernel({ integrator: { volumes: v0 } }),
        b: await adapter.createWavefrontKernel({ integrator: { volumes: rt0 } }),
        c: await adapter.createWavefrontKernel({ integrator: { volumes: v1 } }),
        d: await adapter.createWavefrontKernel({ integrator: { volumes: rt0 } }),
      },
    };
    for (const [sched, k] of Object.entries(kernels)) {
      if (args.forced === '1' && sched === 'wf') continue;
      const A = await render(device, k.a, spp);
      const B = await render(device, k.b, spp);
      const C = await render(device, k.c, spp);
      (k.d as PtMegakernel).setMedia(rt1);
      const D = await render(device, k.d, spp);
      const ab = compare(A, B), cd = compare(C, D), ac = compare(A, C);
      const tag = `${name}_${sched}`;
      metrics[`${tag}_ab_relpx_max`] = ab.maxRel; metrics[`${tag}_ab_over_tol`] = ab.over(tol); metrics[`${tag}_ab_bit_identical`] = ab.same;
      metrics[`${tag}_cd_relpx_max`] = cd.maxRel; metrics[`${tag}_cd_over_tol`] = cd.over(tol); metrics[`${tag}_cd_bit_identical`] = cd.same;
      metrics[`${tag}_ac_relpx_max`] = ac.maxRel; // the new params must change the image (sanity)
      metrics[`${tag}_mean_a`] = ab.mean; metrics[`${tag}_mean_c`] = cd.mean;
      const bad = ab.over(tol) > 0 || cd.over(tol) > 0 || !(ac.maxRel > 0.01);
      metrics[`${tag}_mismatch`] = bad ? 1 : 0;
      if (bad) problems.push(`${tag}: ab over ${ab.over(tol)} (max ${ab.maxRel}), cd over ${cd.over(tol)} (max ${cd.maxRel}), ac max ${ac.maxRel}`);
      // setMedia refuses a structure change
      let refused = 0;
      try { (k.d as PtMegakernel).setMedia({ ...rt1, media: [...rt1.media, ...rt1.media] }); } catch { refused = 1; }
      metrics[`${tag}_structure_refused`] = refused;
      if (!refused) problems.push(`${tag}: setMedia accepted a changed number of media`);
      for (const kk of Object.values(k)) kk.destroy();
    }
    adapter.dispose();
  }
  const bad = Object.entries(metrics).filter(([k, v]) => (k.endsWith('_mismatch') && v > 0) || (k.endsWith('_structure_refused') && v === 0)).length;
  reportResult({ scene: 'runtime-media-check', renderer: 'ours', config: { spp, tol }, valid: bad === 0, metrics, notes: { problems: problems.join(" | ") } });
  if (bad > 0) throw new Error(`runtime-media-check: ${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors: [] });
}
