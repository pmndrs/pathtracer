/**
 * restir-check (P4): ReSTIR DI against plain NEE. Correctness numbers first, timing provisional (the runner records the load; quote only rows without `loaded`).
 *
 *  equiv     M = 1, temporal and spatial off: the estimator is plain NEE without MIS. relBias and relMSE of `ris1` against NEE at `frames` frames, next to the NEE-vs-NEE noise floor.
 *  converge  static camera, `frames` accumulated frames: NEE, RIS (M = 8), RIS + temporal, + spatial in `unbiased` and `biased` modes against a high-frame NEE reference (`reffames`):
 *            relBias (two ReSTIR seeds: their spread says how much of it is noise), relMSE, the NEE floor. The biased modes' bias is the answer to "how biased".
 *  noise     per-frame (1 spp per frame) relMSE against the reference at frame 1 / 4 / 8 of a static camera, averaged over `seeds`, for NEE (averages of 1/2/4/8 frames, for equal-time lookup) and the ReSTIR modes.
 *  motion    camera orbit of `step` degrees per frame for 16 frames: per-frame relMSE against a reference per pose, and the 8x8 block bias (p99 of |block mean / reference block mean - 1|): ghosting shows as a bias
 *            spike next to NEE's own 1 spp block noise; moving vs static for the same frames.
 *  r6        decorrelated reuse: autocorrelation of the per-frame error (luminance) along x at lags 1..32, spatial neighbours `random` vs `fixed`, temporal pick `random` vs `nearest`.
 *  timing    ms per frame at the scene's film (`w` x `h` override): NEE, RIS, RIS + temporal, RIS + temporal + spatial, interleaved `reps` times; provisional.
 *
 * Args: scenes=cornell,quad-light,lights-mix-lamps,many-lights-64  check=equiv,converge,noise,motion,timing,r6  frames=1024  reffames=4096  seeds=8  step=0.5  reps=5  w=1280 h=720  m=8  k=2  radius=16
 */
import { sceneFormatOf, type Caps, type SceneCopy, type SceneCopyBuffers } from '@pmndrs/raytracer';
import { PtMegakernel, type PtIntegratorOptions, type PtRestirOptions, type PtLookAtCamera } from '@pmndrs/pathtracer';
import type { TraceSceneBuffers } from '@pmndrs/bvh';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { getScene } from '../scenes/index.ts';
import { compareImages } from '../metrics/metrics.ts';
import type { Image } from '../src/adapter.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const metrics: Record<string, number> = {};
const notes: Record<string, unknown> = {};
const log = (key: string, value: number) => {
  metrics[key] = Number.isFinite(value) ? value : -1;
};

interface Ctx {
  adapter: PtMegakernelAdapter;
  device: GPUDevice;
  caps: Caps;
  trace: TraceSceneBuffers;
  buffers: SceneCopyBuffers;
  copy: SceneCopy;
  scene: SceneDesc;
  kernels: PtMegakernel[];
}

async function load(name: string, film?: { w: number; h: number }): Promise<Ctx> {
  let scene = getScene(name);
  if (film) scene = { ...scene, film: { ...scene.film, width: film.w, height: film.h } };
  const adapter = new PtMegakernelAdapter({ passesPerSubmit: 8 });
  await adapter.load(scene);
  const dbg = adapter.debug;
  deviceErrors = watchDevice(adapter.gpuContext!.device);
  return { adapter, device: adapter.gpuContext!.device, caps: dbg.caps, trace: dbg.trace, buffers: dbg.sceneBuffers, copy: dbg.copy, scene, kernels: [] };
}

async function makeKernel(ctx: Ctx, integrator: PtIntegratorOptions): Promise<PtMegakernel> {
  const k = await PtMegakernel.create(ctx.device, ctx.caps, { format: sceneFormatOf(ctx.trace), layout: ctx.copy.layout, integrator });
  k.setScene(ctx.trace, ctx.buffers);
  k.setFilm(ctx.scene.film.width, ctx.scene.film.height);
  k.setCamera(ctx.scene.camera);
  k.setLimits({ maxDepth: ctx.scene.integrator.maxDepth, rrStartDepth: ctx.scene.integrator.rrStartDepth });
  ctx.kernels.push(k);
  return k;
}

async function reset(ctx: Ctx, k: PtMegakernel, seed: number): Promise<void> {
  const enc = ctx.device.createCommandEncoder();
  k.reset(enc);
  ctx.device.queue.submit([enc.finish()]);
  k.setSeed(seed);
  await ctx.device.queue.onSubmittedWorkDone();
}

async function frames(ctx: Ctx, k: PtMegakernel, n: number, perSubmit = 16): Promise<void> {
  let left = n;
  while (left > 0) {
    const c = Math.min(perSubmit, left);
    const enc = ctx.device.createCommandEncoder();
    k.encode(enc, c);
    ctx.device.queue.submit([enc.finish()]);
    left -= c;
    await ctx.device.queue.onSubmittedWorkDone();
  }
}

async function image(k: PtMegakernel): Promise<Image> {
  const i = await k.readImage();
  return { width: i.width, height: i.height, channels: 4, data: i.data };
}

/** Accumulated image after `n` frames from a fresh start. */
async function accumulate(ctx: Ctx, k: PtMegakernel, n: number, seed: number): Promise<Image> {
  await reset(ctx, k, seed);
  await frames(ctx, k, n);
  return image(k);
}

const NEE: PtIntegratorOptions = {};
const rs = (o: PtRestirOptions): PtIntegratorOptions => ({ restir: o });

function cmp(test: Image, ref: Image) {
  const m = compareImages(test, ref);
  return { relMse: m.relMse, relBias: m.relBias, firefly: m.fireflyPct, nonFinite: m.nonFiniteTest };
}

async function checkEquiv(name: string, frameCount: number): Promise<void> {
  const ctx = await load(name);
  const nee = await makeKernel(ctx, NEE);
  const neePcg = await makeKernel(ctx, { sampler: 'pcg' });
  const ris1 = await makeKernel(ctx, rs({ candidates: 1, temporal: false, spatial: 0, initialVisibility: false }));
  const a = await accumulate(ctx, nee, frameCount, 0);
  const b = await accumulate(ctx, nee, frameCount, 1);
  const r = await accumulate(ctx, ris1, frameCount, 2);
  const r2 = await accumulate(ctx, ris1, frameCount, 3);
  const floor = cmp(b, a);
  const pcg = cmp(await accumulate(ctx, neePcg, frameCount, 4), a);
  const ours = cmp(r, a);
  const ours2 = cmp(r2, a);
  const p = `equiv__${name}`;
  log(`${p}__floor_relmse`, floor.relMse);
  log(`${p}__floor_relbias`, floor.relBias);
  log(`${p}__ris1_relmse`, ours.relMse);
  log(`${p}__ris1_relbias`, ours.relBias);
  log(`${p}__ris1_seed2_relbias`, ours2.relBias);
  log(`${p}__ris1_ratio`, ours.relMse / floor.relMse);
  // NEE + MIS with independent (PCG) numbers: the like-for-like noise of the same sampler as ReSTIR's; the ratio is what dropping the MIS weight costs.
  log(`${p}__nee_pcg_relmse`, pcg.relMse);
  log(`${p}__nee_pcg_relbias`, pcg.relBias);
  log(`${p}__ris1_ratio_vs_nee_pcg`, ours.relMse / pcg.relMse);
  log(`${p}__nonfinite`, ours.nonFinite);
  dispose(ctx);
}

interface Mode {
  name: string;
  integrator: PtIntegratorOptions;
}

function modes(m: number, k: number, radius: number): Mode[] {
  const base = { candidates: m, spatialRadius: radius };
  return [
    { name: 'ris', integrator: rs({ ...base, temporal: false, spatial: 0, bias: 'unbiased', initialVisibility: false }) },
    { name: 'ris_t_unb', integrator: rs({ ...base, temporal: true, spatial: 0, bias: 'unbiased', initialVisibility: false }) },
    { name: 'ris_ts_unb', integrator: rs({ ...base, temporal: true, spatial: k, bias: 'unbiased', initialVisibility: false }) },
    { name: 'ris_ts_bia', integrator: rs({ ...base, temporal: true, spatial: k, bias: 'biased', initialVisibility: false }) },
    { name: 'ris_ts_bia_vis', integrator: rs({ ...base, temporal: true, spatial: k, bias: 'biased', initialVisibility: true }) },
  ];
}

async function checkConverge(name: string, frameCount: number, refFrames: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const nee = await makeKernel(ctx, NEE);
  const ref = await accumulate(ctx, nee, refFrames, 100);
  const neeN = await accumulate(ctx, nee, frameCount, 1);
  const base = cmp(neeN, ref);
  const p = `converge__${name}`;
  log(`${p}__nee_relmse`, base.relMse);
  log(`${p}__nee_relbias`, base.relBias);
  for (const mode of modes(m, k, radius)) {
    const kernel = await makeKernel(ctx, mode.integrator);
    const a = await accumulate(ctx, kernel, frameCount, 1);
    const b = await accumulate(ctx, kernel, frameCount, 2);
    const ca = cmp(a, ref);
    const cb = cmp(b, ref);
    log(`${p}__${mode.name}_relmse`, ca.relMse);
    log(`${p}__${mode.name}_relbias`, ca.relBias);
    log(`${p}__${mode.name}_relbias_seed2`, cb.relBias);
    log(`${p}__${mode.name}_ratio_vs_nee`, ca.relMse / base.relMse);
    log(`${p}__${mode.name}_nonfinite`, ca.nonFinite);
  }
  dispose(ctx);
}

/** The image of frame t alone, from the running average before and after it. */
async function frameImages(ctx: Ctx, k: PtMegakernel, n: number, seed: number, each?: (t: number, frame: Image) => void): Promise<void> {
  await reset(ctx, k, seed);
  let prev: Float32Array | null = null;
  for (let t = 1; t <= n; t++) {
    await frames(ctx, k, 1, 1);
    const cur = await image(k);
    const out = new Float32Array(cur.data.length);
    for (let i = 0; i < out.length; i++) out[i] = (cur.data[i] as number) * t - (prev ? (prev[i] as number) * (t - 1) : 0);
    prev = cur.data;
    each?.(t, { width: cur.width, height: cur.height, channels: 4, data: out });
  }
}

async function checkNoise(name: string, refFrames: number, seeds: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const nee = await makeKernel(ctx, NEE);
  const ref = await accumulate(ctx, nee, refFrames, 100);
  const p = `noise__${name}`;
  for (const n of [1, 2, 4, 8]) {
    let sum = 0;
    for (let s = 0; s < seeds; s++) sum += cmp(await accumulate(ctx, nee, n, 200 + s), ref).relMse;
    log(`${p}__nee_${n}spp_relmse`, sum / seeds);
  }
  for (const mode of modes(m, k, radius)) {
    const kernel = await makeKernel(ctx, mode.integrator);
    const acc: Record<number, number> = { 1: 0, 4: 0, 8: 0 };
    for (let s = 0; s < seeds; s++) {
      await frameImages(ctx, kernel, 8, 300 + s, (t, frame) => {
        if (t in acc) acc[t] = (acc[t] as number) + cmp(frame, ref).relMse;
      });
    }
    for (const t of [1, 4, 8]) log(`${p}__${mode.name}_frame${t}_relmse`, (acc[t] as number) / seeds);
  }
  dispose(ctx);
}

function orbit(cam: PtLookAtCamera, degrees: number): PtLookAtCamera {
  const a = (degrees * Math.PI) / 180;
  const dx = cam.position[0] - cam.target[0];
  const dz = cam.position[2] - cam.target[2];
  const c = Math.cos(a);
  const s = Math.sin(a);
  return { ...cam, position: [cam.target[0] + dx * c - dz * s, cam.position[1], cam.target[2] + dx * s + dz * c] };
}

/** p99 of |block mean(test) / block mean(ref) - 1| over 8x8 blocks with a reference mean above `floor`. */
function blockBias(test: Image, ref: Image): { p99: number; mean: number } {
  const out: number[] = [];
  const B = 8;
  for (let by = 0; by + B <= test.height; by += B) {
    for (let bx = 0; bx + B <= test.width; bx += B) {
      let t = 0;
      let r = 0;
      for (let y = 0; y < B; y++) {
        for (let x = 0; x < B; x++) {
          const i = ((by + y) * test.width + bx + x) * 4;
          for (let c = 0; c < 3; c++) {
            t += test.data[i + c] as number;
            r += ref.data[i + c] as number;
          }
        }
      }
      if (r / (B * B * 3) > 0.05) out.push(t / r - 1);
    }
  }
  const abs = out.map(Math.abs).sort((a, b) => a - b);
  return { p99: abs[Math.floor(abs.length * 0.99)] ?? 0, mean: out.reduce((a, b) => a + b, 0) / Math.max(1, out.length) };
}

async function checkMotion(name: string, refFrames: number, step: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const nee = await makeKernel(ctx, NEE);
  const poses: PtLookAtCamera[] = [];
  const N = 16;
  for (let t = 0; t < N; t++) poses.push(orbit(ctx.scene.camera, step * t));
  const refs: Image[] = [];
  for (const pose of poses) {
    nee.setCamera(pose);
    refs.push(await accumulate(ctx, nee, refFrames, 100));
  }
  const p = `motion__${name}`;
  // NEE at 1 spp: the block-noise baseline
  let neeBlock = 0;
  let neeMse = 0;
  for (let t = 0; t < N; t++) {
    nee.setCamera(poses[t] as PtLookAtCamera);
    const frame = await accumulate(ctx, nee, 1, 400 + t);
    neeBlock += blockBias(frame, refs[t] as Image).p99 / N;
    neeMse += cmp(frame, refs[t] as Image).relMse / N;
  }
  log(`${p}__nee_1spp_blockp99`, neeBlock);
  log(`${p}__nee_1spp_relmse`, neeMse);
  for (const mode of modes(m, k, radius).filter((x) => x.name !== 'ris')) {
    const kernel = await makeKernel(ctx, mode.integrator);
    for (const moving of [true, false]) {
      let mse = 0;
      let bp99 = 0;
      let bmean = 0;
      let n = 0;
      await reset(ctx, kernel, 500);
      let prev: Float32Array | null = null;
      for (let t = 1; t <= N; t++) {
        kernel.setCamera(moving ? (poses[t - 1] as PtLookAtCamera) : (poses[0] as PtLookAtCamera));
        await frames(ctx, kernel, 1, 1);
        const cur = await image(kernel);
        const f = new Float32Array(cur.data.length);
        for (let i = 0; i < f.length; i++) f[i] = (cur.data[i] as number) * t - (prev ? (prev[i] as number) * (t - 1) : 0);
        prev = cur.data;
        if (t < 4) continue; // history warm-up
        const frame = { width: cur.width, height: cur.height, channels: 4 as const, data: f };
        const ref = refs[moving ? t - 1 : 0] as Image;
        mse += cmp(frame, ref).relMse;
        const bb = blockBias(frame, ref);
        bp99 += bb.p99;
        bmean += bb.mean;
        n++;
      }
      const tag = `${mode.name}_${moving ? 'moving' : 'static'}`;
      log(`${p}__${tag}_relmse`, mse / n);
      log(`${p}__${tag}_blockp99`, bp99 / n);
      log(`${p}__${tag}_blockmean`, bmean / n);
    }
  }
  dispose(ctx);
}

/** Normalised autocorrelation of the luminance error along x at lag `lag`. */
function autocorr(test: Image, ref: Image, lag: number): number {
  const W = test.width;
  const err = new Float32Array(W * test.height);
  let mean = 0;
  for (let i = 0; i < err.length; i++) {
    const l = (c: Image) => 0.2126 * (c.data[i * 4] as number) + 0.7152 * (c.data[i * 4 + 1] as number) + 0.0722 * (c.data[i * 4 + 2] as number);
    err[i] = (l(test) - l(ref)) / (l(ref) + 0.1);
    mean += err[i] as number;
  }
  mean /= err.length;
  let num = 0;
  let den = 0;
  for (let y = 0; y < test.height; y++) {
    for (let x = 0; x < W; x++) {
      const e = (err[y * W + x] as number) - mean;
      den += e * e;
      if (x + lag < W) num += e * ((err[y * W + x + lag] as number) - mean);
    }
  }
  return num / den;
}

async function checkR6(name: string, refFrames: number, seeds: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const nee = await makeKernel(ctx, NEE);
  const ref = await accumulate(ctx, nee, refFrames, 100);
  const base = { candidates: m, spatialRadius: radius, spatial: k, temporal: true, bias: 'biased' as const, initialVisibility: true };
  const variants: [string, PtRestirOptions][] = [
    ['random', { ...base, neighbours: 'random', temporalPick: 'random' }],
    ['fixed', { ...base, neighbours: 'fixed', temporalPick: 'nearest' }],
  ];
  const lags = [1, 2, 4, 8, 16, 32];
  const p = `r6__${name}`;
  for (const [tag, o] of variants) {
    const kernel = await makeKernel(ctx, rs(o));
    const acc = lags.map(() => 0);
    let mse = 0;
    for (let s = 0; s < seeds; s++) {
      await frameImages(ctx, kernel, 8, 600 + s, (t, frame) => {
        if (t !== 8) return;
        lags.forEach((lag, i) => (acc[i] = (acc[i] as number) + autocorr(frame, ref, lag) / seeds));
        mse += cmp(frame, ref).relMse / seeds;
      });
    }
    lags.forEach((lag, i) => log(`${p}__${tag}_ac_lag${lag}`, acc[i] as number));
    log(`${p}__${tag}_frame8_relmse`, mse);
  }
  const neeAc = lags.map(() => 0);
  for (let s = 0; s < seeds; s++) {
    const frame = await accumulate(ctx, nee, 1, 700 + s);
    lags.forEach((lag, i) => (neeAc[i] = (neeAc[i] as number) + autocorr(frame, ref, lag) / seeds));
  }
  lags.forEach((lag, i) => log(`${p}__nee_ac_lag${lag}`, neeAc[i] as number));
  dispose(ctx);
}

async function timeKernel(ctx: Ctx, k: PtMegakernel, n: number): Promise<number> {
  await reset(ctx, k, 0);
  await frames(ctx, k, 4, 4); // warm up (also fills the history)
  const t0 = performance.now();
  await frames(ctx, k, n, 8);
  return (performance.now() - t0) / n;
}

async function checkTiming(name: string, w: number, h: number, reps: number, nFrames: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name, { w, h });
  const cfgs: Mode[] = [
    { name: 'nee', integrator: NEE },
    { name: 'ris', integrator: rs({ candidates: m, temporal: false, spatial: 0, bias: 'biased', initialVisibility: true, spatialRadius: radius }) },
    { name: 'ris_t', integrator: rs({ candidates: m, temporal: true, spatial: 0, bias: 'biased', initialVisibility: true, spatialRadius: radius }) },
    { name: 'ris_ts', integrator: rs({ candidates: m, temporal: true, spatial: k, bias: 'biased', initialVisibility: true, spatialRadius: radius }) },
    { name: 'ris_ts_unb', integrator: rs({ candidates: m, temporal: true, spatial: k, bias: 'unbiased', initialVisibility: false, spatialRadius: radius }) },
  ];
  const kernels: PtMegakernel[] = [];
  for (const c of cfgs) kernels.push(await makeKernel(ctx, c.integrator));
  const samples: number[][] = cfgs.map(() => []);
  for (let r = 0; r < reps; r++) {
    const order = cfgs.map((_, i) => i);
    if (r % 2 === 1) order.reverse();
    for (const i of order) (samples[i] as number[]).push(await timeKernel(ctx, kernels[i] as PtMegakernel, nFrames));
  }
  const p = `timing__${name}__${w}x${h}`;
  cfgs.forEach((c, i) => {
    const s = (samples[i] as number[]).slice().sort((a, b) => a - b);
    log(`${p}__${c.name}_ms_min`, s[0] as number);
    log(`${p}__${c.name}_ms_median`, s[Math.floor(s.length / 2)] as number);
    log(`${p}__${c.name}_ms_max`, s[s.length - 1] as number);
  });
  const neeMin = Math.min(...(samples[0] as number[]));
  cfgs.forEach((c, i) => log(`${p}__${c.name}_ratio_min_vs_nee`, Math.min(...(samples[i] as number[])) / neeMin));
  notes[`${p}__restir_kernel_bytes`] = kernels[1] ? (kernels[1].restir?.wgslBytes ?? 0) : 0;
  notes[`${p}__restir_buffers_bytes`] = kernels[1]?.restir?.stats.bytes ?? 0;
  dispose(ctx);
}

function dispose(ctx: Ctx): void {
  for (const k of ctx.kernels) k.destroy();
  ctx.adapter.dispose();
}

try {
  const args = getArgs();
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const checks = (args.check ?? 'equiv,converge').split(',');
  const scenes = (args.scenes ?? 'cornell,quad-light,lights-mix-lamps,many-lights-64').split(',');
  const frameCount = numberArg(args, 'frames', 1024);
  const refFrames = numberArg(args, 'reffames', 4096);
  const seeds = numberArg(args, 'seeds', 8);
  const step = numberArg(args, 'step', 0.5);
  const reps = numberArg(args, 'reps', 5);
  const m = numberArg(args, 'm', 8);
  const k = numberArg(args, 'k', 2);
  const radius = numberArg(args, 'radius', 16);
  for (const name of scenes) {
    if (checks.includes('equiv')) await checkEquiv(name, frameCount);
    if (checks.includes('converge')) await checkConverge(name, frameCount, refFrames, m, k, radius);
    if (checks.includes('noise')) await checkNoise(name, refFrames, seeds, m, k, radius);
    if (checks.includes('motion')) await checkMotion(name, Math.min(refFrames, 512), step, m, k, radius);
    if (checks.includes('r6')) await checkR6(name, refFrames, seeds, m, k, radius);
    if (checks.includes('timing')) await checkTiming(name, numberArg(args, 'w', 1280), numberArg(args, 'h', 720), reps, numberArg(args, 'timingframes', 16), m, k, radius);
  }
  reportResult({ scene: 'restir-check', renderer: 'ours', config: { checks: checks.join(','), scenes: scenes.join(','), frames: frameCount, refFrames, seeds, m, k, radius }, valid: deviceErrors.length === 0, metrics, notes }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
