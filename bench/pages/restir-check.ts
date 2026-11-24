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
 *  timing    ms per frame at the scene's film (`w` x `h` override): NEE, RIS, RIS + temporal, RIS + temporal + spatial, unbiased, and `sharedHit` (P4.2c), interleaved `reps` times; provisional.
 *  unify     (P4.2b/c/e) on textured, OpenPBR and alpha-MASK lamp scenes as well as the old ones: M = 1 equivalence with NEE (own noise floor and the PCG-NEE ratio), unbiased convergence
 *            (relBias of `ris_ts_unb`, two seeds) of the plain passes, `sharedHit`, `sampler: 'bluenoise'` and both, and of the V-buffer primary (`primary: 'vbuffer'`, the adapter's raster).
 *  prevcam   (P4.2e) one source of truth for the previous camera: the same camera path with `setCamera` only (tracked), `nextFrame`, and `setPrevCamera` must give bit-identical images;
 *            a wrong previous camera must not.
 *
 * `scheduler=wavefront` (P4.2a) runs every kernel of the check as the plain wavefront (`createWavefrontKernel`); `sharedHit` / `shareddiff` / `unify` rows that need it throw there.
 * Args: scenes=cornell,quad-light,lights-mix-lamps,many-lights-64  scheduler=megakernel|wavefront  check=equiv,converge,noise,motion,timing,r6,unify,prevcam  frames=1024  reffames=4096  seeds=8  step=0.5  reps=5  w=1280 h=720  m=8  k=2  radius=16
 */
import { type Caps, type SceneCopy, type SceneCopyBuffers } from '@pmndrs/raytracer';
import { PtMegakernel, ptRestirGbufBytes, type PtIntegratorOptions, type PtRestirOptions, type PtLookAtCamera } from '@pmndrs/pathtracer';
import type { TraceSceneBuffers } from '@pmndrs/bvh';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { getScene } from '../scenes/index.ts';
import { compareImages } from '../metrics/metrics.ts';
import type { Image } from '../src/adapter.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
/** `scheduler=wavefront` page arg: the kernels under test are `PtWavefront` (plain scheduler). */
let wavefront = false;
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
  // layering 'albedo': ReSTIR's target must be deterministic (the adapter's reference default is the stochastic walk); every kernel of the check, NEE included, uses it.
  const adapter = new PtMegakernelAdapter({ passesPerSubmit: 8, layering: 'albedo' });
  await adapter.load(scene);
  const dbg = adapter.debug;
  deviceErrors = watchDevice(adapter.gpuContext!.device);
  return { adapter, device: adapter.gpuContext!.device, caps: dbg.caps, trace: dbg.trace, buffers: dbg.sceneBuffers, copy: dbg.copy, scene, kernels: [] };
}

/** A kernel with the adapter's own options for this scene (textures, OpenPBR, alpha mask, environment) and `integrator` merged over them. */
async function makeKernel(ctx: Ctx, integrator: PtIntegratorOptions): Promise<PtMegakernel> {
  // `scheduler=wavefront` (P4.2a): every kernel of the check is the plain wavefront (same surface for the calls below: setFilm / reset / setSeed / encode / readImage / restir)
  const k = wavefront ? ((await ctx.adapter.createWavefrontKernel({ integrator })) as unknown as PtMegakernel) : await ctx.adapter.createKernel({ integrator });
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

/** Mean of one colour channel (absolute level: V1.7 compares it with a closed form). */
function meanChannel(im: Image, c: number): number {
  let t = 0;
  const n = im.width * im.height;
  for (let i = 0; i < n; i++) t += im.data[i * im.channels + c]!;
  return t / n;
}

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

function modes(m: number, k: number, radius: number, media = false): Mode[] {
  const base = { candidates: m, spatialRadius: radius };
  // V5.4: a scene with media refuses initialVisibility (its binary shadow ray would stop at medium boundaries): that mode is left out there.
  return ([
    { name: 'ris', integrator: rs({ ...base, temporal: false, spatial: 0, bias: 'unbiased', initialVisibility: false }) },
    { name: 'ris_t_unb', integrator: rs({ ...base, temporal: true, spatial: 0, bias: 'unbiased', initialVisibility: false }) },
    { name: 'ris_s_unb', integrator: rs({ ...base, temporal: false, spatial: k, bias: 'unbiased', initialVisibility: false }) },
    { name: 'ris_ts_unb', integrator: rs({ ...base, temporal: true, spatial: k, bias: 'unbiased', initialVisibility: false }) },
    { name: 'ris_ts_bia', integrator: rs({ ...base, temporal: true, spatial: k, bias: 'biased', initialVisibility: false }) },
    { name: 'ris_ts_bia_vis', integrator: rs({ ...base, temporal: true, spatial: k, bias: 'biased', initialVisibility: true }) },
  ] as Mode[]).filter((x) => !media || x.name !== 'ris_ts_bia_vis');
}

const hasMedia = (ctx: Ctx): boolean => Object.keys(ctx.scene.media ?? {}).length > 0;

async function checkConverge(name: string, frameCount: number, refFrames: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const nee = await makeKernel(ctx, NEE);
  const ref = await accumulate(ctx, nee, refFrames, 100);
  const neeN = await accumulate(ctx, nee, frameCount, 1);
  const base = cmp(neeN, ref);
  const p = `converge__${name}`;
  log(`${p}__nee_relmse`, base.relMse);
  log(`${p}__nee_relbias`, base.relBias);
  log(`${p}__ref_mean_g`, meanChannel(ref, 1));
  for (const mode of modes(m, k, radius, hasMedia(ctx))) {
    const kernel = await makeKernel(ctx, mode.integrator);
    const a = await accumulate(ctx, kernel, frameCount, 1);
    log(`${p}__${mode.name}_mean_g`, meanChannel(a, 1));
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

/**
 * `biasscan` (P4.2b bias hunt): relBias of the unbiased modes over `seeds` independent runs against one long NEE reference, as mean +- standard error. One accumulated static-camera run
 * correlates its frames through the reuse, so a single seed reads relBias noise of ~1e-3 on a small film: only the mean over many seeds says anything about bias.
 */
async function checkBiasScan(name: string, frameCount: number, refFrames: number, seeds: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const nee = await makeKernel(ctx, NEE);
  const ref = await accumulate(ctx, nee, refFrames, 100);
  const p = `biasscan__${name}`;
  let refSum = 0;
  for (let i = 0; i < ref.data.length; i += 4) refSum += (ref.data[i] as number) + (ref.data[i + 1] as number) + (ref.data[i + 2] as number);
  log(`${p}__ref_mean`, refSum / ((ref.data.length / 4) * 3));
  const stats = (v: number[]): { mean: number; se: number } => {
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, v.length - 1));
    return { mean, se: sd / Math.sqrt(v.length) };
  };
  const neeBias: number[] = [];
  for (let s = 0; s < seeds; s++) neeBias.push(cmp(await accumulate(ctx, nee, frameCount, 1000 + s), ref).relBias);
  const nb = stats(neeBias);
  log(`${p}__nee_relbias_mean`, nb.mean);
  log(`${p}__nee_relbias_se`, nb.se);
  for (const mode of modes(m, k, radius, hasMedia(ctx)).filter((x) => x.name.endsWith('unb') || x.name === 'ris')) {
    const kernel = await makeKernel(ctx, mode.integrator);
    const v: number[] = [];
    const blk = new Float64Array(32 * 32);
    const blkRef = new Float64Array(32 * 32);
    for (let s = 0; s < seeds; s++) {
      const img = await accumulate(ctx, kernel, frameCount, 2000 + s);
      v.push(cmp(img, ref).relBias);
      for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) {
        const b = Math.min(31, Math.floor((y * 32) / img.height)) * 32 + Math.min(31, Math.floor((x * 32) / img.width));
        const i = (y * img.width + x) * 4;
        for (let c = 0; c < 3; c++) { blk[b] = (blk[b] as number) + (img.data[i + c] as number) / seeds; blkRef[b] = (blkRef[b] as number) + (ref.data[i + c] as number) / seeds; }
      }
    }
    if (mode.name === 'ris_ts_unb') {
      // blocks (8x8 px of a 256 film) whose seed-averaged mean differs from the reference: "y,x:relBias" for |relBias| > 1% and a mean above 0.02
      const out: string[] = [];
      for (let b = 0; b < 1024; b++) {
        const r = (blkRef[b] as number) / 64 / 3;
        if (r > 0.02 && Math.abs((blk[b] as number) / (blkRef[b] as number) - 1) > 0.004) out.push(`${Math.floor(b / 32)},${b % 32}:${(((blk[b] as number) / (blkRef[b] as number) - 1) * 100).toFixed(1)}`);
      }
      notes[`${p}__blocks`] = out.join(' ');
    }
    const st = stats(v);
    log(`${p}__${mode.name}_relbias_mean`, st.mean);
    log(`${p}__${mode.name}_relbias_se`, st.se);
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
  // NEE with independent (PCG) numbers, 1 spp: the like-for-like per-frame noise of ReSTIR's own sampler (CLAUDE.md trap: Owen-Sobol NEE is stratified over frames).
  const neePcg = await makeKernel(ctx, { sampler: 'pcg' });
  let pcgSum = 0;
  for (let s = 0; s < seeds; s++) pcgSum += cmp(await accumulate(ctx, neePcg, 1, 250 + s), ref).relMse;
  log(`${p}__nee_pcg_1spp_relmse`, pcgSum / seeds);
  for (const mode of modes(m, k, radius, hasMedia(ctx))) {
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
    { name: 'ris_ts_shared', integrator: rs({ candidates: m, temporal: true, spatial: k, bias: 'biased', initialVisibility: true, spatialRadius: radius, sharedHit: true }) },
    { name: 'ris_ts_bn', integrator: { ...rs({ candidates: m, temporal: true, spatial: k, bias: 'biased', initialVisibility: true, spatialRadius: radius }), sampler: 'bluenoise' } },
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
  notes[`${p}__restir_gbuf_bytes`] = kernels[1]?.restir?.stats.gbufBytes ?? 0;
  notes[`${p}__restir_shared_buffers_bytes`] = kernels[5]?.restir?.stats.bytes ?? 0;
  // 1080p: the two G-buffer frames now and before P4.2c (2 x 32 B per pixel)
  notes[`gbuf_bytes_1080p_now`] = ptRestirGbufBytes({}, 1920 * 1080);
  notes[`gbuf_bytes_1080p_before`] = 2 * 1920 * 1080 * 32;
  dispose(ctx);
}

/** `unify` (P4.2b/c/e): equivalence at M = 1 and unbiased convergence of every flavour of the passes on a scene (textures, OpenPBR, alpha MASK included). */
async function checkUnify(name: string, frameCount: number, refFrames: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const p = `unify__${name}`;
  const nee = await makeKernel(ctx, NEE);
  const ref = await accumulate(ctx, nee, refFrames, 100);
  const base = cmp(await accumulate(ctx, nee, frameCount, 1), ref);
  log(`${p}__nee_relmse`, base.relMse);
  log(`${p}__nee_relbias`, base.relBias);
  // M = 1, no reuse, no visibility: plain NEE without MIS. The PCG NEE is the like-for-like noise reference.
  const neePcg = await makeKernel(ctx, { sampler: 'pcg' });
  const pcg = cmp(await accumulate(ctx, neePcg, frameCount, 4), ref);
  const ris1 = await makeKernel(ctx, rs({ candidates: 1, temporal: false, spatial: 0, initialVisibility: false }));
  const e1 = cmp(await accumulate(ctx, ris1, frameCount, 2), ref);
  const e2 = cmp(await accumulate(ctx, ris1, frameCount, 3), ref);
  log(`${p}__ris1_relbias`, e1.relBias);
  log(`${p}__ris1_relbias_seed2`, e2.relBias);
  log(`${p}__ris1_relmse`, e1.relMse);
  log(`${p}__ris1_ratio_vs_nee_pcg`, e1.relMse / pcg.relMse);
  log(`${p}__nee_pcg_relbias`, pcg.relBias);
  log(`${p}__nonfinite`, e1.nonFinite);
  // unbiased M = 8 + temporal + k spatial; plain / shared hit / blue noise / both
  const unb: PtRestirOptions = { candidates: m, temporal: true, spatial: k, spatialRadius: radius, bias: 'unbiased', initialVisibility: false };
  const flavours: [string, PtIntegratorOptions][] = [
    ['plain', rs(unb)],
    ['shared', rs({ ...unb, sharedHit: true })],
    ['bluenoise', { ...rs(unb), sampler: 'bluenoise' }],
    ['bluenoise_shared', { ...rs({ ...unb, sharedHit: true }), sampler: 'bluenoise' }],
  ];
  for (const [tag, integrator] of flavours) {
    const kernel = await makeKernel(ctx, integrator);
    const a = cmp(await accumulate(ctx, kernel, frameCount, 1), ref);
    const b = cmp(await accumulate(ctx, kernel, frameCount, 2), ref);
    log(`${p}__${tag}_relbias`, a.relBias);
    log(`${p}__${tag}_relbias_seed2`, b.relBias);
    log(`${p}__${tag}_relmse`, a.relMse);
    log(`${p}__${tag}_ratio_vs_nee`, a.relMse / base.relMse);
    log(`${p}__${tag}_nonfinite`, a.nonFinite);
    if (tag === 'plain') notes[`${p}__gbuf_bytes`] = kernel.restir?.stats.gbufBytes ?? 0;
  }
  // The V-buffer primary: the adapter's raster writes the texture, the passes resolve it. Its own adapter (the V-buffer is the engine's), progressive frames.
  const vb = new PtMegakernelAdapter({ passesPerSubmit: 8, layering: 'albedo', primary: 'vbuffer', restir: unb });
  await vb.load(getScene(name));
  const vimg = (await vb.render({ spp: frameCount, seed: 1 })).image;
  const v = cmp(vimg, ref);
  log(`${p}__vbuffer_relbias`, v.relBias);
  log(`${p}__vbuffer_relmse`, v.relMse);
  log(`${p}__vbuffer_ratio_vs_nee`, v.relMse / base.relMse);
  log(`${p}__vbuffer_nonfinite`, v.nonFinite);
  vb.dispose();
  dispose(ctx);
}

/** `shareddiff`: the same frame with and without `sharedHit` (same seed, same random numbers per pixel): where do the images disagree? */
async function checkSharedDiff(name: string, m: number, k: number): Promise<void> {
  const ctx = await load(name);
  const p = `shareddiff__${name}`;
  const unb: PtRestirOptions = { candidates: m, temporal: true, spatial: k, bias: 'unbiased', initialVisibility: false };
  for (const sampler of ['pcg', 'bluenoise-pairs1', 'bluenoise-kronecker', 'bluenoise'] as const) {
    const extra: PtIntegratorOptions = sampler === 'bluenoise-pairs1' ? { sampler: 'bluenoise', blueNoisePairs: 1 } : sampler === 'bluenoise-kronecker' ? { sampler: 'bluenoise', blueNoiseShift: 'kronecker' } : { sampler };
    const a = await makeKernel(ctx, { ...rs(unb), ...extra });
    const b = await makeKernel(ctx, { ...rs({ ...unb, sharedHit: true }), ...extra });
    for (const n of [1, 4]) {
      const ia = await accumulate(ctx, a, n, 5);
      const ib = await accumulate(ctx, b, n, 5);
      let bad = 0;
      let sa = 0;
      let sb = 0;
      const first: string[] = [];
      for (let i = 0; i < ia.data.length; i += 4) {
        const x = (ia.data[i] as number) + (ia.data[i + 1] as number) + (ia.data[i + 2] as number);
        const y = (ib.data[i] as number) + (ib.data[i + 1] as number) + (ib.data[i + 2] as number);
        sa += x;
        sb += y;
        if (Math.abs(x - y) > 1e-3 * Math.max(1e-2, Math.abs(x))) {
          bad++;
          if (first.length < 6) first.push(`(${(i / 4) % ia.width},${Math.floor(i / 4 / ia.width)}) ${x.toFixed(4)} vs ${y.toFixed(4)}`);
        }
      }
      if (n === 1) {
        const hits = await b.restir!.readHits();
        let miss = 0;
        const firstMiss: string[] = [];
        for (let i = 0; i < hits.length; i += 4) if (hits[i] === 0) {
          miss++;
          if (firstMiss.length < 6) firstMiss.push(`(${(i / 4) % ia.width},${Math.floor(i / 4 / ia.width)})`);
        }
        log(`${p}__${sampler}_hit_buffer_miss_texels`, miss);
        notes[`${p}__${sampler}_hit_buffer_first_miss`] = firstMiss;
      }
      if (n === 1) {
        // where is the shared image black while the traced one is not: counts per 64 x 64 tile (row-major, 8 x 8 tiles of a 512 film)
        const tiles: number[] = Array.from({ length: 64 }, () => 0);
        for (let i = 0; i < ia.data.length; i += 4) {
          const x = (ia.data[i] as number) + (ia.data[i + 1] as number) + (ia.data[i + 2] as number);
          const y = (ib.data[i] as number) + (ib.data[i + 1] as number) + (ib.data[i + 2] as number);
          if (y === 0 && x > 0) {
            const px = (i / 4) % ia.width;
            const py = Math.floor(i / 4 / ia.width);
            tiles[Math.min(7, Math.floor((py * 8) / ia.height)) * 8 + Math.min(7, Math.floor((px * 8) / ia.width))]!++;
          }
        }
        notes[`${p}__${sampler}_black_in_shared_tiles`] = tiles;
        const hits = await b.restir!.readHits();
        const samples: string[] = [];
        for (let i = 0; i < ia.data.length && samples.length < 5; i += 4) {
          const x = (ia.data[i] as number) + (ia.data[i + 1] as number) + (ia.data[i + 2] as number);
          const y = (ib.data[i] as number) + (ib.data[i + 1] as number) + (ib.data[i + 2] as number);
          if (y === 0 && x > 0 && ((i / 4) % 50 === 0)) samples.push(`px ${(i / 4) % ia.width},${Math.floor(i / 4 / ia.width)} traced ${x.toFixed(3)} texel ${hits[i]},${hits[i + 1]},${hits[i + 2]},${(hits[i + 3] as number).toString(16)}`);
        }
        notes[`${p}__${sampler}_black_samples`] = samples;
      }
      log(`${p}__${sampler}_f${n}_differing_pixels`, bad);
      log(`${p}__${sampler}_f${n}_sum_ratio`, sb / sa);
      notes[`${p}__${sampler}_f${n}_first`] = first;
    }
  }
  // which part breaks: no reuse at all, bn + shared
  for (const [tag, o] of [['noreuse', { candidates: 1, temporal: false, spatial: 0, initialVisibility: false }], ['nospatial', { temporal: true, spatial: 0, bias: 'unbiased' as const, initialVisibility: false }]] as [string, PtRestirOptions][]) {
    const a = await makeKernel(ctx, { ...rs(o), sampler: 'bluenoise' });
    const b = await makeKernel(ctx, { ...rs({ ...o, sharedHit: true }), sampler: 'bluenoise' });
    const ia = await accumulate(ctx, a, 1, 5);
    const ib = await accumulate(ctx, b, 1, 5);
    let black = 0;
    for (let i = 0; i < ia.data.length; i += 4) if ((ib.data[i] as number) + (ib.data[i + 1] as number) + (ib.data[i + 2] as number) === 0 && (ia.data[i] as number) > 0) black++;
    log(`${p}__bluenoise_${tag}_black_in_shared`, black);
  }
  dispose(ctx);
}

/** `prevcam` (P4.2e): the previous camera has one source (the megakernel's). Tracked (`setCamera` only), `nextFrame` and an explicit `setPrevCamera` of the same path must agree to the bit. */
async function checkPrevCam(name: string, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const p = `prevcam__${name}`;
  const poses: PtLookAtCamera[] = [];
  for (let t = 0; t < 8; t++) poses.push(orbit(ctx.scene.camera, 1.5 * t));
  const kernel = await makeKernel(ctx, rs({ candidates: m, temporal: true, spatial: k, spatialRadius: radius, bias: 'unbiased', initialVisibility: false }));
  const run = async (how: 'tracked' | 'next' | 'explicit' | 'wrong'): Promise<Float32Array> => {
    await reset(ctx, kernel, 7);
    kernel.setPrevCamera(null);
    for (let t = 0; t < poses.length; t++) {
      if (how === 'next' && t > 0) kernel.nextFrame(poses[t] as PtLookAtCamera);
      else {
        kernel.setCamera(poses[t] as PtLookAtCamera);
        if (how === 'explicit') kernel.setPrevCamera(t > 0 ? (poses[t - 1] as PtLookAtCamera) : null);
        if (how === 'wrong') kernel.setPrevCamera(poses[0] as PtLookAtCamera);
      }
      await frames(ctx, kernel, 1, 1);
    }
    kernel.setPrevCamera(null);
    return (await image(kernel)).data;
  };
  const tracked = await run('tracked');
  const diff = (a: Float32Array, b: Float32Array): number => {
    let worst = 0;
    for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs((a[i] as number) - (b[i] as number)));
    return worst;
  };
  log(`${p}__next_maxdiff`, diff(tracked, await run('next')));
  log(`${p}__explicit_maxdiff`, diff(tracked, await run('explicit')));
  log(`${p}__wrong_maxdiff`, diff(tracked, await run('wrong')));
  dispose(ctx);
}

/**
 * `gi` (P5): ReSTIR GI against the plain 2-bounce path tracer (NEE + MIS, `maxDepth` 2: direct light at x1 and x2, which is what DI + GI estimate; `gi_off` = no GI reuse).
 * DI runs without reuse (`unbiased`, no initial visibility) so the numbers are the GI's. relBias over `seeds` independent runs against one long reference as mean +- standard error
 * (CLAUDE.md trap: a single seed correlates its frames), relMSE of one run, the deeper plain path tracer (scene maxDepth) as the "how much do bounces beyond 2 matter" row, and the plain 2-bounce's own noise.
 */
function giModes(k: number, radius: number): Mode[] {
  const di = { candidates: 4, temporal: false, spatial: 0, bias: 'unbiased' as const, initialVisibility: false };
  const gi = (o: object, bias: 'biased' | 'unbiased') => rs({ ...di, gi: { spatialRadius: radius, bias, ...o } });
  return [
    { name: 'gi_off', integrator: gi({ temporal: false, spatial: 0 }, 'unbiased') },
    { name: 'gi_t_unb', integrator: gi({ temporal: true, spatial: 0 }, 'unbiased') },
    { name: 'gi_s_unb', integrator: gi({ temporal: false, spatial: k }, 'unbiased') },
    { name: 'gi_ts_unb', integrator: gi({ temporal: true, spatial: k }, 'unbiased') },
    { name: 'gi_t_bia', integrator: gi({ temporal: true, spatial: 0 }, 'biased') },
    { name: 'gi_ts_bia', integrator: gi({ temporal: true, spatial: k }, 'biased') },
  ];
}

let args0: Record<string, string> = {};
async function checkGi(name: string, frameCount: number, refFrames: number, seeds: number, k: number, radius: number, only: string[] | null): Promise<void> {
  const ctx = await load(name);
  const nee2 = await makeKernel(ctx, NEE);
  nee2.setLimits({ maxDepth: 2 });
  const nee2Pcg = await makeKernel(ctx, { sampler: 'pcg' });
  nee2Pcg.setLimits({ maxDepth: 2 });
  const neeDeep = await makeKernel(ctx, NEE);
  const ref = await accumulate(ctx, nee2, refFrames, 100);
  const p = `gi__${name}`;
  const stats = (v: number[]): { mean: number; se: number } => {
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, v.length - 1));
    return { mean, se: sd / Math.sqrt(v.length) };
  };
  const run = async (key: string, kernel: PtMegakernel, perSeed: boolean) => {
    const v: number[] = [];
    let mse = 0;
    let nf = 0;
    for (let s = 0; s < (perSeed ? seeds : 1); s++) {
      const c = cmp(await accumulate(ctx, kernel, frameCount, 2000 + s), ref);
      v.push(c.relBias);
      mse += c.relMse / (perSeed ? seeds : 1);
      nf += c.nonFinite;
    }
    const st = stats(v);
    log(`${p}__${key}_relbias_mean`, st.mean);
    log(`${p}__${key}_relbias_se`, st.se);
    log(`${p}__${key}_relmse`, mse);
    log(`${p}__${key}_nonfinite`, nf);
    return mse;
  };
  const neeMse = await run('nee2', nee2, true);
  const pcgMse = await run('nee2_pcg', nee2Pcg, true);
  await run('nee_deep', neeDeep, false);
  for (const mode of giModes(k, radius)) {
    if (only && !only.includes(mode.name)) continue;
    const kernel = await makeKernel(ctx, mode.integrator);
    kernel.setLimits({ maxDepth: 2 });
    const mse = await run(mode.name, kernel, true);
    if (args0.diag === '1') {
      // where does a bias live: sum(test) / sum(ref) - 1 per octave of the reference's luminance, and per image quarter
      const img = await accumulate(ctx, kernel, frameCount, 3000);
      const bins = new Map<number, { t: number; r: number; n: number }>();
      for (let i = 0; i < img.data.length; i += 4) {
        const rl = ((ref.data[i] as number) + (ref.data[i + 1] as number) + (ref.data[i + 2] as number)) / 3;
        const tl = ((img.data[i] as number) + (img.data[i + 1] as number) + (img.data[i + 2] as number)) / 3;
        const b = rl > 0 ? Math.floor(Math.log2(rl)) : -99;
        const e = bins.get(b) ?? { t: 0, r: 0, n: 0 };
        e.t += tl; e.r += rl; e.n++;
        bins.set(b, e);
      }
      notes[`${p}__${mode.name}_bias_by_octave`] = [...bins.entries()].sort((x, y) => x[0] - y[0]).map(([b, e]) => `${b}:${(e.t / e.r - 1).toFixed(4)}(n${e.n},w${(e.r / [...bins.values()].reduce((a2, v) => a2 + v.r, 0)).toFixed(3)})`).join(' ');
    }
    log(`${p}__${mode.name}_ratio_vs_nee2`, mse / neeMse);
    log(`${p}__${mode.name}_ratio_vs_nee2_pcg`, mse / pcgMse);
  }
  dispose(ctx);
}

/**
 * `giframes` (P5): the per-frame picture. Static camera: relMSE of frame 1 / 4 / 8 (one sample per frame, the real-time chain's input) against the plain depth-2 reference, averaged over `seeds`,
 * for the plain path tracer (Owen-Sobol and PCG: the mean of 1 / 4 / 8 frames, the equal-time lookup), ReSTIR DI alone and DI + GI, biased and unbiased. Moving camera (orbit `step` degrees per
 * frame): per-frame relMSE against a reference per pose and the 8 x 8 block bias (ghosting) after a 3 frame warm-up. Every kernel is `maxDepth` 2: DI alone continues the path at x2, DI + GI replaces that by the GI sample.
 */
function giFrameModes(m: number, k: number, radius: number): Mode[] {
  const di = (bias: 'biased' | 'unbiased') => ({ candidates: m, spatialRadius: radius, temporal: true, spatial: k, bias, initialVisibility: bias === 'biased' });
  const gi = (bias: 'biased' | 'unbiased') => ({ spatialRadius: radius, spatial: k, temporal: true, bias });
  return [
    { name: 'di_unb', integrator: rs(di('unbiased')) },
    { name: 'di_bia', integrator: rs(di('biased')) },
    { name: 'digi_unb', integrator: rs({ ...di('unbiased'), gi: gi('unbiased') }) },
    { name: 'digi_bia', integrator: rs({ ...di('biased'), gi: gi('biased') }) },
    // DI biased with initial visibility (the cheap DI) + GI unbiased: isolates the GI part where DI's unbiased mode is the noisy one (Sponza sun)
    { name: 'dibia_giunb', integrator: rs({ ...di('biased'), gi: gi('unbiased') }) },
  ];
}

async function checkGiFrames(name: string, refFrames: number, step: number, seeds: number, m: number, k: number, radius: number): Promise<void> {
  const ctx = await load(name);
  const nee2 = await makeKernel(ctx, NEE);
  nee2.setLimits({ maxDepth: 2 });
  const neePcg = await makeKernel(ctx, { sampler: 'pcg' });
  neePcg.setLimits({ maxDepth: 2 });
  const N = 12;
  const poses: PtLookAtCamera[] = [];
  for (let t = 0; t < N; t++) poses.push(orbit(ctx.scene.camera, step * t));
  const refs: Image[] = [];
  for (const pose of poses) {
    nee2.setCamera(pose);
    refs.push(await accumulate(ctx, nee2, refFrames, 100));
  }
  const p = `giframes__${name}`;
  const ref0 = refs[0] as Image;
  for (const [tag, kernel] of [['nee2', nee2], ['nee2_pcg', neePcg]] as const) {
    kernel.setCamera(poses[0] as PtLookAtCamera);
    for (const n of [1, 4, 8]) {
      let sum = 0;
      for (let sd = 0; sd < seeds; sd++) sum += cmp(await accumulate(ctx, kernel, n, 200 + sd), ref0).relMse / seeds;
      log(`${p}__${tag}_static_f${n}_relmse`, sum);
    }
    let mse = 0;
    let bp = 0;
    for (let t = 0; t < N; t++) {
      kernel.setCamera(poses[t] as PtLookAtCamera);
      const f = await accumulate(ctx, kernel, 1, 400 + t);
      mse += cmp(f, refs[t] as Image).relMse / N;
      bp += blockBias(f, refs[t] as Image).p99 / N;
    }
    log(`${p}__${tag}_moving_f1_relmse`, mse);
    log(`${p}__${tag}_moving_blockp99`, bp);
  }
  for (const mode of giFrameModes(m, k, radius)) {
    if (args0.modes && !args0.modes.split(',').includes(mode.name)) continue;
    const kernel = await makeKernel(ctx, mode.integrator);
    kernel.setLimits({ maxDepth: 2 });
    kernel.setCamera(poses[0] as PtLookAtCamera);
    const acc: Record<number, number> = { 1: 0, 4: 0, 8: 0 };
    for (let sd = 0; sd < seeds; sd++) {
      await frameImages(ctx, kernel, 8, 300 + sd, (t, frame) => {
        if (t in acc) acc[t] = (acc[t] as number) + cmp(frame, ref0).relMse / seeds;
      });
    }
    for (const n of [1, 4, 8]) log(`${p}__${mode.name}_static_f${n}_relmse`, acc[n] as number);
    for (const moving of [true, false]) {
      let mse = 0;
      let bp = 0;
      let bm = 0;
      let cnt = 0;
      await reset(ctx, kernel, 500);
      let prev: Float32Array | null = null;
      for (let t = 1; t <= N; t++) {
        kernel.setCamera((moving ? poses[t - 1] : poses[0]) as PtLookAtCamera);
        await frames(ctx, kernel, 1, 1);
        const cur = await image(kernel);
        const f = new Float32Array(cur.data.length);
        for (let i = 0; i < f.length; i++) f[i] = (cur.data[i] as number) * t - (prev ? (prev[i] as number) * (t - 1) : 0);
        prev = cur.data;
        if (t < 4) continue;
        const frame = { width: cur.width, height: cur.height, channels: 4 as const, data: f };
        const ref = refs[moving ? t - 1 : 0] as Image;
        mse += cmp(frame, ref).relMse;
        const bb = blockBias(frame, ref);
        bp += bb.p99;
        bm += bb.mean;
        cnt++;
      }
      const tag = `${mode.name}_${moving ? 'moving' : 'staticcam'}`;
      log(`${p}__${tag}_relmse`, mse / cnt);
      log(`${p}__${tag}_blockp99`, bp / cnt);
      log(`${p}__${tag}_blockmean`, bm / cnt);
    }
  }
  dispose(ctx);
}

function dispose(ctx: Ctx): void {
  for (const k of ctx.kernels) k.destroy();
  ctx.adapter.dispose();
}

try {
  const args = getArgs();
  args0 = args;
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  wavefront = args.scheduler === 'wavefront';
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
    if (checks.includes('biasscan')) await checkBiasScan(name, frameCount, refFrames, seeds, m, k, radius);
    if (checks.includes('gi')) await checkGi(name, frameCount, refFrames, seeds, k, radius, args.modes ? args.modes.split(',') : null);
    if (checks.includes('giframes')) await checkGiFrames(name, Math.min(refFrames, 512), step, seeds, m, k, radius);
    if (checks.includes('noise')) await checkNoise(name, refFrames, seeds, m, k, radius);
    if (checks.includes('motion')) await checkMotion(name, Math.min(refFrames, 512), step, m, k, radius);
    if (checks.includes('r6')) await checkR6(name, refFrames, seeds, m, k, radius);
    if (checks.includes('unify')) await checkUnify(name, frameCount, refFrames, m, k, radius);
    if (checks.includes('prevcam')) await checkPrevCam(name, m, k, radius);
    if (checks.includes('shareddiff')) await checkSharedDiff(name, m, k);
    if (checks.includes('timing')) await checkTiming(name, numberArg(args, 'w', 1280), numberArg(args, 'h', 720), reps, numberArg(args, 'timingframes', 16), m, k, radius);
  }
  reportResult({ scene: 'restir-check', renderer: 'ours', config: { checks: checks.join(','), scheduler: wavefront ? 'wavefront' : 'megakernel', scenes: scenes.join(','), frames: frameCount, refFrames, seeds, m, k, radius }, valid: deviceErrors.length === 0, metrics, notes }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
