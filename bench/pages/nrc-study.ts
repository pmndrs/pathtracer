/**
 * nrc-study (track N, N6c / plan M5, Q2): does the neural radiance cache make a better image than plain path tracing in the same time?
 *
 * Sponza (sponza-white), the scene's own camera and settings (maxDepth 8, RR from 4, NEE), against the Mitsuba 3 reference at 4096 spp
 * (bench/references/sponza-white, box-downsampled for a smaller film). Both sides use the same megakernel; the cache side only adds the hook.
 *
 *  1. Equal time: the median wall time of one cache frame (training paths in the render pass, `steps` training steps, inference,
 *     resolve) and of one plain sample per pixel, each submitted alone; plain gets k = round(cache / plain) samples per frame.
 *  2. Cold start: fresh network and empty images; both accumulate frame by frame (the cache learns while it renders). At each
 *     checkpoint: relMSE, signed relative bias and LDR-FLIP of both against the reference.
 *  3. Warm cache: the cache trains `warm` more frames without accumulating, then both images restart; same checkpoints. This is the
 *     steady state of a still or slowly moving camera.
 *  4. Stability (warm cache, no accumulation): mean relative frame-to-frame change of single frames, cache side vs plain at k spp.
 *  5. Floors: relMSE and FLIP of pbrt-v4 against Mitsuba (both 4096 spp): numbers below these are not measurable.
 *
 * 6. Adaptation (scenario=adapt or all; real-time mode: every frame cleared, a new seed, training on): single-frame error at
 *    1..64 frames after arriving at the reference view, for a fresh network (cold), a cache warmed on the opposite view then a
 *    hard cut (cut), the same warm-up then a 60-frame camera flight to the view (fly), against a cache warmed at the view
 *    (steady) and plain path tracing (k spp per frame).
 *
 * 7. Unseen region (scenario=unseen or all): the upper gallery's corridor, which no ground-floor view shows; reference = our own
 *    plain path tracer at refSpp (4096). Single-frame error after a cut from a cache warmed on the ground floor, vs cold and steady.
 *
 * 8. scenario=cold (damping A/B, N6f): only the fresh-network curves (ground floor, unseen corridor) and the cut into the corridor.
 *
 * 9. lighting=vknrc (N6.2, WN6): the same study on `sponza-white-vknrc`, sponza-white under VkNRC's fixed lighting (constant sky 10, no sun,
 *    7 scattering vertices, no RR), so the images compare with the native VkNRC baseline. References come from
 *    bench/references/sponza-white-vknrc/ when rendered there; until then the static reference is our own path tracer at refSpp at the
 *    study's film (no pbrt-vs-Mitsuba floor in that case).
 *
 * Args: scenario=static|adapt|unseen|cold|all  lighting=scene|vknrc  refSpp=4096  refC=1 (load the saved corridor reference)
 *       outputInit=zero|he  warmupFrames=0  lrWarmupFrames=0  ema=warmup|plain|off (N6f; defaults zero + warmup)  res=1280x720  frames=1,4,16,64,256  warm=300  infer=sgm|portable  subgroups=4  precision=f16  steps=2  training=inline|replay
 *       trainSelect=tile|pixel  queryDepth=1  lr=0.002  upload=1 (PFMs of the 16- and 256-frame images)  flip=1
 * Timing rows are wall clock per submit (best effort on a shared machine); the quality numbers are deterministic for a seed.
 */
import { DEFAULT_SHAPE, type Precision } from '@pathtracer/mlp';
import { PtMegakernel, type PtImage, type PtLookAtCamera } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { compareImages } from '../metrics/metrics.ts';
import { flipLdr } from '../metrics/flip.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { Image } from '../src/adapter.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import { sponzaWhite, sponzaWhiteVknrc } from '../scenes/tier2/sponza-white.ts';
import { NrcCache } from '../nrc/cache.ts';
import { inferArmOptions } from './mlp-common.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

/** Box-downsamples a 3- or 4-channel image by an integer factor (the reference film is box-filtered too). */
function downsample(img: Image, factor: number): Image {
  if (factor === 1) return img;
  const w = Math.floor(img.width / factor);
  const h = Math.floor(img.height / factor);
  const c = img.channels;
  const out = new Float32Array(w * h * c);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      for (let ch = 0; ch < c; ch++) {
        let s = 0;
        for (let dy = 0; dy < factor; dy++) for (let dx = 0; dx < factor; dx++) s += img.data[((y * factor + dy) * img.width + x * factor + dx) * c + ch] as number;
        out[(y * w + x) * c + ch] = s / (factor * factor);
      }
  return { width: w, height: h, channels: c, data: out } as Image;
}

const asImage = (p: PtImage): Image => ({ width: p.width, height: p.height, channels: 4, data: p.data }) as Image;

async function fetchPfm(path: string): Promise<Image> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status} (copy the git-ignored references into this checkout)`);
  return readPFM(new Uint8Array(await r.arrayBuffer())) as Image;
}

/** Like `fetchPfm`, but null when the file is not there (a scene whose references were not rendered yet). */
async function fetchPfmOrNull(path: string): Promise<Image | null> {
  const r = await fetch(path);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return readPFM(new Uint8Array(await r.arrayBuffer())) as Image;
}

try {
  const args = getArgs();
  const [width, height] = (args.res ?? '1280x720').split('x').map(Number) as [number, number];
  const checkpoints = (args.frames ?? '1,4,16,64,256').split(',').map(Number).sort((a, b) => a - b);
  const maxFrames = checkpoints[checkpoints.length - 1] as number;
  const warm = numberArg(args, 'warm', 300);
  const steps = numberArg(args, 'steps', 2);
  const doFlip = numberArg(args, 'flip', 1) === 1;
  const doUpload = numberArg(args, 'upload', 1) === 1;
  const factor = 1280 / width;
  if (!Number.isInteger(factor) || 720 / height !== factor) throw new Error('res must be 1280x720 divided by an integer (640x360, 320x180)');

  const lighting = args.lighting === 'vknrc' ? 'vknrc' : 'scene';
  const scene = lighting === 'vknrc' ? sponzaWhiteVknrc : sponzaWhite;
  const refDir = `/bench/references/${scene.name}`;
  const adapter = new PtMegakernelAdapter();
  await adapter.load(scene);
  const gpu = adapter.gpuContext!;
  adapterSummary = gpu.summary;
  deviceErrors = watchDevice(gpu.device);
  const device = gpu.device;
  const plain = adapter.renderer as PtMegakernel;
  const camera = scene.camera as PtLookAtCamera;
  const limits = { maxDepth: scene.integrator.maxDepth, ...(scene.integrator.rrStartDepth ? { rrStartDepth: scene.integrator.rrStartDepth } : {}) };
  plain.setFilm(width, height);
  plain.setCamera(camera);
  plain.setLimits(limits);

  // Scene box for the position encoding (the glb is in world space).
  const { input, caps, trace, sceneBuffers } = adapter.debug;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const inst of input.instances) {
    const m = inst.objectToWorld;
    const p = input.meshes[inst.meshIndex]!.positions;
    for (let i = 0; i < p.length; i += 3) {
      const v = [p[i]!, p[i + 1]!, p[i + 2]!];
      const w = [0, 1, 2].map((a) => m[a]! * v[0]! + m[4 + a]! * v[1]! + m[8 + a]! * v[2]! + m[12 + a]!);
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a]!, w[a]!);
        hi[a] = Math.max(hi[a]!, w[a]!);
      }
    }
  }

  let precision = (args.precision ?? 'f16') as Precision;
  const arm = inferArmOptions(args.infer ?? 'sgm', gpu.adapter, device, DEFAULT_SHAPE, precision, { records: 64, subgroups: numberArg(args, 'subgroups', 4) });
  precision = arm.precision;
  const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], { width, height, camera, ...limits }, { min: lo as [number, number, number], max: hi as [number, number, number] }, {
    precision,
    ...(arm.sgm ? { sgm: arm.sgm } : {}),
    ...(arm.infer ? { infer: arm.infer } : {}),
    training: args.training === 'replay' ? 'replay' : 'inline',
    trainSelect: args.trainSelect === 'pixel' ? 'pixel' : 'tile',
    queryDepth: numberArg(args, 'queryDepth', 1),
    stepsPerFrame: steps,
    seed: numberArg(args, 'seed', 1),
    outputInit: args.outputInit === 'he' ? 'he' : 'zero',
    warmupFrames: numberArg(args, 'warmupFrames', 0),
    lrWarmupFrames: numberArg(args, 'lrWarmupFrames', 0),
    ema: args.ema === 'plain' || args.ema === 'off' ? args.ema : 'warmup',
    ...(args.lr ? { learningRate: numberArg(args, 'lr', 0.002) } : {}),
  });

  // sponza-white must have its references; sponza-white-vknrc falls back to our own path tracer at refSpp until they are rendered.
  const refFull = lighting === 'vknrc' ? await fetchPfmOrNull(`${refDir}/mitsuba3-llvm_ad_rgb-4096spp.pfm`) : await fetchPfm(`${refDir}/mitsuba3-llvm_ad_rgb-4096spp.pfm`);
  const pbrtFull = refFull ? await fetchPfmOrNull(`${refDir}/pbrt-v4-cpu-4096spp.pfm`) : null;
  if (lighting !== 'vknrc' && !pbrtFull) throw new Error(`${refDir}/pbrt-v4-cpu-4096spp.pfm: missing (copy the git-ignored references into this checkout)`);
  const pbrt = pbrtFull ? downsample(pbrtFull, factor) : null;

  const submit = async (encode: (e: GPUCommandEncoder) => void): Promise<number> => {
    const e = device.createCommandEncoder();
    encode(e);
    const t0 = performance.now();
    device.queue.submit([e.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - t0;
  };
  const median = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] as number;
  const metrics: Record<string, number> = {};
  const notes: Record<string, unknown> = { infer: arm.description, training: cache.settings.training, trainSelect: cache.settings.trainSelect, trainPaths: cache.trainPaths };

  let ref: Image;
  if (refFull) ref = downsample(refFull, factor);
  else {
    const refSpp = numberArg(args, 'refSpp', 4096);
    plain.setSeed(778);
    await submit((e) => plain.reset(e));
    for (let n = 0; n < refSpp; n += 64) await submit((e) => plain.encode(e, Math.min(64, refSpp - n)));
    ref = asImage(await plain.readImage());
    metrics.static_ref_ours_spp = refSpp;
    notes.reference = `ours at ${refSpp} spp (no ${refDir}/mitsuba3-llvm_ad_rgb-4096spp.pfm)`;
    if (doUpload) await uploadImage('static-reference', ref as never);
  }

  // 5. Floors (only with both native references).
  if (pbrt) {
    const m = compareImages(pbrt, ref);
    metrics.floor_relmse_pbrt_vs_mitsuba = m.relMse;
    if (doFlip) metrics.floor_flip_pbrt_vs_mitsuba = flipLdr(pbrt, ref).mean;
  }

  // 1. Equal time (after 30 warm-up frames so pipelines and caches are hot).
  for (let i = 0; i < 30; i++) await submit((e) => cache.encodeFrame(e));
  const tCache: number[] = [];
  const tPlain: number[] = [];
  for (let i = 0; i < 25; i++) {
    tCache.push(await submit((e) => cache.encodeFrame(e)));
    tPlain.push(await submit((e) => plain.encode(e, 1)));
  }
  const cacheMs = median(tCache);
  const plainMs = median(tPlain);
  const k = Math.max(1, Math.round(cacheMs / plainMs));
  Object.assign(metrics, { ms_cache_frame: cacheMs, ms_plain_spp: plainMs, equal_time_plain_spp: k });

  const measure = async (tag: string, frames: number, upload: boolean): Promise<void> => {
    const [a, b] = await Promise.all([plain.readImage(), cache.render.readImage()]);
    for (const [side, img] of [['plain', asImage(a)], ['nrc', asImage(b)]] as const) {
      const m = compareImages(img, ref);
      metrics[`${tag}_${side}_relmse_f${frames}`] = m.relMse;
      metrics[`${tag}_${side}_relbias_f${frames}`] = m.relBias;
      if (doFlip) metrics[`${tag}_${side}_flip_f${frames}`] = flipLdr(img, ref).mean;
      if (upload) await uploadImage(`${tag}-${side}-f${frames}`, img as never);
    }
  };
  const series = async (tag: string): Promise<void> => {
    plain.setSeed(1);
    cache.render.setSeed(1);
    await submit((e) => {
      plain.reset(e);
      cache.reset(e);
    });
    let done = 0;
    for (const F of checkpoints) {
      while (done < F) {
        await submit((e) => {
          cache.encodeFrame(e);
          plain.encode(e, k);
        });
        done++;
      }
      await measure(tag, F, doUpload && (F === 16 || F === maxFrames));
    }
  };

  const scenario = args.scenario ?? 'static';
  const runStatic = scenario === 'static' || scenario === 'all';
  const runAdapt = scenario === 'adapt' || scenario === 'all';
  const runCold = scenario === 'cold';

  // 2. Cold start: a fresh network.
  if (runStatic) {
  cache.resetNetwork();
  await series('cold');

  // 3. Warm cache: keep training without accumulating, then restart both images.
  for (let i = 0; i < warm; i++) {
    cache.render.setSeed(2000 + i); // each cleared frame draws new samples (a clear restarts the sample index)
    await submit((e) => {
      cache.reset(e);
      cache.encodeFrame(e);
    });
  }
  await series('warm');

  }

  // 4. Stability: single frames, no accumulation.
  if (runStatic) {
    const lumOf = (d: Float32Array, i: number) => 0.2126 * (d[i * 4] as number) + 0.7152 * (d[i * 4 + 1] as number) + 0.0722 * (d[i * 4 + 2] as number);
    const refLum = (i: number) => 0.2126 * (ref.data[i * ref.channels] as number) + 0.7152 * (ref.data[i * ref.channels + 1] as number) + 0.0722 * (ref.data[i * ref.channels + 2] as number);
    const prev: Record<string, Float32Array | null> = { plain: null, nrc: null };
    const change: Record<string, number[]> = { plain: [], nrc: [] };
    for (let f = 0; f < 9; f++) {
      // A new seed per frame, as a real-time renderer draws new samples every frame (reset restarts the sample index).
      plain.setSeed(1000 + f);
      cache.render.setSeed(1000 + f);
      await submit((e) => {
        plain.reset(e);
        cache.reset(e);
        cache.encodeFrame(e);
        plain.encode(e, k);
      });
      const [a, b] = await Promise.all([plain.readImage(), cache.render.readImage()]);
      for (const [side, img] of [['plain', a.data], ['nrc', b.data]] as const) {
        const p = prev[side];
        if (p) {
          let s = 0;
          const n = width * height;
          for (let i = 0; i < n; i++) s += Math.abs(lumOf(img, i) - lumOf(p, i)) / (refLum(i) + 0.01);
          change[side]!.push(s / n);
        }
        prev[side] = img;
      }
    }
    metrics.stability_plain_rel_change = median(change.plain!);
    metrics.stability_nrc_rel_change = median(change.nrc!);
  }

  // 6. Adaptation (real-time mode) and 7. an unseen region: shared helpers.
  const viewB = camera;
  const viewA: PtLookAtCamera = { ...camera, position: [8, 1.7, 0], target: [-10, 2.6, 0] };
  /** The upper gallery's north corridor: vault and back wall the ground-floor views never see, lit almost only by bounced light. */
  const viewC: PtLookAtCamera = { ...camera, position: [-8, 4.95, -4.5], target: [10, 4.95, -4.5] };
  const setCam = (c: PtLookAtCamera) => {
    plain.setCamera(c);
    cache.setCamera(c);
  };
  let seedN = 5000;
  const rtFrame = async (): Promise<void> => {
    const sd = seedN++;
    plain.setSeed(sd);
    cache.render.setSeed(sd);
    await submit((e) => {
      plain.reset(e);
      cache.reset(e);
      cache.encodeFrame(e);
      plain.encode(e, k);
    });
  };
  const single = async (against: Image): Promise<{ plain: number; nrc: number; nrcBias: number }> => {
    const [a, b] = await Promise.all([plain.readImage(), cache.render.readImage()]);
    const mb = compareImages(asImage(b), against);
    return { plain: compareImages(asImage(a), against).relMse, nrc: mb.relMse, nrcBias: mb.relBias };
  };
  const ts = [1, 2, 4, 8, 16, 32, 64];
  const warmAt = async (view: PtLookAtCamera) => {
    cache.resetNetwork();
    setCam(view);
    for (let i = 0; i < warm; i++) await rtFrame();
  };
  const curve = async (tag: string, against: Image) => {
    let t = 0;
    for (const T of ts) {
      while (t < T) {
        await rtFrame();
        t++;
      }
      const m = await single(against);
      metrics[`${tag}_nrc_relmse_t${T}`] = m.nrc;
      metrics[`${tag}_nrc_relbias_t${T}`] = m.nrcBias;
    }
  };
  /** Steady state at `view` (a cache warmed there) and plain path tracing, medians of 8 single frames. */
  const steady = async (tag: string, view: PtLookAtCamera, against: Image) => {
    await warmAt(view);
    const pe: number[] = [];
    const ne: number[] = [];
    const nb: number[] = [];
    for (let i = 0; i < 8; i++) {
      await rtFrame();
      const m = await single(against);
      pe.push(m.plain);
      ne.push(m.nrc);
      nb.push(m.nrcBias);
    }
    metrics[`${tag}_plain_relmse`] = median(pe);
    metrics[`${tag}_steady_nrc_relmse`] = median(ne);
    metrics[`${tag}_steady_nrc_relbias`] = median(nb);
  };

  if (runAdapt) {
    await steady('adapt', viewB, ref);
    cache.resetNetwork();
    setCam(viewB);
    await curve('adapt_cold', ref);
    await warmAt(viewA);
    setCam(viewB);
    await curve('adapt_cut', ref);
    // Fly: the same warm-up, then 60 frames moving A -> B (training all the way), turning around by yaw (lerping the
    // targets would pass through a straight-up view).
    await warmAt(viewA);
    const yawOf = (c: PtLookAtCamera) => Math.atan2(c.target[2] - c.position[2], c.target[0] - c.position[0]);
    const yawA = yawOf(viewA);
    const yawB = yawOf(viewB) + (yawOf(viewB) < yawA ? 2 * Math.PI : 0) - 2 * Math.PI;
    for (let i = 1; i <= 60; i++) {
      const f = i / 60;
      const p = viewA.position.map((v, j) => v + (viewB.position[j]! - v) * f) as [number, number, number];
      const yaw = yawA + (yawB - yawA) * f;
      setCam({ ...viewB, position: p, target: [p[0] + Math.cos(yaw), p[1] + 0.05, p[2] + Math.sin(yaw)] });
      await rtFrame();
    }
    setCam(viewB);
    await curve('adapt_fly', ref);
  }

  // 8. Cold start only (scenario=cold, the N6f damping A/B): fresh-network curves at the ground-floor view and in the unseen
  // corridor, and the cut into the corridor (does damping slow a warm cache down?).
  if (runCold) {
    cache.resetNetwork();
    setCam(viewB);
    await curve('adapt_cold', ref);
  }

  // 7. Unseen region: the reference there is our own path tracer at refSpp (it matches Mitsuba on Sponza, P2.9); refC=1 loads
  // the saved one (bench/references/sponza-white/ours-unseen-gallery-640x360-4096spp.pfm) instead of rendering it.
  if (scenario === 'unseen' || scenario === 'all' || runCold) {
    let refC: Image;
    if (args.refC === '1') {
      const saved = await fetchPfm(`${refDir}/ours-unseen-gallery-640x360-4096spp.pfm`);
      if (saved.width % width !== 0) throw new Error(`refC=1: the saved corridor reference is ${saved.width} wide; run at a width that divides it`);
      refC = downsample(saved, saved.width / width);
    } else {
      const refSpp = numberArg(args, 'refSpp', 4096);
      plain.setCamera(viewC);
      plain.setSeed(777);
      await submit((e) => plain.reset(e));
      for (let n = 0; n < refSpp; n += 64) await submit((e) => plain.encode(e, Math.min(64, refSpp - n)));
      refC = asImage(await plain.readImage());
      metrics.unseen_ref_spp = refSpp;
      if (doUpload) await uploadImage('unseen-reference', refC as never);
    }
    if (!runCold) await steady('unseen', viewC, refC);
    cache.resetNetwork();
    setCam(viewC);
    await curve('unseen_cold', refC);
    // Cut: a cache warmed on the ground-floor view (it never saw this corridor), then a hard cut.
    await warmAt(viewB);
    setCam(viewC);
    await curve('unseen_cut', refC);
    if (doUpload && !runCold) {
      const [a, b] = await Promise.all([plain.readImage(), cache.render.readImage()]);
      await uploadImage('unseen-cut-plain-t64', asImage(a) as never);
      await uploadImage('unseen-cut-nrc-t64', asImage(b) as never);
    }
  }

  reportResult(
    { scene: scene.name, renderer: 'nrc-study', config: { lighting, width, height, checkpoints, warm, steps, precision, infer: arm.description, training: cache.settings.training, trainSelect: cache.settings.trainSelect, queryDepth: cache.settings.queryDepth }, valid: deviceErrors.length === 0, metrics, notes },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
