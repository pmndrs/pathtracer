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
 * Args: scenario=static|adapt|unseen|cold|lighting|all  lighting=scene|vknrc  scene=sponza-glossy (N7b)  ref=ours (no Mitsuba PFM: our own path tracer at refSpp)  refSpp=4096  refC=1 (load the saved corridor reference)
 *       network=default|compact  outputInit=zero|he  warmupFrames=0  lrWarmupFrames=0  ema=warmup|plain|off (N6f; defaults zero + warmup)  res=1280x720  frames=1,4,16,64,256  warm=300  infer=tiled|sgm|portable  subgroups=4  precision=f16  steps=2  training=inline|replay
 *       trainSelect=tile|pixel  queryDepth=1  lr=0.002  upload=1 (PFMs of the 16- and 256-frame images)  flip=1
 *       scene=nrc-materials|nrc-materials-grazing directionInput=view|reflected crop=metal motionViews=5 motionRefSpp=1024 motionSpp=64 (frozen sweep)
 * Timing rows are wall clock per submit (best effort on a shared machine); the quality numbers are deterministic for a seed.
 */
import { MLP_SHAPES, layerCount, layerIn, layerOut, type Precision } from '@pathtracer/mlp';
import { PtMegakernel, type PtImage, type PtLookAtCamera } from '@pmndrs/pathtracer';
import { buildSceneCopy, writeSceneCopy } from '@pmndrs/raytracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { compareImages, REL_MSE_EPS } from '../metrics/metrics.ts';
import { flipLdr } from '../metrics/flip.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { Image } from '../src/adapter.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import { sponzaWhite, sponzaWhiteVknrc } from '../scenes/tier2/sponza-white.ts';
import { sponzaGlossy } from '../scenes/tier2/sponza-glossy.ts';
import { cardMetrics } from '../nrc/card-metrics.ts';
import { nrcMaterialsCard, nrcMaterials, nrcMaterialsGrazing } from '../scenes/tier2/nrc-materials.ts';
import { NrcCache, materialArgs } from '../nrc/cache.ts';
import { terminationArgs } from '../nrc/termination.ts';
import { phase10Settings } from '../nrc/phase10.ts';
import { hashGridArgs } from '../nrc/hash-grid.ts';
import { DEFAULT_SPATIAL_FEATURES } from '../nrc/spatial-features.ts';
import { inferArmOptions, networkArg } from './mlp-common.ts';

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

// Glossy swatches in the material fixture and a fixed floor crop in Sponza.
const materialCrop = [0.1, 0.57, 0.9, 0.91] as const;
const floorCrop = [0.25, 2 / 3, 0.75, 17 / 18] as const;
function glossyCrop(img: Image, bounds: readonly [number, number, number, number] = materialCrop): Image {
  const x = Math.floor(img.width * bounds[0]);
  const y = Math.floor(img.height * bounds[1]);
  const width = Math.floor(img.width * bounds[2]) - x;
  const height = Math.floor(img.height * bounds[3]) - y;
  const data = new Float32Array(width * height * img.channels);
  for (let row = 0; row < height; row++) {
    const start = ((y + row) * img.width + x) * img.channels;
    data.set(img.data.subarray(start, start + width * img.channels), row * width * img.channels);
  }
  return { width, height, channels: img.channels, data };
}

function temporalRelMse(before: Image, after: Image, refBefore: Image, refAfter: Image): number {
  // All four inputs are same-size RGBA crops. Use signed frame differences, so matching the
  // still highlights alone cannot hide the wrong direction or amplitude of their motion.
  let error = 0;
  for (let i = 0; i < after.data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const j = i + c;
      const delta = after.data[j]! - before.data[j]!;
      const referenceDelta = refAfter.data[j]! - refBefore.data[j]!;
      error += (delta - referenceDelta) ** 2 / (referenceDelta ** 2 + REL_MSE_EPS);
    }
  }
  return error / (after.width * after.height * 3);
}

function luminanceAt(image: Image, j: number): number {
  return 0.2126 * image.data[j]! + 0.7152 * image.data[j + 1]! + 0.0722 * image.data[j + 2]!;
}

function highlightThreshold(ref: Image): number {
  const values = Array.from({ length: ref.width * ref.height }, (_, px) => luminanceAt(ref, px * ref.channels));
  values.sort((a, b) => a - b);
  return values[Math.floor(0.9 * values.length)]!;
}

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
  // scene=sponza-glossy (N7b): metal floor and columns, view-dependent radiance at cache vertices; needs ref=ours (no native references).
  const grazing = args.scene === nrcMaterialsGrazing.name;
  const materialScene = grazing || args.scene === nrcMaterials.name;
  const glossyScene = args.scene === sponzaGlossy.name;
  const cropBounds = materialScene
    ? (args.crop === 'metal' && !grazing ? [0.12, 0.60, 0.44, 0.89] as const : materialCrop)
    : floorCrop;
  const cardScene = args.scene === nrcMaterialsCard.name;
  if ((materialScene || cardScene) && args.scenario && args.scenario !== 'static') throw new Error('material fixtures require scenario=static');
  const scene = cardScene ? nrcMaterialsCard : grazing ? nrcMaterialsGrazing : materialScene ? nrcMaterials : glossyScene ? sponzaGlossy : lighting === 'vknrc' ? sponzaWhiteVknrc : sponzaWhite;
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
  const refineControl = args.refineStill === '1';
  const spatialArm = args.phase11;
  if (spatialArm && !['fixed', 'learned', 'ema', 'rgb'].includes(spatialArm)) throw new Error('phase11 must be fixed, learned, ema or rgb');
  if (spatialArm && (args.phase10 || args.webnnReplay === '1')) throw new Error('Phase 11 uses its own study preset');
  if (refineControl && (!cardScene || args.phase10 || spatialArm)) throw new Error('The refinement control is only available for the standalone blue-card study');
  const preset = args.phase10 ? phase10Settings(args.phase10) : spatialArm ? {
    ...phase10Settings(spatialArm === 'rgb' ? 'rgb' : 'mlp'),
    ema: spatialArm === 'ema' ? 'warmup' as const : 'off' as const,
    ...(spatialArm === 'learned' ? { spatialFeatures: {
      log2Entries: numberArg(args, 'featureLog2', DEFAULT_SPATIAL_FEATURES.log2Entries),
      learningRate: numberArg(args, 'featureLr', DEFAULT_SPATIAL_FEATURES.learningRate),
    } } : {}),
  } : {};
  const network = preset.network ?? networkArg(args.network);
  const arm = inferArmOptions(args.infer ?? (args.network || args.phase10 || spatialArm ? 'tiled' : 'sgm'), gpu.adapter, device, MLP_SHAPES[network], precision, { records: 64, subgroups: numberArg(args, 'subgroups', 4) });
  precision = arm.precision;
  if (spatialArm) await device.queue.onSubmittedWorkDone();
  const cacheStart = performance.now();
  const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], { width, height, camera, ...limits }, { min: lo as [number, number, number], max: hi as [number, number, number] }, {
    ...terminationArgs(args), ...(args.model === 'hash' ? { model: 'hash' as const, hash: hashGridArgs(args) } : {}),
    precision, network,
    ...(arm.sgm ? { sgm: arm.sgm } : {}),
    ...(arm.infer ? { infer: arm.infer } : {}),
    training: args.training === 'replay' ? 'replay' : 'inline',
    ...materialArgs(args.factorization, args.materialInputs, args.directionInput),
    fullSuffixEvery: numberArg(args, 'fullSuffixEvery', 0),
    positionBands: numberArg(args, 'positionBands', 12),
    updateBlock: numberArg(args, 'updateBlock', 4),
    trainSelect: args.trainSelect === 'pixel' ? 'pixel' : 'tile',
    queryDepth: numberArg(args, 'queryDepth', 1),
    stepsPerFrame: steps,
    seed: numberArg(args, 'seed', 1),
    outputInit: args.outputInit === 'he' ? 'he' : 'zero',
    warmupFrames: numberArg(args, 'warmupFrames', 0),
    lrWarmupFrames: numberArg(args, 'lrWarmupFrames', 0),
    ema: args.ema === 'plain' || args.ema === 'off' ? args.ema : 'warmup',
    ...(args.lr ? { learningRate: numberArg(args, 'lr', 0.002) } : {}),
    ...preset,
  });
  if (spatialArm) await device.queue.onSubmittedWorkDone();
  const cacheStartupMs = performance.now() - cacheStart;

  // sponza-white must have its references; sponza-white-vknrc falls back to our own path tracer at refSpp until they are rendered.
  // ref=ours (WN5: a checkout without the git-ignored PFMs) skips the native references and uses our own path tracer at refSpp; it
  // does not even ask for them, since a 404 in the console fails the run.
  const oursAllowed = cardScene || materialScene || lighting === 'vknrc' || args.ref === 'ours';
  const refFull =
    cardScene || materialScene || args.ref === 'ours' ? null : oursAllowed ? await fetchPfmOrNull(`${refDir}/mitsuba3-llvm_ad_rgb-4096spp.pfm`) : await fetchPfm(`${refDir}/mitsuba3-llvm_ad_rgb-4096spp.pfm`);
  const pbrtFull = refFull ? await fetchPfmOrNull(`${refDir}/pbrt-v4-cpu-4096spp.pfm`) : null;
  if (!oursAllowed && !pbrtFull) throw new Error(`${refDir}/pbrt-v4-cpu-4096spp.pfm: missing (copy the git-ignored references into this checkout, or pass ref=ours)`);
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
  const notes: Record<string, unknown> = { infer: cache.hash ? cache.description : arm.description, training: cache.settings.training, trainSelect: cache.settings.trainSelect, trainPaths: cache.trainPaths };
  if (spatialArm) metrics.ms_cache_startup = cacheStartupMs;

  const refSpp = numberArg(args, 'refSpp', 4096);
  const renderReference = async (): Promise<Image> => {
    plain.setSeed(778);
    await submit((e) => plain.reset(e));
    for (let n = 0; n < refSpp; n += 64) await submit((e) => plain.encode(e, Math.min(64, refSpp - n)));
    return asImage(await plain.readImage());
  };
  let ref: Image;
  if (refFull) ref = downsample(refFull, factor);
  else {
    ref = await renderReference();
    metrics.static_ref_ours_spp = refSpp;
    notes.reference = `ours at ${refSpp} spp (no ${refDir}/mitsuba3-llvm_ad_rgb-4096spp.pfm)`;
    if (doUpload) await uploadImage('static-reference', ref as never);
    if (doUpload && (materialScene || glossyScene)) await uploadImage('static-reference-glossy', glossyCrop(ref, cropBounds) as never);
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

  if ((args.network && cache.mlp) || args.phase10 || spatialArm) {
    const stageTimes = { inference: [] as number[], trainSteps: [] as number[] };
    for (let f = 0; f < 23; f++) {
      cache.nextFrame();
      for (const stage of cache.stages) {
        const ms = await submit((e) => cache.encodeStage(e, stage));
        if (f >= 3 && (stage === 'inference' || stage === 'trainSteps')) stageTimes[stage].push(ms);
      }
    }
    metrics.ms_cache_inference_stage = median(stageTimes.inference);
    metrics.ms_cache_training_steps = median(stageTimes.trainSteps);
  }
  if (cache.hash) metrics.hash_bytes = cache.hash.bytes;
  if ((args.network || args.phase10 || spatialArm) && cache.mlp) {
    const mlp = cache.mlp;
    metrics.mlp_macs_per_query = mlp.weightCount;
    metrics.mlp_buffer_bytes = mlp.bufferBytes;
    metrics.mlp_inference_weight_bytes = (mlp.sgmWeights ?? mlp.useWeights).size;
    const queryCount = await cache.readQueryCount();
    metrics.mlp_inference_queries = queryCount;
    metrics.mlp_inference_padded_records = Math.ceil(queryCount / mlp.inferGroupRecords) * mlp.inferGroupRecords - queryCount;
    const widths = Array.from({ length: layerCount(mlp.shape) }, (_, l) => layerOut(mlp.shape, l));
    if (mlp.inferBackend === 'portable-tiled') notes.tiledInferenceIdleLanesPerRecord = widths.map((width) => 16 - Math.ceil(width / 4));
    if (mlp.forwardKind === 'tiled') {
      notes.tiledForwardIdleLanesPerRecord = widths.map((width) => 16 - Math.ceil(width / 4));
      notes.tiledBackwardIdleLanesPerRecord = widths.slice(1).map((_, l) => 16 - layerIn(mlp.shape, l + 1) / 4).reverse();
    }
    const { records, batch } = await cache.readTrainCount();
    Object.assign(metrics, { mlp_training_records: records, mlp_training_batch: batch });
  }

  const scoreCard = (tag: string, img: Image) => {
    for (const [key, value] of Object.entries(cardMetrics(img, ref))) metrics[`${tag}_${key}`] = value;
  };
  const measure = async (tag: string, frames: number, upload: boolean): Promise<void> => {
    const [a, b] = await Promise.all([plain.readImage(), cache.render.readImage()]);
    for (const [side, img] of [['plain', asImage(a)], ['nrc', asImage(b)]] as const) {
      if (cardScene) scoreCard(`${tag}_${side}_f${frames}`, img);
      const m = compareImages(img, ref);
      metrics[`${tag}_${side}_relmse_f${frames}`] = m.relMse;
      metrics[`${tag}_${side}_relbias_f${frames}`] = m.relBias;
      if (doFlip) metrics[`${tag}_${side}_flip_f${frames}`] = flipLdr(img, ref).mean;
      if (upload) await uploadImage(`${tag}-${side}-f${frames}`, img as never);
      if (materialScene || glossyScene) {
        const crop = glossyCrop(img, cropBounds);
        metrics[`${tag}_${side}_glossy_relmse_f${frames}`] = compareImages(crop, glossyCrop(ref, cropBounds)).relMse;
        if (upload) await uploadImage(`${tag}-${side}-f${frames}-glossy`, crop as never);
      }
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
    const savedDepth = cache.settings.queryDepth;
    if (refineControl) cache.settings.frozenDisplayOnly = true;
    for (const F of checkpoints) {
      while (done < F) {
        const refine = refineControl && done >= 64;
        if (refineControl) cache.setQueryDepth(refine ? limits.maxDepth + 1 : savedDepth);
        await submit((e) => {
          cache.encodeFrame(e, { train: !refine });
          plain.encode(e, k);
        });
        done++;
      }
      await measure(tag, F, doUpload && (F === 16 || F === maxFrames || ((materialScene || glossyScene) && F === 1)));
    }
    cache.setQueryDepth(savedDepth);
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
  if (args.phase10) cache.hash?.resetStats();
  await series('warm');
  if (spatialArm) {
    const budget = numberArg(args, 'phase11BudgetMs', 300);
    const frozenSpp = numberArg(args, 'phase11FrozenSpp', 64);
    if (!Number.isFinite(budget) || budget <= 0 || !Number.isSafeInteger(frozenSpp) || frozenSpp <= 0) {
      throw new Error('Phase 11 requires a positive time budget and a positive frozen sample count');
    }
    if (cache.spatial) {
      const stats = await cache.readSpatialStats();
      metrics.feature_bytes = cache.spatial.bytes;
      metrics.feature_inference_bytes = cache.spatial.values.size;
      metrics.feature_gradient_adds_per_record_max = cache.spatial.gradientAddsPerRecord;
      metrics.feature_updated_vertices = stats.updatedVertices;
      metrics.feature_collision_records = stats.records;
      for (const { resolution, vertices, aliases } of stats.levels) {
        metrics[`feature_r${resolution}_vertices`] = vertices;
        metrics[`feature_r${resolution}_aliases`] = aliases;
      }
    }
    const scoreSpatial = async (tag: string, image: Image) => {
      const metric = compareImages(image, ref);
      metrics[`${tag}_relmse`] = metric.relMse;
      metrics[`${tag}_relbias`] = metric.relBias;
      if (cardScene) scoreCard(tag, image);
      if (materialScene || glossyScene) {
        const crop = glossyCrop(image, cropBounds);
        metrics[`${tag}_glossy_relmse`] = compareImages(crop, glossyCrop(ref, cropBounds)).relMse;
        if (doUpload) await uploadImage(`${tag}-glossy`, crop);
      }
      if (doUpload) await uploadImage(tag, image);
    };
    const wasDisplayOnly = cache.settings.frozenDisplayOnly;
    cache.settings.frozenDisplayOnly = true;
    cache.render.setSeed(72583);
    await submit(e => cache.reset(e));
    const frozenTimes: number[] = [];
    for (let f = 0; f < frozenSpp; f++) frozenTimes.push(await submit(e => cache.encodeFrame(e, { train: false })));
    metrics.ms_cache_frozen_frame = median(frozenTimes);
    await scoreSpatial('spatial_frozen', asImage(await cache.render.readImage()));
    cache.settings.frozenDisplayOnly = wasDisplayOnly;

    cache.render.setSeed(48107);
    await submit(e => cache.reset(e));
    let elapsed = 0;
    let frames = 0;
    while (elapsed < budget) {
      elapsed += await submit(e => cache.encodeFrame(e));
      frames++;
    }
    metrics.spatial_equal_time_frames = frames;
    metrics.spatial_equal_time_ms = elapsed;
    await scoreSpatial('spatial_equal_time', asImage(await cache.render.readImage()));
    Object.assign(notes, { spatialFrozenSpp: frozenSpp, spatialBudgetMs: budget });
  }
  if (args.webnnReplay === '1') {
    const { replayWebnn } = await import('../nrc/webnn-replay.ts');
    Object.assign(metrics, await replayWebnn(cache, args));
  }
  if (args.phase10 && cache.hash) {
    const st = await cache.hash.readStats();
    metrics.warm_hash_cells = st.cells;
    metrics.warm_hash_directional_cells = await cache.hash.readDirectionalCells();
    metrics.warm_hash_dropped_per_frame = st.dropped / maxFrames;
    metrics.warm_hash_misses_per_pixel = st.misses / (maxFrames * width * height);
    metrics.warm_hash_coarse_per_pixel = st.coarse / (maxFrames * width * height);
  }

  }

  if (cardScene) {
    // Hold the weights, camera, and random sequence fixed. Only the recording and subtraction of training paths changes.
    const locked: Image[] = [];
    for (const displayOnly of [false, true]) {
      cache.settings.frozenDisplayOnly = displayOnly;
      cache.render.setSeed(18213);
      await submit(e => cache.reset(e));
      for (let batch = 0; batch < 8; batch++) {
        await submit(e => { for (let j = 0; j < 8; j++) cache.encodeFrame(e, { train: false }); });
      }
      const img = asImage(await cache.render.readImage());
      locked.push(img);
      const tag = displayOnly ? 'heldout_display_only' : 'heldout_with_training_paths';
      scoreCard(tag, img);
      if (doUpload) await uploadImage(tag, img);
    }
    let worst = 0;
    for (let i = 0; i < locked[0]!.data.length; i++) {
      if (i % 4 !== 3) worst = Math.max(worst, Math.abs(locked[0]!.data[i]! - locked[1]!.data[i]!));
    }
    metrics.heldout_path_layout_max_rgb_delta = worst;
    if (!(worst < 1e-4)) throw new Error(`Frozen training-path subtraction changed card RGB by ${worst}`);
  }

  // 4. Stability: single frames, no accumulation.
  if (runStatic) {
    const lumOf = (d: Float32Array, i: number) => 0.2126 * (d[i * 4] as number) + 0.7152 * (d[i * 4 + 1] as number) + 0.0722 * (d[i * 4 + 2] as number);
    const refLum = (i: number) => 0.2126 * (ref.data[i * ref.channels] as number) + 0.7152 * (ref.data[i * ref.channels + 1] as number) + 0.0722 * (ref.data[i * ref.channels + 2] as number);
    const prev: Record<string, Float32Array | null> = { plain: null, nrc: null };
    const change: Record<string, number[]> = { plain: [], nrc: [] };
    const reflectionChanges: number[] = [];
    const reflectionBox = cardScene ? [86 / 320, 131 / 180, 234 / 320, 168 / 180] : cropBounds;
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
          if (args.phase10 && side === 'nrc') {
            let delta = 0, referenceLight = 0;
            for (let y = Math.floor(height * reflectionBox[1]!); y < Math.floor(height * reflectionBox[3]!); y++) {
              for (let x = Math.floor(width * reflectionBox[0]!); x < Math.floor(width * reflectionBox[2]!); x++) {
                const i = y * width + x;
                delta += Math.abs(lumOf(img, i) - lumOf(p, i));
                referenceLight += refLum(i);
              }
            }
            reflectionChanges.push(delta / Math.max(referenceLight, 1e-8));
          }
        }
        prev[side] = img;
      }
    }
    metrics.stability_plain_rel_change = median(change.plain!);
    metrics.stability_nrc_rel_change = median(change.nrc!);
    if (reflectionChanges.length) metrics.reflection_frame_change = median(reflectionChanges);
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
  const rtFrame = async (timeCache = false, plainSpp = k): Promise<number> => {
    const sd = seedN++;
    plain.setSeed(sd);
    cache.render.setSeed(sd);
    const cacheFrame = (e: GPUCommandEncoder) => {
      cache.reset(e);
      cache.encodeFrame(e);
    };
    const plainFrame = (e: GPUCommandEncoder) => {
      plain.reset(e);
      plain.encode(e, plainSpp);
    };
    if (timeCache) {
      const ms = await submit(cacheFrame);
      await submit(plainFrame);
      return ms;
    }
    return submit((e) => { cacheFrame(e); plainFrame(e); });
  };
  if (materialScene) {
    const viewCount = Math.max(2, Math.floor(numberArg(args, 'motionViews', 5)));
    const refSpp = Math.max(1, Math.floor(numberArg(args, 'motionRefSpp', 256)));
    const views = Array.from({ length: viewCount }, (_, i): PtLookAtCamera => {
      const x = -0.7 + 1.4 * i / (viewCount - 1);
      return { ...camera, position: [camera.position[0] + x, camera.position[1], camera.position[2]], target: [camera.target[0] + 0.35 * x, camera.target[1], camera.target[2]] };
    });
    const references: Image[] = [];
    for (const [i, view] of views.entries()) {
      plain.setCamera(view);
      plain.setSeed(9000 + i);
      await submit((e) => plain.reset(e));
      for (let n = 0; n < refSpp; n += 64) await submit((e) => plain.encode(e, Math.min(64, refSpp - n)));
      references.push(glossyCrop(asImage(await plain.readImage()), cropBounds));
    }
    const previous: Partial<Record<'plain' | 'nrc', Image>> = {};
    for (const [i, view] of views.entries()) {
      setCam(view);
      await rtFrame();
      const [plainImage, cacheImage] = await Promise.all([plain.readImage(), cache.render.readImage()]);
      const currentRef = references[i]!;
      for (const [side, image] of [['plain', plainImage], ['nrc', cacheImage]] as const) {
        const crop = glossyCrop(asImage(image), cropBounds);
        metrics[`motion_${side}_glossy_relmse_v${i + 1}`] = compareImages(crop, currentRef).relMse;
        const prior = previous[side];
        if (prior) metrics[`motion_${side}_glossy_temporal_relmse_v${i + 1}`] = temporalRelMse(prior, crop, references[i - 1]!, currentRef);
        previous[side] = crop;
        if (doUpload) await uploadImage(`motion-${side}-v${i + 1}-glossy`, crop as never);
      }
      if (doUpload) await uploadImage(`motion-reference-v${i + 1}-glossy`, currentRef as never);
    }
    const frozenSpp = numberArg(args, 'motionSpp', 0);
    if (!Number.isSafeInteger(frozenSpp) || frozenSpp < 0) throw new Error('motionSpp must be a nonnegative integer');
    if (frozenSpp > 0) {
      // One trained snapshot for the whole camera sweep. Average out primary NEE noise independently of the reference.
      cache.settings.frozenDisplayOnly = true;
      const thresholds = references.map(highlightThreshold);
      let prev: Image | undefined;
      for (const [i, view] of views.entries()) {
        cache.setCamera(view);
        cache.render.setSeed(12000 + i);
        await submit(e => cache.reset(e));
        for (let n = 0; n < frozenSpp; n += 8) {
          await submit(e => {
            for (let j = n; j < Math.min(n + 8, frozenSpp); j++) cache.encodeFrame(e, { train: false });
          });
        }
        const image = glossyCrop(asImage(await cache.render.readImage()), cropBounds);
        const ref = references[i]!;
        const prefix = `frozen_v${i + 1}_`;
        metrics[prefix + 'relmse'] = compareImages(image, ref).relMse;
        if (prev) metrics[prefix + 'temporal_relmse'] = temporalRelMse(prev, image, references[i - 1]!, ref);
        // Relative energy error on the brightest 10% of reference pixels, with the same mask for both arms.
        let estimated = 0, reference = 0, deltaError = 0, deltaEnergy = 0;
        const priorRef = references[i - 1];
        for (let j = 0; j < ref.data.length; j += ref.channels) {
          const y = luminanceAt(ref, j);
          const hot = y >= thresholds[i]!;
          const predicted = luminanceAt(image, j);
          if (hot) { estimated += predicted; reference += y; }
          if (prev && priorRef) {
            const priorY = luminanceAt(priorRef, j);
            if (hot || priorY >= thresholds[i - 1]!) {
              // The union includes highlights moving out of a pixel as well as those moving into it.
              const delta = y - priorY;
              deltaEnergy += delta * delta;
              deltaError += (predicted - luminanceAt(prev, j) - delta) ** 2;
            }
          }
        }
        if (!(reference > 0)) throw new Error('Empty highlight reference');
        metrics[prefix + 'highlight_bias'] = estimated / reference - 1;
        if (deltaEnergy > 0) metrics[prefix + 'highlight_motion_nmse'] = deltaError / deltaEnergy;
        if (doUpload) await uploadImage(`frozen-nrc-v${i + 1}-glossy`, image);
        prev = image;
      }
    }
    Object.assign(notes, { motionViews: viewCount, motionRefSpp: refSpp, motionSpp: frozenSpp, glossyCrop: cropBounds });
  }
  const single = async (against: Image): Promise<{ plain: number; nrc: number; nrcBias: number }> => {
    const [a, b] = await Promise.all([plain.readImage(), cache.render.readImage()]);
    const mb = compareImages(asImage(b), against);
    return { plain: compareImages(asImage(a), against).relMse, nrc: mb.relMse, nrcBias: mb.relBias };
  };
  if (cache.settings.terminationStats) Object.assign(metrics, await cache.readTerminationStats());
  const ts = [1, 2, 4, 8, 16, 32, 64];
  /** Hash arm (M7): counts per rendered pixel since `hashMark`. Misses include training tails. Only display misses may continue. */
  const hashMark = () => cache.hash?.resetStats();
  const hashMiss = async (tag: string, frames: number) => {
    if (!cache.hash) return;
    const st = await cache.hash.readStats();
    metrics[`${tag}_hash_miss`] = st.misses / (frames * width * height);
    metrics[`${tag}_hash_coarse`] = st.coarse / (frames * width * height);
    metrics[`${tag}_hash_continued`] = st.continued / (frames * width * height);
    metrics[`${tag}_hash_cells`] = st.cells;
    if (st.varianceRejects !== undefined) metrics[`${tag}_variance_rejected`] = st.varianceRejects / (frames * width * height);
    metrics.hash_bytes = cache.hash.bytes;
    if (st.touchedFraction !== undefined) metrics[`${tag}_hash_touched_fraction`] = st.touchedFraction;
  };
  const warmAt = async (view: PtLookAtCamera) => {
    cache.resetNetwork();
    setCam(view);
    for (let i = 0; i < warm; i++) await rtFrame();
  };
  const curve = async (tag: string, against: Image) => {
    let t = 0;
    hashMark();
    for (const T of ts) {
      while (t < T) {
        const firstCut = tag === 'unseen_cut' && t === 0;
        const ms = await rtFrame(firstCut);
        if (firstCut) metrics.unseen_cut_t1_cache_ms = ms;
        t++;
      }
      if (T === 1) {
        await hashMiss(`${tag}_t1`, 1);
        if (doUpload && tag === 'unseen_cut') {
          notes.image = 'unseen-cut-nrc-t1';
          await uploadImage('unseen-cut-nrc-t1', asImage(await cache.render.readImage()) as never);
          await uploadImage('unseen-cut-plain-t1', asImage(await plain.readImage()) as never);
        }
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
    hashMark();
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
    await hashMiss(`${tag}_steady`, 8);
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

  if (scenario === 'lighting') {
    if (!input.sun) throw new Error('scenario=lighting requires a directional light');
    const [x, y, z] = input.sun.direction;
    const direction = [-x, y, -z] as const;
    notes.sunDirectionBefore = input.sun.direction;
    notes.sunDirectionAfter = direction;
    notes.lightingWarmFrames = warm;
    if (glossyScene) notes.glossyCrop = cropBounds;
    await warmAt(viewB);
    await rtFrame(false, 1);

    const capture = async (tag: string, reference: Image, upload: boolean): Promise<void> => {
      const [a, b] = await Promise.all([plain.readImage(), cache.render.readImage()]);
      for (const [side, img] of [['plain', asImage(a)], ['nrc', asImage(b)]] as const) {
        const error = compareImages(img, reference);
        metrics[`${tag}_${side}_relmse`] = error.relMse;
        metrics[`${tag}_${side}_relbias`] = error.relBias;
        if (upload) await uploadImage(`${tag}-${side}`, img);
        if (glossyScene) {
          const crop = glossyCrop(img, cropBounds);
          const cropError = compareImages(crop, glossyCrop(reference, cropBounds));
          metrics[`${tag}_${side}_glossy_relmse`] = cropError.relMse;
          metrics[`${tag}_${side}_glossy_relbias`] = cropError.relBias;
          if (upload) await uploadImage(`${tag}-${side}-glossy`, crop);
        }
      }
    };
    await capture('lighting_before', ref, doUpload);

    // Both renderers already bind these buffers. Keep the trained cache and optimizer as the direct lighting changes.
    const editedCopy = buildSceneCopy({ ...input, sun: { ...input.sun, direction }, options: { ...input.options, surface: { ...input.options?.surface, positions: true } } });
    if (!writeSceneCopy(device, sceneBuffers, editedCopy)) throw new Error('lighting edit did not fit the existing scene buffers');
    const afterReference = await renderReference();
    metrics.lighting_reference_change_relmse = compareImages(ref, afterReference).relMse;
    if (doUpload) {
      await uploadImage('lighting-after-reference', afterReference);
      if (glossyScene) await uploadImage('lighting-after-reference-glossy', glossyCrop(afterReference, cropBounds));
    }
    const checkpointSet = new Set(checkpoints);
    for (let f = 1; f <= maxFrames; f++) {
      await rtFrame(false, 1);
      if (checkpointSet.has(f)) await capture(`lighting_after_f${f}`, afterReference, doUpload && (f === 1 || f === maxFrames));
    }
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
    { scene: scene.name, renderer: 'nrc-study', config: { ...(args.phase10 ? { phase10: args.phase10, refineStill: false } : {}), ...(spatialArm ? { phase11: spatialArm, refineStill: false, ema: cache.settings.ema, spatialFeatures: cache.settings.spatialFeatures } : {}), ...terminationArgs(args), lighting, width, height, checkpoints, warm, steps, precision, infer: arm.description, training: cache.settings.training, trainSelect: cache.settings.trainSelect, updateBlock: cache.settings.updateBlock, fullSuffixEvery: cache.settings.fullSuffixEvery, positionBands: cache.settings.positionBands, materialInputs: cache.settings.materialInputs, factorization: cache.settings.factorization, directionInput: cache.settings.directionInput, queryDepth: cache.settings.queryDepth, model: cache.settings.model, ...(cache.hash ? { hash: cache.hash.settings } : { network, shape: cache.mlp!.shape }) }, valid: deviceErrors.length === 0, metrics, notes },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
