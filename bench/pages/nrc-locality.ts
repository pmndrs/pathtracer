/**
 * Train both sides of a thin partition, or train before moving an opaque card.
 * All probes accumulate samples against an independent, same-geometry PT reference.
 */
import { buildTraceScene, createMeshBvhBuilder, uploadTraceScene } from '@pmndrs/bvh';
import { buildSceneCopy, writeSceneCopy } from '@pmndrs/raytracer';
import type { PtImage, PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { compareImages, REL_MSE_EPS } from '../metrics/metrics.ts';
import { NrcCache } from '../nrc/cache.ts';
import { phase10Settings } from '../nrc/phase10.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import { movingCardFixture, thinWallFixture, type LocalityView } from '../scenes/tier2/nrc-locality.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import type { Image } from '../src/adapter.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const asImage = ({ width, height, data }: PtImage): Image => ({ width, height, channels: 4, data });

function cropImage(image: Image, bounds: LocalityView['crop']): Image {
  const [x0, y0, x1, y1] = bounds;
  const left = Math.floor(x0 * image.width), top = Math.floor(y0 * image.height);
  const width = Math.floor(x1 * image.width) - left;
  const height = Math.floor(y1 * image.height) - top;
  const { channels } = image;
  const data = new Float32Array(width * height * channels);
  for (let y = 0; y < height; y++) {
    const start = ((top + y) * image.width + left) * channels;
    data.set(image.data.subarray(start, start + width * channels), y * width * channels);
  }
  return { width, height, channels, data };
}

function channelMean(image: Image, channel: number): number {
  let sum = 0;
  for (let i = channel; i < image.data.length; i += image.channels) sum += image.data[i]!;
  return sum / (image.width * image.height);
}

function changeError(before: Image, after: Image, refBefore: Image, refAfter: Image): number {
  let error = 0;
  for (let i = 0; i < before.data.length; i += before.channels) {
    for (let channel = 0; channel < 3; channel++) {
      const j = i + channel;
      const delta = refAfter.data[j]! - refBefore.data[j]!;
      const estimated = after.data[j]! - before.data[j]!;
      error += (estimated - delta) ** 2 / (delta ** 2 + REL_MSE_EPS);
    }
  }
  return error / (before.width * before.height * 3);
}

async function main() {
  const args = getArgs();
  const arm = args.arm ?? 'mlp';
  const fixtureName = args.fixture ?? 'thin';
  if (fixtureName !== 'thin' && fixtureName !== 'moving') throw new Error('fixture must be thin or moving');
  const [width, height] = (args.res ?? '256x144').split('x').map(Number) as [number, number];
  const refSpp = numberArg(args, 'refSpp', 1024);
  const probeSpp = numberArg(args, 'probeSpp', 64);
  const warm = numberArg(args, 'warm', 256);
  const thickness = numberArg(args, 'thickness', 0.008);
  const checkpoints = [...new Set((args.checkpoints ?? '1,4,16,64,256').split(',').map(Number))].sort((a, b) => a - b);
  const upload = args.upload === '1';
  if ([width, height, refSpp, probeSpp, warm, ...checkpoints].some(n => !Number.isSafeInteger(n) || n <= 0)
      || Math.min(width, height) < 16 || thickness <= 0 || thickness > 0.5) {
    throw new Error('Expected positive integral spp and frame counts, resolution at least 16x16, and thickness in (0, 0.5] meters');
  }
  if (fixtureName === 'thin' && warm < 2) throw new Error('Thin wall needs at least one training frame per side');

  const fixture = fixtureName === 'thin' ? thinWallFixture(width, height, thickness) : movingCardFixture(width, height);
  const { scene, bounds, views, movedScene } = fixture;
  const adapter = new PtMegakernelAdapter({ scenePacking: 'packed' });
  await adapter.load(scene);
  const { device, summary } = adapter.gpuContext!;
  adapterSummary = summary;
  deviceErrors = watchDevice(device);
  const { caps, trace, sceneBuffers } = adapter.debug;
  const plain = adapter.renderer as PtMegakernel;
  const settings = phase10Settings(arm);
  const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], {
    width, height, camera: scene.camera, ...scene.integrator,
  }, bounds, { ...settings, frozenDisplayOnly: true });
  const metrics: Record<string, number> = { nrc_train_paths: cache.trainPaths };
  let movedTrace: typeof trace | undefined;

  const submit = (encode: (e: GPUCommandEncoder) => void) => {
    const e = device.createCommandEncoder();
    encode(e);
    device.queue.submit([e.finish()]);
    return device.queue.onSubmittedWorkDone();
  };

  const pathTrace = async (view: LocalityView, spp: number, seed: number) => {
    plain.setCamera(view.camera);
    plain.setSeed(seed);
    await submit(e => plain.reset(e));
    for (let n = 0; n < spp; n += 64) await submit(e => plain.encode(e, Math.min(64, spp - n)));
    return asImage(await plain.readImage());
  };

  const probe = async (view: LocalityView, seed: number) => {
    const start = cache.trained;
    cache.setCamera(view.camera);
    cache.render.setSeed(seed);
    await submit(e => cache.reset(e));
    for (let n = 0; n < probeSpp; n += 16) {
      await submit(e => {
        for (let i = n; i < Math.min(n + 16, probeSpp); i++) cache.encodeFrame(e, { train: false });
      });
    }
    const trained = cache.trained;
    if (trained.frames !== start.frames || trained.steps !== start.steps) throw new Error('Frozen probe modified training counters');
    return asImage(await cache.render.readImage());
  };

  const train = async (frames: number, firstSample: number, seed: number) => {
    cache.render.setSeed(seed);
    await submit(e => cache.reset(e));
    cache.render.setFrameIndex(firstSample);
    for (let n = 0; n < frames;) {
      // Camera remains fixed through each submission, including the hash LOD origin.
      const count = Math.min(frames - n, views.length === 1 ? 16 : 1);
      cache.setCamera(views[n % views.length]!.camera);
      await submit(e => {
        for (let i = 0; i < count; i++) cache.encodeFrame(e);
      });
      n += count;
    }
  };

  const score = async (tag: string, image: Image, reference: Image, view: LocalityView) => {
    const sample = cropImage(image, view.crop);
    const target = cropImage(reference, view.crop);
    for (const [region, a, b] of [['all', image, reference], ['crop', sample, target]] as const) {
      const m = compareImages(a, b);
      if (m.nonFiniteTest || m.nonFiniteRef) throw new Error(`Non-finite radiance: ${tag} ${region}`);
      metrics[`${tag}_${region}_relmse`] = m.relMse;
      metrics[`${tag}_${region}_relbias`] = m.relBias;
    }
    if (view.dominant !== undefined) {
      const opposite = 2 - view.dominant;
      const dominantReference = channelMean(target, view.dominant);
      metrics[`${tag}_crop_dominant_ratio`] = channelMean(sample, view.dominant) / dominantReference;
      metrics[`${tag}_crop_opposite_ratio`] = channelMean(sample, opposite) / dominantReference;
      metrics[`${tag}_crop_opposite_excess`] = (channelMean(sample, opposite) - channelMean(target, opposite)) / dominantReference;
    }
    if (upload) {
      await uploadImage(tag, image);
      await uploadImage(`${tag}_crop`, sample);
    }
    return sample;
  };

  try {
    await train(warm, 0, 12);
    const counts = await cache.readTrainCount();
    if (!counts.records || cache.trained.frames !== warm) throw new Error('Warm-up did not train the cache for the requested frames');
    metrics.nrc_last_warm_records = counts.records;
    if (cache.hash) metrics.nrc_hash_bytes = cache.hash.bytes;

    for (const [i, view] of views.entries()) {
      const reference = await pathTrace(view, refSpp, 500 + i);
      const tag = movedScene ? 'before' : view.tag;
      await score(`${tag}_reference`, reference, reference, view);
      const plainBefore = await score(`${tag}_plain`, await pathTrace(view, probeSpp, 700), reference, view);
      const cacheBefore = await score(`${tag}_frozen`, await probe(view, 700), reference, view);
      const displayQueries = await cache.readQueryCount();
      if (!displayQueries) throw new Error(`The ${tag} probe did not query the cache`);
      metrics[`${tag}_display_queries`] = displayQueries;
      if (!movedScene) continue;

      // Rebuild traversal from the translated instance and update the shading copy too.
      // Both renderers see the new geometry at the same camera. The cache stays trained.
      const moved = sceneDescToCopyInput(movedScene);
      const builder = createMeshBvhBuilder();
      try {
        const data = await buildTraceScene(
          moved.meshes.map(mesh => ({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : null })),
          moved.instances.map(instance => ({ ...instance, static: true })),
          { build: mesh => builder.build(mesh), mergeStaticInstances: true, nodeFormat: trace.nodeFormat },
        );
        movedTrace = uploadTraceScene(device, data, 'moved NRC geometry');
      } finally {
        builder.dispose();
      }
      const copy = buildSceneCopy({ ...moved, options: { ...moved.options, surface: { positions: true } } });
      if (!writeSceneCopy(device, sceneBuffers, copy)) throw new Error('Moved scene copy did not fit the original allocation');
      plain.setScene(movedTrace, sceneBuffers);
      cache.render.setScene(movedTrace, sceneBuffers);

      const referenceAfter = await pathTrace(view, refSpp, 501);
      const cropBefore = cropImage(reference, view.crop);
      const cropAfter = await score('moved_reference', referenceAfter, referenceAfter, view);
      metrics.moved_reference_crop_change = compareImages(cropAfter, cropBefore).relMse;
      if (!(metrics.moved_reference_crop_change > 0.01)) throw new Error('Moved geometry did not change the reference reflection');
      const plainAfter = await score('moved_plain', await pathTrace(view, probeSpp, 701), referenceAfter, view);
      metrics.moved_plain_crop_change_relmse = changeError(plainBefore, plainAfter, cropBefore, cropAfter);

      const stale = await score('moved_frozen', await probe(view, 701), referenceAfter, view);
      metrics.moved_frozen_crop_change_relmse = changeError(cacheBefore, stale, cropBefore, cropAfter);
      let done = 0;
      for (const frame of checkpoints) {
        await train(frame - done, done, 13);
        done = frame;
        const prefix = `moved_online_f${frame}`;
        const current = await score(prefix, await probe(view, 701), referenceAfter, view);
        metrics[`${prefix}_crop_change_relmse`] = changeError(cacheBefore, current, cropBefore, cropAfter);
      }
      if (cache.trained.frames !== warm + done) throw new Error('Unexpected number of online training frames');
    }

    if (cache.hash) {
      const stats = await cache.hash.readStats();
      metrics.nrc_hash_cells = stats.cells;
      metrics.nrc_hash_misses = stats.misses;
      metrics.nrc_hash_coarse = stats.coarse;
      metrics.nrc_hash_dropped = stats.dropped;
      metrics.nrc_hash_continued = stats.continued;
    }
    await drainDevice(device);
    if (deviceErrors.length) throw new Error(deviceErrors.join(' | '));
    reportResult({
      scene: scene.name, renderer: 'nrc-locality', valid: true,
      config: { arm, width, height, warm, refSpp, probeSpp, ...(movedScene ? { checkpoints: checkpoints.join(',') } : { thickness }) },
      notes: { cache: cache.description, views, training: cache.trained },
      metrics,
    }, { adapter: adapterSummary, deviceErrors });
  } finally {
    cache.destroy();
    movedTrace?.destroy();
    adapter.dispose();
  }
}

main().catch(error => reportError(error, { adapter: adapterSummary, deviceErrors }));
