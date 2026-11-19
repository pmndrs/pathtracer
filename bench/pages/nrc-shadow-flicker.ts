import type { PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { NrcCache } from '../nrc/cache.ts';
import { phase10Settings } from '../nrc/phase10.ts';
import { NrcPipelineSession, sceneBounds } from '../nrc/pipeline-session.ts';
import { readTextureRgba16f } from '../realtime/chain.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, uploadImage, watchDevice } from '../runner/page-client.ts';
import { nrcMaterialsShadow as scene } from '../scenes/tier2/nrc-materials.ts';
import type { Image } from '../src/adapter.ts';

const width = 480, height = 270;
const frames = 64, warm = 256, refSpp = 8192;
const camera = scene.camera;
const regions = { shadow: [248, 20, 48, 28], lit: [425, 160, 48, 28], edge: [304, 230, 24, 20] } as const;
const seed = numberArg(getArgs(), 'seed', 1);
const settings = { ...phase10Settings('mlp'), seed, frozenDisplayOnly: true };
const metrics: Record<string, number> = {};
const notes: Record<string, unknown> = { regions, camera };
const deviceErrors: string[] = [];

type Film = Pick<Image, 'width' | 'height' | 'data'> & { channels?: number };
function crop(img: Film, [x, y, w, h]: readonly [number, number, number, number]): Float64Array {
  const pixels = new Float64Array(w * h);
  const stride = img.channels ?? 4;
  for (let i = 0; i < pixels.length; i++) {
    const k = stride * ((y + Math.floor(i / w)) * img.width + x + i % w);
    pixels[i] = 0.2126 * img.data[k]! + 0.7152 * img.data[k + 1]! + 0.0722 * img.data[k + 2]!;
  }
  return pixels;
}
type Series = Record<string, Float64Array[]>;
const series: Record<string, Series> = {};
function store(arm: string, img: Film) {
  const run = series[arm] ??= {};
  for (const [name, rect] of Object.entries(regions)) (run[name] ??= []).push(crop(img, rect));
}

function summarize() {
  for (const name of Object.keys(regions)) {
    const reference = series.reference![name]![0]!;
    const n = reference.length;
    const mean = reference.reduce((a, b) => a + b) / n;
    metrics[name + '_reference_luminance'] = mean;
    for (const [arm, run] of Object.entries(series)) {
      if (arm === 'reference') continue;
      const images = run[name]!;
      let pixelDelta2 = 0, patchDelta2 = 0, transitions = 0, lateError2 = 0, lateBias = 0;
      for (let t = 0; t < images.length; t++) {
        const cur = images[t]!, prev = images[t - 1];
        let err2 = 0, bias = 0, sumDelta = 0;
        for (let i = 0; i < n; i++) {
          const error = cur[i]! - reference[i]!;
          err2 += error * error;
          bias += error;
          // Score the final 48 frames in every arm, including the filter with newly reset history.
          if (prev && t >= 16) {
            const delta = cur[i]! - prev[i]!;
            pixelDelta2 += delta * delta;
            sumDelta += delta;
          }
        }
        if ([1, 4, 16, 64].includes(t + 1)) metrics[arm + '_' + name + '_error_f' + (t + 1)] = Math.sqrt(err2 / n) / mean;
        if (t >= 16) {
          transitions++;
          lateError2 += err2;
          lateBias += bias;
          patchDelta2 += (sumDelta / n) ** 2;
        }
      }
      const prefix = arm + '_' + name;
      metrics[prefix + '_pixel_flicker'] = Math.sqrt(pixelDelta2 / (transitions * n)) / mean;
      metrics[prefix + '_patch_flicker'] = Math.sqrt(patchDelta2 / transitions) / mean;
      metrics[prefix + '_error'] = Math.sqrt(lateError2 / (transitions * n)) / mean;
      metrics[prefix + '_bias'] = lateBias / (transitions * n * mean);
    }
    // The paired residual uses identical random paths. Its RMS is not an additive fraction of total variance.
    const diff = series.online![name]!.map((im, t) => im.map((v, i) => v - series.frozen![name]![t]![i]!));
    let delta2 = 0;
    for (let t = 16; t < frames; t++) for (let i = 0; i < n; i++) delta2 += (diff[t]![i]! - diff[t - 1]![i]!) ** 2;
    metrics[name + '_paired_delta_to_online_rms'] = Math.sqrt(delta2 / ((frames - 16) * n)) / mean / metrics['online_' + name + '_pixel_flicker']!;
    const firstDelta = diff[0]!.reduce((v, x) => Math.max(v, Math.abs(x)), 0);
    metrics[name + '_first_pair_max_delta'] = firstDelta;
    if (firstDelta > 1e-4) throw new Error('Paired arms start with different predictions in ' + name);
  }
}

async function run() {
  const adapter = new PtMegakernelAdapter();
  await adapter.load(scene);
  const { device, summary } = adapter.gpuContext!;
  const errors = watchDevice(device);
  const { trace, sceneBuffers, caps } = adapter.debug;
  const plain = adapter.renderer as PtMegakernel;
  const film = { width, height, camera, maxDepth: 6, rrStartDepth: 4 };
  plain.setFilm(width, height);
  plain.setCamera(camera);
  plain.setLimits(film);
  const createCache = () => NrcCache.create(device, caps, plain, [trace, sceneBuffers], film, sceneBounds(adapter), {
    ...settings, precision: 'f16', infer: 'tiled',
  });
  let cache = await createCache();
  async function submit(work: (encoder: GPUCommandEncoder) => void) {
    const begin = performance.now();
    const encoder = device.createCommandEncoder();
    work(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - begin;
  }
  for (let i = 0; i < warm; i++) await submit(e => cache.encodeFrame(e));
  plain.setSeed(109805);
  await submit(e => plain.reset(e));
  for (let i = 0; i < refSpp; i += 32) await submit(e => plain.encode(e, 32));
  const ref = await plain.readImage();
  store('reference', ref);
  await uploadImage('reference', { ...ref, channels: 4 });

  // Frozen first on the SAME cache. No weights or optimizer moments are reloaded between arms.
  for (const arm of ['frozen', 'online', 'plain'] as const) {
    const ms: number[] = [];
    const renderer = arm === 'plain' ? plain : cache.render;
    for (let f = 0; f < frames; f++) {
      renderer.setSeed(10000 * seed + f);
      ms.push(await submit(e => {
        renderer.reset(e);
        if (arm === 'plain') plain.encode(e, 1);
        else cache.encodeFrame(e, { train: arm === 'online' });
      }));
      const img = await renderer.readImage();
      store(arm, img);
      if (f === 15) await uploadImage(arm + '-16', { ...img, channels: 4 });
    }
    metrics[arm + '_ms'] = ms.slice(4).sort((a, b) => a - b)[30]!;
    if (arm === 'frozen' && cache.trained.frames !== warm) throw new Error('Frozen arm updated the cache');
  }
  cache.destroy();
  cache = await createCache();
  for (let i = 0; i < warm; i++) await submit(e => cache.encodeFrame(e));
  cache.render.setSeed(10000 * seed);
  await submit(e => cache.reset(e));
  for (let i = 0; i < frames; i++) {
    // Live still mode keeps the Sobol sample index advancing and averages successive training frames.
    await submit(e => cache.encodeFrame(e));
    const img = await cache.render.readImage();
    store('accumulated', img);
    if (i === 15) await uploadImage('accumulated-16', { ...img, channels: 4 });
  }
  cache.destroy();
  adapter.dispose();
  deviceErrors.push(...errors);

  const session = await NrcPipelineSession.create({ scene, width, height, displayWidth: width, displayHeight: height, cache: settings, denoise: 'svgf' });
  const pipelineErrors = watchDevice(session.device);
  for (let i = 0; i < warm; i += 8) {
    session.trainOnly(camera, 8);
    await session.device.queue.onSubmittedWorkDone();
  }
  session.cutNext();
  for (let i = 0; i < frames; i++) {
    await session.frame(camera, { arms: ['nrc'] });
    for (const stage of ['raw', 'denoised'] as const) {
      const img = await readTextureRgba16f(session.device, session.arms.nrc.stage(stage));
      store(stage, img);
      if (i === 15) await uploadImage('pipeline-' + stage + '-16', img);
    }
  }
  notes.pipelineTiming = await session.timeStages(24);
  await drainDevice(session.device);
  deviceErrors.push(...pipelineErrors);
  session.destroy();
  if (deviceErrors.length) throw new Error(deviceErrors.join('\n'));
  summarize();
  reportResult({ scene: scene.name, renderer: 'shadow-flicker', config: { width, height, warm, frames, refSpp, seed }, metrics, notes }, { adapter: summary, deviceErrors });
}

run().catch(error => reportError(error, { deviceErrors }));
