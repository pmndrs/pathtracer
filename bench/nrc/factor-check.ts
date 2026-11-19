import { layerOffset, recordLoss, type MlpNetwork, type Precision } from '@pathtracer/mlp';
import type { PtFrame, PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { NrcCache, type NrcDirectionInput, type NrcFactorization } from './cache.ts';

export function reconstructionError(frame: PtFrame): number {
  let error = 0;
  for (let i = 0; i < frame.radiance.length; i++) {
    if (i % 4 === 3) continue;
    const reconstructed = frame.diffuse[i]! * frame.albedo[i]! + frame.specular[i]! * frame.specularFactor[i]! + frame.emission[i]!;
    error = Math.max(error, Math.abs(reconstructed - frame.radiance[i]!));
  }
  return error;
}

const checker: SceneDesc = {
  version: 1, tier: 0, name: 'saturated-checker', purpose: 'Measure black leakage and color contrast against a unit sky.',
  camera: { type: 'perspective', position: [0, 2, 0], target: [0, 0, 0], up: [0, 0, -1], yFovDeg: 65 },
  film: { width: 64, height: 64, filter: 'box' },
  integrator: { maxDepth: 4 }, environment: { type: 'uniform', radiance: [1, 1, 1] },
  textures: { checker: { generator: 'checker', colorSpace: 'srgb', width: 256, height: 256, cells: 16, a: [255, 0, 64], b: [0, 255, 64] } },
  materials: { card: { type: 'diffuse', reflectance: [1, 1, 1], reflectanceTexture: 'checker' } },
  meshes: [{ name: 'card', geometry: { type: 'quad', size: [4, 4] }, uv: { scale: [1, 1] }, material: 'card' }],
};

// A single sampled suffix makes the CPU loss oracle independent of GPU target compaction order.
function metalScene(surface: string): SceneDesc {
  if (!['rough', 'smooth', 'fuzz'].includes(surface)) throw new Error('surface must be rough, smooth or fuzz');
  return {
    ...checker, textures: {}, name: 'dark-metal', purpose: 'Measure the training target of a rough, dark metal against the same traced suffix.',
    film: { width: 1, height: 1, filter: 'box' },
    camera: { ...checker.camera, position: [0, 1, 3], yFovDeg: 2 },
    materials: { card: { type: 'openpbr', baseColor: [0.01, 0.25, 0.64], baseMetalness: 1, specularRoughness: surface === 'smooth' ? 0 : 0.3, fuzzWeight: surface === 'fuzz' ? 1 : 0 } },
    meshes: [{ name: 'card', geometry: { type: 'quad', size: [100, 100] }, material: 'card' }],
  };
}

export async function checkFactorization(
  kind: 'checker' | 'room' | 'metal', room: SceneDesc, mode: NrcFactorization, precision: Precision, infer: 'tiled' | 'perRecord',
  setDiagnostics: (summary: AdapterSummary, errors: string[]) => void, surface = 'rough', network: MlpNetwork = 'default', directionInput: NrcDirectionInput = 'view',
): Promise<Record<string, number>> {
  const scene: SceneDesc = kind === 'checker' ? checker : kind === 'metal' ? metalScene(surface) : { ...room, materials: { wall: { type: 'diffuse', reflectance: [0, 0.5, 0.75] } } };
  const { width, height } = scene.film;
  const adapter = new PtMegakernelAdapter({ nee: false, rrStartDepth: 0, frame: true, scenePacking: 'packed' });
  await adapter.load(scene);
  const { device, summary } = adapter.gpuContext!;
  const errors = watchDevice(device);
  setDiagnostics(summary, errors);
  const { caps, trace, sceneBuffers } = adapter.debug;
  const plain = adapter.renderer as PtMegakernel;
  const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], { width, height, camera: scene.camera, maxDepth: scene.integrator.maxDepth },
    { min: [-2, -1, -2], max: [2, 1, 2] }, {
      precision, infer, network, directionInput, materialInputs: 'evaluated', factorization: mode, queryDepth: 0, trainDepth: 4, updateBlock: 2,
      ...(kind !== 'checker' ? { learningRate: 0, ema: 'off', stepsPerFrame: 1 } : {}),
      ...(kind === 'metal' ? { trainDepth: 1, updateBlock: 1, batch: 1 } : {}),
    });
  plain.setSeed(cache.settings.seed);
  const metrics: Record<string, number> = {};
  function submit(encode: (e: GPUCommandEncoder) => void) {
    const e = device.createCommandEncoder();
    encode(e);
    device.queue.submit([e.finish()]);
  }
  try {
    if (kind !== 'checker') {
      // Unit emission in the room gives physical indirect [0,1,3] and demodulated indirect [0,2,4].
      const prediction = kind === 'metal' ? [0.5, 0.5, 0.5] : mode === 'off' ? [0, 1, 3] : [0, 2, 4];
      const mlp = cache.mlp!;
      const { shape } = mlp;
      const weights = new Float32Array(mlp.weightCount);
      weights[shape.inputs - 1] = 1;
      for (let l = 1; l < shape.hidden; l++) weights[layerOffset(shape, l)] = 1;
      const output = layerOffset(shape, shape.hidden);
      for (const [c, v] of prediction.entries()) weights[output + c * shape.width] = v;
      mlp.uploadWeights(weights);
    }
    if (kind === 'metal') {
      submit(e => { plain.encode(e, 1); cache.encodeFrame(e); });
      const [traced, frame, { loss, batch }] = await Promise.all([plain.readFrame(), cache.render.readFrame(), cache.readLoss()]);
      const factor = mode === 'reflectance' && surface === 'rough' ? [1 / 16, 0.25, 0.64] : [1, 1, 1];
      const target = factor.map((scale, c) => traced.radiance[c]! / scale);
      const fixed = [0.5, 0.5, 0.5];
      metrics.normalized_loss_error = Math.abs(loss - recordLoss(fixed, target));
      metrics.beauty_error = Math.max(...factor.map((scale, c) => Math.abs(frame.radiance[c]! - scale * fixed[c]!)));
      metrics.planes_error = reconstructionError(frame);
      if (!(batch === 1 && target[0]! > 0 && metrics.normalized_loss_error < 0.005 && metrics.beauty_error < 2e-5 && metrics.planes_error < 2e-5)) {
        throw new Error(`Metal target ${target} loss ${loss} batch ${batch}, errors ${JSON.stringify(metrics)}`);
      }
    } else if (kind === 'room') {
      for (const depth of [0, 4, 6]) {
        cache.setQueryDepth(depth);
        submit(e => { cache.reset(e); cache.encodeFrame(e); });
        const [frame, { loss, batch }] = await Promise.all([cache.render.readFrame(), cache.readLoss()]);
        let error = 0;
        for (let i = 0; i < frame.radiance.length; i++) {
          if (i % 4 !== 3) error = Math.max(error, Math.abs(frame.radiance[i]! - [1, 2, 4][i % 4]!));
        }
        const planes = reconstructionError(frame);
        Object.assign(metrics, { ['beauty_max_k' + depth]: error, ['planes_max_k' + depth]: planes, ['loss_k' + depth]: loss });
        if (!(error < 2e-5 && planes < 2e-5 && batch > 0 && loss < 1e-8)) throw new Error('Factorized fixed point failed ' + JSON.stringify(metrics));
      }
    } else {
      // Compare one fresh, same-seed frame at each training age. Every traced suffix from this plane sees the unit sky.
      for (let f = 1; f <= 256; f++) {
        const checkpoint = [1, 16, 64, 256].includes(f);
        cache.render.setFrameIndex(f);
        plain.setFrameIndex(f);
        submit(e => { cache.encodeFrame(e); if (checkpoint) plain.encode(e, 1); });
        if (f % 8 === 0) await device.queue.onSubmittedWorkDone();
        if (!checkpoint) continue;
        const [frame, ref] = await Promise.all([cache.render.readFrame(), plain.readFrame()]);
        let dark = 0, light = 0, darkCount = 0, lightCount = 0, squared = 0, energy = 0;
        for (let i = 0; i < ref.radiance.length; i++) {
          if (i % 4 === 3) continue;
          if (Math.abs(frame.albedo[i]! - ref.albedo[i]!) > 1e-6) throw new Error('Mismatched checker sample positions');
          const v = ref.radiance[i]!, d = frame.radiance[i]! - v;
          if (i % 4 < 2 && v < 1e-6) { dark += d; darkCount++; }
          if (i % 4 < 2 && v > 0.999999) { light += d; lightCount++; }
          squared += d * d;
          energy += v * v;
        }
        if (Math.min(darkCount, lightCount) < width * height / 2) throw new Error('Checker reference lacks black or saturated samples.');
        const darkBias = dark / darkCount, lightBias = light / lightCount;
        const planes = reconstructionError(frame);
        const error = squared / energy;
        if (!(planes < 2e-5 && Number.isFinite(error) && (mode === 'off' || Math.abs(darkBias) < 1e-5))) throw new Error('Invalid checker image or reconstruction ' + planes);
        metrics['black_bias_f' + f] = darkBias;
        metrics['saturated_bias_f' + f] = lightBias;
        metrics['contrast_f' + f] = 1 + lightBias - darkBias;
        metrics['normalized_mse_f' + f] = error;
        metrics['planes_max_f' + f] = planes;
        if (f === 16 || f === 256) {
          for (const [label, image] of [['traced', ref], ['nrc', frame]] as const) {
            await uploadImage(label + '-f' + f, { width, height, channels: 4, data: image.radiance });
          }
        }
      }
    }
    await drainDevice(device);
    if (errors.length) throw new Error(errors.join('; '));
    return metrics;
  } finally {
    cache.destroy();
    adapter.dispose();
  }
}
