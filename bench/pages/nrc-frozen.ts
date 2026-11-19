/** Paired frozen-frame regression. Both arms use the same cache and the same trained weight snapshot. */
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { NrcCache } from '../nrc/cache.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { tessellateSphere } from '../scenes/geometry.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import type { PtMegakernel } from '@pmndrs/pathtracer';
import type { Precision } from '@pathtracer/mlp';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

async function main() {
  const args = getArgs();
  const mirror = args.material === 'mirror';
  const hashMiss = args.model === 'hash';
  const width = numberArg(args, 'width', 32);
  const height = numberArg(args, 'height', 32);
  const repeats = numberArg(args, 'repeats', 0);
  const maxDepth = 8;
  const trainDepth = 4;
  const pixels = width * height;
  const mesh = tessellateSphere(1, 64, 32, { flat: true });
  const scene: SceneDesc = {
    version: 1,
    name: mirror ? 'frozen-mirror-room' : 'frozen-diffuse-room',
    purpose: 'Closed emissive room with cache queries on every display path.',
    tier: 0,
    camera: { type: 'perspective', position: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], yFovDeg: 60 },
    film: { width, height, filter: 'box' },
    integrator: { maxDepth },
    materials: {
      wall: mirror
        ? { type: 'conductor', eta: [0.2, 0.924, 1.102], k: [3.912, 2.452, 2.142], roughness: 0 }
        : { type: 'diffuse', reflectance: [0.5, 0.45, 0.4] },
    },
    meshes: [{
      name: 'room',
      geometry: { type: 'triangles', positions: Array.from(mesh.positions), indices: Array.from(mesh.indices), normals: Array.from(mesh.normals) },
      material: 'wall',
      emission: [1, 0.6, 0.35],
      twoSided: true,
    }],
  };
  const adapter = new PtMegakernelAdapter({ nee: false, rrStartDepth: 0, frame: true, scenePacking: 'packed' });
  await adapter.load(scene);
  const { device, summary } = adapter.gpuContext!;
  adapterSummary = summary;
  deviceErrors = watchDevice(device);
  const { caps, trace, sceneBuffers } = adapter.debug;
  const precision: Precision = args.precision === 'f16' ? 'f16' : 'f32';
  const cache = await NrcCache.create(device, caps, adapter.renderer as PtMegakernel, [trace, sceneBuffers], {
    width, height, camera: scene.camera, maxDepth,
  }, { min: [-1, -1, -1], max: [1, 1, 1] }, {
    precision, infer: args.infer === 'tiled' ? 'tiled' : 'perRecord',
    trainDepth, queryDepth: 1, updateBlock: 2, stepsPerFrame: 1, batch: 1024,
    trainSelect: args.trainSelect === 'pixel' ? 'pixel' : 'tile',
    ...(hashMiss ? { model: 'hash' as const, hash: { continueOnMiss: true, log2Entries: 10, minSamples: 1e6, fallbackLevels: 0 } } : {}),
  });

  const metrics: Record<string, number> = {};
  const submit = (run: (encoder: GPUCommandEncoder) => void) => {
    const encoder = device.createCommandEncoder();
    run(encoder);
    device.queue.submit([encoder.finish()]);
  };
  submit((encoder) => {
    for (let frame = 0; frame < 4; frame++) cache.encodeFrame(encoder);
  });
  const trained = cache.trained;
  const originalRecords = await cache.readTrainCount();
  if (!originalRecords.records) throw new Error('Frozen comparison requires trained records.');

  async function sample(displayOnly: boolean) {
    cache.settings.frozenDisplayOnly = displayOnly;
    cache.render.setFrameIndex(0);
    submit((encoder) => cache.encodeFrame(encoder, { train: false }));
    const [image, queries] = await Promise.all([cache.render.readFrame(), cache.readQueryCount()]);
    return { image, queries };
  }

  for (const depth of ['gated', 0, 1, trainDepth, 6] as const) {
    cache.settings.warmupFrames = depth === 'gated' ? trained.frames + 1 : 0;
    cache.setQueryDepth(depth === 'gated' ? 1 : depth);
    const control = await sample(false);
    cache.hash?.resetStats();
    const frozen = await sample(true);
    const displayQueries = depth === 'gated' ? 0 : pixels;
    if (frozen.queries !== displayQueries) {
      throw new Error(`Frozen ${depth} emitted ${frozen.queries} queries, expected ${displayQueries} display queries.`);
    }
    if (depth !== trainDepth && !(control.queries > frozen.queries)) {
      throw new Error(`Control ${depth} emitted no training-only queries.`);
    }
    if (cache.hash) {
      const { continued } = await cache.hash.readStats();
      if (continued !== displayQueries) throw new Error(`Frozen ${depth} continued ${continued} undertrained hash queries, expected ${displayQueries}.`);
      metrics[`continuations_frozen_${depth}`] = continued;
    }
    metrics[`queries_control_${depth}`] = control.queries;
    metrics[`queries_frozen_${depth}`] = frozen.queries;
    for (const plane of ['radiance', 'diffuse', 'specular', 'emission', 'normalViewZ', 'motionRoughness', 'albedo', 'specularFactor'] as const) {
      let error = 0;
      const a = control.image[plane];
      const b = frozen.image[plane];
      for (let i = 0; i < a.length; i++) error = Math.max(error, Math.abs(a[i]! - b[i]!));
      if (!(error <= 1e-5)) throw new Error(`Frozen ${depth} ${plane} max error ${error}.`);
      metrics[`error_${plane}_${depth}`] = error;
    }
  }
  const records = await cache.readTrainCount();
  if (trained.frames !== cache.trained.frames || trained.steps !== cache.trained.steps || records.records !== originalRecords.records || records.batch !== originalRecords.batch) {
    throw new Error('Frozen comparison changed optimizer state or training record counters.');
  }

  // State transitions must work between dispatches within one submitted command buffer.
  cache.setQueryDepth(1);
  submit((encoder) => {
    cache.encodeFrame(encoder, { train: true });
    cache.encodeFrame(encoder, { train: false });
  });
  if (await cache.readQueryCount() !== pixels) throw new Error('Mixed-submit frozen frame retained the training hook.');
  submit((encoder) => {
    cache.encodeFrame(encoder, { train: false });
    cache.encodeFrame(encoder, { train: true });
  });
  if (await cache.readQueryCount() <= pixels) throw new Error('Mixed-submit training frame retained the frozen hook.');

  if (repeats > 0) {
    const timings = { legacy: [] as number[], frozen: [] as number[] };
    for (let r = -3; r < repeats; r++) {
      for (const arm of r % 2 === 0 ? ['legacy', 'frozen'] as const : ['frozen', 'legacy'] as const) {
        cache.settings.frozenDisplayOnly = arm === 'frozen';
        cache.render.setFrameIndex(r + 3);
        const start = performance.now();
        submit((encoder) => cache.encodeFrame(encoder, { train: false }));
        await device.queue.onSubmittedWorkDone();
        if (r >= 0) timings[arm].push(performance.now() - start);
      }
    }
    for (const [arm, samples] of Object.entries(timings)) {
      samples.sort((a, b) => a - b);
      for (const q of [50, 95]) metrics[`wall_ms_${arm}_p${q}`] = samples[Math.ceil(samples.length * q / 100) - 1]!;
    }
  }

  await drainDevice(device);
  if (deviceErrors.length) throw new Error(deviceErrors.join(' | '));
  reportResult({ scene: scene.name, renderer: 'nrc-frozen', valid: true, config: { width, height, precision, repeats, model: cache.settings.model }, metrics }, { adapter: adapterSummary, deviceErrors });
  cache.destroy();
  adapter.dispose();
}

main().catch((error) => reportError(error, { adapter: adapterSummary, deviceErrors }));
