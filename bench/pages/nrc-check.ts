/**
 * nrc-check (track N, N6a gate): the NRC cache on main's megakernel against a closed form.
 *
 * Closed emissive diffuse room (sphere r = 1, camera in the centre, emission E two-sided, albedo rho, NEE off, no RR), the scene of
 * pt-outputs-check. Every path is deterministic, the true indirect radiance at every vertex is rho E / (1 - rho), and the training
 * targets built from replays have it as their fixed point (with the cache as the tail). After `frames` frames of training, a fresh
 * image with the cache at vertex `queryDepth` must read E / (1 - rho) = 2 in every pixel (plain PT to maxDepth M reads
 * E (1 - rho^(M+1)) / (1 - rho)). Reports the mean and worst pixel error, the training records and queries per frame.
 *
 * Args: frames=300  queryDepth=1  trainDepth=4  precision=f32|f16  infer=portable|sgm|sgm-emu  subgroups=4  steps=4  batch=4096  lr=0.01  spp=16
 *       training=inline|replay  updateBlock=2  trainSelect=tile|pixel (inline)  tol=0.03 (mean)  tolMax=0.1
 */
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, watchDevice } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { tessellateSphere } from '../scenes/geometry.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { PtMegakernel } from '@pmndrs/pathtracer';
import type { Precision } from '@pathtracer/mlp';
import { NrcCache } from '../nrc/cache.ts';
import { inferArmOptions } from './mlp-common.ts';
import { DEFAULT_SHAPE } from '@pathtracer/mlp';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const RHO = 0.5;
const E = 1;
const WIDTH = 32;

function room(maxDepth: number): SceneDesc {
  const m = tessellateSphere(1, 64, 32, { flat: true });
  return {
    version: 1,
    name: 'closed-room',
    purpose: 'Closed diffuse emissive sphere: L = E / (1 - rho).',
    tier: 0,
    camera: { type: 'perspective', position: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], yFovDeg: 60 },
    film: { width: WIDTH, height: WIDTH, filter: 'box' },
    integrator: { maxDepth },
    materials: { wall: { type: 'diffuse', reflectance: [RHO, RHO, RHO] } },
    meshes: [{ name: 'room', geometry: { type: 'triangles', positions: Array.from(m.positions), indices: Array.from(m.indices), normals: Array.from(m.normals) }, material: 'wall', emission: [E, E, E], twoSided: true }],
  };
}

try {
  const args = getArgs();
  const frames = numberArg(args, 'frames', 300);
  const queryDepth = numberArg(args, 'queryDepth', 1);
  const trainDepth = numberArg(args, 'trainDepth', 4);
  const spp = numberArg(args, 'spp', 16);
  const tol = numberArg(args, 'tol', 0.03);
  const tolMax = numberArg(args, 'tolMax', 0.1);
  let precision = (args.precision ?? 'f32') as Precision;
  const maxDepth = 8;

  const adapter = new PtMegakernelAdapter({ nee: false, rrStartDepth: 0 });
  await adapter.load(room(maxDepth));
  const gpu = adapter.gpuContext!;
  adapterSummary = gpu.summary;
  deviceErrors = watchDevice(gpu.device);
  const device = gpu.device;
  const plain = adapter.renderer as PtMegakernel;
  const { caps, trace, sceneBuffers } = adapter.debug;
  const scene = room(maxDepth);
  const camera = scene.camera;

  const arm = inferArmOptions(args.infer ?? 'portable', gpu.adapter, device, DEFAULT_SHAPE, precision, { records: 64, subgroups: numberArg(args, 'subgroups', 4) });
  precision = arm.precision;
  const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], { width: WIDTH, height: WIDTH, camera, maxDepth }, { min: [-1, -1, -1], max: [1, 1, 1] }, {
    precision,
    ...(arm.sgm ? { sgm: arm.sgm } : {}),
    training: args.training === 'replay' ? 'replay' : 'inline',
    updateBlock: numberArg(args, 'updateBlock', 2),
    trainSelect: args.trainSelect === 'pixel' ? 'pixel' : 'tile',
    queryDepth,
    trainDepth,
    trainWidth: WIDTH,
    trainHeight: WIDTH,
    stepsPerFrame: numberArg(args, 'steps', 4),
    batch: numberArg(args, 'batch', 4096),
    learningRate: numberArg(args, 'lr', 0.01),
  });

  // Train while rendering (the render pass is part of a frame), a few frames per submit with a wait in between.
  for (let f = 0; f < frames; f += 10) {
    const encoder = device.createCommandEncoder({ label: 'nrc frames' });
    for (let i = 0; i < 10 && f + i < frames; i++) cache.encodeFrame(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const trainCounts = await cache.readTrainCount();

  // Fresh image with the trained cache, no more training.
  {
    const encoder = device.createCommandEncoder({ label: 'nrc eval' });
    cache.reset(encoder);
    for (let i = 0; i < spp; i++) cache.encodeFrame(encoder, { train: false });
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const queries = await cache.readQueryCount();
  const image = await cache.render.readImage();
  const expected = E / (1 - RHO);
  let sum = 0;
  let worst = 0;
  let n = 0;
  for (let i = 0; i < image.data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const v = image.data[i + c] as number;
      sum += v;
      worst = Math.max(worst, Math.abs(v - expected) / expected);
      n++;
    }
  }
  const mean = sum / n;
  const meanRel = Math.abs(mean - expected) / expected;
  const plainDepthLimited = (E * (1 - RHO ** (maxDepth + 1))) / (1 - RHO);
  const ok = meanRel <= tol && worst <= tolMax && deviceErrors.length === 0;
  await drainDevice(device);
  reportResult(
    {
      scene: 'nrc-check',
      renderer: 'ours',
      config: { frames, queryDepth, trainDepth, precision, spp, infer: arm.description, training: cache.settings.training, updateBlock: cache.settings.updateBlock, trainSelect: cache.settings.trainSelect },
      valid: ok,
      metrics: { train_paths: cache.trainPaths, mean, expected, mean_rel_error: meanRel, max_rel_error: worst, plain_depth_limited: plainDepthLimited, train_records: trainCounts.records, train_batch: trainCounts.batch, queries_last_pass: queries },
      notes: { infer: cache.mlp.inferBackend },
    },
    { adapter: adapterSummary, deviceErrors },
  );
  if (!ok) throw new Error(`nrc-check: mean ${mean.toFixed(4)} vs ${expected} (rel ${meanRel.toExponential(2)}), worst pixel rel ${worst.toExponential(2)}${deviceErrors.length ? `; device errors: ${deviceErrors.join('; ')}` : ''}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
