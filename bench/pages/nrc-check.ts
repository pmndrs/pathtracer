/**
 * nrc-check (track N, N6a gate): the NRC cache on main's megakernel against a closed form.
 *
 * Closed emissive diffuse room (sphere r = 1, camera in the centre, emission E two-sided, albedo rho, NEE off, no RR), the scene of
 * pt-outputs-check. Every path is deterministic, the true indirect radiance at every vertex is rho E / (1 - rho), and the training
 * targets built from replays have it as their fixed point (with the cache as the tail). After `frames` frames of training, a fresh
 * image with the cache at vertex `queryDepth` must read E / (1 - rho) = 2 in every pixel (plain PT to maxDepth M reads
 * E (1 - rho^(M+1)) / (1 - rho)). Reports the mean and worst pixel error, the training records and queries per frame.
 *
 * Args: network=default|compact  frames=300  queryDepth=1  trainDepth=4  precision=f32|f16  infer=portable|tiled|sgm|sgm-emu  subgroups=4  forward=sgm fwdLayout=row|blocked gradient=sgm (training, N2r-d)  steps=4  batch=4096  lr=0.01  spp=16
 *       check=spread checks adaptive stop depths; check=mirror checks a moving reflected card; check=delta-chain checks a secondary mirror.
 *       check=hash-miss checks empty, undertrained and valid black cells against plain and truncated paths.
 *       check=directions exposes the direction feature lanes for directionInput=view|reflected.
 *       check=materials verifies evaluated shading normal, roughness, reflectance, and lobe.
 *       check=factor-room|factor-metal|checker tests bootstrap, rough-metal training, or saturated texture contrast with factorization=off|lambert|reflectance.
 *       check=depths checks inline gate/target/stop-depth semantics with an analytic frozen network.
 *       training=inline|replay  updateBlock=2  trainSelect=tile|pixel (inline)  tol=0.03 (mean)  tolMax=0.1
 */
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, watchDevice } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { tessellateSphere, translation } from '../scenes/geometry.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { PtMegakernel } from '@pmndrs/pathtracer';
import type { Precision, SgmScalar } from '@pathtracer/mlp';
import { NrcCache, directionArg, factorizationArg, materialArgs } from '../nrc/cache.ts';
import { checkFactorization, reconstructionError } from '../nrc/factor-check.ts';
import { checkMaterialInputs } from '../nrc/material-check.ts';
import { terminationArgs } from '../nrc/termination.ts';
import { hashGridArgs } from '../nrc/hash-grid.ts';
import { forwardArgs, forwardArmOptions, gradientArmOptions, inferArmOptions, networkArg, readBuffer } from './mlp-common.ts';
import { MLP_SHAPES, recordLoss, layerOffset } from '@pathtracer/mlp';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const RHO = 0.5;
const E = 1;
const WIDTH = 32;

async function checkWeightReload(cache: NrcCache): Promise<Record<string, number>> {
  const { device } = cache;
  const weights = new Float32Array(await readBuffer(device, cache.mlp!.master));
  // Keep cache inference enabled in these frozen frames even though loading resets the warm-up counter.
  cache.settings.warmupFrames = 0;
  async function render() {
    const encoder = device.createCommandEncoder();
    cache.render.setSeed(17);
    cache.reset(encoder);
    cache.encodeFrame(encoder, { train: false });
    device.queue.submit([encoder.finish()]);
    return (await cache.render.readImage()).data;
  }
  cache.loadWeights(weights);
  const loaded = await render();
  cache.resetNetwork();
  const reset = await render();
  cache.loadWeights(weights);
  const reloaded = await render();
  let resetDelta = 0;
  let reloadDelta = 0;
  for (let i = 0; i < loaded.length; i++) {
    resetDelta = Math.max(resetDelta, Math.abs(loaded[i]! - reset[i]!));
    reloadDelta = Math.max(reloadDelta, Math.abs(loaded[i]! - reloaded[i]!));
  }
  // GPU record ordering can change f32 summation by a few ulps between submissions.
  if (!(resetDelta > 1e-6) || !(reloadDelta <= 1e-6) || cache.trained.frames !== 0 || cache.trained.steps !== 0) {
    throw new Error(`nrc-check: checkpoint reset delta ${resetDelta}, reload delta ${reloadDelta}, trained ${cache.trained.steps} steps`);
  }
  return { reset_max_delta: resetDelta, reload_max_delta: reloadDelta };
}

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

// This room has a constant indirect suffix of 1. Route a constant NRC input through one neuron in each layer.
async function checkDepths(cache: NrcCache, plain: PtMegakernel): Promise<Record<string, number>> {
  const mlp = cache.mlp!;
  const weights = new Float32Array(mlp.weightCount);
  const shape = mlp.shape;
  weights[shape.inputs - 1] = 1;
  for (let layer = 1; layer < shape.hidden; layer++) weights[layerOffset(shape, layer)] = 1;
  for (let c = 0; c < 3; c++) weights[layerOffset(shape, shape.hidden) + c * shape.width] = 1;
  mlp.uploadWeights(weights);
  const device = cache.device;
  const metrics: Record<string, number> = {};
  const truncated = (depth: number) => E * (1 - RHO ** (depth + 1)) / (1 - RHO);
  async function checkImage(name: string, expected: number | Float32Array) {
    const { data } = await cache.render.readImage();
    let worst = 0;
    for (let i = 0; i < data.length; i++) {
      if (i % 4 !== 3) worst = Math.max(worst, Math.abs(data[i]! - (typeof expected === 'number' ? expected : expected[i]!)));
    }
    metrics[name] = worst;
    if (!(worst < 1e-5)) throw new Error(`nrc depths: ${name} max error ${worst}`);
  }

  // Every training tile reaches M. Gated and active targets should both be the analytic fixed point (loss zero).
  for (const mode of ['gated', 'active']) {
    const encoder = device.createCommandEncoder();
    cache.reset(encoder);
    cache.encodeFrame(encoder);
    plain.reset(encoder);
    plain.encode(encoder, 1);
    device.queue.submit([encoder.finish()]);
    const queries = await cache.readQueryCount();
    const expected = cache.trainPaths + (mode === 'active' ? WIDTH * WIDTH : 0);
    if (queries !== expected) throw new Error(`nrc depths: ${mode} emitted ${queries} queries, expected ${expected}`);
    metrics[mode + '_queries'] = queries;
    const { batch, loss } = await cache.readLoss();
    if (!batch || !(loss < 1e-8)) throw new Error(`nrc depths: ${mode} target loss ${loss} in batch ${batch}`);
    metrics[mode + '_loss'] = loss;
    await checkImage(mode, mode === 'gated' ? (await plain.readImage()).data : E / (1 - RHO));
  }

  // With a black prediction each pixel reveals exactly where it stopped. Open the gate and change the depth in one submit.
  mlp.uploadWeights(weights.fill(0));
  cache.settings.warmupFrames = cache.trained.frames + 1;
  const encoder = device.createCommandEncoder();
  cache.reset(encoder);
  cache.encodeFrame(encoder); // plain to 8, trains at 4, opens gate
  for (const depth of [6, 4, 1]) {
    cache.setQueryDepth(depth);
    cache.encodeFrame(encoder, { train: false });
  }
  device.queue.submit([encoder.finish()]);
  await checkImage('batched', (truncated(8) + truncated(6) + truncated(4) + truncated(1)) / 4);
  return metrics;
}

// With a black network and one training layer, every target in the emissive room has the same closed form.
async function checkFullTail(cache: NrcCache): Promise<Record<string, number>> {
  const full = cache.settings.fullSuffixEvery === 1;
  const truncated = (depth: number) => E * (1 - RHO ** (depth + 1)) / (1 - RHO);
  const target = truncated(full ? 8 : 1) - E;
  const expectedLoss = recordLoss([0, 0, 0], [target, target, target]);
  const metrics: Record<string, number> = {};
  for (const depth of [0, 1, 3, 9]) {
    cache.setQueryDepth(depth);
    const encoder = cache.device.createCommandEncoder();
    cache.reset(encoder);
    cache.encodeFrame(encoder);
    cache.device.queue.submit([encoder.finish()]);
    const [frame, { batch, loss }, queries] = await Promise.all([cache.render.readFrame(), cache.readLoss(), cache.readQueryCount()]);
    let error = 0;
    for (let i = 0; i < frame.radiance.length; i++) {
      if (i % 4 !== 3) error = Math.max(error, Math.abs(frame.radiance[i]! - truncated(Math.min(8, depth))));
    }
    const planeError = reconstructionError(frame);
    const expectedQueries = (depth <= 8 ? WIDTH * WIDTH : 0) + (full || depth === 1 ? 0 : cache.trainPaths);
    const lossError = Math.abs(loss / expectedLoss - 1);
    if (!(batch > 0 && error < 2e-5 && planeError < 2e-5 && lossError < 1e-4 && queries === expectedQueries)) {
      throw new Error(`full tail K=${depth}: beauty ${error}, planes ${planeError}, loss ${loss}/${expectedLoss}, queries ${queries}/${expectedQueries}`);
    }
    Object.assign(metrics, { ['loss_rel_error_k' + depth]: lossError, ['beauty_error_k' + depth]: error, ['planes_error_k' + depth]: planeError });
  }
  return metrics;
}

async function checkHashMiss(cache: NrcCache, plain: PtMegakernel) {
  const { device } = cache;
  const hash = cache.hash!;
  const metrics: Record<string, number> = {};
  // A unit sphere fits in eight cells of size two around the origin. Give each one a known black target.
  const records = new Float32Array(8 * 16);
  for (let i = 0; i < 8; i++) {
    records.set([i & 1 ? 0.5 : -0.5, i & 2 ? 0.5 : -0.5, i & 4 ? 0.5 : -0.5], 16 * i);
  }
  const storage = (data: Float32Array | Uint32Array) => {
    const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer);
    return buffer;
  };
  const targets = storage(records);
  const count = storage(new Uint32Array([0, 8]));
  hash.bindRecords(targets, count, 8);
  plain.setSeed(cache.settings.seed);
  for (const state of ['empty', 'undertrained', 'black']) {
    if (state !== 'empty') {
      const e = device.createCommandEncoder();
      hash.encodeUpdate(e);
      device.queue.submit([e.finish()]);
    }
    for (const k of [1, 4, 6]) {
      cache.setQueryDepth(k);
      hash.resetStats();
      const e = device.createCommandEncoder();
      plain.reset(e);
      plain.encode(e, 1);
      cache.reset(e);
      cache.encodeFrame(e, { train: false });
      device.queue.submit([e.finish()]);
      const [{ data: traced }, { data: actual }, stats] = await Promise.all([plain.readImage(), cache.render.readImage(), hash.readStats()]);
      let error = 0;
      for (let i = 0; i < actual.length; i++) {
        if (i % 4 === 3) continue;
        const expected = state === 'black' ? E * (1 - RHO ** (k + 1)) / (1 - RHO) : traced[i]!;
        error = Math.max(error, Math.abs(actual[i]! - expected));
      }
      metrics[state + '_max_error_k' + k] = error;
      if (!(error < 1e-5) || stats.continued !== (state === 'black' ? 0 : WIDTH * WIDTH)) {
        throw new Error(`hash-miss: ${state} K=${k}, max error ${error}, continued ${stats.continued}`);
      }
    }
  }
  targets.destroy();
  count.destroy();
  return metrics;
}

// The camera looks down. A colored emissive card above the camera is visible only in the conductor.
function reflectionScene(maxDepth: number, roughness: number): SceneDesc {
  return {
    ...room(maxDepth), name: 'reflected-card',
    camera: { type: 'perspective', position: [0, 1, 0], target: [0, 0, 0], up: [0, 0, -1], yFovDeg: 60 },
    materials: { floor: { type: 'conductor', eta: [0.2, 0.4, 1], k: [3, 2.5, 2], roughness }, card: { type: 'black' } },
    meshes: [
      { name: 'floor', geometry: { type: 'quad', size: [20, 20] }, material: 'floor' },
      { name: 'card', geometry: { type: 'quad', size: [1.4, 1.2] }, material: 'card', transform: translation(0, 2, 0), emission: [4, 1, 0.2], twoSided: true },
    ],
  };
}

async function checkSpread(cache: NrcCache, plain: PtMegakernel, scene: SceneDesc, mirror: boolean) {
  const metrics: Record<string, number> = {};
  const { device } = cache;
  const chain = scene.name === 'diffuse-mirror-diffuse';
  // A black cache prediction on the second diffuse hit must equal a two-bounce traced reference.
  if (chain) plain.setLimits({ maxDepth: 2 });
  for (const [frame, x] of (mirror ? [-0.35, 0, 0.35] : [0]).entries()) {
    if (mirror) {
      const camera = { ...scene.camera, position: [x, 1, 0] as [number, number, number], target: [x, 0, 0] as [number, number, number] };
      plain.setCamera(camera);
      cache.setCamera(camera);
    }
    plain.setSeed(1);
    const e = device.createCommandEncoder();
    plain.reset(e);
    plain.encode(e, 1);
    cache.reset(e);
    cache.encodeFrame(e, { train: false });
    device.queue.submit([e.finish()]);
    const [{ data }, { data: baseline }, stats] = await Promise.all([cache.render.readImage(), plain.readImage(), cache.readTerminationStats()]);
    const queries = stats.display_queries_rough! + stats.display_queries_glossy! + stats.display_queries_delta!;
    let worst = 0;
    let bright = 0;
    let depthSum = 0;
    let inferredQueries = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (mirror || chain) {
        for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(data[i + c]! - baseline[i + c]!));
        if (baseline[i]! > 0.1) bright++;
      } else {
        // A black prediction in this emissive room leaves E * (1 - rho^(depth+1)) / (1-rho).
        const depth = Math.round(Math.log(1 - data[i]! * (1 - RHO) / E) / Math.log(RHO) - 1);
        const expected = E * (1 - RHO ** (depth + 1)) / (1 - RHO);
        for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(data[i + c]! - expected));
        if (depth < 8) { depthSum += depth; inferredQueries++; }
      }
    }
    if (!(worst < 1e-5)) throw new Error(`spread frame ${frame} max error ${worst}`);
    if (mirror && (bright === 0 || bright === WIDTH * WIDTH || queries !== 0)) throw new Error(`mirror test: lit ${bright}, queries ${queries}`);
    if (chain && (stats.vertices_delta! < WIDTH * WIDTH / 2 || stats.display_queries_delta !== 0 ||
        stats.display_queries_rough! < WIDTH * WIDTH / 2 || stats.display_depth_sum !== 2 * queries)) {
      throw new Error(`diffuse-mirror-diffuse expected termination on the second diffuse hit: ${JSON.stringify(stats)}`);
    }
    if (!mirror && !chain && (depthSum !== stats.display_depth_sum || inferredQueries !== queries)) throw new Error(`image implies ${inferredQueries} queries with depth sum ${depthSum}: ${JSON.stringify(stats)}`);
    if (!mirror && !chain && cache.settings.spreadThreshold === 300 && !(depthSum > queries && queries > 0)) throw new Error('spread test expected deferred queries beyond the minimum depth');
    Object.assign(metrics, stats, { ['max_error_' + frame]: worst, ['lit_pixels_' + frame]: bright });
  }
  return metrics;
}

async function main() {
  const args = getArgs();
  const network = networkArg(args.network);
  if (args.check === 'materials' || args.check === 'directions' || args.check === 'checker' || args.check === 'factor-room' || args.check === 'factor-metal') {
    const precision = args.precision === 'f16' ? 'f16' : 'f32';
    const infer = args.infer === 'tiled' ? 'tiled' : 'perRecord';
    const factorization = factorizationArg(args.factorization);
    const directionInput = directionArg(args.directionInput);
    const diagnostics = (summary: AdapterSummary, errors: string[]) => {
      adapterSummary = summary;
      deviceErrors = errors;
    };
    const metrics = args.check === 'materials' || args.check === 'directions'
      ? await checkMaterialInputs(precision, infer, diagnostics, factorization, network, args.check === 'directions' || args.directionInput ? directionInput : undefined)
      : await checkFactorization(args.check === 'factor-metal' ? 'metal' : args.check === 'checker' ? 'checker' : 'room', room(8), factorization, precision, infer, diagnostics, args.surface, network, directionInput);
    reportResult({ scene: 'nrc-' + args.check, renderer: 'nrc-' + args.check, valid: true, config: { precision, infer, factorization, network, directionInput, ...(args.check === 'factor-metal' ? { surface: args.surface ?? 'rough' } : {}) }, metrics }, { adapter: adapterSummary, deviceErrors });
    return;
  }
  const depths = args.check === 'depths';
  const fullTail = args.check === 'full-tail';
  const mirror = args.check === 'mirror';
  const chain = args.check === 'delta-chain';
  const spread = args.check === 'spread' || mirror || chain;
  const hashMiss = args.check === 'hash-miss';
  const frames = numberArg(args, 'frames', 300);
  const queryDepth = numberArg(args, 'queryDepth', 1);
  const trainDepth = numberArg(args, 'trainDepth', 4);
  const spp = numberArg(args, 'spp', 16);
  const tol = numberArg(args, 'tol', 0.03);
  const tolMax = numberArg(args, 'tolMax', 0.1);
  let precision = (args.precision ?? 'f32') as Precision;
  const maxDepth = 8;

  const adapter = new PtMegakernelAdapter({ nee: false, rrStartDepth: 0, ...(fullTail ? { frame: true, scenePacking: 'packed' as const } : {}) });
  const scene = mirror ? reflectionScene(maxDepth, numberArg(args, 'roughness', 0)) : room(maxDepth);
  if (chain) {
    scene.name = 'diffuse-mirror-diffuse';
    scene.camera = reflectionScene(maxDepth, 0).camera;
    scene.materials = { diffuse: { type: 'diffuse', reflectance: [RHO, RHO, RHO] }, mirror: { type: 'conductor', eta: [0.2, 0.4, 1], k: [3, 2.5, 2], roughness: 0 } };
    scene.meshes = [
      { name: 'receiver', geometry: { type: 'quad', size: [2000, 2000] }, material: 'diffuse', emission: [1, 1, 1], twoSided: true },
      { name: 'mirror', geometry: { type: 'quad', size: [2000, 2000] }, material: 'mirror', transform: translation(0, 2, 0) },
    ];
  }
  await adapter.load(scene);
  const gpu = adapter.gpuContext!;
  adapterSummary = gpu.summary;
  deviceErrors = watchDevice(gpu.device);
  const device = gpu.device;
  const plain = adapter.renderer as PtMegakernel;
  const { caps, trace, sceneBuffers } = adapter.debug;
  const camera = scene.camera;

  const shape = MLP_SHAPES[network];
  const arm = inferArmOptions(args.infer ?? 'portable', gpu.adapter, device, shape, precision, { records: numberArg(args, 'sgmRecords', 64), subgroups: numberArg(args, 'subgroups', 4), ...(args.acc ? { accumulate: args.acc as SgmScalar } : {}), ...(args.layout ? { layout: args.layout as 'row' | 'blocked' } : {}), ...(args.pad ? { pad: numberArg(args, 'pad', 8) } : {}) });
  precision = arm.precision;
  const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], { width: WIDTH, height: WIDTH, camera, maxDepth }, { min: [-1, -1, -1], max: [1, 1, 1] }, {
    ...terminationArgs(args), ...(args.model === 'hash' ? { model: 'hash' as const, hash: hashGridArgs(args) } : {}),
    precision, network,
    ...(arm.sgm ? { sgm: arm.sgm } : {}),
    ...(arm.infer ? { infer: arm.infer } : {}),
    // forward=sgm (+ fwdLayout, fwdRecords, fwdSubgroups, acc): the subgroup-matrix forward-backward for training (N2r-d).
    ...(args.forward === 'sgm' ? { trainSgm: forwardArmOptions('sgm', gpu.adapter, device, shape, precision, forwardArgs(args, 1)).trainSgm } : {}),
    ...(args.gradient === 'sgm' ? { gradientSgm: gradientArmOptions('sgm', gpu.adapter, device, shape, precision).gradientSgm } : {}),
    training: args.training === 'replay' ? 'replay' : 'inline',
    ...materialArgs(args.factorization, args.materialInputs, args.directionInput),
    updateBlock: numberArg(args, 'updateBlock', 2),
    trainSelect: args.trainSelect === 'pixel' ? 'pixel' : 'tile',
    outputInit: args.outputInit === 'he' ? 'he' : 'zero',
    warmupFrames: numberArg(args, 'warmupFrames', 0),
    lrWarmupFrames: numberArg(args, 'lrWarmupFrames', 0),
    ema: args.ema === 'plain' || args.ema === 'off' ? args.ema : 'warmup',
    queryDepth,
    trainDepth,
    trainWidth: WIDTH,
    trainHeight: WIDTH,
    stepsPerFrame: numberArg(args, 'steps', 4),
    batch: numberArg(args, 'batch', 4096),
    learningRate: numberArg(args, 'lr', 0.01),
    ...(hashMiss ? { model: 'hash' as const, training: 'inline' as const, trainDepth: 4, warmupFrames: 0, hash: { continueOnMiss: true, log2Entries: 10, normalBins: false, minCell: 2, cellPixels: 1, minSamples: 1.5, fallbackLevels: 0 } } : {}),
    ...(spread ? { model: 'mlp' as const, termination: 'spread' as const, terminationStats: true, spreadThreshold: numberArg(args, 'spreadThreshold', chain ? 0 : mirror ? 0.01 : 300), queryDepth: mirror ? 0 : 1, trainDepth: 4, outputInit: 'zero' as const, warmupFrames: 0 } : {}),
    ...(fullTail ? { fullSuffixEvery: numberArg(args, 'fullSuffixEvery', 1), trainDepth: 1, learningRate: 0, outputInit: 'zero' as const, stepsPerFrame: 1 } : {}),
    ...(depths ? { model: 'mlp' as const, training: 'inline' as const, trainSelect: 'tile' as const, updateBlock: 2, trainDepth: 4, queryDepth: 1, warmupFrames: 1, warmupStepsPerFrame: 1, stepsPerFrame: 1, learningRate: 0 } : {}),
  });

  if (fullTail || depths || hashMiss || spread) {
    const metrics = fullTail ? await checkFullTail(cache) : spread ? await checkSpread(cache, plain, scene, mirror) : hashMiss ? await checkHashMiss(cache, plain) : await checkDepths(cache, plain);
    await drainDevice(device);
    if (deviceErrors.length) throw new Error(deviceErrors.join('; '));
    reportResult({ scene: scene.name, renderer: spread || fullTail ? 'nrc-' + args.check : hashMiss ? 'nrc-hash-miss' : 'nrc-depths', valid: true, config: { precision, ...(cache.mlp ? { network, shape } : {}) }, metrics }, { adapter: adapterSummary, deviceErrors });
    cache.destroy();
    adapter.dispose();
    return;
  }

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
  const reload = args.network && cache.mlp ? await checkWeightReload(cache) : {};
  const ok = meanRel <= tol && worst <= tolMax && deviceErrors.length === 0;
  await drainDevice(device);
  // A miss must fail the run: the first report wins, so throw before reporting a pass.
  if (!ok) throw new Error(`nrc-check: mean ${mean.toFixed(4)} vs ${expected} (rel ${meanRel.toExponential(2)}), worst pixel rel ${worst.toExponential(2)}${deviceErrors.length ? `; device errors: ${deviceErrors.join('; ')}` : ''}`);
  reportResult(
    {
      scene: 'nrc-check',
      renderer: 'ours',
      config: { frames, queryDepth, trainDepth, precision, spp, infer: arm.description, training: cache.settings.training, updateBlock: cache.settings.updateBlock, trainSelect: cache.settings.trainSelect, network, shape },
      valid: ok,
      metrics: { train_paths: cache.trainPaths, mean, expected, mean_rel_error: meanRel, max_rel_error: worst, plain_depth_limited: plainDepthLimited, train_records: trainCounts.records, train_batch: trainCounts.batch, queries_last_pass: queries, ...reload },
      notes: { infer: cache.description },
    },
    { adapter: adapterSummary, deviceErrors },
  );
}

main().catch((error) => reportError(error, { adapter: adapterSummary, deviceErrors }));
