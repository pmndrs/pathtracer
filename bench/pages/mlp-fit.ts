/**
 * Research track N, N1 "learn an image" (docs/design/nrc.md §4): the MLP fits a procedural HDR image from (u, v).
 * Every arm trains from the same seed on the same random samples; the page uploads the target and the fitted image
 * at each snapshot as PFM (viewer A/B), and reports relMSE / PSNR per snapshot plus GPU ms per training step and per
 * million inferences.
 *
 * Training samples are random sub-pixel (u, v) drawn on the GPU (hash of record and Adam step), bilinear from the
 * target, so the CPU only encodes commands. Steps are submitted in chunks with a wait between them so the page stays
 * responsive to the runner (CLAUDE.md trap); timings come from per-pass GPU timestamps, never wall clock.
 *
 * oracleSteps=N (N > 0) instead replays the first N steps on the CPU oracle with the same samples (the JS twin of the
 * WGSL sampler below) and fails if the weights drift: the end-to-end check of hooks, step seeding and the Adam schedule.
 *
 * Args: precision=f32|f16  size=256  batch=16384  snapshots=16,64,256,1024  seed=1  ema=0|1  lr=0.002  oracleSteps=0
 * A/B: --ab precision=f32,f16
 */

import { AdamState, DEFAULT_SHAPE, DEFAULT_TRAIN, MlpEngine, batchGradient, encodeImage, hashU32, initWeights, type Precision } from '@pathtracer/mlp';
import { compareImages } from '../metrics/metrics.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';


/** Procedural HDR target, linear RGB, row 0 = top. Features at several scales so encodings and widths separate. */
export function targetPixel(u: number, v: number): [number, number, number] {
  // Sky gradient.
  let r = 0.15 + 0.85 * v;
  let g = 0.3 + 0.6 * v;
  let b = 0.8 - 0.2 * v;
  // Sun: a disc at 30 with a soft rim and a glow.
  const d = Math.hypot(u - 0.72, v - 0.22);
  const disc = Math.min(Math.max((0.06 - d) / 0.004, 0), 1);
  const glow = 2 * Math.exp(-((d / 0.12) ** 2));
  r += 30 * disc + glow;
  g += 27 * disc + 0.8 * glow;
  b += 20 * disc + 0.4 * glow;
  // Chirp band: stripes from 4 to 96 cycles across, tests high frequencies.
  if (v > 0.42 && v < 0.54) {
    const cycles = 4 + 92 * u;
    const s = 0.5 + 0.5 * Math.sin(2 * Math.PI * cycles * u);
    r = 0.05 + 0.9 * s;
    g = 0.05 + 0.9 * s;
    b = 0.05 + 0.9 * s;
  }
  // Checkerboard floor with a soft shadow.
  if (v > 0.62) {
    const check = (Math.floor(u * 12) + Math.floor((v - 0.62) * 12 / 0.38 * 2)) % 2 === 0 ? 0.7 : 0.08;
    const shadow = 1 - 0.7 * Math.exp(-(((u - 0.35) / 0.15) ** 2) - (((v - 0.8) / 0.08) ** 2));
    r = check * shadow * 1.2;
    g = check * shadow * 1.0;
    b = check * shadow * 0.9;
  }
  return [r, g, b];
}

/** JS twin of the WGSL training sampler: same hash, same (u, v), bilinear target. */
function trainSample(r: number, step: number, seed: number, size: number, target: Float32Array): { u: number; v: number; t: [number, number, number] } {
  const key = hashU32((r ^ hashU32((Math.imul(step, 0x9e3779b9) + seed) >>> 0)) >>> 0);
  const u = (hashU32(key) >>> 8) / 16777216;
  const v = (hashU32((key + 1) >>> 0) >>> 8) / 16777216;
  const fetch = (x: number, y: number, c: number) => target[((Math.min(Math.max(y, 0), size - 1) * size) + Math.min(Math.max(x, 0), size - 1)) * 3 + c] as number;
  const px = u * size - 0.5;
  const py = v * size - 0.5;
  const ix = Math.floor(px);
  const iy = Math.floor(py);
  const fx = px - ix;
  const fy = py - iy;
  const t = [0, 1, 2].map((c) => {
    const a = fetch(ix, iy, c) * (1 - fx) + fetch(ix + 1, iy, c) * fx;
    const b = fetch(ix, iy + 1, c) * (1 - fx) + fetch(ix + 1, iy + 1, c) * fx;
    return a * (1 - fy) + b * fy;
  }) as [number, number, number];
  return { u, v, t };
}

const IMAGE_HOOKS_COMMON = /* wgsl */ `
struct ImageDims { width: u32, height: u32, seed: u32, _pad: u32 }
@group(1) @binding(0) var<storage, read> imgTarget: array<f32>;
@group(1) @binding(1) var<uniform> imgDims: ImageDims;

fn imgHash(x: u32) -> u32 {
  let state = x * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn imgRand(x: u32) -> f32 { return f32(imgHash(x) >> 8u) / 16777216.0; }

fn imgFetch(x: i32, y: i32) -> vec3f {
  let cx = u32(clamp(x, 0, i32(imgDims.width) - 1));
  let cy = u32(clamp(y, 0, i32(imgDims.height) - 1));
  let i = (cy * imgDims.width + cx) * 3u;
  return vec3f(imgTarget[i], imgTarget[i + 1u], imgTarget[i + 2u]);
}
// Bilinear at uv (pixel centres at (i + 0.5) / size).
fn imgSample(uv: vec2f) -> vec3f {
  let p = uv * vec2f(f32(imgDims.width), f32(imgDims.height)) - 0.5;
  let f = fract(p);
  let i = vec2i(floor(p));
  let a = mix(imgFetch(i.x, i.y), imgFetch(i.x + 1, i.y), f.x);
  let b = mix(imgFetch(i.x, i.y + 1), imgFetch(i.x + 1, i.y + 1), f.x);
  return mix(a, b, f.y);
}
`;

const IMAGE_HOOKS_TRAIN = `${IMAGE_HOOKS_COMMON}
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
  let key = imgHash(r ^ imgHash(nnState.step * 0x9e3779b9u + imgDims.seed));
  let uv = vec2f(imgRand(key), imgRand(key + 1u));
  *x = nnEncodeImage(uv);
  return imgSample(uv);
}
`;

const IMAGE_HOOKS_INFER = /* wgsl */ `
struct ImageDims { width: u32, height: u32, seed: u32, _pad: u32 }
@group(1) @binding(1) var<uniform> imgDims: ImageDims;
@group(1) @binding(2) var<storage, read_write> imgOut: array<f32>;
fn nnInferLoad(r: u32) -> array<f32, NN_IN> {
  let uv = (vec2f(f32(r % imgDims.width), f32(r / imgDims.width)) + 0.5) / vec2f(f32(imgDims.width), f32(imgDims.height));
  return nnEncodeImage(uv);
}
fn nnInferStore(r: u32, y: vec3f) { imgOut[r * 3u] = y.x; imgOut[r * 3u + 1u] = y.y; imgOut[r * 3u + 2u] = y.z; }
`;

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  return s.length ? (s[Math.floor(s.length / 2)] as number) : NaN;
}

try {
  const args = getArgs();
  const precision = (args.precision ?? 'f32') as Precision;
  const size = numberArg(args, 'size', 256);
  const batch = numberArg(args, 'batch', 16384);
  const seed = numberArg(args, 'seed', 1);
  const useEma = numberArg(args, 'ema', 0) === 1;
  const learningRate = numberArg(args, 'lr', DEFAULT_TRAIN.learningRate);
  const snapshots = (args.snapshots ?? '16,64,256,1024').split(',').map(Number).sort((a, b) => a - b);
  if (snapshots.some((s) => !Number.isInteger(s) || s <= 0)) throw new Error(`bad snapshots: ${args.snapshots}`);
  const steps = snapshots[snapshots.length - 1] as number;
  const pixels = size * size;
  const oracleSteps = numberArg(args, 'oracleSteps', 0);

  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);
  const timed = device.features.has('timestamp-query');

  // Target image (CPU), uploaded once; also the reference PFM.
  const target = new Float32Array(pixels * 3);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) target.set(targetPixel((x + 0.5) / size, (y + 0.5) / size), (y * size + x) * 3);

  const mlp = await MlpEngine.create(device, {
    precision,
    capacity: batch,
    inferCapacity: pixels,
    useEma,
    train: { ...DEFAULT_TRAIN, learningRate },
    trainHooks: IMAGE_HOOKS_TRAIN,
    inferHooks: IMAGE_HOOKS_INFER,
  });
  const initial = initWeights(DEFAULT_SHAPE, seed);
  mlp.uploadWeights(initial);

  const targetBuffer = storageBuffer(device, target, 'target');
  const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([size, size, seed, 0]));
  const counts = storageBuffer(device, new Uint32Array([batch, pixels, 0, 0]), 'counts');
  const trainHooks = device.createBindGroup({ layout: mlp.trainLayout, entries: [{ binding: 0, resource: { buffer: targetBuffer } }, { binding: 1, resource: { buffer: dims } }] });

  if (oracleSteps > 0) {
    const encoder = device.createCommandEncoder();
    for (let k = 0; k < oracleSteps; k++) mlp.encodeTrainStep(encoder, counts, 0, trainHooks);
    device.queue.submit([encoder.finish()]);
    const gpu = new Float32Array(await readBuffer(device, mlp.master, mlp.weightCount * 4));
    const oracle = new AdamState(initial, { ...DEFAULT_TRAIN, learningRate });
    const inputs = new Float32Array(batch * DEFAULT_SHAPE.inputs);
    const targets = new Float32Array(batch * 3);
    const losses: number[] = [];
    for (let step = 1; step <= oracleSteps; step++) {
      for (let r = 0; r < batch; r++) {
        const s = trainSample(r, step, seed, size, target);
        inputs.set(encodeImage(Math.fround(s.u), Math.fround(s.v), DEFAULT_SHAPE.inputs), r * DEFAULT_SHAPE.inputs);
        targets.set(s.t, r * 3);
      }
      const g = batchGradient(DEFAULT_SHAPE, oracle.weights, inputs, targets, batch);
      losses.push(g.loss);
      oracle.apply(g.gradient);
    }
    let scale = 0;
    const errors = Array.from(gpu, (w, i) => {
      const cpu = (oracle.weights[i] as number) - (initial[i] as number);
      scale = Math.max(scale, Math.abs(cpu));
      return Math.abs(w - (initial[i] as number) - cpu);
    });
    errors.sort((a, b) => a - b);
    const p99 = (errors[Math.floor(errors.length * 0.99)] as number) / scale;
    const p50 = (errors[Math.floor(errors.length * 0.5)] as number) / scale;
    const tol = precision === 'f32' ? 1e-2 : 1e-1;
    await drainDevice(device);
    if (!(p99 <= tol)) throw new Error(`oracle replay: weight-update p99 rel error ${p99.toExponential(2)} > ${tol} after ${oracleSteps} steps`);
    reportResult(
      { scene: 'mlp-fit-procedural', renderer: `wgsl-${precision}`, config: { precision, size, batch, seed, oracleSteps }, metrics: { oracle_update_p50: p50, oracle_update_p99: p99, oracle_loss_first: losses[0] as number, oracle_loss_last: losses[losses.length - 1] as number } },
      { adapter: adapterSummary, deviceErrors },
    );
  } else {
  const querySet = timed ? device.createQuerySet({ type: 'timestamp', count: 2 * (steps + snapshots.length) }) : null;
  const ts = (slot: number): GPUComputePassTimestampWrites | undefined => (querySet ? { querySet, beginningOfPassWriteIndex: 2 * slot, endOfPassWriteIndex: 2 * slot + 1 } : undefined);

  const outputs: GPUBuffer[] = [];
  let done = 0;
  for (const [si, snap] of snapshots.entries()) {
    while (done < snap) {
      const chunk = Math.min(snap - done, 32);
      const encoder = device.createCommandEncoder();
      for (let k = 0; k < chunk; k++, done++) mlp.encodeTrainStep(encoder, counts, 0, trainHooks, ts(done));
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone(); // keeps the page answering the runner; not inside any timed pass
    }
    const out = storageBuffer(device, pixels * 12, `fit-${snap}`);
    const inferHooks = device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 1, resource: { buffer: dims } }, { binding: 2, resource: { buffer: out } }] });
    const encoder = device.createCommandEncoder();
    mlp.encodeInfer(encoder, counts, 1, inferHooks, ts(steps + si));
    device.queue.submit([encoder.finish()]);
    outputs.push(out);
  }

  // Everything below is after the work: readbacks, metrics, uploads.
  let trainMs: number[] = [];
  let inferMs: number[] = [];
  if (querySet) {
    const resolve = device.createBuffer({ size: querySet.count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, querySet.count, resolve, 0);
    device.queue.submit([encoder.finish()]);
    const t = new BigUint64Array(await readBuffer(device, resolve));
    const span = (slot: number) => Number((t[2 * slot + 1] as bigint) - (t[2 * slot] as bigint)) / 1e6;
    trainMs = Array.from({ length: steps }, (_, i) => span(i));
    inferMs = snapshots.map((_, i) => span(steps + i));
  }
  const loss = new Float32Array(await readBuffer(device, mlp.loss, batch * 4));
  const lastLoss = loss.reduce((a, b) => a + b, 0) / batch;

  await uploadImage('target', { width: size, height: size, channels: 3, data: target });
  const metrics: Record<string, number> = { steps, batch, last_loss: lastLoss };
  for (const [si, snap] of snapshots.entries()) {
    const data = new Float32Array(await readBuffer(device, outputs[si] as GPUBuffer, pixels * 12));
    const image = { width: size, height: size, channels: 3 as const, data };
    const m = compareImages(image, { width: size, height: size, channels: 3, data: target });
    metrics[`relmse_step${snap}`] = m.relMse;
    metrics[`psnr_step${snap}`] = m.psnr;
    metrics[`nonfinite_step${snap}`] = m.nonFiniteTest;
    await uploadImage(`fit-${precision}-step${snap}`, image);
  }
  if (timed) {
    metrics.gpu_ms_train_step_median = median(trainMs.slice(Math.min(4, trainMs.length - 1)));
    metrics.gpu_ms_train_step_min = Math.min(...trainMs);
    metrics.gpu_ms_infer_per_mquery = (median(inferMs) * 1e6) / pixels;
  }
  await drainDevice(device);

  const software = adapterSummary.isFallbackAdapter || /swiftshader/i.test(`${adapterSummary.architecture} ${adapterSummary.description}`);
  reportResult(
    {
      scene: 'mlp-fit-procedural',
      renderer: `wgsl-${precision}`,
      config: { precision, size, batch, seed, useEma, learningRate, snapshots, shape: DEFAULT_SHAPE },
      metrics,
      valid: timed && !software,
      notes: { timed, software, trainMsFirst: trainMs.slice(0, 4) },
    },
    { adapter: adapterSummary, deviceErrors },
  );
  }
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
