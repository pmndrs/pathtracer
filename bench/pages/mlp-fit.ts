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
 * Inference arm for the snapshots: infer=portable|sgm|sgm-emu (training stays portable; see mlp-check).
 *
 * Args: precision=f32|f16  size=256  batch=16384  snapshots=16,64,256,1024  seed=1  ema=0|1  lr=0.01  oracleSteps=0
 *       octaves=8  phases=2 (image encoding layout, N1.3; octaves=nyquist means log2(size); 12 and 1 = the original)
 *       gradient=tiled|perWeight (weight-gradient arm, N1.2)
 * A/B: --ab precision=f32,f16   --ab octaves=12,8   --ab phases=1,2   --ab gradient=perWeight,tiled
 */

import { AdamState, DEFAULT_SHAPE, DEFAULT_TRAIN, MlpEngine, batchGradient, encodeImage, initWeights, type ImageEncoding, type Precision } from '@pathtracer/mlp';
import { compareImages } from '../metrics/metrics.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { inferArmOptions, readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';
import { IMAGE_TASK_ENCODING, IMAGE_TASK_LEARNING_RATE, imageHooksInfer, imageHooksTrain, targetImage, trainSample } from './mlp-image.ts';


let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  return s.length ? (s[Math.floor(s.length / 2)] as number) : NaN;
}

try {
  const args = getArgs();
  let precision = (args.precision ?? 'f32') as Precision;
  const size = numberArg(args, 'size', 256);
  const batch = numberArg(args, 'batch', 16384);
  const seed = numberArg(args, 'seed', 1);
  const useEma = numberArg(args, 'ema', 0) === 1;
  const learningRate = numberArg(args, 'lr', IMAGE_TASK_LEARNING_RATE);
  const snapshots = (args.snapshots ?? '16,64,256,1024').split(',').map(Number).sort((a, b) => a - b);
  if (snapshots.some((s) => !Number.isInteger(s) || s <= 0)) throw new Error(`bad snapshots: ${args.snapshots}`);
  const steps = snapshots[snapshots.length - 1] as number;
  const pixels = size * size;
  const oracleSteps = numberArg(args, 'oracleSteps', 0);
  const encoding: ImageEncoding = {
    octaves: args.octaves === 'nyquist' ? Math.round(Math.log2(size)) : numberArg(args, 'octaves', IMAGE_TASK_ENCODING.octaves),
    phases: numberArg(args, 'phases', IMAGE_TASK_ENCODING.phases) as 1 | 2,
  };

  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);
  const timed = device.features.has('timestamp-query');

  // Target image (CPU), uploaded once; also the reference PFM.
  const target = targetImage(size);

  const arm = inferArmOptions(args.infer ?? 'portable', adapter, device, DEFAULT_SHAPE, precision, { records: numberArg(args, 'sgmRecords', 64), subgroups: numberArg(args, 'subgroups', 1), tile: numberArg(args, 'tile', 8) });
  precision = arm.precision;
  const mlp = await MlpEngine.create(device, {
    ...(arm.sgm ? { sgm: arm.sgm } : {}),
    precision,
    capacity: batch,
    inferCapacity: pixels,
    useEma,
    gradient: args.gradient === 'perWeight' ? 'perWeight' : 'tiled',
    train: { ...DEFAULT_TRAIN, learningRate },
    trainHooks: imageHooksTrain(encoding),
    inferHooks: imageHooksInfer(encoding),
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
        inputs.set(encodeImage(Math.fround(s.u), Math.fround(s.v), DEFAULT_SHAPE.inputs, undefined, encoding), r * DEFAULT_SHAPE.inputs);
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
      config: { arm: arm.description, precision, size, batch, seed, useEma, learningRate, snapshots, shape: DEFAULT_SHAPE },
      metrics,
      valid: timed && !software,
      notes: { timed, software, encoding: `${encoding.octaves}x${encoding.phases}`, trainMsFirst: trainMs.slice(0, 4) },
    },
    { adapter: adapterSummary, deviceErrors },
  );
  }
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
