/**
 * Research track N, N2 microbench: GPU time of inference and of one training step for one arm, on the image task
 * (encoding computed in-kernel, so no input buffers). Use the runner's A/B for arm comparisons, e.g.
 *   node bench/runner/run.ts --page mlp-timing --ab infer=portable,sgm --args precision=f16 --repeat 3
 *   node bench/runner/run.ts --page mlp-timing --ab precision=f32,f16
 * Timings come from per-pass timestamps; the only readback is after all timed passes. Inference FLOP counts
 * 2 × weights per record (multiply-adds of the unpadded network), so the sgm arm's padding is not credited.
 *
 * Args: precision=f32|f16  infer=portable|sgm|sgm-emu  size=1024 (inference records = size²)  repeats=20  warmup=3
 *       batch=16384  trainSteps=20  sgmRecords=64  subgroups=1
 */

import { DEFAULT_SHAPE, DEFAULT_TRAIN, MlpEngine, initWeights, weightCount, type Precision } from '@pathtracer/mlp';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { inferArmOptions, readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';
import { IMAGE_HOOKS_INFER, IMAGE_HOOKS_TRAIN, imageBindGroups, targetImage } from './mlp-image.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const median = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] ?? NaN;

try {
  const args = getArgs();
  const size = numberArg(args, 'size', 1024);
  const repeats = numberArg(args, 'repeats', 20);
  const warmup = numberArg(args, 'warmup', 3);
  const batch = numberArg(args, 'batch', 16384);
  const trainSteps = numberArg(args, 'trainSteps', 20);
  const records = size * size;

  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);
  if (!device.features.has('timestamp-query')) throw new Error('mlp-timing needs timestamp-query (no wall-clock fallback).');

  const arm = inferArmOptions(args.infer ?? 'portable', adapter, device, DEFAULT_SHAPE, (args.precision ?? 'f32') as Precision, {
    records: numberArg(args, 'sgmRecords', 64),
    subgroups: numberArg(args, 'subgroups', 1),
    tile: numberArg(args, 'tile', 8),
  });
  const mlp = await MlpEngine.create(device, {
    precision: arm.precision,
    capacity: batch,
    inferCapacity: records,
    train: DEFAULT_TRAIN,
    trainHooks: IMAGE_HOOKS_TRAIN,
    inferHooks: IMAGE_HOOKS_INFER,
    ...(arm.sgm ? { sgm: arm.sgm } : {}),
  });
  mlp.uploadWeights(initWeights(DEFAULT_SHAPE, 1));

  const target = storageBuffer(device, targetImage(size), 'target');
  const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, new Uint32Array([size, size, 1, 0]));
  const out = storageBuffer(device, records * 12, 'out');
  const counts = storageBuffer(device, new Uint32Array([batch, records, 0, 0]), 'counts');
  const groups = imageBindGroups(device, mlp.trainLayout, mlp.inferLayout, target, dims, out);

  const slots = (warmup + repeats) + (warmup + trainSteps);
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2 * slots });
  const ts = (slot: number): GPUComputePassTimestampWrites => ({ querySet, beginningOfPassWriteIndex: 2 * slot, endOfPassWriteIndex: 2 * slot + 1 });

  // Inference passes, then training steps; one submit per pass group, a wait between groups only.
  let slot = 0;
  for (let i = 0; i < warmup + repeats; i++) {
    const encoder = device.createCommandEncoder();
    mlp.encodeInfer(encoder, counts, 1, groups.infer, ts(slot++));
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  for (let i = 0; i < warmup + trainSteps; i++) {
    const encoder = device.createCommandEncoder();
    mlp.encodeTrainStep(encoder, counts, 0, groups.train, ts(slot++));
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();

  const resolve = device.createBuffer({ size: querySet.count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(querySet, 0, querySet.count, resolve, 0);
  device.queue.submit([encoder.finish()]);
  const t = new BigUint64Array(await readBuffer(device, resolve));
  const span = (s: number) => Number((t[2 * s + 1] as bigint) - (t[2 * s] as bigint)) / 1e6;
  const inferMs = Array.from({ length: repeats }, (_, i) => span(warmup + i));
  const trainMs = Array.from({ length: trainSteps }, (_, i) => span(warmup + repeats + warmup + i));
  // Sanity: outputs must be finite (one readback, after timing).
  const sample = new Float32Array(await readBuffer(device, out, Math.min(records, 4096) * 12));
  const nonFinite = sample.filter((v) => !Number.isFinite(v)).length;
  await drainDevice(device);
  if (nonFinite) throw new Error(`${nonFinite} non-finite outputs in the first 4096 records`);

  const inferMedian = median(inferMs);
  const flopPerRecord = 2 * weightCount(DEFAULT_SHAPE);
  const software = adapterSummary.isFallbackAdapter || /swiftshader/i.test(`${adapterSummary.architecture} ${adapterSummary.description}`);
  reportResult(
    {
      scene: 'mlp-timing-image',
      renderer: arm.sgm ? (arm.sgm.emulate ? 'wgsl-sgm-emu' : 'wgsl-sgm') : `wgsl-${arm.precision}`,
      config: { arm: arm.description, precision: arm.precision, size, records, batch, repeats, trainSteps, sgm: arm.sgm ?? null },
      metrics: {
        gpu_ms_infer_median: inferMedian,
        gpu_ms_infer_min: Math.min(...inferMs),
        gpu_ms_per_mquery: (inferMedian * 1e6) / records,
        infer_tflops: (flopPerRecord * records) / (inferMedian * 1e-3) / 1e12,
        gpu_ms_train_step_median: median(trainMs),
        gpu_ms_train_step_min: Math.min(...trainMs),
      },
      valid: !software,
      notes: { software, inferMs, trainMs },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
