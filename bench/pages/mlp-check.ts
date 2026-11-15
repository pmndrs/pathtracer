/**
 * Research track N (docs/design/nrc.md §4): the WGSL MLP against the f64 CPU oracle.
 *  1. inference outputs for a random batch;
 *  2. the batch gradient of one training step;
 *  3. master weights after `steps` Adam steps (the oracle re-derives its gradient from its own f32 weights each step).
 * Errors are relative to the largest magnitude of each quantity. Fails when a check exceeds its tolerance.
 *
 * Inference arm: infer=portable (default) | sgm (subgroup matrices; config from the adapter, sets precision) |
 * sgm-emu (the same kernel with plain-WGSL stand-ins, f32: runs anywhere). sgm options: tile=8|16 (sgm-emu only),
 * subgroups=1|4, sgmRecords=64.
 *
 * Args: precision=f32|f16 (default f32) records=256 steps=5 seed=1 gradient=tiled|perWeight (default perWeight)
 *       gradSplits=64 gradChunk=32 (tiled arm) forward=tiled|perRecord (default tiled) fwdRecords=64 f16 / 48 f32 (tiled forward)
 */

import { AdamState, type ForwardKind, type GradientKind, BUFFER_HOOKS_INFER, BUFFER_HOOKS_TRAIN, DEFAULT_SHAPE, MlpEngine, Rng, batchGradient, encodeImage, forward, initWeights, type Precision } from '@pathtracer/mlp';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { inferArmOptions, readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';

// Tolerances relative to max |value|. f16 activations/weights carry ~1e-3 relative rounding.
const TOLERANCE: Record<Precision, { infer: number; gradient: number; weights: number }> = {
  f32: { infer: 1e-5, gradient: 1e-4, weights: 1e-5 },
  f16: { infer: 2e-2, gradient: 5e-2, weights: 5e-3 },
};

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

function maxRelError(gpu: ArrayLike<number>, cpu: ArrayLike<number>, n = cpu.length): { rel: number; scale: number; worstIndex: number } {
  let scale = 0;
  for (let i = 0; i < n; i++) scale = Math.max(scale, Math.abs(cpu[i] as number));
  let worst = 0;
  let worstIndex = -1;
  for (let i = 0; i < n; i++) {
    const e = Math.abs((gpu[i] as number) - (cpu[i] as number));
    if (!(e <= worst)) {
      worst = Number.isFinite(e) ? e : Infinity;
      worstIndex = i;
    }
  }
  return { rel: worst / Math.max(scale, 1e-30), scale, worstIndex };
}

try {
  const args = getArgs();
  let precision = (args.precision ?? 'f32') as Precision;
  if (precision !== 'f32' && precision !== 'f16') throw new Error(`precision must be f32 or f16, got ${precision}`);
  const inferArm = args.infer ?? 'portable';
  const subgroups = numberArg(args, 'subgroups', 1);
  const sgmRecords = numberArg(args, 'sgmRecords', 64);
  const n = numberArg(args, 'records', 256);
  const steps = numberArg(args, 'steps', 5);
  const seed = numberArg(args, 'seed', 1);
  const shape = DEFAULT_SHAPE;
  const gradientKind = (args.gradient ?? 'tiled') as GradientKind;
  if (gradientKind !== 'perWeight' && gradientKind !== 'tiled') throw new Error(`gradient must be perWeight or tiled, got ${gradientKind}`);
  const forwardKind = (args.forward ?? 'tiled') as ForwardKind;
  if (forwardKind !== 'perRecord' && forwardKind !== 'tiled') throw new Error(`forward must be perRecord or tiled, got ${forwardKind}`);
  const forwardOptions = args.fwdRecords ? { records: numberArg(args, 'fwdRecords', 64) } : {};
  const gradientOptions = { splits: numberArg(args, 'gradSplits', 64), chunk: numberArg(args, 'gradChunk', 32) };

  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);

  // Inputs: the image encoding of random (u, v); targets: a smooth HDR-ish function of (u, v).
  const rng = new Rng(seed);
  const inputs = new Float32Array(n * shape.inputs);
  const targets = new Float32Array(n * 3);
  for (let r = 0; r < n; r++) {
    const u = rng.next();
    const v = rng.next();
    inputs.set(encodeImage(u, v, shape.inputs), r * shape.inputs);
    targets.set([0.2 + 3 * u * u, 0.5 + Math.sin(6 * v), 1 + u * v], r * 3);
  }
  const weights = initWeights(shape, seed);

  const arm = inferArmOptions(inferArm, adapter, device, shape, precision, { records: sgmRecords, subgroups, tile: numberArg(args, 'tile', 8) });
  precision = arm.precision;
  const sgm = arm.sgm;
  const mlp = await MlpEngine.create(device, { precision, capacity: n, inferCapacity: n, trainHooks: BUFFER_HOOKS_TRAIN, inferHooks: BUFFER_HOOKS_INFER, gradient: gradientKind, gradientOptions, forward: forwardKind, forwardOptions, ...(sgm ? { sgm } : {}) });
  const inputBuffer = storageBuffer(device, inputs, 'inputs');
  const targetBuffer = storageBuffer(device, targets, 'targets');
  const outputBuffer = storageBuffer(device, n * 3 * 4, 'outputs');
  const counts = storageBuffer(device, new Uint32Array([n, n, 0, 0]), 'counts');
  const trainHooks = device.createBindGroup({ layout: mlp.trainLayout, entries: [{ binding: 0, resource: { buffer: inputBuffer } }, { binding: 1, resource: { buffer: targetBuffer } }] });
  const inferHooks = device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 0, resource: { buffer: inputBuffer } }, { binding: 2, resource: { buffer: outputBuffer } }] });

  mlp.uploadWeights(weights);

  // 1. Inference with the initial weights.
  let encoder = device.createCommandEncoder();
  mlp.encodeInfer(encoder, counts, 1, inferHooks);
  device.queue.submit([encoder.finish()]);
  const gpuOut = new Float32Array(await readBuffer(device, outputBuffer, n * 12));
  const cpuOut = new Float64Array(n * 3);
  for (let r = 0; r < n; r++) cpuOut.set(forward(shape, weights, inputs.subarray(r * shape.inputs, (r + 1) * shape.inputs)).output, r * 3);
  const infer = maxRelError(gpuOut, cpuOut);

  // 2. One training step: compare the gradient.
  encoder = device.createCommandEncoder();
  mlp.encodeTrainStep(encoder, counts, 0, trainHooks);
  device.queue.submit([encoder.finish()]);
  const gpuGrad = new Float32Array(await readBuffer(device, mlp.gradient, mlp.weightCount * 4));
  const oracle = new AdamState(weights);
  const first = batchGradient(shape, oracle.weights, inputs, targets, n);
  const gradient = maxRelError(gpuGrad, first.gradient);
  oracle.apply(first.gradient);

  // 3. The remaining steps; compare master weights.
  encoder = device.createCommandEncoder();
  for (let s = 1; s < steps; s++) mlp.encodeTrainStep(encoder, counts, 0, trainHooks);
  device.queue.submit([encoder.finish()]);
  for (let s = 1; s < steps; s++) oracle.apply(batchGradient(shape, oracle.weights, inputs, targets, n).gradient);
  const gpuWeights = new Float32Array(await readBuffer(device, mlp.master, mlp.weightCount * 4));
  // Compare the change from the initial weights: absolute weights hide small update errors.
  const gpuDelta = gpuWeights.map((w, i) => w - (weights[i] as number));
  const cpuDelta = oracle.weights.map((w, i) => w - (weights[i] as number));
  const update = maxRelError(gpuDelta, cpuDelta);
  // Adam divides by sqrt(v): where the gradient is ~0 its sign decides a full lr step, so f32 rounding there moves a
  // weight by up to lr per step. Report the distribution and the gradient size at the worst weight.
  const updateErrors = Array.from(gpuDelta, (g, i) => Math.abs(g - (cpuDelta[i] as number)) / Math.max(update.scale, 1e-30)).sort((a, b) => a - b);
  const updateP99 = updateErrors[Math.floor(updateErrors.length * 0.99)] ?? 0;
  const worstGrad = Math.abs(first.gradient[update.worstIndex] ?? 0) / Math.max(gradient.scale, 1e-30);

  await drainDevice(device);
  const tol = TOLERANCE[precision];
  const failures: string[] = [];
  if (!(infer.rel <= tol.infer)) failures.push(`inference rel error ${infer.rel.toExponential(2)} > ${tol.infer}`);
  if (!(gradient.rel <= tol.gradient)) failures.push(`gradient rel error ${gradient.rel.toExponential(2)} > ${tol.gradient} (worst weight ${gradient.worstIndex})`);
  if (!(updateP99 <= tol.weights * 100)) failures.push(`weight update p99 rel error ${updateP99.toExponential(2)} > ${tol.weights * 100} (max ${update.rel.toExponential(2)} at weight ${update.worstIndex}, |g|/max ${worstGrad.toExponential(2)})`);
  if (failures.length) throw new Error(failures.join('; '));

  reportResult(
    {
      scene: 'mlp-random',
      renderer: inferArm === 'portable' ? `wgsl-${precision}` : `wgsl-${inferArm}`,
      config: { precision, forward: forwardKind, forwardOptions, gradient: gradientKind, gradientOptions, records: n, steps, seed, shape, infer: inferArm, sgm: sgm ?? null },
      metrics: {
        infer_rel_error: infer.rel,
        gradient_rel_error: gradient.rel,
        update_rel_error: update.rel,
        loss_first: first.loss,
      },
      notes: { arm: arm.description, inferScale: infer.scale, gradientScale: gradient.scale, updateScale: update.scale, updateP99, worstUpdateIndex: update.worstIndex, worstUpdateRelGradient: worstGrad },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
