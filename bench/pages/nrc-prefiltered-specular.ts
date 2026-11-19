/** Matched second-hit outgoing and prefiltered-incident MLPs, with traced and split-sum oracles. */
import { DEFAULT_TRAIN, MLP_SHAPES, MlpEngine, initWeights, layerOffset, type Precision } from '@pathtracer/mlp';
import {
  SPECULAR_BANDS, SPECULAR_LUT_SIZE, SPECULAR_REFERENCE_WGSL, SPECULAR_ROUGHNESS, SPECULAR_VIEWS,
  specularInferHooks, specularTrainHooks,
} from '../nrc/prefiltered-specular.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import { analyzeTimestamps } from '../runner/timestamps.ts';
import { readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';

const LUMA = [0.2126, 0.7152, 0.0722] as const;
const SHAPE = MLP_SHAPES.compact;

interface Evaluation {
  width: number;
  height: number;
  raw: Float32Array;
  split: Float32Array;
  supported: Uint8Array;
  mu: Float32Array;
}

interface ImageMetrics {
  rrmse: number;
  luminance_bias: number;
  blue_peak: number;
  blue_energy: number;
  blue_rms_width_px: number;
}

function evaluateImage(
  source: Float32Array, evaluation: Evaluation, pixelGroup: (pixel: number, band: number, view: number) => boolean,
): ImageMetrics {
  const { width, height, supported, raw } = evaluation;
  let squaredError = 0;
  let squaredRef = 0;
  let lumaError = 0;
  let lumaRef = 0;
  let energy = 0;
  let peak = 0;
  let radius2 = 0;
  let center = 0;
  let pixels = 0;
  for (const [i, active] of supported.entries()) {
    const view = Math.floor(i / (width * height));
    const band = Math.floor((Math.floor(i / width) % height + 0.5) * SPECULAR_BANDS / height);
    if (!active || !pixelGroup(i, band, view)) continue;
    pixels++;
    for (const [c, weight] of LUMA.entries()) {
      const ref = raw[i * 3 + c]!;
      const error = source[i * 3 + c]! - ref;
      squaredError += error * error;
      squaredRef += ref * ref;
      lumaError += error * weight;
      lumaRef += ref * weight;
    }
    const blue = Math.max(0, source[i * 3 + 2]! - 0.55 * (source[i * 3]! + source[i * 3 + 1]!));
    const column = i % width;
    peak = Math.max(peak, blue);
    energy += blue;
    center += blue * column;
    radius2 += blue * column * column;
  }
  return {
    rrmse: Math.sqrt(squaredError / squaredRef),
    luminance_bias: lumaError / lumaRef,
    blue_peak: peak,
    blue_energy: energy / pixels,
    blue_rms_width_px: energy > 0 ? Math.sqrt(Math.max(0, radius2 / energy - (center / energy) ** 2)) : 0,
  };
}

function score(
  metrics: Record<string, number>, name: string, image: Float32Array, evaluation: Evaluation,
): void {
  for (const [group, viewStart] of [['oblique', 0], ['grazing', 3]] as const) {
    const selectView = (_pixel: number, _band: number, view: number) => view >= viewStart && view < viewStart + 3;
    for (const [key, value] of Object.entries(evaluateImage(image, evaluation, selectView))) {
      metrics[`${name}_${group}_${key}`] = value;
    }
    for (const [band, roughness] of SPECULAR_ROUGHNESS.entries()) {
      const selectBand = (_pixel: number, row: number, view: number) => row === band && selectView(_pixel, row, view);
      for (const [key, value] of Object.entries(evaluateImage(image, evaluation, selectBand))) {
        metrics[`${name}_${group}_r${Math.round(roughness * 100)}_${key}`] = value;
      }
    }
  }
}

function extractRgb(buffer: Float32Array, pixels: number, stride: number, offset = 0): Float32Array {
  const rgb = new Float32Array(3 * pixels);
  for (let i = 0; i < pixels; i++) rgb.set(buffer.subarray(i * stride + offset, i * stride + offset + 3), 3 * i);
  if (!rgb.every((value) => Number.isFinite(value) && value >= 0)) throw new Error('Invalid radiance in specular readback');
  return rgb;
}

async function makeReferences(
  device: GPUDevice, params: GPUBuffer, width: number, height: number, lut: GPUBuffer, references: GPUBuffer,
): Promise<{ image: Evaluation; energyError: number; normalizationError: number }> {
  const pixels = width * height * SPECULAR_VIEWS.length;
  const furnace = storageBuffer(device, 16 * 16, 'conductor furnace checks');
  const module = device.createShaderModule({ label: 'split-sum reference and furnace', code: SPECULAR_REFERENCE_WGSL });
  const make = (entryPoint: string) => device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint } });
  const [lutPipeline, referencePipeline, furnacePipeline] = await Promise.all([make('buildLut'), make('reference'), make('furnace')]);

  function dispatch(
    encoder: GPUCommandEncoder, pipeline: GPUComputePipeline, groups: number, resources: [number, GPUBuffer][],
  ) {
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: resources.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  try {
    const encoder = device.createCommandEncoder({ label: 'reference, split-sum and furnace' });
    dispatch(encoder, lutPipeline, Math.ceil(SPECULAR_LUT_SIZE ** 2 / 64), [[0, params], [1, lut]]);
    dispatch(encoder, referencePipeline, Math.ceil(pixels / 64), [[0, params], [2, lut], [3, references]]);
    dispatch(encoder, furnacePipeline, 1, [[0, params], [2, lut], [4, furnace]]);
    device.queue.submit([encoder.finish()]);

    const white = new Float32Array(await readBuffer(device, furnace));
    const energyError = Math.max(...white.filter((_, i) => i % 4 !== 3));
    const normalizationError = Math.max(...white.filter((_, i) => i % 4 === 3).map((v) => Math.abs(v - 1)));
    if (!(energyError < 0.012 && normalizationError < 0.003)) {
      throw new Error(`Uniform-radiance furnace failed: energy ${energyError}, prefilter normalization ${normalizationError}`);
    }

    const values = new Float32Array(await readBuffer(device, references));
    const raw = extractRgb(values, pixels, 8);
    const split = extractRgb(values, pixels, 8, 4);
    const supported = Uint8Array.from(values.filter((_, i) => i % 8 === 3));
    const mu = values.filter((_, i) => i % 8 === 7);
    if (supported.reduce((sum, value) => sum + value, 0) !== width * height * SPECULAR_VIEWS.length * 2 / 3) {
      throw new Error('Unexpected number of eligible isotropic conductor hits');
    }
    return { image: { width, height, raw, split, supported, mu }, energyError, normalizationError };
  } finally {
    furnace.destroy();
  }
}

interface Training {
  batch: number;
  steps: number;
  seed: number;
  precision: Precision;
}

async function train(
  device: GPUDevice, arm: 'raw' | 'prefiltered', settings: Training, params: GPUBuffer,
  references: GPUBuffer, lut: GPUBuffer, evaluation: Evaluation, metrics: Record<string, number>,
): Promise<Float32Array> {
  const prefilter = arm === 'prefiltered';
  const pixels = evaluation.supported.length;
  const { batch, steps, seed, precision } = settings;
  const engine = await MlpEngine.create(device, {
    shape: SHAPE, precision, capacity: batch, inferCapacity: pixels, useEma: true,
    train: { ...DEFAULT_TRAIN, emaWarmup: true },
    trainHooks: specularTrainHooks(prefilter), inferHooks: specularInferHooks(prefilter),
  });
  const counts = storageBuffer(device, new Uint32Array([batch, pixels, 0, 0]), 'training and inference counts');
  const output = storageBuffer(device, pixels * 16, 'reconstructed outgoing radiance');
  const table = device.features.has('timestamp-query') ? device.createQuerySet({ type: 'timestamp', count: 2 * (steps + 1) }) : undefined;
  const stamps = (i: number): GPUComputePassTimestampWrites | undefined =>
    table ? { querySet: table, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } : undefined;

  try {
    const initial = initWeights(SHAPE, seed);
    initial.fill(0, layerOffset(SHAPE, SHAPE.hidden));
    engine.uploadWeights(initial);
    const trainGroup = device.createBindGroup({
      layout: engine.trainLayout,
      entries: [{ binding: 0, resource: { buffer: params } }],
    });
    const inferGroup = device.createBindGroup({
      layout: engine.inferLayout,
      entries: [
        { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: references } },
        { binding: 2, resource: { buffer: output } },
        ...(prefilter ? [{ binding: 3, resource: { buffer: lut } }] : []),
      ],
    });

    // Bounded submissions keep the GPU runner responsive. Image readback follows all timed work.
    await device.queue.onSubmittedWorkDone();
    for (let done = 0; done < steps;) {
      const encoder = device.createCommandEncoder({ label: `${arm} updates` });
      const end = Math.min(steps, done + 16);
      for (; done < end; done++) engine.encodeTrainStep(encoder, counts, 0, trainGroup, stamps(done));
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    const infer = device.createCommandEncoder({ label: `${arm} frozen inference` });
    engine.encodeInfer(infer, counts, 1, inferGroup, stamps(steps));
    device.queue.submit([infer.finish()]);

    if (table) {
      const resolved = device.createBuffer({ size: table.count * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      try {
        const resolve = device.createCommandEncoder();
        resolve.resolveQuerySet(table, 0, table.count, resolved, 0);
        device.queue.submit([resolve.finish()]);
        const { quantized, durationsNs, stepNs } = analyzeTimestamps(new BigUint64Array(await readBuffer(device, resolved)));
        if (!quantized && durationsNs.every((v) => v > 0)) {
          const time = durationsNs.slice(0, steps).sort((a, b) => a - b);
          metrics[`${arm}_train_gpu_ms_p50`] = time[Math.floor(steps / 2)]! / 1e6;
          metrics[`${arm}_infer_gpu_ms`] = durationsNs[steps]! / 1e6;
          metrics[`${arm}_timestamp_step_ns`] = Number(stepNs);
        }
      } finally {
        resolved.destroy();
      }
    }
    const values = new Float32Array(await readBuffer(device, output));
    const image = extractRgb(values, pixels, 4);
    const fallback = evaluation.supported.reduce((n, eligible, i) => {
      if (values[i * 4 + 3] !== eligible) throw new Error(`${arm} changed the cache eligibility mask at pixel ${i}`);
      if (eligible) return n;
      if (!image.subarray(i * 3, i * 3 + 3).every((v, c) => v === evaluation.raw[i * 3 + c])) {
        throw new Error(`${arm} changed a traced mirror/glass pixel at ${i}`);
      }
      return n + 1;
    }, 0);
    metrics[`${arm}_traced_fallback_pixels`] = fallback;
    return image;
  } finally {
    engine.destroy();
    counts.destroy();
    output.destroy();
    table?.destroy();
  }
}

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const width = numberArg(args, 'width', 72);
  const height = numberArg(args, 'height', 48);
  const seed = numberArg(args, 'seed', 1);
  const refSpp = numberArg(args, 'refSpp', 4096);
  const lutSpp = numberArg(args, 'lutSpp', 4096);
  const steps = numberArg(args, 'steps', 512);
  const batch = numberArg(args, 'batch', 4096);
  const trainSpp = numberArg(args, 'trainSpp', 32);
  const precision = args.precision ?? 'f16';
  const images = args.images !== '0';
  const arm = args.arm ?? 'both';
  if (precision !== 'f16' && precision !== 'f32') throw new Error('precision must be f16 or f32');
  if (arm !== 'both' && arm !== 'raw' && arm !== 'prefiltered') {
    throw new Error('arm must be both, raw, or prefiltered');
  }
  if (![width, height, seed, refSpp, lutSpp, steps, batch, trainSpp].every(Number.isSafeInteger) ||
      width < 8 || width > 256 || height < 12 || height > 192 || height % SPECULAR_BANDS !== 0 ||
      seed < 0 || seed > 0xffff_ffff || refSpp < 64 || refSpp > 16_384 || lutSpp < 256 || lutSpp > 16_384 ||
      steps < 1 || steps > 1024 || batch < 64 || batch > 16_384 || trainSpp < 1 || trainSpp > 256) {
    throw new Error('Invalid dimensions, seed, sample count, or training budget');
  }

  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);
  const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([width, height, seed, trainSpp, refSpp, lutSpp, 0, 0]));
  const lut = storageBuffer(device, SPECULAR_LUT_SIZE ** 2 * 16, 'conductor integration LUT');
  const references = storageBuffer(device, width * height * SPECULAR_VIEWS.length * 32, 'traced and split-sum references');
  const { image: evaluation, energyError, normalizationError } = await makeReferences(device, params, width, height, lut, references);
  const metrics: Record<string, number> = {
    furnace_energy_max_abs: energyError, furnace_prefilter_normalization_max_abs: normalizationError, lut_bytes: lut.size,
    view_oblique_min_n_dot_v: evaluation.mu.subarray(0, width * height * 3).reduce((a, b) => Math.min(a, b), 1),
    view_grazing_max_n_dot_v: evaluation.mu.subarray(width * height * 3).reduce((a, b) => Math.max(a, b), 0),
  };

  score(metrics, 'traced', evaluation.raw, evaluation);
  score(metrics, 'split_oracle', evaluation.split, evaluation);
  if (images) {
    const upload = (name: string, data: Float32Array) =>
      uploadImage(name, { width, height: height * SPECULAR_VIEWS.length, channels: 3, data });
    await upload('traced', evaluation.raw);
    await upload('split-oracle', evaluation.split);
  }
  const models: readonly ('raw' | 'prefiltered')[] = arm === 'both' ? ['raw', 'prefiltered'] : [arm];
  for (const model of models) {
    const image = await train(device, model, { batch, steps, seed, precision }, params, references, lut, evaluation, metrics);
    score(metrics, model, image, evaluation);
    if (model === 'prefiltered') {
      const target = { ...evaluation, raw: evaluation.split };
      metrics.prefiltered_fit_oblique_rrmse = evaluateImage(image, target, (_pixel, _band, view) => view < 3).rrmse;
      metrics.prefiltered_fit_grazing_rrmse = evaluateImage(image, target, (_pixel, _band, view) => view >= 3).rrmse;
    }
    if (images) await uploadImage(model, { width, height: height * SPECULAR_VIEWS.length, channels: 3, data: image });
  }
  for (const buffer of [params, lut, references]) buffer.destroy();
  await drainDevice(device);
  if (deviceErrors.length) throw new Error(deviceErrors.join('\n'));
  reportResult({
    scene: 'nrc-prefiltered-specular',
    renderer: `compact-tiled-${precision}${arm === 'both' ? '' : `-${arm}`}`,
    config: { width, height, seed, refSpp, lutSpp, steps, batch, trainSpp, precision, images, arm },
    metrics,
    notes: {
      image: arm === 'both' ? 'traced' : arm,
      views_top_to_bottom: SPECULAR_VIEWS,
      roughness_top_to_bottom: SPECULAR_ROUGHNESS,
      traced_bands: ['delta mirror', 'glass'],
    },
  }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
