/** Phase 10 GPU checks on deterministic angular fields with independent CPU references. */
import { HashGrid, hashGridStorage, type HashGridSettings } from '../nrc/hash-grid.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import { analyzeTimestamps } from '../runner/timestamps.ts';
import { readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';

type Vec3 = [number, number, number];
type Arm = 'rgb' | 'bins' | 'sh';
type Field = (direction: Vec3) => Vec3;
const ARMS: readonly Arm[] = ['rgb', 'bins', 'sh'];

interface Fixture {
  name: string;
  train: Vec3[];
  queries: Vec3[];
  roughness: number;
  radiance: Field;
}

const IMAGE_WIDTH = 64;
const IMAGE_ROWS = 8;
const QUERIES = IMAGE_WIDTH * IMAGE_ROWS;
const ROUGHNESS = 0.3;
const DIR_BINS = 4;
const GOLDEN_RATIO_CONJUGATE = (Math.sqrt(5) - 1) / 2;
const P: Vec3 = [0.25, 0.25, 0.25];

function direction(y: number, phi: number): Vec3 {
  const radius = Math.sqrt(1 - y * y);
  return [radius * Math.cos(phi), y, radius * Math.sin(phi)];
}

function samples(count: number, minY: number, maxY: number): Vec3[] {
  return Array.from({ length: count }, (_, i) => direction(
    minY + (maxY - minY) * (i + 0.5) / count,
    2 * Math.PI * ((i * GOLDEN_RATIO_CONJUGATE + 0.137) % 1),
  ));
}

function queryDirections(minY: number): Vec3[] {
  return Array.from({ length: QUERIES }, (_, i) => direction(
    1 - (1 - minY) * (Math.floor(i / IMAGE_WIDTH) + 0.5) / IMAGE_ROWS,
    2 * Math.PI * ((i % IMAGE_WIDTH) + 0.277) / IMAGE_WIDTH,
  ));
}

const linear: Field = ([x, y, z]) => [
  1 + 0.45 * x - 0.2 * y + 0.3 * z,
  0.8 - 0.2 * x + 0.3 * y + 0.15 * z,
  0.65 + 0.15 * x + 0.3 * y - 0.2 * z,
];

function lobe(power: number, floor: number): Field {
  return ([x, y, z]) => {
    const peak = Math.max(0, 0.8 * x + 0.48 * y + 0.36 * z) ** power;
    return [floor + 3 * peak, floor + 1.95 * peak, floor + 0.9 * peak];
  };
}

const sphere = queryDirections(-1);
const hemisphere = queryDirections(0);
const fixtures: Fixture[] = [
  { name: 'linear', train: samples(1024, -1, 1), queries: sphere, roughness: ROUGHNESS, radiance: linear },
  { name: 'broad', train: samples(1024, 0, 1), queries: hemisphere, roughness: ROUGHNESS, radiance: lobe(2, 0.25) },
  { name: 'sparse', train: samples(8, 0, 1), queries: hemisphere, roughness: ROUGHNESS, radiance: linear },
  { name: 'skewed', train: samples(1024, 0.995, 1), queries: hemisphere, roughness: ROUGHNESS, radiance: linear },
  { name: 'diffuse', train: samples(256, 0, 1), queries: hemisphere, roughness: 1, radiance: linear },
  { name: 'mirror', train: samples(256, 0, 1), queries: hemisphere, roughness: 0.02, radiance: linear },
  { name: 'peaked', train: samples(1024, 0, 1), queries: hemisphere, roughness: ROUGHNESS, radiance: lobe(12, 0.015) },
];

function makeRecord(direction: Vec3, roughness: number, target: Vec3 = [0, 0, 0], point: Vec3 = P): number[] {
  const [x, y, z] = direction;
  return [
    ...point, 0,
    Math.acos(y) / Math.PI, Math.atan2(z, x) / (2 * Math.PI) + 0.5, 0, 0.5,
    1, 1, 1, roughness,
    ...target, 0,
  ];
}

/** Independent CPU control for the octahedral chart, written with scalar coordinates. */
function octBin([x, y, z]: Vec3): number {
  const scale = Math.abs(x) + Math.abs(y) + Math.abs(z);
  let column = x / scale;
  let row = z / scale;
  if (y < 0) {
    [column, row] = [(1 - Math.abs(row)) * Math.sign(x), (1 - Math.abs(column)) * Math.sign(z)];
  }
  const index = (v: number) => Math.min(DIR_BINS - 1, Math.floor((v + 1) * DIR_BINS / 2));
  return index(column) + DIR_BINS * index(row);
}

function referenceMeans(fixture: Fixture, binned: boolean): Map<number, Vec3> {
  const sums = new Map<number, { rgb: Vec3; count: number }>();
  for (const d of fixture.train) {
    const bin = binned ? octBin(d) : 0;
    const cell = sums.get(bin) ?? { rgb: [0, 0, 0], count: 0 };
    const rgb = fixture.radiance(d);
    for (let c = 0; c < 3; c++) cell.rgb[c]! += rgb[c]!;
    cell.count++;
    sums.set(bin, cell);
  }
  return new Map([...sums].map(([bin, { rgb, count }]) => [bin, [rgb[0] / count, rgb[1] / count, rgb[2] / count]]));
}

/** Solve the four-coefficient CPU fit directly in double precision, before radiance bounds. */
function referenceUnboundedFit(fixture: Fixture): Field {
  const rows = Array.from({ length: 4 }, () => new Float64Array(7));
  for (const direction of fixture.train) {
    const basis = [1, ...direction];
    const rhs = [...basis, ...fixture.radiance(direction)];
    for (const [i, row] of rows.entries()) {
      for (const [j, value] of rhs.entries()) row[j]! += basis[i]! * value;
    }
  }
  for (let i = 1; i < 4; i++) rows[i]![i]! += 0.02 * fixture.train.length;
  for (const [i, row] of rows.entries()) {
    const pivot = row[i]!;
    for (let j = i; j < row.length; j++) row[j]! /= pivot;
    for (const [k, target] of rows.entries()) {
      if (k === i) continue;
      const factor = target[i]!;
      for (let j = i; j < row.length; j++) target[j]! -= factor * row[j]!;
    }
  }
  return (d) => {
    const out: Vec3 = [0, 0, 0];
    const basis = [1, ...d];
    for (const [i, row] of rows.entries()) {
      for (let c = 0; c < 3; c++) out[c]! += basis[i]! * row[c + 4]!;
    }
    return out;
  };
}

function comparePredictions(
  arm: Arm, fixture: Fixture, predictions: Float32Array, raw: Float32Array, metrics: Record<string, number>,
): Float32Array {
  const binned = arm === 'bins' && fixture.roughness < 0.5;
  const means = referenceMeans(fixture, binned);
  const mean = means.get(0);
  const expectMean = arm === 'rgb' || arm === 'bins' ||
    (arm === 'sh' && ['sparse', 'skewed', 'diffuse', 'mirror'].includes(fixture.name));
  const pixels = new Float32Array(3 * QUERIES);
  let squaredError = 0;
  let squaredTarget = 0;
  let hits = 0;
  let hitSquaredError = 0;
  let hitSquaredTarget = 0;
  let maxReferenceError = 0;
  let rawMin = Infinity;
  let rawMax = -Infinity;
  let rawNegatives = 0;
  let clipped = 0;
  let fitted = 0;
  let darkChannels = 0;
  let darkError = 0;
  const fail = (i: number, message: string) => { throw new Error(`${arm} ${fixture.name} query ${i}: ${message}`); };
  for (const [i, d] of fixture.queries.entries()) {
    const target = fixture.radiance(d);
    const expected = binned ? means.get(octBin(d)) : mean;
    const status = predictions[i * 4 + 3];
    const hit = status === 0;
    if (status !== (expected ? 0 : -1)) fail(i, `expected ${expected ? 'hit' : 'miss'}, got level ${status}`);
    if (hit) hits++;
    if (raw[i * 4 + 3] === 1) fitted++;
    for (let c = 0; c < 3; c++) {
      const actual = predictions[i * 4 + c]!;
      const original = raw[i * 4 + c]!;
      const truth = target[c]!;
      if (!Number.isFinite(actual) || actual < 0 || !Number.isFinite(original)) {
        fail(i, `invalid RGB channel ${c}: effective ${actual}, raw ${original}`);
      }
      if (expectMean) maxReferenceError = Math.max(maxReferenceError, Math.abs(actual - (expected?.[c] ?? 0)));
      if (arm === 'sh' && fixture.name === 'linear') {
        maxReferenceError = Math.max(maxReferenceError, Math.abs(actual - truth));
      }
      pixels[i * 3 + c] = actual;
      squaredError += (actual - truth) ** 2;
      squaredTarget += truth ** 2;
      if (fixture.name === 'peaked' && truth < 0.025) {
        darkChannels++;
        darkError += actual - truth;
      }
      if (hit) {
        hitSquaredError += (actual - truth) ** 2;
        hitSquaredTarget += truth ** 2;
        rawMin = Math.min(rawMin, original);
        rawMax = Math.max(rawMax, original);
        rawNegatives += Number(original < 0);
        clipped += Number(Math.abs(original - actual) > 1e-5);
      }
    }
  }
  const tolerance = arm === 'sh' && fixture.name === 'linear' ? 0.04 : 1 / 256 + 1e-5;
  if (maxReferenceError > tolerance) fail(-1, `largest reference error ${maxReferenceError} exceeds ${tolerance}`);
  if (arm === 'sh' && fitted !== (expectMean ? 0 : QUERIES)) {
    fail(-1, `expected ${expectMean ? 0 : QUERIES} fitted queries, got ${fitted}`);
  }
  const prefix = `${arm}_${fixture.name}`;
  metrics[`${prefix}_rrmse`] = Math.sqrt(squaredError / squaredTarget);
  if (hits > 0) {
    metrics[`${prefix}_hit_rrmse`] = Math.sqrt(hitSquaredError / hitSquaredTarget);
    metrics[`${prefix}_raw_rgb_min`] = rawMin;
    metrics[`${prefix}_raw_rgb_max`] = rawMax;
  }
  metrics[`${prefix}_hit_fraction`] = hits / QUERIES;
  metrics[`${prefix}_raw_negative_channels`] = rawNegatives;
  metrics[`${prefix}_clipped_channels`] = clipped;
  if (darkChannels > 0) metrics[`${prefix}_dark_channel_mean_bias`] = darkError / darkChannels;
  if (arm === 'sh') metrics[`${prefix}_fit_fraction`] = fitted / QUERIES;
  if (expectMean || fixture.name === 'linear') metrics[`${prefix}_reference_max_abs`] = maxReferenceError;
  if (arm === 'sh' && fixture.name === 'peaked') {
    const fit = referenceUnboundedFit(fixture);
    const unbounded = fixture.queries.flatMap(fit);
    metrics.sh_peaked_unbounded_cpu_negative_channels = unbounded.filter((v) => v < 0).length;
    metrics.sh_peaked_unbounded_cpu_rgb_min = Math.min(...unbounded);
  }
  return pixels;
}

async function makeProbe(device: GPUDevice, grid: HashGrid, queries: GPUBuffer, output: GPUBuffer) {
  const hook = grid.displayHook;
  const code = /* wgsl */ `
    ${hook.wgsl}
    @group(3) @binding(6) var<storage, read> probes: array<vec4f>;
    @group(3) @binding(7) var<storage, read_write> answers: array<vec4f>;

    @compute @workgroup_size(64)
    fn probe(@builtin(global_invocation_id) id: vec3u) {
      let i = id.x;
      let b = i * 4u;
      let p = probes[b].xyz;
      let angles = probes[b + 1u];
      let roughness = probes[b + 2u].w;
      let predicted = hgSample(p, angles, roughness, i, 0u);
      answers[i] = predicted;
      var raw = vec4f(predicted.xyz, 0.0);
      ${grid.settings.directionalSh ? /* wgsl */ `
      if (predicted.w == 0.0 && hgIsModeratelyGlossy(roughness)) {
        let slot = hgFind(hgKey(p, hgBin(hgNormal(angles.zw)), hgLevel(p)));
        if (slot >= 0 && hgDirectionalReady(u32(slot))) {
          raw = vec4f(hgDirectionalValue(u32(slot), hgNormal(angles.xy)), 1.0);
        }
      }` : ''}
      answers[i + ${QUERIES}u] = raw;
    }
  `;
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'probe' },
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(3),
    entries: [...hook.entries, { binding: 6, resource: { buffer: queries } }, { binding: 7, resource: { buffer: output } }],
  });
  return (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({ label: 'inline directional oracle' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(3, group);
    pass.dispatchWorkgroups(QUERIES / 64);
    pass.end();
  };
}

function timingRecords(cells: number, count: number): Float32Array<ArrayBuffer> {
  return new Float32Array(samples(count, 0, 1).flatMap((d, i) =>
    makeRecord(d, ROUGHNESS, linear(d), [0.25 + i % cells, 0.25, 0.25])));
}

async function timeUpdates(device: GPUDevice, grid: HashGrid, frames: number): Promise<{ wallMs: number; gpuMs?: number; timestampStepNs?: number }> {
  const batchSize = 8;
  const batches = Math.ceil(frames / batchSize);
  const timestamps = device.features.has('timestamp-query') ? device.createQuerySet({ type: 'timestamp', count: frames * 4 }) : undefined;
  const resolveBuffer = timestamps
    ? device.createBuffer({ size: frames * 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : undefined;

  function batchCommands(count: number, batch?: number): GPUCommandBuffer {
    const encoder = device.createCommandEncoder({ label: 'directional hash batch' });
    for (let i = 0; i < count; i++) {
      const stamp = 4 * ((batch ?? 0) * batchSize + i);
      const timing = timestamps && batch !== undefined ? {
        update: { querySet: timestamps, beginningOfPassWriteIndex: stamp, endOfPassWriteIndex: stamp + 1 },
        resolve: { querySet: timestamps, beginningOfPassWriteIndex: stamp + 2, endOfPassWriteIndex: stamp + 3 },
      } : undefined;
      grid.encodeUpdate(encoder, timing);
    }
    return encoder.finish();
  }

  try {
    device.queue.submit([batchCommands(batchSize)]);
    await device.queue.onSubmittedWorkDone();
    let wallMs = 0;
    for (let batch = 0; batch < batches; batch++) {
      const commands = batchCommands(Math.min(batchSize, frames - batch * batchSize), batch);
      const start = performance.now();
      device.queue.submit([commands]);
      await device.queue.onSubmittedWorkDone();
      wallMs += performance.now() - start;
    }
    if (!timestamps || !resolveBuffer) return { wallMs: wallMs / frames };
    const resolve = device.createCommandEncoder();
    resolve.resolveQuerySet(timestamps, 0, timestamps.count, resolveBuffer, 0);
    device.queue.submit([resolve.finish()]);
    const values = new BigUint64Array(await readBuffer(device, resolveBuffer));
    const { durationsNs, stepNs, quantized } = analyzeTimestamps(values);
    const updateNs = durationsNs.reduce((sum, value, i) => sum + (i % 2 === 0 ? value : 0), 0);
    const resolveNs = durationsNs.reduce((sum, value, i) => sum + (i % 2 === 1 ? value : 0), 0);
    const valid = !quantized && values.every((value) => value > 0n) && durationsNs.every((value) => value >= 0);
    return {
      wallMs: wallMs / frames,
      ...(valid && updateNs + resolveNs > 0 ? { gpuMs: (updateNs + resolveNs) / 1e6 / frames } : {}),
      timestampStepNs: valid ? Number(stepNs) : 0,
    };
  } finally {
    timestamps?.destroy();
    resolveBuffer?.destroy();
  }
}

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const log2Entries = numberArg(args, 'hgLog2', 16);
  const memoryMiB = numberArg(args, 'memoryMiB', 10);
  const timeFrames = numberArg(args, 'timeFrames', 64);
  const activeMaintenance = args.hgActive !== '0';
  const images = args.images === '1';
  const arms = (args.arms ?? ARMS.join(',')).split(',') as Arm[];
  if (arms.some((arm) => !ARMS.includes(arm)) || new Set(arms).size !== arms.length) {
    throw new Error('arms must list one or more distinct cache arms: rgb, bins, sh');
  }
  if (!Number.isInteger(log2Entries) || log2Entries < 10 || log2Entries > 20) {
    throw new Error('hgLog2 must be an integer in 10..20');
  }
  if (!Number.isInteger(timeFrames) || timeFrames < 0 || timeFrames > 256) {
    throw new Error('timeFrames must be an integer in 0..256');
  }
  if (memoryMiB <= 0) throw new Error('memoryMiB must be positive');

  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);
  const timeCount = 16384;
  const timings = timeFrames > 0 ? [
    { name: 'distributed', records: timingRecords(256, timeCount) },
    { name: 'hot', records: timingRecords(1, timeCount) },
  ] : [];
  const capacity = timeFrames > 0 ? timeCount : Math.max(...fixtures.map(({ train }) => train.length));
  const records = storageBuffer(device, capacity * 64, 'directional training records');
  const count = storageBuffer(device, 16, 'directional counts');
  const queries = storageBuffer(device, QUERIES * 64, 'directional query records');
  const predictions = storageBuffer(device, QUERIES * 16, 'directional predictions');
  const inline = storageBuffer(device, QUERIES * 32, 'directional probe predictions');
  const metrics: Record<string, number> = {};
  const settings: Partial<HashGridSettings> = {
    activeMaintenance, log2Entries, minCell: 1, cellPixels: 1,
    ...(args.hgLog2 === undefined ? { memoryBudgetBytes: 4 * Math.floor(memoryMiB * 1024 ** 2 / 4) } : {}),
    minSamples: 1, fallbackLevels: 0, pyramid: false, historyFrames: 32, staleFrames: 2,
  };
  const view = { yFovDeg: 60, filmHeight: 1024 };

  for (const arm of arms) {
    const grid = await HashGrid.create(device, {
      ...settings, dirBins: arm === 'bins' ? DIR_BINS : 0, directionalSh: arm === 'sh',
    }, view);
    try {
      grid.bindQueries(queries, count, predictions, QUERIES);
      const probe = await makeProbe(device, grid, queries, inline);
      const pixels = new Float32Array(fixtures.length * QUERIES * 3);
      for (const [fixtureIndex, fixture] of fixtures.entries()) {
        grid.clear();
        const known = new Float32Array(fixture.train.flatMap((d) => makeRecord(d, fixture.roughness, fixture.radiance(d))));
        const heldOut = new Float32Array(fixture.queries.flatMap((d) => makeRecord(d, fixture.roughness)));
        grid.bindRecords(records, count, fixture.train.length);
        device.queue.writeBuffer(records, 0, known);
        device.queue.writeBuffer(queries, 0, heldOut);
        device.queue.writeBuffer(count, 0, new Uint32Array([QUERIES, fixture.train.length]));
        const encoder = device.createCommandEncoder({ label: fixture.name });
        grid.encodeUpdate(encoder);
        grid.encodeLookup(encoder);
        probe(encoder);
        device.queue.submit([encoder.finish()]);
        const output = new Float32Array(await readBuffer(device, predictions));
        const probeOutput = new Float32Array(await readBuffer(device, inline));
        if (!output.every((value, i) => Math.abs(value - probeOutput[i]!) <= 1e-5)) {
          throw new Error(`${arm} ${fixture.name}: deferred and inline predictions differ`);
        }
        const raw = probeOutput.subarray(QUERIES * 4);
        const state = await grid.readStats();
        if (state.dropped !== 0) throw new Error(`${arm} ${fixture.name}: dropped ${state.dropped} oracle records`);
        const expectedCells = referenceMeans(fixture, arm === 'bins' && fixture.roughness < 0.5).size;
        if (state.cells !== expectedCells) {
          const ages = await grid.readCellAges();
          throw new Error(`${arm} ${fixture.name}: expected ${expectedCells} cells, got ${JSON.stringify(ages)}`);
        }
        pixels.set(comparePredictions(arm, fixture, output, raw, metrics), fixtureIndex * QUERIES * 3);
        metrics[`${arm}_${fixture.name}_cells`] = state.cells;
      }

      // End-of-frame ages expire three untouched updates after the training frame.
      device.queue.writeBuffer(count, 0, new Uint32Array([QUERIES, 0]));
      const expiry = device.createCommandEncoder({ label: 'expire directional history' });
      for (let i = 0; i < 3; i++) grid.encodeUpdate(expiry);
      grid.encodeLookup(expiry);
      device.queue.submit([expiry.finish()]);
      const expired = new Float32Array(await readBuffer(device, predictions));
      if (!expired.every((value, i) => value === (i % 4 === 3 ? -1 : 0))) {
        throw new Error(`${arm}: stale directional history remained visible`);
      }

      metrics[`${arm}_entries`] = grid.entries;
      metrics[`${arm}_bytes`] = grid.bytes;
      // Logical accumulator traffic counts the four-byte read and write of every 32-bit atomic add.
      // It excludes hash probes, maintenance, uniform reads, and physical cache-line traffic.
      const accounting = hashGridStorage(grid.settings);
      const accumulatorAtomics = accounting.updateAtomicAddsPerCell + accounting.directionalUpdateAtomicAddsPerCell;
      metrics[`${arm}_accumulator_atomics_per_record`] = accumulatorAtomics;
      metrics[`${arm}_accumulator_logical_rmw_bytes_per_record`] = accumulatorAtomics * 8;
      grid.bindRecords(records, count, capacity);
      for (const workload of timings) {
        grid.clear();
        device.queue.writeBuffer(records, 0, workload.records);
        device.queue.writeBuffer(count, 0, new Uint32Array([0, timeCount]));
        const prefix = `${arm}_${workload.name}`;
        const { wallMs, gpuMs, timestampStepNs } = await timeUpdates(device, grid, timeFrames);
        const state = await grid.readStats();
        metrics[`${prefix}_update_wall_ms`] = wallMs;
        if (gpuMs !== undefined) metrics[`${prefix}_update_gpu_ms`] = gpuMs;
        if (timestampStepNs !== undefined) metrics[`${prefix}_timestamp_step_ns`] = timestampStepNs;
        const dropFraction = state.dropped / (state.frames * timeCount);
        metrics[`${prefix}_cells`] = state.cells;
        if (state.touchedFraction !== undefined) metrics[`${prefix}_touched_cells`] = state.touchedFraction * grid.entries;
        metrics[`${prefix}_drop_fraction`] = dropFraction;
        metrics[`${prefix}_accumulator_logical_rmw_bytes_per_update`] = accumulatorAtomics * 8 * timeCount * (1 - dropFraction);
      }
      if (images) await uploadImage(arm, { width: IMAGE_WIDTH, height: fixtures.length * IMAGE_ROWS, channels: 3, data: pixels });
    } finally {
      grid.destroy();
    }
  }

  if (images) {
    const data = new Float32Array(fixtures.flatMap(({ queries, radiance }) => queries.flatMap(radiance)));
    await uploadImage('analytic', { width: IMAGE_WIDTH, height: fixtures.length * IMAGE_ROWS, channels: 3, data });
  }
  for (const buffer of [records, count, queries, predictions, inline]) buffer.destroy();

  await drainDevice(device);
  if (deviceErrors.length) throw new Error(deviceErrors.join('\n'));
  reportResult({
    scene: 'nrc-directional-check',
    renderer: arms.length === 1 ? `hash-${arms[0]}` : 'ours',
    config: {
      memoryMiB: args.hgLog2 === undefined ? memoryMiB : null,
      log2Entries: args.hgLog2 === undefined ? null : log2Entries,
      timeFrames, timeCount, activeMaintenance, images, arms,
    },
    metrics,
    notes: {
      image: arms[0],
      angular_rows: fixtures.map(({ name }) => name),
      angular_row_height: IMAGE_ROWS,
      timing: 'Eight updates per submit after eight warm-up updates. GPU timing sums the update and resolve pass durations.',
    },
  }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
