/** Deterministic full-table versus active maintenance check using identical hash update records and queries. */
import { HashGrid } from '../nrc/hash-grid.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { readBuffer, requestMlpDevice, storageBuffer } from './mlp-common.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const HOME: [number, number, number] = [0, 0, 0];
const CUT: [number, number, number] = [0, 0, 1000];

function record(z: number, value: number): number[] {
  return [0.25, 0.25, z + 0.25, 0, 0, 0, 0, 0, 1, 1, 1, 1, value, 2 * value, 3 * value, 0];
}

try {
  const args = getArgs();
  const log2Entries = numberArg(args, 'hgLog2', 20);
  const checkVariance = args.checkVariance === '1';
  const timeFrames = numberArg(args, 'timeFrames', 0);
  if (!Number.isSafeInteger(timeFrames) || timeFrames < 0) throw new Error('timeFrames must be a nonnegative integer');
  const { adapter, device } = await requestMlpDevice();
  adapterSummary = summarizeAdapter(adapter);
  deviceErrors = watchDevice(device);

  const inlineCoarse = record(0, 0);
  inlineCoarse.splice(4, 4, 2, 3, 4, 1);
  inlineCoarse[15] = 4;
  const queryData = new Float32Array([...record(0, 0), ...record(1000, 0), ...inlineCoarse]);
  const queries = storageBuffer(device, queryData, 'parity queries');
  const records = storageBuffer(device, 3 * 64, 'parity records');
  const count = storageBuffer(device, 16, 'parity counts');
  const predictions = storageBuffer(device, 3 * 16, 'parity predictions');
  const metrics: Record<string, number> = {};
  const reference = new Map<string, { ages: string; predictions: Float32Array }>();

  async function runArm(grid: HashGrid) {
    const active = grid.settings.activeMaintenance;
    const arm = active ? 'active' : 'full';
    device.queue.writeBuffer(queries, 0, queryData);
    grid.bindQueries(queries, count, predictions, 3);
    grid.bindRecords(records, count, 3);

    let checkpoints = 0;
    async function check(
      name: string, camera: [number, number, number], train: number[][], updates: number,
      frames: number, ages: number[], rgb: [number | null, number | null], touched: number,
    ) {
      grid.setCameraPosition(camera);
      grid.resetStats();
      if (train.length) device.queue.writeBuffer(records, 0, new Float32Array(train.flat()));
      device.queue.writeBuffer(count, 0, new Uint32Array([3, train.length]));
      const encoder = device.createCommandEncoder({ label: name });
      for (let i = 0; i < updates; i++) grid.encodeUpdate(encoder);
      grid.encodeLookup(encoder);
      device.queue.submit([encoder.finish()]);
      // Map and destroy one staging buffer at a time.
      const cellAges = await grid.readCellAges();
      const stats = await grid.readStats();
      const output = new Float32Array(await readBuffer(device, predictions));
      const fail = (detail: string) => { throw new Error(`${arm} ${name}: ${detail}`); };
      if (JSON.stringify(cellAges.map((cell) => cell.frame).sort((a, b) => a - b)) !== JSON.stringify(ages)) {
        fail(`expected age stamps ${ages}, saw ${JSON.stringify(cellAges)}`);
      }
      const expected = [
        ...rgb.flatMap((value) => value === null ? [0, 0, 0, -1] : [value, 2 * value, 3 * value, 0]),
        2, 3, 4, 1,
      ];
      if (!output.every((value, i) => Math.abs(value - expected[i]!) < 1e-5)) {
        fail(`predicted ${output}, expected ${expected}`);
      }
      const checksumAges = JSON.stringify(cellAges);
      if (active) {
        const baseline = reference.get(name)!;
        if (checksumAges !== baseline.ages) fail('occupied checksums or ages differ between arms');
        if (!output.every((value, i) => value === baseline.predictions[i])) fail('predictions differ between arms');
      } else {
        reference.set(name, { ages: checksumAges, predictions: output });
      }
      if (stats.frames !== frames || stats.cells !== ages.length || stats.misses !== rgb.filter((value) => value === null).length ||
          stats.continued !== 1 || stats.coarse !== 1 || stats.dropped !== 0) fail(`unexpected stats ${JSON.stringify(stats)}`);
      if (active && stats.touchedFraction !== touched / grid.entries) fail('touched-slot count differs from expected');
      checkpoints++;
    }

    await check('cold home', HOME, [record(0, 1)], 1, 1, [0, 0], [1, null], 2);
    await check('camera cut', CUT, [record(1000, 2)], 1, 2, [0, 0, 1, 1], [null, 2], 2);
    await check('exact age boundary', CUT, [], 1, 3, [0, 0, 1, 1], [null, 2], 0);
    await check('return and refresh on expiration frame', HOME, [record(0, 3), record(0, 3)], 1, 4, [1, 1, 3, 3], [13 / 5, null], 2);
    await check('evict cut cells', CUT, [], 1, 5, [3, 3], [null, null], 0);
    await check('batched empty frames evict home cells', HOME, [], 2, 7, [], [null, null], 0);
    await check('reinsert without old history', HOME, [record(0, 4)], 1, 8, [7, 7], [4, null], 2);
    grid.clear();
    await check('clear populated grid', HOME, [], 0, 0, [], [null, null], 0);

    device.queue.writeBuffer(grid.stats, 0, new Uint32Array([0xffff_fffe]));
    await check('warm before frame wrap', HOME, [record(0, 1)], 1, 0xffff_ffff, [0xffff_fffe, 0xffff_fffe], [1, null], 2);
    await check('retain across frame wrap', HOME, [], 2, 1, [0xffff_fffe, 0xffff_fffe], [1, null], 0);
    await check('evict after frame wrap', HOME, [], 1, 2, [], [null, null], 0);

    metrics.checkpoints = checkpoints;
    metrics[`${arm}_bytes`] = grid.bytes;
    if (checkVariance) {
      const confidenceOutput = storageBuffer(device, 2 * 32, 'variance outputs');
      const code = /* wgsl */ `
        ${grid.displayHook.wgsl}
        @group(3) @binding(6) var<storage, read_write> output: array<vec4f>;
        @compute @workgroup_size(2)
        fn probe(@builtin(local_invocation_index) i: u32) {
          let point = vec3f(0.25 + f32(i), 0.25, 0.25);
          let sample = hgSampleWithConfidence(point, vec4f(0.0), 1.0, 0u, 0u);
          output[i * 2u] = sample.prediction;
          let raw = hgSample(point, vec4f(0.0), 1.0, 0u, 0u);
          output[i * 2u + 1u] = vec4f(select(0.0, 1.0, sample.varianceRejected), raw.xyz);
        }
      `;
      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'probe' },
      });
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(3),
        entries: [...grid.displayHook.entries, { binding: 6, resource: { buffer: confidenceOutput } }],
      });
      let varianceChecks = 0;
      async function varianceCheck(name: string, values: number[], expected: number | null, rejected: boolean) {
        const train = values.flatMap((value) => [...record(0, 0).slice(0, 12), value, value, value, 0]);
        if (values.length) device.queue.writeBuffer(records, 0, new Float32Array(train));
        device.queue.writeBuffer(count, 0, new Uint32Array([0, values.length]));
        const encoder = device.createCommandEncoder({ label: name });
        grid.encodeUpdate(encoder);
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(3, group);
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const actual = new Float32Array(await readBuffer(device, confidenceOutput));
        const test = [0, 1].flatMap((level) => [
          expected ?? 0, expected ?? 0, expected ?? 0, expected === null ? -1 : level,
          Number(rejected), expected ?? 0, expected ?? 0, expected ?? 0,
        ]);
        if (!actual.every((value, i) => Math.abs(value - test[i]!) < 1e-4)) {
          throw new Error(`${arm} ${name}: saw ${actual}, expected ${test}`);
        }
        varianceChecks++;
      }

      // One sample establishes each cell. Three samples in one later frame mix two endpoints.
      // Two maximum-square terms force a carry in the second-moment accumulator.
      await varianceCheck('constant zero', [0], 0, false);
      await varianceCheck('mixed frame and wide square sum', [0, 1000, 1000], 2000 / 3.5, true);
      await varianceCheck('no new samples', [], 2000 / 3.5, true);
      await varianceCheck('stale boundary retains moments', [], 2000 / 3.5, true);
      await varianceCheck('stale eviction', [], null, false);
      await varianceCheck('constant reinsert after eviction', [1000], 1000, false);
      await varianceCheck('mixed moments before clear', [0, 0], 200, true);
      grid.clear();
      await varianceCheck('constant reinsert after clear', [1000], 1000, false);

      device.queue.writeBuffer(queries, 0, new Float32Array([...inlineCoarse.slice(0, 15), 5]));
      device.queue.writeBuffer(count, 0, new Uint32Array([1, 0]));
      const encoder = device.createCommandEncoder({ label: 'forward raw prediction after variance rejection' });
      grid.encodeLookup(encoder);
      device.queue.submit([encoder.finish()]);
      const stats = await grid.readStats();
      const output = new Float32Array(await readBuffer(device, predictions));
      if (stats.varianceRejects !== 1 || stats.continued !== 1 || stats.coarse !== 1 || stats.misses !== 0 ||
          ![2, 3, 4, 1].every((value, j) => output[j] === value)) {
        throw new Error(`${arm} variance rejection lost the stored prediction or hit statistics: ${output} ${JSON.stringify(stats)}`);
      }
      confidenceOutput.destroy();
      metrics.variance_checkpoints = varianceChecks + 1;
    }
    if (timeFrames > 0) {
      // Keep at most eight updates in flight. Measure completion from submission on a drained queue.
      const batchFrames = 8;
      device.queue.writeBuffer(records, 0, new Float32Array(record(0, 1)));
      device.queue.writeBuffer(count, 0, new Uint32Array([3, 1]));

      function updateBatch(frames: number): GPUCommandBuffer {
        const encoder = device.createCommandEncoder({ label: `${arm} sparse updates` });
        for (let i = 0; i < frames; i++) grid.encodeUpdate(encoder);
        return encoder.finish();
      }

      device.queue.submit([updateBatch(batchFrames)]);
      await device.queue.onSubmittedWorkDone();
      let elapsed = 0;
      for (let i = 0; i < timeFrames; i += batchFrames) {
        const commands = updateBatch(Math.min(batchFrames, timeFrames - i));
        const start = performance.now();
        device.queue.submit([commands]);
        await device.queue.onSubmittedWorkDone();
        elapsed += performance.now() - start;
      }
      metrics.time_batch_frames = Math.min(batchFrames, timeFrames);
      metrics[`${arm}_update_wall_ms`] = elapsed / timeFrames;
      metrics[`${arm}_touched_fraction`] = 2 / grid.entries;
    }
  }

  // The reference is compact CPU data. Only one GPU grid is resident at a time.
  for (const activeMaintenance of [false, true]) {
    const grid = await HashGrid.create(device, {
      activeMaintenance, varianceThreshold: checkVariance ? 0.1 : -1, log2Entries, minCell: 1, cellPixels: 1,
      historyFrames: 2, staleFrames: 2, minSamples: 1, fallbackLevels: 1, pyramid: true,
    }, { yFovDeg: 60, filmHeight: 128 });
    try {
      await runArm(grid);
      await device.queue.onSubmittedWorkDone();
    } finally {
      grid.destroy();
    }
  }
  for (const buffer of [queries, records, count, predictions]) buffer.destroy();

  await drainDevice(device);
  if (deviceErrors.length) throw new Error(deviceErrors.join('\n'));
  reportResult({
    scene: 'nrc-hash-maintenance',
    renderer: 'ours',
    config: { log2Entries, timeFrames, checkVariance },
    valid: true,
    metrics,
  }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
