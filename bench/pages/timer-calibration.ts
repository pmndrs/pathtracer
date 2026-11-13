/**
 * Timer calibration: is `timestamp-query` quantized in this session?
 *
 * Chrome rounds GPU timestamps to 100 us unless it runs with --enable-webgpu-developer-features, and a quantized
 * timer makes every sub-millisecond pass read as 0 or 0.1 ms. Run this page at the start of every timing session
 * (`node bench/runner/run.ts --page timer-calibration`); the runner records `ts_quantized` in the CSV.
 * Modeled on ~/Developer/upscaler/bench/timer-overhead.html (a probe page driven over CDP), but it measures the
 * timer's resolution rather than its overhead.
 *
 * Method: time a compute pass at three workload sizes (empty, light, heavy), `samples` passes each, all
 * timestamps resolved into one buffer and read back once at the end. Then:
 *  - step: the greatest common divisor of all differences between timestamps, in ns. Real ticks give a few ns;
 *    rounding to 100 us gives 100000;
 *  - zero share: the share of passes that measured 0 ns;
 *  - distinct: how many different pass durations appeared.
 * Quantized = step >= 10 us. The performance.now() step is reported too (CPU timer, for context).
 *
 * Args: samples=100 (passes per workload)
 */

import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice, drainDevice } from '../runner/page-client.ts';
import type { AdapterSummary } from '../runner/contract.ts';
import { analyzeTimestamps } from '../runner/timestamps.ts';

const SHADER = /* wgsl */ `
struct Params { iterations: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> sink: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  var acc = f32(id.x);
  for (var i = 0u; i < params.iterations; i++) {
    acc = acc * 1.0001 + f32(i);
  }
  sink[id.x] = acc;
}
`;

const WORKLOADS = [
  { name: 'empty', iterations: 0 },
  { name: 'light', iterations: 256 },
  { name: 'heavy', iterations: 65536 },
] as const;
const GROUPS = 256;

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? (sorted[middle] as number) : ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2;
}

/** Smallest nonzero gap between consecutive performance.now() readings, in microseconds. */
function performanceNowStepUs(): number {
  let smallest = Infinity;
  for (let i = 0; i < 20_000; i++) {
    const a = performance.now();
    let b = performance.now();
    while (b === a) b = performance.now();
    smallest = Math.min(smallest, b - a);
  }
  return smallest * 1000;
}

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const samples = numberArg(args, 'samples', 100);

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  if (!adapter.features.has('timestamp-query')) {
    reportResult(
      { renderer: 'timer-calibration', metrics: { ts_supported: 0 }, valid: false, notes: { problem: 'adapter has no timestamp-query feature' } },
      { adapter: adapterSummary },
    );
    throw new Error('stop'); // swallowed below; the report is already published
  }
  const device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
  deviceErrors = watchDevice(device);

  const passCount = samples * WORKLOADS.length;
  const querySet = device.createQuerySet({ type: 'timestamp', count: passCount * 2 });
  const resolveBuffer = device.createBuffer({ size: passCount * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: passCount * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const sink = device.createBuffer({ size: GROUPS * 64 * 4, usage: GPUBufferUsage.STORAGE });
  const module = device.createShaderModule({ code: SHADER });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });

  const bindGroups = WORKLOADS.map((workload) => {
    const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniform, 0, new Uint32Array([workload.iterations, 0, 0, 0]));
    return device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: sink } },
      ],
    });
  });

  // Warm up (pipeline, driver), then time. Interleave nothing: each workload's passes are contiguous.
  {
    const warm = device.createCommandEncoder();
    const pass = warm.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroups[2] as GPUBindGroup);
    pass.dispatchWorkgroups(GROUPS);
    pass.end();
    device.queue.submit([warm.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const wallStart = performance.now();
  const encoder = device.createCommandEncoder();
  let index = 0;
  for (const [w] of WORKLOADS.entries()) {
    for (let s = 0; s < samples; s++) {
      const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index * 2, endOfPassWriteIndex: index * 2 + 1 } });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroups[w] as GPUBindGroup);
      pass.dispatchWorkgroups(GROUPS);
      pass.end();
      index++;
    }
  }
  encoder.resolveQuerySet(querySet, 0, passCount * 2, resolveBuffer, 0);
  encoder.copyBufferToBuffer(resolveBuffer, 0, readback, 0, passCount * 16);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ); // the one readback, after all timed work
  const wallMs = performance.now() - wallStart;
  const stamps = new BigUint64Array(readback.getMappedRange().slice(0));
  readback.unmap();

  // Durations per workload and the timestamp step.
  const analysis = analyzeTimestamps(stamps);
  const durations = WORKLOADS.map((_, w) => analysis.durationsNs.slice(w * samples, (w + 1) * samples));
  const quantized = analysis.quantized;

  await drainDevice(device);
  reportResult(
    {
      renderer: 'timer-calibration',
      config: { samples, groups: GROUPS, workloads: WORKLOADS.map((workload) => workload.iterations) },
      metrics: {
        ts_supported: 1,
        ts_quantized: quantized ? 1 : 0,
        ts_step_ns: Number(analysis.stepNs),
        ts_distinct_durations: analysis.distinctDurations,
        ts_zero_interval_share: analysis.zeroShare,
        ts_min_nonzero_ns: analysis.minNonzeroNs,
        gpu_ns_median_empty: median(durations[0] as number[]),
        gpu_ns_median_light: median(durations[1] as number[]),
        gpu_ns_median_heavy: median(durations[2] as number[]),
        wall_ms: wallMs,
        performance_now_step_us: performanceNowStepUs(),
      },
      // Not a verdict on the machine: a quantized session cannot support timing, so say so in the CSV.
      valid: !quantized,
      notes: {
        quantized,
        advice: quantized ? 'Timestamps are rounded (about 100 us). Run Chrome with --enable-webgpu-developer-features; the runner passes it, so check for a wrapper or a changed Chrome.' : 'Timestamps are fine-grained.',
      },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  if (!(error instanceof Error && error.message === 'stop')) reportError(error, { adapter: adapterSummary, deviceErrors });
}
