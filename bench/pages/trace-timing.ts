/**
 * Trace core throughput (P1 slice 1): rays/s of the batch dispatcher for L1 (closest hit) and L0 (visibility).
 *
 * Validity: the runner's load check marks rows taken on a busy machine `valid=0` with note `loaded`; quote only the
 * others. (Slice 1's first numbers, P1.2, were taken under load and stay provisional.)
 *
 * Ray sets, built once outside the timed region (one setup readback of the primary hits is allowed there):
 *  - primary: camera rays (coherent), L1;
 *  - bounce: cosine-distributed rays from the primary hits (incoherent), L1 and L0 (any hit up to `aoDistance`);
 *  - shadow: segments from the primary hits toward the sun or an emitter, L0.
 * All (kernel, set) jobs first run untimed, round after round, for `warmupMs`; then `repeats` rounds, each job in its own
 * pass with GPU timestamps, interleaved per round; the 10th-percentile pass time gives rays/s (the median is kept too:
 * on a shared machine other apps' GPU work lands inside passes in bursts, see the comment at the metrics). One readback of the resolved timestamps at the very end.
 *
 * Args: scene=sponza-white rays=1048576 workgroupSize=64 rayLayout=aos hitRecord=full repeats=30 warmupMs=2000
 * A/B a setting with the runner: `--ab workgroupSize=64,128` or `--ab rayLayout=aos,soa`.
 */
import { buildTraceScene, createMeshBvhBuilder, uploadTraceScene, type TraceSceneBuffers } from '@pmndrs/bvh';
import {
  BATCH_WORKGROUP_SIZES,
  RtBatchTracer,
  aosToSoa,
  detectCaps,
  readHits,
  requestDeviceForCaps,
  type BatchWorkgroupSize,
  type Ray,
  type RayLayout,
} from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { cameraRays, createRandom, loadTraceScene, packRays, shadowRays, surfacePoints, type Vec3 } from './trace-scenes.ts';

const STORAGE_IO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

function upload(device: GPUDevice, data: Float32Array<ArrayBuffer>): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: STORAGE_IO });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

async function readBack(device: GPUDevice, buffer: GPUBuffer, size = buffer.size): Promise<ArrayBuffer> {
  const copy = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, copy, 0, size);
  device.queue.submit([encoder.finish()]);
  await copy.mapAsync(GPUMapMode.READ);
  const data = copy.getMappedRange().slice(0);
  copy.destroy();
  return data;
}

/** Cosine-weighted direction about `n`. */
function cosineDirection(n: Vec3, random: () => number): Vec3 {
  const r = Math.sqrt(random());
  const phi = 2 * Math.PI * random();
  const x = r * Math.cos(phi);
  const y = r * Math.sin(phi);
  const z = Math.sqrt(Math.max(0, 1 - x * x - y * y));
  const t: Vec3 = Math.abs(n[0]) > 0.9 ? [0, 1, 0] : [1, 0, 0];
  const b1: Vec3 = [n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
  const l = Math.hypot(...b1);
  const u: Vec3 = [b1[0] / l, b1[1] / l, b1[2] / l];
  const v: Vec3 = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
  return [0, 1, 2].map((k) => u[k]! * x + v[k]! * y + n[k]! * z) as Vec3;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? (sorted[mid] as number) : ((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2;
}

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  const sceneName = args.scene ?? 'sponza-white';
  const count = numberArg(args, 'rays', 1 << 20);
  const workgroupSize = numberArg(args, 'workgroupSize', 64) as BatchWorkgroupSize;
  if (!BATCH_WORKGROUP_SIZES.includes(workgroupSize)) throw new Error(`workgroupSize must be one of ${BATCH_WORKGROUP_SIZES.join(', ')}`);
  const rayLayout = (args.rayLayout ?? 'aos') as RayLayout;
  const hitRecord = (args.hitRecord ?? 'full') as 'full' | 'compact';
  const repeats = numberArg(args, 'repeats', 30);
  // Untimed warm-up by time, not by count: the Apple GPU needs about a second of steady work before it reaches full
  // clock, and with 3 warm-up batches the same pass flipped between ~9 and ~4.5 ms inside one run (2026-11-14).
  const warmupMs = numberArg(args, 'warmupMs', 2000);

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'trace-timing' });
  deviceErrors = watchDevice(device);
  if (!caps.timestampQuery) throw new Error('trace-timing needs timestamp-query (no wall-clock fallback: it would time the CPU too).');

  // Scene: bvh defaults (quantized nodes, record leaves, static instances merged), as SceneTrees would make it.
  const scene = await loadTraceScene(sceneName);
  const builder = createMeshBvhBuilder();
  const data = await buildTraceScene(scene.meshes, scene.instances, { build: (geometry) => builder.build(geometry), mergeStaticInstances: true });
  const buildersUsedWorkers = builder.usesWorker;
  builder.dispose();
  const gpuScene: TraceSceneBuffers = uploadTraceScene(device, data, sceneName);
  const format = { nodeFormat: gpuScene.nodeFormat, triangleLeaves: gpuScene.triangleLeaves, positionLayout: gpuScene.positionLayout };
  const common = { ...format, workgroupSize, rayLayout, hitRecord };
  const closest = await RtBatchTracer.create(device, caps, { ...common, mode: 'closest', label: 'timing closest' });
  const visible = await RtBatchTracer.create(device, caps, { ...common, mode: 'visible', label: 'timing visible' });
  // Setup only: a full-record AoS tracer to find the primary hits.
  const setupTracer = await RtBatchTracer.create(device, caps, { ...format, mode: 'closest', label: 'timing setup' });

  // ---- Rays (setup, untimed) ----
  const random = createRandom(4242);
  const primaryRays = cameraRays(scene, count, random);
  if (!primaryRays.length) throw new Error(`scene ${sceneName} has no camera`);
  const primaryPacked = packRays(primaryRays);
  const primaryBuffer = upload(device, primaryPacked);
  const setupHits = device.createBuffer({ size: count * setupTracer.resultBytes, usage: STORAGE_IO });
  {
    const encoder = device.createCommandEncoder();
    setupTracer.encode(encoder, gpuScene, { rays: primaryBuffer, results: setupHits, count });
    device.queue.submit([encoder.finish()]);
  }
  const hits = readHits(await readBack(device, setupHits), count);
  setupHits.destroy();
  const points = surfacePoints(primaryRays, hits.map((hit) => (hit.t >= 0 ? hit : null)));
  const eps = 1e-4 * scene.scale;
  const aoDistance = numberArg(args, 'aoDistance', 0.1 * scene.scale);
  const bounceRays: Ray[] = [];
  for (let i = 0; bounceRays.length < count && points.length; i++) {
    const p = points[i % points.length]!;
    bounceRays.push({ origin: [p.position[0] + p.normal[0] * eps, p.position[1] + p.normal[1] * eps, p.position[2] + p.normal[2] * eps], tMin: 0, direction: cosineDirection(p.normal, random), tMax: 1e30 });
  }
  const aoRays = bounceRays.map((ray) => ({ ...ray, tMax: aoDistance }));
  const shadow = shadowRays(scene, points, count, random);

  interface Job {
    name: string;
    tracer: RtBatchTracer;
    rays: Ray[];
  }
  const jobs: Job[] = [
    { name: 'l1_primary', tracer: closest, rays: primaryRays },
    { name: 'l1_bounce', tracer: closest, rays: bounceRays },
    { name: 'l0_shadow', tracer: visible, rays: shadow },
    { name: 'l0_ao', tracer: visible, rays: aoRays },
  ].filter((job) => job.rays.length > 0);

  const querySet = device.createQuerySet({ type: 'timestamp', count: jobs.length * repeats * 2 });
  const resolve = device.createBuffer({ size: jobs.length * repeats * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const prepared = jobs.map((job) => {
    const packed = packRays(job.rays);
    const raysIn: GPUBuffer | [GPUBuffer, GPUBuffer] = rayLayout === 'soa' ? (() => { const s = aosToSoa(packed); return [upload(device, s.a), upload(device, s.b)] as [GPUBuffer, GPUBuffer]; })() : upload(device, packed);
    const results = device.createBuffer({ size: job.rays.length * job.tracer.resultBytes, usage: STORAGE_IO });
    return { ...job, raysIn, results, n: job.rays.length };
  });
  await device.queue.onSubmittedWorkDone();

  // ---- Warm-up (untimed): every kernel, round after round, until warmupMs of steady GPU work has passed ----
  let warmupRounds = 0;
  const warmupStart = performance.now();
  while (performance.now() - warmupStart < warmupMs) {
    const encoder = device.createCommandEncoder({ label: 'timing warm-up' });
    for (const job of prepared) job.tracer.encode(encoder, gpuScene, { rays: job.raysIn, results: job.results, count: job.n });
    device.queue.submit([encoder.finish()]);
    warmupRounds++;
    // Keep at most ~2 rounds queued so the GPU never idles and the CPU loop sees real time pass.
    if (warmupRounds % 2 === 0) await device.queue.onSubmittedWorkDone();
  }

  // ---- Timed region: GPU work only, no readback, no CPU writes. Kernels interleave per repeat, so a clock change
  // during the region hits every kernel alike instead of whichever ran first. ----
  for (let r = 0; r < repeats; r++) {
    const encoder = device.createCommandEncoder({ label: `timing round ${r}` });
    for (const [j, job] of prepared.entries()) {
      const slot = (j * repeats + r) * 2;
      job.tracer.encode(encoder, gpuScene, { rays: job.raysIn, results: job.results, count: job.n, timestampWrites: { querySet, beginningOfPassWriteIndex: slot, endOfPassWriteIndex: slot + 1 } });
    }
    device.queue.submit([encoder.finish()]);
  }
  {
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, jobs.length * repeats * 2, resolve, 0);
    device.queue.submit([encoder.finish()]);
  }
  const stamps = new BigUint64Array(await readBack(device, resolve));

  const metrics: Record<string, number> = {};
  const passMs: Record<string, number[]> = {};
  for (const [j, job] of prepared.entries()) {
    const ms: number[] = [];
    for (let r = 0; r < repeats; r++) {
      const slot = (j * repeats + r) * 2;
      ms.push(Number((stamps[slot + 1] as bigint) - (stamps[slot] as bigint)) / 1e6);
    }
    passMs[job.name] = ms;
    const sorted = [...ms].sort((a, b) => a - b);
    const p10 = sorted[Math.floor(0.1 * (sorted.length - 1))] as number;
    const p90 = sorted[Math.floor(0.9 * (sorted.length - 1))] as number;
    // Headline = 10th percentile, not the median. On the shared Mac other apps' GPU work (the Claude app, a Chrome
    // window) lands inside our passes in bursts: the same pass flips between ~9 and ~4 ms, together across kernels
    // (2026-11-14). The fast passes are the uncontended cost; the median mostly measures the interference.
    metrics[`${job.name}_ms`] = p10;
    metrics[`${job.name}_median_ms`] = median(ms);
    metrics[`${job.name}_min_ms`] = sorted[0] as number;
    // Spread inside the run: near 1 means no interference; ~2 or more means contended passes.
    metrics[`${job.name}_p90_over_p10`] = p90 / p10;
    metrics[`${job.name}_mrays_per_s`] = job.n / (p10 * 1e-3) / 1e6;
    metrics[`${job.name}_rays`] = job.n;
  }
  await drainDevice(device);
  for (const job of prepared) {
    if (Array.isArray(job.raysIn)) job.raysIn.forEach((buffer) => buffer.destroy());
    else (job.raysIn as GPUBuffer).destroy();
    job.results.destroy();
  }

  reportResult(
    {
      scene: sceneName,
      renderer: 'rt-batch',
      config: { count, workgroupSize, rayLayout, hitRecord, repeats, warmupMs, format, aoDistance },
      metrics,
      // The runner's load check (load-check.ts) marks a run valid=0 `loaded` when the machine was busy.
      valid: true,
      notes: {
        passMs,
        warmupRounds,
        bvhBuildWorkers: buildersUsedWorkers,
        hitRate: points.length / count,
        triangles: data.meshes.reduce((sum, mesh) => sum + mesh.triangleCount, 0),
      },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
