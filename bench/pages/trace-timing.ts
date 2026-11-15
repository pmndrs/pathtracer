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
 *
 * Two timing methods, both in one run (`timing=passes|batch|both`, default both), so they can be A/B'd:
 *  - passes (the old one): after an untimed warm-up of `warmupMs`, `repeats` rounds, each job in its own pass with GPU
 *    timestamps; the 10th-percentile pass gives `<job>_ms`, `_mrays_per_s`, spread `_p90_over_p10`;
 *  - batch (P1.6a): per job, N back-to-back passes of the same dispatch in ONE command buffer, N calibrated so a sample
 *    lasts >= `batchSeconds` (0.5), timestamps around the whole batch, per-pass ms = batch / N. `samples` (even, 6)
 *    samples per job: `_batch_ms` (median), `_batch_min_ms`, `_batch_max_ms`, spread `_batch_spread` (max/min),
 *    `_batch_mrays_per_s`. The clock stays up inside a long batch, which 4 ms single passes never get.
 * `pageAb=workgroupSize=64,128` (or rayLayout=aos,soa / hitRecord=full,compact) runs both variants inside the page,
 * interleaved ABBA; metrics are then prefixed `A_` / `B_` and `ratio_<job>_batch` (+ `_min`/`_max`, paired per sample)
 * and `ratio_<job>_p10` give B/A throughput. Repeated dispatches write the same outputs: results are compared with a
 * single-pass snapshot after the timed region (`results_identical`, the run is invalid otherwise). One readback of the
 * resolved timestamps per method after its region; no mapAsync inside any timed region.
 *
 * Args: scene=sponza-white rays=1048576 workgroupSize=64 rayLayout=aos hitRecord=full repeats=30 warmupMs=2000
 *       timing=both batchSeconds=0.5 samples=6 pageAb=<key>=<A>,<B>
 * A/B a setting across page runs with the runner: `--ab workgroupSize=64,128`; or in-page with `--args pageAb=...`.
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
  // ---- Variants: one (the page args), or two when `pageAb=<key>=<A>,<B>` asks for an in-page A/B ----
  // The A/B runs inside one page and one GPU session, interleaved ABBA, so a clock change hits both alike (P1.6a).
  interface Variant {
    name: string;
    workgroupSize: BatchWorkgroupSize;
    rayLayout: RayLayout;
    hitRecord: 'full' | 'compact';
  }
  const base: Variant = { name: 'A', workgroupSize, rayLayout, hitRecord };
  let variants: Variant[] = [base];
  let pageAb: string | null = null;
  if (args.pageAb) {
    const eq = args.pageAb.indexOf('=');
    const key = args.pageAb.slice(0, eq);
    const values = args.pageAb.slice(eq + 1).split(',');
    if (!['workgroupSize', 'rayLayout', 'hitRecord'].includes(key) || values.length !== 2) throw new Error(`pageAb must be <workgroupSize|rayLayout|hitRecord>=<A>,<B>, got "${args.pageAb}"`);
    pageAb = args.pageAb;
    variants = values.map((value, i) => ({ ...base, name: i === 0 ? 'A' : 'B', [key]: key === 'workgroupSize' ? (Number(value) as BatchWorkgroupSize) : value }) as Variant);
    for (const v of variants) if (!BATCH_WORKGROUP_SIZES.includes(v.workgroupSize)) throw new Error(`workgroupSize must be one of ${BATCH_WORKGROUP_SIZES.join(', ')}`);
  }
  const timingMode = args.timing ?? 'both'; // passes (old p10-of-30), batch (new), both
  if (!['passes', 'batch', 'both'].includes(timingMode)) throw new Error('timing must be passes, batch or both');
  const batchSeconds = numberArg(args, 'batchSeconds', 0.5);
  const samples = numberArg(args, 'samples', 6);
  if (samples % 2) throw new Error('samples must be even (ABBA pairs)');
  const tracers = await Promise.all(
    variants.map(async (v) => {
      const common = { ...format, workgroupSize: v.workgroupSize, rayLayout: v.rayLayout, hitRecord: v.hitRecord };
      return {
        closest: await RtBatchTracer.create(device, caps, { ...common, mode: 'closest', label: `timing closest ${v.name}` }),
        visible: await RtBatchTracer.create(device, caps, { ...common, mode: 'visible', label: `timing visible ${v.name}` }),
      };
    }),
  );
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
    kind: 'closest' | 'visible';
    rays: Ray[];
  }
  const jobs: Job[] = ([
    { name: 'l1_primary', kind: 'closest', rays: primaryRays },
    { name: 'l1_bounce', kind: 'closest', rays: bounceRays },
    { name: 'l0_shadow', kind: 'visible', rays: shadow },
    { name: 'l0_ao', kind: 'visible', rays: aoRays },
  ] as Job[]).filter((job) => job.rays.length > 0);

  // prepared[v][j]: variant v's tracer, ray buffers and results for job j.
  const prepared = variants.map((v, vi) =>
    jobs.map((job) => {
      const tracer = tracers[vi]![job.kind];
      const packed = packRays(job.rays);
      const raysIn: GPUBuffer | [GPUBuffer, GPUBuffer] = v.rayLayout === 'soa' ? (() => { const s = aosToSoa(packed); return [upload(device, s.a), upload(device, s.b)] as [GPUBuffer, GPUBuffer]; })() : upload(device, packed);
      const results = device.createBuffer({ size: job.rays.length * tracer.resultBytes, usage: STORAGE_IO });
      const snapshot = device.createBuffer({ size: results.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      return { ...job, tracer, raysIn, results, snapshot, n: job.rays.length };
    }),
  );
  type Prepared = (typeof prepared)[number][number];
  const encodeOne = (encoder: GPUCommandEncoder, job: Prepared, timestampWrites?: GPUComputePassTimestampWrites): void =>
    job.tracer.encode(encoder, gpuScene, { rays: job.raysIn, results: job.results, count: job.n, timestampWrites });
  await device.queue.onSubmittedWorkDone();

  // ---- Warm-up (untimed): every kernel of every variant, round after round, until warmupMs of steady GPU work has passed ----
  let warmupRounds = 0;
  const warmupStart = performance.now();
  while (performance.now() - warmupStart < warmupMs) {
    const encoder = device.createCommandEncoder({ label: 'timing warm-up' });
    for (const jobsOfVariant of prepared) for (const job of jobsOfVariant) encodeOne(encoder, job);
    device.queue.submit([encoder.finish()]);
    warmupRounds++;
    // Keep at most ~2 rounds queued so the GPU never idles and the CPU loop sees real time pass.
    if (warmupRounds % 2 === 0) await device.queue.onSubmittedWorkDone();
  }
  // Snapshot of one single pass's results per (variant, job): the batches below must reproduce them exactly.
  {
    const encoder = device.createCommandEncoder({ label: 'timing snapshot' });
    for (const jobsOfVariant of prepared) for (const job of jobsOfVariant) {
      encodeOne(encoder, job);
      encoder.copyBufferToBuffer(job.results, 0, job.snapshot, 0, job.results.size);
    }
    device.queue.submit([encoder.finish()]);
  }

  const nv = variants.length;
  const nj = jobs.length;
  const sorted = (xs: number[]) => [...xs].sort((a, b) => a - b);
  const pct = (xs: number[], q: number) => sorted(xs)[Math.floor(q * (xs.length - 1))] as number;
  const passMs: Record<string, number[][]> = {}; // job -> variant -> 30 single-pass times
  const batchMs: Record<string, number[][]> = {}; // job -> variant -> per-pass ms of each batch sample
  const batchCounts: Record<string, number[]> = {}; // job -> variant -> N

  // ---- Method 1 (old): `repeats` rounds, each (variant, job) in its own pass with GPU timestamps; interleaved per round ----
  if (timingMode !== 'batch') {
    const slots = nv * nj * repeats * 2;
    const querySet = device.createQuerySet({ type: 'timestamp', count: slots });
    const resolve = device.createBuffer({ size: slots * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const slotOf = (v: number, j: number, r: number) => ((v * nj + j) * repeats + r) * 2;
    for (let r = 0; r < repeats; r++) {
      const encoder = device.createCommandEncoder({ label: `timing round ${r}` });
      for (let j = 0; j < nj; j++) {
        for (let k = 0; k < nv; k++) {
          const v = r % 2 ? nv - 1 - k : k; // alternate the variant order each round
          const slot = slotOf(v, j, r);
          encodeOne(encoder, prepared[v]![j]!, { querySet, beginningOfPassWriteIndex: slot, endOfPassWriteIndex: slot + 1 });
        }
      }
      device.queue.submit([encoder.finish()]);
    }
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
    device.queue.submit([encoder.finish()]);
    const stamps = new BigUint64Array(await readBack(device, resolve));
    for (let j = 0; j < nj; j++) {
      passMs[jobs[j]!.name] = variants.map((_, v) => {
        const ms: number[] = [];
        for (let r = 0; r < repeats; r++) {
          const slot = slotOf(v, j, r);
          ms.push(Number((stamps[slot + 1] as bigint) - (stamps[slot] as bigint)) / 1e6);
        }
        return ms;
      });
    }
    querySet.destroy();
    resolve.destroy();
  }

  // ---- Method 2 (new, P1.6a): per (variant, job), N back-to-back passes of the same dispatch in ONE command buffer, N from a
  // calibration so a sample lasts >= batchSeconds; timestamps around the whole batch, per-pass ms = batch / N. Samples
  // interleave ABBA (variants alternate order each sample). The GPU clock stays up inside a long batch, which the 4 ms
  // single passes of method 1 never get. No mapAsync until every sample has been submitted. ----
  if (timingMode !== 'passes') {
    // Calibration (also warm-up): grow a batch until it takes >= 100 ms wall-clock, then scale to batchSeconds.
    const counts: number[][] = jobs.map(() => variants.map(() => 1));
    for (let j = 0; j < nj; j++) {
      for (let v = 0; v < nv; v++) {
        let n = 8;
        let perPass = 0;
        for (let guard = 0; guard < 12; guard++) {
          const encoder = device.createCommandEncoder({ label: 'timing calibration' });
          for (let i = 0; i < n; i++) encodeOne(encoder, prepared[v]![j]!);
          const t0 = performance.now();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          const dt = performance.now() - t0;
          perPass = dt / n;
          if (dt >= 100) break;
          n *= 2;
        }
        counts[j]![v] = Math.max(1, Math.ceil((batchSeconds * 1000) / perPass));
      }
    }
    // A and B run the same number of passes per job, so the paired ratio compares equal amounts of work: use A's N for both.
    for (let j = 0; j < nj; j++) for (let v = 0; v < nv; v++) counts[j]![v] = counts[j]![0]!;
    const slots = nv * nj * samples * 2;
    const querySet = device.createQuerySet({ type: 'timestamp', count: slots });
    const resolve = device.createBuffer({ size: slots * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const slotOf = (v: number, j: number, s: number) => ((v * nj + j) * samples + s) * 2;
    for (let s = 0; s < samples; s++) {
      for (let j = 0; j < nj; j++) {
        for (let k = 0; k < nv; k++) {
          const v = s % 2 ? nv - 1 - k : k;
          const job = prepared[v]![j]!;
          const n = counts[j]![v]!;
          const slot = slotOf(v, j, s);
          const encoder = device.createCommandEncoder({ label: `timing batch ${job.name} ${variants[v]!.name} sample ${s}` });
          for (let i = 0; i < n; i++) {
            const writes: GPUComputePassTimestampWrites = { querySet };
            if (i === 0) writes.beginningOfPassWriteIndex = slot;
            if (i === n - 1) writes.endOfPassWriteIndex = slot + 1;
            encodeOne(encoder, job, i === 0 || i === n - 1 ? writes : undefined);
          }
          device.queue.submit([encoder.finish()]);
          // One sample in flight at a time: Dawn/Metal can merge queued submits into one command buffer, and macOS kills a
          // command buffer that runs ~5 s ("GPU Timeout Error", device lost). onSubmittedWorkDone is not a readback.
          await device.queue.onSubmittedWorkDone();
        }
      }
    }
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
    device.queue.submit([encoder.finish()]);
    const stamps = new BigUint64Array(await readBack(device, resolve));
    for (let j = 0; j < nj; j++) {
      batchCounts[jobs[j]!.name] = counts[j]!;
      batchMs[jobs[j]!.name] = variants.map((_, v) => {
        const ms: number[] = [];
        for (let s = 0; s < samples; s++) {
          const slot = slotOf(v, j, s);
          ms.push(Number((stamps[slot + 1] as bigint) - (stamps[slot] as bigint)) / 1e6 / counts[j]![v]!);
        }
        return ms;
      });
    }
    querySet.destroy();
    resolve.destroy();
  }

  // ---- Same work in, same hits out: after all the repeated dispatches, results equal the single-pass snapshot ----
  const resultsIdentical: Record<string, boolean> = {};
  for (let v = 0; v < nv; v++) {
    for (let j = 0; j < nj; j++) {
      const job = prepared[v]![j]!;
      const [after, before] = await Promise.all([readBack(device, job.results), readBack(device, job.snapshot)]);
      const a = new Uint32Array(after);
      const b = new Uint32Array(before);
      let same = a.length === b.length;
      for (let i = 0; same && i < a.length; i++) if (a[i] !== b[i]) same = false;
      resultsIdentical[`${variants[v]!.name}_${job.name}`] = same;
    }
  }

  const metrics: Record<string, number> = {};
  const prefix = (v: number) => (nv > 1 ? `${variants[v]!.name}_` : '');
  for (let j = 0; j < nj; j++) {
    const job = jobs[j]!;
    for (let v = 0; v < nv; v++) {
      const key = `${prefix(v)}${job.name}`;
      const ms = passMs[job.name]?.[v];
      if (ms) {
        const p10 = pct(ms, 0.1);
        const p90 = pct(ms, 0.9);
        // Method 1 headline = 10th percentile, not the median. On the shared Mac other apps' GPU work lands inside our
        // passes in bursts, and the clock drops between short passes: the same pass flips between ~9 and ~4 ms (2026-11-14).
        metrics[`${key}_ms`] = p10;
        metrics[`${key}_median_ms`] = median(ms);
        metrics[`${key}_min_ms`] = sorted(ms)[0] as number;
        metrics[`${key}_p90_over_p10`] = p90 / p10;
        metrics[`${key}_mrays_per_s`] = job.rays.length / (p10 * 1e-3) / 1e6;
      }
      const bm = batchMs[job.name]?.[v];
      if (bm) {
        const lo = Math.min(...bm);
        const hi = Math.max(...bm);
        metrics[`${key}_batch_ms`] = median(bm); // per pass, median of the samples
        metrics[`${key}_batch_min_ms`] = lo;
        metrics[`${key}_batch_max_ms`] = hi;
        metrics[`${key}_batch_spread`] = hi / lo; // max/min of the samples (compare with p90_over_p10)
        metrics[`${key}_batch_mrays_per_s`] = job.rays.length / (median(bm) * 1e-3) / 1e6;
        metrics[`${key}_batch_passes`] = batchCounts[job.name]![v]!;
      }
    }
    metrics[`${job.name}_rays`] = job.rays.length;
    // In-page A/B: B/A throughput ratio (= tA / tB), paired per ABBA pair, median and min..max.
    if (nv === 2) {
      const pm = passMs[job.name];
      if (pm) {
        metrics[`ratio_${job.name}_p10`] = pct(pm[0]!, 0.1) / pct(pm[1]!, 0.1);
      }
      const bm = batchMs[job.name];
      if (bm) {
        const ratios = bm[0]!.map((t, s) => t / (bm[1]![s] as number));
        metrics[`ratio_${job.name}_batch`] = median(ratios);
        metrics[`ratio_${job.name}_batch_min`] = Math.min(...ratios);
        metrics[`ratio_${job.name}_batch_max`] = Math.max(...ratios);
      }
    }
  }
  const allIdentical = Object.values(resultsIdentical).every(Boolean);
  metrics.results_identical = allIdentical ? 1 : 0;
  await drainDevice(device);
  for (const jobsOfVariant of prepared) {
    for (const job of jobsOfVariant) {
      if (Array.isArray(job.raysIn)) job.raysIn.forEach((buffer) => buffer.destroy());
      else (job.raysIn as GPUBuffer).destroy();
      job.results.destroy();
      job.snapshot.destroy();
    }
  }

  reportResult(
    {
      scene: sceneName,
      renderer: 'rt-batch',
      config: { count, workgroupSize, rayLayout, hitRecord, repeats, warmupMs, format, aoDistance, timing: timingMode, pageAb, batchSeconds, samples },
      metrics,
      // The runner's load check (load-check.ts) marks a run valid=0 `loaded` when the machine was busy.
      valid: allIdentical,
      notes: {
        passMs,
        batchMs,
        batchPasses: batchCounts,
        resultsIdentical,
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
