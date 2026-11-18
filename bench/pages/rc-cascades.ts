/**
 * Cascade-shaped tracing workload over Sponza (P5.rc1), the way gi's Split Radiance Cascades would call `RtIntervalTracer`. NOT a GI technique
 * (that is gi's): a minimal sparse 3D radiance-cascades probe grid, enough to measure rays/s for short interval rays from many probes against
 * plain closest-hit, and to show the API end to end (interval rays -> rtShade at the hit -> naive merge).
 *
 * Probes (sparse, as Split RC): camera primary hits on a width x height grid; cascade 0 has one probe per Δ0 cell (the hit pushed out 0.5 Δ0
 * along its normal), cascade n the cell of spacing Δ0 2^n holding its child's probe (so a parent always exists). Cascade n traces
 * (N0 2^n)^2 directions per probe (N0 = `dirs0`, default 4: 16 / 64 / 256 / 1024) over the interval `(e_{n-1}, e_n]`, e_n = t0 growth^n, t0 = 2 Δ0.
 * Spacing x2, directions x4, interval x4 per cascade, the 3D RC scaling of the Split RC paper (probes / 8, rays per probe x 4).
 *
 * Per cascade it times, interleaved ABBA, back to back in one pass (GPU timestamps around >= `sample_ms` of identical dispatches):
 *   probes-hit      RtIntervalTracer, source probes, 24 B hit records: no ray buffer, one origin per workgroup
 *   rays-hit        RtIntervalTracer, source rays: the same rays from a 32 B ray buffer (what gi would write itself)
 *   plain-clamped   RtBatchTracer closest, the same ray buffer with tMin / tMax = the interval (inclusive; no half-open semantics)
 *   plain-full      RtBatchTracer closest, the same origins and directions, full length (what a trace without intervals costs)
 *   probes-shaded   RtIntervalTracer, source probes, rtShade (lambert: one sun/sky light sample + shadow ray) at every hit
 * and the Split RC form: one ray per pixel from the visible surface over (0, e_top], hit / shaded / plain-full.
 *
 * Then (merge=1) a naive merge, cascade N-1 down to 0 (nearest parent probe, 2x2 child directions averaged, a miss takes the upper cascade's
 * radiance, the top cascade's miss takes the sky), and a check against brute force: the mean radiance L_0 over all cascade-0 directions of every
 * `stride`-th probe against unbounded rays (0, inf] from the same probes with 32 x 32 directions through the same shading.
 *
 * Timings are provisional: best effort on a shared machine (the runner records load), Mrays/s = rays / GPU time of the pass.
 *
 * Args: width=1280 (film width, height 9/16 of it)  cascades=4  spacing=0.25 (Δ0, m)  dirs0=4  growth=4  sample_ms=300  rounds=5  merge=1  split=1  timing=1
 */
import { buildTraceScene, createMeshBvhBuilder, uploadTraceScene, type TraceSceneBuffers } from '@pmndrs/bvh';
import {
  RT_T_INF,
  RtBatchTracer,
  RtIntervalTracer,
  bundleDirection,
  buildSceneCopy,
  cascadeEnd,
  detectCaps,
  packRays,
  readHits,
  requestDeviceForCaps,
  sceneFormatOf,
  uploadSceneCopy,
  type Ray,
} from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { fetchSceneGlb, readGlbMeshes } from '../adapters/shared/glb.ts';
import type { TriangleMesh } from '../scenes/geometry.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import { sponzaWhite } from '../scenes/tier2/sponza-white.ts';

const STORAGE_IO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const FAR = 1e30;
const f32 = Math.fround;

function upload(device: GPUDevice, data: ArrayBufferView<ArrayBuffer>, usage = STORAGE_IO): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

function empty(device: GPUDevice, bytes: number): GPUBuffer {
  return device.createBuffer({ size: Math.max(16, bytes), usage: STORAGE_IO });
}

async function readBack(device: GPUDevice, buffer: GPUBuffer, bytes = buffer.size): Promise<ArrayBuffer> {
  const encoder = device.createCommandEncoder();
  const copy = device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  encoder.copyBufferToBuffer(buffer, 0, copy, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await copy.mapAsync(GPUMapMode.READ);
  const data = copy.getMappedRange().slice(0);
  copy.destroy();
  return data;
}

function mergeMeshes(parts: readonly TriangleMesh[]): TriangleMesh {
  let vertices = 0;
  let indices = 0;
  for (const part of parts) {
    vertices += part.positions.length / 3;
    indices += part.indices.length;
  }
  const positions = new Float32Array(vertices * 3);
  const normals = new Float32Array(vertices * 3);
  const out = new Uint32Array(indices);
  let vOffset = 0;
  let iOffset = 0;
  for (const part of parts) {
    positions.set(part.positions, vOffset * 3);
    normals.set(part.normals, vOffset * 3);
    for (let i = 0; i < part.indices.length; i++) out[iOffset + i] = (part.indices[i] as number) + vOffset;
    vOffset += part.positions.length / 3;
    iOffset += part.indices.length;
  }
  return { positions, normals, indices: out };
}

type V3 = [number, number, number];

function cameraGrid(width: number, height: number): Ray[] {
  const camera = sponzaWhite.camera as unknown as { position: V3; target: V3; up: V3; yFovDeg: number };
  const f = [0, 1, 2].map((k) => camera.target[k]! - camera.position[k]!);
  const fl = Math.hypot(...f);
  const fw = f.map((x) => x / fl);
  const r = [fw[1]! * camera.up[2]! - fw[2]! * camera.up[1]!, fw[2]! * camera.up[0]! - fw[0]! * camera.up[2]!, fw[0]! * camera.up[1]! - fw[1]! * camera.up[0]!];
  const rl = Math.hypot(...r);
  const right = r.map((x) => x / rl);
  const up = [right[1]! * fw[2]! - right[2]! * fw[1]!, right[2]! * fw[0]! - right[0]! * fw[2]!, right[0]! * fw[1]! - right[1]! * fw[0]!];
  const tanY = Math.tan((camera.yFovDeg * Math.PI) / 360);
  const rays: Ray[] = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const px = ((x + 0.5) / width) * 2 - 1;
      const py = 1 - ((y + 0.5) / height) * 2;
      const d = [0, 1, 2].map((k) => fw[k]! + right[k]! * px * tanY * (width / height) + up[k]! * py * tanY);
      const l = Math.hypot(...d);
      rays.push({ origin: [...camera.position], tMin: 0, direction: d.map((v) => v / l) as V3, tMax: FAR });
    }
  }
  return rays;
}

function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Cascade {
  n: number;
  spacing: number;
  /** Directions per axis; D = dirs^2 per probe. */
  dirs: number;
  t0: number;
  t1: number;
  probes: Float32Array<ArrayBuffer>;
  count: number;
  /** Index of each probe's nearest parent in cascade n + 1 (empty for the top cascade). */
  parent: Uint32Array<ArrayBuffer>;
}

const cellKey = (c: number[]): number => ((c[0]! + 2048) * 4096 + (c[1]! + 2048)) * 4096 + (c[2]! + 2048);

function buildCascades(points: { position: V3; normal: V3 }[], count: number, spacing0: number, dirs0: number, growth: number): Cascade[] {
  const cascades: Cascade[] = [];
  const t0 = f32(2 * spacing0);
  // cascade 0: one probe per spacing cell of the hit pushed out of the surface
  let positions: V3[] = [];
  {
    const seen = new Map<number, number>();
    for (const p of points) {
      const q = [0, 1, 2].map((k) => p.position[k]! + p.normal[k]! * 0.5 * spacing0) as V3;
      const key = cellKey(q.map((v) => Math.floor(v / spacing0)));
      if (!seen.has(key)) {
        seen.set(key, positions.length);
        positions.push(q.map(f32) as V3);
      }
    }
  }
  for (let n = 0; n < count; n++) {
    const spacing = spacing0 * 2 ** n;
    const probes = new Float32Array(positions.length * 4);
    positions.forEach((p, i) => probes.set([p[0], p[1], p[2], 1], i * 4));
    const cascade: Cascade = { n, spacing, dirs: dirs0 * 2 ** n, t0: n === 0 ? 0 : cascadeEnd(n - 1, t0, growth), t1: cascadeEnd(n, t0, growth), probes, count: positions.length, parent: new Uint32Array(0) };
    cascades.push(cascade);
    if (n + 1 < count) {
      const nextSpacing = spacing * 2;
      const seen = new Map<number, number>();
      const next: V3[] = [];
      const parent = new Uint32Array(positions.length);
      positions.forEach((p, i) => {
        const cell = p.map((v) => Math.floor(v / nextSpacing));
        const key = cellKey(cell);
        let index = seen.get(key);
        if (index === undefined) {
          index = next.length;
          seen.set(key, index);
          next.push(cell.map((c) => f32((c + 0.5) * nextSpacing)) as V3);
        }
        parent[i] = index;
      });
      cascade.parent = parent;
      positions = next;
    }
  }
  return cascades;
}

/** Rays of a cascade as a 32 B ray buffer: probe-major, cell-minor, the twin of the probe kernel's directions. */
function cascadeRays(c: Cascade, intervalOf: (c: Cascade) => [number, number]): Float32Array<ArrayBuffer> {
  const D = c.dirs * c.dirs;
  const [tMin, tMax] = intervalOf(c);
  const out = new Float32Array(c.count * D * 8);
  for (let p = 0; p < c.count; p++) {
    const o = [c.probes[p * 4] as number, c.probes[p * 4 + 1] as number, c.probes[p * 4 + 2] as number];
    for (let d = 0; d < D; d++) {
      const dir = bundleDirection(d % c.dirs, Math.floor(d / c.dirs), c.dirs);
      out.set([o[0]!, o[1]!, o[2]!, tMin, dir[0], dir[1], dir[2], tMax], (p * D + d) * 8);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Naive merge (the technique is gi's; this is the smallest thing that shows the data flow and checks the intervals)
// ---------------------------------------------------------------------------------------------------------------

const MERGE_WGSL = /* wgsl */ `
struct MergeParams { probes: u32, dirs: u32, parentDirs: u32, top: u32, sky: vec4f }
@group(0) @binding(0) var<uniform> mp: MergeParams;
@group(0) @binding(1) var<storage, read> shaded: array<vec4f>;
@group(0) @binding(2) var<storage, read> parents: array<u32>;
@group(0) @binding(3) var<storage, read> upper: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> merged: array<vec4f>;

@compute @workgroup_size(64)
fn mergeMain(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let i = id.y * groups.x * 64u + id.x;
    let d = mp.dirs * mp.dirs;
    if (i >= mp.probes * d) { return; }
    let r = shaded[i];
    var l = r.xyz;
    if (r.w < 0.0) {
        if (mp.top != 0u) {
            l = mp.sky.xyz;
        } else {
            let p = i / d;
            let c = i - p * d;
            let x = c % mp.dirs;
            let y = c / mp.dirs;
            let base = parents[p] * mp.parentDirs * mp.parentDirs;
            if (mp.parentDirs == mp.dirs) {
                l = upper[base + c].xyz;
            } else {
                var sum = vec3f(0.0);
                for (var k = 0u; k < 4u; k++) {
                    sum += upper[base + (2u * y + (k >> 1u)) * mp.parentDirs + 2u * x + (k & 1u)].xyz;
                }
                l = sum * 0.25;
            }
        }
    }
    merged[i] = vec4f(l, 1.0);
}
`;

// ---------------------------------------------------------------------------------------------------------------

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const metrics: Record<string, number> = {};
const notes: Record<string, unknown> = {};

try {
  const args = getArgs();
  const width = numberArg(args, 'width', 1280);
  const height = Math.round((width * 9) / 16);
  const cascadeCount = numberArg(args, 'cascades', 4);
  const spacing0 = numberArg(args, 'spacing', 0.25);
  const dirs0 = numberArg(args, 'dirs0', 4);
  const growth = numberArg(args, 'growth', 4);
  const sampleMs = numberArg(args, 'sample_ms', 300);
  const rounds = numberArg(args, 'rounds', 6);
  const doMerge = numberArg(args, 'merge', 1) !== 0;
  const doSplit = numberArg(args, 'split', 1) !== 0;
  const doTiming = numberArg(args, 'timing', 1) !== 0;
  if (rounds % 2) throw new Error('rounds must be even (ABBA)');

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'rc-cascades' });
  deviceErrors = watchDevice(device);
  if (doTiming && !caps.timestampQuery) throw new Error('rc-cascades needs timestamp-query for timing (no wall-clock fallback: it would time the CPU too); pass timing=0.');

  // ---- scene: Sponza from the web export, one merged mesh, the scene copy for rtShade ----
  const t00 = performance.now();
  const stage = (what: string) => console.log(`[rc-cascades] ${what} at ${((performance.now() - t00) / 1000).toFixed(1)} s`);
  const { glb } = await fetchSceneGlb(sponzaWhite);
  const merged = mergeMeshes(readGlbMeshes(glb));
  const input = sceneDescToCopyInput(sponzaWhite, { resolveFile: () => merged as never });
  for (const instance of input.instances) instance.objectToWorld = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const builder = createMeshBvhBuilder();
  const data = await buildTraceScene(
    input.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : null })),
    input.instances.map((instance) => ({ ...instance, static: true })),
    { build: (geometry) => builder.build(geometry), mergeStaticInstances: true },
  );
  builder.dispose();
  const trace: TraceSceneBuffers = uploadTraceScene(device, data, 'sponza');
  const copy = buildSceneCopy({ ...input, options: { ...input.options, surface: { ...input.options?.surface, positions: true } } });
  const sceneCopy = uploadSceneCopy(device, copy, { label: 'sponza scene' });
  const format = sceneFormatOf(trace);
  stage(`scene ready (${merged.indices.length / 3} triangles)`);

  // ---- tracers ----
  const sky = (sponzaWhite.environment as unknown as { radiance: V3 }).radiance;
  const make = (o: Parameters<typeof RtIntervalTracer.create>[2]) => RtIntervalTracer.create(device, caps, { ...format, ...o, label: `${o.source}/${o.output}${o.hitRecord === 'compact' ? '/compact' : ''}` });
  const probesHit = await make({ source: 'probes', output: 'hit', hitRecord: 'compact' });
  const raysHit = await make({ source: 'rays', output: 'hit', hitRecord: 'compact' });
  const probesShaded = await make({ source: 'probes', output: 'shaded', scene: { layout: copy.layout } });
  const raysShaded = await make({ source: 'rays', output: 'shaded', scene: { layout: copy.layout } });
  const plain = await RtBatchTracer.create(device, caps, { ...format, mode: 'closest', hitRecord: 'compact', label: 'plain closest compact' });
  const setup = await RtBatchTracer.create(device, caps, { ...format, mode: 'closest', label: 'setup closest' });
  metrics.kernel_bytes_probes_hit = probesHit.kernelBytes;
  metrics.kernel_bytes_rays_hit = raysHit.kernelBytes;
  metrics.kernel_bytes_probes_shaded = probesShaded.kernelBytes;
  stage('kernels compiled');

  // ---- primary hits (setup, untimed) ----
  const primary = cameraGrid(width, height);
  const pixels = primary.length;
  const primaryBuffer = upload(device, packRays(primary));
  const primaryResults = empty(device, pixels * setup.resultBytes);
  {
    const encoder = device.createCommandEncoder();
    setup.encode(encoder, trace, { rays: primaryBuffer, results: primaryResults, count: pixels });
    device.queue.submit([encoder.finish()]);
  }
  const primaryHits = readHits(await readBack(device, primaryResults), pixels);
  const points: { position: V3; normal: V3; ray: number }[] = [];
  primaryHits.forEach((hit, i) => {
    if (hit.t < 0) return;
    const ray = primary[i] as Ray;
    const flip = hit.normal[0] * ray.direction[0] + hit.normal[1] * ray.direction[1] + hit.normal[2] * ray.direction[2] > 0 ? -1 : 1;
    points.push({ position: [0, 1, 2].map((k) => ray.origin[k]! + ray.direction[k]! * hit.t) as V3, normal: hit.normal.map((v) => v * flip) as V3, ray: i });
  });
  metrics.pixels = pixels;
  metrics.visible_pixels = points.length;
  const cascades = buildCascades(points, cascadeCount, spacing0, dirs0, growth);
  const fmt = (c: Cascade) => `c${c.n}: ${c.count} probes x ${c.dirs * c.dirs} dirs = ${c.count * c.dirs * c.dirs} rays, (${c.t0}, ${c.t1}]`;
  cascades.forEach((c) => {
    metrics[`c${c.n}_probes`] = c.count;
    metrics[`c${c.n}_dirs`] = c.dirs * c.dirs;
    metrics[`c${c.n}_rays`] = c.count * c.dirs * c.dirs;
    metrics[`c${c.n}_t0`] = c.t0;
    metrics[`c${c.n}_t1`] = c.t1;
  });
  notes.cascades = cascades.map(fmt);
  metrics.total_rays = cascades.reduce((sum, c) => sum + c.count * c.dirs * c.dirs, 0);
  stage(`${points.length} visible pixels; ${notes.cascades as string[]}`);

  // ---- GPU buffers per cascade ----
  interface Gpu {
    c: Cascade;
    probes: GPUBuffer;
    intervalRays: GPUBuffer;
    fullRays: GPUBuffer;
    hitOut: GPUBuffer;
    plainOut: GPUBuffer;
    shadedOut: GPUBuffer;
    rays: number;
  }
  const gpus: Gpu[] = cascades.map((c) => {
    const rays = c.count * c.dirs * c.dirs;
    return {
      c,
      probes: upload(device, c.probes),
      intervalRays: upload(device, cascadeRays(c, (k) => [k.t0, k.t1])),
      fullRays: upload(device, cascadeRays(c, () => [0, FAR])),
      hitOut: empty(device, rays * 24),
      plainOut: empty(device, rays * 24),
      shadedOut: empty(device, rays * 16),
      rays,
    };
  });
  stage('cascade buffers uploaded');

  // ---- timing ----
  interface Variant {
    name: string;
    rays: number;
    /** Records dispatch number `i` of the sample into `pass`. */
    record(pass: GPUComputePassEncoder, i: number): void;
    begin(): void;
  }
  const variants: Variant[] = [];
  for (const g of gpus) {
    const c = g.c;
    const probeJob = (results: GPUBuffer, i: number) => ({ source: g.probes, results, count: c.count, interval: { t0: c.t0, t1: c.t1 }, directions: c.dirs, reuseParams: i > 0 });
    variants.push(
      { name: `c${c.n}.probes-hit`, rays: g.rays, begin: () => probesHit.beginFrame(), record: (pass, i) => probesHit.dispatch(pass, trace, probeJob(g.hitOut, i)) },
      { name: `c${c.n}.rays-hit`, rays: g.rays, begin: () => raysHit.beginFrame(), record: (pass, i) => raysHit.dispatch(pass, trace, { source: g.intervalRays, results: g.hitOut, count: g.rays, reuseParams: i > 0 }) },
      { name: `c${c.n}.plain-clamped`, rays: g.rays, begin: () => {}, record: (pass) => plain.dispatch(pass, trace, { rays: g.intervalRays, results: g.plainOut, count: g.rays }) },
      { name: `c${c.n}.plain-full`, rays: g.rays, begin: () => {}, record: (pass) => plain.dispatch(pass, trace, { rays: g.fullRays, results: g.plainOut, count: g.rays }) },
      { name: `c${c.n}.probes-shaded`, rays: g.rays, begin: () => probesShaded.beginFrame(), record: (pass, i) => probesShaded.dispatch(pass, trace, { ...probeJob(g.shadedOut, i), sceneCopy }) },
    );
  }
  // Split RC form: one cosine-free hemisphere ray per visible pixel from its surface point, over (0, e_top]
  let splitBuffers: { intervalRays: GPUBuffer; fullRays: GPUBuffer; hitOut: GPUBuffer; plainOut: GPUBuffer; shadedOut: GPUBuffer; n: number } | null = null;
  if (doSplit) {
    const top = (cascades[cascades.length - 1] as Cascade).t1;
    const origin = (p: { position: V3; normal: V3 }) => p.position.map((v, k) => v + p.normal[k]! * 1e-3) as V3;
    const mk = (tMin: number, tMax: number) => {
      const r = random(77);
      return packRays(
        points.map((p) => {
          // direction in the hemisphere of the normal: uniform sphere, flipped inward-pointing ones (Split RC algorithm 3)
          const z = 2 * r() - 1;
          const phi = 2 * Math.PI * r();
          const s = Math.sqrt(1 - z * z);
          const d: V3 = [s * Math.cos(phi), s * Math.sin(phi), z];
          const sign = d[0] * p.normal[0] + d[1] * p.normal[1] + d[2] * p.normal[2] < 0 ? -1 : 1;
          return { origin: origin(p), tMin, direction: d.map((v) => v * sign) as V3, tMax };
        }),
      );
    };
    const n = points.length;
    splitBuffers = { intervalRays: upload(device, mk(0, top)), fullRays: upload(device, mk(0, FAR)), hitOut: empty(device, n * 24), plainOut: empty(device, n * 24), shadedOut: empty(device, n * 16), n };
    const s = splitBuffers;
    variants.push(
      { name: 'split.rays-hit', rays: n, begin: () => raysHit.beginFrame(), record: (pass, i) => raysHit.dispatch(pass, trace, { source: s.intervalRays, results: s.hitOut, count: n, reuseParams: i > 0 }) },
      { name: 'split.rays-shaded', rays: n, begin: () => raysShaded.beginFrame(), record: (pass, i) => raysShaded.dispatch(pass, trace, { source: s.intervalRays, results: s.shadedOut, count: n, sceneCopy, reuseParams: i > 0 }) },
      { name: 'split.plain-clamped', rays: n, begin: () => {}, record: (pass) => plain.dispatch(pass, trace, { rays: s.intervalRays, results: s.plainOut, count: n }) },
      { name: 'split.plain-full', rays: n, begin: () => {}, record: (pass) => plain.dispatch(pass, trace, { rays: s.fullRays, results: s.plainOut, count: n }) },
    );
    metrics.split_rays = n;
  }

  if (doTiming) {
    // N per variant from a calibration (grow until a batch takes >= 80 ms wall), then every sample of a variant runs that N.
    const counts = new Map<string, number>();
    const record = (encoder: GPUCommandEncoder, v: Variant, n: number, writes?: GPUComputePassTimestampWrites) => {
      v.begin();
      const pass = encoder.beginComputePass({ label: v.name, timestampWrites: writes });
      for (let i = 0; i < n; i++) v.record(pass, i);
      pass.end();
    };
    for (const v of variants) {
      let n = 2;
      let per = 0;
      for (let guard = 0; guard < 14; guard++) {
        const encoder = device.createCommandEncoder();
        record(encoder, v, n);
        const t0 = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const dt = performance.now() - t0;
        per = dt / n;
        if (dt >= 80) break;
        n *= 2;
      }
      counts.set(v.name, Math.max(1, Math.ceil(sampleMs / per)));
    }
    stage('calibrated');
    const slots = variants.length * rounds * 2;
    const querySet = device.createQuerySet({ type: 'timestamp', count: slots });
    const resolve = device.createBuffer({ size: slots * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const slotOf = (v: number, r: number) => (v * rounds + r) * 2;
    // Group variants by cascade-or-split so one family runs together, interleaving ABBA inside the family.
    const families = [...new Set(variants.map((v) => v.name.split('.')[0] as string))];
    for (const family of families) {
      const members = variants.map((v, i) => ({ v, i })).filter(({ v }) => v.name.startsWith(`${family}.`));
      for (let r = 0; r < rounds; r++) {
        for (let k = 0; k < members.length; k++) {
          const { v, i } = members[r % 2 ? members.length - 1 - k : k] as { v: Variant; i: number };
          const encoder = device.createCommandEncoder();
          record(encoder, v, counts.get(v.name) as number, { querySet, beginningOfPassWriteIndex: slotOf(i, r), endOfPassWriteIndex: slotOf(i, r) + 1 });
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
        }
      }
    }
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
    device.queue.submit([encoder.finish()]);
    const stamps = new BigUint64Array(await readBack(device, resolve));
    const table: Record<string, { mrays: number[]; ms_per_dispatch: number[] }> = {};
    variants.forEach((v, i) => {
      const n = counts.get(v.name) as number;
      const mrays: number[] = [];
      const ms: number[] = [];
      for (let r = 0; r < rounds; r++) {
        const dt = Number((stamps[slotOf(i, r) + 1] as bigint) - (stamps[slotOf(i, r)] as bigint)) / 1e6;
        ms.push(dt / n);
        mrays.push((v.rays * n) / (dt / 1000) / 1e6);
      }
      const sorted = [...mrays].sort((a, b) => a - b);
      table[v.name] = { mrays, ms_per_dispatch: ms };
      metrics[`${v.name}.mrays_median`] = sorted[Math.floor(sorted.length / 2)] as number;
      metrics[`${v.name}.mrays_min`] = sorted[0] as number;
      metrics[`${v.name}.mrays_max`] = sorted[sorted.length - 1] as number;
      metrics[`${v.name}.ms_median`] = [...ms].sort((a, b) => a - b)[Math.floor(ms.length / 2)] as number;
      metrics[`${v.name}.dispatches`] = n;
    });
    notes.timing = table;
    // one frame's worth: the four cascades through probes-shaded, median ms per dispatch summed
    metrics.frame_probes_shaded_ms = cascades.reduce((sum, c) => sum + (metrics[`c${c.n}.probes-shaded.ms_median`] as number), 0);
    metrics.frame_probes_hit_ms = cascades.reduce((sum, c) => sum + (metrics[`c${c.n}.probes-hit.ms_median`] as number), 0);
    metrics.frame_plain_full_ms = cascades.reduce((sum, c) => sum + (metrics[`c${c.n}.plain-full.ms_median`] as number), 0);
    metrics.frame_mrays_probes_hit = metrics.total_rays! / metrics.frame_probes_hit_ms! / 1000;
    querySet.destroy();
    resolve.destroy();
    stage('timed');
  }

  // ---- naive merge + brute-force check ----
  if (doMerge) {
    const merge = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: MERGE_WGSL }), entryPoint: 'mergeMain' } });

    /** Shades every cascade (one seed), merges top-down, returns cascade 0's merged radiance (`vec4f` per ray, probe-major). */
    const runMerge = async (set: Cascade[], seed: number, sameSeed = false): Promise<Float32Array> => {
      const local = set.map((c) => {
        const rays = c.count * c.dirs * c.dirs;
        return { c, rays, probes: upload(device, c.probes), shaded: empty(device, rays * 16), merged: empty(device, rays * 16) };
      });
      probesShaded.beginFrame();
      {
        const encoder = device.createCommandEncoder();
        for (const g of local) probesShaded.encode(encoder, trace, { source: g.probes, results: g.shaded, count: g.c.count, interval: { t0: g.c.t0, t1: g.c.t1 }, directions: g.c.dirs, seed: sameSeed ? seed : seed + g.c.n, sceneCopy });
        device.queue.submit([encoder.finish()]);
      }
      const temp: GPUBuffer[] = [];
      for (let n = set.length - 1; n >= 0; n--) {
        const g = local[n] as (typeof local)[number];
        const c = g.c;
        const top = n === set.length - 1;
        const params = upload(device, new Uint32Array(new Float32Array([0, 0, 0, 0, sky[0], sky[1], sky[2], 0]).buffer), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(params, 0, new Uint32Array([c.count, c.dirs, top ? c.dirs : (set[n + 1] as Cascade).dirs, top ? 1 : 0]));
        const parents = upload(device, top ? new Uint32Array(4) : c.parent);
        const upper = top ? empty(device, 16) : (local[n + 1] as (typeof local)[number]).merged;
        temp.push(params, parents, ...(top ? [upper] : []));
        const group = device.createBindGroup({
          layout: merge.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: g.shaded } },
            { binding: 2, resource: { buffer: parents } },
            { binding: 3, resource: { buffer: upper } },
            { binding: 4, resource: { buffer: g.merged } },
          ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(merge);
        pass.setBindGroup(0, group);
        const groups = Math.ceil(g.rays / 64);
        pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
        pass.end();
        device.queue.submit([encoder.finish()]);
      }
      const first = local[0] as (typeof local)[number];
      const out = new Float32Array(await readBack(device, first.merged, first.rays * 16));
      for (const g of local) for (const buffer of [g.probes, g.shaded, g.merged]) buffer.destroy();
      for (const buffer of temp) buffer.destroy();
      return out;
    };

    /** Mean radiance (channel average) over cascade 0's directions per probe of `l0`, against unbounded rays (0, inf] with 32 x 32 directions through the same shading. */
    const against = async (l0: Float32Array, c0: Cascade, indices: number[], bruteDirs = 32, bruteSeed = 99, bruteJitter: [number, number] = [0.31, 0.77]) => {
      const D0 = c0.dirs * c0.dirs;
      const subProbes = new Float32Array(indices.length * 4);
      indices.forEach((p, i) => subProbes.set(c0.probes.subarray(p * 4, p * 4 + 4), i * 4));
      const nb = bruteDirs * bruteDirs;
      const bruteProbes = upload(device, subProbes);
      const bruteOut = empty(device, indices.length * nb * 16);
      probesShaded.beginFrame();
      const encoder = device.createCommandEncoder();
      probesShaded.encode(encoder, trace, { source: bruteProbes, results: bruteOut, count: indices.length, interval: { t0: 0, t1: RT_T_INF }, directions: bruteDirs, seed: bruteSeed, jitter: bruteJitter, sceneCopy });
      device.queue.submit([encoder.finish()]);
      const brute = new Float32Array(await readBack(device, bruteOut));
      bruteProbes.destroy();
      bruteOut.destroy();
      const skyMean = (sky[0] + sky[1] + sky[2]) / 3;
      const meanRc: number[] = [];
      const meanBrute: number[] = [];
      indices.forEach((p, i) => {
        let a = 0;
        for (let d = 0; d < D0; d++) a += ((l0[(p * D0 + d) * 4] as number) + (l0[(p * D0 + d) * 4 + 1] as number) + (l0[(p * D0 + d) * 4 + 2] as number)) / 3;
        meanRc.push(a / D0);
        let b = 0;
        for (let d = 0; d < nb; d++) {
          const o = (i * nb + d) * 4;
          b += (brute[o + 3] as number) < 0 ? skyMean : ((brute[o] as number) + (brute[o + 1] as number) + (brute[o + 2] as number)) / 3;
        }
        meanBrute.push(b / nb);
      });
      const sumRc = meanRc.reduce((x, y) => x + y, 0);
      const sumBrute = meanBrute.reduce((x, y) => x + y, 0);
      const rel = meanRc.map((x, i) => Math.abs(x - (meanBrute[i] as number)) / Math.max(meanBrute[i] as number, 1e-3)).sort((x, y) => x - y);
      return { raw: brute, probes: indices.length, rc: sumRc / indices.length, brute: sumBrute / indices.length, ratio: sumRc / sumBrute, p50: rel[Math.floor(rel.length * 0.5)] as number, p90: rel[Math.floor(rel.length * 0.9)] as number };
    };

    // (a) the sparse hierarchy above: nearest parent at the cell centre, so a parent can sit on the far side of a wall (the known naive-RC leak / occlusion bias: expect below 1)
    const c0 = cascades[0] as Cascade;
    const stride = Math.max(1, Math.floor(c0.count / 1024));
    const subset: number[] = [];
    for (let p = 0; p < c0.count; p += stride) subset.push(p);
    const sparse = await against(await runMerge(cascades, 4242), c0, subset);
    for (const [k, v] of Object.entries(sparse)) if (typeof v === 'number') metrics[`merge_sparse_${k}`] = v;

    // (b) co-located cascades: every cascade's probe p sits at cascade 0's probe p, parent = itself. Merging then only averages cones of rays from ONE origin, so the mean radiance
    // must equal the brute-force one (unbiased) up to shading noise: this checks the interval partition end to end (a hit lost or counted twice at a cascade seam shows here).
    const colocatedProbes = new Float32Array(subset.length * 4);
    subset.forEach((p, i) => colocatedProbes.set(c0.probes.subarray(p * 4, p * 4 + 4), i * 4));
    const identity = Uint32Array.from({ length: subset.length }, (_, i) => i);
    const colocated: Cascade[] = cascades.map((c) => ({ ...c, probes: colocatedProbes, count: subset.length, parent: identity }));
    const colocatedL0 = await runMerge(colocated, 777);
    const co = await against(colocatedL0, colocated[0] as Cascade, subset.map((_, i) => i));
    for (const [k, v] of Object.entries(co)) if (typeof v === 'number') metrics[`merge_colocated_${k}`] = v;

    // (c) the exact one: the same co-located probes, but the directions per axis do NOT grow and every cascade shades with one seed, so cascade n's ray for a cell is the very ray of cascade 0 and of the
    // unbounded brute-force run (same ray index, same light sample). Merging is then "first interval with a hit" and must reproduce the unbounded ray's radiance cell by cell: any ray lost or
    // counted twice at a cascade seam, or any interval that reports a hit outside itself, shows as a differing cell.
    const chainDirs = 8;
    const chain: Cascade[] = colocated.map((c) => ({ ...c, dirs: chainDirs }));
    const chainL0 = await runMerge(chain, 31337, true);
    const chainCheck = await against(chainL0, chain[0] as Cascade, subset.map((_, i) => i), chainDirs, 31337, [0.5, 0.5]);
    const brute = chainCheck.raw;
    const skyChannels = [sky[0], sky[1], sky[2]];
    let cells = 0;
    let differing = 0;
    let worst = 0;
    for (let p = 0; p < subset.length; p++) {
      for (let c = 0; c < chainDirs * chainDirs; c++) {
        const o = (p * chainDirs * chainDirs + c) * 4;
        const missed = (brute[o + 3] as number) < 0;
        let bad = false;
        for (let k = 0; k < 3; k++) {
          const want = missed ? (skyChannels[k] as number) : (brute[o + k] as number);
          const got = chainL0[o + k] as number;
          const e = Math.abs(got - want) / Math.max(Math.abs(want), 1e-3);
          worst = Math.max(worst, e);
          if (e > 1e-4) bad = true;
        }
        cells++;
        if (bad) differing++;
      }
    }
    metrics.merge_chain_cells = cells;
    metrics.merge_chain_differing = differing;
    metrics.merge_chain_worst_relative = worst;
    stage('merged and checked');
    // a handful of cells may differ legitimately: a hit within an ulp of a cascade end, or two triangles at one t (ties)
    if (differing > Math.max(2, cells * 1e-3)) throw new Error(`chain merge differs from the unbounded rays in ${differing} of ${cells} cells (worst relative ${worst.toExponential(2)}): an interval seam loses or doubles a hit`);
  }

  await drainDevice(device);
  if (deviceErrors.length) throw new Error(`device errors: ${deviceErrors.join('; ')}`);
  reportResult(
    {
      scene: 'sponza-white',
      renderer: 'rt-interval-cascades',
      config: { width, height, cascades: cascadeCount, spacing: spacing0, dirs0, growth, sampleMs, rounds, caps: { tier: caps.tier, subgroups: caps.subgroups } },
      metrics,
      notes,
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
