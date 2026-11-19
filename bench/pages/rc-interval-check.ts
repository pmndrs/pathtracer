/**
 * Interval queries for gi's Split Radiance Cascades (P5.rc1): `rtClosestInterval` and the batched `RtIntervalTracer`, checked on the GPU.
 *
 * Per scene (edges, cornell, quad-light, sponza-white) and per scene format (quantized + records + merged, f32 + indexed):
 *  1. CASCADE PARTITION. Every ray is cut into six intervals `(e_k, e_{k+1}]` (cascade ends e_n = (diagonal / 64) 4^n, then (4 diagonal, inf]),
 *     once with the paper's `open-low` convention and once with `closed-low`. Against bvh's CPU oracle (`createCpuTracer`, fed the very f32 bounds
 *     the twin `intervalBounds` computes) with the V27 classification of trace-core (ties, grazing, tPosition, watertight slips; a hit within 1e-5
 *     relative of an interval end is counted as `boundary`: f32 on the GPU and f64 on the CPU may land on different sides). On top, exact GPU-only
 *     invariants: every reported t lies inside the interval's f32 bounds, and the interval that owns the unbounded closest hit reports exactly that
 *     hit (bit for bit) while every interval before it reports a miss.
 *  2. BOUNDARY EXACTNESS. For every ray the unbounded closest hit t_h is taken from the plain batch tracer; then intervals cut exactly at t_h, at
 *     its f32 neighbour below and above are traced, in both conventions: (0, t_h] and [0, next(t_h)) hit it, (0, prev(t_h)] and [0, t_h) miss,
 *     (t_h, inf] and (next(t_h), inf] report only something farther, [t_h, inf) and (prev(t_h), inf] report it again, bit for bit.
 *  3. OUTPUTS. `visible` and `distance` outputs agree with `hit`; an indirect tracer (count in a GPU buffer, clamped to the capacity) equals the
 *     direct one bit for bit.
 *  4. PROBE BUNDLES. The `probes` source (position + scale, one equal-area octahedral direction grid per probe) against the `rays` source fed with
 *     the TypeScript twin of the same rays: same hit (instance, triangle, t within 1e-4 relative) on all but a tiny fraction (GPU sin / cos / sqrt are
 *     a few ulp off the doubles, which can flip an edge ray), scale 0 probes skipped, indirect probes bit-identical to direct.
 *  5. WGSL HELPERS. `rtCascadeIndex` / `rtCascadeEnd` / `rtCascadeStart` exactly, `rtOctEqualAreaDecode` against its twin.
 *  6. SHADED OUTPUT. `output: 'shaded'` (one `vec4f` per ray, radiance + t) against an inline reference kernel in this page (`rtClosestInterval` +
 *     `rtShade` with the same hash) on Cornell and quad-light, and the t channel against `distance`.
 *
 * Args: scenes=edges,cornell,quad-light,sponza-white  rays=2048 (per ray set)  shaded=1  packing=split|packed
 */
import { CpuSceneBvh, buildMeshBvhWithTinyBvh, buildTraceScene, createCpuTracer, createMeshBvhBuilder, createTinyBvhBuilder, uploadTraceScene, type CpuTracer, type MeshBvhBuilder, type TinyBvhBuilder, type TraceSceneBuffers, type TraceSceneData } from '@pmndrs/bvh';
import {
  DEFAULT_SCENE_PACKING,
  RT_T_INF,
  RtBatchTracer,
  RtIntervalTracer,
  bundleDirection,
  buildSceneCopy,
  cascadeEnd,
  cascadeIndex,
  cascadeStart,
  detectCaps,
  inInterval,
  intervalBounds,
  nextDownF32,
  nextUpF32,
  octEqualAreaDecode,
  packRays,
  readHits,
  requestDeviceForCaps,
  rtMinifyWgsl,
  rtModuleWgsl,
  rtPackSceneAccess,
  rtSceneBindGroupEntries,
  rtSceneCopyBindGroupEntries,
  rtSceneCopyLayoutEntries,
  rtWalkEntries,
  sceneFormatOf,
  uploadSceneCopy,
  withBindings,
  type Caps,
  type Hit,
  type IntervalConvention,
  type Ray,
  type SceneCopyPacking,
  type ResolvedSceneFormat,
} from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import { cornell, quadLight } from '../scenes/tier0/index.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { T_FLOOR, compareClosest, makeEdgeIndex, type AllHits, type EdgeIndex } from './trace-compare.ts';
import { cameraRays, createRandom, edgeRays, incoherentRays, insideRays, loadTraceScene, type TraceTestScene } from './trace-scenes.ts';

const FAR = 1e30;
const ALL = 0xffffffff;
const f32 = Math.fround;
const STORAGE_IO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const BOUNDARY_RELATIVE = 1e-5;

interface Format extends ResolvedSceneFormat {
  name: string;
  merge: boolean;
}
const FORMATS: Format[] = [
  { name: 'quantized-records-merged', nodeFormat: 'quantized', triangleLeaves: 'records', positionLayout: 'vec4', merge: true },
  { name: 'f32-indexed', nodeFormat: 'f32', triangleLeaves: 'indexed', positionLayout: 'vec4', merge: false },
];

const tally = {
  rays: 0,
  intervalRays: 0,
  hits: 0,
  cpuMismatch: 0,
  ties: 0,
  grazing: 0,
  tPosition: 0,
  boundary: 0,
  ownerTies: 0,
  walkLevel: 0,
  plainChecked: 0,
  plainTmaxMiss: 0,
  plainTminMiss: 0,
  visibleNearEnd: 0,
  slips: 0,
  outside: 0,
  partition: 0,
  exactChecked: 0,
  exactFail: 0,
  outputFail: 0,
  probeRays: 0,
  probeDiff: 0,
  probeFail: 0,
  helperFail: 0,
  shadedRays: 0,
  shadedFail: 0,
  maxProbeT: 0,
};
const examples: string[] = [];
const noted = new Map<string, number>();
function note(category: string, text: string): void {
  const n = noted.get(category) ?? 0;
  noted.set(category, n + 1);
  if (n < 4) examples.push(`[${category}] ${text}`);
}

// ---------------------------------------------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------------------------------------------

function upload(device: GPUDevice, data: ArrayBufferView<ArrayBuffer>, usage = STORAGE_IO): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

async function readBack(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const encoder = device.createCommandEncoder();
  const copy = device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  encoder.copyBufferToBuffer(buffer, 0, copy, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await copy.mapAsync(GPUMapMode.READ);
  const data = copy.getMappedRange().slice(0);
  copy.destroy();
  return data;
}

function empty(device: GPUDevice, bytes: number): GPUBuffer {
  return device.createBuffer({ size: Math.max(16, bytes), usage: STORAGE_IO });
}

/** One interval job on the rays source (`tMin` / `tMax` of each packed ray are the interval). Returns the raw result bytes. */
async function runRays(device: GPUDevice, tracer: RtIntervalTracer, scene: TraceSceneBuffers, packed: Float32Array<ArrayBuffer>, mask = ALL, indirect = false): Promise<ArrayBuffer> {
  const n = packed.length / 8;
  const rays = upload(device, packed);
  const results = empty(device, n * tracer.resultBytes);
  const counts = indirect ? upload(device, new Uint32Array([n + 777, n, 0, 0])) : null;
  tracer.beginFrame();
  const encoder = device.createCommandEncoder();
  tracer.encode(encoder, scene, { source: rays, results, count: counts ? { buffer: counts, word: 1, capacity: n } : n, mask });
  device.queue.submit([encoder.finish()]);
  const data = await readBack(device, results, n * tracer.resultBytes);
  for (const buffer of [rays, results, counts]) buffer?.destroy();
  return data;
}

async function runBase(device: GPUDevice, tracer: RtBatchTracer, scene: TraceSceneBuffers, packed: Float32Array<ArrayBuffer>, mask = ALL): Promise<{ hits: Hit[]; words: Uint32Array }> {
  const n = packed.length / 8;
  const rays = upload(device, packed);
  const results = empty(device, n * tracer.resultBytes);
  const encoder = device.createCommandEncoder();
  tracer.encode(encoder, scene, { rays, results, count: n, mask });
  device.queue.submit([encoder.finish()]);
  const data = await readBack(device, results, n * tracer.resultBytes);
  rays.destroy();
  results.destroy();
  return { hits: readHits(data, n, tracer.options.hitRecord), words: new Uint32Array(data) };
}

const HIT_WORDS = 10;
/**
 * The same hit, or a legitimate tie: the same f32 t (bit for bit) on another triangle of the same instance that shares a corner or lies in one
 * plane (the walk's child order depends on the range, so two triangles at one t can swap).
 */
function sameOrTie(a: Hit, b: Hit, edges: EdgeIndex, direction: readonly number[]): 'same' | 'tie' | 'no' {
  if (a.t < 0 || b.t < 0) return a.t < 0 && b.t < 0 ? 'same' : 'no';
  if (Math.fround(a.t) !== Math.fround(b.t)) return 'no';
  if (a.instance === b.instance && a.triangle === b.triangle) return 'same';
  // Two triangles at one t: on an edge or vertex of either (also across instances: two walls meeting), or coplanar / sharing a corner.
  if (edges.kind(a.instance, a.triangle, a.barycentrics[0], a.barycentrics[1], direction) !== 'interior' || edges.kind(b.instance, b.triangle, b.barycentrics[0], b.barycentrics[1], direction) !== 'interior') return 'tie';
  // Coincident faces of two instances (a box standing on the floor): both hits interior, same t, one world plane. Which one a walk names depends on its
  // child order, which the range changes (seen on D3D12 / RTX 3060, Cornell, 2026-11-19).
  if (a.instance !== b.instance) return edges.coplanarWorld(a.instance, a.triangle, b.instance, b.triangle) ? 'tie' : 'no';
  const r = edges.relation(a.instance, a.triangle, b.triangle);
  return r.sharedCorners > 0 || r.coplanar ? 'tie' : 'no';
}
const f32Of = (u: number): number => new Float32Array(new Uint32Array([u]).buffer)[0] as number;

// ---------------------------------------------------------------------------------------------------------------
// Scenes and rays
// ---------------------------------------------------------------------------------------------------------------

async function buildScene(scene: TraceTestScene, format: Format, builder: MeshBvhBuilder): Promise<TraceSceneData> {
  return buildTraceScene(scene.meshes, scene.instances, {
    build: (geometry) => builder.build(geometry, { triangleLeaves: format.triangleLeaves, nodeFormat: format.nodeFormat === 'quantized' ? 'quantized' : 'f32' }),
    mergeStaticInstances: format.merge,
    triangleLeaves: format.triangleLeaves,
    positionLayout: format.positionLayout,
    nodeFormat: format.nodeFormat,
  });
}

function makeAllHits(scene: TraceTestScene, tiny: TinyBvhBuilder): AllHits {
  let bvh: CpuSceneBvh | null = null;
  return (ray, mask) => {
    bvh ??= CpuSceneBvh.build(scene.meshes, scene.instances, (geometry) => buildMeshBvhWithTinyBvh(geometry, tiny));
    return bvh.raycast(ray, { mask }).map((hit) => ({ instance: hit.instance, triangle: hit.triangle, t: hit.t, u: hit.u, v: hit.v, frontFace: hit.frontFace }));
  };
}

interface RaySet {
  name: string;
  rays: Ray[];
}

function raySets(scene: TraceTestScene, perSet: number): RaySet[] {
  const random = createRandom(0x1ec + scene.name.length);
  const sets: RaySet[] = [];
  const add = (name: string, rays: Ray[]) => {
    if (rays.length) sets.push({ name, rays: rays.map((r) => ({ origin: r.origin, direction: r.direction, tMin: 0, tMax: FAR })) });
  };
  add('primary', cameraRays(scene, perSet, random));
  add('incoherent', incoherentRays(scene, perSet, random));
  add('edges', edgeRays(scene, perSet, random));
  if (scene.closed.length) add('inside', insideRays(scene, perSet, random));
  return sets;
}

/** AoS ray records, `count` per source ray: origin, direction and the given (tMin, tMax) bounds. */
function packBounded(rays: readonly Ray[], bounds: (rayIndex: number) => [number, number][]): Float32Array<ArrayBuffer> {
  const out: Ray[] = [];
  rays.forEach((ray, i) => {
    for (const [tMin, tMax] of bounds(i)) out.push({ origin: ray.origin, direction: ray.direction, tMin, tMax });
  });
  return packRays(out);
}

// ---------------------------------------------------------------------------------------------------------------
// 1-3. Cascade partition, boundary exactness, outputs
// ---------------------------------------------------------------------------------------------------------------

interface Tracers {
  base: RtBatchTracer;
  open: RtIntervalTracer;
  lo: RtIntervalTracer;
  openVisible: RtIntervalTracer;
  loVisible: RtIntervalTracer;
  openDistance: RtIntervalTracer;
  openIndirect: RtIntervalTracer;
  probes: RtIntervalTracer;
  probesIndirect: RtIntervalTracer;
}

async function makeTracers(device: GPUDevice, caps: Caps, format: Format): Promise<Tracers> {
  const fmt = { nodeFormat: format.nodeFormat, triangleLeaves: format.triangleLeaves, positionLayout: format.positionLayout };
  const make = (o: Parameters<typeof RtIntervalTracer.create>[2]) => RtIntervalTracer.create(device, caps, { ...fmt, ...o, label: `${format.name} ${o.source}/${o.output}/${o.convention ?? 'open-low'}${o.indirect ? '/indirect' : ''}` });
  return {
    base: await RtBatchTracer.create(device, caps, { ...fmt, mode: 'closest', label: `${format.name} base` }),
    open: await make({ source: 'rays', output: 'hit' }),
    lo: await make({ source: 'rays', output: 'hit', convention: 'closed-low' }),
    openVisible: await make({ source: 'rays', output: 'visible' }),
    loVisible: await make({ source: 'rays', output: 'visible', convention: 'closed-low' }),
    openDistance: await make({ source: 'rays', output: 'distance' }),
    openIndirect: await make({ source: 'rays', output: 'hit', indirect: true }),
    probes: await make({ source: 'probes', output: 'hit' }),
    probesIndirect: await make({ source: 'probes', output: 'hit', indirect: true }),
  };
}

function intervalVerdict(gpu: Hit, cpu: ReturnType<CpuTracer['closest']>, edges: EdgeIndex, tFloor: number, ray: Ray, allHits: () => ReturnType<AllHits>) {
  const verdict = compareClosest(gpu, cpu, edges, tFloor, ray, allHits, false);
  if (verdict.verdict !== 'mismatch') return verdict;
  // f32 (GPU) and f64 (CPU) can land on different sides of an interval end when the hit sits on it.
  const near = (t: number) => Math.abs(t - ray.tMin) <= BOUNDARY_RELATIVE * Math.max(ray.tMin, tFloor) || (ray.tMax < 1e20 && Math.abs(t - ray.tMax) <= BOUNDARY_RELATIVE * Math.max(ray.tMax, tFloor));
  if ((gpu.t >= 0 && near(gpu.t)) || (cpu && near(cpu.t))) return { ...verdict, verdict: 'boundary' as const };
  return verdict;
}

/** Interval mismatches against the CPU, re-run on the plain inclusive walk with the same bounds to tell the interval logic from the walk itself. */
const suspects: { ray: Ray; where: string; gpu: Hit; why: string; slip: boolean }[] = [];

async function checkSet(device: GPUDevice, scene: TraceTestScene, set: RaySet, tr: Tracers, gpuScene: TraceSceneBuffers, cpu: CpuTracer, allHits: AllHits, edges: EdgeIndex, label: string): Promise<void> {
  const rays = set.rays;
  const n = rays.length;
  tally.rays += n;
  const tFloor = T_FLOOR * scene.scale;
  const ends = [0, ...Array.from({ length: 5 }, (_, k) => cascadeEnd(k, f32(scene.scale / 64), 4)), RT_T_INF];
  const K = ends.length - 1;
  const baseRun = await runBase(device, tr.base, gpuScene, packRays(rays));
  const baseHits = baseRun.hits;
  tally.hits += baseHits.filter((h) => h.t >= 0).length;

  for (const convention of ['open-low', 'closed-low'] as IntervalConvention[]) {
    const tracer = convention === 'open-low' ? tr.open : tr.lo;
    const packed = packBounded(rays, () => Array.from({ length: K }, (_, k) => [ends[k] as number, ends[k + 1] as number] as [number, number]));
    const data = await runRays(device, tracer, gpuScene, packed);
    const hits = readHits(data, n * K);
    const words = new Uint32Array(data);
    tally.intervalRays += n * K;
    for (let i = 0; i < n; i++) {
      const ray = rays[i] as Ray;
      const where = `${label} ${set.name}#${i} ${convention}`;
      const th = (baseHits[i] as Hit).t;
      // Owner: the interval that holds the unbounded closest hit.
      const owner = th >= 0 ? Array.from({ length: K }, (_, k) => k).find((k) => inInterval(th, ends[k] as number, ends[k + 1] as number, convention)) : undefined;
      if (th >= 0 && owner === undefined && th > 1.2e-38) {
        tally.partition++;
        note('partition', `${where}: t_h ${th} belongs to no interval`);
      }
      for (let k = 0; k < K; k++) {
        const hit = hits[i * K + k] as Hit;
        const b = intervalBounds(ends[k] as number, ends[k + 1] as number, convention);
        if (hit.t >= 0 && (hit.t < b.tMin || hit.t > b.tMax)) {
          tally.outside++;
          note('outside', `${where} k=${k}: t ${hit.t} outside [${b.tMin}, ${b.tMax}]`);
        }
        if (owner !== undefined) {
          if (k === owner) {
            const same = sameOrTie(hit, baseHits[i] as Hit, edges, ray.direction);
            if (same === 'tie') tally.ownerTies++;
            if (same === 'no') {
              tally.partition++;
              note('partition', `${where} k=${k}: owner of t_h ${th} reports ${hit.t} ${hit.instance}/${hit.triangle}, unbounded ${(baseHits[i] as Hit).instance}/${(baseHits[i] as Hit).triangle}`);
            }
          }
          if (k < owner && hit.t >= 0) {
            tally.partition++;
            note('partition', `${where} k=${k}: before the owner (${owner}) but hit t ${hit.t} (t_h ${th})`);
          }
          if (k > owner && hit.t >= 0 && !(hit.t > th)) {
            tally.partition++;
            note('partition', `${where} k=${k}: after the owner but t ${hit.t} <= t_h ${th}`);
          }
        } else if (th < 0 && hit.t >= 0) {
          tally.partition++;
          note('partition', `${where} k=${k}: unbounded ray missed but interval hit t ${hit.t}`);
        }
        // CPU oracle on the same f32 bounds.
        const cpuHit = b.empty ? null : cpu.closest({ origin: ray.origin, direction: ray.direction, tMin: b.tMin, tMax: b.tMax }, { mask: ALL });
        const bounded: Ray = { origin: ray.origin, direction: ray.direction, tMin: b.tMin, tMax: b.tMax };
        const verdict = intervalVerdict(hit, cpuHit, edges, tFloor, bounded, () => allHits(ray, ALL));
        if (verdict.verdict === 'tie' || verdict.verdict === 'primaryTie') tally.ties++;
        else if (verdict.verdict === 'grazing') tally.grazing++;
        else if (verdict.verdict === 'tPosition') tally.tPosition++;
        else if (verdict.verdict === 'boundary') tally.boundary++;
        else if (verdict.verdict === 'mismatch') {
          // Decided after the loop: a disagreement the plain inclusive walk over the same bounds shares is the walk's (V27 territory), not the interval logic's.
          const list = allHits(ray, ALL).filter((h) => h.t >= b.tMin * 0.99 && h.t <= Math.min(b.tMax, 1e9) * 1.01).slice(0, 6);
          const cpuNote = `CPU all-hits in range ${list.map((h) => `${h.instance}/${h.triangle}@${h.t.toPrecision(8)} bary ${h.u.toExponential(2)},${h.v.toExponential(2)}`).join(' | ')}`;
          suspects.push({ ray: bounded, where: `${where} k=${k}`, gpu: hit, why: `(${b.tMin}, ${b.tMax}): ${verdict.why}; ${cpuNote}`, slip: !!verdict.slip });
        }
      }
    }

    // 3. outputs: visible and distance agree with the hit output (open-low only for distance), indirect equals direct.
    const vis = convention === 'open-low' ? tr.openVisible : tr.loVisible;
    const visData = new Uint32Array(await runRays(device, vis, gpuScene, packed));
    for (let r = 0; r < n * K; r++) {
      if ((visData[r] !== 0) !== ((hits[r] as Hit).t < 0)) {
        // The any-hit forms use the exact bounds, the closest forms a widened walk cut back (see interval.ts): they may differ for a hit within ~1e-6 of an end.
        const k = r % K;
        const b = intervalBounds(ends[k] as number, ends[k + 1] as number, convention);
        const t = (hits[r] as Hit).t;
        const nearEnd = (x: number) => Math.abs(x - b.tMin) <= 4e-6 * Math.max(b.tMin, tFloor * 10) || (b.tMax < 1e20 && Math.abs(x - b.tMax) <= 4e-6 * Math.max(b.tMax, tFloor * 10));
        if (t >= 0 && nearEnd(t)) tally.visibleNearEnd++;
        else {
          tally.outputFail++;
          note('visible', `${label} ${set.name} ${convention} ray ${r}: visible ${visData[r]} but hit t ${t} in (${b.tMin}, ${b.tMax})`);
          break;
        }
      }
    }
    if (convention === 'open-low') {
      const dist = new Float32Array(await runRays(device, tr.openDistance, gpuScene, packed));
      const indirect = new Uint32Array(await runRays(device, tr.openIndirect, gpuScene, packed, ALL, true));
      for (let r = 0; r < n * K; r++) {
        if (dist[r] !== (hits[r] as Hit).t && !(dist[r] === -1 && (hits[r] as Hit).t < 0)) {
          tally.outputFail++;
          note('distance', `${label} ${set.name} ray ${r}: ${dist[r]} vs ${(hits[r] as Hit).t}`);
          break;
        }
      }
      for (let w = 0; w < n * K * HIT_WORDS; w++) {
        if (indirect[w] !== words[w]) {
          tally.outputFail++;
          note('indirect', `${label} ${set.name}: indirect differs from direct at word ${w}`);
          break;
        }
      }
    }
  }

  if (suspects.length) {
    const plainRun = await runBase(device, tr.base, gpuScene, packRays(suspects.map((x) => x.ray)));
    suspects.forEach((x, i) => {
      const p = plainRun.hits[i] as Hit;
      if (p.t === x.gpu.t && p.instance === x.gpu.instance && p.triangle === x.gpu.triangle) {
        tally.walkLevel++;
        note('walk-level', `${x.where} ${x.why}`);
      } else {
        tally.cpuMismatch++;
        if (x.slip) tally.slips++;
        note('cpu', `${x.where} ${x.why}; plain walk over the same bounds: t ${p.t} ${p.instance}/${p.triangle}`);
      }
    });
    suspects.length = 0;
  }

  // The plain inclusive walk (RtBatchTracer) on the very same boundaries, for the record (V290): rays whose tMax (or tMin) IS their own unbounded hit t.
  {
    const hitting = baseHits.map((h, i) => ({ h, i })).filter(({ h }) => h.t > 1e-20 && h.t < 1e20);
    if (hitting.length) {
      const upper = await runBase(device, tr.base, gpuScene, packRays(hitting.map(({ h, i }) => ({ origin: rays[i]!.origin, direction: rays[i]!.direction, tMin: 0, tMax: h.t }))));
      const lower = await runBase(device, tr.base, gpuScene, packRays(hitting.map(({ h, i }) => ({ origin: rays[i]!.origin, direction: rays[i]!.direction, tMin: h.t, tMax: FAR }))));
      hitting.forEach(({ h, i }, r) => {
        tally.plainChecked++;
        if (upper.hits[r]!.t !== h.t || (upper.hits[r]!.triangle !== h.triangle && sameOrTie(upper.hits[r]!, h, edges, rays[i]!.direction) === 'no')) tally.plainTmaxMiss++;
        if (lower.hits[r]!.t !== h.t || (lower.hits[r]!.triangle !== h.triangle && sameOrTie(lower.hits[r]!, h, edges, rays[i]!.direction) === 'no')) tally.plainTminMiss++;
      });
    }
  }

  // 2. boundary exactness at the GPU's own t_h
  const queries: { ray: number; kind: string; t0: number; t1: number; convention: IntervalConvention; expect: 'eq' | 'miss' | 'after' }[] = [];
  baseHits.forEach((hit, i) => {
    if (!(hit.t > 1e-20 && hit.t < 1e20)) return;
    const c = [hit.t, nextDownF32(hit.t), nextUpF32(hit.t)] as const;
    const names = ['t', 'prev', 'next'] as const;
    c.forEach((cut, j) => {
      queries.push({ ray: i, kind: `(0,${names[j]}]`, t0: 0, t1: cut, convention: 'open-low', expect: j === 1 ? 'miss' : 'eq' });
      queries.push({ ray: i, kind: `(${names[j]},inf]`, t0: cut, t1: RT_T_INF, convention: 'open-low', expect: j === 1 ? 'eq' : 'after' });
      queries.push({ ray: i, kind: `[0,${names[j]})`, t0: 0, t1: cut, convention: 'closed-low', expect: j === 2 ? 'eq' : 'miss' });
      queries.push({ ray: i, kind: `[${names[j]},inf)`, t0: cut, t1: RT_T_INF, convention: 'closed-low', expect: j === 2 ? 'after' : 'eq' });
    });
  });
  for (const convention of ['open-low', 'closed-low'] as IntervalConvention[]) {
    const list = queries.filter((q) => q.convention === convention);
    if (!list.length) continue;
    const packed = packRays(list.map((q) => ({ origin: rays[q.ray]!.origin, direction: rays[q.ray]!.direction, tMin: q.t0, tMax: q.t1 })));
    const data = await runRays(device, convention === 'open-low' ? tr.open : tr.lo, gpuScene, packed);
    const hits = readHits(data, list.length);
    list.forEach((q, r) => {
      tally.exactChecked++;
      const hit = hits[r] as Hit;
      const th = (baseHits[q.ray] as Hit).t;
      let ok: boolean;
      if (q.expect === 'eq') {
        const same = sameOrTie(hit, baseHits[q.ray] as Hit, edges, (rays[q.ray] as Ray).direction);
        if (same === 'tie') tally.ownerTies++;
        ok = same !== 'no';
      }
      else if (q.expect === 'miss') ok = hit.t < 0;
      else ok = hit.t < 0 || hit.t > th;
      if (!ok) {
        tally.exactFail++;
        const base = baseHits[q.ray] as Hit;
        const bary = (h: Hit) => `bary ${h.barycentrics[0].toExponential(3)},${h.barycentrics[1].toExponential(3)}`;
        note('exact', `${label} ${set.name}#${q.ray} ${q.kind} expects ${q.expect}: t_h ${th} on ${base.instance}/${base.triangle} ${bary(base)}, got ${hit.t} on ${hit.instance}/${hit.triangle} ${bary(hit)}`);
      }
    });
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 4. Probe bundles
// ---------------------------------------------------------------------------------------------------------------

async function checkProbes(device: GPUDevice, scene: TraceTestScene, sets: RaySet[], tr: Tracers, gpuScene: TraceSceneBuffers, label: string): Promise<void> {
  const source = (sets.find((s) => s.name === 'incoherent') ?? sets[0]) as RaySet;
  const probeCount = Math.min(source.rays.length, 384);
  const N = 8;
  const D = N * N;
  const jitter: [number, number] = [0.37, 0.61];
  const t0 = f32(scene.scale * 0.002);
  const t1 = f32(scene.scale * 0.45);
  const scale = (p: number) => (p % 7 === 0 ? 0 : p % 5 === 0 ? 2 : 1);
  const positions = new Float32Array(probeCount * 4);
  const rays: Ray[] = [];
  for (let p = 0; p < probeCount; p++) {
    const o = (source.rays[p] as Ray).origin;
    positions.set([o[0], o[1], o[2], scale(p)], p * 4);
    for (let c = 0; c < D; c++) rays.push({ origin: [f32(o[0]), f32(o[1]), f32(o[2])], direction: bundleDirection(c % N, Math.floor(c / N), N, jitter).map(f32) as [number, number, number], tMin: f32(t0 * scale(p)), tMax: f32(t1 * scale(p)) });
  }
  // expected: rays source on the open-low tracer; a probe with scale 0 is skipped (miss), which an empty interval also gives (tMax 0 < tMin)
  const expected = await runRays(device, tr.open, gpuScene, packRays(rays.map((r, i) => (Math.floor(i / D) % 7 === 0 ? { ...r, tMin: 1, tMax: 0 } : r))));
  const exp = readHits(expected, rays.length);
  const run = async (tracer: RtIntervalTracer, indirect: boolean): Promise<ArrayBuffer> => {
    const probes = upload(device, positions);
    const results = empty(device, rays.length * tracer.resultBytes);
    const counts = indirect ? upload(device, new Uint32Array([probeCount + 99, probeCount, 0, 0])) : null;
    tracer.beginFrame();
    const encoder = device.createCommandEncoder();
    tracer.encode(encoder, gpuScene, { source: probes, results, count: counts ? { buffer: counts, word: 1, capacity: probeCount } : probeCount, interval: { t0, t1 }, directions: N, jitter });
    device.queue.submit([encoder.finish()]);
    const data = await readBack(device, results, rays.length * tracer.resultBytes);
    for (const buffer of [probes, results, counts]) buffer?.destroy();
    return data;
  };
  const direct = await run(tr.probes, false);
  const got = readHits(direct, rays.length);
  let differ = 0;
  for (let r = 0; r < rays.length; r++) {
    tally.probeRays++;
    const a = got[r] as Hit;
    const b = exp[r] as Hit;
    const probe = Math.floor(r / D);
    if (scale(probe) === 0) {
      if (a.t >= 0) {
        tally.probeFail++;
        note('probe-skip', `${label}: probe ${probe} has scale 0 but ray ${r} hit t ${a.t}`);
      }
      continue;
    }
    if (a.t < 0 && b.t < 0) continue;
    const same = a.t >= 0 && b.t >= 0 && a.instance === b.instance && a.triangle === b.triangle && Math.abs(a.t - b.t) <= 1e-4 * Math.max(b.t, 1e-3 * scene.scale);
    if (same) tally.maxProbeT = Math.max(tally.maxProbeT, Math.abs(a.t - b.t) / Math.max(b.t, 1e-3 * scene.scale));
    else differ++;
  }
  tally.probeDiff += differ;
  const live = rays.length - Array.from({ length: probeCount }, (_, p) => p).filter((p) => scale(p) === 0).length * D;
  if (differ > Math.max(2, live * 0.005)) {
    tally.probeFail++;
    note('probe', `${label}: ${differ} of ${live} probe rays differ from the rays source fed the twin's directions`);
  }
  const indirect = await run(tr.probesIndirect, true);
  const a = new Uint32Array(direct);
  const b = new Uint32Array(indirect);
  for (let w = 0; w < a.length; w++) {
    if (a[w] !== b[w]) {
      tally.probeFail++;
      note('probe-indirect', `${label}: indirect probes differ from direct at word ${w}`);
      break;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// 5. WGSL helpers
// ---------------------------------------------------------------------------------------------------------------

async function checkHelpers(device: GPUDevice, caps: Caps): Promise<void> {
  const kernel = /* wgsl */ `
@group(0) @binding(0) var<storage, read> hIn: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> hOut: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> hDir: array<vec4f>;
@compute @workgroup_size(64)
fn hMain(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= arrayLength(&hIn)) { return; }
    let v = hIn[i];
    let count = u32(v.w);
    hOut[i] = vec4u(rtCascadeIndex(v.x, v.y, v.z, count), bitcast<u32>(rtCascadeEnd(3u, v.y, v.z)), bitcast<u32>(rtCascadeStart(2u, v.y, v.z)), bitcast<u32>(rtCascadeEnd(0u, v.y, v.z)));
    let uv = vec2f(fract(v.x * 0.61803398875), fract(v.x * 0.7548776662466927));
    hDir[i] = vec4f(rtOctEqualAreaDecode(uv), 0.0);
}
`;
  const code = rtModuleWgsl(caps, { intervals: true }) + kernel;
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'hMain' } });
  const random = createRandom(99);
  const samples: number[] = [];
  const cases: [number, number, number][] = [[0.5, 4, 5], [0.3, 3.3, 6], [1.37, 1.7, 6], [0.01, 2, 8]];
  for (const [t0, growth, count] of cases) {
    const bounds = Array.from({ length: count }, (_, n) => cascadeEnd(n, f32(t0), f32(growth)));
    const ts = [...bounds, ...bounds.map(nextUpF32), ...bounds.map(nextDownF32), ...Array.from({ length: 200 }, () => f32(random() * (bounds[count - 1] as number) * 1.4))];
    for (const t of ts) samples.push(f32(t), f32(t0), f32(growth), count);
  }
  const input = new Float32Array(samples);
  const n = input.length / 4;
  const inBuffer = upload(device, input);
  const out = empty(device, n * 16);
  const dir = empty(device, n * 16);
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuffer } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: dir } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const words = new Uint32Array(await readBack(device, out, n * 16));
  const dirs = new Float32Array(await readBack(device, dir, n * 16));
  let maxDir = 0;
  for (let i = 0; i < n; i++) {
    const [t, t0, growth, count] = [input[i * 4] as number, input[i * 4 + 1] as number, input[i * 4 + 2] as number, input[i * 4 + 3] as number];
    const want = [cascadeIndex(t, t0, growth, count), cascadeEnd(3, t0, growth), cascadeStart(2, t0, growth), cascadeEnd(0, t0, growth)];
    const got = [words[i * 4] as number, f32Of(words[i * 4 + 1] as number), f32Of(words[i * 4 + 2] as number), f32Of(words[i * 4 + 3] as number)];
    if (want.some((w, k) => w !== got[k])) {
      tally.helperFail++;
      note('helpers', `cascade t=${t} t0=${t0} growth=${growth} count=${count}: gpu ${got.join(',')} twin ${want.join(',')}`);
    }
    const twin = octEqualAreaDecode((t * 0.61803398875) % 1, (t * 0.7548776662466927) % 1);
    for (let k = 0; k < 3; k++) maxDir = Math.max(maxDir, Math.abs((dirs[i * 4 + k] as number) - (twin[k] as number)));
  }
  if (maxDir > 2e-4) {
    tally.helperFail++;
    note('helpers', `rtOctEqualAreaDecode differs from its twin by ${maxDir}`);
  }
  metrics.oct_decode_max_abs_error = maxDir;
  for (const buffer of [inBuffer, out, dir]) buffer.destroy();
}

// ---------------------------------------------------------------------------------------------------------------
// 6. Shaded output
// ---------------------------------------------------------------------------------------------------------------

const REFERENCE_KERNEL = /* wgsl */ `
struct RefParams { count: u32, mask: u32, seed: u32, mode: u32 }
@group(1) @binding(0) var<uniform> refParams: RefParams;
@group(1) @binding(1) var<storage, read> refRays: array<RtRay>;
@group(1) @binding(2) var<storage, read_write> refOut: array<vec4f>;

fn refPcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

@compute @workgroup_size(64)
fn refMain(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= refParams.count) { return; }
    let r = refRays[i];
    let hit = rtClosestInterval(r.origin, r.direction, r.tMin, r.tMax, refParams.mask);
    var out = vec4f(0.0, 0.0, 0.0, -1.0);
    if (hit.t >= 0.0) {
        let a = refPcg(i + refPcg(refParams.seed));
        let b = refPcg(a);
        let c = refPcg(b);
        let u = vec3f(f32(a >> 8u), f32(b >> 8u), f32(c >> 8u)) * (1.0 / 16777216.0);
        let s = rtShade(hit, RtRay(r.origin, 0.0, r.direction, hit.t), refParams.mode, u, refParams.mask);
        let col = rtShadeTotal(s);
        out = vec4f(select(vec3f(0.0), col, all(abs(col) < vec3f(3.0e38))), hit.t);
    }
    refOut[i] = out;
}
`;

async function checkShaded(device: GPUDevice, caps: Caps, packing: SceneCopyPacking, perSet: number): Promise<void> {
  for (const desc of [cornell, quadLight] as SceneDesc[]) {
    const input = sceneDescToCopyInput(desc);
    const builder = createMeshBvhBuilder();
    const data = await buildTraceScene(
      input.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : null })),
      input.instances.map((instance) => ({ ...instance, static: true })),
      { build: (geometry) => builder.build(geometry), mergeStaticInstances: true },
    );
    builder.dispose();
    const trace = uploadTraceScene(device, data, desc.name);
    const copy = buildSceneCopy({ ...input, options: { ...input.options, surface: { ...input.options?.surface, positions: true } } });
    const buffers = uploadSceneCopy(device, copy, { label: `${desc.name} scene`, packing });
    const format = sceneFormatOf(trace);
    const modes = ['flat', 'lambert', 'full'] as const;
    const tracers = await Promise.all(modes.map((shadeMode) => RtIntervalTracer.create(device, caps, { ...format, source: 'rays', output: 'shaded', shadeMode, scene: { layout: copy.layout, packing }, label: `${desc.name} shaded ${shadeMode}` })));
    const distance = await RtIntervalTracer.create(device, caps, { ...format, source: 'rays', output: 'distance', label: `${desc.name} distance` });
    const bytes = tracers.map((t) => t.kernelBytes);
    metrics[`shaded_kernel_bytes_${desc.name}`] = Math.max(...bytes);

    // Reference kernel: the inline module + the page's own entry point.
    const module = rtModuleWgsl(caps, { ...format, traceSlot: { group: 0, firstBinding: 0 }, intervals: true, scene: { slot: { group: 2, firstBinding: 0 }, layout: copy.layout, packing }, shade: true, minify: true });
    const code = `${module}\n${rtMinifyWgsl(REFERENCE_KERNEL)}`;
    const layouts = [
      device.createBindGroupLayout({ entries: withBindings(rtWalkEntries(format)) }),
      device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      }),
      device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, undefined, packing) }),
    ];
    const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: layouts }), compute: { module: device.createShaderModule({ code: packing === 'packed' ? rtPackSceneAccess(code) : code }), entryPoint: 'refMain' } });

    const random = createRandom(0x5ade + desc.name.length);
    // Primary rays of the scene camera, cut into the cascade intervals (t0 = 0.4 m, growth 2): every ray yields up to 5 interval rays.
    const camera = desc.camera as unknown as { position: number[]; target: number[]; up: number[]; yFovDeg: number };
    const side = Math.ceil(Math.sqrt(perSet));
    const dirs: Ray[] = [];
    for (let i = 0; i < perSet; i++) {
      const px = ((i % side) + random()) / side;
      const py = (Math.floor(i / side) + random()) / side;
      const f = [0, 1, 2].map((k) => (camera.target[k] as number) - (camera.position[k] as number));
      const fl = Math.hypot(...f);
      const fw = f.map((x) => x / fl);
      const right = [fw[1]! * camera.up[2]! - fw[2]! * camera.up[1]!, fw[2]! * camera.up[0]! - fw[0]! * camera.up[2]!, fw[0]! * camera.up[1]! - fw[1]! * camera.up[0]!];
      const rl = Math.hypot(...right);
      const rt = right.map((x) => x / rl);
      const up = [rt[1]! * fw[2]! - rt[2]! * fw[1]!, rt[2]! * fw[0]! - rt[0]! * fw[2]!, rt[0]! * fw[1]! - rt[1]! * fw[0]!];
      const tanY = Math.tan((camera.yFovDeg * Math.PI) / 360);
      const x = (2 * px - 1) * tanY * (desc.film.width / desc.film.height);
      const y = (1 - 2 * py) * tanY;
      const d = [0, 1, 2].map((k) => fw[k]! + rt[k]! * x + up[k]! * y);
      const dl = Math.hypot(...d);
      dirs.push({ origin: camera.position as [number, number, number], direction: d.map((v) => v / dl) as [number, number, number], tMin: 0, tMax: FAR });
    }
    const ends = [0, 0.4, 0.8, 1.6, 3.2, RT_T_INF];
    const packed = packBounded(dirs, () => ends.slice(0, -1).map((e, k) => [e, ends[k + 1] as number] as [number, number]));
    const count = packed.length / 8;
    const rays = upload(device, packed);
    const sceneGroup = device.createBindGroup({ layout: layouts[0] as GPUBindGroupLayout, entries: rtSceneBindGroupEntries(trace) });
    const copyGroup = device.createBindGroup({ layout: layouts[2] as GPUBindGroupLayout, entries: rtSceneCopyBindGroupEntries(buffers) });
    const distBuf = empty(device, count * 4);
    {
      distance.beginFrame();
      const encoder = device.createCommandEncoder();
      distance.encode(encoder, trace, { source: rays, results: distBuf, count });
      device.queue.submit([encoder.finish()]);
    }
    const dist = new Float32Array(await readBack(device, distBuf, count * 4));
    for (const [m, tracer] of tracers.entries()) {
      const mode = m; // RT_SHADE_MODE: flat 0, lambert 1, full 2
      for (const seed of [0, 12345]) {
        const results = empty(device, count * 16);
        tracer.beginFrame();
        const encoder = device.createCommandEncoder();
        tracer.encode(encoder, trace, { source: rays, results, count, seed, sceneCopy: buffers });
        device.queue.submit([encoder.finish()]);
        const got = new Float32Array(await readBack(device, results, count * 16));
        // reference
        const params = upload(device, new Uint32Array([count, ALL, seed, mode]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const refOut = empty(device, count * 16);
        const refGroup = device.createBindGroup({
          layout: layouts[1] as GPUBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: rays } },
            { binding: 2, resource: { buffer: refOut } },
          ],
        });
        const refEncoder = device.createCommandEncoder();
        const pass = refEncoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, sceneGroup);
        pass.setBindGroup(1, refGroup);
        pass.setBindGroup(2, copyGroup);
        pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
        device.queue.submit([refEncoder.finish()]);
        const want = new Float32Array(await readBack(device, refOut, count * 16));
        let shadedHits = 0;
        let lit = 0;
        for (let i = 0; i < count; i++) {
          tally.shadedRays++;
          const a = got.subarray(i * 4, i * 4 + 4);
          const b = want.subarray(i * 4, i * 4 + 4);
          if (a[3] !== (dist[i] as number) && !(a[3] === -1 && (dist[i] as number) < 0)) {
            tally.shadedFail++;
            note('shaded-t', `${desc.name} ${modes[m]} ray ${i}: shaded t ${a[3]} vs distance ${dist[i]}`);
          }
          if ((a[3] as number) >= 0) shadedHits++;
          if ((a[0] as number) + (a[1] as number) + (a[2] as number) > 0) lit++;
          let bad = false;
          for (let k = 0; k < 4; k++) {
            const x = a[k] as number;
            const y = b[k] as number;
            if (!(Math.abs(x - y) <= 1e-4 * Math.max(Math.abs(x), Math.abs(y), 1e-3))) bad = true;
          }
          if (bad) {
            tally.shadedFail++;
            note('shaded', `${desc.name} ${modes[m]} seed ${seed} ray ${i}: ${Array.from(a).join(',')} vs reference ${Array.from(b).join(',')}`);
          }
        }
        if (shadedHits === 0 || (modes[m] !== 'flat' && lit === 0)) {
          tally.shadedFail++;
          note('shaded', `${desc.name} ${modes[m]} seed ${seed}: ${shadedHits} hits, ${lit} lit: the check saw nothing`);
        }
        for (const buffer of [results, params, refOut]) buffer.destroy();
      }
    }
    for (const buffer of [rays, distBuf]) buffer.destroy();
    for (const tracer of [...tracers, distance]) tracer.destroy();
    buffers.destroy();
    trace.destroy();
    await drainDevice(device);
  }
}

// ---------------------------------------------------------------------------------------------------------------

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const metrics: Record<string, number> = {};

try {
  const args = getArgs();
  const sceneNames = (args.scenes ?? 'edges,cornell,quad-light,sponza-white').split(',').filter(Boolean);
  const perSet = numberArg(args, 'rays', 2048);
  const packing = (args.packing as SceneCopyPacking | undefined) ?? DEFAULT_SCENE_PACKING;
  if (packing !== 'split' && packing !== 'packed') throw new Error(`packing must be split or packed, got '${packing}'`);

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'rc-interval-check' });
  deviceErrors = watchDevice(device);

  const builder = createMeshBvhBuilder();
  const tiny = await createTinyBvhBuilder();
  const tracerByFormat = new Map<string, Tracers>();
  for (const format of FORMATS) tracerByFormat.set(format.name, await makeTracers(device, caps, format));
  const t0 = performance.now();
  const stage = (what: string) => console.log(`[rc-interval-check] ${what} at ${((performance.now() - t0) / 1000).toFixed(1)} s`);

  for (const name of sceneNames) {
    const scene = await loadTraceScene(name);
    const allHits = makeAllHits(scene, tiny);
    const edges = makeEdgeIndex(scene);
    const sets = raySets(scene, perSet);
    for (const format of FORMATS) {
      const data = await buildScene(scene, format, builder);
      const cpu = createCpuTracer(data);
      const gpuScene = uploadTraceScene(device, data, `${name} ${format.name}`);
      const tr = tracerByFormat.get(format.name) as Tracers;
      for (const set of sets) await checkSet(device, scene, set, tr, gpuScene, cpu, allHits, edges, `${name}/${format.name}`);
      await checkProbes(device, scene, sets, tr, gpuScene, `${name}/${format.name}`);
      gpuScene.destroy();
      stage(`${name}/${format.name} checked`);
    }
    await drainDevice(device);
  }
  await checkHelpers(device, caps);
  stage('helpers checked');
  if ((args.shaded ?? '1') !== '0') {
    await checkShaded(device, caps, packing, Math.min(perSet, 1024));
    stage('shaded checked');
  }
  metrics.kernel_bytes_rays_hit = Math.max(...[...tracerByFormat.values()].map((t) => t.open.kernelBytes));
  metrics.kernel_bytes_probes_hit = Math.max(...[...tracerByFormat.values()].map((t) => t.probes.kernelBytes));

  const failing = ['cpuMismatch', 'outside', 'partition', 'exactFail', 'outputFail', 'probeFail', 'helperFail', 'shadedFail'] as const;
  const failures = failing.filter((key) => tally[key] > 0).map((key) => `${key}: ${tally[key]}`);
  await drainDevice(device);
  if (deviceErrors.length) failures.push(`device errors: ${deviceErrors.join('; ')}`);
  for (const [key, value] of Object.entries(tally)) metrics[key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`)] = value;
  if (failures.length) throw new Error(`rc-interval-check failed: ${failures.join(', ')}\n${examples.join('\n')}`);

  reportResult(
    {
      scene: sceneNames.join('+'),
      renderer: 'rt-interval',
      config: { perSet, packing, formats: FORMATS.map((format) => format.name), caps: { tier: caps.tier, subgroups: caps.subgroups } },
      metrics,
      notes: { examples, builderUsesWorker: builder.usesWorker },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
