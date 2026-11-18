/**
 * Trace core correctness (P1 slice 1): L0 `rtVisible` and L1 `rtClosest` against bvh's CPU oracle, through both the
 * batch dispatcher (queue ABI v0, every D11 setting) and the inline module (with `rtSurface`).
 *
 * Per scene and per scene format:
 *  1. the scene is built once (tinybvh trees, `buildTraceScene`), uploaded with bvh's `uploadTraceScene`, and the very
 *     same `TraceSceneData` goes to `createCpuTracer` (the step-for-step CPU port of the WGSL walk, which
 *     `CpuSceneBvh.raycast` also runs). Same trees on both sides: t within 1e-5 relative, same instance, triangle,
 *     front face and material, and identical visibility, except where f32 (GPU) and f64 (bvh's CPU walk) may
 *     legitimately differ next to an edge: ties, grazing contacts and t position error are classified, counted and
 *     reported per ray set (see "Edge classification" below, DIVERGES V27). A watertight slip always fails;
 *  2. variants of the batch kernel (SoA rays, indirect count, compact record, cones, workgroup 128) must reproduce the
 *     base batch bit for bit; the inline kernel must too, and its `rtSurface` must match `referenceSurface` (doubles);
 *  3. with `sceneTrees=1` (default), the production path: `SceneTrees` builds its own trees (GPU first trees, then
 *     tinybvh refinements; both states are traced) and is compared with a CPU-built scene of the same format.
 *
 * Every GPU result is read back once, after its submit; nothing is timed here.
 *
 * Args: scenes=cornell,quad-light,edges,sponza-white  rays=8192 (per ray set)  sceneTrees=1
 */
import {
  CpuSceneBvh,
  SceneTrees,
  buildMeshBvhWithTinyBvh,
  buildTraceScene,
  createCpuTracer,
  createMeshBvhBuilder,
  createTinyBvhBuilder,
  uploadTraceScene,
  type CpuTracer,
  type MeshBvhBuilder,
  type TinyBvhBuilder,
  type TraceSceneBuffers,
  type TraceSceneData,
} from '@pmndrs/bvh';
import {
  RtBatchTracer,
  aosToSoa,
  buildSceneCopy,
  detectCaps,
  packCones,
  readHits,
  referenceSurface,
  requestDeviceForCaps,
  rtBatchArgsKernelWgsl,
  rtModuleWgsl,
  rtSceneBindGroupEntries,
  rtSceneCopyBindGroupEntries,
  rtSceneCopyLayoutEntries,
  DEFAULT_SCENE_PACKING,
  type SceneCopyPacking,
  rtWalkEntries,
  unpackRay,
  uploadSceneCopy,
  withBindings,
  type BatchKernelOptions,
  type Caps,
  type Hit,
  type Ray,
  type ResolvedSceneFormat,
  type SceneCopyBuffers,
  type SurfaceInstanceInput,
} from '@pmndrs/raytracer';
import { createBvhWorkerPool } from '@pmndrs/bvh/worker';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import {
  MASK_NO_EMITTERS,
  cameraRays,
  createRandom,
  edgeRays,
  incoherentRays,
  insideRays,
  loadTraceScene,
  packRays,
  shadowRays,
  surfacePoints,
  type TraceTestScene,
} from './trace-scenes.ts';
import { NORMAL_COSINE, T_FLOOR, T_RELATIVE, compareClosest, makeEdgeIndex, tClose, type AllHits, type EdgeIndex, type Verdict } from './trace-compare.ts';

const CONE = { width: 0.01, spread: 0.002 };
const ALL = 0xffffffff;

interface Format extends ResolvedSceneFormat {
  name: string;
  merge: boolean;
}
const FORMATS: Format[] = [
  { name: 'quantized-records-merged', nodeFormat: 'quantized', triangleLeaves: 'records', positionLayout: 'vec4', merge: true },
  { name: 'f32-indexed', nodeFormat: 'f32', triangleLeaves: 'indexed', positionLayout: 'vec4', merge: false },
];

interface RaySet {
  name: string;
  packed: Float32Array<ArrayBuffer>;
  count: number;
  mask: number;
  /** Every ray must hit (closed box). */
  mustHit: boolean;
}

interface Tally {
  rays: number;
  closestMismatch: number;
  visibleMismatch: number;
  ties: number;
  /** Primary rays only: a different triangle at the same t that shares a corner or plane with the oracle's (listed in notes). */
  primaryTie: number;
  grazing: number;
  tPosition: number;
  slipsOnBoxEdge: number;
  slipsElsewhere: number;
  escapes: number;
  variantMismatch: number;
  inlineMismatch: number;
  surfaceMismatch: number;
  coneMismatch: number;
  hits: number;
  blocked: number;
  maxTRelative: number;
  maxSurfacePositionError: number;
  examples: string[];
  /** Every primary-ray tie / grazing / tPosition, with its triangles (the primary set is strict: see compareClosest). */
  primaryDetails: string[];
  /** Ray-set breakdown of the counted (not failed) categories. */
  bySet: Record<string, { rays: number; ties: number; primaryTie: number; grazing: number; tPosition: number }>;
}
const newTally = (): Tally => ({ rays: 0, closestMismatch: 0, visibleMismatch: 0, ties: 0, primaryTie: 0, grazing: 0, tPosition: 0, slipsOnBoxEdge: 0, slipsElsewhere: 0, escapes: 0, variantMismatch: 0, inlineMismatch: 0, surfaceMismatch: 0, coneMismatch: 0, hits: 0, blocked: 0, maxTRelative: 0, maxSurfacePositionError: 0, examples: [], primaryDetails: [], bySet: {} });

const noted = new WeakMap<Tally, Map<string, number>>();
/** Keeps up to three examples per category. */
function note(tally: Tally, category: string, text: string): void {
  let counts = noted.get(tally);
  if (!counts) noted.set(tally, (counts = new Map()));
  const n = counts.get(category) ?? 0;
  counts.set(category, n + 1);
  if (n < 3) tally.examples.push(text);
}

// ---------------------------------------------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------------------------------------------

const STORAGE_IO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

function upload(device: GPUDevice, data: ArrayBufferView<ArrayBuffer>, usage = STORAGE_IO): GPUBuffer {
  const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(data.byteLength / 16) * 16), usage });
  device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

async function readBack(device: GPUDevice, buffers: GPUBuffer[]): Promise<ArrayBuffer[]> {
  const encoder = device.createCommandEncoder();
  const staging = buffers.map((buffer) => {
    const copy = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(buffer, 0, copy, 0, buffer.size);
    return copy;
  });
  device.queue.submit([encoder.finish()]);
  return Promise.all(
    staging.map(async (copy) => {
      await copy.mapAsync(GPUMapMode.READ);
      const data = copy.getMappedRange().slice(0);
      copy.destroy();
      return data;
    }),
  );
}

/** The page's own kernel on the inline module: L1 with a cone, L0, and the full `rtSurface` per ray. */
/** Scene copy binding (P3.11, page arg `packing=split|packed`, default: the library default, packed since D32). */
let PACKING: SceneCopyPacking = DEFAULT_SCENE_PACKING;

const INLINE_KERNEL = /* wgsl */ `
struct PageParams { count: u32, mask: u32, pad0: u32, pad1: u32, cone: vec4f }
@group(1) @binding(0) var<storage, read> pageRays: array<RtRay>;
@group(1) @binding(1) var<storage, read_write> pageOut: array<vec4u>;
@group(1) @binding(2) var<uniform> pageParams: PageParams;

@compute @workgroup_size(64)
fn pageMain(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let i = id.y * groups.x * 64u + id.x;
    if (i >= pageParams.count) { return; }
    let ray = pageRays[i];
    let hit = rtClosestCone(ray, pageParams.mask, RtCone(pageParams.cone.x, pageParams.cone.y));
    let visible = rtVisible(ray, pageParams.mask);
    let s = rtSurface(hit, ray, RT_SURF_ALL);
    let o = i * 6u;
    pageOut[o] = vec4u(bitcast<u32>(hit.t), hit.instance, hit.triangle, select(0u, 1u, hit.frontFace) | select(0u, 2u, visible));
    pageOut[o + 1u] = vec4u(bitcast<vec2u>(hit.bary), hit.material, bitcast<u32>(hit.coneWidth));
    pageOut[o + 2u] = vec4u(bitcast<vec3u>(s.geometricNormal), select(0u, 1u, s.frontFace));
    pageOut[o + 3u] = vec4u(bitcast<vec3u>(s.position), s.material);
    pageOut[o + 4u] = vec4u(bitcast<vec3u>(s.shadingNormal), 0u);
    pageOut[o + 5u] = vec4u(bitcast<vec2u>(s.uv), 0u, 0u);
}
`;

interface Kernels {
  closest: RtBatchTracer;
  visible: RtBatchTracer;
  /** Must reproduce `closest` / `visible` bit for bit. */
  closestVariants: { name: string; tracer: RtBatchTracer }[];
  visibleVariants: { name: string; tracer: RtBatchTracer }[];
  inline: { pipeline: GPUComputePipeline; layouts: GPUBindGroupLayout[]; bytes: number };
}

async function makeKernels(device: GPUDevice, caps: Caps, format: Format): Promise<Kernels> {
  const base = { nodeFormat: format.nodeFormat, triangleLeaves: format.triangleLeaves, positionLayout: format.positionLayout };
  const make = (options: BatchKernelOptions, name: string) => RtBatchTracer.create(device, caps, { ...base, ...options, label: `${format.name} ${name}` });
  const variants = async (mode: 'closest' | 'visible') => {
    const list: [string, BatchKernelOptions][] = [
      ['soa', { mode, rayLayout: 'soa' }],
      ['indirect', { mode, indirect: true }],
      ['wg128', { mode, workgroupSize: 128 }],
    ];
    if (mode === 'closest') list.push(['compact', { mode, hitRecord: 'compact' }], ['cones', { mode, cones: true }]);
    return Promise.all(list.map(async ([name, options]) => ({ name, tracer: await make(options, `${mode} ${name}`) })));
  };

  const surfaceLayout = { normals: true, uvs: true, positions: true };
  const code = rtModuleWgsl(caps, { ...base, scene: { slot: { group: 2, firstBinding: 0 }, layout: surfaceLayout, materials: false, lights: false, packing: PACKING } }) + INLINE_KERNEL;
  const layouts = [
    device.createBindGroupLayout({ entries: withBindings(rtWalkEntries(base)) }),
    device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    }),
    device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, undefined, PACKING) }),
  ];
  const pipeline = await device.createComputePipelineAsync({
    label: `${format.name} inline`,
    layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
    compute: { module: device.createShaderModule({ code }), entryPoint: 'pageMain' },
  });
  return {
    closest: await make({ mode: 'closest' }, 'closest'),
    visible: await make({ mode: 'visible' }, 'visible'),
    closestVariants: await variants('closest'),
    visibleVariants: await variants('visible'),
    inline: { pipeline, layouts, bytes: code.length },
  };
}

interface GpuAnswers {
  hits: Hit[];
  hitWords: Uint32Array;
  visible: Uint32Array;
  closestVariants: { name: string; words: Uint32Array; widths?: Float32Array }[];
  visibleVariants: { name: string; words: Uint32Array }[];
  inline: Uint32Array | null;
}

/** Runs every kernel on one ray set: one encoder, one submit, then the readbacks. */
async function traceOnGpu(device: GPUDevice, kernels: Kernels | null, base: { closest: RtBatchTracer; visible: RtBatchTracer }, scene: TraceSceneBuffers, surface: SceneCopyBuffers | null, set: RaySet): Promise<GpuAnswers> {
  const n = set.count;
  const rays = upload(device, set.packed);
  const soa = aosToSoa(set.packed);
  const raysA = upload(device, soa.a);
  const raysB = upload(device, soa.b);
  // Word 0 is larger than the capacity (the clamp is under test), word 1 is the count.
  const counts = upload(device, new Uint32Array([n + 1000, n, 0, 0]));
  const cones = upload(device, packCones(Array.from({ length: n }, () => CONE)));
  const results = (bytes: number) => device.createBuffer({ size: Math.max(16, n * bytes), usage: STORAGE_IO });
  const own: GPUBuffer[] = [rays, raysA, raysB, counts, cones];

  const encoder = device.createCommandEncoder({ label: `trace-core ${set.name}` });
  const hitBuffer = results(base.closest.resultBytes);
  const visibleBuffer = results(4);
  base.closest.encode(encoder, scene, { rays, results: hitBuffer, count: n, mask: set.mask });
  base.visible.encode(encoder, scene, { rays, results: visibleBuffer, count: n, mask: set.mask });
  own.push(hitBuffer, visibleBuffer);

  const runVariant = (tracer: RtBatchTracer) => {
    const out = results(tracer.resultBytes);
    const widths = tracer.options.cones ? results(4) : null;
    own.push(out);
    if (widths) own.push(widths);
    tracer.encode(encoder, scene, {
      rays: tracer.options.rayLayout === 'soa' ? [raysA, raysB] : rays,
      results: out,
      count: tracer.options.indirect ? { buffer: counts, word: 0, capacity: n } : n,
      mask: set.mask,
      cones: widths ? { cones, widths } : undefined,
    });
    return { out, widths };
  };
  const closestRuns = (kernels?.closestVariants ?? []).map((variant) => ({ name: variant.name, ...runVariant(variant.tracer) }));
  const visibleRuns = (kernels?.visibleVariants ?? []).map((variant) => ({ name: variant.name, ...runVariant(variant.tracer) }));

  let inlineOut: GPUBuffer | null = null;
  if (kernels && surface) {
    inlineOut = results(6 * 16);
    const params = upload(device, new Float32Array([0, 0, 0, 0, CONE.width, CONE.spread, 0, 0]), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(params, 0, new Uint32Array([n, set.mask]));
    own.push(inlineOut, params);
    const groups = [
      device.createBindGroup({ layout: kernels.inline.layouts[0] as GPUBindGroupLayout, entries: rtSceneBindGroupEntries(scene) }),
      device.createBindGroup({
        layout: kernels.inline.layouts[1] as GPUBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: rays } },
          { binding: 1, resource: { buffer: inlineOut } },
          { binding: 2, resource: { buffer: params } },
        ],
      }),
      device.createBindGroup({ layout: kernels.inline.layouts[2] as GPUBindGroupLayout, entries: rtSceneCopyBindGroupEntries(surface) }),
    ];
    const pass = encoder.beginComputePass({ label: 'trace-core inline' });
    pass.setPipeline(kernels.inline.pipeline);
    groups.forEach((group, i) => pass.setBindGroup(i, group));
    const workgroups = Math.ceil(n / 64);
    const x = Math.min(workgroups, 65535);
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
    pass.end();
  }
  device.queue.submit([encoder.finish()]);

  const toRead = [hitBuffer, visibleBuffer, ...closestRuns.flatMap((run) => (run.widths ? [run.out, run.widths] : [run.out])), ...visibleRuns.map((run) => run.out), ...(inlineOut ? [inlineOut] : [])];
  const data = await readBack(device, toRead);
  let cursor = 0;
  const next = () => data[cursor++] as ArrayBuffer;
  const hitData = next();
  const answers: GpuAnswers = {
    hits: readHits(hitData, n, base.closest.options.hitRecord),
    hitWords: new Uint32Array(hitData, 0, (n * base.closest.resultBytes) / 4),
    visible: new Uint32Array(next(), 0, n),
    closestVariants: closestRuns.map((run) => ({ name: run.name, words: new Uint32Array(next()), widths: run.widths ? new Float32Array(next(), 0, n) : undefined })),
    visibleVariants: visibleRuns.map((run) => ({ name: run.name, words: new Uint32Array(next(), 0, n) })),
    inline: inlineOut ? new Uint32Array(next(), 0, n * 24) : null,
  };
  for (const buffer of own) buffer.destroy();
  return answers;
}

// ---------------------------------------------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------------------------------------------


interface Oracle {
  cpu: CpuTracer;
  /** Every hit along a ray (CpuSceneBvh, records trees), built on first use. */
  allHits: AllHits;
  edges: EdgeIndex;
}

/** A visibility disagreement is grazing when every blocker involved is next to a boundary edge or at an interval end. */
function visibilityVerdict(ray: Ray, mask: number, gpuVisible: boolean, gpuHit: Hit, oracle: Oracle): Verdict {
  const atEnd = (t: number) => tClose(t, ray.tMax, 1e-30) <= T_RELATIVE || tClose(t, ray.tMin, 1e-6) <= T_RELATIVE;
  if (gpuVisible) {
    // CPU found a blocker the GPU did not.
    const closest = oracle.cpu.closest(ray, { mask });
    const fromWalk = closest ? [{ instance: closest.instanceIndex, triangle: closest.triangleIndex, t: closest.t, u: closest.u, v: closest.v, frontFace: closest.frontFace }] : [];
    const blockers = [...fromWalk, ...oracle.allHits(ray, mask)].filter((h) => h.t >= ray.tMin * (1 - T_RELATIVE) && h.t <= ray.tMax * (1 + T_RELATIVE));
    const grazing = blockers.length > 0 && blockers.every((h) => atEnd(h.t) || oracle.edges.kind(h.instance, h.triangle, h.u, h.v, ray.direction) === 'tangent');
    return grazing ? { verdict: 'grazing' } : { verdict: 'mismatch', why: `gpu visible, cpu blocked by ${blockers.map((h) => `${h.instance}/${h.triangle}@${h.t.toPrecision(6)}(${oracle.edges.kind(h.instance, h.triangle, h.u, h.v, ray.direction)})`).join(' ')}; on mesh box: ${blockers.map((h) => oracle.edges.onBox(h.instance, h.triangle, h.u, h.v)).join(',')}` };
  }
  // GPU blocked where the CPU sees through: the GPU's closest hit is the evidence.
  if (gpuHit.t >= 0 && (atEnd(gpuHit.t) || oracle.edges.kind(gpuHit.instance, gpuHit.triangle, gpuHit.barycentrics[0], gpuHit.barycentrics[1], ray.direction) === 'tangent')) return { verdict: 'grazing' };
  return { verdict: 'mismatch', why: `gpu blocked (closest ${gpuHit.t >= 0 ? `${gpuHit.instance}/${gpuHit.triangle}@${gpuHit.t}` : 'miss'}), cpu visible` };
}

function compareSet(tally: Tally, label: string, scene: TraceTestScene, set: RaySet, oracle: Oracle, gpu: GpuAnswers, surfaceCheck: boolean): void {
  const n = set.count;
  const recordWords = gpu.hitWords.length / n;
  tally.rays += n;
  for (let i = 0; i < n; i++) {
    const ray = unpackRay(set.packed, i);
    const cpuHit = oracle.cpu.closest(ray, { mask: set.mask });
    const cpuBlocked = oracle.cpu.any(ray, { mask: set.mask });
    const gpuHit = gpu.hits[i] as Hit;
    const where = `${label} ${set.name}#${i}`;
    if (cpuHit) tally.hits++;
    if (cpuBlocked) tally.blocked++;
    if (set.mustHit && gpuHit.t < 0) {
      tally.escapes++;
      note(tally, 'escape', `${where}: escaped the closed box`);
    }

    const result = compareClosest(gpuHit, cpuHit, oracle.edges, T_FLOOR * scene.scale, ray, () => oracle.allHits(ray, set.mask), set.name === 'primary');
    if (result.verdict === 'match' && result.tRelative !== undefined) tally.maxTRelative = Math.max(tally.maxTRelative, result.tRelative);
    const perSet = (tally.bySet[set.name] ??= { rays: 0, ties: 0, primaryTie: 0, grazing: 0, tPosition: 0 });
    perSet.rays++;
    if (result.verdict === 'tie' || result.verdict === 'primaryTie' || result.verdict === 'grazing' || result.verdict === 'tPosition') perSet[result.verdict === 'tie' ? 'ties' : result.verdict]++;
    if (set.name === 'primary' && result.verdict !== 'match' && result.verdict !== 'mismatch') {
      tally.primaryDetails.push(`${where} ${result.verdict} (${result.why ?? ''}): gpu ${gpuHit.instance}/${gpuHit.triangle} t=${gpuHit.t} bary ${gpuHit.barycentrics.map((x) => x.toExponential(3))} cpu ${cpuHit ? `${cpuHit.instanceIndex}/${cpuHit.triangleIndex} t=${cpuHit.t} bary ${cpuHit.u.toExponential(3)},${cpuHit.v.toExponential(3)}` : 'miss'} dir ${ray.direction.join(',')}`);
    }
    if (result.verdict === 'tie') tally.ties++;
    else if (result.verdict === 'primaryTie') tally.primaryTie++;
    else if (result.verdict === 'grazing') tally.grazing++;
    else if (result.verdict === 'tPosition') tally.tPosition++;
    else if (result.verdict === 'mismatch') {
      tally.closestMismatch++;
      if (result.slip === 'boxEdge') tally.slipsOnBoxEdge++;
      if (result.slip === 'other') tally.slipsElsewhere++;
      note(tally, 'closest', `${where} closest: ${result.why}`);
    }
    const gpuVisible = gpu.visible[i] !== 0;
    if (gpuVisible === cpuBlocked) {
      const verdict = visibilityVerdict(ray, set.mask, gpuVisible, gpuHit, oracle);
      if (verdict.verdict === 'grazing') tally.grazing++;
      else {
        tally.visibleMismatch++;
        note(tally, 'visible', `${where} visible: ${verdict.why}`);
      }
    }
    // Closest-hit and any-hit on the GPU run the same triangle test: they must agree exactly (a hit inside the interval blocks it).
    if ((gpuHit.t >= 0) === gpuVisible) {
      tally.visibleMismatch++;
      note(tally, 'l0l1', `${where}: L1 ${gpuHit.t >= 0 ? 'hit' : 'missed'} but L0 says ${gpuVisible ? 'visible' : 'blocked'}`);
    }

    // Variants reproduce the base batch bit for bit.
    for (const variant of gpu.closestVariants) {
      const words = variant.name === 'compact' ? 6 : recordWords;
      for (let w = 0; w < words; w++) {
        if (variant.words[i * words + w] !== gpu.hitWords[i * recordWords + w]) {
          tally.variantMismatch++;
          note(tally, 'variant', `${where} closest/${variant.name}: word ${w} differs`);
          break;
        }
      }
      if (variant.widths) {
        const expected = gpuHit.t >= 0 ? CONE.width + CONE.spread * gpuHit.t : 0;
        const got = variant.widths[i] as number;
        if (Math.abs(got - expected) > 1e-6 * Math.max(1, Math.abs(expected))) {
          tally.coneMismatch++;
          note(tally, 'cone', `${where} cone width ${got}, expected ${expected}`);
        }
      }
    }
    for (const variant of gpu.visibleVariants) {
      if (variant.words[i] !== gpu.visible[i]) {
        tally.variantMismatch++;
        note(tally, 'variant', `${where} visible/${variant.name} differs`);
      }
    }

    // Inline module: same answers as the batch, cone width, and rtSurface against the doubles oracle.
    if (gpu.inline) {
      const o = i * 24;
      const w = gpu.inline;
      const same =
        w[o] === gpu.hitWords[i * recordWords] &&
        w[o + 1] === gpuHit.instance &&
        w[o + 2] === gpuHit.triangle &&
        ((w[o + 3] as number) & 1) === (gpuHit.frontFace ? 1 : 0) &&
        (((w[o + 3] as number) >> 1) & 1) === gpu.visible[i] &&
        w[o + 4] === gpu.hitWords[i * recordWords + 4] &&
        w[o + 5] === gpu.hitWords[i * recordWords + 5] &&
        w[o + 6] === gpuHit.material;
      if (!same) {
        tally.inlineMismatch++;
        note(tally, 'inline', `${where} inline differs from batch`);
      }
      const f = new Float32Array(w.buffer, w.byteOffset + o * 4, 24);
      const coneWidth = f[7] as number;
      const expectedCone = gpuHit.t >= 0 ? CONE.width + CONE.spread * gpuHit.t : 0;
      if (Math.abs(coneWidth - expectedCone) > 1e-6 * Math.max(1, Math.abs(expectedCone))) {
        tally.coneMismatch++;
        note(tally, 'cone', `${where} inline cone ${coneWidth} expected ${expectedCone}`);
      }
      if (surfaceCheck && gpuHit.t >= 0) {
        const reference = referenceSurface(scene.meshes, scene.instances as SurfaceInstanceInput[], gpuHit, ray, { positionFromVertices: true });
        const position = [f[12], f[13], f[14]] as number[];
        const shading = [f[16], f[17], f[18]] as number[];
        const uv = [f[20], f[21]] as number[];
        // Barycentric position against the ray's own point: independent of the table and of the decode.
        const rayPoint = [0, 1, 2].map((k) => (ray.origin[k] as number) + (ray.direction[k] as number) * gpuHit.t);
        const positionError = Math.max(...[0, 1, 2].map((k) => Math.abs((position[k] as number) - (rayPoint[k] as number))), ...[0, 1, 2].map((k) => Math.abs((position[k] as number) - (reference.position[k] as number))));
        tally.maxSurfacePositionError = Math.max(tally.maxSurfacePositionError, positionError / scene.scale);
        const cos = shading.reduce((sum, value, k) => sum + value * (reference.shadingNormal[k] as number), 0);
        const uvError = Math.max(Math.abs((uv[0] as number) - reference.uv[0]), Math.abs((uv[1] as number) - reference.uv[1]));
        const uvScale = Math.max(1, Math.abs(reference.uv[0]), Math.abs(reference.uv[1]));
        const geometric = [f[8], f[9], f[10]] as number[];
        const geometricSame = geometric.every((value, k) => value === gpuHit.normal[k]) && (w[o + 11] === 1) === gpuHit.frontFace && w[o + 15] === gpuHit.material;
        if (positionError > 1e-4 * scene.scale || !(cos >= NORMAL_COSINE) || uvError > 1e-4 * uvScale || !geometricSame) {
          tally.surfaceMismatch++;
          note(tally, 'surface', `${where} rtSurface: position err ${positionError.toExponential(2)}, shading cos ${cos}, uv err ${uvError.toExponential(2)}, geometric same ${geometricSame}`);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Scene building
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

function makeAllHits(scene: TraceTestScene, tiny: TinyBvhBuilder): Oracle['allHits'] {
  let bvh: CpuSceneBvh | null = null;
  return (ray, mask) => {
    bvh ??= CpuSceneBvh.build(scene.meshes, scene.instances, (geometry) => buildMeshBvhWithTinyBvh(geometry, tiny));
    return bvh.raycast(ray, { mask }).map((hit) => ({ instance: hit.instance, triangle: hit.triangle, t: hit.t, u: hit.u, v: hit.v, frontFace: hit.frontFace }));
  };
}

function raySets(scene: TraceTestScene, perSet: number, oracle: CpuTracer): RaySet[] {
  const random = createRandom(0x5eed + scene.name.length);
  const sets: RaySet[] = [];
  const add = (name: string, rays: Ray[], mask = ALL, mustHit = false) => {
    if (rays.length) sets.push({ name, packed: packRays(rays), count: rays.length, mask, mustHit });
  };
  const primary = cameraRays(scene, perSet, random);
  add('primary', primary);
  const incoherent = incoherentRays(scene, perSet, random);
  add('incoherent', incoherent);
  add('edges', edgeRays(scene, perSet, random));
  if (scene.closed.length) add('inside', insideRays(scene, perSet, random), ALL, true);
  // Shadow segments start on surfaces the oracle found (packed rays, so the f32 values the GPU sees).
  const source = primary.length ? primary : incoherent;
  const packed = packRays(source);
  const hits = source.map((_, i) => oracle.closest(unpackRay(packed, i)));
  const points = surfacePoints(source.map((_, i) => unpackRay(packed, i)), hits);
  const shadow = shadowRays(scene, points, perSet, random);
  add('shadow', shadow);
  if (scene.emitters.length) add('shadow-masked', shadow, MASK_NO_EMITTERS);
  return sets;
}

// ---------------------------------------------------------------------------------------------------------------

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

try {
  const args = getArgs();
  if (args.packing && args.packing !== 'split' && args.packing !== 'packed') throw new Error(`packing must be split or packed, got '${args.packing}'`);
  PACKING = (args.packing as SceneCopyPacking | undefined) ?? DEFAULT_SCENE_PACKING;
  const sceneNames = (args.scenes ?? 'cornell,quad-light,edges,sponza-white').split(',').filter(Boolean);
  const perSet = numberArg(args, 'rays', 8192);
  const withSceneTrees = (args.sceneTrees ?? '1') !== '0';

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'trace-core' });
  deviceErrors = watchDevice(device);

  const builder = createMeshBvhBuilder();
  const tiny = await createTinyBvhBuilder();
  const kernelsByFormat = new Map<string, Kernels>();
  for (const format of FORMATS) kernelsByFormat.set(format.name, await makeKernels(device, caps, format));
  const argsKernelBytes = rtBatchArgsKernelWgsl(caps).length;

  const metrics: Record<string, number> = {};
  const failures: string[] = [];
  const perScene: Record<string, unknown> = {};
  const workerNotes: Record<string, unknown> = {};
  const total = newTally();
  const addInto = (into: Tally, from: Tally) => {
    for (const key of Object.keys(from) as (keyof Tally)[]) {
      if (key === 'bySet') {
        for (const [name, entry] of Object.entries(from.bySet)) {
          const target = (into.bySet[name] ??= { rays: 0, ties: 0, primaryTie: 0, grazing: 0, tPosition: 0 });
          target.rays += entry.rays;
          target.ties += entry.ties;
          target.primaryTie += entry.primaryTie;
          target.grazing += entry.grazing;
          target.tPosition += entry.tPosition;
        }
      } else if (key === 'primaryDetails') into.primaryDetails.push(...from.primaryDetails)
      else if (key === 'examples') into.examples.push(...from.examples.slice(0, Math.max(0, 40 - into.examples.length)));
      else if (key === 'maxTRelative' || key === 'maxSurfacePositionError') into[key] = Math.max(into[key], from[key]);
      else into[key] += from[key];
    }
  };

  for (const name of sceneNames) {
    const t0 = performance.now();
    const stage = (what: string) => console.log(`[trace-core] ${name}: ${what} at ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    const scene = await loadTraceScene(name);
    stage('loaded');
    const copy = buildSceneCopy({ meshes: scene.meshes, instances: scene.instances as SurfaceInstanceInput[], materials: [], options: { surface: { positions: true } } });
    const surface = uploadSceneCopy(device, copy, { label: `${name} scene`, packing: PACKING });
    const allHits = makeAllHits(scene, tiny);
    const edges = makeEdgeIndex(scene);
    let sets: RaySet[] | null = null;
    const sceneReport: Record<string, unknown> = { triangles: scene.meshes.reduce((sum, mesh, i) => sum + (mesh.indices.length / 3) * scene.instances.filter((inst) => inst.meshIndex === i).length, 0), instances: scene.instances.length };

    for (const format of FORMATS) {
      const data = await buildScene(scene, format, builder);
      stage(`${format.name} built`);
      const cpu = createCpuTracer(data);
      sets ??= raySets(scene, perSet, cpu);
      stage('rays ready');
      const gpuScene = uploadTraceScene(device, data, `${name} ${format.name}`);
      const kernels = kernelsByFormat.get(format.name) as Kernels;
      const tally = newTally();
      for (const set of sets) {
        const answers = await traceOnGpu(device, kernels, kernels, gpuScene, surface, set);
        compareSet(tally, `${name}/${format.name}`, scene, set, { cpu, allHits, edges }, answers, true);
        stage(`${format.name} ${set.name} compared`);
      }
      gpuScene.destroy();
      sceneReport[format.name] = tally;
      addInto(total, tally);
    }

    if (withSceneTrees) {
      // Production path: SceneTrees' own trees (GPU first, tinybvh refinements), default formats.
      // GPU first trees are traced first (held refinements), then tinybvh's refined trees after the swap.
      // Default worker mode ('auto'), but through our own pool so a silent fallback to inline builds shows its reason.
      const pool = createBvhWorkerPool();
      const trees = await SceneTrees.create(device, { label: `${name} SceneTrees`, holdRefinements: true, workers: pool, warn: () => {} });
      const handles = scene.meshes.map((mesh) => trees.addMesh(mesh));
      for (const instance of scene.instances) {
        trees.addInstance({ mesh: handles[instance.meshIndex] as number, objectToWorld: instance.objectToWorld, instanceIndex: instance.instanceIndex, mask: instance.mask, material: instance.material, static: instance.static });
      }
      const frame = () => {
        const encoder = device.createCommandEncoder();
        trees.update(encoder);
        device.queue.submit([encoder.finish()]);
      };
      const nextFrame = () => new Promise((resolve) => setTimeout(resolve, 8));
      const deadline = performance.now() + 120_000;
      const pump = async (until: () => boolean, what: string) => {
        frame();
        while (!until()) {
          if (performance.now() > deadline) throw new Error(`${name}: SceneTrees never got ${what} (counts ${JSON.stringify(trees.counts)})`);
          await nextFrame();
          frame();
        }
        frame();
        await device.queue.onSubmittedWorkDone();
      };
      let refined = false;
      void trees.whenRefined().then(() => (refined = true));
      // bvh: one update() before whenBuilt() starts the builds; one more places what they made.
      frame();
      await trees.whenBuilt();
      frame();
      await device.queue.onSubmittedWorkDone();
      workerNotes[`${name} SceneTrees`] = { usesWorkers: trees.usesWorkers, poolUsesWorkers: pool.usesWorkers, inlineReason: pool.inlineReason };
      stage('SceneTrees first trees');
      const stages: { label: string; counts: unknown }[] = [];
      const runStage = async (label: string) => {
        const buffers = trees.traceBuffers;
        const format = { nodeFormat: buffers.nodeFormat, triangleLeaves: buffers.triangleLeaves, positionLayout: buffers.positionLayout };
        const closest = await RtBatchTracer.create(device, caps, { ...format, mode: 'closest', label: `${label} closest` });
        const visible = await RtBatchTracer.create(device, caps, { ...format, mode: 'visible', label: `${label} visible` });
        // The oracle for this path runs on its own CPU-built scene (different trees: ties are expected and counted).
        const cpu = createCpuTracer(await buildScene(scene, { name: 'oracle', ...format, merge: false }, builder));
        const tally = newTally();
        for (const set of sets ?? []) {
          const answers = await traceOnGpu(device, null, { closest, visible }, buffers, null, set);
          compareSet(tally, `${name}/${label}`, scene, set, { cpu, allHits, edges }, answers, false);
        }
        stage(`${label} compared`);
        sceneReport[label] = { ...tally, format, counts: trees.counts };
        stages.push({ label, counts: trees.counts });
        addInto(total, tally);
        closest.destroy();
        visible.destroy();
      };
      await runStage('SceneTrees-gpu');
      await pump(() => refined, 'its tinybvh trees');
      trees.releaseRefinements();
      await pump(() => trees.counts.onGpuTrees === 0, 'every refinement placed');
      await runStage('SceneTrees-refined');
      trees.dispose();
      pool.dispose();
    }
    surface.destroy();
    perScene[name] = sceneReport;
    await drainDevice(device);
  }

  for (const key of ['rays', 'hits', 'blocked', 'closestMismatch', 'visibleMismatch', 'ties', 'primaryTie', 'grazing', 'tPosition', 'slipsOnBoxEdge', 'slipsElsewhere', 'escapes', 'variantMismatch', 'inlineMismatch', 'surfaceMismatch', 'coneMismatch'] as const) metrics[`${key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`)}`] = total[key];
  metrics.max_t_relative = total.maxTRelative;
  metrics.max_surface_position_error_rel = total.maxSurfacePositionError;
  const kernelBytes = Math.max(...[...kernelsByFormat.values()].flatMap((k) => [k.inline.bytes]), argsKernelBytes);
  metrics.max_kernel_bytes = kernelBytes;

  const failing = ['closestMismatch', 'visibleMismatch', 'escapes', 'variantMismatch', 'inlineMismatch', 'surfaceMismatch', 'coneMismatch'] as const;
  for (const key of failing) if (total[key] > 0) failures.push(`${key}: ${total[key]}`);
  await drainDevice(device);
  if (deviceErrors.length) failures.push(`device errors: ${deviceErrors.join('; ')}`);
  if (failures.length) throw new Error(`trace-core failed: ${failures.join(', ')}\n${total.examples.join('\n')}`);

  reportResult(
    {
      scene: sceneNames.join('+'),
      renderer: 'rt-trace-core',
      config: { packing: PACKING, perSet, formats: FORMATS.map((f) => f.name), sceneTrees: withSceneTrees, caps: { tier: caps.tier, subgroups: caps.subgroups } },
      metrics,
      notes: { primaryDetails: total.primaryDetails, workers: { oracleBuilderUsesWorker: builder.usesWorker, ...workerNotes }, perScene, bySet: total.bySet, examples: total.examples },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
