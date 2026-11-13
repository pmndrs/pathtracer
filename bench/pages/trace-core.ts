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
  type CpuHit,
  type CpuTracer,
  type MeshBvhBuilder,
  type TinyBvhBuilder,
  type TraceSceneBuffers,
  type TraceSceneData,
} from '@pmndrs/bvh';
import {
  RtBatchTracer,
  aosToSoa,
  normalMatrixRows,
  buildSurfaceTable,
  detectCaps,
  packCones,
  readHits,
  referenceSurface,
  requestDeviceForCaps,
  rtBatchArgsKernelWgsl,
  rtModuleWgsl,
  rtSceneBindGroupEntries,
  rtSurfaceBindGroupEntries,
  rtSurfaceLayoutEntries,
  rtWalkEntries,
  unpackRay,
  uploadSurfaceTable,
  withBindings,
  type BatchKernelOptions,
  type Caps,
  type Hit,
  type Ray,
  type SceneFormat,
  type SurfaceBuffers,
  type SurfaceInstanceInput,
} from '@pmndrs/raytracer';
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

const T_RELATIVE = 1e-5;
/** t is compared relative to max(t, T_FLOOR * scene diagonal): f32 error in t scales with the coordinates, not with t. */
const T_FLOOR = 1e-3;
const NORMAL_COSINE = 0.9999;
const CONE = { width: 0.01, spread: 0.002 };
const ALL = 0xffffffff;

interface Format extends Required<SceneFormat> {
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
  /** Ray-set breakdown of the counted (not failed) categories. */
  bySet: Record<string, { rays: number; ties: number; grazing: number; tPosition: number }>;
}
const newTally = (): Tally => ({ rays: 0, closestMismatch: 0, visibleMismatch: 0, ties: 0, grazing: 0, tPosition: 0, slipsOnBoxEdge: 0, slipsElsewhere: 0, escapes: 0, variantMismatch: 0, inlineMismatch: 0, surfaceMismatch: 0, coneMismatch: 0, hits: 0, blocked: 0, maxTRelative: 0, maxSurfacePositionError: 0, examples: [], bySet: {} });

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
  const code = rtModuleWgsl(caps, { ...base, surface: { slot: { group: 2, firstBinding: 0 }, layout: surfaceLayout } }) + INLINE_KERNEL;
  const layouts = [
    device.createBindGroupLayout({ entries: withBindings(rtWalkEntries(base)) }),
    device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    }),
    device.createBindGroupLayout({ entries: rtSurfaceLayoutEntries() }),
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
async function traceOnGpu(device: GPUDevice, kernels: Kernels | null, base: { closest: RtBatchTracer; visible: RtBatchTracer }, scene: TraceSceneBuffers, surface: SurfaceBuffers | null, set: RaySet): Promise<GpuAnswers> {
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
      device.createBindGroup({ layout: kernels.inline.layouts[2] as GPUBindGroupLayout, entries: rtSurfaceBindGroupEntries(surface) }),
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

/**
 * Edge classification. bvh's CPU walk runs the f32-stored scene in double arithmetic, the GPU in f32, so a ray within
 * rounding distance of an edge or vertex may get a different answer. What is allowed depends on the geometry there:
 *  - `crossing`: the ray passes THROUGH the surface at a shared edge or vertex (every triangle around it faces the ray
 *    the same way). One of the triangles must be hit: a miss is a watertightness slip and fails. A different triangle
 *    of the same edge at the same t is a tie (both answers are right): counted, not failed.
 *  - `tangent`: the ray only touches the surface there (a convex edge or corner of a solid seen from outside, or a
 *    boundary edge of an open mesh). Hit or miss are both right within rounding: counted as grazing, not failed.
 * Everything is counted and reported; nothing passes silently.
 */
const EDGE_BARY = 1e-4;

type EdgeKind = 'interior' | 'crossing' | 'tangent';

interface EdgeIndex {
  kind(instance: number, triangle: number, u: number, v: number, direction: readonly number[]): EdgeKind;
  /** The two triangles of one instance share a corner (by position bits). */
  adjacent(instance: number, a: number, b: number): boolean;
  /** How many object-space coordinates of the point lie on the mesh's bounding box (2 = on a box edge, 3 = a corner). */
  onBox(instance: number, triangle: number, u: number, v: number): number;
}

function makeEdgeIndex(scene: TraceTestScene): EdgeIndex {
  const bits = new Uint32Array(1);
  const floats = new Float32Array(bits.buffer);
  const perMesh = scene.meshes.map((mesh) => {
    const key = (vertex: number) => {
      const parts: number[] = [];
      for (let k = 0; k < 3; k++) {
        floats[0] = mesh.positions[vertex * 3 + k] as number;
        parts.push(bits[0] as number);
      }
      return parts.join(',');
    };
    const corner = (t: number, k: number) => key(mesh.indices[t * 3 + k] as number);
    const edgeKey = (x: string, y: string) => (x < y ? `${x}|${y}` : `${y}|${x}`);
    const byEdge = new Map<string, number[]>();
    const byVertex = new Map<string, number[]>();
    const triangles = mesh.indices.length / 3;
    const normals = new Float64Array(triangles * 3);
    for (let t = 0; t < triangles; t++) {
      const p = [0, 1, 2].map((k) => [0, 1, 2].map((c) => mesh.positions[(mesh.indices[t * 3 + k] as number) * 3 + c] as number));
      const e1 = [0, 1, 2].map((c) => (p[1]![c] as number) - (p[0]![c] as number));
      const e2 = [0, 1, 2].map((c) => (p[2]![c] as number) - (p[0]![c] as number));
      normals.set([e1[1]! * e2[2]! - e1[2]! * e2[1]!, e1[2]! * e2[0]! - e1[0]! * e2[2]!, e1[0]! * e2[1]! - e1[1]! * e2[0]!], t * 3);
      for (let k = 0; k < 3; k++) {
        const e = edgeKey(corner(t, k), corner(t, (k + 1) % 3));
        byEdge.set(e, [...(byEdge.get(e) ?? []), t]);
        const v = corner(t, k);
        byVertex.set(v, [...(byVertex.get(v) ?? []), t]);
      }
    }
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i++) {
      lo[i % 3] = Math.min(lo[i % 3] as number, mesh.positions[i] as number);
      hi[i % 3] = Math.max(hi[i % 3] as number, mesh.positions[i] as number);
    }
    return { corner, edgeKey, byEdge, byVertex, normals, lo, hi, mesh };
  });
  const recordOf = (instance: number) => scene.instances.find((candidate) => candidate.instanceIndex === instance);
  return {
    kind(instance, triangle, u, v, direction) {
      const record = recordOf(instance);
      const m = record ? perMesh[record.meshIndex] : undefined;
      if (!record || !m) return 'interior';
      const w = [1 - u - v, u, v];
      const small = [0, 1, 2].filter((k) => (w[k] as number) < EDGE_BARY);
      if (small.length === 0) return 'interior';
      // Triangles around the edge (one small weight) or the vertex (two).
      let around: number[];
      if (small.length === 1) {
        const k = small[0] as number;
        around = m.byEdge.get(m.edgeKey(m.corner(triangle, (k + 1) % 3), m.corner(triangle, (k + 2) % 3))) ?? [triangle];
        if (around.length < 2) return 'tangent'; // boundary edge of an open mesh
      } else {
        const vertex = [0, 1, 2].find((k) => !small.includes(k)) ?? 0;
        around = m.byVertex.get(m.corner(triangle, vertex)) ?? [triangle];
      }
      // Ray direction in object space (inverse of the linear part), then which way each triangle faces it.
      const rows = normalMatrixRows(record.objectToWorld);
      const d = [0, 1, 2].map((r) => (rows[r] as number) * (direction[0] as number) + (rows[3 + r] as number) * (direction[1] as number) + (rows[6 + r] as number) * (direction[2] as number));
      const signs = new Set(around.map((t) => Math.sign(m.normals[t * 3]! * d[0]! + m.normals[t * 3 + 1]! * d[1]! + m.normals[t * 3 + 2]! * d[2]!)));
      if (small.length === 2) {
        // A vertex: crossing only when it is interior to a closed fan facing one way.
        const openFan = [...new Set(around.flatMap((t) => [0, 1, 2].map((k) => m.edgeKey(m.corner(t, k), m.corner(t, (k + 1) % 3)))))].some((e) => (m.byEdge.get(e)?.length ?? 0) < 2 && e.includes(m.corner(triangle, [0, 1, 2].find((k) => !small.includes(k)) ?? 0)));
        if (openFan) return 'tangent';
      }
      return signs.size === 1 && !signs.has(0) ? 'crossing' : 'tangent';
    },
    adjacent(instance, a, b) {
      const record = recordOf(instance);
      const m = record ? perMesh[record.meshIndex] : undefined;
      if (!m) return false;
      const corners = new Set([0, 1, 2].map((k) => m.corner(a, k)));
      return [0, 1, 2].some((k) => corners.has(m.corner(b, k)));
    },
    onBox(instance, triangle, u, v) {
      const record = recordOf(instance);
      const m = record ? perMesh[record.meshIndex] : undefined;
      if (!m) return 0;
      const w = [1 - u - v, u, v];
      let count = 0;
      for (let k = 0; k < 3; k++) {
        let p = 0;
        for (let c = 0; c < 3; c++) p += (w[c] as number) * (m.mesh.positions[(m.mesh.indices[triangle * 3 + c] as number) * 3 + k] as number);
        const extent = Math.max((m.hi[k] as number) - (m.lo[k] as number), 1e-9);
        if (Math.abs(p - (m.lo[k] as number)) < 1e-5 * extent || Math.abs(p - (m.hi[k] as number)) < 1e-5 * extent) count++;
      }
      return count;
    },
  };
}

type Verdict = { verdict: 'match' | 'tie' | 'grazing' | 'tPosition' | 'mismatch'; why?: string; tRelative?: number; slip?: 'boxEdge' | 'other' };

function tClose(a: number, b: number, floor: number): number {
  return Math.abs(a - b) / Math.max(Math.abs(b), floor);
}

/** The GPU's closest hit against the oracle's. t is compared relative to max(t, T_FLOOR * scene scale). */
function compareClosest(gpu: Hit, cpu: CpuHit | null, edges: EdgeIndex, tFloor: number, ray: Ray, allHits: () => ReturnType<AllHits>): Verdict {
  const gpuHit = gpu.t >= 0;
  const gpuEdge = gpuHit ? edges.kind(gpu.instance, gpu.triangle, gpu.barycentrics[0], gpu.barycentrics[1], ray.direction) : 'interior';
  const cpuEdge = cpu ? edges.kind(cpu.instanceIndex, cpu.triangleIndex, cpu.u, cpu.v, ray.direction) : 'interior';
  if (!cpu) {
    if (!gpuHit) return { verdict: 'match' };
    return gpuEdge === 'tangent' ? { verdict: 'grazing' } : { verdict: 'mismatch', why: `gpu hit t=${gpu.t} inst=${gpu.instance} tri=${gpu.triangle} (${gpuEdge}), cpu missed` };
  }
  if (!gpuHit) {
    if (cpuEdge === 'tangent') return { verdict: 'grazing' };
    const box = edges.onBox(cpu.instanceIndex, cpu.triangleIndex, cpu.u, cpu.v);
    return { verdict: 'mismatch', slip: cpuEdge === 'crossing' ? (box >= 2 ? 'boxEdge' : 'other') : undefined, why: `gpu missed, cpu hit t=${cpu.t} inst=${cpu.instanceIndex} tri=${cpu.triangleIndex} bary ${cpu.u.toExponential(3)},${cpu.v.toExponential(3)} (${cpuEdge}${cpuEdge === 'crossing' ? `: watertight slip, ${box} coords on the mesh box` : ''})` };
  }
  const tRelative = tClose(gpu.t, cpu.t, tFloor);
  const sameTriangle = gpu.instance === cpu.instanceIndex && gpu.triangle === cpu.triangleIndex;
  // t further than 1e-5 relative is still the same point when the gap, measured along the surface normal, is within
  // 1e-5 of the coordinates' magnitude: f32 rounding of the (object-space) ray scales with the coordinates, and at
  // grazing angles it moves the hit along the ray. Counted as 'tPosition', never as a plain match.
  const cos = Math.abs(ray.direction[0] * cpu.normal[0] + ray.direction[1] * cpu.normal[1] + ray.direction[2] * cpu.normal[2]);
  const point = Math.hypot(ray.origin[0] + ray.direction[0] * cpu.t, ray.origin[1] + ray.direction[1] * cpu.t, ray.origin[2] + ray.direction[2] * cpu.t);
  const positional = tRelative > T_RELATIVE && Math.abs(gpu.t - cpu.t) * cos <= T_RELATIVE * Math.max(cpu.t, point, tFloor);
  if (tRelative > T_RELATIVE && !positional) {
    // Grazing past a tangent edge onto something behind it, on either side.
    const nearer = gpu.t < cpu.t ? gpuEdge : cpuEdge;
    if (!sameTriangle && nearer === 'tangent') return { verdict: 'grazing', tRelative };
    return { verdict: 'mismatch', why: `t gpu ${gpu.t} cpu ${cpu.t} (rel ${tRelative.toExponential(2)}, cos ${cos.toFixed(3)}, tri gpu ${gpu.instance}/${gpu.triangle} cpu ${cpu.instanceIndex}/${cpu.triangleIndex})`, tRelative };
  }
  if (!sameTriangle) {
    if (cpuEdge !== 'interior' || gpuEdge !== 'interior') return { verdict: 'tie', tRelative };
    // Neighbours at the same t: thin triangles make barycentrics a poor distance to the edge.
    if (gpu.instance === cpu.instanceIndex && edges.adjacent(gpu.instance, gpu.triangle, cpu.triangleIndex)) return { verdict: 'tie', tRelative };
    // Coplanar surfaces (a block standing on the floor): the GPU's triangle really is hit at the same t.
    const twin = allHits().find((h) => h.instance === gpu.instance && h.triangle === gpu.triangle && Math.abs(h.t - gpu.t) * cos <= T_RELATIVE * Math.max(h.t, point, tFloor));
    if (twin) return { verdict: 'tie', tRelative };
    return { verdict: 'mismatch', why: `inst/tri gpu ${gpu.instance}/${gpu.triangle} bary ${gpu.barycentrics.map((x) => x.toExponential(3))} (${gpuEdge}) cpu ${cpu.instanceIndex}/${cpu.triangleIndex} bary ${cpu.u.toExponential(3)},${cpu.v.toExponential(3)} (${cpuEdge}) at t=${cpu.t}`, tRelative };
  }
  if (gpu.frontFace !== cpu.frontFace) return { verdict: 'mismatch', why: `frontFace gpu ${gpu.frontFace} cpu ${cpu.frontFace}`, tRelative };
  if (gpu.material !== cpu.material) return { verdict: 'mismatch', why: `material gpu ${gpu.material} cpu ${cpu.material}`, tRelative };
  const normalCos = gpu.normal[0] * cpu.normal[0] + gpu.normal[1] * cpu.normal[1] + gpu.normal[2] * cpu.normal[2];
  if (!(normalCos >= NORMAL_COSINE)) return { verdict: 'mismatch', why: `normal cos ${normalCos}`, tRelative };
  return { verdict: positional ? 'tPosition' : 'match', tRelative };
}

type AllHits = (ray: Ray, mask: number) => { instance: number; triangle: number; t: number; u: number; v: number; frontFace: boolean }[];

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

    const result = compareClosest(gpuHit, cpuHit, oracle.edges, T_FLOOR * scene.scale, ray, () => oracle.allHits(ray, set.mask));
    if (result.verdict === 'match' && result.tRelative !== undefined) tally.maxTRelative = Math.max(tally.maxTRelative, result.tRelative);
    const perSet = (tally.bySet[set.name] ??= { rays: 0, ties: 0, grazing: 0, tPosition: 0 });
    perSet.rays++;
    if (result.verdict === 'tie' || result.verdict === 'grazing' || result.verdict === 'tPosition') perSet[result.verdict === 'tie' ? 'ties' : result.verdict]++;
    if (result.verdict === 'tie') tally.ties++;
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
    build: (geometry) => builder.build(geometry, { triangleLeaves: format.triangleLeaves, nodeFormat: format.nodeFormat }),
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
  const sceneNames = (args.scenes ?? 'cornell,quad-light,edges,sponza-white').split(',').filter(Boolean);
  const perSet = numberArg(args, 'rays', 8192);
  const withSceneTrees = (args.sceneTrees ?? '1') !== '0';

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'trace-core' });
  deviceErrors = watchDevice(device);

  const builder = createMeshBvhBuilder({ worker: 'never' });
  const tiny = await createTinyBvhBuilder();
  const kernelsByFormat = new Map<string, Kernels>();
  for (const format of FORMATS) kernelsByFormat.set(format.name, await makeKernels(device, caps, format));
  const argsKernelBytes = rtBatchArgsKernelWgsl(caps).length;

  const metrics: Record<string, number> = {};
  const failures: string[] = [];
  const perScene: Record<string, unknown> = {};
  const total = newTally();
  const addInto = (into: Tally, from: Tally) => {
    for (const key of Object.keys(from) as (keyof Tally)[]) {
      if (key === 'bySet') {
        for (const [name, entry] of Object.entries(from.bySet)) {
          const target = (into.bySet[name] ??= { rays: 0, ties: 0, grazing: 0, tPosition: 0 });
          target.rays += entry.rays;
          target.ties += entry.ties;
          target.grazing += entry.grazing;
          target.tPosition += entry.tPosition;
        }
      } else if (key === 'examples') into.examples.push(...from.examples.slice(0, Math.max(0, 40 - into.examples.length)));
      else if (key === 'maxTRelative' || key === 'maxSurfacePositionError') into[key] = Math.max(into[key], from[key]);
      else into[key] += from[key];
    }
  };

  for (const name of sceneNames) {
    const t0 = performance.now();
    const stage = (what: string) => console.log(`[trace-core] ${name}: ${what} at ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    const scene = await loadTraceScene(name);
    stage('loaded');
    const table = buildSurfaceTable(scene.meshes, scene.instances as SurfaceInstanceInput[], { positions: true });
    const surface = uploadSurfaceTable(device, table, `${name} surface`);
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
      const trees = await SceneTrees.create(device, { label: `${name} SceneTrees`, holdRefinements: true, workers: { worker: 'never' }, warn: () => {} });
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
      await pump(() => trees.timeline.firstTreesMs !== null && !trees.building, 'its first trees');
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
    }
    surface.destroy();
    perScene[name] = sceneReport;
    await drainDevice(device);
  }

  for (const key of ['rays', 'hits', 'blocked', 'closestMismatch', 'visibleMismatch', 'ties', 'grazing', 'tPosition', 'slipsOnBoxEdge', 'slipsElsewhere', 'escapes', 'variantMismatch', 'inlineMismatch', 'surfaceMismatch', 'coneMismatch'] as const) metrics[`${key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`)}`] = total[key];
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
      config: { perSet, formats: FORMATS.map((f) => f.name), sceneTrees: withSceneTrees, caps: { tier: caps.tier, subgroups: caps.subgroups } },
      metrics,
      notes: { perScene, bySet: total.bySet, examples: total.examples },
    },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
