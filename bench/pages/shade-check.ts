/**
 * shade-check (P1.3): L2 `rtShade` on the GPU against its TypeScript twin, closed forms and the path tracer. Correctness first; the
 * last section is a best-effort throughput note (the quiet pass owns real timings).
 *
 *  1. Twin: primary rays of Cornell, quad-light and three material variants (plastic, copper, half-metal; with a sun and a sky), random
 *     light samples; every mode (flat, lambert, full). GPU `rtShade` against `shade()` (f64) fed with the GPU's hit and its shadow-ray
 *     answer: emission, direct, light index, occluded flag. Also `rtShadePoint` normals against the hit.
 *  2. Closed forms, Monte Carlo over 2^18 light samples: a Lambert plane point under a quad light (form factor), under a delta sun and
 *     a uniform sky with the uniform and the cosine sky sampler (mean and standard deviation: the A/B for envSampling).
 *  3. Hooks: `cache: 'external'` (an irradiance cache) and `bsdf: 'external'` (the path tracer's `ptBsdfEval`) against the twin with the
 *     same hooks; then builtin vs path tracer BSDF on copper and plastic (an A/B, reported not asserted beyond a loose bound).
 *  4. Path tracer: Cornell mean radiance of `rtShade` (lambert and full, many samples) against `PtMegakernel` with maxDepth 1 (the same
 *     expectation, an independent estimator).
 *  5. Throughput (best effort): Mrays/s of flat / lambert / full over Cornell primary hits, wall clock, no readback inside the timed region.
 *
 * Args: rays=4096 (per twin case)  tol=2e-3  mc=262144 (Monte Carlo samples)  spp=1024 (path-tracer comparison)  perf=1 (0 skips section 5)
 */
import { buildTraceScene, createMeshBvhBuilder, uploadTraceScene, type TraceSceneBuffers } from '@pmndrs/bvh';
import {
  buildSceneCopy,
  detectCaps,
  readMaterial,
  requestDeviceForCaps,
  rtMinifyWgsl,
  rtModuleWgsl,
  rtSceneBindGroupEntries,
  rtSceneCopyBindGroupEntries,
  rtSceneCopyLayoutEntries,
  rtPackSceneAccess,
  type SceneCopyPacking,
  rtWalkEntries,
  sceneFormatOf,
  shade,
  shadeAlbedo,
  shadeDiffuseFromIrradiance,
  shadeMiss,
  shadeTotal,
  stripWgslComments,
  uploadSceneCopy,
  withBindings,
  type Caps,
  type RtShadeMode,
  type RtShadeOptions,
  type SceneCopy,
  type SceneCopyBuffers,
  type SceneCopyInput,
  type ShadeHooks,
  type ShadePoint,
} from '@pmndrs/raytracer';
import { PtMegakernel, ptBsdfAdapterWgsl, ptBsdfEval, ptBsdfFromMaterial, ptBsdfWgsl, ptCameraBasis, ptFrameFromNormal, type PtLookAtCamera } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import { cornell, quadLight } from '../scenes/tier0/index.ts';
import { QUAD_LIGHT } from '../scenes/tier0/quad-light.ts';
import type { SceneDesc } from '../scenes/schema.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const problems: string[] = [];
const metrics: Record<string, number> = {};
const notes: Record<string, unknown> = {};
const note = (m: string) => {
  if (problems.length < 30) problems.push(m);
};

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type V3 = [number, number, number];
type F32 = Float32Array<ArrayBuffer>;
const MODES: RtShadeMode[] = ['flat', 'lambert', 'full'];
const MODE_ID: Record<RtShadeMode, number> = { flat: 0, lambert: 1, full: 2 };
const OUT = 24;
const rel = (a: number, b: number, floor: number): number => Math.abs(a - b) / (Math.max(Math.abs(a), Math.abs(b)) + floor);

// ---- harness ----------------------------------------------------------------------------------------------------------

/** The check kernel: per-ray debug output (`shadeRays`) and a camera accumulator (`shadeAccum`) over `rtShade`. */
const KERNEL = /* wgsl */ `
struct ShParams { count: u32, mode: u32, mask: u32, pad: u32 }
struct ShCam { dims: vec4u, mode: vec4u, origin: vec4f, right: vec4f, up: vec4f, forward: vec4f }
@group(2) @binding(0) var<storage, read> shRays: array<vec4f>;
@group(2) @binding(1) var<storage, read> shU: array<vec4f>;
@group(2) @binding(2) var<storage, read_write> shOut: array<f32>;
@group(2) @binding(3) var<uniform> shParams: ShParams;
@group(2) @binding(4) var<storage, read_write> shAccum: array<vec4f>;
@group(2) @binding(5) var<uniform> shCam: ShCam;

fn shW3(base: u32, v: vec3f) {
    shOut[base] = v.x;
    shOut[base + 1u] = v.y;
    shOut[base + 2u] = v.z;
}

@compute @workgroup_size(64)
fn shadeRays(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= shParams.count) {
        return;
    }
    let a = shRays[i * 2u];
    let b = shRays[i * 2u + 1u];
    let ray = RtRay(a.xyz, a.w, b.xyz, b.w);
    let hit = rtClosest(ray, RT_MASK_ALL);
    let s = rtShade(hit, ray, shParams.mode, shU[i].xyz, shParams.mask);
    let o = i * ${OUT}u;
    shW3(o, s.emission);
    shW3(o + 3u, s.direct);
    shW3(o + 6u, s.indirect);
    shOut[o + 9u] = select(-1.0, f32(s.light), s.light != RT_NONE);
    shOut[o + 10u] = select(0.0, 1.0, s.occluded);
    shOut[o + 11u] = hit.t;
    shOut[o + 12u] = f32(hit.material);
    shOut[o + 13u] = select(0.0, 1.0, hit.frontFace);
    if (hit.t >= 0.0) {
        let p = rtShadePoint(hit, ray);
        shW3(o + 14u, p.position);
        shW3(o + 17u, p.normal);
        shW3(o + 20u, p.geometricNormal);
    }
}

fn shHash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn shRand(seed: u32, k: u32) -> f32 {
    return f32(shHash(seed ^ shHash(k + 0x9e3779b9u)) >> 8u) * 0.000000059604644775390625;
}

@compute @workgroup_size(8, 8)
fn shadeAccum(@builtin(global_invocation_id) id: vec3u) {
    let dims = shCam.dims;
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let index = id.y * dims.x + id.x;
    let seed = shHash(index * 2654435761u + dims.w);
    let k = dims.z * 8u;
    let jx = shRand(seed, k);
    let jy = shRand(seed, k + 1u);
    let u = vec3f(shRand(seed, k + 2u), shRand(seed, k + 3u), shRand(seed, k + 4u));
    let ndc = vec2f(((f32(id.x) + jx) / f32(dims.x)) * 2.0 - 1.0, 1.0 - ((f32(id.y) + jy) / f32(dims.y)) * 2.0);
    let ray = RtRay(shCam.origin.xyz, 0.0, normalize(shCam.forward.xyz + shCam.right.xyz * ndc.x + shCam.up.xyz * ndc.y), 1.0e30);
    let hit = rtClosest(ray, RT_MASK_ALL);
    let s = rtShade(hit, ray, shCam.mode.x, u, shCam.mode.y);
    let l = rtShadeTotal(s);
    let y = dot(l, vec3f(0.2126, 0.7152, 0.0722));
    shAccum[index] = shAccum[index] + vec4f(l, y * y);
}
`;

/** The path tracer's BSDF as the `bsdf: 'external'` hook: `ptBsdfEval(ptBsdfFromMaterial(m), wo, wi).f` in the shading frame. */
const PT_BSDF_HOOK = /* wgsl */ `
fn rtShadeBsdf(p: RtShadePoint, wi: vec3f) -> vec3f {
    let frame = ptFrameFromNormal(p.normal);
    return ptBsdfEval(ptBsdfFromMaterial(p.material), ptToLocal(frame, p.wo), ptToLocal(frame, wi)).f;
}
`;

/** An irradiance cache stand-in: E = (1, 2, 3) * (0.5 + 0.5 * n.y), turned into radiance by the helper. */
const CACHE_HOOK = /* wgsl */ `
fn rtShadeIndirect(p: RtShadePoint) -> vec3f {
    return rtShadeDiffuseFromIrradiance(p, vec3f(1.0, 2.0, 3.0) * (0.5 + 0.5 * p.normal.y));
}
`;
const cacheHookTs = (p: ShadePoint): V3 => shadeDiffuseFromIrradiance(p, [1 * (0.5 + 0.5 * p.normal[1]), 2 * (0.5 + 0.5 * p.normal[1]), 3 * (0.5 + 0.5 * p.normal[1])]);
const ptBsdfHookTs = (p: ShadePoint, wi: V3): V3 => {
  const frame = ptFrameFromNormal(p.normal);
  const local = (v: V3): V3 => [v[0] * frame.t[0] + v[1] * frame.t[1] + v[2] * frame.t[2], v[0] * frame.b[0] + v[1] * frame.b[1] + v[2] * frame.b[2], v[0] * frame.n[0] + v[1] * frame.n[1] + v[2] * frame.n[2]];
  return [...ptBsdfEval(ptBsdfFromMaterial(p.material), local(p.wo), local(wi)).f] as V3;
};

interface HarnessOptions {
  name: string;
  shade?: RtShadeOptions;
  extraWgsl?: string;
}

/** Scene copy binding (P3.11, page arg `packing=split|packed`, default split): every harness of the page uses it. */
let PACKING: SceneCopyPacking = 'split';

class Harness {
  readonly device: GPUDevice;
  readonly caps: Caps;
  readonly input: SceneCopyInput;
  readonly copy: SceneCopy;
  readonly trace: TraceSceneBuffers;
  readonly buffers: SceneCopyBuffers;
  readonly wgslBytes: number;
  readonly walkGroup: GPUBindGroup;
  readonly sceneGroup: GPUBindGroup;
  private readonly layouts: GPUBindGroupLayout[];
  private readonly rays: GPUComputePipeline;
  private readonly accum: GPUComputePipeline;
  private readonly format: ReturnType<typeof sceneFormatOf>;

  private constructor(device: GPUDevice, caps: Caps, input: SceneCopyInput, copy: SceneCopy, trace: TraceSceneBuffers, buffers: SceneCopyBuffers, code: string, layouts: GPUBindGroupLayout[], rays: GPUComputePipeline, accum: GPUComputePipeline) {
    this.device = device;
    this.caps = caps;
    this.input = input;
    this.copy = copy;
    this.trace = trace;
    this.buffers = buffers;
    this.wgslBytes = code.length;
    this.layouts = layouts;
    this.rays = rays;
    this.accum = accum;
    this.format = sceneFormatOf(trace);
    this.walkGroup = device.createBindGroup({ layout: layouts[0] as GPUBindGroupLayout, entries: rtSceneBindGroupEntries(trace) });
    this.sceneGroup = device.createBindGroup({ layout: layouts[1] as GPUBindGroupLayout, entries: rtSceneCopyBindGroupEntries(buffers) });
  }

  static async create(device: GPUDevice, caps: Caps, input: SceneCopyInput, o: HarnessOptions): Promise<Harness> {
    const builder = createMeshBvhBuilder();
    const data = await buildTraceScene(
      input.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : null })),
      input.instances.map((instance) => ({ ...instance, static: true })),
      { build: (geometry) => builder.build(geometry), mergeStaticInstances: true },
    );
    builder.dispose();
    const trace = uploadTraceScene(device, data, o.name);
    const copy = buildSceneCopy({ ...input, options: { ...input.options, surface: { ...input.options?.surface, positions: true } } });
    const buffers = uploadSceneCopy(device, copy, { label: `${o.name} scene`, packing: PACKING });
    const format = sceneFormatOf(trace);
    const module = rtModuleWgsl(caps, { ...format, rayMasks: false, traceSlot: { group: 0, firstBinding: 0 }, scene: { slot: { group: 1, firstBinding: 0 }, layout: copy.layout, packing: PACKING }, shade: o.shade ?? true });
    const ptParts = o.shade?.bsdf === 'external' ? `${stripWgslComments(ptBsdfWgsl({}))}\n${ptBsdfAdapterWgsl()}\n` : '';
    const code = rtMinifyWgsl(PACKING === 'packed' ? rtPackSceneAccess(`${module}\n${ptParts}${o.extraWgsl ?? ''}\n${KERNEL}`) : `${module}\n${ptParts}${o.extraWgsl ?? ''}\n${KERNEL}`);
    const COMPUTE = 0x4;
    const layouts = [
      device.createBindGroupLayout({ entries: withBindings(rtWalkEntries(format)) }),
      device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, undefined, PACKING) }),
      device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
          { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
          { binding: 3, visibility: COMPUTE, buffer: { type: 'uniform' } },
          { binding: 4, visibility: COMPUTE, buffer: { type: 'storage' } },
          { binding: 5, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
        ],
      }),
    ];
    const shaderModule = device.createShaderModule({ label: o.name, code });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`WGSL (${o.name}) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n')}`);
    const layout = device.createPipelineLayout({ bindGroupLayouts: layouts });
    const rays = await device.createComputePipelineAsync({ label: `${o.name} rays`, layout, compute: { module: shaderModule, entryPoint: 'shadeRays' } });
    const accum = await device.createComputePipelineAsync({ label: `${o.name} accum`, layout, compute: { module: shaderModule, entryPoint: 'shadeAccum' } });
    return new Harness(device, caps, input, copy, trace, buffers, code, layouts, rays, accum);
  }

  private io(entries: GPUBindGroupEntry[]): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.layouts[2] as GPUBindGroupLayout, entries });
  }

  private storage(data: ArrayBufferView<ArrayBuffer>, extra = 0): GPUBuffer {
    const b = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  /** Shades a ray list. `rays`: 8 floats per ray (origin, tMin, direction, tMax); `u`: 4 per ray. Returns `OUT` floats per ray. */
  async shadeRays(rays: F32, u: F32, mode: RtShadeMode, mask = 0xff): Promise<F32> {
    const count = rays.length / 8;
    const d = this.device;
    const raysBuf = this.storage(rays);
    const uBuf = this.storage(u);
    const outBuf = this.storage(new Float32Array(count * OUT), GPUBufferUsage.COPY_SRC);
    const accumBuf = this.storage(new Float32Array(4));
    const params = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(params, 0, new Uint32Array([count, MODE_ID[mode], mask, 0]));
    const cam = d.createBuffer({ size: 256 + 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const group = this.io([
      { binding: 0, resource: { buffer: raysBuf } },
      { binding: 1, resource: { buffer: uBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: accumBuf } },
      { binding: 5, resource: { buffer: cam, size: 112 } },
    ]);
    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.rays);
    pass.setBindGroup(0, this.walkGroup);
    pass.setBindGroup(1, this.sceneGroup);
    pass.setBindGroup(2, group, [0]);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    d.queue.submit([encoder.finish()]);
    const result: F32 = new Float32Array(await readBack(d, outBuf, count * OUT * 4));
    for (const b of [raysBuf, uBuf, outBuf, accumBuf, params, cam]) b.destroy();
    return result;
  }

  /**
   * Camera accumulation: `passes` samples per pixel into one f32 buffer (rgb sum, w = sum of squared luminance). Returns the per-pixel
   * mean and the standard error of the luminance. `timed`: the wall clock of the dispatches only (readback after).
   */
  async accumulate(camera: PtLookAtCamera, width: number, height: number, mode: RtShadeMode, passes: number, seed = 1): Promise<{ mean: Float32Array; stdError: Float32Array; wallMs: number }> {
    const d = this.device;
    const basis = ptCameraBasis(camera, width, height);
    const stride = 256;
    const slots = new ArrayBuffer(stride * passes);
    for (let s = 0; s < passes; s++) {
      const u = new Uint32Array(slots, s * stride, 8);
      const f = new Float32Array(slots, s * stride + 32, 20);
      u.set([width, height, s, seed, MODE_ID[mode], 0xff, 0, 0]);
      f.set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0]);
    }
    const cam = d.createBuffer({ size: stride * passes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(cam, 0, slots);
    const accum = this.storage(new Float32Array(width * height * 4), GPUBufferUsage.COPY_SRC);
    const dummy = this.storage(new Float32Array(4));
    const dummyOut = this.storage(new Float32Array(4));
    const dummyParams = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const group = this.io([
      { binding: 0, resource: { buffer: dummy } },
      { binding: 1, resource: { buffer: dummy } },
      { binding: 2, resource: { buffer: dummyOut } },
      { binding: 3, resource: { buffer: dummyParams } },
      { binding: 4, resource: { buffer: accum } },
      { binding: 5, resource: { buffer: cam, size: 112 } },
    ]);
    const encoder = d.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.accum);
    pass.setBindGroup(0, this.walkGroup);
    pass.setBindGroup(1, this.sceneGroup);
    for (let s = 0; s < passes; s++) {
      pass.setBindGroup(2, group, [s * stride]);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    }
    pass.end();
    await d.queue.onSubmittedWorkDone();
    const start = performance.now();
    d.queue.submit([encoder.finish()]);
    await d.queue.onSubmittedWorkDone();
    const wallMs = performance.now() - start;
    const sums = new Float32Array(await readBack(d, accum, width * height * 16));
    const mean = new Float32Array(width * height * 4);
    const stdError = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      for (let c = 0; c < 3; c++) mean[i * 4 + c] = (sums[i * 4 + c] as number) / passes;
      mean[i * 4 + 3] = 1;
      const y = 0.2126 * (mean[i * 4] as number) + 0.7152 * (mean[i * 4 + 1] as number) + 0.0722 * (mean[i * 4 + 2] as number);
      const variance = Math.max(0, (sums[i * 4 + 3] as number) / passes - y * y);
      stdError[i] = Math.sqrt(variance / passes);
    }
    for (const b of [cam, accum, dummy, dummyOut, dummyParams]) b.destroy();
    return { mean, stdError, wallMs };
  }

  destroy(): void {
    this.trace.destroy();
    this.buffers.destroy();
  }
}

async function readBack(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}

// ---- scenes ----------------------------------------------------------------------------------------------------------

function withLights(input: SceneCopyInput): SceneCopyInput {
  return { ...input, sun: { direction: [-0.3, -0.8, -0.2], irradiance: [20, 18, 15], angularRadius: 0.03 }, environment: { radiance: [0.3, 0.4, 0.5], scale: 1 } };
}

/** Quad-light with the floor (material 0) replaced. */
function quadWithFloor(floor: Record<string, unknown>): SceneCopyInput {
  const base = sceneDescToCopyInput(quadLight);
  const materials = [...base.materials];
  materials[0] = { ...materials[0], ...floor };
  return withLights({ meshes: base.meshes, instances: base.instances, materials, options: base.options });
}

const SCENES: { name: string; desc: SceneDesc; input: SceneCopyInput }[] = [
  { name: 'cornell', desc: cornell, input: sceneDescToCopyInput(cornell) },
  { name: 'quad-light', desc: quadLight, input: sceneDescToCopyInput(quadLight) },
  { name: 'quad-plastic', desc: quadLight, input: quadWithFloor({ baseColor: [0.6, 0.2, 0.2], specularWeight: 1, specularRoughness: 0.3, ior: 1.5 }) },
  { name: 'quad-copper', desc: quadLight, input: quadWithFloor({ conductorEtaK: true, conductorEta: [0.2, 0.92, 1.1], conductorK: [3.9, 2.45, 2.2], metalness: 1, specularRoughness: 0.35, specularWeight: 1 }) },
  { name: 'quad-halfmetal', desc: quadLight, input: quadWithFloor({ baseColor: [0.8, 0.6, 0.3], metalness: 0.5, specularRoughness: 0.5, specularWeight: 1 }) },
];

/** Primary rays of a scene's camera on a width x height grid, jittered by `rand`. */
function primaryRays(desc: SceneDesc, count: number, rand: () => number): { rays: F32; u: F32 } {
  const side = Math.ceil(Math.sqrt(count));
  const camera = desc.camera as unknown as PtLookAtCamera;
  const basis = ptCameraBasis(camera, side, side);
  const rays = new Float32Array(count * 8);
  const u = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    const px = (i % side) + rand();
    const py = Math.floor(i / side) + rand();
    const nx = (px / side) * 2 - 1;
    const ny = 1 - (py / side) * 2;
    const d = [0, 1, 2].map((c) => basis.forward[c]! + basis.right[c]! * nx + basis.up[c]! * ny);
    const len = Math.hypot(d[0]!, d[1]!, d[2]!);
    rays.set([...basis.origin, 0, d[0]! / len, d[1]! / len, d[2]! / len, 1e30], i * 8);
    u.set([rand(), rand(), rand(), 0], i * 4);
  }
  return { rays, u };
}

// ---- 1 + 3. twin -----------------------------------------------------------------------------------------------------

interface TwinResult {
  mismatches: number;
  flips: number;
  maxRel: number;
  hits: number;
  rays: number;
}

function compareToTwin(label: string, h: Harness, rays: Float32Array, u: Float32Array, out: Float32Array, mode: RtShadeMode, hooks: ShadeHooks, options: RtShadeOptions, tol: number): TwinResult {
  const count = rays.length / 8;
  const r: TwinResult = { mismatches: 0, flips: 0, maxRel: 0, hits: 0, rays: count };
  let scale = 0;
  for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) scale = Math.max(scale, Math.abs(out[i * OUT + 3 + c] as number), Math.abs(out[i * OUT + c] as number));
  const floor = Math.max(1e-6, scale * 1e-5);
  const bad = (what: string, i: number, g: number, c: number) => {
    r.mismatches++;
    note(`${label} ${mode} ray ${i} ${what}: gpu ${g} twin ${c}`);
  };
  for (let i = 0; i < count; i++) {
    const o = i * OUT;
    const g = out.subarray(o, o + OUT);
    const dir: V3 = [rays[i * 8 + 4] as number, rays[i * 8 + 5] as number, rays[i * 8 + 6] as number];
    const uu: V3 = [u[i * 4] as number, u[i * 4 + 1] as number, u[i * 4 + 2] as number];
    if ((g[11] as number) < 0) {
      const miss = shadeMiss(h.copy, dir, { sunVisible: options.sunVisible ?? false });
      for (let c = 0; c < 3; c++) {
        const e = rel(g[c] as number, miss[c] as number, floor);
        r.maxRel = Math.max(r.maxRel, e);
        if (e > tol) bad(`miss emission[${c}]`, i, g[c] as number, miss[c] as number);
      }
      continue;
    }
    r.hits++;
    const material = readMaterial(h.copy, g[12] as number);
    const p: ShadePoint = {
      position: [g[14] as number, g[15] as number, g[16] as number],
      normal: [g[17] as number, g[18] as number, g[19] as number],
      geometricNormal: [g[20] as number, g[21] as number, g[22] as number],
      wo: [-dir[0], -dir[1], -dir[2]],
      frontFace: g[13] === 1,
      material,
      albedo: shadeAlbedo(material),
    };
    // Normals must face the viewer.
    if (p.normal[0] * p.wo[0] + p.normal[1] * p.wo[1] + p.normal[2] * p.wo[2] < -1e-3 && p.frontFace) bad('shading normal faces away', i, 0, 0);
    const gpuOccluded = g[10] === 1;
    const t = shade(h.copy, p, mode, uu, () => !gpuOccluded, hooks, options);
    const gLight = (g[9] as number) < 0 ? 0xffffffff : (g[9] as number);
    if (gLight !== t.light) {
      // A sample whose light weight sits at the f32 noise floor may flip between "contributes" and "does not": counted, not compared.
      r.flips++;
      continue;
    }
    if (gpuOccluded !== t.occluded) {
      r.flips++;
      continue;
    }
    const gv = [g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8]] as number[];
    const tv = [...t.emission, ...t.direct, ...t.indirect];
    for (let c = 0; c < 9; c++) {
      const e = rel(gv[c] as number, tv[c] as number, floor);
      r.maxRel = Math.max(r.maxRel, e);
      if (e > tol || !Number.isFinite(gv[c] as number)) bad(['emission', 'direct', 'indirect'][Math.floor(c / 3)] + `[${c % 3}]`, i, gv[c] as number, tv[c] as number);
    }
  }
  return r;
}

async function checkTwin(device: GPUDevice, caps: Caps, count: number, tol: number): Promise<void> {
  for (const scene of SCENES) {
    for (const envSampling of scene.name === 'cornell' ? (['uniform'] as const) : (['uniform', 'cosine'] as const)) {
      const options: RtShadeOptions = { envSampling };
      const h = await Harness.create(device, caps, scene.input, { name: `${scene.name} ${envSampling}`, shade: options });
      const rand = rng(77);
      const { rays, u } = primaryRays(scene.desc, count, rand);
      for (const mode of MODES) {
        const out = await h.shadeRays(rays, u, mode);
        const r = compareToTwin(`${scene.name}/${envSampling}`, h, rays, u, out, mode, {}, options, tol);
        const key = `twin_${scene.name.replace('-', '_')}_${envSampling}_${mode}`;
        metrics[`${key}_mismatch`] = r.mismatches;
        metrics[`${key}_flips`] = r.flips;
        metrics[`${key}_max_rel`] = r.maxRel;
        metrics[`${key}_hits`] = r.hits;
      }
      if (envSampling === 'uniform') metrics[`kernel_bytes_${scene.name.replace('-', '_')}`] = h.wgslBytes;
      h.destroy();
    }
  }
  // flat with the albedo content.
  {
    const scene = SCENES[2] as (typeof SCENES)[number];
    const options: RtShadeOptions = { flat: 'albedo' };
    const h = await Harness.create(device, caps, scene.input, { name: 'flat albedo', shade: options });
    const { rays, u } = primaryRays(scene.desc, Math.min(count, 1024), rng(5));
    const out = await h.shadeRays(rays, u, 'flat');
    const r = compareToTwin('flat-albedo', h, rays, u, out, 'flat', {}, options, tol);
    metrics.twin_flat_albedo_mismatch = r.mismatches;
    h.destroy();
  }
  // Hooks: irradiance cache + the path tracer's BSDF.
  for (const scene of SCENES.filter((s) => s.name !== 'cornell')) {
    const options: RtShadeOptions = { bsdf: 'external', cache: 'external' };
    const h = await Harness.create(device, caps, scene.input, { name: `${scene.name} hooks`, shade: options, extraWgsl: `${PT_BSDF_HOOK}${CACHE_HOOK}` });
    const { rays, u } = primaryRays(scene.desc, count, rng(91));
    for (const mode of MODES) {
      const out = await h.shadeRays(rays, u, mode);
      const r = compareToTwin(`${scene.name}/hooks`, h, rays, u, out, mode, { bsdf: ptBsdfHookTs, indirect: cacheHookTs }, options, tol * 5);
      const key = `hooks_${scene.name.replace('-', '_')}_${mode}`;
      metrics[`${key}_mismatch`] = r.mismatches;
      metrics[`${key}_max_rel`] = r.maxRel;
      metrics[`${key}_flips`] = r.flips;
    }
    h.destroy();
  }
}

// ---- 3b. builtin vs pt BSDF ------------------------------------------------------------------------------------------

/**
 * Mean over the hits of |builtin - pt| / pt (direct term), same rays and the same light samples. An A/B, not a check. Half metal is left out:
 * the path tracer's material adapter thresholds metalness at 0.5 (a diffuse or a conductor, nothing between), the builtin BSDF mixes.
 */
async function abBsdf(device: GPUDevice, caps: Caps, count: number): Promise<void> {
  for (const scene of SCENES.filter((s) => s.name !== 'cornell' && s.name !== 'quad-halfmetal')) {
    const builtin = await Harness.create(device, caps, scene.input, { name: `${scene.name} builtin`, shade: {} });
    const external = await Harness.create(device, caps, scene.input, { name: `${scene.name} pt`, shade: { bsdf: 'external' }, extraWgsl: PT_BSDF_HOOK });
    const { rays, u } = primaryRays(scene.desc, count, rng(123));
    const a = await builtin.shadeRays(rays, u, 'full');
    const b = await external.shadeRays(rays, u, 'full');
    let sumA = 0;
    let sumB = 0;
    let sumAbs = 0;
    for (let i = 0; i < count; i++) {
      if ((a[i * OUT + 11] as number) < 0) continue;
      const ya = (a[i * OUT + 3] as number) + (a[i * OUT + 4] as number) + (a[i * OUT + 5] as number);
      const yb = (b[i * OUT + 3] as number) + (b[i * OUT + 4] as number) + (b[i * OUT + 5] as number);
      sumA += ya;
      sumB += yb;
      sumAbs += Math.abs(ya - yb);
    }
    const key = `ab_bsdf_${scene.name.replace('-', '_')}`;
    metrics[`${key}_mean_builtin`] = sumA / count;
    metrics[`${key}_mean_pt`] = sumB / count;
    metrics[`${key}_mean_ratio`] = sumA / sumB;
    metrics[`${key}_abs_diff_over_mean`] = sumAbs / Math.max(sumB, 1e-12);
    builtin.destroy();
    external.destroy();
  }
  // Copper: Schlick vs exact complex Fresnel is the only intended difference; expect a few percent at most.
  const ratio = metrics.ab_bsdf_quad_copper_mean_ratio as number;
  if (!(Math.abs(ratio - 1) < 0.06)) {
    metrics.ab_copper_mismatch = 1;
    note(`builtin vs pt on copper: mean ratio ${ratio}`);
  } else metrics.ab_copper_mismatch = 0;
}

// ---- 2. closed forms -----------------------------------------------------------------------------------------------

function centredFormFactor(a: number, b: number, h: number): number {
  const corner = (x: number, y: number) => (1 / (2 * Math.PI)) * ((x / Math.sqrt(1 + x * x)) * Math.atan(y / Math.sqrt(1 + x * x)) + (y / Math.sqrt(1 + y * y)) * Math.atan(x / Math.sqrt(1 + y * y)));
  return 4 * corner(a / 2 / h, b / 2 / h);
}

function stats(values: Float32Array, offset: number, count: number): { mean: number; std: number } {
  let sum = 0;
  let sq = 0;
  for (let i = 0; i < count; i++) {
    const v = values[i * OUT + offset] as number;
    sum += v;
    sq += v * v;
  }
  const mean = sum / count;
  return { mean, std: Math.sqrt(Math.max(0, sq / count - mean * mean)) };
}

/** `count` rays from `from` through `point` (each with its own random light sample numbers). */
function pointRays(count: number, point: V3, from: V3, rand: () => number): { rays: F32; u: F32 } {
  const d: V3 = [point[0] - from[0], point[1] - from[1], point[2] - from[2]];
  const len = Math.hypot(...d);
  const rays = new Float32Array(count * 8);
  const u = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    rays.set([...from, 0, d[0] / len, d[1] / len, d[2] / len, 1e30], i * 8);
    u.set([rand(), rand(), rand(), 0], i * 4);
  }
  return { rays, u };
}

async function checkClosedForms(device: GPUDevice, caps: Caps, count: number): Promise<void> {
  // (a) quad light.
  {
    const h = await Harness.create(device, caps, sceneDescToCopyInput(quadLight), { name: 'quad closed form' });
    const { rays, u } = pointRays(count, [0, 0, 0], [0, 0.5, 0], rng(2024));
    const expected = QUAD_LIGHT.albedo * centredFormFactor(QUAD_LIGHT.size, QUAD_LIGHT.size, QUAD_LIGHT.height) * QUAD_LIGHT.radiance;
    for (const mode of ['lambert', 'full'] as const) {
      const out = await h.shadeRays(rays, u, mode);
      let sum = 0;
      for (let i = 0; i < count; i++) sum += (out[i * OUT + 3] as number) + (out[i * OUT] as number);
      const mean = sum / count;
      metrics[`closed_quad_${mode}_mean`] = mean;
      metrics[`closed_quad_${mode}_expected`] = expected;
      metrics[`closed_quad_${mode}_rel_dev`] = mean / expected - 1;
      const bad = Math.abs(mean / expected - 1) > 0.01;
      metrics[`closed_quad_${mode}_mismatch`] = bad ? 1 : 0;
      if (bad) note(`quad closed form ${mode}: mean ${mean} expected ${expected}`);
    }
    h.destroy();
  }
  // (b) sun + sky.
  const sunDir: V3 = [-0.5, -0.7, -0.3];
  const E = 3;
  const Ls = 0.2;
  const albedo = 0.5;
  const cosTheta = -sunDir[1] / Math.hypot(...sunDir);
  const expected = (albedo / Math.PI) * E * cosTheta + albedo * Ls;
  const planeDesc = (radius: number): SceneDesc => ({
    version: 1,
    name: 'shade-sun-plane',
    purpose: 'Lambertian plane under a sun and a uniform sky.',
    tier: 0,
    camera: { type: 'perspective', position: [0, 3, 0.5], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: 64, height: 64, filter: 'box' },
    integrator: { maxDepth: 1 },
    materials: { floor: { type: 'diffuse', reflectance: [albedo, albedo, albedo] } },
    meshes: [{ name: 'plane', geometry: { type: 'quad', size: [40, 40] }, material: 'floor' }],
    lights: [{ type: 'directional', direction: sunDir, irradiance: [E, E, E], angularRadius: radius }],
    environment: { type: 'uniform', radiance: [Ls, Ls, Ls] },
  });
  const sdev: Record<string, number> = {};
  for (const radius of [0, 0.05]) {
    for (const envSampling of ['uniform', 'cosine'] as const) {
      const h = await Harness.create(device, caps, sceneDescToCopyInput(planeDesc(radius)), { name: `sun-plane r${radius} ${envSampling}`, shade: { envSampling } });
      const { rays, u } = pointRays(count, [0, 0, 0], [0, 1, 0], rng(99));
      const out = await h.shadeRays(rays, u, 'lambert');
      const s = stats(out, 3, count);
      const key = `closed_sun_plane_r${radius === 0 ? '0' : '005'}_${envSampling}`;
      metrics[`${key}_mean`] = s.mean;
      metrics[`${key}_std`] = s.std;
      metrics[`${key}_expected`] = expected;
      metrics[`${key}_rel_dev`] = s.mean / expected - 1;
      // 4 sigma of the sample mean, at least 1%.
      const tolerance = Math.max(0.01, (4 * s.std) / Math.sqrt(count) / expected);
      const bad = Math.abs(s.mean / expected - 1) > tolerance;
      metrics[`${key}_mismatch`] = bad ? 1 : 0;
      if (bad) note(`${key}: mean ${s.mean} expected ${expected} (tolerance ${tolerance})`);
      sdev[`${radius}_${envSampling}`] = s.std;
      h.destroy();
    }
  }
  metrics.closed_sun_plane_std_ratio_uniform_over_cosine = (sdev['0.05_uniform'] as number) / (sdev['0.05_cosine'] as number);
}

// ---- 4. path tracer -------------------------------------------------------------------------------------------------

async function checkAgainstPathTracer(device: GPUDevice, caps: Caps, spp: number): Promise<void> {
  const desc = cornell;
  const width = 64;
  const height = 64;
  const input = sceneDescToCopyInput(desc);
  const h = await Harness.create(device, caps, input, { name: 'cornell vs pt', shade: {} });
  const camera = desc.camera as unknown as PtLookAtCamera;
  const lambert = await h.accumulate(camera, width, height, 'lambert', spp, 3);
  const full = await h.accumulate(camera, width, height, 'full', spp, 4);

  const pt = await PtMegakernel.create(device, caps, { format: sceneFormatOf(h.trace), layout: h.copy.layout, scenePacking: PACKING, integrator: {}, label: 'pt maxDepth 1' });
  pt.setScene(h.trace, h.buffers);
  pt.setFilm(width, height);
  pt.setCamera(camera);
  pt.setLimits({ maxDepth: 1 });
  pt.setSeed(11);
  {
    const encoder = device.createCommandEncoder();
    pt.reset(encoder);
    device.queue.submit([encoder.finish()]);
  }
  for (let done = 0; done < spp; ) {
    const n = Math.min(64, spp - done);
    const encoder = device.createCommandEncoder();
    pt.encode(encoder, n);
    device.queue.submit([encoder.finish()]);
    done += n;
    await device.queue.onSubmittedWorkDone();
  }
  const reference = await pt.readImage();

  for (const [name, image] of [['lambert', lambert], ['full', full]] as const) {
    let sumA = 0;
    let sumB = 0;
    for (let i = 0; i < width * height; i++) {
      sumA += (image.mean[i * 4 + 1] as number);
      sumB += (reference.data[i * 4 + 1] as number);
    }
    // Per-pixel agreement in standard errors (shade's own standard error stands in for both: the path tracer's MIS has lower variance).
    let z2 = 0;
    let zMax = 0;
    let n = 0;
    for (let i = 0; i < width * height; i++) {
      const ya = 0.2126 * (image.mean[i * 4] as number) + 0.7152 * (image.mean[i * 4 + 1] as number) + 0.0722 * (image.mean[i * 4 + 2] as number);
      const yb = 0.2126 * (reference.data[i * 4] as number) + 0.7152 * (reference.data[i * 4 + 1] as number) + 0.0722 * (reference.data[i * 4 + 2] as number);
      const se = Math.sqrt(2) * (image.stdError[i] as number);
      if (se > 1e-6) {
        const z = (ya - yb) / se;
        z2 += z * z;
        zMax = Math.max(zMax, Math.abs(z));
        n++;
      } else if (Math.abs(ya - yb) > 1e-4 * (1 + yb)) {
        z2 += 100;
        n++;
      }
    }
    const meanRel = sumA / sumB - 1;
    metrics[`pt_cornell_${name}_mean_rel_dev`] = meanRel;
    metrics[`pt_cornell_${name}_rms_z`] = Math.sqrt(z2 / Math.max(1, n));
    metrics[`pt_cornell_${name}_max_z`] = zMax;
    // Lambert and full share the Lambertian Cornell; the global mean must agree within 0.5%.
    const bad = Math.abs(meanRel) > 0.005 || Math.sqrt(z2 / Math.max(1, n)) > 1.6;
    metrics[`pt_cornell_${name}_mismatch`] = bad ? 1 : 0;
    if (bad) note(`pt cornell ${name}: mean rel dev ${meanRel}, rms z ${Math.sqrt(z2 / Math.max(1, n))}`);
  }
  pt.destroy();
  h.destroy();
}

// ---- 5. throughput --------------------------------------------------------------------------------------------------

async function throughput(device: GPUDevice, caps: Caps): Promise<void> {
  const h = await Harness.create(device, caps, sceneDescToCopyInput(cornell), { name: 'cornell perf', shade: {} });
  const camera = cornell.camera as unknown as PtLookAtCamera;
  const width = 512;
  const height = 512;
  const passes = 16;
  const per: Record<string, number> = {};
  const samples: Record<string, number[]> = { flat: [], lambert: [], full: [] };
  for (let round = 0; round < 5; round++) {
    for (const mode of MODES) {
      const r = await h.accumulate(camera, width, height, mode, passes, round + 1);
      samples[mode]!.push(r.wallMs);
    }
  }
  for (const mode of MODES) {
    const sorted = [...(samples[mode] as number[])].sort((a, b) => a - b);
    per[mode] = sorted[0] as number;
    metrics[`perf_cornell_${mode}_ms_min`] = sorted[0] as number;
    metrics[`perf_cornell_${mode}_ms_max`] = sorted[sorted.length - 1] as number;
    metrics[`perf_cornell_${mode}_mrays_per_s_best`] = (width * height * passes) / 1e6 / ((sorted[0] as number) / 1000);
  }
  notes.perf = `best-effort, wall clock around ${passes} passes of ${width}x${height} (closest + at most one shadow ray per sample), min..max of 5 interleaved rounds; machine may be loaded, not a result`;
  h.destroy();
}

// ---- main -----------------------------------------------------------------------------------------------------------

try {
  const args = getArgs();
  const count = numberArg(args, 'rays', 4096);
  const tol = numberArg(args, 'tol', 2e-3);
  const mc = numberArg(args, 'mc', 262144);
  const spp = numberArg(args, 'spp', 1024);
  if (args.packing && args.packing !== 'split' && args.packing !== 'packed') throw new Error(`packing must be split or packed, got '${args.packing}'`);
  PACKING = (args.packing as SceneCopyPacking | undefined) ?? 'split';
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'shade-check' });
  deviceErrors = watchDevice(device);

  await checkTwin(device, caps, count, tol);
  await drainDevice(device);
  await checkClosedForms(device, caps, mc);
  await abBsdf(device, caps, Math.max(count, 16384));
  await checkAgainstPathTracer(device, caps, spp);
  if (numberArg(args, 'perf', 1) !== 0) await throughput(device, caps);
  await drainDevice(device);

  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult(
    { scene: 'shade-check', renderer: 'ours', config: { rays: count, tol, mc, spp, packing: PACKING }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems, ...notes } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (total > 0) throw new Error(`shade-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
