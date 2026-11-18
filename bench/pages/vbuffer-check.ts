/**
 * vbuffer-check (P3.6): visibility-buffer primaries against traced primaries. Correctness only; no timing.
 *
 *  1. `raster`: the bench rasteriser's V-buffer against the closest hit of the SAME camera rays (`ptCameraDirection` of pixel + 0.5 + jitter, traced with the batch tracer) at a few
 *     frame jitters. Per pixel: equal (instance, triangle); "near" (a different triangle at the same point: duplicate or coplanar triangles, an edge shared by two triangles);
 *     hit/miss disagreements and "hard" mismatches (a different surface: silhouettes), split by edge pixels (a 4-neighbour whose traced triangle differs) and interior pixels.
 *     Over the equal pixels: max / mean / p99 distance between the barycentric position (`ptVbufferResolve`, the TS twin of the shader) and the traced one, max normal difference, material mismatches.
 *  2. `resolve`: the WGSL `ptVbufferHit` against the traced hits, through a V-buffer built from those traced hits (so the rasteriser is out of the picture): `t`, front face, barycentrics, normal, material.
 *  3. `image`: both schedulers, traced vs V-buffer at `spp` (1024): relMSE(vbuffer, traced A) next to the noise floor relMSE(traced B, traced A) (two independent realisations), relBias, non-finite pixels;
 *     and the two schedulers against each other with V-buffer primaries (same V-buffers, same sampler numbers: they must agree to float noise).
 *
 * Args: scenes=cornell,MetalRoughSpheres,tex-mask,sponza-white  spp=1024  check=raster,resolve,image  jitters=3  specpp=16
 */
import { PT_VBUFFER_NO_MATERIAL, ptCameraBasis, ptCameraDirection, ptVbufferPack, ptVbufferResolve, ptVbufferUnpack, ptVbufferWgsl, ptRaytracerModuleOptions } from '@pmndrs/pathtracer';
import { DEFAULT_SCENE_PACKING, type SceneCopyPacking } from '@pmndrs/raytracer';
import { RtBatchTracer, rtPackSceneAccess, rtModuleWgsl, rtSceneBindGroupEntries, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtWalkEntries, sceneFormatOf, withBindings } from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { getScene } from '../scenes/index.ts';
import { compareImages } from '../metrics/metrics.ts';
import { PtMegakernelAdapter, type PtAdapterSettings } from '../adapters/pt-megakernel/index.ts';

/** Scene copy binding (P3.11, page arg `packing=split|packed`, default: the library default, packed since D32). */
let PACKING: SceneCopyPacking = DEFAULT_SCENE_PACKING;
const rtPackIf = (code: string): string => (PACKING === 'packed' ? rtPackSceneAccess(code) : code);

let adapterSummary: AdapterSummary | null = null;
const deviceErrors: string[] = [];
const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string) => {
  if (problems.length < 40) problems.push(m);
};
let mismatches = 0;
const fail = (m: string) => {
  mismatches++;
  note(m);
};

const JITTERS: [number, number][] = [[0, 0], [0.31, -0.27], [-0.44, 0.12], [0.08, 0.41], [-0.2, -0.37]];

async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return data;
}

/** Camera rays for pixel + 0.5 + jitter as a 32 B AoS array (origin, tMin, direction, tMax). */
function cameraRays(adapter: PtMegakernelAdapter, scene: ReturnType<typeof getScene>, jitter: readonly [number, number]): { rays: Float32Array<ArrayBuffer>; basis: ReturnType<typeof ptCameraBasis> } {
  const { width, height } = scene.film;
  const basis = ptCameraBasis(scene.camera, width, height);
  void adapter;
  const rays = new Float32Array(width * height * 8);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const d = ptCameraDirection(basis, width, height, x + 0.5 + jitter[0], y + 0.5 + jitter[1]);
      const o = (y * width + x) * 8;
      rays.set([basis.origin[0], basis.origin[1], basis.origin[2], 0, d[0], d[1], d[2], 1e30], o);
    }
  }
  return { rays, basis };
}

async function traceHits(adapter: PtMegakernelAdapter, rays: Float32Array<ArrayBuffer>): Promise<{ f: Float32Array; u: Uint32Array }> {
  const { device, trace, caps } = { ...adapter.debug, device: adapter.gpuContext!.device };
  const tracer = await RtBatchTracer.create(device, caps, { ...sceneFormatOf(trace), mode: 'closest', label: 'vbuffer-check closest' });
  const count = rays.length / 8;
  const rayBuffer = device.createBuffer({ size: rays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(rayBuffer, 0, rays);
  const results = device.createBuffer({ size: count * tracer.resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const encoder = device.createCommandEncoder();
  tracer.encode(encoder, trace, { rays: rayBuffer, results, count });
  device.queue.submit([encoder.finish()]);
  const raw = await readBuffer(device, results, count * tracer.resultBytes);
  rayBuffer.destroy();
  results.destroy();
  tracer.destroy();
  return { f: new Float32Array(raw), u: new Uint32Array(raw) };
}

const percentile = (values: number[], p: number): number => {
  if (values.length === 0) return 0;
  const sorted = Float64Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] as number;
};

/** 1. rasteriser vs traced primaries. */
async function checkRaster(name: string, jitterCount: number): Promise<void> {
  const scene = getScene(name);
  const adapter = new PtMegakernelAdapter({ primary: 'vbuffer', scenePacking: PACKING });
  await adapter.load(scene);
  const dbg = adapter.debug;
  const raster = dbg.raster!;
  const { width, height } = scene.film;
  const device = adapter.gpuContext!.device;
  const total = { n: 0, equal: 0, near: 0, hitMiss: 0, hard: 0, edgeN: 0, edgeEqual: 0, interiorN: 0, interiorEqual: 0, interiorHard: 0, interiorHitMiss: 0, materialMismatch: 0, equalN: 0, nearOpposite: 0, nearFrontDiffers: 0 };
  const posErr: number[] = [];
  let normalErrMax = 0;
  for (const jitter of JITTERS.slice(0, jitterCount)) {
    const { rays, basis } = cameraRays(adapter, scene, jitter);
    const traced = await traceHits(adapter, rays);
    const encoder = device.createCommandEncoder();
    raster.encode(encoder, basis, jitter);
    device.queue.submit([encoder.finish()]);
    const words = await raster.read();
    const n = width * height;
    const key = new Float64Array(n);
    for (let i = 0; i < n; i++) key[i] = (traced.f[i * 10] as number) < 0 ? -1 : (traced.u[i * 10 + 1] as number) * 4294967296 + (traced.u[i * 10 + 2] as number);
    for (let i = 0; i < n; i++) {
      const x = i % width;
      const y = (i / width) | 0;
      const k = key[i] as number;
      const edge = (x > 0 && key[i - 1] !== k) || (x < width - 1 && key[i + 1] !== k) || (y > 0 && key[i - width] !== k) || (y < height - 1 && key[i + width] !== k);
      const r = ptVbufferUnpack(words, i * 4);
      const th = (traced.f[i * 10] as number) >= 0;
      total.n++;
      if (edge) total.edgeN++;
      else total.interiorN++;
      let equal = false;
      if (!th && !r.hit) equal = true;
      else if (th && r.hit && (traced.u[i * 10 + 1] as number) === r.instance && (traced.u[i * 10 + 2] as number) === r.triangle) equal = true;
      if (equal) {
        total.equal++;
        if (edge) total.edgeEqual++;
        else total.interiorEqual++;
        if (th) {
          total.equalN++;
          const t = traced.f[i * 10] as number;
          const res = ptVbufferResolve(dbg.copy, r);
          const o = i * 8;
          const px = (rays[o] as number) + (rays[o + 4] as number) * t;
          const py = (rays[o + 1] as number) + (rays[o + 5] as number) * t;
          const pz = (rays[o + 2] as number) + (rays[o + 6] as number) * t;
          posErr.push(Math.hypot(res.position[0] - px, res.position[1] - py, res.position[2] - pz));
          normalErrMax = Math.max(normalErrMax, Math.hypot(res.normal[0] - (traced.f[i * 10 + 7] as number), res.normal[1] - (traced.f[i * 10 + 8] as number), res.normal[2] - (traced.f[i * 10 + 9] as number)));
          if (r.material !== (traced.u[i * 10 + 6] as number)) total.materialMismatch++;
        }
      } else if (th !== r.hit) {
        total.hitMiss++;
        if (!edge) total.interiorHitMiss++;
      } else {
        const t = traced.f[i * 10] as number;
        const res = ptVbufferResolve(dbg.copy, r);
        const o = i * 8;
        const d = Math.hypot(res.position[0] - ((rays[o] as number) + (rays[o + 4] as number) * t), res.position[1] - ((rays[o + 1] as number) + (rays[o + 5] as number) * t), res.position[2] - ((rays[o + 2] as number) + (rays[o + 6] as number) * t));
        if (d < 1e-3 * t + 1e-4) {
          total.near++;
          // a different triangle at the same point: do the two twins face the ray the same way? (opposite-wound duplicates shade differently)
          const dotN = res.normal[0] * (traced.f[i * 10 + 7] as number) + res.normal[1] * (traced.f[i * 10 + 8] as number) + res.normal[2] * (traced.f[i * 10 + 9] as number);
          if (dotN < 0) total.nearOpposite++;
          const dir = [rays[o + 4] as number, rays[o + 5] as number, rays[o + 6] as number];
          if (res.normal[0] * dir[0]! + res.normal[1] * dir[1]! + res.normal[2] * dir[2]! < 0 !== ((traced.u[i * 10 + 3] as number) === 1)) total.nearFrontDiffers++;
        } else {
          total.hard++;
          if (!edge) total.interiorHard++;
        }
      }
    }
  }
  const pct = (a: number, b: number) => (b > 0 ? (100 * a) / b : 100);
  const p = `raster_${name}`;
  metrics[`${p}_pixels`] = total.n;
  metrics[`${p}_equal_pct`] = pct(total.equal, total.n);
  metrics[`${p}_equal_interior_pct`] = pct(total.interiorEqual, total.interiorN);
  metrics[`${p}_equal_edge_pct`] = pct(total.edgeEqual, total.edgeN);
  metrics[`${p}_near_pct`] = pct(total.near, total.n);
  metrics[`${p}_hit_miss_pct`] = pct(total.hitMiss, total.n);
  metrics[`${p}_hard_pct`] = pct(total.hard, total.n);
  metrics[`${p}_near_opposite_normal_n`] = total.nearOpposite;
  metrics[`${p}_near_front_face_differs_n`] = total.nearFrontDiffers;
  metrics[`${p}_near_n`] = total.near;
  metrics[`${p}_hard_interior_n`] = total.interiorHard;
  metrics[`${p}_hit_miss_interior_n`] = total.interiorHitMiss;
  metrics[`${p}_pos_err_max`] = posErr.reduce((a, b) => Math.max(a, b), 0);
  metrics[`${p}_pos_err_mean`] = posErr.reduce((a, b) => a + b, 0) / Math.max(1, posErr.length);
  metrics[`${p}_pos_err_p99`] = percentile(posErr, 0.99);
  metrics[`${p}_normal_err_max`] = normalErrMax;
  metrics[`${p}_material_mismatch`] = total.materialMismatch;
  if (total.materialMismatch > 0) fail(`${name}: ${total.materialMismatch} equal-triangle pixels with a different material`);
  if (pct(total.interiorEqual, total.interiorN) < 99.9) fail(`${name}: only ${pct(total.interiorEqual, total.interiorN).toFixed(3)}% of interior pixels have the traced triangle`);
  if (normalErrMax > 1e-4) fail(`${name}: normal differs by ${normalErrMax}`);
  adapter.dispose();
}

/** 2. WGSL `ptVbufferHit` against traced hits, through a V-buffer built from those hits. */
async function checkResolve(name: string): Promise<void> {
  const scene = getScene(name);
  const adapter = new PtMegakernelAdapter({ primary: 'vbuffer', scenePacking: PACKING });
  await adapter.load(scene);
  const dbg = adapter.debug;
  const device = adapter.gpuContext!.device;
  const { width, height } = scene.film;
  const n = width * height;
  const jitter = [0.23, -0.31] as const;
  const { rays } = cameraRays(adapter, scene, jitter);
  const traced = await traceHits(adapter, rays);
  // a V-buffer from the traced hits
  const rowBytes = Math.ceil((width * 16) / 256) * 256;
  const upload = new Uint32Array((rowBytes / 4) * height);
  for (let i = 0; i < n; i++) {
    const hit = (traced.f[i * 10] as number) >= 0;
    const w = ptVbufferPack({ hit, instance: traced.u[i * 10 + 1] as number, triangle: traced.u[i * 10 + 2] as number, material: traced.u[i * 10 + 6] as number, bary: [traced.f[i * 10 + 4] as number, traced.f[i * 10 + 5] as number] });
    upload.set(w, ((i / width) | 0) * (rowBytes / 4) + (i % width) * 4);
  }
  const texture = device.createTexture({ size: [width, height], format: 'rgba32uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture }, upload, { bytesPerRow: rowBytes }, [width, height]);
  const format = sceneFormatOf(dbg.trace);
  const textures = dbg.copy.textureLayout ?? undefined;
  const code = rtPackIf(`${rtModuleWgsl(dbg.caps, ptRaytracerModuleOptions({ format, layout: dbg.copy.layout, scenePacking: PACKING, ...(textures ? { textures } : {}) }))}
${ptVbufferWgsl()}
@group(2) @binding(0) var<storage, read> checkRays: array<RtRay>;
@group(2) @binding(1) var<storage, read_write> checkOut: array<u32>;
@compute @workgroup_size(64)
fn checkMain(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i >= ${n}u) { return; }
    let h = ptVbufferHit(vec2u(i % ${width}u, i / ${width}u), checkRays[i]);
    let o = i * 10u;
    checkOut[o] = bitcast<u32>(h.t);
    checkOut[o + 1u] = h.instance;
    checkOut[o + 2u] = h.triangle;
    checkOut[o + 3u] = select(0u, 1u, h.frontFace);
    checkOut[o + 4u] = bitcast<u32>(h.bary.x);
    checkOut[o + 5u] = bitcast<u32>(h.bary.y);
    checkOut[o + 6u] = h.material;
    checkOut[o + 7u] = bitcast<u32>(h.normal.x);
    checkOut[o + 8u] = bitcast<u32>(h.normal.y);
    checkOut[o + 9u] = bitcast<u32>(h.normal.z);
}`);
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) throw new Error(`vbuffer-check resolve kernel: ${errors.slice(0, 3).map((m) => `${m.lineNum}: ${m.message}`).join(' | ')}`);
  const empty = device.createBindGroupLayout({ entries: [] });
  const sceneLayout = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, dbg.copy.textureLayout ?? undefined, PACKING) });
  const ioLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint' } },
    ],
  });
  void withBindings;
  void rtWalkEntries;
  void rtSceneBindGroupEntries;
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [empty, sceneLayout, ioLayout] }), compute: { module, entryPoint: 'checkMain' } });
  const rayBuffer = device.createBuffer({ size: rays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(rayBuffer, 0, rays);
  const out = device.createBuffer({ size: n * 40, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({ layout: empty, entries: [] }));
  pass.setBindGroup(1, device.createBindGroup({ layout: sceneLayout, entries: rtSceneCopyBindGroupEntries(dbg.sceneBuffers) }));
  pass.setBindGroup(2, device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: rayBuffer } }, { binding: 1, resource: { buffer: out } }, { binding: 9, resource: texture.createView() }] }));
  pass.dispatchWorkgroups(Math.ceil(n / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const raw = await readBuffer(device, out, n * 40);
  const f = new Float32Array(raw);
  const u = new Uint32Array(raw);
  let tRel = 0;
  let baryErr = 0;
  let normalErr = 0;
  let missMismatch = 0;
  let idMismatch = 0;
  let materialMismatch = 0;
  let frontMismatch = 0;
  let hits = 0;
  for (let i = 0; i < n; i++) {
    const th = (traced.f[i * 10] as number) >= 0;
    const rh = (f[i * 10] as number) >= 0;
    if (th !== rh) {
      missMismatch++;
      continue;
    }
    if (!th) continue;
    hits++;
    if (u[i * 10 + 1] !== traced.u[i * 10 + 1] || u[i * 10 + 2] !== traced.u[i * 10 + 2]) idMismatch++;
    if (u[i * 10 + 6] !== traced.u[i * 10 + 6]) materialMismatch++;
    if (u[i * 10 + 3] !== traced.u[i * 10 + 3]) frontMismatch++;
    const t = traced.f[i * 10] as number;
    tRel = Math.max(tRel, Math.abs((f[i * 10] as number) - t) / t);
    baryErr = Math.max(baryErr, Math.abs((f[i * 10 + 4] as number) - (traced.f[i * 10 + 4] as number)), Math.abs((f[i * 10 + 5] as number) - (traced.f[i * 10 + 5] as number)));
    normalErr = Math.max(normalErr, Math.hypot((f[i * 10 + 7] as number) - (traced.f[i * 10 + 7] as number), (f[i * 10 + 8] as number) - (traced.f[i * 10 + 8] as number), (f[i * 10 + 9] as number) - (traced.f[i * 10 + 9] as number)));
  }
  const p = `resolve_${name}`;
  metrics[`${p}_hits`] = hits;
  metrics[`${p}_miss_mismatch`] = missMismatch;
  metrics[`${p}_id_mismatch`] = idMismatch;
  metrics[`${p}_material_mismatch`] = materialMismatch;
  metrics[`${p}_front_mismatch`] = frontMismatch;
  metrics[`${p}_t_rel_max`] = tRel;
  metrics[`${p}_bary_abs_max`] = baryErr;
  metrics[`${p}_normal_abs_max`] = normalErr;
  if (missMismatch + idMismatch + materialMismatch > 0) fail(`${name}: WGSL resolve disagrees with the traced hit on ${missMismatch} miss, ${idMismatch} id, ${materialMismatch} material pixels`);
  if (frontMismatch > Math.max(2, hits * 1e-4)) fail(`${name}: ${frontMismatch} front-face flags differ`);
  if (tRel > 1e-3) fail(`${name}: t differs by ${tRel} (relative)`);
  if (baryErr > 2e-5) fail(`${name}: barycentrics differ by ${baryErr}`);
  if (normalErr > 1e-4) fail(`${name}: normal differs by ${normalErr}`);
  texture.destroy();
  rayBuffer.destroy();
  out.destroy();
  adapter.dispose();
}

async function renderImage(name: string, settings: PtAdapterSettings, spp: number) {
  const adapter = new PtMegakernelAdapter({ passesPerSubmit: 8, scenePacking: PACKING, ...settings });
  await adapter.load(getScene(name));
  const result = await adapter.render({ spp });
  const image = result.image;
  const kernelBytes = result.stats?.kernel_bytes as number | undefined;
  adapter.dispose();
  return { image, kernelBytes };
}

/** 3. images: traced vs V-buffer, both schedulers, plus the schedulers against each other. */
async function checkImage(name: string, spp: number, specpp: number): Promise<void> {
  for (const scheduler of ['megakernel', 'wavefront'] as const) {
    const a = await renderImage(name, { scheduler, seed: 0 }, spp);
    const b = await renderImage(name, { scheduler, seed: 1 }, spp);
    const vPaired = await renderImage(name, { scheduler, seed: 0, primary: 'vbuffer' }, spp);
    const v = await renderImage(name, { scheduler, seed: 2, primary: 'vbuffer' }, spp);
    const floor = compareImages(b.image, a.image);
    const vb = compareImages(v.image, a.image);
    const paired = compareImages(vPaired.image, a.image);
    const p = `image_${name}_${scheduler}`;
    // vb (seed 2) and traced A (seed 0) are independent realisations, like B (seed 1) and A: their relMSE should be the noise floor. The paired one shares the bounce
    // sampler numbers (the dimension map does not shift: only the camera jitter differs), so it is far below the floor.
    metrics[`${p}_relmse_vbuffer_vs_traced`] = vb.relMse;
    metrics[`${p}_relmse_noise_floor`] = floor.relMse;
    metrics[`${p}_relmse_ratio`] = vb.relMse / Math.max(floor.relMse, 1e-30);
    metrics[`${p}_relmse_paired_same_seed`] = paired.relMse;
    metrics[`${p}_relbias_vbuffer`] = vb.relBias;
    metrics[`${p}_relbias_noise_floor`] = floor.relBias;
    metrics[`${p}_nonfinite`] = vb.nonFiniteTest + vb.nonFiniteRef;
    metrics[`${p}_firefly_pct_vbuffer`] = vb.fireflyPct;
    metrics[`${p}_firefly_pct_floor`] = floor.fireflyPct;
    if (vb.nonFiniteTest > 0) fail(`${name} ${scheduler}: ${vb.nonFiniteTest} non-finite pixels with V-buffer primaries`);
    if (vb.relMse > 1.5 * floor.relMse + 1e-7) fail(`${name} ${scheduler}: relMSE(vbuffer, traced) ${vb.relMse} is above 1.5 x the noise floor ${floor.relMse}`);
    if (Math.abs(vb.relBias) > 3 * Math.abs(floor.relBias) + 2e-3) fail(`${name} ${scheduler}: relBias ${vb.relBias} (floor ${floor.relBias})`);
  }
  // schedulers agree with V-buffer primaries (same V-buffer frames, same sampler numbers)
  const m = await renderImage(name, { scheduler: 'megakernel', primary: 'vbuffer', seed: 3 }, specpp);
  const w = await renderImage(name, { scheduler: 'wavefront', primary: 'vbuffer', seed: 3 }, specpp);
  const sched = compareImages(w.image, m.image);
  metrics[`sched_${name}_relmse_wavefront_vs_megakernel`] = sched.relMse;
  metrics[`kernel_bytes_${name}_megakernel_vbuffer`] = m.kernelBytes ?? 0;
  if (sched.relMse > 1e-6) fail(`${name}: wavefront and megakernel disagree with V-buffer primaries (relMSE ${sched.relMse})`);
}

try {
  const args = getArgs();
  const spp = numberArg(args, 'spp', 1024);
  const specpp = numberArg(args, 'specpp', 16);
  const jitters = numberArg(args, 'jitters', 3);
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  void watchDevice;
  const scenes = (args.scenes ?? 'cornell,MetalRoughSpheres,tex-mask,sponza-white').split(',');
  if (args.packing && args.packing !== 'split' && args.packing !== 'packed') throw new Error(`packing must be split or packed, got '${args.packing}'`);
  PACKING = (args.packing as SceneCopyPacking | undefined) ?? DEFAULT_SCENE_PACKING;
  const only = (args.check ?? 'raster,resolve,image').split(',');
  for (const name of scenes) {
    if (only.includes('raster')) await checkRaster(name, jitters);
    if (only.includes('resolve')) await checkResolve(name);
    if (only.includes('image')) await checkImage(name, spp, specpp);
  }
  void PT_VBUFFER_NO_MATERIAL;
  reportResult(
    { scene: 'vbuffer-check', renderer: 'ours', config: { spp, scenes: scenes.join(','), checks: only.join(',') }, valid: mismatches === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches }, notes: { problems } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (mismatches > 0) throw new Error(`vbuffer-check: ${mismatches} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
