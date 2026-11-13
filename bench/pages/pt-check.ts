/**
 * pt-check (P2.2 / P1.4g / P2.4): the GPU's WGSL against its TypeScript twins. Correctness only, no timing.
 *
 *  1. Sampler: `ptSample2D` (Owen-Sobol and PCG) for a grid of (pixel seed, sample index, pair) values: exact equality with
 *     `ptSample2D` in `dimensions.ts` (integer arithmetic, 24-bit floats).
 *  2. Material adapter: `ptBsdfFromMaterial(rtMaterial(id))` against `ptBsdfFromMaterial(copy.materials[id])` for the Tier 0
 *     material kinds plus the corners (metal without eta/k, blend weights at the thresholds, default material, id past the table).
 *  3. Light sampling: `rtSampleLightLi(index, p, u)` against `sampleLightLi` (f64) for an emissive quad, a sun with an angular
 *     radius, a delta sun and a uniform environment, on random shading points and stratified u; plus the MIS helpers
 *     (`rtSunContains`, `rtSunPdf`, `rtTriangleSolidAnglePdf`).
 *
 *  4. End to end, sun + sky: a Lambertian plane (albedo 0.5) under a sun of irradiance E and a uniform sky of radiance Ls, direct
 *     lighting only (maxDepth 1): every pixel is `albedo/pi * E * cos(theta) + albedo * Ls`. Variants: sun as a cone (MIS with BSDF
 *     hits of the disc), as a delta light, cone without NEE (BSDF sampling finds the disc), cone with sky sampled by BSDF only.
 *
 * Args: cases=4000 (per check)  tol=2e-4  spp=256 (sun-plane)
 */
import {
  SCENE_NONE,
  buildSceneCopy,
  rtOwenSobolWgsl,
  rtSceneWgsl,
  sampleLightLi,
  uploadSceneCopy,
  rtSceneCopyBindGroupEntries,
  rtSceneCopyLayoutEntries,
  type RtMaterialInput,
  type SceneCopyInput,
} from '@pmndrs/raytracer';
import { ptBsdfAdapterWgsl, ptBsdfFromMaterial, ptBsdfWgsl, ptPixelSeed, ptSample2D, ptSamplerWgsl, type PtSamplerKind } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import { quadLight } from '../scenes/tier0/index.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { PtMegakernelAdapter, type PtAdapterSettings } from '../adapters/pt-megakernel/index.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

const problems: string[] = [];
const metrics: Record<string, number> = {};
const note = (m: string) => {
  if (problems.length < 20) problems.push(m);
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

async function compile(device: GPUDevice, code: string, label: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) throw new Error(`WGSL (${label}) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n')}`);
  return module;
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

const storage = (device: GPUDevice, data: ArrayBufferView<ArrayBuffer>, extra = 0): GPUBuffer => {
  const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

// ---- 1. sampler -----------------------------------------------------------------------------------------------------

async function checkSampler(device: GPUDevice, count: number): Promise<void> {
  for (const kind of ['owen-sobol', 'pcg'] as PtSamplerKind[]) {
    const code = `${rtOwenSobolWgsl()}${ptSamplerWgsl(kind)}
@group(0) @binding(0) var<storage, read> inp: array<u32>;
@group(0) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&inp) / 3u) { return; }
  let v = ptSample2D(inp[i * 3u], inp[i * 3u + 1u], inp[i * 3u + 2u]);
  outp[i * 2u] = v.x;
  outp[i * 2u + 1u] = v.y;
}`;
    const rand = rng(99);
    const input = new Uint32Array(count * 3);
    for (let i = 0; i < count; i++) {
      input[i * 3] = ptPixelSeed(Math.floor(rand() * 1024), Math.floor(rand() * 1024), Math.floor(rand() * 4));
      // Include indices past 2^16 (block wrap) and pairs of deep paths.
      input[i * 3 + 1] = i % 5 === 0 ? Math.floor(rand() * 200000) : Math.floor(rand() * 4096);
      input[i * 3 + 2] = i % 7 === 0 ? Math.floor(rand() * 4100) : Math.floor(rand() * 40);
    }
    const inBuf = storage(device, input);
    const outBuf = storage(device, new Float32Array(count * 2), GPUBufferUsage.COPY_SRC);
    const module = await compile(device, code, `sampler ${kind}`);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const got = new Float32Array(await readBack(device, outBuf, count * 8));
    let bad = 0;
    for (let i = 0; i < count; i++) {
      const [x, y] = ptSample2D(kind, input[i * 3] as number, input[i * 3 + 1] as number, input[i * 3 + 2] as number);
      if (got[i * 2] !== x || got[i * 2 + 1] !== y) {
        bad++;
        note(`sampler ${kind} case ${i} (seed ${input[i * 3]}, index ${input[i * 3 + 1]}, pair ${input[i * 3 + 2]}): gpu ${got[i * 2]},${got[i * 2 + 1]} cpu ${x},${y}`);
      }
    }
    metrics[`sampler_${kind.replace('-', '_')}_mismatch`] = bad;
    metrics[`sampler_${kind.replace('-', '_')}_cases`] = count;
  }
}

// ---- scene with every light kind ------------------------------------------------------------------------------------

const EXTRA_MATERIALS: RtMaterialInput[] = [
  { conductorEtaK: true, conductorEta: [0.2, 0.9, 1.1], conductorK: [3.9, 2.4, 2.1], metalness: 1, specularRoughness: 0.35 },
  { baseWeight: 0, baseColor: [0, 0, 0], transmissionWeight: 1, specularWeight: 1, ior: 1.33, specularRoughness: 0.2 },
  { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 },
  { metalness: 1, baseColor: [0.9, 0.5, 0.1], conductorEtaK: false, specularRoughness: 0.5 },
  { metalness: 0.4, conductorEtaK: true, baseColor: [0.2, 0.4, 0.6] },
  { transmissionWeight: 0.4, baseColor: [0.3, 0.3, 0.3] },
  { baseColor: [0.25, 0.5, 0.75], baseWeight: 0.5, specularWeight: 0 },
];

function lightScene(sunRadius: number): SceneCopyInput {
  const base = sceneDescToCopyInput(quadLight);
  return {
    meshes: base.meshes,
    instances: base.instances,
    materials: [...base.materials, ...EXTRA_MATERIALS],
    sun: { direction: [-0.3, -0.8, -0.2], irradiance: [50, 46, 40], angularRadius: sunRadius },
    environment: { radiance: [0.3, 0.4, 0.5], scale: 2 },
  };
}

// ---- 2 + 3. adapter and lights --------------------------------------------------------------------------------------

const rel = (a: number, b: number, abs: number): number => Math.abs(a - b) / (Math.max(Math.abs(a), Math.abs(b)) + abs);

async function checkScene(device: GPUDevice, name: string, input: SceneCopyInput, count: number, tol: number): Promise<void> {
  const copy = buildSceneCopy(input);
  const buffers = uploadSceneCopy(device, copy, { label: name });
  const materialCount = copy.materials.length + 2; // + the default record and an id past the table
  const lightCount = copy.lights.length;
  const rand = rng(1234);
  const L_IN = 8;
  const L_OUT = 20;
  const lightInput = new Float32Array(count * L_IN);
  const lightIndexBits = new Uint32Array(lightInput.buffer);
  const cases: { index: number; p: [number, number, number]; u: [number, number] }[] = [];
  for (let i = 0; i < count; i++) {
    const index = Math.floor(rand() * lightCount);
    const p: [number, number, number] = [Math.fround((rand() - 0.5) * 6), Math.fround(rand() * 0.9 - 0.2), Math.fround((rand() - 0.5) * 6)];
    const u: [number, number] = [Math.fround(rand()), Math.fround(rand())];
    cases.push({ index, p, u });
    lightIndexBits[i * L_IN] = index;
    lightInput.set([p[0], p[1], p[2], u[0], u[1], 0, 0], i * L_IN + 1);
  }
  const code = `${rtSceneWgsl({ group: 0, firstBinding: 0, sampling: true })}${ptBsdfWgsl({})}${ptBsdfAdapterWgsl()}
@group(1) @binding(0) var<storage, read> lightIn: array<u32>;
@group(1) @binding(1) var<storage, read_write> lightOut: array<f32>;
@group(1) @binding(2) var<storage, read_write> matOut: array<f32>;
@compute @workgroup_size(64)
fn lights(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&lightIn) / ${L_IN}u) { return; }
  let b = i * ${L_IN}u;
  let index = lightIn[b];
  let p = vec3f(bitcast<f32>(lightIn[b + 1u]), bitcast<f32>(lightIn[b + 2u]), bitcast<f32>(lightIn[b + 3u]));
  let u = vec2f(bitcast<f32>(lightIn[b + 4u]), bitcast<f32>(lightIn[b + 5u]));
  let li = rtSampleLightLi(index, p, u);
  let o = i * ${L_OUT}u;
  lightOut[o] = select(0.0, 1.0, li.valid);
  lightOut[o + 1u] = select(0.0, 1.0, li.delta);
  lightOut[o + 2u] = li.wi.x; lightOut[o + 3u] = li.wi.y; lightOut[o + 4u] = li.wi.z;
  lightOut[o + 5u] = min(li.distance, 1.0e30);
  lightOut[o + 6u] = li.point.x; lightOut[o + 7u] = li.point.y; lightOut[o + 8u] = li.point.z;
  lightOut[o + 9u] = li.pdf;
  lightOut[o + 10u] = li.normal.x; lightOut[o + 11u] = li.normal.y; lightOut[o + 12u] = li.normal.z;
  lightOut[o + 13u] = li.radiance.x; lightOut[o + 14u] = li.radiance.y; lightOut[o + 15u] = li.radiance.z;
  var sunContains = 0.0;
  var sunPdf = 0.0;
  if (rtHasSun()) {
    let s = rtSun();
    sunContains = select(0.0, 1.0, rtSunContains(s, li.wi));
    sunPdf = rtSunPdf(s, li.wi);
  }
  lightOut[o + 16u] = sunContains;
  lightOut[o + 17u] = sunPdf;
}
@compute @workgroup_size(64)
fn materials(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= ${materialCount}u) { return; }
  let p = ptBsdfFromMaterial(rtMaterial(i));
  let o = i * 16u;
  matOut[o] = f32(p.kind); matOut[o + 1u] = p.ior; matOut[o + 2u] = p.alpha.x; matOut[o + 3u] = p.alpha.y;
  matOut[o + 4u] = p.reflectance.x; matOut[o + 5u] = p.reflectance.y; matOut[o + 6u] = p.reflectance.z;
  matOut[o + 7u] = p.eta.x; matOut[o + 8u] = p.eta.y; matOut[o + 9u] = p.eta.z;
  matOut[o + 10u] = p.k.x; matOut[o + 11u] = p.k.y; matOut[o + 12u] = p.k.z;
}`;
  const module = await compile(device, code, `${name} lights+adapter`);
  const sceneLayout = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries() });
  const ioLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] });
  const lightsPipeline = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'lights' } });
  const materialsPipeline = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'materials' } });
  const inBuf = storage(device, lightInput);
  const lightOutBuf = storage(device, new Float32Array(count * L_OUT), GPUBufferUsage.COPY_SRC);
  const matOutBuf = storage(device, new Float32Array(materialCount * 16), GPUBufferUsage.COPY_SRC);
  const sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: rtSceneCopyBindGroupEntries(buffers) });
  const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: lightOutBuf } }, { binding: 2, resource: { buffer: matOutBuf } }] });
  const encoder = device.createCommandEncoder();
  for (const [pipeline, n] of [[lightsPipeline, count], [materialsPipeline, materialCount]] as const) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, sceneGroup);
    pass.setBindGroup(1, ioGroup);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
  }
  device.queue.submit([encoder.finish()]);
  const lightOut = new Float32Array(await readBack(device, lightOutBuf, count * L_OUT * 4));
  const matOut = new Float32Array(await readBack(device, matOutBuf, materialCount * 16 * 4));

  // Materials: id past the table and the default record (index materials.length) both read the default material.
  let matBad = 0;
  for (let id = 0; id < materialCount; id++) {
    const m = id < copy.materials.length ? copy.materials[id]! : copy.defaultMaterial;
    const p = ptBsdfFromMaterial(m);
    const g = Array.from(matOut.subarray(id * 16, id * 16 + 13));
    const c = [p.kind, p.ior, p.alpha[0], p.alpha[1], ...p.reflectance, ...p.eta, ...p.k];
    c.forEach((cv, k) => {
      if (rel(g[k] as number, cv, 1e-4) > tol) {
        matBad++;
        note(`${name} material ${id} field ${k}: gpu ${g[k]} cpu ${cv}`);
      }
    });
  }
  metrics[`${name}_material_mismatch`] = matBad;
  metrics[`${name}_materials`] = materialCount;

  // Lights.
  let bad = 0;
  let maxRel = 0;
  let invalidMismatch = 0;
  const kindCounts = { triangle: 0, sun: 0, environment: 0 };
  cases.forEach((c, i) => {
    const light = copy.lights[c.index]!;
    kindCounts[light.kind]++;
    const g = lightOut.subarray(i * L_OUT, i * L_OUT + L_OUT);
    const cpu = sampleLightLi(light, c.p, c.u[0], c.u[1]);
    if ((g[0] === 1) !== cpu.valid) {
      // A sample within f32 noise of the one-sided boundary or of a zero distance may flip: counted, not compared.
      invalidMismatch++;
      return;
    }
    if (!cpu.valid) return;
    if ((g[1] === 1) !== cpu.delta) {
      bad++;
      note(`${name} light ${c.index} case ${i}: delta flag gpu ${g[1]} cpu ${cpu.delta}`);
      return;
    }
    const pairs: [number, number, string][] = [
      [g[2] as number, cpu.wi[0], 'wi.x'],
      [g[3] as number, cpu.wi[1], 'wi.y'],
      [g[4] as number, cpu.wi[2], 'wi.z'],
      [g[5] as number, Math.min(cpu.distance, 1e30), 'distance'],
      [g[9] as number, cpu.pdf, 'pdf'],
      [g[13] as number, cpu.radiance[0], 'L.r'],
      [g[14] as number, cpu.radiance[1], 'L.g'],
      [g[15] as number, cpu.radiance[2], 'L.b'],
    ];
    if (light.kind === 'triangle') pairs.push([g[6] as number, cpu.point[0], 'p.x'], [g[7] as number, cpu.point[1], 'p.y'], [g[8] as number, cpu.point[2], 'p.z'], [g[10] as number, cpu.normal[0], 'n.x'], [g[11] as number, cpu.normal[1], 'n.y'], [g[12] as number, cpu.normal[2], 'n.z']);
    for (const [gv, cv, what] of pairs) {
      const r = rel(gv, cv, what.startsWith('wi') || what.startsWith('p.') || what.startsWith('n.') ? 1e-2 : 1e-6);
      maxRel = Math.max(maxRel, r);
      if (r > tol * 5 || !Number.isFinite(gv)) {
        bad++;
        note(`${name} ${light.kind} light ${c.index} case ${i} ${what}: gpu ${gv} cpu ${cv}`);
      }
    }
    if (light.kind === 'sun' && !cpu.delta) {
      if (g[16] !== 1) {
        bad++;
        note(`${name} sun case ${i}: rtSunContains is false for a sampled direction`);
      }
      if (rel(g[17] as number, 1 / light.solidAngle, 0) > tol * 5) {
        bad++;
        note(`${name} sun case ${i}: rtSunPdf ${g[17]} vs ${1 / light.solidAngle}`);
      }
    }
  });
  metrics[`${name}_light_mismatch`] = bad;
  metrics[`${name}_light_validity_flips`] = invalidMismatch;
  metrics[`${name}_light_max_rel`] = maxRel;
  metrics[`${name}_lights_triangle`] = kindCounts.triangle;
  metrics[`${name}_lights_sun`] = kindCounts.sun;
  metrics[`${name}_lights_environment`] = kindCounts.environment;
  if (copy.sunLight === SCENE_NONE) note(`${name}: scene copy has no sun`);
  buffers.destroy();
}

// ---- 4. sun + sky end to end -----------------------------------------------------------------------------------------

const SUN_DIRECTION: [number, number, number] = [-0.5, -0.7, -0.3];
const SUN_IRRADIANCE = 3;
const SKY = 0.2;
const ALBEDO = 0.5;

function sunPlane(angularRadius: number): SceneDesc {
  return {
    version: 1,
    name: 'sun-plane',
    purpose: 'Lambertian plane under a sun and a uniform sky, direct lighting only: closed-form radiance.',
    tier: 0,
    camera: { type: 'perspective', position: [0, 3, 0.5], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: 128, height: 128, filter: 'box' },
    integrator: { maxDepth: 1 },
    materials: { floor: { type: 'diffuse', reflectance: [ALBEDO, ALBEDO, ALBEDO] } },
    meshes: [{ name: 'plane', geometry: { type: 'quad', size: [40, 40] }, material: 'floor' }],
    lights: [{ type: 'directional', direction: SUN_DIRECTION, irradiance: [SUN_IRRADIANCE, SUN_IRRADIANCE, SUN_IRRADIANCE], angularRadius }],
    environment: { type: 'uniform', radiance: [SKY, SKY, SKY] },
  };
}

async function checkSunPlane(spp: number): Promise<void> {
  const length = Math.hypot(...SUN_DIRECTION);
  const cosTheta = -SUN_DIRECTION[1] / length;
  const expected = ((ALBEDO / Math.PI) * SUN_IRRADIANCE * cosTheta) + ALBEDO * SKY;
  const variants: { name: string; radius: number; settings: PtAdapterSettings; expected: number }[] = [
    { name: 'cone', radius: 0.05, settings: {}, expected },
    { name: 'delta', radius: 0, settings: {}, expected },
    { name: 'cone_nee_off', radius: 0.05, settings: { nee: false }, expected },
    { name: 'cone_env_none', radius: 0.05, settings: { envSampling: 'none' }, expected },
  ];
  for (const variant of variants) {
    const adapter = new PtMegakernelAdapter(variant.settings);
    await adapter.load(sunPlane(variant.radius));
    const result = await adapter.render({ spp });
    const data = result.image.data;
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    const n = result.image.width * result.image.height;
    for (let i = 0; i < n; i++) {
      const v = data[i * 4 + 1] as number;
      sum += v;
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
    const mean = sum / n;
    const relDev = (mean - variant.expected) / variant.expected;
    metrics[`sun_plane_${variant.name}_mean`] = mean;
    metrics[`sun_plane_${variant.name}_rel_dev`] = relDev;
    metrics[`sun_plane_${variant.name}_min`] = min;
    metrics[`sun_plane_${variant.name}_max`] = max;
    const tolerance = 0.01;
    if (Math.abs(relDev) > tolerance) {
      metrics[`sun_plane_${variant.name}_mismatch`] = 1;
      note(`sun-plane ${variant.name}: mean ${mean} vs ${variant.expected} (rel dev ${relDev})`);
    } else metrics[`sun_plane_${variant.name}_mismatch`] = 0;
    adapter.dispose();
  }
}

try {
  const args = getArgs();
  const count = numberArg(args, 'cases', 4000);
  const tol = numberArg(args, 'tol', 2e-4);
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice();
  deviceErrors = watchDevice(device);

  await checkSampler(device, count);
  await checkScene(device, 'sun-cone', lightScene(0.05), count, tol);
  await checkScene(device, 'sun-delta', lightScene(0), count, tol);
  await drainDevice(device);
  await checkSunPlane(numberArg(args, 'spp', 256));

  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult(
    {
      scene: 'pt-check',
      renderer: 'ours',
      config: { cases: count, tol },
      valid: total === 0 && deviceErrors.length === 0,
      metrics: { ...metrics, mismatches: total },
      notes: { problems },
    },
    { adapter: adapterSummary, deviceErrors },
  );
  if (total > 0) throw new Error(`pt-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
