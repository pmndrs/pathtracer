/**
 * texture-check (P1.4c): material textures on the GPU against their TypeScript twins and against closed forms. Correctness only, no timing.
 *
 *  1. Lookups: `rtTexSample` (trilinear, every bucket / wrap / colour space, uvs far outside [0, 1]) against `sampleSceneTexture`; the
 *     hardware filters with fixed-point weights, so the tolerance is 8-bit granularity (default 0.01 absolute, linear light). `rtTexLodOf`
 *     against `textureLod`; `rtApplyMaterialTextures` against the glTF semantics (colour, alpha, roughness from G, metalness from B,
 *     emission) written out in this file; `rtApplyNormalMap` against `applyNormalMap` (pure f32 maths, 1e-4).
 *  2. End to end (the whole megakernel, direct lighting from a delta sun, no sky, so every pixel has a closed form):
 *     - base colour: a plane whose sRGB texture is one colour: radiance = decode(code) / pi * E * cos(theta);
 *     - normal map: a tangent-space normal tilted 30 degrees toward +u, toward image up (-z on the quad), mirrored in v, and on a rotated
 *       instance: radiance = albedo / pi * E * max(0, n' . l) with n' the perturbed normal (the 8-bit texel decoded), which also pins the glTF
 *       green-up and tangent conventions;
 *     - emission: radiance = emissive * decode(code) at a camera hit.
 *
 * Args: cases=3000  tol=0.01 (lookup, absolute)  spp=16
 */
import {
  applyNormalMap,
  buildSceneCopy,
  rtSceneWgsl,
  rtSceneCopyBindGroupEntries,
  rtSceneCopyLayoutEntries,
  sampleSceneTexture,
  srgbToLinear,
  textureLod,
  uploadSceneCopy,
  type SceneCopyInput,
  type TextureInput,
} from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { compose, rotationY } from '../scenes/geometry.ts';
import type { SceneDesc, TextureDesc } from '../scenes/schema.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';

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

// ---- 1. lookups ---------------------------------------------------------------------------------------------------------

function randomTexture(rand: () => number, w: number, h: number, colorSpace: 'srgb' | 'linear', wrap: 'repeat' | 'clamp' | 'mirror', label: string, smooth = false): TextureInput {
  const data = new Uint8Array(w * h * 4);
  for (let i = 0; i < data.length; i++) data[i] = smooth ? 128 + Math.round(100 * Math.sin(i * 0.013 + (i & 3))) : Math.floor(rand() * 256);
  return { width: w, height: h, data, colorSpace, wrap, label };
}

const L_IN = 16;
const L_OUT = 24;

async function checkLookups(device: GPUDevice, count: number, tol: number): Promise<void> {
  const rand = rng(77);
  const textures: TextureInput[] = [
    randomTexture(rand, 64, 64, 'srgb', 'repeat', 'colour'),
    randomTexture(rand, 100, 60, 'linear', 'clamp', 'mr (non power of two)'),
    randomTexture(rand, 32, 32, 'linear', 'mirror', 'normal'),
    randomTexture(rand, 256, 256, 'srgb', 'repeat', 'emission', true),
    randomTexture(rand, 64, 64, 'srgb', 'repeat', 'second layer'),
  ];
  const quad = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]), normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]) };
  const input: SceneCopyInput = {
    meshes: [quad],
    instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: 0 }],
    materials: [{ baseColor: [0.8, 0.6, 0.4], specularRoughness: 0.7, metalness: 0.9, emissionColor: [1, 2, 3], baseColorTexture: 0, roughnessMetalTexture: 1, normalTexture: 2, emissionTexture: 3 }],
    textures,
  };
  const copy = buildSceneCopy(input); // the library default: levels 1+ come from the GPU mip generator (D34)
  const twin = buildSceneCopy({ ...input, options: { textures: { mips: 'cpu' } } }).textures!; // the CPU twin needs every level: it is the oracle
  const buffers = uploadSceneCopy(device, copy, { label: 'texture-check' });
  metrics.lookup_arrays = copy.textures!.buckets.length;
  metrics.lookup_texture_bytes = copy.textures!.byteLength;
  const material = copy.materials[0]!;

  const code = `${rtSceneWgsl({ group: 0, firstBinding: 0, textures: copy.textureLayout })}
@group(1) @binding(0) var<storage, read> inp: array<u32>;
@group(1) @binding(1) var<storage, read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= arrayLength(&inp) / ${L_IN}u) { return; }
  let b = i * ${L_IN}u;
  let slot = inp[b];
  let uv = vec2f(bitcast<f32>(inp[b + 1u]), bitcast<f32>(inp[b + 2u]));
  let fp = bitcast<f32>(inp[b + 3u]);
  let tangent = vec4f(bitcast<f32>(inp[b + 4u]), bitcast<f32>(inp[b + 5u]), bitcast<f32>(inp[b + 6u]), bitcast<f32>(inp[b + 7u]));
  let n = vec3f(bitcast<f32>(inp[b + 8u]), bitcast<f32>(inp[b + 9u]), bitcast<f32>(inp[b + 10u]));
  let texel = vec3f(bitcast<f32>(inp[b + 11u]), bitcast<f32>(inp[b + 12u]), bitcast<f32>(inp[b + 13u]));
  let r = rtTexRecord(slot);
  let lod = rtTexLodOf(r, fp);
  let s = rtTexSampleRecord(r, uv, lod);
  let m = rtApplyMaterialTextures(rtMaterial(0u), uv, fp);
  let nm = rtApplyNormalMap(n, tangent, texel);
  let o = i * ${L_OUT}u;
  outp[o] = lod;
  outp[o + 1u] = s.x; outp[o + 2u] = s.y; outp[o + 3u] = s.z; outp[o + 4u] = s.w;
  outp[o + 5u] = m.baseColor.x; outp[o + 6u] = m.baseColor.y; outp[o + 7u] = m.baseColor.z; outp[o + 8u] = m.opacity;
  outp[o + 9u] = m.specularRoughness; outp[o + 10u] = m.metalness;
  outp[o + 11u] = m.emissionColor.x; outp[o + 12u] = m.emissionColor.y; outp[o + 13u] = m.emissionColor.z;
  outp[o + 14u] = nm.x; outp[o + 15u] = nm.y; outp[o + 16u] = nm.z;
}`;
  const module = device.createShaderModule({ label: 'texture-check lookups', code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length > 0) throw new Error(`WGSL (texture-check) failed to compile:\n${errors.map((m) => `  line ${m.lineNum}: ${m.message}`).join('\n')}`);
  const sceneLayout = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, copy.textureLayout) });
  const ioLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, ioLayout] }), compute: { module, entryPoint: 'main' } });

  const input32 = new Float32Array(new ArrayBuffer(count * L_IN * 4));
  const inputBits = new Uint32Array(input32.buffer);
  const cases: { slot: number; u: number; v: number; fp: number; tangent: number[]; hand: number; n: number[]; texel: number[] }[] = [];
  const unit = () => {
    const z = 2 * rand() - 1;
    const a = 2 * Math.PI * rand();
    const r = Math.sqrt(1 - z * z);
    return [Math.fround(r * Math.cos(a)), Math.fround(r * Math.sin(a)), Math.fround(z)];
  };
  for (let i = 0; i < count; i++) {
    const slot = i % 5;
    const u = Math.fround(-1.5 + 4 * rand());
    const v = Math.fround(-1.5 + 4 * rand());
    const fp = Math.fround(i % 3 === 0 ? 0 : 10 ** (-4 + 4.3 * rand()));
    const tangent = unit();
    const hand = rand() < 0.5 ? -1 : 1;
    const n = unit();
    const texel = [Math.fround(rand()), Math.fround(rand()), Math.fround(0.5 + 0.5 * rand())];
    cases.push({ slot, u, v, fp, tangent, hand, n, texel });
    inputBits[i * L_IN] = slot;
    input32.set([u, v, fp, ...tangent, hand, ...n, ...texel], i * L_IN + 1);
  }
  const mk = (data: Float32Array<ArrayBuffer>, extra = 0) => {
    const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extra });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const inBuf = mk(input32);
  const outBuf = mk(new Float32Array(count * L_OUT), GPUBufferUsage.COPY_SRC);
  const sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: rtSceneCopyBindGroupEntries(buffers) });
  const ioGroup = device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: inBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, sceneGroup);
  pass.setBindGroup(1, ioGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  const out = new Float32Array(await readBack(device, outBuf, count * L_OUT * 4));

  const stats = { sample: 0, lod: 0, material: 0, normal: 0 };
  const bad = { sample: 0, lod: 0, material: 0, normal: 0 };
  const sq = (n: number) => n * n;
  void sq;
  cases.forEach((c, i) => {
    const g = out.subarray(i * L_OUT, i * L_OUT + L_OUT);
    const slot = twin.slots[c.slot]!;
    const lod = textureLod(slot.width, slot.height, c.fp);
    const s = sampleSceneTexture(twin, c.slot, c.u, c.v, lod);
    const lodErr = Math.abs((g[0] as number) - lod);
    stats.lod = Math.max(stats.lod, lodErr);
    if (lodErr > 1e-4) {
      bad.lod++;
      note(`lod case ${i} slot ${c.slot} fp ${c.fp}: gpu ${g[0]} cpu ${lod}`);
    }
    const sampleErr = Math.max(...s.map((v, k) => Math.abs((g[1 + k] as number) - v)));
    stats.sample = Math.max(stats.sample, sampleErr);
    if (sampleErr > tol) {
      bad.sample++;
      note(`sample case ${i} slot ${c.slot} uv ${c.u},${c.v} lod ${lod}: gpu ${Array.from(g.subarray(1, 5))} cpu ${s}`);
    }
    // glTF semantics of rtApplyMaterialTextures, from the texture twin and the decoded material.
    const t = (k: number) => sampleSceneTexture(twin, k, c.u, c.v, textureLod(twin.slots[k]!.width, twin.slots[k]!.height, c.fp));
    const base = t(0);
    const mr = t(1);
    const em = t(3);
    const want = [...material.baseColor.map((v, k) => v * (base[k] as number)), material.opacity * base[3], material.specularRoughness * mr[1], material.metalness * mr[2], ...material.emissionColor.map((v, k) => v * (em[k] as number))];
    // Different slots round their own lod; compare against the slot-specific lookups, tolerance as for the lookups.
    const matErr = Math.max(...want.map((v, k) => Math.abs((g[5 + k] as number) - v)));
    stats.material = Math.max(stats.material, matErr);
    // Emission is scaled by up to 3 (the material's [1, 2, 3]), so its filtering error is too.
    if (matErr > 3 * tol) {
      bad.material++;
      note(`material case ${i}: gpu ${Array.from(g.subarray(5, 14))} cpu ${want}`);
    }
    const nm = applyNormalMap(c.n, c.tangent, c.hand, c.texel);
    const nErr = Math.max(...nm.map((v, k) => Math.abs((g[14 + k] as number) - v)));
    stats.normal = Math.max(stats.normal, nErr);
    if (nErr > 1e-4) {
      bad.normal++;
      note(`normal map case ${i}: gpu ${Array.from(g.subarray(14, 17))} cpu ${nm}`);
    }
  });
  for (const k of ['sample', 'lod', 'material', 'normal'] as const) {
    metrics[`lookup_${k}_max_err`] = stats[k];
    metrics[`lookup_${k}_mismatch`] = bad[k];
  }
  metrics.lookup_cases = count;
}

// ---- 2. end to end ------------------------------------------------------------------------------------------------------

const SUN: [number, number, number] = [-0.5, -0.7, -0.3];
const E = 3;

function plane(name: string, textures: Record<string, TextureDesc>, materials: SceneDesc['materials'], transform?: number[], sun = true): SceneDesc {
  return {
    version: 1,
    name,
    purpose: 'Textured plane under a delta sun, direct lighting only: closed-form radiance.',
    tier: 1,
    camera: { type: 'perspective', position: [0, 3, 0.5], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: 64, height: 64, filter: 'box' },
    integrator: { maxDepth: 1 },
    textures,
    materials,
    meshes: [{ name: 'plane', geometry: { type: 'quad', size: [40, 40] }, material: 'm', uv: { scale: [4, 4] }, ...(transform ? { transform } : {}) }],
    ...(sun ? { lights: [{ type: 'directional' as const, direction: SUN, irradiance: [E, E, E] as [number, number, number] }] } : {}),
  };
}

function toward(): [number, number, number] {
  const l = Math.hypot(...SUN);
  return [-SUN[0] / l, -SUN[1] / l, -SUN[2] / l];
}

async function render(scene: SceneDesc, spp: number): Promise<{ mean: [number, number, number]; min: number; max: number }> {
  const adapter = new PtMegakernelAdapter({});
  await adapter.load(scene);
  const result = await adapter.render({ spp });
  adapter.dispose();
  const d = result.image.data;
  const n = result.image.width * result.image.height;
  const mean: [number, number, number] = [0, 0, 0];
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) mean[c] = (mean[c] as number) + (d[i * 4 + c] as number) / n;
    min = Math.min(min, d[i * 4 + 1] as number);
    max = Math.max(max, d[i * 4 + 1] as number);
  }
  return { mean, min, max };
}

function expectColour(name: string, got: { mean: [number, number, number]; min: number; max: number }, want: [number, number, number], rel = 5e-3): void {
  const dev = Math.max(...want.map((w, c) => Math.abs((got.mean[c] as number) - w) / Math.max(w, 1e-6)));
  metrics[`${name}_rel_dev`] = dev;
  metrics[`${name}_mean_g`] = got.mean[1];
  metrics[`${name}_spread`] = got.max - got.min;
  if (dev > rel) {
    metrics[`${name}_mismatch`] = 1;
    note(`${name}: mean ${got.mean} vs closed form ${want} (rel dev ${dev})`);
  } else metrics[`${name}_mismatch`] = 0;
}

async function checkEndToEnd(spp: number): Promise<void> {
  const l = toward();
  const solid = (color: [number, number, number, number?], colorSpace: 'srgb' | 'linear'): TextureDesc => ({ generator: 'solid', color, width: 8, height: 8, colorSpace });

  // Base colour: codes (128, 160, 200).
  {
    const code: [number, number, number] = [128, 160, 200];
    const scene = plane('tex-base', { t: solid(code, 'srgb') }, { m: { type: 'diffuse', reflectance: [1, 1, 1], reflectanceTexture: 't' } });
    const want = code.map((c) => (srgbToLinear(c) / Math.PI) * E * l[1]) as [number, number, number];
    expectColour('base_colour', await render(scene, spp), want);
  }

  // Normal maps. n_ts = (sx, sy, cos) encoded in 8 bits; the closed form decodes the bytes like the GPU does.
  const alpha = Math.PI / 6;
  const encode = (v: number) => Math.round((0.5 + 0.5 * v) * 255);
  const decode = (code: number) => (code / 255) * 2 - 1;
  const rotY = 0.7;
  const cases: { name: string; ts: [number, number]; transform?: number[]; world: (n: number[]) => number[] }[] = [
    { name: 'normal_u', ts: [Math.sin(alpha), 0], world: (n) => n },
    // Image up on the quad is -z (v grows with +z), so a +y texel tilts the normal toward -z.
    { name: 'normal_up', ts: [0, Math.sin(alpha)], world: (n) => n },
    { name: 'normal_u_down', ts: [-Math.sin(alpha), -Math.sin(alpha) * 0.5], world: (n) => n },
    { name: 'normal_rotated', ts: [Math.sin(alpha), 0.2], transform: compose(rotationY(rotY)), world: (n) => [Math.cos(rotY) * n[0]! + Math.sin(rotY) * n[2]!, n[1]!, -Math.sin(rotY) * n[0]! + Math.cos(rotY) * n[2]!] },
  ];
  for (const c of cases) {
    const code: [number, number, number] = [encode(c.ts[0]), encode(c.ts[1]), encode(Math.sqrt(1 - c.ts[0] ** 2 - c.ts[1] ** 2))];
    const ts = [decode(code[0]), decode(code[1]), decode(code[2])];
    const il = 1 / Math.hypot(ts[0]!, ts[1]!, ts[2]!);
    // Tangent space of the (unrotated) quad: T = +x, N = +y, image-up bitangent = -z.
    const local = [ts[0]! * il, ts[2]! * il, -ts[1]! * il];
    const world = c.world(local);
    const albedo = 0.5;
    const cos = Math.max(0, world[0]! * l[0] + world[1]! * l[1] + world[2]! * l[2]);
    const scene = plane(c.name, { n: solid(code, 'linear') }, { m: { type: 'pbr', baseColor: [albedo, albedo, albedo], roughness: 1, metalness: 0, normalTexture: 'n' } }, c.transform);
    const want = (albedo / Math.PI) * E * cos;
    expectColour(c.name, await render(scene, spp), [want, want, want]);
  }

  // Emission: a camera hit sees emissive * decode(code); the base colour is black so nothing else is added.
  {
    const code: [number, number, number] = [200, 100, 50];
    const scene = plane('tex-emission', { t: solid(code, 'srgb') }, { m: { type: 'pbr', baseColor: [0, 0, 0], roughness: 1, metalness: 0, emissive: [2, 2, 2], emissiveTexture: 't' } }, undefined, false);
    expectColour('emission', await render(scene, spp), code.map((c) => 2 * srgbToLinear(c)) as [number, number, number]);
  }

  // Metallic-roughness texture with metalness 0 everywhere (B = 0) and roughness from G: still the Lambertian (pt thresholds metalness at 0.5).
  {
    const scene = plane('tex-mr', { t: solid([0, 128, 0], 'linear') }, { m: { type: 'pbr', baseColor: [0.5, 0.5, 0.5], roughness: 1, metalness: 1, metallicRoughnessTexture: 't' } });
    const want = (0.5 / Math.PI) * E * l[1];
    expectColour('mr_metalness_from_b', await render(scene, spp), [want, want, want]);
  }
}

try {
  const args = getArgs();
  const count = numberArg(args, 'cases', 3000);
  const tol = numberArg(args, 'tol', 0.01);
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const device = await adapter.requestDevice();
  deviceErrors = watchDevice(device);
  await checkLookups(device, count, tol);
  await drainDevice(device);
  await checkEndToEnd(numberArg(args, 'spp', 16));

  const total = Object.entries(metrics).filter(([k]) => k.endsWith('_mismatch')).reduce((sum, [, v]) => sum + v, 0);
  reportResult(
    { scene: 'texture-check', renderer: 'ours', config: { cases: count, tol }, valid: total === 0 && deviceErrors.length === 0, metrics: { ...metrics, mismatches: total }, notes: { problems } },
    { adapter: adapterSummary, deviceErrors },
  );
  if (total > 0) throw new Error(`texture-check: ${total} mismatches. First:\n${problems.join('\n')}`);
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
