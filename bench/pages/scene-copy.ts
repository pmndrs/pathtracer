/**
 * scene-copy (P1.4): correctness of the scene copy's WGSL against its TypeScript twins. No timing.
 *
 * Builds scene copies (Cornell and quad-light from the bench converter, a synthetic scene with 300 random materials,
 * mirrored instances, a mixed emissive mesh, a sun and an environment, and a 20,000-light stress scene), uploads them, and
 * on the GPU:
 *  - `rtMaterial` for every id (plus ids past the table and `RT_NO_MATERIAL`): every field, bit for bit against `readMaterial`;
 *  - `rtLight` for every emissive triangle, `rtSun`, `rtEnvironment`, `rtSceneInfo`: bit for bit;
 *  - `rtSampleLight` for stratified and random u: index and pmf identical to `sampleAlias`, the leftover u within 1e-5; the
 *    GPU's pick counts against power-proportional expectations;
 *  - `rtLightIndexOfHit` / `rtLightPmfOfHit` for every (instance, triangle) of the small scenes and a sample of the big one;
 *  - `rtSurface` (scene source) against `readSurfaceFromTable` (the same CPU decode the slice-1 table is tested with).
 * The module under test is the full inline module (bvh walk + core + scene copy), so a WGSL compile error anywhere fails the page.
 *
 * Args: scenes=cornell,quad-light,mixed,many-lights   lights=20000 (size of the stress scene)   packing=both|split|packed (P3.11: `packed` binds the
 * scene copy as ONE storage buffer, header + table + data; every check above runs on both and must stay bit-exact; `tRaw` also reads raw table / data words)
 */
import {
  SCENE_HEADER,
  SCENE_NONE,
  SURFACE,
  buildSceneCopy,
  createMaterial,
  defaultMaterial,
  detectCaps,
  f32Bits,
  readLight,
  readLightIndexOfHit,
  readLightPmfOfHit,
  readMaterial,
  readSurfaceFromTable,
  requestDeviceForCaps,
  rtModuleWgsl,
  rtSceneCopyBindGroupEntries,
  rtSceneCopyLayoutEntries,
  rtPackSceneAccess,
  sampleAlias,
  type SceneCopyPacking,
  uploadSceneCopy,
  writeSceneCopy,
  type RtMaterial,
  type RtTriangleLight,
  type SceneCopy,
  type SceneCopyInput,
  type SurfaceHitInput,
} from '@pmndrs/raytracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, watchDevice } from '../runner/page-client.ts';
import { buildSceneCopyFromDesc } from '../scenes/to-scene-copy.ts';
import { cornell, quadLight } from '../scenes/tier0/index.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];

// ---------------------------------------------------------------------------------------------------------------
// Field lists: the kernel's writers and the TypeScript flatteners come from the same list, so they cannot drift.
// ---------------------------------------------------------------------------------------------------------------

type FieldType = 'f32' | 'u32' | 'bool' | 'vec3f';
type Field = readonly [name: string, type: FieldType];

const U32_FIELDS = new Set(['alphaMode', 'baseColorTexture', 'roughnessMetalTexture', 'normalTexture', 'emissionTexture']);
// The thin-film fields live in the material record but not in the WGSL `RtMaterial` (read through `ptMaterialFilm`; `lobe-oracle` checks them).
const MATERIAL_FIELDS: Field[] = Object.entries(defaultMaterial()).filter(([name]) => !name.startsWith('thinFilm')).map(([name, value]) => [name, Array.isArray(value) ? 'vec3f' : typeof value === 'boolean' ? 'bool' : U32_FIELDS.has(name) ? 'u32' : 'f32'] as Field);
const LIGHT_FIELDS: Field[] = [
  ['kind', 'u32'], ['position', 'vec3f'], ['edge1', 'vec3f'], ['edge2', 'vec3f'], ['normal', 'vec3f'], ['area', 'f32'], ['pmf', 'f32'], ['radiance', 'vec3f'], ['material', 'u32'], ['instance', 'u32'], ['triangle', 'u32'], ['twoSided', 'bool'],
];
const SUN_FIELDS: Field[] = [['index', 'u32'], ['direction', 'vec3f'], ['angularRadius', 'f32'], ['cosAngularRadius', 'f32'], ['solidAngle', 'f32'], ['irradiance', 'vec3f'], ['pmf', 'f32']];
const ENV_FIELDS: Field[] = [['index', 'u32'], ['radiance', 'vec3f'], ['scale', 'f32'], ['rotationY', 'f32'], ['mapSlot', 'u32'], ['pmf', 'f32']];

const words = (fields: readonly Field[]) => fields.reduce((n, [, type]) => n + (type === 'vec3f' ? 3 : 1), 0);

/** WGSL statements writing `value.<field>` into `out` at `base + k`. */
function writerWgsl(fields: readonly Field[], value: string, out: string, base: string): string {
  let k = 0;
  const lines: string[] = [];
  for (const [name, type] of fields) {
    const v = `${value}.${name}`;
    if (type === 'vec3f') {
      for (const c of ['x', 'y', 'z']) lines.push(`    ${out}[${base} + ${k++}u] = bitcast<u32>(${v}.${c});`);
    } else if (type === 'f32') lines.push(`    ${out}[${base} + ${k++}u] = bitcast<u32>(${v});`);
    else if (type === 'u32') lines.push(`    ${out}[${base} + ${k++}u] = ${v};`);
    else lines.push(`    ${out}[${base} + ${k++}u] = select(0u, 1u, ${v});`);
  }
  return lines.join('\n');
}

function flatten(fields: readonly Field[], record: Record<string, unknown>): number[] {
  const out: number[] = [];
  for (const [name, type] of fields) {
    const v = record[name];
    if (type === 'vec3f') for (const c of v as number[]) out.push(f32Bits(c));
    else if (type === 'f32') out.push(f32Bits(v as number));
    else if (type === 'u32') out.push((v as number) >>> 0);
    else out.push(v ? 1 : 0);
  }
  return out;
}

const KERNELS = /* wgsl */ `
struct TParams { count: u32, a: u32, b: u32, c: u32 }
@group(1) @binding(0) var<storage, read> tIn: array<u32>;
@group(1) @binding(1) var<storage, read_write> tOut: array<u32>;
@group(1) @binding(2) var<uniform> tParams: TParams;

@compute @workgroup_size(64)
fn tMaterials(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= tParams.count) { return; }
    let m = rtMaterial(tIn[id.x]);
    let o = id.x * ${words(MATERIAL_FIELDS)}u;
${writerWgsl(MATERIAL_FIELDS, 'm', 'tOut', 'o')}
}

@compute @workgroup_size(64)
fn tLights(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= tParams.count) { return; }
    let l = rtLight(id.x);
    let o = id.x * ${words(LIGHT_FIELDS)}u;
${writerWgsl(LIGHT_FIELDS, 'l', 'tOut', 'o')}
}

@compute @workgroup_size(1)
fn tSunEnv() {
    let s = rtSun();
    let e = rtEnvironment();
    let i = rtSceneInfo();
${writerWgsl(SUN_FIELDS, 's', 'tOut', '0u')}
${writerWgsl(ENV_FIELDS, 'e', 'tOut', `${words(SUN_FIELDS)}u`)}
    let base = ${words(SUN_FIELDS) + words(ENV_FIELDS)}u;
    tOut[base] = i.instanceSlots;
    tOut[base + 1u] = i.materialCount;
    tOut[base + 2u] = i.lightCount;
    tOut[base + 3u] = i.triangleLightCount;
    tOut[base + 4u] = bitcast<u32>(i.totalPower);
    tOut[base + 5u] = bitcast<u32>(i.sceneRadius);
    tOut[base + 6u] = select(0u, 1u, rtHasSun());
    tOut[base + 7u] = select(0u, 1u, rtHasEnvironment());
}

@compute @workgroup_size(64)
fn tSample(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let i = id.x;
    if (i >= tParams.count) { return; }
    let s = rtSampleLight(bitcast<f32>(tIn[i]));
    tOut[i * 4u] = s.index;
    tOut[i * 4u + 1u] = bitcast<u32>(s.pmf);
    tOut[i * 4u + 2u] = bitcast<u32>(s.u);
    tOut[i * 4u + 3u] = rtLightCount();
}

@compute @workgroup_size(64)
fn tHits(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= tParams.count) { return; }
    let instance = tIn[id.x * 2u];
    let triangle = tIn[id.x * 2u + 1u];
    tOut[id.x * 2u] = rtLightIndexOfHit(instance, triangle);
    tOut[id.x * 2u + 1u] = bitcast<u32>(rtLightPmfOfHit(instance, triangle));
}

@compute @workgroup_size(64)
fn tRaw(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= tParams.count) { return; }
    let w = rtSceneData[tIn[id.x * 2u]];
    let t = rtSceneTable[tIn[id.x * 2u + 1u]];
    tOut[id.x * 5u] = w;
    tOut[id.x * 5u + 1u] = t.x; tOut[id.x * 5u + 2u] = t.y; tOut[id.x * 5u + 3u] = t.z; tOut[id.x * 5u + 4u] = t.w;
}

@compute @workgroup_size(64)
fn tSurface(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= tParams.count) { return; }
    let q = id.x * 16u;
    let f = array<f32, 16>(
        bitcast<f32>(tIn[q]), bitcast<f32>(tIn[q + 1u]), bitcast<f32>(tIn[q + 2u]), bitcast<f32>(tIn[q + 3u]),
        bitcast<f32>(tIn[q + 4u]), bitcast<f32>(tIn[q + 5u]), bitcast<f32>(tIn[q + 6u]), 0.0,
        bitcast<f32>(tIn[q + 8u]), bitcast<f32>(tIn[q + 9u]), 0.0, 0.0,
        bitcast<f32>(tIn[q + 12u]), bitcast<f32>(tIn[q + 13u]), bitcast<f32>(tIn[q + 14u]), 0.0);
    let ray = RtRay(vec3f(f[0], f[1], f[2]), 0.0, vec3f(f[4], f[5], f[6]), 1.0e30);
    let hit = RtHit(f[3], tIn[q + 7u], tIn[q + 10u], tIn[q + 11u] != 0u, vec2f(f[8], f[9]), tIn[q + 15u], vec3f(f[12], f[13], f[14]), 0.0);
    let s = rtSurface(hit, ray, RT_SURF_ALL);
    let o = id.x * 16u;
    tOut[o] = bitcast<u32>(s.position.x); tOut[o + 1u] = bitcast<u32>(s.position.y); tOut[o + 2u] = bitcast<u32>(s.position.z);
    tOut[o + 3u] = bitcast<u32>(s.geometricNormal.x); tOut[o + 4u] = bitcast<u32>(s.geometricNormal.y); tOut[o + 5u] = bitcast<u32>(s.geometricNormal.z);
    tOut[o + 6u] = select(0u, 1u, s.frontFace);
    tOut[o + 7u] = bitcast<u32>(s.shadingNormal.x); tOut[o + 8u] = bitcast<u32>(s.shadingNormal.y); tOut[o + 9u] = bitcast<u32>(s.shadingNormal.z);
    tOut[o + 10u] = bitcast<u32>(s.uv.x); tOut[o + 11u] = bitcast<u32>(s.uv.y);
    tOut[o + 12u] = s.material;
}
`;

// ---------------------------------------------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------------------------------------------

function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

const matrix = (t: [number, number, number], angle: number, s: [number, number, number]): number[] => {
  const c = Math.cos(angle);
  const sn = Math.sin(angle);
  return [c * s[0], 0, -sn * s[0], 0, 0, s[1], 0, 0, sn * s[2], 0, c * s[2], 0, t[0], t[1], t[2], 1];
};

function randomMaterials(count: number, seed: number): RtMaterial[] {
  const rnd = random(seed);
  const c3 = (scale = 1): [number, number, number] => [rnd() * scale, rnd() * scale, rnd() * scale];
  const out: RtMaterial[] = [];
  for (let i = 0; i < count; i++) {
    const edge = i < 6;
    out.push(
      createMaterial({
        baseColor: edge ? ([[0, 0, 0], [1, 1, 1], [3e38, 1e-38, 0.5]] as const)[i % 3]!.map((v) => v) as [number, number, number] : c3(2),
        baseWeight: edge ? i % 2 : rnd(),
        metalness: rnd(),
        specularRoughness: edge ? i % 2 : rnd(),
        specularWeight: rnd(),
        specularColor: c3(),
        ior: 1 + rnd() * 2.5,
        specularAnisotropy: rnd(),
        anisotropyRotation: rnd(),
        conductorEta: c3(5),
        conductorK: c3(9),
        transmissionWeight: rnd(),
        transmissionColor: c3(),
        transmissionDepth: rnd() * 20,
        coatWeight: rnd(),
        coatColor: c3(),
        coatRoughness: rnd(),
        coatIor: 1 + rnd(),
        sheenWeight: rnd(),
        sheenColor: c3(),
        sheenRoughness: rnd(),
        emissionColor: rnd() < 0.15 ? c3(30) : [0, 0, 0],
        opacity: rnd(),
        alphaCutoff: rnd(),
        alphaMode: Math.floor(rnd() * 3),
        thinWalled: rnd() < 0.5,
        conductorEtaK: rnd() < 0.5,
        emissionTwoSided: rnd() < 0.5,
        doubleSided: rnd() < 0.5,
        // Texture slots must name a texture the scene has (`buildSceneCopy` checks, P1.4c): none here; the slot packing is material.test.ts and texture-check.
        baseColorTexture: 0xffff,
        roughnessMetalTexture: 0xffff,
        normalTexture: 0xffff,
        emissionTexture: 0xffff,
      }),
    );
  }
  return out;
}

/** Quad (XZ plane, normal -Y) with normals and UVs. */
const quadMesh = () => ({
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
  indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  normals: new Float32Array([0, -1, 0.2, 0, -1, -0.2, 0, -1, 0.1, 0, -1, 0]),
  uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
});

function mixedScene(): SceneCopyInput {
  const strip = {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0, 1, 3, 0, 0, 3, 0, 1]),
    indices: new Uint32Array([0, 1, 2, 2, 1, 3, 1, 4, 3, 3, 4, 5]),
    normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    uvs: new Float32Array(16).map((_, i) => i * 0.1),
    triangleMaterials: [1, 0xffff, 2, 0xffff],
  };
  const materials = randomMaterials(300, 11);
  materials[1] = createMaterial({ emissionColor: [2, 4, 6] });
  materials[2] = createMaterial({ emissionColor: [1, 1, 1], emissionTwoSided: true });
  return {
    meshes: [quadMesh(), strip],
    instances: [
      { meshIndex: 0, instanceIndex: 0, objectToWorld: matrix([0, 3, 0], 0.4, [2, 1, 3]), material: 1 },
      { meshIndex: 0, instanceIndex: 1, objectToWorld: matrix([5, 3, 0], -0.7, [-1.5, 1, 1]), material: 2 },
      { meshIndex: 1, instanceIndex: 3, objectToWorld: matrix([0, 0, 5], 0.2, [1, 1, 1]), material: 0 },
      { meshIndex: 0, instanceIndex: 4, objectToWorld: matrix([0, 0, 0], 0, [1, 1, 1]), material: 1, visible: false },
      { meshIndex: 0, instanceIndex: 6, objectToWorld: matrix([1, 1, 1], 1.1, [0.5, 2, 0.5]), material: 299 },
      { meshIndex: 0, instanceIndex: 7, objectToWorld: matrix([0, 0, 0], 0, [1, 1, 1]) },
    ],
    materials,
    sun: { direction: [0.3, -0.8, 0.4], irradiance: [50, 46, 40], angularRadius: 0.00465 },
    environment: { radiance: [2, 2.3, 3], scale: 1.5, rotationY: 0.5, mapSlot: 3 },
    options: { surface: { positions: false } },
  };
}

/** One mesh of `count` emissive triangles whose emission varies over four decades: stresses the alias table. */
function manyLights(count: number): SceneCopyInput {
  const rnd = random(5);
  const positions = new Float32Array(count * 9);
  const indices = new Uint32Array(count * 3);
  for (let t = 0; t < count; t++) {
    const base = [rnd() * 10, rnd() * 10, rnd() * 10];
    for (let k = 0; k < 3; k++) for (let c = 0; c < 3; c++) positions[t * 9 + k * 3 + c] = (base[c] as number) + (k === 0 ? 0 : rnd() * (0.05 + 0.5 * rnd()));
    indices.set([t * 3, t * 3 + 1, t * 3 + 2], t * 3);
  }
  const triangleMaterials = new Uint16Array(count).map((_, t) => 1 + (t % 7));
  const materials = [createMaterial({}), ...[0.01, 0.1, 1, 10, 100, 1000, 100].map((v) => createMaterial({ emissionColor: [v, v * 0.5, v * 0.25] }))];
  return { meshes: [{ positions, indices, triangleMaterials }], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: matrix([0, 0, 0], 0, [1, 1, 1]), material: 0 }], materials, sun: { direction: [0, -1, 0], irradiance: [5, 5, 5] } };
}

function loadScene(name: string, lightsCount: number): { copy: SceneCopy; input: SceneCopyInput } {
  if (name === 'cornell') return buildSceneCopyFromDesc(cornell, { copy: { surface: { positions: true } } });
  if (name === 'quad-light') return buildSceneCopyFromDesc(quadLight);
  const input = name === 'mixed' ? mixedScene() : name === 'many-lights' ? manyLights(lightsCount) : null;
  if (!input) throw new Error(`unknown scene '${name}'`);
  return { copy: buildSceneCopy(input), input };
}

// ---------------------------------------------------------------------------------------------------------------
// GPU plumbing
// ---------------------------------------------------------------------------------------------------------------

interface Runner {
  run(entry: string, input: Uint32Array, count: number, outWords: number, scene: ReturnType<typeof uploadSceneCopy>): Promise<Uint32Array>;
  bytes: number;
}

async function makeRunner(device: GPUDevice, code: string, packing: SceneCopyPacking): Promise<Runner> {
  const module = device.createShaderModule({ label: 'scene-copy test', code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`WGSL compile errors:\n${errors.slice(0, 10).map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n')}`);
  const sceneLayout = device.createBindGroupLayout({ entries: rtSceneCopyLayoutEntries(0, undefined, packing) });
  const ioLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
  });
  const emptyLayout = device.createBindGroupLayout({ entries: [] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [emptyLayout, ioLayout, sceneLayout] });
  const pipelines = new Map<string, GPUComputePipeline>();
  const pipeline = async (entry: string) => {
    let p = pipelines.get(entry);
    if (!p) {
      p = await device.createComputePipelineAsync({ label: entry, layout: pipelineLayout, compute: { module, entryPoint: entry } });
      pipelines.set(entry, p);
    }
    return p;
  };
  return {
    bytes: code.length,
    async run(entry, input, count, outWords, scene) {
      const pipe = await pipeline(entry);
      const make = (usage: number, size: number) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
      const inBuffer = make(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, input.byteLength);
      if (input.byteLength) device.queue.writeBuffer(inBuffer, 0, input.buffer as ArrayBuffer, input.byteOffset, input.byteLength);
      const outBytes = Math.max(16, outWords * 4);
      const outBuffer = make(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, outBytes);
      const read = make(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, outBytes);
      const params = make(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 16);
      device.queue.writeBuffer(params, 0, new Uint32Array([count, 0, 0, 0]));
      const io = device.createBindGroup({ layout: ioLayout, entries: [{ binding: 0, resource: { buffer: inBuffer } }, { binding: 1, resource: { buffer: outBuffer } }, { binding: 2, resource: { buffer: params } }] });
      const sceneGroup = device.createBindGroup({ layout: sceneLayout, entries: rtSceneCopyBindGroupEntries(scene) });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(1, io);
      pass.setBindGroup(2, sceneGroup);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / 64)));
      pass.end();
      encoder.copyBufferToBuffer(outBuffer, 0, read, 0, outBytes);
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const result = new Uint32Array(read.getMappedRange().slice(0)).subarray(0, outWords);
      read.unmap();
      for (const b of [inBuffer, outBuffer, read, params]) b.destroy();
      return result;
    },
  };
}

class Tally {
  checks = 0;
  mismatches: string[] = [];
  count: Record<string, number> = {};
  worst: Record<string, number> = {};
  expectEqual(category: string, what: string, a: ArrayLike<number>, b: ArrayLike<number>): void {
    this.count[category] = (this.count[category] ?? 0) + a.length;
    for (let i = 0; i < a.length; i++) {
      this.checks++;
      if (a[i] !== b[i]) this.mismatches.push(`${category} ${what}[${i}]: gpu 0x${(a[i] as number).toString(16)} cpu 0x${(b[i] as number).toString(16)}`);
    }
  }
  near(category: string, what: string, gpu: number, cpu: number, tolerance: number): void {
    this.count[category] = (this.count[category] ?? 0) + 1;
    this.checks++;
    const error = Math.abs(gpu - cpu) / Math.max(1, Math.abs(cpu));
    this.worst[category] = Math.max(this.worst[category] ?? 0, error);
    if (!(error <= tolerance)) this.mismatches.push(`${category} ${what}: gpu ${gpu} cpu ${cpu}`);
  }
}

const asF32 = (word: number): number => new Float32Array(Uint32Array.of(word).buffer)[0] as number;

async function checkScene(name: string, copy: SceneCopy, input: SceneCopyInput, runner: Runner, scene: ReturnType<typeof uploadSceneCopy>, tally: Tally): Promise<void> {
  const tag = (s: string) => `${name}/${s}`;

  // Materials: every id, plus out-of-range ones.
  const ids = [...Array.from({ length: copy.materials.length + 1 }, (_, i) => i), copy.materials.length + 7, 0xffff, 0x10000, SCENE_NONE];
  const mWords = words(MATERIAL_FIELDS);
  const mOut = await runner.run('tMaterials', Uint32Array.from(ids), ids.length, ids.length * mWords, scene);
  ids.forEach((id, i) => tally.expectEqual('material_words', tag(`material ${id}`), mOut.subarray(i * mWords, (i + 1) * mWords), flatten(MATERIAL_FIELDS, readMaterial(copy, id) as unknown as Record<string, unknown>)));

  // Raw words: the first, the last and 5000 random data words, table vec4s (the same indices; every one in range).
  {
    const rnd = random(0x5eed);
    const tableVec4s = copy.table.length / 4;
    const n = 5000;
    const pairs = new Uint32Array(n * 2);
    for (let i = 0; i < n; i++) {
      pairs[i * 2] = i === 0 ? 0 : i === 1 ? copy.data.length - 1 : Math.floor(rnd() * copy.data.length);
      pairs[i * 2 + 1] = i === 0 ? 0 : i === 1 ? tableVec4s - 1 : Math.floor(rnd() * tableVec4s);
    }
    const raw = await runner.run('tRaw', pairs, n, n * 5, scene);
    for (let i = 0; i < n; i++) {
      const d = pairs[i * 2] as number;
      const t = pairs[i * 2 + 1] as number;
      tally.expectEqual('raw_words', tag(`raw ${i} (data ${d}, table ${t})`), raw.subarray(i * 5, i * 5 + 5), [copy.data[d] as number, ...Array.from(copy.table.subarray(t * 4, t * 4 + 4))]);
    }
  }

  // Light records.
  const triangleLights = copy.triangleLightCount;
  const lWords = words(LIGHT_FIELDS);
  if (triangleLights > 0) {
    const lOut = await runner.run('tLights', new Uint32Array(1), triangleLights, triangleLights * lWords, scene);
    for (let i = 0; i < triangleLights; i++) {
      const read = readLight(copy, i);
      const light = read.light as RtTriangleLight;
      const record = { kind: 0, position: light.p0, edge1: light.edge1, edge2: light.edge2, normal: light.normal, area: light.area, pmf: read.pmf, radiance: light.radiance, material: light.material, instance: light.instance, triangle: light.triangle, twoSided: light.twoSided };
      tally.expectEqual('light_words', tag(`light ${i}`), lOut.subarray(i * lWords, (i + 1) * lWords), flatten(LIGHT_FIELDS, record));
    }
  }

  // Sun, environment, info.
  const sWords = words(SUN_FIELDS);
  const eWords = words(ENV_FIELDS);
  const seOut = await runner.run('tSunEnv', new Uint32Array(1), 1, sWords + eWords + 8, scene);
  const sun = copy.sunLight === SCENE_NONE ? null : readLight(copy, copy.sunLight);
  const env = copy.environmentLight === SCENE_NONE ? null : readLight(copy, copy.environmentLight);
  if (sun && sun.light.kind === 'sun') {
    tally.expectEqual('sun_words', tag('sun'), seOut.subarray(0, sWords), flatten(SUN_FIELDS, { index: copy.sunLight, direction: sun.light.direction, angularRadius: sun.light.angularRadius, cosAngularRadius: Math.fround(Math.cos(sun.light.angularRadius)), solidAngle: sun.light.solidAngle, irradiance: sun.light.irradiance, pmf: sun.pmf }));
  } else tally.expectEqual('sun_words', tag('no sun'), [seOut[0] as number, seOut[sWords + eWords + 6] as number], [SCENE_NONE, 0]);
  if (env && env.light.kind === 'environment') {
    tally.expectEqual('env_words', tag('env'), seOut.subarray(sWords, sWords + eWords), flatten(ENV_FIELDS, { index: copy.environmentLight, radiance: env.light.radiance, scale: env.light.scale, rotationY: env.light.rotationY, mapSlot: env.light.mapSlot, pmf: env.pmf }));
  } else tally.expectEqual('env_words', tag('no env'), [seOut[sWords] as number, seOut[sWords + eWords + 7] as number], [SCENE_NONE, 0]);
  const base = sWords + eWords;
  tally.expectEqual('info_words', tag('info'), seOut.subarray(base, base + 6), [copy.table[SCENE_HEADER.instanceSlots] as number, copy.materials.length, copy.lights.length, copy.triangleLightCount, copy.table[SCENE_HEADER.totalPower] as number, copy.table[SCENE_HEADER.sceneRadius] as number]);

  // Alias sampling: stratified plus random u, then counts against power.
  if (copy.lights.length > 0) {
    const n = 1 << 18;
    const rnd = random(77);
    const us = new Float32Array(n);
    for (let i = 0; i < n; i++) us[i] = i % 2 === 0 ? Math.min((i / 2 + 0.5) / (n / 2), 1 - 2 ** -24) : Math.min(rnd(), 1 - 2 ** -24);
    us[0] = 0;
    us[1] = 1 - 2 ** -24;
    const sampled = await runner.run('tSample', new Uint32Array(us.buffer), n, n * 4, scene);
    const counts = new Float64Array(copy.lights.length);
    let wrongIndex = 0;
    let wrongPmf = 0;
    let worstU = 0;
    for (let i = 0; i < n; i++) {
      const cpu = sampleAlias(copy.alias, us[i] as number);
      const index = sampled[i * 4] as number;
      if (index !== cpu.index) wrongIndex++;
      if (sampled[i * 4 + 1] !== f32Bits(cpu.pmf)) wrongPmf++;
      worstU = Math.max(worstU, Math.abs(asF32(sampled[i * 4 + 2] as number) - cpu.u));
      if (i % 2 === 0) counts[index] = (counts[index] as number) + 1;
    }
    tally.checks += 3;
    tally.count.sample_u = (tally.count.sample_u ?? 0) + n;
    if (wrongIndex) tally.mismatches.push(`${tag('sample')}: ${wrongIndex} picks differ from the CPU twin`);
    if (wrongPmf) tally.mismatches.push(`${tag('sample')}: ${wrongPmf} pmf values differ`);
    tally.worst.sample_leftover_u = Math.max(tally.worst.sample_leftover_u ?? 0, worstU);
    if (worstU > 1e-5) tally.mismatches.push(`${tag('sample')}: leftover u differs by ${worstU}`);
    // Stratified half: the expected count of light i is (n/2) * pmf. A light fed by k alias segments is off by about sqrt(k) at
    // worst (every segment end rounds one sample), so the bound is 4 + sqrt(expected) / 2; worst_sample_count_error is the
    // largest deviation over that bound (<= 1 passes).
    let worstRatio = 0;
    copy.lights.forEach((_, i) => {
      const expected = (n / 2) * readLight(copy, i).pmf;
      worstRatio = Math.max(worstRatio, Math.abs((counts[i] as number) - expected) / (4 + 0.5 * Math.sqrt(expected)));
    });
    tally.worst.sample_count_error = Math.max(tally.worst.sample_count_error ?? 0, worstRatio);
    tally.checks++;
    if (worstRatio > 1) tally.mismatches.push(`${tag('sample')}: a light was picked ${worstRatio.toFixed(2)}x past the allowed deviation from its expected count`);
  }

  // Triangle -> light map: every (instance, triangle) of small scenes, a sample of the big one.
  const queries: number[] = [];
  for (const instance of input.instances) {
    const mesh = input.meshes[instance.meshIndex]!;
    const triangles = (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3;
    const step = Math.max(1, Math.floor(triangles / 4096));
    for (let t = 0; t < triangles; t += step) queries.push(instance.instanceIndex, t);
  }
  queries.push(copy.table[SCENE_HEADER.instanceSlots] as number, 0, 1_000_000, 0); // unknown instances (a hit never reports an out-of-range triangle, so none is tested)
  const hits = await runner.run('tHits', Uint32Array.from(queries), queries.length / 2, queries.length, scene);
  for (let i = 0; i < queries.length / 2; i++) {
    const index = readLightIndexOfHit(copy, queries[i * 2] as number, queries[i * 2 + 1] as number);
    tally.expectEqual('hit_light_words', tag(`hit ${queries[i * 2]}/${queries[i * 2 + 1]}`), [hits[i * 2] as number, hits[i * 2 + 1] as number], [index, f32Bits(readLightPmfOfHit(copy, queries[i * 2] as number, queries[i * 2 + 1] as number))]);
  }

  // rtSurface.
  const rnd = random(31);
  const surfaceQueries: { hit: SurfaceHitInput; ray: { origin: [number, number, number]; direction: [number, number, number] } }[] = [];
  for (const instance of input.instances) {
    const mesh = input.meshes[instance.meshIndex]!;
    const triangles = (mesh.indices ? mesh.indices.length : mesh.positions.length / 3) / 3;
    for (let k = 0; k < Math.min(triangles, 64); k++) {
      const t = triangles <= 64 ? k : Math.floor(rnd() * triangles);
      const u = rnd() * 0.5;
      const v = rnd() * 0.5;
      const d: [number, number, number] = [rnd() - 0.5, -0.5 - rnd(), rnd() - 0.5];
      const len = Math.hypot(...d);
      const direction = d.map((c) => Math.fround(c / len)) as [number, number, number];
      surfaceQueries.push({
        hit: { t: 1 + rnd() * 4, instance: instance.instanceIndex, triangle: t, barycentrics: [u, v], frontFace: rnd() < 0.5, material: Math.floor(rnd() * 300), normal: [0, 1, 0] },
        ray: { origin: [rnd() * 4, 5, rnd() * 4].map(Math.fround) as [number, number, number], direction },
      });
    }
  }
  const q = new Uint32Array(surfaceQueries.length * 16);
  surfaceQueries.forEach(({ hit, ray }, i) => {
    const o = i * 16;
    q.set([f32Bits(ray.origin[0]), f32Bits(ray.origin[1]), f32Bits(ray.origin[2]), f32Bits(hit.t), f32Bits(ray.direction[0]), f32Bits(ray.direction[1]), f32Bits(ray.direction[2]), hit.instance, f32Bits(hit.barycentrics[0]), f32Bits(hit.barycentrics[1]), hit.triangle, hit.frontFace ? 1 : 0, f32Bits(hit.normal[0]), f32Bits(hit.normal[1]), f32Bits(hit.normal[2]), hit.material], o);
  });
  const sOut = await runner.run('tSurface', q, surfaceQueries.length, surfaceQueries.length * 16, scene);
  const scale = Math.max(1, copy.bounds.radius * 2);
  surfaceQueries.forEach(({ hit, ray }, i) => {
    const cpu = readSurfaceFromTable(copy.surface, hit, ray, SURFACE.ALL);
    const o = i * 16;
    const f = (k: number) => asF32(sOut[o + k] as number);
    const what = tag(`surface ${hit.instance}/${hit.triangle}`);
    for (let c = 0; c < 3; c++) {
      tally.near('surface_position', what, f(c), cpu.position[c] as number, 2e-5 * scale);
      tally.near('surface_geometric_normal', what, f(3 + c), cpu.geometricNormal[c] as number, 0);
      tally.near('surface_shading_normal', what, f(7 + c), cpu.shadingNormal[c] as number, 5e-6);
    }
    for (let c = 0; c < 2; c++) tally.near('surface_uv', what, f(10 + c), cpu.uv[c] as number, 5e-6);
    tally.expectEqual('surface_flags', what, [sOut[o + 6] as number, sOut[o + 12] as number], [cpu.frontFace ? 1 : 0, cpu.material]);
  });
}

try {
  const args = getArgs();
  const sceneNames = (args.scenes ?? 'cornell,quad-light,mixed,many-lights').split(',').filter(Boolean);
  const lightsCount = numberArg(args, 'lights', 20000);
  const packingArg = args.packing ?? 'both';
  if (!['both', 'split', 'packed'].includes(packingArg)) throw new Error(`packing must be both, split or packed, got '${packingArg}'`);
  const packings: SceneCopyPacking[] = packingArg === 'both' ? ['split', 'packed'] : [packingArg as SceneCopyPacking];

  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'scene-copy' });
  deviceErrors = watchDevice(device);

  const tally = new Tally();
  const notes: Record<string, unknown> = {};
  const metrics: Record<string, number> = {};
  const runners = new Map<string, Runner>();
  let moduleBytes = 0;
  for (const name of sceneNames) {
    const { copy, input } = loadScene(name, lightsCount);
    for (const packing of packings) {
      const key = `${packing}/${copy.layout.normals}/${copy.layout.uvs}/${copy.layout.positions}`;
      let runner = runners.get(key);
      if (!runner) {
        // The full inline module: bvh walk (never called here), core, scene copy at group 2, then the test kernels (rewritten for the packed binding).
        const code = rtModuleWgsl(caps, { scene: { slot: { group: 2, firstBinding: 0 }, layout: copy.layout, packing } }) + (packing === 'packed' ? rtPackSceneAccess(KERNELS) : KERNELS);
        runner = await makeRunner(device, code, packing);
        runners.set(key, runner);
        moduleBytes = Math.max(moduleBytes, runner.bytes);
      }
      const gpu = uploadSceneCopy(device, copy, { label: `${name} ${packing}`, packing });
      const before = tally.mismatches.length;
      await checkScene(`${name}/${packing}`, copy, input, runner, gpu, tally);
      // In-place rewrite (reflections' `setTransforms` / `setLights` pattern): a rebuilt copy with a brighter sun into the same buffers, every check again.
      if (input.sun && before === tally.mismatches.length) {
        const copy2 = buildSceneCopy({ ...input, sun: { ...input.sun, irradiance: input.sun.irradiance.map((v) => v * 2) as [number, number, number] } });
        const fits = writeSceneCopy(device, gpu, copy2);
        tally.expectEqual('rewrite_fits', `${name}/${packing}`, [fits ? 1 : 0], [1]);
        if (fits) await checkScene(`${name}/${packing}/rewritten`, copy2, input, runner, gpu, tally);
      }
      notes[`${name}/${packing}`] = { lights: copy.lights.length, triangleLights: copy.triangleLightCount, materials: copy.materials.length, bytes: copy.byteLength, storageBytes: gpu.storageBytes, layout: copy.layout, fingerprint: copy.fingerprint, mismatches: tally.mismatches.length - before };
      gpu.destroy();
    }
  }

  await drainDevice(device);
  for (const [k, v] of Object.entries(tally.count)) metrics[`compared_${k}`] = v;
  for (const [k, v] of Object.entries(tally.worst)) metrics[`worst_${k}`] = v;
  metrics.checks = tally.checks;
  metrics.mismatches = tally.mismatches.length;
  metrics.module_bytes = moduleBytes;
  if (deviceErrors.length) tally.mismatches.push(`device errors: ${deviceErrors.join('; ')}`);
  if (tally.mismatches.length) throw new Error(`scene-copy failed: ${tally.mismatches.length} mismatches\n${tally.mismatches.slice(0, 40).join('\n')}`);

  reportResult({ scene: sceneNames.join('+'), renderer: 'rt-scene-copy', config: { scenes: sceneNames, lights: lightsCount, packings, caps: { tier: caps.tier, subgroups: caps.subgroups } }, metrics, notes }, { adapter: adapterSummary, deviceErrors });
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
