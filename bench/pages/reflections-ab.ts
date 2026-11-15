/**
 * P1.5 A/B: reflections' traced-hit shading on the old trace core vs ours, against our path tracer at maxDepth 1.
 *
 *   REFLECTIONS_REPO=<reflections checkout> node bench/runner/run.ts --page reflections-ab [--args frames=1024,spp=1024,size=128]
 *
 * The reflections code is loaded from that checkout by path (`/@fs/...`; the bench server allows `REFLECTIONS_REPO`), not
 * linked: this repo does not depend on reflections. For every scene, the same rays (camera rays, jittered per frame) go through
 * reflections' own WGSL seam (`HIT_WGSL` + the provider's `code(slot)` + a `HitShading`'s `shadeTracedHit`, normals turned to
 * the ray as its traced pass does):
 *
 *   A   `createPmndrsBvhTraceProvider` (the @pmndrs/bvh wrapper reflections had) + `environmentHitShading`: emission + albedo x
 *       (the sky's cosine-weighted mean + shadowed directional light / pi) + the specular lobe on the sky. That is reflections'
 *       hit shading without probes (with probes, the probe irradiance replaces the sky mean; it is not built here).
 *   B   `createRaytracerTraceProvider` (@pmndrs/raytracer: the same bvh walk, its scene copy and `rtShade`) +
 *       `raytracerHitShading`: emission + one light sample (triangle lamps, sun, sky) with its shadow ray, builtin diffuse + GGX.
 *   PT  `PtMegakernel` over B's own bvh buffers and scene copy, maxDepth 1 (emission + direct light with MIS), legacy BSDF
 *       adapter; scenes with `openpbr` materials also in OpenPBR mode with the library's layering (D24: albedo + internal, the
 *       headline) and the stochastic walk (reference mode), so a glossy plastic shows which BSDF model B's builtin matches.
 *
 * Metrics per scene: hit agreement of A and B on pixel-centre rays (t, triangle, instance, material); the mean luminance of A
 * and B over PT on pixels every frame of both hit; the share of those pixels darker than 0.9 x PT. Images (linear PFM, row 0 =
 * top) `<scene>-A`, `<scene>-B`, `<scene>-pt` for the viewer. Correctness only: no timing here (reflections' `--rt-bvh-timing`
 * times the trace pass).
 */
import * as bvh from '@pmndrs/bvh';
import * as rt from '@pmndrs/raytracer';
import { createMaterial, detectCaps, requestDeviceForCaps, sceneFormatOf, type Caps, type RtMaterial, type RtMaterialInput } from '@pmndrs/raytracer';
import { PtMegakernel, ptCameraBasis, ptMaterialFeatures, type PtLookAtCamera } from '@pmndrs/pathtracer';
import type { AdapterSummary } from '../runner/contract.ts';
import { drainDevice, getArgs, numberArg, reportError, reportResult, summarizeAdapter, uploadImage, watchDevice } from '../runner/page-client.ts';
import { sceneDescToCopyInput, type DescSceneCopyInput } from '../scenes/to-scene-copy.ts';
import { cornell } from '../scenes/tier0/index.ts';
import { lightsMix, lightsMixLamps } from '../scenes/ab/lights-mix.ts';
import { compose, rotationX, translation } from '../scenes/geometry.ts';
import type { SceneDesc } from '../scenes/schema.ts';

let adapterSummary: AdapterSummary | null = null;
let deviceErrors: string[] = [];
const metrics: Record<string, number> = {};
const notes: Record<string, unknown> = {};

// ---- the reflections modules (loaded by path) ---------------------------------------------------------------------

type V3 = readonly [number, number, number];
interface Slot { group: number; firstBinding: number }
interface ReflProvider {
  readonly name: string;
  code(slot: Slot): string;
  readonly entries: readonly Omit<GPUBindGroupLayoutEntry, 'binding'>[];
  resources(): readonly GPUBindingResource[];
  destroy(): void;
}
interface ReflRaytracerProvider extends ReflProvider {
  readonly traceBuffers: bvh.TraceSceneBuffers;
  readonly sceneCopy: rt.SceneCopy;
  readonly sceneCopyBuffers: rt.SceneCopyBuffers;
}
interface ReflShading { wgsl: string; layout?: GPUBindGroupLayout }
interface ReflScene {
  meshes: { positions: Float32Array; indices?: Uint32Array | null; normals?: Float32Array | null; triangleMaterials?: Uint32Array | null }[];
  instances: { mesh: number; matrixWorld: ArrayLike<number>; material?: number }[];
  materials: Record<string, unknown>[];
  lights?: { sun?: unknown; environment?: unknown };
}
interface Refl {
  HIT_WGSL: string;
  createPmndrsBvhTraceProvider(device: GPUDevice, scene: ReflScene, options: { module: unknown }): Promise<ReflProvider>;
  createRaytracerTraceProvider(device: GPUDevice, scene: ReflScene, options: { module: unknown; bvhModule: unknown }): Promise<ReflRaytracerProvider>;
  environmentHitShading(device: GPUDevice, mapping: 'equirect'): ReflShading & {
    createBindGroup(view: GPUTextureView): GPUBindGroup;
    setLighting(k: number, lights: { directional?: { direction: V3; color: V3 }[] }): void;
    destroy(): void;
  };
  raytracerHitShading(device: GPUDevice): ReflShading & { bindGroup: GPUBindGroup; setFrame(frame: number): void; destroy(): void };
}

async function loadReflections(repo: string): Promise<Refl> {
  const at = (file: string) => import(/* @vite-ignore */ `/@fs${repo}/packages/core/src/trace/${file}`) as Promise<Record<string, unknown>>;
  const [provider, pmndrs, raytracer, shading] = await Promise.all([at('provider.ts'), at('pmndrsBvh.ts'), at('raytracer.ts'), at('hitShading.ts')]);
  return { ...provider, ...pmndrs, ...raytracer, ...shading } as unknown as Refl;
}

// ---- scenes ----------------------------------------------------------------------------------------------------------

/** A grey floor, a red plastic sphere and a gold sphere (OpenPBR, roughness 0.3) under a lamp, a sun and a sky. */
const glossy: SceneDesc = {
  version: 1,
  name: 'reflections-glossy',
  purpose: 'P1.5: glossy plastic and metal under a lamp, a sun and a sky (reflections A/B only).',
  tier: 1,
  camera: { type: 'perspective', position: [0, 1.6, 3.6], target: [0, 0.5, 0], up: [0, 1, 0], yFovDeg: 40 },
  film: { width: 128, height: 128, filter: 'box' },
  integrator: { maxDepth: 1 },
  materials: {
    floor: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] },
    plastic: { type: 'openpbr', baseColor: [0.7, 0.12, 0.1], specularRoughness: 0.3 },
    gold: { type: 'openpbr', baseColor: [1.0, 0.78, 0.34], baseMetalness: 1, specularRoughness: 0.3 },
    lightBack: { type: 'black' },
  },
  meshes: [
    { name: 'floor', geometry: { type: 'quad', size: [8, 8] }, material: 'floor' },
    { name: 'plastic', geometry: { type: 'sphere', radius: 0.5, segments: 64, rings: 32 }, material: 'plastic', transform: translation(-0.6, 0.5, 0) },
    { name: 'gold', geometry: { type: 'sphere', radius: 0.5, segments: 64, rings: 32 }, material: 'gold', transform: translation(0.6, 0.5, 0) },
    { name: 'lamp', geometry: { type: 'quad', size: [0.8, 0.8] }, material: 'lightBack', transform: compose(translation(0, 2.2, 0.8), rotationX(Math.PI)), emission: [30, 28, 24] },
  ],
  lights: [{ type: 'directional', direction: [0.4, -0.75, -0.5], irradiance: [6, 5.7, 5.1], angularRadius: 0.00465 }],
  environment: { type: 'uniform', radiance: [0.25, 0.3, 0.4] },
};

const SCENES: SceneDesc[] = [cornell, lightsMixLamps, lightsMix, glossy];

const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
const norm = (v: readonly number[]): V3 => {
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  return [v[0]! / l, v[1]! / l, v[2]! / l];
};

/** The seam material of a decoded @pmndrs/raytracer material (what three's `defaultHitMaterial` would give for it). */
function seamMaterial(m: RtMaterial, input: RtMaterialInput): Record<string, unknown> {
  const k = m.baseWeight * (1 - m.metalness) * (1 - m.transmissionWeight);
  const base = m.baseColor.map((c) => c * m.baseWeight);
  const d = (m.ior - 1) / (m.ior + 1);
  const metal = m.conductorEtaK
    ? m.conductorEta.map((eta, i) => ((eta - 1) ** 2 + m.conductorK[i]! ** 2) / ((eta + 1) ** 2 + m.conductorK[i]! ** 2))
    : base;
  const f0 = m.specularColor.map((s, i) => (1 - m.metalness) * s * m.specularWeight * d * d + m.metalness * metal[i]!);
  const hasSpec = m.specularWeight > 0 || m.metalness > 0;
  return {
    albedo: m.baseColor.map((c) => c * k),
    emissive: [...m.emissionColor],
    ...(hasSpec ? { specular: f0, roughness: m.specularRoughness } : {}),
    clearcoat: m.coatWeight,
    clearcoatRoughness: m.coatRoughness,
    // B reads the exact material (its scene copy is the path tracer's); A reads the seam fields above
    openpbr: { ...input },
  };
}

function reflScene(input: DescSceneCopyInput, decoded: RtMaterial[]): ReflScene {
  return {
    meshes: input.meshes.map((m) => ({
      positions: m.positions,
      indices: m.indices ? Uint32Array.from(m.indices as ArrayLike<number>) : null,
      normals: m.normals ? Float32Array.from(m.normals as ArrayLike<number>) : null,
      triangleMaterials: m.triangleMaterials ? Uint32Array.from(m.triangleMaterials as ArrayLike<number>) : null,
    })),
    instances: input.instances.map((i) => ({ mesh: i.meshIndex, matrixWorld: Array.from(i.objectToWorld), ...(i.material !== undefined ? { material: i.material } : {}) })),
    materials: input.materials.map((m, i) => seamMaterial(decoded[i] as RtMaterial, m)),
  };
}

// ---- the harness kernel (group 0) ----------------------------------------------------------------------------------

const HARNESS = /* wgsl */ `
struct AbCam { origin: vec4f, right: vec4f, up: vec4f, forward: vec4f, dims: vec4u }
@group(0) @binding(0) var<uniform> abCam: AbCam;
@group(0) @binding(1) var<storage, read_write> abAccum: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> abHits: array<vec4u>;
var<private> hsoPixel: vec2u = vec2u(0u);
// as the traced pass: shadow-ray origin offset along N by 3e-3 x the distance to the camera (D8)
fn traceLightVisible(P: vec3f, N: vec3f, L: vec3f, tMax: f32) -> bool {
  return !traceAny(P + N * (3e-3 * length(P - abCam.origin.xyz)), L, tMax);
}
fn rtHitSpecularOn() -> bool { return true; }
fn abHash(v: u32) -> u32 {
  let state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn abRand(seed: u32, k: u32) -> f32 { return f32(abHash(seed ^ abHash(k + 0x9e3779b9u)) >> 8u) * 5.9604644775390625e-8; }

@compute @workgroup_size(8, 8)
fn abMain(@builtin(global_invocation_id) id: vec3u) {
  let dims = abCam.dims;
  if (id.x >= dims.x || id.y >= dims.y) { return; }
  let index = id.y * dims.x + id.x;
  let centre = dims.w == 1u;
  let seed = abHash(index * 2654435761u + dims.z * 0x68bc21ebu);
  let j = select(vec2f(abRand(seed, 0u), abRand(seed, 1u)), vec2f(0.5), centre);
  let ndc = vec2f(((f32(id.x) + j.x) / f32(dims.x)) * 2.0 - 1.0, 1.0 - ((f32(id.y) + j.y) / f32(dims.y)) * 2.0);
  let dir = normalize(abCam.forward.xyz + abCam.right.xyz * ndc.x + abCam.up.xyz * ndc.y);
  let origin = abCam.origin.xyz;
  let hit = traceFirstHit(origin, dir, 1e30);
  if (centre) {
    abHits[index] = vec4u(bitcast<u32>(hit.t), hit.triangle, hit.instance, hit.material);
    return;
  }
  if (hit.t < 0.0) { return; }
  // the traced pass's normal: geometric (or -dir), toward the ray, then the shading normal on that side (D1700)
  let hitP = origin + dir * hit.t;
  var hitN = select(-dir, hit.normal, dot(hit.normal, hit.normal) > 0.0);
  hitN = select(hitN, -hitN, dot(hitN, dir) > 0.0);
  let sn = traceHitShadingNormal(hit);
  if (dot(sn, sn) > 1e-20) {
    let n = normalize(sn);
    hitN = select(n, -n, dot(n, hitN) < 0.0);
  }
  hsoPixel = id.xy;
  let L = shadeTracedHit(hit, hitP, hitN, dir);
  abAccum[index] = abAccum[index] + vec4f(L, 1.0);
}
`;

interface Frames { sum: Float32Array; frames: number }

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

class Harness {
  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPUComputePipeline,
    private readonly layout0: GPUBindGroupLayout,
    private readonly providerGroup: GPUBindGroup,
    private readonly shadingGroup: GPUBindGroup | null,
    readonly wgslBytes: number,
  ) {}

  static async create(device: GPUDevice, refl: Refl, provider: ReflProvider, shading: ReflShading, shadingGroup: GPUBindGroup | null): Promise<Harness> {
    const code = `${refl.HIT_WGSL}\n${provider.code({ group: 2, firstBinding: 0 })}\n${shading.wgsl}\n${HARNESS}`;
    const module = device.createShaderModule({ code, label: `reflections-ab ${provider.name}` });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`reflections-ab ${provider.name}: ${errors.map((e) => `${e.lineNum}:${e.linePos} ${e.message} | ${code.split('\n')[e.lineNum - 1]}`).join('\n')}`);
    const C = GPUShaderStage.COMPUTE;
    const layout0 = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'storage' } },
        { binding: 2, visibility: C, buffer: { type: 'storage' } },
      ],
    });
    const layout2 = device.createBindGroupLayout({ entries: provider.entries.map((e, binding) => ({ ...e, binding }) as GPUBindGroupLayoutEntry) });
    const empty = device.createBindGroupLayout({ entries: [] });
    const pipeline = await device.createComputePipelineAsync({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout0, empty, layout2, shading.layout ?? empty] }),
      compute: { module, entryPoint: 'abMain' },
    });
    const providerGroup = device.createBindGroup({ layout: layout2, entries: provider.resources().map((resource, binding) => ({ binding, resource })) });
    return new Harness(device, pipeline, layout0, providerGroup, shadingGroup, code.length);
  }

  private camera(camera: PtLookAtCamera, width: number, height: number, frame: number, centre: boolean): GPUBuffer {
    const basis = ptCameraBasis(camera, width, height);
    const data = new ArrayBuffer(80);
    new Float32Array(data, 0, 16).set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0]);
    new Uint32Array(data, 64, 4).set([width, height, frame, centre ? 1 : 0]);
    const b = this.device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  private dispatch(cams: GPUBuffer[], accum: GPUBuffer, hits: GPUBuffer, width: number, height: number, beforeEach?: (k: number) => void): void {
    cams.forEach((cam, k) => {
      beforeEach?.(k);
      const group = this.device.createBindGroup({ layout: this.layout0, entries: [{ binding: 0, resource: { buffer: cam } }, { binding: 1, resource: { buffer: accum } }, { binding: 2, resource: { buffer: hits } }] });
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, group);
      pass.setBindGroup(2, this.providerGroup);
      if (this.shadingGroup) pass.setBindGroup(3, this.shadingGroup);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
    });
  }

  /** Pixel-centre rays: [t bits, triangle, instance, material] per pixel. */
  async centreHits(camera: PtLookAtCamera, width: number, height: number): Promise<Uint32Array> {
    const d = this.device;
    const hits = d.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const accum = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    const cam = this.camera(camera, width, height, 0, true);
    this.dispatch([cam], accum, hits, width, height);
    const out = new Uint32Array(await readBack(d, hits, width * height * 16));
    for (const b of [hits, accum, cam]) b.destroy();
    return out;
  }

  /** `frames` jittered samples per pixel: rgb sum and the number of frames that hit, per pixel. */
  async accumulate(camera: PtLookAtCamera, width: number, height: number, frames: number, setFrame?: (frame: number) => void): Promise<Frames> {
    const d = this.device;
    const accum = d.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const hits = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
    for (let done = 0; done < frames; ) {
      const n = Math.min(64, frames - done);
      const cams = Array.from({ length: n }, (_, k) => this.camera(camera, width, height, done + k + 1, false));
      this.dispatch(cams, accum, hits, width, height, (k) => setFrame?.(done + k + 1));
      await d.queue.onSubmittedWorkDone();
      cams.forEach((c) => c.destroy());
      done += n;
    }
    const sum = new Float32Array(await readBack(d, accum, width * height * 16));
    accum.destroy();
    hits.destroy();
    return { sum, frames };
  }
}

// ---- one scene ---------------------------------------------------------------------------------------------------------

async function runScene(device: GPUDevice, caps: Caps, refl: Refl, desc: SceneDesc, size: number, frames: number, spp: number): Promise<void> {
  const name = desc.name;
  const input = sceneDescToCopyInput(desc);
  const decoded = rt.buildSceneCopy(input).materials;
  const scene = reflScene(input, decoded);
  const camera = desc.camera as unknown as PtLookAtCamera;
  const width = size, height = size;

  // A: the @pmndrs/bvh wrapper + environment hit shading (sky mean + shadowed sun)
  const providerA = await refl.createPmndrsBvhTraceProvider(device, scene, { module: bvh });
  const env = refl.environmentHitShading(device, 'equirect');
  const sky = input.environment?.radiance ?? [0, 0, 0];
  const skyScale = input.environment?.scale ?? 1;
  const skyTexture = device.createTexture({ size: [1, 1], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: skyTexture }, new Float32Array([sky[0]! * skyScale, sky[1]! * skyScale, sky[2]! * skyScale, 1]), { bytesPerRow: 16 }, [1, 1]);
  const sun = input.sun;
  env.setLighting(1, { directional: sun ? [{ direction: norm(sun.direction.map((x) => -x)), color: [...sun.irradiance] as unknown as V3 }] : [] });
  const harnessA = await Harness.create(device, refl, providerA, env, env.createBindGroup(skyTexture.createView()));

  // B: the @pmndrs/raytracer provider + rtShade
  const providerB = await refl.createRaytracerTraceProvider(device, { ...scene, lights: { sun: input.sun ?? null, environment: input.environment ?? null } }, { module: rt, bvhModule: bvh });
  const shadeB = refl.raytracerHitShading(device);
  const harnessB = await Harness.create(device, refl, providerB, shadeB, shadeB.bindGroup);
  metrics[`${name}_wgsl_bytes_A`] = harnessA.wgslBytes;
  metrics[`${name}_wgsl_bytes_B`] = harnessB.wgslBytes;

  // hit agreement on pixel-centre rays
  const ha = await harnessA.centreHits(camera, width, height);
  const hb = await harnessB.centreHits(camera, width, height);
  const fa = new Float32Array(ha.buffer), fb = new Float32Array(hb.buffer);
  let bothMiss = 0, bothHit = 0, agree = 0, onlyA = 0, onlyB = 0, sameIds = 0;
  for (let i = 0; i < width * height; i++) {
    const ta = fa[i * 4]!, tb = fb[i * 4]!;
    if (ta < 0 && tb < 0) { bothMiss++; agree++; continue; }
    if (ta < 0) { onlyB++; continue; }
    if (tb < 0) { onlyA++; continue; }
    bothHit++;
    if (Math.abs(ta - tb) <= 1e-5 * Math.max(ta, tb)) agree++;
    if (ha[i * 4 + 1] === hb[i * 4 + 1] && ha[i * 4 + 2] === hb[i * 4 + 2] && ha[i * 4 + 3] === hb[i * 4 + 3]) sameIds++;
  }
  metrics[`${name}_rays`] = width * height;
  metrics[`${name}_hit_agree`] = agree;
  metrics[`${name}_hit_only_A`] = onlyA;
  metrics[`${name}_hit_only_B`] = onlyB;
  metrics[`${name}_hits_same_ids`] = sameIds;
  metrics[`${name}_hits_both`] = bothHit;

  // shading
  const imgA = await harnessA.accumulate(camera, width, height, frames);
  const imgB = await harnessB.accumulate(camera, width, height, frames, (f) => shadeB.setFrame(f));

  // PT maxDepth 1 over B's buffers. Scenes with `openpbr` materials run in OpenPBR mode, once per layering: the library default
  // (D24: 'albedo' + 'internal', the headline), the stochastic walk (reference mode), and the legacy adapter (a non-metal is Lambert)
  const openpbr = Object.values(desc.materials).some((m) => m.type === 'openpbr');
  type PtMode = 'legacy' | 'albedo' | 'stochastic';
  const modes: PtMode[] = openpbr ? ['albedo', 'stochastic', 'legacy'] : ['legacy'];
  const renderPt = async (mode: PtMode) => {
    const materials = mode === 'legacy' ? undefined : {
      features: ptMaterialFeatures(input.materials.map((m) => createMaterial(m))),
      layered: mode === 'albedo' ? { layering: 'albedo' as const, albedoBase: 'internal' as const } : { layering: 'stochastic' as const, maxDepth: 64, nSamples: 1 },
      layerThickness: 1e-6, transmissionTint: true,
    };
    const pt = await PtMegakernel.create(device, caps, {
      format: sceneFormatOf(providerB.traceBuffers), layout: providerB.sceneCopy.layout,
      integrator: materials ? { materials } : {}, label: `reflections-ab pt ${name} ${mode}`,
    });
    pt.setScene(providerB.traceBuffers, providerB.sceneCopyBuffers);
    pt.setFilm(width, height);
    pt.setCamera(camera);
    pt.setLimits({ maxDepth: 1 });
    pt.setSeed(7);
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
    const image = await pt.readImage();
    pt.destroy();
    return image;
  };
  const refs = new Map<PtMode, Float32Array>();
  for (const mode of modes) refs.set(mode, (await renderPt(mode)).data);
  const ref = { data: refs.get(modes[0]!)! };
  notes[`${name}_pt_modes`] = modes;

  // compare on pixels every frame of both hit
  const toImage = (f: Frames) => {
    const data = new Float32Array(width * height * 4);
    for (let i = 0; i < width * height; i++) for (let c = 0; c < 3; c++) data[i * 4 + c] = f.sum[i * 4 + c]! / f.frames;
    for (let i = 0; i < width * height; i++) data[i * 4 + 3] = 1;
    return data;
  };
  const a = toImage(imgA), b = toImage(imgB), p = ref.data;
  let sa = 0, sb = 0, sp = 0, n = 0, darkA = 0, darkB = 0, brightA = 0, brightB = 0;
  for (let i = 0; i < width * height; i++) {
    if (imgA.sum[i * 4 + 3] !== frames || imgB.sum[i * 4 + 3] !== frames) continue;
    const ya = lum(a[i * 4]!, a[i * 4 + 1]!, a[i * 4 + 2]!), yb = lum(b[i * 4]!, b[i * 4 + 1]!, b[i * 4 + 2]!), yp = lum(p[i * 4]!, p[i * 4 + 1]!, p[i * 4 + 2]!);
    sa += ya; sb += yb; sp += yp; n++;
    if (yp > 1e-4) {
      if (ya < 0.9 * yp) darkA++;
      if (yb < 0.9 * yp) darkB++;
      if (ya > 1.1 * yp) brightA++;
      if (yb > 1.1 * yp) brightB++;
    }
  }
  // per material (pixel-centre hit's material), emitters apart: the camera sees the lamps' backs in lights-mix*, where A adds
  // emission on either side and the path tracer (and B) only on the front
  // (pixels next to an emitter are left out: their jittered samples land on the lamp, which outweighs the surface)
  const emitterAt = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return false;
    const k = y * width + x;
    return fb[k * 4]! >= 0 && (decoded[hb[k * 4 + 3]!]?.emissionLuminance ?? 0) > 0;
  };
  const perMaterial = new Map<number, { a: number; b: number; p: number; n: number }>();
  const inMaterial = new Int32Array(width * height).fill(-1);
  for (let i = 0; i < width * height; i++) {
    if (imgA.sum[i * 4 + 3] !== frames || imgB.sum[i * 4 + 3] !== frames || fb[i * 4]! < 0) continue;
    const id = hb[i * 4 + 3]!;
    const x = i % width, y = Math.floor(i / width);
    const emissive = (decoded[id]?.emissionLuminance ?? 0) > 0;
    if (!emissive && [-1, 0, 1].some((dy) => [-1, 0, 1].some((dx) => emitterAt(x + dx, y + dy)))) continue;
    const e = perMaterial.get(id) ?? { a: 0, b: 0, p: 0, n: 0 };
    e.a += lum(a[i * 4]!, a[i * 4 + 1]!, a[i * 4 + 2]!); e.b += lum(b[i * 4]!, b[i * 4 + 1]!, b[i * 4 + 2]!); e.p += lum(p[i * 4]!, p[i * 4 + 1]!, p[i * 4 + 2]!); e.n++;
    perMaterial.set(id, e);
    inMaterial[i] = id;
  }
  let surf = { a: 0, b: 0, p: 0, n: 0 };
  for (const [id, e] of perMaterial) {
    const m = decoded[id];
    const label = `${name}_mat_${(input.materialNames[id] ?? String(id)).replace(/[^A-Za-z0-9]+/g, '_').replace(/_+$/, '')}`;
    metrics[`${label}_px`] = e.n;
    metrics[`${label}_A_over_pt`] = e.a / e.p;
    metrics[`${label}_B_over_pt`] = e.b / e.p;
    for (const mode of modes.slice(1)) {
      const r = refs.get(mode)!;
      let sum = 0;
      for (let i = 0; i < width * height; i++) if (inMaterial[i] === id) sum += lum(r[i * 4]!, r[i * 4 + 1]!, r[i * 4 + 2]!);
      metrics[`${label}_A_over_pt_${mode}`] = e.a / sum;
      metrics[`${label}_B_over_pt_${mode}`] = e.b / sum;
    }
    if (m && m.emissionLuminance <= 0) surf = { a: surf.a + e.a, b: surf.b + e.b, p: surf.p + e.p, n: surf.n + e.n };
  }
  metrics[`${name}_surfaces_px`] = surf.n;
  metrics[`${name}_surfaces_A_over_pt`] = surf.a / surf.p;
  metrics[`${name}_surfaces_B_over_pt`] = surf.b / surf.p;
  metrics[`${name}_compared_px`] = n;
  metrics[`${name}_A_over_pt`] = sa / sp;
  metrics[`${name}_B_over_pt`] = sb / sp;
  metrics[`${name}_A_darker_10pc_frac`] = darkA / Math.max(1, n);
  metrics[`${name}_B_darker_10pc_frac`] = darkB / Math.max(1, n);
  metrics[`${name}_A_brighter_10pc_frac`] = brightA / Math.max(1, n);
  metrics[`${name}_B_brighter_10pc_frac`] = brightB / Math.max(1, n);
  await uploadImage(`${name}-A`, { width, height, channels: 4, data: a });
  await uploadImage(`${name}-B`, { width, height, channels: 4, data: b });
  for (const mode of modes) await uploadImage(`${name}-pt${mode === 'legacy' ? '' : `-${mode}`}`, { width, height, channels: 4, data: refs.get(mode)! });

  providerA.destroy();
  providerB.destroy();
  env.destroy();
  shadeB.destroy();
  skyTexture.destroy();
}

try {
  const args = getArgs();
  const repo = args.reflections ?? '';
  if (!repo) throw new Error('reflections-ab: pass --args reflections=<checkout> (and run the runner with REFLECTIONS_REPO=<checkout> so the server serves it)');
  const size = numberArg(args, 'size', 128);
  const frames = numberArg(args, 'frames', 1024);
  const spp = numberArg(args, 'spp', 1024);
  const only = args.scenes ? args.scenes.split(',') : null;
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  adapterSummary = summarizeAdapter(adapter);
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'reflections-ab' });
  deviceErrors = watchDevice(device);
  const refl = await loadReflections(repo);
  for (const desc of SCENES) {
    if (only && !only.includes(desc.name)) continue;
    await runScene(device, caps, refl, desc, size, frames, spp);
    await drainDevice(device);
  }
  notes.method = 'A = reflections @pmndrs/bvh wrapper + environmentHitShading (no probes); B = @pmndrs/raytracer provider + raytracerHitShading (rtShade FULL); PT = PtMegakernel maxDepth 1 over B buffers';
  reportResult(
    { scene: 'reflections-ab', renderer: 'ours', config: { size, frames, spp, repo }, valid: deviceErrors.length === 0, metrics, notes },
    { adapter: adapterSummary, deviceErrors },
  );
} catch (error) {
  reportError(error, { adapter: adapterSummary, deviceErrors });
}
