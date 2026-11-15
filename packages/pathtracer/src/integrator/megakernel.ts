/**
 * Host side of the megakernel path tracer (P2.4): kernel generation, pipeline, accumulation buffer, per-pass dispatch.
 *
 *     const pt = await PtMegakernel.create(device, caps, { format, layout: copy.layout, integrator: { sampler: 'owen-sobol' } });
 *     pt.setScene(traceBuffers, sceneCopyBuffers);                          // bvh walk buffers + the scene copy
 *     pt.setFilm(width, height); pt.setCamera(cameraDesc);
 *     pt.setLimits({ maxDepth: 8, rrStartDepth: 4 });                       // pbrt semantics; rrStartDepth omitted = no RR
 *     pt.encode(encoder, 16);                                               // 16 sample passes, one dispatch each
 *     const image = await pt.readImage();                                   // accum / samples, outside any timed region
 *
 * One dispatch per sample pass, so no single dispatch outlives a GPU watchdog and a pass is the unit of progress. Each pass
 * gets its own record in a ring of uniform slots (dynamic offsets): `queue.writeBuffer` is ordered with `submit`, so a slot
 * is never rewritten before the pass that read it ran, as long as no more than `PT_PARAM_SLOTS` passes are outstanding.
 * Accumulation is f32 `array<vec4f>` (sum of radiance, w = samples); `readImage` divides. Nothing here reads back unless you call it.
 *
 * Real-time frames (P4.1, `docs/design/realtime.md`): `mode: 'realtime'` (or `integrator: { frame: true, sampler: 'bluenoise' }` + `blueNoise`) turns a pass into ONE 1 spp frame
 * that overwrites the accumulator and writes the denoiser guides:
 *     const pt = await PtMegakernel.create(device, caps, { format, layout, mode: 'realtime', blueNoise, integrator: { ... } });
 *     pt.setCamera(cam0);  encode(1);  ...  pt.nextFrame(cam1);  pt.encode(encoder, 1);     // the old camera is the previous one (motion)
 *     const { accum, planes } = pt.frameBuffers;                                              // or: const frame = await pt.readFrame();
 */
import { rtLightSampleWgsl, rtModuleWgsl, rtOwenSobolWgsl, rtPackSceneAccess, DEFAULT_SCENE_PACKING, type SceneCopyPacking, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtMinifyWgsl, rtCreateComputePipelineAsync, rtSceneBindGroupEntries, rtTransformWgsl, rtWalkEntries, stripWgslComments, withBindings } from '@pmndrs/raytracer';
import type { RtWgslBuild } from '@pmndrs/raytracer';
import type { BlueNoiseTexture, Caps, SceneCopyBuffers, SceneFormat, SurfaceLayout, TextureLayout } from '@pmndrs/raytracer';
import { ptBsdfAdapterWgsl, ptBsdfReducedWgsl, ptLayeredDispatchWgsl, ptReducedKernelFeatures } from '../bsdf/adapter.ts';
import { ptBsdfWgsl } from '../bsdf/bsdf-wgsl.ts';
import { ptLayeredWgsl } from '../bsdf/layered-wgsl.ts';
import { PT_LAYER_CORR_BINDING, ptLayerCorrCreateTexture, ptLayerCorrLayoutEntry, ptLayerCorrNeedsTexture } from '../bsdf/layered-correction.ts';
import { ptFuzzTableF32 } from '../bsdf/fuzz.ts';
import { ptFuzzWgsl } from '../bsdf/fuzz-wgsl.ts';
import { ptThinFilmWgsl } from '../bsdf/thinfilm-wgsl.ts';
import { ptCameraBasis, type PtLookAtCamera } from './camera.ts';
import { PT_BLUENOISE_BINDING, ptSamplerWgsl } from './dimensions.ts';
import { ptApplyMode, ptModePreset, type PtMode } from './modes.ts';
import { PT_VBUFFER_BINDING, ptVbufferWgsl } from './vbuffer.ts';
import { PtRestirPasses } from '../restir/restir.ts';
import { PT_RESTIR_SHADE_BINDING } from '../restir/restir-wgsl.ts';
import { PT_PARAMS_BYTES, PT_WORKGROUP, ptIntegratorWgsl, resolveIntegratorOptions, type PtIntegratorOptions } from './integrator-wgsl.ts';
import { PT_AOV_VEC4, PT_COUNTER_WORDS, PT_FRAME_BINDING, PT_FRAME_PARAMS_BINDING, PT_FRAME_PARAMS_BYTES, PT_FRAME_PLANES, ptDecodeAovs, ptDecodeFrame, ptOutputsWgsl, ptSummarizeCounters, type PtAovs, type PtCounters, type PtFrame } from './outputs-wgsl.ts';

// GPUBufferUsage bits spelled out so this file loads in Node (tests).
const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const UNIFORM = 0x40;
const MAP_READ = 0x1;
const COMPUTE = 0x4;

/** Group 2 binding of the debug counters (0 params, 1 accumulator, 2 fuzz table). The hooks' own bind group is group 3. */
export const PT_MEGAKERNEL_COUNTERS_BINDING = 3;

/** Uniform slots in the ring (one per outstanding sample pass). */
export const PT_PARAM_SLOTS = 1024;

type TraceScene = Parameters<typeof rtSceneBindGroupEntries>[0];

export interface PtLimits {
  /** Scattering events per path, pbrt `maxdepth` semantics (1 = direct lighting only; the emission of a camera hit is free). */
  maxDepth: number;
  /** Russian roulette from this scattering vertex (1 = first surface hit); omitted or 0 = off. */
  rrStartDepth?: number;
  /** `textureLod: 'cone'` only: multiplies the pixel angle the ray cone starts with. Default 1; pbrt-v4's rule is `max(1/8, 1/sqrt(spp))`. */
  textureConeScale?: number;
}

export interface PtMegakernelOptions {
  /** bvh formats of the scene the kernel will run on (`sceneFormatOf(traceBuffers)`). */
  format: SceneFormat;
  /** Surface layout of the scene copy (`copy.layout`). */
  layout: SurfaceLayout;
  integrator?: PtIntegratorOptions;
  /**
   * Mode preset (P4.1, `modes.ts`): `'reference'` (today's defaults), `'preview'`, `'realtime'`. Fills the integrator settings the caller left undefined (explicit options win),
   * merges the D24 layering into layered materials, and `setLimits` starts from the mode's depth. `scenePacking` is NOT changed: pass `ptModePreset(mode).scenePacking` to both
   * `uploadSceneCopy` and this option if you want the mode's binding.
   */
  mode?: PtMode;
  /**
   * `integrator.sampler: 'bluenoise'` only (and the realtime mode, which uses the sampler when this is given): the blue-noise texture, a vec2 one (`gf-vec2-64` / `gf-vec2-128`,
   * `fetchBlueNoise` / `decodeBlueNoisePng`; D16, our own textures), power-of-two sides. Uploaded as an `rg32float` texture at group 2 binding `PT_BLUENOISE_BINDING`.
   */
  blueNoise?: BlueNoiseTexture;
  /** Texture arrays of the scene (`copy.textureLayout`, P1.4c): compiles the texture lookups and binds them next to the scene copy. Omit for an untextured scene. */
  textures?: TextureLayout | null;
  /**
   * How the scene copy is bound (P3.11): `split` (two storage buffers, the default `DEFAULT_SCENE_PACKING`) or `packed` (one: frees a binding in the kernel).
   * Upload the scene copy with the same `uploadSceneCopy(..., { packing })`; `setScene` throws on a mismatch.
   */
  scenePacking?: SceneCopyPacking;
  /**
   * Drop the indentation of every line and the spaces around `{ } ( ) , ;` (every line break stays, so compiler line numbers
   * are kept). Default true: it is 17% of the text (63.3 KB -> 52.7 KB). Comments must already be gone (they are).
   */
  minify?: boolean;
  /**
   * Drop top-level declarations `ptMain` and the bindings cannot reach (`rtPruneWgsl`: bvh closest-point queries, unused light and
   * cone helpers, ...). Default true; the text shrinks by about 4.7 KB and every result is bit-identical. Dropped lines shift
   * compiler line numbers after them; `ptMegakernelBuild(...).map` translates them back (so does the error `create` throws).
   */
  prune?: boolean;
  /**
   * Rename top-level functions, structs and constants to 1-2 letter names (`rtShortenWgsl`; entry point, bindings and overrides
   * keep theirs). Default true; saves about 8 KB, results bit-identical. Compiler messages then show short names;
   * the source map (`ptMegakernelBuild(...).map`, applied by `create` on a compile error) gives the original names back.
   */
  shortenNames?: boolean;
  /** Also pass a Source Map v3 as `GPUShaderModuleDescriptor.sourceMap` (browsers may ignore it). Default false. */
  sourceMap?: boolean;
  label?: string;
}

export interface PtImage {
  width: number;
  height: number;
  /** RGBA float32, row 0 = top, radiance averaged over `samples`. alpha = 1. */
  data: Float32Array;
  samples: number;
}

/** Indentation and the spaces around `{ } ( ) , ;` removed; line breaks kept. Only safe on comment-free source. */
export const minifyWgsl = rtMinifyWgsl;

/** The scene-independent WGSL both schedulers append their integrator to: light sampling, Owen-Sobol, BSDF, layered / film / fuzz lobes, material adapter, sampler (in this order). */
export function ptShadingModulesWgsl(options: Pick<PtMegakernelOptions, 'layout' | 'integrator' | 'textures'>): { integratorOptions: PtIntegratorOptions; parts: string[] } {
  const integratorOptions: PtIntegratorOptions = { ...options.integrator, ...(options.textures ? { textures: true } : {}) };
  const integrator = resolveIntegratorOptions(integratorOptions);
  const mats = integrator.materials;
  if (integrator.sampler === 'bluenoise' && integrator.primary === 'vbuffer') throw new Error("ptMegakernelBuild: sampler 'bluenoise' with primary 'vbuffer' is allowed (the jitter pair is not consumed) only through the engine's own jitter; remove one");
  if (integrator.primary === 'vbuffer' && !options.layout.positions) throw new Error("ptMegakernelBuild: integrator.primary 'vbuffer' needs a surface layout with positions (the hit position comes from the barycentrics)");
  if (mats?.features.aniso && !options.layout.tangents) throw new Error('ptMegakernelBuild: anisotropic materials (features.aniso) need a surface layout with tangents');
  const bsdfOptions = { ...(options.integrator ?? {}), ...(mats?.features.thin ? { thin: true } : {}), ...(mats?.features.aniso ? { aniso: true } : {}), ...(mats?.features.thinFilm ? { thinFilm: true } : {}) };
  const layered = mats?.features.layered ? [stripWgslComments(ptLayeredWgsl({ ...mats.layered, vndf: bsdfOptions.vndf, transport: bsdfOptions.transport, ...(mats.features.coatedMetal ? { conductorBase: true } : {}), ...(mats.features.thinFilm ? { thinFilm: true } : {}) })), ptLayeredDispatchWgsl(!!mats.features.coatedMetal)] : [];
  const film = mats?.features.thinFilm ? [stripWgslComments(ptThinFilmWgsl())] : [];
  const fuzz = mats?.features.fuzz ? [stripWgslComments(ptFuzzWgsl({ layered: !!mats.features.layered }))] : [];
  const parts = [rtLightSampleWgsl({ envMap: integrator.envMap, envImportance: integrator.envSampling !== 'uniform' }), rtOwenSobolWgsl(), stripWgslComments(ptBsdfWgsl(bsdfOptions)), ...layered, ptBsdfAdapterWgsl(mats ?? undefined), ...(integrator.lod !== 'off' && mats ? [ptBsdfReducedWgsl(mats)] : []), ...film, ...fuzz, ptSamplerWgsl(integrator.sampler, integrator.blueNoise)];
  return { integratorOptions, parts };
}

/**
 * R1: the integrator options of the wavefront's secondary shade kernel (`materialLod: 'secondary'`): every vertex it shades is reduced, so the layered /
 * fuzz / thin film / anisotropy / coated-metal code is not compiled (`ptReducedKernelFeatures`) while the reduced adapter still knows which of those the scene has (`reducedFrom`).
 */
export function ptSecondaryIntegratorOptions(integrator: PtIntegratorOptions = {}): PtIntegratorOptions {
  const mats = integrator.materials;
  return { ...integrator, lodKernel: 'secondary', materials: mats ? { ...mats, features: ptReducedKernelFeatures(mats.features), reducedFrom: mats.features } : mats };
}

/** The raytracer module options both schedulers use for the walk (group 0) and the scene copy (group 1). */
export function ptRaytracerModuleOptions(options: Pick<PtMegakernelOptions, 'format' | 'layout' | 'textures' | 'scenePacking'>): Parameters<typeof rtModuleWgsl>[1] {
  return {
    ...options.format,
    rayMasks: false,
    traceSlot: { group: 0, firstBinding: 0 },
    scene: { slot: { group: 1, firstBinding: 0 }, layout: options.layout, ...(options.textures ? { textures: options.textures } : {}), ...(options.scenePacking ? { packing: options.scenePacking } : {}) },
  };
}

/** The kernel's source and the map that reads its compiler messages (`rtMapCompilationInfo`). `create` compiles `code`. */
export function ptMegakernelBuild(caps: Caps, options: Pick<PtMegakernelOptions, 'format' | 'layout' | 'integrator' | 'textures' | 'scenePacking' | 'minify' | 'prune' | 'shortenNames'>): RtWgslBuild {
  const { integratorOptions, parts } = ptShadingModulesWgsl(options);
  const raytracer = rtModuleWgsl(caps, ptRaytracerModuleOptions(options));
  const resolved = resolveIntegratorOptions(integratorOptions);
  // Optional outputs (hooks, counters, AOVs; `outputs-wgsl.ts`) are appended only when asked, so a kernel without them keeps its text.
  const outputs = ptOutputsWgsl(resolved, PT_MEGAKERNEL_COUNTERS_BINDING);
  const ours = [...parts, ...(outputs ? [outputs] : []), ...(resolved.primary === 'vbuffer' ? [ptVbufferWgsl()] : []), ptIntegratorWgsl(integratorOptions)].join('\n');
  const packed = (options.scenePacking ?? DEFAULT_SCENE_PACKING) === 'packed';
  // `raytracer` already went through the caps preprocessor (its `enable` lines are first); ours has no `#if`.
  const { code, map } = rtTransformWgsl(`${raytracer}\n${packed ? rtPackSceneAccess(ours) : ours}`, { prune: options.prune, shorten: options.shortenNames, minify: options.minify, keep: ['ptMain'] });
  return { code, map };
}

/** The kernel's source for these options (what `create` compiles). Exported for size checks and the tests. */
export function ptMegakernelWgsl(caps: Caps, options: Pick<PtMegakernelOptions, 'format' | 'layout' | 'integrator' | 'textures' | 'scenePacking' | 'minify' | 'prune' | 'shortenNames'>): string {
  return ptMegakernelBuild(caps, options).code;
}

/** The blue-noise texture must be a vec2 one with power-of-two sides (the lookup wraps with a mask). */
export function validateBlueNoise(tex: BlueNoiseTexture | undefined): asserts tex is BlueNoiseTexture {
  if (!tex) throw new Error("PtMegakernel: integrator.sampler 'bluenoise' needs the `blueNoise` option (a vec2 texture such as gf-vec2-64: fetchBlueNoise / decodeBlueNoisePng)");
  const { width, height, channels, depth } = tex.meta;
  if (channels !== 2 || depth !== 1) throw new Error(`PtMegakernel: blue noise '${tex.meta.name}' must be a vec2 texture of one slice (channels ${channels}, depth ${depth})`);
  if ((width & (width - 1)) !== 0 || (height & (height - 1)) !== 0) throw new Error(`PtMegakernel: blue noise '${tex.meta.name}' must have power-of-two sides, got ${width} x ${height}`);
}

/** `rg32float` texture of the blue-noise data (textureLoad, `unfilterable-float`). */
export function createBlueNoiseTexture(device: GPUDevice, tex: BlueNoiseTexture, label: string): GPUTexture {
  const { width, height } = tex.meta;
  const texture = device.createTexture({ label: `${label} blue noise`, size: [width, height], format: 'rg32float', usage: 0x04 | 0x02 /* TEXTURE_BINDING | COPY_DST */ });
  device.queue.writeTexture({ texture }, tex.data as unknown as GPUAllowSharedBufferSource, { bytesPerRow: width * 8, rowsPerImage: height }, [width, height]);
  return texture;
}

export class PtMegakernel {
  readonly device: GPUDevice;
  readonly options: PtMegakernelOptions;
  readonly wgslBytes: number;
  private readonly pipeline: GPUComputePipeline;
  private readonly walkLayout: GPUBindGroupLayout;
  private readonly sceneLayout: GPUBindGroupLayout;
  private readonly ioLayout: GPUBindGroupLayout;
  private readonly paramsBuffer: GPUBuffer;
  /** The fuzz LTC table (`features.fuzz` only, group 2 binding 2). */
  private readonly fuzzTable: GPUBuffer | null;
  /** P2.32b: the `albedoCorrection: 'texture'` table (group 2 binding `PT_LAYER_CORR_BINDING`), only for such a kernel. */
  private corrTexture: GPUTexture | null = null;
  private readonly hookLayout: GPUBindGroupLayout | null;
  private hookGroup: GPUBindGroup | null = null;
  private vbufferView: GPUTextureView | null = null;
  /** P4.1: the blue-noise texture (`sampler: 'bluenoise'`), the previous-camera ring and the per-frame planes (`integrator.frame`). */
  private blueTexture: GPUTexture | null = null;
  private frameParamsBuffer: GPUBuffer | null = null;
  private frameBuffer: GPUBuffer | null = null;
  private prevCamera: PtLookAtCamera | null = null;
  private jitter: [number, number] = [0, 0];
  /** ReSTIR DI passes (`integrator.restir`, P4): run before every sample dispatch. */
  restir: PtRestirPasses | null = null;
  private counterBuffer: GPUBuffer | null = null;
  private readonly slotBytes: number;
  private slot = 0;
  private walkGroup: GPUBindGroup | null = null;
  private sceneGroup: GPUBindGroup | null = null;
  private ioGroup: GPUBindGroup | null = null;
  private accum: GPUBuffer | null = null;
  private width = 0;
  private height = 0;
  private camera: PtLookAtCamera | null = null;
  private limits: PtLimits = { maxDepth: 5 };
  private seed = 0;
  /** Sample passes encoded since the last `reset`. */
  samples = 0;

  private constructor(device: GPUDevice, options: PtMegakernelOptions, pipeline: GPUComputePipeline, layouts: GPUBindGroupLayout[], wgslBytes: number, fuzzTable: GPUBuffer | null) {
    this.fuzzTable = fuzzTable;
    this.hookLayout = layouts[3] ?? null;
    this.device = device;
    this.options = options;
    if (options.mode) {
      const lim = ptModePreset(options.mode).limits;
      if (lim) this.limits = { maxDepth: lim.maxDepth, ...(lim.rrStartDepth ? { rrStartDepth: lim.rrStartDepth } : {}), ...(lim.textureConeScale !== undefined ? { textureConeScale: lim.textureConeScale } : {}) };
    }
    this.pipeline = pipeline;
    this.walkLayout = layouts[0] as GPUBindGroupLayout;
    this.sceneLayout = layouts[1] as GPUBindGroupLayout;
    this.ioLayout = layouts[2] as GPUBindGroupLayout;
    this.wgslBytes = wgslBytes;
    this.slotBytes = Math.max(PT_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
    this.paramsBuffer = device.createBuffer({ label: 'pt params', size: this.slotBytes * PT_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
    if (options.integrator?.frame) this.frameParamsBuffer = device.createBuffer({ label: 'pt frame params', size: this.slotBytes * PT_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
  }

  static async create(device: GPUDevice, caps: Caps, requested: PtMegakernelOptions): Promise<PtMegakernel> {
    const options = ptApplyMode(requested);
    const label = options.label ?? 'pt megakernel';
    const bn = options.integrator?.sampler === 'bluenoise';
    if (bn) validateBlueNoise(options.blueNoise);
    if (bn && options.integrator?.restir) throw new Error("PtMegakernel: integrator.sampler 'bluenoise' with integrator.restir is not supported yet (the ReSTIR passes compile the sampler text but do not bind the blue-noise texture: TASKS P4.2e)");
    const { code, map } = ptMegakernelBuild(caps, options);
    const corrTexture = !!options.integrator?.materials?.features.layered && ptLayerCorrNeedsTexture(options.integrator.materials.layered);
    const layouts = [
      device.createBindGroupLayout({ label: `${label} walk`, entries: withBindings(rtWalkEntries(options.format)) }),
      device.createBindGroupLayout({ label: `${label} scene`, entries: rtSceneCopyLayoutEntries(0, options.textures, options.scenePacking) }),
      device.createBindGroupLayout({
        label: `${label} io`,
        entries: [
          { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_PARAMS_BYTES } },
          { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
          ...(options.integrator?.materials?.features.fuzz ? [{ binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' as const } }] : []),
          ...(options.integrator?.counters ? [{ binding: PT_MEGAKERNEL_COUNTERS_BINDING, visibility: COMPUTE, buffer: { type: 'storage' as const } }] : []),
          ...(options.integrator?.primary === 'vbuffer' ? [{ binding: PT_VBUFFER_BINDING, visibility: COMPUTE, texture: { sampleType: 'uint' as const, viewDimension: '2d' as const } }] : []),
          ...(corrTexture ? [ptLayerCorrLayoutEntry()] : []),
          ...(options.integrator?.restir ? [{ binding: PT_RESTIR_SHADE_BINDING, visibility: COMPUTE, buffer: { type: 'read-only-storage' as const } }] : []),
          ...(bn ? [{ binding: PT_BLUENOISE_BINDING, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float' as const, viewDimension: '2d' as const } }] : []),
          ...(options.integrator?.frame ? [{ binding: PT_FRAME_PARAMS_BINDING, visibility: COMPUTE, buffer: { type: 'uniform' as const, hasDynamicOffset: true, minBindingSize: PT_FRAME_PARAMS_BYTES } }, { binding: PT_FRAME_BINDING, visibility: COMPUTE, buffer: { type: 'storage' as const } }] : []),
        ],
      }),
      ...(options.integrator?.hooks ? [device.createBindGroupLayout({ label: `${label} hooks`, entries: options.integrator.hooks.bindings ?? [] })] : []),
    ];
    // On a compile error this throws with original line numbers, names and source text (rtCreateComputePipelineAsync).
    const pipeline = await rtCreateComputePipelineAsync(device, { label, layout: device.createPipelineLayout({ bindGroupLayouts: layouts }), code, map, entryPoint: 'ptMain', sourceMap: options.sourceMap });
    let fuzzTable: GPUBuffer | null = null;
    if (options.integrator?.materials?.features.fuzz) {
      const table = ptFuzzTableF32();
      fuzzTable = device.createBuffer({ label: `${label} fuzz ltc`, size: table.byteLength, usage: STORAGE | COPY_DST });
      device.queue.writeBuffer(fuzzTable, 0, table);
    }
    const kernel = new PtMegakernel(device, options, pipeline, layouts, code.length, fuzzTable);
    if (options.integrator?.restir) kernel.restir = await PtRestirPasses.create(device, caps, options, layouts[0] as GPUBindGroupLayout, layouts[1] as GPUBindGroupLayout);
    if (corrTexture) kernel.corrTexture = ptLayerCorrCreateTexture(device, label);
    if (bn) kernel.blueTexture = createBlueNoiseTexture(device, options.blueNoise as BlueNoiseTexture, label);
    return kernel;
  }

  setScene(trace: TraceScene, scene: SceneCopyBuffers): void {
    if ((this.options.scenePacking ?? DEFAULT_SCENE_PACKING) !== scene.packing) throw new Error(`PtMegakernel.setScene: the kernel was built for scene packing '${this.options.scenePacking ?? DEFAULT_SCENE_PACKING}' but the scene copy is '${scene.packing}' (build both with the same packing)`);
    if (!!this.options.textures !== !!scene.textures) throw new Error(`PtMegakernel.setScene: the kernel was built ${this.options.textures ? 'with' : 'without'} textures but the scene copy has ${scene.textures ? 'some' : 'none'}`);
    this.walkGroup = this.device.createBindGroup({ label: 'pt walk', layout: this.walkLayout, entries: rtSceneBindGroupEntries(trace) });
    this.sceneGroup = this.device.createBindGroup({ label: 'pt scene', layout: this.sceneLayout, entries: rtSceneCopyBindGroupEntries(scene) });
  }

  /**
   * The accumulation buffer (`array<vec4f>`: radiance sum, w = samples; AOVs after `width * height` when on), or null before `setFilm`.
   * For a cache that adds its contribution after the pass (track N: `throughput x prediction` per terminated path) and for live display
   * without a readback. Replaced by `setFilm` on a size change.
   */
  get accumulator(): GPUBuffer | null {
    return this.accum;
  }

  /** Layout of the hooks' bind group (`@group(3)`; `options.integrator.hooks.bindings`), or null without hooks. Build your bind group on it, then `setHookBindGroup`. */
  get hookBindGroupLayout(): GPUBindGroupLayout | null {
    return this.hookLayout;
  }

  /** The hooks' bind group (group 3). Not needed for a hook without bindings. */
  setHookBindGroup(group: GPUBindGroup): void {
    if (!this.hookLayout) throw new Error('PtMegakernel.setHookBindGroup: the kernel was built without hooks');
    this.hookGroup = group;
  }

  /** True when the kernel writes debug counters: its timings are not results (the adapter marks such rows invalid). */
  get instrumented(): boolean {
    return !!this.options.integrator?.counters;
  }

  /**
   * `primary: 'vbuffer'`: the V-buffer texture (`rgba32uint`, format in `vbuffer.ts`, `width x height`, row 0 = top) the next `encode` reads. The engine rasterises into it
   * before the encode (same command encoder is fine: passes run in order). Pass the texture or a view of it.
   */
  setVbuffer(texture: GPUTexture | GPUTextureView): void {
    if (this.options.integrator?.primary !== 'vbuffer') throw new Error("PtMegakernel.setVbuffer: the kernel was built without integrator.primary: 'vbuffer'");
    this.vbufferView = 'createView' in texture ? texture.createView() : texture;
    this.ioGroup = null;
  }

  /** `primary: 'vbuffer'`: the sub-pixel offset the rasteriser used for the frame the next `encode` shades, in pixels, x right / y down, 0 = pixel centre, [-0.5, 0.5]. */
  setVbufferJitter(x: number, y: number): void {
    this.jitter = [x, y];
  }

  setFilm(width: number, height: number): void {
    if (width === this.width && height === this.height && this.accum) return;
    this.width = width;
    this.height = height;
    this.accum?.destroy();
    this.counterBuffer?.destroy();
    this.counterBuffer = null;
    this.frameBuffer?.destroy();
    this.frameBuffer = null;
    // The AOVs (when on) live in the tail of the accumulator: width * height radiance sums, then PT_AOV_VEC4 vec4 per pixel.
    const vec4PerPixel = this.options.integrator?.aovs ? 1 + PT_AOV_VEC4 : 1;
    this.accum = this.device.createBuffer({ label: 'pt accum', size: width * height * 16 * vec4PerPixel, usage: STORAGE | COPY_SRC | COPY_DST });
    if (this.options.integrator?.frame) {
      const bytes = width * height * 16 * PT_FRAME_PLANES;
      if (bytes > this.device.limits.maxStorageBufferBindingSize) throw new RangeError(`PtMegakernel.setFilm: the per-frame planes need ${bytes} bytes, past maxStorageBufferBindingSize (${this.device.limits.maxStorageBufferBindingSize}); render a smaller film`);
      this.frameBuffer = this.device.createBuffer({ label: 'pt frame planes', size: bytes, usage: STORAGE | COPY_SRC | COPY_DST });
    }
    if (this.options.integrator?.counters) this.counterBuffer = this.device.createBuffer({ label: 'pt counters', size: width * height * PT_COUNTER_WORDS * 4, usage: STORAGE | COPY_SRC | COPY_DST });
    this.restir?.setFilm(width, height);
    this.ioGroup = null;
    this.samples = 0;
  }

  setCamera(camera: PtLookAtCamera): void {
    this.camera = camera;
  }

  /**
   * `integrator.frame`: the camera of the PREVIOUS frame, which the motion vectors reproject to (static scenes: both cameras describe the same world). Unset or null = the
   * current camera (zero motion). Both cameras are the unjittered ones.
   */
  setPrevCamera(camera: PtLookAtCamera | null): void {
    this.prevCamera = camera;
  }

  /** `integrator.frame`: one call per frame: the old camera becomes the previous one and `camera` the current. Not for the first frame: use `setCamera`. */
  nextFrame(camera: PtLookAtCamera): void {
    this.prevCamera = this.camera;
    this.camera = camera;
  }

  /** The frame index the sampler sees (`sampleIndex`). In `frame` mode it counts frames (it advances with every `encode`); set it to replay a frame. */
  setFrameIndex(index: number): void {
    this.samples = index >>> 0;
  }

  setLimits(limits: PtLimits): void {
    this.limits = limits;
  }

  /** Scrambling seed base (the per-pixel seeds derive from it). Changing it gives an independent realisation. */
  setSeed(seed: number): void {
    this.seed = seed >>> 0;
  }

  /** Zeroes the accumulation buffer and the sample counter. Encodes into `encoder`. */
  reset(encoder: GPUCommandEncoder): void {
    if (this.accum) encoder.clearBuffer(this.accum);
    if (this.counterBuffer) encoder.clearBuffer(this.counterBuffer);
    this.restir?.reset(encoder);
    if (this.frameBuffer) encoder.clearBuffer(this.frameBuffer);
    this.samples = 0;
  }

  private paramsRecord(sampleIndex: number, finalRegion = 0): ArrayBuffer {
    if (!this.camera) throw new Error('PtMegakernel: setCamera() first');
    const basis = ptCameraBasis(this.camera, this.width, this.height);
    const buffer = new ArrayBuffer(PT_PARAMS_BYTES);
    const u = new Uint32Array(buffer);
    const f = new Float32Array(buffer);
    u.set([this.width, this.height, sampleIndex >>> 0, this.seed, this.limits.maxDepth >>> 0, (this.limits.rrStartDepth ?? 0) >>> 0, 0, finalRegion * this.width * this.height]);
    f[6] = this.limits.textureConeScale ?? 1;
    // With a V-buffer the (otherwise unused) w lanes of origin and right carry the frame's sample position inside the pixel (0.5 + offset): `ptPath` reads them in place of pair 0.
    const vb = this.options.integrator?.primary === 'vbuffer';
    f.set([...basis.origin, vb ? 0.5 + this.jitter[0] : 0, ...basis.right, vb ? 0.5 + this.jitter[1] : 0, ...basis.up, 0, ...basis.forward, 0], 8);
    return buffer;
  }

  private frameParamsRecord(): ArrayBuffer {
    const basis = ptCameraBasis(this.prevCamera ?? (this.camera as PtLookAtCamera), this.width, this.height);
    const buffer = new ArrayBuffer(PT_FRAME_PARAMS_BYTES);
    new Float32Array(buffer).set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0]);
    return buffer;
  }

  /** Records `passes` sample passes (one dispatch each) into `encoder`. At most 256 per call; at most `PT_PARAM_SLOTS` outstanding. */
  encode(encoder: GPUCommandEncoder, passes: number, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.walkGroup || !this.sceneGroup) throw new Error('PtMegakernel: setScene() first');
    if (!this.accum) throw new Error('PtMegakernel: setFilm() first');
    if (passes > 256) throw new RangeError('PtMegakernel.encode: at most 256 passes per call');
    const vbuffer = this.options.integrator?.primary === 'vbuffer';
    if (vbuffer && passes !== 1) throw new RangeError("PtMegakernel.encode: with primary 'vbuffer' one encode is one frame (one V-buffer, one jitter): passes must be 1");
    if (vbuffer && !this.vbufferView) throw new Error('PtMegakernel: setVbuffer() first');
    const frame = !!this.options.integrator?.frame;
    if (frame && passes !== 1) throw new RangeError('PtMegakernel.encode: with integrator.frame one encode is one frame (one sample, one previous camera): passes must be 1');
    this.ioGroup ??= this.device.createBindGroup({
      label: 'pt io',
      layout: this.ioLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer, offset: 0, size: PT_PARAMS_BYTES } },
        { binding: 1, resource: { buffer: this.accum } },
        ...(this.fuzzTable ? [{ binding: 2, resource: { buffer: this.fuzzTable } }] : []),
        ...(this.counterBuffer ? [{ binding: PT_MEGAKERNEL_COUNTERS_BINDING, resource: { buffer: this.counterBuffer } }] : []),
        ...(this.vbufferView ? [{ binding: PT_VBUFFER_BINDING, resource: this.vbufferView }] : []),
        ...(this.corrTexture ? [{ binding: PT_LAYER_CORR_BINDING, resource: this.corrTexture.createView() }] : []),
        ...(this.restir?.reservoirs ? [{ binding: PT_RESTIR_SHADE_BINDING, resource: { buffer: this.restir.reservoirs } }] : []),
        ...(this.blueTexture ? [{ binding: PT_BLUENOISE_BINDING, resource: this.blueTexture.createView() }] : []),
        ...(this.frameBuffer && this.frameParamsBuffer ? [{ binding: PT_FRAME_PARAMS_BINDING, resource: { buffer: this.frameParamsBuffer, offset: 0, size: PT_FRAME_PARAMS_BYTES } }, { binding: PT_FRAME_BINDING, resource: { buffer: this.frameBuffer } }] : []),
      ],
    });
    if (this.hookLayout && !this.hookGroup) {
      if ((this.options.integrator?.hooks?.bindings ?? []).length > 0) throw new Error('PtMegakernel: setHookBindGroup() first (the hooks declare bindings)');
      this.hookGroup = this.device.createBindGroup({ label: 'pt hooks (empty)', layout: this.hookLayout, entries: [] });
    }
    const pass = encoder.beginComputePass({ label: 'pt samples', timestampWrites });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.walkGroup);
    pass.setBindGroup(1, this.sceneGroup);
    if (this.hookGroup) pass.setBindGroup(3, this.hookGroup);
    const gx = Math.ceil(this.width / PT_WORKGROUP[0]);
    const gy = Math.ceil(this.height / PT_WORKGROUP[1]);
    for (let i = 0; i < passes; i++) {
      let finalRegion = 0;
      if (this.restir) {
        if (!this.camera) throw new Error('PtMegakernel: setCamera() first');
        finalRegion = this.restir.encodeFrame(pass, this.walkGroup, this.sceneGroup, this.camera, this.seed, this.samples);
        // The ReSTIR passes switched pipeline and groups: set the shade pass's again.
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.walkGroup);
        pass.setBindGroup(1, this.sceneGroup);
        if (this.hookGroup) pass.setBindGroup(3, this.hookGroup);
      }
      const offset = (this.slot % PT_PARAM_SLOTS) * this.slotBytes;
      this.slot++;
      this.device.queue.writeBuffer(this.paramsBuffer, offset, this.paramsRecord(this.samples, finalRegion));
      if (frame && this.frameParamsBuffer) this.device.queue.writeBuffer(this.frameParamsBuffer, offset, this.frameParamsRecord());
      pass.setBindGroup(2, this.ioGroup, frame ? [offset, offset] : [offset]);
      pass.dispatchWorkgroups(gx, gy);
      this.samples++;
    }
    pass.end();
  }

  /** Copies the accumulator, maps it and averages. The caller keeps this out of timed regions. */
  async readImage(): Promise<PtImage> {
    if (!this.accum) throw new Error('PtMegakernel: nothing rendered');
    const size = this.width * this.height * 16;
    const staging = this.device.createBuffer({ label: 'pt readback', size, usage: MAP_READ | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'pt readback' });
    encoder.copyBufferToBuffer(this.accum, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1 /* GPUMapMode.READ */);
    const sums = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const resolved = resolveIntegratorOptions(this.options.integrator ?? {});
    const n = resolved.accumulate ? Math.max(1, this.samples) : 1;
    const data = new Float32Array(sums.length);
    for (let i = 0; i < sums.length; i += 4) {
      data[i] = (sums[i] as number) / n;
      data[i + 1] = (sums[i + 1] as number) / n;
      data[i + 2] = (sums[i + 2] as number) / n;
      data[i + 3] = 1;
    }
    return { width: this.width, height: this.height, data, samples: this.samples };
  }

  private async readBuffer(buffer: GPUBuffer, offset: number, size: number): Promise<ArrayBuffer> {
    const staging = this.device.createBuffer({ label: 'pt readback', size, usage: MAP_READ | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'pt readback' });
    encoder.copyBufferToBuffer(buffer, offset, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1 /* GPUMapMode.READ */);
    const data = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return data;
  }

  /** The AOVs (`integrator.aovs`), averaged over the samples so far. Outside any timed region. */
  async readAovs(): Promise<PtAovs> {
    if (!this.accum) throw new Error('PtMegakernel: nothing rendered');
    if (!this.options.integrator?.aovs) throw new Error('PtMegakernel.readAovs: the kernel was built without integrator.aovs');
    const sums = new Float32Array(await this.readBuffer(this.accum, 0, this.width * this.height * 16 * (1 + PT_AOV_VEC4)));
    return ptDecodeAovs(sums, this.width, this.height, this.samples);
  }

  /** `integrator.frame`: the last frame's beauty and guide planes, read back (outside any timed region). The GPU buffers themselves: `frameBuffers`. */
  async readFrame(): Promise<PtFrame> {
    if (!this.accum || !this.frameBuffer) throw new Error('PtMegakernel.readFrame: the kernel was built without integrator.frame (or setFilm() was not called)');
    const n = this.width * this.height;
    const radiance = new Float32Array(await this.readBuffer(this.accum, 0, n * 16));
    const planes = new Float32Array(await this.readBuffer(this.frameBuffer, 0, n * 16 * PT_FRAME_PLANES));
    return ptDecodeFrame(radiance, planes, this.width, this.height);
  }

  /**
   * `integrator.frame`: the GPU buffers a temporal filter reads. `accum` is the beauty (vec4f per pixel, the last dispatch's sample, a = 1); `planes` holds `PT_FRAME_PLANES` planes of
   * `width * height` vec4f each, plane-major (plane p starts at byte `p * width * height * 16`; `PtFrame` names them). Valid until the next `encode` writes them.
   */
  get frameBuffers(): { accum: GPUBuffer; planes: GPUBuffer; width: number; height: number } {
    if (!this.accum || !this.frameBuffer) throw new Error('PtMegakernel.frameBuffers: the kernel was built without integrator.frame (or setFilm() was not called)');
    return { accum: this.accum, planes: this.frameBuffer, width: this.width, height: this.height };
  }

  /** The debug counters (`integrator.counters`) summed over the samples so far. Outside any timed region. */
  async readCounters(): Promise<PtCounters> {
    if (!this.counterBuffer) throw new Error('PtMegakernel.readCounters: the kernel was built without integrator.counters');
    const raw = new Uint32Array(await this.readBuffer(this.counterBuffer, 0, this.width * this.height * PT_COUNTER_WORDS * 4));
    return { width: this.width, height: this.height, samples: this.samples, raw, summary: ptSummarizeCounters(raw, this.width, this.height, this.samples) };
  }

  destroy(): void {
    this.counterBuffer?.destroy();
    this.paramsBuffer.destroy();
    this.fuzzTable?.destroy();
    this.corrTexture?.destroy();
    this.restir?.destroy();
    this.blueTexture?.destroy();
    this.frameParamsBuffer?.destroy();
    this.frameBuffer?.destroy();
    this.accum?.destroy();
    this.accum = null;
  }
}
