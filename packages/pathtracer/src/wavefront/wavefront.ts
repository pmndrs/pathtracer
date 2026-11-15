/**
 * Host side of the wavefront path tracer (P3.1), the second scheduler beside `PtMegakernel` (D11: build both, then decide).
 *
 *     const pt = await PtWavefront.create(device, caps, { format, layout: copy.layout, integrator: { sampler: 'owen-sobol' } });
 *     pt.setScene(traceBuffers, sceneCopyBuffers);  pt.setFilm(w, h);  pt.setCamera(cam);  pt.setLimits({ maxDepth: 8, rrStartDepth: 4 });
 *     pt.encode(encoder, 4);                        // 4 sample passes, every kernel of every bounce in one compute pass
 *     const image = await pt.readImage();           // outside any timed region
 *
 * Same surface as the megakernel (create / setScene / setFilm / setCamera / setLimits / setSeed / reset / encode / readImage / destroy).
 * One sample pass = camera + args(INIT) + for each depth `intersect, shade, args, shadow`, with GPU-written indirect dispatch args:
 * no readback and no CPU decision inside the frame. The queue protocol and buffer layout are in `layout.ts`. Slice 1: one sample in
 * flight, one path per pixel, no sorting / compaction (P3.3+); alpha MASK is handled in the intersect and shadow kernels (P3.7, R2 `alphaPolicy`). P3.2 adds the opt-in refill modes and `layers` (see `PtWavefrontOptions`);
 * with `refill`, `encode` does not wait for its paths: `await finish()` before `readImage`.
 *
 * The params uniform is a ring of dynamic-offset slots like the megakernel's, with one record per depth iteration
 * (`PTWF_PARAM_SLOTS`; `maxPassesPerEncode()` says how many sample passes fit in one `encode`).
 */
import { rtCreateComputePipelineAsync, rtModuleWgsl, rtPackSceneAccess, DEFAULT_SCENE_PACKING, rtSceneBindGroupEntries, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtTransformWgsl, rtWalkEntries, withBindings } from '@pmndrs/raytracer';
import type { Caps, RtWgslBuild, SceneCopyBuffers } from '@pmndrs/raytracer';
import { ptFuzzTableF32 } from '../bsdf/fuzz.ts';
import { PT_LAYER_CORR_BINDING, ptLayerCorrCreateTexture, ptLayerCorrLayoutEntry, ptLayerCorrNeedsTexture } from '../bsdf/layered-correction.ts';
import { ptCameraBasis, type PtLookAtCamera } from '../integrator/camera.ts';
import { PT_PARAMS_BYTES, ptMaskShadow, resolveIntegratorOptions, type PtIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { PT_AOV_VEC4, PT_COUNTER_WORDS, ptDecodeAovs, ptOutputsWgsl, ptSummarizeCounters, type PtAovs, type PtCounters } from '../integrator/outputs-wgsl.ts';
import { PT_VBUFFER_BINDING, ptVbufferWgsl } from '../integrator/vbuffer.ts';
import { ptRaytracerModuleOptions, ptSecondaryIntegratorOptions, ptShadingModulesWgsl, type PtImage, type PtLimits, type PtMegakernelOptions } from '../integrator/megakernel.ts';
import { PTWF_ARGS_BYTES, PTWF_ARGS_MAIN, PTWF_ARGS_REFILL, PTWF_ARGS_SHADOW, PTWF_HIT_VEC4, PTWF_INIT, PTWF_COUNT_IDLE, PTWF_COUNTERS, PTWF_PARAM_SLOTS, PTWF_RECORDS_PER_ENCODE, PTWF_SHADOW_VEC4, PTWF_START_ITER, ptwfBufferBytes, ptwfRecordsPerSample, ptwfStateBytes } from './layout.ts';
import { PTWF_BINDING, ptwfArgsWgsl, ptwfCameraWgsl, ptwfIntersectWgsl, ptwfShadeWgsl, ptwfShadowWgsl, ptwfUnsupported, type PtwfConfig } from './wavefront-wgsl.ts';

const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const UNIFORM = 0x40;
const INDIRECT = 0x100;
const MAP_READ = 0x1;
const COMPUTE = 0x4;

/**
 * Options are the megakernel's (`format`, `layout`, `integrator`, `textures`, text transforms), so one adapter setting switches schedulers, plus the
 * scheduling settings of P3.2 (each is an A/B switch, D11):
 *  - `refill` (default false): a slot whose path ends gets the pixel's next sample, so the queues stay full. `true` / `'inline'`: the shade thread starts it at once;
 *    `'batch'`: the slot waits in an idle queue and one dispatch starts all idle slots when the live queue drops below `refillThreshold` (default 0.25, Cycles' rule) of the
 *    slots (primary rays of a batch are coherent again). `encode(n)` then asks for n samples per pixel but does not wait for them: call `finish()` (async, one tiny
 *    readback per chunk, early-out when the queues are empty) before reading the image.
 *  - `layers` (default 1, needs `refill`): path slots per pixel (samples in flight), for films too small to fill the GPU.
 */
export interface PtWavefrontOptions extends PtMegakernelOptions {
  refill?: boolean | 'inline' | 'batch';
  refillThreshold?: number;
  layers?: number;
}

function scheduling(options: Pick<PtWavefrontOptions, 'refill' | 'layers' | 'refillThreshold'>): PtwfConfig {
  const layers = options.layers ?? 1;
  if (!Number.isInteger(layers) || layers < 1 || layers > 64) throw new RangeError(`PtWavefront: layers must be an integer in 1..64, got ${layers}`);
  if (layers > 1 && !options.refill) throw new Error('PtWavefront: layers > 1 needs refill: true (a slot must be able to start the next sample)');
  const threshold = options.refillThreshold ?? 0.25;
  if (!(threshold > 0 && threshold <= 1)) throw new RangeError(`PtWavefront: refillThreshold must be in (0, 1], got ${threshold}`);
  return { refill: !!options.refill, batch: options.refill === 'batch', threshold, layers };
}

export const PTWF_KERNELS = ['camera', 'intersect', 'shade', 'shadow', 'args'] as const;
/** Group 2 bindings each kernel declares (`PTWF_BINDING` numbers; 2 = fuzz table, only with `features.fuzz`). Must match the `bindings([...])` of the WGSL. */
const IO_BINDINGS: Record<(typeof PTWF_KERNELS)[number], readonly number[]> = { camera: [0, 1, 3, 6], intersect: [0, 3, 4, 5, 6], shade: [0, 1, 2, 3, 4, 5, 6, 7], shadow: [0, 1, 5, 6, 7], args: [0, 5] };
const USES_WALK: Record<(typeof PTWF_KERNELS)[number], boolean> = { camera: false, intersect: true, shade: false, shadow: true, args: false };
/** Kernels that bind the scene copy (group 1): shade always; with alpha MASK also intersect (cutout loop) and, for an exact shadow policy, shadow (`ptVisibleMasked`). */
function usesScene(kernel: (typeof PTWF_KERNELS)[number], integrator: PtIntegratorOptions | undefined, textures: unknown): boolean {
  if (kernel === 'shade') return true;
  const o = resolveIntegratorOptions({ ...integrator, ...(textures ? { textures: true } : {}) });
  if (o.primary === 'vbuffer' && kernel === 'intersect') return true; // resolves the V-buffer texel through the scene copy's surface table
  return o.alphaMask && (kernel === 'intersect' || (kernel === 'shadow' && ptMaskShadow(o)));
}
export type PtwfKernel = (typeof PTWF_KERNELS)[number];
/** The five kernels, plus (`integrator.materialLod: 'secondary'`, R1) `shadeSecondary`: the smaller shade kernel for depth >= `materialLodDepth` (the `shade` kernel then only runs depth < N). */
export type PtWavefrontBuild = Record<PtwfKernel, RtWgslBuild> & { shadeSecondary?: RtWgslBuild };

/**
 * The shade kernel has no ray traversal, so it drops the walk's group 0 declarations (a declared binding must be in its pipeline layout, and
 * walk + scene + wavefront buffers would pass the 10 storage buffers per stage of Apple GPUs). The surface, material and light code does not read them.
 */
function stripWalkBindings(code: string): string {
  return code.replace(/^@group\(0\) @binding\(\d+\)[^\n]*\n/gm, '');
}

/** The five kernels' source and source maps. Throws for options the wavefront does not support (`ptwfUnsupported`). */
export function ptWavefrontBuild(caps: Caps, options: Pick<PtWavefrontOptions, 'format' | 'layout' | 'integrator' | 'textures' | 'scenePacking' | 'minify' | 'prune' | 'shortenNames' | 'refill' | 'layers' | 'refillThreshold'>): PtWavefrontBuild {
  const wf = scheduling(options);
  if (resolveIntegratorOptions(options.integrator ?? {}).primary === 'vbuffer') {
    if (wf.refill) throw new Error("PtWavefront: integrator.primary 'vbuffer' does not support refill / layers (a slot starts its next sample at its own time; a V-buffer is one frame)");
    if (!options.layout.positions) throw new Error("PtWavefront: integrator.primary 'vbuffer' needs a surface layout with positions");
  }
  if (wf.layers > 1) {
    const o = resolveIntegratorOptions(options.integrator ?? {});
    if (o.aovs || o.counters || o.hooks) throw new Error('PtWavefront: layers > 1 does not support aovs / counters / hooks yet (they index per-pixel buffers by the path slot)');
  }
  // R1: the shade kernel only ever sees depth < N when a secondary kernel takes over, so it is built as the default ('full') kernel, byte for byte.
  const lod = resolveIntegratorOptions(options.integrator ?? {});
  const lodOn = lod.lod !== 'off';
  if (lodOn && wf.refill) throw new Error("PtWavefront: integrator.materialLod 'secondary' is not supported with refill yet (a refilled slot mixes depths in one queue: P3.x)");
  if (lodOn) options = { ...options, integrator: { ...options.integrator, materialLod: 'full' } };
  const { integratorOptions, parts } = ptShadingModulesWgsl(options);
  const why = ptwfUnsupported(integratorOptions);
  if (why) throw new Error(`PtWavefront: unsupported: ${why}`);
  const t = { prune: options.prune, shorten: options.shortenNames, minify: options.minify };
  const walk = rtModuleWgsl(caps, { ...options.format, rayMasks: false, traceSlot: { group: 0, firstBinding: 0 } });
  const full = rtModuleWgsl(caps, ptRaytracerModuleOptions(options));
  // Optional outputs (hooks, counters, AOVs; `integrator/outputs-wgsl.ts`) join the shade kernel only: it is the one that sees every vertex.
  const outputs = ptOutputsWgsl(resolveIntegratorOptions(integratorOptions), PTWF_BINDING.counters);
  const sampler = parts.slice(1, 2).concat(parts.slice(-1)).join('\n'); // Owen-Sobol + the sampler's ptSample2D
  // The appended integrator text is written against the split bindings; `full` (rtModuleWgsl) already has the packed ones.
  const px = (code: string): string => ((options.scenePacking ?? DEFAULT_SCENE_PACKING) === 'packed' ? rtPackSceneAccess(code) : code);
  let shadeSecondary: RtWgslBuild | undefined;
  if (lodOn) {
    const second = ptShadingModulesWgsl({ ...options, integrator: { ...ptSecondaryIntegratorOptions(options.integrator), materialLod: 'secondary', materialLodDepth: lod.lodDepth } });
    const secondOutputs = ptOutputsWgsl(resolveIntegratorOptions(second.integratorOptions), PTWF_BINDING.counters);
    shadeSecondary = rtTransformWgsl(px(`${stripWalkBindings(full)}\n${second.parts.join('\n')}\n${secondOutputs ? `${secondOutputs}\n` : ''}${ptwfShadeWgsl(second.integratorOptions, wf)}`), { ...t, keep: ['ptwfShade'] });
  }
  return {
    ...(shadeSecondary ? { shadeSecondary } : {}),
    camera: rtTransformWgsl(px(`${sampler}\n${ptwfCameraWgsl(integratorOptions, wf)}`), { ...t, keep: ['ptwfCamera'] }),
    intersect: rtTransformWgsl(px(`${usesScene('intersect', integratorOptions, null) ? full : walk}\n${resolveIntegratorOptions(integratorOptions).primary === 'vbuffer' ? `${ptVbufferWgsl()}\n` : ''}${ptwfIntersectWgsl(integratorOptions, wf)}`), { ...t, keep: ['ptwfIntersect'] }),
    shade: rtTransformWgsl(px(`${stripWalkBindings(full)}\n${parts.join('\n')}\n${outputs ? `${outputs}\n` : ''}${ptwfShadeWgsl(integratorOptions, wf)}`), { ...t, keep: ['ptwfShade'] }),
    shadow: rtTransformWgsl(px(`${usesScene('shadow', integratorOptions, null) ? full : walk}\n${ptwfShadowWgsl(integratorOptions, wf)}`), { ...t, keep: ['ptwfShadow'] }),
    args: rtTransformWgsl(ptwfArgsWgsl(wf), { ...t, keep: ['ptwfArgsMain'] }),
  };
}

/** Bytes of each kernel's WGSL. */
export function ptWavefrontKernelBytes(build: PtWavefrontBuild): Record<PtwfKernel, number> & { shadeSecondary?: number } {
  return { ...(Object.fromEntries(PTWF_KERNELS.map((k) => [k, build[k].code.length])) as Record<PtwfKernel, number>), ...(build.shadeSecondary ? { shadeSecondary: build.shadeSecondary.code.length } : {}) };
}

interface PtwfLayouts {
  hooks: GPUBindGroupLayout | null;
  empty: GPUBindGroupLayout;
  walk: GPUBindGroupLayout;
  scene: GPUBindGroupLayout;
  io: Record<PtwfKernel, GPUBindGroupLayout>;
  args: GPUBindGroupLayout;
  /** Which kernels bind the scene copy (`usesScene`). */
  scene_: Record<PtwfKernel, boolean>;
}

/** Group 2 bindings of a kernel for this build: the fuzz table (2) only with fuzz, the debug counters (8) only with counters (shade kernel). */
function ioBindings(kernel: PtwfKernel, fuzz: boolean, counters: boolean, refill: boolean, vbuffer = false, corr = false): number[] {
  // refill: the camera kernel (start pass) also appends to the queue, so it binds the counters (5)
  return [...(kernel === 'camera' && refill ? [0, 1, 3, 5, 6] : IO_BINDINGS[kernel]), ...(kernel === 'shade' && counters ? [PTWF_BINDING.counters] : []), ...(kernel === 'intersect' && vbuffer ? [PT_VBUFFER_BINDING] : []), ...(kernel === 'shade' && corr ? [PT_LAYER_CORR_BINDING] : [])].filter((binding) => binding !== 2 || fuzz);
}

type TraceScene = Parameters<typeof rtSceneBindGroupEntries>[0];

export class PtWavefront {
  readonly device: GPUDevice;
  readonly options: PtWavefrontOptions;
  /** Largest kernel's WGSL bytes (the D26 budget applies per kernel). */
  readonly wgslBytes: number;
  readonly kernelBytes: Record<PtwfKernel, number> & { shadeSecondary?: number };
  private readonly pipelines: Record<PtwfKernel, GPUComputePipeline>;
  /** R1: the shade kernel of depth >= `lodDepth` (`integrator.materialLod: 'secondary'`), or null. */
  private readonly pipelineSecondary: GPUComputePipeline | null;
  private readonly lodDepth: number;
  private readonly layouts: PtwfLayouts;
  private readonly paramsBuffer: GPUBuffer;
  private readonly argsBuffer: GPUBuffer;
  private readonly fuzzTable: GPUBuffer | null;
  /** P2.32b: the `albedoCorrection: 'texture'` table (shade kernel, group 2 binding `PT_LAYER_CORR_BINDING`). */
  private corrTexture: GPUTexture | null = null;
  private readonly slotBytes: number;
  private readonly medium: boolean;
  private hookGroup: GPUBindGroup | null = null;
  private vbufferView: GPUTextureView | null = null;
  private jitter: [number, number] = [0, 0];
  private counterBuffer: GPUBuffer | null = null;
  private readonly wf: PtwfConfig;
  private slot = 0;
  private walkGroup: GPUBindGroup | null = null;
  private sceneGroup: GPUBindGroup | null = null;
  private ioGroups: Record<PtwfKernel, GPUBindGroup> | null = null;
  private emptyGroup: GPUBindGroup | null = null;
  private argsGroup: GPUBindGroup | null = null;
  private buffers: { accum: GPUBuffer; state: GPUBuffer; hit: GPUBuffer; count: GPUBuffer; queue: GPUBuffer; shadow: GPUBuffer } | null = null;
  private width = 0;
  private height = 0;
  private camera: PtLookAtCamera | null = null;
  private limits: PtLimits = { maxDepth: 5 };
  private seed = 0;
  /** Sample passes encoded since the last `reset` (refill: the target samples per pixel). */
  samples = 0;
  /** Refill: iterations encoded since the last `reset` (the parity of the next one continues the queue ping-pong). */
  private iterBase = 0;
  /** Refill: iterations a pass (one sample per pixel) took in the last drained render; sizes the next `encode` (`finish` learns it). */
  private itersPerPass = 0;
  private stage: GPUBuffer[] = [];
  /** Refill: fewest and most samples any pixel had in the last `readImage` (equal to `samples` for both once `finish` has drained). */
  sampleCount = { min: 0, max: 0 };
  /** Refill: what the last `finish` did (chunks submitted, iterations encoded, wall ms). */
  lastFinish = { chunks: 0, iterations: 0, ms: 0 };

  private constructor(device: GPUDevice, options: PtWavefrontOptions, pipelines: Record<PtwfKernel, GPUComputePipeline>, layouts: PtwfLayouts, bytes: Record<PtwfKernel, number> & { shadeSecondary?: number }, fuzzTable: GPUBuffer | null, medium: boolean, wf: PtwfConfig, secondary: GPUComputePipeline | null, lodDepth: number) {
    this.device = device;
    this.pipelineSecondary = secondary;
    this.lodDepth = lodDepth;
    this.wf = wf;
    this.options = options;
    this.pipelines = pipelines;
    this.layouts = layouts;
    this.kernelBytes = bytes;
    this.wgslBytes = Math.max(...Object.values(bytes));
    this.fuzzTable = fuzzTable;
    this.medium = medium;
    this.slotBytes = Math.max(PT_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
    this.paramsBuffer = device.createBuffer({ label: 'ptwf params', size: this.slotBytes * PTWF_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
    this.argsBuffer = device.createBuffer({ label: 'ptwf args', size: PTWF_ARGS_BYTES, usage: STORAGE | INDIRECT });
  }

  static async create(device: GPUDevice, caps: Caps, options: PtWavefrontOptions): Promise<PtWavefront> {
    const label = options.label ?? 'pt wavefront';
    const build = ptWavefrontBuild(caps, options);
    const wf = scheduling(options);
    const features = options.integrator?.materials?.features;
    const corrTexture = !!features?.layered && ptLayerCorrNeedsTexture(options.integrator?.materials?.layered);
    device.pushErrorScope('validation');
    // One layout per kernel with only the buffers it declares: a shared layout would bind 3 (walk) + 2 (scene) + 7 buffers and pass the
    // 10-storage-buffers-per-stage limit of Apple GPUs (Chrome/Metal). A kernel that has no walk (or no scene) gets an empty group there.
    const empty = device.createBindGroupLayout({ label: `${label} empty`, entries: [] });
    const walk = device.createBindGroupLayout({ label: `${label} walk`, entries: withBindings(rtWalkEntries(options.format)) });
    const scene = device.createBindGroupLayout({ label: `${label} scene`, entries: rtSceneCopyLayoutEntries(0, options.textures, options.scenePacking) });
    const ioEntries = (kernel: PtwfKernel): GPUBindGroupLayoutEntry[] =>
      ioBindings(kernel, !!features?.fuzz, !!options.integrator?.counters, wf.refill, options.integrator?.primary === 'vbuffer', corrTexture).map((binding) =>
        binding === 0 ? { binding, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_PARAMS_BYTES } } : binding === PT_LAYER_CORR_BINDING ? ptLayerCorrLayoutEntry() : binding === PT_VBUFFER_BINDING ? { binding, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } } : { binding, visibility: COMPUTE, buffer: { type: binding === 2 ? 'read-only-storage' : 'storage' } },
      ) as GPUBindGroupLayoutEntry[];
    const io = Object.fromEntries(PTWF_KERNELS.map((k) => [k, device.createBindGroupLayout({ label: `${label} io ${k}`, entries: ioEntries(k) })])) as Record<PtwfKernel, GPUBindGroupLayout>;
    const argsLayout = device.createBindGroupLayout({ label: `${label} args`, entries: [{ binding: 0, visibility: COMPUTE, buffer: { type: 'storage' } }] });
    const hooks = options.integrator?.hooks ? device.createBindGroupLayout({ label: `${label} hooks`, entries: options.integrator.hooks.bindings ?? [] }) : null;
    const scenePerKernel = Object.fromEntries(PTWF_KERNELS.map((k) => [k, usesScene(k, options.integrator, options.textures)])) as Record<PtwfKernel, boolean>;
    const layouts = { hooks, empty, walk, scene, io, args: argsLayout, scene_: scenePerKernel };
    const pipelineLayout = (k: PtwfKernel): GPUPipelineLayout =>
      device.createPipelineLayout({ label: `${label} ${k}`, bindGroupLayouts: [USES_WALK[k] ? walk : empty, scenePerKernel[k] ? scene : empty, io[k], ...(k === 'args' ? [argsLayout] : []), ...(k === 'shade' && hooks ? [hooks] : [])] });
    const pipelineLayouts = Object.fromEntries(PTWF_KERNELS.map((k) => [k, pipelineLayout(k)])) as Record<PtwfKernel, GPUPipelineLayout>;
    const layoutError = await device.popErrorScope();
    if (layoutError) throw new Error(`PtWavefront: invalid bind group / pipeline layout: ${layoutError.message}`);
    const entry: Record<PtwfKernel, string> = { camera: 'ptwfCamera', intersect: 'ptwfIntersect', shade: 'ptwfShade', shadow: 'ptwfShadow', args: 'ptwfArgsMain' };
    const made = await Promise.all(PTWF_KERNELS.map((k) => rtCreateComputePipelineAsync(device, { label: `${label} ${k}`, layout: pipelineLayouts[k], code: build[k].code, map: build[k].map, entryPoint: entry[k], sourceMap: options.sourceMap })));
    const pipelines = Object.fromEntries(PTWF_KERNELS.map((k, i) => [k, made[i]])) as Record<PtwfKernel, GPUComputePipeline>;
    const secondary = build.shadeSecondary ? await rtCreateComputePipelineAsync(device, { label: `${label} shade secondary`, layout: pipelineLayouts.shade, code: build.shadeSecondary.code, map: build.shadeSecondary.map, entryPoint: entry.shade, sourceMap: options.sourceMap }) : null;
    let fuzzTable: GPUBuffer | null = null;
    if (features?.fuzz) {
      const table = ptFuzzTableF32();
      fuzzTable = device.createBuffer({ label: `${label} fuzz ltc`, size: table.byteLength, usage: STORAGE | COPY_DST });
      device.queue.writeBuffer(fuzzTable, 0, table);
    }
    const kernel = new PtWavefront(device, options, pipelines, layouts, ptWavefrontKernelBytes(build), fuzzTable, !!features?.medium, wf, secondary, resolveIntegratorOptions(options.integrator ?? {}).lodDepth);
    if (corrTexture) kernel.corrTexture = ptLayerCorrCreateTexture(device, label);
    return kernel;
  }

  setScene(trace: TraceScene, scene: SceneCopyBuffers): void {
    if ((this.options.scenePacking ?? DEFAULT_SCENE_PACKING) !== scene.packing) throw new Error(`PtWavefront.setScene: the kernel was built for scene packing '${this.options.scenePacking ?? DEFAULT_SCENE_PACKING}' but the scene copy is '${scene.packing}' (build both with the same packing)`);
    if (!!this.options.textures !== !!scene.textures) throw new Error(`PtWavefront.setScene: the kernel was built ${this.options.textures ? 'with' : 'without'} textures but the scene copy has ${scene.textures ? 'some' : 'none'}`);
    this.walkGroup = this.device.createBindGroup({ label: 'ptwf walk', layout: this.layouts.walk, entries: rtSceneBindGroupEntries(trace) });
    this.sceneGroup = this.device.createBindGroup({ label: 'ptwf scene', layout: this.layouts.scene, entries: rtSceneCopyBindGroupEntries(scene) });
  }

  /** Bytes of path state per path, and of all per-path buffers, for the current film (the spec budget is 96 B of state). */
  memory(): { stateBytesPerPath: number; totalBytes: number } {
    const o = { medium: this.medium, textures: !!this.options.textures };
    const paths = this.width * this.height * this.wf.layers;
    return { stateBytesPerPath: ptwfStateBytes(o), totalBytes: ptwfBufferBytes(paths, o).total + paths * 16 };
  }

  /** `primary: 'vbuffer'`: the V-buffer texture (`rgba32uint`, `vbuffer.ts`, `width x height`, row 0 = top) the next `encode` reads. */
  setVbuffer(texture: GPUTexture | GPUTextureView): void {
    if (this.options.integrator?.primary !== 'vbuffer') throw new Error("PtWavefront.setVbuffer: the kernels were built without integrator.primary: 'vbuffer'");
    this.vbufferView = 'createView' in texture ? texture.createView() : texture;
    this.ioGroups = null;
  }

  /** `primary: 'vbuffer'`: the sub-pixel offset the rasteriser used for the frame the next `encode` shades (pixels, x right / y down, 0 = pixel centre, [-0.5, 0.5]). */
  setVbufferJitter(x: number, y: number): void {
    this.jitter = [x, y];
  }

  setFilm(width: number, height: number): void {
    if (width === this.width && height === this.height && this.buffers) return;
    const paths = width * height * this.wf.layers;
    const o = { medium: this.medium, textures: !!this.options.textures };
    const bytes = ptwfBufferBytes(paths, o);
    const max = this.device.limits.maxStorageBufferBindingSize;
    if (bytes.state > max || paths * PTWF_HIT_VEC4 * 16 > max) throw new RangeError(`PtWavefront.setFilm: ${width}x${height} needs ${(bytes.state / 2 ** 20).toFixed(0)} MiB of path state, the device binds at most ${(max / 2 ** 20).toFixed(0)} MiB per storage buffer`);
    this.width = width;
    this.height = height;
    this.destroyBuffers();
    const make = (label: string, size: number, usage = STORAGE) => this.device.createBuffer({ label: `ptwf ${label}`, size, usage });
    this.buffers = {
      accum: make('accum', paths * 16 * (this.options.integrator?.aovs ? 1 + PT_AOV_VEC4 : 1), STORAGE | COPY_SRC | COPY_DST),
      state: make('state', bytes.state, STORAGE | COPY_DST),
      hit: make('hit', bytes.hit),
      count: make('counters', bytes.counters, STORAGE | COPY_SRC | COPY_DST),
      queue: make('queue', bytes.queue),
      shadow: make('shadow rays', paths * PTWF_SHADOW_VEC4 * 16),
    };
    this.counterBuffer = this.options.integrator?.counters ? make('debug counters', paths * PT_COUNTER_WORDS * 4, STORAGE | COPY_SRC | COPY_DST) : null;
    this.ioGroups = null;
    this.samples = 0;
    this.iterBase = 0;
  }

  /** Layout of the hooks' bind group (`@group(3)` of the shade kernel), or null without hooks. */
  get hookBindGroupLayout(): GPUBindGroupLayout | null {
    return this.layouts.hooks;
  }

  /** The hooks' bind group (group 3 of the shade kernel). Not needed for a hook without bindings. */
  setHookBindGroup(group: GPUBindGroup): void {
    if (!this.layouts.hooks) throw new Error('PtWavefront.setHookBindGroup: the kernels were built without hooks');
    this.hookGroup = group;
  }

  /** True when the kernels write debug counters: timings are not results. */
  get instrumented(): boolean {
    return !!this.options.integrator?.counters;
  }

  setCamera(camera: PtLookAtCamera): void {
    this.camera = camera;
  }

  setLimits(limits: PtLimits): void {
    this.limits = limits;
  }

  setSeed(seed: number): void {
    this.seed = seed >>> 0;
  }

  reset(encoder: GPUCommandEncoder): void {
    if (this.counterBuffer) encoder.clearBuffer(this.counterBuffer);
    if (this.buffers) {
      encoder.clearBuffer(this.buffers.accum);
      if (this.wf.refill) {
        // queues, counters and path state persist across encodes: zero counters + state (zero depth word = dormant slot) = a fresh film
        encoder.clearBuffer(this.buffers.count);
        encoder.clearBuffer(this.buffers.state);
      }
    }
    this.samples = 0;
    this.iterBase = 0;
  }

  /** How many sample passes one `encode` call can take with the current `maxDepth` (the params ring must not wrap inside one encoder). */
  maxPassesPerEncode(): number {
    if (this.wf.refill) return Infinity; // only the parity of an iteration is ever read: 4 uniform records per encode, any length
    return Math.max(1, Math.floor(PTWF_PARAM_SLOTS / ptwfRecordsPerSample(this.limits.maxDepth)));
  }

  /** `sampleIndex` is the pass number in the plain scheduler and the TARGET samples per pixel in refill mode (`dims.z`). */
  private paramsRecord(sampleIndex: number, iter: number): ArrayBuffer {
    if (!this.camera) throw new Error('PtWavefront: setCamera() first');
    const basis = ptCameraBasis(this.camera, this.width, this.height);
    const buffer = new ArrayBuffer(PT_PARAMS_BYTES);
    const u = new Uint32Array(buffer);
    const f = new Float32Array(buffer);
    u.set([this.width, this.height, sampleIndex >>> 0, this.seed, this.limits.maxDepth >>> 0, (this.limits.rrStartDepth ?? 0) >>> 0, 0, iter >>> 0]);
    f[6] = this.limits.textureConeScale ?? 1;
    const vb = this.options.integrator?.primary === 'vbuffer';
    f.set([...basis.origin, vb ? 0.5 + this.jitter[0] : 0, ...basis.right, vb ? 0.5 + this.jitter[1] : 0, ...basis.up, 0, ...basis.forward, 0], 8);
    return buffer;
  }

  /** Records `passes` sample passes into one compute pass of `encoder`. At most `maxPassesPerEncode()`. */
  encode(encoder: GPUCommandEncoder, passes: number, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.walkGroup || !this.sceneGroup) throw new Error('PtWavefront: setScene() first');
    const b = this.buffers;
    if (!b) throw new Error('PtWavefront: setFilm() first');
    if (this.wf.refill) {
      this.samples += passes;
      this.encodeIterations(encoder, Math.max(1, Math.round((passes * this.iterationsPerPass()) / this.wf.layers)), true, timestampWrites);
      return;
    }
    if (this.options.integrator?.primary === 'vbuffer') {
      if (passes !== 1) throw new RangeError("PtWavefront.encode: with primary 'vbuffer' one encode is one frame (one V-buffer, one jitter): passes must be 1");
      if (!this.vbufferView) throw new Error('PtWavefront: setVbuffer() first');
    }
    if (passes > this.maxPassesPerEncode()) throw new RangeError(`PtWavefront.encode: at most ${this.maxPassesPerEncode()} passes per call at maxDepth ${this.limits.maxDepth}`);
    this.ioGroups ??= this.makeIoGroups(b);
    this.argsGroup ??= this.device.createBindGroup({ label: 'ptwf args', layout: this.layouts.args, entries: [{ binding: 0, resource: { buffer: this.argsBuffer } }] });
    this.emptyGroup ??= this.device.createBindGroup({ label: 'ptwf empty', layout: this.layouts.empty, entries: [] });
    const walkGroup = this.walkGroup;
    const sceneGroup = this.sceneGroup;
    const ioGroups = this.ioGroups;
    const argsGroup = this.argsGroup;
    const emptyGroup = this.emptyGroup;
    if (this.layouts.hooks && !this.hookGroup) {
      if ((this.options.integrator?.hooks?.bindings ?? []).length > 0) throw new Error('PtWavefront: setHookBindGroup() first (the hooks declare bindings)');
      this.hookGroup = this.device.createBindGroup({ label: 'ptwf hooks (empty)', layout: this.layouts.hooks, entries: [] });
    }
    const hookGroup = this.hookGroup;
    const maxDepth = this.limits.maxDepth;
    const pass = encoder.beginComputePass({ label: 'ptwf samples', timestampWrites });
    const gx = Math.ceil(this.width / 8);
    const gy = Math.ceil(this.height / 8);
    let offset = 0;
    const record = (sampleIndex: number, iter: number): void => {
      offset = (this.slot % PTWF_PARAM_SLOTS) * this.slotBytes;
      this.slot++;
      this.device.queue.writeBuffer(this.paramsBuffer, offset, this.paramsRecord(sampleIndex, iter));
    };
    // Every pipeline has its own layout, so a pipeline switch invalidates the groups: set them all again.
    const use = (kernel: PtwfKernel, secondary = false): void => {
      pass.setPipeline(secondary && this.pipelineSecondary ? this.pipelineSecondary : this.pipelines[kernel]);
      pass.setBindGroup(0, USES_WALK[kernel] ? walkGroup : emptyGroup);
      pass.setBindGroup(1, this.layouts.scene_[kernel] ? sceneGroup : emptyGroup);
      pass.setBindGroup(2, ioGroups[kernel], [offset]);
      if (kernel === 'args') pass.setBindGroup(3, argsGroup);
      else if (kernel === 'shade' && hookGroup) pass.setBindGroup(3, hookGroup);
    };
    for (let i = 0; i < passes; i++) {
      record(this.samples, PTWF_INIT);
      use('camera');
      pass.dispatchWorkgroups(gx, gy);
      use('args');
      pass.dispatchWorkgroups(1);
      for (let d = 0; d <= maxDepth; d++) {
        record(this.samples, d);
        use('intersect');
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        use('shade', d >= this.lodDepth); // R1: from depth lodDepth on, the reduced-material shade kernel (when built)
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        if (d === maxDepth) break; // depth maxDepth ends every path before NEE and the bounce: nothing left to trace
        use('args');
        pass.dispatchWorkgroups(1);
        use('shadow');
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_SHADOW);
      }
      this.samples++;
    }
    pass.end();
  }

  /** Iterations the next `encode(passes)` asks for per pass: the learned figure, else a guess (a short path; `finish` does the rest). */
  private iterationsPerPass(): number {
    return this.itersPerPass > 0 ? this.itersPerPass : Math.min(this.limits.maxDepth + 1, 16);
  }

  /**
   * Refill mode: `iterations` depth iterations (optionally preceded by the start pass that gives dormant slots their next sample) in one compute pass.
   * Four uniform records cover every iteration (parity 0 / 1 and the two start records): the kernels only read the parity of `iter`.
   */
  private encodeIterations(encoder: GPUCommandEncoder, iterations: number, start: boolean, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.walkGroup || !this.sceneGroup) throw new Error('PtWavefront: setScene() first');
    const b = this.buffers;
    if (!b) throw new Error('PtWavefront: setFilm() first');
    this.ioGroups ??= this.makeIoGroups(b);
    this.argsGroup ??= this.device.createBindGroup({ label: 'ptwf args', layout: this.layouts.args, entries: [{ binding: 0, resource: { buffer: this.argsBuffer } }] });
    this.emptyGroup ??= this.device.createBindGroup({ label: 'ptwf empty', layout: this.layouts.empty, entries: [] });
    const walkGroup = this.walkGroup;
    const sceneGroup = this.sceneGroup;
    const ioGroups = this.ioGroups;
    const argsGroup = this.argsGroup;
    const emptyGroup = this.emptyGroup;
    if (this.layouts.hooks && !this.hookGroup) {
      if ((this.options.integrator?.hooks?.bindings ?? []).length > 0) throw new Error('PtWavefront: setHookBindGroup() first (the hooks declare bindings)');
      this.hookGroup = this.device.createBindGroup({ label: 'ptwf hooks (empty)', layout: this.layouts.hooks, entries: [] });
    }
    const hookGroup = this.hookGroup;
    const base = this.slot;
    this.slot += PTWF_RECORDS_PER_ENCODE;
    const offsets: number[] = [];
    for (let i = 0; i < PTWF_RECORDS_PER_ENCODE; i++) {
      const offset = ((base + i) % PTWF_PARAM_SLOTS) * this.slotBytes;
      offsets.push(offset);
      this.device.queue.writeBuffer(this.paramsBuffer, offset, this.paramsRecord(this.samples, i)); // iter = 0, 1 (parity), 2, 3 (start | parity)
    }
    const pass = encoder.beginComputePass({ label: 'ptwf refill', timestampWrites });
    const use = (kernel: PtwfKernel, offset: number): void => {
      pass.setPipeline(this.pipelines[kernel]);
      pass.setBindGroup(0, USES_WALK[kernel] ? walkGroup : emptyGroup);
      pass.setBindGroup(1, this.layouts.scene_[kernel] ? sceneGroup : emptyGroup);
      pass.setBindGroup(2, ioGroups[kernel], [offset]);
      if (kernel === 'args') pass.setBindGroup(3, argsGroup);
      else if (kernel === 'shade' && hookGroup) pass.setBindGroup(3, hookGroup);
    };
    if (start) {
      const startOffset = offsets[PTWF_START_ITER + (this.iterBase & 1)] as number;
      use('camera', startOffset);
      pass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8), this.wf.layers);
      use('args', startOffset);
      pass.dispatchWorkgroups(1);
    }
    for (let i = 0; i < iterations; i++) {
      const offset = offsets[(this.iterBase + i) & 1] as number;
      use('intersect', offset);
      pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
      use('shade', offset);
      pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
      use('args', offset);
      pass.dispatchWorkgroups(1);
      if (this.wf.batch) {
        // batch refill: the camera kernel over the film (zero groups unless the args kernel found the live queue below the threshold); it appends to the NEXT queue, which is what the start record of parity (d + 1) selects
        use('camera', offsets[PTWF_START_ITER + ((this.iterBase + i + 1) & 1)] as number);
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_REFILL);
      }
      use('shadow', offset);
      pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_SHADOW);
    }
    pass.end();
    this.iterBase += iterations;
  }

  /**
   * Refill mode: runs iterations until every path of every `encode`d sample has finished (the next sample of a slot starts inline, so this is
   * the tail only). Chunks of iterations (32, doubling to 256) are submitted back to back; after each one an 8-byte copy of the queue counters is
   * mapped while the NEXT chunk already runs on the GPU (one chunk of lag, so the GPU never waits for the CPU), and the loop stops when the
   * counters of the chunk before say nothing is queued. The over-run is at most two chunks of empty (zero-workgroup) iterations. This is the one
   * readback of the refill scheduler, once per render, never inside the per-frame encode; timed runs include it (it is part of finishing).
   */
  async finish(): Promise<void> {
    if (!this.wf.refill) return;
    const t0 = performance.now();
    const b = this.buffers;
    if (!b) throw new Error('PtWavefront: setFilm() first');
    this.stage = this.stage.length ? this.stage : [0, 1].map((i) => this.device.createBuffer({ label: `ptwf finish stage ${i}`, size: 32, usage: MAP_READ | COPY_DST }));
    const startIter = this.iterBase;
    const read = async (buffer: GPUBuffer): Promise<number> => {
      await buffer.mapAsync(1 /* GPUMapMode.READ */);
      const w = new Uint32Array(buffer.getMappedRange().slice(0));
      buffer.unmap();
      return (w[0] as number) + (w[1] as number) + (w[PTWF_COUNT_IDLE] as number); // both queues + the idle slots (batch refill)
    };
    let chunk = 32;
    let previous: Promise<number> | null = null;
    let beforeLastTwo = this.iterBase;
    let beforeLast = this.iterBase;
    let chunks = 0;
    for (let i = 0; ; i++) {
      const encoder = this.device.createCommandEncoder({ label: 'ptwf finish' });
      this.encodeIterations(encoder, chunk, false);
      encoder.copyBufferToBuffer(b.count, 0, this.stage[i & 1] as GPUBuffer, 0, PTWF_COUNTERS * 4);
      this.device.queue.submit([encoder.finish()]);
      chunks++;
      const mine = read(this.stage[i & 1] as GPUBuffer);
      beforeLastTwo = beforeLast;
      beforeLast = this.iterBase - chunk;
      if (previous && (await previous) === 0) {
        await mine;
        break;
      }
      previous = mine;
      chunk = Math.min(chunk * 2, 256);
    }
    await this.device.queue.onSubmittedWorkDone();
    // everything up to the chunk before the last two was needed (the last two ran dry): the learned cost of a pass
    if (this.samples > 0) this.itersPerPass = Math.max(1, (Math.max(startIter, beforeLastTwo) * this.wf.layers) / this.samples);
    this.lastFinish = { chunks, iterations: this.iterBase - startIter, ms: performance.now() - t0 };
  }

  private makeIoGroups(b: NonNullable<PtWavefront['buffers']>): Record<PtwfKernel, GPUBindGroup> {
    const resources: Record<number, GPUBindingResource | undefined> = {
      0: { buffer: this.paramsBuffer, offset: 0, size: PT_PARAMS_BYTES },
      1: { buffer: b.accum },
      2: this.fuzzTable ? { buffer: this.fuzzTable } : undefined,
      3: { buffer: b.state },
      4: { buffer: b.hit },
      5: { buffer: b.count },
      6: { buffer: b.queue },
      7: { buffer: b.shadow },
      8: this.counterBuffer ? { buffer: this.counterBuffer } : undefined,
      [PT_VBUFFER_BINDING]: this.vbufferView ?? undefined,
      [PT_LAYER_CORR_BINDING]: this.corrTexture ? this.corrTexture.createView() : undefined,
    };
    return Object.fromEntries(
      PTWF_KERNELS.map((k) => [k, this.device.createBindGroup({ label: `ptwf io ${k}`, layout: this.layouts.io[k], entries: ioBindings(k, !!this.fuzzTable, !!this.counterBuffer, this.wf.refill, this.options.integrator?.primary === 'vbuffer', !!this.corrTexture).filter((binding) => resources[binding]).map((binding) => ({ binding, resource: resources[binding] as GPUBindingResource })) })]),
    ) as Record<PtwfKernel, GPUBindGroup>;
  }

  async readImage(): Promise<PtImage> {
    if (!this.buffers) throw new Error('PtWavefront: nothing rendered');
    const size = this.width * this.height * this.wf.layers * 16;
    const staging = this.device.createBuffer({ label: 'ptwf readback', size, usage: MAP_READ | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'ptwf readback' });
    encoder.copyBufferToBuffer(this.buffers.accum, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1 /* GPUMapMode.READ */);
    const sums = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const pixels = this.width * this.height;
    const data = new Float32Array(pixels * 4);
    this.sampleCount = { min: Infinity, max: 0 };
    if (this.wf.refill) {
      // slot = layer * pixels + pixel; w = samples started in the slot. Drained (finish()), the summed w is exactly `samples` for every pixel.
      for (let p = 0; p < pixels; p++) {
        let r = 0, g = 0, bl = 0, w = 0;
        for (let l = 0; l < this.wf.layers; l++) {
          const i = (l * pixels + p) * 4;
          r += sums[i] as number;
          g += sums[i + 1] as number;
          bl += sums[i + 2] as number;
          w += sums[i + 3] as number;
        }
        if (w < this.sampleCount.min) this.sampleCount.min = w;
        if (w > this.sampleCount.max) this.sampleCount.max = w;
        const n = Math.max(1, w);
        data[p * 4] = r / n;
        data[p * 4 + 1] = g / n;
        data[p * 4 + 2] = bl / n;
        data[p * 4 + 3] = 1;
      }
    } else {
      const n = Math.max(1, this.samples);
      for (let i = 0; i < sums.length; i += 4) {
        data[i] = (sums[i] as number) / n;
        data[i + 1] = (sums[i + 1] as number) / n;
        data[i + 2] = (sums[i + 2] as number) / n;
        data[i + 3] = 1;
      }
    }
    return { width: this.width, height: this.height, data, samples: this.samples };
  }

  private async readBuffer(buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const staging = this.device.createBuffer({ label: 'ptwf readback', size, usage: MAP_READ | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'ptwf readback' });
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1 /* GPUMapMode.READ */);
    const data = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return data;
  }

  /** The AOVs (`integrator.aovs`), averaged over the samples so far. Outside any timed region. */
  async readAovs(): Promise<PtAovs> {
    if (!this.buffers) throw new Error('PtWavefront: nothing rendered');
    if (!this.options.integrator?.aovs) throw new Error('PtWavefront.readAovs: the kernels were built without integrator.aovs');
    const sums = new Float32Array(await this.readBuffer(this.buffers.accum, this.width * this.height * 16 * (1 + PT_AOV_VEC4)));
    return ptDecodeAovs(sums, this.width, this.height, this.samples);
  }

  /** The debug counters (`integrator.counters`) summed over the samples so far. Outside any timed region. */
  async readCounters(): Promise<PtCounters> {
    if (!this.counterBuffer) throw new Error('PtWavefront.readCounters: the kernels were built without integrator.counters');
    const raw = new Uint32Array(await this.readBuffer(this.counterBuffer, this.width * this.height * PT_COUNTER_WORDS * 4));
    return { width: this.width, height: this.height, samples: this.samples, raw, summary: ptSummarizeCounters(raw, this.width, this.height, this.samples) };
  }

  private destroyBuffers(): void {
    this.counterBuffer?.destroy();
    this.counterBuffer = null;
    if (!this.buffers) return;
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
    this.buffers = null;
  }

  destroy(): void {
    this.paramsBuffer.destroy();
    this.argsBuffer.destroy();
    this.fuzzTable?.destroy();
    this.corrTexture?.destroy();
    for (const buffer of this.stage) buffer.destroy();
    this.stage = [];
    this.destroyBuffers();
  }
}
