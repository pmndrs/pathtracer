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
import { loadBlueNoise, rtCreateComputePipelineAsync, rtModuleWgsl, rtPackSceneAccess, DEFAULT_SCENE_PACKING, rtSceneBindGroupEntries, sceneFormatOf, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtTransformWgsl, rtWalkEntries, withBindings } from '@pmndrs/raytracer';
import type { BlueNoiseTexture, Caps, RtWgslBuild, SceneCopyBuffers } from '@pmndrs/raytracer';
import { ptFuzzTableF32 } from '../bsdf/fuzz.ts';
import { PT_LAYER_CORR_BINDING, ptLayerCorrCreateTexture, ptLayerCorrLayoutEntry, ptLayerCorrNeedsTexture } from '../bsdf/layered-correction.ts';
import { ptCameraBasis, type PtLookAtCamera } from '../integrator/camera.ts';
import { PT_BLUENOISE_BINDING } from '../integrator/dimensions.ts';
import { ptApplyMode, ptModePreset } from '../integrator/modes.ts';
import { PT_PARAMS_BYTES, ptMaskShadow, ptShadowNeedsScene, resolveIntegratorOptions, type PtIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { PT_AOV_VEC4, PT_COUNTER_WORDS, PT_FRAME_PARAMS_BINDING, PT_FRAME_PARAMS_BYTES, PT_FRAME_PLANES, ptDecodeAovs, ptDecodeFrame, ptFrameBase, ptFrameTailBytes, ptHookWantsRadiance, ptOutputsWgsl, ptSummarizeCounters, type PtAovs, type PtCounters, type PtFrame } from '../integrator/outputs-wgsl.ts';
import { PtRestirPasses, ptRestirUnsupported } from '../restir/restir.ts';
import { PT_RESTIR_SHADE_BINDING } from '../restir/restir-wgsl.ts';
import { PT_RESTIR_GI_SHADE_BINDING } from '../restir/restir-gi-wgsl.ts';
import { PT_VBUFFER_BINDING, ptVbufferWgsl } from '../integrator/vbuffer.ts';
import { PT_SPEC_BINDING } from '../spectral/spectral-wgsl.ts';
import { defaultRgb2SpecTable } from '../spectral/rgb2spec.ts';
import { PT_VOLUME_BINDING, PT_VOLUME_MAJORANT_BINDING, PT_VOLUME_SAMPLER_BINDING, ptVolumeHasGrid } from '../volume/volume-wgsl.ts';
import { createBlueNoiseTexture, createSpectralTexture, createVolumeSampler, createVolumeTexture, ptNeedsRayMasks, ptRaytracerModuleOptions, ptSecondaryIntegratorOptions, ptShadingModulesWgsl, validateBlueNoise, type PtImage, type PtLimits, type PtMegakernelOptions } from '../integrator/megakernel.ts';
import { PTWF_ARGS_BYTES, PTWF_ARGS_MAIN, PTWF_STRIDE, PTWF_WORKGROUP, PTWF_ARGS_REFILL, PTWF_ARGS_SHADOW, PTWF_HIT_VEC4, PTWF_INIT, PTWF_COUNT_IDLE, PTWF_COUNTERS, PTWF_PARAM_SLOTS, PTWF_RECORDS_PER_ENCODE, PTWF_SHADOW_VEC4, PTWF_START_ITER, ptwfBufferBytes, ptwfRecordsPerSample, ptwfStateBytes } from './layout.ts';
import { PTWF_BINDING, ptwfArgsWgsl, ptwfCameraWgsl, ptwfIntersectWgsl, ptwfPrimaryWgsl, ptwfShadeWgsl, ptwfShadowWgsl, ptwfSortWgsl, ptwfUnsupported, type PtwfConfig } from './wavefront-wgsl.ts';

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
  /** `'ordered'` (P3.10): inline start in the shade kernel, but the restarted slots are appended by the `primary` kernel in slot (pixel) order, so primary rays stay coherent. */
  refill?: boolean | 'inline' | 'batch' | 'ordered';
  refillThreshold?: number;
  layers?: number;
  /**
   * Queue appends of the shade kernel (P3.3, D11 A/B): `'atomic'` (default) one `atomicAdd` per appending thread; `'subgroup'` one `atomicAdd` per subgroup
   * (`subgroupExclusiveAdd` ranks the lanes, so the queue keeps the lane order; needs the `subgroups` feature, throws without it); `'auto'` = subgroup when the device has the feature.
   * The shade body is restructured so the appends sit in uniform control flow (`ptwfShadeEntrySubgroup`). The queue is always compact (only live paths are appended): this is the whole of "compaction".
   */
  appends?: 'atomic' | 'subgroup' | 'auto';
  /**
   * Coherence sorting (P3.4, D11 A/B), each block-wise (a workgroup sorts its 64 queue entries in place, no global payload sort): `'material'` sorts the queue between intersect
   * and shade by miss / material id; `'octant'` sorts the next queue by ray-direction octant before its intersect; `'both'`; `'none'` (default). Costs one extra dispatch per iteration each.
   */
  sort?: 'none' | 'material' | 'octant' | 'both';
}

function scheduling(options: Pick<PtWavefrontOptions, 'refill' | 'layers' | 'refillThreshold' | 'appends' | 'sort'>, caps?: Caps): PtwfConfig {
  const layers = options.layers ?? 1;
  if (!Number.isInteger(layers) || layers < 1 || layers > 64) throw new RangeError(`PtWavefront: layers must be an integer in 1..64, got ${layers}`);
  if (layers > 1 && !options.refill) throw new Error('PtWavefront: layers > 1 needs refill: true (a slot must be able to start the next sample)');
  const threshold = options.refillThreshold ?? 0.25;
  if (!(threshold > 0 && threshold <= 1)) throw new RangeError(`PtWavefront: refillThreshold must be in (0, 1], got ${threshold}`);
  const appends = options.appends ?? 'atomic';
  if (appends !== 'atomic' && appends !== 'subgroup' && appends !== 'auto') throw new RangeError(`PtWavefront: appends must be 'atomic' | 'subgroup' | 'auto', got ${String(appends)}`);
  if (appends === 'subgroup' && caps && !caps.subgroups) throw new Error("PtWavefront: appends 'subgroup' needs the `subgroups` device feature (use 'auto' to fall back to atomic appends)");
  const sort = options.sort ?? 'none';
  if (sort !== 'none' && sort !== 'material' && sort !== 'octant' && sort !== 'both') throw new RangeError(`PtWavefront: sort must be 'none' | 'material' | 'octant' | 'both', got ${String(sort)}`);
  const subgroup = appends === 'subgroup' || (appends === 'auto' && !!caps?.subgroups);
  // optional fields are left out when off, so a default config is the same object shape (and kernel text) as before P3.3 / P3.4 / P3.10
  return { refill: !!options.refill, batch: options.refill === 'batch', threshold, layers, ...(options.refill === 'ordered' ? { ordered: true } : {}), ...(subgroup ? { subgroup: true } : {}), ...(sort === 'material' || sort === 'both' ? { sortHit: true } : {}), ...(sort === 'octant' || sort === 'both' ? { sortRay: true } : {}) };
}

export const PTWF_KERNELS = ['camera', 'intersect', 'shade', 'shadow', 'args'] as const;
/** Kernels that exist only with a setting: `sortHit` / `sortRay` (`sort`, P3.4), `primary` (`refill: 'ordered'`, P3.10). */
export const PTWF_EXTRA_KERNELS = ['sortHit', 'sortRay', 'primary'] as const;
export type PtwfBaseKernel = (typeof PTWF_KERNELS)[number];
export type PtwfExtraKernel = (typeof PTWF_EXTRA_KERNELS)[number];
/** Group 2 bindings each kernel declares (`PTWF_BINDING` numbers; 2 = fuzz table, only with `features.fuzz`). Must match the `bindings([...])` of the WGSL. */
const IO_BINDINGS: Record<PtwfKernel, readonly number[]> = { camera: [0, 1, 3, 6], intersect: [0, 3, 4, 5, 6], shade: [0, 1, 2, 3, 4, 5, 6, 7], shadow: [0, 1, 5, 6, 7], args: [0, 5], sortHit: [0, 4, 5, 6], sortRay: [0, 3, 5, 6], primary: [0, 3, 5, 6] };
const USES_WALK: Record<PtwfKernel, boolean> = { camera: false, intersect: true, shade: false, shadow: true, args: false, sortHit: false, sortRay: false, primary: false };
/** Kernels that bind the scene copy (group 1): shade always; with alpha MASK also intersect (cutout loop) and, for an exact shadow policy, shadow (`ptVisibleMasked`). */
function usesScene(kernel: PtwfKernel, integrator: PtIntegratorOptions | undefined, textures: unknown): boolean {
  if (kernel === 'shade') return true;
  if (kernel === 'sortHit' || kernel === 'sortRay' || kernel === 'primary') return false;
  const o = resolveIntegratorOptions({ ...integrator, ...(textures ? { textures: true } : {}) });
  if (o.primary === 'vbuffer' && kernel === 'intersect') return true; // resolves the V-buffer texel through the scene copy's surface table
  return o.alphaMask && (kernel === 'intersect' || (kernel === 'shadow' && ptShadowNeedsScene(o)));
}
export type PtwfKernel = PtwfBaseKernel | PtwfExtraKernel;
/** The five kernels, plus (`integrator.materialLod: 'secondary'`, R1) `shadeSecondary`: the smaller shade kernel for depth >= `materialLodDepth` (the `shade` kernel then only runs depth < N), plus the settings' extra kernels. */
export type PtWavefrontBuild = Record<PtwfBaseKernel, RtWgslBuild> & { shadeSecondary?: RtWgslBuild; sortHit?: RtWgslBuild; sortRay?: RtWgslBuild; primary?: RtWgslBuild };
type KernelBytes = Record<PtwfBaseKernel, number> & Partial<Record<'shadeSecondary' | PtwfExtraKernel, number>>;

/**
 * The shade kernel has no ray traversal, so it drops the walk's group 0 declarations (a declared binding must be in its pipeline layout, and
 * walk + scene + wavefront buffers would pass the 10 storage buffers per stage of Apple GPUs). The surface, material and light code does not read them.
 */
function stripWalkBindings(code: string): string {
  return code.replace(/^@group\(0\) @binding\(\d+\)[^\n]*\n/gm, '');
}

/** The five kernels' source and source maps. Throws for options the wavefront does not support (`ptwfUnsupported`). */
export function ptWavefrontBuild(caps: Caps, options: Pick<PtWavefrontOptions, 'format' | 'layout' | 'integrator' | 'textures' | 'scenePacking' | 'minify' | 'prune' | 'shortenNames' | 'refill' | 'layers' | 'refillThreshold' | 'appends' | 'sort'>): PtWavefrontBuild {
  const wf = scheduling(options, caps);
  const requested = resolveIntegratorOptions({ ...options.integrator, ...(options.textures ? { textures: true } : {}) });
  // P4.1b / P4.2a: frames and ReSTIR run on the PLAIN scheduler only: one sample per pixel, path = pixel = G-buffer texel. A refill slot restarts at its own time (not sample-aligned).
  if ((requested.frame || requested.restir) && wf.refill) throw new Error(`PtWavefront: integrator.${requested.frame ? 'frame' : 'restir'} needs the plain scheduler (refill: false, layers: 1): ${requested.frame ? 'a frame is one sample per pixel' : 'the ReSTIR G-buffer must be aligned with the sample the shade kernel shades'}`);
  if (requested.restir) {
    const why = ptRestirUnsupported({ ...options, integrator: options.integrator });
    if (why) throw new Error(`PtWavefront: ${why}`);
  }
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
  if (lodOn && (lod.volumes || lod.spectral)) throw new Error(`PtWavefront: integrator.materialLod 'secondary' with integrator.${lod.volumes ? 'volumes' : 'spectral'} is not ported (the reduced shade kernel has no medium vertex / wavelength code; V270)`);
  if (lodOn) options = { ...options, integrator: { ...options.integrator, materialLod: 'full' } };
  const { integratorOptions, parts } = ptShadingModulesWgsl(options);
  const why = ptwfUnsupported(integratorOptions);
  if (why) throw new Error(`PtWavefront: unsupported: ${why}`);
  const t = { prune: options.prune, shorten: options.shortenNames, minify: options.minify };
  const walk = rtModuleWgsl(caps, { ...options.format, rayMasks: ptNeedsRayMasks(options), traceSlot: { group: 0, firstBinding: 0 } });
  const full = rtModuleWgsl(caps, ptRaytracerModuleOptions(options));
  // Optional outputs (hooks, counters, AOVs; `integrator/outputs-wgsl.ts`) join the shade kernel only: it is the one that sees every vertex.
  const outputs = ptOutputsWgsl(resolveIntegratorOptions(integratorOptions), PTWF_BINDING.counters, 'tail');
  const sampler = parts.slice(1, 2).concat(parts.slice(-1)).join('\n'); // Owen-Sobol + the sampler's ptSample2D
  // The appended integrator text is written against the split bindings; `full` (rtModuleWgsl) already has the packed ones.
  const px = (code: string): string => ((options.scenePacking ?? DEFAULT_SCENE_PACKING) === 'packed' ? rtPackSceneAccess(code) : code);
  // R1 x alpha MASK x cone LOD: the cutout test of the intersect kernel uses the widened footprint of depth >= N like the megakernel's (see `PtwfConfig.lodTexDepth`)
  const ro = resolveIntegratorOptions(integratorOptions);
  const intersectWf: PtwfConfig = lodOn && ro.alphaMask && ro.textures && ro.textureLod === 'cone' ? { ...wf, lodTexDepth: lod.lodDepth } : wf;
  let shadeSecondary: RtWgslBuild | undefined;
  if (lodOn) {
    const second = ptShadingModulesWgsl({ ...options, integrator: { ...ptSecondaryIntegratorOptions(options.integrator), materialLod: 'secondary', materialLodDepth: lod.lodDepth } });
    const secondOutputs = ptOutputsWgsl(resolveIntegratorOptions(second.integratorOptions), PTWF_BINDING.counters, 'tail');
    // R1b: with refill a queue mixes depths, so each shade kernel skips the entries of the other class (`lodGate`); the plain scheduler's depth is uniform per dispatch, no gate
    const secondWf = wf.refill ? { ...wf, lodGate: { role: 'second' as const, depth: lod.lodDepth } } : wf;
    shadeSecondary = rtTransformWgsl(px(`${stripWalkBindings(full)}\n${second.parts.join('\n')}\n${secondOutputs ? `${secondOutputs}\n` : ''}${ptwfShadeWgsl(second.integratorOptions, secondWf)}`), { ...t, keep: ['ptwfShade'] });
  }
  return {
    ...(shadeSecondary ? { shadeSecondary } : {}),
    camera: rtTransformWgsl(px(`${sampler}\n${ptwfCameraWgsl(integratorOptions, wf)}`), { ...t, keep: ['ptwfCamera'] }),
    intersect: rtTransformWgsl(px(`${usesScene('intersect', integratorOptions, null) ? full : walk}\n${resolveIntegratorOptions(integratorOptions).primary === 'vbuffer' ? `${ptVbufferWgsl()}\n` : ''}${ptwfIntersectWgsl(integratorOptions, intersectWf)}`), { ...t, keep: ['ptwfIntersect'] }),
    shade: rtTransformWgsl(px(`${stripWalkBindings(full)}\n${parts.join('\n')}\n${outputs ? `${outputs}\n` : ''}${ptwfShadeWgsl(integratorOptions, lodOn && wf.refill ? { ...wf, lodGate: { role: 'first' as const, depth: lod.lodDepth } } : wf)}`), { ...t, keep: ['ptwfShade'] }),
    shadow: rtTransformWgsl(px(`${usesScene('shadow', integratorOptions, null) ? full : walk}\n${ptwfShadowWgsl(integratorOptions, wf)}`), { ...t, keep: ['ptwfShadow'] }),
    args: rtTransformWgsl(ptwfArgsWgsl(wf), { ...t, keep: ['ptwfArgsMain'] }),
    ...(wf.sortHit ? { sortHit: rtTransformWgsl(ptwfSortWgsl('hit', wf), { ...t, keep: ['ptwfSort'] }) } : {}),
    ...(wf.sortRay ? { sortRay: rtTransformWgsl(ptwfSortWgsl('ray', wf), { ...t, keep: ['ptwfSort'] }) } : {}),
    ...(wf.ordered ? { primary: rtTransformWgsl(ptwfPrimaryWgsl(wf), { ...t, keep: ['ptwfPrimary'] }) } : {}),
  };
}

/** Bytes of each kernel's WGSL. */
export function ptWavefrontKernelBytes(build: PtWavefrontBuild): KernelBytes {
  return { ...(Object.fromEntries(PTWF_KERNELS.map((k) => [k, build[k].code.length])) as Record<PtwfBaseKernel, number>), ...(build.shadeSecondary ? { shadeSecondary: build.shadeSecondary.code.length } : {}), ...Object.fromEntries(PTWF_EXTRA_KERNELS.filter((k) => build[k]).map((k) => [k, (build[k] as RtWgslBuild).code.length])) };
}

interface PtwfLayouts {
  hooks: GPUBindGroupLayout | null;
  /** `ptOnRadiance` only: group 3 of the shadow kernel (the hook's `radianceBindings`). */
  hooksRadiance: GPUBindGroupLayout | null;
  empty: GPUBindGroupLayout;
  walk: GPUBindGroupLayout;
  scene: GPUBindGroupLayout;
  io: Record<PtwfKernel, GPUBindGroupLayout>;
  args: GPUBindGroupLayout;
  /** Which kernels bind the scene copy (`usesScene`). */
  scene_: Record<PtwfKernel, boolean>;
}

/** Group 2 bindings of a kernel for this build: the fuzz table (2) only with fuzz, the debug counters (8) only with counters (shade kernel). */
function ioBindings(kernel: PtwfKernel, fuzz: boolean, counters: boolean, refill: boolean, vbuffer = false, corr = false, extra: PtwfExtraBindings = {}): number[] {
  // refill: the camera kernel (start pass) also appends to the queue, so it binds the counters (5)
  // P4.1b / P4.2a: the blue-noise texture (camera + shade: both draw sampler pairs), the previous camera (shade: motion of the primary hit) and the ReSTIR reservoirs (shade: the first vertex's light)
  return [...(kernel === 'camera' && refill ? [0, 1, 3, 5, 6] : IO_BINDINGS[kernel]), ...(kernel === 'shade' && counters ? [PTWF_BINDING.counters] : []), ...(kernel === 'intersect' && vbuffer ? [PT_VBUFFER_BINDING] : []), ...(kernel === 'shade' && corr ? [PT_LAYER_CORR_BINDING] : []), ...(kernel === 'shade' && extra.restir ? [PT_RESTIR_SHADE_BINDING] : []), ...(kernel === 'shade' && extra.restirGi ? [PT_RESTIR_GI_SHADE_BINDING] : []), ...((kernel === 'camera' || kernel === 'shade') && extra.blueNoise ? [PT_BLUENOISE_BINDING] : []), ...(kernel === 'shade' && extra.frame ? [PT_FRAME_PARAMS_BINDING] : []), ...((kernel === 'intersect' || kernel === 'shadow') && extra.volumeMajorant ? [PT_VOLUME_MAJORANT_BINDING] : []), ...((kernel === 'intersect' || kernel === 'shadow') && extra.volumeGrid ? [PT_VOLUME_BINDING] : []), ...((kernel === 'intersect' || kernel === 'shadow') && extra.volumeHw ? [PT_VOLUME_SAMPLER_BINDING] : []), ...(kernel === 'shade' && extra.spectral ? [PT_SPEC_BINDING] : [])].filter((binding) => binding !== 2 || fuzz);
}

/** Group 2 bindings the real-time options add (P4.1b / P4.2a): the blue-noise texture, the per-frame outputs' previous-camera uniform, the ReSTIR reservoirs. */
interface PtwfExtraBindings {
  blueNoise?: boolean;
  frame?: boolean;
  restir?: boolean;
  /** ReSTIR GI (P5): the GI reservoirs, read by the shade kernel. */
  restirGi?: boolean;
  /** P5 media (V270): the density grid (and the brick majorant grid) the intersect and shadow kernels read; the spectral table the shade kernel reads. */
  volumeGrid?: boolean;
  volumeMajorant?: boolean;
  /** V1.2: `volumes.densityFilter: 'hardware'` (the grid is bound filterable, plus a sampler). */
  volumeHw?: boolean;
  spectral?: boolean;
}

type TraceScene = Parameters<typeof rtSceneBindGroupEntries>[0];

export class PtWavefront {
  readonly device: GPUDevice;
  readonly options: PtWavefrontOptions;
  /** Largest kernel's WGSL bytes (the D26 budget applies per kernel). */
  readonly wgslBytes: number;
  readonly kernelBytes: KernelBytes;
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
  private hookRadianceGroup: GPUBindGroup | null = null;
  private vbufferView: GPUTextureView | null = null;
  private jitter: [number, number] = [0, 0];
  /** P4.1b / P4.2a: which real-time bindings the kernels carry (blue noise, frame planes' previous camera, ReSTIR reservoirs). */
  private extra: PtwfExtraBindings = {};
  private blueTexture: GPUTexture | null = null;
  private blueTextureView: GPUTextureView | null = null;
  private corrTextureView: GPUTextureView | null = null;
  /** P5 (V270): the density grid / brick majorant of grid media (intersect and shadow kernels), the spectral table (shade kernel). */
  private volumeTexture: GPUTexture | null = null;
  private volumeSampler: GPUSampler | null = null;
  private majorantTexture: GPUTexture | null = null;
  private specTexture: GPUTexture | null = null;
  /** Ring of previous-camera records, one slot per params slot (`integrator.frame`, shade kernel binding `PT_FRAME_PARAMS_BINDING`). */
  private frameParamsBuffer: GPUBuffer | null = null;
  private prevCamera: PtLookAtCamera | null = null;
  /** The camera of the last encoded frame (ReSTIR's previous camera when none was set explicitly). `reset` forgets it. */
  private lastEncodedCamera: PtLookAtCamera | null = null;
  /** Region of the final ReSTIR reservoirs the shade kernel reads (`up.w` of the params record), in vec4 units. */
  private finalRegionBase = 0;
  /** ReSTIR DI passes (`integrator.restir`, P4.2a): run before the camera kernel of every sample pass. Same class and settings as the megakernel's. */
  restir: PtRestirPasses | null = null;
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

  private constructor(device: GPUDevice, options: PtWavefrontOptions, pipelines: Record<PtwfKernel, GPUComputePipeline>, layouts: PtwfLayouts, bytes: KernelBytes, fuzzTable: GPUBuffer | null, medium: boolean, wf: PtwfConfig, secondary: GPUComputePipeline | null, lodDepth: number) {
    this.device = device;
    this.pipelineSecondary = secondary;
    this.lodDepth = lodDepth;
    this.wf = wf;
    this.options = options;
    // Same as the megakernel: a mode starts the path limits from its preset (`setLimits` replaces them).
    const lim = options.mode ? ptModePreset(options.mode).limits : null;
    if (lim) this.limits = { maxDepth: lim.maxDepth, ...(lim.rrStartDepth ? { rrStartDepth: lim.rrStartDepth } : {}), ...(lim.textureConeScale !== undefined ? { textureConeScale: lim.textureConeScale } : {}) };
    this.pipelines = pipelines;
    this.layouts = layouts;
    this.kernelBytes = bytes;
    this.wgslBytes = Math.max(...Object.values(bytes));
    this.fuzzTable = fuzzTable;
    this.medium = medium;
    this.slotBytes = Math.max(PT_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
    this.paramsBuffer = device.createBuffer({ label: 'ptwf params', size: this.slotBytes * PTWF_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
    this.argsBuffer = device.createBuffer({ label: 'ptwf args', size: PTWF_ARGS_BYTES, usage: STORAGE | INDIRECT });
    if (options.integrator?.frame) this.frameParamsBuffer = device.createBuffer({ label: 'ptwf frame params', size: this.slotBytes * PTWF_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
  }

  static async create(device: GPUDevice, caps: Caps, requested: PtWavefrontOptions): Promise<PtWavefront> {
    const moded = ptApplyMode(requested, 'wavefront');
    // The `bluenoise` sampler (real-time mode) defaults to the texture shipped in the package.
    const options = moded.integrator?.sampler === 'bluenoise' && !moded.blueNoise ? { ...moded, blueNoise: await loadBlueNoise() } : moded;
    const label = options.label ?? 'pt wavefront';
    const build = ptWavefrontBuild(caps, options);
    const wf = scheduling(options, caps);
    const features = options.integrator?.materials?.features;
    const corrTexture = !!features?.layered && ptLayerCorrNeedsTexture(options.integrator?.materials?.layered);
    const resolved = resolveIntegratorOptions({ ...options.integrator, ...(options.textures ? { textures: true } : {}) });
    const extra: PtwfExtraBindings = { blueNoise: resolved.sampler === 'bluenoise', frame: resolved.frame, restir: resolved.restir, restirGi: resolved.restirGi, ...(ptVolumeHasGrid(resolved.volumes) ? { volumeGrid: true } : {}), ...(ptVolumeHasGrid(resolved.volumes) && resolved.volumes?.densityFilter === 'hardware' ? { volumeHw: true } : {}), ...(ptVolumeHasGrid(resolved.volumes) && resolved.volumes?.majorant === 'brick' ? { volumeMajorant: true } : {}), ...(resolved.spectral ? { spectral: true } : {}) };
    if (extra.volumeGrid && !options.volumeGrid) throw new Error('PtWavefront: integrator.volumes has a grid medium: pass the `volumeGrid` option (resolution + density)');
    if (extra.volumeHw && options.volumeGrid?.format !== 'r16float' && !device.features.has('float32-filterable')) throw new Error("PtWavefront: integrator.volumes.densityFilter 'hardware' with an r32float grid needs the 'float32-filterable' device feature (this device lacks it): use volumeGrid.format 'r16float' or densityFilter 'manual'");
    if (extra.volumeMajorant && !options.volumeGrid?.majorant) throw new Error("PtWavefront: integrator.volumes.majorant 'brick' needs `volumeGrid.majorant` (brickMajorants)");
    const specTable = resolved.spectral ? (options.spectralTable ?? defaultRgb2SpecTable(resolved.spectral.res)) : null;
    if (resolved.spectral && specTable && specTable.res !== resolved.spectral.res) throw new Error(`PtWavefront: integrator.spectral.res is ${resolved.spectral.res} but the spectral table has res ${specTable.res} (pass \`spectral: { res: ${specTable.res} }\`)`);
    if (extra.blueNoise) validateBlueNoise(options.blueNoise);
    if (options.textures?.binding === 'table') throw new Error("PtWavefront: textures with binding 'table' (resource table, P1.4c2) run on the megakernel only for now; build the textures with binding 'arrays'");
    device.pushErrorScope('validation');
    // One layout per kernel with only the buffers it declares: a shared layout would bind 3 (walk) + 2 (scene) + 7 buffers and pass the
    // 10-storage-buffers-per-stage limit of Apple GPUs (Chrome/Metal). A kernel that has no walk (or no scene) gets an empty group there.
    const empty = device.createBindGroupLayout({ label: `${label} empty`, entries: [] });
    const walk = device.createBindGroupLayout({ label: `${label} walk`, entries: withBindings(rtWalkEntries(options.format)) });
    const scene = device.createBindGroupLayout({ label: `${label} scene`, entries: rtSceneCopyLayoutEntries(0, options.textures, options.scenePacking) });
    const ioEntries = (kernel: PtwfKernel): GPUBindGroupLayoutEntry[] =>
      ioBindings(kernel, !!features?.fuzz, !!options.integrator?.counters, wf.refill, options.integrator?.primary === 'vbuffer', corrTexture, extra).map((binding) =>
        binding === 0 ? { binding, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_PARAMS_BYTES } } : binding === PT_FRAME_PARAMS_BINDING ? { binding, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_FRAME_PARAMS_BYTES } } : binding === PT_BLUENOISE_BINDING ? { binding, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } } : binding === PT_LAYER_CORR_BINDING ? ptLayerCorrLayoutEntry() : binding === PT_VBUFFER_BINDING ? { binding, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } } : binding === PT_VOLUME_SAMPLER_BINDING ? { binding, visibility: COMPUTE, sampler: { type: 'filtering' } } : binding === PT_VOLUME_BINDING || binding === PT_VOLUME_MAJORANT_BINDING ? { binding, visibility: COMPUTE, texture: { sampleType: binding === PT_VOLUME_BINDING && extra.volumeHw ? 'float' : 'unfilterable-float', viewDimension: '3d' } } : binding === PT_SPEC_BINDING ? { binding, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } } : binding === PT_RESTIR_SHADE_BINDING || binding === PT_RESTIR_GI_SHADE_BINDING ? { binding, visibility: COMPUTE, buffer: { type: 'read-only-storage' } } : { binding, visibility: COMPUTE, buffer: { type: binding === 2 ? 'read-only-storage' : 'storage' } },
      ) as GPUBindGroupLayoutEntry[];
    const io = Object.fromEntries([...PTWF_KERNELS, ...PTWF_EXTRA_KERNELS].map((k) => [k, device.createBindGroupLayout({ label: `${label} io ${k}`, entries: ioEntries(k) })])) as Record<PtwfKernel, GPUBindGroupLayout>;
    const argsLayout = device.createBindGroupLayout({ label: `${label} args`, entries: [{ binding: 0, visibility: COMPUTE, buffer: { type: 'storage' } }] });
    const hooks = options.integrator?.hooks ? device.createBindGroupLayout({ label: `${label} hooks`, entries: options.integrator.hooks.bindings ?? [] }) : null;
    // `ptOnRadiance`: the shadow kernel reports next-event light after its visibility test, with the bindings the callback uses.
    const radianceHooks = options.integrator?.hooks && ptHookWantsRadiance(options.integrator.hooks)
      ? device.createBindGroupLayout({ label: `${label} hooks radiance`, entries: (options.integrator.hooks.bindings ?? []).filter((e) => options.integrator?.hooks?.radianceBindings?.includes(e.binding) ?? true) })
      : null;
    const scenePerKernel = Object.fromEntries([...PTWF_KERNELS, ...PTWF_EXTRA_KERNELS].map((k) => [k, usesScene(k, options.integrator, options.textures)])) as Record<PtwfKernel, boolean>;
    const layouts = { hooks, hooksRadiance: radianceHooks, empty, walk, scene, io, args: argsLayout, scene_: scenePerKernel };
    const pipelineLayout = (k: PtwfKernel): GPUPipelineLayout =>
      device.createPipelineLayout({ label: `${label} ${k}`, bindGroupLayouts: [USES_WALK[k] ? walk : empty, scenePerKernel[k] ? scene : empty, io[k], ...(k === 'args' ? [argsLayout] : []), ...(k === 'shade' && hooks ? [hooks] : []), ...(k === 'shadow' && radianceHooks ? [radianceHooks] : [])] });
    const pipelineLayouts = Object.fromEntries([...PTWF_KERNELS, ...PTWF_EXTRA_KERNELS].map((k) => [k, pipelineLayout(k)])) as Record<PtwfKernel, GPUPipelineLayout>;
    const layoutError = await device.popErrorScope();
    if (layoutError) throw new Error(`PtWavefront: invalid bind group / pipeline layout: ${layoutError.message}`);
    const entry: Record<PtwfKernel, string> = { camera: 'ptwfCamera', intersect: 'ptwfIntersect', shade: 'ptwfShade', shadow: 'ptwfShadow', args: 'ptwfArgsMain', sortHit: 'ptwfSort', sortRay: 'ptwfSort', primary: 'ptwfPrimary' };
    const built = [...PTWF_KERNELS, ...PTWF_EXTRA_KERNELS].filter((k): k is PtwfKernel => !!build[k]);
    const made = await Promise.all(built.map((k) => rtCreateComputePipelineAsync(device, { label: `${label} ${k}`, layout: pipelineLayouts[k], code: (build[k] as RtWgslBuild).code, map: (build[k] as RtWgslBuild).map, entryPoint: entry[k], sourceMap: options.sourceMap })));
    const pipelines = Object.fromEntries(built.map((k, i) => [k, made[i]])) as Record<PtwfKernel, GPUComputePipeline>;
    const secondary = build.shadeSecondary ? await rtCreateComputePipelineAsync(device, { label: `${label} shade secondary`, layout: pipelineLayouts.shade, code: build.shadeSecondary.code, map: build.shadeSecondary.map, entryPoint: entry.shade, sourceMap: options.sourceMap }) : null;
    let fuzzTable: GPUBuffer | null = null;
    if (features?.fuzz) {
      const table = ptFuzzTableF32();
      fuzzTable = device.createBuffer({ label: `${label} fuzz ltc`, size: table.byteLength, usage: STORAGE | COPY_DST });
      device.queue.writeBuffer(fuzzTable, 0, table);
    }
    const kernel = new PtWavefront(device, options, pipelines, layouts, ptWavefrontKernelBytes(build), fuzzTable, !!features?.medium, wf, secondary, resolveIntegratorOptions(options.integrator ?? {}).lodDepth);
    kernel.extra = extra;
    if (corrTexture) kernel.corrTexture = ptLayerCorrCreateTexture(device, label);
    if (extra.blueNoise) kernel.blueTexture = createBlueNoiseTexture(device, options.blueNoise as BlueNoiseTexture, label);
    if (extra.restir) kernel.restir = await PtRestirPasses.create(device, caps, options, walk, scene);
    if (extra.volumeHw) kernel.volumeSampler = createVolumeSampler(device, label);
    if (extra.volumeGrid) kernel.volumeTexture = createVolumeTexture(device, options.volumeGrid as NonNullable<PtWavefrontOptions['volumeGrid']>, label);
    if (extra.volumeMajorant) {
      const mj = (options.volumeGrid as NonNullable<PtWavefrontOptions['volumeGrid']>).majorant as { resolution: readonly [number, number, number]; values: Float32Array };
      kernel.majorantTexture = createVolumeTexture(device, { resolution: mj.resolution, density: mj.values }, `${label} majorant`);
    }
    if (specTable) kernel.specTexture = createSpectralTexture(device, specTable, label);
    return kernel;
  }

  setScene(trace: TraceScene, scene: SceneCopyBuffers): void {
    // A wide scene read by a binary walk (or the reverse) is garbage, not an error: the node formats must agree (X1.i).
    if (trace.nodeFormat !== sceneFormatOf(this.options.format).nodeFormat) throw new Error(`PtWavefront.setScene: the kernel walks '${sceneFormatOf(this.options.format).nodeFormat}' nodes but the scene is '${trace.nodeFormat}'`);
    if ((this.options.scenePacking ?? DEFAULT_SCENE_PACKING) !== scene.packing) throw new Error(`PtWavefront.setScene: the kernel was built for scene packing '${this.options.scenePacking ?? DEFAULT_SCENE_PACKING}' but the scene copy is '${scene.packing}' (build both with the same packing)`);
    if (!!this.options.textures !== !!scene.textures) throw new Error(`PtWavefront.setScene: the kernel was built ${this.options.textures ? 'with' : 'without'} textures but the scene copy has ${scene.textures ? 'some' : 'none'}`);
    this.walkGroup = this.device.createBindGroup({ label: 'ptwf walk', layout: this.layouts.walk, entries: rtSceneBindGroupEntries(trace) });
    this.sceneGroup = this.device.createBindGroup({ label: 'ptwf scene', layout: this.layouts.scene, entries: rtSceneCopyBindGroupEntries(scene) });
  }

  /** Bytes of path state per path, and of all per-path buffers, for the current film (the spec budget is 96 B of state). */
  memory(): { stateBytesPerPath: number; totalBytes: number } {
    const o = { medium: this.medium, textures: !!this.options.textures, ...(this.options.integrator?.volumes ? { volumes: true } : {}) };
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
    const o = { medium: this.medium, textures: !!this.options.textures, ...(this.options.integrator?.volumes ? { volumes: true } : {}) };
    const bytes = ptwfBufferBytes(paths, o);
    const max = this.device.limits.maxStorageBufferBindingSize;
    if (bytes.state > max || paths * PTWF_HIT_VEC4 * 16 > max) throw new RangeError(`PtWavefront.setFilm: ${width}x${height} needs ${(bytes.state / 2 ** 20).toFixed(0)} MiB of path state, the device binds at most ${(max / 2 ** 20).toFixed(0)} MiB per storage buffer`);
    this.width = width;
    this.height = height;
    this.destroyBuffers();
    const make = (label: string, size: number, usage = STORAGE) => this.device.createBuffer({ label: `ptwf ${label}`, size, usage });
    // `integrator.frame`: the planes live in the accumulator's tail (`ptFrameBase`), so no kernel needs a binding for them; one plane-major region like the megakernel's `planes` buffer
    const frameBytes = this.options.integrator?.frame ? ptFrameTailBytes(paths) : 0;
    if (frameBytes > max) throw new RangeError(`PtWavefront.setFilm: the beauty + per-frame planes need ${frameBytes} bytes, past maxStorageBufferBindingSize (${max}); render a smaller film`);
    this.buffers = {
      accum: make('accum', frameBytes || paths * 16 * (this.options.integrator?.aovs ? 1 + PT_AOV_VEC4 : 1), STORAGE | COPY_SRC | COPY_DST),
      state: make('state', bytes.state, STORAGE | COPY_DST),
      hit: make('hit', bytes.hit),
      count: make('counters', bytes.counters, STORAGE | COPY_SRC | COPY_DST),
      queue: make('queue', bytes.queue),
      // `ptOnRadiance` adds a 4th section: (vertex, flags, sample, pixel) for the shadow kernel's report.
      shadow: make('shadow rays', paths * (this.layouts.hooksRadiance || this.options.integrator?.volumes ? PTWF_SHADOW_VEC4 + 1 : PTWF_SHADOW_VEC4) * 16),
    };
    this.counterBuffer = this.options.integrator?.counters ? make('debug counters', paths * PT_COUNTER_WORDS * 4, STORAGE | COPY_SRC | COPY_DST) : null;
    this.restir?.setFilm(width, height);
    this.ioGroups = null;
    this.samples = 0;
    this.iterBase = 0;
  }

  /** Layout of the hooks' bind group (`@group(3)` of the shade kernel), or null without hooks. */
  get hookBindGroupLayout(): GPUBindGroupLayout | null {
    return this.layouts.hooks;
  }

  /** `ptOnRadiance` only: layout of the shadow kernel's hook group (the hook's `radianceBindings`), or null. */
  get hookRadianceBindGroupLayout(): GPUBindGroupLayout | null {
    return this.layouts.hooksRadiance;
  }

  /** `ptOnRadiance` only: the shadow kernel's hook group (group 3), built on `hookRadianceBindGroupLayout` from the same buffers as `setHookBindGroup`. */
  setHookRadianceBindGroup(group: GPUBindGroup): void {
    if (!this.layouts.hooksRadiance) throw new Error('PtWavefront.setHookRadianceBindGroup: the hooks define no ptOnRadiance');
    this.hookRadianceGroup = group;
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

  /** `integrator.frame` / ReSTIR: the camera of the PREVIOUS frame, which motion vectors and the reservoir reprojection go through (as `PtMegakernel.setPrevCamera`). Unset or null = the current camera. */
  setPrevCamera(camera: PtLookAtCamera | null): void {
    this.prevCamera = camera;
  }

  /** One call per frame: the old camera becomes the previous one and `camera` the current. Not for the first frame: use `setCamera`. */
  nextFrame(camera: PtLookAtCamera): void {
    this.prevCamera = this.camera;
    this.camera = camera;
  }

  /** The frame index the sampler sees (`sampleIndex`). In `frame` mode it counts frames (it advances with every `encode`); set it to replay a frame. */
  setFrameIndex(index: number): void {
    this.samples = index >>> 0;
  }

  /** The accumulation buffer (`array<vec4f>`: radiance sums, w = samples; with `integrator.frame` the planes follow at `planesOffset`), or null before `setFilm`. */
  get accumulator(): GPUBuffer | null {
    return this.buffers?.accum ?? null;
  }

  /**
   * `integrator.frame`: the GPU buffers a temporal filter reads, as `PtMegakernel.frameBuffers` but with ONE buffer: `accum` is the beauty (vec4f per pixel, a = 1) and `planes` is the same buffer, whose
   * `PT_FRAME_PLANES` planes of `width * height` vec4f start at byte `planesOffset` (256-byte aligned, so it is a valid storage binding offset), plane-major like the megakernel's.
   */
  get frameBuffers(): { accum: GPUBuffer; planes: GPUBuffer; planesOffset: number; width: number; height: number } {
    if (!this.buffers || !this.options.integrator?.frame) throw new Error('PtWavefront.frameBuffers: the kernels were built without integrator.frame (or setFilm() was not called)');
    return { accum: this.buffers.accum, planes: this.buffers.accum, planesOffset: ptFrameBase(this.width * this.height) * 16, width: this.width, height: this.height };
  }

  /** `integrator.frame`: the last frame's beauty and guide planes, read back (outside any timed region). Same layout as `PtMegakernel.readFrame`. */
  async readFrame(): Promise<PtFrame> {
    if (!this.buffers || !this.options.integrator?.frame) throw new Error('PtWavefront.readFrame: the kernels were built without integrator.frame (or setFilm() was not called)');
    const n = this.width * this.height;
    const all = new Float32Array(await this.readBuffer(this.buffers.accum, ptFrameTailBytes(n)));
    const base = ptFrameBase(n) * 4;
    return ptDecodeFrame(all.slice(0, n * 4), all.slice(base, base + PT_FRAME_PLANES * n * 4), this.width, this.height);
  }

  private frameParamsRecord(): ArrayBuffer {
    const basis = ptCameraBasis(this.prevCamera ?? (this.camera as PtLookAtCamera), this.width, this.height);
    const buffer = new ArrayBuffer(PT_FRAME_PARAMS_BYTES);
    new Float32Array(buffer).set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0]);
    return buffer;
  }

  private blueView(): GPUTextureView | null {
    if (!this.blueTexture) return null;
    return (this.blueTextureView ??= this.blueTexture.createView());
  }

  private corrView(): GPUTextureView | null {
    if (!this.corrTexture) return null;
    return (this.corrTextureView ??= this.corrTexture.createView());
  }

  setLimits(limits: PtLimits): void {
    this.limits = limits;
  }

  setSeed(seed: number): void {
    this.seed = seed >>> 0;
  }

  reset(encoder: GPUCommandEncoder): void {
    this.restir?.reset(encoder);
    this.lastEncodedCamera = null;
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
    return Math.max(1, Math.floor(PTWF_PARAM_SLOTS / ptwfRecordsPerSample(this.iterDepth())));
  }

  /** Depth iterations a sample can need: `maxDepth`, plus `maxVolumeDepth` with `volumes.separateDepth` (a medium scatter is an iteration that does not count toward `maxDepth`). */
  private iterDepth(): number {
    return this.options.integrator?.volumes?.separateDepth ? this.limits.maxDepth + Math.min(this.limits.maxVolumeDepth ?? this.limits.maxDepth, 16383) : this.limits.maxDepth;
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
    if (this.options.integrator?.volumes?.separateDepth) u[23] = Math.min(this.limits.maxVolumeDepth ?? this.limits.maxDepth, 16383) >>> 0; // forward.w (V1.3)
    if (this.finalRegionBase !== 0) u[19] = this.finalRegionBase; // up.w: the ReSTIR final-reservoir region (P4.2a), a u32 the shade kernel bitcasts
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
    if (this.options.integrator?.frame && passes !== 1) throw new RangeError('PtWavefront.encode: with integrator.frame one encode is one frame (one sample, one previous camera): passes must be 1');
    if (passes > this.maxPassesPerEncode()) throw new RangeError(`PtWavefront.encode: at most ${this.maxPassesPerEncode()} passes per call at maxDepth ${this.iterDepth()}`);
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
    if (this.layouts.hooksRadiance && !this.hookRadianceGroup) {
      if (this.layouts.hooksRadiance && (this.options.integrator?.hooks?.bindings ?? []).some((e) => this.options.integrator?.hooks?.radianceBindings?.includes(e.binding) ?? true)) throw new Error('PtWavefront: setHookRadianceBindGroup() first (ptOnRadiance uses bindings)');
      this.hookRadianceGroup = this.device.createBindGroup({ label: 'ptwf hooks radiance (empty)', layout: this.layouts.hooksRadiance, entries: [] });
    }
    const hookRadianceGroup = this.hookRadianceGroup;
    const maxDepth = this.iterDepth();
    const pass = encoder.beginComputePass({ label: 'ptwf samples', timestampWrites });
    const gx = Math.ceil(this.width / 8);
    const gy = Math.ceil(this.height / 8);
    let offset = 0;
    const frame = !!this.options.integrator?.frame;
    const record = (sampleIndex: number, iter: number): void => {
      offset = (this.slot % PTWF_PARAM_SLOTS) * this.slotBytes;
      this.slot++;
      this.device.queue.writeBuffer(this.paramsBuffer, offset, this.paramsRecord(sampleIndex, iter));
      // the previous camera is read by the depth-0 shade only (the motion of the primary hit): one record per frame, at the slot of iteration 0
      if (frame && iter === 0 && this.frameParamsBuffer) this.device.queue.writeBuffer(this.frameParamsBuffer, offset, this.frameParamsRecord());
    };
    // Every pipeline has its own layout, so a pipeline switch invalidates the groups: set them all again.
    const use = (kernel: PtwfKernel, secondary = false): void => {
      pass.setPipeline(secondary && this.pipelineSecondary ? this.pipelineSecondary : this.pipelines[kernel]);
      pass.setBindGroup(0, USES_WALK[kernel] ? walkGroup : emptyGroup);
      pass.setBindGroup(1, this.layouts.scene_[kernel] ? sceneGroup : emptyGroup);
      pass.setBindGroup(2, ioGroups[kernel], kernel === 'shade' && frame ? [offset, offset] : [offset]);
      if (kernel === 'args') pass.setBindGroup(3, argsGroup);
      else if (kernel === 'shade' && hookGroup) pass.setBindGroup(3, hookGroup);
      else if (kernel === 'shadow' && hookRadianceGroup) pass.setBindGroup(3, hookRadianceGroup);
    };
    for (let i = 0; i < passes; i++) {
      if (this.restir) {
        if (!this.camera) throw new Error('PtWavefront: setCamera() first');
        // P4.2a: the four ReSTIR passes run before the sample (same passes module as the megakernel's, same previous-camera rule); the camera kernel's jitter pair and the G-buffer's are the same sampler draw
        this.restir.setInputs({ blueNoise: this.blueView(), vbuffer: this.vbufferView, layerCorrection: this.corrView() });
        this.finalRegionBase = this.restir.encodeFrame(pass, walkGroup, sceneGroup, { camera: this.camera, prev: this.prevCamera ?? this.lastEncodedCamera, seed: this.seed, frame: this.samples, coneScale: this.limits.textureConeScale ?? 1, jitter: this.jitter }) * this.width * this.height;
      }
      record(this.samples, PTWF_INIT);
      use('camera');
      pass.dispatchWorkgroups(gx, gy);
      use('args');
      pass.dispatchWorkgroups(1);
      for (let d = 0; d <= maxDepth; d++) {
        record(this.samples, d);
        use('intersect');
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        if (this.wf.sortHit && d < maxDepth) {
          use('sortHit');
          pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        }
        use('shade', d >= this.lodDepth); // R1: from depth lodDepth on, the reduced-material shade kernel (when built)
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        if (d === maxDepth) break; // depth maxDepth ends every path before NEE and the bounce: nothing left to trace
        use('args');
        pass.dispatchWorkgroups(1);
        if (this.wf.sortRay) {
          use('sortRay'); // the queue shade just filled (the args kernel has written its count into the main args)
          pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        }
        use('shadow');
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_SHADOW);
      }
      this.samples++;
      this.lastEncodedCamera = this.camera;
    }
    pass.end();
  }

  /** Iterations the next `encode(passes)` asks for per pass: the learned figure, else a guess (a short path; `finish` does the rest). */
  private iterationsPerPass(): number {
    return this.itersPerPass > 0 ? this.itersPerPass : Math.min(this.iterDepth() + 1, 16);
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
    if (this.layouts.hooksRadiance && !this.hookRadianceGroup) {
      if (this.layouts.hooksRadiance && (this.options.integrator?.hooks?.bindings ?? []).some((e) => this.options.integrator?.hooks?.radianceBindings?.includes(e.binding) ?? true)) throw new Error('PtWavefront: setHookRadianceBindGroup() first (ptOnRadiance uses bindings)');
      this.hookRadianceGroup = this.device.createBindGroup({ label: 'ptwf hooks radiance (empty)', layout: this.layouts.hooksRadiance, entries: [] });
    }
    const hookRadianceGroup = this.hookRadianceGroup;
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
      else if (kernel === 'shadow' && hookRadianceGroup) pass.setBindGroup(3, hookRadianceGroup);
    };
    const groups = Math.ceil((this.width * this.height * this.wf.layers) / PTWF_WORKGROUP);
    const ordered = { x: Math.min(groups, PTWF_STRIDE), y: Math.ceil(groups / PTWF_STRIDE) };
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
      if (this.wf.sortHit) {
        use('sortHit', offset);
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
      }
      use('shade', offset);
      pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
      if (this.pipelineSecondary) {
        // R1b: a queue mixes depths; each shade kernel skips the other class's entries (`lodGate`)
        pass.setPipeline(this.pipelineSecondary);
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
      }
      if (this.wf.ordered) {
        use('primary', offset); // restarted slots join the next queue in slot order
        pass.dispatchWorkgroups(ordered.x, ordered.y);
      }
      use('args', offset);
      pass.dispatchWorkgroups(1);
      if (this.wf.batch) {
        // batch refill: the camera kernel over the film (zero groups unless the args kernel found the live queue below the threshold); it appends to the NEXT queue, which is what the start record of parity (d + 1) selects
        use('camera', offsets[PTWF_START_ITER + ((this.iterBase + i + 1) & 1)] as number);
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_REFILL);
      }
      if (this.wf.sortRay) {
        use('sortRay', offset);
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
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
      [PT_LAYER_CORR_BINDING]: this.corrTexture ? this.corrView() ?? undefined : undefined,
      [PT_RESTIR_SHADE_BINDING]: this.restir?.reservoirs ? { buffer: this.restir.reservoirs } : undefined,
      [PT_RESTIR_GI_SHADE_BINDING]: this.restir?.giReservoirs ? { buffer: this.restir.giReservoirs } : undefined,
      [PT_BLUENOISE_BINDING]: this.blueView() ?? undefined,
      [PT_FRAME_PARAMS_BINDING]: this.frameParamsBuffer ? { buffer: this.frameParamsBuffer, offset: 0, size: PT_FRAME_PARAMS_BYTES } : undefined,
      [PT_VOLUME_BINDING]: this.volumeTexture ? this.volumeTexture.createView({ dimension: '3d' }) : undefined,
      [PT_VOLUME_SAMPLER_BINDING]: this.volumeSampler ?? undefined,
      [PT_VOLUME_MAJORANT_BINDING]: this.majorantTexture ? this.majorantTexture.createView({ dimension: '3d' }) : undefined,
      [PT_SPEC_BINDING]: this.specTexture ? this.specTexture.createView() : undefined,
    };
    return Object.fromEntries(
      [...PTWF_KERNELS, ...PTWF_EXTRA_KERNELS].map((k) => [k, this.device.createBindGroup({ label: `ptwf io ${k}`, layout: this.layouts.io[k], entries: ioBindings(k, !!this.fuzzTable, !!this.counterBuffer, this.wf.refill, this.options.integrator?.primary === 'vbuffer', !!this.corrTexture, this.extra).filter((binding) => resources[binding]).map((binding) => ({ binding, resource: resources[binding] as GPUBindingResource })) })]),
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
      // a frame (`integrator.frame`, or `accumulate: false`) overwrites the accumulator: the sample is the image
      const n = resolveIntegratorOptions(this.options.integrator ?? {}).accumulate ? Math.max(1, this.samples) : 1;
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
    this.blueTexture?.destroy();
    this.volumeTexture?.destroy();
    this.volumeSampler = null;
    this.majorantTexture?.destroy();
    this.specTexture?.destroy();
    this.frameParamsBuffer?.destroy();
    this.restir?.destroy();
    for (const buffer of this.stage) buffer.destroy();
    this.stage = [];
    this.destroyBuffers();
  }
}
