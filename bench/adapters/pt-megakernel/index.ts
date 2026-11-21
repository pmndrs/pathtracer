/**
 * Our reference path tracer (`@pmndrs/pathtracer` megakernel, P2.4) behind `RendererAdapter`, so `matrix.ts --renderers pt-megakernel`
 * and `baseline.ts` treat it like the competitors.
 *
 * Scene: `SceneDesc` -> `sceneDescToCopyInput` (bench/scenes/to-scene-copy.ts) -> bvh `buildTraceScene` + the scene copy
 * (`buildSceneCopy`, barycentric positions). One instance per mesh, all static. `file` geometry is loaded from the shared web
 * export (`fetchSceneGlb`, the same glb the web competitors read; baked world-space triangles, all primitives merged into one
 * mesh). An `hdri` environment is fetched as a PFM from `/bench/assets/<path>` and copied into the scene copy (P1.4d); it is
 * importance-sampled by luminance unless `envSampling` says otherwise (`pt_env=uniform` is the A/B baseline).
 *
 * Timing: wall clock around the dispatches, `onSubmittedWorkDone` at the end of the timed region, readback excluded (snapshot
 * readbacks are subtracted from later times). Best effort: this adapter is for correctness; precision timing is the quiet pass.
 */
import { benchAssets, blueNoiseAssets } from '../../site-paths.ts';
import { buildTraceScene, createMeshBvhBuilder, uploadTraceScene, type NodeFormat, type TraceSceneBuffers } from '@pmndrs/bvh';
import { ALPHA_MODE, fetchBlueNoise, createMaterial, detectCaps, requestDeviceForCaps, sceneFormatOf, uploadSceneCopy, buildSceneCopy, lightSelectionName, DEFAULT_LIGHT_SELECTION, type Caps, type LightSelection, type SceneCopy, type SceneCopyBuffers, type SceneCopyInput } from '@pmndrs/raytracer';
import { sceneDensityFiles, sceneVolumes } from '../../scenes/volumes.ts';
import type { Rgb2SpecTable } from '@pmndrs/pathtracer';
import { brickMajorants, ptRoundGridF16, ptQuantizeGridR8, PtMegakernel, PtWavefront, ptwfAutoPolicy, ptApplyMode, ptMegakernelWgsl, ptModePreset, type PtMode, resolveRestirOptions, PT_PAIR, ptCameraBasis, ptMaterialFeatures, ptPixelSeed, ptSample2D, PT_COUNTER, PT_COUNTER_WORDS, type PtAovs, type PtCounters, type PtImage, type PtIntegratorOptions, type PtLayeredOptions, type PtMaterialOptions, type PtMegakernelOptions, type PtWavefrontOptions } from '@pmndrs/pathtracer';
import type { FeatureSupport, Image, RenderBudget, RenderResult, RendererAdapter, RendererInfo } from '../../src/adapter.ts';
import type { TriangleMesh } from '../../scenes/geometry.ts';
import { resolveOpenPbr } from '../../scenes/openpbr.ts';
import type { MaterialDesc, MeshDesc, SceneDesc } from '../../scenes/schema.ts';
import { sceneDescToCopyInput } from '../../scenes/to-scene-copy.ts';
import { collectMaterialXPrograms } from '../../scenes/materialx.ts';
import { alphaCoverageOf, classifyAlphaCards, INSTANCE_MASK_MIXED, type AlphaClassStats } from '../../scenes/alpha-classify.ts';
import { uvTangents } from '../../scenes/tangents.ts';
import { setImageTextureLoader } from '../../scenes/textures.ts';
import { readPFM } from '../../metrics/pfm.ts';
import { fetchSceneGlb, readGlbMeshes } from '../shared/glb.ts';
import { VbufferRaster } from '../shared/vbuffer-raster.ts';
import type { GpuContext } from '../shared/gpu.ts';

/** Experiment switches (page args `pt_*`); defaults are the reference configuration. */
export interface PtAdapterSettings extends PtIntegratorOptions {
  /** `barycentric` (default): hit positions from the triangle's vertices; `ray`: origin + t * direction. P1.8 A/B. */
  positions?: 'barycentric' | 'ray';
  /** P5 media: shadow-ray transmittance through a homogeneous medium, `analytic` (default) or `ratio` (pbrt's ratio tracking; V241 A/B). */
  volumeShadow?: 'analytic' | 'ratio' | 'residual' | 'telescoping';
  /** P5 grid media: `global` (default) or `brick` majorant (V243 A/B). */
  volumeMajorant?: 'global' | 'brick';
  volumeTrRoulette?: boolean;
  /** V4.2: forced in-scatter NEE in homogeneous media (`integrator.volumes.forcedNee`, megakernel). */
  volumeForcedNee?: boolean;
  /** V4.1: equiangular distance sampling for the forced NEE (`integrator.volumes.equiangular`, implies forcedNee, megakernel). */
  volumeEquiangular?: boolean;
  /** V1.2: `'hardware'` reads grid density with one filtered `textureSampleLevel` (default `'manual'`). */
  volumeDensityFilter?: 'manual' | 'hardware';
  /** V3.2: `'atlas'` stores the grid as a sparse brick atlas + indirection (`volumes.gridLayout`); default dense. */
  volumeGridLayout?: 'dense' | 'atlas';
  /** P5.v15 / P5.v16: `volumes.outputs` (megakernel): uploads `vol_order01`, `vol_orders2plus`, `vol_transmittance`, `vol_stderr`, `vol_first_scatter` next to the image. */
  volumeOutputs?: boolean;
  /** P5.v18: the grid texture format (`r16float` rounds the density to f16 first). */
  volumeGridFormat?: 'r32float' | 'r16float' | 'r8unorm';
  /** maxDepth override (the scene's own otherwise). */
  maxDepth?: number;
  /** V1.3: separate cap for medium scatter events (`integrator.volumes.separateDepth`); surface vertices keep `maxDepth`. Unset = media count toward `maxDepth` as before. */
  maxVolumeDepth?: number;
  /** rrStartDepth override: a number, or 0 for off; the scene's own otherwise. */
  rrStartDepth?: number;
  /** Sun angular radius override in radians (0 = delta light, as pbrt and Mitsuba); the scene's own otherwise. A/B for V35. */
  sunAngularRadius?: number;
  /** Light-selection weights for NEE (P1.4f): `power` (default), `kind`, `{ capped: share }`. Noise only. */
  lightSelection?: LightSelection;
  /** `textureLod: 'cone'` only: scale of the ray cone's starting spread. `'auto'` (default) = pbrt-v4's `max(1/8, 1/sqrt(spp))` of the render's spp; or a number (1 = a full pixel). */
  textureConeScale?: number | 'auto';
  /**
   * OpenPBR layering (P2.21; only scenes with an `openpbr` material read these). `layering` default `'stochastic'`, pbrt's `coateddiffuse` walk,
   * the reference mode (the class B scenes agree with the reference only this way: `albedo` OpenPBR scaling is ~0.2 brighter on a bright
   * base, bsdf.md); `'albedo'` is the cheap default of the library. `albedoBase` (albedo only) `'openpbr'` | `'internal'`; `exitMis`, `layerDepth`
   * (walk length, default 64 as the exporters' `maxdepth`) and `layerThickness` (default 1e-6, the exporters' transparent layer) are the stochastic walk's.
   */
  layering?: 'albedo' | 'stochastic';
  albedoBase?: 'openpbr' | 'internal';
  /** P2.32b (albedo layering, internal base): fitted correction of the base lobe, `'none'` (default) | `'lut'` | `'texture'`. */
  albedoCorrection?: 'none' | 'lut' | 'texture';
  exitMis?: 'pbrt' | 'none';
  layerDepth?: number;
  layerThickness?: number;
  /** Tint thin-walled transmission by `transmission_color` (default true; pbrt and Mitsuba `thindielectric` cannot tint, so the reference differs on tinted thin walls). */
  transmissionTint?: boolean;
  /** Render `openpbr` materials that use lobes we do not implement (fuzz, thin film, dispersion, subsurface, anisotropy) with those lobes ignored, instead of skipping the scene. Default false. */
  coverage?: boolean;
  /**
   * Source of the anisotropy axis (P2.26): `'attribute'` (default) the glTF `TANGENT` the web export carries, interpolated and made orthogonal to the
   * shading normal; `'uv'` dp/du of each triangle's uvs (`uvTangents`), the tangent pbrt-v4 and Mitsuba 3 derive themselves.
   */
  tangent?: 'attribute' | 'uv';
  /** Scrambling seed base. */
  seed?: number;
  /** Scheduler (P3.1): `megakernel` (default) or `wavefront` (queues + indirect dispatch; same estimator, alpha MASK in intersect and shadow (P3.7)). The `pt-wavefront` renderer id sets it. */
  scheduler?: 'megakernel' | 'wavefront';
  /** Wavefront scheduling (P3.2): `refill` starts a slot's next sample inline when its path ends (A/B; `finish()` drains the tail inside the timed region), `layers` = slots per pixel (needs refill). */
  refill?: boolean | 'inline' | 'batch' | 'ordered';
  refillThreshold?: number;
  /** Wavefront queue appends (P3.3): `atomic` (default), `subgroup` (one `atomicAdd` per subgroup) or `auto` (subgroup when the device has the feature). */
  appends?: 'atomic' | 'subgroup' | 'auto';
  /** Wavefront block sorting (P3.4): `none` (default), `material` (shade queue by material), `octant` (next queue by ray octant) or `both`. */
  sort?: 'none' | 'material' | 'octant' | 'both';
  /** P3.9: pick scheduler, refill, layers, appends and sort with `ptwfAutoPolicy` from the film, maxDepth and the device (overrides those settings; the choice is recorded in the row's stats). */
  auto?: boolean;
  layers?: number;
  /** Scene copy binding (P3.11): `split` (two storage buffers) or `packed` (one). Default: the library default (`DEFAULT_SCENE_PACKING`, `packed` since D32). Images are per-pixel identical; only the addressing differs. */
  scenePacking?: 'split' | 'packed';
  /** bvh scene options (X1.i, D25; page args `pt_bvh`, `pt_bvh_decode`, `pt_bvh_walk`): `nodeFormat` `quantized` (default) | `f32` | the experimental wide `bvh4` | `cwbvh8` (hits are the same, tie order aside); `decode` `exact` | `folded` (wide only, default folded); `walk` `nested` | `single` (bvh4 only). */
  /** Texture mip levels (X5, page arg `pt_mips`): `gpu` (default since D34: level 0 only on the CPU, levels 1+ by `@pmndrs/downsampler` in `uploadSceneCopy`; codes differ from the CPU chain by up to 2) or `cpu` (every level built in `buildSceneTextures`). */
  mips?: 'cpu' | 'gpu';
  /** Texture binding (P1.4c2, page arg `pt_tex_binding`): `arrays` (default, every device) or `table` (resource table, Tier 3; megakernel only; Chrome on Metal fails compute passes with a table, V296). Same texels: images equal `arrays` unless `arrays` had to merge buckets. */
  textureBinding?: 'arrays' | 'table';
  bvh?: { nodeFormat?: NodeFormat; decode?: 'exact' | 'folded'; walk?: 'nested' | 'single' };
  /** Sample passes per submit. Default 8 (the wavefront clamps it to what its params ring takes at the scene's maxDepth). */
  passesPerSubmit?: number;
  /**
   * Denoise the final frame with OIDN (`denoise: 'oidn'`, P2 exit item): needs the AOVs (turned on here). The raw image stays the result (metrics run on it);
   * the denoised one is an extra image (`extraImages`, uploaded as `<name>__denoised`). Page arg `pt_denoise=oidn`. Quality only, never in a timed region.
   */
  denoise?: 'oidn';
  /** OIDN weights base URL for the Denoiser package (`.tza` files); default `/@fs/<Denoiser checkout>/packages/denoiser/tzas` of the bench server. */
  denoiseTzaUrl?: string;
  /** P4.1: mode preset (`reference | preview | realtime`, `modes.ts`); explicit settings win. Not for the progressive matrix runs of `realtime` (its `frame` outputs need one encode per frame: `realtime-check` drives those). */
  mode?: PtMode;
  /** P4.1: name of the blue-noise texture the `bluenoise` sampler reads (default `gf-vec2-64`; our own textures, `packages/raytracer/assets/bluenoise`). */
  blueNoise?: string;
  /** Kernel text options (P2.10): drop unreachable declarations / shorten top-level names. Default true; false is the A/B baseline. */
  prune?: boolean;
  shortenNames?: boolean;
}

/**
 * P5: the RGB -> spectrum table of resolution `res`: the prebuilt `bench/assets/generated/rgb2spec-<res>.f32` (`node bench/spectral-table.ts`) when the server has it, else
 * undefined (the kernel then fits it in the page: ~4 s at res 64).
 */
async function fetchSpectralTable(res: number): Promise<Rgb2SpecTable | undefined> {
  try {
    const r = await fetch(`${benchAssets}/generated/rgb2spec-${res}.f32`);
    if (!r.ok) return undefined;
    const data = new Float32Array(await r.arrayBuffer());
    return data.length === 3 * res * res * res * 3 ? { res, data } : undefined;
  } catch {
    return undefined;
  }
}

/** Concatenates glb primitives (already world space) into one triangle mesh. */
function mergeMeshes(parts: readonly TriangleMesh[]): TriangleMesh {
  let vertices = 0;
  let indices = 0;
  for (const part of parts) {
    vertices += part.positions.length / 3;
    indices += part.indices.length;
  }
  const positions = new Float32Array(vertices * 3);
  const normals = new Float32Array(vertices * 3);
  const out = new Uint32Array(indices);
  let vOffset = 0;
  let iOffset = 0;
  for (const part of parts) {
    positions.set(part.positions, vOffset * 3);
    normals.set(part.normals, vOffset * 3);
    for (let i = 0; i < part.indices.length; i++) out[iOffset + i] = part.indices[i]! + vOffset;
    vOffset += part.positions.length / 3;
    iOffset += part.indices.length;
  }
  return { positions, normals, indices: out };
}

export const PT_ADAPTER_ID = 'pt-megakernel';
export const PT_WAVEFRONT_ADAPTER_ID = 'pt-wavefront';

/** Why an `openpbr` material is not rendered exactly: the OpenPBR lobes the megakernel does not have yet (P2.23+, class C), or null. */
export function openPbrUnsupported(desc: MaterialDesc, options: { spectral?: boolean } = {}): string | null {
  if (desc.type !== 'openpbr') return null;
  const m = resolveOpenPbr(desc);
  const missing: string[] = [];
  if (m.transmissionDispersionScale > 0 && !options.spectral) missing.push('dispersion (needs pt_spectral=1: P2.28 / P5)');
  if (m.subsurfaceWeight > 0) missing.push('diffuse transmission (P2.29)');
  // P2.26: anisotropic metals have the tangent frame and the split GGX alphas; a dielectric lobe (and a coat over the metal) is still isotropic
  if (m.specularRoughnessAnisotropy > 0 && (m.baseMetalness === 0 || m.coatWeight > 0)) missing.push('anisotropy on a dielectric or under a coat (P2.26 does metals only)');
  return missing.length > 0 ? `uses ${missing.join(', ')}, not implemented` : null;
}

const toImage = (image: PtImage): Image => ({ width: image.width, height: image.height, channels: 4, data: image.data });

export class PtMegakernelAdapter implements RendererAdapter {
  readonly info: RendererInfo;
  readonly features: FeatureSupport = {
    materials: ['diffuse', 'conductor', 'dielectric', 'black', 'pbr', 'openpbr', 'interface'],
    media: true,
    materialUnsupported: (m) => (this.settings.coverage ? null : openPbrUnsupported(m, { spectral: !!this.settings.spectral })),
    textures: true,
    emissiveMeshes: true,
    directionalLight: true,
    environment: ['uniform', 'hdri'],
    fileGeometry: true,
    rawOutput: true,
    notes:
      'Megakernel, NEE (one shadow ray) + power-heuristic MIS, Owen-Sobol, f32 accumulation. A sun with an angular radius is sampled as a cone (the references use a delta light). ' +
      '`openpbr`: metal (complex Fresnel from base colour), stochastic metalness mix, thick and thin-walled dielectric, absorbing interior medium (Beer), dielectric-over-diffuse and coat via the layered BSDF (stochastic in reference mode), emission; fuzz, thin film, dispersion, subsurface and anisotropy skip the scene (`pt_coverage=1` renders them ignored). ' +
      'File geometry (Sponza) comes from the shared web glb export. HDR environment maps (equirect PFM): bilinear lookup, luminance importance sampling with MIS.',
  };

  gpuContext: GpuContext | null = null;
  approximations: string[] = [];
  /** Extra images of the last `render` (AOVs, counter heatmaps, the denoised frame): the competitor page uploads them next to the main image. */
  extraImages: { name: string; image: Image }[] = [];
  /** AOVs and counters of the last `render` (when `aovs` / `counters` are on). */
  lastAovs: PtAovs | null = null;
  lastCounters: PtCounters | null = null;
  private settings: PtAdapterSettings;
  /** The scheduler instance (`PtMegakernel` or `PtWavefront`), for checks that bind hook buffers or read outputs themselves. */
  get renderer(): PtMegakernel | PtWavefront | null {
    return this.kernel;
  }
  private kernel: PtMegakernel | PtWavefront | null = null;
  private trace: TraceSceneBuffers | null = null;
  private sceneBuffers: SceneCopyBuffers | null = null;
  private scene: SceneDesc | null = null;
  private lastStats: Record<string, number> = {};
  private limits: { maxDepth: number; rrStartDepth?: number; maxVolumeDepth?: number } = { maxDepth: 1 };
  /** `primary: 'vbuffer'`: the bench rasteriser that writes the V-buffer before each sample (not part of the library). */
  private raster: VbufferRaster | null = null;
  private lastLoad: { caps: Caps; copy: SceneCopy; input: SceneCopyInput } | null = null;
  private lastKernelOptions: PtMegakernelOptions | null = null;

  /**
   * Check pages: another megakernel on the same device and scene copy, built from this adapter's own kernel options (materials, textures, alpha mask, environment, mode) with
   * `integrator` overrides merged over them (`restir: null` drops ReSTIR). Film, camera, limits and seed are set as in `load`; the caller destroys it.
   */
  async createKernel(override: { integrator?: PtIntegratorOptions; blueNoise?: string } = {}): Promise<PtMegakernel> {
    const base = this.lastKernelOptions;
    if (!base || !this.trace || !this.sceneBuffers || !this.gpuContext || !this.scene || !this.lastLoad) throw new Error('pt-megakernel: load() first');
    const integrator = { ...base.integrator, ...override.integrator } as PtIntegratorOptions;
    const blueNoise = integrator.sampler === 'bluenoise' && !base.blueNoise ? await fetchBlueNoise(blueNoiseAssets, override.blueNoise ?? 'gf-vec2-64') : base.blueNoise;
    const kernel = await PtMegakernel.create(this.gpuContext.device, this.lastLoad.caps, { ...base, ...(blueNoise ? { blueNoise } : {}), integrator });
    kernel.setScene(this.trace, this.sceneBuffers);
    kernel.setFilm(this.scene.film.width, this.scene.film.height);
    kernel.setCamera(this.scene.camera);
    kernel.setLimits(this.limits);
    kernel.setSeed(this.settings.seed ?? 0);
    return kernel;
  }
  /**
   * Check pages (P4.1b / P4.2a): a WAVEFRONT kernel on the same device and scene copy, from this adapter's kernel options (as `createKernel`, whatever scheduler the adapter itself runs), with
   * `integrator` overrides merged over them. The plain scheduler (no refill) unless `wavefront` says otherwise. Film, camera, limits and seed are set as in `load`; the caller destroys it.
   */
  async createWavefrontKernel(override: { integrator?: PtIntegratorOptions; blueNoise?: string; wavefront?: Partial<PtWavefrontOptions> } = {}): Promise<PtWavefront> {
    const base = this.lastKernelOptions;
    if (!base || !this.trace || !this.sceneBuffers || !this.gpuContext || !this.scene || !this.lastLoad) throw new Error('pt-megakernel: load() first');
    const integrator = { ...base.integrator, ...override.integrator } as PtIntegratorOptions;
    const blueNoise = integrator.sampler === 'bluenoise' && !base.blueNoise ? await fetchBlueNoise(blueNoiseAssets, override.blueNoise ?? 'gf-vec2-64') : base.blueNoise;
    const kernel = await PtWavefront.create(this.gpuContext.device, this.lastLoad.caps, { ...base, ...(blueNoise ? { blueNoise } : {}), integrator, ...(override.wavefront ?? {}) });
    kernel.setScene(this.trace, this.sceneBuffers);
    kernel.setFilm(this.scene.film.width, this.scene.film.height);
    kernel.setCamera(this.scene.camera);
    kernel.setLimits(this.limits);
    kernel.setSeed(this.settings.seed ?? 0);
    return kernel;
  }
  /** What the last `load` built, for checks that drive the primary-hit path themselves (`vbuffer-check`): the device's caps, the CPU scene copy and its input, the V-buffer rasteriser and the GPU buffers. */
  get debug(): { caps: Caps; copy: SceneCopy; input: SceneCopyInput; raster: VbufferRaster | null; trace: TraceSceneBuffers; sceneBuffers: SceneCopyBuffers } {
    if (!this.lastLoad || !this.trace || !this.sceneBuffers) throw new Error('pt-megakernel: load() first');
    return { ...this.lastLoad, raster: this.raster, trace: this.trace, sceneBuffers: this.sceneBuffers };
  }

  constructor(settings: PtAdapterSettings = {}, id?: string) {
    // OIDN needs albedo + normal.
    this.settings = settings.denoise ? { ...settings, aovs: true } : settings;
    this.info = {
      id: id ?? (settings.scheduler === 'wavefront' ? PT_WAVEFRONT_ADAPTER_ID : PT_ADAPTER_ID),
      name: settings.scheduler === 'wavefront' ? '@pmndrs/pathtracer wavefront' : '@pmndrs/pathtracer megakernel',
      version: '0.0.0',
      commit: 'workspace',
      backend: 'webgpu',
      config: {
        scheduler: settings.scheduler ?? 'megakernel',
        ...(settings.scheduler === 'wavefront' ? { refill: settings.refill === true ? 'inline' : settings.refill || 'off', refillThreshold: settings.refillThreshold ?? 0.25, layers: settings.layers ?? 1 } : {}),
        ...(settings.scheduler === 'wavefront' && settings.appends && settings.appends !== 'atomic' ? { appends: settings.appends } : {}),
        ...(settings.scheduler === 'wavefront' && settings.sort && settings.sort !== 'none' ? { sort: settings.sort } : {}),
        ...(settings.alphaPolicy && settings.alphaPolicy !== 'exact' ? { alphaPolicy: settings.alphaPolicy } : {}),
        ...(settings.alphaShadow && settings.alphaShadow !== 'policy' ? { alphaShadow: settings.alphaShadow, ...(settings.alphaShadow === 'roulette' ? { alphaRoulette: settings.alphaRoulette ?? 0.25 } : {}), ...(settings.alphaShadow === 'firstK' ? { alphaFirstK: settings.alphaFirstK ?? 1 } : {}), ...(settings.alphaShadow === 'transmit' ? { alphaTransmit: settings.alphaTransmit ?? 0.5 } : {}) } : {}),
        ...(settings.volumeShadow ? { volumeShadow: settings.volumeShadow } : {}),
        ...(settings.volumeMajorant ? { volumeMajorant: settings.volumeMajorant } : {}),
        ...(settings.volumeTrRoulette ? { volumeTrRoulette: true } : {}),
        ...(settings.volumeForcedNee ? { volumeForcedNee: true } : {}),
        ...(settings.volumeEquiangular ? { volumeEquiangular: true } : {}),
        ...(settings.volumeDensityFilter === 'hardware' ? { volumeDensityFilter: 'hardware' } : {}),
        ...(settings.volumeGridLayout === 'atlas' ? { volumeGridLayout: 'atlas' } : {}),
        ...(settings.volumeOutputs ? { volumeOutputs: 1 } : {}),
        ...(settings.volumeGridFormat ? { volumeGridFormat: settings.volumeGridFormat } : {}),
        ...(settings.primary === 'vbuffer' ? { primary: 'vbuffer' } : {}),
        ...(settings.scenePacking ? { scenePacking: settings.scenePacking } : {}),
        ...(settings.bvh && Object.keys(settings.bvh).length ? { bvh: settings.bvh } : {}),
        ...(settings.mips === 'cpu' ? { mips: 'cpu' } : {}),
        ...(settings.textureBinding === 'table' ? { textureBinding: 'table' } : {}),
        sampler: settings.sampler ?? 'owen-sobol',
        ...(settings.mode ? { mode: settings.mode } : {}),
        ...(settings.sampler === 'bluenoise' || settings.mode === 'realtime' ? { blueNoise: settings.blueNoise ?? 'gf-vec2-64', ...(settings.blueNoiseShift ? { blueNoiseShift: settings.blueNoiseShift } : {}), ...(settings.blueNoisePairs ? { blueNoisePairs: settings.blueNoisePairs } : {}) } : {}),
        ...(settings.frame ? { frame: true } : {}),
        nee: settings.nee ?? true,
        envSampling: settings.envSampling ?? 'auto',
        sunVisible: settings.sunVisible ?? false,
        positions: settings.positions ?? 'barycentric',
        ...(settings.sunAngularRadius !== undefined ? { sunAngularRadius: settings.sunAngularRadius } : {}),
        vndf: settings.vndf ?? 'standard',
        transport: settings.transport ?? 'radiance',
        ...(settings.maxDepth !== undefined ? { maxDepth: settings.maxDepth } : {}),
        ...(settings.maxVolumeDepth !== undefined ? { maxVolumeDepth: settings.maxVolumeDepth } : {}),
        ...(settings.rrStartDepth !== undefined ? { rrStartDepth: settings.rrStartDepth } : {}),
        seed: settings.seed ?? 0,
        ...(settings.counters ? { counters: true } : {}),
        ...(settings.aovs || settings.denoise ? { aovs: true } : {}),
        ...(settings.denoise ? { denoise: settings.denoise } : {}),
        lightSelection: lightSelectionName(settings.lightSelection ?? DEFAULT_LIGHT_SELECTION),
        textureLod: settings.textureLod ?? 0,
        layering: settings.layering ?? 'stochastic',
        ...(settings.albedoBase ? { albedoBase: settings.albedoBase } : {}),
        ...(settings.albedoCorrection && settings.albedoCorrection !== 'none' ? { albedoCorrection: settings.albedoCorrection } : {}),
        ...(settings.exitMis ? { exitMis: settings.exitMis } : {}),
        layerDepth: settings.layerDepth ?? 64,
        layerThickness: settings.layerThickness ?? 1e-6,
        ...(settings.restir ? { restir: JSON.stringify(resolveRestirOptions(settings.restir)) } : {}),
        ...(settings.spectral ? { spectral: JSON.stringify(settings.spectral) } : {}),
        ...(settings.materialLod === 'secondary' ? { materialLod: 'secondary', materialLodDepth: settings.materialLodDepth ?? 1 } : settings.materialLod === 'full' ? { materialLod: 'full' } : {}),
        transmissionTint: settings.transmissionTint ?? true,
        ...(settings.textureLod === 'cone' ? { textureConeScale: settings.textureConeScale ?? 'auto' } : {}),
      },
    };
  }

  /** The environment sampling in force: the setting, else importance for a map and uniform otherwise. */
  private envSampling(hasMap: boolean): NonNullable<PtAdapterSettings['envSampling']> {
    return this.settings.envSampling ?? (hasMap ? 'importance' : 'uniform');
  }

  async load(scene: SceneDesc): Promise<{ loadMs: number; compileMs?: number }> {
    const t0 = performance.now();
    this.scene = scene;
    if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
    const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: this.info.id });
    // Imported here: page-client reads `location` when loaded, so the module must not load in Node (unit tests import this file).
    const { summarizeAdapter } = await import('../../runner/page-client.ts');
    this.gpuContext = { adapter, device, summary: summarizeAdapter(adapter) };

    const files = scene.meshes.filter((mesh) => mesh.geometry.type === 'file');
    let resolveFile: ((geometry: unknown, mesh: MeshDesc) => TriangleMesh) | undefined;
    let fileMeshes: MeshDesc[] = [];
    if (files.length > 0) {
      const { glb, approximations, images } = await fetchSceneGlb(scene);
      this.approximations = approximations;
      // glTF image textures: the Node exporter decoded them (export-scenes.ts); the scene copy reads them through the registered loader
      if (images.size > 0) {
        setImageTextureLoader((desc) => {
          const image = images.get(`${desc.path}#${desc.image}`);
          if (!image) throw new Error(`pt-megakernel: texture ${desc.path}#${desc.image} is not in the web export of '${scene.name}'`);
          return image;
        });
      }
      const prims = readGlbMeshes(glb);
      if (prims.length === scene.meshes.length) {
        // The export has one node per mesh, in scene order, already in world space (Tier 1 glTF scenes: one file mesh per glTF material).
        const byMesh = new Map(scene.meshes.map((mesh, i) => [mesh, prims[i] as TriangleMesh]));
        resolveFile = (_g, mesh) => byMesh.get(mesh) as TriangleMesh;
        fileMeshes = files;
      } else {
        if (files.length > 1) throw new Error(`pt-megakernel: scene '${scene.name}' has ${files.length} file meshes but the web export has ${prims.length} nodes for ${scene.meshes.length} meshes`);
        const merged = mergeMeshes(prims);
        resolveFile = () => merged;
        fileMeshes = files;
      }
    }
    let hdriMap: Image | undefined;
    if (scene.environment?.type === 'hdri') {
      const url = `${benchAssets}/${scene.environment.path}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`environment ${scene.environment.path}: HTTP ${response.status}. Run: node bench/scenes/tier0/generate-assets.ts`);
      hdriMap = readPFM(await response.arrayBuffer());
    }
    let input = sceneDescToCopyInput(scene, { ...(resolveFile ? { resolveFile: resolveFile as never } : {}), ...(hdriMap ? { hdriMap } : {}) });
    // The glb is baked into world space (the exporter applied each mesh's transform): the instance transform must not apply it again.
    for (const mesh of fileMeshes) input.instances[scene.meshes.indexOf(mesh)]!.objectToWorld = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    if (this.settings.tangent === 'uv') for (const mesh of input.meshes) if (mesh.tangents && mesh.uvs) mesh.tangents = uvTangents({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : Uint32Array.from({ length: mesh.positions.length / 3 }, (_, i) => i), uvs: mesh.uvs as Float32Array });
    if (input.sun && this.settings.sunAngularRadius !== undefined) input.sun = { ...input.sun, angularRadius: this.settings.sunAngularRadius };
    // R2b `alphaShadow: 'transmit'`: every cutout card goes into the mask-`alphaMixedMask` instance group (no triangle split), the walk tells cards from the rest
    // D31: a `preview` / `realtime` mode resolves to `transmit` too (unless the caller picked another shadow mode or a solid policy), so the cards need the mask the same way.
    const modeTransmit = this.settings.alphaShadow === undefined && !!this.settings.mode && ptModePreset(this.settings.mode).integrator.alphaShadow === 'transmit' && (!this.settings.alphaPolicy || this.settings.alphaPolicy === 'exact');
    if (this.settings.alphaShadow === 'transmit' || modeTransmit) {
      const mask = this.settings.alphaMixedMask ?? INSTANCE_MASK_MIXED;
      input.instances.forEach((inst) => {
        if (inst.material !== undefined && createMaterial(input.materials[inst.material] ?? {}).alphaMode === ALPHA_MODE.mask) inst.mask = mask;
      });
    }
    // R2b `alphaShadow: 'classified'`: split the cutout cards into mixed and solid instances (and drop the transparent triangles) before the trees are built
    let alphaClasses: AlphaClassStats[] = [];
    if (this.settings.alphaShadow === 'classified') {
      if (this.settings.textureLod === 'cone') this.approximations.push("alphaShadow 'classified': the classification is exact for level-0 texture lookups; textureLod 'cone' can differ at card edges");
      const classified = classifyAlphaCards(input, { ...(this.settings.alphaMixedMask !== undefined ? { mixedMask: this.settings.alphaMixedMask } : {}) });
      input = classified.input;
      alphaClasses = classified.stats;
    }
    const builder = createMeshBvhBuilder();
    const data = await buildTraceScene(
      input.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : null })),
      input.instances.map((instance) => ({ ...instance, static: true })),
      { build: (geometry) => builder.build(geometry), mergeStaticInstances: true, ...(this.settings.bvh?.nodeFormat ? { nodeFormat: this.settings.bvh.nodeFormat } : {}) },
    );
    builder.dispose();
    this.trace = uploadTraceScene(device, data, scene.name);
    const textureOptions = { ...(this.settings.mips === 'cpu' ? { mips: 'cpu' as const } : {}), ...(this.settings.textureBinding === 'table' ? { binding: 'table' as const } : {}) };
    const copy = buildSceneCopy({ ...input, options: { ...input.options, ...(Object.keys(textureOptions).length ? { textures: { ...input.options?.textures, ...textureOptions } } : {}), ...(this.settings.lightSelection ? { lightSelection: this.settings.lightSelection } : {}), surface: { ...input.options?.surface, positions: (this.settings.positions ?? 'barycentric') === 'barycentric' } } });
    this.sceneBuffers = uploadSceneCopy(device, copy, { label: `${scene.name} scene`, ...(this.settings.scenePacking ? { packing: this.settings.scenePacking } : {}) });
    this.lastLoad = { caps, copy, input };
    const loadMs = performance.now() - t0;

    // OpenPBR mode only when the scene has an `openpbr` material (the legacy adapter keeps the other scenes' kernel and images unchanged).
    // A metallic-roughness texture decides metalness per texel (factor x B in [0, factor]): the kernel needs both lobes and the stochastic mix, whatever the factor says.
    // MaterialX codegen (P5, approach b): the generated WGSL of the scene's `materialx` materials goes to the kernel as `integrator.materialx`.
    const mtlx = collectMaterialXPrograms(scene, input.materialNames);
    if (mtlx) this.approximations.push(...mtlx.programs.notes.map((n) => `materialx: ${n}`));
    let materials: PtMaterialOptions | null = null;
    if (Object.values(scene.materials).some((m) => m.type === 'openpbr')) {
      const modeLayered = this.settings.mode ? ptModePreset(this.settings.mode).layered : null;
      const layered: PtLayeredOptions = { layering: this.settings.layering ?? modeLayered?.layering ?? 'stochastic', maxDepth: this.settings.layerDepth ?? 64, nSamples: 1, ...((this.settings.albedoBase ?? modeLayered?.albedoBase) ? { albedoBase: (this.settings.albedoBase ?? modeLayered?.albedoBase) as 'openpbr' | 'internal' } : {}), ...(this.settings.albedoCorrection ? { albedoCorrection: this.settings.albedoCorrection } : {}), ...(this.settings.exitMis ? { exitMis: this.settings.exitMis } : {}) };
      const features = ptMaterialFeatures([...input.materials.map((m) => createMaterial(m.roughnessMetalTexture !== undefined && (m.metalness ?? 0) > 0 ? { ...m, metalness: 0.5 } : m)), ...(mtlx?.featureProbes ?? []).map((p) => createMaterial(p))]);
      // The thin film (P2.27) lives in the albedo layering's Fresnel; the stochastic walk has no film (DIVERGES V82).
      if (features.thinFilm && layered.layering === 'stochastic') {
        layered.layering = 'albedo';
        this.approximations.push('thin film: layering albedo (the stochastic walk has no film)');
      }
      materials = { features, layered, layerThickness: this.settings.layerThickness ?? 1e-6, transmissionTint: this.settings.transmissionTint ?? true };
    }
    const t1 = performance.now();
    const alphaMask = !!copy.textureLayout && input.materials.some((m) => m.alphaMode === ALPHA_MODE.mask);
    if (alphaMask && this.settings.alphaShadow && this.settings.alphaShadow !== 'policy') this.approximations.push(`alpha shadow ${this.settings.alphaShadow}${this.settings.alphaShadow === 'classified' ? ' (exact for level-0 lookups)' : this.settings.alphaShadow === 'roulette' ? ' (unbiased, noisier)' : ' (biased, not a reference)'}`);
    if (alphaMask && this.settings.alphaPolicy && this.settings.alphaPolicy !== 'exact') this.approximations.push(`alpha policy ${this.settings.alphaPolicy}: cutouts ignored for some ray classes (biased, not a reference)`);
    if (this.settings.auto) {
      const choice = ptwfAutoPolicy({ width: scene.film.width, height: scene.film.height, maxDepth: this.settings.maxDepth ?? scene.integrator.maxDepth, subgroups: caps.subgroups });
      const { auto: _auto, ...rest } = this.settings;
      void _auto;
      this.settings = { ...rest, scheduler: choice.scheduler, refill: choice.refill, layers: choice.layers, appends: choice.appends, sort: choice.sort };
      this.autoReason = choice.reason;
    }
    const { denoise: _denoise, denoiseTzaUrl: _tza, refill: _refill, refillThreshold: _thr, layers: _layers, appends: _appends, sort: _sort, mode: _mode, blueNoise: _bnName, ...integratorSettings } = this.settings;
    void _mode;
    void _bnName;
    const needsBlueNoise = integratorSettings.sampler === 'bluenoise' || this.settings.mode === 'realtime'; // P4.1b: the wavefront runs blue noise too
    const blueNoise = needsBlueNoise ? await fetchBlueNoise(blueNoiseAssets, this.settings.blueNoise ?? 'gf-vec2-64') : undefined;
    void _denoise;
    void _tza;
    void _refill;
    void _thr;
    void _layers;
    void _appends;
    void _sort;
    const densityFiles = new Map<string, Float32Array>();
    for (const file of sceneDensityFiles(scene)) {
      const response = await fetch(`${benchAssets}/${file}`);
      if (!response.ok) throw new Error(`density file ${file}: HTTP ${response.status}. Run: node bench/scenes/volumes/vdb-grid.ts`);
      densityFiles.set(file, new Float32Array(await response.arrayBuffer()));
    }
    const volumes0 = sceneVolumes(scene, input, densityFiles);
    // P5.v18: r16float grid = the density rounded to f16 everywhere (texture, densityMax, brick majorants), so the majorant bounds what the GPU reads
    const volumes = volumes0?.grid && this.settings.volumeGridFormat === 'r16float' ? roundVolumesF16(volumes0) : volumes0?.grid && this.settings.volumeGridFormat === 'r8unorm' ? quantizeVolumesR8(volumes0) : volumes0; // r8unorm (V3.1): same rule, with the scale in the medium table
    const spectralTable = integratorSettings.spectral ? await fetchSpectralTable(typeof integratorSettings.spectral === 'object' ? integratorSettings.spectral.res ?? 64 : 64) : undefined;
    const kernelOptions = { ...(volumes?.grid ? { volumeGrid: { ...volumes.grid, ...(this.settings.volumeGridFormat === 'r16float' ? { format: 'r16float' as const } : this.settings.volumeGridFormat === 'r8unorm' ? { format: 'r8unorm' as const, scale: volumes.options.media.find((m) => m.grid)?.grid?.densityScale } : {}), ...(this.settings.volumeMajorant === 'brick' ? { majorant: brickMajorants(volumes.grid.resolution, volumes.grid.density, (this.settings.volumeShadow === 'residual' || this.settings.volumeShadow === 'telescoping') ? { minima: true } : {}) } : {}) } } : {}), ...(spectralTable ? { spectralTable } : {}), ...(this.settings.mode ? { mode: this.settings.mode } : {}), ...(blueNoise ? { blueNoise } : {}), format: sceneFormatOf({ nodeFormat: this.trace.nodeFormat, triangleLeaves: this.trace.triangleLeaves, positionLayout: this.trace.positionLayout, ...(this.settings.bvh?.decode ? { wideDecode: this.settings.bvh.decode } : {}), ...(this.settings.bvh?.walk ? { wideWalk: this.settings.bvh.walk } : {}) }), layout: copy.layout, textures: copy.textureLayout, ...(this.settings.scenePacking ? { scenePacking: this.settings.scenePacking } : {}), integrator: { ...integratorSettings, ...(alphaMask ? { alphaMask } : {}), ...(alphaMask && (this.settings.alphaShadow === 'coverage' || this.settings.alphaShadow === 'stochastic') ? { alphaCoverage: this.settings.alphaCoverage ?? alphaCoverageOf(input) } : {}), envMap: !!hdriMap, envSampling: this.envSampling(!!hdriMap), materials, ...(mtlx ? { materialx: { wgsl: mtlx.programs.wgsl, normal: mtlx.programs.normal } } : {}), ...(volumes ? { volumes: { ...volumes.options, ...(this.settings.maxVolumeDepth !== undefined ? { separateDepth: true } : {}), ...(this.settings.volumeShadow ? { shadowTracking: this.settings.volumeShadow } : {}), ...(this.settings.volumeMajorant ? { majorant: this.settings.volumeMajorant } : {}), ...(this.settings.volumeTrRoulette ? { trRoulette: true } : {}), ...(this.settings.volumeForcedNee ? { forcedNee: true } : {}), ...(this.settings.volumeEquiangular ? { equiangular: true } : {}), ...(this.settings.volumeDensityFilter === 'hardware' ? { densityFilter: 'hardware' as const } : {}), ...(this.settings.volumeGridLayout === 'atlas' ? { gridLayout: 'atlas' as const } : {}), ...(this.settings.volumeOutputs ? { outputs: true } : {}) } } : {}) }, ...(this.settings.prune !== undefined ? { prune: this.settings.prune } : {}), ...(this.settings.shortenNames !== undefined ? { shortenNames: this.settings.shortenNames } : {}), label: this.info.id };
    // D32: `auto` also routes a megakernel over the ~50 KB DXC class to the wavefront (the kernel text is only known here, after the materials / textures are decided).
    if (this.autoReason !== '' && this.settings.scheduler !== 'wavefront' && !volumes && !integratorSettings.spectral) {
      const kernelBytes = ptMegakernelWgsl(caps, ptApplyMode(kernelOptions as unknown as PtMegakernelOptions)).length;
      const choice = ptwfAutoPolicy({ width: scene.film.width, height: scene.film.height, maxDepth: this.settings.maxDepth ?? scene.integrator.maxDepth, subgroups: caps.subgroups, kernelBytes });
      if (choice.scheduler === 'wavefront') {
        this.settings = { ...this.settings, scheduler: choice.scheduler, refill: choice.refill, layers: choice.layers, appends: choice.appends, sort: choice.sort };
        this.autoReason = choice.reason;
      }
    }
    this.lastKernelOptions = kernelOptions as unknown as PtMegakernelOptions;
    this.kernel = this.settings.scheduler === 'wavefront' ? await PtWavefront.create(device, caps, { ...kernelOptions, ...(this.settings.refill !== undefined ? { refill: this.settings.refill } : {}), ...(this.settings.refillThreshold !== undefined ? { refillThreshold: this.settings.refillThreshold } : {}), ...(this.settings.layers !== undefined ? { layers: this.settings.layers } : {}), ...(this.settings.appends !== undefined ? { appends: this.settings.appends } : {}), ...(this.settings.sort !== undefined ? { sort: this.settings.sort } : {}) }) : await PtMegakernel.create(device, caps, kernelOptions);
    this.kernel.setScene(this.trace, this.sceneBuffers);
    this.kernel.setFilm(scene.film.width, scene.film.height);
    this.kernel.setCamera(scene.camera);
    if (this.settings.primary === 'vbuffer') {
      if ((this.settings.positions ?? 'barycentric') !== 'barycentric') throw new Error("pt-megakernel: primary 'vbuffer' needs positions 'barycentric'");
      this.raster?.destroy();
      this.raster = VbufferRaster.create(device, input, scene.film.width, scene.film.height);
      this.kernel.setVbuffer(this.raster.texture);
    }
    this.limits = { maxDepth: this.settings.maxDepth ?? scene.integrator.maxDepth, rrStartDepth: this.settings.rrStartDepth ?? scene.integrator.rrStartDepth, ...(this.settings.maxVolumeDepth !== undefined ? { maxVolumeDepth: this.settings.maxVolumeDepth } : {}) };
    this.kernel.setLimits(this.limits);
    this.kernel.setSeed(this.settings.seed ?? 0);
    this.lastStats = { kernel_bytes: this.kernel.wgslBytes, ...(alphaClasses.length > 0 ? { alpha_tri_total: alphaClasses.reduce((a, c) => a + c.triangles, 0), alpha_tri_mixed: alphaClasses.reduce((a, c) => a + c.mixed, 0), alpha_tri_solid: alphaClasses.reduce((a, c) => a + c.solid, 0), alpha_tri_transparent: alphaClasses.reduce((a, c) => a + c.transparent, 0) } : {}), ...(this.kernel instanceof PtWavefront ? { ...Object.fromEntries(Object.entries(this.kernel.kernelBytes).map(([k, v]) => [`kernel_bytes_${k}`, v])), path_state_bytes: this.kernel.memory().stateBytesPerPath, wavefront_bytes: this.kernel.memory().totalBytes } : {}), ...(materials ? { openpbr: 1, mat_mix: +materials.features.mix, mat_thin: +materials.features.thin, mat_layered: +materials.features.layered, mat_medium: +materials.features.medium, ...(materials.features.aniso ? { mat_aniso: 1 } : {}) } : {}), scene_copy_bytes: copy.byteLength, env_map: hdriMap ? 1 : 0, env_importance: hdriMap && this.envSampling(true) === 'importance' ? 1 : 0, ...(copy.textures ? { texture_arrays: copy.textures.buckets.length, texture_bytes: copy.textures.byteLength, texture_count: copy.textures.slots.length } : {}) };
    return { loadMs, compileMs: performance.now() - t1 };
  }

  async render(budget: RenderBudget): Promise<RenderResult> {
    const kernel = this.kernel;
    const gpu = this.gpuContext;
    if (!kernel || !gpu) throw new Error('pt-megakernel: load() first');
    const spp = budget.spp ?? 64;
    const device = gpu.device;
    const perSubmit = Math.min(this.settings.passesPerSubmit ?? 8, kernel instanceof PtWavefront ? kernel.maxPassesPerEncode() : Infinity);
    if (budget.seed !== undefined) kernel.setSeed(budget.seed);
    const coneSetting = this.settings.textureConeScale ?? 'auto';
    kernel.setLimits({ ...this.limits, textureConeScale: coneSetting === 'auto' ? Math.max(0.125, 1 / Math.sqrt(spp)) : coneSetting });
    {
      const encoder = device.createCommandEncoder({ label: 'pt reset' });
      kernel.reset(encoder);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    const targets = [...(budget.snapshotsAtSpp ?? []).filter((s) => s < spp), spp];
    const snapshots: NonNullable<RenderResult['snapshots']> = [];
    let readbackMs = 0;
    let done = 0;
    const start = performance.now();
    for (const target of targets) {
      while (done < target) {
        const n = Math.min(perSubmit, target - done);
        const encoder = device.createCommandEncoder({ label: 'pt passes' });
        if (this.raster && this.scene) {
          // V-buffer primaries: one raster frame + one pass per sample; the frame's sub-pixel offset is the Owen-Sobol jitter pair of a fixed pixel seed (the engine owns the jitter).
          const basis = ptCameraBasis(this.scene.camera, this.raster.width, this.raster.height);
          const seed = ptPixelSeed(0, 0, this.settings.seed ?? 0);
          for (let i = 0; i < n; i++) {
            const [u, v] = ptSample2D('owen-sobol', seed, done + i, PT_PAIR.jitter);
            const jitter = [u - 0.5, v - 0.5] as const;
            this.raster.encode(encoder, basis, jitter);
            kernel.setVbufferJitter(jitter[0], jitter[1]);
            kernel.encode(encoder, 1);
          }
        } else kernel.encode(encoder, n);
        device.queue.submit([encoder.finish()]);
        done += n;
        await device.queue.onSubmittedWorkDone();
      }
      if (kernel instanceof PtWavefront) await kernel.finish(); // refill: drain the paths still in flight (no-op for the plain scheduler); part of the timed region
      const wallMs = performance.now() - start - readbackMs;
      if (target < spp) {
        const r0 = performance.now();
        snapshots.push({ spp: target, wallMs, image: toImage(await kernel.readImage()) });
        readbackMs += performance.now() - r0;
      }
    }
    const wallMs = performance.now() - start - readbackMs;
    const image = toImage(await kernel.readImage());
    await this.readOutputs(kernel, image);
    const wfStats: Record<string, number> = kernel instanceof PtWavefront && this.settings.refill ? { wf_refill: this.settings.refill === 'batch' ? 2 : this.settings.refill === 'ordered' ? 3 : 1, wf_layers: this.settings.layers ?? 1, wf_finish_chunks: kernel.lastFinish.chunks, wf_finish_iterations: kernel.lastFinish.iterations, wf_finish_ms: kernel.lastFinish.ms, wf_min_samples: kernel.sampleCount.min, wf_max_samples: kernel.sampleCount.max } : {};
    return { image, spp, timings: { wallMs }, snapshots, stats: { ...this.lastStats, ...(kernel.instrumented ? { instrumented: 1 } : {}), ...this.outputStats, ...wfStats } };
  }

  private autoReason = '';
  private outputStats: Record<string, number> = {};

  /** AOVs, counters and the denoised frame after the timed region (readbacks and the denoiser are never inside it). */
  private async readOutputs(kernel: PtMegakernel | PtWavefront, image: Image): Promise<void> {
    this.extraImages = [];
    this.outputStats = {};
    this.lastAovs = null;
    this.lastCounters = null;
    const plane = (name: string, data: Float32Array, channels: 1 | 3, width: number, height: number): void => {
      const rgba = new Float32Array(width * height * 4);
      for (let i = 0; i < width * height; i++) for (let c = 0; c < 3; c++) rgba[i * 4 + c] = data[i * channels + (channels === 1 ? 0 : c)] as number;
      this.extraImages.push({ name, image: { width, height, channels: 4, data: rgba } });
    };
    if (this.settings.aovs) {
      const aovs = await kernel.readAovs();
      this.lastAovs = aovs;
      const viewDepth = new Float32Array(aovs.viewZ.length);
      for (let i = 0; i < viewDepth.length; i++) viewDepth[i] = -(aovs.viewZ[i] as number);
      plane('aov_albedo', aovs.albedo, 3, aovs.width, aovs.height);
      plane('aov_normal', aovs.normal, 3, aovs.width, aovs.height);
      plane('aov_depth', viewDepth, 1, aovs.width, aovs.height);
      this.outputStats.aov_coverage = aovs.coverage.reduce((a, b) => a + b, 0) / Math.max(1, aovs.coverage.length);
    }
    if (this.settings.volumeOutputs && kernel instanceof PtMegakernel) {
      const v = await kernel.readVolumeOutputs();
      const rest = v.radiance.map((x, i) => x - (v.order01[i] as number));
      plane('vol_order01', v.order01, 3, v.width, v.height);
      plane('vol_orders2plus', rest, 3, v.width, v.height);
      plane('vol_transmittance', v.transmittance, 1, v.width, v.height);
      plane('vol_stderr', v.stdError, 3, v.width, v.height);
      plane('vol_first_scatter', v.firstScatterDistance, 1, v.width, v.height);
      const mean = (a: Float32Array) => a.reduce((s, x) => s + x, 0) / Math.max(1, a.length);
      this.outputStats.vol_order01_mean = mean(v.order01);
      this.outputStats.vol_orders2plus_mean = mean(rest);
      this.outputStats.vol_transmittance_mean = mean(v.transmittance);
      this.outputStats.vol_scatter_fraction = mean(v.scatterFraction);
      // standard error of the IMAGE mean from the per-pixel errors (pixels independent): sqrt(sum se^2) / n, per channel averaged
      let se2 = 0;
      for (const x of v.stdError) se2 += x * x;
      this.outputStats.vol_mean_stderr = Math.sqrt(se2 / 3) / (v.width * v.height);
    }
    if (this.settings.counters) {
      const counters = await kernel.readCounters();
      this.lastCounters = counters;
      const word = (w: number) => {
        const out = new Float32Array(counters.width * counters.height);
        for (let p = 0; p < out.length; p++) out[p] = (counters.raw[p * PT_COUNTER_WORDS + w] as number) / Math.max(1, counters.samples);
        return out;
      };
      const rays = word(PT_COUNTER.extensionRays).map((v, p) => v + (counters.raw[p * PT_COUNTER_WORDS + PT_COUNTER.shadowRays] as number) / Math.max(1, counters.samples));
      plane('counter_depth', word(PT_COUNTER.depth), 1, counters.width, counters.height);
      plane('counter_rays', rays, 1, counters.width, counters.height);
      plane('counter_shadow_rays', word(PT_COUNTER.shadowRays), 1, counters.width, counters.height);
      const { summary } = counters;
      this.outputStats = { ...this.outputStats, counter_depth_mean: summary.depth, counter_extension_rays_mean: summary.extensionRays, counter_shadow_rays_mean: summary.shadowRays, ...Object.fromEntries(Object.entries(summary.reasons).map(([k, v]) => [`counter_term_${k}`, v])) };
    }
    if (this.settings.denoise === 'oidn' && this.lastAovs && this.scene && this.gpuContext) {
      const { denoiseWithOidn } = await import('../shared/oidn.ts');
      const t0 = performance.now();
      const clean = await denoiseWithOidn(this.gpuContext.device, image, this.lastAovs, this.scene.camera, { ...(this.settings.denoiseTzaUrl ? { tzaUrl: this.settings.denoiseTzaUrl } : {}) });
      this.outputStats.denoise_ms = performance.now() - t0;
      this.extraImages.push({ name: 'denoised', image: clean });
    }
  }

  dispose(): void {
    this.kernel?.destroy();
    this.raster?.destroy();
    this.raster = null;
    this.trace?.destroy();
    this.sceneBuffers?.destroy();
    this.kernel = null;
    this.trace = null;
    this.sceneBuffers = null;
  }
}

export const createPtMegakernel = (settings?: PtAdapterSettings, id?: string): PtMegakernelAdapter => new PtMegakernelAdapter(settings, id);

/** The scene's volumes with the grid density rounded to f16 and every grid medium's densityMax recomputed from the rounded values (P5.v18). */
function roundVolumesF16(v: NonNullable<ReturnType<typeof sceneVolumes>>): NonNullable<ReturnType<typeof sceneVolumes>> {
  const density = ptRoundGridF16(v.grid!.density);
  let max = 0;
  for (const x of density) max = Math.max(max, x);
  return { ...v, grid: { ...v.grid!, density }, options: { ...v.options, media: v.options.media.map((m) => (m.grid ? { ...m, grid: { ...m.grid, densityMax: max } } : m)) } };
}

/** V3.1: the grid density quantised to r8unorm (per-grid scale = max); densityMax, the brick majorants and the medium's `densityScale` follow from the dequantised values. */
function quantizeVolumesR8(v: NonNullable<ReturnType<typeof sceneVolumes>>): NonNullable<ReturnType<typeof sceneVolumes>> {
  const q = ptQuantizeGridR8(v.grid!.density);
  let max = 0;
  for (const x of q.dequantized) max = Math.max(max, x);
  return { ...v, grid: { ...v.grid!, density: q.dequantized }, options: { ...v.options, media: v.options.media.map((m) => (m.grid ? { ...m, grid: { ...m.grid, densityMax: max, densityScale: q.scale } } : m)) } };
}
