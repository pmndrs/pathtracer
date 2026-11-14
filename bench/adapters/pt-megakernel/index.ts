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
import { buildTraceScene, createMeshBvhBuilder, uploadTraceScene, type TraceSceneBuffers } from '@pmndrs/bvh';
import { createMaterial, detectCaps, requestDeviceForCaps, sceneFormatOf, uploadSceneCopy, buildSceneCopy, lightSelectionName, DEFAULT_LIGHT_SELECTION, type LightSelection, type SceneCopyBuffers } from '@pmndrs/raytracer';
import { PtMegakernel, ptMaterialFeatures, type PtImage, type PtIntegratorOptions, type PtLayeredOptions, type PtMaterialOptions } from '@pmndrs/pathtracer';
import type { FeatureSupport, Image, RenderBudget, RenderResult, RendererAdapter, RendererInfo } from '../../src/adapter.ts';
import type { TriangleMesh } from '../../scenes/geometry.ts';
import { resolveOpenPbr } from '../../scenes/openpbr.ts';
import type { MaterialDesc, MeshDesc, SceneDesc } from '../../scenes/schema.ts';
import { sceneDescToCopyInput } from '../../scenes/to-scene-copy.ts';
import { readPFM } from '../../metrics/pfm.ts';
import { fetchSceneGlb, readGlbMeshes } from '../shared/glb.ts';
import type { GpuContext } from '../shared/gpu.ts';

/** Experiment switches (page args `pt_*`); defaults are the reference configuration. */
export interface PtAdapterSettings extends PtIntegratorOptions {
  /** `barycentric` (default): hit positions from the triangle's vertices; `ray`: origin + t * direction. P1.8 A/B. */
  positions?: 'barycentric' | 'ray';
  /** maxDepth override (the scene's own otherwise). */
  maxDepth?: number;
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
  exitMis?: 'pbrt' | 'none';
  layerDepth?: number;
  layerThickness?: number;
  /** Tint thin-walled transmission by `transmission_color` (default true; pbrt and Mitsuba `thindielectric` cannot tint, so the reference differs on tinted thin walls). */
  transmissionTint?: boolean;
  /** Render `openpbr` materials that use lobes we do not implement (fuzz, thin film, dispersion, subsurface, anisotropy) with those lobes ignored, instead of skipping the scene. Default false. */
  coverage?: boolean;
  /** Scrambling seed base. */
  seed?: number;
  /** Sample passes per submit. Default 8. */
  passesPerSubmit?: number;
  /** Kernel text options (P2.10): drop unreachable declarations / shorten top-level names. Default true; false is the A/B baseline. */
  prune?: boolean;
  shortenNames?: boolean;
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

/** Why an `openpbr` material is not rendered exactly: the OpenPBR lobes the megakernel does not have yet (P2.23+, class C), or null. */
export function openPbrUnsupported(desc: MaterialDesc): string | null {
  if (desc.type !== 'openpbr') return null;
  const m = resolveOpenPbr(desc);
  const missing: string[] = [];
  if (m.fuzzWeight > 0) missing.push('fuzz/sheen (P2.25)');
  if (m.thinFilmWeight > 0) missing.push('thin film (P2.27)');
  if (m.transmissionDispersionScale > 0) missing.push('dispersion (P2.28)');
  if (m.subsurfaceWeight > 0) missing.push('diffuse transmission (P2.29)');
  if (m.specularRoughnessAnisotropy > 0) missing.push('anisotropy (P2.26)');
  return missing.length > 0 ? `uses ${missing.join(', ')}, not implemented` : null;
}

const toImage = (image: PtImage): Image => ({ width: image.width, height: image.height, channels: 4, data: image.data });

export class PtMegakernelAdapter implements RendererAdapter {
  readonly info: RendererInfo;
  readonly features: FeatureSupport = {
    materials: ['diffuse', 'conductor', 'dielectric', 'black', 'pbr', 'openpbr'],
    materialUnsupported: (m) => (this.settings.coverage ? null : openPbrUnsupported(m)),
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
  private readonly settings: PtAdapterSettings;
  private kernel: PtMegakernel | null = null;
  private trace: TraceSceneBuffers | null = null;
  private sceneBuffers: SceneCopyBuffers | null = null;
  private scene: SceneDesc | null = null;
  private lastStats: Record<string, number> = {};
  private limits: { maxDepth: number; rrStartDepth?: number } = { maxDepth: 1 };

  constructor(settings: PtAdapterSettings = {}) {
    this.settings = settings;
    this.info = {
      id: PT_ADAPTER_ID,
      name: '@pmndrs/pathtracer megakernel',
      version: '0.0.0',
      commit: 'workspace',
      backend: 'webgpu',
      config: {
        sampler: settings.sampler ?? 'owen-sobol',
        nee: settings.nee ?? true,
        envSampling: settings.envSampling ?? 'auto',
        sunVisible: settings.sunVisible ?? false,
        positions: settings.positions ?? 'barycentric',
        ...(settings.sunAngularRadius !== undefined ? { sunAngularRadius: settings.sunAngularRadius } : {}),
        vndf: settings.vndf ?? 'standard',
        transport: settings.transport ?? 'radiance',
        ...(settings.maxDepth !== undefined ? { maxDepth: settings.maxDepth } : {}),
        ...(settings.rrStartDepth !== undefined ? { rrStartDepth: settings.rrStartDepth } : {}),
        seed: settings.seed ?? 0,
        lightSelection: lightSelectionName(settings.lightSelection ?? DEFAULT_LIGHT_SELECTION),
        textureLod: settings.textureLod ?? 0,
        layering: settings.layering ?? 'stochastic',
        ...(settings.albedoBase ? { albedoBase: settings.albedoBase } : {}),
        ...(settings.exitMis ? { exitMis: settings.exitMis } : {}),
        layerDepth: settings.layerDepth ?? 64,
        layerThickness: settings.layerThickness ?? 1e-6,
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
    const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter), { label: 'pt-megakernel' });
    // Imported here: page-client reads `location` when loaded, so the module must not load in Node (unit tests import this file).
    const { summarizeAdapter } = await import('../../runner/page-client.ts');
    this.gpuContext = { adapter, device, summary: summarizeAdapter(adapter) };

    const files = scene.meshes.filter((mesh) => mesh.geometry.type === 'file');
    let resolveFile: ((geometry: unknown, mesh: MeshDesc) => TriangleMesh) | undefined;
    let fileMeshes: MeshDesc[] = [];
    if (files.length > 0) {
      const { glb, approximations } = await fetchSceneGlb(scene);
      this.approximations = approximations;
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
      const url = `/bench/assets/${scene.environment.path}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`environment ${scene.environment.path}: HTTP ${response.status}. Run: node bench/scenes/tier0/generate-assets.ts`);
      hdriMap = readPFM(await response.arrayBuffer());
    }
    const input = sceneDescToCopyInput(scene, { ...(resolveFile ? { resolveFile: resolveFile as never } : {}), ...(hdriMap ? { hdriMap } : {}) });
    // The glb is baked into world space (the exporter applied each mesh's transform): the instance transform must not apply it again.
    for (const mesh of fileMeshes) input.instances[scene.meshes.indexOf(mesh)]!.objectToWorld = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    if (input.sun && this.settings.sunAngularRadius !== undefined) input.sun = { ...input.sun, angularRadius: this.settings.sunAngularRadius };
    const builder = createMeshBvhBuilder();
    const data = await buildTraceScene(
      input.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices ? Uint32Array.from(mesh.indices) : null })),
      input.instances.map((instance) => ({ ...instance, static: true })),
      { build: (geometry) => builder.build(geometry), mergeStaticInstances: true },
    );
    builder.dispose();
    this.trace = uploadTraceScene(device, data, scene.name);
    const copy = buildSceneCopy({ ...input, options: { ...input.options, ...(this.settings.lightSelection ? { lightSelection: this.settings.lightSelection } : {}), surface: { ...input.options?.surface, positions: (this.settings.positions ?? 'barycentric') === 'barycentric' } } });
    this.sceneBuffers = uploadSceneCopy(device, copy, { label: `${scene.name} scene` });
    const loadMs = performance.now() - t0;

    // OpenPBR mode only when the scene has an `openpbr` material (the legacy adapter keeps the other scenes' kernel and images unchanged).
    let materials: PtMaterialOptions | null = null;
    if (Object.values(scene.materials).some((m) => m.type === 'openpbr')) {
      const layered: PtLayeredOptions = { layering: this.settings.layering ?? 'stochastic', maxDepth: this.settings.layerDepth ?? 64, nSamples: 1, ...(this.settings.albedoBase ? { albedoBase: this.settings.albedoBase } : {}), ...(this.settings.exitMis ? { exitMis: this.settings.exitMis } : {}) };
      materials = { features: ptMaterialFeatures(input.materials.map((m) => createMaterial(m))), layered, layerThickness: this.settings.layerThickness ?? 1e-6, transmissionTint: this.settings.transmissionTint ?? true };
    }
    const t1 = performance.now();
    this.kernel = await PtMegakernel.create(device, caps, { format: sceneFormatOf(this.trace), layout: copy.layout, textures: copy.textureLayout, integrator: { ...this.settings, envMap: !!hdriMap, envSampling: this.envSampling(!!hdriMap), materials }, ...(this.settings.prune !== undefined ? { prune: this.settings.prune } : {}), ...(this.settings.shortenNames !== undefined ? { shortenNames: this.settings.shortenNames } : {}), label: 'pt-megakernel' });
    this.kernel.setScene(this.trace, this.sceneBuffers);
    this.kernel.setFilm(scene.film.width, scene.film.height);
    this.kernel.setCamera(scene.camera);
    this.limits = { maxDepth: this.settings.maxDepth ?? scene.integrator.maxDepth, rrStartDepth: this.settings.rrStartDepth ?? scene.integrator.rrStartDepth };
    this.kernel.setLimits(this.limits);
    this.kernel.setSeed(this.settings.seed ?? 0);
    this.lastStats = { kernel_bytes: this.kernel.wgslBytes, ...(materials ? { openpbr: 1, mat_mix: +materials.features.mix, mat_thin: +materials.features.thin, mat_layered: +materials.features.layered, mat_medium: +materials.features.medium } : {}), scene_copy_bytes: copy.byteLength, env_map: hdriMap ? 1 : 0, env_importance: hdriMap && this.envSampling(true) === 'importance' ? 1 : 0, ...(copy.textures ? { texture_arrays: copy.textures.buckets.length, texture_bytes: copy.textures.byteLength, texture_count: copy.textures.slots.length } : {}) };
    return { loadMs, compileMs: performance.now() - t1 };
  }

  async render(budget: RenderBudget): Promise<RenderResult> {
    const kernel = this.kernel;
    const gpu = this.gpuContext;
    if (!kernel || !gpu) throw new Error('pt-megakernel: load() first');
    const spp = budget.spp ?? 64;
    const device = gpu.device;
    const perSubmit = this.settings.passesPerSubmit ?? 8;
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
        kernel.encode(encoder, n);
        device.queue.submit([encoder.finish()]);
        done += n;
        await device.queue.onSubmittedWorkDone();
      }
      const wallMs = performance.now() - start - readbackMs;
      if (target < spp) {
        const r0 = performance.now();
        snapshots.push({ spp: target, wallMs, image: toImage(await kernel.readImage()) });
        readbackMs += performance.now() - r0;
      }
    }
    const wallMs = performance.now() - start - readbackMs;
    const image = toImage(await kernel.readImage());
    return { image, spp, timings: { wallMs }, snapshots, stats: this.lastStats };
  }

  dispose(): void {
    this.kernel?.destroy();
    this.trace?.destroy();
    this.sceneBuffers?.destroy();
    this.kernel = null;
    this.trace = null;
    this.sceneBuffers = null;
  }
}

export const createPtMegakernel = (settings?: PtAdapterSettings): PtMegakernelAdapter => new PtMegakernelAdapter(settings);
