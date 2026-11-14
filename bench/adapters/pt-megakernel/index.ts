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
import { detectCaps, requestDeviceForCaps, sceneFormatOf, uploadSceneCopy, buildSceneCopy, lightSelectionName, DEFAULT_LIGHT_SELECTION, type LightSelection, type SceneCopyBuffers } from '@pmndrs/raytracer';
import { PtMegakernel, type PtImage, type PtIntegratorOptions } from '@pmndrs/pathtracer';
import type { FeatureSupport, Image, RenderBudget, RenderResult, RendererAdapter, RendererInfo } from '../../src/adapter.ts';
import type { TriangleMesh } from '../../scenes/geometry.ts';
import type { SceneDesc } from '../../scenes/schema.ts';
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

const toImage = (image: PtImage): Image => ({ width: image.width, height: image.height, channels: 4, data: image.data });

export class PtMegakernelAdapter implements RendererAdapter {
  readonly info: RendererInfo;
  readonly features: FeatureSupport = {
    materials: ['diffuse', 'conductor', 'dielectric', 'black'],
    emissiveMeshes: true,
    directionalLight: true,
    environment: ['uniform', 'hdri'],
    fileGeometry: true,
    rawOutput: true,
    notes:
      'Megakernel, NEE (one shadow ray) + power-heuristic MIS, Owen-Sobol, f32 accumulation. A sun with an angular radius is sampled as a cone (the references use a delta light). ' +
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
    let resolveFile: (() => TriangleMesh) | undefined;
    if (files.length > 1) throw new Error(`pt-megakernel: scene '${scene.name}' has ${files.length} file meshes; the web export is one glb per scene`);
    if (files.length === 1) {
      const { glb, approximations } = await fetchSceneGlb(scene);
      this.approximations = approximations;
      const merged = mergeMeshes(readGlbMeshes(glb));
      resolveFile = () => merged;
    }
    let hdriMap: Image | undefined;
    if (scene.environment?.type === 'hdri') {
      const url = `/bench/assets/${scene.environment.path}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`environment ${scene.environment.path}: HTTP ${response.status}. Run: node bench/scenes/tier0/generate-assets.ts`);
      hdriMap = readPFM(await response.arrayBuffer());
    }
    const input = sceneDescToCopyInput(scene, { ...(resolveFile ? { resolveFile } : {}), ...(hdriMap ? { hdriMap } : {}) });
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

    const t1 = performance.now();
    this.kernel = await PtMegakernel.create(device, caps, { format: sceneFormatOf(this.trace), layout: copy.layout, integrator: { ...this.settings, envMap: !!hdriMap, envSampling: this.envSampling(!!hdriMap) }, ...(this.settings.prune !== undefined ? { prune: this.settings.prune } : {}), ...(this.settings.shortenNames !== undefined ? { shortenNames: this.settings.shortenNames } : {}), label: 'pt-megakernel' });
    this.kernel.setScene(this.trace, this.sceneBuffers);
    this.kernel.setFilm(scene.film.width, scene.film.height);
    this.kernel.setCamera(scene.camera);
    this.kernel.setLimits({ maxDepth: this.settings.maxDepth ?? scene.integrator.maxDepth, rrStartDepth: this.settings.rrStartDepth ?? scene.integrator.rrStartDepth });
    this.kernel.setSeed(this.settings.seed ?? 0);
    this.lastStats = { kernel_bytes: this.kernel.wgslBytes, scene_copy_bytes: copy.byteLength, env_map: hdriMap ? 1 : 0, env_importance: hdriMap && this.envSampling(true) === 'importance' ? 1 : 0 };
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
