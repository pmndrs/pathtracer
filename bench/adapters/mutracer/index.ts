/**
 * mutracer (@meimu/mutracer, MIT; github.com/pissang/mutracer) behind `RendererAdapter`.
 * Licence rule: we run and measure the published package; none of its code is copied into this repository.
 *
 * The scene goes glTF -> three.js scene (GLTFLoader) -> `fromTHREEScene`, the camera is built from the scene lookAt and
 * converted with `fromTHREECamera` (exact transform and fov), the environment is a float equirect texture.
 *
 * Off or neutral: denoiser (none attached), bevel shading (default on: it rounds the flat-shaded facets of the Tier 0
 * spheres), radiance clamp (0), caustics (undefined), adaptive sampling (undefined), fixed batch size (no
 * targetBatchTimeMs). The linear HDR accumulation is read with `readPixels('hdr')`, not the display canvas.
 * Left at library defaults: Sobol sampling, 4 direct-light candidates per vertex (its NEE strategy).
 *
 * Timing: one `converge({samples})` per requested spp (each restarts, so snapshots are independent images with
 * their own wall time); the timed region is the awaited converge call, readback comes after.
 */
import { Engine, Render, VERSION, fromTHREECamera, fromTHREEScene } from '@meimu/mutracer';
import type { FeatureSupport, Image, RenderBudget, RenderResult, RendererAdapter, RendererInfo } from '../../src/adapter.ts';
import type { SceneDesc } from '../../scenes/schema.ts';
import { loadEnvironment } from '../shared/env.ts';
import { nowMs, type GpuContext } from '../shared/gpu.ts';
import { summarizeAdapter } from '../../runner/page-client.ts';
import { buildThreeScene } from '../shared/three-scene.ts';

export const MUTRACER_PACKAGE = '@meimu/mutracer';
export const MUTRACER_VERSION = '0.3.1';

/** mutracer clamps `bounces` to [1, 50] and caps total path events at min(3 x bounces, 50). */
const MAX_BOUNCES = 50;

/**
 * Scene maxDepth (pbrt: scattering events, 1 = direct lighting only) -> mutracer `bounces`. Calibrated against Mitsuba
 * renders of the Cornell box at depth 1-4 (means 0.16014, 0.18588, 0.19907, 0.20542): mutracer bounces 1, 2, 3 read
 * 0.18586, 0.19915, 0.20553, i.e. bounces = maxDepth - 1 (the camera hit's own direct lighting is not a "bounce").
 * Direct-lighting-only (maxDepth 1) is therefore not expressible; it is exact only where nothing inter-reflects.
 */
export function bouncesFor(maxDepth: number): number {
  return Math.min(Math.max(maxDepth - 1, 1), MAX_BOUNCES);
}

export class MutracerAdapter implements RendererAdapter {
  private readonly bouncesOverride: number | undefined;

  constructor(bouncesOverride?: number) {
    this.bouncesOverride = bouncesOverride;
  }

  readonly info: RendererInfo = {
    id: 'mutracer',
    name: 'mutracer',
    version: MUTRACER_VERSION,
    commit: `npm ${MUTRACER_PACKAGE}@${MUTRACER_VERSION} (library reports ${String(VERSION)})`,
    backend: 'webgpu',
    config: { bevel: false, clamping: 0, sampler: 'sobol', directLightCandidates: 4, samplesPerBatch: 8, bvh: 'builtin/bvh2' },
  };
  readonly features: FeatureSupport = {
    // Conductors: OpenPBR metal is base_color (F0) + F82 tint, not eta/k. Dielectric: path-event cap of 50 breaks the
    // maxDepth 1000 furnace, but the material itself is a smooth dielectric (kept; the cap is reported as a divergence).
    materials: ['diffuse', 'dielectric', 'black'],
    emissiveMeshes: true,
    directionalLight: false,
    environment: ['uniform', 'hdri'],
    rawOutput: true,
    notes:
      'bounces = maxDepth - 1 (min 1, so direct-only maxDepth 1 is not expressible). Bounces clamp to 50 and total path events to min(3 x bounces, 50): scenes needing maxDepth > 51 (furnace-dielectric: 1000) are truncated. ' +
      'Conductors (eta/k) are OpenPBR F0/F82-tint metals: not exact. Emission follows the glTF (two-sided). ' +
      'Raw output: no denoiser, bevel off, clamping 0, HDR accumulation read with readPixels("hdr"). ' +
      'Cannot be turned off: 4 RIS direct-light candidates per vertex (library default, its NEE strategy); 16-pixel alignment of the render target.',
  };

  gpuContext: (GpuContext) | null = null;
  approximations: string[] = [];
  private engine: Engine | null = null;
  private mt: Render | null = null;
  private scene: SceneDesc | null = null;

  async load(scene: SceneDesc, options?: { device?: GPUDevice; baseUrl?: string }): Promise<{ loadMs: number; compileMs?: number }> {
    const t0 = nowMs();
    this.scene = scene;
    const env = await loadEnvironment(scene, options?.baseUrl ?? '/bench/assets');
    const three = await buildThreeScene(scene, env);
    this.approximations = three.approximations;

    const engine = new Engine({ backend: 'webgpu' });
    const errors: unknown[] = [];
    engine.on('error', (error: unknown) => errors.push(error));
    await engine.init();
    if (engine.backend !== 'webgpu') throw new Error(`mutracer fell back to ${String(engine.backend)}: ${String(engine.info?.webgpuFallbackReason)}`);
    engine.scene = fromTHREEScene(three.scene);
    engine.camera = fromTHREECamera(three.camera);
    this.engine = engine;

    const adapter = await navigator.gpu.requestAdapter();
    if (adapter && engine.device) this.gpuContext = { adapter, device: engine.device, summary: summarizeAdapter(adapter) };

    this.mt = this.makeRender(this.bouncesOverride ?? bouncesFor(scene.integrator.maxDepth));
    engine.update();
    const loadMs = nowMs() - t0;

    // Warm-up: compiles shaders and builds the BVH for the first render (the cost is reported, the image discarded).
    const c0 = nowMs();
    await this.mt.converge({ samples: 1 });
    await engine.device?.queue.onSubmittedWorkDone();
    const compileMs = nowMs() - c0;
    if (errors.length > 0) throw new Error(`mutracer reported ${errors.length} error(s): ${String(errors[0])}`);
    return { loadMs, compileMs };
  }

  private makeRender(bounces: number): Render {
    const scene = this.scene!;
    return new Render(this.engine!, {
      width: scene.film.width,
      height: scene.film.height,
      bounces,
      clamping: 0,
      bevel: false,
      maxSamples: 1 << 20,
    });
  }

  private async read(): Promise<Image> {
    const scene = this.scene!;
    const pixels = await this.mt!.readPixels('hdr');
    const { width: w, height: h } = scene.film;
    if (pixels.width === w && pixels.height === h) return { width: w, height: h, channels: 4, data: pixels.data as Float32Array };
    // The target is rounded up to a multiple of 16; the image is the top-left w x h block.
    const data = new Float32Array(w * h * 4);
    for (let y = 0; y < h; y++) data.set((pixels.data as Float32Array).subarray(y * pixels.width * 4, y * pixels.width * 4 + w * 4), y * w * 4);
    return { width: w, height: h, channels: 4, data };
  }

  async render_(spp: number): Promise<{ wallMs: number; image: Image }> {
    const render = this.mt!;
    render.restart();
    await this.engine!.device?.queue.onSubmittedWorkDone();
    const t0 = nowMs();
    await render.converge({ samples: spp });
    await this.engine!.device?.queue.onSubmittedWorkDone();
    const wallMs = nowMs() - t0;
    return { wallMs, image: await this.read() };
  }

  async render(budget: RenderBudget): Promise<RenderResult> {
    if (!this.mt || !this.engine) throw new Error('load() first');
    if (budget.spp === undefined || budget.ms !== undefined) throw new Error('mutracer adapter renders to a fixed spp (ms budgets are not supported)');
    const snapshots: NonNullable<RenderResult['snapshots']> = [];
    for (const s of (budget.snapshotsAtSpp ?? []).filter((v) => v < budget.spp!)) {
      const r = await this.render_(s);
      snapshots.push({ spp: s, wallMs: r.wallMs, image: r.image });
    }
    const final = await this.render_(budget.spp);
    return { image: final.image, spp: budget.spp, timings: { wallMs: final.wallMs }, snapshots, stats: { bouncesUsed: this.bouncesOverride ?? bouncesFor(this.scene!.integrator.maxDepth) } };
  }

  dispose(): void {
    this.mt?.dispose();
    this.engine?.dispose();
    this.mt = null;
    this.engine = null;
  }
}

export const createMutracer = (bounces?: number): MutracerAdapter => new MutracerAdapter(bounces);
