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
 * The readback is premultiplied by its coverage alpha (it is straight colour otherwise; see `MutracerSettings.premultiply`).
 *
 * Timing: one `converge({samples})` per requested spp (each restarts, so snapshots are independent images with
 * their own wall time); the timed region is the awaited converge call, readback comes after.
 */
import { DataTexture, EquirectangularReflectionMapping, FloatType, LinearFilter, RGBAFormat } from 'three';
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

/** Experiment switches (P0.26 A/B); defaults are the bench configuration. */
export interface MutracerSettings {
  /** `directLightCandidates` (library default 4, 1..16). */
  candidates?: number;
  /** `samplingPattern` (library default sobol). */
  sampler?: 'sobol' | 'zsobol' | 'random';
  /** `bevel` (bench default false; library default true). */
  bevel?: boolean;
  /** `dispersionWavelengths` (library default 3). */
  dispersion?: 1 | 3;
  /** `converge({ sampleSequenceOffset })`. */
  offset?: number;
  /**
   * Multiply the RGB read back by the coverage alpha (default true). mutracer's `readPixels('hdr')` returns straight
   * (unassociated) colour: the mean over the samples that hit something, with the hit fraction in alpha. Pixels on a
   * silhouette against an empty background (Cornell box rim) then read as fully lit; the references average black
   * misses in. P0.26: this alone was the whole "mutracer stops converging" plateau.
   */
  premultiply?: boolean;
  /** Experiment: replace a uniform environment's 4x2 map by a constant map of this many texels across (height half). */
  envWidth?: number;
}

export class MutracerAdapter implements RendererAdapter {
  private readonly bouncesOverride: number | undefined;
  private readonly settings: MutracerSettings;

  constructor(bouncesOverride?: number, settings: MutracerSettings = {}) {
    this.bouncesOverride = bouncesOverride;
    this.settings = settings;
    this.info = {
      ...this.info,
      config: {
        ...this.info.config,
        bevel: settings.bevel ?? false,
        sampler: settings.sampler ?? 'sobol',
        directLightCandidates: settings.candidates ?? 4,
        ...(settings.dispersion !== undefined ? { dispersionWavelengths: settings.dispersion } : {}),
        ...(settings.offset !== undefined ? { sampleSequenceOffset: settings.offset } : {}),
        premultiply: settings.premultiply ?? true,
      },
    };
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
    directionalLight: true,
    environment: ['uniform', 'hdri'],
    rawOutput: true,
    notes:
      'bounces = maxDepth - 1 (min 1, so direct-only maxDepth 1 is not expressible). Bounces clamp to 50 and total path events to min(3 x bounces, 50): scenes needing maxDepth > 51 (furnace-dielectric: 1000) are truncated. ' +
      'Conductors (eta/k) are OpenPBR F0/F82-tint metals: not exact. Emission follows the glTF (two-sided). ' +
      'Raw output: no denoiser, bevel off, clamping 0, HDR accumulation read with readPixels("hdr") and multiplied by its coverage alpha (straight colour otherwise: silhouette pixels against an empty background read fully lit). ' +
      'Cannot be turned off: 4 RIS direct-light candidates per vertex (library default, its NEE strategy); 16-pixel alignment of the render target.',
  };

  gpuContext: (GpuContext) | null = null;
  approximations: string[] = [];
  private engine: Engine | null = null;
  private mt: Render | null = null;
  private scene: SceneDesc | null = null;
  alphaProbe: number[] = [];

  async load(scene: SceneDesc, options?: { device?: GPUDevice; baseUrl?: string }): Promise<{ loadMs: number; compileMs?: number }> {
    const t0 = nowMs();
    this.scene = scene;
    const env = await loadEnvironment(scene, options?.baseUrl ?? '/bench/assets');
    const three = await buildThreeScene(scene, env);
    this.approximations = three.approximations;
    const envWidth = this.settings.envWidth;
    const envTex = three.scene.environment as { image?: { data?: Float32Array } } | null;
    if (envWidth && scene.environment?.type === 'uniform' && envTex?.image?.data) {
      const [r, g, b] = [envTex.image.data[0]!, envTex.image.data[1]!, envTex.image.data[2]!];
      const data = new Float32Array(envWidth * (envWidth / 2) * 4);
      for (let i = 0; i < data.length; i += 4) data.set([r, g, b, 1], i);
      const tex = new DataTexture(data, envWidth, envWidth / 2, RGBAFormat, FloatType);
      tex.mapping = EquirectangularReflectionMapping;
      tex.magFilter = LinearFilter;
      tex.minFilter = LinearFilter;
      tex.generateMipmaps = false;
      tex.flipY = false;
      tex.needsUpdate = true;
      three.scene.environment = tex;
      three.scene.background = tex;
    }

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
      bevel: this.settings.bevel ?? false,
      maxSamples: 1 << 20,
      ...(this.settings.candidates !== undefined ? { directLightCandidates: this.settings.candidates } : {}),
      ...(this.settings.sampler !== undefined ? { samplingPattern: this.settings.sampler } : {}),
      ...(this.settings.dispersion !== undefined ? { dispersionWavelengths: this.settings.dispersion } : {}),
    });
  }

  private async read(): Promise<Image> {
    const scene = this.scene!;
    const pixels = await this.mt!.readPixels('hdr');
    const { width: w, height: h } = scene.film;
    const src = pixels.data as Float32Array;
    this.alphaProbe = [0, 0, 0, 0].map((_, k) => src[((k === 0 ? 8 : k === 1 ? 100 : k === 2 ? 7 : 256) * pixels.width + (k === 2 ? 256 : 256)) * 4 + 3]!);
    if (this.settings.premultiply ?? true) for (let i = 0; i < src.length; i += 4) { const a = src[i + 3]!; src[i] = src[i]! * a; src[i + 1] = src[i + 1]! * a; src[i + 2] = src[i + 2]! * a; }
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
    await render.converge({ samples: spp, ...(this.settings.offset !== undefined ? { sampleSequenceOffset: this.settings.offset } : {}) });
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
    return { image: final.image, spp: budget.spp, timings: { wallMs: final.wallMs }, snapshots, stats: { alphaRim8: this.alphaProbe[0] ?? -1, alphaMid100: this.alphaProbe[1] ?? -1, alphaTop7: this.alphaProbe[2] ?? -1, bouncesUsed: this.bouncesOverride ?? bouncesFor(this.scene!.integrator.maxDepth) } };
  }

  dispose(): void {
    this.mt?.dispose();
    this.engine?.dispose();
    this.mt = null;
    this.engine = null;
  }
}

export const createMutracer = (bounces?: number, settings?: MutracerSettings): MutracerAdapter => new MutracerAdapter(bounces, settings);
