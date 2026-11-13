/**
 * three-gpu-pathtracer (WebGPU) behind `RendererAdapter`. One class, two schedulers: the wavefront tracer (the
 * library default) and the megakernel (`useMegakernel(true)`); `tgpt-wavefront` and `tgpt-megakernel` are thin wrappers.
 *
 * We run the library; nothing of it is copied here. Everything that is not raw path tracing is switched off:
 *  - low-res preview and cross-fade (dynamicLowRes, renderDelay, fadeDuration, minSamples), canvas blit (renderToCanvas);
 *  - adaptive frame budget and adaptive preview resolution (adaptiveFrameBudget, adaptiveDynamicRes);
 *  - firefly filters that bias the image (filterGlossyFactor = 0, clampDirect = clampIndirect = 0);
 *  - no denoiser, no upscaler, renderer tone mapping off (the linear accumulation target is read, not the canvas).
 * Kept at library defaults on purpose: Sobol sampling, MIS, frame budget (250000 path steps per call unless `budget=`).
 *
 * Timing: `renderSample()` is called back to back with no readback inside the timed region. The number of calls
 * that reaches the requested spp is found first in an untimed calibration pass (sample counts polled after every
 * call, which is also where snapshot images are read); the timed pass replays exactly that many calls.
 */
import * as THREE from 'three/webgpu';
import { TRANSMISSIVE_BACKGROUND_ENVIRONMENT, WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';
import type { FeatureSupport, Image, RenderBudget, RenderResult, RendererAdapter, RendererInfo } from '../../src/adapter.ts';
import type { SceneDesc } from '../../scenes/schema.ts';
import { loadEnvironment } from '../shared/env.ts';
import { createMaxDevice, nowMs, type GpuContext } from '../shared/gpu.ts';
import { convertQuadEmittersToRectLights } from '../shared/rect-lights.ts';
import { buildThreeScene } from '../shared/three-scene.ts';
import { TGPT_VERSION } from './version.ts';

export type Scheduler = 'wavefront' | 'megakernel';

export interface TgptOptions {
  /** Path steps per `renderSample()` call (library default 250000). Bigger = fewer JS calls. */
  frameBudget?: number;
  /** How emissive quads are lit: `rect` = RectAreaLight (one-sided, next-event estimation; default), `mesh` = glTF emissive mesh (found by BSDF sampling only). */
  emitters?: 'rect' | 'mesh';
  /** Overrides maxBounces (calibration). Default: the scene's maxDepth mapped by `bounceOffset`. */
  bounces?: number;
  /** Where the PFM environments live (default /bench/assets). */
  assetsUrl?: string;
}

const MAX_CALLS = 2_000_000;

export class TgptAdapter implements RendererAdapter {
  readonly info: RendererInfo;
  readonly features: FeatureSupport = {
    // Conductors are not here: glTF carries F0 only, so grazing-angle Fresnel differs from eta/k (DIVERGES V7).
    materials: ['diffuse', 'dielectric', 'black'],
    emissiveMeshes: true,
    directionalLight: true,
    environment: ['uniform', 'hdri'],
    rawOutput: true,
    notes:
      'Emissive meshes are lit by BSDF sampling only (library: "do not take advantage of MIS"), so small lights are noisy. ' +
      'Emission is two-sided (glTF). Conductors approximated by metallic F0 (not listed as supported). ' +
      'Raw output: low-res preview, fade, adaptive budgets, clamps, glossy filter, denoiser, upscaler and canvas blit all off. ' +
      'Cannot be turned off: alpha-weighted sample blending (alpha is forced to 1 by an opaque clear colour); the library ' +
      'measures sample counts itself every 250 ms (a tiny tally kernel plus a 20-byte readback) while renderSample runs.',
  };

  private gpu: GpuContext | null = null;
  private renderer: THREE.WebGPURenderer | null = null;
  private tracer: WebGPUPathTracer | null = null;
  private scene: SceneDesc | null = null;
  approximations: string[] = [];
  private emitterMode: 'rect' | 'mesh' = 'rect';
  private keptEmissiveMeshes = false;

  private readonly scheduler: Scheduler;
  private readonly options: TgptOptions;

  constructor(scheduler: Scheduler, options: TgptOptions = {}) {
    this.scheduler = scheduler;
    this.options = options;
    this.info = {
      id: `tgpt-${scheduler}`,
      name: `three-gpu-pathtracer ${scheduler}`,
      version: TGPT_VERSION.package,
      commit: `npm three-gpu-pathtracer@${TGPT_VERSION.package}, three@${TGPT_VERSION.three}, three-mesh-bvh@${TGPT_VERSION.bvh}`,
      backend: 'webgpu',
      config: { scheduler, emitters: options.emitters ?? 'rect', frameBudget: options.frameBudget ?? 250000, sampler: 'sobol', mis: true, clampDirect: 0, clampIndirect: 0, filterGlossy: 0 },
    };
  }

  /**
   * Scene maxDepth (pbrt: scattering events, 1 = direct lighting only) -> the library's `maxBounces`. Calibrated on
   * quad-light (emissive mesh, direct lighting only, closed-form irradiance) and furnace-diffuse (environment), both
   * schedulers alike: a light found by next-event estimation (environment) counts at the vertex it is sampled from
   * (maxBounces = maxDepth), but an emissive mesh is only found by BSDF sampling, one vertex later (maxBounces =
   * maxDepth + 1 reproduces pbrt's depth). With both kinds of light in one scene the environment part is one depth long.
   */
  bouncesFor(scene: SceneDesc): number {
    const meshEmitters = this.emitterMode === 'mesh' || this.keptEmissiveMeshes;
    return scene.integrator.maxDepth + (meshEmitters && scene.meshes.some((m) => m.emission) ? 1 : 0);
  }

  get gpuContext(): GpuContext | null {
    return this.gpu;
  }

  async load(scene: SceneDesc, options?: { device?: GPUDevice; baseUrl?: string }): Promise<{ loadMs: number; compileMs?: number }> {
    const t0 = nowMs();
    this.scene = scene;
    this.gpu = await createMaxDevice();
    const { device } = this.gpu;
    const canvas = document.createElement('canvas');
    canvas.width = scene.film.width;
    canvas.height = scene.film.height;
    const renderer = new THREE.WebGPURenderer({ canvas, device, antialias: false });
    renderer.toneMapping = THREE.NoToneMapping;
    // Opaque black: with a transparent clear colour the library's alpha-weighted blend drops samples that escape.
    renderer.setClearColor(0x000000, 1);
    await renderer.init();
    this.renderer = renderer;

    const env = await loadEnvironment(scene, options?.baseUrl ?? this.options.assetsUrl ?? '/bench/assets');
    const three = await buildThreeScene(scene, env);
    this.approximations = three.approximations;
    this.emitterMode = this.options.emitters ?? 'rect';
    this.keptEmissiveMeshes = false;
    if (this.emitterMode === 'rect') {
      const conversion = convertQuadEmittersToRectLights(three.scene, scene);
      this.keptEmissiveMeshes = conversion.kept.length > 0;
    }

    const tracer = new WebGPUPathTracer(renderer);
    tracer.synchronizeRenderSize = false;
    tracer.setSize(scene.film.width, scene.film.height);
    tracer.setScene(three.scene, three.camera);
    // useMegakernel() builds a fresh backend that keeps only some settings (not maxBounces or frameBudget), so the
    // scheduler is chosen first and every setting is applied after it.
    if (this.scheduler === 'megakernel') (tracer as unknown as { useMegakernel(on: boolean): void }).useMegakernel(true);
    tracer.renderToCanvas = false;
    tracer.dynamicLowRes = false;
    tracer.adaptiveDynamicRes = false;
    tracer.adaptiveFrameBudget = false;
    tracer.renderDelay = 0;
    tracer.fadeDuration = 0;
    tracer.minSamples = 0;
    tracer.filterGlossyFactor = 0;
    tracer.clampDirect = 0;
    tracer.clampIndirect = 0;
    tracer.stableNoise = true;
    tracer.frameBudget = this.options.frameBudget ?? 250000;
    tracer.maxBounces = this.options.bounces ?? this.bouncesFor(scene);
    tracer.transmissiveBackground = TRANSMISSIVE_BACKGROUND_ENVIRONMENT;
    this.tracer = tracer;
    const loadMs = nowMs() - t0;

    // Kernels compile on the first call (synchronous compile): time it, then discard that sample.
    const c0 = nowMs();
    tracer.maxSamples = 1;
    tracer.renderSample();
    await device.queue.onSubmittedWorkDone();
    const compileMs = nowMs() - c0;
    return { loadMs, compileMs };
  }

  private async counts(): Promise<{ min: number; max: number; avg: number }> {
    return this.tracer!.getSampleCountsAsync();
  }

  /** Reads the linear float accumulation target (not the canvas). Row 0 = top. */
  private async readImage(): Promise<Image> {
    const target = this.tracer!.target as THREE.Texture | null;
    if (!target) throw new Error('path tracer has no output target');
    const size = target.image as { width: number; height: number };
    const { width, height } = size;
    const backend = this.renderer!.backend as unknown as {
      copyTextureToBuffer(t: THREE.Texture, x: number, y: number, w: number, h: number, face: number): Promise<Float32Array>;
    };
    const data = await backend.copyTextureToBuffer(target, 0, 0, width, height, 0);
    const scene = this.scene!;
    if (width !== scene.film.width || height !== scene.film.height) throw new Error(`target is ${width}x${height}, scene wants ${scene.film.width}x${scene.film.height}`);
    return { width, height, channels: 4, data: new Float32Array(data.buffer, data.byteOffset, data.length) };
  }

  async render(budget: RenderBudget): Promise<RenderResult> {
    const tracer = this.tracer;
    if (!tracer || !this.gpu) throw new Error('load() first');
    if (budget.spp === undefined || budget.ms !== undefined) throw new Error('tgpt adapter renders to a fixed spp (ms budgets are not supported)');
    const spp = budget.spp;
    const { device } = this.gpu;
    const snapshotSpps = (budget.snapshotsAtSpp ?? []).filter((s) => s < spp);
    const targets = [...snapshotSpps, spp];

    // ---- Calibration pass (untimed): how many calls reach each target; snapshot images. ----------------------------------
    tracer.maxSamples = spp;
    tracer.reset();
    const callsAt = new Map<number, number>();
    const snapshotImages = new Map<number, { image: Image; avg: number }>();
    let calls = 0;
    let next = 0;
    while (next < targets.length) {
      tracer.renderSample();
      calls++;
      const counts = await this.counts();
      while (next < targets.length && counts.min >= targets[next]!) {
        const target = targets[next]!;
        callsAt.set(target, calls);
        if (target < spp) {
          await device.queue.onSubmittedWorkDone();
          snapshotImages.set(target, { image: await this.readImage(), avg: counts.avg });
        }
        next++;
      }
      if (calls > MAX_CALLS) throw new Error(`gave up after ${MAX_CALLS} renderSample calls (min ${counts.min}/${spp} spp)`);
    }
    const totalCalls = callsAt.get(spp)!;

    // ---- Timed pass: the same number of calls, no readback inside. ----------------------------------------------------------
    tracer.reset();
    await device.queue.onSubmittedWorkDone();
    const checkpoints = new Map<number, number>();
    const checkpointAtCall = new Map<number, number>(snapshotSpps.map((s) => [callsAt.get(s)!, s]));
    const t0 = nowMs();
    for (let call = 1; call <= totalCalls; call++) {
      tracer.renderSample();
      const snapshotSpp = checkpointAtCall.get(call);
      if (snapshotSpp !== undefined) {
        await device.queue.onSubmittedWorkDone();
        checkpoints.set(snapshotSpp, nowMs() - t0);
      }
    }
    await device.queue.onSubmittedWorkDone();
    const wallMs = nowMs() - t0;

    // ---- After the clock: verify the replay really reached spp, read the result. ----------------------------------------------
    let counts = await this.counts();
    let extraCalls = 0;
    while (counts.min < spp && extraCalls < 10000) {
      tracer.renderSample();
      extraCalls++;
      counts = await this.counts();
    }
    await device.queue.onSubmittedWorkDone();
    const image = await this.readImage();

    return {
      image,
      spp,
      timings: { wallMs },
      snapshots: snapshotSpps.map((s) => ({ spp: s, wallMs: checkpoints.get(s) ?? NaN, image: snapshotImages.get(s)!.image })),
      stats: {
        renderSampleCalls: totalCalls,
        replayShortfallCalls: extraCalls,
        minSppReached: counts.min,
        avgSppReached: counts.avg,
        frameBudget: tracer.frameBudget,
        ...Object.fromEntries(snapshotSpps.map((s) => [`snapshot${s}AvgSpp`, snapshotImages.get(s)!.avg])),
      },
    };
  }

  dispose(): void {
    this.tracer?.dispose();
    this.renderer?.dispose();
    this.tracer = null;
    this.renderer = null;
    this.gpu = null;
  }
}
