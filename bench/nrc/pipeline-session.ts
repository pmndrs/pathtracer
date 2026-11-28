/**
 * NRC pipeline session (track N, M9): one scene, two arms through the identical denoise + upscale chain (`bench/realtime/chain.ts`).
 *
 *  - plain: the frame-mode megakernel at N samples per pixel per frame (N >= 1, set for equal time);
 *  - nrc:   the same kernel + the cache hook (`NrcCache`, training inline), one sample per pixel per frame.
 *
 * Both kernels are built from one adapter (`integrator.frame`, blue-noise sampler, scene copy packed); the scene's own path length and
 * Russian roulette, so both arms estimate the same image as the 4096 spp references. Used by `nrc-pipeline` (demo) and
 * `nrc-pipeline-study` (headless twin). No readback in `frame()`.
 */
import { MLP_SHAPES, type Precision } from '@pathtracer/mlp';
import type { PtLookAtCamera, PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { inferArmOptions } from '../pages/mlp-common.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { cameraKey } from '../realtime/chain-math.ts';
import { NrcCache, type NrcSettings } from './cache.ts';
import { PipelineArm, type DenoiserKind, type OidnArmSettings, type PipelineSettings } from '../realtime/chain.ts';
import { GradientSamples } from '../realtime/gradient-samples.ts';

export interface SessionOptions {
  scene: SceneDesc;
  width: number;
  height: number;
  displayWidth: number;
  displayHeight: number;
  /** MLP inference arm (`inferArmOptions`): tiled (default), portable (per-record), sgm. */
  infer?: string;
  precision?: Precision;
  subgroups?: number;
  maxDepth?: number;
  cache?: Partial<NrcSettings>;
  denoiser?: PipelineSettings['denoiser'];
  upscaler?: PipelineSettings['upscaler'];
  /** Denoiser of both arms: 'svgf' (default), 'oidn' (with `oidn` settings), 'none'. */
  denoise?: DenoiserKind;
  oidn?: Partial<OidnArmSettings>;
  /** Reactive mask from the SVGF history length for the upscaler (`PipelineSettings.reactive`; then `arm.reactiveEnabled` switches it). Default off. */
  reactive?: PipelineSettings['reactive'];
  /** Who owns the camera jitter (`PipelineSettings.jitter`): 'sampler' (default, per pixel), 'upscaler' (its per-frame sequence) or 'centre'. */
  jitter?: PipelineSettings['jitter'];
  /**
   * Texture level of detail of both arms' kernels: 'cone' (default since 2026-11-27: ray-cone footprint, the library's `mode: 'realtime'`
   * choice; N11.17 cut the interior albedo-guide change 8.7% -> 3.0% per frame) or 0 (every lookup at mip 0, which aliases the albedo /
   * roughness / normal guides at 1 spp; the setting of every chain study before that date). `textureConeScale` scales the cone's start in
   * render pixels; default render / display width (0.5 at 2x upscaling: a display pixel's footprint, the usual upscaler mip bias of
   * log2(render / display)). Classroom, 640x360 -> 2x: a full render pixel (1) blurred the wood (-17% detail on the floor), 0.5 kept most of it
   * (-8%) with the same flicker gain (N11.17).
   */
  textureLod?: 'cone' | 0;
  textureConeScale?: number;
}

export type ArmName = 'plain' | 'nrc';

export class NrcPipelineSession {
  readonly adapter!: PtMegakernelAdapter;
  readonly device!: GPUDevice;
  readonly plain!: PtMegakernel;
  cache!: NrcCache;
  readonly arms!: Record<ArmName, PipelineArm>;
  readonly options!: SessionOptions;
  readonly bounds!: { min: [number, number, number]; max: [number, number, number] };
  readonly inferDescription!: string;
  /** Plain samples per pixel per frame. */
  plainSpp = 1;
  nrcSpp = 1;
  accumulateStill = false;
  svgfHistoryLimit = 0;
  /**
   * N11.18: make A-SVGF gradient samples for the denoiser (`antiLag: 'gradient'`): every frame the previous frame is traced again through
   * the current scene (`bench/realtime/gradient-samples.ts`; one more path-traced frame per frame). 1 sample per pixel, not with
   * `accumulateStill`. The arms hand the textures to SVGF as `inputs.gradient` while this is on.
   */
  gradientSamples = false;
  private grad: Partial<Record<ArmName, GradientSamples>> = {};
  private lastTrain = true;
  readonly samples = { plain: 0, nrc: 0 };
  private stillKey = { plain: '', nrc: '' };
  private camera: PtLookAtCamera | null = null;
  private cut = true;

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  /** The texture cone's start in render pixels: the option, else one display pixel (render / display width, at most 1). */
  static coneScale(o: SessionOptions): number {
    return o.textureConeScale ?? Math.min(1, o.width / o.displayWidth);
  }

  static async create(o: SessionOptions): Promise<NrcPipelineSession> {
    const adapter = new PtMegakernelAdapter({ nee: true, frame: true, sampler: 'bluenoise', scenePacking: 'packed', ...(o.jitter && o.jitter !== 'sampler' ? { cameraJitter: 'engine' as const } : {}), ...((o.textureLod ?? 'cone') === 'cone' ? { textureLod: 'cone' as const, textureConeScale: NrcPipelineSession.coneScale(o) } : {}), ...(o.maxDepth ? { maxDepth: o.maxDepth } : {}) });
    await adapter.load(o.scene);
    const gpu = adapter.gpuContext!;
    const device = gpu.device;
    const plain = adapter.renderer as PtMegakernel;
    plain.setFilm(o.width, o.height);
    const limits = NrcPipelineSession.limits(o);
    plain.setLimits(limits);
    plain.setCamera(o.scene.camera as PtLookAtCamera);
    const bounds = sceneBounds(adapter);
    let precision: Precision = o.precision ?? 'f16';
    if (precision === 'f16' && !device.features.has('shader-f16')) precision = 'f32';
    const arm = inferArmOptions(o.infer ?? 'tiled', gpu.adapter, device, MLP_SHAPES[o.cache?.network ?? 'default'], precision, { records: 64, subgroups: o.subgroups ?? 4 });
    const { caps, trace, sceneBuffers } = adapter.debug;
    const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], { width: o.width, height: o.height, camera: o.scene.camera as PtLookAtCamera, ...limits }, bounds, {
      precision: arm.precision,
      ...(arm.sgm ? { sgm: arm.sgm } : {}),
      ...(arm.infer ? { infer: arm.infer } : {}),
      training: 'inline',
      ...(o.cache ?? {}),
    });
    const ps: PipelineSettings = { width: o.width, height: o.height, displayWidth: o.displayWidth, displayHeight: o.displayHeight, ...(o.denoiser ? { denoiser: o.denoiser } : {}), ...(o.upscaler ? { upscaler: o.upscaler } : {}), ...(o.denoise ? { denoise: o.denoise } : {}), ...(o.oidn ? { oidn: o.oidn } : {}), ...(o.reactive ? { reactive: o.reactive } : {}), ...(o.jitter ? { jitter: o.jitter } : {}) };
    if ((o.textureLod ?? 'cone') === 'cone') for (const k of [plain, cache.render]) k.setLimits({ ...limits, textureConeScale: NrcPipelineSession.coneScale(o) });
    const [pa, na] = await Promise.all([PipelineArm.create(device, ps, 'plain'), PipelineArm.create(device, ps, 'nrc')]);
    return new NrcPipelineSession({ adapter, device, plain, cache, arms: { plain: pa, nrc: na }, options: o, bounds, inferDescription: cache.hash ? cache.description : arm.description });
  }

  static limits(o: SessionOptions): { maxDepth: number; rrStartDepth?: number } {
    const rr = o.scene.integrator.rrStartDepth;
    return { maxDepth: o.maxDepth ?? o.scene.integrator.maxDepth, ...(rr ? { rrStartDepth: rr } : {}) };
  }

  /** A camera cut on the next frame: zero motion, and both arms drop their denoiser and upscaler history. */
  cutNext(): void {
    this.cut = true;
    this.stillKey = { plain: '', nrc: '' };
  }

  private setCamera(camera: PtLookAtCamera): void {
    if (this.cut || !this.camera) {
      this.plain.setCamera(camera);
      this.plain.setPrevCamera(camera);
      this.cache.setCamera(camera);
      this.cache.render.setPrevCamera(camera);
      this.arms.plain.resetHistory();
      this.arms.nrc.resetHistory();
      this.cut = false;
    } else {
      this.plain.nextFrame(camera);
      this.cache.nextCamera(camera);
    }
    this.camera = camera;
  }

  /**
   * Warm-up gate (N11.7): the cache arm shows plain path tracing (1 spp, through the same chain) until the network has trained `frames`
   * frames, training on the same pass meanwhile; `steps` training steps per frame during the gate (default 16, D30; 0 = the cache's `stepsPerFrame`).
   * Takes effect on the next frame; a gate already passed stays open (set it before `cache.resetNetwork()` to gate a fresh network).
   */
  setGate(frames: number, steps = 16): void {
    this.cache.settings.warmupFrames = Math.max(0, Math.round(frames));
    this.cache.settings.warmupStepsPerFrame = Math.max(0, Math.round(steps));
  }

  /** Both arms switch denoiser (the comparison needs the same chain). */
  setDenoiser(kind: DenoiserKind, oidn: Partial<OidnArmSettings> = {}): void {
    this.arms.plain.setDenoiser(kind, oidn);
    this.arms.nrc.setDenoiser(kind, oidn);
  }

  /**
   * One frame of the chosen arms: path tracing (plain N spp / cache frame) + pack in one submit, then denoise + composite + upscale
   * (each arm submits its own; OIDN is awaited). `train`: the cache trains this frame (also while its image is held). `denoise: false`
   * sends the raw image to the upscaler. An arm whose OIDN still frame is held for this camera neither traces nor denoises.
   * Returns the arms that rendered.
   */
  async frame(camera: PtLookAtCamera, o: { arms?: readonly ArmName[]; train?: boolean; denoise?: boolean } = {}): Promise<ArmName[]> {
    this.retrace(o.arms ?? ['plain', 'nrc']); // before the kernels learn about this frame: they still hold last frame's camera and jitter
    this.setCamera(camera); // first: a cut resets the arms, which releases a held frame
    const arms = (o.arms ?? ['plain', 'nrc']).filter((a) => o.denoise === false || !this.arms[a].heldFor(camera));
    const encoder = this.device.createCommandEncoder({ label: 'nrc pipeline frame' });
    const key = this.accumulateStill ? cameraKey(camera) : '';
    if (arms.includes('plain')) this.encodePlain(encoder, key);
    if (arms.includes('nrc')) this.encodeCache(encoder, o.train !== false, key && key + '/' + this.cache.warming);
    this.device.queue.submit([encoder.finish()]);
    for (const a of arms) await this.arms[a].post(camera, { progressiveInput: this.accumulateStill, ...(o.denoise === false ? { denoise: false } : {}) });
    return arms;
  }

  /** The cache side only, no image chain (warming the network). */
  trainOnly(camera: PtLookAtCamera, frames: number): void {
    this.setCamera(camera);
    for (let f = 0; f < frames; f++) {
      const encoder = this.device.createCommandEncoder({ label: 'nrc warm' });
      this.cache.encodeFrame(encoder, { train: true });
      this.device.queue.submit([encoder.finish()]);
      if (f + 1 < frames) this.cache.nextCamera(camera);
    }
  }

  private gradientOn(): boolean {
    return this.gradientSamples && !this.accumulateStill && this.plainSpp === 1 && this.nrcSpp === 1;
  }

  /** The gradient samples' first half: last frame again, with the scene as it is now (its own submit: the kernels' uniforms are per submit). */
  private retrace(arms: readonly ArmName[]): void {
    const on = this.gradientOn() && !this.cut && !!this.camera;
    for (const arm of ['plain', 'nrc'] as const) {
      const g = this.grad[arm];
      this.arms[arm].gradient = on && arms.includes(arm) ? (this.grad[arm] ??= new GradientSamples(this.device, this.options.width, this.options.height, arm)) : null;
      if (!g) continue;
      if (!on || !arms.includes(arm) || !g.have || (arm === 'nrc' && this.cache.warming)) {
        g.invalidate();
        continue;
      }
      const encoder = this.device.createCommandEncoder({ label: `gradient retrace ${arm}` });
      if (arm === 'plain') {
        this.plain.setFrameIndex(this.plain.samples - 1);
        this.plain.encode(encoder, 1);
        g.keepRetrace(encoder, this.plain.frameBuffers.planes);
      } else {
        const c = this.cache;
        const count = (c as unknown as { queryCount: GPUBuffer }).queryCount;
        // the cache's own frame counter (word 1; the resolve pass bumps it) picks the training tiles: replay last frame's value
        encoder.copyBufferToBuffer(this.nrcCounter(), 0, count, 4, 4);
        c.render.setFrameIndex(c.render.samples - 1);
        c.encodeStage(encoder, 'renderPass', this.lastTrain);
        c.encodeStage(encoder, 'inference', this.lastTrain);
        g.keepRetrace(encoder, c.render.frameBuffers.planes);
      }
      this.device.queue.submit([encoder.finish()]);
    }
  }

  private counterCopy: GPUBuffer | null = null;
  private nrcCounter(): GPUBuffer {
    return (this.counterCopy ??= this.device.createBuffer({ label: 'nrc frame counter copy', size: 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
  }

  private beginSamples(encoder: GPUCommandEncoder, arm: ArmName, count: number, key: string): void {
    const pipeline = this.arms[arm];
    pipeline.setSignalHistoryLimit(key && pipeline.kind === 'svgf' ? this.svgfHistoryLimit : 0);
    if (!key || key !== this.stillKey[arm]) {
      this.arms[arm].beginFrame(encoder);
      this.samples[arm] = 0;
    }
    this.stillKey[arm] = key;
    this.samples[arm] += count;
  }

  /** The arm's jitter for this frame onto its kernel (with an engine-owned jitter; the sampler's own jitter needs nothing). */
  private applyJitter(arm: ArmName): void {
    const j = this.arms[arm].nextJitter();
    if ((this.options.jitter ?? 'sampler') === 'sampler') return;
    (arm === 'plain' ? this.plain : this.cache.render).setCameraJitter(j.x, j.y);
  }

  encodePlain(encoder: GPUCommandEncoder, key = ''): void {
    const arm = this.arms.plain;
    this.applyJitter('plain');
    this.beginSamples(encoder, 'plain', this.plainSpp, key);
    for (let k = 0; k < this.plainSpp; k++) {
      this.plain.encode(encoder, 1);
      arm.accumulate(encoder, this.plain.frameBuffers);
    }
    if (this.gradientOn()) (this.grad.plain ??= new GradientSamples(this.device, this.options.width, this.options.height, 'plain')).encode(encoder, this.plain.frameBuffers.planes);
    else this.grad.plain?.invalidate();
    arm.pack(encoder, this.samples.plain);
  }

  encodeCache(encoder: GPUCommandEncoder, train: boolean, key = ''): void {
    const arm = this.arms.nrc;
    this.applyJitter('nrc');
    this.beginSamples(encoder, 'nrc', this.nrcSpp, key);
    const grad = this.gradientOn() && !this.cache.warming;
    if (grad) encoder.copyBufferToBuffer((this.cache as unknown as { queryCount: GPUBuffer }).queryCount, 4, this.nrcCounter(), 0, 4);
    for (let i = 0; i < this.nrcSpp; i++) {
      // Update weights once, after the last render sample, including at the warm-up boundary.
      this.lastTrain = i === this.nrcSpp - 1 && train && this.cache.settings.stepsPerFrame > 0;
      if (grad) {
        // as `encodeFrame`, with the gradient pass between the image and the weight update (the planes are final after 'inference')
        (this.cache as unknown as { frame: number }).frame++;
        this.cache.encodeStage(encoder, 'renderPass', this.lastTrain);
        this.cache.encodeStage(encoder, 'inference', this.lastTrain);
      } else {
        this.cache.encodeFrame(encoder, { train: this.lastTrain });
      }
      arm.accumulate(encoder, this.cache.render.frameBuffers);
      if (grad && this.lastTrain) this.cache.encodeStage(encoder, 'trainSteps', true);
    }
    if (grad) (this.grad.nrc ??= new GradientSamples(this.device, this.options.width, this.options.height, 'nrc')).encode(encoder, this.cache.render.frameBuffers.planes);
    else this.grad.nrc?.invalidate();
    arm.pack(encoder, this.samples.nrc);
  }

  /**
   * Wall time per stage, each submitted alone and waited for (best effort: about +-0.3 ms per submit; no timestamps). Median of `frames`
   * after 3 warm-up frames. Uses the current camera; the cache keeps training.
   */
  async timeStages(frames = 20): Promise<Record<string, number>> {
    const camera = this.camera ?? (this.options.scene.camera as PtLookAtCamera);
    this.setCamera(camera);
    const run = async (fn: (e: GPUCommandEncoder) => void): Promise<number> => {
      const e = this.device.createCommandEncoder({ label: 'nrc pipeline timing' });
      fn(e);
      const t0 = performance.now();
      this.device.queue.submit([e.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    const post = async (arm: PipelineArm, denoise: boolean): Promise<number> => {
      // A history reset per call keeps OIDN on its `moving` network (the per-frame cost while the camera moves).
      arm.resetHistory();
      const t0 = performance.now();
      await arm.post(camera, { denoise });
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    const oidnStill = async (arm: PipelineArm): Promise<number> => {
      arm.primeStill(camera);
      const t0 = performance.now();
      await arm.denoiseOidn(camera, false);
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    await this.arms.plain.ready();
    const t: Record<string, number[]> = { plainSpp: [], cacheFrame: [], plainPack: [], post: [], postNoDenoise: [], ...(this.arms.plain.kind === 'oidn' ? { oidnStill: [] } : {}) };
    for (let f = 0; f < frames + 3; f++) {
      const keep = f >= 3;
      const ms = {
        plainSpp: await run((e) => this.plain.encode(e, 1)),
        cacheFrame: await run((e) => this.cache.encodeFrame(e, { train: this.cache.settings.stepsPerFrame > 0 })),
        plainPack: await run((e) => {
          this.arms.plain.beginFrame(e);
          this.arms.plain.accumulate(e, this.plain.frameBuffers);
          this.arms.plain.pack(e, 1);
        }),
        post: await post(this.arms.plain, true),
        postNoDenoise: await post(this.arms.plain, false),
        ...(this.arms.plain.kind === 'oidn' ? { oidnStill: await oidnStill(this.arms.plain) } : {}),
      };
      if (keep) for (const [k, v] of Object.entries(ms)) t[k]!.push(v);
    }
    const med = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)] as number;
    const r: Record<string, number> = {};
    for (const [k, v] of Object.entries(t)) r[k] = med(v);
    r.denoiseComposite = Math.max(0, r.post! - r.postNoDenoise!);
    r.upscale = r.postNoDenoise!;
    r.equalSpp = Math.max(1, Math.round(r.cacheFrame! / r.plainSpp!));
    r.plainArm = r.equalSpp * r.plainSpp! + r.plainPack! + r.post!;
    r.nrcArm = r.cacheFrame! + r.plainPack! + r.post!;
    this.arms.plain.resetHistory();
    return r;
  }

  destroy(): void {
    for (const g of Object.values(this.grad)) g?.destroy();
    this.counterCopy?.destroy();
    this.arms.plain.destroy();
    this.arms.nrc.destroy();
    this.cache.destroy();
    this.adapter.dispose();
  }
}

/** World-space box of the scene (the cache's position encoding). */
export function sceneBounds(adapter: PtMegakernelAdapter): { min: [number, number, number]; max: [number, number, number] } {
  const { input } = adapter.debug;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const inst of input.instances) {
    const m = inst.objectToWorld;
    const p = input.meshes[inst.meshIndex]!.positions;
    for (let i = 0; i < p.length; i += 3) {
      const x = p[i]!, y = p[i + 1]!, z = p[i + 2]!;
      const w = [m[0]! * x + m[4]! * y + m[8]! * z + m[12]!, m[1]! * x + m[5]! * y + m[9]! * z + m[13]!, m[2]! * x + m[6]! * y + m[10]! * z + m[14]!];
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a]!, w[a]!);
        hi[a] = Math.max(hi[a]!, w[a]!);
      }
    }
  }
  return { min: lo as [number, number, number], max: hi as [number, number, number] };
}
