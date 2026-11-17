/**
 * NRC pipeline session (track N, M9): one scene, two arms through the identical denoise + upscale chain (`pipeline.ts`).
 *
 *  - plain: the frame-mode megakernel at N samples per pixel per frame (N >= 1, set for equal time);
 *  - nrc:   the same kernel + the cache hook (`NrcCache`, training inline), one sample per pixel per frame.
 *
 * Both kernels are built from one adapter (`integrator.frame`, blue-noise sampler, scene copy packed); the scene's own path length and
 * Russian roulette, so both arms estimate the same image as the 4096 spp references. Used by `nrc-pipeline` (demo) and
 * `nrc-pipeline-study` (headless twin). No readback in `frame()`.
 */
import { DEFAULT_SHAPE, type Precision } from '@pathtracer/mlp';
import type { PtLookAtCamera, PtMegakernel } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { inferArmOptions } from '../pages/mlp-common.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { NrcCache, type NrcSettings } from './cache.ts';
import { PipelineArm, type DenoiserKind, type OidnArmSettings, type PipelineSettings } from './pipeline.ts';

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
  private camera: PtLookAtCamera | null = null;
  private cut = true;

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  static async create(o: SessionOptions): Promise<NrcPipelineSession> {
    const adapter = new PtMegakernelAdapter({ nee: true, frame: true, sampler: 'bluenoise', scenePacking: 'packed', ...(o.maxDepth ? { maxDepth: o.maxDepth } : {}) });
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
    const arm = inferArmOptions(o.infer ?? 'tiled', gpu.adapter, device, DEFAULT_SHAPE, precision, { records: 64, subgroups: o.subgroups ?? 4 });
    const { caps, trace, sceneBuffers } = adapter.debug;
    const cache = await NrcCache.create(device, caps, plain, [trace, sceneBuffers], { width: o.width, height: o.height, camera: o.scene.camera as PtLookAtCamera, ...limits }, bounds, {
      precision: arm.precision,
      ...(arm.sgm ? { sgm: arm.sgm } : {}),
      ...(arm.infer ? { infer: arm.infer } : {}),
      training: 'inline',
      ...(o.cache ?? {}),
    });
    const ps: PipelineSettings = { width: o.width, height: o.height, displayWidth: o.displayWidth, displayHeight: o.displayHeight, ...(o.denoiser ? { denoiser: o.denoiser } : {}), ...(o.upscaler ? { upscaler: o.upscaler } : {}), ...(o.denoise ? { denoise: o.denoise } : {}), ...(o.oidn ? { oidn: o.oidn } : {}) };
    const [pa, na] = await Promise.all([PipelineArm.create(device, ps, 'plain'), PipelineArm.create(device, ps, 'nrc')]);
    return new NrcPipelineSession({ adapter, device, plain, cache, arms: { plain: pa, nrc: na }, options: o, bounds, inferDescription: arm.description });
  }

  static limits(o: SessionOptions): { maxDepth: number; rrStartDepth?: number } {
    const rr = o.scene.integrator.rrStartDepth;
    return { maxDepth: o.maxDepth ?? o.scene.integrator.maxDepth, ...(rr ? { rrStartDepth: rr } : {}) };
  }

  /** A camera cut on the next frame: zero motion, and both arms drop their denoiser and upscaler history. */
  cutNext(): void {
    this.cut = true;
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
    this.setCamera(camera); // first: a cut resets the arms, which releases a held frame
    const arms = (o.arms ?? ['plain', 'nrc']).filter((a) => o.denoise === false || !this.arms[a].heldFor(camera));
    const encoder = this.device.createCommandEncoder({ label: 'nrc pipeline frame' });
    if (arms.includes('plain')) this.encodePlain(encoder);
    if (arms.includes('nrc')) this.encodeCache(encoder, o.train !== false);
    this.device.queue.submit([encoder.finish()]);
    for (const a of arms) await this.arms[a].post(camera, { ...(o.denoise === false ? { denoise: false } : {}) });
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

  encodePlain(encoder: GPUCommandEncoder): void {
    const arm = this.arms.plain;
    arm.beginFrame(encoder);
    for (let k = 0; k < this.plainSpp; k++) {
      this.plain.encode(encoder, 1);
      arm.accumulate(encoder, this.plain.frameBuffers);
    }
    arm.pack(encoder, this.plainSpp);
  }

  encodeCache(encoder: GPUCommandEncoder, train: boolean): void {
    const arm = this.arms.nrc;
    arm.beginFrame(encoder);
    this.cache.encodeFrame(encoder, { train: train && this.cache.settings.stepsPerFrame > 0 });
    arm.accumulate(encoder, this.cache.render.frameBuffers);
    arm.pack(encoder, 1);
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
