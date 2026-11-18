/**
 * Real-time pipeline session (P4.5 / P4.6 / P4.7, docs/design/realtime.md section 5): the library's real-time path through the shared chain
 * (`chain.ts`), with NO cache. The plain path tracer is the input:
 *
 *   PtMegakernel { mode: 'realtime', frame planes, maxDepth 2, optional ReSTIR DI }   half resolution (960 x 540)
 *     -> PipelineArm: accumulate / pack (planes -> textures, depth guide from viewZ)
 *     -> Denoiser `DiffuseSpecularDenoiser` (SVGF + anti-lag) -> composite -> (reactive mask from the history length)
 *     -> pmndrs/upscaler temporal path (jitter: false)                                  full resolution (1920 x 1080)
 *     -> tonemap (ACES, sRGB, rgba8unorm)
 *
 * The P4.7 preview pipeline is the same session with `preview: true`: maxDepth 4, OIDN small (`moving: 'fast'`, every frame), no history hold.
 *
 * Jitter owner: the path tracer's sampler (pair 0, per pixel; Owen-Sobol with ReSTIR, blue noise without), as in the track N chain
 * (nrc-pipeline.md section 4): the upscaler runs with `jitter: false`, the denoiser gets `jitter` [0, 0], and motion is jitter-free by construction.
 */
import type { PtLookAtCamera, PtMegakernel, PtRestirOptions } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { PipelineArm, type DenoiserKind, type OidnArmSettings, type PipelineSettings } from './chain.ts';

export interface RealtimeSessionOptions {
  scene: SceneDesc;
  width: number;
  height: number;
  displayWidth: number;
  displayHeight: number;
  /** Path length (the `realtime` preset's 2; the preview pipeline's 4). */
  maxDepth?: number;
  /** ReSTIR DI at the first vertex (`integrator.restir`). Owen-Sobol sampler then (modes.ts: the passes do not bind the blue-noise texture yet). */
  restir?: PtRestirOptions | false;
  /** Denoiser of the chain: 'svgf' (default), 'oidn', 'none'. */
  denoise?: DenoiserKind;
  oidn?: Partial<OidnArmSettings>;
  denoiser?: PipelineSettings['denoiser'];
  upscaler?: PipelineSettings['upscaler'];
  /** Reactive mask for the upscaler from the SVGF history length (default off). */
  reactive?: PipelineSettings['reactive'];
  /** The P4.7 preview pipeline: maxDepth 4 and OIDN small every frame unless the options above say otherwise. */
  preview?: boolean;
  /** Extra adapter settings (A/B knobs: `sampler`, `blueNoiseShift`, `materialLod` ...). */
  adapter?: Record<string, unknown>;
}

export interface BatchTime {
  /** Per-iteration ms: median, min, max over the rounds. */
  median: number;
  min: number;
  max: number;
  iterations: number;
  rounds: number;
}

export class RealtimePipelineSession {
  readonly adapter!: PtMegakernelAdapter;
  readonly device!: GPUDevice;
  readonly kernel!: PtMegakernel;
  readonly arm!: PipelineArm;
  readonly options!: RealtimeSessionOptions;
  readonly limits!: { maxDepth: number; rrStartDepth: number };
  private camera: PtLookAtCamera | null = null;
  private cut = true;
  /** Frames since the last cut. */
  frames = 0;

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  static async create(o: RealtimeSessionOptions): Promise<RealtimePipelineSession> {
    const maxDepth = o.maxDepth ?? (o.preview ? 4 : 2);
    const limits = { maxDepth, rrStartDepth: 0 };
    const adapter = new PtMegakernelAdapter({ nee: true, mode: 'realtime', scenePacking: 'packed', maxDepth, rrStartDepth: 0, ...(o.restir ? { restir: o.restir } : {}), ...(o.adapter ?? {}) });
    await adapter.load(o.scene);
    const device = adapter.gpuContext!.device;
    const kernel = adapter.renderer as PtMegakernel;
    kernel.setFilm(o.width, o.height);
    kernel.setLimits(limits);
    kernel.setCamera(o.scene.camera as PtLookAtCamera);
    const denoise: DenoiserKind = o.denoise ?? (o.preview ? 'oidn' : 'svgf');
    const oidn: Partial<OidnArmSettings> | undefined = o.oidn ?? (o.preview ? { moving: 'fast', stillSamples: Infinity, hold: false } : undefined);
    const ps: PipelineSettings = {
      width: o.width,
      height: o.height,
      displayWidth: o.displayWidth,
      displayHeight: o.displayHeight,
      denoise,
      ...(oidn ? { oidn } : {}),
      ...(o.denoiser ? { denoiser: o.denoiser } : {}),
      ...(o.upscaler ? { upscaler: o.upscaler } : {}),
      ...(o.reactive ? { reactive: o.reactive } : {}),
    };
    const arm = await PipelineArm.create(device, ps, o.preview ? 'preview' : 'realtime');
    return new RealtimePipelineSession({ adapter, device, kernel, arm, options: o, limits });
  }

  /** A camera cut on the next frame: zero motion, the denoiser and upscaler histories (and ReSTIR's) restart. */
  cutNext(): void {
    this.cut = true;
  }

  /** The kernel's camera bookkeeping for one frame (first frame / cut: no motion; else the old camera becomes the previous one). */
  private setCamera(camera: PtLookAtCamera): void {
    if (this.cut || !this.camera) {
      this.kernel.setCamera(camera);
      this.kernel.setPrevCamera(camera);
      this.kernel.restir?.dropHistory();
      this.arm.resetHistory();
      this.cut = false;
      this.frames = 0;
    } else {
      this.kernel.nextFrame(camera);
    }
    this.camera = camera;
    this.frames++;
  }

  /** Records the path tracing of one frame and the pack into `encoder` (one sample per pixel). */
  encodeTrace(encoder: GPUCommandEncoder): void {
    this.arm.beginFrame(encoder);
    this.kernel.encode(encoder, 1);
    this.arm.accumulate(encoder, this.kernel.frameBuffers);
    this.arm.pack(encoder, 1);
  }

  /** The kernel's own pass only (no accumulate / pack): the path-trace stage on its own. */
  encodeKernel(encoder: GPUCommandEncoder): void {
    this.kernel.encode(encoder, 1);
  }

  /**
   * One frame: trace + pack in one submit, then denoise + composite + upscale (+ tonemap). `denoise: false` sends the raw sample to the upscaler.
   * No readback; OIDN is awaited by the chain.
   */
  async frame(camera: PtLookAtCamera, o: { denoise?: boolean; tonemap?: boolean } = {}): Promise<void> {
    this.setCamera(camera);
    const encoder = this.device.createCommandEncoder({ label: 'realtime frame' });
    this.encodeTrace(encoder);
    this.device.queue.submit([encoder.finish()]);
    await this.arm.post(camera, { ...(o.denoise === false ? { denoise: false } : {}), ...(o.tonemap ? { tonemap: true } : {}) });
  }

  /** Switch the chain's denoiser (resets the histories). */
  setDenoiser(kind: DenoiserKind, oidn: Partial<OidnArmSettings> = {}): void {
    this.arm.setDenoiser(kind, oidn);
  }

  /**
   * Wall time of a repeated piece of GPU work, per iteration: N iterations recorded into ONE command buffer (>= ~0.25 s), `rounds` rounds, each
   * awaited with `onSubmittedWorkDone` before the next (one batch in flight: CLAUDE.md Traps). Best effort: the machine's load is the caller's to record.
   */
  async timeEncoded(record: (encoder: GPUCommandEncoder) => void, o: { minMs?: number; rounds?: number; maxIterations?: number } = {}): Promise<BatchTime> {
    const minMs = o.minMs ?? 250;
    const rounds = o.rounds ?? 5;
    const batch = async (n: number): Promise<number> => {
      const e = this.device.createCommandEncoder({ label: 'realtime timing' });
      for (let i = 0; i < n; i++) record(e);
      const t0 = performance.now();
      this.device.queue.submit([e.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    await batch(2);
    const probe = (await batch(4)) / 4;
    const n = Math.max(2, Math.min(o.maxIterations ?? 64, Math.ceil(minMs / Math.max(probe, 0.05))));
    const ms: number[] = [];
    for (let r = 0; r < rounds; r++) ms.push((await batch(n)) / n);
    ms.sort((a, b) => a - b);
    return { median: ms[Math.floor(ms.length / 2)] as number, min: ms[0] as number, max: ms[ms.length - 1] as number, iterations: n, rounds };
  }

  /** Same for work that submits its own command buffers (`arm.post`): `call` is invoked N times back to back, then awaited once. */
  async timeCalls(call: () => Promise<void>, o: { minMs?: number; rounds?: number; maxIterations?: number } = {}): Promise<BatchTime> {
    const minMs = o.minMs ?? 250;
    const rounds = o.rounds ?? 5;
    const batch = async (n: number): Promise<number> => {
      await this.device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      for (let i = 0; i < n; i++) await call();
      await this.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    await batch(2);
    const probe = (await batch(4)) / 4;
    const n = Math.max(2, Math.min(o.maxIterations ?? 64, Math.ceil(minMs / Math.max(probe, 0.05))));
    const ms: number[] = [];
    for (let r = 0; r < rounds; r++) ms.push((await batch(n)) / n);
    ms.sort((a, b) => a - b);
    return { median: ms[Math.floor(ms.length / 2)] as number, min: ms[0] as number, max: ms[ms.length - 1] as number, iterations: n, rounds };
  }

  /**
   * Per-stage timing at the current camera (set one with a `frame` first): the kernel pass (with ReSTIR: its passes included), accumulate + pack,
   * the post chain without the denoiser (copy + upscale), the full post (denoise + composite + upscale), the tonemap, and the whole frame
   * (trace + pack + post + tonemap, frames queued back to back). Provisional: wall clock of batched submissions, no GPU timestamps.
   */
  async timeStages(o: { minMs?: number; rounds?: number } = {}): Promise<Record<string, BatchTime>> {
    const camera = this.camera ?? (this.options.scene.camera as PtLookAtCamera);
    if (!this.camera) await this.frame(camera);
    const r: Record<string, BatchTime> = {};
    r.trace = await this.timeEncoded((e) => this.encodeKernel(e), o);
    r.pack = await this.timeEncoded((e) => {
      this.arm.beginFrame(e);
      this.arm.accumulate(e, this.kernel.frameBuffers);
      this.arm.pack(e, 1);
    }, o);
    r.postNoDenoise = await this.timeCalls(() => this.arm.post(camera, { denoise: false }), o);
    r.post = await this.timeCalls(() => this.arm.post(camera), o);
    r.tonemap = await this.timeCalls(async () => this.arm.tonemap(), o);
    r.frame = await this.timeCalls(() => this.frame(camera, { tonemap: true }), o);
    return r;
  }

  destroy(): void {
    this.arm.destroy();
    this.adapter.dispose();
  }
}
