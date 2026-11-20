/**
 * A progressive renderer for the examples: a `PtMegakernel` (or `PtWavefront`: the same calls) over a `GpuScene`, sized to the canvas,
 * restarting when the camera or the scene changes and stopping at `maxSamples`. This is the whole loop of a progressive path tracer:
 * `reset`, `encode`, draw the accumulator.
 */
import { PtMegakernel, PtWavefront, ptSceneOptions, type PtLookAtCamera, type PtSceneOptionsOverrides, type PtWavefrontOptions } from '@pmndrs/pathtracer';
import type { GpuScene } from '@pmndrs/raytracer';
import type { ExampleContext } from './example.ts';
import { AccumulationView } from './present.ts';
import { samplingFor, type Sampling } from './sampling.ts';

export interface ProgressiveSettings {
  /** Sample passes per frame (more converges faster, but the page reacts slower). */
  samplesPerFrame: number;
  /** Stop accumulating here; the GPU then idles. */
  maxSamples: number;
  /** Scattering events per path (1 = direct light only). */
  maxDepth: number;
}

/** Which scheduler to compile: the megakernel (default) or the wavefront with its scheduling options. */
export type SchedulerChoice = { scheduler?: 'megakernel' } | ({ scheduler: 'wavefront' } & Pick<PtWavefrontOptions, 'refill' | 'layers' | 'appends' | 'sort'>);

export class ProgressiveRenderer {
  readonly view: AccumulationView;
  readonly settings: ProgressiveSettings = { samplesPerFrame: 2, maxSamples: 4096, maxDepth: 8 };
  private pt: PtMegakernel | PtWavefront;
  private scene: GpuScene;
  private camera: PtLookAtCamera;
  private dirty = true;
  private width = 0;
  private height = 0;
  private started = performance.now();
  /** Exposure and tone mapping of the last drawn frame: a converged image is drawn again only when they change. */
  private drawn = '';

  private readonly ctx: ExampleContext;
  /** The page's "Samples" controls (stop and samples per frame override `settings` when set). */
  private readonly sampling: Sampling;
  private samplingVersion = 0;

  private constructor(ctx: ExampleContext, pt: PtMegakernel | PtWavefront, scene: GpuScene, camera: PtLookAtCamera) {
    this.ctx = ctx;
    this.sampling = samplingFor(ctx, { perFrame: true });
    this.samplingVersion = this.sampling.version;
    ctx.stats.set('image', 'path traced, accumulated · no raster · no denoiser · no upscaler');
    this.pt = pt;
    this.scene = scene;
    this.camera = camera;
    this.view = new AccumulationView(ctx.device, ctx.format);
    ctx.onResize(() => (this.dirty = true));
  }

  /** Compiles a kernel for `scene` (its materials, textures and formats decide the kernel text) and binds the scene. */
  static async create(ctx: ExampleContext, scene: GpuScene, camera: PtLookAtCamera, overrides: PtSceneOptionsOverrides = {}, scheduler: SchedulerChoice = {}): Promise<ProgressiveRenderer> {
    const options = ptSceneOptions(scene, overrides);
    const pt = scheduler.scheduler === 'wavefront' ? await PtWavefront.create(ctx.device, ctx.caps, { ...options, ...scheduler }) : await PtMegakernel.create(ctx.device, ctx.caps, options);
    pt.setScene(scene.trace, scene.buffers);
    const bytes = pt instanceof PtWavefront ? Object.values(pt.kernelBytes).reduce((a, b) => a + (b ?? 0), 0) : pt.wgslBytes;
    ctx.stats.set('kernel', `${(bytes / 1024).toFixed(1)} KB${pt instanceof PtWavefront ? ` in ${Object.keys(pt.kernelBytes).length} kernels` : ''}`);
    const renderer = new ProgressiveRenderer(ctx, pt, scene, camera);
    // Headless checks read the image through this (outside any frame: it maps a buffer).
    (globalThis as { __progressive?: ProgressiveRenderer }).__progressive = renderer;
    return renderer;
  }

  get samples(): number {
    return this.pt.samples;
  }

  /** The mean radiance so far (RGBA float32, row 0 = top). A readback: for checks and saving, not for every frame. */
  async readImage(): ReturnType<PtMegakernel['readImage']> {
    if (this.pt instanceof PtWavefront) await this.pt.finish();
    return this.pt.readImage();
  }

  /** The kernel object (for settings this wrapper does not expose). */
  get kernel(): PtMegakernel | PtWavefront {
    return this.pt;
  }

  /** Starts the image again (call after changing a setting that changes the image). */
  restart(): void {
    this.dirty = true;
  }

  setCamera(camera: PtLookAtCamera): void {
    this.camera = camera;
    this.dirty = true;
  }

  /** Switches to another scene. Its kernel options must match the current kernel's (same materials and textures), else `replace` a new renderer. */
  setScene(scene: GpuScene): void {
    this.pt.setScene(scene.trace, scene.buffers);
    this.scene = scene;
    this.dirty = true;
  }

  /** Records one frame: restart if needed, a few sample passes, then the accumulator to the canvas. */
  frame(encoder: GPUCommandEncoder): boolean {
    const [w, h] = this.ctx.filmSize();
    const maxSamples = this.sampling.stopAt || this.settings.maxSamples;
    const perFrame = this.sampling.perFrame || this.settings.samplesPerFrame;
    if (this.sampling.version !== this.samplingVersion) {
      this.samplingVersion = this.sampling.version;
      this.dirty = true;
    }
    const look = `${this.view.exposure}|${this.view.toneMapping}`;
    const done = !this.dirty && w === this.width && h === this.height && this.pt.samples >= maxSamples && look === this.drawn;
    if (done) return false;
    this.drawn = look;
    if (w !== this.width || h !== this.height) {
      this.pt.setFilm(w, h);
      this.width = w;
      this.height = h;
      this.dirty = true;
    }
    if (this.dirty) {
      this.pt.setCamera(this.camera);
      this.pt.setLimits({ maxDepth: this.settings.maxDepth, rrStartDepth: 3 });
      this.pt.reset(encoder);
      this.dirty = false;
      this.started = performance.now();
    }
    const perEncode = this.pt instanceof PtWavefront ? this.pt.maxPassesPerEncode() : 256;
    const n = Math.min(perFrame, maxSamples - this.pt.samples, perEncode);
    if (n > 0) this.pt.encode(encoder, n);
    this.view.encode(encoder, this.pt.accumulator!, w, h, this.ctx.target());
    const seconds = (performance.now() - this.started) / 1000;
    this.ctx.stats.set('samples', `${this.pt.samples} / ${maxSamples} spp`);
    this.ctx.stats.set('film', `${w} × ${h}`);
    this.ctx.stats.set('time', `${seconds.toFixed(1)} s`);
    return true;
  }

  destroy(): void {
    this.pt.destroy();
    this.view.destroy();
    void this.scene;
  }
}
