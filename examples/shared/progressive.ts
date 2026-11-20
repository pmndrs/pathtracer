/**
 * A progressive renderer for the examples: a `PtMegakernel` over a `GpuScene`, sized to the canvas, restarting when the camera or the
 * scene changes and stopping at `maxSamples`. This is the whole loop of a progressive path tracer: `reset`, `encode`, draw the accumulator.
 */
import { PtMegakernel, ptSceneOptions, type PtLookAtCamera, type PtSceneOptionsOverrides } from '@pmndrs/pathtracer';
import type { GpuScene } from '@pmndrs/raytracer';
import type { ExampleContext } from './example.ts';
import { AccumulationView } from './present.ts';

export interface ProgressiveSettings {
  /** Sample passes per frame (more converges faster, but the page reacts slower). */
  samplesPerFrame: number;
  /** Stop accumulating here; the GPU then idles. */
  maxSamples: number;
  /** Scattering events per path (1 = direct light only). */
  maxDepth: number;
}

export class ProgressiveRenderer {
  readonly view: AccumulationView;
  readonly settings: ProgressiveSettings = { samplesPerFrame: 2, maxSamples: 4096, maxDepth: 8 };
  private pt: PtMegakernel;
  private scene: GpuScene;
  private camera: PtLookAtCamera;
  private dirty = true;
  private width = 0;
  private height = 0;
  private started = performance.now();
  /** Exposure and tone mapping of the last drawn frame: a converged image is drawn again only when they change. */
  private drawn = '';

  private readonly ctx: ExampleContext;

  private constructor(ctx: ExampleContext, pt: PtMegakernel, scene: GpuScene, camera: PtLookAtCamera) {
    this.ctx = ctx;
    this.pt = pt;
    this.scene = scene;
    this.camera = camera;
    this.view = new AccumulationView(ctx.device, ctx.format);
    ctx.onResize(() => (this.dirty = true));
  }

  /** Compiles a kernel for `scene` (its materials, textures and formats decide the kernel text) and binds the scene. */
  static async create(ctx: ExampleContext, scene: GpuScene, camera: PtLookAtCamera, overrides: PtSceneOptionsOverrides = {}): Promise<ProgressiveRenderer> {
    const pt = await PtMegakernel.create(ctx.device, ctx.caps, ptSceneOptions(scene, overrides));
    pt.setScene(scene.trace, scene.buffers);
    ctx.stats.set('kernel', `${(pt.wgslBytes / 1024).toFixed(1)} KB`);
    const renderer = new ProgressiveRenderer(ctx, pt, scene, camera);
    // Headless checks read the image through this (outside any frame: it maps a buffer).
    (globalThis as { __progressive?: ProgressiveRenderer }).__progressive = renderer;
    return renderer;
  }

  get samples(): number {
    return this.pt.samples;
  }

  /** The mean radiance so far (RGBA float32, row 0 = top). A readback: for checks and saving, not for every frame. */
  readImage(): ReturnType<PtMegakernel['readImage']> {
    return this.pt.readImage();
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
    const look = `${this.view.exposure}|${this.view.toneMapping}`;
    const done = !this.dirty && w === this.width && h === this.height && this.pt.samples >= this.settings.maxSamples && look === this.drawn;
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
    const n = Math.min(this.settings.samplesPerFrame, this.settings.maxSamples - this.pt.samples);
    if (n > 0) this.pt.encode(encoder, n);
    this.view.encode(encoder, this.pt.accumulator!, w, h, this.ctx.target());
    const seconds = (performance.now() - this.started) / 1000;
    this.ctx.stats.set('samples', `${this.pt.samples} / ${this.settings.maxSamples}`);
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
