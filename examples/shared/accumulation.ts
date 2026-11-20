/**
 * A progressive image for the examples' own kernels: an `array<vec4f>` (rgb sum, w = samples) sized to the canvas, cleared when the view
 * changes, drawn with `AccumulationView`, and left alone once `maxFrames` frames are in (the GPU then idles).
 */
import type { ExampleContext } from './example.ts';
import { AccumulationView } from './present.ts';
import { samplingFor, type Sampling } from './sampling.ts';

export class Accumulation {
  readonly view: AccumulationView;
  maxFrames = 1024;
  buffer: GPUBuffer | null = null;
  width = 0;
  height = 0;
  /** Frames recorded since the last reset. */
  frame = 0;
  private dirty = true;
  private drawn = '';
  private readonly ctx: ExampleContext;
  private readonly sampling: Sampling;
  private samplingVersion = 0;

  constructor(ctx: ExampleContext) {
    this.ctx = ctx;
    this.sampling = samplingFor(ctx, { perFrame: false });
    this.samplingVersion = this.sampling.version;
    ctx.stats.set('image', 'ray traced in a compute shader, accumulated · no raster · no denoiser');
    this.view = new AccumulationView(ctx.device, ctx.format);
    ctx.onResize(() => (this.dirty = true));
  }

  /** Starts the image again on the next frame. */
  reset(): void {
    this.dirty = true;
  }

  /**
   * Records one frame: `record(encoder, frame)` adds samples into `buffer` (cleared first after a reset), then the image is drawn.
   * Returns false when there is nothing to do.
   */
  frameWith(encoder: GPUCommandEncoder, record: (encoder: GPUCommandEncoder, frame: number) => void): boolean {
    const [w, h] = this.ctx.filmSize();
    if (w !== this.width || h !== this.height || !this.buffer) {
      this.buffer?.destroy();
      this.buffer = this.ctx.device.createBuffer({ label: 'accumulation', size: w * h * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      this.width = w;
      this.height = h;
      this.dirty = true;
    }
    const maxFrames = this.sampling.stopAt || this.maxFrames;
    if (this.sampling.version !== this.samplingVersion) {
      this.samplingVersion = this.sampling.version;
      this.dirty = true;
    }
    const look = `${this.view.exposure}|${this.view.toneMapping}`;
    if (!this.dirty && this.frame >= maxFrames && look === this.drawn) return false;
    this.drawn = look;
    if (this.dirty) {
      encoder.clearBuffer(this.buffer);
      this.frame = 0;
      this.dirty = false;
    }
    if (this.frame < maxFrames) {
      record(encoder, this.frame);
      this.frame++;
    }
    this.view.encode(encoder, this.buffer, w, h, this.ctx.target());
    this.ctx.stats.set('samples', `${this.frame} / ${maxFrames} frames`);
    this.ctx.stats.set('film', `${w} × ${h}`);
    return true;
  }

  /** The mean so far (RGBA float32, row 0 = top). A readback: for checks, not for every frame. */
  async read(): Promise<{ width: number; height: number; data: Float32Array; samples: number }> {
    const { device } = this.ctx;
    const size = this.width * this.height * 16;
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffer!, 0, staging, 0, size);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const sums = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    const data = new Float32Array(sums.length);
    for (let i = 0; i < sums.length; i += 4) {
      const n = sums[i + 3]! || 1;
      data[i] = sums[i]! / n;
      data[i + 1] = sums[i + 1]! / n;
      data[i + 2] = sums[i + 2]! / n;
      data[i + 3] = 1;
    }
    return { width: this.width, height: this.height, data, samples: this.frame };
  }
}

/** Headless checks read the current example's image through `globalThis.__progressive` (same as the path tracer examples). */
export function exposeForChecks(accumulation: Accumulation): void {
  (globalThis as { __progressive?: unknown }).__progressive = { get samples() { return accumulation.frame; }, readImage: () => accumulation.read() };
}
