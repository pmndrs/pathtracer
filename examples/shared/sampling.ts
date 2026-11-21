/**
 * The "Samples" folder every progressive example gets: where to stop accumulating and how many samples to add per frame, so a reader can
 * stop at ONE sample and see the raw noise of a traced image (a clean image a moment after a camera move is accumulation, not raster or a
 * denoiser). `ProgressiveRenderer` and `Accumulation` read it every frame; 0 means "the example's own setting". `?spp=N` in the URL sets
 * the stop.
 */
import type { ExampleContext } from './example.ts';

export interface Sampling {
  /** Stop accumulating at this many samples per pixel (0: the example's default). */
  stopAt: number;
  /** Samples added per frame (0: the example's default). Path tracer examples only. */
  perFrame: number;
  /** Bumped by "Restart" and by a lower stop: renderers start the image again when it changes. */
  version: number;
}

const STOPS: Record<string, number> = {
  'example default': 0,
  '1 (one frame: the raw noise)': 1,
  '2': 2,
  '4': 4,
  '16': 16,
  '64': 64,
  '256': 256,
  '1024': 1024,
  '4096': 4096,
};
const PER_FRAME: Record<string, number> = { 'example default': 0, '1': 1, '2': 2, '4': 4, '8': 8, '16': 16 };

const bySession = new WeakMap<ExampleContext, Sampling>();

/** The page's sampling settings, creating the "Samples" folder on first use. `perFrame: false` hides the per-frame choice. */
export function samplingFor(ctx: ExampleContext, options: { perFrame: boolean }): Sampling {
  let s = bySession.get(ctx);
  if (s) return s;
  const fromUrl = Number(new URLSearchParams(location.search).get('spp') ?? 0);
  s = { stopAt: Number.isFinite(fromUrl) && fromUrl > 0 ? Math.floor(fromUrl) : 0, perFrame: 0, version: 0 };
  bySession.set(ctx, s);
  const sampling = s;
  const folder = ctx.pane.addFolder({ title: 'Samples', expanded: true });
  folder.addBinding(sampling, 'stopAt', { label: 'stop at (spp)', options: sampling.stopAt && !Object.values(STOPS).includes(sampling.stopAt) ? { ...STOPS, [String(sampling.stopAt)]: sampling.stopAt } : STOPS }).on('change', () => sampling.version++);
  if (options.perFrame) folder.addBinding(sampling, 'perFrame', { label: 'samples / frame', options: PER_FRAME }).on('change', () => sampling.version++);
  folder.addButton({ title: 'Restart the image' }).on('click', () => sampling.version++);
  return sampling;
}
