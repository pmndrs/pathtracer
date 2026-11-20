/**
 * Convergence: how fast does a path traced image approach the right answer, and does the sampler matter?
 *
 * The page first renders a reference (`reference` samples per pixel, its own seed), then renders the same view again from scratch with
 * another seed and measures the error against the reference at 1, 2, 4, ... samples: relMSE, the mean over pixels of
 * (x - ref)^2 / (ref^2 + 0.01) on luminance. Plain Monte Carlo error falls as 1/spp (the dashed guide); Owen-scrambled Sobol points are
 * stratified and fall faster at low sample counts on smooth regions, PCG (independent random numbers) follows 1/spp. Near the reference's
 * own sample count the curve flattens: the reference's noise is a floor (about relMSE(1 spp) / reference spp).
 *
 * Readbacks happen between batches, outside any timed region: this page measures error, not speed.
 */
import { PtMegakernel, ptSceneOptions, type PtSamplerKind } from '@pmndrs/pathtracer';
import { buildGpuScene, type GpuScene } from '@pmndrs/raytracer';
import { CORNELL_CAMERA, cornellBox } from '../../shared/cornell.ts';
import { defineExample } from '../../shared/example.ts';
import { LAMPS_CAMERA, lampsScene } from '../../shared/lamps.ts';
import { AccumulationView } from '../../shared/present.ts';

const SAMPLERS: PtSamplerKind[] = ['owen-sobol', 'pcg'];
const COLOURS: Record<string, string> = { 'owen-sobol': '#caf543', pcg: '#58b4ff' };
const SCENES = { 'Cornell box': 'cornell', '64 lamps': 'lamps' } as const;

type Series = { sampler: string; points: [number, number][] };

/** Luminance relMSE of `x` against `ref` (RGBA float32). Non-finite pixels are skipped and counted. */
function relMse(x: Float32Array, ref: Float32Array): number {
  let sum = 0, n = 0;
  for (let i = 0; i < ref.length; i += 4) {
    const r = 0.2126 * ref[i]! + 0.7152 * ref[i + 1]! + 0.0722 * ref[i + 2]!;
    const v = 0.2126 * x[i]! + 0.7152 * x[i + 1]! + 0.0722 * x[i + 2]!;
    if (!Number.isFinite(v) || !Number.isFinite(r)) continue;
    sum += ((v - r) * (v - r)) / (r * r + 0.01);
    n++;
  }
  return n ? sum / n : NaN;
}

/** A log-log chart of the series, with a 1/spp guide through the first point of the first series. */
function chart(series: Series[], maxSpp: number): string {
  const W = 360, H = 220, L = 46, R = 12, T = 12, B = 34;
  const all = series.flatMap((s) => s.points).filter(([, e]) => e > 0);
  if (!all.length) return '';
  const ymax = Math.max(...all.map(([, e]) => e)), ymin = Math.min(...all.map(([, e]) => e));
  const ly0 = Math.floor(Math.log10(ymin)), ly1 = Math.ceil(Math.log10(ymax));
  const lx1 = Math.log2(maxSpp);
  const x = (spp: number) => L + (Math.log2(spp) / Math.max(1, lx1)) * (W - L - R);
  const y = (e: number) => T + ((ly1 - Math.log10(e)) / Math.max(1, ly1 - ly0)) * (H - T - B);
  const grid: string[] = [];
  for (let d = ly0; d <= ly1; d++) grid.push(`<line x1="${L}" x2="${W - R}" y1="${y(10 ** d)}" y2="${y(10 ** d)}" class="grid"/><text x="${L - 6}" y="${y(10 ** d) + 4}" text-anchor="end">1e${d}</text>`);
  for (let k = 0; k <= lx1; k += Math.max(1, Math.round(lx1 / 6))) grid.push(`<text x="${x(2 ** k)}" y="${H - B + 16}" text-anchor="middle">${2 ** k}</text>`);
  const lines = series.map((s) => `<polyline fill="none" stroke="${COLOURS[s.sampler] ?? '#fff'}" stroke-width="2" points="${s.points.filter(([, e]) => e > 0).map(([n, e]) => `${x(n).toFixed(1)},${y(e).toFixed(1)}`).join(' ')}"/>`);
  const first = series[0]?.points[0];
  const guide = first ? `<line x1="${x(first[0])}" y1="${y(first[1])}" x2="${x(maxSpp)}" y2="${y((first[1] * first[0]) / maxSpp)}" class="guide"/>` : '';
  const legend = series.map((s, i) => `<g transform="translate(${L + 8 + i * 110},${T + 6})"><rect width="14" height="3" y="5" fill="${COLOURS[s.sampler]}"/><text x="20" y="10">${s.sampler}</text></g>`).join('') + `<g transform="translate(${L + 8 + series.length * 110},${T + 6})"><line x1="0" x2="14" y1="7" y2="7" class="guide"/><text x="20" y="10">1/spp</text></g>`;
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="relMSE against samples per pixel, log-log">${grid.join('')}${guide}${lines.join('')}${legend}<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">samples per pixel</text></svg>`;
}

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const params = { scene: (search.get('scene') === 'lamps' ? 'lamps' : 'cornell') as 'cornell' | 'lamps', reference: Number(search.get('ref') ?? 1024), maxDepth: 6 };
  const style = document.createElement('style');
  style.textContent = `
    .conv { position: fixed; left: 16px; bottom: 16px; padding: 12px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); max-width: calc(100vw - 32px); overflow: auto; }
    .conv svg text { fill: var(--muted); font: 11px var(--font); } .conv .grid { stroke: var(--border); } .conv .guide { stroke: var(--muted); stroke-dasharray: 4 3; }
    .conv p { margin: 0 0 6px; color: var(--muted); font-size: 12px; }`;
  document.head.append(style);
  const panel = document.createElement('div');
  panel.className = 'conv';
  document.body.append(panel);
  const view = new AccumulationView(device, ctx.format);
  stats.set('image', 'path traced, accumulated · no raster · no denoiser · no upscaler');

  let run = 0;
  let lastImage: Float32Array | null = null;
  let checkpoints = 0;
  const go = async () => {
    const mine = ++run;
    const series: Series[] = [];
    let scene: GpuScene | null = null;
    const kernels: PtMegakernel[] = [];
    try {
      ctx.progress('Building the scene…');
      scene = await buildGpuScene(device, params.scene === 'lamps' ? lampsScene() : cornellBox(), { label: params.scene });
      const camera = params.scene === 'lamps' ? LAMPS_CAMERA : CORNELL_CAMERA;
      const [w, h] = ctx.filmSize();
      const draw = (pt: PtMegakernel, encoder: GPUCommandEncoder) => view.encode(encoder, pt.accumulator!, w, h, ctx.target());
      // `n` more samples in batches of at most 64, drawn after each batch so the page shows progress.
      const add = async (pt: PtMegakernel, n: number) => {
        while (n > 0 && mine === run) {
          const k = Math.min(n, 64);
          const e = device.createCommandEncoder();
          pt.encode(e, k);
          draw(pt, e);
          device.queue.submit([e.finish()]);
          await device.queue.onSubmittedWorkDone();
          n -= k;
        }
      };
      const start = (pt: PtMegakernel, seed: number) => {
        pt.setFilm(w, h);
        pt.setCamera(camera);
        pt.setLimits({ maxDepth: params.maxDepth, rrStartDepth: 3 });
        pt.setSeed(seed);
        const e = device.createCommandEncoder();
        pt.reset(e);
        device.queue.submit([e.finish()]);
      };
      for (const sampler of SAMPLERS) {
        ctx.progress(`Compiling (${sampler})…`);
        const pt = await PtMegakernel.create(device, caps, ptSceneOptions(scene, { integrator: { sampler } }));
        pt.setScene(scene.trace, scene.buffers);
        kernels.push(pt);
      }
      // The reference: Owen-Sobol at `reference` spp with its own seed.
      ctx.progress(`Rendering the reference (${params.reference} spp)…`);
      start(kernels[0]!, 9001);
      await add(kernels[0]!, params.reference);
      if (mine !== run) return;
      const ref = (await kernels[0]!.readImage()).data;
      ctx.progress(null);
      for (let s = 0; s < SAMPLERS.length; s++) {
        const pt = kernels[s]!;
        const entry: Series = { sampler: SAMPLERS[s]!, points: [] };
        series.push(entry);
        start(pt, 1 + s);
        let done = 0;
        for (let target = 1; target <= params.reference / 4 && mine === run; target *= 2) {
          await add(pt, target - done);
          done = target;
          const img = (await pt.readImage()).data;
          lastImage = img;
          checkpoints++;
          entry.points.push([target, relMse(img, ref)]);
          stats.set(SAMPLERS[s]!, `${target} spp: relMSE ${entry.points.at(-1)![1].toExponential(2)}`);
          panel.innerHTML = `<p>relMSE against a ${params.reference} spp reference (log-log). Floor ≈ ${(series[0]!.points[0]![1] / params.reference).toExponential(1)}.</p>${chart(series, params.reference / 4)}`;
        }
      }
    } catch (error) {
      ctx.error('The run failed', error);
    } finally {
      await device.queue.onSubmittedWorkDone();
      for (const k of kernels) k.destroy();
      scene?.destroy();
    }
  };

  // Headless checks: one checkpoint is a "sample" here; the image is the last measured one.
  (globalThis as { __progressive?: unknown }).__progressive = {
    get samples() { return checkpoints; },
    readImage: async () => ({ data: lastImage ?? new Float32Array(4) }),
  };

  const f = pane.addFolder({ title: 'Convergence' });
  f.addBinding(params, 'scene', { options: SCENES });
  f.addBinding(params, 'reference', { label: 'reference spp', options: { 64: 64, 256: 256, 1024: 1024, 4096: 4096 } });
  f.addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 12, step: 1 });
  f.addButton({ title: 'Run' }).on('click', () => void go());
  await go();
});
