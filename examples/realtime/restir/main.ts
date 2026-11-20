/**
 * ReSTIR DI in real time (experimental): one sample per pixel per frame over 64 lamps, with and without reservoir resampling.
 *
 * Plain next-event estimation picks one light per pixel and frame, so with many small lights most pixels pick one that contributes
 * little. ReSTIR DI (`integrator.restir`) draws several candidates per pixel, keeps one in proportion to its contribution, then merges the
 * kept samples of the previous frame (temporal) and of neighbouring pixels (spatial). Look at a single frame (averaging off): the same
 * one sample per pixel is far less noisy. Biased mode is cheaper and less noisy still, but darker where neighbours disagree.
 */
import { PtMegakernel, ptSceneOptions, type PtRestirOptions } from '@pmndrs/pathtracer';
import { buildGpuScene } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { FrameView } from '../../shared/frame-view.ts';
import { LAMPS_CAMERA, lampsScene } from '../../shared/lamps.ts';
import { OrbitCamera } from '../../shared/orbit.ts';

defineExample(async (ctx) => {
  const { device, caps, pane, stats, canvas } = ctx;
  const search = new URLSearchParams(location.search);
  const params = {
    restir: (search.get('restir') ?? 'unbiased') as 'off' | 'unbiased' | 'biased',
    candidates: 8,
    temporal: true,
    spatial: 2,
    accumulate: search.get('accumulate') === '1',
    spin: search.get('spin') === '1',
    maxDepth: 1,
    exposure: 1,
  };
  stats.set('image', 'path traced, 1 sample per pixel per frame · no raster · no denoiser · no upscaler');
  ctx.setRenderScale(0.5);
  const orbit = new OrbitCamera(canvas, LAMPS_CAMERA);
  const view = new FrameView(device, ctx.format);
  view.accumulate = params.accumulate;
  const scene = await buildGpuScene(device, lampsScene(), { label: 'lamps' });
  stats.set('lights', '64 lamps, sun, sky');
  let pt: PtMegakernel | null = null;
  let size = [0, 0];

  const rebuild = async () => {
    ctx.progress('Compiling…');
    try {
      const restir: PtRestirOptions | null = params.restir === 'off' ? null : { bias: params.restir, candidates: params.candidates, temporal: params.temporal, spatial: params.spatial };
      const next = await PtMegakernel.create(device, caps, ptSceneOptions(scene, { mode: 'realtime', integrator: { restir } }));
      next.setScene(scene.trace, scene.buffers);
      next.setLimits({ maxDepth: params.maxDepth, rrStartDepth: 0, textureConeScale: 1 });
      stats.set('kernel', `${(next.wgslBytes / 1024).toFixed(1)} KB`);
      const old = pt;
      pt = next;
      size = [0, 0];
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('This combination does not compile', error);
    }
  };
  orbit.onChange = () => view.restart();

  const r = pane.addFolder({ title: 'ReSTIR DI (experimental)' });
  r.addBinding(params, 'restir', { label: 'light sampling', options: { 'Plain NEE (one light)': 'off', 'ReSTIR, unbiased': 'unbiased', 'ReSTIR, biased': 'biased' } }).on('change', () => void rebuild());
  r.addBinding(params, 'candidates', { min: 1, max: 32, step: 1 }).on('change', (e) => e.last && void rebuild());
  r.addBinding(params, 'temporal', { label: 'temporal reuse' }).on('change', () => void rebuild());
  r.addBinding(params, 'spatial', { label: 'spatial neighbours', min: 0, max: 8, step: 1 }).on('change', (e) => e.last && void rebuild());
  const f = pane.addFolder({ title: 'Frame' });
  f.addBinding(params, 'accumulate', { label: 'average while still' }).on('change', () => {
    view.accumulate = params.accumulate;
    view.restart();
  });
  f.addBinding(params, 'spin', { label: 'orbit by itself' });
  f.addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 4, step: 1 }).on('change', () => {
    pt?.setLimits({ maxDepth: params.maxDepth, rrStartDepth: 0, textureConeScale: 1 });
    view.restart();
  });
  f.addBinding(params, 'exposure', { min: 0.1, max: 8 }).on('change', () => (view.exposure = params.exposure));

  (globalThis as { __realtime?: unknown }).__realtime = {
    get frames() { return view.rendered; },
    read: () => view.read(size[0]!, size[1]!),
    size: () => size,
  };

  await rebuild();
  ctx.loop((encoder, dt) => {
    if (!pt) return false;
    if (params.spin) {
      orbit.yaw += Math.min(dt, 50) * 0.0003;
      view.restart();
    }
    const [w, h] = ctx.filmSize();
    if (params.accumulate && view.count >= 2048 && w === size[0] && h === size[1]) return false;
    if (w !== size[0] || h !== size[1]) {
      pt.setFilm(w, h);
      pt.setCamera(orbit.camera);
      size = [w, h];
      view.restart();
    } else pt.nextFrame(orbit.camera);
    pt.encode(encoder, 1);
    view.encode(encoder, pt.frameBuffers, ctx.target());
    stats.set('film', `${w} × ${h}`);
    stats.set('averaged', params.accumulate ? `${view.count} frames` : 'off');
    return true;
  });
});
