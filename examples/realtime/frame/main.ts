/**
 * Real-time frames: `mode: 'realtime'` renders ONE sample per pixel per frame, with the outputs a denoiser and an upscaler need.
 *
 *   PtMegakernel.create(..., { mode: 'realtime' })   frame planes, blue-noise sampler (the texture ships in @pmndrs/raytracer), maxDepth 2,
 *                                                     reduced materials after the first bounce, cheap cutout shadows
 *   pt.nextFrame(camera); pt.encode(encoder, 1)       one frame; the previous camera gives the motion vectors
 *   pt.frameBuffers                                   the beauty and 7 planes on the GPU (demodulated diffuse and specular, albedo,
 *                                                     specular factor, normal + view depth, motion + roughness, emission)
 *
 * This page shows the planes as they are, or their average while the camera is still (where a temporal filter would take them). Blue
 * noise puts a single frame's error at high frequencies, which is what a spatial filter removes best; compare it with Owen-Sobol.
 */
import { PtMegakernel, ptSceneOptions, type PtLookAtCamera, type PtSamplerKind } from '@pmndrs/pathtracer';
import type { GpuScene } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { FRAME_VIEWS, FrameView, type FrameViewMode } from '../../shared/frame-view.ts';
import { scenePicker, type ScenePicker } from '../../shared/scene-picker.ts';

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const params = {
    view: Number(search.get('view') ?? 0) as FrameViewMode,
    sampler: (search.get('sampler') ?? 'bluenoise') as PtSamplerKind,
    accumulate: search.get('accumulate') !== '0',
    spin: search.get('spin') === '1',
    maxDepth: 2,
    exposure: 1,
    scale: 0.5,
  };
  const view = new FrameView(device, ctx.format);
  view.mode = params.view;
  view.accumulate = params.accumulate;
  let pt: PtMegakernel | null = null;
  let scene: GpuScene | null = null;
  let camera: PtLookAtCamera | null = null;
  let size = [0, 0];
  stats.set('image', 'path traced, 1 sample per pixel per frame · no raster · no denoiser · no upscaler');
  // Real-time paths render below display resolution (an upscaler brings them back); the planes are 7 x 16 B per pixel.
  ctx.setRenderScale(params.scale);

  const rebuild = async () => {
    if (!scene || !camera) return;
    ctx.progress('Compiling…');
    try {
      const next = await PtMegakernel.create(device, caps, ptSceneOptions(scene, { mode: 'realtime', integrator: { sampler: params.sampler } }));
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

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  // scenePicker calls back before it returns: read the orbit camera only once it has.
  let picker: ScenePicker | null = null;
  picker = await scenePicker(ctx, sceneFolder, 'Damaged Helmet', async (next, loaded) => {
    scene = next;
    camera = loaded.camera;
    await rebuild();
  }, () => {
    if (!picker) return;
    camera = picker.orbit.camera;
    view.restart();
  });
  sceneFolder.addBinding(params, 'spin', { label: 'orbit by itself' });

  const frameFolder = pane.addFolder({ title: 'Frame' });
  frameFolder.addBinding(params, 'view', { options: FRAME_VIEWS }).on('change', () => {
    view.mode = params.view;
    view.restart();
  });
  frameFolder.addBinding(params, 'accumulate', { label: 'average while still' }).on('change', () => {
    view.accumulate = params.accumulate;
    view.restart();
  });
  frameFolder.addBinding(params, 'exposure', { min: 0.1, max: 8 }).on('change', () => (view.exposure = params.exposure));
  const render = pane.addFolder({ title: 'Path tracer' });
  render.addBinding(params, 'sampler', { options: { 'Blue noise': 'bluenoise', 'Owen-Sobol': 'owen-sobol', PCG: 'pcg' } }).on('change', () => void rebuild().then(() => view.restart()));
  render.addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 6, step: 1 }).on('change', () => {
    pt?.setLimits({ maxDepth: params.maxDepth, rrStartDepth: 0, textureConeScale: 1 });
    view.restart();
  });
  render.addBinding(params, 'scale', { label: 'render scale', min: 0.25, max: 1, step: 0.25 }).on('change', () => ctx.setRenderScale(params.scale));

  // Headless checks: the averaged view as linear values, and the frame count.
  (globalThis as { __realtime?: unknown }).__realtime = {
    get frames() { return view.rendered; },
    read: () => view.read(size[0]!, size[1]!),
    size: () => size,
  };

  ctx.loop((encoder, dt) => {
    if (!pt || !camera) return false;
    if (params.spin && picker) {
      picker.orbit.yaw += Math.min(dt, 50) * 0.0004;
      camera = picker.orbit.camera;
      view.restart();
    }
    const [w, h] = ctx.filmSize();
    // A still camera averaged this long has converged as far as the eye can tell: let the GPU idle.
    if (params.accumulate && view.count >= 2048 && w === size[0] && h === size[1]) return false;
    if (w !== size[0] || h !== size[1]) {
      pt.setFilm(w, h);
      pt.setCamera(camera);
      size = [w, h];
      view.restart();
    } else pt.nextFrame(camera);
    pt.encode(encoder, 1);
    view.encode(encoder, pt.frameBuffers, ctx.target());
    stats.set('film', `${w} × ${h}`);
    stats.set('averaged', params.accumulate ? `${view.count} frames` : 'off');
    return true;
  });
});
