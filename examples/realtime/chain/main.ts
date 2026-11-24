/**
 * The real-time chain: one path-traced sample per pixel per frame, denoised and upscaled.
 *
 *   PtMegakernel { mode: 'realtime' }      1 sample per pixel per frame at the render size (half the display by default), with the frame planes
 *   RealtimeChain (examples/shared)        pack the planes -> denoiser SVGF (`denoiser/realtime`) -> composite -> `@pmndrs/upscaler` temporal
 *   StageView                              any stage to the canvas, or the raw frame beside the final one
 *
 * The raw frame is what the path tracer gives in a frame's budget: one sample, very noisy. The denoiser averages it over time and space
 * using the planes (normals, depth, motion, albedo split off so texture detail is not blurred), and the upscaler brings it to display size.
 */
import { PtMegakernel, ptSceneOptions, type PtLookAtCamera } from '@pmndrs/pathtracer';
import type { GpuScene } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { RealtimeChain, StageView, type ChainStage, type UpscalePath } from '../../shared/realtime-chain.ts';
import { scenePicker, type ScenePicker } from '../../shared/scene-picker.ts';

type ViewMode = 'split' | ChainStage;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const params = {
    view: (search.get('view') ?? 'split') as ViewMode,
    ratio: Number(search.get('ratio') ?? 2),
    denoise: search.get('denoise') !== '0',
    upscaler: (search.get('upscaler') ?? 'temporal') as UpscalePath,
    maxDepth: 2,
    exposure: 1,
    spin: search.get('spin') === '1',
  };
  const show = new StageView(device, ctx.format);
  let pt: PtMegakernel | null = null;
  let chain: RealtimeChain | null = null;
  let scene: GpuScene | null = null;
  let camera: PtLookAtCamera | null = null;
  let chainKey = '';
  let building = false;
  let moved = true;
  let frames = 0;
  stats.set('image', 'path traced, 1 sample per pixel per frame · denoised (SVGF) · upscaled (temporal) · no raster');

  const rebuildKernel = async () => {
    if (!scene || !camera) return;
    ctx.progress('Compiling…');
    try {
      const next = await PtMegakernel.create(device, caps, ptSceneOptions(scene, { mode: 'realtime' }));
      next.setScene(scene.trace, scene.buffers);
      next.setLimits({ maxDepth: params.maxDepth, rrStartDepth: 0, textureConeScale: 1 });
      stats.set('kernel', `${(next.wgslBytes / 1024).toFixed(1)} KB`);
      const old = pt;
      pt = next;
      chainKey = '';
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('The path tracer does not compile', error);
    }
  };

  /** The chain is sized for one render and display size: rebuilt when the canvas or the ratio changes. */
  const ensureChain = async (display: [number, number]) => {
    if (!scene || !pt || !camera) return;
    const w = Math.max(8, Math.round(display[0] / params.ratio));
    const h = Math.max(8, Math.round(display[1] / params.ratio));
    // The denoiser reads view depth beyond `far` as sky: reach past the whole scene from the camera, with room to orbit out.
    const b = scene.copy.bounds;
    const reach = Math.hypot(camera.position[0] - b.center[0], camera.position[1] - b.center[1], camera.position[2] - b.center[2]) + b.radius;
    const far = Math.max(200, 4 * reach);
    const key = [w, h, ...display, far].join(',');
    if (key === chainKey || building) return;
    building = true;
    try {
      const next = await RealtimeChain.create(device, { width: w, height: h, displayWidth: display[0], displayHeight: display[1], near: 0.01, far });
      next.denoise = params.denoise;
      await next.setUpscalePath(params.upscaler);
      const old = chain;
      chain = next;
      chainKey = key;
      pt.setFilm(w, h);
      moved = true;
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('The chain failed to start', error);
    } finally {
      building = false;
    }
  };

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  // scenePicker calls back before it returns: read the orbit camera only once it has.
  let picker: ScenePicker | null = null;
  picker = await scenePicker(ctx, sceneFolder, 'Cornell box', async (next, loaded) => {
    scene = next;
    camera = loaded.camera;
    await rebuildKernel();
  }, () => {
    if (!picker) return;
    camera = picker.orbit.camera;
  });
  sceneFolder.addBinding(params, 'spin', { label: 'orbit by itself' });

  const viewFolder = pane.addFolder({ title: 'View' });
  viewFolder.addBinding(params, 'view', {
    options: { 'Raw frame | final': 'split', 'Final (denoised + upscaled)': 'final', 'Denoised (render size)': 'denoised', 'Raw frame (1 spp)': 'raw' },
  });
  viewFolder.addBinding(params, 'exposure', { min: 0.1, max: 8 }).on('change', () => (show.exposure = params.exposure));

  const chainFolder = pane.addFolder({ title: 'Chain' });
  chainFolder.addBinding(params, 'ratio', { label: 'upscale ratio', options: { '1x (no upscale)': 1, '1.5x': 1.5, '2x': 2, '3x': 3 } }).on('change', () => (chainKey = ''));
  chainFolder.addBinding(params, 'denoise', { label: 'denoiser' }).on('change', () => {
    if (chain) chain.denoise = params.denoise;
    chain?.resetHistory();
  });
  chainFolder.addBinding(params, 'upscaler', { options: { 'Temporal (accumulates frames)': 'temporal', 'Spatial (FSR1, one frame)': 'spatial', Bilinear: 'bilinear' } }).on('change', () => void chain?.setUpscalePath(params.upscaler));
  chainFolder.addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 6, step: 1 }).on('change', () => {
    pt?.setLimits({ maxDepth: params.maxDepth, rrStartDepth: 0, textureConeScale: 1 });
    chain?.resetHistory();
  });

  // Headless checks: frames drawn, the final stage's size and its linear values.
  (globalThis as { __realtime?: unknown }).__realtime = {
    get frames() { return frames; },
    size: () => (chain ? [chain.s.displayWidth, chain.s.displayHeight] : [0, 0]),
    /** The final stage (denoised + upscaled) as linear values; `stage` picks another one. */
    read: (stage: ChainStage = 'final') => chain?.read(stage) ?? Promise.resolve(new Float32Array(0)),
  };

  let last: PtLookAtCamera | null = null;
  ctx.loop((encoder, dt) => {
    if (!camera) return false;
    if (params.spin && picker) {
      picker.orbit.yaw += Math.min(dt, 50) * 0.0004;
      camera = picker.orbit.camera;
    }
    const display = ctx.filmSize();
    void ensureChain(display);
    if (!pt || !chain || chainKey === '' || chain.s.displayWidth !== display[0] || chain.s.displayHeight !== display[1]) return false;

    // One sample per pixel, traced and packed in its own submit: the denoiser and the upscaler submit theirs after it.
    const trace = device.createCommandEncoder({ label: 'chain trace' });
    if (moved || !last) {
      pt.setCamera(camera);
      pt.setPrevCamera(camera);
      chain.resetHistory();
      moved = false;
    } else pt.nextFrame(camera);
    last = camera;
    chain.beginFrame(trace);
    pt.encode(trace, 1);
    chain.accumulate(trace, pt.frameBuffers);
    chain.pack(trace);
    device.queue.submit([trace.finish()]);
    chain.post(camera, Math.min(dt, 100) / 1000);

    const target = ctx.target();
    if (params.view === 'split') {
      const half = Math.round(display[0] / 2);
      show.encode(encoder, chain.stage('raw'), target, display, { slot: 0, columns: [0, half] });
      show.encode(encoder, chain.stage('final'), target, display, { slot: 1, columns: [half, display[0]], load: true, divider: true });
    } else show.encode(encoder, chain.stage(params.view), target, display);
    frames++;
    stats.set('render', `${chain.s.width} × ${chain.s.height}`);
    stats.set('display', `${display[0]} × ${display[1]}`);
    stats.set('view', params.view === 'split' ? 'left: raw frame · right: final' : params.view);
    return true;
  });
});
