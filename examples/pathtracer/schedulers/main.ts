/**
 * Megakernel or wavefront: the same estimator, scheduled two ways.
 *
 * `PtMegakernel` traces a whole path per thread in one kernel: simple, and fast while paths are short and the kernel is small.
 * `PtWavefront` splits a sample into kernels (camera, intersect, shade, shadow) connected by GPU queues and indirect dispatch: each kernel
 * is small, threads stay busy on long paths, and `refill` starts a pixel's next sample as soon as its path ends. The images are the same
 * per pixel (same random numbers per pixel and vertex); only the time differs. `ptwfAutoPolicy` picks one for a film, path length and device.
 */
import { ptMegakernelWgsl, ptSceneOptions, ptwfAutoPolicy, type PtLookAtCamera } from '@pmndrs/pathtracer';
import type { GpuScene } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { ProgressiveRenderer, type SchedulerChoice } from '../../shared/progressive.ts';
import { scenePicker, type ScenePicker } from '../../shared/scene-picker.ts';

defineExample(async (ctx) => {
  const { pane, stats, caps } = ctx;
  const params = {
    scheduler: (new URLSearchParams(location.search).get('scheduler') ?? 'megakernel') as 'megakernel' | 'wavefront',
    refill: 'off' as 'off' | 'batch' | 'ordered' | 'inline',
    sort: 'none' as 'none' | 'material' | 'octant' | 'both',
    appends: 'auto' as 'atomic' | 'subgroup' | 'auto',
    samplesPerFrame: 4,
    maxDepth: 8,
    suggestion: '',
  };
  let renderer: ProgressiveRenderer | null = null;
  let scene: GpuScene | null = null;
  let camera: PtLookAtCamera = { position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 };

  const choice = (): SchedulerChoice =>
    params.scheduler === 'megakernel' ? {} : { scheduler: 'wavefront', refill: params.refill === 'off' ? false : params.refill, sort: params.sort, appends: params.appends };

  const suggest = () => {
    if (!scene) return;
    const [width, height] = ctx.filmSize();
    const kernelBytes = ptMegakernelWgsl(caps, ptSceneOptions(scene)).length;
    const pick = ptwfAutoPolicy({ width, height, maxDepth: params.maxDepth, subgroups: caps.subgroups, kernelBytes });
    params.suggestion = `${pick.scheduler}${pick.scheduler === 'wavefront' && pick.refill ? `, refill ${pick.refill}` : ''}`;
    stats.set('auto policy', params.suggestion);
    console.info(`ptwfAutoPolicy: ${pick.reason}`);
  };

  /** Compiles the chosen scheduler for the current scene. */
  const rebuild = async () => {
    if (!scene) return;
    ctx.progress(`Compiling the ${params.scheduler}…`);
    const next = await ProgressiveRenderer.create(ctx, scene, camera, {}, choice());
    next.settings.samplesPerFrame = params.samplesPerFrame;
    next.settings.maxDepth = params.maxDepth;
    const old = renderer;
    renderer = next;
    ctx.progress(null);
    suggest();
    await ctx.device.queue.onSubmittedWorkDone();
    old?.destroy();
  };

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  // scenePicker calls back before it returns: keep the orbit camera through the callback's own reference.
  let picker: ScenePicker | null = null;
  picker = await scenePicker(ctx, sceneFolder, 'Toy Car', async (next, loaded) => {
    scene = next;
    camera = loaded.camera;
    await rebuild();
  }, () => {
    if (!picker) return;
    camera = picker.orbit.camera;
    renderer?.setCamera(camera);
  });

  const sched = pane.addFolder({ title: 'Scheduler' });
  sched.addBinding(params, 'scheduler', { options: { Megakernel: 'megakernel', Wavefront: 'wavefront' } }).on('change', () => void rebuild());
  sched.addBinding(params, 'refill', { label: 'refill (wavefront)', options: { Off: 'off', 'Batch (pixel order)': 'batch', Ordered: 'ordered', Inline: 'inline' } }).on('change', () => params.scheduler === 'wavefront' && void rebuild());
  sched.addBinding(params, 'sort', { label: 'sort (wavefront)', options: { None: 'none', 'By material': 'material', 'By ray octant': 'octant', Both: 'both' } }).on('change', () => params.scheduler === 'wavefront' && void rebuild());
  sched.addBinding(params, 'appends', { label: 'appends (wavefront)', options: { Auto: 'auto', Atomic: 'atomic', Subgroup: 'subgroup' } }).on('change', () => params.scheduler === 'wavefront' && void rebuild());

  const render = pane.addFolder({ title: 'Render' });
  render.addBinding(params, 'samplesPerFrame', { label: 'samples / frame', min: 1, max: 16, step: 1 }).on('change', () => renderer && (renderer.settings.samplesPerFrame = params.samplesPerFrame));
  render.addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 32, step: 1 }).on('change', () => {
    if (!renderer) return;
    renderer.settings.maxDepth = params.maxDepth;
    renderer.restart();
    suggest();
  });

  ctx.onResize(suggest);
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
