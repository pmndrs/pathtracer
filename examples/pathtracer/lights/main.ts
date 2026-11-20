/**
 * Many lights: 64 small coloured lamps over a floor with a few boxes, under a dim sky and a low sun.
 *
 * Next-event estimation picks ONE light per vertex and traces one shadow ray to it, so which light it picks decides the noise. The scene
 * copy's light table holds a pick probability per light (`lightSelection`, a scene-copy build option):
 *   pbrt (default)  pbrt-v4's rule: infinite lights (sun, sky) by power share against the rest, emitters by power within
 *   power           every light by its power
 *   kind            the same chance for each kind (emitters, sun, sky), by power within a kind
 *   capped 25%      by power, no light kind above a quarter
 * ReSTIR DI (experimental) instead draws several candidates per pixel, keeps one by its contribution, and reuses good picks from the
 * previous frame and from neighbours: far less noise per sample with many lights, at the cost of extra passes.
 */
import { buildGpuScene, type GpuScene, type LightSelection } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { LAMPS_CAMERA, lampsScene } from '../../shared/lamps.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

const STRATEGIES: Record<string, LightSelection> = { 'pbrt (default)': 'pbrt', power: 'power', kind: 'kind', 'capped 25%': { capped: 0.25 } };

defineExample(async (ctx) => {
  const { device, pane, canvas, stats } = ctx;
  const params = { strategy: 'pbrt (default)', restir: new URLSearchParams(location.search).has('restir'), maxDepth: 4 };
  const orbit = new OrbitCamera(canvas, LAMPS_CAMERA);
  let scene: GpuScene = await buildGpuScene(device, lampsScene(STRATEGIES[params.strategy]!), { label: 'lamps' });
  let renderer: ProgressiveRenderer | null = null;
  stats.set('lights', '64 lamps, sun, sky');

  const rebuild = async () => {
    ctx.progress('Compiling…');
    try {
      const next = await ProgressiveRenderer.create(ctx, scene, orbit.camera, { integrator: { restir: params.restir ? {} : null } });
      next.settings.maxDepth = params.maxDepth;
      next.settings.samplesPerFrame = 1;
      const old = renderer;
      renderer = next;
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('This combination does not compile', error);
    }
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  const sampling = pane.addFolder({ title: 'Light sampling' });
  sampling.addBinding(params, 'strategy', { label: 'light selection', options: Object.fromEntries(Object.keys(STRATEGIES).map((k) => [k, k])) }).on('change', async () => {
    // The selection lives in the scene copy's light table: rebuild the scene, keep the kernel.
    const old = scene;
    scene = await buildGpuScene(device, lampsScene(STRATEGIES[params.strategy]!), { label: 'lamps' });
    renderer?.setScene(scene);
    await device.queue.onSubmittedWorkDone();
    old.destroy();
  });
  sampling.addBinding(params, 'restir', { label: 'ReSTIR DI (experimental)' }).on('change', () => void rebuild());
  pane.addFolder({ title: 'Render' }).addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 8, step: 1 }).on('change', () => {
    if (!renderer) return;
    renderer.settings.maxDepth = params.maxDepth;
    renderer.restart();
  });

  await rebuild();
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
