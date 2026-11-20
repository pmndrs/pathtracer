/**
 * Hello, path tracer: a Cornell box built from triangles, path traced progressively.
 *
 * The whole program is four steps: describe the scene (`cornellBox`, plain triangles and OpenPBR materials), build its GPU form
 * (`buildGpuScene`: bvh trees + the scene copy), compile a path tracer for it (`ptSceneOptions` + `PtMegakernel`, inside
 * `ProgressiveRenderer`), and each frame add a few samples and draw the running mean.
 */
import { buildGpuScene, type GpuScene } from '@pmndrs/raytracer';
import { cornellBox, CORNELL_CAMERA } from '../../shared/cornell.ts';
import { defineExample } from '../../shared/example.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

defineExample(async (ctx) => {
  const { device, pane, canvas } = ctx;
  ctx.progress('Building the scene…');
  const light = { intensity: 1, color: { r: 1, g: 0.71, b: 0.24 } };
  const lightRadiance = (): [number, number, number] => [17 * light.intensity * light.color.r, 17 * light.intensity * light.color.g, 17 * light.intensity * light.color.b];
  let scene: GpuScene = await buildGpuScene(device, cornellBox({ light: lightRadiance() }), { label: 'cornell' });

  ctx.progress('Compiling the path tracer…');
  const renderer = await ProgressiveRenderer.create(ctx, scene, CORNELL_CAMERA);
  ctx.progress(null);

  const orbit = new OrbitCamera(canvas, CORNELL_CAMERA);
  orbit.onChange = () => renderer.setCamera(orbit.camera);

  // The light's radiance lives in the scene copy's material table: a new scene with the same triangles and materials binds to the same kernel.
  const rebuild = async () => {
    const old = scene;
    scene = await buildGpuScene(device, cornellBox({ light: lightRadiance() }), { label: 'cornell' });
    renderer.setScene(scene);
    await device.queue.onSubmittedWorkDone();
    old.destroy();
  };

  const render = pane.addFolder({ title: 'Render' });
  render.addBinding(renderer.settings, 'samplesPerFrame', { label: 'samples / frame', min: 1, max: 16, step: 1 });
  render.addBinding(renderer.settings, 'maxSamples', { label: 'max samples', options: { 256: 256, 1024: 1024, 4096: 4096, 16384: 16384 } }).on('change', () => renderer.restart());
  render.addBinding(renderer.settings, 'maxDepth', { label: 'max bounces', min: 1, max: 16, step: 1 }).on('change', () => renderer.restart());
  const scale = { value: 1 };
  render.addBinding(scale, 'value', { label: 'render scale', min: 0.25, max: 1, step: 0.05 }).on('change', (e) => e.last && ctx.setRenderScale(scale.value));

  const view = pane.addFolder({ title: 'View' });
  view.addBinding(renderer.view, 'exposure', { min: 0, max: 4, step: 0.01 });
  view.addBinding(renderer.view, 'toneMapping', { label: 'tone mapping', options: { ACES: 'aces', Reinhard: 'reinhard', 'None (clamp)': 'clamp' } });

  const lightFolder = pane.addFolder({ title: 'Light' });
  lightFolder.addBinding(light, 'intensity', { min: 0, max: 4, step: 0.05 }).on('change', (e) => e.last && void rebuild());
  lightFolder.addBinding(light, 'color', { color: { type: 'float' } }).on('change', (e) => e.last && void rebuild());

  pane.addButton({ title: 'Reset camera' }).on('click', () => orbit.set(CORNELL_CAMERA));

  ctx.loop((encoder) => renderer.frame(encoder));
});
