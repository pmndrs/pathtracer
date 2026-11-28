/**
 * glTF models: `loadGltf` -> `buildGpuScene` -> a kernel compiled for the model's materials (`ptSceneOptions`).
 *
 * The model's materials become OpenPBR parameters (core metallic-roughness plus the KHR_materials_* extensions the scene copy has),
 * its textures are decoded in the page, and its first directional light (if any) becomes the sun. A glTF has no sky: this page adds a
 * uniform one and an optional sun. Changing the light rebuilds the scene copy; changing the model compiles a new kernel.
 */
import { buildGpuScene, loadGltf, type GltfSceneInput, type GpuScene, type SunInput } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

/** Khronos glTF-Sample-Assets at a pinned commit (each model has its own licence; see the repository's README per model). */
const ASSETS = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/';
const MODELS: Record<string, string> = {
  'Damaged Helmet': `${ASSETS}DamagedHelmet/glTF-Binary/DamagedHelmet.glb`,
  'Sheen Chair': `${ASSETS}SheenChair/glTF-Binary/SheenChair.glb`,
  'Toy Car': `${ASSETS}ToyCar/glTF-Binary/ToyCar.glb`,
  'Clear Coat Test': `${ASSETS}ClearCoatTest/glTF-Binary/ClearCoatTest.glb`,
  'Metal Rough Spheres': `${ASSETS}MetalRoughSpheres/glTF-Binary/MetalRoughSpheres.glb`,
  'Dragon Attenuation': `${ASSETS}DragonAttenuation/glTF-Binary/DragonAttenuation.glb`,
};

defineExample(async (ctx) => {
  const { device, pane, canvas, stats } = ctx;
  const params = {
    model: MODELS['Damaged Helmet']!,
    sky: { r: 0.8, g: 0.9, b: 1 },
    skyIntensity: 1,
    sun: true,
    sunElevation: 50,
    sunAzimuth: 30,
    sunIntensity: 3,
    fileCamera: true,
  };

  let input: GltfSceneInput | null = null;
  let scene: GpuScene | null = null;
  let renderer: ProgressiveRenderer | null = null;
  let orbit: OrbitCamera | null = null;
  let loading = 0;

  const sunFromParams = (): SunInput | null => {
    if (!params.sun) return null;
    const el = (params.sunElevation * Math.PI) / 180, az = (params.sunAzimuth * Math.PI) / 180;
    // `direction` is the way the light travels: from the sun towards the ground.
    return { direction: [-Math.cos(el) * Math.sin(az), -Math.sin(el), -Math.cos(el) * Math.cos(az)], irradiance: [params.sunIntensity, params.sunIntensity * 0.95, params.sunIntensity * 0.88], angularRadius: 0.00465 };
  };
  const applyLights = (target: GltfSceneInput) => {
    target.environment = { radiance: [params.sky.r * params.skyIntensity, params.sky.g * params.skyIntensity, params.sky.b * params.skyIntensity] };
    target.sun = sunFromParams();
  };

  /** Lights live in the scene copy: rebuild the scene and bind it to the same kernel (the materials did not change). */
  const rebuildLights = async () => {
    if (!input || !renderer || !scene) return;
    applyLights(input);
    const old = scene;
    scene = await buildGpuScene(device, input, { label: 'model' });
    renderer.setScene(scene);
    await device.queue.onSubmittedWorkDone();
    old.destroy();
  };

  const load = async (source: string | ArrayBuffer, name: string) => {
    const ticket = ++loading;
    try {
      ctx.progress(`Loading ${name}…`);
      const next = await loadGltf(source);
      if (ticket !== loading) return;
      applyLights(next);
      ctx.progress('Building the trees…');
      const nextScene = await buildGpuScene(device, next, { label: 'model' });
      ctx.progress('Compiling a kernel for these materials…');
      const camera = (params.fileCamera && next.cameras[0]?.camera) || { position: [0, 0, 3] as [number, number, number], target: [0, 0, 0] as [number, number, number], up: [0, 1, 0] as [number, number, number], yFovDeg: 40 };
      const nextRenderer = await ProgressiveRenderer.create(ctx, nextScene, camera);
      if (ticket !== loading) {
        nextRenderer.destroy();
        nextScene.destroy();
        return;
      }
      const oldRenderer = renderer, oldScene = scene;
      input = next;
      scene = nextScene;
      renderer = nextRenderer;
      sync();
      orbit?.dispose();
      // The file camera's target is only a point one unit ahead of it: the orbit pivots on the model (same picture, same direction).
      orbit = new OrbitCamera(canvas, camera, next.bounds);
      orbit.onChange = () => renderer?.setCamera(orbit!.camera);
      renderer.setCamera(orbit.camera);
      if (!(params.fileCamera && next.cameras[0])) orbit.frame(next.bounds.min, next.bounds.max);
      stats.set('triangles', next.gltf.stats.triangles.toLocaleString());
      stats.set('materials', next.materials.length);
      stats.set('textures', next.textures?.length ?? 0);
      if (next.notes.length) console.info(`glTF notes for ${name}:\n${next.notes.join('\n')}`);
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      oldRenderer?.destroy();
      oldScene?.destroy();
    } catch (error) {
      if (ticket === loading) ctx.error(`Could not load ${name}`, error);
    }
  };

  const model = pane.addFolder({ title: 'Model' });
  model.addBinding(params, 'model', { options: MODELS }).on('change', () => void load(params.model, Object.keys(MODELS).find((k) => MODELS[k] === params.model) ?? 'model'));
  model.addBinding(params, 'fileCamera', { label: 'file camera' });

  const lights = pane.addFolder({ title: 'Light' });
  const relight = (e: { last: boolean }) => e.last && void rebuildLights();
  lights.addBinding(params, 'sky', { color: { type: 'float' } }).on('change', relight);
  lights.addBinding(params, 'skyIntensity', { label: 'sky', min: 0, max: 4, step: 0.05 }).on('change', relight);
  lights.addBinding(params, 'sun').on('change', relight);
  lights.addBinding(params, 'sunElevation', { label: 'elevation', min: 1, max: 90, step: 1 }).on('change', relight);
  lights.addBinding(params, 'sunAzimuth', { label: 'azimuth', min: -180, max: 180, step: 1 }).on('change', relight);
  lights.addBinding(params, 'sunIntensity', { label: 'sun', min: 0, max: 10, step: 0.1 }).on('change', relight);

  const settings = { samplesPerFrame: 2, maxSamples: 4096, maxDepth: 8, exposure: 1, toneMapping: 'aces' as const, scale: 1 };
  const sync = () => {
    if (!renderer) return;
    Object.assign(renderer.settings, { samplesPerFrame: settings.samplesPerFrame, maxSamples: settings.maxSamples, maxDepth: settings.maxDepth });
    Object.assign(renderer.view, { exposure: settings.exposure, toneMapping: settings.toneMapping });
  };
  const render = pane.addFolder({ title: 'Render' });
  render.addBinding(settings, 'samplesPerFrame', { label: 'samples / frame', min: 1, max: 16, step: 1 }).on('change', sync);
  render.addBinding(settings, 'maxSamples', { label: 'max samples', options: { 256: 256, 1024: 1024, 4096: 4096, 16384: 16384 } }).on('change', () => (sync(), renderer?.restart()));
  render.addBinding(settings, 'maxDepth', { label: 'max bounces', min: 1, max: 16, step: 1 }).on('change', () => (sync(), renderer?.restart()));
  render.addBinding(settings, 'scale', { label: 'render scale', min: 0.25, max: 1, step: 0.05 }).on('change', (e) => e.last && ctx.setRenderScale(settings.scale));
  const view = pane.addFolder({ title: 'View' });
  view.addBinding(settings, 'exposure', { min: 0, max: 4, step: 0.01 }).on('change', sync);
  view.addBinding(settings, 'toneMapping', { label: 'tone mapping', options: { ACES: 'aces', Reinhard: 'reinhard', 'None (clamp)': 'clamp' } }).on('change', sync);
  pane.addButton({ title: 'Frame model' }).on('click', () => input && orbit?.frame(input.bounds.min, input.bounds.max));

  ctx.onDrop(['.glb'], (file) => void file.arrayBuffer().then((bytes) => load(bytes, file.name)));
  await load(params.model, 'Damaged Helmet');
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
