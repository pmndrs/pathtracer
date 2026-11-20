/**
 * The scene a Ray tracer example traces: a dropdown over `SCENES`, dropped .glb files, and the orbit camera. `onScene` runs after each
 * (re)load with the new GPU scene (the example rebuilds its kernel: the scene's formats and textures are compiled in).
 */
import { buildGpuScene, type GpuScene } from '@pmndrs/raytracer';
import type { FolderApi } from 'tweakpane';
import type { ExampleContext } from './example.ts';
import { loadScene, SCENES, type LoadedScene } from './models.ts';
import { OrbitCamera } from './orbit.ts';

export interface ScenePicker {
  readonly orbit: OrbitCamera;
  current(): { scene: GpuScene; loaded: LoadedScene } | null;
}

export async function scenePicker(ctx: ExampleContext, folder: FolderApi, initial: string, onScene: (scene: GpuScene, loaded: LoadedScene) => Promise<void>, onCamera: () => void): Promise<ScenePicker> {
  // `?scene=<label>` starts on another scene (links from the docs, headless checks).
  const asked = new URLSearchParams(location.search).get('scene');
  if (asked && SCENES[asked]) initial = asked;
  const state = { scene: SCENES[initial]! };
  let current: { scene: GpuScene; loaded: LoadedScene } | null = null;
  let ticket = 0;
  const orbit = new OrbitCamera(ctx.canvas, { position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 });
  orbit.onChange = onCamera;

  const load = async (source: string | ArrayBuffer, name: string) => {
    const mine = ++ticket;
    try {
      ctx.progress(`Loading ${name}…`);
      const loaded = await loadScene(source, name);
      if (mine !== ticket) return;
      ctx.progress('Building the trees…');
      const scene = await buildGpuScene(ctx.device, loaded.input, { label: name });
      if (mine !== ticket) return void scene.destroy();
      ctx.progress('Compiling…');
      const old = current;
      current = { scene, loaded };
      orbit.set(loaded.camera, false);
      await onScene(scene, loaded);
      onCamera();
      ctx.progress(null);
      await ctx.device.queue.onSubmittedWorkDone();
      old?.scene.destroy();
    } catch (error) {
      if (mine === ticket) ctx.error(`Could not load ${name}`, error);
    }
  };

  folder.addBinding(state, 'scene', { options: SCENES }).on('change', () => void load(state.scene, Object.keys(SCENES).find((k) => SCENES[k] === state.scene) ?? 'scene'));
  folder.addButton({ title: 'Frame scene' }).on('click', () => current && orbit.frame(current.loaded.bounds.min, current.loaded.bounds.max));
  ctx.onDrop(['.glb'], (file) => void file.arrayBuffer().then((bytes) => load(bytes, file.name)));
  await load(state.scene, initial);
  return { orbit, current: () => current };
}
