import { benchAssets } from '../site-paths.ts';
import type { SceneDesc } from './schema.ts';

/** IDs match the prepared GLB, texture sidecars and scene hash used by the hero. */
export const interiorNames = ['hero-classroom', 'hero-sponza'] as const;
export type InteriorName = typeof interiorNames[number];
export function isInterior(name: string): name is InteriorName {
  return interiorNames.some(id => id === name);
}

export function interiorPath(name: InteriorName): string {
  return `generated/web/${name}.scene.json`;
}

export function missingInterior(name: InteriorName): string {
  return `Prepare ${name} from the pathtracer root with python3 bench/hero/prepare.py ${name.slice(5)}.`;
}

export function parseInterior(name: InteriorName, json: string): SceneDesc {
  const scene: SceneDesc = JSON.parse(json);
  if (scene.name !== name) throw new Error(`Expected ${name} in ${interiorPath(name)}, found ${scene.name}.`);
  return scene;
}

export async function loadInterior(name: InteriorName): Promise<SceneDesc> {
  const response = await fetch(benchAssets + '/' + interiorPath(name));
  if (!response.ok) throw new Error(missingInterior(name));
  return parseInterior(name, await response.text());
}
