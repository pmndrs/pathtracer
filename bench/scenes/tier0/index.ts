import type { SceneDesc } from '../schema.ts';
import { cornell } from './cornell.ts';
import { envSphere } from './env-sphere.ts';
import { furnaceConductor, furnaceDielectric, furnaceDiffuse } from './furnace.ts';
import { quadLight } from './quad-light.ts';

export { cornell, envSphere, furnaceConductor, furnaceDielectric, furnaceDiffuse, quadLight };

export const tier0Scenes: readonly SceneDesc[] = [furnaceDiffuse, furnaceDielectric, furnaceConductor, cornell, quadLight, envSphere];

export function getTier0Scene(name: string): SceneDesc {
  const s = tier0Scenes.find((x) => x.name === name);
  if (!s) throw new Error(`unknown tier0 scene '${name}' (have: ${tier0Scenes.map((x) => x.name).join(', ')})`);
  return s;
}
