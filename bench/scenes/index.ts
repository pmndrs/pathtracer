import type { SceneDesc } from './schema.ts';
import { tier0Scenes } from './tier0/index.ts';
import { tier2Scenes } from './tier2/index.ts';

export { tier0Scenes, tier2Scenes };

/** Every bench scene (tier 0 analytic, tier 2 quality + speed). Scenes with `file` geometry need `node bench/assets/fetch.ts`. */
export const allScenes: readonly SceneDesc[] = [...tier0Scenes, ...tier2Scenes];

export function getScene(name: string): SceneDesc {
  const s = allScenes.find((x) => x.name === name);
  if (!s) throw new Error(`unknown scene '${name}' (have: ${allScenes.map((x) => x.name).join(', ')})`);
  return s;
}
