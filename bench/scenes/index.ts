import type { SceneDesc } from './schema.ts';
import { tier0Scenes } from './tier0/index.ts';
import { tier2Scenes } from './tier2/index.ts';
import { abScenes } from './ab/lights-mix.ts';
import { tier1Scenes } from './tier1/textured.ts';

export { tier0Scenes, tier1Scenes, tier2Scenes, abScenes };

/** Every bench scene (tier 0 analytic, tier 2 quality + speed, `ab` experiment scenes without references). Scenes with `file` geometry need `node bench/assets/fetch.ts`. */
export const allScenes: readonly SceneDesc[] = [...tier0Scenes, ...tier1Scenes, ...tier2Scenes, ...abScenes];

export function getScene(name: string): SceneDesc {
  const s = allScenes.find((x) => x.name === name);
  if (!s) throw new Error(`unknown scene '${name}' (have: ${allScenes.map((x) => x.name).join(', ')})`);
  return s;
}
