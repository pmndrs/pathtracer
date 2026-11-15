import type { SceneDesc } from './schema.ts';
import { tier0Scenes } from './tier0/index.ts';
import { tier2Scenes } from './tier2/index.ts';
import { abScenes as lightScenes } from './ab/lights-mix.ts';
import { openpbrCheckScenes } from './ab/openpbr-check.ts';
import { foliageScenes } from './ab/foliage-mask.ts';
import { lodCornell } from './ab/lod-cornell.ts';
import { manyLightsScenes } from './ab/many-lights.ts';
import { tier1Scenes } from './tier1/textured.ts';
import { tier1GltfSceneDescs } from './tier1/gltf-scenes.ts';
import { tier1TexturedGltfSceneDescs } from './tier1/gltf-scenes-tex.ts';
import { ablateScene, parseAblationName } from './tier1/ablations.ts';

/** Experiment scenes without pbrt/Mitsuba references: light selection A/B (P1.4f) and the analytic `openpbr` checks (P2.21, P2.22). */
export const abScenes: readonly SceneDesc[] = [...lightScenes, ...openpbrCheckScenes, ...foliageScenes, lodCornell, ...manyLightsScenes];
export { tier0Scenes, tier1Scenes, tier1GltfSceneDescs, tier1TexturedGltfSceneDescs, tier2Scenes, openpbrCheckScenes };

/** Every bench scene (tier 0 analytic, tier 2 quality + speed, `ab` experiment scenes without references). Scenes with `file` geometry need `node bench/assets/fetch.ts`. */
export const allScenes: readonly SceneDesc[] = [...tier0Scenes, ...tier1Scenes, ...tier1GltfSceneDescs, ...tier1TexturedGltfSceneDescs, ...tier2Scenes, ...abScenes];

/** A scene by name; `<scene>--no-<feature>` is the scene with that feature stripped (Tier 1 ablations, `tier1/ablations.ts`). */
export function getScene(name: string): SceneDesc {
  const s = allScenes.find((x) => x.name === name);
  if (!s) {
    const ab = parseAblationName(name);
    if (ab) return ablateScene(getScene(ab.base), ab.tag);
  }
  if (!s) throw new Error(`unknown scene '${name}' (have: ${allScenes.map((x) => x.name).join(', ')})`);
  return s;
}
