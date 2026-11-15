/**
 * Node-side `SceneDesc` of a glTF model read from the downloaded file: materials, textures and the default material come from
 * `loadGltfScene`, the rest (camera, lights, film, integrator, overrides) from the sidecar (`GltfSceneBuild`, see `gltf-desc.ts`).
 * The static Tier 1 scenes (`tier1/gltf-scenes.ts`) take the material table from the committed goldens instead; a test checks both agree.
 */
import { resolve } from 'node:path';
import { ASSETS_DIR } from './assets-dir.ts';
import { buildGltfSceneDesc } from './gltf-desc.ts';
import type { GltfModelRef, GltfSceneBuild } from './gltf-desc.ts';
import { gltfTextureDescs, loadGltfScene } from './gltf-scene.ts';
import type { GltfScene } from './gltf-scene.ts';
import type { SceneDesc } from './schema.ts';

export type GltfSidecar = Omit<GltfSceneBuild, 'model' | 'materials' | 'textures' | 'defaultMaterial'> & {
  /** Add a mesh for primitives without a glTF material when the model has any (default false: they are dropped, e.g. unlabelled text). */
  includeDefaultMaterial?: boolean;
};

export function gltfSceneDescFromFile(model: GltfModelRef, sidecar: GltfSidecar, assetsDir: string = ASSETS_DIR): { scene: SceneDesc; loaded: GltfScene } {
  const loaded = loadGltfScene(resolve(assetsDir, model.path), { sha256: model.sha256, normals: sidecar.normals ?? 'geometric', uvs: true });
  const { includeDefaultMaterial, ...rest } = sidecar;
  const scene = buildGltfSceneDesc({
    ...rest,
    model,
    materials: loaded.materials.map((m) => ({ index: m.index, ...(m.name !== undefined ? { name: m.name } : {}), material: m.material })),
    ...(includeDefaultMaterial && loaded.defaultMaterial ? { defaultMaterial: loaded.defaultMaterial.material } : {}),
    textures: gltfTextureDescs(loaded, model.path, model.sha256),
  });
  return { scene, loaded };
}
