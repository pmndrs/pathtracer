/**
 * Node side of the web adapters: writes each scene's glTF (through bench/exporters/gltf.ts) where the browser can fetch it.
 *
 *   node bench/adapters/export-scenes.ts [scene ...]      (default: every bench scene, Tier 0 and Tier 2; `file` meshes need `node bench/assets/fetch.ts`)
 *
 * `<scene>--no-<feature>` names (Tier 1 ablation variants, bench/scenes/tier1/ablations.ts) resolve through `getScene`.
 * Output: bench/assets/generated/web/<scene>.glb and <scene>.json ({ sceneHash, approximations }). Git-ignored.
 * Also makes sure the procedural environment PFM exists.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { generateTexture } from '../scenes/textures.ts';
import '../scenes/gltf-images.ts'; // registers the image decoder
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toGltf } from '../exporters/gltf.ts';
import { ASSETS_DIR, generateAssets } from '../scenes/tier0/generate-assets.ts';
import { ENV_FILE } from '../scenes/tier0/env-sky.ts';
import { allScenes, getScene } from '../scenes/index.ts';
import type { SceneDesc } from '../scenes/schema.ts';

export const WEB_DIR = join(ASSETS_DIR, 'generated', 'web');

export function exportScene(scene: SceneDesc): string {
  // `openpbr` materials are written as core + KHR_materials_* (bench/exporters/gltf-openpbr.ts, P2.31); our own pt-megakernel reads only the triangles.
  const out = toGltf(scene, { assetsDir: ASSETS_DIR, spp: 0 });
  const file = out.files[0];
  if (!file || typeof file.data === 'string') throw new Error(`glTF exporter returned no binary file for ${scene.name}`);
  mkdirSync(WEB_DIR, { recursive: true });
  writeFileSync(join(WEB_DIR, `${scene.name}.glb`), file.data);
  const sceneHash = createHash('sha256').update(JSON.stringify(scene)).digest('hex');
  // glTF image textures: decoded here (PNG/JPEG, the same decoder the native exporters use) and written as raw RGBA8, so the browser reads exactly the texels the references read
  const textures: { key: string; file: string; width: number; height: number }[] = [];
  for (const desc of Object.values(scene.textures ?? {})) {
    if (desc.generator !== 'image') continue;
    const key = `${desc.path}#${desc.image}`;
    if (textures.some((t) => t.key === key)) continue;
    const t = generateTexture(desc);
    const file = `${scene.name}.tex${textures.length}.rgba`;
    writeFileSync(join(WEB_DIR, file), t.data);
    textures.push({ key, file, width: t.width, height: t.height });
  }
  writeFileSync(join(WEB_DIR, `${scene.name}.json`), `${JSON.stringify({ scene: scene.name, sceneHash, approximations: out.approximations, ...(textures.length ? { textures } : {}) }, null, 2)}\n`);
  return join(WEB_DIR, `${scene.name}.glb`);
}

export function exportScenes(names: readonly string[] = []): string[] {
  if (!existsSync(join(ASSETS_DIR, ENV_FILE))) generateAssets();
  const scenes = names.length > 0 ? names.map((n) => getScene(n)) : allScenes;
  return scenes.map(exportScene);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  for (const path of exportScenes(process.argv.slice(2))) console.log('wrote', path);
}
