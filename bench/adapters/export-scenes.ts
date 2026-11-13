/**
 * Node side of the web adapters: writes each scene's glTF (through bench/exporters/gltf.ts) where the browser can fetch it.
 *
 *   node bench/adapters/export-scenes.ts [scene ...]      (default: every Tier 0 scene)
 *
 * Output: bench/assets/generated/web/<scene>.glb and <scene>.json ({ sceneHash, approximations }). Git-ignored.
 * Also makes sure the procedural environment PFM exists.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toGltf } from '../exporters/gltf.ts';
import { ASSETS_DIR, generateAssets } from '../scenes/tier0/generate-assets.ts';
import { ENV_FILE } from '../scenes/tier0/env-sky.ts';
import { tier0Scenes } from '../scenes/tier0/index.ts';
import type { SceneDesc } from '../scenes/schema.ts';

export const WEB_DIR = join(ASSETS_DIR, 'generated', 'web');

export function exportScene(scene: SceneDesc): string {
  const out = toGltf(scene, { assetsDir: ASSETS_DIR, spp: 0 });
  const file = out.files[0];
  if (!file || typeof file.data === 'string') throw new Error(`glTF exporter returned no binary file for ${scene.name}`);
  mkdirSync(WEB_DIR, { recursive: true });
  writeFileSync(join(WEB_DIR, `${scene.name}.glb`), file.data);
  const sceneHash = createHash('sha256').update(JSON.stringify(scene)).digest('hex');
  writeFileSync(join(WEB_DIR, `${scene.name}.json`), `${JSON.stringify({ scene: scene.name, sceneHash, approximations: out.approximations }, null, 2)}\n`);
  return join(WEB_DIR, `${scene.name}.glb`);
}

export function exportScenes(names: readonly string[] = []): string[] {
  if (!existsSync(join(ASSETS_DIR, ENV_FILE))) generateAssets();
  const scenes = names.length > 0 ? names.map((n) => tier0Scenes.find((s) => s.name === n) ?? failUnknown(n)) : tier0Scenes;
  return scenes.map(exportScene);
}

function failUnknown(name: string): never {
  throw new Error(`unknown tier 0 scene "${name}" (have: ${tier0Scenes.map((s) => s.name).join(', ')})`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  for (const path of exportScenes(process.argv.slice(2))) console.log('wrote', path);
}
