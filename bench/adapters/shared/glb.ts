/**
 * The scene's glTF, as written by the shared exporter (bench/exporters/gltf.ts). The exporter runs on Node only (it uses
 * Buffer and, for `file` meshes, the filesystem), so `node bench/adapters/export-scenes.ts` writes
 * bench/assets/generated/web/<scene>.glb plus a <scene>.json sidecar (scene hash, exporter approximations), and the
 * page fetches them. The hash of the SceneDesc JSON is checked so an edited scene cannot silently render stale geometry.
 */
import type { SceneDesc } from '../../scenes/schema.ts';

export const WEB_EXPORT_DIR = '/bench/assets/generated/web';

export interface SceneGlb {
  glb: ArrayBuffer;
  /** What glTF could not express exactly (from the exporter). */
  approximations: string[];
}

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export async function fetchSceneGlb(scene: SceneDesc): Promise<SceneGlb> {
  const hint = 'Run: node bench/adapters/export-scenes.ts';
  const side = await fetch(`${WEB_EXPORT_DIR}/${scene.name}.json`);
  if (!side.ok) throw new Error(`no exported glTF for scene "${scene.name}" (HTTP ${side.status}). ${hint}`);
  const meta = (await side.json()) as { sceneHash: string; approximations: string[] };
  const hash = await sha256Hex(JSON.stringify(scene));
  if (hash !== meta.sceneHash) throw new Error(`exported glTF for "${scene.name}" is stale (scene definition changed). ${hint}`);
  const glb = await fetch(`${WEB_EXPORT_DIR}/${scene.name}.glb`);
  if (!glb.ok) throw new Error(`no exported glb for scene "${scene.name}" (HTTP ${glb.status}). ${hint}`);
  return { glb: await glb.arrayBuffer(), approximations: meta.approximations };
}
