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

/** Minimal GLB reader for the bench's own export (bench/exporters/gltf.ts): float VEC3 POSITION/NORMAL, u32 indices. */
export function readGlbMeshes(glb: ArrayBuffer): { positions: Float32Array; normals: Float32Array; indices: Uint32Array }[] {
  const view = new DataView(glb);
  if (view.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB');
  let offset = 12;
  let json: { accessors: { bufferView: number; byteOffset?: number; componentType: number; count: number; type: string }[]; bufferViews: { byteOffset?: number; byteLength: number; byteStride?: number }[]; meshes: { primitives: { attributes: Record<string, number>; indices?: number }[] }[]; nodes: { mesh?: number; matrix?: number[] }[] } | null = null;
  let bin: Uint8Array | null = null;
  while (offset + 8 <= glb.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const data = new Uint8Array(glb, offset + 8, length);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(data));
    else if (type === 0x004e4942) bin = data;
    offset += 8 + length;
  }
  if (!json || !bin) throw new Error('GLB without JSON or BIN chunk');
  const binary = bin;
  const doc = json;
  const read = (index: number, kind: 'f32' | 'u32') => {
    const accessor = doc.accessors[index];
    if (!accessor) throw new Error(`accessor ${index} missing`);
    const bufferView = doc.bufferViews[accessor.bufferView];
    if (!bufferView || bufferView.byteStride) throw new Error('strided views are not supported');
    const components = accessor.type === 'VEC3' ? 3 : 1;
    const byteOffset = binary.byteOffset + (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const bytes = binary.buffer.slice(byteOffset, byteOffset + accessor.count * components * 4);
    if (kind === 'f32' && accessor.componentType !== 5126) throw new Error('positions must be float');
    if (kind === 'u32' && accessor.componentType !== 5125) throw new Error('indices must be u32');
    return kind === 'f32' ? new Float32Array(bytes) : new Uint32Array(bytes);
  };
  const out: { positions: Float32Array; normals: Float32Array; indices: Uint32Array }[] = [];
  for (const node of doc.nodes) {
    if (node.mesh === undefined) continue;
    if (node.matrix) throw new Error('mesh nodes with a matrix are not expected in the bench export');
    for (const primitive of doc.meshes[node.mesh]?.primitives ?? []) {
      out.push({
        positions: read(primitive.attributes.POSITION as number, 'f32') as Float32Array,
        normals: read(primitive.attributes.NORMAL as number, 'f32') as Float32Array,
        indices: read(primitive.indices as number, 'u32') as Uint32Array,
      });
    }
  }
  return out;
}
