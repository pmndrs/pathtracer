/** Node-side loader for the schema's `file` geometry (kept out of geometry.ts so browser code never pulls in node:fs). */
import { resolve } from 'node:path';
import { loadGltfTriangles } from './gltf-loader.ts';
import type { GltfLoadStats } from './gltf-loader.ts';
import { loadGltfScene } from './gltf-scene.ts';
import type { GltfScene } from './gltf-scene.ts';
import type { TriangleMesh } from './geometry.ts';
import type { GeometryDesc } from './schema.ts';
import { ASSETS_DIR } from './assets-dir.ts';
import './gltf-images.ts'; // registers the image texture decoder

export { ASSETS_DIR };

type FileGeometry = Extract<GeometryDesc, { type: 'file' }>;

// A scene with N materials resolves N `select` meshes from one file: keep the last loaded scene so the file is read once.
let lastScene: { key: string; scene: GltfScene } | undefined;

function sceneOf(path: string, g: FileGeometry): GltfScene {
  // uvs are always loaded (only textured materials have any) and stripped per mesh below, so meshes that differ only in `uvs` share one load.
  const a = g.attributes ?? {};
  const key = `${path}|${g.sha256 ?? ''}|${a.normals ?? 'geometric'}|${a.tangents ? 1 : 0}|${a.colors ? 1 : 0}`;
  if (lastScene?.key !== key) {
    lastScene = { key, scene: loadGltfScene(path, { ...(g.sha256 ? { sha256: g.sha256 } : {}), ...(a.normals ? { normals: a.normals } : {}), uvs: true, tangents: a.tangents ?? false, colors: a.colors ?? false }) };
  }
  return lastScene.scene;
}

/** Local-space triangles of a `file` geometry (assets resolved against `assetsDir`, default bench/assets). */
export function loadFileGeometry(g: FileGeometry, assetsDir: string = ASSETS_DIR): { mesh: TriangleMesh; stats?: GltfLoadStats } {
  const path = resolve(assetsDir, g.path);
  switch (g.format) {
    case 'glb':
    case 'gltf': {
      if (!g.select && !g.attributes) return loadGltfTriangles(path, { sha256: g.sha256 });
      const scene = sceneOf(path, g);
      const wanted = g.select?.material;
      const groups = wanted === undefined ? [] : scene.groups.filter((x) => x.materialIndex === wanted);
      if (wanted !== undefined && groups.length === 0) throw new Error(`file geometry '${g.path}': no triangles use glTF material ${wanted}`);
      if (wanted === undefined) throw new Error(`file geometry '${g.path}': \`attributes\` without \`select\` is not supported (use select.material, or leave both off for the flat mesh)`);
      const group = groups[0]!;
      const { uvs, ...rest } = group.mesh;
      return { mesh: g.attributes?.uvs && uvs ? { ...rest, uvs } : rest, stats: { fileTriangles: group.triangles, degenerateDropped: 0, triangles: group.triangles, primitives: 0, bounds: scene.stats.bounds } };
    }
    default:
      throw new Error(`file geometry '${g.path}': format '${g.format}' is not supported by the Node loader yet (glb/gltf only)`);
  }
}
