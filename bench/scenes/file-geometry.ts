/** Node-side loader for the schema's `file` geometry (kept out of geometry.ts so browser code never pulls in node:fs). */
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { loadGltfTriangles } from './gltf-loader.ts';
import type { GltfLoadStats } from './gltf-loader.ts';
import type { TriangleMesh } from './geometry.ts';
import type { GeometryDesc } from './schema.ts';

/** bench/assets: downloads (git-ignored, see MANIFEST.json) and generated assets. */
export const ASSETS_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'assets');

type FileGeometry = Extract<GeometryDesc, { type: 'file' }>;

/** Local-space triangles of a `file` geometry (assets resolved against `assetsDir`, default bench/assets). */
export function loadFileGeometry(g: FileGeometry, assetsDir: string = ASSETS_DIR): { mesh: TriangleMesh; stats?: GltfLoadStats } {
  const path = resolve(assetsDir, g.path);
  switch (g.format) {
    case 'glb':
    case 'gltf':
      return loadGltfTriangles(path, { sha256: g.sha256 });
    default:
      throw new Error(`file geometry '${g.path}': format '${g.format}' is not supported by the Node loader yet (glb/gltf only)`);
  }
}
