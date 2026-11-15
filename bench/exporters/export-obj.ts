/**
 * Writes a scene as OBJ + MTL for VkNRC (track N, N6.2 / WN6; bench/exporters/obj.ts), plus `vknrc.json` (VkNRC's normalisation and the scene camera in its terms).
 *
 *   node bench/exporters/export-obj.ts [scene] [--out <dir>]      (default scene: sponza-white-vknrc; Sponza needs `node bench/assets/fetch.ts`)
 *
 * Output (git-ignored): bench/assets/generated/vknrc/<scene>/scene.obj, scene.mtl, vknrc.json. Then: `VkNRC <dir>/scene.obj`.
 */
import { parseArgs } from 'node:util';
import { join } from 'node:path';
import { ASSETS_DIR } from '../scenes/file-geometry.ts';
import { getScene } from '../scenes/index.ts';
import { writeExport } from './index.ts';
import { toObj } from './obj.ts';

const { values, positionals } = parseArgs({ allowPositionals: true, options: { out: { type: 'string' } } });
const scene = getScene(positionals[0] ?? 'sponza-white-vknrc');
const dir = values.out ?? join(ASSETS_DIR, 'generated', 'vknrc', scene.name);
const result = toObj(scene, { assetsDir: ASSETS_DIR });
const file = writeExport(result, dir);
const s = result.sidecar;
console.log(`${file}: ${s.vertices} vertices, ${s.triangles} triangles; VkNRC camera position ${s.camera.position.map((v) => v.toFixed(5)).join(' ')}, yaw ${s.camera.yaw.toFixed(5)}, pitch ${s.camera.pitch.toFixed(5)}`);
for (const a of result.approximations) console.log(`  approximation: ${a}`);
