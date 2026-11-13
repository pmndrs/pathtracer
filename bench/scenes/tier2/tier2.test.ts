import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import { ASSETS_DIR, loadFileGeometry } from '../file-geometry.ts';
import { allScenes, getScene } from '../index.ts';
import { sponzaWhite, SPONZA_GLTF } from './sponza-white.ts';

describe('tier2 scenes', () => {
  it('are registered, uniquely named and well formed', () => {
    expect(new Set(allScenes.map((s) => s.name)).size).toBe(allScenes.length);
    expect(getScene('sponza-white')).toBe(sponzaWhite);
    expect(sponzaWhite.tier).toBe(2);
    expect(sponzaWhite.film).toEqual({ width: 1280, height: 720, filter: 'box' });
    expect(sponzaWhite.integrator).toEqual({ maxDepth: 8, rrStartDepth: 4 });
    expect(Object.values(sponzaWhite.materials)).toEqual([{ type: 'diffuse', reflectance: [0.5, 0.5, 0.5] }]);
    for (const m of sponzaWhite.meshes) expect(sponzaWhite.materials[m.material]).toBeDefined();
  });
  it('pins the same sha256 as bench/assets/MANIFEST.json', () => {
    const manifest = JSON.parse(readFileSync(join(ASSETS_DIR, 'MANIFEST.json'), 'utf8')) as { assets: { files: { path: string; sha256: string }[] }[] };
    const files = manifest.assets.flatMap((a) => a.files);
    expect(files.find((f) => f.path === SPONZA_GLTF.path)?.sha256).toBe(SPONZA_GLTF.sha256);
  });
  // Needs the git-ignored download (node bench/assets/fetch.ts); skipped on a fresh checkout.
  const have = existsSync(join(ASSETS_DIR, SPONZA_GLTF.path)) && existsSync(join(ASSETS_DIR, 'downloads/sponza/Sponza.bin'));
  it.skipIf(!have)('loads Sponza: 262,267 file triangles, one degenerate dropped, plausible bounds', () => {
    const g = sponzaWhite.meshes[0]!.geometry;
    if (g.type !== 'file') throw new Error('expected file geometry');
    const { stats } = loadFileGeometry(g);
    expect(stats!.fileTriangles).toBe(262267);
    expect(stats!.triangles).toBe(262266);
    expect(stats!.bounds.max[0] - stats!.bounds.min[0]).toBeGreaterThan(25); // atrium is ~30 m long
    expect(createHash('sha256').update(readFileSync(join(ASSETS_DIR, SPONZA_GLTF.path))).digest('hex')).toBe(SPONZA_GLTF.sha256);
  });
});
