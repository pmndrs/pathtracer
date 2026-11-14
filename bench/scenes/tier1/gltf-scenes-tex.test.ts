import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { getScene } from '../index.ts';
import { gltfSceneDescFromFile } from '../gltf-scene-desc.ts';
import { sceneDescToCopyInput } from '../to-scene-copy.ts';
import { loadFileGeometry } from '../file-geometry.ts';
import { generateTexture } from '../textures.ts';
import { ASSETS_DIR } from '../assets-dir.ts';
import { goldenMaterials, tier1GltfScenes } from './gltf-scenes.ts';
import { tier1TexturedGltfScenes } from './gltf-scenes-tex.ts';

const round = (v: unknown): unknown => {
  if (typeof v === 'number') return Number.isInteger(v) ? v : Number(v.toPrecision(7));
  if (Array.isArray(v)) return v.map(round);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).filter(([, x]) => x !== undefined).map(([k, x]) => [k, round(x)]));
  return v;
};
const plain = (v: unknown): unknown => JSON.parse(JSON.stringify(v));

describe('Tier 1 wave 2 (textured, unmodified) scenes', () => {
  const names = ['IORTestGrid', 'IORTestGrid-A', 'SheenTestGrid', 'AttenuationTest', 'MetalRoughSpheres'];
  it('are registered by name', () => {
    expect(tier1TexturedGltfScenes.map((t) => t.scene.name)).toEqual(names);
    for (const n of names) expect(getScene(n).tier).toBe(1);
  });
  for (const t of tier1TexturedGltfScenes) {
    describe(t.scene.name, () => {
      it('keeps every glTF material of the model (nothing dropped or replaced) and its textures', () => {
        const dropped = t.sidecar.dropMaterials ?? [];
        expect(t.scene.meshes).toHaveLength(goldenMaterials(t.golden).length - dropped.length);
        if (t.scene.name !== 'IORTestGrid-A') expect(dropped).toEqual([]);
        else expect(dropped).toHaveLength(10);
        expect(t.sidecar.replace).toBeUndefined();
        const textured = t.scene.meshes.filter((m) => m.uv);
        expect(textured.length).toBeGreaterThan(0);
        for (const m of textured) expect(m.geometry).toMatchObject({ type: 'file', attributes: { uvs: true } });
        for (const tex of Object.values(t.scene.textures ?? {})) expect(tex.generator).toBe('image');
      });
      it('looks at the same thing as its wave 1 variant (camera, film, integrator, sky)', () => {
        const w1 = tier1GltfScenes.find((s) => s.scene.name === (t.scene.name === 'MetalRoughSpheres' ? 'MetalRoughSpheresNoTextures' : `${t.scene.name.replace(/-A$/, '')}-notex`))!;
        if (t.scene.name !== 'MetalRoughSpheres') {
          expect(t.scene.camera).toEqual(w1.scene.camera);
          expect(t.scene.film).toEqual(w1.scene.film);
          expect(t.scene.integrator).toEqual(w1.scene.integrator);
        }
        expect(t.scene.environment).toEqual(w1.scene.environment);
      });
      it.skipIf(!existsSync(join(ASSETS_DIR, t.model.path)))('equals the scene built from the downloaded model; textures decode to their declared size; uvs reach the scene copy', () => {
        const { scene } = gltfSceneDescFromFile(t.model, { ...t.sidecar });
        expect(round(plain(scene))).toEqual(round(plain(t.scene)));
        for (const tex of Object.values(t.scene.textures ?? {})) {
          const g = generateTexture(tex);
          expect([g.width, g.height, g.data.length]).toEqual([tex.width, tex.height, tex.width * tex.height * 4]);
        }
        const input = sceneDescToCopyInput(t.scene, { resolveFile: (g) => loadFileGeometry(g).mesh });
        expect(input.textures).toHaveLength(Object.keys(t.scene.textures ?? {}).length);
        t.scene.meshes.forEach((mesh, i) => expect(!!input.meshes[i]!.uvs).toBe(!!mesh.uv));
      });
    });
  }
  it('MetalRoughSpheres: clamp-to-edge sRGB base colour + linear metallic-roughness, one mesh, no alpha', () => {
    const s = getScene('MetalRoughSpheres');
    expect(Object.values(s.textures!).map((t) => [t.colorSpace, t.wrap])).toEqual([['srgb', 'clamp'], ['linear', 'clamp']]);
    expect(s.meshes).toHaveLength(1);
    expect(Object.values(s.materials)[0]).toMatchObject({ baseColorTexture: 't0', metallicRoughnessTexture: 't1_lin' });
  });
});

describe('tex-mask (alpha MASK)', () => {
  it('is a mask material over a checker whose b cells have alpha below the cutoff', () => {
    const s = getScene('tex-mask');
    expect(s.materials['screen']).toMatchObject({ alphaMode: 'mask', alphaCutoff: 0.5, baseColorTexture: 'screen' });
    const t = generateTexture(s.textures!['screen']!);
    const a = (x: number, y: number) => t.data[(y * t.width + x) * 4 + 3]!;
    expect([a(1, 1), a(17, 1), a(17, 17)]).toEqual([255, 100, 255]); // marker cell, b cell, a cell; the marker's alpha stays 255
    const input = sceneDescToCopyInput(s);
    expect(input.materials[1]).toMatchObject({ alphaMode: 1, alphaCutoff: 0.5 });
  });
});
