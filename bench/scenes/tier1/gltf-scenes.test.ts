import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { getScene } from '../index.ts';
import { loadFileGeometry } from '../file-geometry.ts';
import { loadGltfTriangles } from '../gltf-loader.ts';
import { gltfSceneDescFromFile } from '../gltf-scene-desc.ts';
import { gltfTextureDescs, loadGltfScene } from '../gltf-scene.ts';
import { sceneDescToCopyInput } from '../to-scene-copy.ts';
import { generateTexture } from '../textures.ts';
import { ASSETS_DIR } from '../assets-dir.ts';
import { SPONZA_GLTF } from '../tier2/sponza-white.ts';
import { GOLDEN_DIR, tier1Models } from './goldens.ts';
import { goldenMaterials, tier1GltfScenes } from './gltf-scenes.ts';

/** 7 significant digits, as the goldens are written (goldens.ts). */
const round = (v: unknown): unknown => {
  if (typeof v === 'number') return Number.isInteger(v) ? v : Number(v.toPrecision(7));
  if (Array.isArray(v)) return v.map(round);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).filter(([, x]) => x !== undefined).map(([k, x]) => [k, round(x)]));
  return v;
};
const plain = (v: unknown): unknown => JSON.parse(JSON.stringify(v));

/**
 * Per model: triangles in the file (docs/design/tier1.md section 1.1; counts every node's instance of a shared mesh, so
 * `AnisotropyRotationTest` is 13,876 = 7,476 + a second use of one mesh), triangles dropped as zero-area (sphere poles), and the
 * number of textures the four bench slots read (the file's other images are guides, labels, or slots with no scene-copy field).
 */
const EXPECTED: Record<string, { fileTriangles: number; degenerate: number; textures: number; materials: number }> = {
  MetalRoughSpheresNoTextures: { fileTriangles: 1_040_409, degenerate: 196, textures: 0, materials: 98 },
  MetalRoughSpheres: { fileTriangles: 501_776, degenerate: 0, textures: 2, materials: 1 },
  TransmissionThinwallTestGrid: { fileTriangles: 13_982, degenerate: 0, textures: 1, materials: 13 },
  TransmissionRoughnessTest: { fileTriangles: 77_792, degenerate: 0, textures: 9, materials: 9 },
  TransmissionTest: { fileTriangles: 128_775, degenerate: 0, textures: 8, materials: 14 },
  AttenuationTest: { fileTriangles: 292, degenerate: 0, textures: 2, materials: 18 },
  IORTestGrid: { fileTriangles: 30_945, degenerate: 0, textures: 1, materials: 23 },
  SpecularTest: { fileTriangles: 44_828, degenerate: 0, textures: 1, materials: 24 },
  ClearCoatTest: { fileTriangles: 37_116, degenerate: 0, textures: 5, materials: 19 },
  SheenTestGrid: { fileTriangles: 21_646, degenerate: 0, textures: 1, materials: 19 },
  IridescenceDielectricSpheres: { fileTriangles: 617_406, degenerate: 16_212, textures: 1, materials: 344 },
  IridescenceMetallicSpheres: { fileTriangles: 617_406, degenerate: 16_212, textures: 1, materials: 344 },
  AnisotropyStrengthTest: { fileTriangles: 97_240, degenerate: 0, textures: 1, materials: 50 },
  AnisotropyRotationTest: { fileTriangles: 13_876, degenerate: 0, textures: 5, materials: 6 },
  AnisotropyDiscTest: { fileTriangles: 2_788, degenerate: 0, textures: 0, materials: 12 },
  EmissiveStrengthTest: { fileTriangles: 90, degenerate: 0, textures: 1, materials: 6 },
  DispersionTest: { fileTriangles: 51_827, degenerate: 0, textures: 2, materials: 27 },
  DiffuseTransmissionTest: { fileTriangles: 4_980, degenerate: 0, textures: 3, materials: 29 },
};

interface GoldenJson { header: { materialCount: number }; materials: { i: number; name?: string; class: string; openpbr: Record<string, unknown>; notes?: string[]; textures?: string[]; unsupported?: string[] }[] }

describe('Tier 1 models through the glTF scene loader (needs node bench/assets/fetch.ts)', () => {
  const models = tier1Models();
  it('has an expectation for every Tier 1 model', () => {
    expect(Object.keys(EXPECTED).sort()).toEqual(models.map((m) => m.name).sort());
  });
  for (const model of models) {
    const want = EXPECTED[model.name]!;
    describe(model.name, () => {
      it.skipIf(!existsSync(model.file))('loads: triangle count, material table equals the P2.17 golden, texture count', () => {
        const s = loadGltfScene(model.file, { sha256: model.sha256, tangents: true, colors: true });
        expect(s.stats.fileTriangles).toBe(want.fileTriangles);
        expect(s.stats.degenerateDropped).toBe(want.degenerate);
        expect(s.stats.triangles).toBe(want.fileTriangles - want.degenerate);
        expect(s.groups.reduce((a, g) => a + g.triangles, 0)).toBe(s.stats.triangles);
        expect(s.materials).toHaveLength(want.materials);
        expect(s.textures).toHaveLength(want.textures);
        const golden = JSON.parse(readFileSync(join(GOLDEN_DIR, `${model.name}.json`), 'utf8')) as GoldenJson;
        expect(s.materials).toHaveLength(golden.header.materialCount);
        s.materials.forEach((m, i) => {
          const g = golden.materials[i]!;
          expect(m.index).toBe(g.i);
          expect(m.name).toBe(g.name);
          expect(m.converted.class).toBe(g.class);
          expect(round({ ...m.converted.material, type: undefined }), `material ${i}`).toEqual(plain(g.openpbr));
          expect(round(m.converted.notes.length ? m.converted.notes : undefined)).toEqual(g.notes);
          expect(round(m.converted.textures.length ? m.converted.textures : undefined)).toEqual(g.textures);
          expect(round(m.converted.unsupported.length ? m.converted.unsupported : undefined)).toEqual(g.unsupported);
        });
        // every material with a texture slot got its uvs; every group's triangles are drawn from materials that exist
        for (const g of s.groups) {
          const m = g.materialIndex >= 0 ? s.materials[g.materialIndex]! : s.defaultMaterial!;
          // (an anisotropic material also gets TEXCOORD_0 when tangents are carried: pbrt and Mitsuba derive the GGX axis from it, P2.26)
          expect(!!g.mesh.uvs, `${g.name} uvs`).toBe(Object.keys(m.slots).length > 0 || (m.material.specularRoughnessAnisotropy ?? 0) > 0);
        }
      });
    });
  }

  it.skipIf(!existsSync(tier1Models().find((m) => m.name === 'DiffuseTransmissionTest')!.file))('DiffuseTransmissionTest: a sun, unlit emitters, one BLEND material', () => {
    const m = tier1Models().find((x) => x.name === 'DiffuseTransmissionTest')!;
    const s = loadGltfScene(m.file);
    expect(s.lights).toHaveLength(1);
    expect(s.lights[0]!.type).toBe('directional');
    expect(s.materials.filter((x) => x.material.emissionLuminance === 1 && x.material.baseWeight === 0)).toHaveLength(8);
    expect(s.materials.filter((x) => x.material.alphaMode === 'blend')).toHaveLength(1);
  });
  it.skipIf(!existsSync(tier1Models().find((m) => m.name === 'TransmissionTest')!.file))('TransmissionTest: its camera is read', () => {
    const m = tier1Models().find((x) => x.name === 'TransmissionTest')!;
    expect(loadGltfScene(m.file).cameras).toHaveLength(1);
  });
  it.skipIf(!existsSync(tier1Models().find((m) => m.name === 'ClearCoatTest')!.file))('ClearCoatTest: tangents carried on request; its JPEG decodes to the declared size through the scene copy', () => {
    const m = tier1Models().find((x) => x.name === 'ClearCoatTest')!;
    const s = loadGltfScene(m.file, { tangents: true });
    expect(s.groups.every((g) => g.mesh.tangents && g.mesh.tangents.length === g.mesh.positions.length / 3 * 4)).toBe(true);
    const descs = gltfTextureDescs(s, `downloads/tier1/ClearCoatTest/ClearCoatTest.glb`, m.sha256);
    const jpg = Object.values(descs).find((d) => d.width === 1024)!;
    const t = generateTexture(jpg);
    expect([t.width, t.height, t.data.length]).toEqual([1024, 1024, 1024 * 1024 * 4]);
  });
});

describe('Tier 1 wave 1 scenes', () => {
  const names = ['AttenuationTest-notex', 'TransmissionThinwallTestGrid-notex', 'EmissiveStrengthTest-notex', 'MetalRoughSpheresNoTextures', 'MetalRoughSpheresNoTextures-grey-metal', 'IORTestGrid-notex', 'IORTestGrid-notex-A', 'IORTestGrid-notex-C', 'ClearCoatTest-notex', 'ClearCoatTest-notex-metal', 'AnisotropyStrengthTest-notex', 'SheenTestGrid-notex'];
  it('are registered by name', () => {
    expect(tier1GltfScenes.map((t) => t.scene.name)).toEqual(names);
    for (const n of names) expect(getScene(n).tier).toBe(1);
  });
  for (const t of tier1GltfScenes) {
    describe(t.scene.name, () => {
      it('has one file mesh per kept glTF material, every one factor-only, framed by a camera that looks at the model', () => {
        const dropped = new Set((t.sidecar.dropMaterials ?? []).map(String));
        const kept = goldenMaterials(t.golden).filter((m) => !((m.name !== undefined && dropped.has(m.name)) || dropped.has(String(m.index))));
        expect(t.scene.meshes).toHaveLength(kept.length);
        for (const mesh of t.scene.meshes) {
          expect(mesh.geometry.type).toBe('file');
          expect(mesh.uv).toBeUndefined();
        }
        expect(t.scene.textures).toBeUndefined();
        const { position, target, up } = t.scene.camera;
        const f = [target[0] - position[0], target[1] - position[1], target[2] - position[2]];
        expect(f[2]).toBeLessThan(0); // looks down -Z from +Z
        expect(up).toEqual([0, 1, 0]);
        expect(t.scene.environment?.type).toBe('hdri');
      });
      it.skipIf(!existsSync(join(ASSETS_DIR, t.model.path)))('equals the scene built from the downloaded model, resolves to triangles, and every drawn material exists', () => {
        const { scene, loaded } = gltfSceneDescFromFile(t.model, { ...t.sidecar });
        expect(round(plain(scene))).toEqual(round(plain(t.scene)));
        const input = sceneDescToCopyInput(t.scene, { resolveFile: (g) => loadFileGeometry(g).mesh });
        expect(input.meshes).toHaveLength(t.scene.meshes.length);
        expect(input.instances).toHaveLength(t.scene.meshes.length);
        const triangles = input.meshes.reduce((a, m) => a + (m.indices as ArrayLike<number>).length / 3, 0);
        const expected = t.scene.meshes.reduce((a, mesh) => a + loaded.groups.find((g) => g.materialIndex === (mesh.geometry as { select: { material: number } }).select.material)!.triangles, 0);
        expect(triangles).toBe(expected);
        if (t.scene.name === 'MetalRoughSpheresNoTextures') {
          expect(triangles).toBe(98 * 10_598);
          expect(loaded.defaultMaterial?.index).toBe(-1); // the text plane, dropped (no includeDefaultMaterial)
        }
        // wave 1 has no textures, so no mesh needed uvs
        expect(input.textures).toBeUndefined();
      });
    });
  }
});

describe('Sponza is unchanged by loader v2', () => {
  const path = join(ASSETS_DIR, SPONZA_GLTF.path);
  const sha = (a: ArrayBufferView): string => createHash('sha256').update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest('hex');
  it.skipIf(!existsSync(path))('the flat loader output is byte-identical to the pre-v2 loader (pinned hashes) and the scene loader finds the same triangles', () => {
    const r = loadGltfTriangles(path, { sha256: SPONZA_GLTF.sha256 });
    // recorded from gltf-loader.ts before the accessor code moved to gltf-read.ts
    expect(sha(r.mesh.positions)).toBe('1eafd4b1cc67058c022092e60eadcc2d254c2cfaa1698b61c5d3f47dfe636504');
    expect(sha(r.mesh.normals)).toBe('748b7e1eea25efbda8af9b849ff289371adf8c10328555785f2a173af25ddced');
    expect(sha(r.mesh.indices)).toBe('61e15eb4e29bb957ef796385a75cd1e39571381ba01bbbf139ed7af1bf823875');
    expect(r.stats).toMatchObject({ fileTriangles: 262_267, degenerateDropped: 1, triangles: 262_266, primitives: 103 });
    const s = loadGltfScene(path, { sha256: SPONZA_GLTF.sha256, uvs: false });
    expect(s.stats).toMatchObject({ fileTriangles: 262_267, degenerateDropped: 1, triangles: 262_266, primitives: 103 });
    // same triangles (the scene loader groups them by material, so compare as multisets of vertex triples + normals)
    const key = (p: Float32Array, n: Float32Array, t: number): string => `${Array.from(p.subarray(t * 9, t * 9 + 9)).join(',')}|${Array.from(n.subarray(t * 9, t * 9 + 3)).join(',')}`;
    const flat = new Map<string, number>();
    for (let t = 0; t < r.mesh.indices.length / 3; t++) { const k = key(r.mesh.positions, r.mesh.normals, t); flat.set(k, (flat.get(k) ?? 0) + 1); }
    for (const g of s.groups) for (let t = 0; t < g.triangles; t++) { const k = key(g.mesh.positions, g.mesh.normals, t); const c = flat.get(k); expect(c, 'triangle missing from the flat mesh').toBeGreaterThan(0); flat.set(k, c! - 1); }
    expect([...flat.values()].every((c) => c === 0)).toBe(true);
  });
  it.skipIf(!existsSync(path))('the sponza-white scene still resolves to the flat mesh (no select, no attributes)', () => {
    const g = getScene('sponza-white').meshes[0]!.geometry;
    if (g.type !== 'file') throw new Error('expected file geometry');
    expect(g.select).toBeUndefined();
    expect(g.attributes).toBeUndefined();
    expect(sha(loadFileGeometry(g).mesh.positions)).toBe('1eafd4b1cc67058c022092e60eadcc2d254c2cfaa1698b61c5d3f47dfe636504');
  });
  it.skipIf(!existsSync(path))('its materials load; image files that were not downloaded drop the texture slot with a note, not the geometry', () => {
    const s = loadGltfScene(path, { sha256: SPONZA_GLTF.sha256 });
    expect(s.materials.length).toBeGreaterThan(20);
    expect(s.stats.triangles).toBe(262_266);
    // MANIFEST.json fetches Sponza.gltf + Sponza.bin only (P1.4c3: add its images to fetch Sponza with textures); with them the slots resolve.
    if (s.textures.length === 0) expect(s.notes.filter((n) => /cannot be read/.test(n)).length).toBeGreaterThan(0);
    else expect(generateTexture(Object.values(gltfTextureDescs(s, SPONZA_GLTF.path, SPONZA_GLTF.sha256))[0]!).data.length).toBeGreaterThan(0);
  });
});
