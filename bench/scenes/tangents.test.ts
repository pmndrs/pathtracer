// P2.26 tangent convention check: glTF TANGENT (what a glTF consumer uses) against dp/du of the uvs (what pbrt-v4 and Mitsuba 3 derive on their own).
import { existsSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { toMitsuba, toPbrt } from '../exporters/index.ts';
import { ASSETS_DIR } from './assets-dir.ts';
import { loadFileGeometry } from './file-geometry.ts';
import { getScene } from './index.ts';
import { loadGltfScene } from './gltf-scene.ts';
import { tangentAngleDeg, uvTangents } from './tangents.ts';
import { tier1Models } from './tier1/goldens.ts';

describe('uvTangents', () => {
  it('a quad whose u grows along +x has the tangent +x, and w = +1 when "up" (decreasing v) is cross(n, t)', () => {
    // n = +z, t = +x, cross(n, t) = +y; v decreases toward +y (glTF: v down), so the bitangent along +y points up: w = +1
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
    const uvs = new Float32Array([0, 1, 1, 1, 1, 0, 0, 0]);
    const t = uvTangents({ positions, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), uvs });
    for (let v = 0; v < 4; v++) expect([...t.slice(v * 4, v * 4 + 4)]).toEqual([1, 0, 0, 1]);
    const flipped = uvTangents({ positions, indices: new Uint32Array([0, 1, 2, 0, 2, 3]), uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]) });
    expect(flipped[3]).toBe(-1);
  });
  it('is the triangle dp/du of an un-welded mesh and the area-weighted mean on a shared vertex', () => {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
    const uvs = new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]);
    const t = uvTangents({ positions, indices: new Uint32Array([0, 1, 2, 0, 3, 1]), uvs });
    expect(Math.hypot(t[0]!, t[1]!, t[2]!)).toBeCloseTo(1, 6);
    expect(t[4]).toBeCloseTo(1, 6); // vertex 1 sits only in triangle 0 and 1; dp/du is +x in the first
  });
});

const model = tier1Models().find((m) => m.name === 'AnisotropyStrengthTest')!;
describe.skipIf(!existsSync(model.file))('AnisotropyStrengthTest: glTF TANGENT vs dp/du of the uvs (P2.26 tangent check)', () => {
  it('at the corners of every flat triangle (the references\' frame) the two agree within 6 degrees (mean under 4): not swapped, flipped or 90 degrees off', () => {
    const s = loadGltfScene(model.file, { sha256: model.sha256, tangents: true });
    let worstTri = 0;
    let sumTri = 0;
    let count = 0;
    for (const g of s.groups) {
      if (!g.mesh.uvs || !g.mesh.tangents) continue;
      const uv = uvTangents(g.mesh as { positions: Float32Array; indices: Uint32Array; uvs: Float32Array });
      for (let v = 0; v < g.mesh.positions.length / 3; v++) {
        const n: [number, number, number] = [g.mesh.normals[v * 3]!, g.mesh.normals[v * 3 + 1]!, g.mesh.normals[v * 3 + 2]!];
        const a = tangentAngleDeg(uv, v, g.mesh.tangents, v, n);
        worstTri = Math.max(worstTri, a);
        sumTri += a;
        count++;
      }
    }
    expect(count).toBeGreaterThan(40_000);
    console.log(`AnisotropyStrengthTest tangent check: ${count} triangle corners, angle(TANGENT, dp/du) max ${worstTri.toFixed(2)} deg, mean ${(sumTri / count).toFixed(3)} deg`);
    expect(worstTri).toBeLessThan(6);
    expect(sumTri / count).toBeLessThan(4);
  });
  it('the scene meshes of anisotropic materials carry uvs and tangents; the exporters get uvs for exactly those meshes', () => {
    const scene = getScene('AnisotropyStrengthTest-notex');
    const m1 = scene.meshes.find((m) => m.name === 'm1')!;
    const m0 = scene.meshes.find((m) => m.name === 'm0')!;
    const a = loadFileGeometry(m1.geometry as never).mesh;
    const b = loadFileGeometry(m0.geometry as never).mesh;
    expect(a.uvs?.length).toBe((a.positions.length / 3) * 2);
    expect(a.tangents?.length).toBe((a.positions.length / 3) * 4);
    expect(b.uvs).toBeUndefined(); // anisotropy 0: isotropic, no uv frame needed
    for (const toNative of [toPbrt, toMitsuba]) {
      const out = toNative(scene, { spp: 16, assetsDir: ASSETS_DIR });
      expect(out.approximations.filter((x) => /no uvs/.test(x))).toEqual([]);
      const ply = out.files.find((f) => f.path.includes('-m1.'))!;
      expect(Buffer.from(ply.data as Uint8Array).subarray(0, 400).toString('latin1')).toContain('property float u');
    }
  });
});
