import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { sampleSceneTexture } from '@pmndrs/raytracer';
import { toGltf, toMitsuba, toPbrt } from '../exporters/index.ts';
import { encodePly } from '../exporters/ply.ts';
import { encodePng } from '../exporters/png.ts';
import { unsupportedReason } from '../adapters/registry.ts';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { meshToWorld, tessellate, tessellateQuad, tessellateSphere } from './geometry.ts';
import { getScene } from './index.ts';
import { buildSceneCopyFromDesc, materialFromDesc, sceneDescToCopyInput } from './to-scene-copy.ts';
import { generateTexture } from './textures.ts';
import { texChecker } from './tier1/textured.ts';

const opts = { assetsDir: '', spp: 16 };
const text = (f: { data: Buffer | string }) => f.data.toString();

describe('procedural textures', () => {
  it('checker: cells, orientation marker in the top-left cell of the tile, row 0 = top', () => {
    const t = generateTexture({ generator: 'checker', cells: 4, a: [255, 255, 255], b: [0, 0, 0], marker: [255, 0, 0], width: 16, height: 16, colorSpace: 'srgb' });
    const px = (x: number, y: number) => Array.from(t.data.subarray((y * 16 + x) * 4, (y * 16 + x) * 4 + 4));
    expect(px(0, 0)).toEqual([255, 0, 0, 255]);
    expect(px(5, 0)).toEqual([0, 0, 0, 255]); // cell (1, 0) is b
    expect(px(0, 5)).toEqual([0, 0, 0, 255]);
    expect(px(5, 5)).toEqual([255, 255, 255, 255]);
    expect(t.wrap).toBe('repeat');
  });

  it('bumps: a flat normal on the zero-gradient lines, tangent-space unit vectors, +x = +u', () => {
    const t = generateTexture({ generator: 'bumps', periods: 1, amplitude: 1, width: 64, height: 64, colorSpace: 'linear' });
    for (let i = 0; i < t.data.length; i += 4) {
      const n = [0, 1, 2].map((k) => (t.data[i + k]! / 255) * 2 - 1);
      expect(Math.hypot(n[0]!, n[1]!, n[2]!)).toBeCloseTo(1, 1);
      expect(n[2]).toBeGreaterThan(0);
    }
    // At u = 1/8 (sin(2 pi u) rising, v = 1/4 so sin(2 pi v) = 1): dh/du = cos(pi/4) > 0, so the normal leans toward -u.
    const o = (16 * 64 + 8) * 4; // y = 16 -> v = 0.258, x = 8 -> u = 0.133
    expect(t.data[o]!).toBeLessThan(128);
  });

  it('stripes-mr: green holds roughness, blue metalness', () => {
    const t = generateTexture({ generator: 'stripes-mr', periods: 2, a: [0.25, 0], b: [1, 1], width: 16, height: 4, colorSpace: 'linear' });
    expect(Array.from(t.data.subarray(0, 4))).toEqual([0, 64, 0, 255]);
    expect(Array.from(t.data.subarray(4 * 4, 4 * 4 + 4))).toEqual([0, 255, 255, 255]);
  });
});

describe('uvs on the analytic shapes', () => {
  it('quad: u along +X, v along +Z, glTF origin top-left; scale multiplies', () => {
    const q = tessellateQuad([4, 2], { scale: [3, 5] });
    expect(Array.from(q.uvs!)).toEqual([0, 5, 3, 5, 3, 0, 0, 0]);
    // vertex 0 is (-2, 0, +1): u = 0, v = max; vertex 2 is (+2, 0, -1): u = max, v = 0.
    expect([q.positions[0], q.positions[2]]).toEqual([-2, 1]);
    expect(tessellateQuad([1, 1]).uvs).toBeUndefined();
  });

  it('sphere: un-welded, same triangles and normals, uvs continuous across the seam, poles do not smear', () => {
    const plain = tessellateSphere(1, 16, 8);
    const s = tessellateSphere(1, 16, 8, { uv: {} });
    expect(s.indices.length).toBe(plain.indices.length);
    expect(s.uvs!.length).toBe((s.positions.length / 3) * 2);
    for (let t = 0; t < s.indices.length; t += 3) {
      const u = [0, 1, 2].map((k) => s.uvs![(t + k) * 2]!);
      const v = [0, 1, 2].map((k) => s.uvs![(t + k) * 2 + 1]!);
      expect(Math.max(...u) - Math.min(...u)).toBeLessThan(0.5); // no triangle spans the texture
      v.forEach((x) => expect(x).toBeGreaterThanOrEqual(0));
      v.forEach((x) => expect(x).toBeLessThanOrEqual(1 + 1e-6));
    }
    // v = 0 at the north pole (+Y), 1 at the south pole; u = 0 at +X, 0.25 toward +Z.
    for (let i = 0; i < s.positions.length / 3; i++) {
      const y = s.positions[i * 3 + 1]!;
      expect(s.uvs![i * 2 + 1]!).toBeCloseTo(Math.acos(Math.min(1, Math.max(-1, y))) / Math.PI, 5);
      if (Math.abs(y) < 0.99) {
        const u = s.uvs![i * 2]!;
        const expected = Math.atan2(s.positions[i * 3 + 2]!, s.positions[i * 3]!) / (2 * Math.PI);
        expect((((u - expected) % 1) + 1) % 1 < 1e-5 || (((u - expected) % 1) + 1) % 1 > 1 - 1e-5).toBe(true);
      }
      expect(Math.hypot(s.normals[i * 3]!, s.normals[i * 3 + 1]!, s.normals[i * 3 + 2]!)).toBeCloseTo(1, 6);
    }
  });

  it('other shapes refuse uvs; meshToWorld carries them through the transform', () => {
    expect(() => tessellate({ type: 'box', size: [1, 1, 1] }, { uv: {} })).toThrow(/quad and sphere/);
    const w = meshToWorld({ name: 'q', geometry: { type: 'quad', size: [1, 1] }, material: 'm', uv: { scale: [2, 2] }, transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1] });
    expect(w.uvs).toBeDefined();
    expect(w.positions[0]).toBeCloseTo(4.5, 6);
  });
});

describe('texture files for the native renderers', () => {
  it('encodePng writes a valid 8-bit RGB PNG that inflates back to the texels (row 0 = top)', () => {
    const rgba = Uint8Array.from([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 10, 20, 30, 255]);
    const png = encodePng(2, 2, rgba);
    expect(png.subarray(0, 8)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
    expect(png.readUInt32BE(16)).toBe(2);
    expect(png[24]).toBe(8);
    expect(png[25]).toBe(2); // RGB: all alpha 255
    const idat = png.indexOf('IDAT');
    const length = png.readUInt32BE(idat - 4);
    const raw = inflateSync(png.subarray(idat + 4, idat + 4 + length));
    expect(Array.from(raw)).toEqual([0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30].slice(0, 7).concat([0, 0, 0, 255, 10, 20, 30]));
    expect(encodePng(1, 1, Uint8Array.from([1, 2, 3, 100]))[25]).toBe(6); // RGBA when alpha is not opaque
  });

  it('encodePly writes u v only for meshes with uvs, flipping v on request', () => {
    const q = tessellateQuad([1, 1], { scale: [1, 1] });
    const plain = encodePly(tessellateQuad([1, 1]));
    const withUv = encodePly(q);
    expect(withUv.length - plain.length).toBe(4 * 8 + 'property float u\nproperty float v\n'.length);
    expect(withUv.toString('latin1', 0, 300)).toContain('property float u\nproperty float v');
    expect(plain.toString('latin1', 0, 300)).not.toContain('property float u');
    const flipped = encodePly(q, { flipV: true });
    const headerEnd = withUv.indexOf('end_header\n') + 'end_header\n'.length;
    // Vertex 0 has uv (0, 1): v is at byte 28 of the record.
    expect(withUv.readFloatLE(headerEnd + 28)).toBe(1);
    expect(flipped.readFloatLE(headerEnd + 28)).toBe(0);
  });

  it('pbrt: an imagemap per used texture (bilinear, sRGB, repeat), the material names it, v is flipped; mitsuba: a bitmap, raw false, v as is', () => {
    const p = toPbrt(texChecker, opts);
    const pbrt = text(p.files.find((f) => f.path === 'scene.pbrt')!);
    expect(pbrt).toContain('Texture "floor" "spectrum" "imagemap" "string filename" "tex-floor.png" "string filter" "bilinear" "string wrap" "repeat" "string encoding" "sRGB"');
    expect(pbrt).toContain('Material "diffuse" "texture reflectance" "floor"');
    expect(p.files.map((f) => f.path)).toEqual(expect.arrayContaining(['tex-floor.png', 'tex-ball.png']));
    expect(p.approximations.join(' ')).toMatch(/mip level/);
    const m = toMitsuba(texChecker, opts);
    const xml = text(m.files.find((f) => f.path === 'scene.xml')!);
    expect(xml).toContain('<texture type="bitmap" name="reflectance"><string name="filename" value="tex-floor.png"/><string name="filter_type" value="bilinear"/><string name="wrap_mode" value="repeat"/><boolean name="raw" value="false"/></texture>');
    const ply = (r: typeof p) => r.files.find((f) => f.path.endsWith('floor.ply'))!.data as Buffer;
    const at = (b: Buffer) => b.readFloatLE(b.indexOf('end_header\n') + 'end_header\n'.length + 28);
    expect(at(ply(p))).toBe(-3); // pbrt: v' = 1 - v (vertex 0 has v = the uv scale 4; repeat wraps)
    expect(at(ply(m))).toBe(4); // Mitsuba: unchanged (uv scale 4)
  });

  it('the exporters refuse what the native renderers cannot express', () => {
    const pbr = { ...texChecker, materials: { ...texChecker.materials, floor: { type: 'pbr' as const, baseColor: [1, 1, 1] as [number, number, number], roughness: 1, metalness: 0 } } };
    expect(() => toPbrt(pbr, opts)).toThrow(/pbr/);
    expect(() => toMitsuba(pbr, opts)).toThrow(/pbr/);
    const tinted = { ...texChecker, materials: { ...texChecker.materials, floor: { type: 'diffuse' as const, reflectance: [0.5, 0.5, 0.5] as [number, number, number], reflectanceTexture: 'floor' } } };
    expect(() => toPbrt(tinted, opts)).toThrow(/reflectance \[1, 1, 1\]/);
    // glTF (web renderers) carries the factors and says the textures are dropped; those renderers skip the scene anyway.
    expect(toGltf(texChecker, opts).approximations.join(' ')).toMatch(/dropped/);
  });
});

describe('textured bench scenes', () => {
  it('tex-checker and tex-checker-hf: well formed, every textured mesh has uvs, sizes are powers of two so every renderer reads the same texels', () => {
    for (const name of ['tex-checker', 'tex-checker-hf']) {
      const scene = getScene(name);
      expect(scene.tier).toBe(1);
      for (const t of Object.values(scene.textures!)) {
        expect(Number.isInteger(Math.log2(t.width))).toBe(true);
        expect(Number.isInteger(Math.log2(t.height))).toBe(true);
      }
      for (const mesh of scene.meshes) expect(mesh.uv, mesh.name).toBeDefined();
    }
  });

  it('converts to a scene copy with one texture per desc, slots by key order, sRGB base colour, uvs on the meshes', () => {
    const { copy, input } = buildSceneCopyFromDesc(texChecker);
    expect(input.textures).toHaveLength(2);
    expect(copy.textures!.slots.map((s) => s.label)).toEqual(['floor', 'ball']);
    expect(copy.materials.map((m) => m.baseColorTexture)).toEqual([0, 1]);
    expect(copy.layout.uvs).toBe(true);
    expect(copy.layout.positions).toBe(true);
    expect(copy.textureLayout!.buckets.map((b) => b.width)).toEqual([64, 128]);
    // The marker (top-left cell, orange) sits at uv (0.05, 0.05): the lookup returns it in linear light.
    const marker = sampleSceneTexture(copy.textures!, 0, 0.03, 0.03, 0);
    expect(marker[0]).toBeGreaterThan(0.7);
    expect(marker[2]).toBeLessThan(0.05);
  });

  it('a textured material on a mesh without uvs is an error, and an unknown texture too', () => {
    const noUv = { ...texChecker, meshes: texChecker.meshes.map(({ uv: _uv, ...m }) => m) };
    expect(() => sceneDescToCopyInput(noUv)).toThrow(/no `uv`/);
    expect(() => sceneDescToCopyInput({ ...texChecker, textures: {} })).toThrow(/unknown texture/);
    expect(() => materialFromDesc({ type: 'diffuse', reflectance: [1, 1, 1], reflectanceTexture: 'x' })).toThrow(/not defined/);
  });

  it('pbr maps factors and every texture slot', () => {
    const m = materialFromDesc(
      { type: 'pbr', baseColor: [0.1, 0.2, 0.3], roughness: 0.4, metalness: 0.6, emissive: [1, 2, 3], baseColorTexture: 'a', metallicRoughnessTexture: 'b', normalTexture: 'c', emissiveTexture: 'd' },
      (n) => 'abcd'.indexOf(n),
    );
    expect(m).toMatchObject({ baseColor: [0.1, 0.2, 0.3], specularRoughness: 0.4, metalness: 0.6, emissionColor: [1, 2, 3], baseColorTexture: 0, roughnessMetalTexture: 1, normalTexture: 2, emissionTexture: 3 });
  });

  it('only renderers that list textures get the scene', () => {
    const features = { materials: ['diffuse'] as const, emissiveMeshes: true, directionalLight: true, environment: ['uniform'] as const, rawOutput: true };
    expect(unsupportedReason(features, texChecker)).toMatch(/textures/);
    expect(unsupportedReason({ ...features, textures: true }, texChecker)).toBeNull();
    expect(new PtMegakernelAdapter().features.textures).toBe(true);
    expect(unsupportedReason(new PtMegakernelAdapter().features, getScene('tex-checker-hf'))).toBeNull();
  });
});
