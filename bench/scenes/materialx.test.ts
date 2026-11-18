import { describe, expect, it } from 'vitest';
import { RESERVED, balanced, declaredIdentifiers } from '../../packages/raytracer/src/wgsl-names.testutil.ts';
import { getScene } from './index.ts';
import { collectMaterialXPrograms, buildMaterialX } from './materialx.ts';
import { syntheticImage } from './materialx-images.ts';
import { generateTexture } from './textures.ts';
import { materialFromDesc, sceneDescToCopyInput } from './to-scene-copy.ts';

describe('MaterialX bench scenes', () => {
  it('constant open_pbr_surface and its native openpbr description give the same material record', () => {
    const a = sceneDescToCopyInput(getScene('mtlx-const-open-pbr'));
    const b = sceneDescToCopyInput(getScene('mtlx-const-open-pbr-ref'));
    expect(a.materials).toEqual(b.materials);
  });
  it('constant standard_surface maps to the hand-mapped OpenPBR (V261)', () => {
    const a = sceneDescToCopyInput(getScene('mtlx-const-std'));
    const b = sceneDescToCopyInput(getScene('mtlx-const-std-ref'));
    expect(a.materials).toEqual(b.materials);
    // The standard_surface defaults that differ from OpenPBR's reached the record: specular_roughness was authored 0.25, thin film off.
    expect(a.materials[1]!.specularRoughness).toBe(0.25);
  });
  it('rejects mode constants for a graph that varies', () => {
    expect(() => buildMaterialX({ key: 'x-const-varying', mode: 'constants', source: getScene('mtlx-uv-bake') && '<materialx version="1.38"><texcoord name="uv" type="vector2"/><ramplr name="r" type="float"><input name="texcoord" type="vector2" nodename="uv"/></ramplr><standard_surface name="S" type="surfaceshader"><input name="specular_roughness" type="float" nodename="r"/></standard_surface></materialx>' })).toThrow(/varies/);
  });
  it('bake: textures are lazy, sized as asked, and carry the graph (checker orientation: the marker cell is top-left)', () => {
    const scene = getScene('mtlx-uv-bake-256');
    expect(Object.keys(scene.textures!).sort()).toEqual(['uv-bake-256.base', 'uv-bake-256.mr', 'uv-bake-256.normal']);
    const base = generateTexture(scene.textures!['uv-bake-256.base']!);
    expect([base.width, base.height]).toEqual([256, 256]);
    expect(base.colorSpace).toBe('srgb');
    const mr = generateTexture(scene.textures!['uv-bake-256.mr']!);
    // Metalness ramps along u (smoothstep 0.55..0.75 of u): 0 at the left, 255 at the right; roughness is the rings.
    expect(mr.data[(128 * 256 + 2) * 4 + 2]).toBe(0);
    expect(mr.data[(128 * 256 + 253) * 4 + 2]).toBe(255);
    const mat = scene.materials.subject as unknown as { baseColor: number[]; specularRoughness: number; baseMetalness: number; baseColorTexture: string };
    expect(mat.baseColor).toEqual([1, 1, 1]);
    expect(mat.specularRoughness).toBe(1);
    expect(mat.baseColorTexture).toBe('uv-bake-256.base');
  });
  it('codegen: one program for the subject, normal map generated, images mapped to scene textures', () => {
    const scene = getScene('mtlx-uv-codegen');
    const input = sceneDescToCopyInput(scene);
    const k = collectMaterialXPrograms(scene, input.materialNames)!;
    expect(k.programs.normal).toBe(true);
    expect(k.programs.wgsl).toContain('fn ptMtlxApply(');
    expect(k.programs.wgsl).toContain('fn ptMtlxNormal(');
    const slots = [...k.programs.wgsl.matchAll(/ptMtlxImg\((\d+)u/g)].map((m) => Number(m[1]));
    expect(new Set(slots)).toEqual(new Set([0, 1, 2]));
    // The record is the fallback: roughness at uv (0.5, 0.5), a metalness probe for the kernel features.
    expect(k.featureProbes.some((p) => p.metalness === 0.5)).toBe(true);
  });
  it('codegen of a position-driven graph reads no image and gets a placeholder texture', () => {
    const scene = getScene('mtlx-proc-codegen');
    expect(Object.keys(scene.textures!)).toEqual(['proc-codegen.placeholder']);
    const k = collectMaterialXPrograms(scene, sceneDescToCopyInput(scene).materialNames)!;
    expect(k.programs.normal).toBe(false);
    expect(k.programs.wgsl).not.toContain('ptMtlxImg');
    expect(k.programs.wgsl).toContain('sin(');
  });
  it('generated WGSL is balanced and declares no reserved identifier or duplicate function', () => {
    for (const name of ['mtlx-uv-codegen', 'mtlx-proc-codegen']) {
      const scene = getScene(name);
      const { wgsl } = collectMaterialXPrograms(scene, sceneDescToCopyInput(scene).materialNames)!.programs;
      expect(balanced(wgsl)).toBe(true);
      expect([...declaredIdentifiers(wgsl)].filter((n) => RESERVED.has(n))).toEqual([]);
      const fns = [...wgsl.matchAll(/^fn (\w+)/gm)].map((m) => m[1]);
      expect(new Set(fns).size).toBe(fns.length);
      // Every generated helper that is called is declared (a missing `need` only fails on the GPU).
      const called = new Set([...wgsl.matchAll(/\b(ptMtlx\w+)\(/g)].map((m) => m[1]));
      for (const c of called) expect(fns, `ptMtlx function ${c}`).toContain(c);
    }
  });
  it('synthetic images are what the scenes say they are', () => {
    const c = syntheticImage('gen:checker8')!;
    expect([c.data[0], c.data[1], c.data[2]]).toEqual([230, 40, 40]); // marker cell, top-left
    expect(syntheticImage('gen:ramp')!.data[(5 * 128 + 127) * 4]).toBe(255);
    expect(syntheticImage('nope.png')).toBeUndefined();
    void materialFromDesc;
  });
});
