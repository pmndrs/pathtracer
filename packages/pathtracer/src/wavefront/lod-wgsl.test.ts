// R1 (ray-class material LOD) at the kernel-text level: 'full' leaves the text alone, 'secondary' adds the reduced branch (megakernel) or a second, smaller shade kernel (wavefront).
import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptMaterialFeatures, type PtMaterialOptions } from '../bsdf/adapter.ts';
import { createMaterial } from '@pmndrs/raytracer';
import { ptMegakernelBuild } from '../integrator/megakernel.ts';
import { RESERVED, balanced, declaredIdentifiers } from '../../../raytracer/src/wgsl-names.testutil.ts';
import { ptWavefrontBuild, ptWavefrontKernelBytes } from './wavefront.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true, tangents: true };
const texLayout = { normals: true, uvs: true, positions: true, tangents: true };
const bucket = { width: 256, height: 256, colorSpace: 'srgb' as const, wrap: 'repeat' as const, layers: 2, mipLevels: 9 };

// a table with a layered dielectric, a coated metal, an anisotropic metal, fuzz and a thin film: every lobe the reduction removes
const table = [
  createMaterial({ baseColor: [0.8, 0.5, 0.2], baseWeight: 1, specularWeight: 1, ior: 1.5, specularRoughness: 0.3, sheenWeight: 1, sheenColor: [1, 1, 1] }),
  createMaterial({ metalness: 1, conductorEtaK: true, conductorEta: [0.2, 0.9, 1.1], conductorK: [3.9, 2.4, 2.1], specularRoughness: 0.3, coatWeight: 1, coatIor: 1.5, coatRoughness: 0.1 }),
  createMaterial({ metalness: 1, baseColor: [0.9, 0.6, 0.2], baseWeight: 1, specularRoughness: 0.4, specularAnisotropy: 0.5 }),
  createMaterial({ baseColor: [0.9, 0.9, 0.9], baseWeight: 1, specularWeight: 1, ior: 1.5, specularRoughness: 0.2, thinFilmWeight: 1, thinFilmThickness: 0.4, thinFilmIor: 1.4 }),
];
const features = ptMaterialFeatures(table);
// the stochastic walk has no thin film (the adapter falls back to albedo layering for such a scene): its table leaves the film material out
const { thinFilm: _film, ...withoutFilm } = ptMaterialFeatures(table.slice(0, 3));
void _film;
const materials = (layering: 'albedo' | 'stochastic'): PtMaterialOptions => ({ features: layering === 'albedo' ? features : withoutFilm, layered: { layering, maxDepth: 64 } });

describe('materialLod (R1)', () => {
  it('the table has every feature the reduction removes', () => {
    expect(features).toMatchObject({ layered: true, coatedMetal: true, aniso: true, fuzz: true, thinFilm: true });
  });

  for (const layering of ['albedo', 'stochastic'] as const) {
    it(`megakernel (${layering}): 'full' is the default text, 'secondary' adds the reduced branch only`, () => {
      const opts = (extra: object) => ({ format: {}, layout, integrator: { materials: materials(layering), ...extra } });
      const none = ptMegakernelBuild(caps, opts({})).code;
      const full = ptMegakernelBuild(caps, opts({ materialLod: 'full' })).code;
      const secondary = ptMegakernelBuild(caps, opts({ materialLod: 'secondary' })).code;
      expect(full).toBe(none);
      expect(secondary).not.toBe(none);
      const text = ptMegakernelBuild(caps, { ...opts({ materialLod: 'secondary' }), shortenNames: false, minify: false }).code;
      expect(text).toContain('ptBsdfReduced');
      expect(text).toContain('PT_LOD_DEPTH: u32 = 1u');
      expect(text).toContain('select(mat.sheenWeight, 0.0, lod)');
      expect(text).toContain('mat.specularAnisotropy > 0.0 && !lod');
      expect(text).toContain('ptLayeredEval'); // the full lobes stay compiled for depth < N
      expect(ptMegakernelBuild(caps, { ...opts({ materialLod: 'secondary', materialLodDepth: 3 }), shortenNames: false, minify: false }).code).toContain('PT_LOD_DEPTH: u32 = 3u');
      expect(balanced(secondary)).toBe(true);
    });
  }

  it('wavefront: the shade kernel stays the default text, the secondary kernel has no layered / fuzz / film / aniso code and is smaller', () => {
    const base = { format: {}, layout, integrator: { materials: materials('albedo') } };
    const off = ptWavefrontBuild(caps, base);
    const on = ptWavefrontBuild(caps, { ...base, integrator: { ...base.integrator, materialLod: 'secondary' } });
    expect(off.shadeSecondary).toBeUndefined();
    expect(on.shadeSecondary).toBeDefined();
    for (const k of ['camera', 'intersect', 'shade', 'shadow', 'args'] as const) expect(on[k].code, k).toBe(off[k].code);
    const bytes = ptWavefrontKernelBytes(on);
    expect(bytes.shadeSecondary).toBeLessThan(bytes.shade);
    const readable = ptWavefrontBuild(caps, { ...base, integrator: { ...base.integrator, materialLod: 'secondary' }, shortenNames: false });
    const text = (readable.shadeSecondary as { code: string }).code;
    expect(text).toContain('fn ptBsdfReduced');
    expect(text).not.toContain('ptLayeredEval');
    expect(text).not.toContain('ptFuzzedEval');
    expect(text).not.toContain('ptThinFilm');
    expect(text).not.toContain('ptFrameFromTangent');
    expect(text).toContain('PT_LAYER_LUT'); // its own copy of the albedo table: the reduced material reads E(mu, r)
    expect(balanced(text)).toBe(true);
  });

  it('wavefront with textures: the footprint of a reduced vertex is widened (cone LOD) and the secondary kernel is under the 50 KB budget', () => {
    const base = { format: {}, layout: texLayout, textures: { buckets: [bucket] }, integrator: { textureLod: 'cone' as const, materials: materials('stochastic') } };
    const on = ptWavefrontBuild(caps, { ...base, integrator: { ...base.integrator, materialLod: 'secondary' }, shortenNames: false });
    expect((on.shadeSecondary as { code: string }).code).toContain('PT_LOD_TEX_BIAS');
    const bytes = ptWavefrontKernelBytes(ptWavefrontBuild(caps, { ...base, integrator: { ...base.integrator, materialLod: 'secondary' } }));
    expect(bytes.shadeSecondary).toBeLessThan(50_000);
  });

  it('refill modes are not supported yet (a refilled slot mixes depths in one queue)', () => {
    expect(() => ptWavefrontBuild(caps, { format: {}, layout, integrator: { materials: materials('albedo'), materialLod: 'secondary' }, refill: true })).toThrow(/materialLod/);
  });

  it('no WGSL identifier of the reduced adapter is a reserved word', () => {
    const on = ptWavefrontBuild(caps, { format: {}, layout, integrator: { materials: materials('albedo'), materialLod: 'secondary' }, shortenNames: false, minify: false });
    const names = declaredIdentifiers((on.shadeSecondary as { code: string }).code);
    for (const n of names) expect(RESERVED.has(n), n).toBe(false);
  });
});
