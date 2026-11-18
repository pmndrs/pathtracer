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

  it('alpha MASK x cone LOD (P4.1c): the intersect kernel decides cutouts with the widened footprint of depth >= N, as the megakernel does; no other combination changes its text', () => {
    const build = (integrator: object, extra: object = {}) => ptWavefrontBuild(caps, { format: {}, layout: texLayout, textures: { buckets: [bucket] }, integrator: { alphaMask: true, ...integrator }, shortenNames: false, minify: false, ...extra });
    const cone = { textureLod: 'cone' as const };
    const on = build({ ...cone, materialLod: 'secondary', materialLodDepth: 2 });
    expect(on.intersect.code).toContain('select(1.0, 2.0, st.depth >= 2u)');
    expect(build({ ...cone, materialLod: 'secondary' }).intersect.code).toContain('st.depth >= 1u)');
    // the megakernel's cutout test is on the same lod-widened `fp` (the line the intersect kernel mirrors)
    const mk = ptMegakernelBuild(caps, { format: {}, layout: texLayout, textures: { buckets: [bucket] }, integrator: { alphaMask: true, ...cone, materialLod: 'secondary', materialLodDepth: 2 }, shortenNames: false, minify: false }).code;
    expect(mk).toContain('select(1.0, PT_LOD_TEX_BIAS, lod)');
    // off: material LOD full, level-0 textures, no cutouts -> the plain text
    for (const [name, integrator] of [['materialLod full', { ...cone }], ['textureLod 0', { materialLod: 'secondary' as const }]] as const) {
      expect(build(integrator).intersect.code, name).not.toContain('PT_LOD_TEX_BIAS');
      expect(build(integrator).intersect.code, name).not.toMatch(/st\.depth >= \d+u\)/);
    }
    expect(ptWavefrontBuild(caps, { format: {}, layout: texLayout, textures: { buckets: [bucket] }, integrator: { ...cone, materialLod: 'secondary' }, shortenNames: false, minify: false }).intersect.code).not.toMatch(/st\.depth >= \d+u\)/);
    // the other kernels are the ones without the option
    const off = build({ ...cone });
    for (const k of ['camera', 'shade', 'shadow', 'args'] as const) expect(on[k].code, k).toBe(off[k].code);
    expect(balanced(on.intersect.code)).toBe(true);
  });

  it('refill modes (R1b): both shade kernels exist, each skipping the other depth class (a refilled queue mixes depths)', () => {
    for (const refill of [true, 'batch', 'ordered'] as const) {
      const b = ptWavefrontBuild(caps, { format: {}, layout, integrator: { materials: materials('albedo'), materialLod: 'secondary', materialLodDepth: 1 }, refill, shortenNames: false, minify: false });
      expect(b.shadeSecondary, String(refill)).toBeDefined();
      expect(b.shade.code).toMatch(/& 0x1fffffffu\) >= 1u\) \{\s*return;/);
      expect((b.shadeSecondary as { code: string }).code).toMatch(/& 0x1fffffffu\) < 1u\) \{\s*return;/);
    }
  });

  it('no WGSL identifier of the reduced adapter is a reserved word', () => {
    const on = ptWavefrontBuild(caps, { format: {}, layout, integrator: { materials: materials('albedo'), materialLod: 'secondary' }, shortenNames: false, minify: false });
    const names = declaredIdentifiers((on.shadeSecondary as { code: string }).code);
    for (const n of names) expect(RESERVED.has(n), n).toBe(false);
  });
});
