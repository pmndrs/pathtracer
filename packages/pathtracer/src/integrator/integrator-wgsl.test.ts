import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { PT_PARAMS_BYTES, ptIntegratorWgsl, resolveIntegratorOptions } from './integrator-wgsl.ts';
import { PT_NO_FEATURES, type PtMaterialFeatures, type PtMaterialOptions } from '../bsdf/adapter.ts';
import { ptMegakernelWgsl } from './megakernel.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const bare = buildCaps({ features: new Set(), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };

function declared(wgsl: string): string[] {
  return [...wgsl.matchAll(/^(?:@[^\n]*?\s)?(?:struct|fn|const|alias|override|var(?:<[^>]*>)?)\s+([A-Za-z_]\w*)/gm)].map((m) => m[1] as string);
}

describe('integrator WGSL', () => {
  it('resolves defaults: Owen-Sobol, NEE on, uniform env sampling, sun invisible to camera', () => {
    expect(resolveIntegratorOptions()).toEqual({ sampler: 'owen-sobol', blueNoise: {}, frame: false, frameEmission: 'none', accumulate: true, nee: true, envSampling: 'uniform', envMap: false, sunVisible: false, textures: false, textureLod: 0, alphaMask: false, alphaPolicy: 'exact', alphaShadow: 'exact', alphaRoulette: 0.25, alphaFirstK: 1, alphaCoverage: [], alphaMixedMask: 4, alphaTransmit: 0.5, materials: null, hooks: null, counters: false, aovs: false, primary: 'trace', lod: 'off', lodDepth: 1, restir: false, restirShared: false, restirGi: false, volumes: null, originOffset: 0, spectral: null, mtlx: false, mtlxNormal: false });
  });

  it('bakes the settings in as module constants', () => {
    const a = ptIntegratorWgsl({});
    expect(a).toContain('PT_NEE: bool = true');
    expect(a).toContain('PT_ENV_SAMPLING: bool = true');
    expect(a).toContain('PT_SUN_VISIBLE: bool = false');
    const b = ptIntegratorWgsl({ nee: false, envSampling: 'none', sunVisible: true });
    expect(b).toContain('PT_NEE: bool = false');
    expect(b).toContain('PT_ENV_SAMPLING: bool = false');
    expect(b).toContain('PT_SUN_VISIBLE: bool = true');
  });

  it('the params struct is 96 bytes: 6 vec4', () => {
    expect(PT_PARAMS_BYTES).toBe(96);
    expect(ptIntegratorWgsl()).toMatch(/struct PtParams \{\s*dims: vec4u,\s*limits: vec4u,\s*origin: vec4f,\s*right: vec4f,\s*up: vec4f,\s*forward: vec4f,\s*\}/);
  });

  it('uses the pbrt order: emission, depth check, NEE, BSDF sample, Russian roulette', () => {
    const w = ptIntegratorWgsl();
    const at = (s: string, from = 0) => w.indexOf(s, from);
    expect(at('mat.emissionColor')).toBeGreaterThan(0);
    expect(at('depth >= maxDepth')).toBeGreaterThan(at('mat.emissionColor'));
    const depthCheck = at('depth >= maxDepth');
    const nee = at('ptSampleLd(', depthCheck);
    const sample = at('ptBsdfSample(', nee);
    expect(nee).toBeGreaterThan(depthCheck);
    expect(sample).toBeGreaterThan(nee);
    expect(at('rrStart > 0u && depth >= rrStart', sample)).toBeGreaterThan(sample);
  });

  it('keeps radiance as vec4 and f32 accumulation', () => {
    const w = ptIntegratorWgsl();
    expect(w).toContain('alias PtRadiance = vec4f');
    expect(w).toContain('ptAccum: array<vec4f>');
  });

  it('declares no name twice across the whole kernel, for both samplers and with NEE off', () => {
    for (const integrator of [{}, { sampler: 'pcg' as const }, { nee: false }, { vndf: 'cap' as const, transport: 'importance' as const }]) {
      // Not minified: the declaration scan keys on the first column, where only top-level declarations start.
      const names = declared(ptMegakernelWgsl(caps, { format: {}, layout, integrator, minify: false }));
      const dup = names.filter((n, i) => names.indexOf(n) !== i);
      expect(dup).toEqual([]);
    }
  });

  it('every `pt`/`rt` function the integrator calls is declared in the kernel', () => {
    const code = ptMegakernelWgsl(caps, { format: {}, layout, minify: false, shortenNames: false });
    const names = new Set(declared(code));
    const called = new Set([...ptIntegratorWgsl().matchAll(/\b((?:pt|rt)[A-Z]\w*)\(/g)].map((m) => m[1] as string));
    called.delete('PtRadiance');
    for (const n of called) expect(names.has(n) || /^Rt|^Pt/.test(n), n).toBe(true);
    const missing = [...called].filter((n) => !names.has(n) && !/^(Rt|Pt)/.test(n));
    expect(missing).toEqual([]);
  });

  it('puts the bindings where the host expects them: walk group 0, scene group 1, io group 2', () => {
    const code = ptMegakernelWgsl(caps, { format: {}, layout, scenePacking: 'split', minify: false, shortenNames: false });
    expect(code).toMatch(/@group\(0\) @binding\(0\)/);
    expect(code).toMatch(/@group\(1\) @binding\(0\) var<storage, read> rtSceneTable/);
    expect(code).toMatch(/@group\(1\) @binding\(1\) var<storage, read> rtSceneData/);
    expect(code).toMatch(/@group\(2\) @binding\(0\) var<uniform> ptParams/);
    expect(code).toMatch(/@group\(2\) @binding\(1\) var<storage, read_write> ptAccum/);
  });

  it('starts with the caps enable lines exactly once (a second buildWgsl would put them mid-file)', () => {
    const code = ptMegakernelWgsl(caps, { format: {}, layout });
    expect(code.startsWith('enable f16;\nenable subgroups;')).toBe(true);
    expect(code.match(/^enable /gm)?.length).toBe(2);
    expect(ptMegakernelWgsl(bare, { format: {}, layout }).match(/^enable /gm)).toBeNull();
  });

  it('stays within the kernel budget class: records the size (target ~50 KB, see TASKS P2.4 follow-up)', () => {
    const minified = ptMegakernelWgsl(caps, { format: {}, layout }).length;
    const plain = ptMegakernelWgsl(caps, { format: {}, layout, minify: false }).length;
    expect(minified).toBeLessThan(plain);
    // Hard ceiling so the size cannot creep (P2.10: 40.4 KB with pruning + short names; the bvh walk is ~14 KB of it and is not
    // ours to cut). Windows DXC compile time grows with size: the kernel stays under the 50 KB class.
    expect(minified).toBeLessThan(41_500);
    expect(plain).toBeLessThan(51_000);
    // The A/B baselines (everything on the page, long names) stay available and are the pre-P2.10 sizes.
    const raw = ptMegakernelWgsl(caps, { format: {}, layout, prune: false, shortenNames: false }).length;
    expect(raw).toBeGreaterThan(52_000);
    expect(raw).toBeLessThan(54_000);
  });

  it('an environment-map scene adds the map code (about 2.8 KB minified) and stays under 44.5 KB; uniform sampling of a map is the A/B baseline', () => {
    const none = ptMegakernelWgsl(caps, { format: {}, layout }).length;
    const map = ptMegakernelWgsl(caps, { format: {}, layout, integrator: { envMap: true, envSampling: 'importance' } });
    const readable = (integrator: object) => ptMegakernelWgsl(caps, { format: {}, layout, integrator, shortenNames: false });
    expect(map.length).toBeGreaterThan(none + 2_000);
    expect(map.length).toBeLessThan(44_500);
    expect(readable({ envMap: true, envSampling: 'importance' })).toContain('RT_ENV_IMPORTANCE: bool = true');
    expect(readable({ envMap: true, envSampling: 'uniform' })).toContain('RT_ENV_IMPORTANCE: bool = false');
    expect(readable({})).not.toContain('rtEnvMapLookup');
  });

  describe('material textures (P1.4c)', () => {
    const texLayout = { normals: true, uvs: true, positions: true };
    const bucket = (srgb: boolean, wrap: 'repeat' | 'clamp' | 'mirror' = 'repeat') => ({ width: 256, height: 256, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap, layers: 2, mipLevels: 9 });
    const textures = { buckets: [bucket(false), bucket(true), bucket(true, 'clamp')] };
    const build = (extra: object = {}) => ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures, ...extra });

    it('emits no texture code without textures: the untextured kernel text does not mention them', () => {
      const code = ptMegakernelWgsl(caps, { format: {}, layout, minify: false, shortenNames: false });
      expect(code).not.toMatch(/rtTex|rtApplyMaterialTextures|PT_TEX_CONE|RtCone\(/);
      expect(ptIntegratorWgsl({})).not.toContain('cone');
    });

    it('alpha mask (P1.4c5): the mask code exists only with `alphaMask` and textures; the textured kernel without it is unchanged', () => {
      const plain = build({ minify: false, shortenNames: false });
      const masked = build({ minify: false, shortenNames: false, integrator: { alphaMask: true } });
      expect(plain).not.toMatch(/ptMaskedOut|ptVisibleMasked|PT_MASK_SKIPS|skips/);
      expect(masked).toContain('fn ptMaskedOut');
      expect(masked).toContain('ptVisibleMasked(ray, RT_MASK_ALL)');
      expect(masked).toContain('mat.alphaMode == RT_ALPHA_MASK && mat.opacity < mat.alphaCutoff');
      expect(build({ minify: false, shortenNames: false, integrator: { alphaMask: false } })).toBe(plain);
      // without textures there is no base colour alpha to read: the option is ignored and the kernel text is the plain one
      const untex = ptMegakernelWgsl(caps, { format: {}, layout, minify: false, shortenNames: false });
      expect(ptMegakernelWgsl(caps, { format: {}, layout, minify: false, shortenNames: false, integrator: { alphaMask: true } })).toBe(untex);
    });

    it('alpha policy (R2): exact is the default text; the opaque policies drop the shadow loop and / or the indirect cutout test', () => {
      const o = { minify: false, shortenNames: false };
      const exact = build({ ...o, integrator: { alphaMask: true } });
      expect(build({ ...o, integrator: { alphaMask: true, alphaPolicy: 'exact' } })).toBe(exact);
      const shadow = build({ ...o, integrator: { alphaMask: true, alphaPolicy: 'opaqueShadow' } });
      expect(shadow).not.toContain('ptVisibleMasked');
      expect(shadow).toContain('rtVisible(ray, RT_MASK_ALL)');
      expect(shadow).toContain('if (mat.alphaMode == RT_ALPHA_MASK && mat.opacity < mat.alphaCutoff)');
      const indirect = build({ ...o, integrator: { alphaMask: true, alphaPolicy: 'opaqueIndirect' } });
      expect(indirect).toContain('ptVisibleMasked(ray, RT_MASK_ALL)');
      expect(indirect).toContain('if (depth == 0u && mat.alphaMode == RT_ALPHA_MASK && mat.opacity < mat.alphaCutoff)');
      const both = build({ ...o, integrator: { alphaMask: true, alphaPolicy: 'opaqueSecondary' } });
      expect(both).not.toContain('ptVisibleMasked');
      expect(both).toContain('depth == 0u && mat.alphaMode == RT_ALPHA_MASK');
      // no alphaMask: the policy is ignored
      expect(build({ ...o, integrator: { alphaPolicy: 'opaqueSecondary' } })).toBe(build(o));
    });

    it('alpha shadow (R2b): `policy` is the default text; each mode replaces only the shadow visibility; options without alphaMask change nothing', () => {
      const o = { minify: false, shortenNames: false };
      const exact = build({ ...o, integrator: { alphaMask: true } });
      expect(build({ ...o, integrator: { alphaMask: true, alphaShadow: 'policy' } })).toBe(exact);
      expect(build({ ...o, integrator: { alphaMask: true, alphaShadow: 'policy', alphaPolicy: 'opaqueShadow' } })).toBe(build({ ...o, integrator: { alphaMask: true, alphaPolicy: 'opaqueShadow' } }));
      const roulette = build({ ...o, integrator: { alphaMask: true, alphaShadow: 'roulette', alphaRoulette: 0.125 } });
      expect(roulette).toContain('fn ptShadowWeight(ray: RtRay, h: vec3u) -> f32');
      expect(roulette).toContain('const PT_ALPHA_Q: f32 = 0.125;');
      expect(roulette).toContain('ptVisibleMasked(ray, RT_MASK_ALL)'); // the exact loop is the correction
      expect(roulette).toContain('let vis = ptShadowWeight(ray, vec3u(pixelSeed, sampleIndex, vertex));');
      expect(roulette).toContain('(w * vis / pl)');
      // the indirect side still follows alphaPolicy (exact by default: cutouts at depth >= 1 are tested)
      expect(roulette).toContain('if (mat.alphaMode == RT_ALPHA_MASK && mat.opacity < mat.alphaCutoff)');
      expect(build({ ...o, integrator: { alphaMask: true, alphaShadow: 'roulette', alphaPolicy: 'opaqueIndirect' } })).toContain('depth == 0u && mat.alphaMode == RT_ALPHA_MASK');
      const coverage = build({ ...o, integrator: { alphaMask: true, alphaShadow: 'coverage', alphaCoverage: [1, 0.5, 0.25] } });
      expect(coverage).toContain('var<private> PT_ALPHA_COVERAGE: array<f32, 3> = array<f32, 3>(1.0, 0.5, 0.25);');
      expect(coverage).not.toContain('ptVisibleMasked'); // no texture fetch in the shadow ray
      expect(coverage).not.toContain('ptAlphaRand');
      const stochastic = build({ ...o, integrator: { alphaMask: true, alphaShadow: 'stochastic', alphaCoverage: [1, 0.5] } });
      expect(stochastic).toContain('ptAlphaRand(h.x, h.y, h.z + i) < c');
      expect(stochastic).not.toContain('ptVisibleMasked');
      const firstK = build({ ...o, integrator: { alphaMask: true, alphaShadow: 'firstK', alphaFirstK: 2 } });
      expect(firstK).toContain('const PT_ALPHA_K: u32 = 2u;');
      expect(firstK).toContain('rtVisible(ray, RT_MASK_ALL)'); // the solid walk after the exact ones
      const transmit = build({ ...o, integrator: { alphaMask: true, alphaShadow: 'transmit', alphaTransmit: 0.25 } });
      expect(transmit).toContain('const PT_ALPHA_BETA: f32 = 0.25;');
      expect(transmit).toContain('rtVisible(ray, RT_MASK_ALL & ~PT_ALPHA_CARDS)');
      expect(transmit).not.toContain('ptVisibleMasked'); // no texture fetch, no loop
      const classified = build({ ...o, integrator: { alphaMask: true, alphaShadow: 'classified', alphaMixedMask: 8 } });
      expect(classified).toContain('const PT_ALPHA_MIXED: u32 = 0x8u;');
      expect(classified).toContain('rtVisible(ray, RT_MASK_ALL & ~PT_ALPHA_MIXED)');
      expect(classified).toContain('ptVisibleMasked(ray, PT_ALPHA_MIXED)');
      // ignored without alphaMask; bad parameters throw
      expect(build({ ...o, integrator: { alphaShadow: 'roulette' } })).toBe(build(o));
      expect(() => resolveIntegratorOptions({ alphaMask: true, textures: true, alphaShadow: 'roulette', alphaRoulette: 0 })).toThrow(/alphaRoulette/);
      expect(() => resolveIntegratorOptions({ alphaMask: true, textures: true, alphaShadow: 'coverage' })).toThrow(/alphaCoverage/);
      expect(() => resolveIntegratorOptions({ alphaMask: true, textures: true, alphaShadow: 'firstK', alphaFirstK: 0 })).toThrow(/alphaFirstK/);
    });

    it('binds the arrays then their samplers after the two scene buffers, in group 1', () => {
      const code = ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures, scenePacking: 'split', minify: false, shortenNames: false });
      for (let i = 0; i < 3; i++) {
        expect(code).toContain(`@group(1) @binding(${2 + i}) var rtTex${i}: texture_2d_array<f32>;`);
        expect(code).toContain(`@group(1) @binding(${5 + i}) var rtSmp${i}: sampler;`);
      }
      expect(code).not.toContain('rtTex3');
      expect(code).toContain('RT_SURF_TEXFRAME');
    });

    it('lod setting is a module constant: 0 (reference) or cone', () => {
      const readable = (textureLod: 0 | 'cone') => ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures, integrator: { textureLod }, minify: false, shortenNames: false });
      expect(readable(0)).toContain('PT_TEX_CONE: bool = false');
      expect(readable('cone')).toContain('PT_TEX_CONE: bool = true');
    });

    it('size: about 3 KB over the untextured kernel, under 45.5 KB, 47.5 KB with an environment map', () => {
      const plain = ptMegakernelWgsl(caps, { format: {}, layout: texLayout }).length;
      const textured = build().length;
      expect(textured - plain).toBeGreaterThan(2_000);
      expect(textured).toBeLessThan(45_500);
      expect(build({ integrator: { envMap: true, envSampling: 'importance' } }).length).toBeLessThan(47_500);
    });

    it('textures need a surface layout with uvs and positions', () => {
      expect(() => ptMegakernelWgsl(caps, { format: {}, layout: { normals: true, uvs: false, positions: true }, textures })).toThrow(/uvs and positions/);
    });
  });

  it('pruning and short names keep the entry point and the bindings, and drop the bvh closest-point queries', () => {
    const code = ptMegakernelWgsl(caps, { format: {}, layout });
    expect(code).toContain('fn ptMain(');
    expect(code).toMatch(/@group\(2\)\s*@binding\(1\)\s*var<storage,\s*read_write>\s*ptAccum/);
    expect(ptMegakernelWgsl(caps, { format: {}, layout, shortenNames: false })).toContain('fn rtClosest');
    expect(ptMegakernelWgsl(caps, { format: {}, layout, prune: false, shortenNames: false })).toContain('bvhClosestPointToTriangle');
    expect(ptMegakernelWgsl(caps, { format: {}, layout, shortenNames: false })).not.toContain('bvhClosestPointToTriangle');
  });

  it('minify keeps the line count (compiler line numbers stay valid)', () => {
    const a = ptMegakernelWgsl(caps, { format: {}, layout });
    const b = ptMegakernelWgsl(caps, { format: {}, layout, minify: false });
    expect(a.split('\n').length).toBe(b.split('\n').length);
  });

  it('indexed leaves and f32 nodes still compose', () => {
    for (const format of [{ triangleLeaves: 'indexed' as const }, { nodeFormat: 'f32' as const }]) {
      expect(ptMegakernelWgsl(caps, { format, layout }).length).toBeGreaterThan(30_000);
    }
  });

  describe('OpenPBR mode (P2.21 layered, P2.22 thin + medium, P2.30 mix): only the lobes a scene uses are compiled', () => {
    const feats = (f: Partial<PtMaterialFeatures>, layered: PtMaterialOptions['layered'] = { layering: 'albedo' }): PtMaterialOptions => ({ features: { ...PT_NO_FEATURES, ...f }, layered });
    const build = (materials: PtMaterialOptions | null, extra: object = {}, readable = false) => ptMegakernelWgsl(caps, { format: {}, layout, integrator: { materials, ...extra }, ...(readable ? { minify: false, shortenNames: false } : {}) });
    const readable = (f: Partial<PtMaterialFeatures>, layered?: PtMaterialOptions['layered']) => build(feats(f, layered), {}, true);

    it('without `materials` the kernel is the Tier 0 kernel (checked byte for byte against the pre-OpenPBR commit with bench/size-snapshot.ts): no thin, layered, mix or medium code', () => {
      const code = readable({});
      const legacy = ptMegakernelWgsl(caps, { format: {}, layout, minify: false, shortenNames: false });
      expect(legacy).not.toMatch(/PT_BSDF_THIN|PT_BSDF_LAYERED|ptMediumSigma|ptLayerSeed|ptLayeredEval|sigma/);
      expect(code).not.toBe(legacy);
      expect(ptIntegratorWgsl({ materials: null })).toBe(ptIntegratorWgsl({}));
    });

    it('each feature adds its code and nothing else', () => {
      const none = readable({});
      expect(none).not.toMatch(/PT_BSDF_THIN|PT_BSDF_LAYERED|ptMediumSigma|ptLayerSeed|ptLayeredEval|var sigma/);
      expect(readable({ thin: true })).toMatch(/PT_BSDF_THIN/);
      expect(readable({ thin: true })).not.toMatch(/PT_BSDF_LAYERED|ptMediumSigma|var sigma/);
      const medium = readable({ medium: true });
      expect(medium).toMatch(/var sigma = vec3f\(0\.0\);/);
      expect(medium).toMatch(/beta = beta \* exp\(-sigma \* hit\.t\);/);
      expect(medium).toMatch(/sigma = select\(vec3f\(0\.0\), ptMediumSigma\(mat\), wo\.z > 0\.0\);/);
      expect(medium).not.toMatch(/PT_BSDF_THIN|PT_BSDF_LAYERED/);
      expect(readable({ mix: true })).toMatch(/ptBsdfFromMaterial\(mat, ptSample2D\(pixelSeed, sampleIndex, ptVertexPair\(depth, PT_SLOT_RR\)\)\.y\)/);
      expect(readable({})).toMatch(/ptBsdfFromMaterial\(mat, 0\.0\)/);
      const layered = readable({ layered: true });
      expect(layered).toMatch(/PT_BSDF_LAYERED/);
      expect(layered).toMatch(/ptLayerSeed = pixelSeed/);
      expect(layered).not.toMatch(/PT_FLAG_PROPORTIONAL\) != 0u/);
      expect(readable({ layered: true }, { layering: 'stochastic' })).toMatch(/PT_FLAG_PROPORTIONAL\) != 0u/);
    });

    it('the layered BSDF compiles only in the chosen layering: albedo carries the table, stochastic the walk', () => {
      const a = readable({ layered: true }, { layering: 'albedo' });
      const st = readable({ layered: true }, { layering: 'stochastic' });
      expect(a).toMatch(/ptLayerAlbedoLut/);
      expect(a).not.toMatch(/ptLayeredF\(/);
      expect(st).toMatch(/fn ptLayeredF\(/);
      expect(st).not.toMatch(/ptLayerAlbedoLut/);
    });

    it('fuzz (P2.25) and thin film (P2.27) add their code only for scenes that use them, and the kernel without them is unchanged', () => {
      const base = readable({ layered: true });
      expect(base).not.toMatch(/ptFuzz|ptFilm|ptFrConductor/);
      const fz = readable({ layered: true, fuzz: true });
      expect(fz).toMatch(/@group\(2\) @binding\(2\) var<storage, read> ptFuzzTable/);
      expect(fz).toMatch(/ptFuzzW = mat\.sheenWeight;/);
      expect(fz).toMatch(/ptFuzzedEval\(bp, wo, wiLocal\)/);
      expect(fz).toMatch(/ptFuzzedSample\(bp, wo,/);
      expect(fz).not.toMatch(/ptFilm/);
      const flm = readable({ layered: true, thinFilm: true });
      expect(flm).toMatch(/ptFilm = select\(vec3f\(0\.0\), ptMaterialFilm\(hit\.material\)/);
      expect(flm).toMatch(/ptFrConductor\(dwo, p\.eta, p\.k\)/);
      expect(flm).not.toMatch(/ptFuzz/);
      expect(() => readable({ layered: true, thinFilm: true }, { layering: 'stochastic' })).toThrow(/albedo/);
      // fuzz without a layered lobe: the base is dispatched straight to ptBsdfEval
      expect(readable({ fuzz: true })).toMatch(/ptBsdfEval\(p, wo, wi\)/);
    });

    it('sizes (minified, short names; P2.21/22/30): thin +0.5 KB, medium +0.3 KB, layered albedo +7.9 KB, stochastic +9.1 KB; all lobes with an env map under 53.7 KB', () => {
      const legacy = ptMegakernelWgsl(caps, { format: {}, layout }).length;
      const size = (f: Partial<PtMaterialFeatures>, layered?: PtMaterialOptions['layered'], extra: object = {}) => build(feats(f, layered), extra).length;
      const all = { layered: true, mix: true, thin: true, medium: true };
      expect(size({})).toBeLessThan(legacy + 100);
      expect(size({ mix: true })).toBeLessThan(legacy + 150);
      expect(size({ thin: true })).toBeLessThan(legacy + 600);
      expect(size({ medium: true })).toBeLessThan(legacy + 400);
      // Caps raised by 150 B (P-W1): the constant-bounded walk loops (rtBoundWalkLoops) add ~130 B to every kernel with both walks.
      // D32: the default albedo layering carries the `texture` correction (+0.8 KB of text) and the scene copy is packed (-0.1 KB).
      expect(size({ layered: true }, { layering: 'albedo' })).toBeLessThan(49_600);
      expect(size({ layered: true }, { layering: 'stochastic' })).toBeLessThan(50_050);
      expect(size(all, { layering: 'albedo' })).toBeLessThan(50_400);
      expect(size(all, { layering: 'stochastic' })).toBeLessThan(50_850);
      expect(size(all, { layering: 'stochastic' }, { envMap: true, envSampling: 'importance' })).toBeLessThan(53_650);
    });
  });
});

describe('integrator.originOffset (P5.v17, V302)', () => {
  it('0 / absent keeps the kernel text; a floor shifts every origin along the side normal before the ulp offset', () => {
    expect(ptIntegratorWgsl({ originOffset: 0 })).toBe(ptIntegratorWgsl({}));
    const text = ptIntegratorWgsl({ originOffset: 0.001953125 });
    expect(text).toContain('const PT_ORIGIN_FLOOR: f32 = 0.001953125;');
    expect(text).toContain('let p = p0 + PT_ORIGIN_FLOOR * n;');
    expect(text.split('fn ptOffsetOrigin(').length - 1).toBe(1);
    expect(() => resolveIntegratorOptions({ originOffset: -1 })).toThrow(RangeError);
    expect(() => resolveIntegratorOptions({ originOffset: Number.NaN })).toThrow(RangeError);
  });
});
