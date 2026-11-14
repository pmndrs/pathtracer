import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { PT_PARAMS_BYTES, ptIntegratorWgsl, resolveIntegratorOptions } from './integrator-wgsl.ts';
import { ptMegakernelWgsl } from './megakernel.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const bare = buildCaps({ features: new Set(), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };

function declared(wgsl: string): string[] {
  return [...wgsl.matchAll(/^(?:@[^\n]*?\s)?(?:struct|fn|const|alias|override|var(?:<[^>]*>)?)\s+([A-Za-z_]\w*)/gm)].map((m) => m[1] as string);
}

describe('integrator WGSL', () => {
  it('resolves defaults: Owen-Sobol, NEE on, uniform env sampling, sun invisible to camera', () => {
    expect(resolveIntegratorOptions()).toEqual({ sampler: 'owen-sobol', nee: true, envSampling: 'uniform', envMap: false, sunVisible: false, textures: false, textureLod: 0 });
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
    const code = ptMegakernelWgsl(caps, { format: {}, layout, minify: false, shortenNames: false });
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

    it('binds the arrays then their samplers after the two scene buffers, in group 1', () => {
      const code = ptMegakernelWgsl(caps, { format: {}, layout: texLayout, textures, minify: false, shortenNames: false });
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
});
