import { traceModuleWgsl } from '@pmndrs/bvh';
import { describe, expect, it } from 'vitest';
import { buildCaps } from '../caps/index.ts';
import { rtModuleStorageBuffers, rtModuleWgsl, type RtModuleOptions } from '../trace/index.ts';
import { RESERVED, balanced, declared, declaredIdentifiers } from '../wgsl-names.testutil.ts';
import { RT_SHADE_MODE, type RtShadeOptions } from './shade.ts';
import { rtShadeWgsl } from './shade-wgsl.ts';

const noFeatures = buildCaps({ features: new Set(), limits: {} }, new Set());
const allFeatures = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const slot = { group: 2, firstBinding: 0 };
const layout = { normals: true, uvs: true, positions: true };
const KERNEL_BYTE_LIMIT = 50_000;

const variants: RtShadeOptions[] = [];
for (const bsdf of ['builtin', 'external'] as const)
  for (const cache of ['none', 'external'] as const)
    for (const envSampling of ['uniform', 'cosine'] as const)
      for (const flat of ['cache', 'albedo'] as const) variants.push({ bsdf, cache, envSampling, flat, sunVisible: envSampling === 'cosine' });

const module = (shade: RtShadeOptions | boolean, extra: Partial<RtModuleOptions> = {}, caps = noFeatures): string => rtModuleWgsl(caps, { scene: { slot, layout }, shade, ...extra });

describe('rtShade WGSL', () => {
  it('declares no identifier WGSL reserves, in any variant', () => {
    for (const shade of variants) {
      const bad = [...declaredIdentifiers(rtShadeWgsl(shade)), ...declaredIdentifiers(module(shade))].filter((name) => RESERVED.has(name));
      expect(bad, JSON.stringify(shade)).toEqual([]);
    }
  });

  it('every variant preprocesses cleanly, is balanced and declares nothing twice or outside the rt prefix', () => {
    const walk = new Set(declared(traceModuleWgsl({ triangleLeaves: 'indexed' })));
    for (const shade of variants) {
      for (const caps of [noFeatures, allFeatures]) {
        const code = module(shade, { triangleLeaves: 'indexed' }, caps);
        const what = JSON.stringify(shade);
        expect(code, what).not.toMatch(/^\s*#(if|elif|else|endif)\b/m);
        expect(balanced(code), what).toBe(true);
        const seen = new Set<string>();
        for (const name of declared(code)) {
          expect(seen.has(name), `${what} duplicate ${name}`).toBe(false);
          seen.add(name);
          if (!walk.has(name)) expect(name, what).toMatch(/^(rt|Rt|RT_)/);
        }
      }
    }
  });

  it('the public functions exist; the hooks are the module\'s unless declared external', () => {
    const code = module(true);
    for (const fn of ['rtShade', 'rtShadePoint', 'rtShadeAt', 'rtShadeTotal', 'rtShadeMiss', 'rtShadeDirect', 'rtShadeDiffuseFromIrradiance', 'rtOffsetOrigin', 'rtSideNormal', 'rtShadeIndirect', 'rtShadeBsdf', 'rtSampleLightLi']) expect(code, fn).toContain(`fn ${fn}(`);
    const external = module({ cache: 'external', bsdf: 'external' });
    expect(external).not.toContain('fn rtShadeIndirect(');
    expect(external).not.toContain('fn rtShadeBsdf(');
    // A consumer's declarations then complete the module (WGSL has no declaration order).
    const mine = `${external}\nfn rtShadeIndirect(p: RtShadePoint) -> vec3f { return p.albedo; }\nfn rtShadeBsdf(p: RtShadePoint, wi: vec3f) -> vec3f { return vec3f(1.0); }`;
    expect(declared(mine).filter((n) => n === 'rtShadeIndirect' || n === 'rtShadeBsdf')).toEqual(['rtShadeIndirect', 'rtShadeBsdf']);
    expect(balanced(mine)).toBe(true);
  });

  it('mode constants match the TypeScript ones; options become module constants', () => {
    const code = rtShadeWgsl({ envSampling: 'cosine', flat: 'albedo', sunVisible: true, minAlpha: 0.05 });
    for (const [name, value] of Object.entries(RT_SHADE_MODE)) expect(code).toContain(`const RT_SHADE_${name.toUpperCase()}: u32 = ${value}u;`);
    expect(code).toContain('const RT_SHADE_ENV_COSINE: bool = true;');
    expect(code).toContain('const RT_SHADE_FLAT_ALBEDO: bool = true;');
    expect(code).toContain('const RT_SHADE_SUN_VISIBLE: bool = true;');
    expect(code).toContain(`const RT_SHADE_MIN_ALPHA: f32 = ${Math.fround(0.05)};`);
  });

  it('needs the scene copy with surface, materials and lights, and adds light sampling itself', () => {
    expect(() => rtModuleWgsl(noFeatures, { shade: true })).toThrow(/scene/);
    expect(() => module(true, { scene: { slot, layout, lights: false } })).toThrow(/lights/);
    expect(() => module(true, { scene: { slot, layout, surface: false } })).toThrow(/surface/);
    expect(module(false)).not.toContain('fn rtShade(');
    expect(module(false)).not.toContain('fn rtSampleLightLi(');
    expect(module(true)).toContain('fn rtSampleLightLi(');
    expect(rtModuleStorageBuffers({ scene: { slot, layout }, shade: true })).toBe(rtModuleStorageBuffers({ scene: { slot, layout } }));
  });

  it('stays under the kernel cap with `minify` (without it the module with shade and indexed leaves is just over)', () => {
    let largest = 0;
    let unminified = 0;
    for (const shade of variants) {
      for (const triangleLeaves of ['records', 'indexed'] as const) {
        largest = Math.max(largest, module(shade, { triangleLeaves, minify: true }).length);
        unminified = Math.max(unminified, module(shade, { triangleLeaves }).length);
      }
    }
    expect(largest + 4_000).toBeLessThan(KERNEL_BYTE_LIMIT); // room for a kernel of 4 KB
    expect(unminified).toBeGreaterThan(largest * 1.1);
    // What shade (and the light sampling it needs) costs, unminified.
    const withShade = module(true, { triangleLeaves: 'indexed' }).length;
    const without = module(false, { triangleLeaves: 'indexed' }).length;
    expect(withShade - without).toBeLessThan(12_000);
    // Minifying keeps every line break (error line numbers) and changes no token.
    const plain = module(true);
    const small = module(true, { minify: true });
    expect(small.split('\n').length).toBe(plain.split('\n').length);
    expect(small.replace(/\s+/g, '')).toBe(plain.replace(/\s+/g, ''));
  });
});
