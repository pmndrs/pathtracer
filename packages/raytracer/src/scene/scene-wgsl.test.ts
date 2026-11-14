import { traceModuleWgsl } from '@pmndrs/bvh';
import { describe, expect, it } from 'vitest';
import { buildCaps } from '../caps/index.ts';
import { rtModuleStorageBuffers, rtModuleWgsl, rtSceneEntries, stripWgslComments, type RtModuleOptions } from '../trace/index.ts';
import { SCENE_COPY_STORAGE_BUFFERS } from './constants.ts';
import { defaultMaterial } from './material.ts';
import { rtLightsWgsl, rtMaterialWgsl, rtSceneBindingsWgsl, rtSceneWgsl } from './scene-wgsl.ts';
import { SCENE_HEADER } from './constants.ts';
import { RESERVED, balanced, declared, declaredIdentifiers } from '../wgsl-names.testutil.ts';

const noFeatures = buildCaps({ features: new Set(), limits: {} }, new Set());
const allFeatures = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const KERNEL_BYTE_LIMIT = 50_000;
const slot = { group: 2, firstBinding: 0 };

function fieldsOf(wgsl: string, struct: string): string[] {
  const body = new RegExp(`struct\\s+${struct}\\s*\\{([^}]*)\\}`).exec(wgsl)?.[1];
  if (!body) throw new Error(`struct ${struct} not found`);
  return body
    .split('\n')
    .map((line) => line.replace(/\/\/.*$/, '').trim())
    .filter(Boolean)
    .map((line) => line.split(':')[0]!.trim());
}

function sceneOptions(): RtModuleOptions[] {
  const out: RtModuleOptions[] = [];
  for (const normals of [false, true])
    for (const uvs of [false, true])
      for (const positions of [false, true])
        for (const triangleLeaves of ['records', 'indexed'] as const)
          for (const nodeFormat of ['f32', 'quantized'] as const)
            out.push({ triangleLeaves, nodeFormat, scene: { slot, layout: { normals, uvs, positions } } });
  return out;
}

describe('scene copy WGSL', () => {
  it('declares no identifier that WGSL reserves', () => {
    const code = rtModuleWgsl(allFeatures, { triangleLeaves: 'indexed', scene: { slot, layout: { normals: true, uvs: true, positions: true } } });
    const bad = [...declaredIdentifiers(rtSceneWgsl({ group: 0, firstBinding: 0 })), ...declaredIdentifiers(code)].filter((name) => RESERVED.has(name));
    expect(bad).toEqual([]);
    // The detector does catch the keyword that failed on the GPU.
    expect([...declaredIdentifiers('let meta = 1u;')].filter((n) => RESERVED.has(n))).toEqual(['meta']);
  });

  it('every layout combination preprocesses cleanly, stays under the kernel cap and binds two buffers', () => {
    let largest = 0;
    for (const options of sceneOptions()) {
      for (const caps of [noFeatures, allFeatures]) {
        const code = rtModuleWgsl(caps, options);
        const what = JSON.stringify(options);
        expect(code, what).not.toMatch(/^\s*#(if|elif|else|endif)\b/m);
        expect(balanced(code), what).toBe(true);
        largest = Math.max(largest, code.length);
        const bindings = [...code.matchAll(/@group\(2\)\s*@binding\((\d+)\)/g)].map((m) => Number(m[1]));
        expect(bindings, what).toEqual([0, 1]);
        for (const fn of ['rtSurface', 'rtMaterial', 'rtMaterialAlpha', 'rtSampleLight', 'rtLight', 'rtSun', 'rtEnvironment', 'rtLightPmf', 'rtLightIndexOfHit', 'rtLightPmfOfHit', 'rtSceneInfo', 'rtVisible', 'rtClosest']) expect(code, `${fn} ${what}`).toContain(`fn ${fn}(`);
        expect(code, what).not.toContain('rtSurfaceTable');
      }
    }
    // Headroom: the cap is 50,000 and this is the module alone (a kernel adds its own few hundred bytes).
    expect(largest).toBeLessThan(KERNEL_BYTE_LIMIT - 2_000);
  });

  it('comment stripping keeps line numbers and the walk\'s size without it would not fit', () => {
    const walk = traceModuleWgsl({ triangleLeaves: 'indexed' });
    const stripped = stripWgslComments(walk);
    expect(stripped.split('\n').length).toBe(walk.split('\n').length);
    expect(stripped.length).toBeLessThan(walk.length * 0.85);
    expect(stripped).not.toContain('//');
    const sample = 'let a = 1; // trailing\n/* block\n over lines */ let b = 2;\n';
    expect(stripWgslComments(sample)).toBe('let a = 1;\n\n let b = 2;\n');
    const unstripped = rtModuleWgsl(noFeatures, { triangleLeaves: 'indexed', scene: { slot, layout: { normals: true, uvs: true, positions: true } }, stripComments: false });
    expect(unstripped.length).toBeGreaterThan(rtModuleWgsl(noFeatures, { triangleLeaves: 'indexed', scene: { slot, layout: { normals: true, uvs: true, positions: true } } }).length);
  });

  it('declares no name twice, and nothing outside the rt prefix beyond the bvh walk', () => {
    const walk = new Set(declared(traceModuleWgsl({ triangleLeaves: 'indexed' })));
    const code = rtModuleWgsl(noFeatures, { triangleLeaves: 'indexed', scene: { slot, layout: { normals: true, uvs: true, positions: true } } });
    const names = declared(code);
    const seen = new Set<string>();
    for (const name of names) {
      expect(seen.has(name), `duplicate ${name}`).toBe(false);
      seen.add(name);
    }
    expect(names.filter((n) => !walk.has(n) && !/^(rt|Rt|RT_)/.test(n))).toEqual([]);
  });

  it('can leave out materials, lights or surface', () => {
    const layout = { normals: true, uvs: false, positions: false };
    const noMaterials = rtModuleWgsl(noFeatures, { scene: { slot, layout, materials: false } });
    expect(noMaterials).not.toContain('fn rtMaterial(');
    expect(noMaterials).toContain('fn rtSampleLight(');
    const noLights = rtModuleWgsl(noFeatures, { scene: { slot, layout, lights: false } });
    expect(noLights).not.toContain('fn rtSampleLight(');
    expect(noLights).toContain('fn rtMaterial(');
    const noSurface = rtModuleWgsl(noFeatures, { scene: { slot, layout, surface: false } });
    expect(noSurface).not.toContain('fn rtSurface(');
    expect(noSurface).toContain('fn rtMaterial(');
    expect(noSurface).toContain('fn rtLightIndexOfHit(');
  });

  it('RtMaterial has exactly the TypeScript material\'s fields', () => {
    expect(fieldsOf(rtMaterialWgsl(), 'RtMaterial').sort()).toEqual(Object.keys(defaultMaterial()).sort());
  });

  it('light structs name the fields the docs promise', () => {
    const code = rtLightsWgsl();
    expect(fieldsOf(code, 'RtLight')).toEqual(['kind', 'position', 'edge1', 'edge2', 'normal', 'area', 'pmf', 'radiance', 'material', 'instance', 'triangle', 'twoSided']);
    expect(fieldsOf(code, 'RtLightSample')).toEqual(['index', 'pmf', 'u']);
    expect(fieldsOf(code, 'RtSun')).toEqual(['index', 'direction', 'angularRadius', 'cosAngularRadius', 'solidAngle', 'irradiance', 'pmf']);
    expect(fieldsOf(code, 'RtEnvironment')).toEqual(['index', 'radiance', 'scale', 'rotationY', 'mapSlot', 'pmf']);
  });

  it('header words the shader reads by position are where the TypeScript header puts them', () => {
    // rtSceneInfo reads rtSceneTable[0].x, [1].x/.z, [2].x, [3].x/.y; rtMaterial [1].x/.y; lights [1].z/.w and [2].y/.z.
    const vec4 = (word: number) => [Math.floor(word / 4), word % 4];
    const at = (word: number) => `${vec4(word)[0]}.${'xyzw'[vec4(word)[1]!]}`;
    expect([at(SCENE_HEADER.instanceSlots), at(SCENE_HEADER.materialCount), at(SCENE_HEADER.materialBase), at(SCENE_HEADER.lightCount), at(SCENE_HEADER.lightBase), at(SCENE_HEADER.triangleLightCount), at(SCENE_HEADER.sunLight), at(SCENE_HEADER.envLight), at(SCENE_HEADER.totalPower), at(SCENE_HEADER.sceneRadius)]).toEqual([
      '0.x', '1.x', '1.y', '1.z', '1.w', '2.x', '2.y', '2.z', '3.x', '3.y',
    ]);
    const info = rtSceneBindingsWgsl({ group: 0, firstBinding: 0 });
    expect(info).toContain('rtSceneTable[0].x, h1.x, h1.z, h2.x, bitcast<f32>(h3.x), bitcast<f32>(h3.y)');
    expect(rtLightsWgsl()).toContain('return rtSceneTable[1].z;');
    expect(rtLightsWgsl()).toContain('rtSceneTable[2].y');
    expect(rtLightsWgsl()).toContain('rtSceneTable[2].z');
  });

  it('binding budget: scene copy 2, walk 3 (4 indexed)', () => {
    expect(SCENE_COPY_STORAGE_BUFFERS).toBe(2);
    expect(rtSceneEntries()).toHaveLength(2);
    const scene = { slot, layout: { normals: true, uvs: true, positions: true } };
    expect(rtModuleStorageBuffers({ scene })).toBe(5);
    expect(rtModuleStorageBuffers({ triangleLeaves: 'indexed', scene })).toBe(6);
    expect(rtSceneWgsl({ group: 0, firstBinding: 4 })).toContain('@group(0) @binding(5)');
  });
});
