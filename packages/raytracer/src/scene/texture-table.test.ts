/**
 * P1.4c2: the resource-table texture binding (`TextureOptions.binding: 'table'`). Text and layout checks only; the GPU check is
 * `node bench/runner/run.ts --page bindless-check` (SwiftShader on a Mac, Chrome on Windows / Linux).
 */
import { describe, expect, it } from 'vitest';
import { buildCaps } from '../caps/index.ts';
import { rtModuleWgsl } from '../trace/index.ts';
import { rtSceneCopyLayoutEntries, rtUsesResourceTable } from './gpu.ts';
import { rtTextureTableEnable, rtTexturesWgsl } from './scene-wgsl.ts';
import { buildSceneTextures, textureLayoutOf, textureTableSize, type TextureInput } from './texture.ts';

const tex = (w: number, h: number, wrap: TextureInput['wrap'] = 'repeat', colorSpace: TextureInput['colorSpace'] = 'linear'): TextureInput => ({ width: w, height: h, data: new Uint8Array(w * h * 4).fill(128), colorSpace, wrap });

/** Ten distinct (size, colour space, wrap) groups: more than the 8 arrays the `arrays` binding allows. */
const tenGroups = (): TextureInput[] => [
  tex(4, 4), tex(8, 8), tex(16, 16), tex(32, 32), tex(64, 64),
  tex(4, 4, 'clamp'), tex(8, 8, 'clamp'), tex(16, 16, 'mirror'), tex(4, 4, 'repeat', 'srgb'), tex(8, 8, 'repeat', 'srgb'),
];

const noTable = buildCaps({ features: new Set(['subgroups']), limits: {} }, new Set());
const prefixed = buildCaps({ features: new Set(['subgroups', 'chromium-experimental-sampling-resource-table']), limits: {} }, new Set());
const unprefixed = buildCaps({ features: new Set(['sampling-resource-table']), limits: {} }, new Set());
const surface = { normals: true, uvs: true, positions: true } as const;

describe('texture binding: table', () => {
  it('never merges buckets (arrays merges ten groups down to eight)', () => {
    const arrays = buildSceneTextures(tenGroups(), { mips: 'cpu' });
    const table = buildSceneTextures(tenGroups(), { mips: 'cpu', binding: 'table' });
    expect(arrays.buckets.length).toBe(8);
    expect(table.buckets.length).toBe(10);
    // Merging pads textures up: the table stores fewer bytes.
    expect(table.byteLength).toBeLessThan(arrays.byteLength);
    expect(table.binding).toBe('table');
    expect('binding' in arrays).toBe(false);
  });

  it('gives the same buckets and slots as arrays when arrays did not merge', () => {
    const inputs = tenGroups().slice(0, 6);
    const a = buildSceneTextures(inputs, { mips: 'cpu' });
    const t = buildSceneTextures(inputs, { mips: 'cpu', binding: 'table' });
    expect(t.slots).toEqual(a.slots);
    expect(t.buckets.map((b) => [b.width, b.height, b.colorSpace, b.wrap, b.layers])).toEqual(a.buckets.map((b) => [b.width, b.height, b.colorSpace, b.wrap, b.layers]));
    t.buckets.forEach((b, i) => b.mips.forEach((m, l) => expect(Buffer.from(m).equals(Buffer.from(a.buckets[i]!.mips[l]!))).toBe(true)));
  });

  it('carries the binding in the layout only when it is a table (arrays layouts keep their shape)', () => {
    const t = textureLayoutOf(buildSceneTextures(tenGroups(), { binding: 'table' }));
    const a = textureLayoutOf(buildSceneTextures(tenGroups()));
    expect(t.binding).toBe('table');
    expect(Object.keys(a)).toEqual(['buckets']);
    expect(textureTableSize(t)).toBe(20);
    expect(rtUsesResourceTable(t)).toBe(true);
    expect(rtUsesResourceTable(a)).toBe(false);
    expect(rtUsesResourceTable(null)).toBe(false);
  });

  it('declares no texture bindings and looks up by bucket through getResource', () => {
    const layout = textureLayoutOf(buildSceneTextures(tenGroups(), { binding: 'table' }));
    const code = rtTexturesWgsl({ group: 1, firstBinding: 0, layout, packing: 'split' });
    expect(code).not.toMatch(/@binding/);
    expect(code).not.toMatch(/switch/);
    expect(code).toContain('getResource<texture_2d_array<f32>>(r.x)');
    expect(code).toContain('getResource<sampler>(10u + r.x)');
    const arrays = rtTexturesWgsl({ group: 1, firstBinding: 0, layout: textureLayoutOf(buildSceneTextures(tenGroups())), packing: 'split' });
    expect(arrays.match(/@binding/g)?.length).toBe(16);
  });

  it('adds no texture or sampler entries to the scene copy layout', () => {
    const t = textureLayoutOf(buildSceneTextures(tenGroups(), { binding: 'table' }));
    expect(rtSceneCopyLayoutEntries(0, t, 'split').length).toBe(2);
    expect(rtSceneCopyLayoutEntries(0, t, 'packed').length).toBe(1);
    expect(rtSceneCopyLayoutEntries(0, textureLayoutOf(buildSceneTextures(tenGroups())), 'split').length).toBe(2 + 16);
  });

  it('puts the enable line first, with the spelling of the feature the device has', () => {
    const layout = textureLayoutOf(buildSceneTextures([tex(4, 4)], { binding: 'table' }));
    const scene = { slot: { group: 1, firstBinding: 0 }, packing: 'split' as const, layout: surface, textures: layout };
    const code = rtModuleWgsl(prefixed, { scene });
    expect(code.split('\n')[0]).toBe('enable chromium_experimental_resource_table;');
    expect(code.split('\n')[1]).toBe('enable subgroups;');
    expect(rtModuleWgsl(unprefixed, { scene }).split('\n')[0]).toBe('enable resource_table;');
    expect(rtTextureTableEnable(undefined)).toBe('enable resource_table;');
    expect(() => rtModuleWgsl(noTable, { scene })).toThrow(/resource table/);
    // The same scene with arrays has no such line.
    expect(rtModuleWgsl(prefixed, { scene: { ...scene, textures: textureLayoutOf(buildSceneTextures([tex(4, 4)])) } })).not.toMatch(/resource_table/);
  });
});
