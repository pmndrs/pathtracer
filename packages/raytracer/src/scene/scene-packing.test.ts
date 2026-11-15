/**
 * Scene copy packing (P3.11): the packed buffer's layout, the WGSL rewrite, the binding numbers and the guarantee that the split text is untouched.
 * The GPU side (every accessor bit-exact on both packings) is `--page scene-copy`.
 */
import { describe, expect, it } from 'vitest';
import { buildCaps } from '../caps/index.ts';
import { rtModuleStorageBuffers, rtModuleWgsl, rtSceneEntries } from '../trace/index.ts';
import { DEFAULT_SCENE_PACKING, PACKED_HEADER, PACKED_TABLE_BASE_VEC4S, sceneCopyStorageBuffers } from './constants.ts';
import { buildSceneCopy, packSceneCopy, packedSceneCopyBytes, readPackedDataWord, readPackedTableVec4 } from './scene-copy.ts';
import { rtLightsWgsl, rtPackSceneAccess, rtSceneBufferBindings, rtSceneWgsl } from './scene-wgsl.ts';
import { balanced } from '../wgsl-names.testutil.ts';

const caps = buildCaps({ features: new Set(['subgroups']), limits: {} }, new Set());
const slot = { group: 1, firstBinding: 0 };
const layout = { normals: true, uvs: true, positions: true };
const textures = { buckets: [{ width: 8, height: 8, colorSpace: 'srgb' as const, wrap: 'repeat' as const, layers: 1, mipLevels: 4 }, { width: 8, height: 8, colorSpace: 'linear' as const, wrap: 'clamp' as const, layers: 1, mipLevels: 4 }] };

function scene() {
  // 3 triangles (9 index words, odd data length on purpose), a sun and an environment.
  return buildSceneCopy({
    meshes: [{ positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]), indices: new Uint32Array([0, 1, 2, 0, 2, 3, 1, 2, 3]), normals: new Float32Array(12).fill(0.5), uvs: new Float32Array(8) }],
    instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: 0 }],
    materials: [{ emissionColor: [1, 2, 3] }],
    sun: { direction: [0, -1, 0], irradiance: [1, 1, 1] },
  });
}

describe('packed scene copy layout', () => {
  const copy = scene();
  const packed = packSceneCopy(copy);

  it('is [header | table | data padded to whole vec4s], 16-byte aligned, table and data unchanged', () => {
    expect(packed.words.length % 4).toBe(0);
    expect(packed.byteLength).toBe(packedSceneCopyBytes(copy));
    expect(packed.tableBase).toBe(PACKED_TABLE_BASE_VEC4S);
    expect(packed.dataBase).toBe(PACKED_TABLE_BASE_VEC4S + copy.table.length / 4);
    expect(copy.data.length % 4).not.toBe(0); // the fixture exercises the padding
    expect(packed.byteLength).toBe(16 + copy.table.byteLength + Math.ceil(copy.data.length / 4) * 16);
    expect(Array.from(packed.words.subarray(PACKED_HEADER.dataBase, 4))).toEqual([packed.dataBase, packed.tableBase, copy.data.length, packed.words.length / 4]);
    expect(Array.from(packed.words.subarray(packed.tableBase * 4, packed.dataBase * 4))).toEqual(Array.from(copy.table));
    expect(Array.from(packed.words.subarray(packed.dataBase * 4, packed.dataBase * 4 + copy.data.length))).toEqual(Array.from(copy.data));
    expect(packed.words.subarray(packed.dataBase * 4 + copy.data.length).every((w) => w === 0)).toBe(true); // the padding is zero
  });

  it('the CPU twins of rtScene[i + 1] and rtSceneWord(i) read every word back', () => {
    for (let i = 0; i < copy.data.length; i++) expect(readPackedDataWord(packed.words, i)).toBe(copy.data[i]);
    for (let i = 0; i < copy.table.length / 4; i++) expect(Array.from(readPackedTableVec4(packed.words, i))).toEqual(Array.from(copy.table.subarray(i * 4, i * 4 + 4)));
  });

  it('counts the storage buffers: 2 split, 1 packed', () => {
    expect(sceneCopyStorageBuffers('split')).toBe(2);
    expect(sceneCopyStorageBuffers('packed')).toBe(1);
    expect(rtSceneBufferBindings('split')).toBe(2);
    expect(rtSceneBufferBindings('packed')).toBe(1);
    expect(rtSceneEntries('split')).toHaveLength(2);
    expect(rtSceneEntries('packed')).toHaveLength(1);
    expect(rtModuleStorageBuffers({ scene: { slot, layout, packing: 'packed' } })).toBe(rtModuleStorageBuffers({ scene: { slot, layout, packing: 'split' } }) - 1);
  });

  it('the library default stays split until the quiet-pass A/B (flip DEFAULT_SCENE_PACKING when it is in)', () => {
    expect(DEFAULT_SCENE_PACKING).toBe('split');
  });
});

describe('rtPackSceneAccess', () => {
  it('rewrites table and data reads, nested and with weak operators', () => {
    expect(rtPackSceneAccess('let a = rtSceneTable[2].z;')).toBe('let a = rtScene[3u].z;');
    expect(rtPackSceneAccess('let a = rtSceneTable[i];')).toBe('let a = rtScene[i + 1u];');
    expect(rtPackSceneAccess('let a = rtSceneTable[rtSceneTable[3].z + slot];')).toBe('let a = rtScene[rtScene[4u].z + slot + 1u];');
    expect(rtPackSceneAccess('let a = rtSceneTable[h1.y + slot * 8u + 7u].yzw;')).toBe('let a = rtScene[h1.y + slot * 8u + 7u + 1u].yzw;');
    expect(rtPackSceneAccess('let a = rtSceneTable[select(a, b, c < d)];')).toBe('let a = rtScene[select(a, b, c < d) + 1u];');
    expect(rtPackSceneAccess('let a = rtSceneTable[i & 3u];')).toBe('let a = rtScene[(i & 3u) + 1u];');
    expect(rtPackSceneAccess('x = bitcast<f32>(rtSceneData[base + (k - 1u) * stride]);')).toBe('x = bitcast<f32>(rtSceneWord(base + (k - 1u) * stride));');
    expect(rtPackSceneAccess('rtSceneData[rtSceneTable[record + 7u].x + triangle]')).toBe('rtSceneWord(rtScene[record + 7u + 1u].x + triangle)');
  });

  it('is idempotent and leaves a source with neither name alone', () => {
    const once = rtPackSceneAccess('let a = rtSceneTable[rtSceneTable[3].z + slot]; let b = rtSceneData[i];');
    expect(rtPackSceneAccess(once)).toBe(once);
    expect(rtPackSceneAccess('fn f() {}')).toBe('fn f() {}');
    expect(() => rtPackSceneAccess('rtSceneTable[i')).toThrow(/unbalanced/);
  });
});

describe('packed WGSL', () => {
  const split = (extra = {}) => rtModuleWgsl(caps, { triangleLeaves: 'indexed', scene: { slot, layout, ...extra }, shade: true });

  it('declares one scene binding and the textures right after it; nothing of the split names is left', () => {
    const code = split({ packing: 'packed', textures });
    expect(balanced(code)).toBe(true);
    expect(code).toContain('@group(1) @binding(0) var<storage, read> rtScene: array<vec4u>;');
    expect(code).toContain('fn rtSceneWord(i: u32) -> u32');
    expect(code).not.toMatch(/rtSceneTable|rtSceneData/);
    expect(code).toContain('@group(1) @binding(1) var rtTex0: texture_2d_array<f32>;');
    expect(code).toContain('@group(1) @binding(3) var rtSmp0: sampler;');
    expect(code).toContain('rtScene[rtScene[4u].z + slot + 1u]');
    const bindings = [...code.matchAll(/@group\(1\)\s*@binding\((\d+)\)/g)].map((m) => Number(m[1]));
    expect(bindings).toEqual([0, 1, 2, 3, 4]);
  });

  it('keeps the split text: nothing changes unless packing is asked for', () => {
    const plain = split();
    expect(split({ packing: 'split' })).toBe(plain);
    expect(plain).toContain('@group(1) @binding(1) var<storage, read> rtSceneData: array<u32>;');
    expect(plain).not.toMatch(/rtSceneWord|rtScene\[/);
    expect(rtSceneWgsl({ group: 0, firstBinding: 0 })).toBe(rtSceneWgsl({ group: 0, firstBinding: 0, packing: 'split' }));
    expect(rtLightsWgsl()).toContain('rtSceneTable[1].z');
  });

  it('stays within a few hundred bytes of the split text (the cost of the data addressing)', () => {
    const a = split().length;
    const b = split({ packing: 'packed' }).length;
    expect(Math.abs(b - a)).toBeLessThan(1_500);
  });
});
