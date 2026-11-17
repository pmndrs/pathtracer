import { describe, expect, it } from 'vitest';
import { RESERVED, balanced, declaredIdentifiers } from '../../packages/raytracer/src/wgsl-names.testutil.ts';
import { DEFAULT_HASH_GRID, hashGridArgs, hashGridWgsl } from './hash-grid.ts';

describe('hash grid (M7)', () => {
  it('generates balanced WGSL with no reserved identifiers', () => {
    for (const [fb, pyramid] of [[0, false], [2, false], [2, true]] as const) {
      for (const [name, code] of Object.entries(hashGridWgsl(fb, pyramid))) {
        expect(balanced(code), name).toBe(true);
        expect([...declaredIdentifiers(code)].filter((n) => RESERVED.has(n)), name).toEqual([]);
      }
    }
  });

  it('bakes the fallback count and the bucket loop bounds in as constants (Tint loop counters on D3D12)', () => {
    const { lookup, update } = hashGridWgsl(3);
    expect(lookup).toContain('const HG_FALLBACK = 3u;');
    expect(lookup).toMatch(/for \(var i = 0u; i < 8u; i\+\+\)/);
    expect(update).toMatch(/for \(var i = 0u; i < 8u; i\+\+\)/);
  });

  it('maps page args onto the settings and leaves the rest at the defaults', () => {
    expect(hashGridArgs({})).toEqual({});
    expect(hashGridArgs({ hgMinSamples: '4', hgNormals: '0', hgCellPx: '16' })).toEqual({ minSamples: 4, normalBins: false, cellPixels: 16 });
    expect({ ...DEFAULT_HASH_GRID, ...hashGridArgs({ hgLog2: '22' }) }.log2Entries).toBe(22);
  });
});
