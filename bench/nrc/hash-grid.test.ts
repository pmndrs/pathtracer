import { describe, expect, it } from 'vitest';
import { RESERVED, balanced, declaredIdentifiers } from '../../packages/raytracer/src/wgsl-names.testutil.ts';
import { DEFAULT_HASH_GRID, hashGridArgs, hashGridStorage, hashGridWgsl } from './hash-grid.ts';

describe('hash grid (M7)', () => {
  it('generates balanced WGSL with no reserved identifiers', () => {
    for (const [fb, pyramid, active, variance] of [[0, false, false, false], [2, false, false, false], [2, true, false, false], [2, true, true, false], [2, true, false, true], [2, true, true, true]] as const) {
      for (const [name, code] of Object.entries(hashGridWgsl(fb, pyramid, active, variance))) {
        expect(balanced(code), name).toBe(true);
        expect([...declaredIdentifiers(code)].filter((n) => RESERVED.has(n)), name).toEqual([]);
      }
    }
    for (const active of [false, true]) {
      for (const variance of [false, true]) {
        for (const [name, code] of Object.entries(hashGridWgsl(2, true, active, variance, true, true))) {
          expect(balanced(code), name).toBe(true);
          expect([...declaredIdentifiers(code)].filter((n) => RESERVED.has(n)), name).toEqual([]);
        }
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
    expect(DEFAULT_HASH_GRID.activeMaintenance).toBe(false);
    expect(DEFAULT_HASH_GRID.directionalSh).toBe(false);
    expect(DEFAULT_HASH_GRID.varianceThreshold).toBe(-1);
    expect(hashGridArgs({ hgVariance: '0.1' })).toEqual({ varianceThreshold: 0.1 });
    expect(hashGridArgs({ hgActive: '1' })).toEqual({ activeMaintenance: true });
    expect(hashGridArgs({ hgActive: '0' })).toEqual({ activeMaintenance: false });
    expect(hashGridArgs({ hgMinSamples: '4', hgNormals: '0', hgCellPx: '16' })).toEqual({ minSamples: 4, normalBins: false, cellPixels: 16 });
    expect({ ...DEFAULT_HASH_GRID, ...hashGridArgs({ hgLog2: '22' }) }.log2Entries).toBe(22);
    expect(hashGridArgs({ hgDirectionalSh: '1', hgBudgetBytes: '10485844' })).toEqual({ directionalSh: true, memoryBudgetBytes: 10485844 });
  });

  it('fits all directional controls into the same explicit table allocation', () => {
    const memoryBudgetBytes = 10 * 1024 ** 2 + 84;
    const rgb = hashGridStorage({ memoryBudgetBytes });
    const bins = hashGridStorage({ memoryBudgetBytes, dirBins: 4 });
    const sh = hashGridStorage({ memoryBudgetBytes, directionalSh: true });
    expect(rgb).toEqual(bins);
    expect(rgb).toMatchObject({ entries: 262144, bytes: memoryBudgetBytes, bytesPerEntry: 40, updateAtomicAddsPerCell: 4 });
    expect(sh).toMatchObject({ entries: 40960, bytes: memoryBudgetBytes, bytesPerEntry: 256, directionalUpdateAtomicAddsPerCell: 21 });
    expect(() => hashGridStorage({ directionalSh: true, dirBins: 4 })).toThrow('mutually exclusive');
  });
});
