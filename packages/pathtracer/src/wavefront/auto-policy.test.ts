import { describe, expect, it } from 'vitest';
import { ptwfAutoLayers, ptwfAutoPolicy } from './auto-policy.ts';

describe('wavefront auto policy (P3.9)', () => {
  it('short paths: megakernel whatever the film', () => {
    expect(ptwfAutoPolicy({ width: 256, height: 256, maxDepth: 1, subgroups: true })).toMatchObject({ scheduler: 'megakernel', refill: false });
    expect(ptwfAutoPolicy({ width: 1920, height: 1080, maxDepth: 2, subgroups: true }).scheduler).toBe('megakernel');
  });

  it('big film: plain wavefront; mid film: inline refill, one layer; small film: layers up to 1-2 M slots', () => {
    expect(ptwfAutoPolicy({ width: 1280, height: 720, maxDepth: 8, subgroups: true })).toMatchObject({ scheduler: 'wavefront', refill: false, layers: 1, appends: 'subgroup', sort: 'none' });
    expect(ptwfAutoPolicy({ width: 512, height: 512, maxDepth: 32, subgroups: false })).toMatchObject({ scheduler: 'wavefront', refill: 'inline', layers: 1, appends: 'atomic' });
    const small = ptwfAutoPolicy({ width: 384, height: 256, maxDepth: 16, subgroups: true });
    expect(small).toMatchObject({ refill: 'inline', layers: 16 });
    expect(384 * 256 * small.layers).toBeGreaterThan(1 << 20);
    expect(384 * 256 * small.layers).toBeLessThanOrEqual(1 << 21);
  });

  it('layers: monotone in the film, 1..32', () => {
    let last = 33;
    for (const pixels of [1000, 10_000, 50_000, 98_304, 150_000, 262_144, 1_000_000]) {
      const l = ptwfAutoLayers(pixels);
      expect(l).toBeGreaterThanOrEqual(1);
      expect(l).toBeLessThanOrEqual(32);
      expect(l).toBeLessThanOrEqual(last);
      last = l;
    }
    expect(() => ptwfAutoPolicy({ width: 0, height: 4, maxDepth: 4, subgroups: true })).toThrow(RangeError);
  });
});
