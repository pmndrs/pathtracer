import { describe, expect, it } from 'vitest';
import { ptPhaseLobes } from './phase.ts';
import { brickMajorants, ptMediumFrom, ptVolumeGridWithMajorant } from './volume.ts';

/** V6.6 API ergonomics: a phase function in `ptMediumFrom`, and `volumeGrid` building its own brick majorants. */
describe('V6.6 API ergonomics', () => {
  const sa = [0.1, 0.2, 0.3] as const, ss = [1, 2, 3] as const;
  it('ptMediumFrom with a phase object equals the manual medium', () => {
    const lobes = ptPhaseLobes({ type: 'mie', diameter: 20 });
    expect(ptMediumFrom(sa, ss, { type: 'mie', diameter: 20 }, 2)).toEqual({ ...ptMediumFrom(sa, ss, lobes.gA, 2), lobes });
    const hg2 = ptPhaseLobes({ type: 'hg2', g1: 0.7, g2: -0.3, w2: 0.2 });
    expect(ptMediumFrom(sa, ss, { type: 'hg2', g1: 0.7, g2: -0.3, w2: 0.2 })).toEqual({ ...ptMediumFrom(sa, ss, 0.7), lobes: hg2 });
    expect(ptMediumFrom(sa, ss, { type: 'hgDraine', gHG: 0.5, gD: 0.9, alpha: 5, wD: 0.3 }).lobes).toEqual({ gA: 0.5, gB: 0.9, w: 0.3, alpha: 5 });
  });
  it("ptMediumFrom({ type: 'hg' }) and a plain number are the same medium (no lobes)", () => {
    expect(ptMediumFrom(sa, ss, { type: 'hg', g: 0.4 })).toEqual(ptMediumFrom(sa, ss, 0.4));
    expect(ptMediumFrom(sa, ss, 0.4).lobes).toBeUndefined();
    expect(ptMediumFrom(sa, ss)).toEqual(ptMediumFrom(sa, ss, 0));
  });
  it("volumeGrid gets its brick majorants (and minima for residual / telescoping) when majorant is 'brick'", () => {
    const resolution = [20, 12, 9] as const;
    const density = new Float32Array(20 * 12 * 9).map((_, i) => ((i * 2654435761) >>> 0) / 4294967296);
    const grid: { resolution: readonly [number, number, number]; density: Float32Array; majorant?: unknown } = { resolution, density };
    const made = ptVolumeGridWithMajorant(grid, { majorant: 'brick' })!;
    const ref = brickMajorants(resolution, density);
    expect(made.majorant).toEqual(ref);
    expect((made.majorant as { minima?: Float32Array }).minima).toBeUndefined();
    expect(ptVolumeGridWithMajorant(grid, { majorant: 'brick', shadowTracking: 'residual' })!.majorant).toEqual(brickMajorants(resolution, density, { minima: true }));
    expect((ptVolumeGridWithMajorant(grid, { majorant: 'brick', shadowTracking: 'telescoping' })!.majorant as { minima?: Float32Array }).minima).toBeDefined();
    // untouched: no brick request, a caller's own majorant, no grid
    expect(ptVolumeGridWithMajorant(grid, { majorant: 'global' })).toBe(grid);
    expect(ptVolumeGridWithMajorant(grid, undefined)).toBe(grid);
    const own = { ...grid, majorant: ref };
    expect(ptVolumeGridWithMajorant(own, { majorant: 'brick' })).toBe(own);
    expect(ptVolumeGridWithMajorant(undefined, { majorant: 'brick' })).toBeUndefined();
  });
});
