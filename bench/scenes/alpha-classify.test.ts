import { describe, expect, it } from 'vitest';
import { alphaCoverageOf, classifyAlphaCards, classifyTriangle, INSTANCE_MASK_MIXED, type AlphaLevel } from './alpha-classify.ts';
import { getScene } from './index.ts';
import { sceneDescToCopyInput } from './to-scene-copy.ts';

/** 4 x 1 texture: alpha 1, 1, 0, 0 (contour at x = 2 texel boundary). */
const level: AlphaLevel = { width: 4, height: 1, wrap: 'clamp', at: (i) => (Math.min(3, Math.max(0, i)) < 2 ? 1 : 0) };
const tri = (x0: number, x1: number): number[] => [x0 / 4, 0, x1 / 4, 0, x0 / 4, 1];

describe('alpha classification (R2b)', () => {
  it('a triangle is pure when it stays on one side of the bilinear contour, mixed when it crosses it', () => {
    expect(classifyTriangle(tri(0.2, 1.4), level, 1, 0.5)).toBe('solid');
    expect(classifyTriangle(tri(2.6, 3.8), level, 1, 0.5)).toBe('transparent');
    expect(classifyTriangle(tri(1.2, 2.8), level, 1, 0.5)).toBe('mixed');
    // touching the contour along an edge (x = 2) is zero-measure contact
    expect(classifyTriangle(tri(0.5, 2), level, 1, 0.5)).toBe('solid');
    expect(classifyTriangle(tri(2, 3.5), level, 1, 0.5)).toBe('transparent');
  });

  it('foliage-mask cards (2 triangles over a 6 x 6 checker) are all mixed; the aligned grid has none mixed; coverage is 0.5', () => {
    const plain = sceneDescToCopyInput(getScene('foliage-mask'));
    expect(classifyAlphaCards(plain).stats.every((s) => s.mixed === s.triangles)).toBe(true);
    expect(alphaCoverageOf(plain)).toEqual([1, 0.5]);
    const grid = classifyAlphaCards(sceneDescToCopyInput(getScene('foliage-mask-grid16')));
    expect(grid.stats.reduce((a, s) => a + s.mixed, 0)).toBe(0);
    expect(grid.stats.reduce((a, s) => a + s.solid, 0)).toBe(grid.stats.reduce((a, s) => a + s.transparent, 0));
    // solid halves moved to an opaque twin; transparent triangles dropped
    const nSolid = grid.stats.reduce((a, s) => a + s.solid, 0);
    const total = grid.input.instances.reduce((a, i) => a + (grid.input.meshes[i.meshIndex]!.indices!.length / 3), 0);
    expect(total - 2 * 0).toBeGreaterThanOrEqual(nSolid);
    expect(grid.input.materialNames).toContain('leaf+solid');
    expect(grid.input.instances.some((i) => i.mask === INSTANCE_MASK_MIXED)).toBe(false);
  });

  it('a partly mixed grid keeps its mixed triangles in instances with the mixed mask', () => {
    const r = classifyAlphaCards(sceneDescToCopyInput(getScene('foliage-mask-grid12')));
    const mixed = r.stats.reduce((a, s) => a + s.mixed, 0);
    expect(mixed).toBeGreaterThan(0);
    const inMixed = r.input.instances.filter((i) => i.mask === INSTANCE_MASK_MIXED).reduce((a, i) => a + r.input.meshes[i.meshIndex]!.indices!.length / 3, 0);
    expect(inMixed).toBe(mixed);
  });
});
