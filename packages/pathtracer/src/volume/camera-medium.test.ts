import { describe, expect, it } from 'vitest';
import { ptDetectCameraMedium, ptWindingNumber, type PtMediumBoundary } from './camera-medium.ts';

function box(h: number, medium: number, c: [number, number, number] = [0, 0, 0], flip = false): PtMediumBoundary {
  const positions: number[] = [];
  for (let i = 0; i < 8; i++) positions.push(c[0] + (i & 1 ? h : -h), c[1] + (i & 2 ? h : -h), c[2] + (i & 4 ? h : -h));
  // faces as quads (outward wound), split into triangles
  const quads = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];
  const indices: number[] = [];
  for (const [a, b, c2, d] of quads) indices.push(a!, b!, c2!, a!, c2!, d!);
  if (flip) for (let t = 0; t < indices.length; t += 3) [indices[t + 1], indices[t + 2]] = [indices[t + 2]!, indices[t + 1]!];
  return { medium, positions, indices };
}
function sphere(r: number, medium: number, n = 24): PtMediumBoundary {
  const positions: number[] = [0, r, 0, 0, -r, 0];
  const indices: number[] = [];
  const rings = n / 2;
  for (let i = 1; i < rings; i++) for (let j = 0; j < n; j++) {
    const th = (i / rings) * Math.PI, ph = (j / n) * 2 * Math.PI;
    positions.push(r * Math.sin(th) * Math.cos(ph), r * Math.cos(th), r * Math.sin(th) * Math.sin(ph));
  }
  const at = (i: number, j: number): number => 2 + (i - 1) * n + (j % n);
  for (let j = 0; j < n; j++) indices.push(0, at(1, j + 1), at(1, j)); // outward: north cap
  for (let i = 1; i < rings - 1; i++) for (let j = 0; j < n; j++) indices.push(at(i, j), at(i, j + 1), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j));
  for (let j = 0; j < n; j++) indices.push(1, at(rings - 1, j), at(rings - 1, j + 1));
  return { medium, positions, indices };
}

describe('camera medium detection (V3.3)', () => {
  it('winding number of a box: 1 inside, 0 outside, -1 for an inward-wound one', () => {
    const b = box(1, 1);
    expect(ptWindingNumber([0.1, 0.2, -0.3], b.positions, b.indices)).toBeCloseTo(1, 6);
    expect(ptWindingNumber([2, 0, 0], b.positions, b.indices)).toBeCloseTo(0, 6);
    const f = box(1, 1, [0, 0, 0], true);
    expect(ptWindingNumber([0, 0, 0], f.positions, f.indices)).toBeCloseTo(-1, 6);
  });
  it('a point on an axis-aligned face plane through a vertex row (ray-parity degenerate) still resolves', () => {
    const b = box(1, 1);
    expect(ptDetectCameraMedium([0, 0, 0], [b])).toBe(1);
    expect(ptDetectCameraMedium([0, 1.5, 1], [b])).toBe(0);
  });
  it('a sphere', () => {
    const s = sphere(1, 3);
    expect(ptDetectCameraMedium([0.3, 0.3, 0.3], [s])).toBe(3);
    expect(ptDetectCameraMedium([0.9, 0.9, 0], [s])).toBe(0);
  });
  it('a point outside every boundary is vacuum', () => {
    expect(ptDetectCameraMedium([5, 5, 5], [box(1, 1), sphere(2, 2)])).toBe(0);
  });
  it('nested boxes: the innermost wins, whatever the order', () => {
    const outer = box(4, 1), inner = box(1, 2);
    expect(ptDetectCameraMedium([0, 0, 0], [outer, inner])).toBe(2);
    expect(ptDetectCameraMedium([0, 0, 0], [inner, outer])).toBe(2);
    expect(ptDetectCameraMedium([2, 0, 0], [outer, inner])).toBe(1);
    expect(ptDetectCameraMedium([6, 0, 0], [outer, inner])).toBe(0);
  });
  it('separate boxes and a vacuum boundary (medium 0) is ignored', () => {
    expect(ptDetectCameraMedium([10, 0, 0], [box(1, 1), box(1, 2, [10, 0, 0])])).toBe(2);
    expect(ptDetectCameraMedium([0, 0, 0], [box(1, 0)])).toBe(0);
  });
});
