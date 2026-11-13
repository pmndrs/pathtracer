import { describe, expect, it } from 'vitest';
import {
  compose, cross3, determinant3, identity, meshToWorld, multiply, rotationX, rotationY, scaling, surfaceArea,
  tessellate, tessellateBox, tessellateQuad, tessellateSphere, transformDirection, transformPoint, translation,
  unweld, lookAtCameraToWorld,
} from './geometry.ts';
import type { TriangleMesh } from './geometry.ts';

/** Geometric normal of each triangle dotted with (centroid − origin): > 0 means outward-facing for convex shapes. */
function outwardness(m: TriangleMesh): number[] {
  const out: number[] = [];
  for (let t = 0; t < m.indices.length; t += 3) {
    const p = [0, 1, 2].map((k) => {
      const i = m.indices[t + k]!;
      return [m.positions[i * 3]!, m.positions[i * 3 + 1]!, m.positions[i * 3 + 2]!] as [number, number, number];
    });
    const n = cross3(
      [p[1]![0] - p[0]![0], p[1]![1] - p[0]![1], p[1]![2] - p[0]![2]],
      [p[2]![0] - p[0]![0], p[2]![1] - p[0]![1], p[2]![2] - p[0]![2]],
    );
    const c = [0, 1, 2].map((k) => (p[0]![k]! + p[1]![k]! + p[2]![k]!) / 3);
    out.push(n[0] * c[0]! + n[1] * c[1]! + n[2] * c[2]!);
  }
  return out;
}

describe('tessellation', () => {
  it('quad: +Y normal, area = size product', () => {
    const q = tessellateQuad([2, 3]);
    expect(surfaceArea(q)).toBeCloseTo(6, 6);
    const n = cross3(
      [q.positions[3]! - q.positions[0]!, 0, 0],
      [q.positions[6]! - q.positions[0]!, 0, q.positions[8]! - q.positions[2]!],
    );
    expect(n[1]).toBeGreaterThan(0);
  });

  it('box: 12 triangles, outward winding, exact area', () => {
    const b = tessellateBox([1, 2, 3]);
    expect(b.indices.length / 3).toBe(12);
    expect(surfaceArea(b)).toBeCloseTo(2 * (2 + 3 + 6), 6);
    for (const o of outwardness(b)) expect(o).toBeGreaterThan(0);
  });

  it('sphere: triangle count, outward winding, area and volume converge', () => {
    const s = tessellateSphere(1, 64, 32);
    expect(s.indices.length / 3).toBe(2 * 64 * (32 - 1));
    expect(s.positions.length / 3).toBe(2 + 64 * 31);
    for (const o of outwardness(s)) expect(o).toBeGreaterThan(0);
    expect(surfaceArea(s)).toBeGreaterThan(4 * Math.PI * 0.99);
    expect(surfaceArea(s)).toBeLessThan(4 * Math.PI);
    for (let i = 0; i < s.positions.length; i += 3) {
      expect(Math.hypot(s.positions[i]!, s.positions[i + 1]!, s.positions[i + 2]!)).toBeCloseTo(1, 6);
    }
  });

  it('is deterministic', () => {
    const a = tessellate({ type: 'sphere', radius: 2, segments: 17, rings: 9 });
    const b = tessellate({ type: 'sphere', radius: 2, segments: 17, rings: 9 });
    expect(Buffer.from(a.positions.buffer).equals(Buffer.from(b.positions.buffer))).toBe(true);
    expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
  });

  it('unweld gives flat per-triangle normals', () => {
    const f = unweld(tessellateSphere(1, 8, 4));
    expect(f.positions.length / 3).toBe(f.indices.length);
    for (let i = 0; i < f.normals.length; i += 9) {
      expect(f.normals[i]).toBeCloseTo(f.normals[i + 3]!, 6);
      expect(f.normals[i + 1]).toBeCloseTo(f.normals[i + 4]!, 6);
    }
  });
});

describe('transforms', () => {
  it('translation/rotation/scale compose right-to-left', () => {
    const m = compose(translation(1, 2, 3), rotationY(Math.PI / 2), scaling(2));
    const p = transformPoint(m, [1, 0, 0]);
    // scale -> (2,0,0); rotY(90deg) maps +X to -Z; then translate.
    expect(p[0]).toBeCloseTo(1, 6);
    expect(p[1]).toBeCloseTo(2, 6);
    expect(p[2]).toBeCloseTo(3 - 2, 6);
  });

  it('multiply with identity is a no-op; column-major layout', () => {
    const t = translation(4, 5, 6);
    expect(multiply(t, identity())).toEqual(t);
    expect(t[12]).toBe(4);
  });

  it('rotationX(pi) flips a quad to face down; direction ignores translation', () => {
    const m = compose(translation(0, 1, 0), rotationX(Math.PI));
    const d = transformDirection(m, [0, 1, 0]);
    expect(d[1]).toBeCloseTo(-1, 6);
  });

  it('mirrored transforms keep triangles outward-facing', () => {
    const w = meshToWorld({ name: 'm', geometry: { type: 'box', size: [1, 1, 1] }, material: 'x', transform: scaling(-1, 1, 1) });
    expect(determinant3(scaling(-1, 1, 1))).toBeLessThan(0);
    for (const o of outwardness(w)) expect(o).toBeGreaterThan(0);
  });

  it('meshToWorld moves vertices', () => {
    const w = meshToWorld({ name: 'm', geometry: { type: 'quad', size: [1, 1] }, material: 'x', transform: translation(0, 5, 0) });
    for (let i = 1; i < w.positions.length; i += 3) expect(w.positions[i]).toBeCloseTo(5, 6);
  });

  it('lookAt camera-to-world looks down -Z toward the target', () => {
    const m = lookAtCameraToWorld([0, 0, 4], [0, 0, 0], [0, 1, 0]);
    const f = transformDirection(m, [0, 0, -1]);
    expect(f[2]).toBeCloseTo(-1, 6);
    expect(transformPoint(m, [0, 0, 0])).toEqual([0, 0, 4]);
  });
});
