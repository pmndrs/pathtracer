import { describe, expect, it } from 'vitest';
import { ptCameraBasis, ptCameraDirection } from './camera.ts';

const cam = { position: [0, 0, 2] as const, target: [0, 0, 0] as const, up: [0, 1, 0] as const, yFovDeg: 40 };

describe('pinhole camera', () => {
  it('looks down -Z with +X to the right and +Y up, row 0 = top', () => {
    const b = ptCameraBasis(cam, 256, 256);
    expect(b.forward).toEqual([0, 0, -1]);
    expect(b.right[0]).toBeGreaterThan(0);
    expect(b.up[1]).toBeGreaterThan(0);
    const centre = ptCameraDirection(b, 256, 256, 128, 128);
    expect(centre[2]).toBeCloseTo(-1, 12);
    const topLeft = ptCameraDirection(b, 256, 256, 0, 0);
    expect(topLeft[0]).toBeLessThan(0);
    expect(topLeft[1]).toBeGreaterThan(0);
    const bottomRight = ptCameraDirection(b, 256, 256, 256, 256);
    expect(bottomRight[0]).toBeGreaterThan(0);
    expect(bottomRight[1]).toBeLessThan(0);
  });

  it('yFov is the vertical field of view; the aspect ratio widens x only', () => {
    const b = ptCameraBasis(cam, 512, 256);
    const top = ptCameraDirection(b, 512, 256, 256, 0);
    const angle = Math.atan2(top[1], -top[2]);
    expect(angle).toBeCloseTo((20 * Math.PI) / 180, 12);
    const left = ptCameraDirection(b, 512, 256, 0, 128);
    expect(Math.atan2(-left[0], -left[2])).toBeCloseTo(Math.atan(Math.tan((20 * Math.PI) / 180) * 2), 12);
  });

  it('a camera not on an axis keeps the basis orthonormal (before the field-of-view scale)', () => {
    const b = ptCameraBasis({ position: [3, 2, 1], target: [0, 0.5, 0], up: [0, 1, 0], yFovDeg: 60 }, 100, 100);
    const t = Math.tan(Math.PI / 6);
    const dot = (p: readonly number[], q: readonly number[]) => p[0]! * q[0]! + p[1]! * q[1]! + p[2]! * q[2]!;
    expect(Math.hypot(...b.forward)).toBeCloseTo(1, 12);
    expect(Math.hypot(...b.right)).toBeCloseTo(t, 12);
    expect(Math.hypot(...b.up)).toBeCloseTo(t, 12);
    expect(dot(b.forward, b.right)).toBeCloseTo(0, 12);
    expect(dot(b.forward, b.up)).toBeCloseTo(0, 12);
    expect(dot(b.right, b.up)).toBeCloseTo(0, 12);
  });

  it('refuses a degenerate camera', () => {
    expect(() => ptCameraBasis({ ...cam, target: [0, 0, 2] }, 8, 8)).toThrow(/zero-length/);
    expect(() => ptCameraBasis({ ...cam, up: [0, 0, 1] }, 8, 8)).toThrow(/zero-length/);
  });
});
