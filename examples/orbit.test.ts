import { describe, expect, it } from 'vitest';
import { boundsCenter, boundsRadius, clampDistance, clampTarget, fitDistance, pivotOnSubject } from './shared/orbit-math.ts';

const cam = (position: [number, number, number], target: [number, number, number]) => ({ position, target, up: [0, 1, 0] as [number, number, number], yFovDeg: 40 });

describe('pivotOnSubject', () => {
  it('moves a glTF camera target (one unit ahead) to the model along the same view axis', () => {
    // Tiny model at (3, 0.5, -10), camera at the origin looking down -Z.
    const bounds = { min: [2.9, 0.4, -10.1], max: [3.1, 0.6, -9.9] };
    const out = pivotOnSubject(cam([0, 0, 0], [0, 0, -1]), bounds);
    expect(out.position).toEqual([0, 0, 0]);
    expect(out.target[2]).toBeCloseTo(-10, 6);
    expect(out.target[0]).toBeCloseTo(0, 6);
    expect(out.target[1]).toBeCloseTo(0, 6);
  });
  it('falls back to the bounds centre when the model is behind the camera', () => {
    const bounds = { min: [-1, -1, 4], max: [1, 1, 6] };
    expect(pivotOnSubject(cam([0, 0, 0], [0, 0, -1]), bounds).target).toEqual(boundsCenter(bounds));
  });
  it('keeps a target that is already on the subject', () => {
    const bounds = { min: [-1, -1, -1], max: [1, 1, 1] };
    const out = pivotOnSubject(cam([0, 0, 5], [0, 0, 0]), bounds);
    expect(out.target[2]).toBeCloseTo(0, 6);
  });
});

describe('limits', () => {
  it('scales with the subject, not an absolute number', () => {
    expect(clampDistance(1e-9, 100)).toBeCloseTo(2);
    expect(clampDistance(1e-9, 0.1)).toBeCloseTo(0.002);
    expect(clampDistance(1e9, 1)).toBeCloseTo(50);
  });
  it('clamps a pan target near the subject', () => {
    expect(clampTarget([100, 0, 0], [0, 0, 0], 1)).toEqual([2, 0, 0]);
  });
  it('has a usable radius for an empty box and fits the field of view', () => {
    expect(boundsRadius({ min: [0, 0, 0], max: [0, 0, 0] })).toBe(1);
    expect(fitDistance(1, 40)).toBeCloseTo(1.05 / Math.sin(Math.PI / 9));
  });
});
