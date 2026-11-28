import { describe, expect, it } from 'vitest';
import { FLY_MAX_BACKLOG, FlyCamera, flyInputFromCodes } from './fly-camera.ts';

const W = flyInputFromCodes(new Set(['KeyW']));

/** Holds `input` for `seconds`, delivered in animation frames of `frameMs`. */
function run(frameMs: number, seconds: number, input = W): FlyCamera {
  const cam = new FlyCamera();
  cam.speed = 2;
  const frames = Math.round((seconds * 1000) / frameMs);
  for (let i = 0; i < frames; i++) cam.advance(frameMs / 1000, input);
  return cam;
}

describe('FlyCamera', () => {
  it('covers the same distance whatever the frame rate', () => {
    const fast = run(1000 / 120, 2).position[0];
    const slow = run(1000 / 12, 2).position[0];
    const uneven = (() => {
      const cam = new FlyCamera();
      cam.speed = 2;
      let t = 0;
      for (let i = 0; t < 2; i++) {
        const dt = Math.min(2 - t, i % 3 === 0 ? 0.09 : 0.011);
        cam.advance(dt, W);
        t += dt;
      }
      return cam.position[0];
    })();
    // Only a fraction of one 1/120 s step can differ (the backlog a frame leaves behind).
    expect(Math.abs(fast - slow)).toBeLessThan(2 / 120 + 1e-9);
    expect(Math.abs(fast - uneven)).toBeLessThan(2 / 120 + 1e-9);
    // 2 m/s for 2 s, less the ease-in (speed x the response time).
    expect(fast).toBeGreaterThan(3.7);
    expect(fast).toBeLessThan(4);
  });

  it('comes to an exact rest after the keys are released', () => {
    const cam = run(1000 / 60, 1);
    for (let i = 0; i < 60; i++) cam.advance(1 / 60);
    expect(cam.moving).toBe(false);
    const a = JSON.stringify(cam.camera);
    expect(cam.advance(1 / 60)).toBe(false);
    expect(JSON.stringify(cam.camera)).toBe(a);
  });

  it('does not jump after a stall or after idle time', () => {
    const cam = new FlyCamera();
    cam.advance(30); // idle for 30 s
    cam.advance(1 / 60, W);
    expect(cam.position[0]).toBeLessThan(0.01);
    cam.advance(10, W); // a 10 s stall with the key down
    expect(cam.position[0]).toBeLessThan(FLY_MAX_BACKLOG * 1.01 + 0.01);
  });

  it('moves a diagonal at the same speed as an axis, and opposite keys cancel', () => {
    const diag = run(1000 / 60, 2, flyInputFromCodes(new Set(['KeyW', 'KeyD'])));
    const axis = run(1000 / 60, 2);
    expect(Math.hypot(diag.position[0], diag.position[2])).toBeCloseTo(axis.position[0], 6);
    expect(flyInputFromCodes(new Set(['KeyW', 'KeyS', 'ArrowLeft']))).toMatchObject({ forward: 0, right: -1, up: 0, boost: false });
  });

  it('round-trips a look-at camera', () => {
    const cam = new FlyCamera();
    cam.setPose({ position: [1, 2, 3], target: [1.5, 2.2, 2], up: [0, 1, 0], yFovDeg: 40 });
    const c = cam.camera;
    const d = [0.5, 0.2, -1].map((v) => v / Math.hypot(0.5, 0.2, 1));
    c.target.forEach((v, i) => expect(v - c.position[i]!).toBeCloseTo(d[i]!, 6));
    expect(c.yFovDeg).toBe(40);
  });
});
