import { describe, expect, it } from 'vitest';
import { meshToWorld, surfaceArea } from '../geometry.ts';
import { ENV_HEIGHT, ENV_WIDTH, SUN_DIR, equirectDirection, skyRadiance } from './env-sky.ts';
import { getTier0Scene, tier0Scenes } from './index.ts';

describe('tier0 scenes', () => {
  it('are well formed: unique names, known materials, honest purposes', () => {
    const names = new Set<string>();
    for (const s of tier0Scenes) {
      expect(names.has(s.name)).toBe(false);
      names.add(s.name);
      expect(s.tier).toBe(0);
      expect(s.purpose.length).toBeGreaterThan(30);
      for (const m of s.meshes) expect(s.materials[m.material]).toBeDefined();
      if (s.integrator.rrStartDepth === undefined) expect(s.integrator.maxDepth).toBeGreaterThan(0);
    }
    expect(getTier0Scene('cornell').film).toEqual({ width: 512, height: 512, filter: 'box' });
  });

  it('furnace frames: close camera sees only the sphere surface', () => {
    const s = getTier0Scene('furnace-diffuse');
    const half = (s.camera.yFovDeg * Math.PI) / 360;
    const cornerAngle = Math.atan(Math.tan(half) * Math.SQRT2);
    const dist = Math.hypot(...s.camera.position);
    expect(cornerAngle).toBeLessThan(Math.asin(1 / dist) * 0.95);
  });

  it('cornell: light faces down and sits inside the room', () => {
    const s = getTier0Scene('cornell');
    const light = s.meshes.find((m) => m.emission)!;
    const w = meshToWorld(light);
    // geometric normal of the first triangle points -Y
    const p = (i: number) => [w.positions[i * 3]!, w.positions[i * 3 + 1]!, w.positions[i * 3 + 2]!];
    const [a, b, c] = [p(w.indices[0]!), p(w.indices[1]!), p(w.indices[2]!)];
    const ny = (b[2]! - a[2]!) * (c[0]! - a[0]!) - (b[0]! - a[0]!) * (c[2]! - a[2]!);
    expect(ny).toBeLessThan(0);
    expect(surfaceArea(w)).toBeCloseTo(0.13 * 0.105, 5);
  });

  it('quad light faces down', () => {
    const light = getTier0Scene('quad-light').meshes.find((m) => m.emission)!;
    const w = meshToWorld(light);
    for (let i = 1; i < w.normals.length; i += 3) expect(w.normals[i]).toBeCloseTo(-1, 6);
  });
});

describe('env sky', () => {
  it('peak lies at the sun direction and equirect direction maps u=0 to -Z, u=0.25 to +X, v=0 to +Y', () => {
    const d0 = equirectDirection(0, 0.5);
    expect(d0[2]).toBeCloseTo(-1, 6);
    const d1 = equirectDirection(0.25, 0.5);
    expect(d1[0]).toBeCloseTo(1, 6);
    expect(equirectDirection(0.3, 0)[1]).toBeCloseTo(1, 6);
    expect(skyRadiance(SUN_DIR)[0]).toBeGreaterThan(skyRadiance([-SUN_DIR[0], SUN_DIR[1], -SUN_DIR[2]])[0] * 20);
    expect(ENV_WIDTH).toBe(2 * ENV_HEIGHT);
  });
});
