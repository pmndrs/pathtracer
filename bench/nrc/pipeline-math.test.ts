import { describe, expect, it } from 'vitest';
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { ptCameraBasis } from '@pmndrs/pathtracer';
import { PIPE_FAR, PIPE_NEAR, projectionMatrix, resample, viewMatrix, worldMatrix } from './pipeline-math.ts';

const cam: PtLookAtCamera = { position: [-8, 1.7, 0], target: [10, 2.6, 0], up: [0, 1, 0], yFovDeg: 60 };

const mul = (m: Float32Array, v: readonly number[]) => [0, 1, 2, 3].map((r) => m[r]! * v[0]! + m[4 + r]! * v[1]! + m[8 + r]! * v[2]! + m[12 + r]! * v[3]!);

describe('pipeline camera matrices (three.js conventions)', () => {
  it('view and world are inverses', () => {
    const v = viewMatrix(cam);
    const w = worldMatrix(cam);
    for (const p of [[1, 2, 3, 1], [-4, 0.5, 7, 1]]) {
      const back = mul(v, mul(w, p));
      for (let i = 0; i < 4; i++) expect(back[i]).toBeCloseTo(p[i]!, 4);
    }
  });

  it('projects like the kernel (ptCameraBasis: NDC = view-space ratio over tan(fov/2) x aspect) and writes the depth the upscaler linearises', () => {
    const w = 960;
    const h = 540;
    const b = ptCameraBasis(cam, w, h);
    const world = [2, 3.1, -1.4, 1];
    const view = mul(viewMatrix(cam), world);
    const clip = mul(projectionMatrix(cam.yFovDeg, w / h), view);
    const ndc = [clip[0]! / clip[3]!, clip[1]! / clip[3]!, clip[2]! / clip[3]!];
    // The kernel's own projection (ptProject in outputs-wgsl.ts).
    const d = [world[0]! - b.origin[0], world[1]! - b.origin[1], world[2]! - b.origin[2]];
    const dt = (a: readonly number[], c: readonly number[]) => a[0]! * c[0]! + a[1]! * c[1]! + a[2]! * c[2]!;
    const z = dt(d, b.forward);
    expect(ndc[0]).toBeCloseTo(dt(d, b.right) / (dt(b.right, b.right) * z), 4);
    expect(ndc[1]).toBeCloseTo(dt(d, b.up) / (dt(b.up, b.up) * z), 4);
    expect(view[2]).toBeCloseTo(-z, 4);
    // Depth from viewZ (DEPTH_WGSL) and the upscaler's linearizeDepth round trip.
    const dist = z;
    const depth = (PIPE_FAR * (dist - PIPE_NEAR)) / (dist * (PIPE_FAR - PIPE_NEAR));
    expect(ndc[2]).toBeCloseTo(depth, 5);
    expect((PIPE_FAR * PIPE_NEAR) / (PIPE_FAR - depth * (PIPE_FAR - PIPE_NEAR))).toBeCloseTo(dist, 3);
  });
});

describe('resample', () => {
  it('keeps a constant image constant, up and down', () => {
    const img = { width: 16, height: 9, channels: 3, data: new Float32Array(16 * 9 * 3).fill(0.7) };
    for (const [W, H] of [[12, 7], [24, 13], [32, 18]] as const) {
      const r = resample(img, W, H);
      expect(r.width).toBe(W);
      for (const v of r.data) expect(v).toBeCloseTo(0.7, 6);
    }
  });

  it('keeps the mean of a gradient within a pixel', () => {
    const w = 64;
    const h = 36;
    const data = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = x / w + y / h;
    const mean = (d: Float32Array) => d.reduce((s, v) => s + v, 0) / d.length;
    const down = resample({ width: w, height: h, channels: 1, data }, 48, 27);
    const up = resample({ width: w, height: h, channels: 1, data }, 96, 54);
    expect(mean(down.data)).toBeCloseTo(mean(data), 2);
    expect(mean(up.data)).toBeCloseTo(mean(data), 2);
  });
});
