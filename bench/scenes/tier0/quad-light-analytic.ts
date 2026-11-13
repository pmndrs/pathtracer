import { lookAtCameraToWorld, transformDirection } from '../geometry.ts';
import type { SceneDesc } from '../schema.ts';
import { QUAD_LIGHT } from './quad-light.ts';

/**
 * Irradiance E (W/m²) on the plane point (x, 0, z) from the one-sided quad light facing down: both cosines equal h/r,
 * so E = L·h²·∫ dA / r⁴ over the light. Midpoint rule on an n×n grid (error ~1e-6 for n = 200).
 */
export function quadLightIrradiance(x: number, z: number, n = 200): number {
  const { size, height, radiance } = QUAD_LIGHT;
  const dA = (size / n) ** 2;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const lx = -size / 2 + ((i + 0.5) * size) / n;
    for (let j = 0; j < n; j++) {
      const lz = -size / 2 + ((j + 0.5) * size) / n;
      const r2 = (lx - x) ** 2 + (lz - z) ** 2 + height * height;
      sum += dA / (r2 * r2);
    }
  }
  return radiance * height * height * sum;
}

/** Closed form for the point directly under the centre of the light (4 corner rectangles of a/2 × b/2). */
export function quadLightCentreIrradianceClosedForm(): number {
  const { size, height, radiance } = QUAD_LIGHT;
  const A = size / 2 / height, B = size / 2 / height;
  const f = (1 / (2 * Math.PI)) * ((A / Math.sqrt(1 + A * A)) * Math.atan(B / Math.sqrt(1 + A * A)) + (B / Math.sqrt(1 + B * B)) * Math.atan(A / Math.sqrt(1 + B * B)));
  return radiance * Math.PI * 4 * f;
}

/**
 * Expected radiance image (albedo/π · E) for the quad-light scene, direct lighting only. Pixels that are not plain
 * lit plane (miss, light quad, plane edge) in ANY of 4×4 sub-samples are NaN (masked out of the comparison).
 */
export function quadLightAnalyticImage(scene: SceneDesc): Float32Array {
  const { width: W, height: H } = scene.film;
  const cam = scene.camera;
  const m = lookAtCameraToWorld(cam.position, cam.target, cam.up);
  const t = Math.tan((cam.yFovDeg * Math.PI) / 360);
  const aspect = W / H;
  const out = new Float32Array(W * H * 3);
  const sub = 4;
  const half = QUAD_LIGHT.planeSize / 2;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      let acc = 0;
      let ok = true;
      for (let sj = 0; sj < sub && ok; sj++) {
        for (let si = 0; si < sub; si++) {
          const px = (i + (si + 0.5) / sub) / W, py = (j + (sj + 0.5) / sub) / H;
          const d = transformDirection(m, [(2 * px - 1) * t * aspect, (1 - 2 * py) * t, -1]);
          if (d[1] >= 0) { ok = false; break; }
          const s = -cam.position[1] / d[1];
          const x = cam.position[0] + s * d[0], z = cam.position[2] + s * d[2];
          // light quad at y = height: does the ray cross it?
          const sl = (QUAD_LIGHT.height - cam.position[1]) / d[1];
          const lx = cam.position[0] + sl * d[0], lz = cam.position[2] + sl * d[2];
          if (sl > 0 && Math.abs(lx) <= QUAD_LIGHT.size / 2 && Math.abs(lz) <= QUAD_LIGHT.size / 2) { ok = false; break; }
          if (Math.abs(x) > half || Math.abs(z) > half) { ok = false; break; }
          acc += (QUAD_LIGHT.albedo / Math.PI) * quadLightIrradiance(x, z, 40);
        }
      }
      const v = ok ? acc / (sub * sub) : NaN;
      out.set([v, v, v], (j * W + i) * 3);
    }
  }
  return out;
}
