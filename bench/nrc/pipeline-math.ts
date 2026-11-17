/**
 * NRC pipeline math (track N, M9): camera matrices in three.js conventions (column-major, view looks down -Z, WebGPU depth [0, 1]) and
 * the reference resample. No GPU globals, so unit tests import it (pipeline-math.test.ts).
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';

/** Near / far of the projection the depth guide is written with (Sponza is ~30 m long). Only the depth texture and the linearisation read them. */
export const PIPE_NEAR = 0.05;
export const PIPE_FAR = 200;


type V3 = [number, number, number];
export const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
export const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
export const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
export const norm = (a: readonly number[]): V3 => {
  const l = Math.hypot(a[0]!, a[1]!, a[2]!);
  return [a[0]! / l, a[1]! / l, a[2]! / l];
};

/** Exact identity of a camera (a moved camera restarts progressive accumulation). */
export function cameraKey(c: PtLookAtCamera): string {
  return [...c.position, ...c.target, ...c.up, c.yFovDeg].join(',');
}

/** World -> view (column-major), the same basis as `ptCameraBasis` (right = forward x up). */
export function viewMatrix(c: PtLookAtCamera): Float32Array {
  const f = norm(sub(c.target, c.position));
  const r = norm(cross(f, c.up));
  const u = cross(r, f);
  const p = c.position;
  return new Float32Array([r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, p), -dot(u, p), dot(f, p), 1]);
}

/** View -> world (column-major). */
export function worldMatrix(c: PtLookAtCamera): Float32Array {
  const f = norm(sub(c.target, c.position));
  const r = norm(cross(f, c.up));
  const u = cross(r, f);
  const p = c.position;
  return new Float32Array([r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0, p[0], p[1], p[2], 1]);
}

/** three's `makePerspective` for WebGPU (depth 0 at near, 1 at far). */
export function projectionMatrix(yFovDeg: number, aspect: number, near = PIPE_NEAR, far = PIPE_FAR): Float32Array {
  const t = 1 / Math.tan((yFovDeg * Math.PI) / 360);
  const m = new Float32Array(16);
  m[0] = t / aspect;
  m[5] = t;
  m[10] = -far / (far - near);
  m[11] = -1;
  m[14] = (-far * near) / (far - near);
  return m;
}


/** One axis of a separable tent resample (support max(1, scale) source pixels: an area-like filter downscaling, bilinear-like upscaling). */
function resampleAxis(src: Float32Array, w: number, h: number, c: number, size: number, horizontal: boolean): Float32Array {
  const n = horizontal ? w : h;
  const scale = n / size;
  const radius = Math.max(1, scale);
  const outW = horizontal ? size : w;
  const outH = horizontal ? h : size;
  const other = horizontal ? h : w;
  const out = new Float32Array(outW * outH * c);
  const at = (along: number, across: number) => (horizontal ? across * w + along : along * w + across);
  const to = (along: number, across: number) => (horizontal ? across * outW + along : along * outW + across);
  for (let j = 0; j < size; j++) {
    const center = (j + 0.5) * scale;
    const taps: [number, number][] = [];
    let wsum = 0;
    for (let k = Math.max(0, Math.floor(center - radius)); k <= Math.min(n - 1, Math.ceil(center + radius)); k++) {
      const wgt = Math.max(0, 1 - Math.abs(k + 0.5 - center) / radius);
      if (wgt > 0) {
        taps.push([k, wgt]);
        wsum += wgt;
      }
    }
    for (let o = 0; o < other; o++)
      for (let ch = 0; ch < c; ch++) {
        let acc = 0;
        for (const [k, wgt] of taps) acc += (src[at(k, o) * c + ch] as number) * wgt;
        out[to(j, o) * c + ch] = acc / wsum;
      }
  }
  return out;
}

/** Separable tent resample of an image to W x H (brings the 1280x720 reference to the render and display sizes). */
export function resample(img: { width: number; height: number; channels: number; data: Float32Array }, W: number, H: number): { width: number; height: number; channels: number; data: Float32Array } {
  const c = img.channels;
  const a = img.width === W ? img.data : resampleAxis(img.data, img.width, img.height, c, W, true);
  const b = img.height === H ? a : resampleAxis(a, W, img.height, c, H, false);
  return { width: W, height: H, channels: c, data: b };
}
