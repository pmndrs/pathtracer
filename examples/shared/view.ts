/**
 * A camera uniform for the examples' own compute shaders: film size, a frame counter and the camera basis, so a kernel builds its
 * primary ray for pixel (x, y) (row 0 = top) with `exRay(x, y, jitter)`. 64 bytes of basis after 16 bytes of header.
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';

export const VIEW_WGSL = /* wgsl */ `
struct ExView {
  width: u32, height: u32, frame: u32, mode: u32,
  origin: vec4f,  // w: a free float parameter of the example
  right: vec4f,   // tan(fov / 2) * aspect along the camera's right; w: a second parameter
  up: vec4f,      // tan(fov / 2) along the camera's up
  forward: vec4f,
}

// The primary ray through pixel (x, y) at sub-pixel offset jitter in [0, 1)^2 (0.5 = the centre).
fn exRay(view: ExView, x: u32, y: u32, jitter: vec2f) -> RtRay {
  let ndc = vec2f((f32(x) + jitter.x) / f32(view.width), (f32(y) + jitter.y) / f32(view.height)) * 2.0 - 1.0;
  let d = normalize(view.forward.xyz + ndc.x * view.right.xyz - ndc.y * view.up.xyz);
  return RtRay(view.origin.xyz, 0.0, d, 1e30);
}

// PCG hash: a well-mixed u32 from a u32.
fn exHash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// A uniform float in [0, 1) from a seed, advancing it.
fn exRand(seed: ptr<function, u32>) -> f32 {
  *seed = exHash(*seed);
  return f32(*seed >> 8u) * (1.0 / 16777216.0);
}
`;

export const VIEW_BYTES = 80;

/** Writes the view record: `params` fills origin.w and right.w. */
export function writeView(device: GPUDevice, buffer: GPUBuffer, camera: PtLookAtCamera, width: number, height: number, frame: number, mode = 0, params: [number, number] = [0, 0]): void {
  const f = [camera.target[0] - camera.position[0], camera.target[1] - camera.position[1], camera.target[2] - camera.position[2]];
  const fl = Math.hypot(f[0]!, f[1]!, f[2]!);
  const fw = f.map((v) => v / fl) as [number, number, number];
  const r = [fw[1] * camera.up[2] - fw[2] * camera.up[1], fw[2] * camera.up[0] - fw[0] * camera.up[2], fw[0] * camera.up[1] - fw[1] * camera.up[0]];
  const rl = Math.hypot(r[0]!, r[1]!, r[2]!);
  const rn = r.map((v) => v / rl) as [number, number, number];
  const u = [rn[1] * fw[2] - rn[2] * fw[1], rn[2] * fw[0] - rn[0] * fw[2], rn[0] * fw[1] - rn[1] * fw[0]];
  const t = Math.tan((camera.yFovDeg * Math.PI) / 360);
  const aspect = width / height;
  const data = new ArrayBuffer(VIEW_BYTES);
  new Uint32Array(data, 0, 4).set([width, height, frame, mode]);
  new Float32Array(data, 16, 16).set([...camera.position, params[0], ...rn.map((v) => v * t * aspect), params[1], ...u.map((v) => v * t), 0, ...fw, 0]);
  device.queue.writeBuffer(buffer, 0, data);
}
