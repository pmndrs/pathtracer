/**
 * The "learn an image" task shared by mlp-fit, mlp-timing and mlp-live (research track N): a procedural HDR target,
 * the GPU training sampler (random sub-pixel (u, v) from a hash of record and Adam step, bilinear target) with its JS
 * twin for oracle replays, and the group-1 hooks for MlpEngine.
 */

import { hashU32 } from '@pathtracer/mlp';

/** Procedural HDR target, linear RGB, row 0 = top. Features at several scales so encodings and widths separate. */
export function targetPixel(u: number, v: number): [number, number, number] {
  // Sky gradient.
  let r = 0.15 + 0.85 * v;
  let g = 0.3 + 0.6 * v;
  let b = 0.8 - 0.2 * v;
  // Sun: a disc at 30 with a soft rim and a glow.
  const d = Math.hypot(u - 0.72, v - 0.22);
  const disc = Math.min(Math.max((0.06 - d) / 0.004, 0), 1);
  const glow = 2 * Math.exp(-((d / 0.12) ** 2));
  r += 30 * disc + glow;
  g += 27 * disc + 0.8 * glow;
  b += 20 * disc + 0.4 * glow;
  // Chirp band: stripes from 4 to 96 cycles across, tests high frequencies.
  if (v > 0.42 && v < 0.54) {
    const cycles = 4 + 92 * u;
    const s = 0.5 + 0.5 * Math.sin(2 * Math.PI * cycles * u);
    r = 0.05 + 0.9 * s;
    g = 0.05 + 0.9 * s;
    b = 0.05 + 0.9 * s;
  }
  // Checkerboard floor with a soft shadow.
  if (v > 0.62) {
    const check = (Math.floor(u * 12) + Math.floor((v - 0.62) * 12 / 0.38 * 2)) % 2 === 0 ? 0.7 : 0.08;
    const shadow = 1 - 0.7 * Math.exp(-(((u - 0.35) / 0.15) ** 2) - (((v - 0.8) / 0.08) ** 2));
    r = check * shadow * 1.2;
    g = check * shadow * 1.0;
    b = check * shadow * 0.9;
  }
  return [r, g, b];
}

/** JS twin of the WGSL training sampler: same hash, same (u, v), bilinear target. */
export function trainSample(r: number, step: number, seed: number, size: number, target: Float32Array): { u: number; v: number; t: [number, number, number] } {
  const key = hashU32((r ^ hashU32((Math.imul(step, 0x9e3779b9) + seed) >>> 0)) >>> 0);
  const u = (hashU32(key) >>> 8) / 16777216;
  const v = (hashU32((key + 1) >>> 0) >>> 8) / 16777216;
  const fetch = (x: number, y: number, c: number) => target[((Math.min(Math.max(y, 0), size - 1) * size) + Math.min(Math.max(x, 0), size - 1)) * 3 + c] as number;
  const px = u * size - 0.5;
  const py = v * size - 0.5;
  const ix = Math.floor(px);
  const iy = Math.floor(py);
  const fx = px - ix;
  const fy = py - iy;
  const t = [0, 1, 2].map((c) => {
    const a = fetch(ix, iy, c) * (1 - fx) + fetch(ix + 1, iy, c) * fx;
    const b = fetch(ix, iy + 1, c) * (1 - fx) + fetch(ix + 1, iy + 1, c) * fx;
    return a * (1 - fy) + b * fy;
  }) as [number, number, number];
  return { u, v, t };
}

export const IMAGE_HOOKS_COMMON = /* wgsl */ `
struct ImageDims { width: u32, height: u32, seed: u32, _pad: u32 }
@group(1) @binding(0) var<storage, read> imgTarget: array<f32>;
@group(1) @binding(1) var<uniform> imgDims: ImageDims;

fn imgHash(x: u32) -> u32 {
  let state = x * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn imgRand(x: u32) -> f32 { return f32(imgHash(x) >> 8u) / 16777216.0; }

fn imgFetch(x: i32, y: i32) -> vec3f {
  let cx = u32(clamp(x, 0, i32(imgDims.width) - 1));
  let cy = u32(clamp(y, 0, i32(imgDims.height) - 1));
  let i = (cy * imgDims.width + cx) * 3u;
  return vec3f(imgTarget[i], imgTarget[i + 1u], imgTarget[i + 2u]);
}
// Bilinear at uv (pixel centres at (i + 0.5) / size).
fn imgSample(uv: vec2f) -> vec3f {
  let p = uv * vec2f(f32(imgDims.width), f32(imgDims.height)) - 0.5;
  let f = fract(p);
  let i = vec2i(floor(p));
  let a = mix(imgFetch(i.x, i.y), imgFetch(i.x + 1, i.y), f.x);
  let b = mix(imgFetch(i.x, i.y + 1), imgFetch(i.x + 1, i.y + 1), f.x);
  return mix(a, b, f.y);
}
`;

export const IMAGE_HOOKS_TRAIN = `${IMAGE_HOOKS_COMMON}
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
  let key = imgHash(r ^ imgHash(nnState.step * 0x9e3779b9u + imgDims.seed));
  let uv = vec2f(imgRand(key), imgRand(key + 1u));
  *x = nnEncodeImage(uv);
  return imgSample(uv);
}
`;

export const IMAGE_HOOKS_INFER = /* wgsl */ `
struct ImageDims { width: u32, height: u32, seed: u32, _pad: u32 }
@group(1) @binding(1) var<uniform> imgDims: ImageDims;
@group(1) @binding(2) var<storage, read_write> imgOut: array<f32>;
fn nnInferLoad(r: u32) -> array<f32, NN_IN> {
  let uv = (vec2f(f32(r % imgDims.width), f32(r / imgDims.width)) + 0.5) / vec2f(f32(imgDims.width), f32(imgDims.height));
  return nnEncodeImage(uv);
}
fn nnInferStore(r: u32, y: vec3f) { imgOut[r * 3u] = y.x; imgOut[r * 3u + 1u] = y.y; imgOut[r * 3u + 2u] = y.z; }
`;


/** Target image, size × size, linear RGB f32, row 0 = top. */
export function targetImage(size: number): Float32Array {
  const target = new Float32Array(size * size * 3);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) target.set(targetPixel((x + 0.5) / size, (y + 0.5) / size), (y * size + x) * 3);
  return target;
}

/** Group-1 bind groups for the image hooks: training reads the target and dims; inference writes `out`. */
export function imageBindGroups(device: GPUDevice, trainLayout: GPUBindGroupLayout, inferLayout: GPUBindGroupLayout, target: GPUBuffer, dims: GPUBuffer, out: GPUBuffer): { train: GPUBindGroup; infer: GPUBindGroup } {
  return {
    train: device.createBindGroup({ layout: trainLayout, entries: [{ binding: 0, resource: { buffer: target } }, { binding: 1, resource: { buffer: dims } }] }),
    infer: device.createBindGroup({ layout: inferLayout, entries: [{ binding: 1, resource: { buffer: dims } }, { binding: 2, resource: { buffer: out } }] }),
  };
}
