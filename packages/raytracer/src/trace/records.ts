/**
 * CPU side of the queue ABI v0: packing rays and cones for upload, reading hit and visibility records back, and the
 * dispatch-size arithmetic. Pure functions, unit-tested in Node; nothing here touches a GPU.
 */
import type { Hit, Ray, RayCone } from '../contract/ray.ts';
import { HIT_BYTES, HIT_COMPACT_BYTES, RAY_BYTES, RAY_CONE_BYTES } from './constants.ts';
import { HIT_WORD } from './constants.ts';

/** Floats per ray in the AoS layout (`RtRay`, 32 B): origin xyz, tMin, direction xyz, tMax. */
export const RAY_FLOATS = RAY_BYTES / 4;

/** AoS rays (the contract `Ray`, 32 B each), ready for `writeBuffer`. */
export function packRays(rays: readonly Ray[]): Float32Array<ArrayBuffer> {
  const out = new Float32Array(rays.length * RAY_FLOATS);
  rays.forEach((ray, i) => {
    const o = i * RAY_FLOATS;
    out[o] = ray.origin[0];
    out[o + 1] = ray.origin[1];
    out[o + 2] = ray.origin[2];
    out[o + 3] = ray.tMin;
    out[o + 4] = ray.direction[0];
    out[o + 5] = ray.direction[1];
    out[o + 6] = ray.direction[2];
    out[o + 7] = ray.tMax;
  });
  return out;
}

/** SoA rays: `[origin.xyz, tMin]` per ray in `a`, `[direction.xyz, tMax]` per ray in `b` (16 B each). */
export function packRaysSoa(rays: readonly Ray[]): { a: Float32Array<ArrayBuffer>; b: Float32Array<ArrayBuffer> } {
  const a = new Float32Array(rays.length * 4);
  const b = new Float32Array(rays.length * 4);
  rays.forEach((ray, i) => {
    a.set([ray.origin[0], ray.origin[1], ray.origin[2], ray.tMin], i * 4);
    b.set([ray.direction[0], ray.direction[1], ray.direction[2], ray.tMax], i * 4);
  });
  return { a, b };
}

/** AoS packed rays to SoA (same values, bit for bit). */
export function aosToSoa(packed: Float32Array): { a: Float32Array<ArrayBuffer>; b: Float32Array<ArrayBuffer> } {
  const count = packed.length / RAY_FLOATS;
  const a = new Float32Array(count * 4);
  const b = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    a.set(packed.subarray(i * RAY_FLOATS, i * RAY_FLOATS + 4), i * 4);
    b.set(packed.subarray(i * RAY_FLOATS + 4, i * RAY_FLOATS + 8), i * 4);
  }
  return { a, b };
}

export function unpackRay(packed: Float32Array, index: number): Ray {
  const o = index * RAY_FLOATS;
  const f = (k: number) => packed[o + k] as number;
  return { origin: [f(0), f(1), f(2)], tMin: f(3), direction: [f(4), f(5), f(6)], tMax: f(7) };
}

export function packCones(cones: readonly RayCone[]): Float32Array<ArrayBuffer> {
  const out = new Float32Array((cones.length * RAY_CONE_BYTES) / 4);
  cones.forEach((cone, i) => out.set([cone.width, cone.spread], i * 2));
  return out;
}

/** Bytes per result of a batch: closest-hit record (40 or 24 B) or one u32 visibility word. */
export function resultBytes(mode: 'closest' | 'visible', hitRecord: 'full' | 'compact' = 'full'): number {
  if (mode === 'visible') return 4;
  return hitRecord === 'full' ? HIT_BYTES : HIT_COMPACT_BYTES;
}

/**
 * Reads closest-hit records. Compact records report `material` as `0xffffffff` and a zero normal (they carry neither).
 * A miss has `t < 0` (`MISS_T`).
 */
export function readHits(data: ArrayBuffer | ArrayBufferView, count: number, hitRecord: 'full' | 'compact' = 'full'): Hit[] {
  const buffer = ArrayBuffer.isView(data) ? data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) : data;
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  const stride = resultBytes('closest', hitRecord) / 4;
  const hits: Hit[] = [];
  for (let i = 0; i < count; i++) {
    const o = i * stride;
    const full = hitRecord === 'full';
    hits.push({
      t: floats[o + HIT_WORD.t] as number,
      instance: words[o + HIT_WORD.instance] as number,
      triangle: words[o + HIT_WORD.triangle] as number,
      frontFace: words[o + HIT_WORD.frontFace] !== 0,
      barycentrics: [floats[o + HIT_WORD.barycentrics] as number, floats[o + HIT_WORD.barycentrics + 1] as number],
      material: full ? (words[o + HIT_WORD.material] as number) : 0xffffffff,
      normal: full ? [floats[o + HIT_WORD.normal] as number, floats[o + HIT_WORD.normal + 1] as number, floats[o + HIT_WORD.normal + 2] as number] : [0, 0, 0],
    });
  }
  return hits;
}

/** Reads a visibility batch: true = nothing blocks the ray's interval. */
export function readVisibility(data: ArrayBuffer | ArrayBufferView, count: number): boolean[] {
  const words = ArrayBuffer.isView(data) ? new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4) : new Uint32Array(data);
  return Array.from({ length: count }, (_, i) => words[i] !== 0);
}

/**
 * Workgroup grid for `groups` workgroups under the per-dimension limit: `x * y >= groups`, `x <= maxPerDimension`.
 * The kernel folds (x, y) back into one ray index and drops the tail. Mirrors the WGSL `rtBatchArgs` kernel.
 */
export function dispatchGrid(count: number, workgroupSize: number, maxPerDimension: number = 65535): [number, number, number] {
  const groups = Math.ceil(count / workgroupSize);
  if (groups === 0) return [0, 1, 1];
  const x = Math.min(groups, maxPerDimension);
  return [x, Math.ceil(groups / x), 1];
}

/** `RtBatchParams` uniform (16 B): count, mask, capacity, countWord. */
export const BATCH_PARAMS_BYTES = 16;
export function packBatchParams(params: { count: number; mask: number; capacity: number; countWord: number }): Uint32Array<ArrayBuffer> {
  return new Uint32Array([params.count >>> 0, params.mask >>> 0, params.capacity >>> 0, params.countWord >>> 0]);
}
