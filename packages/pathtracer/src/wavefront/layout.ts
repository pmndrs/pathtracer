/**
 * Memory layout and queue protocol of the wavefront scheduler (P3.1) and its TypeScript twins. The WGSL in `wavefront-wgsl.ts`
 * is generated from the same constants; `layout.test.ts` checks the packing and simulates the counter protocol.
 *
 * One path per pixel, so a path's index is its pixel index (`path = y * width + x`). Slice 1 keeps no refill (P3.2).
 *
 * Path state, SoA, `array<vec4u>` (floats are stored as their bit patterns: a small u32 is a denormal f32 and a load is allowed
 * to flush it; u32 never is). Section `k` of path `p` is element `k * N + p`, N = width * height:
 *
 *     0  origin.xyz, prevPdf                       ray the next intersect traces, and the MIS pdf of the last BSDF sample
 *     1  direction.xyz, etaScale
 *     2  beta.xyz, depth | specularBounce << 31    scattering events so far; the last bounce was a delta lobe
 *     3  prevP.xyz, coneSpread                     previous vertex (emission MIS); ray-cone spread (textures only, else 0)
 *     4  sigma.xyz, coneWidth                      interior medium attenuation + ray-cone width (only with medium or textures)
 *
 * = 64 B (plain), 80 B (medium or textures), under the 96 B budget (spec section 5). Radiance is not state: each path adds its
 * emission straight into the accumulator (a pixel has one path, so no two threads write it in one dispatch) and its shadow
 * rays add their light after the shade dispatch; this saves 12 B and the final resolve pass.
 *
 * Hit record, written by `intersect`, read by `shade` (transient, 48 B/path, `array<vec4u>`):
 *     0  t, instance, triangle, material           1  bary.xy, frontFace, 0           2  normal.xyz, 0
 * Shadow ray record, written by `shade`, read by `shadow` (48 B/path): 0 origin.xyz, tMin; 1 direction.xyz, tMax; 2 contribution.xyz, 0.
 *
 * Queues (`ptwfQueue`, u32 path indices, three regions of N): Q0, Q1 (the extension queue ping-pongs by iteration parity) and the
 * shadow queue. Counters (`ptwfCount`, atomics): `[Q0, Q1, shadow0, shadow1]`. Iteration `d` reads `Q[d & 1]`, shade appends to
 * `Q[(d + 1) & 1]` and to the shadow queue counted in `shadow[d & 1]`.
 */

/** Threads per workgroup of the queue kernels. */
export const PTWF_WORKGROUP = 64;
/** Workgroups per row of an indirect dispatch: larger queues fold into y (`maxComputeWorkgroupsPerDimension` is 65535). */
export const PTWF_STRIDE = 4096;
/** `iter` value of the frame-start step (camera kernel and the args kernel's initialisation). */
export const PTWF_INIT = 0xffffffff;
/** Words of the counter array. */
export const PTWF_COUNTERS = 4;
/** Bytes of the indirect args buffer: main dispatch at 0, shadow dispatch at 16. */
export const PTWF_ARGS_BYTES = 32;
export const PTWF_ARGS_MAIN = 0;
export const PTWF_ARGS_SHADOW = 16;
/** Uniform slots in the ring: one record per iteration plus one for the frame start. */
export const PTWF_PARAM_SLOTS = 4096;

export const PTWF_STATE_VEC4_BASE = 4;
export const PTWF_HIT_VEC4 = 3;
export const PTWF_SHADOW_VEC4 = 3;

export interface PtwfStateOptions {
  /** The path carries an interior-medium attenuation (`features.medium`). */
  medium: boolean;
  /** The path carries a ray cone (`textures`). */
  textures: boolean;
}

/** vec4s per path in the state buffer. */
export function ptwfStateVec4s(o: PtwfStateOptions): number {
  return o.medium || o.textures ? PTWF_STATE_VEC4_BASE + 1 : PTWF_STATE_VEC4_BASE;
}

/** Bytes of path state per path (the spec's budget is 96). */
export function ptwfStateBytes(o: PtwfStateOptions): number {
  return ptwfStateVec4s(o) * 16;
}

/** Bytes of every per-path buffer for `paths` paths: `{ state, hit, shadow, queue, counters, total }`. */
export function ptwfBufferBytes(paths: number, o: PtwfStateOptions): { state: number; hit: number; shadow: number; queue: number; counters: number; total: number } {
  const state = paths * ptwfStateBytes(o);
  const hit = paths * PTWF_HIT_VEC4 * 16;
  const shadow = paths * PTWF_SHADOW_VEC4 * 16;
  const queue = paths * 3 * 4;
  const counters = PTWF_COUNTERS * 4;
  return { state, hit, shadow, queue, counters, total: state + hit + shadow + queue + counters };
}

/** `depth | specularBounce << 31` (state section 2, word w). Depth must fit 31 bits. */
export function ptwfPackDepth(depth: number, specular: boolean): number {
  if (!(depth >= 0 && depth < 0x80000000)) throw new RangeError(`ptwfPackDepth: depth ${depth} does not fit 31 bits`);
  return ((specular ? 0x80000000 : 0) | depth) >>> 0;
}

export function ptwfUnpackDepth(word: number): { depth: number; specular: boolean } {
  return { depth: word & 0x7fffffff, specular: (word >>> 31) === 1 };
}

/** Workgroup counts `[x, y, z]` of an indirect dispatch for `count` queued paths: x folds at `PTWF_STRIDE`. TS twin of WGSL `ptwfGroups`. */
export function ptwfGroups(count: number): [number, number, number] {
  const g = Math.ceil(count / PTWF_WORKGROUP);
  return [Math.min(g, PTWF_STRIDE), Math.ceil(g / PTWF_STRIDE), 1];
}

/** Queue slot of a thread of an indirect dispatch (WGSL: `(wid.y * PTWF_STRIDE + wid.x) * 64 + li`). */
export function ptwfSlot(wid: readonly [number, number], localIndex: number): number {
  return (wid[1] * PTWF_STRIDE + wid[0]) * PTWF_WORKGROUP + localIndex;
}

/** Word offset of the three queue regions in `ptwfQueue`. */
export function ptwfQueueOffsets(paths: number): { extension: [number, number]; shadow: number } {
  return { extension: [0, paths], shadow: 2 * paths };
}

/** Uniform records one sample pass writes: the frame start plus one per iteration (`maxDepth + 1` iterations: depths 0..maxDepth). */
export function ptwfRecordsPerSample(maxDepth: number): number {
  return 1 + maxDepth + 1;
}

/** Dispatches one sample pass encodes: camera + init, then per iteration intersect, shade and (except the last) args + shadow. */
export function ptwfDispatchesPerSample(maxDepth: number): number {
  return 2 + 2 * (maxDepth + 1) + 2 * maxDepth;
}

/**
 * The counter protocol, as the args kernel runs it (TS twin of WGSL `ptwfArgs`): `counters` is `[Q0, Q1, shadow0, shadow1]`,
 * `iter` the iteration whose shade just finished (or `PTWF_INIT` before the first). Returns the dispatch args it writes.
 *
 *  - INIT: Q0 = paths, the other counters 0; main args for `paths`.
 *  - iteration d: main args = count of the queue shade just filled (`Q[(d + 1) & 1]`), shadow args = `shadow[d & 1]`; then it zeroes
 *    the queue shade read (`Q[d & 1]`, refilled by iteration d + 1) and `shadow[(d + 1) & 1]` (filled by shade d + 1; its last user,
 *    shadow d - 1, has finished).
 */
export function ptwfArgsStep(counters: Uint32Array, iter: number, paths: number): { main: [number, number, number]; shadow: [number, number, number] } {
  if (iter === PTWF_INIT) {
    counters.set([paths, 0, 0, 0]);
    return { main: ptwfGroups(paths), shadow: [0, 0, 1] };
  }
  const cur = iter & 1;
  const nxt = cur ^ 1;
  const main = ptwfGroups(counters[nxt] as number);
  const shadow = ptwfGroups(counters[2 + cur] as number);
  counters[cur] = 0;
  counters[2 + nxt] = 0;
  return { main, shadow };
}
