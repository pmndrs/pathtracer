/**
 * Memory layout and queue protocol of the wavefront scheduler (P3.1) and its TypeScript twins. The WGSL in `wavefront-wgsl.ts`
 * is generated from the same constants; `layout.test.ts` checks the packing and simulates the counter protocol.
 *
 * One path per pixel, so a path's index is its pixel index (`path = y * width + x`). Slice 1 keeps no refill (P3.2).
 *
 * Refill mode (P3.2, `PtWavefrontOptions.refill`; two flavours: `true` / 'inline' starts the next sample at once in the shade thread, 'batch' queues the slot
 * is counted as idle and, when the live queue drops below a fraction of the slots (Cycles' rule), the args kernel dispatches the camera kernel over the whole film once: it scans
 * the slots in pixel order and starts every one that is not live and still has a sample, so restarted primary rays stay coherent; inline refill appends in random order): a path SLOT is `layer * N + pixel` (N = width * height, K = `layers` slots per pixel, the
 * state / hit / queue buffers are sized for M = K * N slots, `n` in every SoA index below is M). A slot runs the samples
 * `layer, layer + K, layer + 2K, ...` of its pixel one after the other: when its path ends, the shade thread itself starts the slot's next
 * sample (inline refill) and appends it to the next queue, so the queues stay full instead of draining over `maxDepth` iterations. The
 * accumulator is per slot (`ptAccum[slot]`, w = samples STARTED in this slot, no two slots ever write one accumulator); the readback sums
 * the K layers per pixel and divides by the summed w. Sample indices are deterministic: sample `s` of pixel `p` is always run exactly once
 * (slot `s % K` of `p`, its `floor(s / K)`-th), so the Owen-Sobol streams, and the image after T complete samples, equal the megakernel's.
 * A slot that has nothing left to start is dormant: the live bit (bit 29 of the state's depth word) is clear. The queues, counters and
 * path state persist across `encode` calls (the parity of the iteration counter continues), so an encode need not finish its paths.
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
/** Words of the counter array: `[Q0, Q1, shadow0, shadow1, idle]` (idle = slots that finished a path and wait for a refill, batch mode only). */
export const PTWF_COUNTERS = 5;
export const PTWF_COUNT_IDLE = 4;
/** Bytes of the indirect args buffer: main dispatch at 0, shadow dispatch at 16, refill dispatch at 32 (batch refill). */
export const PTWF_ARGS_BYTES = 48;
export const PTWF_ARGS_MAIN = 0;
export const PTWF_ARGS_SHADOW = 16;
export const PTWF_ARGS_REFILL = 32;
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

/** The live bit of the depth word (set on every stored path; clear = the slot is dormant, refill mode). Bit 30 is the AOV `aovDone` flag (`integrator.aovs`). */
export const PTWF_LIVE = 0x20000000;

/** `depth | live << 29 | aovDone << 30 | specularBounce << 31` (state section 2, word w). Depth must fit 29 bits. */
export function ptwfPackDepth(depth: number, specular: boolean): number {
  if (!(depth >= 0 && depth < PTWF_LIVE)) throw new RangeError(`ptwfPackDepth: depth ${depth} does not fit 29 bits`);
  return ((specular ? 0x80000000 : 0) | PTWF_LIVE | depth) >>> 0;
}

export function ptwfUnpackDepth(word: number): { depth: number; specular: boolean; live: boolean } {
  return { depth: word & 0x1fffffff, specular: (word >>> 31) === 1, live: (word & PTWF_LIVE) !== 0 };
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

/**
 * Refill protocol (P3.2). Iteration records: parity 0 and 1 (`iter` 0 / 1) and the start records (`iter` 2 / 3: parity in bit 0). Only the
 * parity of an iteration is ever read by a kernel, so one encode needs 4 uniform slots whatever its length (no ring limit on passes).
 */
export const PTWF_RECORDS_PER_ENCODE = 4;
export const PTWF_START_ITER = 2;

/** Sample index of the `started`-th sample (0-based) of layer `layer` of a pixel, with `layers` slots per pixel. */
export function ptwfSampleOf(started: number, layer: number, layers: number): number {
  return started * layers + layer;
}

/** Does a slot that has started `started` samples have another one to start for a target of `target` samples per pixel? */
export function ptwfHasNext(started: number, layer: number, layers: number, target: number): boolean {
  return ptwfSampleOf(started, layer, layers) < target;
}

/** The args kernel's step for a start record (`iter` 2 | parity): main args for the queue the camera kernel appended to; nothing is zeroed. */
export function ptwfArgsStart(counters: Uint32Array, iter: number): { main: [number, number, number] } {
  return { main: ptwfGroups(counters[iter & 1] as number) };
}

/** Uniform records and dispatches of an encode of `iterations` iterations: camera + args (when `start`), then 4 per iteration. */
export function ptwfRefillDispatches(iterations: number, start: boolean): number {
  return (start ? 2 : 0) + 4 * iterations;
}

/**
 * Batch refill, the args kernel's decision (TS twin of the WGSL): after shade, `live` paths are queued for the next iteration and `idle` slots wait
 * for a next sample. If `live` is below `threshold` x slots (or zero) the camera kernel is dispatched over the film (`refillGrid`, else zero groups) and starts
 * every idle slot (appended to the live queue), so the next main dispatch covers `live + idle`.
 */
export function ptwfRefillDecision(live: number, idle: number, slots: number, threshold: number, width: number, height: number, layers: number): { refill: number; main: [number, number, number]; refillGrid: [number, number, number] } {
  const refill = idle > 0 && (live === 0 || live < Math.floor(slots * threshold)) ? idle : 0;
  return { refill, main: ptwfGroups(live + refill), refillGrid: refill > 0 ? [Math.ceil(width / 8), Math.ceil(height / 8), layers] : [0, 0, 0] };
}

/**
 * P3.3: subgroup-aggregated append (TS twin of WGSL `ptwfAppend`). `wants[i]` = lane i of one subgroup wants a queue slot; one `atomicAdd` of the number of
 * wanting lanes reserves a contiguous range at `counter` and each lane takes `base + (wanting lanes before it)`, so the lanes keep their order. Returns the
 * slot of every wanting lane (-1 for the others) and the new counter.
 */
export function ptwfSubgroupAppend(wants: readonly boolean[], counter: number): { slots: number[]; counter: number } {
  const total = wants.filter(Boolean).length;
  let rank = 0;
  const slots = wants.map((w) => (w ? counter + rank++ : -1));
  return { slots, counter: counter + total };
}

/**
 * P3.4: block-wise counting sort of one queue block (TS twin of WGSL `ptwfSort`): returns, for every entry, its new position inside the block. Position =
 * entries with a smaller key + earlier entries with the same key (stable). `keys.length` is at most `PTWF_WORKGROUP`.
 */
export function ptwfBlockSortPositions(keys: readonly number[]): number[] {
  return keys.map((key, i) => keys.reduce((pos, k, j) => pos + (k < key || (k === key && j < i) ? 1 : 0), 0));
}

/** P3.4 key of the `hit` sort (TS twin of the WGSL): 0 for a miss, else 1 + material % 15. */
export function ptwfHitSortKey(hitT: number, material: number): number {
  return hitT < 0 ? 0 : 1 + (material % 15);
}

/** P3.4 key of the `ray` sort (octant): the sign bits of the direction, x | y << 1 | z << 2. */
export function ptwfRaySortKey(dir: readonly [number, number, number]): number {
  return (dir[0] < 0 ? 1 : 0) | (dir[1] < 0 ? 2 : 0) | (dir[2] < 0 ? 4 : 0);
}

/**
 * P3.10 ordered restart (TS twin of WGSL `ptwfPrimary`): the slots flagged in one 64-slot block are appended to the queue in slot order with one
 * range reservation. `flags[i]` = slot `block * 64 + i` was restarted this iteration. Returns the queue entries written (slot ids) and the new counter.
 */
export function ptwfOrderedAppend(flags: readonly boolean[], block: number, counter: number): { entries: number[]; counter: number } {
  const entries: number[] = [];
  flags.forEach((f, i) => {
    if (f) entries.push(block * PTWF_WORKGROUP + i);
  });
  return { entries, counter: counter + entries.length };
}

/** R1b: which shade kernel handles a path of `depth` when `materialLod: 'secondary'` runs with refill (the gate of `ptwfShadeWgsl`). */
export function ptwfLodClass(depth: number, lodDepth: number): 'first' | 'second' {
  return depth < lodDepth ? 'first' : 'second';
}
