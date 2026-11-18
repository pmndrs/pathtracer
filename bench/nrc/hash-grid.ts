/**
 * World-space hash-grid radiance cache (track N, M7 / N7, docs/design/nrc-hash-cache.md): the non-neural competitor to the NRC
 * network, in the network's place inside `NrcCache` (`model: 'hash'`). Same paths, same training records, same termination: only the
 * store differs, so the A/B isolates "network vs hash grid".
 *
 * Design from the published descriptions only (Binder et al. 2019 path-space filtering, arXiv 1902.05942; SHaRC's integration notes as
 * summarised in research/notes/08-realtime-gi.md; no RTX SDK code, CLAUDE.md licence rules):
 *  - A cell is (level, integer cell of size minCell * 2^level, dominant normal axis + sign). The level grows with the distance to the
 *    camera so a cell covers about `cellPixels` pixels wherever it is (`hgLevel`), SHaRC's LOD rule.
 *  - Open addressing in buckets of `BUCKET` slots: slot word = 32-bit checksum of the cell (0 = empty), claimed by compare-exchange.
 *    Lookups scan the whole bucket (no early stop at an empty slot), so evicting a cell never hides another.
 *  - Update: every training record (the same (vertex, target) pairs the network trains on) adds its target into its cell as u32 fixed
 *    point with stochastic rounding (WGSL has no float atomics) plus a sample count.
 *  - Resolve: per cell, an exponentially weighted sample mean with a time constant of `historyFrames` frames (the history weight decays
 *    only in frames that bring new samples); cells without samples for `staleFrames` frames are evicted.
 *  - `pyramid`: each record also adds into its `fallbackLevels` coarser cells (a multi-resolution grid; off = one level per record).
 *  - Lookup: the query's cell; on a miss (absent, or fewer than `minSamples`) up to `fallbackLevels` coarser cells; still a miss: 0 (the
 *    same as a fresh network with a zero output layer). Optional jitter in the tangent plane (Binder's jittered lookup).
 */

export interface HashGridSettings {
  /** Table size: 2^log2Entries cells (20: 1M cells, 40 MiB). */
  log2Entries: number;
  /** Smallest cell, in scene units (metres). */
  minCell: number;
  /** Cell size target in pixels at the cell's distance from the camera (the LOD rule). */
  cellPixels: number;
  /** Separate cells per dominant normal axis and sign (6 bins), or one cell for all normals. */
  normalBins: boolean;
  /** Time constant of the per-cell running mean, in frames. */
  historyFrames: number;
  /** A cell without new samples for this many frames is evicted. */
  staleFrames: number;
  /** Coarser levels tried on a miss. */
  fallbackLevels: number;
  /** Each record also adds into its `fallbackLevels` coarser cells, so a fallback always finds data (a multi-resolution grid). */
  pyramid: boolean;
  /** A cell with less sample weight than this counts as a miss. */
  minSamples: number;
  /** Lookup jitter as a fraction of the cell size, in the tangent plane (0 = off). */
  jitter: number;
  /**
   * Direction bins at glossy vertices (record roughness < 0.5): the cell also splits by the outgoing direction wo into N × N octahedral
   * bins (0 = off: one view-independent value per cell, SHaRC's default). Lambert vertices (roughness 1) always use one bin.
   */
  dirBins: number;
}

export const DEFAULT_HASH_GRID: HashGridSettings = {
  log2Entries: 20,
  minCell: 0.02,
  // M7 sweeps (sponza-white 640x360, NRC-STUDY.md "Hash grid vs network"): 16 px cells, the pyramid and a 16-sample threshold removed the
  // single-sample blotches after camera changes (worst transient 10.6 -> 2.8) at no bias; 3 fallback levels cut misses after a cut from 26% to 2%.
  // At 720p a 64-sample threshold beat 16 in every scenario (steady 2.48 vs 2.53).
  cellPixels: 16,
  normalBins: true,
  historyFrames: 32,
  staleFrames: 64,
  fallbackLevels: 3,
  pyramid: true,
  minSamples: 64,
  jitter: 0,
  dirBins: 0,
};

/** Bucket size (slots probed per lookup); a compile-time constant so the loops have a constant trip count (Tint loop counters on D3D12). */
const BUCKET = 8;
/** Fixed-point scale of the accumulators, and the per-sample clamp that keeps a u32 sum from overflowing (≥ 16k clamped samples). */
const SCALE = 256;
const CLAMP = 1000;
const MAX_LEVEL = 24;

/** Records are `NRC_RECORD_VEC4` = 4 vec4 each (bench/nrc/cache.ts): (position, pixel), (wo θ/π φ/2π, normal θ/π φ/2π), (albedo, 1), (value, flag). */
const RECORD_VEC4 = 4;

const COMMON_WGSL = (fallback: number, pyramid = false) => /* wgsl */ `
struct HgParams {
  cam: vec4f,
  minCell: f32, jitter: f32, decay: f32, minSamples: f32,
  mask: u32, stale: u32, normals: u32, queryCap: u32,
  recordCap: u32, dirBins: u32, _b: u32, _c: u32,
}
@group(0) @binding(0) var<uniform> hg: HgParams;

fn hgMix(x: u32) -> u32 {
  let s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn hgMix2(x: u32) -> u32 {
  var h = x * 0x85ebca6bu;
  h ^= h >> 13u;
  h *= 0xc2b2ae35u;
  h ^= h >> 16u;
  return h;
}
fn hgRand(a: u32, b: u32) -> f32 { return f32(hgMix(a ^ hgMix(b)) >> 8u) * (1.0 / 16777216.0); }

fn hgNormal(e: vec2f) -> vec3f {
  let th = e.x * 3.141592653589793;
  let ph = (e.y - 0.5) * 6.283185307179586;
  return vec3f(sin(th) * cos(ph), cos(th), sin(th) * sin(ph));
}
fn hgLevel(p: vec3f) -> u32 {
  let size = max(distance(p, hg.cam.xyz) * hg.cam.w, hg.minCell);
  return min(u32(max(floor(log2(size / hg.minCell)), 0.0)), ${MAX_LEVEL}u);
}
fn hgBin(n: vec3f) -> u32 {
  if (hg.normals == 0u) { return 0u; }
  let a = abs(n);
  if (a.x >= a.y && a.x >= a.z) { return select(1u, 2u, n.x < 0.0); }
  if (a.y >= a.z) { return select(3u, 4u, n.y < 0.0); }
  return select(5u, 6u, n.z < 0.0);
}
fn hgCellSize(level: u32) -> f32 { return hg.minCell * exp2(f32(level)); }
// (bucket start, checksum != 0) of the cell holding p at a level.
// Octahedral bin (1-based; 0 = none) of the outgoing direction (record word r1.xy, θ/π and φ/2π) at a glossy vertex.
fn hgDirBin(e: vec2f, rough: f32) -> u32 {
  if (hg.dirBins == 0u || rough >= 0.5) { return 0u; }
  let d = hgNormal(e);
  var o = d.xz / (abs(d.x) + abs(d.y) + abs(d.z));
  if (d.y < 0.0) { o = (1.0 - abs(o.yx)) * select(vec2f(-1.0), vec2f(1.0), o >= vec2f(0.0)); }
  let n = f32(hg.dirBins);
  let c = min(vec2u((o * 0.5 + 0.5) * n), vec2u(hg.dirBins - 1u));
  return 1u + c.x + c.y * hg.dirBins;
}
fn hgKey(p: vec3f, bin: u32, level: u32) -> vec2u {
  let u = bitcast<vec3u>(vec3i(floor(p / hgCellSize(level))));
  let lb = level | (bin << 8u);  // bin: normal bin | direction bin << 4
  let h = hgMix(hgMix(hgMix(hgMix(u.x + 0x9e3779b9u) ^ u.y) ^ u.z) ^ lb);
  let g = hgMix2(hgMix2(hgMix2(hgMix2(u.z ^ 0x27d4eb2fu) + u.x) + u.y) + lb * 0x165667b1u);
  return vec2u(h & hg.mask & ~${BUCKET - 1}u, max(g, 1u));
}
const HG_FALLBACK = ${fallback}u;
const HG_PYRAMID = ${pyramid ? fallback : 0}u;
`;

const LOOKUP_WGSL = (fallback: number) => /* wgsl */ `
${COMMON_WGSL(fallback)}
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> resolved: array<vec4f>;
@group(0) @binding(3) var<storage, read> queries: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> preds: array<vec4f>;
@group(0) @binding(5) var<storage, read> qcount: array<u32>;
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>>;

var<workgroup> hgMisses: atomic<u32>;
var<workgroup> hgCoarse: atomic<u32>;

fn hgFind(k: vec2u) -> i32 {
  for (var i = 0u; i < ${BUCKET}u; i++) {
    if (keys[k.x + i] == k.y) { return i32(k.x + i); }
  }
  return -1;
}

@compute @workgroup_size(64)
fn hgLookup(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    atomicStore(&hgMisses, 0u);
    atomicStore(&hgCoarse, 0u);
  }
  workgroupBarrier();
  let r = gid.x;
  if (r < min(qcount[0], hg.queryCap)) {
    let b = r * ${RECORD_VEC4}u;
    let p0 = queries[b].xyz;
    let n = hgNormal(queries[b + 1u].zw);
    let l0 = hgLevel(p0);
    var p = p0;
    if (hg.jitter > 0.0) {
      let f = atomicLoad(&stats[0]);
      var o = (vec3f(hgRand(r, f * 3u), hgRand(r, f * 3u + 1u), hgRand(r, f * 3u + 2u)) - 0.5) * (hg.jitter * hgCellSize(l0));
      o -= n * dot(o, n);
      p += o;
    }
    let bin = hgBin(n) | (hgDirBin(queries[b + 1u].xy, queries[b + 2u].w) << 4u);
    var out = vec4f(0.0, 0.0, 0.0, -1.0);
    for (var j = 0u; j <= HG_FALLBACK; j++) {
      let s = hgFind(hgKey(p, bin, l0 + j));
      if (s >= 0) {
        let v = resolved[u32(s)];
        if (v.w >= hg.minSamples) {
          out = vec4f(v.xyz, f32(j));
          break;
        }
      }
    }
    preds[r] = out;
    if (out.w < 0.0) { atomicAdd(&hgMisses, 1u); } else if (out.w > 0.0) { atomicAdd(&hgCoarse, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let m = atomicLoad(&hgMisses);
    let c = atomicLoad(&hgCoarse);
    if (m > 0u) { atomicAdd(&stats[1], m); }
    if (c > 0u) { atomicAdd(&stats[2], c); }
  }
}
`;

const UPDATE_WGSL = (fallback: number, pyramid: boolean) => /* wgsl */ `
${COMMON_WGSL(fallback, pyramid)}
@group(0) @binding(1) var<storage, read_write> keys: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> records: array<vec4f>;
@group(0) @binding(4) var<storage, read> rcount: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

// The cell's slot, claimed if absent; -1 when the bucket is full.
fn hgInsert(k: vec2u) -> i32 {
  for (var i = 0u; i < ${BUCKET}u; i++) {
    if (atomicLoad(&keys[k.x + i]) == k.y) { return i32(k.x + i); }
  }
  for (var i = 0u; i < ${BUCKET}u; i++) {
    let slot = k.x + i;
    if (atomicLoad(&keys[slot]) == 0u) {
      var res = atomicCompareExchangeWeak(&keys[slot], 0u, k.y);
      if (!res.exchanged && res.old_value == 0u) { res = atomicCompareExchangeWeak(&keys[slot], 0u, k.y); }
      if (res.exchanged || res.old_value == k.y) { return i32(slot); }
    }
  }
  return -1;
}

@compute @workgroup_size(64)
fn hgUpdate(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r >= min(rcount[1], hg.recordCap)) { return; }
  let b = r * ${RECORD_VEC4}u;
  let p = records[b].xyz;
  let t = records[b + 3u].xyz;
  let bin = hgBin(hgNormal(records[b + 1u].zw)) | (hgDirBin(records[b + 1u].xy, records[b + 2u].w) << 4u);
  let l0 = hgLevel(p);
  let f = atomicLoad(&stats[0]);
  let u = vec3f(hgRand(r, f * 3u + 7u), hgRand(r, f * 3u + 8u), hgRand(r, f * 3u + 9u));
  let q = vec3u(floor(clamp(t, vec3f(0.0), vec3f(${CLAMP}.0)) * ${SCALE}.0 + u));
  // The record's own cell, and with \`pyramid\` the coarser cells a lookup falls back to (so they always hold data).
  for (var j = 0u; j <= HG_PYRAMID; j++) {
    let s = hgInsert(hgKey(p, bin, l0 + j));
    if (s < 0) {
      atomicAdd(&stats[3], 1u);
      continue;
    }
    let a = u32(s) * 4u;
    atomicAdd(&accum[a], q.x);
    atomicAdd(&accum[a + 1u], q.y);
    atomicAdd(&accum[a + 2u], q.z);
    atomicAdd(&accum[a + 3u], 1u);
  }
}
`;

const RESOLVE_WGSL = (fallback: number) => /* wgsl */ `
${COMMON_WGSL(fallback)}
@group(0) @binding(1) var<storage, read_write> keys: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> resolved: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> age: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

@compute @workgroup_size(256)
fn hgResolve(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i > hg.mask) { return; }
  if (atomicLoad(&keys[i]) == 0u) { return; }
  let f = atomicLoad(&stats[0]);
  let a = i * 4u;
  let n = atomicLoad(&accum[a + 3u]);
  if (n > 0u) {
    let sum = vec3f(f32(atomicLoad(&accum[a])), f32(atomicLoad(&accum[a + 1u])), f32(atomicLoad(&accum[a + 2u]))) * (1.0 / ${SCALE}.0);
    let v = resolved[i];
    let h = v.w * hg.decay;
    let w = h + f32(n);
    resolved[i] = vec4f((v.xyz * h + sum) / w, w);
    age[i] = f;
    atomicStore(&accum[a], 0u);
    atomicStore(&accum[a + 1u], 0u);
    atomicStore(&accum[a + 2u], 0u);
    atomicStore(&accum[a + 3u], 0u);
  } else if (f - age[i] > hg.stale) {
    atomicStore(&keys[i], 0u);
    resolved[i] = vec4f(0.0);
  }
}

// One thread after the resolve: the frame counter (stats[0]) that ages cells and seeds the rounding and jitter.
@compute @workgroup_size(1)
fn hgTick() {
  atomicAdd(&stats[0], 1u);
}
`;

/** The grid's three modules (lookup, update, resolve + tick) for `fallbackLevels` coarser lookups; exported for the WGSL name test. */
export function hashGridWgsl(fallbackLevels: number, pyramid = false): { lookup: string; update: string; resolve: string } {
  const fb = Math.max(0, Math.floor(fallbackLevels));
  return { lookup: LOOKUP_WGSL(fb), update: UPDATE_WGSL(fb, pyramid), resolve: RESOLVE_WGSL(fb) };
}

export interface HashGridStats {
  /** Frames resolved since the last clear. */
  frames: number;
  /** Lookups that found nothing (written 0), lookups answered by a coarser level, records dropped (bucket full); cumulative since `resetStats`. */
  misses: number;
  coarse: number;
  dropped: number;
  /** Cells in use (a full scan of the key table: debug only). */
  cells: number;
}

/** The hash grid: lookup over query records, update from training records, resolve. Everything encodes into the caller's encoder. */
export class HashGrid {
  readonly settings: HashGridSettings;
  readonly entries: number;
  private readonly device: GPUDevice;
  private readonly params: GPUBuffer;
  private readonly keys: GPUBuffer;
  private readonly accum: GPUBuffer;
  private readonly resolved: GPUBuffer;
  private readonly age: GPUBuffer;
  readonly stats: GPUBuffer;
  private readonly lookupPipeline: GPUComputePipeline;
  private readonly updatePipeline: GPUComputePipeline;
  private readonly resolvePipeline: GPUComputePipeline;
  private readonly tickPipeline: GPUComputePipeline;
  private readonly resolveGroup: GPUBindGroup;
  private readonly tickGroup: GPUBindGroup;
  private lookupGroup: GPUBindGroup | null = null;
  private updateGroup: GPUBindGroup | null = null;
  private queryCap = 0;
  private recordCap = 0;
  private readonly pixelAngle: number;
  private readonly paramWords = new ArrayBuffer(64);

  private constructor(init: Record<string, unknown>) {
    this.settings = init.settings as HashGridSettings;
    this.entries = init.entries as number;
    this.device = init.device as GPUDevice;
    this.params = init.params as GPUBuffer;
    this.keys = init.keys as GPUBuffer;
    this.accum = init.accum as GPUBuffer;
    this.resolved = init.resolved as GPUBuffer;
    this.age = init.age as GPUBuffer;
    this.stats = init.stats as GPUBuffer;
    this.lookupPipeline = init.lookupPipeline as GPUComputePipeline;
    this.updatePipeline = init.updatePipeline as GPUComputePipeline;
    this.resolvePipeline = init.resolvePipeline as GPUComputePipeline;
    this.tickPipeline = init.tickPipeline as GPUComputePipeline;
    this.resolveGroup = init.resolveGroup as GPUBindGroup;
    this.tickGroup = init.tickGroup as GPUBindGroup;
    this.pixelAngle = init.pixelAngle as number;
  }

  /** `yFovDeg` and `filmHeight` set the pixel footprint the LOD rule uses. */
  static async create(device: GPUDevice, settings: Partial<HashGridSettings>, view: { yFovDeg: number; filmHeight: number }): Promise<HashGrid> {
    const s: HashGridSettings = { ...DEFAULT_HASH_GRID, ...settings };
    if (s.log2Entries < 10 || s.log2Entries > 22) throw new Error('HashGrid: log2Entries must be 10..22 (the resolve dispatch and the 128 MiB binding cap)');
    const entries = 2 ** s.log2Entries;
    const { STORAGE, COPY_DST, COPY_SRC, UNIFORM } = GPUBufferUsage;
    const buf = (label: string, size: number, usage = STORAGE | COPY_DST | COPY_SRC) => device.createBuffer({ label, size, usage });
    const params = buf('hash grid params', 64, UNIFORM | COPY_DST);
    const keys = buf('hash grid keys', entries * 4);
    const accum = buf('hash grid accumulators', entries * 16);
    const resolved = buf('hash grid resolved', entries * 16);
    const age = buf('hash grid age', entries * 4);
    const stats = buf('hash grid stats', 16);
    const code = hashGridWgsl(s.fallbackLevels, s.pyramid);
    const make = (label: string, code: string, entryPoint: string) => device.createComputePipelineAsync({ label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint } });
    const [lookupPipeline, updatePipeline, resolvePipeline, tickPipeline] = await Promise.all([
      make('hash grid lookup', code.lookup, 'hgLookup'),
      make('hash grid update', code.update, 'hgUpdate'),
      make('hash grid resolve', code.resolve, 'hgResolve'),
      make('hash grid tick', code.resolve, 'hgTick'),
    ]);
    const resolveGroup = device.createBindGroup({ layout: resolvePipeline.getBindGroupLayout(0), entries: [params, keys, accum, resolved, age, stats].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const tickGroup = device.createBindGroup({ layout: tickPipeline.getBindGroupLayout(0), entries: [{ binding: 5, resource: { buffer: stats } }] });
    const grid = new HashGrid({
      settings: s, entries, device, params, keys, accum, resolved, age, stats, lookupPipeline, updatePipeline, resolvePipeline, tickPipeline, resolveGroup, tickGroup,
      pixelAngle: (2 * Math.tan((view.yFovDeg * Math.PI) / 360)) / view.filmHeight,
    });
    grid.clear();
    return grid;
  }

  /** Binds the query records (+ their count, word 0) and the prediction buffer the lookup writes. */
  bindQueries(queries: GPUBuffer, count: GPUBuffer, preds: GPUBuffer, capacity: number): void {
    this.queryCap = capacity;
    this.lookupGroup = this.device.createBindGroup({ layout: this.lookupPipeline.getBindGroupLayout(0), entries: [this.params, this.keys, this.resolved, queries, preds, count, this.stats].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.writeParams();
  }

  /** Binds the training records (+ their count, word 1) the update adds into the cells. */
  bindRecords(records: GPUBuffer, count: GPUBuffer, capacity: number): void {
    this.recordCap = capacity;
    this.updateGroup = this.device.createBindGroup({ layout: this.updatePipeline.getBindGroupLayout(0), entries: [this.params, this.keys, this.accum, records, count, this.stats].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.writeParams();
  }

  private camera: [number, number, number] = [0, 0, 0];

  /** The LOD rule measures distances from here. */
  setCameraPosition(position: readonly [number, number, number] | readonly number[]): void {
    this.camera = [position[0] as number, position[1] as number, position[2] as number];
    this.writeParams();
  }

  private writeParams(): void {
    const s = this.settings;
    const f = new Float32Array(this.paramWords);
    const u = new Uint32Array(this.paramWords);
    f.set([...this.camera, this.pixelAngle * s.cellPixels, s.minCell, s.jitter, 1 - 1 / Math.max(1, s.historyFrames), s.minSamples]);
    u.set([this.entries - 1, s.staleFrames, s.normalBins ? 1 : 0, this.queryCap, this.recordCap, Math.max(0, Math.floor(s.dirBins)), 0, 0], 8);
    this.device.queue.writeBuffer(this.params, 0, this.paramWords);
  }

  /** Predictions for the bound queries: rgb, w = level offset of the hit (0 = own cell) or −1 for a miss (rgb 0). */
  encodeLookup(encoder: GPUCommandEncoder): void {
    if (!this.lookupGroup) throw new Error('HashGrid: bindQueries first');
    const pass = encoder.beginComputePass({ label: 'hash grid lookup' });
    pass.setPipeline(this.lookupPipeline);
    pass.setBindGroup(0, this.lookupGroup);
    pass.dispatchWorkgroups(Math.ceil(this.queryCap / 64));
    pass.end();
  }

  /** Adds the bound training records into their cells, then resolves every cell and advances the frame counter. */
  encodeUpdate(encoder: GPUCommandEncoder): void {
    if (!this.updateGroup) throw new Error('HashGrid: bindRecords first');
    const pass = encoder.beginComputePass({ label: 'hash grid update' });
    pass.setPipeline(this.updatePipeline);
    pass.setBindGroup(0, this.updateGroup);
    pass.dispatchWorkgroups(Math.ceil(this.recordCap / 64));
    pass.end();
    const resolve = encoder.beginComputePass({ label: 'hash grid resolve' });
    resolve.setPipeline(this.resolvePipeline);
    resolve.setBindGroup(0, this.resolveGroup);
    resolve.dispatchWorkgroups(this.entries / 256);
    resolve.setPipeline(this.tickPipeline);
    resolve.setBindGroup(0, this.tickGroup);
    resolve.dispatchWorkgroups(1);
    resolve.end();
  }

  /** Empties the table (submitted now; queue order keeps it before any later frame). */
  clear(): void {
    const e = this.device.createCommandEncoder({ label: 'hash grid clear' });
    for (const b of [this.keys, this.accum, this.resolved, this.age, this.stats]) e.clearBuffer(b);
    this.device.queue.submit([e.finish()]);
  }

  /** Debug only (never in a frame loop). */
  async readStats(): Promise<HashGridStats> {
    const staging = this.device.createBuffer({ size: 16 + this.entries * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = this.device.createCommandEncoder();
    e.copyBufferToBuffer(this.stats, 0, staging, 0, 16);
    e.copyBufferToBuffer(this.keys, 0, staging, 16, this.entries * 4);
    this.device.queue.submit([e.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(staging.getMappedRange());
    let cells = 0;
    for (let i = 4; i < v.length; i++) if (v[i] !== 0) cells++;
    const out = { frames: v[0] as number, misses: v[1] as number, coarse: v[2] as number, dropped: v[3] as number, cells };
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Zeroes the miss / coarse / dropped counters (keeps the frame counter). */
  resetStats(): void {
    this.device.queue.writeBuffer(this.stats, 4, new Uint32Array([0, 0, 0]));
  }

  destroy(): void {
    for (const b of [this.params, this.keys, this.accum, this.resolved, this.age, this.stats]) b.destroy();
  }
}

/** Hash grid settings from page args (`hgLog2 hgMinCell hgCellPx hgNormals hgHistory hgStale hgFallback hgPyramid hgMinSamples hgJitter hgDirBins`); absent keys keep the defaults. */
export function hashGridArgs(args: Record<string, string | undefined>): Partial<HashGridSettings> {
  const out: Partial<HashGridSettings> = {};
  const num = (key: string, field: keyof HashGridSettings) => {
    const v = args[key];
    if (v !== undefined) (out as Record<string, number>)[field] = Number(v);
  };
  num('hgLog2', 'log2Entries');
  num('hgMinCell', 'minCell');
  num('hgCellPx', 'cellPixels');
  num('hgHistory', 'historyFrames');
  num('hgStale', 'staleFrames');
  num('hgFallback', 'fallbackLevels');
  num('hgMinSamples', 'minSamples');
  num('hgJitter', 'jitter');
  num('hgDirBins', 'dirBins');
  if (args.hgNormals !== undefined) out.normalBins = args.hgNormals !== '0';
  if (args.hgPyramid !== undefined) out.pyramid = args.hgPyramid !== '0';
  return out;
}
