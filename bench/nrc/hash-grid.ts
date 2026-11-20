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
 *    only in frames that bring new samples); cells without samples for more than `staleFrames` frames are evicted.
 *  - `pyramid`: each record also adds into its `fallbackLevels` coarser cells (a multi-resolution grid; off = one level per record).
 *  - Lookup: the query's cell; on a miss (absent, or fewer than `minSamples`) up to `fallbackLevels` coarser cells; still a miss: 0 (the
 *    same as a fresh network with a zero output layer). Optional jitter in the tangent plane (Binder's jittered lookup).
 */

export interface HashGridSettings {
  /** Resolve only touched cells and check age bounds before scanning a block for eviction. */
  activeMaintenance: boolean;
  /** Maximum display-cell population variance in normalized log2 luminance squared. Negative disables tracking. */
  varianceThreshold: number;
  /** Continue the display path on a miss at the requested bounce. Training targets retain the fixed-depth control. */
  continueOnMiss: boolean;
  /** Table size: 2^log2Entries cells (20: 1M RGB cells, 40 MiB). */
  log2Entries: number;
  /** Optional exact GPU allocation in bytes. Overrides log2Entries and fits whole resolve groups. */
  memoryBudgetBytes?: number;
  /** Smallest cell, in scene units (metres). */
  minCell: number;
  /** Cell size target in pixels at the cell's distance from the camera (the LOD rule). */
  cellPixels: number;
  /** Separate cells per dominant normal axis and sign (6 bins), or one cell for all normals. */
  normalBins: boolean;
  /** Time constant of the per-cell running mean, in frames. */
  historyFrames: number;
  /** Maximum age of an untouched cell, measured in frames since its last sample. */
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
  /** Fit first-order spherical harmonics at moderately glossy vertices in supported cells. */
  directionalSh: boolean;
}

export const DEFAULT_HASH_GRID: HashGridSettings = {
  activeMaintenance: false,
  varianceThreshold: -1,
  continueOnMiss: false,
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
  directionalSh: false,
};

/** Bucket size (slots probed per lookup); a compile-time constant so the loops have a constant trip count (Tint loop counters on D3D12). */
const BUCKET = 8;
const AGE_BLOCK = 64;
const RESOLVE_GROUP_SIZE = 256;
/** Fixed-point scale of the accumulators, and the per-sample clamp that keeps a u32 sum from overflowing (≥ 16k clamped samples). */
const SCALE = 256;
const CLAMP = 1000;
const MAX_LEVEL = 24;
const SH_WORDS = 21; // Eight direction moments, three RGB totals, nine cross-moments, and the sample count.
const SH_COEFFICIENT_BYTES = 48;
const SH_DIRECTION_SCALE = 16384;
const SH_COLOR_SCALE = SCALE / 2;
const MIN_SH_SAMPLES = 16;
const MAX_STORAGE_BINDING_BYTES = 128 * 1024 ** 2;
/** Attempts of the weak compare-exchange on one slot that still reads empty, before the insert is refused (pmndrs/pathtracer#1). */
const CAS_TRIES = 4;

/**
 * `hgInsert(k) -> i32`: the cell's slot in its bucket, claimed if absent; -1 when the bucket is full (or, in practice never, when
 * `casTries` weak exchanges on an empty slot all fail spuriously). The caller declares `keys: array<atomic<u32>>`.
 *
 * One key never gets two slots (pmndrs/pathtracer#1): during an update pass slots only go from 0 to a key, and every thread probes
 * the bucket in the same order and never passes a slot without learning its final key. A slot that reads non-zero is final (take it
 * if it is ours); one that reads 0 is exchanged until it holds a key (ours or another's). WGSL only has the weak exchange, which
 * may fail with the slot still empty; moving on then could claim a second slot behind a concurrent winner, so after `casTries`
 * such failures the insert is refused instead (a dropped sample, counted with the full-bucket drops). Both loops have literal
 * trip counts (Tint loop counters on D3D12). Eviction (`hgEvict`) runs in its own dispatch, never during inserts.
 * Exported for `bench/pages/hash-insert-check.ts`, which runs this exact text; `hash-grid.test.ts` models the protocol.
 */
export function hashGridInsertWgsl(bucket = BUCKET, casTries = CAS_TRIES): string {
  return /* wgsl */ `
fn hgInsert(k: vec2u) -> i32 {
  for (var i = 0u; i < ${bucket}u; i++) {
    if (atomicLoad(&keys[k.x + i]) == k.y) { return i32(k.x + i); }
  }
  for (var i = 0u; i < ${bucket}u; i++) {
    let slot = k.x + i;
    var owner = atomicLoad(&keys[slot]);
    for (var t = 0u; t < ${casTries}u; t++) {
      if (owner != 0u) { break; }
      let res = atomicCompareExchangeWeak(&keys[slot], 0u, k.y);
      if (res.exchanged) { return i32(slot); }
      owner = res.old_value;
    }
    // Still empty after spurious failures: refuse rather than risk a second slot further on.
    if (owner == 0u) { return -1; }
    if (owner == k.y) { return i32(slot); }
  }
  return -1;
}
`;
}

/** Allocation and per-insert payload accounting, excluding key probes and optional active-list bookkeeping. */
export function hashGridStorage(settings: Partial<HashGridSettings> = {}) {
  const s = { ...DEFAULT_HASH_GRID, ...settings };
  if (s.directionalSh && s.dirBins !== 0) throw new Error('HashGrid: directionalSh and dirBins are mutually exclusive');
  const variance = s.varianceThreshold >= 0;
  const valueBytesPerEntry = 16 + (s.directionalSh ? SH_COEFFICIENT_BYTES : 0);
  const snapshotBytesPerEntry = 4 + valueBytesPerEntry + (variance ? 8 : 0);
  const accumBytesPerEntry = 16 + (s.directionalSh ? SH_WORDS * 4 : 0);
  const ageBytesPerEntry = 4 + (s.directionalSh ? SH_WORDS * 4 : 0);
  const momentBytesPerEntry = variance ? 16 : 0;
  const activeBytesPerEntry = s.activeMaintenance ? 4 + 8 / AGE_BLOCK : 0;
  const statsBytes = variance ? 24 : 20;
  const fixedBytes = 64 + statsBytes + (s.activeMaintenance ? 16 : 0);
  const bytesPerEntry = snapshotBytesPerEntry + accumBytesPerEntry + ageBytesPerEntry + momentBytesPerEntry + activeBytesPerEntry;
  const budget = s.memoryBudgetBytes;
  if (budget !== undefined && (!Number.isSafeInteger(budget) || budget % 4 !== 0)) {
    throw new Error('HashGrid: memoryBudgetBytes must be a whole number of 32-bit words');
  }
  if (budget === undefined && (!Number.isInteger(s.log2Entries) || s.log2Entries < 10 || s.log2Entries > 22)) {
    throw new Error('HashGrid: log2Entries must be an integer from 10 to 22');
  }
  const entries = budget === undefined ? 2 ** s.log2Entries
    : Math.floor((budget - fixedBytes) / (bytesPerEntry * RESOLVE_GROUP_SIZE)) * RESOLVE_GROUP_SIZE;
  if (entries < 1024 || entries > 2 ** 22) throw new Error('HashGrid: table must have between 1024 and 4194304 cells');
  const bytes = budget ?? (fixedBytes + entries * bytesPerEntry);
  const padding = bytes - (fixedBytes + entries * bytesPerEntry);
  const snapshotBytes = entries * snapshotBytesPerEntry;
  const accumBytes = entries * accumBytesPerEntry;
  const ageBytes = entries * ageBytesPerEntry + padding;
  if (Math.max(snapshotBytes, accumBytes, ageBytes) > MAX_STORAGE_BINDING_BYTES) {
    throw new Error('HashGrid: table exceeds the 128 MiB storage binding cap');
  }
  return {
    entries, bytes, bytesPerEntry, snapshotBytes, accumBytes, ageBytes, statsBytes,
    valueBytes: entries * valueBytesPerEntry,
    momentBytes: entries * momentBytesPerEntry,
    updateAtomicAddsPerCell: variance ? 6 : 4,
    directionalUpdateAtomicAddsPerCell: s.directionalSh ? SH_WORDS : 0,
  };
}

/** Records are `NRC_RECORD_VEC4` = 4 vec4 each (bench/nrc/cache.ts): (position, pixel), (wo θ/π φ/2π, normal θ/π φ/2π), (albedo, 1), (value, flag). */
const RECORD_VEC4 = 4;

const COMMON_WGSL = (fallback: number, pyramid = false, group = 0, paramsBinding = 0, budgeted = false) => /* wgsl */ `
struct HgParams {
  cam: vec4f,
  minCell: f32, jitter: f32, decay: f32, minSamples: f32,
  mask: u32, stale: u32, normals: u32, queryCap: u32,
  recordCap: u32, dirBins: u32, varianceThreshold: f32, _c: u32,
}
@group(${group}) @binding(${paramsBinding}) var<uniform> hg: HgParams;

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
  return vec2u(${budgeted ? `(h / ${BUCKET}u % ((hg.mask + 1u) / ${BUCKET}u)) * ${BUCKET}u` : `h & hg.mask & ~${BUCKET - 1}u`}, max(g, 1u));
}
const HG_FALLBACK = ${fallback}u;
const HG_PYRAMID = ${pyramid ? fallback : 0}u;
`;

const SH_COMMON_WGSL = /* wgsl */ `
fn hgIsModeratelyGlossy(rough: f32) -> bool { return rough >= 0.12 && rough < 0.5; }
`;

// The path hook and deferred lookup read the same resolved snapshot and apply the same acceptance rule.
function lookupFunctions(fallback: number, group: number, first: number, variance = false, directionalSh = false, budgeted = false): string {
  return /* wgsl */ `
${COMMON_WGSL(fallback, false, group, first, budgeted)}
@group(${group}) @binding(${first + 1}) var<storage, read> hgSnapshot: array<vec4u>;
// Checksums, base values, optional directional coefficients, and optional variance moments.
fn hgChecksum(i: u32) -> u32 { return hgSnapshot[i / 4u][i % 4u]; }
fn hgValue(i: u32) -> vec4f { return bitcast<vec4f>(hgSnapshot[(hg.mask + 1u) / 4u + i]); }

${directionalSh ? /* wgsl */ `
${SH_COMMON_WGSL}
fn hgDirectionalReady(i: u32) -> bool { return hgValue(i).w < 0.0; }
fn hgDirectionalValue(i: u32, d: vec3f) -> vec3f {
  let entries = hg.mask + 1u;
  let offset = entries / 4u + entries + i * 3u;
  let b = vec4f(d, 1.0);
  return vec3f(dot(bitcast<vec4f>(hgSnapshot[offset]), b),
    dot(bitcast<vec4f>(hgSnapshot[offset + 1u]), b), dot(bitcast<vec4f>(hgSnapshot[offset + 2u]), b));
}
` : ''}
fn hgFind(k: vec2u) -> i32 {
  for (var i = 0u; i < ${BUCKET}u; i++) {
    if (hgChecksum(k.x + i) == k.y) { return i32(k.x + i); }
  }
  return -1;
}

${variance ? /* wgsl */ `
struct HgConfidenceSample {
  prediction: vec4f,
  varianceRejected: bool,
}

fn hgVarianceRejects(i: u32) -> bool {
  let entries = hg.mask + 1u;
  let moments = bitcast<vec4f>(hgSnapshot[entries / 4u + entries * ${directionalSh ? 4 : 1}u + i / 2u]);
  let pair = select(moments.xy, moments.zw, i % 2u == 1u);
  // Subtract an FP32 tolerance for cancellation when every sample quantizes identically.
  return max(pair.y - pair.x * pair.x - 2e-7, 0.0) > hg.varianceThreshold;
}

fn hgSampleImpl(p0: vec3f, angles: vec4f, rough: f32, r: u32, f: u32, checkVariance: bool) -> HgConfidenceSample {
  var rejected = false;` : /* wgsl */ `
fn hgSample(p0: vec3f, angles: vec4f, rough: f32, r: u32, f: u32) -> vec4f {`}
  let n = hgNormal(angles.zw);
  let l0 = hgLevel(p0);
  var p = p0;
  if (hg.jitter > 0.0) {
    var o = (vec3f(hgRand(r, f * 3u), hgRand(r, f * 3u + 1u), hgRand(r, f * 3u + 2u)) - 0.5) * (hg.jitter * hgCellSize(l0));
    o -= n * dot(o, n);
    p += o;
  }
  let bin = hgBin(n) | (hgDirBin(angles.xy, rough) << 4u);
  var out = vec4f(0.0, 0.0, 0.0, -1.0);
  for (var j = 0u; j <= HG_FALLBACK; j++) {
    let s = hgFind(hgKey(p, bin, l0 + j));
    if (s >= 0) {
      let v = hgValue(u32(s));
      if (${directionalSh ? 'abs(v.w)' : 'v.w'} >= hg.minSamples) {
        ${directionalSh ? /* wgsl */ `
        var value = v.xyz;
        if (v.w < 0.0 && hgIsModeratelyGlossy(rough)) {
          value = clamp(hgDirectionalValue(u32(s), hgNormal(angles.xy)), vec3f(0.0), vec3f(${CLAMP}.0));
        }
        out = vec4f(value, f32(j));` : 'out = vec4f(v.xyz, f32(j));'}
        ${variance ? 'if (checkVariance) { rejected = hgVarianceRejects(u32(s)); }' : ''}
        break;
      }
    }
  }
  return ${variance ? 'HgConfidenceSample(out, rejected)' : 'out'};
}

${variance ? /* wgsl */ `
fn hgSample(p: vec3f, angles: vec4f, rough: f32, r: u32, f: u32) -> vec4f {
  return hgSampleImpl(p, angles, rough, r, f, false).prediction;
}
fn hgSampleWithConfidence(p: vec3f, angles: vec4f, rough: f32, r: u32, f: u32) -> HgConfidenceSample {
  return hgSampleImpl(p, angles, rough, r, f, true);
}
` : ''}
`;
}

const LOOKUP_WGSL = (fallback: number, variance: boolean, directionalSh: boolean, budgeted: boolean) => /* wgsl */ `
${lookupFunctions(fallback, 0, 0, false, directionalSh, budgeted)}
@group(0) @binding(2) var<storage, read> queries: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> preds: array<vec4f>;
@group(0) @binding(4) var<storage, read> qcount: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

var<workgroup> hgContinued: atomic<u32>;
var<workgroup> hgMisses: atomic<u32>;
var<workgroup> hgCoarse: atomic<u32>;
${variance ? 'var<workgroup> hgVarianceRejected: atomic<u32>;' : ''}

@compute @workgroup_size(64)
fn hgLookup(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    atomicStore(&hgContinued, 0u);
    atomicStore(&hgMisses, 0u);
    atomicStore(&hgCoarse, 0u);
    ${variance ? 'atomicStore(&hgVarianceRejected, 0u);' : ''}
  }
  workgroupBarrier();
  let r = gid.x;
  if (r < min(qcount[0], hg.queryCap)) {
    let b = r * ${RECORD_VEC4}u;
    let queryKind = queries[b + 3u].w;
    let coarseRejected = queryKind == 4.0;
    ${variance ? 'let varianceRejected = queryKind == 5.0;' : ''}
    let inlineDisplay = queryKind == 3.0 || coarseRejected${variance ? ' || varianceRejected' : ''};
    var out = queries[b + 1u];
    if (!inlineDisplay) {
      out = hgSample(queries[b].xyz, queries[b + 1u], queries[b + 2u].w, r, atomicLoad(&stats[0]));
    } else if (out.w < 0.0 || coarseRejected${variance ? ' || varianceRejected' : ''}) {
      atomicAdd(&hgContinued, 1u);
    }
    ${variance ? 'if (varianceRejected) { atomicAdd(&hgVarianceRejected, 1u); }' : ''}
    preds[r] = out;
    if (out.w < 0.0) { atomicAdd(&hgMisses, 1u); } else if (out.w > 0.0) { atomicAdd(&hgCoarse, 1u); }
  }
  workgroupBarrier();
  if (li == 0u) {
    let continued = atomicLoad(&hgContinued);
    if (continued > 0u) { atomicAdd(&stats[4], continued); }
    let m = atomicLoad(&hgMisses);
    let c = atomicLoad(&hgCoarse);
    if (m > 0u) { atomicAdd(&stats[1], m); }
    if (c > 0u) { atomicAdd(&stats[2], c); }
    ${variance ? /* wgsl */ `
    let rejected = atomicLoad(&hgVarianceRejected);
    if (rejected > 0u) { atomicAdd(&stats[5], rejected); }` : ''}
  }
}
`;

// Word 0 is the frame's touched-slot count, followed by one slot per possible entry.
// Then each age block has an occupancy state (0 empty, 1 active, 2 newly active) and oldest age.
const ACTIVE_STORAGE_WGSL = /* wgsl */ `
@group(0) @binding(6) var<storage, read_write> activeCells: array<atomic<u32>>;
fn hgBlockState(block: u32) -> u32 { return hg.mask + 2u + block * 2u; }
`;

const MOMENT_STORAGE_WGSL = /* wgsl */ `
@group(0) @binding(8) var<storage, read_write> momentSums: array<atomic<u32>>;
`;

const SH_UPDATE_WGSL = /* wgsl */ `
${SH_COMMON_WGSL}
fn hgAddDirection(i: u32, d: vec3f, q: vec3u) {
  let a = (hg.mask + 1u) * 4u + i * ${SH_WORDS}u;
  // The zz moment follows from the trace, since the decoded direction has unit length.
  let moments = array<f32, 8>(d.x, d.y, d.z, d.x * d.x, d.x * d.y, d.x * d.z, d.y * d.y, d.y * d.z);
  for (var j = 0u; j < 8u; j++) {
    atomicAdd(&accum[a + j], bitcast<u32>(i32(round(moments[j] * ${SH_DIRECTION_SCALE}.0))));
  }
  for (var j = 0u; j < 3u; j++) {
    let crossMoment = vec3i(round(f32(q[j]) * d));
    atomicAdd(&accum[a + 8u + j], q[j]);
    atomicAdd(&accum[a + 11u + j * 3u], bitcast<u32>(crossMoment.x));
    atomicAdd(&accum[a + 12u + j * 3u], bitcast<u32>(crossMoment.y));
    atomicAdd(&accum[a + 13u + j * 3u], bitcast<u32>(crossMoment.z));
  }
  atomicAdd(&accum[a + 20u], 1u);
}
`;

const UPDATE_WGSL = (fallback: number, pyramid: boolean, active: boolean, variance: boolean, directionalSh: boolean, budgeted: boolean) => /* wgsl */ `
${COMMON_WGSL(fallback, pyramid, 0, 0, budgeted)}
@group(0) @binding(1) var<storage, read_write> keys: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> records: array<vec4f>;
@group(0) @binding(4) var<storage, read> rcount: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

${directionalSh ? SH_UPDATE_WGSL : ''}
${active ? ACTIVE_STORAGE_WGSL : ''}
${variance ? /* wgsl */ `
${MOMENT_STORAGE_WGSL}
// Each sum has a low and high word. No floating-point atomics or batch-size-dependent clipping.
fn hgAddMoments(a: u32, q: u32) {
  let square = q * q;
  let sumLow = atomicAdd(&momentSums[a], q);
  let squareLow = atomicAdd(&momentSums[a + 2u], square);
  if (sumLow > 0xffffffffu - q) { atomicAdd(&momentSums[a + 1u], 1u); }
  if (squareLow > 0xffffffffu - square) { atomicAdd(&momentSums[a + 3u], 1u); }
}
` : ''}
${hashGridInsertWgsl()}
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
  ${directionalSh ? /* wgsl */ `
  let directional = hgIsModeratelyGlossy(records[b + 2u].w);
  var d = vec3f(0.0);
  var qSh = vec3u(0u);
  if (directional) {
    d = hgNormal(records[b + 1u].xy);
    qSh = vec3u(floor(clamp(t, vec3f(0.0), vec3f(${CLAMP}.0)) * ${SH_COLOR_SCALE}.0 + u));
  }` : ''}
  ${variance ? /* wgsl */ `
  let luminance = dot(clamp(t, vec3f(0.0), vec3f(${CLAMP}.0)), vec3f(0.2126, 0.7152, 0.0722));
  let logLuminance = clamp(log2(1.0 + luminance) / log2(${CLAMP + 1}.0), 0.0, 1.0);
  let momentSample = u32(round(logLuminance * 65535.0));` : ''}
  // The record's own cell, and with \`pyramid\` the coarser cells a lookup falls back to (so they always hold data).
  for (var j = 0u; j <= HG_PYRAMID; j++) {
    let s = hgInsert(hgKey(p, bin, l0 + j));
    if (s < 0) {
      atomicAdd(&stats[3], 1u);
      continue;
    }
    let a = u32(s) * 4u;
    ${directionalSh ? 'if (directional) { hgAddDirection(u32(s), d, qSh); }' : ''}
    ${variance ? 'hgAddMoments(a, momentSample);' : ''}
    atomicAdd(&accum[a], q.x);
    atomicAdd(&accum[a + 1u], q.y);
    atomicAdd(&accum[a + 2u], q.z);
    ${active ? /* wgsl */ `
    // The first sample for a slot appends exactly one item, even if others updated RGB first.
    if (atomicAdd(&accum[a + 3u], 1u) == 0u) {
      let index = atomicAdd(&activeCells[0], 1u);
      atomicStore(&activeCells[index + 1u], u32(s));
      let state = hgBlockState(u32(s) / ${AGE_BLOCK}u);
      if (atomicLoad(&activeCells[state]) == 0u) { atomicStore(&activeCells[state], 2u); }
    }` : 'atomicAdd(&accum[a + 3u], 1u);'}
  }
}
`;

const SH_RESOLVE_WGSL = /* wgsl */ `
fn hgDirectionalWeightOffset(i: u32) -> u32 { return hg.mask + 1u + i * ${SH_WORDS}u + 20u; }

fn hgResolveDirection(i: u32, wasReady: bool) -> bool {
  let weightOffset = hgDirectionalWeightOffset(i);
  let frameOffset = (hg.mask + 1u) * 4u + i * ${SH_WORDS}u;
  let n = atomicExchange(&accum[frameOffset + 20u], 0u);
  let h = bitcast<f32>(age[weightOffset]) * hg.decay;
  let w = h + f32(n);
  age[weightOffset] = bitcast<u32>(w);
  let enough = w >= max(hg.minSamples, ${MIN_SH_SAMPLES}.0);
  if (n == 0u) { return enough && wasReady; }

  let historyOffset = weightOffset - 20u;
  var moments: array<f32, 20>;
  for (var j = 0u; j < 20u; j++) {
    let sum = f32(bitcast<i32>(atomicExchange(&accum[frameOffset + j], 0u)));
    let scale = select(${SH_COLOR_SCALE}.0, ${SH_DIRECTION_SCALE}.0, j < 8u);
    moments[j] = (bitcast<f32>(age[historyOffset + j]) * h + sum / scale) / w;
    age[historyOffset + j] = bitcast<u32>(moments[j]);
  }
  if (!enough) { return false; }

  // Center the normal equations so the fit represents a constant plus the three l=1 terms,
  // even for nonuniform directions. The trace gate and ridge suppress unsupported slopes.
  let d = vec3f(moments[0], moments[1], moments[2]);
  if (1.0 - dot(d, d) < 0.02) { return false; }
  let ridge = 0.02;
  let xx = max(0.0, moments[3] - d.x * d.x) + ridge;
  let xy = moments[4] - d.x * d.y;
  let xz = moments[5] - d.x * d.z;
  let yy = max(0.0, moments[6] - d.y * d.y) + ridge;
  let yz = moments[7] - d.y * d.z;
  let zz = max(0.0, 1.0 - moments[3] - moments[6] - d.z * d.z) + ridge;
  let c0 = vec3f(yy * zz - yz * yz, xz * yz - xy * zz, xy * yz - xz * yy);
  let c1 = vec3f(c0.y, xx * zz - xz * xz, xy * xz - xx * yz);
  let c2 = vec3f(c0.z, c1.z, xx * yy - xy * xy);
  let det = dot(vec3f(xx, xy, xz), c0);
  if (det <= 1e-6) { return false; }
  let covarianceInv = mat3x3f(c0, c1, c2) * (1.0 / det);
  let coeffOffset = hg.mask + 1u + i * 3u;
  for (var j = 0u; j < 3u; j++) {
    let mean = max(moments[j + 8u], 0.0);
    let k = 11u + j * 3u;
    let crossMoment = vec3f(moments[k], moments[k + 1u], moments[k + 2u]);
    var slope = covarianceInv * (crossMoment - d * mean);
    // Bound amplification and preserve nonnegative radiance over the full unit sphere.
    let span = length(slope);
    let lowerExcursion = span + dot(d, slope);
    slope *= min(1.0, min(1.5 * mean / max(span, 1e-6), mean / max(lowerExcursion, 1e-6)));
    resolved[coeffOffset + j] = vec4f(slope, mean - dot(d, slope));
  }
  return true;
}
`;

const RESOLVE_WGSL = (fallback: number, active: boolean, variance: boolean, directionalSh: boolean, budgeted: boolean) => /* wgsl */ `
${COMMON_WGSL(fallback, false, 0, 0, budgeted)}
@group(0) @binding(1) var<storage, read_write> keys: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> resolved: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> age: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;

${directionalSh ? SH_RESOLVE_WGSL : ''}
${variance ? /* wgsl */ `
${MOMENT_STORAGE_WGSL}
@group(0) @binding(9) var<storage, read_write> resolvedMoments: array<vec2f>;

fn hgTakeMoment(a: u32) -> f32 {
  return f32(atomicExchange(&momentSums[a + 1u], 0u)) * 4294967296.0 + f32(atomicExchange(&momentSums[a], 0u));
}
` : ''}
fn hgResolveCell(i: u32, f: u32) {
  let a = i * 4u;
  let n = atomicLoad(&accum[a + 3u]);
  if (n > 0u) {
    let sum = vec3f(f32(atomicLoad(&accum[a])), f32(atomicLoad(&accum[a + 1u])), f32(atomicLoad(&accum[a + 2u]))) * (1.0 / ${SCALE}.0);
    let v = resolved[i];
    let h = ${directionalSh ? 'abs(v.w)' : 'v.w'} * hg.decay;
    let w = h + f32(n);
    ${directionalSh ? 'let fitted = hgResolveDirection(i, v.w < 0.0);' : ''}
    // SH uses the sign of the base weight to mark a usable fit without another snapshot word.
    resolved[i] = vec4f((v.xyz * h + sum) / w, ${directionalSh ? 'select(w, -w, fitted)' : 'w'});
    ${variance ? /* wgsl */ `
    let moment1 = hgTakeMoment(a) / 65535.0;
    let moment2 = hgTakeMoment(a + 2u) / (65535.0 * 65535.0);
    resolvedMoments[i] = (resolvedMoments[i] * h + vec2f(moment1, moment2)) / w;` : ''}
    age[i] = f;
    atomicStore(&accum[a], 0u);
    atomicStore(&accum[a + 1u], 0u);
    atomicStore(&accum[a + 2u], 0u);
    atomicStore(&accum[a + 3u], 0u);
  } else if (f - age[i] > hg.stale) {
    hgEvict(i);
  }
}

fn hgEvict(i: u32) {
  atomicStore(&keys[i], 0u);
  resolved[i] = vec4f(0.0);
  ${directionalSh ? 'age[hgDirectionalWeightOffset(i)] = 0u;' : ''}
  ${variance ? 'resolvedMoments[i] = vec2f(0.0);' : ''}
}

${active ? /* wgsl */ `
${ACTIVE_STORAGE_WGSL}
@group(0) @binding(7) var<storage, read_write> resolveDispatch: array<u32>;

@compute @workgroup_size(1)
fn hgPrepare() {
  resolveDispatch[0] = (atomicLoad(&activeCells[0]) + ${RESOLVE_GROUP_SIZE - 1}u) / ${RESOLVE_GROUP_SIZE}u;
  resolveDispatch[1] = 1u;
  resolveDispatch[2] = 1u;
}

@compute @workgroup_size(${RESOLVE_GROUP_SIZE})
fn hgResolve(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= atomicLoad(&activeCells[0])) { return; }
  let i = atomicLoad(&activeCells[gid.x + 1u]);
  hgResolveCell(i, atomicLoad(&stats[0]));
}

// A newer sample cannot make an active block expire sooner. Sweep at the earliest
// possible expiration (or when a block first becomes active), then record a fresh bound.
// Resolving touched cells first lets a sample on the expiration frame keep its history.
@compute @workgroup_size(64)
fn hgReclaim(@builtin(global_invocation_id) gid: vec3u) {
  let first = gid.x * ${AGE_BLOCK}u;
  if (first > hg.mask) { return; }
  let state = hgBlockState(gid.x);
  let occupancy = atomicLoad(&activeCells[state]);
  if (occupancy == 0u) { return; }
  let f = atomicLoad(&stats[0]);
  if (occupancy == 1u && f - atomicLoad(&activeCells[state + 1u]) <= hg.stale) { return; }

  var oldest = 0u;
  var populated = false;
  for (var offset = 0u; offset < ${AGE_BLOCK}u; offset++) {
    let i = first + offset;
    if (atomicLoad(&keys[i]) == 0u) { continue; }
    let elapsed = f - age[i];
    if (elapsed > hg.stale) {
      hgEvict(i);
    } else {
      oldest = max(oldest, elapsed);
      populated = true;
    }
  }
  atomicStore(&activeCells[state], select(0u, 1u, populated));
  atomicStore(&activeCells[state + 1u], f - oldest);
}
` : /* wgsl */ `
@compute @workgroup_size(${RESOLVE_GROUP_SIZE})
fn hgResolve(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i > hg.mask || atomicLoad(&keys[i]) == 0u) { return; }
  hgResolveCell(i, atomicLoad(&stats[0]));
}
`}

// One thread after the resolve: the frame counter (stats[0]) that ages cells and seeds the rounding and jitter.
@compute @workgroup_size(1)
fn hgTick() {
  atomicAdd(&stats[0], 1u);
}
`;

/** The grid's three modules (lookup, update, resolve + tick) for `fallbackLevels` coarser lookups; exported for the WGSL name test. */
export function hashGridWgsl(fallbackLevels: number, pyramid = false, active = false, variance = false, directionalSh = false, budgeted = false): { lookup: string; update: string; resolve: string } {
  const fb = Math.max(0, Math.floor(fallbackLevels));
  return {
    lookup: LOOKUP_WGSL(fb, variance, directionalSh, budgeted),
    update: UPDATE_WGSL(fb, pyramid, active, variance, directionalSh, budgeted),
    resolve: RESOLVE_WGSL(fb, active, variance, directionalSh, budgeted),
  };
}

export interface HashGridStats {
  /** Frames resolved since the last clear. */
  frames: number;
  /** Display paths traced past their requested cache depth after a miss, cumulative since resetStats. */
  continued: number;
  /** Misses including training tails, coarse hits and full-bucket drops, cumulative since resetStats. */
  misses: number;
  coarse: number;
  dropped: number;
  /** Cells in use (a full scan of the key table: debug only). */
  cells: number;
  /** Fraction of all slots touched during the most recent update, available in active mode. */
  touchedFraction?: number;
  /** Display paths continued because the selected cell exceeded the variance threshold. */
  varianceRejects?: number;
}

interface ActiveMaintenance {
  cells: GPUBuffer;
  dispatch: GPUBuffer;
  preparePipeline: GPUComputePipeline;
  prepareGroup: GPUBindGroup;
  reclaimPipeline: GPUComputePipeline;
  reclaimGroup: GPUBindGroup;
}

/** The hash grid: lookup over query records, update from training records, resolve. Everything encodes into the caller's encoder. */
export class HashGrid {
  readonly settings: HashGridSettings;
  readonly entries: number;
  private readonly device: GPUDevice;
  private readonly params: GPUBuffer;
  private readonly snapshot: GPUBuffer;
  private readonly accum: GPUBuffer;
  private readonly momentSums?: GPUBuffer;
  private readonly age: GPUBuffer;
  readonly stats: GPUBuffer;
  private readonly lookupPipeline: GPUComputePipeline;
  private readonly updatePipeline: GPUComputePipeline;
  private readonly resolvePipeline: GPUComputePipeline;
  private readonly tickPipeline: GPUComputePipeline;
  private readonly resolveGroup: GPUBindGroup;
  private readonly tickGroup: GPUBindGroup;
  private readonly active?: ActiveMaintenance;
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
    this.snapshot = init.snapshot as GPUBuffer;
    this.accum = init.accum as GPUBuffer;
    this.momentSums = init.momentSums as GPUBuffer | undefined;
    this.age = init.age as GPUBuffer;
    this.stats = init.stats as GPUBuffer;
    this.lookupPipeline = init.lookupPipeline as GPUComputePipeline;
    this.updatePipeline = init.updatePipeline as GPUComputePipeline;
    this.resolvePipeline = init.resolvePipeline as GPUComputePipeline;
    this.tickPipeline = init.tickPipeline as GPUComputePipeline;
    this.resolveGroup = init.resolveGroup as GPUBindGroup;
    this.tickGroup = init.tickGroup as GPUBindGroup;
    this.active = init.active as ActiveMaintenance | undefined;
    this.pixelAngle = init.pixelAngle as number;
  }

  /** `yFovDeg` and `filmHeight` set the pixel footprint the LOD rule uses. */
  static async create(device: GPUDevice, settings: Partial<HashGridSettings>, view: { yFovDeg: number; filmHeight: number }): Promise<HashGrid> {
    const s: HashGridSettings = { ...DEFAULT_HASH_GRID, ...settings };
    const storage = hashGridStorage(s);
    const { entries } = storage;
    const { STORAGE, COPY_DST, COPY_SRC, UNIFORM, INDIRECT } = GPUBufferUsage;
    const buf = (label: string, size: number, usage = STORAGE | COPY_DST | COPY_SRC) => device.createBuffer({ label, size, usage });
    const params = buf('hash grid params', 64, UNIFORM | COPY_DST);
    const variance = s.varianceThreshold >= 0;
    const snapshot = buf('hash grid keys + resolved cells', storage.snapshotBytes);
    const momentSums = variance ? buf('hash grid moment accumulators', storage.momentBytes) : undefined;
    const accum = buf('hash grid accumulators', storage.accumBytes);
    const age = buf('hash grid age and fit history', storage.ageBytes);
    const stats = buf('hash grid stats', storage.statsBytes);
    const code = hashGridWgsl(s.fallbackLevels, s.pyramid, s.activeMaintenance, variance, s.directionalSh, s.memoryBudgetBytes !== undefined);
    const make = (label: string, code: string, entryPoint: string) => device.createComputePipelineAsync({ label, layout: 'auto', compute: { module: device.createShaderModule({ label, code }), entryPoint } });
    const [lookupPipeline, updatePipeline, resolvePipeline, tickPipeline] = await Promise.all([
      make('hash grid lookup', code.lookup, 'hgLookup'),
      make('hash grid update', code.update, 'hgUpdate'),
      make('hash grid resolve', code.resolve, 'hgResolve'),
      make('hash grid tick', code.resolve, 'hgTick'),
    ]);
    const keyResource = { buffer: snapshot, size: entries * 4 };
    const valueResource = { buffer: snapshot, offset: entries * 4, size: storage.valueBytes };
    const bind = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[]) =>
      device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    let active: ActiveMaintenance | undefined;
    const resolveBindings: GPUBindGroupEntry[] = [
      { buffer: params }, keyResource, { buffer: accum }, valueResource, { buffer: age }, { buffer: stats },
    ].map((resource, binding) => ({ binding, resource }));
    if (momentSums) {
      resolveBindings.push(
        { binding: 8, resource: { buffer: momentSums } },
        { binding: 9, resource: { buffer: snapshot, offset: entries * 4 + storage.valueBytes } },
      );
    }
    if (s.activeMaintenance) {
      const cells = buf('hash grid touched slots and age bounds', (1 + entries + 2 * entries / AGE_BLOCK) * 4);
      const dispatch = buf('hash grid touched resolve dispatch', 12, STORAGE | INDIRECT);
      const [preparePipeline, reclaimPipeline] = await Promise.all([
        make('hash grid prepare touched resolve', code.resolve, 'hgPrepare'),
        make('hash grid reclaim stale cells', code.resolve, 'hgReclaim'),
      ]);
      const cellsBinding = { binding: 6, resource: { buffer: cells } };
      const prepareGroup = bind(preparePipeline, [cellsBinding, { binding: 7, resource: { buffer: dispatch } }]);
      const reclaimGroup = bind(reclaimPipeline, [
        ...resolveBindings.filter(({ binding }) => binding !== 2 && binding !== 8), cellsBinding,
      ]);
      resolveBindings.push(cellsBinding);
      active = { cells, dispatch, preparePipeline, prepareGroup, reclaimPipeline, reclaimGroup };
    }
    const resolveGroup = bind(resolvePipeline, resolveBindings);
    const tickGroup = device.createBindGroup({ layout: tickPipeline.getBindGroupLayout(0), entries: [{ binding: 5, resource: { buffer: stats } }] });
    const grid = new HashGrid({
      settings: s, entries, device, params, snapshot, accum, momentSums, age, stats, lookupPipeline, updatePipeline, resolvePipeline, tickPipeline, resolveGroup, tickGroup, active,
      pixelAngle: (2 * Math.tan((view.yFovDeg * Math.PI) / 360)) / view.filmHeight,
    });
    grid.clear();
    return grid;
  }

  /** Optional group 3 extension for an inline display decision. */
  get displayHook() {
    const buffers = [this.params, this.snapshot];
    return {
      wgsl: lookupFunctions(Math.max(0, Math.floor(this.settings.fallbackLevels)), 3, 4, !!this.momentSums,
        this.settings.directionalSh, this.settings.memoryBudgetBytes !== undefined),
      bindings: buffers.map((_, i): GPUBindGroupLayoutEntry => ({ binding: 4 + i, visibility: GPUShaderStage.COMPUTE, buffer: { type: i === 0 ? 'uniform' : 'read-only-storage' } })),
      entries: buffers.map((buffer, i) => ({ binding: 4 + i, resource: { buffer } })),
    };
  }

  /** Binds the query records (+ their count, word 0) and the prediction buffer the lookup writes. */
  bindQueries(queries: GPUBuffer, count: GPUBuffer, preds: GPUBuffer, capacity: number): void {
    this.queryCap = capacity;
    this.lookupGroup = this.device.createBindGroup({ layout: this.lookupPipeline.getBindGroupLayout(0), entries: [this.params, this.snapshot, queries, preds, count, this.stats].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.writeParams();
  }

  /** Binds the training records (+ their count, word 1) the update adds into the cells. */
  bindRecords(records: GPUBuffer, count: GPUBuffer, capacity: number): void {
    this.recordCap = capacity;
    const buffers = [this.params, this.snapshot, this.accum, records, count, this.stats];
    if (this.active) buffers.push(this.active.cells);
    const entries = buffers.map((buffer, binding) => ({ binding, resource: binding === 1 ? { buffer, size: this.entries * 4 } : { buffer } }));
    if (this.momentSums) entries.push({ binding: 8, resource: { buffer: this.momentSums } });
    this.updateGroup = this.device.createBindGroup({ layout: this.updatePipeline.getBindGroupLayout(0), entries });
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
    f[14] = s.varianceThreshold;
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

  /** Adds the bound training records into their cells, then resolves, evicts stale cells, and advances the frame counter. */
  encodeUpdate(encoder: GPUCommandEncoder, timing?: {
    update?: GPUComputePassTimestampWrites;
    resolve?: GPUComputePassTimestampWrites;
  }): void {
    if (!this.updateGroup) throw new Error('HashGrid: bindRecords first');
    const active = this.active;
    if (active) encoder.clearBuffer(active.cells, 0, 4);
    const pass = encoder.beginComputePass({
      label: 'hash grid update',
      ...(timing?.update ? { timestampWrites: timing.update } : {}),
    });
    pass.setPipeline(this.updatePipeline);
    pass.setBindGroup(0, this.updateGroup);
    pass.dispatchWorkgroups(Math.ceil(this.recordCap / 64));
    pass.end();
    const resolve = encoder.beginComputePass({
      label: 'hash grid resolve',
      ...(timing?.resolve ? { timestampWrites: timing.resolve } : {}),
    });
    if (active) {
      resolve.setPipeline(active.preparePipeline);
      resolve.setBindGroup(0, active.prepareGroup);
      resolve.dispatchWorkgroups(1);
    }
    resolve.setPipeline(this.resolvePipeline);
    resolve.setBindGroup(0, this.resolveGroup);
    if (active) {
      resolve.dispatchWorkgroupsIndirect(active.dispatch, 0);
      resolve.setPipeline(active.reclaimPipeline);
      resolve.setBindGroup(0, active.reclaimGroup);
      resolve.dispatchWorkgroups(Math.ceil(this.entries / (AGE_BLOCK * 64)));
    } else {
      resolve.dispatchWorkgroups(this.entries / RESOLVE_GROUP_SIZE);
    }
    resolve.setPipeline(this.tickPipeline);
    resolve.setBindGroup(0, this.tickGroup);
    resolve.dispatchWorkgroups(1);
    resolve.end();
  }

  /** Empties the table (submitted now; queue order keeps it before any later frame). */
  clear(): void {
    const e = this.device.createCommandEncoder({ label: 'hash grid clear' });
    for (const b of [this.snapshot, this.accum, this.age, this.stats]) e.clearBuffer(b);
    if (this.active) e.clearBuffer(this.active.cells);
    if (this.momentSums) e.clearBuffer(this.momentSums);
    this.device.queue.submit([e.finish()]);
  }

  /** GPU allocation for grid storage, including optional maintenance buffers. */
  get bytes(): number {
    return this.params.size + this.snapshot.size + this.accum.size + this.age.size + this.stats.size
      + (this.active ? this.active.cells.size + this.active.dispatch.size : 0) + (this.momentSums?.size ?? 0);
  }

  /** Debug readback of the number of cells with a usable directional fit. */
  async readDirectionalCells(): Promise<number> {
    if (!this.settings.directionalSh) return 0;
    const size = this.entries * 16;
    const staging = this.device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const e = this.device.createCommandEncoder();
      e.copyBufferToBuffer(this.snapshot, this.entries * 4, staging, 0, size);
      this.device.queue.submit([e.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const v = new Float32Array(staging.getMappedRange());
      let fitted = 0;
      for (let i = 3; i < v.length; i += 4) if (v[i]! < 0) fitted++;
      return fitted;
    } finally {
      staging.destroy();
    }
  }

  /** Debug readback of the checksum and last sample frame for every occupied slot. */
  async readCellAges(): Promise<{ checksum: number; frame: number }[]> {
    const bytes = this.entries * 4;
    const staging = this.device.createBuffer({ size: 2 * bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = this.device.createCommandEncoder();
    e.copyBufferToBuffer(this.snapshot, 0, staging, 0, bytes);
    e.copyBufferToBuffer(this.age, 0, staging, bytes, bytes);
    this.device.queue.submit([e.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(staging.getMappedRange());
    const out: { checksum: number; frame: number }[] = [];
    for (let i = 0; i < this.entries; i++) {
      if (v[i]!) out.push({ checksum: v[i]!, frame: v[this.entries + i]! });
    }
    staging.unmap();
    staging.destroy();
    return out.sort((a, b) => a.checksum - b.checksum || a.frame - b.frame);
  }

  /** Debug only (never in a frame loop). */
  async readStats(): Promise<HashGridStats> {
    const statBytes = this.stats.size;
    const keyEnd = statBytes + this.entries * 4;
    const staging = this.device.createBuffer({ size: keyEnd + (this.active ? 4 : 0), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = this.device.createCommandEncoder();
    e.copyBufferToBuffer(this.stats, 0, staging, 0, statBytes);
    e.copyBufferToBuffer(this.snapshot, 0, staging, statBytes, this.entries * 4);
    if (this.active) e.copyBufferToBuffer(this.active.cells, 0, staging, keyEnd, 4);
    this.device.queue.submit([e.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(staging.getMappedRange());
    let cells = 0;
    for (let i = statBytes / 4; i < keyEnd / 4; i++) if (v[i] !== 0) cells++;
    const out: HashGridStats = { frames: v[0] as number, misses: v[1] as number, coarse: v[2] as number, dropped: v[3] as number, continued: v[4] as number, cells };
    if (this.momentSums) out.varianceRejects = v[5]!;
    if (this.active) out.touchedFraction = (v[keyEnd / 4] as number) / this.entries;
    staging.unmap();
    staging.destroy();
    return out;
  }

  /** Zeroes the miss / coarse / dropped counters (keeps the frame counter). */
  resetStats(): void {
    this.device.queue.writeBuffer(this.stats, 4, new Uint32Array(this.stats.size / 4 - 1));
  }

  destroy(): void {
    for (const b of [this.params, this.snapshot, this.accum, this.age, this.stats]) b.destroy();
    this.momentSums?.destroy();
    this.active?.cells.destroy();
    this.active?.dispatch.destroy();
  }
}

/** Hash grid settings from page args. Absent keys keep the defaults. */
export function hashGridArgs(args: Record<string, string | undefined>): Partial<HashGridSettings> {
  const out: Partial<HashGridSettings> = {};
  const num = (key: string, field: keyof HashGridSettings) => {
    const v = args[key];
    if (v !== undefined) (out as Record<string, number>)[field] = Number(v);
  };
  num('hgLog2', 'log2Entries');
  num('hgBudgetBytes', 'memoryBudgetBytes');
  num('hgMinCell', 'minCell');
  num('hgCellPx', 'cellPixels');
  num('hgHistory', 'historyFrames');
  num('hgStale', 'staleFrames');
  num('hgFallback', 'fallbackLevels');
  num('hgMinSamples', 'minSamples');
  num('hgJitter', 'jitter');
  num('hgDirBins', 'dirBins');
  num('hgVariance', 'varianceThreshold');
  if (args.hgActive !== undefined) out.activeMaintenance = args.hgActive === '1';
  if (args.hgContinue !== undefined) out.continueOnMiss = args.hgContinue === '1';
  if (args.hgDirectionalSh !== undefined) out.directionalSh = args.hgDirectionalSh === '1';
  if (args.hgNormals !== undefined) out.normalBins = args.hgNormals !== '0';
  if (args.hgPyramid !== undefined) out.pyramid = args.hgPyramid !== '0';
  return out;
}
