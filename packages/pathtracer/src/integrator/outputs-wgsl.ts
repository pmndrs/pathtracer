/**
 * Optional integrator outputs, each OFF by default and compiled only when asked (a kernel without them is byte-identical):
 *
 *  - `hooks` (R10, spec section 8, D3): user WGSL `ptOnVertex` / `ptTerminate` called at every scattering vertex by both schedulers.
 *  - `counters` (R4): per-pixel debug counters in one storage buffer (depth, rays, termination reason). Never in a timed run.
 *  - `aovs` (P2 exit item): first non-specular albedo + normal and the primary viewZ, accumulated in the tail of the accumulation buffer.
 *
 * This file only generates WGSL text and holds the TypeScript twins of the layouts; the schedulers call it (`integrator-wgsl.ts`,
 * `../wavefront/wavefront-wgsl.ts`). All WGSL here is order independent (module scope), so it is appended after the shading modules and refers to
 * `ptAccum` / `ptCounters` that the scheduler declares.
 */

/** Interface of the L3 path hooks, typed by this struct, not by `@pmndrs/gi`'s types. */
export interface PtHooksOptions {
  /**
   * WGSL declaring `fn ptOnVertex(v: PtPathVertex) -> u32` (return `PT_HOOK_CONTINUE` or `PT_HOOK_TERMINATE`) and/or
   * `fn ptTerminate(v: PtPathVertex) -> vec3f` (the radiance to add when the path terminates; the integrator multiplies it by `v.throughput`).
   * A missing function gets a default (continue / black). The text may declare its own bindings in `@group(3)` (see `bindings`),
   * structs and helpers; name them with your own prefix (`gi*`), not `pt*` / `rt*` / `bvh*`.
   */
  wgsl: string;
  /**
   * Layout entries of the hook's own bind group (`@group(3)`; the wavefront binds it to the shade kernel only), e.g. a hash table. Default none.
   * Mind the 10 storage buffers per stage of Apple GPUs. Measured (`pt-outputs-check check=budget`, M5 Pro): the megakernel leaves 4 for the hook
   * (walk 3, scene 2, accumulator = 6 + fuzz table + counters), 3 with `counters`; the wavefront's shade kernel leaves 2 (scene 2, accumulator, state,
   * hit, count, queue, shadow rays = 8), 1 with `counters`; the fuzz table takes one more. `create` fails on pipeline creation with the validation
   * message when the sum is over the device limit.
   */
  bindings?: GPUBindGroupLayoutEntry[];
  /**
   * Training ("update") paths: with `updateBlock: B`, one pixel per BxB block is an update path per sample (the pixel rotates with the sample index:
   * `(x % B) + (y % B) * B == sample % B^2`) and its `PtPathVertex.flags` has `PT_VERTEX_UPDATE` set. The integrator does nothing else with the
   * flag: the hook decides (typically: an update path never returns `PT_HOOK_TERMINATE`, so it carries on to the end, and writes its vertices
   * to the cache's update buffer). 0 (default) = no update paths.
   */
  updateBlock?: number;
}

/** Hook return values and `PtPathVertex.flags` bits (the WGSL constants of the same names). */
export const PT_HOOK = { continue: 0, terminate: 1 } as const;
export const PT_VERTEX_UPDATE = 1;

/** Words of `ptCounters` per pixel and what they hold (the WGSL `PT_COUNTER_*` constants). Counts accumulate over samples. */
export const PT_COUNTER_WORDS = 12;
export const PT_COUNTER = {
  /** Scattering events summed over samples (mean depth = depth / samples). */
  depth: 0,
  /** Extension rays traced (closest hit), including alpha-mask pass-throughs. */
  extensionRays: 1,
  /** Shadow rays traced (next-event estimation). */
  shadowRays: 2,
  /** Reserved for bvh node visits (bvh's walk has no counter hook yet: see TASKS.md R4). Always 0. */
  nodeVisits: 3,
  /** First of the termination reasons (`PT_TERM`): word `reasons + reason`. Counts of paths per reason. */
  reasons: 4,
} as const;
/** Why a path ended. Words `PT_COUNTER.reasons + PT_TERM.x`; they sum to the sample count. */
export const PT_TERM = { escaped: 0, absorbed: 1, maxDepth: 2, rr: 3, hook: 4, invalid: 5, maskOverflow: 6 } as const;
export const PT_TERM_NAMES = ['escaped', 'absorbed', 'maxDepth', 'rr', 'hook', 'invalid', 'maskOverflow'] as const;

/** Vec4 slots of the AOVs per pixel, in the tail of the accumulation buffer (after the `width * height` radiance sums). */
export const PT_AOV_VEC4 = 2;

export interface PtOutputsFlags {
  hooks: PtHooksOptions | null;
  counters: boolean;
  aovs: boolean;
}

/** `ptHookUpdateFlag` in TypeScript: is the path of pixel (x, y) in sample `sample` an update path? */
export function ptHookIsUpdate(x: number, y: number, sample: number, block: number): boolean {
  if (!(block > 0)) return false;
  return (x % block) + (y % block) * block === sample % (block * block);
}

const HAS_ON_VERTEX = /\bfn\s+ptOnVertex\b/;
const HAS_TERMINATE = /\bfn\s+ptTerminate\b/;

/** The hook interface: struct, constants, the user's text and defaults for the functions it leaves out. Empty without hooks. */
export function ptHooksWgsl(hooks: PtHooksOptions | null): string {
  if (!hooks) return '';
  const block = Math.max(0, Math.floor(hooks.updateBlock ?? 0));
  return /* wgsl */ `
const PT_HOOK_CONTINUE: u32 = 0u;
const PT_HOOK_TERMINATE: u32 = 1u;
const PT_VERTEX_UPDATE: u32 = 1u;
const PT_HOOK_UPDATE_BLOCK: u32 = ${block}u;

// What a hook sees at a scattering vertex, after emission and next-event estimation, before the BSDF sample.
struct PtPathVertex {
    position: vec3f,
    depth: u32,
    normal: vec3f,
    materialClass: u32,
    wo: vec3f,
    pixel: u32,
    throughput: vec3f,
    flags: u32,
    emitted: vec3f,
    sampleIndex: u32,
    albedo: vec3f,
    material: u32,
}

fn ptHookUpdateFlag(x: u32, y: u32, sampleIndex: u32) -> u32 {
    if (PT_HOOK_UPDATE_BLOCK == 0u) {
        return 0u;
    }
    let b = PT_HOOK_UPDATE_BLOCK;
    return select(0u, PT_VERTEX_UPDATE, (x % b) + (y % b) * b == sampleIndex % (b * b));
}

${hooks.wgsl.trim()}
${HAS_ON_VERTEX.test(hooks.wgsl) ? '' : `
fn ptOnVertex(v: PtPathVertex) -> u32 {
    return PT_HOOK_CONTINUE;
}
`}${HAS_TERMINATE.test(hooks.wgsl) ? '' : `
fn ptTerminate(v: PtPathVertex) -> vec3f {
    return vec3f(0.0);
}
`}`;
}

/**
 * The call both schedulers make (the expression builds the vertex from the locals of their loop: `surf`, `hit`, `ray`, `bp`, `mat`, `beta`,
 * `vertex`, `hookEmitted`, pixel x / y, sample index and pixel index).
 */
export function ptHookVertexExpr(args: { x: string; y: string; pixel: string; sampleIndex: string }): string {
  return `PtPathVertex(surf.position, vertex, ptSideNormal(hit.normal, -ray.direction), bp.kind, -ray.direction, ${args.pixel}, beta, ptHookUpdateFlag(${args.x}, ${args.y}, ${args.sampleIndex}), hookEmitted, ${args.sampleIndex}, ptAovAlbedo(mat), hit.material)`;
}

/** Albedo estimate used by the AOV and the hook vertex (`ptAovAlbedo`) and the three writers. Needs `ptAccum`. */
export function ptAovWgsl(flags: { aovs: boolean; hooks: boolean }): string {
  if (!flags.aovs && !flags.hooks) return '';
  const writers = flags.aovs
    ? `
fn ptAovPrimary(pixel: u32, n: u32, viewZ: f32) {
    let i = n + 2u * pixel;
    ptAccum[i] = ptAccum[i] + vec4f(0.0, 0.0, 0.0, viewZ);
    ptAccum[i + 1u] = ptAccum[i + 1u] + vec4f(0.0, 0.0, 0.0, 1.0);
}

fn ptAovGuide(pixel: u32, n: u32, albedo: vec3f, normal: vec3f) {
    let i = n + 2u * pixel;
    ptAccum[i] = ptAccum[i] + vec4f(albedo, 0.0);
    ptAccum[i + 1u] = ptAccum[i + 1u] + vec4f(normal, 0.0);
}
`
    : '';
  return /* wgsl */ `
// Demodulation albedo of the surface a vertex sits on: base colour (a conductor's is its reflectance at normal incidence in the
// legacy adapter and the base colour in OpenPBR), white for a rough dielectric, which has no body colour.
fn ptAovAlbedo(m: RtMaterial) -> vec3f {
    if (m.transmissionWeight > 0.5) {
        return vec3f(1.0);
    }
    return m.baseColor * m.baseWeight;
}
${writers}`;
}

/** Debug counters (`ptCounters`, `PT_COUNTER_WORDS` u32 per pixel). `binding` is the scheduler's slot in group 2. */
export function ptCountersWgsl(binding: number): string {
  return /* wgsl */ `
@group(2) @binding(${binding}) var<storage, read_write> ptCounters: array<u32>;
const PT_COUNTER_WORDS: u32 = ${PT_COUNTER_WORDS}u;
const PT_COUNTER_REASONS: u32 = ${PT_COUNTER.reasons}u;

fn ptCountRay(pixel: u32, extension: u32, shadow: u32) {
    let i = pixel * PT_COUNTER_WORDS;
    ptCounters[i + ${PT_COUNTER.extensionRays}u] += extension;
    ptCounters[i + ${PT_COUNTER.shadowRays}u] += shadow;
}

fn ptCountEnd(pixel: u32, depth: u32, reason: u32) {
    let i = pixel * PT_COUNTER_WORDS;
    ptCounters[i + ${PT_COUNTER.depth}u] += depth;
    ptCounters[i + PT_COUNTER_REASONS + reason] += 1u;
}
`;
}

/** Everything the enabled outputs add after the shading modules, or '' (then nothing at all is appended). */
export function ptOutputsWgsl(flags: PtOutputsFlags, countersBinding: number | null): string {
  const parts = [ptAovWgsl({ aovs: flags.aovs, hooks: !!flags.hooks }), ptHooksWgsl(flags.hooks), flags.counters && countersBinding !== null ? ptCountersWgsl(countersBinding) : ''].filter((p) => p !== '');
  return parts.join('\n');
}

/** Reader of the accumulated AOV tail: averages over samples. `data` = the whole accumulation buffer (radiance sums then AOV vec4s). */
export interface PtAovs {
  width: number;
  height: number;
  samples: number;
  /** RGB per pixel (row 0 = top): the albedo of the first non-specular vertex, averaged over samples (a miss counts 0). */
  albedo: Float32Array;
  /** Unit world-space normal of that vertex per pixel, xyz (the average over samples, renormalised; 0 where no sample hit). */
  normal: Float32Array;
  /** viewZ of the first hit, negative in front (averaged over the samples that hit; 0 where none did). */
  viewZ: Float32Array;
  /** Fraction of samples whose primary ray hit a surface. */
  coverage: Float32Array;
}

export function ptDecodeAovs(sums: Float32Array, width: number, height: number, samples: number): PtAovs {
  const n = width * height;
  const albedo = new Float32Array(n * 3);
  const normal = new Float32Array(n * 3);
  const viewZ = new Float32Array(n);
  const coverage = new Float32Array(n);
  const s = Math.max(1, samples);
  for (let p = 0; p < n; p++) {
    const a = n * 4 + p * 8;
    const hits = sums[a + 7] as number;
    for (let c = 0; c < 3; c++) albedo[p * 3 + c] = (sums[a + c] as number) / s;
    const nx = sums[a + 4] as number;
    const ny = sums[a + 5] as number;
    const nz = sums[a + 6] as number;
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) {
      normal[p * 3] = nx / len;
      normal[p * 3 + 1] = ny / len;
      normal[p * 3 + 2] = nz / len;
    }
    viewZ[p] = hits > 0 ? (sums[a + 3] as number) / hits : 0;
    coverage[p] = hits / s;
  }
  return { width, height, samples, albedo, normal, viewZ, coverage };
}

/** Counters read back from `ptCounters` plus per-sample averages. */
export interface PtCounters {
  width: number;
  height: number;
  samples: number;
  /** `PT_COUNTER_WORDS` u32 per pixel, raw sums over samples. */
  raw: Uint32Array;
  /** Mean over pixels and samples. */
  summary: PtCounterSummary;
}

export interface PtCounterSummary {
  depth: number;
  extensionRays: number;
  shadowRays: number;
  nodeVisits: number;
  /** Fraction of paths per termination reason (`PT_TERM_NAMES`); sums to 1. */
  reasons: Record<(typeof PT_TERM_NAMES)[number], number>;
}

export function ptSummarizeCounters(raw: Uint32Array, width: number, height: number, samples: number): PtCounterSummary {
  const n = width * height;
  const total = Math.max(1, n * samples);
  const sum = new Float64Array(PT_COUNTER_WORDS);
  for (let p = 0; p < n; p++) for (let w = 0; w < PT_COUNTER_WORDS; w++) sum[w]! += raw[p * PT_COUNTER_WORDS + w] as number;
  const reasons = Object.fromEntries(PT_TERM_NAMES.map((name, i) => [name, (sum[PT_COUNTER.reasons + i] as number) / total])) as PtCounterSummary['reasons'];
  return { depth: (sum[PT_COUNTER.depth] as number) / total, extensionRays: (sum[PT_COUNTER.extensionRays] as number) / total, shadowRays: (sum[PT_COUNTER.shadowRays] as number) / total, nodeVisits: (sum[PT_COUNTER.nodeVisits] as number) / total, reasons };
}

/** One float per pixel of word `word` divided by `samples` (a heatmap / PFM image of a counter). */
export function ptCounterImage(raw: Uint32Array, width: number, height: number, samples: number, word: number): Float32Array {
  const n = width * height;
  const out = new Float32Array(n);
  const s = Math.max(1, samples);
  for (let p = 0; p < n; p++) out[p] = (raw[p * PT_COUNTER_WORDS + word] as number) / s;
  return out;
}
