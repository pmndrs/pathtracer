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
   * Mind the 10 storage buffers per stage of Apple GPUs. Measured (`pt-outputs-check check=budget`, M5 Pro): with the scene copy `split` (two buffers,
   * the default) the megakernel leaves 4 for the hook (walk 3, scene 2, accumulator = 6 + fuzz table + counters), 3 with `counters`; the wavefront's
   * shade kernel leaves 2 (scene 2, accumulator, state, hit, count, queue, shadow rays = 8), 1 with `counters`; the fuzz table takes one more.
   * With `scenePacking: 'packed'` (P3.11, the scene copy as ONE buffer) every number is one higher: megakernel 5 (4 with `counters`), wavefront shade
   * 3 (2 with `counters`). `create` fails on pipeline creation with the validation message when the sum is over the device limit.
   */
  bindings?: GPUBindGroupLayoutEntry[];
  /**
   * Training ("update") paths: with `updateBlock: B`, one pixel per BxB block is an update path per sample (the pixel rotates with the sample index:
   * `(x % B) + (y % B) * B == sample % B^2`) and its `PtPathVertex.flags` has `PT_VERTEX_UPDATE` set. The integrator does nothing else with the
   * flag: the hook decides (typically: an update path never returns `PT_HOOK_TERMINATE`, so it carries on to the end, and writes its vertices
   * to the cache's update buffer). 0 (default) = no update paths.
   */
  updateBlock?: number;
  /**
   * Wavefront only, with `ptOnRadiance` (`ptHookWantsRadiance`): the `bindings` (by binding number) the radiance callback uses. The wavefront adds
   * next-event light in its shadow kernel, after the visibility test, and calls `ptOnRadiance` there, so that kernel binds these as its own `@group(3)`
   * (`PtWavefront.setHookRadianceBindGroup`). It has room for 3 storage buffers on Apple (walk 3 + io 4 = 7 of 10; 1 with an exact alpha-mask shadow
   * policy, which adds the scene copy). Default: all of `bindings`. The hook text then also compiles in the shadow kernel (the hook interface is there;
   * functions the callback does not reach are pruned).
   */
  radianceBindings?: number[];
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

/** Group 2 bindings of the per-frame outputs (`integrator.frame`): the previous camera (uniform, dynamic offset, one slot per pass like `ptParams`) and the planes. */
export const PT_FRAME_PARAMS_BINDING = 13;
export const PT_FRAME_BINDING = 14;
/** Planes of `ptFrame` (each `width * height` vec4f, plane-major): see `PtFrame`. The beauty (total radiance) is the accumulator itself. */
export const PT_FRAME_PLANES = 6;
/** Bytes of one `PtFrameParams` record (the previous camera: origin, right, up, forward as vec4f). */
export const PT_FRAME_PARAMS_BYTES = 64;
/** Demodulation factors are floored here (per channel) so the division is finite; the stored factor is the floored one, so demodulated x factor reproduces the radiance exactly. */
export const PT_DEMOD_EPS = 0.001;

export interface PtOutputsFlags {
  hooks: PtHooksOptions | null;
  counters: boolean;
  aovs: boolean;
  frame?: boolean;
}

/** `ptHookUpdateFlag` in TypeScript: is the path of pixel (x, y) in sample `sample` an update path? */
export function ptHookIsUpdate(x: number, y: number, sample: number, block: number): boolean {
  if (!(block > 0)) return false;
  return (x % block) + (y % block) * block === sample % (block * block);
}

const HAS_ON_VERTEX = /\bfn\s+ptOnVertex\b/;
const HAS_ON_RADIANCE = /\bfn\s+ptOnRadiance\b/;

/**
 * Does the hook text define the optional radiance callback `fn ptOnRadiance(pixel: u32, sampleIndex: u32, index: u32, flags: u32, value: vec3f)`?
 * Only then do the integrators emit its call sites (every other kernel, hooked or not, keeps its exact text). It is called wherever the path adds
 * radiance, with the added value (throughput included, exactly what goes into the pixel) and the index of the vertex whose outgoing light it is
 * part of: emission and next-event estimation at vertex j report j; light arriving along the ray that leaves vertex j (the environment on an
 * escape, `ptTerminate`'s cache value) reports j + 1. So the radiance a path gathers after vertex k's own emission and NEE is the sum of the
 * values with index > k: what a cache's training targets need (track N, V84). `flags` is `PtPathVertex.flags` of the path (`PT_VERTEX_UPDATE`).
 */
export function ptHookWantsRadiance(hooks: PtHooksOptions | null | undefined): boolean {
  return !!hooks && HAS_ON_RADIANCE.test(hooks.wgsl);
}
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
export function ptAovWgsl(flags: { aovs: boolean; hooks: boolean; frame?: boolean }): string {
  if (!flags.aovs && !flags.hooks && !flags.frame) return '';
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
${flags.frame ? ptAovSpecularWgsl() : ''}${writers}`;
}

function ptAovSpecularWgsl(): string {
  return /* wgsl */ `
// Specular demodulation factor of a primary hit: a rough stand-in for the pre-integrated specular term (F0 of the surface: 0.04 for a dielectric, the base colour for a metal, 1 for glass). The engine may supply its own DFG term instead.
fn ptAovSpecular(m: RtMaterial) -> vec3f {
    if (m.transmissionWeight > 0.5) {
        return vec3f(1.0);
    }
    return mix(vec3f(0.04), m.baseColor * m.baseWeight, m.metalness);
}
`;
}

/** The per-frame outputs (P4.1): previous-camera uniform, the planes, the reprojection and the writer \`ptPath\` calls once at its end. Needs \`ptParams\`, \`ptAovAlbedo\` (\`ptAovWgsl\` with \`frame\`). */
export function ptFrameWgsl(): string {
  return /* wgsl */ `
struct PtFrameParams {
    origin: vec4f,
    right: vec4f,
    up: vec4f,
    forward: vec4f,
}
@group(2) @binding(${PT_FRAME_PARAMS_BINDING}) var<uniform> ptFrameParams: PtFrameParams;
@group(2) @binding(${PT_FRAME_BINDING}) var<storage, read_write> ptFrame: array<vec4f>;
const PT_DEMOD_EPS: f32 = ${PT_DEMOD_EPS};

// (ndc.x, ndc.y, viewZ) of world point p seen from a pinhole camera given as the kernel's basis (right and up are scaled by tan(fov/2) (x aspect)).
fn ptProject(origin: vec3f, right: vec3f, up: vec3f, forward: vec3f, p: vec3f) -> vec3f {
    let d = p - origin;
    let z = dot(d, forward);
    return vec3f(dot(d, right) / (dot(right, right) * z), dot(d, up) / (dot(up, up) * z), -z);
}

// Motion of a world point: NDC delta current - previous (+y up, no jitter: both projections are the unjittered cameras) and zPrev - zCur in view space (viewZ negative in front).
fn ptFrameMotion(p: vec3f) -> vec3f {
    let c = ptProject(ptParams.origin.xyz, ptParams.right.xyz, ptParams.up.xyz, ptParams.forward.xyz, p);
    let q = ptProject(ptFrameParams.origin.xyz, ptFrameParams.right.xyz, ptFrameParams.up.xyz, ptFrameParams.forward.xyz, p);
    return vec3f(c.xy - q.xy, q.z - c.z);
}

fn ptFramePlanes(pixel: u32, n: u32, radiance: vec3f, direct: vec3f, bounceT: f32, seen: bool, isSpec: bool, normal: vec3f, viewZ: f32, motion: vec3f, rough: f32, albedo: vec3f, spec: vec3f) {
    let aF = select(vec3f(1.0), max(albedo, vec3f(PT_DEMOD_EPS)), seen);
    let sF = select(vec3f(1.0), max(spec, vec3f(PT_DEMOD_EPS)), seen);
    let useSpec = seen && isSpec;
    let diffuse = select(radiance, direct, useSpec);
    let specular = select(vec3f(0.0), radiance - direct, useSpec);
    let hd = select(0.0, bounceT, seen);
    ptFrame[pixel] = vec4f(diffuse / aF, select(hd, 0.0, useSpec));
    ptFrame[n + pixel] = vec4f(specular / sF, select(0.0, hd, useSpec));
    ptFrame[2u * n + pixel] = vec4f(normal, viewZ);
    ptFrame[3u * n + pixel] = vec4f(motion.xy, rough, motion.z);
    ptFrame[4u * n + pixel] = vec4f(aF, select(0.0, 1.0, useSpec));
    ptFrame[5u * n + pixel] = vec4f(sF, select(0.0, 1.0, seen));
}
`;
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
  const parts = [ptAovWgsl({ aovs: flags.aovs, hooks: !!flags.hooks, frame: !!flags.frame }), flags.frame ? ptFrameWgsl() : '', ptHooksWgsl(flags.hooks), flags.counters && countersBinding !== null ? ptCountersWgsl(countersBinding) : ''].filter((p) => p !== '');
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

/** One frame's outputs read back (`PtMegakernel.readFrame`); every plane is `width * height` vec4 (4 floats per pixel, row 0 = top). */
export interface PtFrame {
  width: number;
  height: number;
  /** Beauty: total radiance of the sample, rgb; a = 1. */
  radiance: Float32Array;
  /** rgb: diffuse-class radiance / `albedo` factor (primary emission and sky included); a: hit distance of the first bounce ray (0 = miss, or a specular-class path). */
  diffuse: Float32Array;
  /** rgb: specular-class radiance / `specularFactor`; a: hit distance of the first bounce ray (0 = miss, or a diffuse-class path). */
  specular: Float32Array;
  /** rgb: world-space unit shading normal of the primary hit (0 on a miss); a: viewZ (negative in front, 0 on a miss). */
  normalViewZ: Float32Array;
  /** r, g: motion, NDC delta current - previous (+y up, jitter-free); b: perceptual roughness; a: motionZ = viewZ(previous) - viewZ(current). */
  motionRoughness: Float32Array;
  /** rgb: the diffuse demodulation factor (albedo of the primary hit, floored at `PT_DEMOD_EPS`; 1 on a miss); a: 1 when the primary hit is specular-class (conductor / dielectric lobe). */
  albedo: Float32Array;
  /** rgb: the specular demodulation factor (approximate pre-integrated specular term, floored; 1 on a miss); a: 1 when the primary ray hit a surface. */
  specularFactor: Float32Array;
}

/** Splits the `ptFrame` planes (and the accumulator) into a `PtFrame`. `planes` = `PT_FRAME_PLANES * width * height * 4` floats, `radiance` = `width * height * 4`. */
export function ptDecodeFrame(radiance: Float32Array, planes: Float32Array, width: number, height: number): PtFrame {
  const n = width * height * 4;
  const plane = (i: number): Float32Array => planes.slice(i * n, (i + 1) * n);
  return { width, height, radiance, diffuse: plane(0), specular: plane(1), normalViewZ: plane(2), motionRoughness: plane(3), albedo: plane(4), specularFactor: plane(5) };
}
