/**
 * Neural radiance cache on main's megakernel path tracer (research track N, TASKS N6a, docs/internal/design/nrc-plan.md §5).
 *
 * Two hooked kernels share one hook text (`NRC_HOOK_WGSL`), both built from the plain kernel's own options so the only
 * difference from plain path tracing is the hook:
 *
 *  - **render** (the image): every path terminates into the cache at its vertex `queryDepth` (after that vertex's
 *    emission and NEE), appends a query record and adds 0. After the pass, inference over the queries and a resolve pass
 *    add `throughput × max(prediction, 0)` into the render kernel's accumulator (one query per pixel per pass, so the
 *    add is a plain store, no atomics).
 *  - **trainer** (training paths): a small film over the same camera. Targets come from deterministic replays instead
 *    of a path-end callback (the hooks expose neither the NEE term of a vertex nor the final escape, V84): with the same
 *    seed and sample index, run k terminates every path at vertex k with 0, so its accumulator holds the radiance up to
 *    and including vertex k's emission and NEE (R_k) and writes the vertex record; the tail run terminates at vertex M
 *    and writes a tail query. Then target_k = (R_M − R_k + β_M · cache(x_M)) / β_k is exactly what the render hook
 *    must return at vertex k (MIS included), with the cache's own prediction as the tail (self-training, N-I5).
 *    Cost: M + 1 trainer passes per frame (DIVERGES N-I9 in nrc-diverges.md).
 *
 * `model: 'hash'` (M7) swaps the network for a world-space hash grid (`hash-grid.ts`): inference becomes a lookup, the training steps one
 * update + resolve; the hook, the targets, the resolve into the image and the fix-up are the same code.
 *
 * Per frame, everything is encoded into one command encoder with no readback: trainer runs → tail inference → targets
 * → `stepsPerFrame` training steps → render pass → inference → resolve.
 */

import { MlpEngine, nrcEncodingWgsl, initWeights, layerOffset, MLP_SHAPES, DEFAULT_TRAIN, type MlpNetwork, type Precision, type MlpEngineOptions } from '@pathtracer/mlp';
import { PtMegakernel, type PtHooksOptions, type PtLookAtCamera, type PtMegakernelOptions } from '@pmndrs/pathtracer';
import type { Caps } from '@pmndrs/raytracer';
import { candidateWgsl, type NrcTerminationOptions } from './termination.ts';
import { HashGrid, type HashGridSettings } from './hash-grid.ts';
import { SpatialFeatures, spatialCollisions, spatialEncodingWgsl, type SpatialFeatureSettings } from './spatial-features.ts';

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;
const COPY_SRC = GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM;

/** vec4 per record: (position, pixel as an exact f32), (direction θ/π φ/2π, normal θ/π φ/2π), (albedo, roughness), (throughput, valid). */
/** Legacy record stride, also used by the hash grid. Evaluated MLP records append specular RGB and BSDF kind. */
export const NRC_RECORD_VEC4 = 4;
const HOOK_SLOT = 256;

/** The hook both NRC kernels compile. mode 0: render query at depth >= stopDepth; mode 1: trainer record at stopDepth. */
export const NRC_HOOK_WGSL = /* wgsl */ `
struct NrcHookParams { mode: u32, stopDepth: u32, layerOffset: u32, capacity: u32 }
@group(3) @binding(0) var<uniform> nrcHook: NrcHookParams;
@group(3) @binding(1) var<storage, read_write> nrcRecords: array<vec4f>;
@group(3) @binding(2) var<storage, read_write> nrcCount: array<atomic<u32>>;

fn nrcSpherical(d: vec3f) -> vec2f {
    return vec2f(acos(clamp(d.y, -1.0, 1.0)) * 0.3183098861837907, atan2(d.z, d.x) * 0.15915494309189535 + 0.5);
}

// Roughness input (N-I10): the legacy material adapter makes a material with metalness < 0.5 a pure Lambertian (roughness 1 for the
// cache) and anything else a GGX conductor with the material's specular roughness.
fn nrcRoughness(v: PtPathVertex) -> f32 {
    let m = rtMaterial(v.material);
    return select(1.0, m.specularRoughness, m.metalness >= 0.5);
}

fn nrcWrite(slot: u32, v: PtPathVertex) {
    let b = slot * ${NRC_RECORD_VEC4}u;
    nrcRecords[b] = vec4f(v.position, f32(v.pixel));
    nrcRecords[b + 1u] = vec4f(nrcSpherical(v.wo), nrcSpherical(v.normal));
    nrcRecords[b + 2u] = vec4f(v.albedo, nrcRoughness(v));
    nrcRecords[b + 3u] = vec4f(v.throughput, 1.0);
}

fn ptOnVertex(v: PtPathVertex) -> u32 {
    if (v.depth < nrcHook.stopDepth) {
        return PT_HOOK_CONTINUE;
    }
    if (nrcHook.mode == 0u) {
        let slot = atomicAdd(&nrcCount[0], 1u);
        if (slot < nrcHook.capacity) {
            nrcWrite(slot, v);
        }
    } else {
        nrcWrite(nrcHook.layerOffset + v.pixel, v);
    }
    return PT_HOOK_TERMINATE;
}

fn ptTerminate(v: PtPathVertex) -> vec3f {
    return vec3f(0.0);
}
`;

/**
 * One-pass training (`training: 'inline'`, TASKS N6d): some of the render pass's own paths are the training paths (`trainSelect`:
 * whole 8 × 8 tiles or single pixels, `nrcTrainSlot`). Their vertices 0..M−1 are recorded per block slot, `ptOnRadiance` sums what they gather per
 * vertex index, and at vertex M they end in an ordinary query (its prediction goes into the image and into the targets).
 * target_k = (Σ_{j>k} gathered_j + β_M · cache(x_M)) / β_k: the same quantity as the replays, with no extra pass.
 */
function inlineHookWgsl(hashLookup?: string, policy?: NrcSettings, mirrorEmission = false, evaluated = false, fuzz = false): string {
  const extended = evaluated && policy?.model !== 'hash';
  const recordVec4 = extended ? NRC_RECORD_VEC4 + 1 : NRC_RECORD_VEC4;
  const direction = policy?.directionInput === 'reflected' ? 'reflect(-v.wo, v.shadingNormal)' : 'v.wo';
  const angles = `vec4f(nrcSpherical(${direction}), nrcSpherical(v.${extended ? 'shadingNormal' : 'normal'}))`;
  const fullEvery = policy?.fullSuffixEvery ?? 0;
  const factorized = (policy?.factorization ?? 'off') !== 'off';
  const materialClass = factorized ? 'nrcRecordedClass(v)' : 'f32(v.materialClass)';
  return /* wgsl */ `
${hashLookup ?? ''}
struct NrcHookParams { mode: u32, stopDepth: u32, layerOffset: u32, capacity: u32, trainDepth: u32, width: u32, block: u32, slotsX: u32, slots: u32, tileEvery: u32, tilesX: u32, sumsOffset: u32 }
@group(3) @binding(0) var<uniform> nrcHook: NrcHookParams;
@group(3) @binding(1) var<storage, read_write> nrcRecords: array<vec4f>;
@group(3) @binding(2) var<storage, read_write> nrcCount: array<atomic<u32>>;
@group(3) @binding(3) var<storage, read_write> nrcTrain: array<vec4f>;
// Training vertices and per-path sums share one storage binding. sumsOffset is in vec4 units.

fn nrcSpherical(d: vec3f) -> vec2f {
    return vec2f(acos(clamp(d.y, -1.0, 1.0)) * 0.3183098861837907, ${evaluated ? 'select(atan2(d.z, d.x), 0.0, all(d.xz == vec2f(0.0)))' : 'atan2(d.z, d.x)'} * 0.15915494309189535 + 0.5);
}

// Roughness input (N-I10): the legacy material adapter makes a material with metalness < 0.5 a pure Lambertian (roughness 1 for the
// cache) and anything else a GGX conductor with the material's specular roughness.
fn nrcRoughness(v: PtPathVertex) -> f32 {
    let m = rtMaterial(v.material);
    return select(1.0, m.specularRoughness, m.metalness >= 0.5);
}

${factorized ? `
fn nrcRecordedClass(v: PtPathVertex) -> f32 {
    // Bit 3 exempts a delta interface or an additional fuzz lobe from base-color demodulation.
    return f32(v.materialClass + select(0u, 8u, v.delta${fuzz ? ' || rtMaterial(v.material).sheenWeight > 0.0' : ''}));
}
` : ''}

// Training-path slot + 1 for this pixel and sample, 0 when the path is not a training path. Tiles (tileEvery > 0): every
// tileEvery-th 8 x 8 tile (the megakernel's workgroup) trains, rotating with the sample, so a workgroup is all training paths or
// none (no lane waits on a long training path; VkNRC picks whole subgroups for the same reason). Pixels: one pixel per
// block x block, rotating (main's updateBlock rule).
fn nrcTrainSlot(pixel: u32, sampleIndex: u32) -> u32 {
    if (nrcHook.mode != 0u) { return 0u; }
    // Rotate with the cache's own frame counter (nrcCount[1], bumped by the resolve pass), not the sample index: clearing the
    // image restarts the sample index, which would pin training to the same tiles while the camera moves.
    let frame = atomicLoad(&nrcCount[1]);
    let x = pixel % nrcHook.width;
    let y = pixel / nrcHook.width;
    if (nrcHook.tileEvery > 0u) {
        // Diagonal pattern: tile (tx, ty) trains when (tx + block * ty) % tileEvery == frame % tileEvery, so each row shifts by
        // block tiles (training tiles scatter instead of lining up in columns when tilesX is a multiple of tileEvery) and
        // every tile trains once per tileEvery frames. One selected tile per tileEvery-wide group of a row: a unique slot.
        let tx = x / 8u;
        let ty = y / 8u;
        if ((tx + nrcHook.block * ty) % nrcHook.tileEvery != frame % nrcHook.tileEvery) {
            return 0u;
        }
        let groupsX = (nrcHook.tilesX + nrcHook.tileEvery - 1u) / nrcHook.tileEvery;
        return (ty * groupsX + tx / nrcHook.tileEvery) * 64u + (y % 8u) * 8u + x % 8u + 1u;
    }
    let b = nrcHook.block;
    if ((x % b) + (y % b) * b != frame % (b * b)) {
        return 0u;
    }
    return (y / b) * nrcHook.slotsX + x / b + 1u;
}

// Appends a query; returns its index + 1 (0 when full). flag 1: the resolve pass adds it to the image; 2: training only.
fn nrcQuery(v: PtPathVertex, flag: f32) -> u32 {
    let q = atomicAdd(&nrcCount[0], 1u);
    if (q >= nrcHook.capacity) {
        return 0u;
    }
    let b = q * ${recordVec4}u;
    nrcRecords[b] = vec4f(v.position, f32(v.pixel));
    nrcRecords[b + 1u] = ${angles};
    nrcRecords[b + 2u] = vec4f(${evaluated ? 'v.diffuseReflectance, v.roughness' : 'v.albedo, nrcRoughness(v)'});
    nrcRecords[b + 3u] = vec4f(v.throughput, flag);${extended ? `
    nrcRecords[b + 4u] = vec4f(v.specularReflectance, ${materialClass});` : ''}
    return q + 1u;
}

${candidateWgsl(policy)}

// Inline hash display queries carry their resolved RGB and level in the otherwise unused angle word.
// A missing/undertrained cell has level -1; a valid black cell is still a hit.
fn nrcAcceptDisplay(v: PtPathVertex, query: u32) -> bool {
    ${policy?.terminationStats ? 'atomicAdd(&nrcCount[7u + nrcSurfaceBin(v)], 1u); atomicAdd(&nrcCount[10u], v.depth);' : ''}
    ${hashLookup ? `if (query == 0u) { return false; }
    let b = (query - 1u) * ${recordVec4}u;
    ${(policy?.hash.varianceThreshold ?? -1) >= 0 ? `let sample = hgSampleWithConfidence(v.position, nrcRecords[b + 1u], nrcRecords[b + 2u].w, query - 1u, atomicLoad(&nrcCount[1]));
    let prediction = sample.prediction;
    if (sample.varianceRejected) {
        nrcRecords[b + 1u] = prediction;
        nrcRecords[b + 3u].w = 5.0;
        return false;
    }` : `let prediction = hgSample(v.position, nrcRecords[b + 1u], nrcRecords[b + 2u].w, query - 1u, atomicLoad(&nrcCount[1]));`}
    nrcRecords[b + 1u] = prediction;
    ${policy?.maxDisplayCoarse !== undefined ? `if (prediction.w > ${policy.maxDisplayCoarse}.0) {
        nrcRecords[b + 3u].w = 4.0; // Keep the actual prediction for a shared training tail at M.
        return false;
    }` : ''}
    nrcRecords[b + 3u].w = 3.0;
    return prediction.w >= 0.0;` : 'return true;'}
}

${fullEvery ? `var<private> nrcFullSuffix: bool;
fn nrcChooseFull(pixel: u32, frame: u32) -> bool {
    let s = (pixel ^ (frame * 0x9e3779b9u)) * 747796405u + 2891336453u;
    let h = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return ((h >> 22u) ^ h) % ${fullEvery}u == 0u;
}` : ''}
fn ptOnVertex(v: PtPathVertex) -> u32 {
    let display = nrcDisplayCandidate(v);
    let train = nrcTrainSlot(v.pixel, v.sampleIndex);
    if (train != 0u) {
        let slot = train - 1u;
        // The image gets the same estimate as every other pixel: a display query at the cache vertex (resolved into the
        // image), and the fix-up pass later removes what the path gathers beyond it. The path itself trains to vertex M.
        let s0 = nrcHook.sumsOffset + slot * (nrcHook.trainDepth + 1u);
        ${fullEvery ? `if (v.depth == 0u) { nrcFullSuffix = nrcChooseFull(v.pixel, atomicLoad(&nrcCount[1])); }
        // Full training suffixes bypass the bootstrap. Any display query still resolves at its original hit.
        if (nrcFullSuffix && v.depth >= nrcHook.trainDepth) {
            if (display && nrcAcceptDisplay(v, nrcQuery(v, 1.0))) { nrcTrain[s0 + 1u].w = f32(v.depth + 1u); }
            return PT_HOOK_CONTINUE;
        }` : ''}
        if (v.depth < nrcHook.trainDepth) {
            let b = (v.depth * nrcHook.slots + slot) * ${recordVec4}u;
            nrcTrain[b] = vec4f(v.position, f32(v.pixel));
            nrcTrain[b + 1u] = ${angles};
            nrcTrain[b + 2u] = vec4f(${evaluated ? 'v.diffuseReflectance, v.roughness' : 'v.albedo, nrcRoughness(v)'});
            nrcTrain[b + 3u] = vec4f(v.throughput, 1.0);${extended ? `
            nrcTrain[b + 4u] = vec4f(v.specularReflectance, ${materialClass});` : ''}
            if (display) {
                if (nrcAcceptDisplay(v, nrcQuery(v, 1.0))) {
                    nrcTrain[s0 + 1u].w = f32(v.depth + 1u);
                }
            }
            return PT_HOOK_CONTINUE;
        }
        if (v.depth == nrcHook.trainDepth) {
            // Only the boundary vertex bootstraps training. Its prediction can also serve a display query at the same depth.
            let tail = nrcQuery(v, select(2.0, 1.0, display));
            nrcTrain[s0].w = f32(tail);
            if (display) {
                return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, nrcAcceptDisplay(v, tail));
            }
            if (nrcTrain[s0 + 1u].w != 0.0) { return PT_HOOK_TERMINATE; }
            return PT_HOOK_CONTINUE;
        }
        // A deeper display query (or a gated plain path) proceeds independently of the recorded training suffix.
    }
    if (display) {
        return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, nrcAcceptDisplay(v, nrcQuery(v, 1.0)));
    }
    return PT_HOOK_CONTINUE;
}

fn ptTerminate(v: PtPathVertex) -> vec3f {
    return vec3f(0.0);
}

fn ptOnRadiance(pixel: u32, sampleIndex: u32, index: u32, flags: u32, value: vec3f) {
    let train = nrcTrainSlot(pixel, sampleIndex);
    if (train == 0u || (index > nrcHook.trainDepth${fullEvery ? ' && !nrcFullSuffix' : ''})) {
        return;
    }
    let slot = train - 1u;
    let s0 = nrcHook.sumsOffset + slot * (nrcHook.trainDepth + 1u);
    let i = s0 + ${fullEvery ? 'min(index, nrcHook.trainDepth)' : 'index'};
    nrcTrain[i] = vec4f(nrcTrain[i].xyz + value, nrcTrain[i].w);${fullEvery ? `
    // The last target bucket can contain several bounces. Track the displayed suffix independently, including when K >= M.
    let displayedFrom = u32(nrcTrain[s0 + 1u].w);
    if (displayedFrom != 0u && index >= displayedFrom) {
        let excess = nrcHook.sumsOffset + nrcHook.slots * (nrcHook.trainDepth + ${mirrorEmission ? '2' : '1'}u) + slot;
        nrcTrain[excess] += vec4f(value, 0.0);
    }` : ''}${mirrorEmission ? `
    if (index > 0u && (flags & PT_HOOK_RADIANCE_EMISSION) != 0u) {
        let firstBeyond = u32(nrcTrain[s0 + 1u].w);
        if (firstBeyond != 0u && index >= firstBeyond) {
            let pure = nrcHook.sumsOffset + nrcHook.slots * (nrcHook.trainDepth + 1u) + slot;
            nrcTrain[pure] += vec4f(value, 0.0);
        }
    }` : ''}
}
`;
}

export const NRC_HOOK_INLINE_WGSL = inlineHookWgsl();

const NRC_HOOK_INLINE_BINDINGS: GPUBindGroupLayoutEntry[] = [
  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ...[1, 2, 3].map((binding): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } })),
];

export type NrcFactorization = 'off' | 'lambert' | 'reflectance';

export function factorizationArg(value = 'off'): NrcFactorization {
  if (value === 'off' || value === 'lambert' || value === 'reflectance') return value;
  throw new Error("factorization must be 'off', 'lambert' or 'reflectance'");
}

export type NrcDirectionInput = 'view' | 'reflected';

export function directionArg(value?: string | null): NrcDirectionInput {
  if (value == null || value === 'view') return 'view';
  if (value === 'reflected') return value;
  throw new Error('directionInput must be view or reflected');
}

export function materialArgs(target?: string | null, inputs?: string | null, direction?: string | null): Partial<Pick<NrcSettings, 'factorization' | 'materialInputs' | 'directionInput'>> {
  const directionInput = directionArg(direction);
  const factorization = factorizationArg(target ?? 'off');
  if (!inputs) return { factorization, directionInput };
  if (inputs !== 'legacy' && inputs !== 'evaluated') throw new Error('materialInputs must be legacy or evaluated');
  return { materialInputs: inputs, factorization, directionInput };
}

// Both hooks consume the same stored, texture-evaluated factor. The target builder and render resolve
// always see physical radiance from a tail query, including when it also supplies the display pixel.
function factorWgsl(mode: NrcFactorization): string {
  if (mode === 'off') return '';
  return /* wgsl */ `
fn nrcFactor(diffuse: vec3f, material: vec4f) -> vec3f {
  if (material.w == 0.0) {
    return select(vec3f(0.0), max(diffuse, vec3f(1.0 / 1024.0)), diffuse > vec3f(0.0));
  }${mode === 'reflectance' ? `
  // F0 only approximates the directional response. Keep dark metal channels learnable at grazing angles.
  if (material.w == 1.0) { return max(material.xyz, vec3f(1.0 / 16.0)); }` : ''}
  return vec3f(1.0);
}
`;
}

const targetsInlineWgsl = (recordVec4: number, mode: NrcFactorization) => /* wgsl */ `
${factorWgsl(mode)}
struct P { slots: u32, layers: u32, batch: u32, capacity: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;
@group(0) @binding(2) var<storage, read> sums: array<vec4f>;
@group(0) @binding(3) var<storage, read> queries: array<vec4f>;
@group(0) @binding(4) var<storage, read> preds: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outSet: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> count: array<atomic<u32>>;

fn finite3(v: vec3f) -> bool {
  let e = (bitcast<vec3u>(v) >> vec3u(23u)) & vec3u(0xffu);
  return all(e != vec3u(0xffu));
}

@compute @workgroup_size(64)
fn nrcTargets(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.slots * p.layers) { return; }
  let k = i / p.slots;
  let slot = i % p.slots;
  let rb = (k * p.slots + slot) * ${recordVec4}u;
  let beta = verts[rb + 3u];
  if (beta.w == 0.0) { return; }
  let s0 = slot * (p.layers + 1u);
  var after = vec3f(0.0);
  for (var j = k + 1u; j <= p.layers; j++) { after += sums[s0 + j].xyz; }
  let tq = u32(sums[s0].w);
  if (tq != 0u) {
    let qb = (tq - 1u) * ${recordVec4}u;
    after += queries[qb + 3u].xyz * max(preds[tq - 1u].xyz, vec3f(0.0));
  }
  let ${mode === 'off' ? 't' : 'radiance'} = max(select(vec3f(0.0), after / beta.xyz, beta.xyz > vec3f(1.0e-8)), vec3f(0.0));
  ${mode === 'off' ? '' : `let factor = nrcFactor(verts[rb + 2u].xyz, verts[rb + 4u]);
  let t = select(vec3f(0.0), radiance / select(vec3f(1.0), factor, factor > vec3f(0.0)), factor > vec3f(0.0));`}
  if (!finite3(t)) { return; }
  let slotOut = atomicAdd(&count[1], 1u);
  if (slotOut >= p.capacity) { return; }
  let o = slotOut * ${recordVec4}u;
  outSet[o] = verts[rb];
  outSet[o + 1u] = verts[rb + 1u];
  outSet[o + 2u] = verts[rb + 2u];
  outSet[o + 3u] = vec4f(t, 1.0);${recordVec4 > NRC_RECORD_VEC4 ? `
  outSet[o + 4u] = verts[rb + 4u];` : ''}
}

@compute @workgroup_size(1)
fn nrcFinish() {
  let n = min(atomicLoad(&count[1]), p.capacity);
  atomicStore(&count[1], n);
  atomicStore(&count[0], min(n, p.batch));
}
`;

const NRC_HOOK_BINDINGS: GPUBindGroupLayoutEntry[] = [
  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
];

/** Shared by the MLP hooks: records → the 64 NRC inputs (VkNRC layout), position normalised to the scene box. */
const NRC_COORDINATES_WGSL = /* wgsl */ `
struct NrcBox { lo: vec4f, scale: vec4f }
fn nrcNormalizePosition(world: vec3f) -> vec3f {
  return clamp((world - nrcBox.lo.xyz) * nrcBox.scale.xyz - 1.0, vec3f(-1.0), vec3f(1.0));
}
`;

export const nrcInputsWgsl = (evaluated: boolean, encoding: string) => /* wgsl */ `
${encoding}
${NRC_COORDINATES_WGSL}
fn nrcInputsOf(r0: vec4f, r1: vec4f, r2: vec4f${evaluated ? ', material: vec4f' : ''}) -> array<f32, NN_IN> {
  let p = nrcNormalizePosition(r0.xyz);
  ${evaluated ? `var inputs = nnEncodeNrc(p, r1.xy, r1.zw, r2.w, r2.xyz, material.xyz);
  // Replace one of the two constant inputs with the evaluated lobe class.
  inputs[NN_IN - 2u] = 1.0 + 0.2 * f32(u32(material.w) & 7u);
  return inputs;` : 'return nnEncodeNrc(p, r1.xy, r1.zw, r2.w, r2.xyz, vec3f(0.0));'}
}
`;

const nrcInferHooks = (recordVec4: number, evaluated: boolean, mode: NrcFactorization, encoding: string) => /* wgsl */ `
${factorWgsl(mode)}
@group(1) @binding(0) var<storage, read> nrcIn: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> nrcOut: array<vec4f>;
@group(1) @binding(2) var<uniform> nrcBox: NrcBox;
${nrcInputsWgsl(evaluated, encoding)}
fn nnInferLoad(r: u32) -> array<f32, NN_IN> {
  let b = r * ${recordVec4}u;
  return nrcInputsOf(nrcIn[b], nrcIn[b + 1u], nrcIn[b + 2u]${evaluated ? ', nrcIn[b + 4u]' : ''});
}
fn nnInferStore(r: u32, y: vec3f) {
  ${mode === 'off' ? 'nrcOut[r] = vec4f(y, 0.0);' : `let b = r * ${recordVec4}u;
  nrcOut[r] = vec4f(y * nrcFactor(nrcIn[b + 2u].xyz, nrcIn[b + 4u]), 0.0);`}
}
`;

const NRC_TRAIN_RECORD_INDEX_WGSL = /* wgsl */ `
fn nrcHash(x: u32) -> u32 {
  let s = x * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn nrcTrainRecordIndex(r: u32) -> u32 {
  let n = max(nrcSetCount[1], 1u);
  return nrcHash(r ^ nrcHash(nnState.step * 0x9e3779b9u + 17u)) % n;
}
`;

const nrcTrainHooks = (recordVec4: number, evaluated: boolean, encoding: string) => /* wgsl */ `
@group(1) @binding(0) var<storage, read> nrcSet: array<vec4f>;
@group(1) @binding(1) var<storage, read> nrcSetCount: array<u32>;
@group(1) @binding(2) var<uniform> nrcBox: NrcBox;
${nrcInputsWgsl(evaluated, encoding)}
${NRC_TRAIN_RECORD_INDEX_WGSL}
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
  let b = nrcTrainRecordIndex(r) * ${recordVec4}u;
  *x = nrcInputsOf(nrcSet[b], nrcSet[b + 1u], nrcSet[b + 2u]${evaluated ? ', nrcSet[b + 4u]' : ''});
  return nrcSet[b + 3u].xyz;
}
`;

const TARGETS_WGSL = /* wgsl */ `
struct P { npix: u32, layers: u32, batch: u32, capacity: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;
@group(0) @binding(2) var<storage, read> radiance: array<vec4f>;
@group(0) @binding(3) var<storage, read> tailPred: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outSet: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> count: array<atomic<u32>>;

fn finite3(v: vec3f) -> bool {
  let e = (bitcast<vec3u>(v) >> vec3u(23u)) & vec3u(0xffu);
  return all(e != vec3u(0xffu));
}

@compute @workgroup_size(64)
fn nrcTargets(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.npix * p.layers) { return; }
  let k = i / p.npix;
  let px = i % p.npix;
  let rb = (k * p.npix + px) * ${NRC_RECORD_VEC4}u;
  let beta = verts[rb + 3u];
  if (beta.w == 0.0) { return; }
  let tail = verts[(p.layers * p.npix + px) * ${NRC_RECORD_VEC4}u + 3u];
  var after = radiance[p.layers * p.npix + px].xyz - radiance[k * p.npix + px].xyz;
  if (tail.w != 0.0) { after += tail.xyz * max(tailPred[px].xyz, vec3f(0.0)); }
  let t = max(select(vec3f(0.0), after / beta.xyz, beta.xyz > vec3f(1.0e-8)), vec3f(0.0));
  if (!finite3(t)) { return; }
  let slot = atomicAdd(&count[1], 1u);
  if (slot >= p.capacity) { return; }
  let o = slot * ${NRC_RECORD_VEC4}u;
  outSet[o] = verts[rb];
  outSet[o + 1u] = verts[rb + 1u];
  outSet[o + 2u] = verts[rb + 2u];
  outSet[o + 3u] = vec4f(t, 1.0);
}

@compute @workgroup_size(1)
fn nrcFinish() {
  let n = min(atomicLoad(&count[1]), p.capacity);
  atomicStore(&count[1], n);
  atomicStore(&count[0], min(n, p.batch));
}
`;

/**
 * Frame mode (`integrator.frame`, P4.1 planes; docs/internal/design/nrc-pipeline.md §2): the cache's indirect light is part of the primary vertex's
 * class, so it also lands in the demodulated diffuse (plane 0) or specular (plane 1) plane, divided by that plane's stored factor (plane 4
 * rgb / plane 5 rgb; plane 4 alpha = 1 marks a specular-class primary). `diffuse * albedo + specular * specFactor = beauty` stays exact.
 * Empty when off, so the plain resolve text is unchanged.
 */
function nrcPlanesWgsl(frame: boolean, binding: number): string {
  if (!frame) return '';
  return /* wgsl */ `
@group(0) @binding(${binding}) var<storage, read_write> planes: array<vec4f>;
fn nrcPlaneAdd(px: u32, n: u32, v: vec3f) {
  if (planes[4u * n + px].w == 1.0) {
    planes[n + px] = planes[n + px] + vec4f(v / planes[5u * n + px].xyz, 0.0);
  } else {
    planes[px] = planes[px] + vec4f(v / planes[4u * n + px].xyz, 0.0);
  }
}
`;
}

/** The resolve pass; `frame`: also add into the demodulated planes (`nrcPlanesWgsl`); `p.npix` = pixel count. */
function resolveWgsl(frame: boolean, recordVec4: number): string {
  return /* wgsl */ `
struct P { capacity: u32, npix: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> queries: array<vec4f>;
@group(0) @binding(2) var<storage, read> preds: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> count: array<u32>;
@group(0) @binding(4) var<storage, read_write> accum: array<vec4f>;
${nrcPlanesWgsl(frame, 5)}
@compute @workgroup_size(64)
fn nrcResolve(@builtin(global_invocation_id) gid: vec3u) {
  let r = gid.x;
  if (r == 0u) {
    count[1] = count[1] + 1u; // the frame counter the hook rotates training tiles with
  }
  if (r >= min(count[0], p.capacity)) { return; }
  let b = r * ${recordVec4}u;
  if (queries[b + 3u].w != 1.0 && queries[b + 3u].w != 3.0) { return; } // a training path's tail: training only
  let px = u32(queries[b].w);
  let beta = queries[b + 3u].xyz;
  let add = beta * max(preds[r].xyz, vec3f(0.0));
  accum[px] = accum[px] + vec4f(add, 0.0);${frame ? '\n  nrcPlaneAdd(px, p.npix, add);' : ''}
}
`;
}

/**
 * Inline training, after the resolve: a training path's pixel already holds everything the path gathered (the integrator added it);
 * remove what it gathered beyond the cache vertex K (reported values with index > K, summed per path by ptOnRadiance), so the pixel
 * keeps the same estimate as every other pixel (its display query at K was resolved into the image). Only when K < M.
 */
function fixupWgsl(frame: boolean, mirrorEmission: boolean, recordVec4: number, fullSuffix: boolean, diffuseFirst: boolean): string {
  return /* wgsl */ `
struct P { slots: u32, queryDepth: u32, trainDepth: u32, npix: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> verts: array<vec4f>;
@group(0) @binding(2) var<storage, read> sums: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4f>;
${nrcPlanesWgsl(frame, 4)}
@compute @workgroup_size(64)
fn nrcFixup(@builtin(global_invocation_id) gid: vec3u) {
  let slot = gid.x;
  if (slot >= p.slots${fullSuffix || diffuseFirst ? '' : ' || p.queryDepth >= p.trainDepth'}) { return; }
  let v0 = slot * ${recordVec4}u; // layer 0: the primary vertex (no record: the camera ray escaped, nothing to remove)
  if (verts[v0 + 3u].w == 0.0) { return; }
  let s0 = slot * (p.trainDepth + 1u);
  let firstBeyond = u32(sums[s0 + 1u].w);
  if (firstBeyond == 0u) { return; } // A miss kept the complete traced suffix in the image.
  ${fullSuffix ? `var beyond = sums[p.slots * (p.trainDepth + ${mirrorEmission ? '2' : '1'}u) + slot].xyz;` : `var beyond = vec3f(0.0);
  for (var j = firstBeyond; j <= p.trainDepth; j++) { beyond += sums[s0 + j].xyz; }`}
  let px = u32(verts[v0].w);
  accum[px] = accum[px] - vec4f(beyond, 0.0);${mirrorEmission ? `
  let pure = sums[p.slots * (p.trainDepth + 1u) + slot].xyz;
  planes[6u * p.npix + px] -= vec4f(pure, 0.0);
  beyond -= pure;` : ''}${frame ? '\n  nrcPlaneAdd(px, p.npix, -beyond);' : ''}
  ${frame ? `if (firstBeyond == 1u) {
    planes[px].w = 0.0;
    planes[p.npix + px].w = 0.0;
  }` : ''}
}
`;
}

export interface NrcSettings extends NrcTerminationOptions {
  /**
   * What stores the cache (M7, Q4): 'mlp' (default) the network; 'hash' a world-space hash grid (`hash-grid.ts`) fed by the same training
   * records and queried by the same render paths, so only the store differs. 'hash' needs inline training; network settings are ignored.
   */
  model: 'mlp' | 'hash';
  /** Hidden width and layer count for the 64-input NRC encoding. */
  readonly network: MlpNetwork;
  /** Evaluate material inputs inline. Hash records use the evaluated roughness and retain geometric normal keys. */
  materialInputs: 'legacy' | 'evaluated';
  /** Replace the outgoing direction with the incident camera ray reflected about the evaluated shading normal. */
  readonly directionInput: NrcDirectionInput;
  /** Active triangle-wave bands per world-space axis, with the 64 input lanes kept fixed. */
  readonly positionBands: number;
  /** Trainable multiresolution position features in the MLP's position lanes. */
  readonly spatialFeatures?: SpatialFeatureSettings;
  /** Learn an indirect target divided by the eligible vertex reflectance. Requires evaluated inline MLP records. */
  factorization: NrcFactorization;
  /** Hash grid settings (`model: 'hash'`). */
  hash: Partial<HashGridSettings>;
  /**
   * Where training paths come from: 'inline' (default) the render pass's own update paths, gathered through `ptOnRadiance` (no extra
   * pass); 'replay' a separate trainer film replayed trainDepth + 1 times per frame (the first version, kept for the A/B).
   */
  training: 'inline' | 'replay';
  /** Use display-only tracing on inline frames encoded with `train: false`. */
  frozenDisplayOnly: boolean;
  /** Inline training: 1 / updateBlock² of the pixels train each frame (default 4: 6.25%). */
  updateBlock: number;
  /**
   * Inline training: which pixels. 'tile' (default) every updateBlock²-th 8 × 8 tile, so a workgroup is all training paths or none;
   * 'pixel' one pixel per updateBlock × updateBlock block (every workgroup waits on its longest, training, path).
   */
  trainSelect: 'tile' | 'pixel';
  precision: Precision;
  /** Inference backend options from `inferArmOptions` (portable when absent). */
  sgm?: MlpEngineOptions['sgm'];
  /** Portable inference kernel when `sgm` is absent: 'tiled' (default, N1.4) or 'perRecord'. */
  infer?: 'perRecord' | 'tiled';
  /** Subgroup-matrix forward-backward for training (`forward: 'sgm'`, from `forwardArmOptions`); tiled when absent. */
  trainSgm?: MlpEngineOptions['trainSgm'];
  /** Subgroup-matrix weight gradient (`gradient: 'sgm'`, needs `trainSgm` in the blocked layout); tiled when absent. */
  gradientSgm?: MlpEngineOptions['gradientSgm'];
  /** Vertex the render paths terminate into the cache at (0 = the primary hit: the "cache only" view). Default 1. */
  queryDepth: number;
  /** Training paths are terminated at vertex M with the cache as the tail; targets for vertices 0..M−1. Default 4. */
  trainDepth: number;
  /** One in N selected inline paths traces its training suffix to the integrator limit. Zero uses only bootstraps. */
  readonly fullSuffixEvery: number;
  /** Replay training: the trainer film (same camera, same aspect). Default 128 × 72. */
  trainWidth: number;
  trainHeight: number;
  stepsPerFrame: number;
  batch: number;
  learningRate: number;
  useEma: boolean;
  seed: number;
  /** Damping a fresh cache (N6f). Output layer of a fresh network: 'zero' (default: it predicts 0 at first) or 'he' (random, like the rest). */
  outputInit: 'he' | 'zero';
  /**
   * Warm-up gate (N11.7): plain path tracing in the image (no cache) until the cache has trained this many frames; the training tiles of
   * the same pass keep recording, so the network trains exactly as it would with the cache on screen. 0 = no gate.
   */
  warmupFrames: number;
  /**
   * Training steps per frame during the warm-up gate (0 = `stepsPerFrame`). Default 16 (maintainer, 2026-11-18, D30): the gate is a step
   * budget, so 16 steps open a 600-step OIDN gate after 38 frames instead of 150, with the same first cache frame and no instability (N11.7).
   */
  warmupStepsPerFrame: number;
  /** Learning rate ramps linearly from 1/n to full over this many training frames. */
  lrWarmupFrames: number;
  /** Inference weights: 'warmup' EMA (default; min(0.99, (1 + t) / (10 + t))), 'plain' EMA, or 'off' (the trained weights). */
  ema: 'off' | 'plain' | 'warmup';
}

export const DEFAULT_NRC_SETTINGS: NrcSettings = {
  termination: 'fixed',
  spreadThreshold: 0.01,
  terminationStats: false,
  model: 'mlp',
  network: 'default',
  materialInputs: 'legacy',
  directionInput: 'view',
  positionBands: 12,
  factorization: 'off',
  hash: {},
  training: 'inline',
  frozenDisplayOnly: false,
  updateBlock: 4,
  trainSelect: 'tile',
  precision: 'f16',
  queryDepth: 1,
  trainDepth: 4,
  fullSuffixEvery: 0,
  trainWidth: 128,
  trainHeight: 72,
  stepsPerFrame: 4,
  batch: 16384,
  learningRate: DEFAULT_TRAIN.learningRate,
  useEma: false,
  seed: 1,
  // N6f A/B (NRC-STUDY.md, 3 seeds): zero output layer + warm-up EMA brought a fresh cache's first-frame error to its steady level
  // (2.36 vs 4.67 ground floor, 5.36 vs 11.0 unseen corridor) and bias to 3-4% by frame 8, with no lag after a cut.
  outputInit: 'zero',
  warmupFrames: 0,
  warmupStepsPerFrame: 16,
  lrWarmupFrames: 0,
  ema: 'warmup',
};

type Stage = 'trainingPaths' | 'trainSteps' | 'renderPass' | 'inference';

export class NrcCache {
  readonly device!: GPUDevice;
  readonly render!: PtMegakernel;
  /** Replay training only. */
  readonly trainer!: PtMegakernel | null;
  /** The network (`model: 'mlp'`), null for the hash grid. */
  readonly mlp!: MlpEngine | null;
  readonly spatial!: SpatialFeatures | null;
  /** The hash grid (`model: 'hash'`), null for the network. */
  readonly hash!: HashGrid | null;
  readonly settings!: NrcSettings;
  readonly width!: number;
  readonly height!: number;
  /** Training paths per frame: trainer pixels (replay) or update-path slots (inline). */
  readonly trainPaths!: number;
  private readonly hookParams!: GPUBuffer;
  private readonly trainGroups!: GPUBindGroup[];
  private readonly trainVerts!: GPUBuffer;
  private readonly trainRadiance!: GPUBuffer | null;
  private readonly tailPred!: GPUBuffer | null;
  private readonly trainSet!: GPUBuffer;
  private readonly trainCount!: GPUBuffer;
  private readonly tailCount!: GPUBuffer | null;
  private readonly queries!: GPUBuffer;
  private readonly preds!: GPUBuffer;
  private readonly queryCount!: GPUBuffer;
  private readonly box!: GPUBuffer;
  private readonly normalizedBox!: Float32Array;
  private readonly recordVec4!: number;
  private readonly targetsPipeline!: GPUComputePipeline;
  private readonly finishPipeline!: GPUComputePipeline;
  private readonly resolvePipeline!: GPUComputePipeline;
  private readonly targetsGroup!: GPUBindGroup;
  private readonly finishGroup!: GPUBindGroup;
  private readonly resolveGroup!: GPUBindGroup;
  private readonly tailInferGroup!: GPUBindGroup | null;
  private readonly renderInferGroup!: GPUBindGroup | null;
  private readonly trainHookGroup!: GPUBindGroup | null;
  private readonly targetParams!: GPUBuffer;
  private readonly resolveParams!: GPUBuffer;
  private readonly queryCapacity!: number;
  private readonly fixupPipeline!: GPUComputePipeline | null;
  private readonly fixupGroup!: GPUBindGroup | null;
  private readonly fixupParams!: GPUBuffer | null;
  private frame = 0;

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  /**
   * Builds the hooked kernel(s) from the plain kernel's options, the MLP and the buffers.
   * `bounds`: the scene box (min, max) for the position encoding.
   */
  static async create(device: GPUDevice, caps: Caps, plain: PtMegakernel, sceneBuffers: Parameters<PtMegakernel['setScene']>, film: { width: number; height: number; camera: PtLookAtCamera; maxDepth: number; rrStartDepth?: number }, bounds: { min: [number, number, number]; max: [number, number, number] }, settings: Partial<NrcSettings> = {}): Promise<NrcCache> {
    const s: NrcSettings = { ...DEFAULT_NRC_SETTINGS, ...settings };
    if (s.spatialFeatures && (s.model === 'hash' || s.trainSgm || s.ema !== 'off' || s.useEma)) {
      throw new Error('Spatial features require a portable MLP training layout and raw inference weights');
    }
    const encoding = nrcEncodingWgsl(s.positionBands, s.spatialFeatures && spatialEncodingWgsl(s.spatialFeatures));
    directionArg(s.directionInput);
    if (s.model === 'hash' && s.positionBands !== 12) throw new Error('positionBands requires an MLP cache');
    if ((s.factorization !== 'off' || s.directionInput === 'reflected') && settings.materialInputs === undefined) s.materialInputs = 'evaluated';
    if (!Number.isSafeInteger(s.fullSuffixEvery) || s.fullSuffixEvery < 0 || s.fullSuffixEvery > 0xffffffff ||
        (s.fullSuffixEvery && s.training !== 'inline')) throw new Error('fullSuffixEvery requires a u32 and inline training when enabled');
    const inline = s.training === 'inline';
    // Frame mode (the real-time planes, nrc-pipeline): the resolve and fix-up also write the demodulated planes.
    const frame = !!plain.options.integrator?.frame;
    const mirrorEmission = frame && (plain.options.integrator?.frameEmission ?? 'mirror') === 'mirror';
    if (frame && !inline) throw new Error("NrcCache: frame mode (integrator.frame) supports training: 'inline' only");
    const useHash = s.model === 'hash';
    if (useHash && !inline) throw new Error("NrcCache: model 'hash' supports training: 'inline' only");
    if (!Number.isFinite(s.spreadThreshold) || s.spreadThreshold < 0) throw new Error('spreadThreshold must be finite and nonnegative');
    if ((s.termination !== 'fixed' || s.terminationStats) && !inline) throw new Error('adaptive termination and its diagnostics require inline paths');
    if ((s.hash.varianceThreshold ?? -1) >= 0 && (!useHash || !s.hash.continueOnMiss)) {
      throw new Error('hash variance confidence requires model=hash and hgContinue=1');
    }
    if (s.maxDisplayCoarse !== undefined && (!useHash || !s.hash.continueOnMiss || !Number.isSafeInteger(s.maxDisplayCoarse) || s.maxDisplayCoarse < 0)) {
      throw new Error('maxDisplayCoarse requires a nonnegative integer and hash continuation');
    }
    factorizationArg(s.factorization);
    const evaluated = s.materialInputs === 'evaluated';
    if (s.directionInput === 'reflected' && !evaluated) throw new Error('NrcCache: reflected directions require evaluated material inputs');
    if (s.factorization !== 'off' && !evaluated) throw new Error('NrcCache: reflectance factorization requires evaluated material inputs');
    if (evaluated && !inline) throw new Error('NrcCache: evaluated material inputs require inline training');
    if (useHash && (s.factorization !== 'off' || s.directionInput !== 'view')) throw new Error('Hash records require raw RGB and outgoing view directions');
    const recordVec4 = evaluated && !useHash ? NRC_RECORD_VEC4 + 1 : NRC_RECORD_VEC4;
    const recordBytes = recordVec4 * 16;
    const M = s.trainDepth;
    const pixels = film.width * film.height;
    const B = Math.max(1, Math.floor(s.updateBlock));
    const slotsX = Math.ceil(film.width / B);
    const tileEvery = s.trainSelect === 'tile' ? B * B : 0;
    const tilesX = Math.ceil(film.width / 8);
    const trainPaths = !inline ? s.trainWidth * s.trainHeight : tileEvery ? Math.ceil(film.height / 8) * Math.ceil(tilesX / tileEvery) * 64 : slotsX * Math.ceil(film.height / B);
    const hash = useHash ? await HashGrid.create(device, s.hash, { yFovDeg: film.camera.yFovDeg, filmHeight: film.height }) : null;
    const displayHook = hash?.settings.continueOnMiss ? hash.displayHook : null;
    const hookCode = inline ? inlineHookWgsl(displayHook?.wgsl, s, mirrorEmission, evaluated, plain.options.integrator?.materials?.features.fuzz) : NRC_HOOK_WGSL;
    const hooks: PtHooksOptions = inline ? { wgsl: hookCode, pathSegment: s.factorization !== 'off' || s.termination === 'spread' || s.terminationStats, bindings: [...NRC_HOOK_INLINE_BINDINGS, ...(displayHook?.bindings ?? [])], ...(mirrorEmission ? { radianceEmissionFlag: true } : {}), ...(evaluated ? { evaluatedMaterial: true } : {}) } : { wgsl: hookCode, bindings: NRC_HOOK_BINDINGS };
    const hooked = (label: string): PtMegakernelOptions => ({ ...plain.options, integrator: { ...plain.options.integrator, hooks }, label });
    const [render, trainer] = await Promise.all([PtMegakernel.create(device, caps, hooked('nrc render')), inline ? Promise.resolve(null) : PtMegakernel.create(device, caps, hooked('nrc trainer'))]);
    render.setScene(...sceneBuffers);
    render.setFilm(film.width, film.height);
    render.setCamera(film.camera);
    render.setLimits({ maxDepth: film.maxDepth, ...(film.rrStartDepth ? { rrStartDepth: film.rrStartDepth } : {}) });
    render.setSeed(s.seed);
    if (trainer) {
      trainer.setScene(...sceneBuffers);
      trainer.setFilm(s.trainWidth, s.trainHeight);
      trainer.setCamera(film.camera);
      // Vertex M must get a hook call (hooks run at vertices < maxDepth); no Russian roulette on training paths.
      trainer.setLimits({ maxDepth: M + 1 });
      if ((M * trainPaths * recordBytes) % device.limits.minStorageBufferOffsetAlignment) throw new Error('NrcCache: trainDepth × trainWidth × trainHeight must be a multiple of 4 (storage offset alignment)');
    }

    const buf = (label: string, size: number, usage = STORAGE | COPY_DST | COPY_SRC) => device.createBuffer({ label, size: Math.max(16, size), usage });
    // The frame-mode hook has ten storage bindings available on Chrome/Metal. Share one for vertices and sums
    // so the optional hash lookup fits too. Target and fix-up passes bind the two aligned ranges individually.
    const vertexBytes = Math.max(16, (inline ? M : M + 1) * trainPaths * recordBytes);
    const alignment = device.limits.minStorageBufferOffsetAlignment;
    const sumsOffset = Math.ceil(vertexBytes / alignment) * alignment;
    const trainVerts = buf('nrc training vertices and sums', inline ? sumsOffset + trainPaths * (M + 1 + Number(mirrorEmission) + Number(s.fullSuffixEvery > 0)) * 16 : vertexBytes);
    const inlineRanges: GPUBufferBinding[] = [{ buffer: trainVerts, size: vertexBytes }, { buffer: trainVerts, offset: sumsOffset }];
    const entriesOf = (resources: GPUBufferBinding[]) => resources.map((resource, binding) => ({ binding, resource }));
    const trainRadiance = inline ? null : buf('nrc train radiance', (M + 1) * trainPaths * 16);
    const tailPred = inline ? null : buf('nrc tail predictions', trainPaths * 16);
    const capacity = M * trainPaths;
    const trainSet = buf('nrc training set', capacity * recordBytes);
    const trainCount = buf('nrc training count', 16);
    const tailCount = inline ? null : buf('nrc tail count', 16);
    if (tailCount) device.queue.writeBuffer(tailCount, 0, new Uint32Array([trainPaths, 0, 0, 0]));
    // One query per pixel, plus (inline) a training path's display query and tail: up to two per training path.
    const queryCapacity = pixels + (inline ? trainPaths : 0);
    const queries = buf('nrc queries', queryCapacity * recordBytes);
    const preds = buf('nrc predictions', queryCapacity * 16);
    const queryCount = buf('nrc query count', s.terminationStats ? 44 : 16);
    const box = buf('nrc box', 32, UNIFORM | COPY_DST);
    const lo = bounds.min;
    const scale = bounds.max.map((h, i) => 2 / Math.max(h - (lo[i] as number), 1e-6));
    const normalizedBox = new Float32Array([...lo, 0, ...scale, 0]);
    device.queue.writeBuffer(box, 0, normalizedBox);

    // Hook params: replay slots 0..M (trainer run k), slot M+1 the render kernel; inline uses slot M+1 only.
    const hookParams = buf('nrc hook params', (M + 2) * HOOK_SLOT, UNIFORM | COPY_DST);
    for (let k = 0; k <= M; k++) device.queue.writeBuffer(hookParams, k * HOOK_SLOT, new Uint32Array([1, k, k * trainPaths, 0]));
    const renderParams = new Uint32Array([0, s.queryDepth, 0, queryCapacity, M, film.width, B, slotsX, trainPaths, tileEvery, tilesX, sumsOffset / 16]);
    device.queue.writeBuffer(hookParams, (M + 1) * HOOK_SLOT, renderParams);
    const paramsEntry = (k: number) => ({ binding: 0, resource: { buffer: hookParams, offset: k * HOOK_SLOT, size: inline ? 48 : 16 } });
    let trainGroups: GPUBindGroup[] = [];
    if (inline) {
      render.setHookBindGroup(device.createBindGroup({
        layout: render.hookBindGroupLayout as GPUBindGroupLayout,
        entries: [paramsEntry(M + 1), ...[queries, queryCount, trainVerts].map((buffer, i) => ({ binding: i + 1, resource: { buffer } })), ...(displayHook?.entries ?? [])],
      }));
    } else {
      const dummyCount = buf('nrc trainer hook count', 16);
      const group = (k: number, records: GPUBuffer, count: GPUBuffer, layout: GPUBindGroupLayout) => device.createBindGroup({ layout, entries: [paramsEntry(k), { binding: 1, resource: { buffer: records } }, { binding: 2, resource: { buffer: count } }] });
      trainGroups = Array.from({ length: M + 1 }, (_, k) => group(k, trainVerts, dummyCount, (trainer as PtMegakernel).hookBindGroupLayout as GPUBindGroupLayout));
      render.setHookBindGroup(group(M + 1, queries, queryCount, render.hookBindGroupLayout as GPUBindGroupLayout));
    }

    const mlp = useHash ? null : await MlpEngine.create(device, {
      shape: MLP_SHAPES[s.network],
      precision: s.precision,
      capacity: s.batch,
      inferCapacity: Math.max(queryCapacity, trainPaths),
      useEma: s.useEma || s.ema !== 'off',
      train: { ...DEFAULT_TRAIN, learningRate: s.learningRate, emaWarmup: s.ema === 'warmup' },
      trainHooks: nrcTrainHooks(recordVec4, evaluated, encoding),
      inferHooks: nrcInferHooks(recordVec4, evaluated, s.factorization, encoding),
      ...(s.sgm ? { sgm: s.sgm } : {}),
      ...(s.infer ? { infer: s.infer } : {}),
      ...(s.trainSgm ? { forward: 'sgm' as const, trainSgm: s.trainSgm } : {}),
      ...(s.gradientSgm ? { gradient: 'sgm' as const, gradientSgm: s.gradientSgm } : {}),
    });
    mlp?.uploadWeights(NrcCache.freshWeights(s, s.seed));
    const spatial = mlp && s.spatialFeatures ? await SpatialFeatures.create(
      mlp, s.spatialFeatures, trainSet, trainCount, box, recordVec4, NRC_COORDINATES_WGSL, NRC_TRAIN_RECORD_INDEX_WGSL,
    ) : null;
    spatial?.reset(s.seed);
    const spatialEntry = spatial ? [{ binding: 3, resource: { buffer: spatial.values } }] : [];
    const tailInferGroup = inline || !mlp
      ? null
      : device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 0, resource: { buffer: trainVerts, offset: M * trainPaths * recordBytes, size: trainPaths * recordBytes } }, { binding: 1, resource: { buffer: tailPred as GPUBuffer } }, { binding: 2, resource: { buffer: box } }, ...spatialEntry] });
    const renderInferGroup = !mlp ? null : device.createBindGroup({ layout: mlp.inferLayout, entries: [{ binding: 0, resource: { buffer: queries } }, { binding: 1, resource: { buffer: preds } }, { binding: 2, resource: { buffer: box } }, ...spatialEntry] });
    const trainHookGroup = !mlp ? null : device.createBindGroup({ layout: mlp.trainLayout, entries: [{ binding: 0, resource: { buffer: trainSet } }, { binding: 1, resource: { buffer: trainCount } }, { binding: 2, resource: { buffer: box } }, ...spatialEntry] });

    const targetsModule = device.createShaderModule({ label: 'nrc targets', code: inline ? targetsInlineWgsl(recordVec4, s.factorization) : TARGETS_WGSL });
    const [targetsPipeline, finishPipeline, resolvePipeline] = await Promise.all([
      device.createComputePipelineAsync({ label: 'nrc targets', layout: 'auto', compute: { module: targetsModule, entryPoint: 'nrcTargets' } }),
      device.createComputePipelineAsync({ label: 'nrc finish', layout: 'auto', compute: { module: targetsModule, entryPoint: 'nrcFinish' } }),
      device.createComputePipelineAsync({ label: 'nrc resolve', layout: 'auto', compute: { module: device.createShaderModule({ label: 'nrc resolve', code: resolveWgsl(frame, recordVec4) }), entryPoint: 'nrcResolve' } }),
    ]);
    const targetParams = buf('nrc targets params', 16, UNIFORM | COPY_DST);
    device.queue.writeBuffer(targetParams, 0, new Uint32Array([trainPaths, M, s.batch, capacity]));
    const targetInputs = inline
      ? [...inlineRanges, { buffer: queries }, { buffer: preds }]
      : [{ buffer: trainVerts }, { buffer: trainRadiance as GPUBuffer }, { buffer: tailPred as GPUBuffer }];
    const targetEntries = entriesOf([{ buffer: targetParams }, ...targetInputs, { buffer: trainSet }, { buffer: trainCount }]);
    const targetsGroup = device.createBindGroup({ layout: targetsPipeline.getBindGroupLayout(0), entries: targetEntries });
    // nrcFinish uses the params and the count only (auto layout drops the rest).
    const finishGroup = device.createBindGroup({ layout: finishPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: targetParams } }, { binding: targetEntries.length - 1, resource: { buffer: trainCount } }] });
    const resolveParams = buf('nrc resolve params', 16, UNIFORM | COPY_DST);
    device.queue.writeBuffer(resolveParams, 0, new Uint32Array([queryCapacity, pixels, 0, 0]));
    const resolveGroup = device.createBindGroup({
      layout: resolvePipeline.getBindGroupLayout(0),
      entries: [resolveParams, queries, preds, queryCount, render.accumulator as GPUBuffer, ...(frame ? [render.frameBuffers.planes] : [])].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    // Inline training: the fix-up that gives training pixels the same image estimate as the rest.
    let fixupPipeline: GPUComputePipeline | null = null;
    let fixupGroup: GPUBindGroup | null = null;
    let fixupParams: GPUBuffer | null = null;
    if (inline) {
      fixupPipeline = await device.createComputePipelineAsync({ label: 'nrc fixup', layout: 'auto', compute: { module: device.createShaderModule({ label: 'nrc fixup', code: fixupWgsl(frame, mirrorEmission, recordVec4, s.fullSuffixEvery > 0, s.termination === 'diffuse-first') }), entryPoint: 'nrcFixup' } });
      fixupParams = buf('nrc fixup params', 16, UNIFORM | COPY_DST);
      device.queue.writeBuffer(fixupParams, 0, new Uint32Array([trainPaths, s.queryDepth, M, pixels]));
      fixupGroup = device.createBindGroup({ layout: fixupPipeline.getBindGroupLayout(0), entries: entriesOf([{ buffer: fixupParams }, ...inlineRanges, { buffer: render.accumulator as GPUBuffer }, ...(frame ? [{ buffer: render.frameBuffers.planes }] : [])]) });
    }

    // The hash grid reads the same query records (lookup instead of inference) and the same training set (update instead of steps).
    if (hash) {
      hash.bindQueries(queries, queryCount, preds, queryCapacity);
      hash.bindRecords(trainSet, trainCount, capacity);
      hash.setCameraPosition(film.camera.position);
    }

    return new NrcCache({
      device, render, trainer, mlp, hash, spatial, settings: s, width: film.width, height: film.height, trainPaths, hookParams, trainGroups, trainVerts, trainRadiance, tailPred, trainSet, trainCount, tailCount, queries, preds, queryCount, box, normalizedBox, recordVec4,
      targetsPipeline, finishPipeline, resolvePipeline, targetsGroup, finishGroup, resolveGroup, tailInferGroup, renderInferGroup, trainHookGroup, targetParams, resolveParams, fixupPipeline, fixupGroup, fixupParams, queryCapacity,
    });
  }

  /** What stores the cache, for logs: the network's inference backend and precision, or the hash grid's size. */
  get description(): string {
    return this.hash ? `hash grid · ${this.hash.entries.toLocaleString()} entries` : `${this.mlp?.inferBackend} ${this.mlp?.precision}`;
  }

  get accumulator(): GPUBuffer {
    return this.render.accumulator as GPUBuffer;
  }

  get samples(): number {
    return this.render.samples;
  }

  /** Clears the image (keeps what the network learned). */
  reset(encoder: GPUCommandEncoder): void {
    this.render.reset(encoder);
  }

  /** Forgets everything the network learned (fresh weights, Adam state). */
  resetNetwork(): void {
    if (this.mlp) {
      this.loadWeights(NrcCache.freshWeights(this.settings, this.settings.seed + this.frame));
      return;
    }
    this.hash?.clear();
    this.trainedFrames = 0;
    this.trainedSteps = 0;
  }

  /** Loads master weights for this network shape and starts a new optimizer and warm-up schedule. */
  loadWeights(weights: Float32Array): void {
    if (!this.mlp) throw new Error('NrcCache: loadWeights requires an MLP cache');
    this.mlp.uploadWeights(weights);
    this.spatial?.reset(this.settings.seed + this.frame);
    this.trainedFrames = 0;
    this.trainedSteps = 0;
  }

  /** He-initialised weights; with outputInit 'zero' the output layer starts at 0, so a fresh cache predicts no light. */
  static freshWeights(s: NrcSettings, seed: number): Float32Array {
    const shape = MLP_SHAPES[s.network];
    const w = initWeights(shape, seed);
    if (s.outputInit === 'zero') w.fill(0, layerOffset(shape, shape.hidden));
    return w;
  }

  /** Frames trained since the network was (re)initialised. */
  private trainedFrames = 0;
  private trainedSteps = 0;

  /** Training frames and steps since the network was (re)initialised. */
  get trained(): { frames: number; steps: number } {
    return { frames: this.trainedFrames, steps: this.trainedSteps };
  }

  /** The warm-up gate is closed: the image is plain path tracing while the network trains (`warmupFrames`). */
  get warming(): boolean {
    return this.trainedFrames < this.settings.warmupFrames;
  }
  private stopWritten = -1;
  private fixupStopWritten = -1;
  private displayOnlyWritten = false;
  private readonly depthWords = new Map<number, GPUBuffer>();
  private readonly displayOnlyWords = new Map<boolean, GPUBuffer>();

  /** The hook's stop depth for this frame: no cache (0xffffffff) during warm-up frames, else the query depth. */
  private applyStopDepth(encoder: GPUCommandEncoder, displayOnly: boolean): void {
    const s = this.settings;
    const stop = this.warming ? 0xffffffff : s.queryDepth;
    const updateHook = stop !== this.stopWritten;
    const updateFixup = this.fixupParams && !displayOnly && stop !== this.fixupStopWritten;
    if (!updateHook && !updateFixup) return;
    // A gate can open between two frames in one command buffer. Queue writes would change both frames before either runs.
    let word = this.depthWords.get(stop);
    if (!word) {
      word = this.device.createBuffer({ label: 'nrc stop depth', size: 4, usage: COPY_SRC, mappedAtCreation: true });
      new Uint32Array(word.getMappedRange())[0] = stop;
      word.unmap();
      this.depthWords.set(stop, word);
    }
    if (updateHook) {
      encoder.copyBufferToBuffer(word, 0, this.hookParams, (s.trainDepth + 1) * HOOK_SLOT + 4, 4);
      this.stopWritten = stop;
    }
    if (this.fixupParams && updateFixup) {
      encoder.copyBufferToBuffer(word, 0, this.fixupParams, 4, 4);
      this.fixupStopWritten = stop;
    }
  }

  private applyDisplayOnly(encoder: GPUCommandEncoder, displayOnly: boolean): void {
    if (displayOnly === this.displayOnlyWritten) return;
    this.displayOnlyWritten = displayOnly;
    let word = this.displayOnlyWords.get(displayOnly);
    if (!word) {
      word = this.device.createBuffer({ label: 'nrc display-only mode', size: 4, usage: COPY_SRC, mappedAtCreation: true });
      new Uint32Array(word.getMappedRange())[0] = Number(displayOnly);
      word.unmap();
      this.displayOnlyWords.set(displayOnly, word);
    }
    // Keep mode changes ordered with tracing when training and frozen frames share a submit.
    encoder.copyBufferToBuffer(word, 0, this.hookParams, (this.settings.trainDepth + 1) * HOOK_SLOT, 4);
  }

  setCamera(camera: PtLookAtCamera): void {
    this.render.setCamera(camera);
    this.hash?.setCameraPosition(camera.position);
    this.trainer?.setCamera(camera);
  }

  /** Frame mode: the next frame's camera (the current one becomes the previous camera the motion vectors reproject to). */
  nextCamera(camera: PtLookAtCamera): void {
    this.render.nextFrame(camera);
    this.hash?.setCameraPosition(camera.position);
    this.trainer?.setCamera(camera);
  }

  /** Vertex the render paths terminate into the cache at (0 = primary hit). Clear the image after a change. */
  setQueryDepth(depth: number): void {
    this.settings.queryDepth = depth;
  }

  /**
   * A frame's stages in the order `encodeFrame` runs them. Replay: training paths, steps, render, inference. Inline: render (it
   * carries the training paths), inference (+ targets), steps (VkNRC's order: inference before training).
   */
  get stages(): readonly Stage[] {
    return this.settings.training === 'inline' ? ['renderPass', 'inference', 'trainSteps'] : ['trainingPaths', 'trainSteps', 'renderPass', 'inference'];
  }

  /**
   * One frame: one render sample pass with the cache, and training unless `train` is false. No readback.
   * (No timestamp writes: about 20 timestamped passes per frame lost the Metal device with "Insufficient Memory" about half
   * the time; time stages by submitting them one at a time instead, `encodeStage`.)
   */
  encodeFrame(encoder: GPUCommandEncoder, options: { train?: boolean } = {}): void {
    this.frame++;
    const train = options.train !== false;
    for (const stage of this.stages) if (train || stage === 'renderPass' || stage === 'inference') this.encodeStage(encoder, stage, train);
  }

  /** One stage of a frame, in `stages` order (`nextFrame` first when calling stages yourself). */
  encodeStage(encoder: GPUCommandEncoder, stage: Stage, train = true): void {
    const { settings: s } = this;
    const M = s.trainDepth;
    const inline = s.training === 'inline';
    const displayOnly = inline && s.frozenDisplayOnly && !train;
    if (!train && (stage === 'trainingPaths' || stage === 'trainSteps')) return;
    if (stage === 'trainingPaths') {
      if (inline || !this.trainer) return;
      const n = this.trainPaths;
      encoder.clearBuffer(this.trainVerts);
      encoder.clearBuffer(this.trainCount);
      this.trainer.setSeed(s.seed * 7919 + this.frame);
      for (let k = 0; k <= M; k++) {
        this.trainer.reset(encoder);
        this.trainer.setHookBindGroup(this.trainGroups[k] as GPUBindGroup);
        this.trainer.encode(encoder, 1);
        encoder.copyBufferToBuffer(this.trainer.accumulator as GPUBuffer, 0, this.trainRadiance as GPUBuffer, k * n * 16, n * 16);
      }
      (this.mlp as MlpEngine).encodeInfer(encoder, this.tailCount as GPUBuffer, 0, this.tailInferGroup as GPUBindGroup);
      this.encodeTargets(encoder);
    } else if (stage === 'trainSteps') {
      if (this.hash) {
        // One update + resolve per training frame (a hash grid has no steps; 0 steps = no training, as for the network).
        const steps = this.warming && s.warmupStepsPerFrame > 0 ? s.warmupStepsPerFrame : s.stepsPerFrame;
        if (steps > 0) {
          this.hash.encodeUpdate(encoder);
          this.trainedFrames++;
          this.trainedSteps++;
        }
        return;
      }
      const mlp = this.mlp as MlpEngine;
      if (s.lrWarmupFrames > 0) mlp.setLearningRate(s.learningRate * Math.min(1, (this.trainedFrames + 1) / s.lrWarmupFrames));
      const steps = this.warming && s.warmupStepsPerFrame > 0 ? s.warmupStepsPerFrame : s.stepsPerFrame;
      for (let i = 0; i < steps; i++) {
        mlp.encodeTrainStep(encoder, this.trainCount, 0, this.trainHookGroup as GPUBindGroup, undefined, undefined, this.spatial?.encodeUpdate);
      }
      if (steps > 0) {
        this.trainedFrames++;
        this.trainedSteps += steps;
      }
    } else if (stage === 'renderPass') {
      this.applyStopDepth(encoder, displayOnly);
      if (inline) this.applyDisplayOnly(encoder, displayOnly);
      if (s.terminationStats) encoder.clearBuffer(this.queryCount, 16);
      encoder.clearBuffer(this.queryCount, 0, 4); // word 1 is the frame counter
      if (inline && !displayOnly) {
        encoder.clearBuffer(this.trainVerts);
      }
      this.render.encode(encoder, 1);
    } else {
      if (this.hash) this.hash.encodeLookup(encoder);
      else (this.mlp as MlpEngine).encodeInfer(encoder, this.queryCount, 0, this.renderInferGroup as GPUBindGroup);
      if (inline && train) {
        encoder.clearBuffer(this.trainCount);
        this.encodeTargets(encoder);
      }
      const pass = encoder.beginComputePass({ label: 'nrc resolve' });
      pass.setPipeline(this.resolvePipeline);
      pass.setBindGroup(0, this.resolveGroup);
      pass.dispatchWorkgroups(Math.ceil((displayOnly ? this.width * this.height : this.queryCapacity) / 64));
      if (!displayOnly && this.fixupPipeline && this.fixupGroup) {
        pass.setPipeline(this.fixupPipeline);
        pass.setBindGroup(0, this.fixupGroup);
        pass.dispatchWorkgroups(Math.ceil(this.trainPaths / 64));
      }
      pass.end();
    }
  }

  private encodeTargets(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: 'nrc targets' });
    pass.setPipeline(this.targetsPipeline);
    pass.setBindGroup(0, this.targetsGroup);
    pass.dispatchWorkgroups(Math.ceil((this.trainPaths * this.settings.trainDepth) / 64));
    pass.setPipeline(this.finishPipeline);
    pass.setBindGroup(0, this.finishGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  /** Starts a frame for `encodeStage` callers (the trainer's seed advances per frame). */
  nextFrame(): void {
    this.frame++;
  }

  /** Debug only (never in a frame loop): training records of the last frame, and render queries. */
  async readTrainCount(): Promise<{ batch: number; records: number }> {
    const v = new Uint32Array(await readU32s(this.device, this.trainCount, 2));
    return { batch: v[0] as number, records: v[1] as number };
  }

  /** Collision and optimizer diagnostics for one sampled training frame. */
  async readSpatialStats() {
    if (!this.spatial) throw new Error('Enable spatial features before requesting their diagnostics');
    const count = Math.min((await this.readTrainCount()).records, 8192);
    const stride = this.recordVec4 * 4;
    const positions = count ? new Float32Array(await readU32s(this.device, this.trainSet, count * stride)) : new Float32Array();
    return {
      records: count,
      updatedVertices: await this.spatial.readUpdatedVertices(),
      levels: spatialCollisions(positions, stride, this.normalizedBox, this.spatial.settings),
    };
  }

  /** Debug only (never in a frame loop): mean relative-L2 loss of the last training step over its batch (the frame's records). */
  async readLoss(): Promise<{ batch: number; loss: number }> {
    const { batch } = await this.readTrainCount();
    if (batch === 0 || !this.mlp) return { batch, loss: NaN };
    const n = Math.min(batch, this.mlp.loss.size / 4);
    const staging = this.device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.mlp.loss, 0, staging, 0, n * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(staging.getMappedRange());
    let sum = 0;
    for (let i = 0; i < n; i++) sum += v[i] as number;
    staging.unmap();
    staging.destroy();
    return { batch, loss: sum / n };
  }

  async readTerminationStats(): Promise<Record<string, number>> {
    if (!this.settings.terminationStats) throw new Error('enable terminationStats before requesting path diagnostics');
    const counts = new Uint32Array(await readU32s(this.device, this.queryCount, 11));
    const names = ['delta', 'glossy', 'rough'];
    const result: Record<string, number> = { display_depth_sum: counts[10]! };
    for (const [i, name] of names.entries()) {
      result['vertices_' + name] = counts[4 + i]!;
      result['display_queries_' + name] = counts[7 + i]!;
    }
    return result;
  }

  /** Borrow the current packed query records for an inference-backend comparison on the same frozen frame. */
  get replayBuffers() {
    if (!this.mlp || !this.renderInferGroup) throw new Error('Inference replay requires an MLP');
    return { queries: this.queries, box: this.box, predictions: this.preds,
      infer: (encoder: GPUCommandEncoder) => this.mlp!.encodeInfer(encoder, this.queryCount, 0, this.renderInferGroup!) };
  }

  async readQueryCount(): Promise<number> {
    return new Uint32Array(await readU32s(this.device, this.queryCount, 1))[0] as number;
  }

  destroy(): void {
    this.render.destroy();
    this.trainer?.destroy();
    this.mlp?.destroy();
    this.spatial?.destroy();
    this.hash?.destroy();
    for (const word of this.depthWords.values()) word.destroy();
    for (const word of this.displayOnlyWords.values()) word.destroy();
    for (const b of [this.hookParams, this.trainVerts, this.trainRadiance, this.tailPred, this.trainSet, this.trainCount, this.tailCount, this.queries, this.preds, this.queryCount, this.box, this.targetParams, this.resolveParams, this.fixupParams]) b?.destroy();
  }
}

async function readU32s(device: GPUDevice, buffer: GPUBuffer, n: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0, n * 4);
  staging.unmap();
  staging.destroy();
  return out;
}
