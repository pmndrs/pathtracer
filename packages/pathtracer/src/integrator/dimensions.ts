/**
 * The sampler's dimension map and its TypeScript twin (spec section 6: "documented dimension map").
 *
 * The reference sampler is Owen-scrambled Sobol with shuffled, independently scrambled 2D pairs (Burley 2020, the
 * construction in `@pmndrs/raytracer` `sampling/owen-sobol.ts`). One sample of one pixel consumes **pairs**; pair `k` of
 * sample `i` is the point `shuffle_k(i)` of an Owen-scrambled (0,2)-sequence with its own seed, so pairs are mutually
 * independent and each pair on its own is a (0,m,2)-net over the first 2^m samples of a pixel. A 1D request takes one
 * coordinate of a pair, which is itself a stratified uniform number.
 *
 *     pair 0                          pixel jitter (x right, y down), the box filter of radius 0.5
 *                                     (not consumed with `integrator.primary: 'vbuffer'`: the engine owns the jitter, `vbuffer.ts`; no other pair moves)
 *     pair 1                          reserved: lens / time
 *     pair 2 + 4 v + 0                vertex v: light point (u for the position on the chosen light / direction on a sun or environment)
 *     pair 2 + 4 v + 1                vertex v: BSDF direction (u1, u2 of `ptBsdfSample`)
 *     pair 2 + 4 v + 2                vertex v: x = light selection, y = BSDF lobe component (`uc` of `ptBsdfSample`)
 *     pair 2 + 4 v + 3                vertex v: x = Russian roulette, y reserved
 *
 * `v` is the zero-based scattering vertex (0 = first surface hit). A path with maxDepth 1000 reaches pair 4001: pairs are
 * hashed, not tabulated, so the depth costs nothing.
 *
 * Sample indices wrap their 16-bit shuffle every 65,536 samples of a pixel; the block number joins the scramble seed, so a
 * pixel never repeats a sequence (stratification is per block of 2^16, enough for any progressive run we do).
 *
 * The debug sampler (`pcg`) uses the same pair indices with a PCG hash of (pixel, sample, pair): independent, unstratified.
 */
import { hash32, nestedUniformScramble, owenSobol2D, shuffleIndex } from '@pmndrs/raytracer';

export const PT_PAIR = {
  jitter: 0,
  lens: 1,
  /** First pair of vertex 0. */
  vertexBase: 2,
  pairsPerVertex: 4,
} as const;

/** Slot of a pair inside a vertex. */
export const PT_VERTEX_SLOT = {
  lightPoint: 0,
  bsdfDirection: 1,
  /** x: light selection, y: BSDF component. */
  selectAndComponent: 2,
  /** x: Russian roulette. */
  russianRoulette: 3,
} as const;

/**
 * `'bluenoise'` (P4.1, real-time mode): the first `blueNoisePairs` pairs of a sample read our own jointly optimised blue-noise texture (D16, `gf-vec2`, `PtMegakernelOptions.blueNoise`)
 * at the pixel (plus a per-pair texel offset, so pairs are independent of each other), animated by a per-frame shift that is the same for every pixel (it keeps the spatial
 * structure and makes the sequence over frames equidistributed, so the accumulated mean is unbiased). Deeper pairs fall back to the Owen-Sobol pair. `sampleIndex` is the FRAME index.
 */
export type PtSamplerKind = 'owen-sobol' | 'pcg' | 'bluenoise';

/** Group 2 binding of the blue-noise texture (`rg32float`, `unfilterable-float`, textureLoad). Taken: 0-3, 9 (V-buffer), 10 (layer correction), 11 (ReSTIR reservoirs). */
export const PT_BLUENOISE_BINDING = 12;

/** Per-frame shift of a blue-noise pair: `owen` (default, P4.1 A/B) = the Owen-Sobol pair of the frame index with a seed shared by all pixels; `kronecker` = frame times an irrational vector (R2 for pair 0, sqrt of primes for the rest; the P0.17 baseline). */
export type PtBlueNoiseShift = 'kronecker' | 'owen';

export interface PtBlueNoiseOptions {
  /** How many leading pairs read the texture (default 6: pixel jitter, lens, and vertex 0's light point, BSDF direction, selection + component). At most 15. */
  pairs?: number;
  shift?: PtBlueNoiseShift;
}

export const PT_BLUENOISE_DEFAULT_PAIRS = 6;
export const PT_BLUENOISE_MAX_PAIRS = 15;
/** Seed shared by all pixels of the `owen` shift (it makes the shift pixel independent: the texture's spatial structure survives). */
export const PT_BLUENOISE_SHIFT_SEED = 0xb10e5eed;

/** Pair index of `slot` at scattering vertex `vertex` (0-based). */
export function ptPairIndex(vertex: number, slot: number): number {
  return PT_PAIR.vertexBase + vertex * PT_PAIR.pairsPerVertex + slot;
}

/** Highest pair index a path of `maxDepth` scattering vertices touches. */
export function ptMaxPair(maxDepth: number): number {
  return ptPairIndex(Math.max(0, maxDepth - 1), PT_VERTEX_SLOT.russianRoulette);
}

/** Bits of the sample index that are shuffled (the rest selects a block). */
export const PT_SHUFFLE_BITS = 16;

const u32 = (v: number): number => v >>> 0;

/** Per-pixel seed. TS twin of WGSL `ptPixelSeed`. */
export function ptPixelSeed(x: number, y: number, seedBase: number): number {
  return hash32(u32(x) ^ hash32(u32(y) ^ u32(seedBase)));
}

/** Seed of one pair of one sample block. */
export function ptPairSeed(pixelSeed: number, pair: number, block: number): number {
  return hash32(u32(pixelSeed ^ hash32(u32(Math.imul(pair, 0x9e3779b9) + 0x7f4a7c15 + Math.imul(block, 0x85ebca6b)))));
}

const UNIT24 = 2 ** -24;

/** The Owen-Sobol pair `pair` of sample `sampleIndex` of a pixel, in [0, 1)^2 with 24 bits (what the WGSL returns). */
export function ptSampleOwenSobol(pixelSeed: number, sampleIndex: number, pair: number): [number, number] {
  const block = sampleIndex >>> PT_SHUFFLE_BITS;
  const local = sampleIndex & ((1 << PT_SHUFFLE_BITS) - 1);
  const seed = ptPairSeed(pixelSeed, pair, block);
  const idx = shuffleIndex(local, seed, PT_SHUFFLE_BITS);
  const p = owenSobol2D(idx, u32(seed ^ 0x68bc21eb)) as [number, number];
  // owenSobol2D returns value / 2^32; the GPU keeps the top 24 bits.
  return [Math.floor(p[0] * 2 ** 24) * UNIT24, Math.floor(p[1] * 2 ** 24) * UNIT24];
}

/** PCG hash (Jarzynski & Olano, "pcg" output permutation of a 32-bit LCG state). */
export function pcgHash(v: number): number {
  const state = u32(Math.imul(v, 747796405) + 2891336453);
  const word = u32(Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737));
  return u32((word >>> 22) ^ word);
}

/** Debug sampler: independent uniform pair from a PCG hash chain over (pixel, sample, pair). */
export function ptSamplePcg(pixelSeed: number, sampleIndex: number, pair: number): [number, number] {
  const a = pcgHash(u32(pixelSeed ^ pcgHash(u32(sampleIndex + Math.imul(pair, 0x9e3779b9)))));
  const b = pcgHash(a);
  return [(a >>> 8) * UNIT24, (b >>> 8) * UNIT24];
}

export function ptSample2D(kind: PtSamplerKind, pixelSeed: number, sampleIndex: number, pair: number): [number, number] {
  if (kind === 'bluenoise') throw new RangeError('ptSample2D: the blue-noise sampler needs the texture and the pixel: use ptSampleBlueNoise');
  return kind === 'pcg' ? ptSamplePcg(pixelSeed, sampleIndex, pair) : ptSampleOwenSobol(pixelSeed, sampleIndex, pair);
}

export { nestedUniformScramble };

const R2 = [0.7548776662466927, 0.5698402909980532] as const;
const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109] as const;

/** Fixed-point (2^32) per-frame step of pair `pair`, component c: R2 for pair 0, fract(sqrt(prime)) for the others (rationally independent, so pairs do not move along a diagonal). */
export function ptBlueNoiseAlpha(pair: number): [number, number] {
  if (pair === 0) return [Math.floor(R2[0] * 2 ** 32), Math.floor(R2[1] * 2 ** 32)];
  const f = (p: number): number => Math.floor((Math.sqrt(p) % 1) * 2 ** 32);
  return [f(PRIMES[2 * (pair - 1)] as number), f(PRIMES[2 * (pair - 1) + 1] as number)];
}

/** Texel offset of a pair (a hash, so each pair reads other texels of the same texture). */
export function ptBlueNoiseOffset(pair: number): [number, number] {
  const h = hash32(u32(Math.imul(pair, 0x9e3779b9) + 0x7f4a7c15));
  return [h & 0xffff, h >>> 16];
}

/** The per-frame shift of a pair in [0, 1)^2, 24 bits. TS twin of WGSL `ptBnShift`. */
export function ptBlueNoiseShift(shift: PtBlueNoiseShift, pair: number, frame: number): [number, number] {
  if (shift === 'owen') return ptSampleOwenSobol(PT_BLUENOISE_SHIFT_SEED, frame, pair);
  const [ax, ay] = ptBlueNoiseAlpha(pair);
  return [(Math.imul(frame, ax) >>> 8) * UNIT24, (Math.imul(frame, ay) >>> 8) * UNIT24];
}

/** A blue-noise texture as the sampler sees it: vec2 texels, `width` x `height` (powers of two), row-major, interleaved. */
export interface PtBlueNoiseData {
  width: number;
  height: number;
  data: Float32Array;
}

/**
 * The blue-noise pair `pair` of frame `frame` at pixel (x, y), TS twin of `ptSample2D` for `pair < pairs` (f32 arithmetic of the final add, so it matches the WGSL to ~1e-7).
 * Pairs at or past `pairs` are the Owen-Sobol pair (`ptSampleOwenSobol(pixelSeed, frame, pair)`).
 */
export function ptSampleBlueNoise(tex: PtBlueNoiseData, x: number, y: number, frame: number, pair: number, options: PtBlueNoiseOptions = {}, pixelSeed = 0): [number, number] {
  if (pair >= (options.pairs ?? PT_BLUENOISE_DEFAULT_PAIRS)) return ptSampleOwenSobol(pixelSeed, frame, pair);
  const [ox, oy] = ptBlueNoiseOffset(pair);
  const cx = (x + ox) & (tex.width - 1);
  const cy = (y + oy) & (tex.height - 1);
  const i = (cy * tex.width + cx) * 2;
  const [sx, sy] = ptBlueNoiseShift(options.shift ?? 'owen', pair, frame);
  const fr = (v: number): number => {
    const t = Math.fround(v);
    return t - Math.floor(t);
  };
  return [fr(Math.fround((tex.data[i] as number) + sx)), fr(Math.fround((tex.data[i + 1] as number) + sy))];
}

function blueNoiseWgsl(options: PtBlueNoiseOptions): string {
  const pairs = Math.min(PT_BLUENOISE_MAX_PAIRS, Math.max(1, Math.floor(options.pairs ?? PT_BLUENOISE_DEFAULT_PAIRS)));
  const owen = options.shift !== 'kronecker';
  const alphas = Array.from({ length: pairs }, (_, k) => ptBlueNoiseAlpha(k)).flat();
  return /* wgsl */ `
@group(2) @binding(${PT_BLUENOISE_BINDING}) var ptBlueNoise: texture_2d<f32>;
var<private> ptBnPixel: vec2u;
const PT_BN_PAIRS: u32 = ${pairs}u;
const PT_BN_OWEN: bool = ${owen};
${owen ? '' : `var<private> PT_BN_ALPHA: array<u32, ${alphas.length}> = array<u32, ${alphas.length}>(${alphas.map((a) => `${a}u`).join(', ')});
`}
fn ptBnShift(pair: u32, frame: u32) -> vec2f {
    if (PT_BN_OWEN) {
        let block = frame >> PT_SHUFFLE_BITS;
        let local = frame & ((1u << PT_SHUFFLE_BITS) - 1u);
        let seed = rtHash32(${PT_BLUENOISE_SHIFT_SEED}u ^ rtHash32(pair * 0x9e3779b9u + 0x7f4a7c15u + block * 0x85ebca6bu));
        let idx = rtShuffleIndex(local, seed, PT_SHUFFLE_BITS);
        let p = rtOwenSobol2D(idx, seed ^ 0x68bc21ebu);
        return vec2f(rtUnit24(p.x), rtUnit24(p.y));
    }
${owen ? '    return vec2f(0.0);' : `    let s = vec2u(PT_BN_ALPHA[2u * pair], PT_BN_ALPHA[2u * pair + 1u]) * frame;
    return vec2f(f32(s.x >> 8u), f32(s.y >> 8u)) * 5.9604644775390625e-8;`}
}
`;
}

/** WGSL for the sampler: needs `rtOwenSobolWgsl()` before it. `kind` picks which `ptSample2D` body is emitted. */
export function ptSamplerWgsl(kind: PtSamplerKind, blueNoise: PtBlueNoiseOptions = {}): string {
  const bn = kind === 'bluenoise';
  return /* wgsl */ `
const PT_PAIR_JITTER: u32 = ${PT_PAIR.jitter}u;
const PT_PAIR_LENS: u32 = ${PT_PAIR.lens}u;
const PT_PAIR_VERTEX_BASE: u32 = ${PT_PAIR.vertexBase}u;
const PT_PAIRS_PER_VERTEX: u32 = ${PT_PAIR.pairsPerVertex}u;
const PT_SLOT_LIGHT_POINT: u32 = ${PT_VERTEX_SLOT.lightPoint}u;
const PT_SLOT_BSDF_DIRECTION: u32 = ${PT_VERTEX_SLOT.bsdfDirection}u;
const PT_SLOT_SELECT_COMPONENT: u32 = ${PT_VERTEX_SLOT.selectAndComponent}u;
const PT_SLOT_RR: u32 = ${PT_VERTEX_SLOT.russianRoulette}u;
const PT_SHUFFLE_BITS: u32 = ${PT_SHUFFLE_BITS}u;
const PT_SAMPLER_PCG: bool = ${kind === 'pcg'};

fn ptPixelSeed(x: u32, y: u32, seedBase: u32) -> u32 {
    return rtHash32(x ^ rtHash32(y ^ seedBase));
}

fn ptPcgHash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn ptSample2D(pixelSeed: u32, sampleIndex: u32, pair: u32) -> vec2f {
${bn ? `    if (pair < PT_BN_PAIRS) {
        let dim = textureDimensions(ptBlueNoise);
        let h = rtHash32(pair * 0x9e3779b9u + 0x7f4a7c15u);
        let c = vec2u(ptBnPixel.x + (h & 0xffffu), ptBnPixel.y + (h >> 16u)) & (dim - vec2u(1u));
        return fract(textureLoad(ptBlueNoise, vec2i(c), 0).xy + ptBnShift(pair, sampleIndex));
    }
` : ''}    if (PT_SAMPLER_PCG) {
        let a = ptPcgHash(pixelSeed ^ ptPcgHash(sampleIndex + pair * 0x9e3779b9u));
        let b = ptPcgHash(a);
        return vec2f(rtUnit24(a), rtUnit24(b));
    }
    let block = sampleIndex >> PT_SHUFFLE_BITS;
    let local = sampleIndex & ((1u << PT_SHUFFLE_BITS) - 1u);
    let seed = rtHash32(pixelSeed ^ rtHash32(pair * 0x9e3779b9u + 0x7f4a7c15u + block * 0x85ebca6bu));
    let idx = rtShuffleIndex(local, seed, PT_SHUFFLE_BITS);
    let p = rtOwenSobol2D(idx, seed ^ 0x68bc21ebu);
    return vec2f(rtUnit24(p.x), rtUnit24(p.y));
}

fn ptVertexPair(vertex: u32, slot: u32) -> u32 {
    return PT_PAIR_VERTEX_BASE + vertex * PT_PAIRS_PER_VERTEX + slot;
}
${bn ? blueNoiseWgsl(blueNoise) : ''}`;
}
