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

export type PtSamplerKind = 'owen-sobol' | 'pcg';

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
  return kind === 'pcg' ? ptSamplePcg(pixelSeed, sampleIndex, pair) : ptSampleOwenSobol(pixelSeed, sampleIndex, pair);
}

export { nestedUniformScramble };

/** WGSL for the sampler: needs `rtOwenSobolWgsl()` before it. `kind` picks which `ptSample2D` body is emitted. */
export function ptSamplerWgsl(kind: PtSamplerKind): string {
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
    if (PT_SAMPLER_PCG) {
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
`;
}
