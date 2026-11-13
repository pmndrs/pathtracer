/**
 * WGSL twin of `owen-sobol.ts` (prefix `rt`): the same integer operations in the same order, so the GPU and the CPU
 * reference produce identical 32-bit fixed-point values (u32 multiplication wraps in WGSL exactly like `Math.imul`).
 *
 *     fn rtHash32(x: u32) -> u32                              lowbias32, TS `hash32`
 *     fn rtNestedUniformScramble(x: u32, seed: u32) -> u32    Owen scramble of a 32-bit fixed-point value
 *     fn rtShuffleIndex(index: u32, seed: u32, bits: u32) -> u32
 *     fn rtSobolDim1(index: u32) -> u32
 *     fn rtOwenSobol2D(index: u32, seed: u32) -> vec2u        fixed point; TS `owenSobol2D` * 2^32
 *     fn rtUnit24(x: u32) -> f32                              top 24 bits as a float in [0, 1), exact
 *
 * No comments in the generated text (kernel size cap).
 */
export function rtOwenSobolWgsl(): string {
  return /* wgsl */ `
fn rtHash32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn rtLaineKarras(x0: u32, seed: u32) -> u32 {
    var x = x0 + seed;
    x = x ^ (x * 0x6c50b47cu);
    x = x ^ (x * 0xb82f1e52u);
    x = x ^ (x * 0xc7afe638u);
    x = x ^ (x * 0x8d22f6e6u);
    return x;
}

fn rtNestedUniformScramble(x: u32, seed: u32) -> u32 {
    return reverseBits(rtLaineKarras(reverseBits(x), seed));
}

fn rtShuffleIndex(index: u32, seed: u32, bits: u32) -> u32 {
    return rtNestedUniformScramble(index << (32u - bits), seed) >> (32u - bits);
}

fn rtSobolDim1(index: u32) -> u32 {
    var v = 0x80000000u;
    var r = 0u;
    var i = index;
    loop {
        if (i == 0u) {
            break;
        }
        if ((i & 1u) != 0u) {
            r = r ^ v;
        }
        i = i >> 1u;
        v = v ^ (v >> 1u);
    }
    return r;
}

fn rtOwenSobol2D(index: u32, seed: u32) -> vec2u {
    let s0 = rtHash32(seed);
    let s1 = rtHash32(seed ^ 0x9e3779b9u);
    return vec2u(rtNestedUniformScramble(reverseBits(index), s0), rtNestedUniformScramble(rtSobolDim1(index), s1));
}

fn rtUnit24(x: u32) -> f32 {
    return f32(x >> 8u) * 5.9604645e-8;
}
`;
}
