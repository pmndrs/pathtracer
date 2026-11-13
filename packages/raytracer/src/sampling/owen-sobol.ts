/**
 * Owen-scrambled Sobol (0,2)-sequence, hash-based, 2D. CPU reference of the WGSL version used by the
 * progressive reference mode (research/notes/04-sampling-reuse.md section 3.1).
 *
 * Construction after Burley, "Practical Hash-based Owen Scrambling" (JCGT 9(4), 2020): reverse the bits, apply a
 * Laine-Karras style hash permutation (each output bit depends only on the more significant input bits, which is
 * what a nested uniform scramble is), reverse again. Index shuffling for decorrelating dimension pairs uses the
 * same scramble on the sample index. Original code; only the construction and the four multiplier constants of
 * the Laine-Karras hash come from the paper.
 */

/** Reverse the 32 bits of `v`. */
export function reverseBits32(v: number): number {
  v = ((v >>> 1) & 0x55555555) | ((v & 0x55555555) << 1);
  v = ((v >>> 2) & 0x33333333) | ((v & 0x33333333) << 2);
  v = ((v >>> 4) & 0x0f0f0f0f) | ((v & 0x0f0f0f0f) << 4);
  v = ((v >>> 8) & 0x00ff00ff) | ((v & 0x00ff00ff) << 8);
  return ((v >>> 16) | (v << 16)) >>> 0;
}

/** Laine-Karras hash permutation, applied to bit-reversed input. */
export function laineKarras(x: number, seed: number): number {
  x = (x + seed) >>> 0;
  x = (x ^ Math.imul(x, 0x6c50b47c)) >>> 0;
  x = (x ^ Math.imul(x, 0xb82f1e52)) >>> 0;
  x = (x ^ Math.imul(x, 0xc7afe638)) >>> 0;
  x = (x ^ Math.imul(x, 0x8d22f6e6)) >>> 0;
  return x;
}

/** Nested uniform (Owen) scramble of a 32-bit fixed-point value. */
export function nestedUniformScramble(x: number, seed: number): number {
  return reverseBits32(laineKarras(reverseBits32(x), seed));
}

/** Sobol dimension 0 (van der Corput radical inverse base 2) as a 32-bit fixed-point value. */
export function sobolDim0(index: number): number {
  return reverseBits32(index);
}

/** Sobol dimension 1 as a 32-bit fixed-point value. */
export function sobolDim1(index: number): number {
  let v = 0x80000000;
  let r = 0;
  for (let i = index >>> 0; i; i >>>= 1, v ^= v >>> 1) if (i & 1) r ^= v;
  return r >>> 0;
}

/** 32-bit integer hash (lowbias32, Wellons); good enough to derive scramble seeds from pixel ids. */
export function hash32(x: number): number {
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

/**
 * Point `index` of the Owen-scrambled 2D Sobol sequence, written to `out[0..1]` in [0, 1).
 * `seed` selects the scramble (use a per-pixel hash). The first 2^k points for any k form a (0,k,2)-net.
 */
export function owenSobol2D(index: number, seed: number, out: Float64Array | number[] = [0, 0]): Float64Array | number[] {
  const s0 = hash32(seed);
  const s1 = hash32(seed ^ 0x9e3779b9);
  out[0] = nestedUniformScramble(sobolDim0(index), s0) / 4294967296;
  out[1] = nestedUniformScramble(sobolDim1(index), s1) / 4294967296;
  return out;
}

/**
 * Shuffled index for decorrelating dimension pairs of the same pixel: a nested random permutation of [0, 2^bits)
 * (aligned blocks map to aligned blocks), so the first 2^k shuffled indices still address one aligned block of the net.
 */
export function shuffleIndex(index: number, seed: number, bits: number): number {
  return nestedUniformScramble((index << (32 - bits)) >>> 0, seed) >>> (32 - bits);
}
