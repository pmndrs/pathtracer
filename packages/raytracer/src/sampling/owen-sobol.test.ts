import { describe, expect, it } from 'vitest';
import { nestedUniformScramble, owenSobol2D, reverseBits32, shuffleIndex, sobolDim0, sobolDim1 } from './owen-sobol.ts';

/** Every elementary interval of area 2^-m holds exactly one of the 2^m points: a (0, m, 2)-net in base 2. */
function isNet(pts: Array<[number, number]>, m: number): boolean {
  for (let a = 0; a <= m; a++) {
    const nx = 2 ** a;
    const ny = 2 ** (m - a);
    const seen = new Uint8Array(nx * ny);
    for (const [x, y] of pts) {
      const cell = Math.floor(y * ny) * nx + Math.floor(x * nx);
      if (seen[cell]) return false;
      seen[cell] = 1;
    }
  }
  return true;
}

describe('owen-scrambled sobol', () => {
  it('reverseBits32 is an involution', () => {
    for (const v of [0, 1, 0x80000000, 0x12345678, 0xffffffff]) expect(reverseBits32(reverseBits32(v))).toBe(v >>> 0);
  });

  it('unscrambled dims 0/1 are the Sobol (0,2)-sequence', () => {
    const pts: Array<[number, number]> = [];
    for (let i = 0; i < 256; i++) pts.push([sobolDim0(i) / 2 ** 32, sobolDim1(i) / 2 ** 32]);
    for (let m = 1; m <= 8; m++) expect(isNet(pts.slice(0, 2 ** m), m)).toBe(true);
    expect(sobolDim1(1)).toBe(0x80000000);
    expect(sobolDim1(2)).toBe(0xc0000000);
  });

  it('every prefix of 2^m scrambled points stays a (0,m,2)-net, for many seeds', () => {
    for (let seed = 0; seed < 20; seed++) {
      const pts: Array<[number, number]> = [];
      for (let i = 0; i < 256; i++) {
        const o = owenSobol2D(i, seed, [0, 0]);
        expect(o[0]).toBeGreaterThanOrEqual(0);
        expect(o[0]).toBeLessThan(1);
        pts.push([o[0]!, o[1]!]);
      }
      for (let m = 1; m <= 8; m++) expect(isNet(pts.slice(0, 2 ** m), m)).toBe(true);
    }
  });

  it('nested scramble preserves aligned dyadic blocks and is unbiased over seeds', () => {
    // Values sharing their top k bits map to values sharing their top k bits.
    for (let seed = 1; seed < 50; seed++) {
      const a = nestedUniformScramble(0x40000000, seed);
      const b = nestedUniformScramble(0x40001234, seed);
      expect(a >>> 18).toBe(b >>> 18);
    }
    let mean = 0;
    for (let seed = 0; seed < 4000; seed++) mean += nestedUniformScramble(0, seed * 7919) / 2 ** 32 / 4000;
    expect(Math.abs(mean - 0.5)).toBeLessThan(0.02);
  });

  it('shuffleIndex is a permutation of [0, 2^bits)', () => {
    for (const seed of [1, 99, 123456]) {
      const seen = new Set<number>();
      for (let i = 0; i < 64; i++) seen.add(shuffleIndex(i, seed, 6));
      expect(seen.size).toBe(64);
      expect(Math.max(...seen)).toBe(63);
    }
  });
});
