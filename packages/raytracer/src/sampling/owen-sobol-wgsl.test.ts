import { describe, expect, it } from 'vitest';
import { laineKarras } from './owen-sobol.ts';
import { rtOwenSobolWgsl } from './owen-sobol-wgsl.ts';

const wgsl = rtOwenSobolWgsl();

describe('Owen-Sobol WGSL twin (the GPU comparison is bench page pt-check)', () => {
  it('declares the public functions', () => {
    for (const name of ['rtHash32', 'rtLaineKarras', 'rtNestedUniformScramble', 'rtShuffleIndex', 'rtSobolDim1', 'rtOwenSobol2D', 'rtUnit24']) expect(wgsl).toContain(`fn ${name}(`);
  });

  it('uses the same multiplier constants as the TypeScript reference, in the same order', () => {
    // laineKarras in owen-sobol.ts: 0x6c50b47c, 0xb82f1e52, 0xc7afe638, 0x8d22f6e6; hash32: 0x7feb352d, 0x846ca68b.
    const order = [...wgsl.matchAll(/0x[0-9a-f]{8}u/g)].map((m) => m[0]);
    expect(order).toEqual(['0x7feb352du', '0x846ca68bu', '0x6c50b47cu', '0xb82f1e52u', '0xc7afe638u', '0x8d22f6e6u', '0x80000000u', '0x9e3779b9u']);
    expect(laineKarras(0, 0)).toBeGreaterThanOrEqual(0);
  });

  it('is comment-free with balanced braces, and rtUnit24 scales by 2^-24', () => {
    expect(wgsl).not.toContain('//');
    expect(wgsl.split('{').length).toBe(wgsl.split('}').length);
    expect(wgsl).toContain(`${2 ** -24}`.replace('5.960464477539063e-8', '5.9604645e-8'));
  });
});
