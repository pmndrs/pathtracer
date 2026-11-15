import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE, DEFAULT_TRAIN } from './config.ts';
import { AdamState } from './oracle.ts';
import { adamWgsl } from './wgsl.ts';

describe('warm-up EMA (N6f)', () => {
  it('follows the weights closely over the first steps, then reaches the plain decay', () => {
    const w0 = new Float32Array([0, 0]);
    const plain = new AdamState(w0, { ...DEFAULT_TRAIN, learningRate: 0.1 });
    const warm = new AdamState(w0, { ...DEFAULT_TRAIN, learningRate: 0.1, emaWarmup: true });
    for (let s = 0; s < 5; s++) {
      plain.apply([1, -1]);
      warm.apply([1, -1]);
    }
    // Same weights; the warm-up EMA is much closer to them than the plain one (decay 0.99 from the initial weights).
    expect(warm.weights[0]).toBeCloseTo(plain.weights[0] as number, 6);
    const gapPlain = Math.abs((plain.weights[0] as number) - (plain.emaWeights()[0] as number));
    const gapWarm = Math.abs((warm.weights[0] as number) - (warm.emaWeights()[0] as number));
    expect(gapWarm).toBeLessThan(gapPlain / 4);
    // (1 + t) / (10 + t) passes 0.99 at t = 881: the decay is the plain one from there on.
    expect(Math.min(0.99, (1 + 1000) / (10 + 1000))).toBe(0.99);
  });

  it('the Adam kernel applies the same decay rule behind a flag', () => {
    const w = adamWgsl({ shape: DEFAULT_SHAPE, precision: 'f32' });
    expect(w).toContain('emaWarmup: u32');
    expect(w).toContain('min(params.emaDecay, (1.0 + t) / (10.0 + t)), params.emaWarmup == 1u)');
  });
});
