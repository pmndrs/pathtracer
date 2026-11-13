import { describe, expect, it } from 'vitest';
import { analyzeTimestamps, gcd } from './timestamps.ts';

describe('analyzeTimestamps', () => {
  it('sees fine-grained timestamps as not quantized', () => {
    const base = 1_000_000_007n;
    const stamps = [base, base + 2_333n, base + 10_000n, base + 17_251n, base + 20_001n, base + 31_113n];
    const result = analyzeTimestamps(stamps);
    expect(result.stepNs).toBe(1n);
    expect(result.quantized).toBe(false);
    expect(result.durationsNs).toEqual([2333, 7251, 11112]);
    expect(result.zeroShare).toBe(0);
  });

  it('sees 100 us rounding as quantized, with zero-length passes', () => {
    const q = 100_000n;
    // Short passes round to the same value: durations 0, 100 us, 0.
    const stamps = [5n * q, 5n * q, 7n * q, 8n * q, 9n * q, 9n * q];
    const result = analyzeTimestamps(stamps);
    expect(result.stepNs).toBe(q);
    expect(result.quantized).toBe(true);
    expect(result.zeroShare).toBeCloseTo(2 / 3);
    expect(result.minNonzeroNs).toBe(100_000);
    expect(result.distinctDurations).toBe(2);
  });

  it('treats a dead timer (all equal) as unusable', () => {
    expect(analyzeTimestamps([5n, 5n, 5n, 5n]).quantized).toBe(true);
  });

  it('computes gcd', () => {
    expect(gcd(12n, 18n)).toBe(6n);
    expect(gcd(0n, 7n)).toBe(7n);
  });
});
