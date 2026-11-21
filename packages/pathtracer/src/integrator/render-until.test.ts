import { describe, expect, it } from 'vitest';
import { ptRenderUntil, ptStopBatches, ptStopMeasure, ptStopMet } from './render-until.ts';
import { ptDecodeVolumeOutputs, type PtVolumeOutputs } from '../volume/volume.ts';

/** A flat image of `n` pixels, radiance `m` in every channel, per-sample std `sd` (iid): se = sd / sqrt(N). */
function fake(n: number, m: number, sd: number, samples: number): PtVolumeOutputs {
  const se = sd / Math.sqrt(samples);
  return {
    width: n, height: 1, samples,
    radiance: new Float32Array(n * 3).fill(m), order01: new Float32Array(n * 3), transmittance: new Float32Array(n),
    stdError: new Float32Array(n * 3).fill(se), firstScatterDistance: new Float32Array(n), scatterFraction: new Float32Array(n),
  };
}

describe('ptStopBatches', () => {
  it('doubles the totals and clips at the cap', () => {
    expect(ptStopBatches(64, 1000)).toEqual([64, 64, 128, 256, 488]);
    expect(ptStopBatches(64, 64)).toEqual([64]);
  });
  it('counts samples already accumulated', () => {
    const b = ptStopBatches(8, 100, 50);
    expect(b).toEqual([8, 8, 16, 18]);
    expect(50 + b.reduce((a, x) => a + x, 0)).toBe(100);
  });
});

describe('ptStopMeasure', () => {
  it('image-mean relative SE of iid pixels is se / (m sqrt(n))', () => {
    const n = 100, m = 0.5, se = 0.01;
    const r = ptStopMeasure(fake(n, m, se, 1), {});
    // luminance: sum se_l^2 = n se^2 (weights sum to 1), sum mean = n m
    expect(r.meanRelSe).toBeCloseTo(Math.sqrt(n) * se / (n * m), 6);
  });
  it('pixel quantile ignores a few bad pixels below it and sees them above', () => {
    const o = fake(1000, 1, 1, 1);
    for (let i = 0; i < 5; i++) for (let c = 0; c < 3; c++) o.stdError[i * 3 + c] = 10;
    expect(ptStopMeasure(o, { pixelRelSe: { below: 1, quantile: 0.99 } }).pixelRelSe).toBeCloseTo(1, 5);
    expect(ptStopMeasure(o, { pixelRelSe: { below: 1, quantile: 0.999 } }).pixelRelSe).toBeCloseTo(10, 4);
  });
  it('the floor stops a black pixel from dividing by zero', () => {
    const o = fake(4, 0, 0.5, 1);
    expect(ptStopMeasure(o, { pixelRelSe: { below: 1, floor: 0.1 } }).pixelRelSe).toBeCloseTo(5, 5);
  });
  it('worst-channel takes the largest channel', () => {
    const o = fake(10, 1, 0.1, 1);
    for (let i = 0; i < 10; i++) o.stdError[i * 3 + 2] = 0.4;
    expect(ptStopMeasure(o, { measure: 'worst-channel' }).meanRelSe).toBeCloseTo(Math.sqrt(10) * 0.4 / 10, 6);
  });
  it('agrees with the decoder of the accumulator (se from the second moment)', () => {
    // two samples of 1 and 3 in every channel: mean 2, var 2 (N-1), se = sqrt(2/2) = 1
    const sums = new Float32Array(4 * 4);
    sums.set([4, 4, 4, 1], 0);
    sums.set([0, 0, 0, 0], 4);
    sums.set([10, 10, 10, 0], 8);
    const out = ptDecodeVolumeOutputs(sums, 1, 1, 2);
    expect(ptStopMeasure(out, {}).meanRelSe).toBeCloseTo(0.5, 6);
  });
  it('needs a target', () => {
    expect(() => ptStopMet({ meanRelSe: 0, pixelRelSe: NaN }, {})).toThrow();
  });
});

describe('ptRenderUntil', () => {
  const make = (sd: number) => {
    let total = 0;
    const runs: number[] = [];
    return {
      runs,
      driver: {
        run: async (n: number) => { runs.push(n); total += n; },
        read: async () => fake(256, 0.4, sd, total),
        now: () => 0,
      },
    };
  };
  it('stops at the first batch that meets the target; SE follows 1/sqrt(N)', async () => {
    const { driver, runs } = make(2);
    // se_mean(N) = sd / sqrt(N) / (m sqrt(n)) = 2 / sqrt(N) / (0.4 * 16) = 0.3125 / sqrt(N)
    const r = await ptRenderUntil(driver, { meanRelSe: 0.01 });
    expect(r.met).toBe(true);
    expect(r.reason).toBe('target');
    expect(r.samples).toBe(1024); // 64, 128, 256, 512, 1024: 0.3125/sqrt(1024) = 0.0098 < 0.01, 512 gives 0.0138
    expect(runs).toEqual([64, 64, 128, 256, 512]);
    expect(r.history.map((h) => h.samples)).toEqual([64, 128, 256, 512, 1024]);
    for (const h of r.history) expect(h.meanRelSe * Math.sqrt(h.samples)).toBeCloseTo(0.3125, 5);
    expect(r.achieved.meanRelSe).toBe(r.history[4]!.meanRelSe);
  });
  it('stops at the cap and says so', async () => {
    const { driver } = make(2);
    const r = await ptRenderUntil(driver, { meanRelSe: 1e-6, maxSamples: 300 });
    expect(r.met).toBe(false);
    expect(r.reason).toBe('cap');
    expect(r.samples).toBe(300);
    expect(r.history.map((h) => h.samples)).toEqual([64, 128, 256, 300]);
  });
  it('needs both targets when both are given', async () => {
    const { driver } = make(2);
    // mean target is met at 1024; the pixel target (se / m = 5 / sqrt(N) < 0.1) needs N > 2500
    const r = await ptRenderUntil(driver, { meanRelSe: 0.01, pixelRelSe: { below: 0.1 }, maxSamples: 8192 });
    expect(r.samples).toBe(4096);
    expect(r.met).toBe(true);
  });
  it('continues from samples already accumulated', async () => {
    let total = 100;
    const r = await ptRenderUntil({ run: async (n) => { total += n; }, read: async () => fake(256, 0.4, 2, total), now: () => 0 }, { meanRelSe: 1e-6, maxSamples: 200, firstBatch: 32 }, 100);
    expect(r.samples).toBe(200);
    expect(r.history.map((h) => h.samples)).toEqual([132, 164, 200]);
  });
  it('refuses a call without a target', async () => {
    await expect(ptRenderUntil(make(1).driver, {})).rejects.toThrow();
  });
});
