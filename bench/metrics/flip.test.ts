import { describe, expect, it } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import { flipHdr, flipLdr, hdrFlipExposures, hdrFlipLdr } from './flip.ts';
import { makeImage, readPFM } from './pfm.ts';

const flat = (w: number, h: number, v: number) => makeImage(w, h, () => [v, v, v]);

describe('LDR-FLIP', () => {
  it('identical images -> 0', () => {
    const a = makeImage(24, 24, (x, y) => [x / 24, y / 24, 0.3]);
    expect(flipLdr(a, a, { tonemap: 'clamp' }).mean).toBe(0);
  });

  it('black vs white (uniform): hand value dE_c ~ 0.968 (HyAB 100^0.7 redistributed against cmax ~ 40.7), no features', () => {
    const r = flipLdr(flat(16, 16, 1), flat(16, 16, 0), { tonemap: 'clamp' });
    expect(r.mean).toBeCloseTo(0.968, 2);
    expect(r.map.length).toBe(256);
    expect(r.map[0]).toBeCloseTo(r.map[255]!, 6);
  });

  it('is monotone in the size of a uniform difference and symmetric for uniform fields', () => {
    const ref = flat(16, 16, 0.5);
    const small = flipLdr(flat(16, 16, 0.55), ref, { tonemap: 'clamp' }).mean;
    const big = flipLdr(flat(16, 16, 0.8), ref, { tonemap: 'clamp' }).mean;
    expect(small).toBeGreaterThan(0);
    expect(big).toBeGreaterThan(small);
    const fwd = flipLdr(flat(16, 16, 0.8), ref, { tonemap: 'clamp' }).mean;
    const bwd = flipLdr(ref, flat(16, 16, 0.8), { tonemap: 'clamp' }).mean;
    expect(fwd).toBeCloseTo(bwd, 6);
  });

  it('an isolated bright pixel is localised in the map', () => {
    const ref = flat(32, 32, 0.2);
    const test = makeImage(32, 32, (x, y) => (x === 16 && y === 16 ? [1, 1, 1] : [0.2, 0.2, 0.2]));
    const { map } = flipLdr(test, ref, { tonemap: 'clamp' });
    expect(map[16 * 32 + 16]!).toBeGreaterThan(map[2 * 32 + 2]!);
    expect(map[2 * 32 + 2]!).toBeCloseTo(0, 3);
  });
});

describe('HDR-FLIP', () => {
  const hdr = (w: number, h: number, f: (x: number, y: number) => [number, number, number]) => makeImage(w, h, f);

  it('identical images -> 0 and the exposure range follows the reference (ACES: xMax solves tonemap = 0.85)', () => {
    const a = hdr(24, 24, (x, y) => [0.1 + x, 0.2 + y, 0.05]);
    const r = flipHdr(a, a);
    expect(r.mean).toBe(0);
    const plan = hdrFlipExposures(a, 'aces');
    // brightest luminance ~ 23.2 -> start = log2(xMax / Ymax); stop uses the median
    expect(plan.start).toBeLessThan(plan.stop);
    expect(plan.count).toBe(Math.max(2, Math.ceil(plan.stop - plan.start)));
    expect(r.count).toBe(plan.count);
  });

  it('the first exposure puts the reference maximum at 0.85 display value (definition of start)', () => {
    const a = hdr(8, 8, (x) => [x === 7 ? 30 : 1, x === 7 ? 30 : 1, x === 7 ? 30 : 1]);
    const { start } = hdrFlipExposures(a, 'aces');
    const out = hdrFlipLdr(a, start, 'aces');
    expect(out[3 * 7]).toBeCloseTo(0.85, 3);
  });

  it('a firefly on dark content is caught at the low exposures, but only if the REFERENCE spans them', () => {
    // The exposure range comes from the reference alone. A flat dark reference gives start == stop (2 identical
    // exposures at which a 500 firefly merely saturates, error ~0.07); a reference with a lamp reaches down to the
    // exposure at which 500 is still white while 0.05 is black.
    const flatRef = hdr(32, 32, () => [0.05, 0.05, 0.05]);
    const lampRef = hdr(32, 32, (x, y) => (x === 0 && y === 0 ? [200, 200, 200] : [0.05, 0.05, 0.05]));
    const test = (r: typeof flatRef) => hdr(32, 32, (x, y) => (x === 16 && y === 16 ? [500, 500, 500] : r.data[(y * 32 + x) * 3]! === 200 ? [200, 200, 200] : [0.05, 0.05, 0.05]));
    const flat = flipHdr(test(flatRef), flatRef);
    const lamp = flipHdr(test(lampRef), lampRef);
    expect(flat.count).toBe(2);
    expect(flat.map[16 * 32 + 16]!).toBeLessThan(0.2);
    expect(lamp.map[16 * 32 + 16]!).toBeGreaterThan(0.5);
    expect(lamp.map[2 * 32 + 2]!).toBeCloseTo(0, 3);
  });

  it('non-finite and negative input count as 0 instead of poisoning the mean', () => {
    const ref = hdr(16, 16, () => [0.5, 0.5, 0.5]);
    const test = hdr(16, 16, (x, y) => (x === 3 && y === 3 ? [NaN, -1, Infinity] : [0.5, 0.5, 0.5]));
    expect(Number.isFinite(flipHdr(test, ref).mean)).toBe(true);
  });
});

// Agreement with NVlabs flip 1.7 (pip flip-evaluator) on their own images; needs .cache/flip-validation (git-ignored,
// created by the P0.14 steps in DIVERGES V6) and ~45 s of CPU, so it only runs with FLIP_VALIDATE=1
// (`FLIP_VALIDATE=1 pnpm exec vitest run bench/metrics/flip.test.ts`); `node bench/metrics/validate-flip.ts` prints the same numbers.
const nv = '.cache/flip-validation/work';
describe.skipIf(!process.env.FLIP_VALIDATE || !existsSync(`${nv}/nv_hdr_map.pfm`))('NVlabs flip reference outputs', () => {
  const load = (n: string) => readPFM(readFileSync(`${nv}/${n}`));
  it('LDR-FLIP with the sRGB decode switched on matches their mean (0.159693) and per-pixel map', () => {
    const r = flipLdr(load('test.png.pfm'), load('reference.png.pfm'), { tonemap: 'clamp', srgbDecode: true });
    expect(Math.abs(r.mean - 0.159693)).toBeLessThan(1e-5);
    const theirs = load('nv_ldr_map.pfm');
    let max = 0;
    for (let i = 0; i < r.map.length; i++) max = Math.max(max, Math.abs(r.map[i]! - theirs.data[3 * i]!));
    expect(max).toBeLessThan(1e-4);
  }, 60_000);
  it('HDR-FLIP matches their mean (0.283547), exposure range (-12.5423..0.9427, 14) and per-pixel map', () => {
    const ref = load('reference.exr.pfm');
    const plan = hdrFlipExposures(ref, 'aces');
    expect(plan.start).toBeCloseTo(-12.542346, 3);
    expect(plan.stop).toBeCloseTo(0.9427484, 3);
    expect(plan.count).toBe(14);
    const r = flipHdr(load('test.exr.pfm'), ref);
    expect(Math.abs(r.mean - 0.283547)).toBeLessThan(1e-5);
    const theirs = load('nv_hdr_map.pfm');
    let max = 0;
    for (let i = 0; i < r.map.length; i++) max = Math.max(max, Math.abs(r.map[i]! - theirs.data[3 * i]!));
    expect(max).toBeLessThan(1e-4);
  }, 120_000);
});
