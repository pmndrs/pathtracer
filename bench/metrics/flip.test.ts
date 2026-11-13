import { describe, expect, it } from 'vitest';
import { flipLdr } from './flip.ts';
import { makeImage } from './pfm.ts';

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
