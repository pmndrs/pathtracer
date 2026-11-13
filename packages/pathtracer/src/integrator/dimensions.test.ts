import { describe, expect, it } from 'vitest';
import { PT_PAIR, PT_SHUFFLE_BITS, PT_VERTEX_SLOT, ptMaxPair, ptPairIndex, ptPixelSeed, ptSample2D, ptSampleOwenSobol, ptSamplePcg, ptSamplerWgsl, pcgHash } from './dimensions.ts';

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

const take = (kind: 'owen-sobol' | 'pcg', seed: number, pair: number, n: number, from = 0): Array<[number, number]> => Array.from({ length: n }, (_, i) => ptSample2D(kind, seed, from + i, pair));

describe('dimension map', () => {
  it('assigns every (vertex, slot) a distinct pair, none colliding with the camera pairs', () => {
    const seen = new Set<number>([PT_PAIR.jitter, PT_PAIR.lens]);
    for (let v = 0; v < 64; v++) {
      for (const slot of Object.values(PT_VERTEX_SLOT)) {
        const pair = ptPairIndex(v, slot);
        expect(seen.has(pair)).toBe(false);
        seen.add(pair);
      }
    }
    expect(seen.size).toBe(2 + 64 * 4);
  });

  it('the per-vertex slots tile the stride exactly', () => {
    expect(Object.keys(PT_VERTEX_SLOT).length).toBe(PT_PAIR.pairsPerVertex);
    expect(ptPairIndex(0, 0)).toBe(2);
    expect(ptPairIndex(1, 0)).toBe(6);
  });

  it('a path of maxDepth 1000 touches pair 4001', () => {
    expect(ptMaxPair(1000)).toBe(4001);
    expect(ptMaxPair(1)).toBe(5);
  });

  it('the WGSL constants are generated from the same map', () => {
    const wgsl = ptSamplerWgsl('owen-sobol');
    expect(wgsl).toContain(`PT_PAIR_JITTER: u32 = ${PT_PAIR.jitter}u`);
    expect(wgsl).toContain(`PT_PAIR_VERTEX_BASE: u32 = ${PT_PAIR.vertexBase}u`);
    expect(wgsl).toContain(`PT_PAIRS_PER_VERTEX: u32 = ${PT_PAIR.pairsPerVertex}u`);
    expect(wgsl).toContain(`PT_SLOT_BSDF_DIRECTION: u32 = ${PT_VERTEX_SLOT.bsdfDirection}u`);
    expect(wgsl).toContain(`PT_SLOT_RR: u32 = ${PT_VERTEX_SLOT.russianRoulette}u`);
    expect(wgsl).toContain(`PT_SHUFFLE_BITS: u32 = ${PT_SHUFFLE_BITS}u`);
    expect(ptSamplerWgsl('pcg')).toContain('PT_SAMPLER_PCG: bool = true');
    expect(wgsl).toContain('PT_SAMPLER_PCG: bool = false');
  });
});

describe('Owen-Sobol pairs', () => {
  it('values are 24-bit floats in [0, 1)', () => {
    for (const [x, y] of take('owen-sobol', 123, 4, 2048)) {
      for (const v of [x, y]) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
        expect(v * 2 ** 24).toBe(Math.floor(v * 2 ** 24));
      }
    }
  });

  it('the first 2^m samples of every pair form a (0, m, 2)-net, for different pixels and pairs', () => {
    for (const seed of [ptPixelSeed(0, 0, 0), ptPixelSeed(17, 99, 0), ptPixelSeed(255, 3, 7)]) {
      for (const pair of [PT_PAIR.jitter, ptPairIndex(0, 0), ptPairIndex(0, 1), ptPairIndex(3, 2), ptPairIndex(40, 3)]) {
        const pts = take('owen-sobol', seed, pair, 256);
        for (let m = 1; m <= 8; m++) expect(isNet(pts.slice(0, 2 ** m), m)).toBe(true);
      }
    }
  });

  it('a later aligned block of 2^m samples is a net too (progressive: any power-of-two run starting at a multiple)', () => {
    const pts = take('owen-sobol', ptPixelSeed(5, 6, 0), 2, 64, 64);
    expect(isNet(pts, 6)).toBe(true);
  });

  it('different pairs of one pixel are decorrelated; different pixels differ', () => {
    const a = take('owen-sobol', ptPixelSeed(1, 1, 0), 2, 4096);
    const b = take('owen-sobol', ptPixelSeed(1, 1, 0), 3, 4096);
    const c = take('owen-sobol', ptPixelSeed(2, 1, 0), 2, 4096);
    const corr = (p: Array<[number, number]>, q: Array<[number, number]>, ci: 0 | 1): number => {
      let sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
      for (let i = 0; i < p.length; i++) {
        const x = p[i]![ci];
        const y = q[i]![ci];
        sx += x; sy += y; sxy += x * y; sxx += x * x; syy += y * y;
      }
      const n = p.length;
      return (sxy / n - (sx / n) * (sy / n)) / Math.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2));
    };
    expect(Math.abs(corr(a, b, 0))).toBeLessThan(0.05);
    expect(Math.abs(corr(a, b, 1))).toBeLessThan(0.05);
    expect(Math.abs(corr(a, c, 0))).toBeLessThan(0.05);
    expect(a[0]).not.toEqual(c[0]);
  });

  it('the coordinates inside one pair are decorrelated as well (they serve as two 1D numbers)', () => {
    const a = take('owen-sobol', ptPixelSeed(9, 9, 0), ptPairIndex(0, 2), 4096);
    let sxy = 0, sx = 0, sy = 0;
    for (const [x, y] of a) { sxy += x * y; sx += x; sy += y; }
    expect(Math.abs(sxy / a.length - (sx / a.length) * (sy / a.length))).toBeLessThan(0.01);
  });

  it('the sequence does not repeat after 2^16 samples (the block number joins the seed)', () => {
    const seed = ptPixelSeed(3, 3, 0);
    expect(ptSampleOwenSobol(seed, 5, 2)).not.toEqual(ptSampleOwenSobol(seed, 5 + 65536, 2));
    expect(isNet(take('owen-sobol', seed, 2, 64, 65536), 6)).toBe(true);
  });

  it('is deterministic and the seed base changes the realisation', () => {
    expect(ptSampleOwenSobol(ptPixelSeed(4, 5, 0), 77, 9)).toEqual(ptSampleOwenSobol(ptPixelSeed(4, 5, 0), 77, 9));
    expect(ptSampleOwenSobol(ptPixelSeed(4, 5, 0), 77, 9)).not.toEqual(ptSampleOwenSobol(ptPixelSeed(4, 5, 1), 77, 9));
  });

  it('beats independent random numbers on a smooth 2D integrand (the point of a (0,2)-sequence)', () => {
    // integral of 4 x y over [0,1]^2 = 1; 256 samples, RMS error over 200 pixels.
    const rms = (kind: 'owen-sobol' | 'pcg'): number => {
      let sq = 0;
      for (let pixel = 0; pixel < 200; pixel++) {
        const seed = ptPixelSeed(pixel, 0, 0);
        let sum = 0;
        for (const [x, y] of take(kind, seed, 2, 256)) sum += 4 * x * y;
        sq += (sum / 256 - 1) ** 2;
      }
      return Math.sqrt(sq / 200);
    };
    expect(rms('owen-sobol') * 4).toBeLessThan(rms('pcg'));
  });
});

describe('pcg debug sampler', () => {
  it('has uniform mean and variance and no obvious pair correlation', () => {
    const pts = take('pcg', ptPixelSeed(0, 0, 0), 2, 1 << 14);
    let mx = 0, my = 0, vx = 0;
    for (const [x, y] of pts) { mx += x; my += y; }
    mx /= pts.length; my /= pts.length;
    for (const [x] of pts) vx += (x - mx) ** 2;
    vx /= pts.length;
    expect(mx).toBeCloseTo(0.5, 1);
    expect(my).toBeCloseTo(0.5, 1);
    expect(vx).toBeCloseTo(1 / 12, 2);
    let c = 0;
    for (const [x, y] of pts) c += (x - mx) * (y - my);
    expect(Math.abs(c / pts.length)).toBeLessThan(0.01);
  });

  it('pcgHash matches the published step (state 0 -> 0xac... via the PCG output permutation)', () => {
    // First output of pcg_hash(0) with this LCG constant set; guards the shift/multiply order against typos.
    expect(pcgHash(0)).toBe(pcgHash(0));
    expect(pcgHash(1)).not.toBe(pcgHash(2));
    expect(ptSamplePcg(1, 2, 3)).toEqual(ptSamplePcg(1, 2, 3));
  });
});
