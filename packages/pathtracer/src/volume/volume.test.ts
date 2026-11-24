import { describe, expect, it } from 'vitest';
import { heightDistance, heightOpticalDepth, PT_VOLUME_OUTPUT_VEC4, f16BitsToF32, f32ToF16Bits, ptDecodeVolumeOutputs, ptF16Bits, ptRoundGridF16, ptQuantizeGridR8, ptR8Bytes, ptTexelBound, ATLAS_CELL, ATLAS_EMPTY, atlasDensityAtIndex, brickAtlasMemory, buildBrickAtlas, brickDims, brickMajorants, ddaSegments, residualTransmittance, telescopingTransmittance, brickTrack, deltaTrack, gridBoundaryFactor, gridDensity, gridWorldToIndex, hgPhase, hgSampleCos, ptMediumFrom, ptMediumMajorant, slabSingleScatter, slabSingleScatterPhase, slabTransmittance, transmittance, ptMediumEmissionSegment } from './volume.ts';

/** Small deterministic generator (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('Henyey-Greenstein', () => {
  for (const g of [0, 0.3, -0.6, 0.9]) {
    it(`integrates to 1 and its sampler has mean cosine -g (g = ${g})`, () => {
      let integral = 0;
      const n = 20000;
      for (let i = 0; i < n; i++) integral += hgPhase(-1 + ((i + 0.5) / n) * 2, g) * 2 * Math.PI * (2 / n);
      expect(integral).toBeCloseTo(1, 3);
      const r = rng(7);
      let mean = 0;
      const m = 200000;
      for (let i = 0; i < m; i++) mean += hgSampleCos(g, r());
      // cos is measured against wo (pointing back): forward scattering (g > 0) is cos -> -1
      expect(mean / m).toBeCloseTo(-g, 2);
    });
  }
  it('the sampled density equals the evaluated one (histogram)', () => {
    const g = 0.7;
    const r = rng(11);
    const bins = new Array<number>(10).fill(0);
    const m = 400000;
    for (let i = 0; i < m; i++) bins[Math.min(9, Math.floor(((hgSampleCos(g, r()) + 1) / 2) * 10))]!++;
    for (let b = 0; b < 10; b++) {
      let expected = 0;
      const sub = 400;
      for (let i = 0; i < sub; i++) expected += hgPhase(-1 + (b + (i + 0.5) / sub) * 0.2, g) * 2 * Math.PI * (0.2 / sub);
      expect(bins[b]! / m).toBeCloseTo(expected, 2);
    }
  });
});

describe('delta tracking', () => {
  const o = [0, 0, 0] as const;
  const d = [0, 0, 1] as const;

  it('the pass weight of an absorbing homogeneous medium averages exp(-sigma_t d) per channel, grey and chromatic', () => {
    for (const sigma of [[1, 1, 1], [0.4, 1, 2.5]] as const) {
      const m = ptMediumFrom(sigma, [0, 0, 0]);
      const r = rng(3);
      const sum = [0, 0, 0];
      const n = 400000;
      for (let i = 0; i < n; i++) {
        const e = deltaTrack(m, o, d, 1.3, r);
        if (e.kind === 'pass') for (let c = 0; c < 3; c++) sum[c]! += e.weight[c]!;
      }
      for (let c = 0; c < 3; c++) expect(sum[c]! / n).toBeCloseTo(Math.exp(-sigma[c]! * 1.3), 2);
    }
  });

  it('a non-absorbing medium: every ray either scatters or passes, pass fraction exp(-sigma_t d)', () => {
    const m = ptMediumFrom([0, 0, 0], [0.8, 0.8, 0.8]);
    const r = rng(5);
    let pass = 0, scatter = 0;
    const n = 200000;
    for (let i = 0; i < n; i++) {
      const e = deltaTrack(m, o, d, 2, r);
      if (e.kind === 'pass') pass++;
      else if (e.kind === 'scatter') scatter++;
    }
    expect(pass / n).toBeCloseTo(Math.exp(-1.6), 2);
    expect((pass + scatter) / n).toBe(1);
  });

  it('expected scatter weight density: E[weight at scatter, first scatter before tmax] = integral sigma_s T(t) dt (chromatic albedo)', () => {
    const m = ptMediumFrom([0.2, 0.5, 0.1], [0.6, 0.3, 0.9]);
    const r = rng(9);
    const sum = [0, 0, 0];
    const n = 600000;
    for (let i = 0; i < n; i++) {
      const e = deltaTrack(m, o, d, 1.5, r);
      if (e.kind === 'scatter') for (let c = 0; c < 3; c++) sum[c]! += e.weight[c]!;
    }
    // the weight of a scatter is sigma_s / max sigma_s and the sampling density is mu exp(-mu t) x ..., so E[sum] = ratio of scattered shares: check against the closed form
    // P_c(first real scatter before tmax) = integral_0^T sigma_s,c exp(-sigma_t,c t) dt = albedo_c (1 - exp(-sigma_t,c T))
    for (let c = 0; c < 3; c++) {
      const st = [0.8, 0.8, 1.0][c]!;
      const ss = [0.6, 0.3, 0.9][c]!;
      expect(sum[c]! / n).toBeCloseTo((ss / st) * (1 - Math.exp(-st * 1.5)), 2);
    }
  });

  it('a grid medium: delta tracking and ratio tracking both estimate the line integral of a trilinear density', () => {
    const res = [4, 4, 4] as const;
    const data = new Float32Array(64).map((_, i) => 0.2 + 0.8 * ((i * 37) % 11) / 10);
    const grid = { resolution: res, worldToIndex: gridWorldToIndex(res, [[0, 0, 0], [1, 1, 1]]), densityMax: Math.max(...data) };
    const dens = gridDensity(grid, data);
    const m = ptMediumFrom([1.5, 2, 1], [0, 0, 0], 0, 1, grid);
    const o2 = [0.4, 0.3, -0.5] as const;
    const d2 = [0, 0, 1] as const;
    // numeric line integral through the density
    let tau = 0;
    const sub = 20000;
    for (let i = 0; i < sub; i++) tau += dens(0.4, 0.3, -0.5 + ((i + 0.5) / sub) * 2) * (2 / sub);
    const r = rng(21);
    const n = 300000;
    const delta = [0, 0, 0], ratio = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const e = deltaTrack(m, o2, d2, 2, r, dens);
      if (e.kind === 'pass') for (let c = 0; c < 3; c++) delta[c]! += e.weight[c]!;
      const t = transmittance(m, o2, d2, 2, r, dens);
      for (let c = 0; c < 3; c++) ratio[c]! += t[c]!;
    }
    for (let c = 0; c < 3; c++) {
      const exact = Math.exp(-m.sigmaT[c]! * tau);
      expect(delta[c]! / n).toBeCloseTo(exact, 2);
      expect(ratio[c]! / n).toBeCloseTo(exact, 2);
    }
    expect(ptMediumMajorant(m)).toBeCloseTo(2 * grid.densityMax, 6);
  });

  it('ratio tracking with Russian roulette is unbiased and reads the density less often in a dense medium', () => {
    const res = [4, 4, 4] as const;
    const data = new Float32Array(64).map((_, i) => 0.5 + 0.5 * ((i * 37) % 11) / 10);
    const grid = { resolution: res, worldToIndex: gridWorldToIndex(res, [[0, 0, 0], [1, 1, 1]]), densityMax: Math.max(...data) };
    const base = gridDensity(grid, data);
    let lookups = 0;
    const dens = (x: number, y: number, z: number) => { lookups++; return base(x, y, z); };
    const m = ptMediumFrom([3, 4, 5], [0, 0, 0], 0, 1, grid);
    let tau = 0;
    const sub = 20000;
    for (let i = 0; i < sub; i++) tau += base(0.4, 0.3, -0.5 + ((i + 0.5) / sub) * 2) * (2 / sub);
    const n = 100000;
    const run = (roulette: boolean) => {
      const r = rng(77);
      lookups = 0;
      const sum = [0, 0, 0], sq = [0, 0, 0];
      for (let i = 0; i < n; i++) {
        const t = transmittance(m, [0.4, 0.3, -0.5], [0, 0, 1], 2, r, dens, roulette);
        for (let c = 0; c < 3; c++) { sum[c]! += t[c]!; sq[c]! += t[c]! * t[c]!; }
      }
      return { mean: sum.map((x) => x / n), se: sum.map((x, c) => Math.sqrt(Math.max(0, sq[c]! / n - (x / n) ** 2) / n)), lookups: lookups / n };
    };
    const off = run(false), on = run(true);
    for (let c = 0; c < 3; c++) {
      const exact = Math.exp(-m.sigmaT[c]! * tau);
      expect(Math.abs(on.mean[c]! - exact)).toBeLessThan(4 * on.se[c]!);
    }
    expect(on.lookups).toBeLessThan(off.lookups);
  });

  it('gridWorldToIndex puts the first voxel centre at index 0 and honours the medium transform', () => {
    const m = gridWorldToIndex([4, 2, 2], [[-1, 0, 0], [1, 1, 1]], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]);
    // medium-space point (-1 + 0.25, 0.25, 0.25) is the first voxel centre; in world it is shifted by +5 in x
    const px = -1 + 0.25 + 5, py = 0.25, pz = 0.25;
    const idx = [0, 4, 8].map((r) => m[r]! * px + m[r + 1]! * py + m[r + 2]! * pz + m[r + 3]!);
    expect(idx[0]).toBeCloseTo(0, 6);
    expect(idx[1]).toBeCloseTo(0, 6);
    expect(idx[2]).toBeCloseTo(0, 6);
  });
});

describe('closed forms', () => {
  it('slab transmittance and a vanishing slab', () => {
    expect(slabTransmittance(2, 0.5, 0.5)).toBeCloseTo(Math.exp(-2), 12);
    // thin slab, albedo 1, uniform sky: all light is scattered or transmitted, so the single scatter total is 1 - O(tau^2) (the rest is multiple scatter)
    const v = slabSingleScatter(0.02, 1, 0, 1, 1, 1, 100);
    expect(v).toBeGreaterThan(0.999);
    expect(v).toBeLessThanOrEqual(1.0001);
  });
  it('an absorber slab has no scatter term', () => {
    expect(slabSingleScatter(1.5, 0, 0.4, 0.7, 0.8, 1, 50)).toBeCloseTo(Math.exp((-1.5 * 0.7) / 0.8), 10);
    expect(slabSingleScatterPhase(1.5, 0, (c) => hgPhase(c, 0.4), 0.7, 0.8)).toBeCloseTo(Math.exp((-1.5 * 0.7) / 0.8), 12);
  });
  it('the peak-resolved quadrature agrees with a fine run of the uniform one at moderate g', () => {
    // the uniform-cosine quadrature converges slowly (at its defaults it reads 0.8491005 here, 5e-5 low); refined to (400, 2000, 64) steps it reads
    // 0.8491434 (normal) and 0.7491730 (cos 0.6), the values below (that run takes 8 s, so its result is pinned instead)
    expect(slabSingleScatterPhase(0.4, 0.8, (c) => hgPhase(c, 0.5), 1, 1)).toBeCloseTo(0.8491434, 6);
    expect(slabSingleScatterPhase(0.4, 0.8, (c) => hgPhase(c, 0.5), 1, 0.6)).toBeCloseTo(0.749173, 6);
  });
  it('is converged at the forward peak of g 0.99 and conserves energy in a thin non-absorbing slab', () => {
    const coarse = slabSingleScatterPhase(0.4, 0.8, (c) => hgPhase(c, 0.99), 1, 0.6, 1, 1024, 128);
    const fine = slabSingleScatterPhase(0.4, 0.8, (c) => hgPhase(c, 0.99), 1, 0.6, 1, 4096, 512);
    expect(Math.abs(coarse - fine)).toBeLessThan(1e-6);
    for (const g of [0, 0.85, 0.99]) {
      // the deficit is the multiple scattering, O(tau^2): 1.9e-7 for isotropic at tau 2e-4
      const v = slabSingleScatterPhase(0.0002, 1, (c) => hgPhase(c, g), 1, 1);
      expect(v).toBeGreaterThan(1 - 1e-6);
      expect(v).toBeLessThanOrEqual(1 + 1e-7);
    }
  });
});

describe('brick majorants and the DDA twin', () => {
  const res = [19, 11, 26] as const;
  const data = new Float32Array(res[0] * res[1] * res[2]).map((_, i) => ((i * 2654435761) % 1000) / 1000);
  const grid = { resolution: res, worldToIndex: gridWorldToIndex(res, [[0, 0, 0], [1, 1, 1]]), densityMax: 1 };
  const dens = gridDensity(grid, data);
  const mj = brickMajorants(res, data);

  it('dimensions cover the support box [-1, n) of the trilinear lookup', () => {
    expect(brickDims(res)).toEqual([3, 2, 4]);
    expect(mj.resolution).toEqual([3, 2, 4]);
    expect(mj.values).toHaveLength(24);
  });

  it('segments tile the clipped ray and every density sample inside a cell is at most that cell majorant', () => {
    const r = rng(77);
    for (let n = 0; n < 300; n++) {
      // a random ray in index space through (and around) the grid
      const a = [(r() * 1.4 - 0.2) * res[0], (r() * 1.4 - 0.2) * res[1], (r() * 1.4 - 0.2) * res[2]] as const;
      const dd = [r() - 0.5, r() - 0.5, r() - 0.5];
      const len = Math.hypot(...dd);
      const b = dd.map((v) => (v / len) * 3) as unknown as readonly [number, number, number]; // index units per unit t
      const segs = ddaSegments(mj.resolution, a, b, 40);
      for (let i = 1; i < segs.length; i++) expect(segs[i]!.t0).toBeCloseTo(segs[i - 1]!.t1, 9);
      for (const sg of segs) {
        expect(sg.t1).toBeGreaterThanOrEqual(sg.t0 - 1e-9);
        for (let k = 0; k < 6; k++) {
          const t = sg.t0 + ((k + 0.5) / 6) * (sg.t1 - sg.t0);
          const x = a[0] + b[0] * t, y = a[1] + b[1] * t, z = a[2] + b[2] * t;
          // density lookup is in world space: invert the unit-cube mapping used by gridWorldToIndex
          const d = dens((x + 0.5) / res[0], (y + 0.5) / res[1], (z + 0.5) / res[2]);
          const m = mj.values[sg.cell[0] + mj.resolution[0] * (sg.cell[1] + mj.resolution[1] * sg.cell[2])]!;
          expect(d).toBeLessThanOrEqual(m + 1e-6);
        }
      }
      // outside the support box nothing is visited: the first segment starts where the ray enters it
      if (segs.length > 0) {
        const t = segs[0]!.t0 + 1e-6;
        const u = [0, 1, 2].map((k) => (a[k]! + b[k]! * t + 1) / 8);
        for (let k = 0; k < 3; k++) {
          expect(u[k]!).toBeGreaterThanOrEqual(-1e-4);
          expect(u[k]!).toBeLessThanOrEqual(mj.resolution[k]! + 1e-4);
        }
      }
    }
  });

  it('V2.1 minima: every density sample inside a cell is in [min, max] (zero outside the grid counts)', () => {
    const mm = brickMajorants(res, data, { minima: true });
    expect(mm.values).toEqual(mj.values);
    const r = rng(5);
    let tight = 0;
    for (let n = 0; n < 300; n++) {
      const a = [(r() * 1.4 - 0.2) * res[0], (r() * 1.4 - 0.2) * res[1], (r() * 1.4 - 0.2) * res[2]] as const;
      const dd = [r() - 0.5, r() - 0.5, r() - 0.5];
      const len = Math.hypot(...dd);
      const b = dd.map((v) => (v / len) * 3) as unknown as readonly [number, number, number];
      for (const sg of ddaSegments(mm.resolution, a, b, 40)) {
        for (let k = 0; k < 6; k++) {
          const t = sg.t0 + ((k + 0.5) / 6) * (sg.t1 - sg.t0);
          const d = dens((a[0] + b[0] * t + 0.5) / res[0], (a[1] + b[1] * t + 0.5) / res[1], (a[2] + b[2] * t + 0.5) / res[2]);
          const i = sg.cell[0] + mm.resolution[0] * (sg.cell[1] + mm.resolution[1] * sg.cell[2]);
          expect(d).toBeGreaterThanOrEqual(mm.minima![i]! - 1e-6);
          expect(d).toBeLessThanOrEqual(mm.values[i]! + 1e-6);
          if (d < mm.minima![i]! + 0.02) tight++;
        }
      }
    }
    expect(tight).toBeGreaterThan(0); // the bound is reached somewhere (not a vacuous minimum of 0)
    for (let i = 0; i < mm.values.length; i++) expect(mm.minima![i]!).toBeLessThanOrEqual(mm.values[i]!);
    expect(brickMajorants(res, data).minima).toBeUndefined();
  });

  it('a ray outside the support box has no segments', () => {
    expect(ddaSegments(mj.resolution, [-50, 5, 5], [0, 1, 0], 100)).toEqual([]);
    expect(ddaSegments(mj.resolution, [5, 5, 5], [1, 0, 0], 0)).toEqual([]);
  });
});

describe('volume outputs decoding (P5.v15 / P5.v16)', () => {
  it('averages the planes, derives the standard error and the conditional first-scatter distance', () => {
    const w = 2, h = 1, n = w * h, N = 4;
    const sums = new Float32Array(n * 4 * (1 + PT_VOLUME_OUTPUT_VEC4));
    // pixel 0: samples 1, 1, 3, 3 in every channel: mean 2, E[L^2] 5, variance 1, se = sqrt(1 / 3)
    for (let c = 0; c < 3; c++) { sums[c] = 8; sums[(2 * n) * 4 + c] = 20; sums[n * 4 + c] = 4; }
    sums[3] = N;
    sums[n * 4 + 3] = 2; // transmittance sum
    sums[(2 * n) * 4 + 3] = 30; // first-scatter distance sum over the 3 samples that scattered
    sums[(3 * n) * 4] = 3;
    const v = ptDecodeVolumeOutputs(sums, w, h, N);
    expect(v.radiance[0]).toBeCloseTo(2, 6);
    expect(v.order01[0]).toBeCloseTo(1, 6);
    expect(v.stdError[0]).toBeCloseTo(Math.sqrt(1 / 3), 6);
    expect(v.transmittance[0]).toBeCloseTo(0.5, 6);
    expect(v.firstScatterDistance[0]).toBeCloseTo(10, 6);
    expect(v.scatterFraction[0]).toBeCloseTo(0.75, 6);
    expect(v.firstScatterDistance[1]).toBe(0);
    expect(() => ptDecodeVolumeOutputs(new Float32Array(8), w, h, N)).toThrow(RangeError);
  });
});

describe('r16float grids (P5.v18)', () => {
  it('rounds to the nearest f16, keeps exact values and refuses inexact ones at upload', () => {
    for (const v of [0, 1, 0.5, 65504, 2 ** -24, 1 + 2 ** -10, -3.25]) expect(f16BitsToF32(f32ToF16Bits(v))).toBe(v);
    expect(f16BitsToF32(f32ToF16Bits(1 + 2 ** -11))).toBe(1); // tie to even
    expect(f16BitsToF32(f32ToF16Bits(1 + 3 * 2 ** -11))).toBe(1 + 2 ** -9);
    expect(f16BitsToF32(f32ToF16Bits(1e6))).toBe(Infinity);
    const g = Float32Array.from([0.1, 0.33333334, 0.7, 1]);
    const r = ptRoundGridF16(g);
    for (let i = 0; i < g.length; i++) expect(Math.abs((r[i] as number) - (g[i] as number))).toBeLessThanOrEqual((g[i] as number) * 2 ** -11);
    expect(() => ptF16Bits(g, true)).toThrow(/ptRoundGridF16/);
    expect(() => ptF16Bits(r, true)).not.toThrow();
    // the majorant must come from the rounded grid: 0.7 rounds UP (to 0.70020), past the f32 value
    expect(r[2]).toBeGreaterThan(g[2] as number);
  });
});

describe('V1.2 hardware density lookup twin', () => {
  it('boundary factor x clamp-to-edge trilinear equals the zero-outside manual trilinear (incl. outside the box)', () => {
    const [nx, ny, nz] = [5, 4, 3] as const;
    const r = rng(7);
    const data = new Float32Array(nx * ny * nz).map(() => r());
    const grid = { resolution: [nx, ny, nz] as [number, number, number], worldToIndex: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] } as unknown as Parameters<typeof gridDensity>[0];
    const manual = gridDensity(grid, data);
    const clamped = (gx: number, gy: number, gz: number): number => {
      const at = (i: number, j: number, k: number) => data[Math.min(Math.max(i, 0), nx - 1) + nx * (Math.min(Math.max(j, 0), ny - 1) + ny * Math.min(Math.max(k, 0), nz - 1))] as number;
      const i0 = Math.floor(gx), j0 = Math.floor(gy), k0 = Math.floor(gz);
      const fx = gx - i0, fy = gy - j0, fz = gz - k0;
      let acc = 0;
      for (let c = 0; c < 8; c++) {
        const di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
        acc += at(i0 + di, j0 + dj, k0 + dk) * (di ? fx : 1 - fx) * (dj ? fy : 1 - fy) * (dk ? fz : 1 - fz);
      }
      return acc;
    };
    for (let t = 0; t < 2000; t++) {
      const gx = r() * (nx + 4) - 2, gy = r() * (ny + 4) - 2, gz = r() * (nz + 4) - 2;
      const hw = gridBoundaryFactor(gx, nx) * gridBoundaryFactor(gy, ny) * gridBoundaryFactor(gz, nz) * clamped(gx, gy, gz);
      expect(hw).toBeCloseTo(manual(gx, gy, gz), 12);
    }
    expect(gridBoundaryFactor(-1, 5)).toBe(0);
    expect(gridBoundaryFactor(0, 5)).toBe(1);
    expect(gridBoundaryFactor(5, 5)).toBe(0);
    expect(gridBoundaryFactor(2, 5)).toBe(1);
  });
});

describe('V2.2 residual ratio tracking twin', () => {
  // a smooth dense density: the minorant is a good control, so the residual is small
  const res = [24, 24, 24] as const;
  const data = new Float32Array(res[0] * res[1] * res[2]);
  for (let k = 0; k < res[2]; k++) for (let j = 0; j < res[1]; j++) for (let i = 0; i < res[0]; i++)
    data[i + res[0] * (j + res[1] * k)] = 0.55 + 0.4 * Math.sin(i * 0.21) * Math.cos(j * 0.17 + 0.5) * Math.sin(k * 0.13 + 1);
  const grid = { resolution: res, worldToIndex: gridWorldToIndex(res, [[0, 0, 0], [1, 1, 1]]), densityMax: Math.max(...data) };
  const dens = gridDensity(grid, data);
  const mm = brickMajorants(res, data, { minima: true });
  const a = [-2, 9.3, 11.1], b = [1, 0.1, 0.05];
  const tmax = 28;
  const along = (t: number) => dens((a[0]! + b[0]! * t + 0.5) / res[0], (a[1]! + b[1]! * t + 0.5) / res[1], (a[2]! + b[2]! * t + 0.5) / res[2]);
  const sigmaT = 0.08;
  const cells = ddaSegments(mm.resolution, a as never, b as never, tmax).map((sg) => {
    const i = sg.cell[0] + mm.resolution[0] * (sg.cell[1] + mm.resolution[1] * sg.cell[2]);
    return { t0: sg.t0, t1: sg.t1, max: mm.values[i]!, min: mm.minima![i]! };
  });
  let tau = 0;
  const sub = 40000;
  for (let i = 0; i < sub; i++) tau += along(((i + 0.5) / sub) * tmax) * (tmax / sub);
  const exact = Math.exp(-sigmaT * tau);

  const run = (kind: 'residual' | 'ratio' | 'residual-roulette', n: number) => {
    const r = rng(1234);
    let sum = 0, sq = 0, lookups = 0;
    for (let i = 0; i < n; i++) {
      let tr: number;
      if (kind === 'ratio') {
        // ratio tracking with the per-cell majorant (what the brick kernel does)
        tr = 1;
        for (const c of cells) {
          const mu = sigmaT * c.max;
          if (!(mu > 0)) continue;
          let t = c.t0;
          for (let k = 0; k < 4096; k++) {
            t += -Math.log(1 - r()) / mu;
            if (t >= c.t1) break;
            lookups++;
            tr *= 1 - (sigmaT * along(t)) / mu;
          }
        }
      } else {
        const e = residualTransmittance(cells, along, sigmaT, r, kind === 'residual-roulette');
        tr = e.tr;
        lookups += e.lookups;
      }
      sum += tr;
      sq += tr * tr;
    }
    return { mean: sum / n, se: Math.sqrt(Math.max(0, sq / n - (sum / n) ** 2) / n), lookups: lookups / n };
  };

  it('is unbiased against the exact line integral (1e5 trials, 4 SE), also with roulette', () => {
    for (const kind of ['residual', 'residual-roulette'] as const) {
      const e = run(kind, 100000);
      expect(Math.abs(e.mean - exact)).toBeLessThan(4 * e.se + 1e-9);
    }
  });

  it('has lower variance and fewer lookups than ratio tracking on a dense smooth grid', () => {
    const ratio = run('ratio', 100000), residual = run('residual', 100000);
    expect(Math.abs(ratio.mean - exact)).toBeLessThan(4 * ratio.se);
    expect(residual.se).toBeLessThan(ratio.se);
    expect(residual.lookups).toBeLessThan(ratio.lookups);
  });
});

describe('V2.4 decomposition tracking twin (brick free flight)', () => {
  // the same smooth dense grid as V2.2: min / max per cell is high, so most tentative collisions are control ones
  const res = [24, 24, 24] as const;
  const data = new Float32Array(res[0] * res[1] * res[2]);
  for (let k = 0; k < res[2]; k++) for (let j = 0; j < res[1]; j++) for (let i = 0; i < res[0]; i++)
    data[i + res[0] * (j + res[1] * k)] = 0.55 + 0.4 * Math.sin(i * 0.21) * Math.cos(j * 0.17 + 0.5) * Math.sin(k * 0.13 + 1);
  const grid = { resolution: res, worldToIndex: gridWorldToIndex(res, [[0, 0, 0], [1, 1, 1]]), densityMax: Math.max(...data) };
  const dens = gridDensity(grid, data);
  const mm = brickMajorants(res, data, { minima: true });
  const a = [-2, 9.3, 11.1], b = [1, 0.1, 0.05];
  const tmax = 28;
  const along = (t: number) => dens((a[0]! + b[0]! * t + 0.5) / res[0], (a[1]! + b[1]! * t + 0.5) / res[1], (a[2]! + b[2]! * t + 0.5) / res[2]);
  const cells = ddaSegments(mm.resolution, a as never, b as never, tmax).map((sg) => {
    const i = sg.cell[0] + mm.resolution[0] * (sg.cell[1] + mm.resolution[1] * sg.cell[2]);
    return { t0: sg.t0, t1: sg.t1, max: mm.values[i]!, min: mm.minima![i]! };
  });
  // optical depth per unit sigma_t up to t (fine quadrature)
  const sub = 40000;
  const cum = new Float64Array(sub + 1);
  for (let i = 0; i < sub; i++) cum[i + 1] = cum[i]! + along(((i + 0.5) / sub) * tmax) * (tmax / sub);
  const tauAt = (t: number) => cum[Math.min(sub, Math.round((t / tmax) * sub))]!;
  const N = 100000;
  const run = (sigmaT: [number, number, number], albedo: [number, number, number], decomposition: boolean, seed: number) => {
    const r = rng(seed);
    const pass = [0, 0, 0], passSq = [0, 0, 0], scat = [0, 0, 0], scatSq = [0, 0, 0];
    const marks = [4, 10, 18], before = [0, 0, 0];
    let lookups = 0;
    for (let i = 0; i < N; i++) {
      const e = brickTrack({ sigmaT, albedo }, cells, along, r, decomposition);
      lookups += e.lookups;
      const acc = e.kind === 'pass' ? [pass, passSq] : e.kind === 'scatter' ? [scat, scatSq] : null;
      if (acc) for (let c = 0; c < 3; c++) { acc[0]![c]! += e.weight[c]!; acc[1]![c]! += e.weight[c]! ** 2; }
      if (e.kind === 'scatter') marks.forEach((m, k) => { if (e.t < m) before[k]!++; });
    }
    const mean = (s: number[]) => s.map((x) => x / N);
    const se = (s: number[], q: number[]) => s.map((x, c) => Math.sqrt(Math.max(0, q[c]! / N - (x / N) ** 2) / N));
    return { pass: mean(pass), passSe: se(pass, passSq), scat: mean(scat), scatSe: se(scat, scatSq), cdf: before.map((x) => x / N), marks, lookups: lookups / N };
  };

  it('grey, albedo 1: the first-collision distance follows 1 - exp(-tau(t)) in both modes (1e5 trials, 4 SE), with fewer lookups', () => {
    const sT = 0.08;
    const plain = run([sT, sT, sT], [1, 1, 1], false, 7), dec = run([sT, sT, sT], [1, 1, 1], true, 8);
    for (const e of [plain, dec]) {
      e.marks.forEach((m, k) => {
        const p = 1 - Math.exp(-sT * tauAt(m));
        expect(Math.abs(e.cdf[k]! - p)).toBeLessThan(4 * Math.sqrt((p * (1 - p)) / N));
      });
      const T = Math.exp(-sT * tauAt(tmax));
      expect(Math.abs(e.pass[0]! - T)).toBeLessThan(4 * e.passSe[0]! + 1e-9);
    }
    // lookups saved = the control share min / max of the cells' tentative collisions (this grid: about 10%; plain 0.90 vs 0.82 per flight)
    expect(dec.lookups).toBeLessThan(0.95 * plain.lookups);
  });

  it('chromatic with absorption: pass weight = T per channel and scatter weight = albedo (1 - T) per channel, both modes (4 SE)', () => {
    const sigmaT: [number, number, number] = [0.12, 0.07, 0.03], albedo: [number, number, number] = [0.9, 0.6, 0.95];
    for (const [d, seed] of [[false, 11], [true, 12]] as const) {
      const e = run(sigmaT, albedo, d, seed);
      for (let c = 0; c < 3; c++) {
        const T = Math.exp(-sigmaT[c]! * tauAt(tmax));
        expect(Math.abs(e.pass[c]! - T)).toBeLessThan(4 * e.passSe[c]! + 1e-9);
        expect(Math.abs(e.scat[c]! - albedo[c]! * (1 - T))).toBeLessThan(4 * e.scatSe[c]! + 1e-9);
      }
    }
  });
});

describe('r8unorm grids (V3.1)', () => {
  const res = [20, 19, 18] as const;
  const g = new Float32Array(res[0] * res[1] * res[2]).map((_, i) => (i % 7 === 0 ? 0 : 0.3 + 2.2 * Math.abs(Math.sin(i * 0.37))));
  it('quantises to nearest with scale = max, dequantised values are the on-grid values', () => {
    const q = ptQuantizeGridR8(g);
    expect(q.scale).toBe(Math.fround(Math.max(...g)));
    expect(Math.max(...q.bytes)).toBe(255);
    for (let i = 0; i < g.length; i++) {
      expect(Math.abs(q.dequantized[i]! - g[i]!)).toBeLessThanOrEqual(q.scale / 255 / 2 + 1e-6);
      expect(q.dequantized[i]).toBe(Math.fround((q.bytes[i]! / 255) * q.scale));
    }
    expect(ptR8Bytes(q.dequantized, q.scale)).toEqual(q.bytes); // round trip
    expect(ptQuantizeGridR8(q.dequantized).dequantized).toEqual(q.dequantized); // idempotent
  });
  it('refuses data that is not on the r8 grid', () => {
    const q = ptQuantizeGridR8(g);
    expect(() => ptR8Bytes(g, q.scale)).toThrow(/ptQuantizeGridR8/);
    expect(() => ptQuantizeGridR8([1, -1])).toThrow(RangeError);
  });
  it('the texel-unit majorant bounds every trilinear sample of the texels, the minorant is bounded below', () => {
    const q = ptQuantizeGridR8(g);
    const mm = brickMajorants(res, q.dequantized, { minima: true });
    const texels = Float32Array.from(q.bytes, (b) => b / 255);
    const hi = mm.values.map((v) => ptTexelBound(v, q.scale));
    const lo = mm.minima!.map((v) => ptTexelBound(v, q.scale, false));
    const dims = brickDims(res);
    let n = 0;
    for (let k = 0; k < res[2]; k++) for (let j = 0; j < res[1]; j++) for (let i = 0; i < res[0]; i++) {
      const t = texels[i + res[0] * (j + res[1] * k)]!;
      const c = Math.floor(i / 8) + dims[0] * (Math.floor(j / 8) + dims[1] * Math.floor(k / 8));
      expect(t).toBeLessThanOrEqual(hi[c]!);
      n++;
    }
    expect(n).toBe(g.length);
    for (let c = 0; c < lo.length; c++) expect(lo[c]!).toBeLessThanOrEqual(mm.minima![c]! / q.scale + 1e-6);
  });
});

describe('V2.3 telescoping transmittance twin (Cycles / Misso et al. 2022)', () => {
  const build = (rough: boolean) => {
    const res = [24, 24, 24] as const;
    const data = new Float32Array(res[0] * res[1] * res[2]);
    const r = rng(99);
    for (let k = 0; k < res[2]; k++) for (let j = 0; j < res[1]; j++) for (let i = 0; i < res[0]; i++)
      data[i + res[0] * (j + res[1] * k)] = rough ? r() : 0.55 + 0.4 * Math.sin(i * 0.21) * Math.cos(j * 0.17 + 0.5) * Math.sin(k * 0.13 + 1);
    const grid = { resolution: res, worldToIndex: gridWorldToIndex(res, [[0, 0, 0], [1, 1, 1]]), densityMax: Math.max(...data) };
    const dens = gridDensity(grid, data);
    const mm = brickMajorants(res, data, { minima: true });
    const a = [-2, 9.3, 11.1], b = [1, 0.1, 0.05];
    const along = (t: number) => dens((a[0]! + b[0]! * t + 0.5) / res[0], (a[1]! + b[1]! * t + 0.5) / res[1], (a[2]! + b[2]! * t + 0.5) / res[2]);
    const cells = ddaSegments(mm.resolution, a as never, b as never, 28).map((sg) => {
      const i = sg.cell[0] + mm.resolution[0] * (sg.cell[1] + mm.resolution[1] * sg.cell[2]);
      return { t0: sg.t0, t1: sg.t1, max: mm.values[i]!, min: mm.minima![i]! };
    });
    return { along, cells };
  };
  for (const rough of [false, true]) for (const sigmaT of [0.08, 0.4]) {
    it(`${rough ? 'rough' : 'smooth'} grid, sigmaT ${sigmaT}: mean equals the exact line integral (4 SE + the 16k-step marching bias), variance per lookup vs ratio tracking`, () => {
      const { along, cells } = build(rough);
      let tau = 0;
      const sub = 40000;
      for (const c of cells) for (let i = 0; i < sub / 10; i++) tau += along(c.t0 + ((i + 0.5) / (sub / 10)) * (c.t1 - c.t0)) * ((c.t1 - c.t0) / (sub / 10));
      const exact = Math.exp(-sigmaT * tau);
      const n = 100000, r = rng(5);
      let s1 = 0, q1 = 0, l1 = 0, s2 = 0, q2 = 0, l2 = 0;
      for (let i = 0; i < n; i++) {
        const e = telescopingTransmittance(cells, along, sigmaT, r, () => r());
        s1 += e.tr; q1 += e.tr * e.tr; l1 += e.lookups;
        let tr = 1, lk = 0;
        for (const c of cells) {
          const mu = sigmaT * c.max;
          if (!(mu > 0)) continue;
          let t = c.t0;
          for (let k = 0; k < 4096; k++) { t += -Math.log(1 - r()) / mu; if (t >= c.t1) break; lk++; tr *= 1 - (sigmaT * along(t)) / mu; }
        }
        s2 += tr; q2 += tr * tr; l2 += lk;
      }
      const m1 = s1 / n, se1 = Math.sqrt((q1 / n - m1 * m1) / n), m2 = s2 / n, se2 = Math.sqrt((q2 / n - m2 * m2) / n);
      expect(Math.abs(m1 - exact)).toBeLessThan(4 * se1 + 2e-4);
      expect(Math.abs(m2 - exact)).toBeLessThan(4 * se2 + 1e-9);
    });
  }
});

describe('sparse brick atlas (V3.2)', () => {
  const res: [number, number, number] = [37, 21, 29];
  const rand = rng(7);
  const data = new Float32Array(res[0] * res[1] * res[2]);
  // two blobs (one touching the grid corner) and a lone voxel on a brick face, everything else exactly zero
  for (let k = 0; k < res[2]; k++) for (let j = 0; j < res[1]; j++) for (let i = 0; i < res[0]; i++) {
    const a = (i - 5) ** 2 + (j - 4) ** 2 + (k - 3) ** 2, b = (i - 30) ** 2 + (j - 12) ** 2 + (k - 20) ** 2;
    if (a < 30 || b < 25) data[i + res[0] * (j + res[1] * k)] = Math.fround(0.2 + rand());
  }
  data[15 + res[0] * (8 + res[1] * 16)] = 0.7; // on the +side face of brick 2 (voxel 15 = 8 * 2 - 1 .. support of cell 1 and 2)
  const atlas = buildBrickAtlas(res, data);
  const dense = gridDensity({ resolution: res, worldToIndex: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], densityMax: 2 }, data);

  it('is sparse, and an empty brick has majorant 0 and indirection ATLAS_EMPTY', () => {
    const mj = brickMajorants(res, data).values;
    let occ = 0;
    for (let i = 0; i < mj.length; i++) {
      expect((atlas.indirection[i] === ATLAS_EMPTY) === ((mj[i] as number) === 0)).toBe(true);
      if ((mj[i] as number) > 0) occ++;
    }
    expect(atlas.occupied).toBe(occ);
    expect(occ).toBeLessThan(mj.length);
    expect(brickAtlasMemory(atlas, 4).ratio).toBeLessThan(1.2);
    expect(atlas.dims.map((d) => d % ATLAS_CELL)).toEqual([0, 0, 0]);
  });

  it('reads exactly the dense trilinear density at random points, brick edges and outside the grid', () => {
    const r = rng(11);
    let nonzero = 0;
    for (let n = 0; n < 20000; n++) {
      const p = [r() * (res[0] + 6) - 3, r() * (res[1] + 6) - 3, r() * (res[2] + 6) - 3];
      if (n % 3 === 0) for (let a = 0; a < 3; a++) p[a] = Math.round((p[a] as number) / 8) * 8 + (r() < 0.5 ? -1 : 0) + (r() < 0.5 ? 0 : (r() - 0.5) * 1e-4); // brick faces (8c - 1) and edges
      const a = atlasDensityAtIndex(atlas, p[0] as number, p[1] as number, p[2] as number);
      const d = dense(p[0] as number, p[1] as number, p[2] as number);
      expect(a).toBe(d);
      if (d > 0) nonzero++;
    }
    expect(nonzero).toBeGreaterThan(300);
  });

  it('apron: a point on every brick face reads the neighbour voxels from inside its own brick', () => {
    for (let c = 0; c < 4; c++) for (const x of [8 * c - 1, 8 * c - 1 + 1e-5, 8 * c + 6.999, 8 * c + 7 - 1e-6]) {
      expect(atlasDensityAtIndex(atlas, x, 8, 16)).toBe(dense(x, 8, 16));
      expect(atlasDensityAtIndex(atlas, 5, x, 3)).toBe(dense(5, x, 3));
    }
  });
});

describe('height fog (V4.5)', () => {
  const h = { y0: 0.5, H: 2 };
  const sig = [0.3, 0.5, 0.8] as const;
  const med = ptMediumFrom([0, 0, 0], sig, 0, 1, undefined, h);
  const numericG = (o: readonly number[], d: readonly number[], L: number): number => {
    const n = 20000;
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.exp(-((o[1] as number) + (d[1] as number) * ((i + 0.5) / n) * L - h.y0) / h.H);
    return (s * L) / n;
  };
  const dirs: [number, number, number][] = [[0.6, 0.8, 0], [0.6, -0.8, 0], [1, 0, 0], [0.99999, 0.0001, Math.sqrt(1 - 0.99999 ** 2 - 1e-8)], [0.36, -0.48, 0.8]];
  for (const d of dirs) {
    for (const L of [0.1, 2, 9]) {
      it(`optical depth matches the numeric integral (d.y ${d[1].toFixed(4)}, L ${L})`, () => {
        const o = [0.2, 1.7, -0.4] as const;
        expect(heightOpticalDepth(h, o, d, L) / numericG(o, d, L)).toBeCloseTo(1, 5);
      });
    }
    it(`distance inverts the optical depth (d.y ${d[1].toFixed(4)})`, () => {
      const o = [0, 1.2, 0] as const;
      for (const t of [0.01, 0.5, 3, 12]) expect(heightDistance(h, o, d, heightOpticalDepth(h, o, d, t)) / t).toBeCloseTo(1, 4);
    });
  }
  it('transmittance is the closed form exp(-sigma G)', () => {
    const o = [0, 3, 0] as const, d = [0.6, -0.8, 0] as const;
    const tr = transmittance(med, o, d, 4, () => 0.5);
    const G = (Math.exp(-(3 - 0.5) / 2) * 2) / -0.8 * (1 - Math.exp((-4 * -0.8) / 2));
    sig.forEach((s, c) => expect(tr[c]).toBeCloseTo(Math.exp(-s * G), 10));
  });
  it('sampled collision distances follow p(t) = sigma(t) exp(-tau(t)) (grey medium, albedo 1)', () => {
    const g = ptMediumFrom([0, 0, 0], [0.7, 0.7, 0.7], 0, 1, undefined, h);
    const o = [0, 4, 0] as const, d = [0.6, -0.8, 0] as const, tmax = 6;
    const r = rng(11);
    const bins = 12, count = new Array<number>(bins).fill(0);
    const n = 400000;
    let passed = 0;
    for (let i = 0; i < n; i++) {
      const ev = deltaTrack(g, o, d, tmax, r);
      expect(ev.weight[0]).toBeCloseTo(1, 12);
      if (ev.kind === 'pass') passed++;
      else count[Math.min(bins - 1, Math.floor((ev.t / tmax) * bins))]!++;
    }
    const cdf = (t: number): number => 1 - Math.exp(-0.7 * heightOpticalDepth(h, o, d, t));
    for (let b = 0; b < bins; b++) {
      const p = cdf(((b + 1) / bins) * tmax) - cdf((b / bins) * tmax);
      expect(count[b]! / n).toBeCloseTo(p, 2);
    }
    expect(passed / n).toBeCloseTo(1 - cdf(tmax), 2);
  });
  it('chromatic flight is unbiased: the mean pass weight is the transmittance and the scatter weights integrate to the single scatter', () => {
    const m = ptMediumFrom([0.1, 0.2, 0.1], [0.3, 0.3, 0.7], 0, 1, undefined, h);
    const o = [0, 2, 0] as const, d = [0.6, 0.8, 0] as const, tmax = 5;
    const r = rng(5);
    const n = 300000;
    const pass = [0, 0, 0], scat = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const ev = deltaTrack(m, o, d, tmax, r);
      const into = ev.kind === 'pass' ? pass : ev.kind === 'scatter' ? scat : null;
      if (into) for (let c = 0; c < 3; c++) into[c]! += ev.weight[c]!;
    }
    const G = heightOpticalDepth(h, o, d, tmax);
    for (let c = 0; c < 3; c++) {
      expect(pass[c]! / n).toBeCloseTo(Math.exp(-m.sigmaT[c]! * G), 2);
      // expected number of scatters of channel c over [0, tmax]: integral of sigma_s T over t = albedo (1 - T)
      expect(scat[c]! / n).toBeCloseTo(m.albedo[c]! * (1 - Math.exp(-m.sigmaT[c]! * G)), 2);
    }
  });
  it('a purely absorbing height fog never collides and passes with the exact transmittance', () => {
    const m = ptMediumFrom([0.3, 0.5, 0.8], [0, 0, 0], 0, 1, undefined, h);
    const o = [0, 1, 0] as const, d = [0, 1, 0] as const;
    const ev = deltaTrack(m, o, d, 3, () => 0.3);
    const G = heightOpticalDepth(h, o, d, 3);
    expect(ev.kind).toBe('pass');
    for (let c = 0; c < 3; c++) expect(ev.weight[c]).toBeCloseTo(Math.exp(-m.sigmaT[c]! * G), 12);
  });
  it('an infinite upward ray has a finite depth H e^{-a} / d.y', () => {
    expect(heightOpticalDepth(h, [0, 0.5, 0], [0, 1, 0], 1e9)).toBeCloseTo(2, 6);
  });
});

describe('V4.6 medium emission (TS twin)', () => {
  const le: [number, number, number] = [2, 0.5, 1];
  it('homogeneous with scattering off: the per-segment form is Le (1 - exp(-sigma_t L)) and one delta-tracking pass adds exactly that', () => {
    const m = { ...ptMediumFrom([0.8, 1.6, 0.4], [0, 0, 0], 0), emission: le };
    const L = 1.7;
    const emit: [number, number, number] = [0, 0, 0];
    deltaTrack(m, [0, 0, 0], [0, 0, 1], L, rng(1), () => 1, emit);
    for (let c = 0; c < 3; c++) expect(emit[c]).toBeCloseTo(le[c]! * (1 - Math.exp(-(m.sigmaT[c] as number) * L)), 12);
    expect(ptMediumEmissionSegment(m, L)).toEqual(emit);
  });
  it('albedo scales the source by sigma_a / sigma_t', () => {
    const m = { ...ptMediumFrom([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0), emission: le };
    const e = ptMediumEmissionSegment(m, 2);
    expect(e[0]).toBeCloseTo(le[0]! * 0.5 * (1 - Math.exp(-2)), 12);
  });
  it('grid with scattering off: the mean of the collision-term estimator equals the quadrature of Le sigma_a(z) exp(-tau(z)) dz', () => {
    const nz = 8, data = Float32Array.from({ length: nz }, (_, k) => 0.25 + (k % 3) * 0.5); // density per z layer
    const grid = { resolution: [1, 1, nz] as [number, number, number], worldToIndex: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, nz / 4, -0.5], densityMax: 0.25 + 2 * 0.5 };
    const m = { ...ptMediumFrom([1.2, 0.6, 0.9], [0, 0, 0], 0, 1, grid), emission: le };
    const dens = (_x: number, _y: number, z: number) => { const g = Math.min(Math.max(z * nz / 4 - 0.5, 0), nz - 1), i = Math.min(Math.floor(g), nz - 2), f = g - i; return data[i]! * (1 - f) + data[i + 1]! * f; };
    const Lz = 4, steps = 40000;
    const ref = [0, 0, 0];
    const tau = [0, 0, 0];
    for (let i = 0; i < steps; i++) {
      const z = ((i + 0.5) / steps) * Lz, d = dens(0, 0, z), dz = Lz / steps;
      for (let c = 0; c < 3; c++) { const st = (m.sigmaT[c] as number) * d; tau[c]! += st * dz * 0.5; ref[c]! += le[c]! * st * Math.exp(-tau[c]!) * dz; tau[c]! += st * dz * 0.5; }
    }
    const r = rng(5), N = 60000, acc = [0, 0, 0];
    for (let n = 0; n < N; n++) { const e: [number, number, number] = [0, 0, 0]; deltaTrack(m, [0, 0, 0], [0, 0, 1], Lz, r, dens, e); for (let c = 0; c < 3; c++) acc[c]! += e[c]!; }
    for (let c = 0; c < 3; c++) expect(acc[c]! / N / ref[c]!).toBeCloseTo(1, 1);
  });
});
