import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { type BlueNoiseMeta, decodeBlueNoisePng } from '../../src/noise/bluenoise.ts';
import { isPermutation, ksUniform, powerSpectrum2D, radialAverage, sliceSpectrum, temporalSpectrum, variance } from './analysis.ts';
import { DualExtrema } from './extrema.ts';
import { decodePng, encodePng } from './png.ts';
import { mulberry32 } from './rng.ts';
import { vectorSwapVec2 } from './vectorSwap.ts';
import { voidAndCluster } from './voidAndCluster.ts';

const ASSETS = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'assets', 'bluenoise');

/** Mean normalised radial power below `cut` x Nyquist (white noise reads about 1). */
function lowBand(img: ArrayLike<number>, w: number, cut = 0.25): number {
  const r = radialAverage(powerSpectrum2D(img, w, w), w, w);
  const v = variance(img);
  let s = 0;
  let c = 0;
  for (let i = 1; i < r.length; i++) if (i / r.length < cut) { s += r[i]!; c++; }
  return s / c / v;
}

describe('DualExtrema', () => {
  it('matches a brute-force scan after random updates', () => {
    const n = 5000;
    const ex = new DualExtrema(n);
    const rand = mulberry32(3);
    for (let i = 0; i < n; i++) {
      ex.hi[i] = rand() < 0.3 ? rand() : -Infinity;
      ex.lo[i] = ex.hi[i] === -Infinity ? rand() : -Infinity;
    }
    ex.rebuild();
    for (let step = 0; step < 300; step++) {
      for (let k = 0; k < 7; k++) {
        const i = Math.floor(rand() * n);
        ex.hi[i] = rand() < 0.5 ? rand() : -Infinity;
        ex.lo[i] = rand() < 0.5 ? rand() : -Infinity;
        ex.touch(i);
      }
      ex.flush();
      let bh = -1;
      let bl = -1;
      for (let i = 0; i < n; i++) {
        if (ex.hi[i]! > -Infinity && (bh < 0 || ex.hi[i]! > ex.hi[bh]!)) bh = i;
        if (ex.lo[i]! > -Infinity && (bl < 0 || ex.lo[i]! > ex.lo[bl]!)) bl = i;
      }
      expect(ex.hi[ex.argMaxFilled()]).toBe(ex.hi[bh]);
      expect(ex.lo[ex.argMinEmpty()]).toBe(ex.lo[bl]);
    }
  });
});

describe('void-and-cluster 2D', () => {
  const r = voidAndCluster({ width: 32, height: 32, seed: 5 });

  it('is a rank permutation with exactly uniform values', () => {
    expect(isPermutation(r.ranks)).toBe(true);
    expect(ksUniform(r.values)).toBeLessThan(1.5 / r.values.length);
  });

  it('has a low-frequency hole (white noise reads ~1)', () => {
    expect(lowBand(r.values, 32)).toBeLessThan(0.02);
    const rand = mulberry32(9);
    const white = Float32Array.from({ length: 1024 }, () => rand());
    expect(lowBand(white, 32)).toBeGreaterThan(0.5);
  });

  it('is blue at every threshold (dither-array property)', () => {
    for (const t of [0.1, 0.3, 0.5]) {
      const bin = Float32Array.from(r.values, (v) => (v < t ? 1 : 0));
      expect(lowBand(bin, 32)).toBeLessThan(0.15);
    }
  });

  it('is deterministic for a seed', () => {
    expect(voidAndCluster({ width: 32, height: 32, seed: 5 }).ranks).toEqual(r.ranks);
    expect(voidAndCluster({ width: 32, height: 32, seed: 6 }).ranks).not.toEqual(r.ranks);
  });
});

describe('void-and-cluster spatiotemporal', () => {
  const W = 32;
  const D = 16;
  const r = voidAndCluster({ width: W, height: W, depth: D, seed: 2 });

  it('ranks the whole volume', () => {
    expect(isPermutation(r.ranks)).toBe(true);
  });

  it('every slice is blue', () => {
    const s = sliceSpectrum(r.values, W, W, D);
    let low = 0;
    for (let i = 1; i < 4; i++) low += s[i]! / 3;
    expect(low).toBeLessThan(0.05);
    for (let z = 0; z < D; z++) expect(lowBand(r.values.subarray(z * W * W, (z + 1) * W * W), W)).toBeLessThan(0.1);
  });

  it('every pixel is high-pass over time', () => {
    const t = temporalSpectrum(r.values, W, W, D);
    expect(t[1]!).toBeLessThan(0.2);
    expect(t[D / 2]!).toBeGreaterThan(1.3);
  });

  it('slices are roughly uniform', () => {
    for (let z = 0; z < D; z++) expect(ksUniform(r.values.subarray(z * W * W, (z + 1) * W * W))).toBeLessThan(0.08);
  });
});

describe('vec2 pixel swapping (Georgiev & Fajardo)', () => {
  const W = 32;
  const g = vectorSwapVec2({ width: W, height: W, iterationsPerPixel: 60, seed: 4 });

  it('keeps both marginals exact permutations of k/(N-1)', () => {
    for (const c of [0, 1]) {
      const ranks = Array.from({ length: W * W }, (_, i) => Math.round(g.values[2 * i + c]! * (W * W - 1)));
      expect(isPermutation(ranks)).toBe(true);
    }
  });

  it('turns shuffled (white) channels blue', () => {
    expect(g.accepted).toBeGreaterThan(0);
    const x = Float32Array.from({ length: W * W }, (_, i) => g.values[2 * i]!);
    expect(lowBand(x, W)).toBeLessThan(0.4);
  });
});

describe('PNG', () => {
  it('round-trips 8/16-bit grey and RGB through every filter type', () => {
    const rand = mulberry32(1);
    for (const bitDepth of [8, 16] as const) {
      for (const channels of [1, 3] as const) {
        const data = Uint16Array.from({ length: 7 * 5 * channels }, () => Math.floor(rand() * (bitDepth === 16 ? 65536 : 256)));
        for (const filter of [0, 1, 2, 3, 4] as const) {
          const img = { width: 7, height: 5, channels, bitDepth, data };
          expect(decodePng(encodePng(img, filter)).data).toEqual(data);
        }
      }
    }
  });

  it('runtime decoder unpacks a vec2 atlas (with an empty tile) like the tool decoder', async () => {
    const size = 4;
    const depth = 3;
    const cols = 2;
    const width = size * cols;
    const height = size * 2;
    const data = new Uint16Array(width * height * 3);
    for (let i = 0; i < data.length; i++) data[i] = (i * 2654435761) % 65536;
    const meta: BlueNoiseMeta = {
      name: 't', kind: 'st', width: size, height: size, depth, channels: 2, f32: 't.f32', png: 't.png',
      atlasColumns: cols, sigmaSpatial: 1.5, sigmaTemporal: 1.5, seed: 0,
    };
    for (const filter of [0, 4] as const) {
      const tex = await decodeBlueNoisePng(encodePng({ width, height, channels: 3, bitDepth: 16, data }, filter), meta);
      for (let z = 0; z < depth; z++) {
        for (let y = 0; y < size; y++) {
          for (let x = 0; x < size; x++) {
            const ax = (z % cols) * size + x;
            const ay = Math.floor(z / cols) * size + y;
            for (let c = 0; c < 2; c++) {
              expect(tex.data[((z * size + y) * size + x) * 2 + c]).toBeCloseTo(data[(ay * width + ax) * 3 + c]! / 65535, 7);
            }
          }
        }
      }
    }
  });
});

describe.skipIf(!existsSync(join(ASSETS, 'manifest.json')))('committed assets', () => {
  const manifest = existsSync(join(ASSETS, 'manifest.json'))
    ? (JSON.parse(readFileSync(join(ASSETS, 'manifest.json'), 'utf8')) as { textures: BlueNoiseMeta[] })
    : { textures: [] };

  it('lists scalar + vec2 at 64 and 128 for vc2d and st, and vec2 for gf', () => {
    const names = manifest.textures.map((t) => t.name).sort();
    expect(names).toEqual([
      'gf-vec2-128', 'gf-vec2-64',
      'st-scalar-128x64', 'st-scalar-64x32', 'st-vec2-128x64', 'st-vec2-64x32',
      'vc2d-scalar-128', 'vc2d-scalar-64', 'vc2d-vec2-128', 'vc2d-vec2-64',
    ]);
  });

  it('every PNG decodes to its manifest size; the 2D ones keep exact ranks and a low-frequency hole', async () => {
    for (const meta of manifest.textures) {
      const tex = await decodeBlueNoisePng(readFileSync(join(ASSETS, meta.png)), meta);
      expect(tex.data.length).toBe(meta.width * meta.height * meta.depth * meta.channels);
      if (meta.depth !== 1) continue;
      const n = meta.width * meta.height;
      for (let c = 0; c < meta.channels; c++) {
        const ch = Float32Array.from({ length: n }, (_, i) => tex.data[i * meta.channels + c]!);
        expect(isPermutation(Array.from(ch, (v) => Math.round(v * (n - 1))))).toBe(true);
        expect(lowBand(ch, meta.width, 0.2)).toBeLessThan(meta.kind === 'gf' ? 0.05 : 0.01);
      }
    }
  });
});
