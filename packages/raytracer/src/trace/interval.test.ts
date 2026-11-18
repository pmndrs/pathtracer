import { describe, expect, it } from 'vitest';
import { traceModuleWgsl } from '@pmndrs/bvh';
import { RESERVED, balanced, declared, declaredIdentifiers } from '../wgsl-names.testutil.ts';
import { buildCaps } from '../caps/index.ts';
import {
  INTERVAL_PARAMS_BYTES,
  INTERVAL_RING_STRIDE,
  RT_INTERVAL_WGSL,
  RT_T_INF,
  RT_T_NORMAL_MIN,
  bundleDirection,
  cascadeEnd,
  cascadeIndex,
  cascadeStart,
  inInterval,
  intervalBounds,
  intervalRayCount,
  intervalResultBytes,
  nextDownF32,
  nextUpF32,
  octEqualAreaDecode,
  packIntervalParams,
  resolveIntervalOptions,
  rtIntervalArgsKernelWgsl,
  rtIntervalEntries,
  rtIntervalEntryWgsl,
  rtIntervalKernelWgsl,
  rtModuleWgsl,
  type IntervalConvention,
  type IntervalKernelOptions,
} from './index.ts';

const noFeatures = buildCaps({ features: new Set(), limits: {} }, new Set());
const allFeatures = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const KERNEL_BYTE_LIMIT = 50_000;
const layout = { normals: true, uvs: true, positions: true };
const f = Math.fround;

function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('f32 neighbours', () => {
  it('nextUp / nextDown move exactly one ulp and invert each other', () => {
    for (const x of [RT_T_NORMAL_MIN, 1e-20, 1e-3, 0.5, 1, 1.5, 2, 3.14159, 1000, 1e20, 1e38]) {
      const x32 = f(x);
      const up = nextUpF32(x32);
      const down = nextDownF32(x32);
      expect(up).toBeGreaterThan(x32);
      expect(down).toBeLessThan(x32);
      expect(nextDownF32(up)).toBe(x32);
      expect(nextUpF32(down)).toBe(x32);
      // no f32 lies strictly between x and its neighbour
      expect(f((x32 + up) / 2) === x32 || f((x32 + up) / 2) === up).toBe(true);
    }
    expect(nextUpF32(1)).toBe(1 + 2 ** -23);
    expect(nextDownF32(1)).toBe(1 - 2 ** -24);
  });
});

describe('interval bounds', () => {
  it('open-low (t0, t1]: a hit at t0 is excluded, at t1 included', () => {
    const t0 = f(0.75);
    const t1 = f(3.5);
    expect(inInterval(t0, t0, t1)).toBe(false);
    expect(inInterval(nextUpF32(t0), t0, t1)).toBe(true);
    expect(inInterval(t1, t0, t1)).toBe(true);
    expect(inInterval(nextUpF32(t1), t0, t1)).toBe(false);
  });

  it('closed-low [t0, t1): a hit at t0 is included, at t1 excluded', () => {
    const t0 = f(0.75);
    const t1 = f(3.5);
    expect(inInterval(t0, t0, t1, 'closed-low')).toBe(true);
    expect(inInterval(nextDownF32(t0), t0, t1, 'closed-low')).toBe(false);
    expect(inInterval(nextDownF32(t1), t0, t1, 'closed-low')).toBe(true);
    expect(inInterval(t1, t0, t1, 'closed-low')).toBe(false);
  });

  it('consecutive intervals partition: every t is in exactly one, for both conventions, including every boundary', () => {
    const rand = random(7);
    for (const convention of ['open-low', 'closed-low'] as IntervalConvention[]) {
      for (let trial = 0; trial < 200; trial++) {
        const cuts = [0, ...Array.from({ length: 5 }, () => f(rand() * 40 + 1e-3)).sort((a, b) => a - b), RT_T_INF];
        const probes: number[] = [];
        for (const c of cuts) probes.push(c, nextUpF32(Math.max(c, RT_T_NORMAL_MIN)), nextDownF32(Math.max(c, RT_T_NORMAL_MIN)));
        for (let i = 0; i < 20; i++) probes.push(f(rand() * 45));
        for (const t of probes) {
          if (!(t >= RT_T_NORMAL_MIN) || t >= RT_T_INF) continue;
          let owners = 0;
          for (let k = 0; k + 1 < cuts.length; k++) if (inInterval(t, cuts[k] as number, cuts[k + 1] as number, convention)) owners++;
          // (0, a] misses t = 0 and closed-low [0, a) takes it; both are covered, t >= min normal always has exactly one owner
          expect(owners, `${convention} t=${t} cuts=${cuts.join(',')}`).toBe(1);
        }
      }
    }
  });

  it('empty intervals are flagged and never contain anything', () => {
    for (const convention of ['open-low', 'closed-low'] as IntervalConvention[]) {
      expect(intervalBounds(2, 2, convention).empty).toBe(true);
      expect(intervalBounds(3, 2, convention).empty).toBe(true);
      expect(intervalBounds(0, 0, convention).empty).toBe(true);
      expect(intervalBounds(0, 1e-39, convention).empty).toBe(true);
      expect(intervalBounds(0, 1, convention).empty).toBe(false);
    }
    expect(inInterval(2, 2, 2)).toBe(false);
  });

  it('t1 = infinity is clamped to the largest finite f32; t0 = 0 never reports t = 0 with open-low', () => {
    expect(intervalBounds(1, Infinity).tMax).toBe(RT_T_INF);
    expect(intervalBounds(0, 1).tMin).toBe(RT_T_NORMAL_MIN);
    expect(inInterval(0, 0, 1)).toBe(false);
    expect(inInterval(0, 0, 1, 'closed-low')).toBe(true);
  });
});

describe('cascades', () => {
  it('end = t0 * growth^n and start = previous end', () => {
    expect(cascadeEnd(0, 0.5, 4)).toBe(0.5);
    expect(cascadeEnd(3, 0.5, 4)).toBe(32);
    expect(cascadeStart(0, 0.5, 4)).toBe(0);
    expect(cascadeStart(2, 0.5, 4)).toBe(2);
  });

  it('cascadeIndex agrees with the open-low partition at every boundary, for growths that are not powers of two', () => {
    const rand = random(11);
    for (const growth of [2, 4, 3.3, 1.7]) {
      const count = 6;
      const t0 = f(0.3 + rand());
      const bounds = Array.from({ length: count }, (_, n) => cascadeEnd(n, t0, growth));
      const samples = [...bounds, ...bounds.map(nextUpF32), ...bounds.map(nextDownF32), ...Array.from({ length: 50 }, () => f(rand() * (bounds[count - 1] as number) * 1.3))];
      for (const t of samples) {
        const n = cascadeIndex(t, t0, growth, count);
        if (n === count) {
          expect(t).toBeGreaterThan(bounds[count - 1] as number);
          continue;
        }
        expect(inInterval(t, cascadeStart(n, t0, growth), cascadeEnd(n, t0, growth)), `growth ${growth} t ${t} n ${n}`).toBe(true);
        for (let m = 0; m < count; m++) if (m !== n) expect(inInterval(t, cascadeStart(m, t0, growth), cascadeEnd(m, t0, growth))).toBe(false);
      }
    }
  });
});

describe('equal-area octahedral directions', () => {
  it('unit length everywhere, corners and edges included', () => {
    const rand = random(3);
    for (let i = 0; i < 2000; i++) {
      const [x, y, z] = octEqualAreaDecode(rand(), rand());
      expect(Math.hypot(x, y, z)).toBeCloseTo(1, 12);
    }
    for (const [u, v] of [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5], [0.5, 0], [0, 0.5], [1, 0.5], [0.5, 1]] as const) {
      const [x, y, z] = octEqualAreaDecode(u, v);
      expect(Math.hypot(x, y, z)).toBeCloseTo(1, 12);
    }
    expect(octEqualAreaDecode(0.5, 0.5)[2]).toBe(1);
  });

  it('equal area: the z of the 32 x 32 cell centres is uniform on [-1, 1] (Archimedes), and cells are equal in solid angle', () => {
    const rand = random(5);
    const zs: number[] = Array.from({ length: 32768 }, () => octEqualAreaDecode(rand(), rand())[2]);
    const bins = new Array<number>(16).fill(0);
    for (const z of zs) bins[Math.min(15, Math.floor(((z + 1) / 2) * 16))]!++;
    for (const count of bins) expect(Math.abs(count - zs.length / 16)).toBeLessThan((zs.length / 16) * 0.08);
    // solid angle of a cell by a fine sampling of its four corners' fan: area of the quad via two spherical triangles
    const solid = (a: number[], b: number[], c: number[]): number => {
      const dot = (p: number[], q: number[]) => (p[0] as number) * (q[0] as number) + (p[1] as number) * (q[1] as number) + (p[2] as number) * (q[2] as number);
      const cross = (p: number[], q: number[]) => [(p[1] as number) * (q[2] as number) - (p[2] as number) * (q[1] as number), (p[2] as number) * (q[0] as number) - (p[0] as number) * (q[2] as number), (p[0] as number) * (q[1] as number) - (p[1] as number) * (q[0] as number)];
      return 2 * Math.atan2(Math.abs(dot(a, cross(b, c))), 1 + dot(a, b) + dot(b, c) + dot(c, a));
    };
    const m = 16;
    const sub = 4;
    const fine = m * sub;
    const corner = (i: number, j: number) => octEqualAreaDecode(i / fine, j / fine);
    const areas = new Array<number>(m * m).fill(0);
    for (let j = 0; j < fine; j++) for (let i = 0; i < fine; i++) {
      areas[Math.floor(j / sub) * m + Math.floor(i / sub)]! += solid(corner(i, j), corner(i + 1, j), corner(i + 1, j + 1)) + solid(corner(i, j), corner(i + 1, j + 1), corner(i, j + 1));
    }
    const mean = (4 * Math.PI) / (m * m);
    expect(areas.reduce((a, b) => a + b, 0)).toBeCloseTo(4 * Math.PI, 2);
    for (const a of areas) expect(Math.abs(a - mean) / mean).toBeLessThan(0.03);
  });

  it('a 2 x 2 split refines a cell: the four child centres sit around the parent centre (nested grids)', () => {
    for (const [x, y] of [[0, 0], [3, 1], [5, 6], [7, 7]] as const) {
      const parent = bundleDirection(x, y, 8);
      let mean = [0, 0, 0];
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]] as const) {
        const child = bundleDirection(2 * x + dx, 2 * y + dy, 16);
        mean = [mean[0]! + child[0], mean[1]! + child[1], mean[2]! + child[2]];
      }
      const len = Math.hypot(...(mean as [number, number, number]));
      const dotp = (mean[0]! * parent[0] + mean[1]! * parent[1] + mean[2]! * parent[2]) / len;
      expect(dotp).toBeGreaterThan(0.99);
    }
  });
});

describe('generated WGSL', () => {
  const combos = (): IntervalKernelOptions[] => {
    const list: IntervalKernelOptions[] = [];
    for (const source of ['rays', 'probes'] as const)
      for (const output of ['hit', 'distance', 'visible'] as const)
        for (const convention of ['open-low', 'closed-low'] as const)
          for (const indirect of [false, true])
            for (const triangleLeaves of ['records', 'indexed'] as const)
              for (const hitRecord of output === 'hit' ? (['full', 'compact'] as const) : (['full'] as const)) list.push({ source, output, convention, indirect, triangleLeaves, hitRecord, workgroupSize: 64 });
    const scene = { layout, packing: 'packed' as const };
    for (const source of ['rays', 'probes'] as const)
      for (const shadeMode of ['flat', 'lambert', 'full'] as const)
        for (const packing of ['packed', 'split'] as const)
          for (const indirect of [false, true]) list.push({ source, output: 'shaded', shadeMode, indirect, scene: { ...scene, packing }, workgroupSize: 64 });
    return list;
  };

  it('every combination preprocesses cleanly, stays under the size cap and keeps balanced brackets', () => {
    let largest = 0;
    for (const combo of combos()) {
      for (const caps of [noFeatures, allFeatures]) {
        const code = rtIntervalKernelWgsl(caps, combo);
        const what = JSON.stringify(combo);
        expect(code, what).not.toMatch(/^\s*#(if|elif|else|endif)\b/m);
        expect(code, what).toContain('fn rtInterval(');
        expect(code, what).toContain('fn rtClosestInterval');
        expect(balanced(code), what).toBe(true);
        largest = Math.max(largest, code.length);
        expect(code.length, what).toBeLessThan(KERNEL_BYTE_LIMIT);
        // bindings: group 1 binding numbers equal the layout entries
        const group1 = [...code.matchAll(/@group\(1\)\s*@binding\((\d+)\)/g)].map((m) => Number(m[1]));
        expect(group1.sort(), what).toEqual(rtIntervalEntries(resolveIntervalOptions(combo)).map((e) => e.binding));
      }
    }
    expect(largest).toBeLessThan(KERNEL_BYTE_LIMIT);
  });

  it('the interval functions are only in the module when asked (default kernels are unchanged)', () => {
    expect(rtModuleWgsl(noFeatures)).not.toContain('rtClosestInterval');
    expect(rtModuleWgsl(noFeatures, { intervals: true })).toContain('fn rtClosestInterval(');
    expect(rtModuleWgsl(noFeatures, { intervals: true })).toContain('fn rtVisibleIntervalLo(');
  });

  it('declares only rt-prefixed names beyond the bvh walk, and no reserved word as an identifier', () => {
    const walk = new Set(declared(traceModuleWgsl({})));
    const kernel = rtIntervalKernelWgsl(noFeatures, { source: 'probes', output: 'shaded', scene: { layout }, indirect: true, minify: false });
    const strays = declared(kernel).filter((name) => !walk.has(name) && !/^(rt|Rt|RT_)/.test(name));
    expect(strays).toEqual([]);
    for (const text of [RT_INTERVAL_WGSL, rtIntervalEntryWgsl(resolveIntervalOptions({ source: 'probes', output: 'hit' })), rtIntervalEntryWgsl(resolveIntervalOptions({ source: 'rays', output: 'visible', convention: 'closed-low' })), rtIntervalEntryWgsl(resolveIntervalOptions({ source: 'probes', output: 'shaded', scene: { layout } })), rtIntervalArgsKernelWgsl(noFeatures, resolveIntervalOptions({ source: 'probes', output: 'hit', indirect: true }))]) {
      for (const name of declaredIdentifiers(text)) expect(RESERVED.has(name), name).toBe(false);
    }
  });

  it('RtIntervalParams is 48 bytes and the packer writes the fields where the struct declares them', () => {
    const code = RT_INTERVAL_WGSL + rtIntervalArgsKernelWgsl(noFeatures, resolveIntervalOptions({ source: 'rays', output: 'hit', indirect: true }));
    const struct = /struct RtIntervalParams \{([^}]*)\}/.exec(code)?.[1] ?? '';
    expect([...struct.matchAll(/(\w+):\s*(\w+)/g)].map((m) => `${m[1]}:${m[2]}`)).toEqual(['count:u32', 'mask:u32', 'capacity:u32', 'countWord:u32', 't0:f32', 't1:f32', 'dirs:u32', 'seed:u32', 'jitter:vec2f', 'pad:vec2f']);
    const bytes = packIntervalParams({ count: 5, mask: 7, capacity: 9, countWord: 2, t0: 0.5, t1: 2, directions: 4, seed: 99, jitter: [0.25, 0.75] });
    expect(bytes.byteLength).toBe(INTERVAL_PARAMS_BYTES);
    const u = new Uint32Array(bytes);
    const fl = new Float32Array(bytes);
    expect([u[0], u[1], u[2], u[3], fl[4], fl[5], u[6], u[7], fl[8], fl[9]]).toEqual([5, 7, 9, 2, 0.5, 2, 4, 99, 0.25, 0.75]);
    expect(INTERVAL_RING_STRIDE % 256).toBe(0);
  });

  it('result sizes and ray counts', () => {
    expect(intervalResultBytes('hit', 'full')).toBe(40);
    expect(intervalResultBytes('hit', 'compact')).toBe(24);
    expect(intervalResultBytes('shaded')).toBe(16);
    expect(intervalResultBytes('distance')).toBe(4);
    expect(intervalRayCount('probes', 100, 8)).toBe(6400);
    expect(intervalRayCount('rays', 100, 8)).toBe(100);
  });

  it("'shaded' needs the scene and the others refuse it", () => {
    expect(() => resolveIntervalOptions({ source: 'rays', output: 'shaded' })).toThrow(/scene/);
    expect(() => resolveIntervalOptions({ source: 'rays', output: 'hit', scene: { layout } })).toThrow(/shaded/);
  });
});
