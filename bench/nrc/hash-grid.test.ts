import { describe, expect, it } from 'vitest';
import { RESERVED, balanced, declaredIdentifiers } from '../../packages/raytracer/src/wgsl-names.testutil.ts';
import { DEFAULT_HASH_GRID, hashGridArgs, hashGridInsertWgsl, hashGridStorage, hashGridWgsl } from './hash-grid.ts';

describe('hash grid (M7)', () => {
  it('generates balanced WGSL with no reserved identifiers', () => {
    for (const [fb, pyramid, active, variance] of [[0, false, false, false], [2, false, false, false], [2, true, false, false], [2, true, true, false], [2, true, false, true], [2, true, true, true]] as const) {
      for (const [name, code] of Object.entries(hashGridWgsl(fb, pyramid, active, variance))) {
        expect(balanced(code), name).toBe(true);
        expect([...declaredIdentifiers(code)].filter((n) => RESERVED.has(n)), name).toEqual([]);
      }
    }
    for (const active of [false, true]) {
      for (const variance of [false, true]) {
        for (const [name, code] of Object.entries(hashGridWgsl(2, true, active, variance, true, true))) {
          expect(balanced(code), name).toBe(true);
          expect([...declaredIdentifiers(code)].filter((n) => RESERVED.has(n)), name).toEqual([]);
        }
      }
    }
  });

  it('bakes the fallback count and the bucket loop bounds in as constants (Tint loop counters on D3D12)', () => {
    const { lookup, update } = hashGridWgsl(3);
    expect(lookup).toContain('const HG_FALLBACK = 3u;');
    expect(lookup).toMatch(/for \(var i = 0u; i < 8u; i\+\+\)/);
    expect(update).toMatch(/for \(var i = 0u; i < 8u; i\+\+\)/);
  });

  it('maps page args onto the settings and leaves the rest at the defaults', () => {
    expect(hashGridArgs({})).toEqual({});
    expect(DEFAULT_HASH_GRID.activeMaintenance).toBe(false);
    expect(DEFAULT_HASH_GRID.directionalSh).toBe(false);
    expect(DEFAULT_HASH_GRID.varianceThreshold).toBe(-1);
    expect(hashGridArgs({ hgVariance: '0.1' })).toEqual({ varianceThreshold: 0.1 });
    expect(hashGridArgs({ hgActive: '1' })).toEqual({ activeMaintenance: true });
    expect(hashGridArgs({ hgActive: '0' })).toEqual({ activeMaintenance: false });
    expect(hashGridArgs({ hgMinSamples: '4', hgNormals: '0', hgCellPx: '16' })).toEqual({ minSamples: 4, normalBins: false, cellPixels: 16 });
    expect({ ...DEFAULT_HASH_GRID, ...hashGridArgs({ hgLog2: '22' }) }.log2Entries).toBe(22);
    expect(hashGridArgs({ hgDirectionalSh: '1', hgBudgetBytes: '10485844' })).toEqual({ directionalSh: true, memoryBudgetBytes: 10485844 });
  });

  it('fits all directional controls into the same explicit table allocation', () => {
    const memoryBudgetBytes = 10 * 1024 ** 2 + 84;
    const rgb = hashGridStorage({ memoryBudgetBytes });
    const bins = hashGridStorage({ memoryBudgetBytes, dirBins: 4 });
    const sh = hashGridStorage({ memoryBudgetBytes, directionalSh: true });
    expect(rgb).toEqual(bins);
    expect(rgb).toMatchObject({ entries: 262144, bytes: memoryBudgetBytes, bytesPerEntry: 40, updateAtomicAddsPerCell: 4 });
    expect(sh).toMatchObject({ entries: 40960, bytes: memoryBudgetBytes, bytesPerEntry: 256, directionalUpdateAtomicAddsPerCell: 21 });
    expect(() => hashGridStorage({ directionalSh: true, dirBins: 4 })).toThrow('mutually exclusive');
  });

  it('emits the insert with literal loop bounds and no unbounded retry (pmndrs/pathtracer#1)', () => {
    const insert = hashGridInsertWgsl();
    expect(insert).not.toMatch(/\bloop\s*\{/);
    expect(insert).toMatch(/for \(var t = 0u; t < 4u; t\+\+\)/);
    expect(hashGridWgsl(2, true).update).toContain(insert);
    expect(balanced(insert)).toBe(true);
  });

  //* Insert protocol model (pmndrs/pathtracer#1)
  // A step machine per thread, one step per atomic op, run under a random scheduler with a weak exchange that fails spuriously.
  // `current` mirrors `hashGridInsertWgsl` op for op; `dd771c8` is the version gi ported, which let one key take two slots.
  type Protocol = (keys: Uint32Array, key: number, tries: number) => Generator<void, number, void>;
  interface Weak { p: number; rand: () => number }
  const weak = { p: 0, rand: Math.random } as Weak;
  const cas = (keys: Uint32Array, slot: number, key: number) => {
    const old = keys[slot];
    if (old !== 0) return { exchanged: false, old };
    if (weak.rand() < weak.p) return { exchanged: false, old: 0 };
    keys[slot] = key;
    return { exchanged: true, old: 0 };
  };
  const current: Protocol = function* (keys, key, tries) {
    for (let i = 0; i < keys.length; i++) { yield; if (keys[i] === key) return i; }
    for (let i = 0; i < keys.length; i++) {
      yield;
      let owner = keys[i];
      for (let t = 0; t < tries; t++) {
        if (owner !== 0) break;
        yield;
        const res = cas(keys, i, key);
        if (res.exchanged) return i;
        owner = res.old;
      }
      if (owner === 0) return -1;
      if (owner === key) return i;
    }
    return -1;
  };
  const dd771c8: Protocol = function* (keys, key) {
    for (let i = 0; i < keys.length; i++) { yield; if (keys[i] === key) return i; }
    for (let i = 0; i < keys.length; i++) {
      yield;
      if (keys[i] === 0) {
        yield;
        let res = cas(keys, i, key);
        if (!res.exchanged && res.old === 0) { yield; res = cas(keys, i, key); }
        if (res.exchanged || res.old === key) return i;
      }
    }
    return -1;
  };
  const mulberry = (seed: number) => () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  // Many threads insert a few cells into one bucket at once (the issue's suggested check). Returns the bucket, each thread's result.
  const run = (protocol: Protocol, seed: number, threads: number, distinct: number, bucket: number, p: number, tries = 4) => {
    const rand = mulberry(seed);
    weak.p = p;
    weak.rand = rand;
    const keys = new Uint32Array(bucket);
    const want = Array.from({ length: threads }, (_, t) => 1 + (t % distinct));
    const live = want.map((k) => ({ k, it: protocol(keys, k, tries), slot: undefined as number | undefined }));
    let pending = live.length;
    while (pending > 0) {
      const th = live[Math.floor(rand() * live.length)]!;
      if (th.slot !== undefined) continue;
      const step = th.it.next();
      if (step.done) { th.slot = step.value; pending--; }
    }
    return { keys, results: live.map(({ k, slot }) => ({ k, slot: slot! })) };
  };
  const duplicates = (keys: Uint32Array) => keys.length - new Set(keys).size - Math.max(0, keys.filter((k) => k === 0).length - 1);

  it('the dd771c8 insert gives one cell two slots under concurrent inserts (the reported bug)', () => {
    let seen = 0;
    for (let seed = 0; seed < 200; seed++) if (duplicates(run(dd771c8, seed, 64, 3, 8, 0).keys) > 0) seen++;
    expect(seen).toBeGreaterThan(0);
  });

  it('the insert never gives a cell two slots and never refuses while the bucket has room', () => {
    for (const p of [0, 0.3, 0.9]) {
      for (let seed = 0; seed < 300; seed++) {
        const { keys, results } = run(current, seed, 64, 3, 8, p);
        expect(duplicates(keys), `p ${p} seed ${seed}`).toBe(0);
        for (const { k, slot } of results) {
          if (slot >= 0) expect(keys[slot]).toBe(k);
          // A refusal with room left can only come from `tries` spurious failures in a row on an empty slot.
          else expect(p, `p ${p} seed ${seed}: refused key ${k} with room`).toBeGreaterThan(0);
        }
        if (p === 0) expect(results.every((r) => r.slot >= 0)).toBe(true);
      }
    }
  });

  it('refuses only when the bucket is full of other cells (more cells than slots)', () => {
    for (let seed = 0; seed < 300; seed++) {
      const { keys, results } = run(current, seed, 64, 12, 8, 0);
      expect(duplicates(keys)).toBe(0);
      expect(keys.every((k) => k !== 0)).toBe(true);
      for (const { k, slot } of results) {
        if (slot >= 0) expect(keys[slot]).toBe(k);
        else expect([...keys]).not.toContain(k);
      }
    }
  });
});
