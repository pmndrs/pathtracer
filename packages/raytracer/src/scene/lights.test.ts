import { describe, expect, it } from 'vitest';
import { LIGHT_RECORD_BYTES, LIGHT_RECORD_VEC4S, SCENE_NONE } from './constants.ts';
import {
  aliasProbabilityOf,
  buildAliasTable,
  makeEnvironment,
  makeSun,
  packLight,
  sampleAlias,
  trianglePower,
  unpackLight,
  type RtTriangleLight,
} from './lights.ts';

describe('alias table', () => {
  const cases: Record<string, number[]> = {
    'two equal': [1, 1],
    'skewed': [1, 2, 3, 4],
    'one dominant': [1000, 1, 1, 1, 1],
    'with zeros': [0, 5, 0, 5, 1],
    'single': [3],
    'powers of two': [1, 2, 4, 8, 16, 32],
    'wide range': [1e-6, 1, 1e6],
  };

  for (const [name, weights] of Object.entries(cases)) {
    it(`${name}: pmf sums to 1 and the table realises the weights`, () => {
      const table = buildAliasTable(weights);
      const total = weights.reduce((a, b) => a + b, 0);
      let sum = 0;
      weights.forEach((w, i) => {
        sum += table.pmf[i]!;
        expect(table.pmf[i]).toBe(Math.fround(w / total));
        // The table's own probability of drawing i (from prob + alias) matches the weight; this checks the construction, not the pmf array.
        expect(aliasProbabilityOf(table, i)).toBeCloseTo(w / total, 6);
      });
      expect(sum).toBeCloseTo(1, 6);
      expect(table.total).toBe(total);
    });

    it(`${name}: stratified samples hit each index with its probability`, () => {
      const table = buildAliasTable(weights);
      const total = weights.reduce((a, b) => a + b, 0);
      const n = 200_000;
      const counts = new Array<number>(weights.length).fill(0);
      let bad = 0;
      for (let k = 0; k < n; k++) {
        const sample = sampleAlias(table, (k + 0.5) / n);
        counts[sample.index]!++;
        if (sample.pmf !== table.pmf[sample.index] || !(sample.u >= 0 && sample.u < 1)) bad++;
      }
      expect(bad).toBe(0);
      weights.forEach((w, i) => expect(counts[i]! / n, `${name}[${i}]`).toBeCloseTo(w / total, 3));
      // A zero-weight light is never picked.
      weights.forEach((w, i) => w === 0 && expect(counts[i]).toBe(0));
    });
  }

  it('random (white) samples also follow the weights within sampling error', () => {
    const weights = [5, 1, 3, 0.5, 10];
    const table = buildAliasTable(weights);
    let seed = 99;
    const rnd = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 2 ** 32;
    };
    const n = 400_000;
    const counts = new Array<number>(weights.length).fill(0);
    for (let k = 0; k < n; k++) counts[sampleAlias(table, rnd()).index]!++;
    const total = weights.reduce((a, b) => a + b, 0);
    weights.forEach((w, i) => expect(Math.abs(counts[i]! / n - w / total)).toBeLessThan(0.004));
  });

  it('the leftover u is uniform given the pick (so it can feed the next dimension)', () => {
    const table = buildAliasTable([1, 2, 3, 4]);
    const buckets = new Array<number>(10).fill(0);
    const n = 100_000;
    for (let k = 0; k < n; k++) buckets[Math.floor(sampleAlias(table, (k + 0.5) / n).u * 10)]!++;
    for (const count of buckets) expect(count / n).toBeCloseTo(0.1, 1);
  });

  it('all-zero weights give a uniform table; empty gives no light', () => {
    const table = buildAliasTable([0, 0, 0]);
    expect(Array.from(table.pmf)).toEqual([Math.fround(1 / 3), Math.fround(1 / 3), Math.fround(1 / 3)]);
    expect(table.total).toBe(0);
    expect(sampleAlias(buildAliasTable([]), 0.5)).toEqual({ index: SCENE_NONE, pmf: 0, u: 0.5 });
  });

  it('u = 0 and the largest u below 1 stay in range', () => {
    const table = buildAliasTable([1, 2, 3]);
    expect(sampleAlias(table, 0).index).toBeLessThan(3);
    expect(sampleAlias(table, 1 - 2 ** -24).index).toBeLessThan(3);
    expect(sampleAlias(table, 1).index).toBeLessThan(3);
  });

  it('rejects negative, NaN and infinite weights', () => {
    expect(() => buildAliasTable([1, -1])).toThrow(RangeError);
    expect(() => buildAliasTable([1, NaN])).toThrow(RangeError);
    expect(() => buildAliasTable([1, Infinity])).toThrow(RangeError);
  });
});

describe('light records', () => {
  const triangle: RtTriangleLight = {
    kind: 'triangle',
    instance: 7,
    triangle: 123456,
    material: 3,
    p0: [1, 2, 3],
    edge1: [0.5, 0, 0],
    edge2: [0, 0, 0.25],
    normal: [0, -1, 0],
    area: 0.0625,
    radiance: [18.4, 14, 6.75],
    twoSided: true,
    power: 0,
  };

  it('96 bytes, 6 vec4u', () => {
    expect(LIGHT_RECORD_BYTES).toBe(LIGHT_RECORD_VEC4S * 16);
  });

  it('every kind round-trips through its record', () => {
    const words = new Uint32Array(LIGHT_RECORD_VEC4S * 4);
    packLight(triangle, 0.25, 0.5, 2, words, 0);
    const t = unpackLight(words);
    expect(t.light).toMatchObject({ kind: 'triangle', instance: 7, triangle: 123456, material: 3, p0: [1, 2, 3], edge1: [0.5, 0, 0], edge2: [0, 0, 0.25], normal: [0, -1, 0], area: 0.0625, twoSided: true });
    expect((t.light as RtTriangleLight).radiance).toEqual([Math.fround(18.4), 14, 6.75]);
    expect([t.pmf, t.aliasProb, t.alias]).toEqual([0.25, 0.5, 2]);

    const sun = makeSun({ direction: [0, -1, 0], irradiance: [50, 46, 40], angularRadius: 0.00465 }, 10);
    packLight(sun, 0.1, 1, 5, words, 0);
    const s = unpackLight(words).light;
    expect(s).toMatchObject({ kind: 'sun', direction: [0, 1, 0], angularRadius: Math.fround(0.00465), irradiance: [50, 46, 40] });
    expect((s as typeof sun).solidAngle).toBe(Math.fround(2 * Math.PI * (1 - Math.cos(0.00465))));

    const env = makeEnvironment({ radiance: [2, 2.3, 3], scale: 2, rotationY: 1, mapSlot: 4 }, 10);
    packLight(env, 0.9, 0.3, 0, words, 0);
    expect(unpackLight(words).light).toMatchObject({ kind: 'environment', scale: 2, rotationY: 1, mapSlot: 4 });
    packLight(makeEnvironment({ radiance: [1, 1, 1] }, 1), 1, 1, 0, words, 0);
    expect((unpackLight(words).light as { mapSlot: number }).mapSlot).toBe(SCENE_NONE);
  });

  it('unknown kinds are rejected', () => {
    expect(() => unpackLight(new Uint32Array(24).fill(9))).toThrow(RangeError);
  });
});

describe('power', () => {
  it('a diffuse emitter radiates pi * area * Y(L); two-sided doubles it', () => {
    expect(trianglePower(2, [1, 1, 1], false)).toBeCloseTo(2 * Math.PI, 12);
    expect(trianglePower(2, [1, 1, 1], true)).toBeCloseTo(4 * Math.PI, 12);
    expect(trianglePower(1, [0, 1, 0], false)).toBeCloseTo(Math.PI * 0.7152, 12);
  });

  it('the sun is the flux through the scene sphere cross section; an environment is 4 pi times that per unit radiance', () => {
    const sun = makeSun({ direction: [0, -1, 0], irradiance: [2, 2, 2] }, 3);
    expect(sun.power).toBeCloseTo(2 * Math.PI * 9, 10);
    expect(makeSun({ direction: [0, -1, 0], irradiance: [2, 2, 2] }, 3, 0.5).power).toBeCloseTo(Math.PI * 9, 10);
    const env = makeEnvironment({ radiance: [1, 1, 1] }, 3);
    expect(env.power).toBeCloseTo(4 * Math.PI * Math.PI * 9, 10);
    expect(makeEnvironment({ radiance: [1, 1, 1] }, 3, 0).power).toBe(0);
  });

  it('the sun direction points toward the sun and is normalised; bad suns are rejected', () => {
    const sun = makeSun({ direction: [0, -2, 0], irradiance: [1, 1, 1] }, 1);
    expect(sun.direction).toEqual([0, 1, 0]);
    expect(() => makeSun({ direction: [0, 0, 0], irradiance: [1, 1, 1] }, 1)).toThrow(RangeError);
    expect(() => makeSun({ direction: [0, 1, 0], irradiance: [1, 1, 1], angularRadius: 2 }, 1)).toThrow(RangeError);
  });
});

