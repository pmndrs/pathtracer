import { describe, expect, it } from 'vitest';
import { SCENE_NONE } from '../scene/constants.ts';
import { buildSceneCopy, type SceneCopy, type SceneCopyInput } from '../scene/scene-copy.ts';
import { createMaterial } from '../scene/material.ts';
import type { Vec3Tuple } from '../scene/lights.ts';
import { builtinBsdf, ggxD, makeShadePoint, resolveShadeOptions, shade, shadeAlbedo, shadeDiffuseFromIrradiance, shadeF0, shadeMiss, shadeTotal, sampleCosineAbout, type ShadePoint, type ShadeVisible } from './shade.ts';

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

type M16 = number[];
/** A unit quad in XZ, authored normal -Y, scaled to a x b and centred at (0, h, 0): an emitter facing down. */
function ceilingQuad(a: number, b: number, h: number): { mesh: SceneCopyInput['meshes'][number]; matrix: M16 } {
  return {
    mesh: { positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]) },
    matrix: [a, 0, 0, 0, 0, 1, 0, 0, 0, 0, b, 0, -a / 2, h, -b / 2, 1],
  };
}

const always: ShadeVisible = () => true;
const surface = (m = createMaterial({ baseColor: [0.5, 0.5, 0.5], specularWeight: 0 })): ShadePoint =>
  makeShadePoint({ material: m, position: [0, 0, 0], shadingNormal: [0, 1, 0], geometricNormal: [0, 1, 0], frontFace: true, rayDirection: [0, -1, 0] });

/** Mean of `shade` over random u. */
function mean(copy: SceneCopy, p: ShadePoint, mode: 'flat' | 'lambert' | 'full', n: number, options = {}, visible: ShadeVisible = always, seed = 7): { mean: number[]; std: number[] } {
  const r = rng(seed);
  const sum = [0, 0, 0];
  const sq = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const s = shade(copy, p, mode, [r(), r(), r()], visible, {}, options);
    const t = shadeTotal(s);
    for (let c = 0; c < 3; c++) {
      sum[c]! += t[c] as number;
      sq[c]! += (t[c] as number) ** 2;
    }
  }
  return { mean: sum.map((v) => v / n), std: sq.map((v, c) => Math.sqrt(Math.max(0, v / n - (sum[c]! / n) ** 2))) };
}

/** Form factor of a rectangle a x b centred above a point at height h, normals facing each other (differential area to rectangle). */
function centredFormFactor(a: number, b: number, h: number): number {
  const corner = (x: number, y: number) => (1 / (2 * Math.PI)) * ((x / Math.sqrt(1 + x * x)) * Math.atan(y / Math.sqrt(1 + x * x)) + (y / Math.sqrt(1 + y * y)) * Math.atan(x / Math.sqrt(1 + y * y)));
  return 4 * corner(a / 2 / h, b / 2 / h);
}

describe('rtShade twin: analytic cases', () => {
  it('Lambert plane under a quad light: albedo * formFactor * Le (1% at 60k samples)', () => {
    const { mesh, matrix } = ceilingQuad(2, 1.2, 1.5);
    const copy = buildSceneCopy({ meshes: [mesh], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: matrix, material: 0 }], materials: [{ emissionColor: [10, 8, 6] }] });
    const ff = centredFormFactor(2, 1.2, 1.5);
    const got = mean(copy, surface(), 'lambert', 60000);
    for (const [c, le] of [10, 8, 6].entries()) expect(got.mean[c]! / (0.5 * ff * le)).toBeCloseTo(1, 1);
    for (const [c, le] of [10, 8, 6].entries()) expect(Math.abs(got.mean[c]! / (0.5 * ff * le) - 1)).toBeLessThan(0.012);
    // FULL with a purely diffuse material is the same estimator.
    const full = mean(copy, surface(), 'full', 60000, {}, always, 7);
    expect(Math.abs(full.mean[0]! / got.mean[0]! - 1)).toBeLessThan(1e-9);
  });

  it('Lambert plane under a delta sun and a uniform sky: albedo/pi * E cos + albedo * Ls, with either sky sampler', () => {
    const copy = buildSceneCopy({
      meshes: [ceilingQuad(1, 1, 1).mesh],
      instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: ceilingQuad(1, 1, 1).matrix, material: 0, visible: false }],
      materials: [{}],
      sun: { direction: [-0.3, -0.8, -0.2], irradiance: [3, 3, 3], angularRadius: 0 },
      environment: { radiance: [0.2, 0.2, 0.2] },
    });
    const d = Math.hypot(0.3, 0.8, 0.2);
    const expected = (0.5 / Math.PI) * 3 * (0.8 / d) + 0.5 * 0.2;
    const std: Record<string, number> = {};
    for (const envSampling of ['uniform', 'cosine'] as const) {
      const got = mean(copy, surface(), 'lambert', 80000, { envSampling });
      expect(Math.abs(got.mean[1]! / expected - 1), envSampling).toBeLessThan(0.015);
      std[envSampling] = got.std[1] as number;
    }
    // The cosine sampler removes the sky's share of the variance; the sun/sky selection noise stays.
    expect(std.cosine!).toBeLessThan(std.uniform!);
  });

  it('a sun disc with an angular radius gives the same mean as a delta sun of the same irradiance', () => {
    const base = { meshes: [ceilingQuad(1, 1, 1).mesh], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: ceilingQuad(1, 1, 1).matrix, material: 0, visible: false }], materials: [{}] };
    const delta = buildSceneCopy({ ...base, sun: { direction: [-0.3, -0.8, -0.2], irradiance: [3, 3, 3], angularRadius: 0 } });
    const disc = buildSceneCopy({ ...base, sun: { direction: [-0.3, -0.8, -0.2], irradiance: [3, 3, 3], angularRadius: 0.05 } });
    const a = mean(delta, surface(), 'lambert', 20000).mean[0]!;
    const b = mean(disc, surface(), 'lambert', 20000).mean[0]!;
    expect(Math.abs(b / a - 1)).toBeLessThan(0.01);
  });

  it('shadow rays: a blocked sample contributes nothing and is reported; the mean scales with the visible fraction', () => {
    const { mesh, matrix } = ceilingQuad(2, 2, 1);
    const copy = buildSceneCopy({ meshes: [mesh], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: matrix, material: 0 }], materials: [{ emissionColor: [5, 5, 5] }] });
    const open = mean(copy, surface(), 'lambert', 20000).mean[0]!;
    const half = mean(copy, surface(), 'lambert', 20000, {}, (_o, dir) => dir[0] > 0).mean[0]!;
    expect(half / open).toBeGreaterThan(0.45);
    expect(half / open).toBeLessThan(0.55);
    const blocked = shade(copy, surface(), 'lambert', [0.3, 0.4, 0.5], () => false);
    expect(blocked.occluded).toBe(true);
    expect(blocked.direct).toEqual([0, 0, 0]);
    expect(blocked.light).not.toBe(SCENE_NONE);
  });
});

describe('rtShade twin: modes, emission, hooks', () => {
  const { mesh, matrix } = ceilingQuad(2, 2, 1);
  const copy = buildSceneCopy({ meshes: [mesh], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: matrix, material: 0 }], materials: [{ emissionColor: [5, 5, 5] }] });

  it('FLAT traces nothing: emission + indirect only, never calls visible', () => {
    let calls = 0;
    const s = shade(copy, surface(createMaterial({ emissionColor: [1, 2, 3] })), 'flat', [0.1, 0.2, 0.3], () => (calls++, true), { indirect: () => [0.1, 0.2, 0.3] });
    expect(calls).toBe(0);
    expect(s.direct).toEqual([0, 0, 0]);
    expect(shadeTotal(s)).toEqual([1.1, 2.2, 3.3]);
    const albedo = shade(copy, surface(createMaterial({ baseColor: [0.2, 0.4, 0.6] })), 'flat', [0, 0, 0], always, {}, { flat: 'albedo' });
    expect(albedo.direct).toEqual([0.2, 0.4, 0.6]);
  });

  it('LAMBERT and FULL trace at most one ray', () => {
    let calls = 0;
    const visible: ShadeVisible = () => (calls++, true);
    for (const mode of ['lambert', 'full'] as const) for (let i = 0; i < 50; i++) shade(copy, surface(), mode, [i / 50, 0.37, 0.61], visible);
    expect(calls).toBeLessThanOrEqual(100);
    expect(calls).toBeGreaterThan(0);
  });

  it('one-sided emission shows on the front face only; a back-face hit flips the normals', () => {
    const m = createMaterial({ emissionColor: [1, 1, 1] });
    const front = makeShadePoint({ material: m, position: [0, 0, 0], shadingNormal: [0, 1, 0], geometricNormal: [0, 1, 0], frontFace: true, rayDirection: [0, -1, 0] });
    const back = makeShadePoint({ material: m, position: [0, 0, 0], shadingNormal: [0, 1, 0], geometricNormal: [0, 1, 0], frontFace: false, rayDirection: [0, 1, 0] });
    expect(shade(copy, front, 'flat', [0, 0, 0], always).emission).toEqual([1, 1, 1]);
    expect(shade(copy, back, 'flat', [0, 0, 0], always).emission).toEqual([0, 0, 0]);
    expect(back.normal).toEqual([-0, -1, -0]);
    const two = makeShadePoint({ material: createMaterial({ emissionColor: [1, 1, 1], emissionTwoSided: true }), position: [0, 0, 0], shadingNormal: [0, 1, 0], geometricNormal: [0, 1, 0], frontFace: false, rayDirection: [0, 1, 0] });
    expect(shade(copy, two, 'flat', [0, 0, 0], always).emission).toEqual([1, 1, 1]);
  });

  it('the cache hook adds to every mode; an irradiance cache goes through albedo / pi', () => {
    const p = surface();
    const hooks = { indirect: (q: ShadePoint) => shadeDiffuseFromIrradiance(q, [Math.PI, 2 * Math.PI, 0]) };
    for (const mode of ['flat', 'lambert', 'full'] as const) expect(shade(copy, p, mode, [0.5, 0.5, 0.5], () => false, hooks).indirect).toEqual([0.5, 1, 0]);
  });

  it("an 'external' BSDF replaces the builtin one in FULL only", () => {
    const p = surface(createMaterial({ baseColor: [0.9, 0.9, 0.9], specularWeight: 0 }));
    const bsdf = () => [1, 1, 1] as [number, number, number];
    const u: Vec3Tuple = [0.2, 0.4, 0.8];
    const ext = shade(copy, p, 'full', u, always, { bsdf }, { bsdf: 'external' });
    const built = shade(copy, p, 'full', u, always);
    expect(Math.abs(ext.direct[0] / built.direct[0] / (Math.PI / 0.9) - 1)).toBeLessThan(1e-6);
    const lam = shade(copy, p, 'lambert', u, always, { bsdf }, { bsdf: 'external' });
    expect(lam.direct[0]).toBeCloseTo(built.direct[0], 9);
    expect(() => shade(copy, p, 'full', u, always, {}, { bsdf: 'external' })).toThrow();
  });

  it('a miss returns the environment, and the sun disc only with sunVisible', () => {
    const lit = buildSceneCopy({ meshes: [mesh], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: matrix, material: 0 }], materials: [{}], sun: { direction: [0, -1, 0], irradiance: [10, 10, 10], angularRadius: 0.1 }, environment: { radiance: [1, 2, 3], scale: 2 } });
    expect(shadeMiss(lit, [1, 0, 0])).toEqual([2, 4, 6]);
    expect(shadeMiss(lit, [0, 1, 0])).toEqual([2, 4, 6]);
    expect(shadeMiss(lit, [0, 1, 0], { sunVisible: true })[0]).toBeGreaterThan(100);
  });

  it('rejects a bad alpha floor', () => {
    expect(() => resolveShadeOptions({ minAlpha: 0 })).toThrow(RangeError);
  });
});

describe('builtin BSDF', () => {
  const white = (roughness: number) => createMaterial({ metalness: 1, baseColor: [1, 1, 1], specularRoughness: roughness });
  const point = (m: ReturnType<typeof createMaterial>, thetaO: number): ShadePoint =>
    makeShadePoint({ material: m, position: [0, 0, 0], shadingNormal: [0, 1, 0], geometricNormal: [0, 1, 0], frontFace: true, rayDirection: [Math.sin(thetaO), -Math.cos(thetaO), 0] });
  const dirOf = (theta: number, phi: number): Vec3Tuple => [Math.sin(theta) * Math.cos(phi), Math.cos(theta), Math.sin(theta) * Math.sin(phi)];

  it('is reciprocal', () => {
    const r = rng(3);
    const m = createMaterial({ baseColor: [0.4, 0.5, 0.6], metalness: 0.5, specularRoughness: 0.5 });
    for (let i = 0; i < 200; i++) {
      const a = dirOf(Math.acos(Math.sqrt(r())), r() * 6.28);
      const b = dirOf(Math.acos(Math.sqrt(r())), r() * 6.28);
      const pa = makeShadePoint({ material: m, position: [0, 0, 0], shadingNormal: [0, 1, 0], geometricNormal: [0, 1, 0], frontFace: true, rayDirection: [-a[0], -a[1], -a[2]] });
      const pb = makeShadePoint({ material: m, position: [0, 0, 0], shadingNormal: [0, 1, 0], geometricNormal: [0, 1, 0], frontFace: true, rayDirection: [-b[0], -b[1], -b[2]] });
      const fab = builtinBsdf(pa, b);
      const fba = builtinBsdf(pb, a);
      for (let c = 0; c < 3; c++) expect(Math.abs((fab[c] as number) - (fba[c] as number))).toBeLessThan(1e-9 * (1 + (fab[c] as number)));
    }
  });

  it('white metal at normal incidence reproduces the single-scatter albedos the path tracer BSDF documents (D15 / bsdf.md furnace)', () => {
    // Brute force: albedo = 2 pi * integral f cos sin(theta) dtheta, wo = n (isotropic). bsdf.md (perceptual roughness r, alpha = r^2): r 0.25 -> 0.9957, 0.5 -> 0.9158, 1 -> 0.3068.
    for (const [roughness, expected] of [[0.25, 0.9957], [0.5, 0.9158], [1, 0.3068]] as const) {
      const p = point(white(roughness), 0);
      const n = 40000;
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const theta = ((i + 0.5) / n) * (Math.PI / 2);
        const f = builtinBsdf(p, dirOf(theta, 0.3))[0] as number;
        sum += f * Math.cos(theta) * Math.sin(theta);
      }
      const albedo = (2 * Math.PI * sum * (Math.PI / 2)) / n;
      expect(albedo, `roughness ${roughness}`).toBeCloseTo(expected, 3);
    }
  });

  it('GGX D integrates to one over projected area', () => {
    for (const a of [0.1, 0.3, 0.8]) {
      const a2 = a * a;
      const n = 200000;
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const theta = ((i + 0.5) / n) * (Math.PI / 2);
        sum += ggxD(Math.cos(theta), a2) * Math.cos(theta) * Math.sin(theta);
      }
      expect(2 * Math.PI * sum * (Math.PI / 2 / n)).toBeCloseTo(1, 3);
    }
  });

  it('is energy conserving for a rough dielectric (diffuse + specular below 1) and uses the right F0', () => {
    const m = createMaterial({ baseColor: [0.9, 0.9, 0.9], specularRoughness: 0.4 });
    expect(shadeF0(m)[0]).toBeCloseTo(0.04, 6);
    expect(shadeAlbedo(m)[0]).toBeCloseTo(0.9, 6);
    for (const theta of [0, 0.7, 1.3]) {
      const p = point(m, theta);
      const n = 4000;
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const t = ((i + 0.5) / n) * (Math.PI / 2);
        for (let j = 0; j < 16; j++) sum += (builtinBsdf(p, dirOf(t, ((j + 0.5) / 16) * 2 * Math.PI))[0] as number) * Math.cos(t) * Math.sin(t);
      }
      const albedo = (sum * (Math.PI / 2 / n) * 2 * Math.PI) / 16;
      expect(albedo).toBeLessThan(1.0);
      expect(albedo).toBeGreaterThan(0.85);
    }
    // eta / k conductor F0.
    const copper = createMaterial({ metalness: 1, conductorEtaK: true, conductorEta: [0.2, 0.9, 1.1], conductorK: [3.9, 2.4, 2.1] });
    const f0 = shadeF0(copper);
    expect(f0[0]).toBeCloseTo((0.8 ** 2 + 3.9 ** 2) / (1.2 ** 2 + 3.9 ** 2), 9);
    expect(f0[0]).toBeGreaterThan(f0[2] as number - 1);
  });

  it('cosine sampling has pdf cos/pi and stays in the hemisphere', () => {
    const r = rng(5);
    for (let i = 0; i < 500; i++) {
      const n: Vec3Tuple = [0.3, 0.8, -0.5];
      const len = Math.hypot(...n);
      const nn: Vec3Tuple = [n[0] / len, n[1] / len, n[2] / len];
      const s = sampleCosineAbout(nn, r(), r());
      const cos = s.wi[0] * nn[0] + s.wi[1] * nn[1] + s.wi[2] * nn[2];
      expect(cos).toBeGreaterThanOrEqual(-1e-12);
      expect(s.pdf).toBeCloseTo(cos / Math.PI, 9);
      expect(Math.hypot(...s.wi)).toBeCloseTo(1, 9);
    }
  });
});
