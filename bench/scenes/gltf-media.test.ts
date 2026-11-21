import { describe, expect, it } from 'vitest';
import { absorberOf, attenuationToSigmaA, gltfVolumesAsMedia, singleScatterAlbedo } from './gltf-media.ts';
import type { SceneDesc } from './schema.ts';

const glass = { type: 'openpbr', transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 2 } as const;
const base = (materials: SceneDesc['materials'], meshes: SceneDesc['meshes']): SceneDesc => ({ version: 1, name: 't', purpose: 'p', tier: 1, camera: { type: 'perspective', position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 }, film: { width: 8, height: 8, filter: 'box' }, integrator: { maxDepth: 4 }, materials, meshes, lights: [], environment: { type: 'uniform', radiance: [1, 1, 1] } } as SceneDesc);

describe('gltfVolumesAsMedia', () => {
  it('sigma_a = -ln(colour) / distance', () => {
    const s = attenuationToSigmaA([0.1, 0.5, 1], 2)!;
    expect(s[0]).toBeCloseTo(Math.log(10) / 2, 12);
    expect(s[1]).toBeCloseTo(Math.log(2) / 2, 12);
    expect(s[2]).toBe(0);
    expect(attenuationToSigmaA([1, 1, 1], 1)).toBeUndefined();
    expect(attenuationToSigmaA([0.5, 0.5, 0.5], 0)).toBeUndefined();
  });
  it('thin, no depth, no transmission: no absorber', () => {
    expect(absorberOf({ ...glass, geometryThinWalled: true })).toBeUndefined();
    expect(absorberOf({ ...glass, transmissionDepth: 0 })).toBeUndefined();
    expect(absorberOf({ ...glass, transmissionWeight: 0 })).toBeUndefined();
    expect(absorberOf({ type: 'diffuse', reflectance: [1, 1, 1] })).toBeUndefined();
  });
  it('moves the absorber into a medium and keeps the dielectric', () => {
    const sc = base({ g: { ...glass }, d: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] } }, [
      { name: 'a', geometry: { type: 'sphere', radius: 1, segments: 8, rings: 4 }, material: 'g' },
      { name: 'b', geometry: { type: 'sphere', radius: 1, segments: 8, rings: 4 }, material: 'd' },
    ] as SceneDesc['meshes']);
    const out = gltfVolumesAsMedia(sc);
    expect(out.name).toBe('t-media');
    expect(out.meshes[0]!.medium).toEqual({ interior: 'absorb-g' });
    expect(out.meshes[1]!.medium).toBeUndefined();
    expect(out.media!['absorb-g']).toMatchObject({ type: 'homogeneous', sigmaS: [0, 0, 0] });
    expect(out.materials['g']).toMatchObject({ transmissionWeight: 1, transmissionColor: [1, 1, 1] });
    expect((out.materials['g'] as { transmissionDepth?: number }).transmissionDepth).toBeUndefined();
    expect(gltfVolumesAsMedia(base({ d: sc.materials['d']! }, [sc.meshes[1]!]))).toEqual(base({ d: sc.materials['d']! }, [sc.meshes[1]!]));
  });
  it('single-scatter albedo inversion (KHR_materials_scatter, Kulla-Conty fit)', () => {
    expect(singleScatterAlbedo([0, 0, 0], 0.3)).toEqual([0, 0, 0]);
    // a = 1: the fit gives s = -2.7e-11, clamped to 0, so rho_ss = 1 exactly (a white multi-scatter medium never absorbs)
    expect(singleScatterAlbedo([1, 1, 1], 0)[0]).toBe(1);
    const a = 0.5;
    const s = 4.09712 + 4.20863 * a - Math.sqrt(9.59217 + 41.6808 * a + 17.7126 * a * a);
    expect(singleScatterAlbedo([a, a, a], 0)[1]).toBeCloseTo(1 - s * s, 12);
    expect(singleScatterAlbedo([a, a, a], 0.5)[2]).toBeCloseTo((1 - s * s) / (1 - 0.5 * s * s), 12);
    const v = singleScatterAlbedo([0.1, 0.5, 0.9], 0);
    expect(v[0]! < v[1]! && v[1]! < v[2]!).toBe(true);
  });
  it('volumeScatter splits sigma_t into sigma_s / sigma_a and sets g', () => {
    const vs = { strength: 1, multiscatterColor: [0.8, 0.5, 0.2] as const, anisotropy: 0.4 };
    const sc = base({ g: { ...glass, volumeScatter: vs } as SceneDesc['materials'][string] }, [{ name: 'a', geometry: { type: 'sphere', radius: 1, segments: 8, rings: 4 }, material: 'g' }] as SceneDesc['meshes']);
    const out = gltfVolumesAsMedia(sc);
    const m = out.media!['absorb-g']!;
    const t = attenuationToSigmaA([0.1, 0.5, 0.9], 2)!;
    const rho = singleScatterAlbedo([0.8, 0.5, 0.2], 0.4);
    for (let i = 0; i < 3; i++) {
      expect(m.sigmaS[i]).toBeCloseTo(t[i]! * rho[i]!, 12);
      expect(m.sigmaA[i]! + m.sigmaS[i]!).toBeCloseTo(t[i]!, 12);
    }
    expect(m.g).toBe(0.4);
    expect('volumeScatter' in out.materials['g']!).toBe(false);
    // strength scales the multi-scatter albedo: 0.5 x [0.8,0.5,0.2] -> less scattering
    const half = gltfVolumesAsMedia(base({ g: { ...glass, volumeScatter: { ...vs, strength: 0.5 } } as SceneDesc['materials'][string] }, sc.meshes)).media!['absorb-g']!;
    expect(half.sigmaS[0]).toBeLessThan(m.sigmaS[0]!);
  });
});
