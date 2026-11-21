import { describe, expect, it } from 'vitest';
import { absorberOf, attenuationToSigmaA, gltfVolumesAsMedia } from './gltf-media.ts';
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
});
