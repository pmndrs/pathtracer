import { describe, expect, it } from 'vitest';
import { ALPHA_MODE, buildSceneCopy, sceneFormatOf } from '@pmndrs/raytracer';
import type { GpuScene, SceneCopyInput } from '@pmndrs/raytracer';
import { ptSceneOptions } from './scene-options.ts';

const quad: SceneCopyInput['meshes'][number] = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), indices: [0, 1, 2, 0, 2, 3], uvs: [0, 1, 1, 1, 1, 0, 0, 0] };

/** A `GpuScene` without a device: only the CPU halves are read by `ptSceneOptions`. */
function fakeScene(input: SceneCopyInput): GpuScene {
  return { input, copy: buildSceneCopy(input), trace: {} as GpuScene['trace'], buffers: {} as GpuScene['buffers'], format: sceneFormatOf({}), packing: 'packed', destroy() {} };
}

describe('ptSceneOptions', () => {
  it('takes formats, layout and packing from the scene and turns on OpenPBR materials', () => {
    const scene = fakeScene({ meshes: [quad], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: 0 }], materials: [{ baseColor: [0.8, 0.8, 0.8], coatWeight: 1 }] });
    const o = ptSceneOptions(scene, { integrator: { nee: false } });
    expect(o.format).toEqual(scene.format);
    expect(o.layout).toEqual(scene.copy.layout);
    expect(o.scenePacking).toBe('packed');
    expect(o.textures).toBeNull();
    expect(o.integrator?.nee).toBe(false);
    expect(o.integrator?.envMap).toBe(false);
    expect(o.integrator?.alphaMask).toBeUndefined();
    expect(o.integrator?.materials?.features.layered).toBe(true);
  });

  it('turns on the alpha mask for a textured cutout and the importance-sampled environment for a map', () => {
    const texture = { width: 2, height: 2, data: new Uint8Array(16).fill(255), colorSpace: 'srgb' as const };
    const scene = fakeScene({
      meshes: [quad], instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], material: 0 }],
      materials: [{ baseColorTexture: 0, alphaMode: ALPHA_MODE.mask, metalness: 1 }], textures: [texture],
      environment: { map: { width: 4, height: 2, channels: 3, data: new Float32Array(24).fill(1) } },
    });
    const o = ptSceneOptions(scene);
    expect(o.textures).not.toBeNull();
    expect(o.integrator?.alphaMask).toBe(true);
    expect(o.integrator?.envMap).toBe(true);
    expect(o.integrator?.envSampling).toBe('importance');
    expect(ptSceneOptions(scene, { integrator: { materials: null } }).integrator?.materials).toBeNull();
  });
});
