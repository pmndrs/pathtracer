import { describe, expect, it } from 'vitest';
import { ABLATIONS, ablateScene, ablationSceneName, parseAblationName } from './ablations.ts';
import { tier1GltfSceneDescs } from './../index.ts';
import { getScene } from '../index.ts';
import { openPbrToGltfMaterial } from '../../exporters/gltf-openpbr.ts';
import type { SceneDesc } from '../schema.ts';

const base = (m: Record<string, unknown>): SceneDesc => ({
  version: 1, name: 's', purpose: '', tier: 1,
  camera: { type: 'perspective', position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
  film: { width: 8, height: 8, filter: 'box' }, integrator: { maxDepth: 4 },
  materials: { a: { type: 'openpbr', ...m } as never, d: { type: 'diffuse', reflectance: [1, 1, 1] } },
  meshes: [],
});

describe('ablations', () => {
  it('names parse and round trip', () => {
    expect(parseAblationName(ablationSceneName('IORTestGrid-notex-A', 'spec'))).toEqual({ base: 'IORTestGrid-notex-A', tag: 'spec' });
    expect(parseAblationName('cornell')).toBeNull();
    expect(parseAblationName('x--no-unknowntag')).toBeNull();
  });
  it('strips only the named feature and leaves other material types alone', () => {
    const s = ablateScene(base({ coatWeight: 1, coatRoughness: 0.1, baseMetalness: 0 }), 'coat');
    expect(s.name).toBe('s--no-coat');
    expect((s.materials['a'] as { coatWeight: number }).coatWeight).toBe(0);
    expect((s.materials['a'] as { coatRoughness: number }).coatRoughness).toBe(0.1);
    expect(s.materials['d']).toEqual({ type: 'diffuse', reflectance: [1, 1, 1] });
  });
  it('throws when no material uses the feature (the variant would equal the base)', () => {
    expect(() => ablateScene(base({ baseMetalness: 0 }), 'coat')).toThrow(/changes no material/);
  });
  it('every wave-1 ablation of the coverage rows changes its scene', () => {
    for (const [scene, tag] of [['MetalRoughSpheresNoTextures', 'metal'], ['MetalRoughSpheresNoTextures', 'spec'], ['TransmissionThinwallTestGrid-notex', 'transmission'], ['AttenuationTest-notex', 'attenuation'], ['IORTestGrid-notex-A', 'ior'], ['IORTestGrid-notex-A', 'spec'], ['IORTestGrid-notex', 'specfactor'], ['ClearCoatTest-notex', 'coat'], ['SheenTestGrid-notex', 'fuzz'], ['EmissiveStrengthTest-notex', 'strength']] as const) {
      const v = getScene(ablationSceneName(scene, tag));
      expect(v.name).toBe(ablationSceneName(scene, tag));
      expect(JSON.stringify(v.materials)).not.toBe(JSON.stringify(getScene(scene).materials));
    }
    expect(tier1GltfSceneDescs.length).toBeGreaterThan(0);
    expect(Object.keys(ABLATIONS).length).toBeGreaterThan(5);
  });
});

describe('openPbrToGltfMaterial', () => {
  it('writes only the extensions a material uses', () => {
    const { material } = openPbrToGltfMaterial({ baseColor: [0.5, 0.5, 0.5], baseMetalness: 0, specularRoughness: 0.4 });
    expect(material.extensions).toBeUndefined();
    expect(material.pbrMetallicRoughness).toEqual({ baseColorFactor: [0.5, 0.5, 0.5, 1], metallicFactor: 0, roughnessFactor: 0.4 });
  });
  it('maps thick transmission with attenuation, thin-walled without a volume', () => {
    const thick = openPbrToGltfMaterial({ transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 2, specularIor: 1.33 }).material.extensions!;
    expect(thick['KHR_materials_volume']).toMatchObject({ attenuationDistance: 2, attenuationColor: [0.1, 0.5, 0.9] });
    expect(thick['KHR_materials_ior']).toEqual({ ior: 1.33 });
    const thin = openPbrToGltfMaterial({ transmissionWeight: 1, geometryThinWalled: true }).material.extensions!;
    expect(thin['KHR_materials_volume']).toBeUndefined();
    expect(thin['KHR_materials_transmission']).toEqual({ transmissionFactor: 1 });
  });
  it('maps clearcoat, sheen, specular and emission strength', () => {
    const e = openPbrToGltfMaterial({ coatWeight: 1, coatRoughness: 0.03, fuzzWeight: 1, fuzzColor: [0.5, 0.5, 1], fuzzRoughness: 0.3, specularWeight: 0.25, emissionColor: [0.1, 0.5, 0.9], emissionLuminance: 4 });
    expect(e.material.extensions!['KHR_materials_clearcoat']).toEqual({ clearcoatFactor: 1, clearcoatRoughnessFactor: 0.03 });
    expect(e.material.extensions!['KHR_materials_sheen']).toEqual({ sheenColorFactor: [0.5, 0.5, 1], sheenRoughnessFactor: 0.3 });
    expect(e.material.extensions!['KHR_materials_specular']).toMatchObject({ specularFactor: 0.25 });
    expect(e.material.extensions!['KHR_materials_emissive_strength']).toEqual({ emissiveStrength: 4 });
    expect(e.material.emissiveFactor).toEqual([0.1, 0.5, 0.9]);
  });
  it('round trips through gltfMaterialToOpenPbr for the wave-1 features', async () => {
    const { gltfMaterialToOpenPbr } = await import('../gltf-openpbr.ts');
    const src = { transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9] as const, transmissionDepth: 1, specularIor: 1.76, specularRoughness: 0, baseColor: [1, 1, 1] as const, baseMetalness: 0, coatWeight: 0.5, coatRoughness: 0.2, coatIor: 1.5 };
    const { material } = openPbrToGltfMaterial(src);
    const back = gltfMaterialToOpenPbr(material as never).material;
    expect(back).toMatchObject({ transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 1, specularIor: 1.76, coatWeight: 0.5, coatRoughness: 0.2 });
  });
});
