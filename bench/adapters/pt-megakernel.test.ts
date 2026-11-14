import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, ptBsdfFromMaterial } from '@pmndrs/pathtracer';
import { createMaterial } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { getScene, allScenes, tier1GltfSceneDescs, tier1TexturedGltfSceneDescs } from '../scenes/index.ts';
import { materialFromDesc, sceneDescToCopyInput } from '../scenes/to-scene-copy.ts';
import { ADAPTER_IDS, OWN_ADAPTER_IDS, unsupportedReason } from './registry.ts';
import { PT_ADAPTER_ID, PtMegakernelAdapter } from './pt-megakernel/index.ts';

const adapter = new PtMegakernelAdapter();

describe('pt-megakernel adapter', () => {
  it('is registered as one of our own renderers and not in the default competitor set', () => {
    expect(OWN_ADAPTER_IDS).toContain(PT_ADAPTER_ID);
    expect([...ADAPTER_IDS]).not.toContain(PT_ADAPTER_ID);
  });

  it('config records every setting that changes the image', () => {
    expect(adapter.info.config).toMatchObject({ sampler: 'owen-sobol', nee: true, envSampling: 'auto', sunVisible: false, positions: 'barycentric', vndf: 'standard', transport: 'radiance', seed: 0 });
    const custom = new PtMegakernelAdapter({ sampler: 'pcg', nee: false, positions: 'ray', maxDepth: 3 });
    expect(custom.info.config).toMatchObject({ sampler: 'pcg', nee: false, positions: 'ray', maxDepth: 3 });
  });

  it('renders every scene: env-sphere through its HDR map, Sponza from the web glb export', () => {
    const skipped = allScenes.filter((scene) => unsupportedReason(adapter.features, scene) !== null).map((scene) => scene.name);
    // The Tier 1 glTF scenes use `openpbr` materials. Waves 1 and 2 render all of them (the fuzz lobe, P2.25, made SheenTestGrid renderable; the thin film, P2.27, too).
    const tier1Gltf = new Set([...tier1GltfSceneDescs, ...tier1TexturedGltfSceneDescs].map((s) => s.name));
    expect(skipped.filter((name) => !tier1Gltf.has(name))).toEqual([]);
    expect(skipped.filter((name) => tier1Gltf.has(name))).toEqual([]);
    expect(unsupportedReason(adapter.features, getScene('SheenTestGrid-notex'))).toBeNull();
    expect(unsupportedReason(adapter.features, getScene('env-sphere'))).toBeNull();
    expect(unsupportedReason(adapter.features, getScene('sponza-white'))).toBeNull();
  });
});

describe('bench materials -> BSDFs through the scene copy records', () => {
  const kindOf = (desc: Parameters<typeof materialFromDesc>[0]): number => ptBsdfFromMaterial(createMaterial(materialFromDesc(desc))).kind;

  it('maps each bench material type to its BSDF', () => {
    expect(kindOf({ type: 'diffuse', reflectance: [0.5, 0.5, 0.5] })).toBe(PT_BSDF_DIFFUSE);
    expect(kindOf({ type: 'conductor', eta: [1, 1, 1], k: [1000, 1000, 1000], roughness: 0.7 })).toBe(PT_BSDF_CONDUCTOR);
    expect(kindOf({ type: 'dielectric', ior: 1.5, roughness: 0 })).toBe(PT_BSDF_DIELECTRIC);
    expect(kindOf({ type: 'black' })).toBe(PT_BSDF_BLACK);
  });

  it('every material of every renderable scene maps to a lobe the kernel implements, with the parameters the scene gave', () => {
    for (const scene of allScenes.filter((s) => unsupportedReason(adapter.features, s) === null)) {
      for (const [name, desc] of Object.entries(scene.materials)) {
        const p = ptBsdfFromMaterial(createMaterial(materialFromDesc(desc, (n) => Object.keys(scene.textures ?? {}).indexOf(n))));
        if (desc.type === 'diffuse') expect(p.reflectance, `${scene.name}/${name}`).toEqual([...desc.reflectance]);
        if (desc.type === 'conductor') {
          expect(p.eta).toEqual([...desc.eta]);
          expect(p.alpha[0]).toBeCloseTo(desc.roughness ** 2, 12);
        }
        if (desc.type === 'dielectric') expect(p.ior).toBe(desc.ior);
      }
    }
  });

  it('the converter gives emitters the black backing material plus emission, so an emitter hit ends the path after its light', () => {
    const input = sceneDescToCopyInput(getScene('quad-light'));
    const light = input.materials.find((_, i) => input.materialNames[i]!.includes('emission'))!;
    expect(ptBsdfFromMaterial(createMaterial(light)).kind).toBe(PT_BSDF_BLACK);
  });
});
