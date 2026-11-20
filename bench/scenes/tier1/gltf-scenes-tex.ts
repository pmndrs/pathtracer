/**
 * Tier 1 wave 2 scenes (docs/internal/design/tier1.md section 4, P2.24): the unmodified textured Khronos models. Wave 1 dropped the labels and
 * replaced every textured backdrop by a grey Lambertian (`<model>-notex`); here the textures stay: `IORTestGrid` and `SheenTestGrid`
 * (a checker backdrop; the IOR one is a rough metal), `AttenuationTest` (label sheet, plain grid backdrop) and `MetalRoughSpheres`
 * (the textured sibling of `MetalRoughSpheresNoTextures`: base colour + metallic-roughness maps on one mesh of spheres).
 *
 * Cameras, film, integrator and sky are those of the wave 1 variants of the same model. Materials come from the committed goldens (they
 * list texture slots by name only), the `TEXTURE_SLOTS` table here names the texture each slot reads, and `gltf-scenes.test.ts`-style
 * checks assert the scene equals one built from the downloaded model by `gltfSceneDescFromFile` (whose `loadGltfScene` reads the real
 * slots, sizes, wraps and samplers). Image textures carry no pixels: `generator: 'image'` points into the pinned glb.
 *
 * Alpha: none of these four models uses `alphaMode` (all opaque, no `KHR_texture_transform`), so MASK is exercised by the synthetic
 * `tex-mask` scene (`tier1/textured.ts`) instead. `AttenuationTest`'s `R3_ThicknessTex_Mat` row has a `thicknessTexture` that no renderer
 * here reads: it keeps its `thicknessFactor` (identically in ours and both references; recorded as a divergence).
 */
import { buildGltfSceneDesc } from '../gltf-desc.ts';
import type { GltfDescMaterial, GltfSceneBuild } from '../gltf-desc.ts';
import type { MaterialDesc, SceneDesc, TextureDesc } from '../schema.ts';
import attenuation from './golden/AttenuationTest.json' with { type: 'json' };
import ior from './golden/IORTestGrid.json' with { type: 'json' };
import metalRough from './golden/MetalRoughSpheres.json' with { type: 'json' };
import sheen from './golden/SheenTestGrid.json' with { type: 'json' };
import { goldenMaterials, goldenModelRef, tier1GltfScenes } from './gltf-scenes.ts';
import type { Tier1GltfScene } from './gltf-scenes.ts';

type Golden = Parameters<typeof goldenModelRef>[0];
type Slot = 'baseColorTexture' | 'metallicRoughnessTexture' | 'normalTexture' | 'emissiveTexture';

/** One image of the model's glb as a scene texture (what `gltfTextureDescs` writes). */
function image(golden: Golden, imageIndex: number, size: number, colorSpace: 'srgb' | 'linear', wrap: 'repeat' | 'clamp' = 'repeat'): TextureDesc {
  const model = goldenModelRef(golden);
  return { generator: 'image', path: model.path, image: imageIndex, sha256: model.sha256, width: size, height: size, colorSpace, wrap, filter: 'linear' };
}

/** The first wave 1 scene of a model: its camera, film, integrator and sky (the wave 2 scene must look at the same thing). */
const wave1 = (name: string): Tier1GltfScene => {
  const t = tier1GltfScenes.find((s) => s.scene.name === name);
  if (!t) throw new Error(`gltf-scenes-tex: no wave 1 scene '${name}'`);
  return t;
};

function make(golden: Golden, name: string, purpose: string, from: string, textures: Record<string, TextureDesc>, slots: Record<string, Partial<Record<Slot, string>>>, extra: Partial<GltfSceneBuild> = {}): Tier1GltfScene {
  const model = goldenModelRef(golden);
  const base = wave1(from).sidecar;
  const materials: GltfDescMaterial[] = goldenMaterials(golden).map((m) => {
    const key = m.name ?? String(m.index);
    const s = slots[key];
    return s ? { ...m, material: { ...(m.material as Extract<MaterialDesc, { type: 'openpbr' }>), ...s } } : m;
  });
  for (const k of Object.keys(slots)) if (!materials.some((m) => (m.name ?? String(m.index)) === k)) throw new Error(`gltf-scenes-tex '${name}': no material '${k}'`);
  const { dropMaterials: _d, replace: _r, ...camera } = base;
  const sidecar = { ...camera, ...extra, name, purpose };
  const build: GltfSceneBuild = { ...sidecar, model, materials, textures };
  return { golden: golden as Tier1GltfScene['golden'], model, sidecar: sidecar as Tier1GltfScene['sidecar'], scene: buildGltfSceneDesc(build) };
}

const iorGolden = ior as Golden, sheenGolden = sheen as Golden, attGolden = attenuation as Golden, mrGolden = metalRough as Golden;

const scenes = {
  ior: make(iorGolden, 'IORTestGrid', 'Khronos IORTestGrid, unmodified: 20 spheres (IOR 1.0-2.42 x specular 0.25 / 1, black opaque and white transmissive bases) in front of a rough-metal checker backdrop (base colour texture, repeat). Tier 1 ior + specular + textured conductor (P2.24). The ten specular 0.25 spheres are class C (coverage).', 'IORTestGrid-notex',
    { t0: image(iorGolden, 0, 256, 'srgb') }, { Backdrop: { baseColorTexture: 't0' } }),
  // IORTestGrid minus its ten specular 0.25 (class C) spheres, as `IORTestGrid-notex-A`: the textured backdrop, the labels and the ten class A spheres alone, so no class C
  // reflection or refraction reaches a gated region. The Tier 1 gate scene of the wave.
  iorA: make(iorGolden, 'IORTestGrid-A', 'IORTestGrid with the ten specularFactor 0.25 (class C) spheres dropped: the class A spheres (specular 1, black opaque and white transmissive, IOR 1.0-2.42) in front of the rough-metal checker backdrop (base colour texture), labels kept. Tier 1 ior + textured conductor gate (P2.24).', 'IORTestGrid-notex',
    { t0: image(iorGolden, 0, 256, 'srgb') }, { Backdrop: { baseColorTexture: 't0' } },
    { dropMaterials: (ior as Golden).materials.filter((m) => /_S0\.25$/.test(m.name ?? '')).map((m) => m.name as string) }),
  sheen: make(sheenGolden, 'SheenTestGrid', 'Khronos SheenTestGrid, unmodified: 16 sheen spheres in front of a rough dielectric checker backdrop (base colour texture, repeat). Tier 1 sheen + textured dielectric (P2.24); sheen is coverage only.', 'SheenTestGrid-notex',
    { t0: image(sheenGolden, 0, 256, 'srgb') }, { Backdrop: { baseColorTexture: 't0' } }),
  attenuation: make(attGolden, 'AttenuationTest', 'Khronos AttenuationTest, unmodified: attenuationDistance/Color and thickness on closed blocks and thin samples, the label sheet (base colour texture with its own alpha, opaque mode) and the plain-grid backdrop. R3_ThicknessTex_Mat keeps its thicknessFactor (the thickness texture is read by no renderer here). Tier 1 volume + textured dielectric (P2.24).', 'AttenuationTest-notex',
    { t0: image(attGolden, 0, 512, 'srgb'), t2: image(attGolden, 2, 256, 'srgb') }, { LabelMaterial: { baseColorTexture: 't0' }, FlatBackdrop: { baseColorTexture: 't2' } }),
  metalRough: make(mrGolden, 'MetalRoughSpheres', 'Khronos MetalRoughSpheres, textured: one 502k-triangle mesh of spheres whose base colour (sRGB) and metallic-roughness (G roughness, B metalness) maps, clamp-to-edge, carry the metalness x roughness grid. Tier 1 core metallic-roughness with textures (P2.24).', 'MetalRoughSpheresNoTextures',
    { t0: image(mrGolden, 0, 1024, 'srgb', 'clamp'), t1_lin: image(mrGolden, 1, 1024, 'linear', 'clamp') }, { '0': { baseColorTexture: 't0', metallicRoughnessTexture: 't1_lin' } },
    {
      camera: { type: 'perspective', position: [0, -0.3, 19.5], target: [0, -0.3, 0], up: [0, 1, 0], yFovDeg: 30 },
      film: { width: 768, height: 768, filter: 'box' },
      integrator: { maxDepth: 8 },
      scale: 1,
    }),
};

/** The wave 2 scenes with their goldens, model references and sidecars (tests rebuild them from the download). */
export const tier1TexturedGltfScenes: readonly Tier1GltfScene[] = Object.values(scenes);

/** The `SceneDesc`s, registered in `scenes/index.ts`. */
export const tier1TexturedGltfSceneDescs: readonly SceneDesc[] = tier1TexturedGltfScenes.map((s) => s.scene);
