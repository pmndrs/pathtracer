/**
 * Tier 1 wave 1 scenes (docs/design/tier1.md sections 1.1 and 4; P2.19): Khronos `glTF-Sample-Assets` test models with a sidecar each
 * (camera, environment, film, integrator, overrides). Wave 1 has no textures: the labels are dropped and the textured backdrop becomes a grey
 * Lambertian, so the scene is named `<model>-notex` (`MetalRoughSpheresNoTextures` has none to begin with).
 *
 * The material tables come from the committed goldens (`golden/<Model>.json`, written by `goldens.ts` from `gltfMaterialToOpenPbr`), so the scenes
 * exist without the download; the geometry is a `file` mesh per glTF material (`select.material`) resolved from `bench/assets/downloads/tier1/`
 * (`node bench/assets/fetch.ts`). `gltf-scenes.test.ts` checks the golden-built scene equals one built from the downloaded model.
 *
 * Cameras were framed from the loaded bounds (every model views down -Z from +Z). Lighting is the procedural sky of `env-sphere`
 * rotated 180 degrees about Y, so its sun hot spot is on the camera side. Film is small (these are coverage scenes; references come later).
 */
import { ENV_FILE } from '../tier0/env-sky.ts';
import { buildGltfSceneDesc } from '../gltf-desc.ts';
import type { GltfDescMaterial, GltfModelRef, GltfSceneBuild } from '../gltf-desc.ts';
import type { GltfSidecar } from '../gltf-scene-desc.ts';
import type { EnvironmentDesc, MaterialDesc, SceneDesc } from '../schema.ts';
import anisotropy from './golden/AnisotropyStrengthTest.json' with { type: 'json' };
import attenuation from './golden/AttenuationTest.json' with { type: 'json' };
import clearcoat from './golden/ClearCoatTest.json' with { type: 'json' };
import emissive from './golden/EmissiveStrengthTest.json' with { type: 'json' };
import ior from './golden/IORTestGrid.json' with { type: 'json' };
import iridescence from './golden/IridescenceDielectricSpheres.json' with { type: 'json' };
import metalRough from './golden/MetalRoughSpheresNoTextures.json' with { type: 'json' };
import sheen from './golden/SheenTestGrid.json' with { type: 'json' };
import thinwall from './golden/TransmissionThinwallTestGrid.json' with { type: 'json' };

interface GoldenFile {
  header: { model: string; sourceSha256: string; materialCount: number };
  materials: { i: number; name?: string; class: string; openpbr: Record<string, unknown> }[];
}

/** The pinned model file of a golden (the glb of every wave 1 model; `.gltf` models would need their format passed). */
export function goldenModelRef(g: GoldenFile, format: 'glb' | 'gltf' = 'glb'): GltfModelRef {
  const m = g.header.model;
  return { path: `downloads/tier1/${m}/${m}.${format}`, format, sha256: g.header.sourceSha256 };
}

/** The material table of a golden as scene materials (sparse `openpbr`, defaults resolved by `resolveOpenPbr`). */
export function goldenMaterials(g: GoldenFile): GltfDescMaterial[] {
  return g.materials.map((m) => ({ index: m.i, ...(m.name !== undefined ? { name: m.name } : {}), material: { type: 'openpbr', ...m.openpbr } as MaterialDesc }));
}

/** The procedural sky, sun on the camera side. */
const SKY: EnvironmentDesc = { type: 'hdri', path: ENV_FILE, scale: 1, rotationYDeg: 180 };
/** The factor-only stand-in for every textured backdrop. */
const GREY: MaterialDesc = { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] };
/** Metal-base variants of the ClearCoatTest "Simple" spheres (P2.23): a rough gold-ish metal bare, and under the same clear coat (coat roughness 0.03, ior 1.5). */
const GOLD: MaterialDesc = { type: 'openpbr', baseColor: [0.9, 0.6, 0.25], baseMetalness: 1, specularRoughness: 0.35 };
const GOLD_COATED: MaterialDesc = { type: 'openpbr', baseColor: [0.9, 0.6, 0.25], baseMetalness: 1, specularRoughness: 0.35, coatWeight: 1, coatRoughness: 0.03, coatIor: 1.5 };

const greyBase = (c: unknown): boolean => Array.isArray(c) && Math.abs((c[0] as number) - (c[2] as number)) < 1e-3;

export interface Tier1GltfScene { golden: GoldenFile; model: GltfModelRef; sidecar: GltfSidecar; scene: SceneDesc }

function make(golden: GoldenFile, sidecar: GltfSidecar, format: 'glb' | 'gltf' = 'glb'): Tier1GltfScene {
  const model = goldenModelRef(golden, format);
  const build: GltfSceneBuild = { ...sidecar, model, materials: goldenMaterials(golden) };
  return { golden, model, sidecar, scene: buildGltfSceneDesc(build) };
}

const sidecars = {
  // R1..R5 volume and thin blocks, 20 x 20 backdrop at z = -2, labels left of the blocks. Drop the labels and the one row that needs a thickness texture.
  attenuation: make(attenuation as GoldenFile, {
    name: 'AttenuationTest-notex',
    purpose: 'Khronos AttenuationTest without textures (labels and the thickness-texture row dropped, backdrop = grey Lambertian): attenuationDistance/Color and thickness on closed blocks and thin samples under the sky. Tier 1 volume coverage (P2.22).',
    camera: { type: 'perspective', position: [1.5, 0.25, 21], target: [1.5, 0.25, 0], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: 600, height: 720, filter: 'box' },
    integrator: { maxDepth: 16 },
    environment: SKY,
    dropMaterials: ['LabelMaterial', 'R3_ThicknessTex_Mat'],
    replace: { FlatBackdrop: GREY },
  }),
  // 5 thin-walled and 5 solid spheres (IOR 1.0 .. 2.42) on a backdrop at z = -0.9; text and its plate dropped.
  thinwall: make(thinwall as GoldenFile, {
    name: 'TransmissionThinwallTestGrid-notex',
    purpose: 'Khronos TransmissionThinwallTestGrid without textures (text dropped, checker backdrop = grey Lambertian): thin-walled (thickness 0) and volume spheres, IOR 1.0-2.42, roughness 0. Tier 1 thin-wall transmission (P2.22).',
    camera: { type: 'perspective', position: [1.11, 3.28, 10.8], target: [1.11, 3.28, 0], up: [0, 1, 0], yFovDeg: 35 },
    film: { width: 1000, height: 500, filter: 'box' },
    integrator: { maxDepth: 16 },
    environment: SKY,
    dropMaterials: ['Text', 'Text Backdrop'],
    replace: { Backdrop: GREY },
  }),
  // Five emissive cubes (factor {0.1, 0.5, 0.9} x strength 1, 2, 4, 8, 16), black base; backdrop replaced.
  emissive: make(emissive as GoldenFile, {
    name: 'EmissiveStrengthTest-notex',
    purpose: 'Khronos EmissiveStrengthTest with the textured backdrop replaced by a grey Lambertian: five cubes, emissiveFactor (0.1, 0.5, 0.9) x emissiveStrength 1-16, black base. Tier 1 emissive_strength (P2.30).',
    camera: { type: 'perspective', position: [0, -0.25, 7], target: [0, -0.25, 0], up: [0, 1, 0], yFovDeg: 40 },
    film: { width: 1120, height: 400, filter: 'box' },
    integrator: { maxDepth: 4 },
    environment: SKY,
    replace: { FlatBackdrop: GREY },
  }),
  // 7 x 7 grid x 2 layers (grey and gold, metalness 0..1 by row, roughness 0..1 by column) of 0.7 mm spheres, 1 mm pitch: scaled x1000. The unlabelled text plane (glTF default material) is dropped.
  metalRough: make(metalRough as GoldenFile, {
    name: 'MetalRoughSpheresNoTextures',
    purpose: 'Khronos MetalRoughSpheresNoTextures (scaled x1000 from millimetres): 98 factor-only spheres, two 7 x 7 layers (grey and gold), metalness by row and roughness by column, 1.04 M triangles. Tier 1 core metallic-roughness; pure-metal rows are reference class A.',
    scale: 1000,
    camera: { type: 'perspective', position: [6.5, 7.5, 12], target: [3, 3, -1.5], up: [0, 1, 0], yFovDeg: 38 },
    film: { width: 768, height: 768, filter: 'box' },
    integrator: { maxDepth: 8 },
    environment: SKY,
  }),
  // The class A gate layer of MetalRoughSpheresNoTextures: the seven grey pure-metal spheres (metalness 1, roughness 0..1; row y = 6, z = 0) alone, so no class B
  // dielectric row or saturated gold sphere sits in the image (P2.31 coverage matrix: the whole-scene image is dominated by the 84 class B rows).
  greyMetal: make(metalRough as GoldenFile, {
    name: 'MetalRoughSpheresNoTextures-grey-metal',
    purpose: 'MetalRoughSpheresNoTextures reduced to its seven grey pure-metal spheres (metalness 1, roughness 0..1) under the sky: reference class A, no class B row in the image. Tier 1 core metallic-roughness (P2.31).',
    scale: 1000,
    camera: { type: 'perspective', position: [3, 6, 8], target: [3, 6, 0], up: [0, 1, 0], yFovDeg: 16 },
    film: { width: 768, height: 192, filter: 'box' },
    integrator: { maxDepth: 8 },
    environment: SKY,
    dropMaterials: (metalRough as GoldenFile).materials.filter((m) => !(m.openpbr['baseMetalness'] === 1 && greyBase(m.openpbr['baseColor']))).map((m) => m.i),
  }),
  // 4 x 5 spheres (IOR 1.0..2.42 x specular 0.25 / 1 x black base T0 / white base T1) in front of a backdrop at z = -0.04.
  ior: make(ior as GoldenFile, {
    name: 'IORTestGrid-notex',
    purpose: 'Khronos IORTestGrid without textures (text dropped, checker backdrop = grey Lambertian): spheres with ior 1.0-2.42 x specularFactor 0.25/1, black opaque and white transmissive bases. Tier 1 ior + specular.',
    camera: { type: 'perspective', position: [0, 0, 1.4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 640, height: 768, filter: 'box' },
    integrator: { maxDepth: 16 },
    environment: SKY,
    dropMaterials: ['Text', 'Text Backdrop'],
    replace: { Backdrop: GREY },
  }),
  // IORTestGrid-notex minus the ten specularFactor 0.25 spheres (class C), which the references approximate through IOR: the class A spheres no longer see them reflected or refracted (TASKS P2.32a).
  iorA: make(ior as GoldenFile, {
    name: 'IORTestGrid-notex-A',
    purpose: 'IORTestGrid-notex with the ten specularFactor 0.25 (class C) spheres dropped: only the class A specular 1 spheres (black opaque and white transmissive, IOR 1.0-2.42) remain, so no class C reflection leaks into their regions. Tier 1 ior.',
    camera: { type: 'perspective', position: [0, 0, 1.4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 640, height: 768, filter: 'box' },
    integrator: { maxDepth: 16 },
    environment: SKY,
    dropMaterials: ['Text', 'Text Backdrop', 'IOR1.0_Black_R0_M0_T0_S0.25', 'IOR1.33_Black_R0_M0_T0_S0.25', 'IOR1.5_Black_R0_M0_T0_S0.25', 'IOR1.76_Black_R0_M0_T0_S0.25', 'IOR2.42_Black_R0_M0_T0_S0.25', 'IOR2.42_White_R0_M0_T1_S0.25', 'IOR1.76_White_R0_M0_T1_S0.25', 'IOR1.5_White_R0_M0_T1_S0.25', 'IOR1.33_White_R0_M0_T1_S0.25', 'IOR1.0_White_R0_M0_T1_S0.25'],
    replace: { Backdrop: GREY },
  }),
  // Only the five black specularFactor 0.25 spheres (class C: coverage, no reference) in front of the backdrop: the specular < 1 feature fills the image (P2.31).
  iorC: make(ior as GoldenFile, {
    name: 'IORTestGrid-notex-C',
    purpose: 'IORTestGrid-notex reduced to its five black opaque specularFactor 0.25 spheres (IOR 1.0-2.42): reference class C (coverage only), the feature is all there is to see. Tier 1 specular factor < 1 (P2.31).',
    camera: { type: 'perspective', position: [0, 0, 1.4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 640, height: 768, filter: 'box' },
    integrator: { maxDepth: 16 },
    environment: SKY,
    dropMaterials: (ior as GoldenFile).materials.filter((m) => !/^IOR[\d.]+_Black_R0_M0_T0_S0\.25$/.test(m.name ?? '') && m.name !== 'Backdrop').map((m) => m.i),
    replace: { Backdrop: GREY },
  }),
  // The factor-only "Simple" row (base, coated, coat only); every other row uses clearcoat/roughness/normal textures. No backdrop in the file: spheres against the sky.
  clearcoat: make(clearcoat as GoldenFile, {
    name: 'ClearCoatTest-notex',
    purpose: 'Khronos ClearCoatTest, factor-only "Simple" row only (base layer, coated, coat only; the textured rows and labels dropped), seen against the sky (the model has no backdrop). Tier 1 clearcoat (P2.23).',
    camera: { type: 'perspective', position: [0, 5.25, 5.2], target: [0, 5.25, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 1024, height: 384, filter: 'box' },
    integrator: { maxDepth: 8 },
    environment: SKY,
    dropMaterials: ['Partial_Base', 'Partial_Coated', 'Partial_Coating', 'RoughVariations_Base', 'RoughVariations_Coated', 'RoughVariations_Coating', 'BaseNorm_Base', 'BaseNorm_Coated', 'BaseNorm_Coating', 'CoatNorm_Base', 'CoatNorm_Coated', 'CoatNorm_Coating', 'SharedNorm_Base', 'SharedNorm_Coated', 'SharedNorm_Coating', 'LabelMaterial'],
  }),
  // The Simple row with the red dielectric base replaced by a rough gold-ish metal: base layer, coated (coat over a metal: pbrt coatedconductor), coat only.
  clearcoatMetal: make(clearcoat as GoldenFile, {
    name: 'ClearCoatTest-notex-metal',
    purpose: 'ClearCoatTest-notex with the Simple row\'s base replaced by a rough gold-ish metal (roughness 0.35): bare metal, the same metal under a clear coat (coat roughness 0.03, ior 1.5; pbrt coatedconductor, Mitsuba has no equivalent), coat only. Tier 1 clearcoat over a metal base (P2.23).',
    camera: { type: 'perspective', position: [0, 5.25, 5.2], target: [0, 5.25, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 1024, height: 384, filter: 'box' },
    integrator: { maxDepth: 8 },
    environment: SKY,
    dropMaterials: ['Partial_Base', 'Partial_Coated', 'Partial_Coating', 'RoughVariations_Base', 'RoughVariations_Coated', 'RoughVariations_Coating', 'BaseNorm_Base', 'BaseNorm_Coated', 'BaseNorm_Coating', 'CoatNorm_Base', 'CoatNorm_Coated', 'CoatNorm_Coating', 'SharedNorm_Base', 'SharedNorm_Coated', 'SharedNorm_Coating', 'LabelMaterial'],
    replace: { Simple_Base: GOLD, Simple_Coated: GOLD_COATED },
  }),
  // 7 x 7 grid of r = 0.4 metal spheres (x -3..3 = roughness 0..1, y 0..6 = anisotropy strength 0..1) and a plane of text labels at z = 0. No backdrop in the
  // model: spheres against the sky. All 49 materials are class A (white pure metal, KHR_materials_anisotropy). The tangent is glTF TANGENT = dp/du of the uvs (tangent check, V75).
  anisotropy: make(anisotropy as GoldenFile, {
    name: 'AnisotropyStrengthTest-notex',
    purpose: 'Khronos AnisotropyStrengthTest without the label plane: 49 white metal spheres (7 x 7: roughness and KHR_materials_anisotropy strength grid), anisotropy axis = glTF TANGENT = dp/du of the uvs, under the sky (the model has no backdrop). Tier 1 anisotropy (P2.26); reference class A.',
    camera: { type: 'perspective', position: [0, 3, 15], target: [0, 3, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 768, height: 768, filter: 'box' },
    integrator: { maxDepth: 8 },
    environment: SKY,
    dropMaterials: ['Label Mat'],
  }),
  // IridescenceDielectricSpheres is a 7 x 7 x 7 grid of r = 1.5 spheres (343 materials, 600k triangles; the file ships `.gltf` + `.bin` only): x = base ior
  // (1..2), y = film thickness (0.1..0.7 um), z = film ior (1..2); the guides plane is not a sphere. The slice z = -3.5 (film ior 1.33, material index
  // 49 ix + 7 iy + 2) keeps 49 spheres (about 88k triangles): base ior x film thickness on a 0.8 grey rough dielectric, seen down -Z so no other layer hides it. Class C (P2.27).
  iridescence: make(iridescence as GoldenFile, {
    name: 'IridescenceDielectricSpheres-slice',
    purpose: 'IridescenceDielectricSpheres reduced to the 7 x 7 slice with film ior 1.33 (49 spheres: base ior 1-2 x film thickness 0.1-0.7 um on a 0.8 grey dielectric, roughness 0.1), seen down -Z under the sky; the other 294 spheres and the guides plane are dropped. Tier 1 iridescence (P2.27); coverage only.',
    camera: { type: 'perspective', position: [0, 0, 30], target: [0, 0, -3.5], up: [0, 1, 0], yFovDeg: 44 },
    film: { width: 768, height: 768, filter: 'box' },
    integrator: { maxDepth: 4 },
    environment: SKY,
    dropMaterials: (iridescence as GoldenFile).materials.filter((m) => m.i % 7 !== 2 || m.i >= 343).map((m) => m.i),
  }, 'gltf'),
  // 4 x 4 spheres sheenColor 0..1 x sheenRoughness 0..1 in front of a backdrop at z = -0.04.
  sheen: make(sheen as GoldenFile, {
    name: 'SheenTestGrid-notex',
    purpose: 'Khronos SheenTestGrid without textures (text dropped, checker backdrop = grey Lambertian): 16 spheres, sheenColor 0-1 x sheenRoughness 0-1. Tier 1 sheen (P2.25); coverage only.',
    camera: { type: 'perspective', position: [0, 0.055, 1.2], target: [0, 0.055, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 768, height: 768, filter: 'box' },
    integrator: { maxDepth: 8 },
    environment: SKY,
    dropMaterials: ['Text', 'Text Backdrop'],
    replace: { Backdrop: GREY },
  }),
};

/** Every static Tier 1 glTF scene with its golden, model reference and sidecar (tests rebuild them from the download). */
export const tier1GltfScenes: readonly Tier1GltfScene[] = Object.values(sidecars);

/** The `SceneDesc`s, registered in `scenes/index.ts` (`--scenes AttenuationTest-notex,...`). */
export const tier1GltfSceneDescs: readonly SceneDesc[] = tier1GltfScenes.map((s) => s.scene);
