import type { SceneDesc } from '../schema.ts';
import { compose, translation } from '../geometry.ts';
import { manyLampMeshes } from './many-lights.ts';

/**
 * ReSTIR DI on textured scenes, OpenPBR materials and alpha-MASK primaries (P4.2b): the `many-lights-64` lamp grid (64 one-sided 0.1 m quads at four power levels, 1 m up) over
 * a plane with a box and a ball, with the materials the ReSTIR passes must evaluate exactly as the shade pass does. No pbrt / Mitsuba reference (the lamp count makes them slow, and
 * the point is agreement of reservoir resampling with plain NEE): judge against our own high-spp NEE render of the same scene.
 *
 *  - `restir-tex-lamps`: legacy materials with textures: a checker floor, a ball with a checker base colour, a tangent-space bump normal map (shading normal != geometric)
 *    and a metallic-roughness texture of stripes alternating rough dielectric and smooth metal (the per-texel metalness picks a lobe), a plain metal box.
 *  - `restir-pbr-lamps`: OpenPBR mode, no textures: a dielectric-over-diffuse floor (layered lobe, `albedo` layering), a rough gold ball (conductor), a box at metalness 0.5
 *    (the stochastic mix: one lobe per pixel and frame, so the G-buffer carries the pick).
 *  - `restir-mask-lamps`: OpenPBR + textures + alpha MASK: a checker screen (half of it holes) floats 0.5 m over the ground, between the lamps and the floor: camera rays pass through
 *    the holes, the visibility of a reservoir sample must too (`ptVisibleMasked`), and the G-buffer must record the hit behind a hole, not the card.
 */
function lampScene(name: string, purpose: string, extra: Partial<SceneDesc> & Pick<SceneDesc, 'materials' | 'meshes'>): SceneDesc {
  const { materials, meshes, ...rest } = extra;
  return {
    version: 1,
    name,
    purpose,
    tier: 1,
    camera: { type: 'perspective', position: [0, 2.2, 3.6], target: [0, 0, 0.2], up: [0, 1, 0], yFovDeg: 50 },
    film: { width: 256, height: 256, filter: 'box' },
    integrator: { maxDepth: 2 },
    materials: { ...materials, lightBack: { type: 'black' } },
    meshes: [...meshes, ...manyLampMeshes(8)],
    ...rest,
  };
}

export const restirTexLamps: SceneDesc = lampScene(
  'restir-tex-lamps',
  '64 lamps over a textured plane, ball (bump normal map) and box (metallic-roughness stripes), legacy materials: ReSTIR DI with texture lookups and normal maps in the target (P4.2b). Own references only.',
  {
    textures: {
      floor: { generator: 'checker', cells: 8, a: [235, 235, 235], b: [60, 110, 190], width: 128, height: 128, colorSpace: 'srgb' },
      ball: { generator: 'checker', cells: 6, a: [220, 60, 50], b: [240, 220, 150], width: 64, height: 64, colorSpace: 'srgb' },
      bump: { generator: 'bumps', periods: 12, amplitude: 0.6, width: 64, height: 64, colorSpace: 'linear' },
      stripes: { generator: 'stripes-mr', periods: 6, a: [0.9, 0], b: [0.25, 1], width: 64, height: 64, colorSpace: 'linear' },
    },
    materials: {
      floor: { type: 'diffuse', reflectance: [1, 1, 1], reflectanceTexture: 'floor' },
      ball: { type: 'pbr', baseColor: [1, 1, 1], roughness: 1, metalness: 1, baseColorTexture: 'ball', normalTexture: 'bump', metallicRoughnessTexture: 'stripes' },
      box: { type: 'pbr', baseColor: [0.9, 0.75, 0.5], roughness: 0.45, metalness: 1 },
    },
    meshes: [
      { name: 'plane', geometry: { type: 'quad', size: [8, 8] }, material: 'floor', uv: { scale: [4, 4] } },
      { name: 'box', geometry: { type: 'box', size: [0.8, 0.6, 0.8] }, material: 'box', transform: translation(-0.9, 0.3, 0.2) },
      { name: 'ball', geometry: { type: 'sphere', radius: 0.45, segments: 48, rings: 24 }, material: 'ball', transform: translation(0.9, 0.45, -0.2), uv: { scale: [2, 1] } },
    ],
  },
);

export const restirPbrLamps: SceneDesc = lampScene(
  'restir-pbr-lamps',
  '64 lamps over an OpenPBR plane (dielectric over diffuse), a rough gold ball and a metalness-0.5 box (stochastic mix): ReSTIR DI with the OpenPBR adapter in the target, layering albedo (P4.2b). Own references only.',
  {
    materials: {
      floor: { type: 'openpbr', baseColor: [0.5, 0.45, 0.4], specularRoughness: 0.3 },
      gold: { type: 'openpbr', baseColor: [1, 0.77, 0.34], baseMetalness: 1, specularRoughness: 0.35 },
      mix: { type: 'openpbr', baseColor: [0.2, 0.5, 0.8], baseMetalness: 0.5, specularRoughness: 0.4 },
    },
    meshes: [
      { name: 'plane', geometry: { type: 'quad', size: [8, 8] }, material: 'floor' },
      { name: 'box', geometry: { type: 'box', size: [0.8, 0.6, 0.8] }, material: 'mix', transform: translation(-0.9, 0.3, 0.2) },
      { name: 'ball', geometry: { type: 'sphere', radius: 0.45, segments: 48, rings: 24 }, material: 'gold', transform: translation(0.9, 0.45, -0.2) },
    ],
  },
);

export const restirMaskLamps: SceneDesc = lampScene(
  'restir-mask-lamps',
  '64 lamps over a ground with an alpha-MASK checker screen (half holes) between them: ReSTIR DI with cutout primaries and masked shadow rays (P4.2b). Own references only.',
  {
    textures: {
      screen: { generator: 'checker', cells: 8, a: [235, 150, 40], b: [60, 110, 190], alphaA: 255, alphaB: 0, width: 128, height: 128, colorSpace: 'srgb' },
    },
    materials: {
      ground: { type: 'openpbr', baseColor: [0.6, 0.6, 0.6], specularRoughness: 0.5, specularWeight: 0.3 },
      screen: { type: 'openpbr', baseColor: [1, 1, 1], baseMetalness: 0, specularWeight: 0, baseColorTexture: 'screen', alphaMode: 'mask', alphaCutoff: 0.5, doubleSided: true },
    },
    meshes: [
      { name: 'ground', geometry: { type: 'quad', size: [8, 8] }, material: 'ground' },
      { name: 'screen', geometry: { type: 'quad', size: [2.4, 2.4] }, material: 'screen', transform: compose(translation(0, 0.5, 0)), uv: {} },
    ],
  },
);

/** `restir-pbr-lamps` with the box on the gold material (no stochastic mix): separates the mix pick from the layered floor in a bias hunt. */
export const restirPbrLampsNoMix: SceneDesc = { ...restirPbrLamps, name: 'restir-pbr-lamps-nomix', purpose: 'restir-pbr-lamps without the metalness-0.5 box (it is gold): no stochastic material pick.', meshes: restirPbrLamps.meshes.map((m) => (m.material === 'mix' ? { ...m, material: 'gold' } : m)) };

/** Bias hunt variants: everything gold (conductor lobe only) and everything Lambert (OpenPBR adapter, specularWeight 0). */
const retarget = (name: string, materials: SceneDesc['materials'], purpose: string): SceneDesc => ({ ...restirPbrLamps, name, purpose, materials: { ...materials, lightBack: { type: 'black' } }, meshes: restirPbrLamps.meshes });
export const restirPbrLampsGold: SceneDesc = retarget('restir-pbr-lamps-gold', { floor: { type: 'openpbr', baseColor: [1, 0.77, 0.34], baseMetalness: 1, specularRoughness: 0.35 }, gold: { type: 'openpbr', baseColor: [1, 0.77, 0.34], baseMetalness: 1, specularRoughness: 0.35 }, mix: { type: 'openpbr', baseColor: [1, 0.77, 0.34], baseMetalness: 1, specularRoughness: 0.35 } }, 'restir-pbr-lamps with every surface rough gold (conductor lobe only).');
export const restirPbrLampsLambert: SceneDesc = retarget('restir-pbr-lamps-lambert', { floor: { type: 'openpbr', baseColor: [0.5, 0.45, 0.4], specularWeight: 0 }, gold: { type: 'openpbr', baseColor: [0.8, 0.6, 0.3], specularWeight: 0 }, mix: { type: 'openpbr', baseColor: [0.2, 0.5, 0.8], specularWeight: 0 } }, 'restir-pbr-lamps with every surface Lambertian through the OpenPBR adapter.');

/** Bias hunt (P4.2b): one material kind at a time on the otherwise Lambertian scene (`lam` = Lambert via OpenPBR, specularWeight 0). */
const lam = (c: [number, number, number]) => ({ type: 'openpbr' as const, baseColor: c, specularWeight: 0 });
const layeredFloor = restirPbrLamps.materials.floor as SceneDesc['materials'][string];
const mixMat = restirPbrLamps.materials.mix as SceneDesc['materials'][string];
const goldMat = restirPbrLamps.materials.gold as SceneDesc['materials'][string];
export const restirPbrLampsFloor: SceneDesc = retarget('restir-pbr-lamps-floor', { floor: layeredFloor, gold: lam([0.8, 0.6, 0.3]), mix: lam([0.2, 0.5, 0.8]) }, 'restir-pbr-lamps with only the floor layered (dielectric over diffuse), the rest Lambert.');
export const restirPbrLampsMixOnly: SceneDesc = retarget('restir-pbr-lamps-mixonly', { floor: lam([0.5, 0.45, 0.4]), gold: lam([0.8, 0.6, 0.3]), mix: mixMat }, 'restir-pbr-lamps with only the box at metalness 0.5 (stochastic mix), the rest Lambert.');
export const restirPbrLampsGoldOnly: SceneDesc = retarget('restir-pbr-lamps-goldonly', { floor: lam([0.5, 0.45, 0.4]), gold: goldMat, mix: lam([0.2, 0.5, 0.8]) }, 'restir-pbr-lamps with only the ball gold, the rest Lambert.');

/**
 * `restir-ground-sun` (V1.7): the offset-bound diagnostic without media. A 2 km Lambert ground (two triangles, vertices 1 km from the origin) under a sun 30 degrees off the zenith, a 40 m black box
 * 100 m aside (out of the camera's view), camera 0.3 m above the origin looking straight down at 1 degree: direct light only, pixel = rho / pi E cos = 0.5 / pi * 3 * cos 30.
 * A shadow ray that starts inside the ground's own triangle error hits it and darkens the pixel.
 */
export const restirGroundSun: SceneDesc = {
  version: 1,
  name: 'restir-ground-sun',
  purpose: 'A 2 km ground under a sun with the camera 0.3 m above the origin: pixel = 0.5 / pi * 3 * cos 30 = 0.41350 (direct, maxDepth 1). Shadow rays leaving the ground at a point far from its vertices (V1.7).',
  tier: 1,
  camera: { type: 'perspective', position: [0, 0.3, 0.0001], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 1 },
  film: { width: 32, height: 32, filter: 'box' },
  integrator: { maxDepth: 1 },
  materials: { ground: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] }, black: { type: 'diffuse', reflectance: [0, 0, 0] } },
  meshes: [
    { name: 'ground', geometry: { type: 'quad', size: [2000, 2000] }, material: 'ground' },
    { name: 'box', geometry: { type: 'box', size: [40, 1, 40] }, material: 'black', transform: translation(100, 1, 0) },
  ],
  lights: [{ type: 'directional', direction: [-0.5, -Math.cos((30 * Math.PI) / 180), 0], irradiance: [3, 3, 3] }],
  expected: { type: 'mean', value: [0.4135, 0.4135, 0.4135], tolerance: 0.002, pixelwise: false },
};

export const restirScenes: readonly SceneDesc[] = [restirGroundSun, restirTexLamps, restirPbrLamps, restirMaskLamps, restirPbrLampsNoMix, restirPbrLampsGold, restirPbrLampsLambert, restirPbrLampsFloor, restirPbrLampsMixOnly, restirPbrLampsGoldOnly];
