import type { SceneDesc } from '../schema.ts';
import { compose, translation } from '../geometry.ts';
import { manyLampMeshes } from './many-lights.ts';

/**
 * ReSTIR DI on textured scenes, OpenPBR materials and alpha-MASK primaries (P4.2b): the `many-lights-64` lamp grid (64 one-sided 0.1 m quads at four power levels, 1 m up) over
 * a plane with a box and a ball, with the materials the ReSTIR passes must evaluate exactly as the shade pass does. No pbrt / Mitsuba reference (the lamp count makes them slow, and
 * the point is agreement of reservoir resampling with plain NEE): judge against our own high-spp NEE render of the same scene.
 *
 *  - `restir-tex-lamps`: legacy materials with textures: a checker floor, a ball with a checker base colour and a tangent-space bump normal map (shading normal != geometric),
 *    a box whose metallic-roughness texture alternates stripes of rough dielectric and smooth metal (the per-texel metalness picks a lobe).
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
      ball: { type: 'pbr', baseColor: [1, 1, 1], roughness: 0.7, metalness: 0, baseColorTexture: 'ball', normalTexture: 'bump' },
      box: { type: 'pbr', baseColor: [0.9, 0.75, 0.5], roughness: 1, metalness: 1, metallicRoughnessTexture: 'stripes' },
    },
    meshes: [
      { name: 'plane', geometry: { type: 'quad', size: [8, 8] }, material: 'floor', uv: { scale: [4, 4] } },
      { name: 'box', geometry: { type: 'box', size: [0.8, 0.6, 0.8] }, material: 'box', transform: translation(-0.9, 0.3, 0.2), uv: {} },
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

export const restirScenes: readonly SceneDesc[] = [restirTexLamps, restirPbrLamps, restirMaskLamps];
