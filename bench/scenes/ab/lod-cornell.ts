import type { SceneDesc } from '../schema.ts';
import { cornell } from '../tier0/cornell.ts';

/**
 * Stress scene for the ray-class material LOD (R1, `integrator.materialLod`): the Cornell box with every surface an `openpbr` material the reduction touches, so indirect
 * light (maxDepth 8) mostly bounces off reduced vertices. Walls and short block: dielectric over diffuse (rough 0.35); red wall: coated (coat roughness 0.1, IOR 1.6); green
 * wall: fuzz; tall block: coated rough gold. No pbrt/Mitsuba reference (a fuzz / coat wall is class B): judge the reduced render against our own full one
 * (`bench/adapters/lod-agreement.ts`).
 */
export const lodCornell: SceneDesc = {
  ...cornell,
  name: 'lod-cornell',
  purpose: 'R1 stress scene: Cornell box with layered, coated, fuzzy and coated-metal OpenPBR surfaces, maxDepth 8, 256x256; compare the reduced secondary material with the full one.',
  tier: 1,
  film: { width: 256, height: 256, filter: 'box' },
  integrator: { maxDepth: 8 },
  materials: {
    white: { type: 'openpbr', baseColor: [0.725, 0.71, 0.68], specularRoughness: 0.35 },
    red: { type: 'openpbr', baseColor: [0.63, 0.065, 0.05], specularRoughness: 0.5, coatWeight: 1, coatRoughness: 0.1, coatIor: 1.6 },
    green: { type: 'openpbr', baseColor: [0.14, 0.45, 0.091], specularWeight: 0, fuzzWeight: 0.8, fuzzColor: [0.9, 1, 0.9], fuzzRoughness: 0.5 },
    gold: { type: 'openpbr', baseColor: [1, 0.78, 0.34], baseMetalness: 1, specularRoughness: 0.35, coatWeight: 1, coatRoughness: 0.3, coatIor: 1.5 },
    black: { type: 'black' },
  },
  meshes: cornell.meshes.map((m) => (m.name === 'tall-block' ? { ...m, material: 'gold' } : m)),
};
