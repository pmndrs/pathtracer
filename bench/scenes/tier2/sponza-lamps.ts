import type { MeshDesc, SceneDesc } from '../schema.ts';
import { sponzaWhite } from './sponza-white.ts';

/**
 * sponza-lamps (P4.5 / P4.6): the many-light Sponza the real-time chain's ReSTIR check needs. sponza-white's geometry and grey material, no sun,
 * a weak uniform sky (0.1) and `SPONZA_LAMPS` small one-sided quad lamps (0.14 m, facing down) on a 16 x 4 grid at y = 1.0 m (below eye level: the camera sees only their black backs, so no emitter is directly visible) over the
 * nave, in four emission levels (the same mixed-power pattern as many-lights-64). Direct light dominates, so the light-selection and
 * visibility noise that ReSTIR DI addresses is the visible noise. One file mesh for the geometry (flat web export, as sponza-white) + one triangle mesh per
 * emission level. No native references: the real-time pages render their own (maxDepth 2).
 */
const LEVELS = [60, 120, 240, 480];
const NX = 16;
const NZ = 4;
export const SPONZA_LAMPS = NX * NZ;

function lampMeshes(y: number): MeshDesc[] {
  const groups: { x: number; z: number }[][] = LEVELS.map(() => []);
  for (let i = 0; i < NX; i++)
    for (let j = 0; j < NZ; j++) groups[(i * 7 + j * 13 + ((i * j) % 5)) % LEVELS.length]!.push({ x: -9 + ((i + 0.5) * 18) / NX, z: -2.7 + ((j + 0.5) * 5.4) / NZ });
  const h = 0.07;
  return groups.map((lamps, level) => {
    const positions: number[] = [];
    const indices: number[] = [];
    const normals: number[] = [];
    for (const lamp of lamps) {
      const base = positions.length / 3;
      positions.push(lamp.x - h, y, lamp.z - h, lamp.x + h, y, lamp.z - h, lamp.x + h, y, lamp.z + h, lamp.x - h, y, lamp.z + h);
      for (let k = 0; k < 4; k++) normals.push(0, -1, 0);
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    const e = LEVELS[level] as number;
    return { name: `lamps-${e}`, geometry: { type: 'triangles', positions, indices, normals }, material: 'lightBack', emission: [e, e, e] } as MeshDesc;
  });
}

export const sponzaLamps: SceneDesc = {
  ...sponzaWhite,
  name: 'sponza-lamps',
  purpose: `sponza-white lit by ${SPONZA_LAMPS} small ceiling lamps in four power levels (no sun, weak sky): many-light direct lighting on 262k triangles, ReSTIR DI vs NEE through the real-time chain (P4.5 / P4.6). No native reference.`,
  materials: { ...sponzaWhite.materials, lightBack: { type: 'black' } },
  meshes: [...sponzaWhite.meshes, ...lampMeshes(1.0)],
  lights: [],
  environment: { type: 'uniform', radiance: [0.1, 0.11, 0.14] },
};

/**
 * sponza-lamps-visible (P4.5a): the same lamps at y = 2.8 m, above the eye (1.7 m) and inside the camera's frustum (it looks up 5 degrees, 60 degree field of view): every lamp
 * is an emitter seen directly by the primary ray (480 / 240 / 120 / 60 radiance against a surround of 0.1-1), the case that broke the demodulated planes (emission divided by
 * the albedo floor). The lighting differs from sponza-lamps (the lamps are 1.8 m higher), so compare chains on this scene, not against sponza-lamps numbers.
 */
export const sponzaLampsVisible: SceneDesc = {
  ...sponzaLamps,
  name: 'sponza-lamps-visible',
  purpose: `sponza-lamps with the ${SPONZA_LAMPS} lamps at y = 2.8 m, in the primary view: directly visible emitters through the real-time chain (P4.5a). No native reference.`,
  meshes: [...sponzaWhite.meshes, ...lampMeshes(2.8)],
};
