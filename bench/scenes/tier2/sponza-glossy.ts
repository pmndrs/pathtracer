import type { MeshDesc, SceneDesc } from '../schema.ts';
import { SPONZA_GLTF, sponzaWhite } from './sponza-white.ts';

/**
 * sponza-glossy (track N, N7b): sponza-white with its floor and ground-floor columns turned into grey rough metal, so the radiance a
 * cache has to store at those vertices depends on the direction it is seen from. The question it answers is whether the network's
 * direction and roughness inputs beat a view-independent hash grid there (M7 / Q4, docs/design/nrc-hash-cache.md).
 *
 * Sponza's glTF materials have no names; picked by geometry (bench analysis 2026-11-18): material 9 is the floor of both storeys
 * (360 m², all faces horizontal, y = 0 and 3.3), 17-19 are the ground-floor column shafts (vertical, y 0..2.27). Everything else
 * keeps sponza-white's grey Lambertian. One file mesh per glTF material (`select.material`), so the web export has one node per
 * material in this order. Conductor: n = 1.5, k = 3 for every channel (normal-incidence reflectance 0.61, grey); the floor is
 * polished (roughness 0.15), the columns satin (0.3). Same lights, camera and path length as sponza-white. No native references:
 * the NRC pages render their own (`ref=ours`).
 */
const SPONZA_MATERIALS = 25;
const FLOOR = 9;
const COLUMNS = new Set([17, 18, 19]);

const meshes: MeshDesc[] = Array.from({ length: SPONZA_MATERIALS }, (_, i) => ({
  name: `sponza-m${i}`,
  geometry: { type: 'file', path: SPONZA_GLTF.path, format: 'gltf', sha256: SPONZA_GLTF.sha256, select: { material: i } },
  material: i === FLOOR ? 'floor' : COLUMNS.has(i) ? 'column' : 'grey',
}));

export const sponzaGlossy: SceneDesc = {
  ...sponzaWhite,
  name: 'sponza-glossy',
  purpose: 'sponza-white with a polished metal floor (roughness 0.15) and satin metal columns (0.3): view-dependent radiance at cache vertices (track N, N7b).',
  materials: {
    grey: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] },
    floor: { type: 'conductor', eta: [1.5, 1.5, 1.5], k: [3, 3, 3], roughness: 0.15 },
    column: { type: 'conductor', eta: [1.5, 1.5, 1.5], k: [3, 3, 3], roughness: 0.3 },
  },
  meshes,
};
