import type { MeshDesc, SceneDesc, Vec3 } from '../schema.ts';
import { cross3 } from '../geometry.ts';

/**
 * Classic Cornell box (Goral et al. data as distributed by the Cornell PCG lab, 556-unit room) with 1 unit = 1 mm, so
 * the room is ~0.556 m: the physical box size. Coordinates are right-handed; the camera sits at z = -0.8 m looking
 * toward +z, which puts the red wall (x = 0.553) on the left of the image.
 * Reflectances are the widely used RGB values (red 0.63/0.065/0.05, green 0.14/0.45/0.091, white 0.725/0.71/0.68).
 */
const mm = (v: readonly number[]): Vec3 => [v[0]! * 0.001, v[1]! * 0.001, v[2]! * 0.001];

type Quad = readonly [Vec3, Vec3, Vec3, Vec3];

/** Merge quads into one triangle mesh, flipping each so its geometric normal points toward `facing`. */
function quads(list: Quad[], facing: (centroid: Vec3) => Vec3): { positions: number[]; indices: number[] } {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const q of list) {
    const c: Vec3 = [
      (q[0][0] + q[1][0] + q[2][0] + q[3][0]) / 4,
      (q[0][1] + q[1][1] + q[2][1] + q[3][1]) / 4,
      (q[0][2] + q[1][2] + q[2][2] + q[3][2]) / 4,
    ];
    const n = cross3(
      [q[1][0] - q[0][0], q[1][1] - q[0][1], q[1][2] - q[0][2]],
      [q[2][0] - q[0][0], q[2][1] - q[0][1], q[2][2] - q[0][2]],
    );
    const f = facing(c);
    const flip = n[0] * (f[0] - c[0]) + n[1] * (f[1] - c[1]) + n[2] * (f[2] - c[2]) < 0;
    const order = flip ? [0, 3, 2, 1] : [0, 1, 2, 3];
    const base = positions.length / 3;
    for (const k of order) positions.push(...q[k]!);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  return { positions, indices };
}

const roomCentre: Vec3 = [0.278, 0.274, 0.28];
const inRoom = () => roomCentre;
const outOf = (centre: Vec3) => (c: Vec3): Vec3 => [2 * c[0] - centre[0], 2 * c[1] - centre[1], 2 * c[2] - centre[2]];

const q = (a: number[], b: number[], c: number[], d: number[]): Quad => [mm(a), mm(b), mm(c), mm(d)];

const floor = q([552.8, 0, 0], [0, 0, 0], [0, 0, 559.2], [549.6, 0, 559.2]);
const ceiling = q([556, 548.8, 0], [556, 548.8, 559.2], [0, 548.8, 559.2], [0, 548.8, 0]);
const back = q([549.6, 0, 559.2], [0, 0, 559.2], [0, 548.8, 559.2], [556, 548.8, 559.2]);
const green = q([0, 0, 559.2], [0, 0, 0], [0, 548.8, 0], [0, 548.8, 559.2]);
const red = q([552.8, 0, 0], [549.6, 0, 559.2], [556, 548.8, 559.2], [556, 548.8, 0]);
// Slightly below the ceiling so the emitter never z-fights with it.
const light = q([343, 548.7, 227], [343, 548.7, 332], [213, 548.7, 332], [213, 548.7, 227]);

const shortBlock: Quad[] = [
  q([130, 165, 65], [82, 165, 225], [240, 165, 272], [290, 165, 114]),
  q([290, 0, 114], [290, 165, 114], [240, 165, 272], [240, 0, 272]),
  q([130, 0, 65], [130, 165, 65], [290, 165, 114], [290, 0, 114]),
  q([82, 0, 225], [82, 165, 225], [130, 165, 65], [130, 0, 65]),
  q([240, 0, 272], [240, 165, 272], [82, 165, 225], [82, 0, 225]),
  q([82, 0, 225], [130, 0, 65], [290, 0, 114], [240, 0, 272]),
];
const tallBlock: Quad[] = [
  q([423, 330, 247], [265, 330, 296], [314, 330, 456], [472, 330, 406]),
  q([423, 0, 247], [423, 330, 247], [472, 330, 406], [472, 0, 406]),
  q([472, 0, 406], [472, 330, 406], [314, 330, 456], [314, 0, 456]),
  q([314, 0, 456], [314, 330, 456], [265, 330, 296], [265, 0, 296]),
  q([265, 0, 296], [265, 330, 296], [423, 330, 247], [423, 0, 247]),
  q([265, 0, 296], [423, 0, 247], [472, 0, 406], [314, 0, 456]),
];

const tri = (list: Quad[], facing: (c: Vec3) => Vec3) => {
  const { positions, indices } = quads(list, facing);
  return { type: 'triangles' as const, positions, indices };
};

const meshes: MeshDesc[] = [
  { name: 'white-walls', geometry: tri([floor, ceiling, back], inRoom), material: 'white' },
  { name: 'red-wall', geometry: tri([red], inRoom), material: 'red' },
  { name: 'green-wall', geometry: tri([green], inRoom), material: 'green' },
  { name: 'short-block', geometry: tri(shortBlock, outOf(mm([185, 82.5, 169]))), material: 'white' },
  { name: 'tall-block', geometry: tri(tallBlock, outOf(mm([368, 165, 351]))), material: 'white' },
  // Faces down (toward the room); the back face is black.
  { name: 'light', geometry: tri([light], inRoom), material: 'black', emission: [18.4, 14.0, 6.75] },
];

export const cornell: SceneDesc = {
  version: 1,
  name: 'cornell',
  purpose: 'Classic Cornell box (metres, red/green/white diffuse, ceiling area light, two blocks), 512x512, maxDepth 32 without Russian roulette: colour bleeding, area-light NEE and full indirect transport; no analytic value (pbrt/Mitsuba converged agreement is the reference).',
  tier: 0,
  camera: { type: 'perspective', position: [0.278, 0.273, -0.8], target: [0.278, 0.273, 0], up: [0, 1, 0], yFovDeg: 39.3 },
  film: { width: 512, height: 512, filter: 'box' },
  integrator: { maxDepth: 32 },
  materials: {
    white: { type: 'diffuse', reflectance: [0.725, 0.71, 0.68] },
    red: { type: 'diffuse', reflectance: [0.63, 0.065, 0.05] },
    green: { type: 'diffuse', reflectance: [0.14, 0.45, 0.091] },
    black: { type: 'black' },
  },
  meshes,
};
