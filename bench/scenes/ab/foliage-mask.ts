import type { SceneDesc } from '../schema.ts';
import { compose, rotationX, rotationY, rotationZ, translation } from '../geometry.ts';

/**
 * `foliage-mask` (R2): many alpha-MASK cutout cards over a lit ground, the case the alpha policy is about. A 6 x 6 grid of 1.0 m cards, each tilted a
 * little differently and lifted to one of three heights, so they overlap and shadow each other and the ground; the texture is a 6 x 6 checker whose
 * `b` cells are holes (alpha 0, so half of every card is air). Sun and uniform sky, `maxDepth` 4, so camera rays pass through holes, shadow rays must
 * pass through holes (exact) or stop at the card (`opaqueShadow`), and bounce rays hit cards from behind. Diffuse only (`specularWeight` 0) so Mitsuba
 * (mask BSDF on `volpath`) and pbrt-v4 (alpha texture) can draw it. Deterministic (no random numbers: the tilts come from a fixed integer hash).
 */
function foliage(): SceneDesc {
  const hash = (i: number, k: number): number => ((((i * 2654435761 + k * 40503) >>> 0) % 1000) / 1000) * 2 - 1; // [-1, 1)
  const meshes: SceneDesc['meshes'] = [{ name: 'ground', geometry: { type: 'quad', size: [10, 10] }, material: 'ground' }];
  const n = 6;
  for (let iz = 0; iz < n; iz++) {
    for (let ix = 0; ix < n; ix++) {
      const i = iz * n + ix;
      const x = (ix - (n - 1) / 2) * 0.85 + 0.15 * hash(i, 1);
      const z = (iz - (n - 1) / 2) * 0.85 + 0.15 * hash(i, 2);
      const y = 0.45 + 0.3 * ((ix + 2 * iz) % 3);
      meshes.push({ name: `card${i}`, geometry: { type: 'quad', size: [1, 1] }, material: 'leaf', transform: compose(translation(x, y, z), rotationY(Math.PI * hash(i, 3)), rotationX(0.35 * hash(i, 4)), rotationZ(0.35 * hash(i, 5))), uv: {} });
    }
  }
  return {
    version: 1,
    name: 'foliage-mask',
    purpose: 'Alpha-MASK foliage stand-in: 36 overlapping checker cards (half holes) over a lit ground, sun and sky. A/B of the per-ray-class alpha policy (R2) and the wavefront alpha MASK (P3.7). Own references only (pbrt-v4 / Mitsuba possible, not required).',
    tier: 1,
    camera: { type: 'perspective', position: [0, 4.2, 5.2], target: [0, 0.4, 0], up: [0, 1, 0], yFovDeg: 42 },
    film: { width: 320, height: 240, filter: 'box' },
    integrator: { maxDepth: 4 },
    textures: {
      leaf: { generator: 'checker', cells: 6, a: [70, 160, 60], b: [200, 220, 90], alphaA: 255, alphaB: 0, width: 96, height: 96, colorSpace: 'srgb' },
    },
    materials: {
      ground: { type: 'diffuse', reflectance: [0.55, 0.5, 0.4] },
      leaf: { type: 'openpbr', baseColor: [1, 1, 1], baseMetalness: 0, specularWeight: 0, baseColorTexture: 'leaf', alphaMode: 'mask', alphaCutoff: 0.5, doubleSided: true },
    },
    meshes,
    lights: [{ type: 'directional', direction: [-0.45, -0.8, -0.35], irradiance: [3.2, 3.0, 2.7] }],
    environment: { type: 'uniform', radiance: [0.45, 0.5, 0.6] },
  };
}

export const foliageMask: SceneDesc = foliage();
export const foliageScenes: readonly SceneDesc[] = [foliageMask];
