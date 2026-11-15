import type { SceneDesc } from '../schema.ts';
import { compose, rotationX, rotationY, rotationZ, translation } from '../geometry.ts';

/**
 * `foliage-mask` (R2): many alpha-MASK cutout cards over a lit ground, the case the alpha policy is about. A 6 x 6 grid of 1.0 m cards, each tilted a
 * little differently and lifted to one of three heights, so they overlap and shadow each other and the ground; the texture is a 6 x 6 checker whose
 * `b` cells are holes (alpha 0, so half of every card is air). Sun and uniform sky, `maxDepth` 4, so camera rays pass through holes, shadow rays must
 * pass through holes (exact) or stop at the card (`opaqueShadow`), and bounce rays hit cards from behind. Diffuse only (`specularWeight` 0) so Mitsuba
 * (mask BSDF on `volpath`) and pbrt-v4 (alpha texture) can draw it. Deterministic (no random numbers: the tilts come from a fixed integer hash).
 */
interface FoliageVariant {
  name: string;
  purpose: string;
  /** Cards per side (default 6). */
  n?: number;
  /** Distance between card centres (default 0.85). */
  spacing?: number;
  /** Card grid cells per side (default 1: two triangles per card). Cutout geometry finer than the texture's cells (R2b). */
  segments?: number;
  /** Checker cells per card side (default 6). */
  cells?: number;
  /** Texture size in texels (default 96; a power of two such as 128 is stored as it is, 96 is resampled to 128 and the checker edges leave the texel grid). */
  texSize?: number;
}

function foliage(v: FoliageVariant = { name: 'foliage-mask', purpose: '' }): SceneDesc {
  const n = v.n ?? 6;
  const spacing = v.spacing ?? 0.85;
  const segments = v.segments ?? 1;
  const hash = (i: number, k: number): number => ((((i * 2654435761 + k * 40503) >>> 0) % 1000) / 1000) * 2 - 1; // [-1, 1)
  const meshes: SceneDesc['meshes'] = [{ name: 'ground', geometry: { type: 'quad', size: [10, 10] }, material: 'ground' }];
  for (let iz = 0; iz < n; iz++) {
    for (let ix = 0; ix < n; ix++) {
      const i = iz * n + ix;
      const x = (ix - (n - 1) / 2) * spacing + 0.15 * hash(i, 1);
      const z = (iz - (n - 1) / 2) * spacing + 0.15 * hash(i, 2);
      const y = 0.45 + 0.3 * ((ix + 2 * iz) % 3);
      meshes.push({ name: `card${i}`, geometry: segments > 1 ? { type: 'quad', size: [1, 1], segments: [segments, segments] } : { type: 'quad', size: [1, 1] }, material: 'leaf', transform: compose(translation(x, y, z), rotationY(Math.PI * hash(i, 3)), rotationX(0.35 * hash(i, 4)), rotationZ(0.35 * hash(i, 5))), uv: {} });
    }
  }
  return {
    version: 1,
    name: v.name,
    purpose: v.purpose || 'Alpha-MASK foliage stand-in: 36 overlapping checker cards (half holes) over a lit ground, sun and sky. A/B of the per-ray-class alpha policy (R2) and the wavefront alpha MASK (P3.7). Own references only (pbrt-v4 / Mitsuba possible, not required).',
    tier: 1,
    camera: { type: 'perspective', position: [0, 4.2, 5.2], target: [0, 0.4, 0], up: [0, 1, 0], yFovDeg: 42 },
    film: { width: 320, height: 240, filter: 'box' },
    integrator: { maxDepth: 4 },
    textures: {
      leaf: { generator: 'checker', cells: v.cells ?? 6, a: [70, 160, 60], b: [200, 220, 90], alphaA: 255, alphaB: 0, width: v.texSize ?? 96, height: v.texSize ?? 96, colorSpace: 'srgb' },
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
/**
 * R2b variants of `foliage-mask` (same camera, lights, material):
 *  - `foliage-mask-dense`: 10 x 10 cards 0.5 m apart (100 cards, 2.8x the cards per square metre): rays cross several cutout cards, so exact and solid shadows differ most.
 *  - `foliage-mask-grid16`: the 6 x 6 cards as 16 x 16 grids over an 8 x 8 checker in a 128 px texture (stored as it is): every triangle lies in one checker cell and the
 *    contour sits on texel boundaries, so no triangle is mixed (cut-out geometry: the best case of a per-triangle classification).
 *  - `foliage-mask-grid12`: 12 x 12 grids against the 8 x 8 cells: a cell edge crosses a third to half of the triangles (the intermediate case).
 */
export const foliageMaskDense: SceneDesc = foliage({ name: 'foliage-mask-dense', n: 10, spacing: 0.5, purpose: 'foliage-mask with 100 cards 0.5 m apart: rays cross several cutout cards (R2b: exact vs solid shadows differ most).' });
export const foliageMaskGrid16: SceneDesc = foliage({ name: 'foliage-mask-grid16', segments: 16, cells: 8, texSize: 128, purpose: 'foliage-mask with every card a 16 x 16 grid aligned with an 8 x 8 checker (128 px): no triangle is mixed (R2b: per-triangle alpha classification, best case).' });
export const foliageMaskGrid12: SceneDesc = foliage({ name: 'foliage-mask-grid12', segments: 12, cells: 8, texSize: 128, purpose: 'foliage-mask with every card a 12 x 12 grid against an 8 x 8 checker (128 px): about half of the triangles straddle a cell edge (R2b: per-triangle alpha classification, unfavourable case).' });
export const foliageScenes: readonly SceneDesc[] = [foliageMask, foliageMaskDense, foliageMaskGrid16, foliageMaskGrid12];
