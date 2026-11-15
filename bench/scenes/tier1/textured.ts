import type { SceneDesc, TextureDesc } from '../schema.ts';
import { compose, translation } from '../geometry.ts';

/**
 * Tier 1 textured scenes (P1.4c), procedural textures only. Both read the same texels in every renderer (PNG for pbrt-v4 / Mitsuba).
 *
 * `tex-checker`: a ground quad and a sphere with an sRGB checker (an orange marker cell in the top-left cell of every tile shows the
 * uv orientation), lit by a delta sun and a uniform sky, diffuse only so pbrt-v4 and Mitsuba can draw it. The ground is tiled 4x4 and its
 * 128 px texture reads about 8 texels per pixel at the far edge, the most pbrt's mip choice (it filters with ray differentials scaled by
 * 1/8 at high spp) leaves on level 0; the sphere carries a second, smaller texture (two size buckets).
 *
 * The glTF maps beyond base colour (metallic-roughness, normal, emissive) have no native reference; `bench/pages/texture-check.ts` checks them against closed forms.
 */
const checker = (cells: number, size: number, a: [number, number, number], b: [number, number, number]): TextureDesc => ({
  generator: 'checker',
  cells,
  a,
  b,
  marker: [230, 120, 20],
  width: size,
  height: size,
  colorSpace: 'srgb',
});

/** `floorSize` x `floorSize` texels with `floorCells` cells across one tile; the high-frequency variant minifies up to ~70 texels per pixel at the far edge. */
function checkerScene(name: string, purpose: string, floorSize: number, floorCells: number): SceneDesc {
  return {
    version: 1,
    name,
    purpose,
    tier: 1,
    camera: { type: 'perspective', position: [0, 1.3, 3.4], target: [0, 0.25, 0], up: [0, 1, 0], yFovDeg: 42 },
    film: { width: 320, height: 240, filter: 'box' },
    integrator: { maxDepth: 3 },
    textures: {
      floor: checker(floorCells, floorSize, [235, 235, 235], [60, 110, 190]),
      ball: checker(4, 64, [220, 60, 50], [240, 220, 150]),
    },
    materials: {
      floor: { type: 'diffuse', reflectance: [1, 1, 1], reflectanceTexture: 'floor' },
      ball: { type: 'diffuse', reflectance: [1, 1, 1], reflectanceTexture: 'ball' },
    },
    meshes: [
      { name: 'floor', geometry: { type: 'quad', size: [6, 6] }, material: 'floor', uv: { scale: [4, 4] } },
      {
        name: 'ball',
        geometry: { type: 'sphere', radius: 0.6, segments: 64, rings: 32 },
        material: 'ball',
        transform: compose(translation(0, 0.6, 0)),
        uv: { scale: [2, 1] },
      },
    ],
    lights: [{ type: 'directional', direction: [-0.5, -0.75, -0.4], irradiance: [3.2, 3.0, 2.7] }],
    environment: { type: 'uniform', radiance: [0.45, 0.5, 0.6] },
  };
}

export const texChecker = checkerScene(
  'tex-checker',
  'Image textures through the scene copy: an sRGB checker with an orientation marker on a ground quad and a sphere, bilinear lookups, sun and sky. Compared with Mitsuba bitmap and pbrt imagemap.',
  128,
  8,
);

export const texCheckerHf = checkerScene(
  'tex-checker-hf',
  'tex-checker with a 1024 px, 128-cell floor texture: up to ~70 texels per pixel at the far edge, so level of detail matters. A/B of textureLod 0 vs ray cones against Mitsuba (pbrt-v4 picks its own mip level here).',
  1024,
  128,
);

/**
 * `tex-mask` (P1.4c5, P2.24): an alpha-MASK cutout. A 2.4 m square screen floats 0.5 m above a grey ground; its base colour texture is an
 * 8 x 8 checker whose orange cells have alpha 1 and whose blue cells alpha 100/255 (below the 0.5 cutoff, so they are holes). The bilinear alpha
 * ramps 255 -> 100 across a texel, so the cutoff contour sits 82% of the way to the hole: it checks the interpolated threshold, not just the cell
 * pattern. The sun throws the screen's shadow on the ground, so shadow rays must pass through the holes. Diffuse only (`specularWeight` 0), so
 * pbrt-v4 (alpha texture) and Mitsuba (mask BSDF on `volpath`) can draw it.
 */
export const texMask: SceneDesc = {
  version: 1,
  name: 'tex-mask',
  purpose: 'Alpha MASK from a base colour texture alpha (hard cutoff 0.5 on the bilinear alpha): a checker screen whose blue cells are holes, over a ground that receives its shadow. Compared with pbrt-v4 (alpha texture) and Mitsuba (mask BSDF).',
  tier: 1,
  camera: { type: 'perspective', position: [0, 2.4, 3.6], target: [0, 0.25, 0], up: [0, 1, 0], yFovDeg: 42 },
  film: { width: 320, height: 240, filter: 'box' },
  integrator: { maxDepth: 3 },
  textures: {
    screen: { generator: 'checker', cells: 8, a: [235, 150, 40], b: [60, 110, 190], marker: [230, 40, 40], alphaA: 255, alphaB: 100, width: 128, height: 128, colorSpace: 'srgb' },
  },
  materials: {
    ground: { type: 'diffuse', reflectance: [0.6, 0.6, 0.6] },
    screen: { type: 'openpbr', baseColor: [1, 1, 1], baseMetalness: 0, specularWeight: 0, baseColorTexture: 'screen', alphaMode: 'mask', alphaCutoff: 0.5, doubleSided: true },
  },
  meshes: [
    { name: 'ground', geometry: { type: 'quad', size: [8, 8] }, material: 'ground' },
    { name: 'screen', geometry: { type: 'quad', size: [2.4, 2.4] }, material: 'screen', transform: compose(translation(0, 0.5, 0)), uv: {} },
  ],
  lights: [{ type: 'directional', direction: [-0.35, -0.8, -0.3], irradiance: [3.2, 3.0, 2.7] }],
  environment: { type: 'uniform', radiance: [0.45, 0.5, 0.6] },
};

export const tier1Scenes: readonly SceneDesc[] = [texChecker, texCheckerHf, texMask];
