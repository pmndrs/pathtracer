import type { SceneDesc } from '../schema.ts';

/**
 * Sponza (Khronos glTF-Sample-Assets, Crytek/Cryengine licence, git-ignored download: see bench/assets/MANIFEST.json)
 * with EVERY material replaced by one grey Lambertian (reflectance 0.5). No textures and no alpha cutouts: the three
 * MASK-mode foliage materials of the original (plants, chains) therefore render as SOLID quads in every renderer, which
 * is what makes pbrt-v4, Mitsuba and the web renderers see exactly the same thing. The loader (scenes/gltf-loader.ts)
 * bakes node transforms, un-welds and uses flat geometric normals, so there is no shading-normal ambiguity either.
 *
 * Lighting: a sun (directional, travel direction `SUN_DIRECTION`, angular radius = the real sun's 0.265 deg; pbrt and
 * Mitsuba treat it as a delta direction) plus a uniform sky of constant radiance so the interior gets real indirect light.
 * Values were chosen so the image mean is ~0.25 (target 0.2-0.5): sun irradiance 50 W/m² (a sunlit floor patch reads
 * 0.5/pi*50*cos ~ 7), sky radiance ~2.4 (open-roof-lit surfaces ~ 1-2), so sun, sky and interior bounce all matter.
 * Camera: the classic view down the atrium from inside its west end (x = -8; the end wall is at x = -11.4), eye
 * height 1.7 m, looking toward +X, 60 deg vertical FOV.
 */
export const SPONZA_GLTF = { path: 'downloads/sponza/Sponza.gltf', sha256: '646c10cbc8fab990ca29f363e90e2d65155f3a3569506852eb1434a9465b9501' } as const;

const sunDir = ((): [number, number, number] => {
  const v = [0.3, -0.8, 0.4];
  const l = Math.hypot(v[0]!, v[1]!, v[2]!);
  return [v[0]! / l, v[1]! / l, v[2]! / l];
})();
/** Direction the sunlight TRAVELS (toward the scene), unit length. */
export const SPONZA_SUN_DIRECTION = sunDir;

export const sponzaWhite: SceneDesc = {
  version: 1,
  name: 'sponza-white',
  purpose: 'Sponza atrium, one grey Lambertian (0.5) for every surface, sun + uniform sky: real indirect lighting on 262k triangles, identical in pbrt-v4, Mitsuba and the web renderers (quality + speed tier).',
  tier: 2,
  camera: { type: 'perspective', position: [-8, 1.7, 0], target: [10, 2.6, 0], up: [0, 1, 0], yFovDeg: 60 },
  film: { width: 1280, height: 720, filter: 'box' },
  integrator: { maxDepth: 8, rrStartDepth: 4 },
  materials: { grey: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] } },
  meshes: [{ name: 'sponza', geometry: { type: 'file', path: SPONZA_GLTF.path, format: 'gltf', sha256: SPONZA_GLTF.sha256 }, material: 'grey' }],
  lights: [{ type: 'directional', direction: sunDir, irradiance: [50, 46, 40], angularRadius: 0.00465 }],
  environment: { type: 'uniform', radiance: [2.0, 2.3, 3.0] },
};
