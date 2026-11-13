import type { SceneDesc } from '../schema.ts';
import { tessellateSphere } from '../geometry.ts';

/**
 * Furnace spheres are flat-shaded (each facet has its own geometric normal): a convex polyhedron then conserves energy
 * exactly, whereas smooth shading normals on a coarse mesh let Mitsuba lose ~1.5% along the silhouette (verified:
 * 304 pixels of the smooth dielectric furnace read 0.98 in Mitsuba, pbrt exact).
 */
const flatSphere = (() => {
  const m = tessellateSphere(1, 128, 64, { flat: true });
  return { type: 'triangles' as const, positions: Array.from(m.positions), indices: Array.from(m.indices), normals: Array.from(m.normals) };
})();

const furnaceBase = {
  version: 1 as const,
  tier: 0 as const,
  film: { width: 256, height: 256, filter: 'box' as const },
  environment: { type: 'uniform' as const, radiance: [1, 1, 1] as const },
};

const sphere = (material: string) => ({
  name: 'sphere',
  geometry: flatSphere,
  material,
});

/**
 * Camera 2 m from the sphere centre with 40 degree vertical fov: the whole frame (corners included) lies inside the
 * sphere's silhouette, so every pixel sees the sphere surface and nothing else.
 */
const closeCamera = { type: 'perspective' as const, position: [0, 0, 2] as const, target: [0, 0, 0] as const, up: [0, 1, 0] as const, yFovDeg: 40 };
/** Camera 4 m away, 40 degree fov: sphere plus a margin of plain environment. */
const farCamera = { ...closeCamera, position: [0, 0, 4] as const };

/** Convex diffuse object in a uniform environment: sees only the environment, so radiance = albedo exactly. */
export const furnaceDiffuse: SceneDesc = {
  ...furnaceBase,
  name: 'furnace-diffuse',
  purpose: 'White furnace: convex flat-shaded Lambertian sphere (albedo 0.5, frame fully inside the silhouette) under uniform L=1 must read exactly the albedo in every pixel; tests env sampling and diffuse BSDF normalisation.',
  camera: closeCamera,
  integrator: { maxDepth: 8 },
  materials: { m: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] } },
  meshes: [sphere('m')],
  // Tolerance is relative, applied to the image mean and to each pixel's expected-vs-converged check by bench/metrics.
  expected: { type: 'uniform', value: [0.5, 0.5, 0.5], tolerance: 0.01 },
};

/** Smooth, absorption-free dielectric conserves energy: every path ends in the environment with weight 1. */
export const furnaceDielectric: SceneDesc = {
  ...furnaceBase,
  name: 'furnace-dielectric',
  purpose: 'White furnace: smooth dielectric flat-shaded sphere (ior 1.5, no absorption) under uniform L=1 must read 1 everywhere, including the background; tests Fresnel/refraction energy conservation and deep path handling (maxDepth 1000: facet-to-facet total internal reflection chains lose 0.1% of the energy at maxDepth 100 in both renderers).',
  camera: farCamera,
  integrator: { maxDepth: 1000 },
  materials: { m: { type: 'dielectric', ior: 1.5, roughness: 0 } },
  meshes: [sphere('m')],
  // Per-pixel tolerance 2%: silhouette pixels of the faceted sphere have heavy-tailed noise (internal reflection chains).
  expected: { type: 'uniform', value: [1, 1, 1], tolerance: 0.02 },
};

/**
 * Near-unit Fresnel (k huge) rough conductor. A single-scattering GGX microfacet model loses the energy of paths that
 * scatter between microfacets, so the image is darker than 1 and the loss grows with roughness. There is no analytic
 * value (it depends on the renderer's multiple-scattering treatment); we record pbrt-v4/Mitsuba as the reference and
 * compare against them. A renderer with multiple-scattering compensation will read closer to 1.
 */
export const furnaceConductor: SceneDesc = {
  ...furnaceBase,
  name: 'furnace-conductor',
  purpose: 'White furnace: rough GGX conductor flat-shaded sphere (roughness 0.7, alpha 0.49, Fresnel ~ 1) under uniform L=1; no analytic value, shows the missing multiple-scattering energy that single-scatter microfacet models lose (compare to pbrt/Mitsuba).',
  camera: closeCamera,
  integrator: { maxDepth: 16 },
  materials: { m: { type: 'conductor', eta: [1, 1, 1], k: [1000, 1000, 1000], roughness: 0.7 } },
  meshes: [sphere('m')],
};
