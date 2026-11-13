import type { SceneDesc } from '../schema.ts';
import { compose, rotationX, translation } from '../geometry.ts';

/** Geometry shared with the analytic check in the unit tests. */
export const QUAD_LIGHT = { size: 1, height: 1, radiance: 10, albedo: 0.5, planeSize: 8 } as const;

/**
 * One-sided 1 m x 1 m quad light 1 m above a diffuse plane (albedo 0.5), direct lighting only (maxDepth 1). The plane
 * radiance below the light centre is analytic: L = albedo/pi * E, with E = Lemit * (form factor of the quad) * pi
 * (see quad-light.test.ts), so the image checks next-event estimation of an area light against a closed form.
 */
export const quadLight: SceneDesc = {
  version: 1,
  name: 'quad-light',
  purpose: 'Single one-sided quad area light over a diffuse plane, direct lighting only: tests area-light sampling (solid-angle/area pdf, one-sidedness) against a closed-form irradiance at the plane point under the light.',
  tier: 0,
  camera: { type: 'perspective', position: [0, 2.5, 3.5], target: [0, 0, 0.3], up: [0, 1, 0], yFovDeg: 45 },
  film: { width: 256, height: 256, filter: 'box' },
  integrator: { maxDepth: 1 },
  materials: {
    floor: { type: 'diffuse', reflectance: [QUAD_LIGHT.albedo, QUAD_LIGHT.albedo, QUAD_LIGHT.albedo] },
    lightBack: { type: 'black' },
  },
  meshes: [
    { name: 'plane', geometry: { type: 'quad', size: [QUAD_LIGHT.planeSize, QUAD_LIGHT.planeSize] }, material: 'floor' },
    {
      name: 'light',
      geometry: { type: 'quad', size: [QUAD_LIGHT.size, QUAD_LIGHT.size] },
      material: 'lightBack',
      // rotate 180 degrees about X so the local +Y normal faces down toward the plane
      transform: compose(translation(0, QUAD_LIGHT.height, 0), rotationX(Math.PI)),
      emission: [QUAD_LIGHT.radiance, QUAD_LIGHT.radiance, QUAD_LIGHT.radiance],
    },
  ],
};
