import type { SceneDesc } from '../schema.ts';
import { compose, rotationX, translation } from '../geometry.ts';

/**
 * A/B scenes for light selection (P1.4f): a diffuse plane under two one-sided quad lights (a small bright one and a larger
 * dim one), a sun and a uniform sky. Every kind of light is present and their power-proportional shares are far from their
 * contribution, so the strategies differ. No pbrt/Mitsuba references: judge against our own high-spp render
 * (`bench/adapters/light-selection-ab.ts`). `lights-mix`: sun and sky are the key light (power over the bounding sphere makes
 * them take ~99% of the picks, so the lamps are under-sampled). `lights-mix-lamps`: the lamps are the key light and the
 * sun and sky are weak.
 */
function mix(name: string, purpose: string, p: { lampA: number; lampB: number; sun: number; sky: number }): SceneDesc {
  const down = (x: number, h: number) => compose(translation(x, h, 0), rotationX(Math.PI));
  return {
    version: 1,
    name,
    purpose,
    tier: 1,
    camera: { type: 'perspective', position: [0, 2.5, 3.5], target: [0, 0, 0.3], up: [0, 1, 0], yFovDeg: 45 },
    film: { width: 256, height: 256, filter: 'box' },
    integrator: { maxDepth: 3 },
    materials: {
      floor: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] },
      lightBack: { type: 'black' },
    },
    meshes: [
      { name: 'plane', geometry: { type: 'quad', size: [8, 8] }, material: 'floor' },
      { name: 'lampA', geometry: { type: 'quad', size: [0.5, 0.5] }, material: 'lightBack', transform: down(-1.5, 0.8), emission: [p.lampA, p.lampA, p.lampA] },
      { name: 'lampB', geometry: { type: 'quad', size: [1, 1] }, material: 'lightBack', transform: down(1.5, 1.5), emission: [p.lampB, p.lampB, p.lampB] },
    ],
    lights: [{ type: 'directional', direction: [0.5, -0.7, -0.3], irradiance: [p.sun, p.sun * 0.95, p.sun * 0.85], angularRadius: 0.00465 }],
    environment: { type: 'uniform', radiance: [p.sky, p.sky * 1.1, p.sky * 1.3] },
  };
}

export const lightsMix = mix('lights-mix', 'Plane under two quad lamps, a sun and a uniform sky; sun and sky are the key light. A/B for light selection weights (P1.4f).', { lampA: 40, lampB: 5, sun: 20, sky: 0.5 });
export const lightsMixLamps = mix('lights-mix-lamps', 'Plane under two quad lamps, a sun and a uniform sky; the lamps are the key light. A/B for light selection weights (P1.4f).', { lampA: 400, lampB: 60, sun: 2, sky: 0.15 });

export const abScenes: readonly SceneDesc[] = [lightsMix, lightsMixLamps];
