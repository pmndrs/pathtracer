import type { SceneDesc } from '../schema.ts';
import { translation } from '../geometry.ts';
import { ENV_FILE } from './env-sky.ts';

/**
 * Diffuse sphere and rough copper sphere on a ground plane under the procedural sky (gradient + sun hot spot; see
 * env-sky.ts). Needs `node bench/scenes/tier0/generate-assets.ts` first (the PFM is not hand-authored).
 */
export const envSphere: SceneDesc = {
  version: 1,
  name: 'env-sphere',
  purpose: 'Diffuse and rough-copper spheres on a plane under a procedural HDR sky with a sun-like hot spot: tests environment importance sampling, MIS against BSDF sampling, and equirect orientation; no analytic value (converged pbrt/Mitsuba agreement is the reference).',
  tier: 0,
  camera: { type: 'perspective', position: [0, 1.6, 5.2], target: [0, 0.55, 0], up: [0, 1, 0], yFovDeg: 35 },
  film: { width: 384, height: 256, filter: 'box' },
  integrator: { maxDepth: 16 },
  materials: {
    ground: { type: 'diffuse', reflectance: [0.5, 0.5, 0.5] },
    clay: { type: 'diffuse', reflectance: [0.8, 0.35, 0.3] },
    copper: { type: 'conductor', eta: [0.2, 0.924, 1.102], k: [3.912, 2.452, 2.142], roughness: 0.3 },
  },
  meshes: [
    { name: 'ground', geometry: { type: 'quad', size: [40, 40] }, material: 'ground' },
    { name: 'clay', geometry: { type: 'sphere', radius: 0.7, segments: 96, rings: 48 }, material: 'clay', transform: translation(-1.0, 0.7, 0) },
    { name: 'copper', geometry: { type: 'sphere', radius: 0.7, segments: 96, rings: 48 }, material: 'copper', transform: translation(1.0, 0.7, 0) },
  ],
  environment: { type: 'hdri', path: ENV_FILE, scale: 1, rotationYDeg: 0 },
};
