import { compose, rotationX, translation } from '../geometry.ts';
import type { SceneDesc } from '../schema.ts';

/**
 * Analytic checks of the `openpbr` path in our megakernel (P2.21, P2.22, P2.30): scenes with a closed-form expectation, no pbrt/Mitsuba
 * reference needed. They need `pt-megakernel` (`openpbr`); registered with the experiment scenes. All four look through a very narrow
 * field of view (1 degree at 5 m, so |cos| > 0.9999 and Fresnel is its normal-incidence value) down -Z.
 *
 *  - `thin-sheet`: a thin-walled sheet (ior 1.5) in front of a unit-ish emitter, black sky. Pixel = T' Lb with the sheet's total transmittance
 *    T' = (1 - R) / (1 + R) = 0.9230769 at normal incidence (R = 0.04; the inner reflections summed, pbrt `ThinDielectricBxDF`), so L = (0.923077, 0.461538, 0.230769).
 *  - `thin-furnace`: the same sheet under a uniform unit sky and no emitter: R' + T' = 1, every pixel is exactly 1.
 *  - `beer-slab`: a closed glass slab (0.5 m thick, ior 1.5, `transmission_color` (0.8, 0.5, 0.2) at `transmission_depth` 0.25 m, so exp(-sigma d) = color^2)
 *    in front of the emitter: pixel = T^2 x / (1 - R^2 x^2) Lb per channel with x = exp(-sigma d), T = 1 - R, R = 0.04.
 *  - `layered-furnace`: a white dielectric-over-diffuse sphere (ior 1.5, roughness 0.1) under a unit sky, maxDepth 200: the layered BSDF
 *    conserves energy to 0.1% (albedo 1 by construction, stochastic 0.999 at this roughness), so the mean is 1 within 0.01.
 */
const NARROW = { type: 'perspective' as const, position: [0, 0, 5] as const, target: [0, 0, 0] as const, up: [0, 1, 0] as const, yFovDeg: 1 };
const FILM = { width: 64, height: 64, filter: 'box' as const };
const facingCamera = (z: number) => compose(translation(0, 0, z), rotationX(Math.PI / 2));
const LB = [1, 0.5, 0.25] as const;
const R = ((1.5 - 1) / (1.5 + 1)) ** 2;
const sheet = { type: 'openpbr' as const, baseColor: [1, 1, 1] as const, transmissionWeight: 1, geometryThinWalled: true, specularRoughness: 0, specularIor: 1.5 };

export const thinSheet: SceneDesc = {
  version: 1,
  name: 'openpbr-thin-sheet',
  purpose: 'Thin-walled smooth dielectric sheet over an emitter at normal incidence: pixel = (1 - R) / (1 + R) of the emitter (P2.22).',
  tier: 1,
  camera: NARROW,
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: { sheet, back: { type: 'black' } },
  meshes: [
    { name: 'sheet', geometry: { type: 'quad', size: [4, 4] }, material: 'sheet', transform: facingCamera(0) },
    { name: 'emitter', geometry: { type: 'quad', size: [8, 8] }, material: 'back', transform: facingCamera(-1), emission: LB },
  ],
  expected: { type: 'mean', value: [LB[0] * ((1 - R) / (1 + R)), LB[1] * ((1 - R) / (1 + R)), LB[2] * ((1 - R) / (1 + R))], tolerance: 0.002 },
};

export const thinFurnace: SceneDesc = {
  version: 1,
  name: 'openpbr-thin-furnace',
  purpose: 'Thin-walled sheet under a unit sky: R + T = 1, every pixel is 1 (P2.22).',
  tier: 1,
  camera: NARROW,
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: { sheet },
  meshes: [{ name: 'sheet', geometry: { type: 'quad', size: [4, 4] }, material: 'sheet', transform: facingCamera(0) }],
  environment: { type: 'uniform', radiance: [1, 1, 1] },
  expected: { type: 'uniform', value: [1, 1, 1], tolerance: 1e-4 },
};

const TRANSMISSION_COLOR = [0.8, 0.5, 0.2] as const;
const DEPTH = 0.25;
const THICKNESS = 0.5;
export const beerSlab: SceneDesc = {
  version: 1,
  name: 'openpbr-beer-slab',
  purpose: 'Closed glass slab with an absorbing interior over an emitter at normal incidence: pixel = T^2 x / (1 - R^2 x^2) per channel, x = exp(-sigma d) (P2.22).',
  tier: 1,
  camera: NARROW,
  film: FILM,
  integrator: { maxDepth: 64 },
  materials: { glass: { type: 'openpbr', baseColor: [1, 1, 1], transmissionWeight: 1, specularRoughness: 0, specularIor: 1.5, transmissionColor: TRANSMISSION_COLOR, transmissionDepth: DEPTH }, back: { type: 'black' } },
  meshes: [
    { name: 'slab', geometry: { type: 'box', size: [4, 4, THICKNESS] }, material: 'glass' },
    { name: 'emitter', geometry: { type: 'quad', size: [8, 8] }, material: 'back', transform: facingCamera(-1), emission: LB },
  ],
  expected: {
    type: 'mean',
    value: TRANSMISSION_COLOR.map((c, i) => {
      const x = c ** (THICKNESS / DEPTH);
      return (LB[i] as number) * (((1 - R) ** 2 * x) / (1 - R * R * x * x));
    }) as unknown as [number, number, number],
    tolerance: 0.003,
  },
};

export const layeredFurnace: SceneDesc = {
  version: 1,
  name: 'openpbr-layered-furnace',
  purpose: 'White dielectric-over-diffuse sphere (ior 1.5, roughness 0.1) under a unit sky: the layered BSDF conserves energy to 0.1%, so the mean is 1 (P2.21).',
  tier: 1,
  camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
  film: { width: 64, height: 64, filter: 'box' },
  integrator: { maxDepth: 200 },
  materials: { plastic: { type: 'openpbr', baseColor: [1, 1, 1], specularRoughness: 0.1, specularIor: 1.5 } },
  meshes: [{ name: 'ball', geometry: { type: 'sphere', radius: 1, segments: 64, rings: 32 }, material: 'plastic' }],
  environment: { type: 'uniform', radiance: [1, 1, 1] },
  expected: { type: 'mean', value: [1, 1, 1], tolerance: 0.01 },
};

export const openpbrCheckScenes: readonly SceneDesc[] = [thinSheet, thinFurnace, beerSlab, layeredFurnace];
