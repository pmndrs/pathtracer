import { compose, rotationX, translation } from '../geometry.ts';
import type { Rgb, SceneDesc } from '../schema.ts';

/**
 * Colour round trip of the spectral mode (P5): 24 flat colours, one per cell of a 6 x 4 grid seen head on from far away (nearly orthographic), so the image of a cell is
 * its colour and `bench/adapters/spectral-swatches.ts` can read it back per cell.
 *
 *  - `spectral-swatches-refl`: a diffuse (reflectance) swatch per cell under a uniform white sky, one bounce: pixel = the swatch RGB in the mean (RGB mode exactly;
 *    spectral mode through the sigmoid uplift, the D65 white light and the CMF estimator: within the table's error, `docs/design/spectral.md`).
 *  - `spectral-swatches-emit`: an emitter per cell (radiance = 4 x the swatch RGB, unbounded: RGBIlluminantSpectrum) on a black sky: pixel = the emitter RGB.
 *  - `spectral-swatches-env`: no geometry, a uniform coloured sky: every pixel = the sky RGB (the environment's uplift).
 */
export const SWATCH_COLORS: readonly Rgb[] = [
  [0.17, 0.08, 0.05], [0.58, 0.32, 0.25], [0.09, 0.19, 0.34], [0.08, 0.15, 0.05], [0.26, 0.22, 0.5], [0.12, 0.52, 0.4],
  [0.7, 0.26, 0.03], [0.05, 0.09, 0.4], [0.55, 0.07, 0.1], [0.1, 0.03, 0.12], [0.35, 0.5, 0.04], [0.75, 0.4, 0.02],
  [0.02, 0.03, 0.25], [0.04, 0.3, 0.05], [0.45, 0.03, 0.03], [0.85, 0.7, 0.0], [0.5, 0.07, 0.3], [0.0, 0.25, 0.5],
  [0.9, 0.9, 0.9], [0.5, 0.5, 0.5], [0.18, 0.18, 0.18], [1.0, 0.78, 0.34], [0.02, 0.02, 0.9], [0.9, 0.02, 0.02],
];
export const SWATCH_GRID = { cols: 6, rows: 4, cell: 64 } as const;
export const SWATCH_EMIT_SCALE = 4;

const CAMERA = { type: 'perspective' as const, position: [0, 0, 40] as const, target: [0, 0, 0] as const, up: [0, 1, 0] as const, yFovDeg: 2 * (Math.atan(2.1 / 40) * 180) / Math.PI };
const FILM = { width: SWATCH_GRID.cols * SWATCH_GRID.cell, height: SWATCH_GRID.rows * SWATCH_GRID.cell, filter: 'box' as const };
const face = (x: number, y: number) => compose(translation(x, y, 0), rotationX(Math.PI / 2));
const center = (i: number): [number, number] => [((i % SWATCH_GRID.cols) - (SWATCH_GRID.cols - 1) / 2), ((SWATCH_GRID.rows - 1) / 2 - Math.floor(i / SWATCH_GRID.cols))];

const refl: SceneDesc = {
  version: 1,
  name: 'spectral-swatches-refl',
  purpose: '24 diffuse swatches under a uniform white sky, one bounce: the pixel of a cell is its reflectance RGB (round trip of the spectral uplift), P5.',
  tier: 1,
  camera: CAMERA,
  film: FILM,
  integrator: { maxDepth: 1 },
  materials: Object.fromEntries(SWATCH_COLORS.map((c, i) => [`s${i}`, { type: 'diffuse' as const, reflectance: c }])),
  meshes: SWATCH_COLORS.map((_, i) => ({ name: `swatch${i}`, geometry: { type: 'quad' as const, size: [0.9, 0.9] as const }, material: `s${i}`, transform: face(...center(i)) })),
  environment: { type: 'uniform', radiance: [1, 1, 1] },
};

const emit: SceneDesc = {
  version: 1,
  name: 'spectral-swatches-emit',
  purpose: '24 emitters (radiance 4 x swatch RGB) seen directly on a black sky: the pixel of a cell is the emitter RGB (round trip of the illuminant uplift), P5.',
  tier: 1,
  camera: CAMERA,
  film: FILM,
  integrator: { maxDepth: 1 },
  materials: { back: { type: 'black' } },
  meshes: SWATCH_COLORS.map((c, i) => ({ name: `emitter${i}`, geometry: { type: 'quad' as const, size: [0.9, 0.9] as const }, material: 'back', transform: face(...center(i)), emission: c.map((v) => v * SWATCH_EMIT_SCALE) as unknown as Rgb })),
};

const env: SceneDesc = {
  version: 1,
  name: 'spectral-swatches-env',
  purpose: 'A uniform sky of a saturated colour seen directly: every pixel is the sky RGB (round trip of the environment uplift), P5.',
  tier: 1,
  camera: CAMERA,
  film: { width: 64, height: 64, filter: 'box' },
  integrator: { maxDepth: 1 },
  materials: {},
  meshes: [],
  environment: { type: 'uniform', radiance: [2.4, 0.8, 0.05] },
  expected: { type: 'uniform', value: [2.4, 0.8, 0.05], tolerance: 0.03 },
};

export const spectralSwatchScenes: readonly SceneDesc[] = [refl, emit, env];
