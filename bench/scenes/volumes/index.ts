import { gridDensity, gridWorldToIndex, slabSingleScatter } from '@pmndrs/pathtracer';
import { compose, rotationX, scaling, translation } from '../geometry.ts';
import type { MediumDesc, MeshDesc, SceneDesc } from '../schema.ts';
import { cornell } from '../tier0/cornell.ts';
import { cloudScenes } from './clouds.ts';

/**
 * Participating-media scenes (P5, docs/internal/design/volumes.md).
 *
 * Analytic checks (`volumeCheckScenes`, no reference renderer needed, closed forms; look through a 1 degree field of view so a pixel is one ray direction):
 *  - `vol-absorb-slab`: a 1 m slab of absorbing medium (interface boundary) under a unit sky, camera along the normal: pixel = exp(-sigma_a d) per channel (delta tracking
 *    with every collision a null or an absorption: only the closed form survives).
 *  - `vol-camera-medium`: the camera inside an absorbing medium that fills space, looking at an emitter 3 m away: pixel = L exp(-sigma_a 3) (`cameraMedium`).
 *  - `vol-furnace-sphere`, `vol-furnace-sphere-back`: a sphere of non-absorbing medium (albedo 1, forward g = 0.6 / backward g = -0.5) under a unit sky, max depth 400: energy
 *    conservation, every pixel is 1 (NEE through the medium, phase sampling, MIS, ratio of weights).
 *  - `vol-slab-single`, `vol-slab-single-oblique`: a thin scattering slab (tau 0.4, albedo 0.8, g 0.5; normal and 53 degrees) under a unit sky, max depth 1 = unscattered light plus
 *    exactly one scattering event: pixel = `slabSingleScatter` (a quadrature of the single-scattering integral over depth and incident direction).
 *  - `vol-grid-slab`: a grid medium whose density varies only along the slab normal (16 layers): pixel = exp(-sigma_t integral density dz), the line integral of the
 *    same trilinear lookup (delta tracking and ratio tracking against the closed form).
 * Reference scenes (`volumeReferenceScenes`, compared with pbrt-v4 and Mitsuba 3 at 4096 spp): `vol-cornell-smoke` (a homogeneous smoke cube in the Cornell box), `vol-milk-sphere`
 * (a scattering medium inside a smooth glass sphere, sun + sky), `vol-grid-cloud` (a procedural density grid under a sun, shadow on a floor).
 */
const NARROW = (position: readonly [number, number, number], target: readonly [number, number, number] = [0, 0, 0]) => ({ type: 'perspective' as const, position, target, up: [0, 1, 0] as const, yFovDeg: 1 });
const FILM = { width: 32, height: 32, filter: 'box' as const };
const SKY = { type: 'uniform' as const, radiance: [1, 1, 1] as const };
const facingCamera = (z: number) => compose(translation(0, 0, z), rotationX(Math.PI / 2));
const LB = [1, 0.5, 0.25] as const;

const slabMesh = (thickness: number, medium: string, size = 40): MeshDesc => ({ name: 'slab', geometry: { type: 'box', size: [size, size, thickness] }, material: 'boundary', medium: { interior: medium } });

const SIGMA_A = [0.3, 0.8, 1.6] as const;
export const volAbsorbSlab: SceneDesc = {
  version: 1,
  name: 'vol-absorb-slab',
  purpose: 'Absorbing homogeneous slab (1 m, sigma_a 0.3/0.8/1.6) under a unit sky, normal incidence: pixel = exp(-sigma_a d) per channel (P5).',
  tier: 1,
  camera: NARROW([0, 0, 5]),
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: { boundary: { type: 'interface' } },
  media: { gas: { type: 'homogeneous', sigmaA: SIGMA_A, sigmaS: [0, 0, 0] } },
  meshes: [slabMesh(1, 'gas')],
  environment: SKY,
  expected: { type: 'mean', value: SIGMA_A.map((s) => Math.exp(-s)) as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
};

const CAM_SIGMA = [0.1, 0.3, 0.5] as const;
export const volCameraMedium: SceneDesc = {
  version: 1,
  name: 'vol-camera-medium',
  purpose: 'Camera inside an absorbing medium that fills space, an emitter 3 m away: pixel = L exp(-sigma_a 3) (P5, cameraMedium).',
  tier: 1,
  camera: NARROW([0, 0, 0], [0, 0, -1]),
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: { back: { type: 'black' } },
  media: { fog: { type: 'homogeneous', sigmaA: CAM_SIGMA, sigmaS: [0, 0, 0] } },
  cameraMedium: 'fog',
  meshes: [{ name: 'emitter', geometry: { type: 'quad', size: [8, 8] }, material: 'back', transform: facingCamera(-3), emission: LB }],
  expected: { type: 'mean', value: LB.map((l, i) => l * Math.exp(-3 * CAM_SIGMA[i]!)) as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
};

const furnace = (name: string, g: number): SceneDesc => ({
  version: 1,
  name,
  purpose: `Sphere of non-absorbing scattering medium (sigma_s 1.5, g ${g}) under a unit sky, interface boundary: energy conservation, every pixel is 1 (P5).`,
  tier: 1,
  camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
  film: { width: 48, height: 48, filter: 'box' },
  integrator: { maxDepth: 400 },
  materials: { boundary: { type: 'interface' } },
  media: { cloud: { type: 'homogeneous', sigmaA: [0, 0, 0], sigmaS: [1.5, 1.5, 1.5], g } },
  meshes: [{ name: 'ball', geometry: { type: 'sphere', radius: 1, segments: 64, rings: 32 }, material: 'boundary', medium: { interior: 'cloud' } }],
  environment: SKY,
  expected: { type: 'mean', value: [1, 1, 1], tolerance: 0.01, pixelwise: false },
});
export const volFurnaceSphere = furnace('vol-furnace-sphere', 0.6);
export const volFurnaceSphereBack = furnace('vol-furnace-sphere-back', -0.5);

const SLAB = { sigmaT: 0.4, albedo: 0.8, g: 0.5, thickness: 1 } as const;
const slabSingle = (name: string, cosine: number): SceneDesc => {
  const sin = Math.sqrt(1 - cosine * cosine);
  const expected = slabSingleScatter(SLAB.sigmaT, SLAB.albedo, SLAB.g, SLAB.thickness, cosine, 1);
  return {
    version: 1,
    name,
    purpose: `Thin scattering slab (tau ${SLAB.sigmaT * SLAB.thickness}, albedo ${SLAB.albedo}, g ${SLAB.g}) under a unit sky seen at cos ${cosine}, max depth 1: unscattered + single scattering, pixel = ${expected.toFixed(6)} (quadrature, P5).`,
    tier: 1,
    camera: NARROW([0, 5 * sin, 5 * cosine]),
    film: FILM,
    integrator: { maxDepth: 1 },
    materials: { boundary: { type: 'interface' } },
    media: { haze: { type: 'homogeneous', sigmaA: [SLAB.sigmaT * (1 - SLAB.albedo), SLAB.sigmaT * (1 - SLAB.albedo), SLAB.sigmaT * (1 - SLAB.albedo)], sigmaS: [SLAB.sigmaT * SLAB.albedo, SLAB.sigmaT * SLAB.albedo, SLAB.sigmaT * SLAB.albedo], g: SLAB.g } },
    meshes: [slabMesh(SLAB.thickness, 'haze')],
    environment: SKY,
    expected: { type: 'mean', value: [expected, expected, expected], tolerance: 0.002, pixelwise: false },
  };
};
export const volSlabSingle = slabSingle('vol-slab-single', 1);
export const volSlabSingleOblique = slabSingle('vol-slab-single-oblique', 0.6);

// --- a density that varies along z only: 16 layers over a 1 m slab -----------------------------------------------------------------------------------------------
const LAYERS = 16;
const layerDensity = (k: number): number => 0.35 + 0.65 * Math.abs(Math.sin(0.7 + k * 0.55));
function slabGrid(): { resolution: [number, number, number]; density: number[]; bounds: [[number, number, number], [number, number, number]] } {
  const nx = 4, ny = 4;
  const density: number[] = [];
  for (let k = 0; k < LAYERS; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) density.push(layerDensity(k));
  return { resolution: [nx, ny, LAYERS], density, bounds: [[-4, -4, -0.5], [4, 4, 0.5]] };
}
const GRID_SIGMA = [0.9, 0.9, 0.9] as const;
function gridSlabExpected(): number[] {
  const g = slabGrid();
  const grid = { resolution: g.resolution, worldToIndex: gridWorldToIndex(g.resolution, g.bounds), densityMax: Math.max(...g.density) };
  const dens = gridDensity(grid, g.density);
  let tau = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) tau += dens(0, 0, -0.5 + ((i + 0.5) / n)) / n;
  return GRID_SIGMA.map((s) => Math.exp(-s * tau));
}
export const volGridSlab: SceneDesc = (() => {
  const g = slabGrid();
  return {
    version: 1,
    name: 'vol-grid-slab',
    purpose: 'Grid medium whose density varies in 16 layers along the normal of a 1 m slab, absorption only, unit sky, normal incidence: pixel = exp(-sigma integral of the trilinear density) (P5).',
    tier: 1,
    camera: NARROW([0, 0, 5]),
    film: FILM,
    integrator: { maxDepth: 8 },
    materials: { boundary: { type: 'interface' } },
    media: { layers: { type: 'grid', resolution: g.resolution, density: g.density, bounds: g.bounds, sigmaA: GRID_SIGMA, sigmaS: [0, 0, 0] } },
    meshes: [{ name: 'slab', geometry: { type: 'box', size: [8, 8, 1] }, material: 'boundary', medium: { interior: 'layers' } }],
    environment: SKY,
    expected: { type: 'mean', value: gridSlabExpected() as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
  };
})();

/**
 * V2.4: `vol-grid-slab` with INTERIOR bricks (a 32^3 grid over the same 8 x 8 x 1 m box, density smooth along z: 0.6 + 0.3 sin(0.7 + 0.2 k)), so the brick minorant of the cells the
 * camera ray crosses is well above 0 (min / max about 0.6-0.9) and decomposition tracking takes most collisions as control ones; in the 4 x 4 x 16 `vol-grid-slab` every brick
 * touches the grid border, its minorant is 0 and decomposition equals delta tracking. Absorption only, unit sky: pixel = exp(-sigma integral of the trilinear density).
 */
const INNER_N = 32;
const innerDensity = (k: number): number => 0.6 + 0.3 * Math.sin(0.7 + 0.2 * k);
function innerSlabGrid(): { resolution: [number, number, number]; density: number[]; bounds: [[number, number, number], [number, number, number]] } {
  const density: number[] = [];
  for (let k = 0; k < INNER_N; k++) for (let j = 0; j < INNER_N; j++) for (let i = 0; i < INNER_N; i++) density.push(innerDensity(k));
  return { resolution: [INNER_N, INNER_N, INNER_N], density, bounds: [[-4, -4, -0.5], [4, 4, 0.5]] };
}
export const volGridSlabInner: SceneDesc = (() => {
  const g = innerSlabGrid();
  const grid = { resolution: g.resolution, worldToIndex: gridWorldToIndex(g.resolution, g.bounds), densityMax: Math.max(...g.density) };
  const dens = gridDensity(grid, g.density);
  let tau = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) tau += dens(0, 0, -0.5 + ((i + 0.5) / n)) / n;
  return {
    ...volGridSlab,
    name: 'vol-grid-slab-inner',
    purpose: 'V2.4: vol-grid-slab on a 32^3 grid with a smooth z profile, so the camera ray crosses interior bricks (minorant > 0): exercises decomposition tracking. Pixel = exp(-sigma integral of the trilinear density).',
    media: { layers: { type: 'grid', resolution: g.resolution, density: g.density, bounds: g.bounds, sigmaA: GRID_SIGMA, sigmaS: [0, 0, 0] } },
    expected: { type: 'mean', value: GRID_SIGMA.map((s) => Math.exp(-s * tau)) as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
  };
})();

// --- the camera inside a grid medium (V3.3) ---------------------------------------------------------------------------------------------------------------------
/** 32 layers over z in [-4, 4] (an 8 m box, the camera at its centre), an emitter 3 m ahead: pixel = L exp(-sigma integral_0^3 density(0, 0, -z) dz). */
const CAM_LAYERS = 32;
const camGridDensity = (k: number): number => 0.2 + 0.5 * Math.abs(Math.sin(0.4 + k * 0.45));
function camGrid(): { resolution: [number, number, number]; density: number[]; bounds: [[number, number, number], [number, number, number]] } {
  const density: number[] = [];
  for (let k = 0; k < CAM_LAYERS; k++) for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) density.push(camGridDensity(k));
  return { resolution: [4, 4, CAM_LAYERS], density, bounds: [[-4, -4, -4], [4, 4, 4]] };
}
function camGridExpected(): number[] {
  const g = camGrid();
  const grid = { resolution: g.resolution, worldToIndex: gridWorldToIndex(g.resolution, g.bounds), densityMax: Math.max(...g.density) };
  const dens = gridDensity(grid, g.density);
  let tau = 0;
  const n = 20000;
  for (let i = 0; i < n; i++) tau += (3 * dens(0, 0, -3 * ((i + 0.5) / n))) / n;
  return CAM_SIGMA.map((s) => LB[CAM_SIGMA.indexOf(s)]! * Math.exp(-s * tau));
}
const camGridScene = (name: string, explicit: boolean): SceneDesc => {
  const g = camGrid();
  return {
    version: 1,
    name,
    purpose: `Camera inside an absorbing GRID medium (density in 32 layers along the view axis, 8 m box), an emitter 3 m ahead: pixel = L exp(-sigma integral of the trilinear density)${explicit ? ' (explicit cameraMedium)' : ' (cameraMedium detected from the boundary mesh)'} (V3.3).`,
    tier: 1,
    camera: NARROW([0, 0, 0], [0, 0, -1]),
    film: FILM,
    integrator: { maxDepth: 8 },
    materials: { boundary: { type: 'interface' }, back: { type: 'black' } },
    media: { layers: { type: 'grid', resolution: g.resolution, density: g.density, bounds: g.bounds, sigmaA: CAM_SIGMA, sigmaS: [0, 0, 0] } },
    ...(explicit ? { cameraMedium: 'layers' } : {}),
    meshes: [
      { name: 'box', geometry: { type: 'box', size: [8, 8, 8] }, material: 'boundary', medium: { interior: 'layers' } },
      { name: 'emitter', geometry: { type: 'quad', size: [8, 8] }, material: 'back', transform: facingCamera(-3), emission: LB },
    ],
    expected: { type: 'mean', value: camGridExpected() as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
  };
};
export const volCameraGrid: SceneDesc = camGridScene('vol-camera-grid', true);
export const volCameraGridAuto: SceneDesc = camGridScene('vol-camera-grid-auto', false);

// --- reference scenes ------------------------------------------------------------------------------------------------------------------------------------
/** Smoke cube in the Cornell box (0.22 m cube where the short block stood). sigma_t 10 / m (tau 2.2 across), albedo 0.9, g 0.3: grey, so pbrt's spectral uplift is exact. */
export const volCornellSmoke: SceneDesc = {
  ...cornell,
  name: 'vol-cornell-smoke',
  purpose: 'Cornell box with a homogeneous smoke cube (0.22 m, sigma_t 10/m, albedo 0.9, g 0.3) on the floor: NEE through a medium, multiple scattering, colour bleeding (P5).',
  tier: 2,
  film: { width: 128, height: 128, filter: 'box' },
  integrator: { maxDepth: 32 },
  materials: { ...cornell.materials, boundary: { type: 'interface' } },
  media: { smoke: { type: 'homogeneous', sigmaA: [1, 1, 1], sigmaS: [9, 9, 9], g: 0.3 } },
  meshes: [
    ...cornell.meshes.filter((m) => m.name !== 'short-block'),
    { name: 'smoke-cube', geometry: { type: 'box', size: [0.22, 0.22, 0.22] }, material: 'boundary', transform: translation(0.185, 0.11, 0.169), medium: { interior: 'smoke' } },
  ],
};

/**
 * The examples' "Fog and clouds" page in the bench Cornell box: fog fills the room up to 4 mm under the ceiling lamp (the example: 1 cm under it in a 2 m room), optical depth
 * across the room ~1.2 as there (sigma_t 2.2/m over 0.55 m), albedo 0.9, g 0.3, both blocks inside the fog. Medium scatter points next to a small bright lamp: the case
 * `vol-cornell-smoke` (a cube on the floor) never tests.
 */
export const volCornellFog: SceneDesc = {
  ...cornell,
  name: 'vol-cornell-fog',
  purpose: 'Cornell box filled with homogeneous fog up to 4 mm under the lamp (sigma_t 2.2/m, albedo 0.9, g 0.3), blocks inside the fog: scatter points next to a small bright emitter (the examples volumes page).',
  tier: 2,
  film: { width: 128, height: 128, filter: 'box' },
  integrator: { maxDepth: 32 },
  materials: { ...cornell.materials, boundary: { type: 'interface' } },
  media: { fog: { type: 'homogeneous', sigmaA: [0.22, 0.22, 0.22], sigmaS: [1.98, 1.98, 1.98], g: 0.3 } },
  meshes: [
    ...cornell.meshes.map((m) => (m.name === 'short-block' || m.name === 'tall-block' ? { ...m, medium: { interior: 'fog', exterior: 'fog' } } : m)),
    { name: 'fog-box', geometry: { type: 'box', size: [0.541, 0.5417, 0.551] }, material: 'boundary', transform: translation(0.2745, 0.27385, 0.2795), medium: { interior: 'fog' } },
  ],
};
/** `vol-cornell-fog` without the two blocks: no opaque surface inside the fog (pbrt and Mitsuba disagree by ~40% with the blocks in; this tells whose export is wrong). */
export const volCornellFogEmpty: SceneDesc = {
  ...volCornellFog,
  name: 'vol-cornell-fog-empty',
  purpose: `${volCornellFog.purpose} Blocks removed: no surface inside the fog.`,
  meshes: volCornellFog.meshes.filter((m) => m.name !== 'short-block' && m.name !== 'tall-block'),
};

export const volMilkSphere: SceneDesc = {
  version: 1,
  name: 'vol-milk-sphere',
  purpose: 'Smooth glass sphere (ior 1.33) full of a scattering medium (sigma_s 2.5, sigma_a 0.05, g 0.2) on a diffuse floor, sun + sky: medium state through refraction, shadow rays blocked by glass (P5).',
  tier: 2,
  camera: { type: 'perspective', position: [0, 0.35, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 35 },
  film: { width: 128, height: 128, filter: 'box' },
  integrator: { maxDepth: 40 },
  materials: { glass: { type: 'dielectric', ior: 1.33, roughness: 0 }, floor: { type: 'diffuse', reflectance: [0.6, 0.6, 0.6] } },
  media: { milk: { type: 'homogeneous', sigmaA: [0.05, 0.05, 0.05], sigmaS: [2.5, 2.5, 2.5], g: 0.2 } },
  meshes: [
    { name: 'floor', geometry: { type: 'quad', size: [8, 8] }, material: 'floor', transform: translation(0, -0.5, 0) },
    { name: 'ball', geometry: { type: 'sphere', radius: 0.5, segments: 64, rings: 32 }, material: 'glass', medium: { interior: 'milk' } },
  ],
  lights: [{ type: 'directional', direction: [-0.4, -1, -0.5], irradiance: [3, 3, 3] }],
  environment: { type: 'uniform', radiance: [0.4, 0.4, 0.4] },
};

/** A 48^3 density: a handful of Gaussian blobs, faded to exactly zero in a 2 voxel border (pbrt reads zero outside the grid, Mitsuba clamps: they agree on a zero border). */
export function cloudDensity(n = 48): Float32Array {
  const blobs: Array<[number, number, number, number, number]> = [
    [0.0, 0.0, 0.0, 0.28, 1.0],
    [0.22, 0.08, 0.05, 0.17, 0.9],
    [-0.2, 0.1, -0.08, 0.2, 0.8],
    [0.05, -0.15, 0.2, 0.15, 0.7],
    [-0.1, 0.22, 0.15, 0.13, 0.6],
    [0.12, -0.05, -0.25, 0.14, 0.75],
  ];
  const out = new Float32Array(n * n * n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5, z = (k + 0.5) / n - 0.5;
    let d = 0;
    for (const [bx, by, bz, r, a] of blobs) d += a * Math.exp(-((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) / (2 * r * r * 0.5));
    // a little high-frequency structure so the majorant is not the mean
    d *= 0.75 + 0.25 * Math.sin(23 * x + 7 * y) * Math.cos(19 * z - 5 * x);
    const edge = Math.min(i, j, k, n - 1 - i, n - 1 - j, n - 1 - k);
    out[i + n * (j + n * k)] = edge < 2 ? 0 : Math.min(1, Math.max(0, d));
  }
  return out;
}

const cloudGrid: MediumDesc = { type: 'grid', resolution: [48, 48, 48], density: cloudDensity(48), bounds: [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], sigmaA: [0.4, 0.4, 0.4], sigmaS: [8, 8, 8], g: 0.4 };
export const volGridCloud: SceneDesc = {
  version: 1,
  name: 'vol-grid-cloud',
  purpose: 'Heterogeneous density grid (48^3, peak sigma_t 8.4/m, albedo 0.95, g 0.4) in a 1 m box above a floor, sun + sky: delta tracking and ratio tracking on a grid, the cloud shadow on the floor (P5).',
  tier: 2,
  camera: { type: 'perspective', position: [0, 0.35, 3], target: [0, -0.05, 0], up: [0, 1, 0], yFovDeg: 35 },
  film: { width: 128, height: 128, filter: 'box' },
  integrator: { maxDepth: 40 },
  materials: { boundary: { type: 'interface' }, floor: { type: 'diffuse', reflectance: [0.6, 0.6, 0.6] } },
  media: { cloud: cloudGrid },
  meshes: [
    { name: 'floor', geometry: { type: 'quad', size: [8, 8] }, material: 'floor', transform: translation(0, -0.6, 0) },
    { name: 'cloud-box', geometry: { type: 'box', size: [1, 1, 1] }, material: 'boundary', medium: { interior: 'cloud' } },
  ],
  lights: [{ type: 'directional', direction: [-0.5, -1, -0.4], irradiance: [3, 3, 3] }],
  environment: { type: 'uniform', radiance: [0.4, 0.4, 0.4] },
};

/**
 * A real OpenVDB density: WebVDB's `smoke.vdb` (OpenVDB sample model, CC BY-SA 4.0: THIRD_PARTY), converted by `node bench/scenes/volumes/vdb-grid.ts` (2^3 box average, 2 zero cells of border, peak
 * normalised to 1) to `bench/assets/generated/volumes/vdb-smoke.f32` (git-ignored derivative). The plume is 109 m tall in the file; the mesh scales it to 1.6 m.
 */
const VDB_SMOKE = { resolution: [60, 115, 60] as const, half: [28.571428571428573, 54.76190476190476, 28.571428571428573] as const, height: 1.6 };
const vdbScale = VDB_SMOKE.height / (2 * VDB_SMOKE.half[1]);
export const volVdbSmoke: SceneDesc = {
  version: 1,
  name: 'vol-vdb-smoke',
  purpose: "WebVDB's smoke.vdb (a real fog volume, 60x115x60 after 2x downsampling) as a heterogeneous medium, sun + sky over a floor (P5; needs `node bench/scenes/volumes/vdb-grid.ts` first).",
  tier: 2,
  camera: { type: 'perspective', position: [0, 0.9, 2.6], target: [0, 0.8, 0], up: [0, 1, 0], yFovDeg: 35 },
  film: { width: 128, height: 128, filter: 'box' },
  integrator: { maxDepth: 40 },
  materials: { boundary: { type: 'interface' }, floor: { type: 'diffuse', reflectance: [0.6, 0.6, 0.6] } },
  media: { plume: { type: 'grid', resolution: VDB_SMOKE.resolution, densityFile: 'generated/volumes/vdb-smoke.f32', bounds: [[-VDB_SMOKE.half[0], -VDB_SMOKE.half[1], -VDB_SMOKE.half[2]], [VDB_SMOKE.half[0], VDB_SMOKE.half[1], VDB_SMOKE.half[2]]], sigmaA: [4, 4, 4], sigmaS: [36, 36, 36], g: 0.2 } },
  meshes: [
    { name: 'floor', geometry: { type: 'quad', size: [8, 8] }, material: 'floor', transform: translation(0, -0.01, 0) },
    { name: 'plume-box', geometry: { type: 'box', size: [2 * VDB_SMOKE.half[0], 2 * VDB_SMOKE.half[1], 2 * VDB_SMOKE.half[2]] }, material: 'boundary', transform: compose(translation(0, VDB_SMOKE.height / 2, 0), scaling(vdbScale)), medium: { interior: 'plume' } },
  ],
  lights: [{ type: 'directional', direction: [-0.5, -1, -0.4], irradiance: [3, 3, 3] }],
  environment: { type: 'uniform', radiance: [0.4, 0.4, 0.4] },
};

/**
 * Real OpenVDB clouds (V3.4b; test assets, CC BY-SA, derivatives are generated and git-ignored). `node bench/scenes/volumes/vdb-grid.ts <vdb> <name> <downsample>`:
 *  - `vdb-wdas-cloud`: WDAS cloud (quarter res), 3^3 box average -> 170x117x209, 2 zero border cells, peak 1; one file unit = 1 m here (the cloud is ~425 x 292 x 522 units).
 *  - `vdb-bunny-cloud`: OpenVDB bunny_cloud, 2^3 average -> 293x290x223, peak 1 (raw peak 2.6); ~47 x 46 x 37 units.
 * Cloud regime like `vol-cl-cloud256`: sun behind (travels -x, -y, +z, so the camera at +z looks at the lit rim) plus sky 0.4, g 0.85, albedo 0.99, maxDepth 512.
 */
function vdbCloud(name: string, purpose: string, file: string, resolution: readonly [number, number, number], lo: readonly [number, number, number], hi: readonly [number, number, number], sigmaT: number, camera: SceneDesc['camera']): SceneDesc {
  const half = [0, 1, 2].map((a) => (hi[a]! - lo[a]!) / 2) as [number, number, number];
  return {
    version: 1,
    name,
    purpose,
    tier: 2,
    camera,
    film: { width: 128, height: 128, filter: 'box' },
    integrator: { maxDepth: 512 },
    materials: { boundary: { type: 'interface' } },
    media: { cloud: { type: 'grid', resolution, densityFile: file, bounds: [[-half[0], -half[1], -half[2]], half], sigmaA: [0.01 * sigmaT, 0.01 * sigmaT, 0.01 * sigmaT], sigmaS: [0.99 * sigmaT, 0.99 * sigmaT, 0.99 * sigmaT], g: 0.85 } },
    meshes: [{ name: 'cloud-box', geometry: { type: 'box', size: [2 * half[0], 2 * half[1], 2 * half[2]] }, material: 'boundary', medium: { interior: 'cloud' } }],
    lights: [{ type: 'directional', direction: [-0.55, -0.45, 0.7], irradiance: [3, 3, 3] }],
    environment: { type: 'uniform', radiance: [0.4, 0.4, 0.4] },
  };
}
export const volVdbWdasCloud: SceneDesc = vdbCloud(
  'vol-vdb-wdas-cloud',
  "Walt Disney Animation Studios cloud (wdas_cloud_quarter.vdb, 3^3 averaged to 170x117x209), peak sigma_t 0.22/unit (max column tau 58, mean over covered columns 14), albedo 0.99, HG g 0.85, sun behind + sky 0.4, maxDepth 512: a real cloud for the equal-spp noise comparison (V3.4b; needs `node bench/scenes/volumes/vdb-grid.ts <vdb> vdb-wdas-cloud 3`).",
  'generated/volumes/vdb-wdas-cloud.f32', [170, 117, 209], [-222.91666135191917, -72.91666492819786, -303.74999275803566], [202.0833285152912, 219.5833280980587, 218.74999478459358], 0.22,
  { type: 'perspective', position: [0, 60, 1250], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 24 },
);
export const volVdbBunnyCloud: SceneDesc = vdbCloud(
  'vol-vdb-bunny-cloud',
  "OpenVDB bunny_cloud.vdb (2^3 averaged to 293x290x223), peak sigma_t 6/unit (max column tau 49, mean over covered columns 5.5), albedo 0.99, HG g 0.85, sun behind + sky 0.4, maxDepth 512 (V3.4b; needs `node bench/scenes/volumes/vdb-grid.ts <vdb> vdb-bunny-cloud 2`).",
  'generated/volumes/vdb-bunny-cloud.f32', [293, 290, 223], [-24.36, -4.12, -17], [22.52, 42.28, 18.68], 6,
  { type: 'perspective', position: [0, 4, 120], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 24 },
);

// --- V4.6 emissive media: a black background, a purely absorbing emissive slab seen along its normal -------------------------------------------------------------
const BLACK = { type: 'uniform' as const, radiance: [0, 0, 0] as const };
const EMIT_LE = [2, 1, 0.5] as const;
const EMIT_SIGMA = [0.4, 0.9, 1.8] as const;
export const volEmitSlab: SceneDesc = {
  version: 1,
  name: 'vol-emit-slab',
  purpose: 'Emissive purely absorbing slab (1 m, sigma_a 0.4/0.9/1.8, Le 2/1/0.5) on a black background, normal incidence: pixel = Le (1 - exp(-sigma_a d)) per channel (V4.6).',
  tier: 1,
  camera: NARROW([0, 0, 5]),
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: { boundary: { type: 'interface' } },
  media: { glow: { type: 'homogeneous', sigmaA: EMIT_SIGMA, sigmaS: [0, 0, 0], Le: EMIT_LE } },
  meshes: [slabMesh(1, 'glow')],
  environment: BLACK,
  expected: { type: 'mean', value: EMIT_SIGMA.map((s, i) => EMIT_LE[i]! * (1 - Math.exp(-s))) as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
};
/** The camera looks down -z from z = 5 into a slab over z in [-0.5, 0.5]: L = integral over depth of Le sigma_a(z) exp(-tau(from the entry z = 0.5 down to z)) dz, by quadrature of the same trilinear density. */
function gridEmitExpected(): number[] {
  const g = slabGrid();
  const dens = gridDensity({ resolution: g.resolution, worldToIndex: gridWorldToIndex(g.resolution, g.bounds), densityMax: Math.max(...g.density) }, g.density);
  const n = 40000;
  return GRID_SIGMA.map((s, c) => {
    let tau = 0, L = 0;
    for (let i = 0; i < n; i++) {
      const d = dens(0, 0, 0.5 - (i + 0.5) / n), dz = 1 / n;
      tau += s * d * dz * 0.5;
      L += EMIT_LE[c]! * s * d * Math.exp(-tau) * dz;
      tau += s * d * dz * 0.5;
    }
    return L;
  });
}
export const volEmitGridSlab: SceneDesc = (() => {
  const g = slabGrid();
  return {
    version: 1,
    name: 'vol-emit-grid-slab',
    purpose: 'Emissive grid slab (16 density layers along the normal, absorption only, Le 2/1/0.5) on a black background: pixel = integral of Le sigma_a(z) exp(-tau) dz of the trilinear density (V4.6).',
    tier: 1,
    camera: NARROW([0, 0, 5]),
    film: FILM,
    integrator: { maxDepth: 8 },
    materials: { boundary: { type: 'interface' } },
    media: { layers: { type: 'grid', resolution: g.resolution, density: g.density, bounds: g.bounds, sigmaA: GRID_SIGMA, sigmaS: [0, 0, 0], Le: EMIT_LE } },
    meshes: [{ name: 'slab', geometry: { type: 'box', size: [8, 8, 1] }, material: 'boundary', medium: { interior: 'layers' } }],
    environment: BLACK,
    expected: { type: 'mean', value: gridEmitExpected() as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
  };
})();

/** A scattering emissive sphere (albedo 0.5, g 0.3, grey so pbrt's spectral uplift is exact) on a black background: our emission + in-scattering against pbrt-v4 `homogeneous` with `Le` (Mitsuba 3 has no emissive medium: no second reference). */
export const volEmitSphere: SceneDesc = {
  version: 1,
  name: 'vol-emit-sphere',
  purpose: 'Emissive scattering sphere (r 0.8, sigma_a = sigma_s = 1.2, g 0.3, Le 1) on a black background: emission plus scattered emission, pbrt-v4 reference only (V4.6).',
  tier: 2,
  camera: { type: 'perspective', position: [0, 0.2, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
  film: { width: 96, height: 96, filter: 'box' },
  integrator: { maxDepth: 32 },
  materials: { boundary: { type: 'interface' } },
  media: { glow: { type: 'homogeneous', sigmaA: [1.2, 1.2, 1.2], sigmaS: [1.2, 1.2, 1.2], g: 0.3, Le: [1, 1, 1] } },
  meshes: [{ name: 'ball', geometry: { type: 'sphere', radius: 0.8, segments: 64, rings: 32 }, material: 'boundary', medium: { interior: 'glow' } }],
  environment: BLACK,
};

// V6.2 nested media (the `volumes.mediumStack` option; pass `pt_vol_stack=1`). No explicit exteriors anywhere: without the stack, leaving the inner boundary sets the medium to vacuum.
const NEST_A = [0.3, 0.5, 0.2] as const;
const NEST_B = [1.0, 0.4, 0.7] as const;
const BOUNDARY = { boundary: { type: 'interface' as const } };
const stackExpect = (l: readonly [number, number, number], a: number, b: number) => ({ type: 'mean' as const, value: l.map((x, i) => x * Math.exp(-(a * NEST_A[i]! + b * NEST_B[i]!))) as unknown as [number, number, number], tolerance: 0.01, pixelwise: false });
export const volNestedAbsorb: SceneDesc = {
  version: 1,
  name: 'vol-nested-absorb',
  purpose: 'An absorbing sphere (radius 1, sigma_b) inside an absorbing box (4 m, sigma_a), no explicit exteriors, unit sky, camera outside along the axis: pixel = exp(-(sigma_a 2 + sigma_b 2)) (V6.2, needs the medium stack).',
  tier: 1,
  camera: NARROW([0, 0, 6]),
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: BOUNDARY,
  media: { outer: { type: 'homogeneous', sigmaA: NEST_A, sigmaS: [0, 0, 0] }, inner: { type: 'homogeneous', sigmaA: NEST_B, sigmaS: [0, 0, 0] } },
  meshes: [
    { name: 'box', geometry: { type: 'box', size: [4, 4, 4] }, material: 'boundary', medium: { interior: 'outer' } },
    { name: 'ball', geometry: { type: 'sphere', radius: 1, segments: 128, rings: 64 }, material: 'boundary', medium: { interior: 'inner' } },
  ],
  environment: SKY,
  expected: stackExpect([1, 1, 1], 2, 2),
};
export const volNestedOverlap: SceneDesc = {
  version: 1,
  name: 'vol-nested-overlap',
  purpose: 'Two overlapping absorbing boxes along the axis (b: z in [0, 2], a: z in [-1, 1]), unit sky: the most recently entered medium wins in the overlap, pixel = exp(-(sigma_b 1 + sigma_a 2)) (V6.2, needs the medium stack).',
  tier: 1,
  camera: NARROW([0, 0, 6]),
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: BOUNDARY,
  media: { a: { type: 'homogeneous', sigmaA: NEST_A, sigmaS: [0, 0, 0] }, b: { type: 'homogeneous', sigmaA: NEST_B, sigmaS: [0, 0, 0] } },
  meshes: [
    { name: 'boxA', geometry: { type: 'box', size: [4, 4, 2] }, material: 'boundary', medium: { interior: 'a' } },
    { name: 'boxB', geometry: { type: 'box', size: [4, 4, 2] }, material: 'boundary', medium: { interior: 'b' }, transform: translation(0, 0, 1) },
  ],
  environment: SKY,
  expected: { type: 'mean', value: NEST_A.map((a, i) => Math.exp(-(NEST_B[i]! * 1 + a * 2))) as unknown as [number, number, number], tolerance: 0.01, pixelwise: false },
};
export const volNestedCamera: SceneDesc = {
  version: 1,
  name: 'vol-nested-camera',
  purpose: 'The camera inside an absorbing box (40 m, sigma_a) that holds an absorbing sphere (radius 1 at z = -2, sigma_b) and an emitter at z = -4: pixel = L exp(-(sigma_a 2 + sigma_b 2)) (V6.2, camera stack).',
  tier: 1,
  camera: NARROW([0, 0, 0], [0, 0, -1]),
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: { ...BOUNDARY, back: { type: 'black' } },
  media: { outer: { type: 'homogeneous', sigmaA: NEST_A, sigmaS: [0, 0, 0] }, inner: { type: 'homogeneous', sigmaA: NEST_B, sigmaS: [0, 0, 0] } },
  meshes: [
    { name: 'box', geometry: { type: 'box', size: [40, 40, 40] }, material: 'boundary', medium: { interior: 'outer' } },
    { name: 'ball', geometry: { type: 'sphere', radius: 1, segments: 128, rings: 64 }, material: 'boundary', medium: { interior: 'inner' }, transform: translation(0, 0, -2) },
    { name: 'emitter', geometry: { type: 'quad', size: [8, 8] }, material: 'back', transform: facingCamera(-4), emission: LB },
  ],
  expected: stackExpect(LB, 2, 2),
};
export const volNestedFurnace: SceneDesc = {
  version: 1,
  name: 'vol-nested-furnace',
  purpose: 'A scattering, non-absorbing sphere (albedo 1) inside a scattering, non-absorbing box (albedo 1), no explicit exteriors, unit sky, maxDepth 400: every pixel is 1 (V6.2: with the stack a path that leaves the sphere is back in the box medium; without it, in vacuum).',
  tier: 1,
  camera: { type: 'perspective', position: [0, 0, 8], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 25 },
  film: { width: 48, height: 48, filter: 'box' },
  integrator: { maxDepth: 400 },
  materials: BOUNDARY,
  media: { outer: { type: 'homogeneous', sigmaA: [0, 0, 0], sigmaS: [0.4, 0.4, 0.4], g: 0.3 }, inner: { type: 'homogeneous', sigmaA: [0, 0, 0], sigmaS: [1.5, 1.5, 1.5], g: -0.4 } },
  meshes: [
    { name: 'box', geometry: { type: 'box', size: [4, 4, 4] }, material: 'boundary', medium: { interior: 'outer' } },
    { name: 'ball', geometry: { type: 'sphere', radius: 1, segments: 64, rings: 32 }, material: 'boundary', medium: { interior: 'inner' } },
  ],
  environment: SKY,
  expected: { type: 'mean', value: [1, 1, 1], tolerance: 0.01, pixelwise: false },
};

/**
 * V5.1d control: `vol-cornell-smoke` with the cube lifted 1 mm off the floor. In the original the cube's bottom face (an interface) is coplanar with the floor, so a ray from a scatter point
 * toward the floor hits two triangles at the same t up to an ulp: which one wins depends on the last bit of the scatter point, i.e. on how the compiler contracted its arithmetic. Two differently
 * compiled kernels (the megakernel with and without AOVs, V5.1a; the wavefront with and without frame planes, V5.1d) then flip ~0.1% of the paths. Without the contact they agree per pixel.
 */
export const volCornellSmokeLift: SceneDesc = {
  ...volCornellSmoke,
  name: 'vol-cornell-smoke-lift',
  purpose: 'vol-cornell-smoke with the smoke cube 1 mm above the floor (no coplanar interface / floor faces): per-pixel scheduler agreement control (V5.1d).',
  meshes: volCornellSmoke.meshes.map((m) => (m.name === 'smoke-cube' ? { ...m, transform: translation(0.185, 0.111, 0.169) } : m)),
};

export const volumeCheckScenes: readonly SceneDesc[] = [volEmitSlab, volEmitGridSlab, volAbsorbSlab, volCameraMedium, volFurnaceSphere, volFurnaceSphereBack, volSlabSingle, volSlabSingleOblique, volGridSlab, volCameraGrid, volCameraGridAuto];
export const volumeReferenceScenes: readonly SceneDesc[] = [volCornellSmoke, volCornellFog, volCornellFogEmpty, volMilkSphere, volGridCloud, volVdbSmoke, volVdbWdasCloud, volVdbBunnyCloud];
export const volumeStackScenes: readonly SceneDesc[] = [volNestedAbsorb, volNestedOverlap, volNestedCamera, volNestedFurnace];
export const volumeScenes: readonly SceneDesc[] = [...volumeCheckScenes, ...volumeStackScenes, ...volumeReferenceScenes, volEmitSphere, volGridSlabInner, volCornellSmokeLift, ...cloudScenes];
