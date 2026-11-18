import { gridDensity, gridWorldToIndex, slabSingleScatter } from '@pmndrs/pathtracer';
import { compose, rotationX, translation } from '../geometry.ts';
import type { MediumDesc, MeshDesc, SceneDesc } from '../schema.ts';
import { cornell } from '../tier0/cornell.ts';

/**
 * Participating-media scenes (P5, docs/design/volumes.md).
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
const NARROW = (position: readonly [number, number, number]) => ({ type: 'perspective' as const, position, target: [0, 0, 0] as const, up: [0, 1, 0] as const, yFovDeg: 1 });
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
  expected: { type: 'mean', value: SIGMA_A.map((s) => Math.exp(-s)) as unknown as [number, number, number], tolerance: 0.003 },
};

const CAM_SIGMA = [0.1, 0.3, 0.5] as const;
export const volCameraMedium: SceneDesc = {
  version: 1,
  name: 'vol-camera-medium',
  purpose: 'Camera inside an absorbing medium that fills space, an emitter 3 m away: pixel = L exp(-sigma_a 3) (P5, cameraMedium).',
  tier: 1,
  camera: NARROW([0, 0, 0]),
  film: FILM,
  integrator: { maxDepth: 8 },
  materials: { back: { type: 'black' } },
  media: { fog: { type: 'homogeneous', sigmaA: CAM_SIGMA, sigmaS: [0, 0, 0] } },
  cameraMedium: 'fog',
  meshes: [{ name: 'emitter', geometry: { type: 'quad', size: [8, 8] }, material: 'back', transform: facingCamera(-3), emission: LB }],
  expected: { type: 'mean', value: LB.map((l, i) => l * Math.exp(-3 * CAM_SIGMA[i]!)) as unknown as [number, number, number], tolerance: 0.003 },
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
  expected: { type: 'mean', value: [1, 1, 1], tolerance: 0.01 },
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
    expected: { type: 'mean', value: [expected, expected, expected], tolerance: 0.002 },
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
    expected: { type: 'mean', value: gridSlabExpected() as unknown as [number, number, number], tolerance: 0.004 },
  };
})();

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

export const volumeCheckScenes: readonly SceneDesc[] = [volAbsorbSlab, volCameraMedium, volFurnaceSphere, volFurnaceSphereBack, volSlabSingle, volSlabSingleOblique, volGridSlab];
export const volumeReferenceScenes: readonly SceneDesc[] = [volCornellSmoke, volMilkSphere, volGridCloud];
export const volumeScenes: readonly SceneDesc[] = [...volumeCheckScenes, ...volumeReferenceScenes];
