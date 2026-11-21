import { hgPhase, phaseEval, ptPhaseLobes, slabSingleScatterPhase } from '@pmndrs/pathtracer';
import type { MediumDesc, MeshDesc, SceneDesc } from '../schema.ts';

/**
 * The cloud regime (P5.v13, docs/internal/CLOUDS-HANDOFF.md item 1): strongly forward scattering (HG g 0.8 to 0.99), albedo 0.99 to 1, optical thickness 10 to 200,
 * hundreds of scattering orders (`maxDepth` 512, no Russian roulette), a delta sun plus a uniform sky. Grey sigma everywhere (pbrt's spectral uplift is then exact, V62).
 *
 * Closed forms (ours only):
 *  - `vol-cl-furnace-g<G>-t<T>`: a non-absorbing sphere (diameter 1 m, optical diameter T) under a unit sky; the camera sees only the sphere (every pixel's ray crosses it).
 *    Every pixel is 1 once `maxDepth` is high enough; the scenes carry maxDepth 4096 and `pt_maxdepth` sweeps the cap (how many orders a cloud needs).
 *  - `vol-cl-single-g<G>[-oblique]`: the single-scattering slab of `vol-slab-single` (tau 0.4, albedo 0.8, max depth 1) at a high g; the expected value is the peak-resolved
 *    quadrature `slabSingleScatterPhase` (the uniform-cosine one of the older scenes is 1% off at g 0.85).
 * References (pbrt-v4 `volpath` and Mitsuba 3 `volpath`, at their own noise floor):
 *  - `vol-cl-sph-g<G>-a<A>-t<T>`: the same sphere, scattering, under the sun (partly behind it: the forward lobe lights the rim) and a sky of 0.4, maxDepth 512.
 *  - `vol-cl-slab-<top|base>-g<G>-a<A>-t<T>`: a 1 m slab (40 m wide), sun from above at 25 degrees off the normal, seen from above (reflection) or below (transmission).
 *  - `vol-cl-cloud256`: a 256^3 grid (2 m voxels, a 512 m box) with a peak sigma_t of 0.3 per metre (optical depth through the core about 60), g 0.85, albedo 0.99:
 *    the clouds page's own scale. Needs `node bench/scenes/volumes/cloud-grid.ts` (writes the git-ignored density file).
 */

const pct = (v: number): string => String(Math.round(v * 100)).padStart(2, '0');
const SUN_SPHERE = { type: 'directional' as const, direction: [-0.55, -0.45, 0.7] as const, irradiance: [3, 3, 3] as const };
const SKY_04 = { type: 'uniform' as const, radiance: [0.4, 0.4, 0.4] as const };
const ball = (medium: string): MeshDesc => ({ name: 'ball', geometry: { type: 'sphere', radius: 0.5, segments: 96, rings: 48 }, material: 'boundary', medium: { interior: medium } });
const grey = (tau: number, albedo: number, g: number) => ({ type: 'homogeneous' as const, sigmaA: [tau * (1 - albedo), tau * (1 - albedo), tau * (1 - albedo)] as const, sigmaS: [tau * albedo, tau * albedo, tau * albedo] as const, g });

export const CLOUD_G = [0.8, 0.85, 0.9, 0.95, 0.99] as const;
export const CLOUD_TAU = [10, 50, 200] as const;

const furnace = (g: number, tau: number): SceneDesc => ({
  version: 1,
  name: `vol-cl-furnace-g${pct(g)}-t${tau}`,
  purpose: `Non-absorbing sphere (optical diameter ${tau}, HG g ${g}) under a unit sky, camera inside the silhouette: every pixel is 1 once maxDepth is high enough (P5.v13).`,
  tier: 1,
  // the sphere subtends 7.2 degrees half-angle at 4 m; a 12 degree field keeps every pixel's ray inside it
  camera: { type: 'perspective', position: [0, 0, 4], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 12 },
  film: { width: 32, height: 32, filter: 'box' },
  integrator: { maxDepth: 4096 },
  materials: { boundary: { type: 'interface' } },
  media: { cloud: grey(tau, 1, g) },
  meshes: [ball('cloud')],
  environment: { type: 'uniform', radiance: [1, 1, 1] },
  expected: { type: 'mean', value: [1, 1, 1], tolerance: 0.002, pixelwise: false },
});
export const cloudFurnaceScenes: readonly SceneDesc[] = CLOUD_G.flatMap((g) => CLOUD_TAU.map((t) => furnace(g, t)));

const SINGLE = { sigmaT: 0.4, albedo: 0.8, thickness: 1 } as const;
const single = (g: number, cosine: number): SceneDesc => {
  const sin = Math.sqrt(1 - cosine * cosine);
  const expected = slabSingleScatterPhase(SINGLE.sigmaT, SINGLE.albedo, (c) => hgPhase(c, g), SINGLE.thickness, cosine, 1, 1024, 128);
  return {
    version: 1,
    name: `vol-cl-single-g${pct(g)}${cosine === 1 ? '' : '-oblique'}`,
    purpose: `Thin scattering slab (tau 0.4, albedo 0.8, HG g ${g}) under a unit sky seen at cos ${cosine}, max depth 1: unscattered + single scattering, pixel = ${expected.toFixed(6)} (peak-resolved quadrature, P5.v13).`,
    tier: 1,
    camera: { type: 'perspective', position: [0, 5 * sin, 5 * cosine], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 1 },
    film: { width: 32, height: 32, filter: 'box' },
    integrator: { maxDepth: 1 },
    materials: { boundary: { type: 'interface' } },
    media: { haze: grey(SINGLE.sigmaT, SINGLE.albedo, g) },
    meshes: [{ name: 'slab', geometry: { type: 'box', size: [40, 40, SINGLE.thickness] }, material: 'boundary', medium: { interior: 'haze' } }],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
    expected: { type: 'mean', value: [expected, expected, expected], tolerance: 0.002, pixelwise: false },
  };
};
export const cloudSingleScenes: readonly SceneDesc[] = [single(0.85, 1), single(0.85, 0.6), single(0.95, 0.6), single(0.99, 0.6)];

// --- P5.v14 phase functions ---------------------------------------------------------------------------------------------------------------------------------
type PhaseDesc = NonNullable<MediumDesc['phase']>;
/** The clouds runtime's art default (0.2 at 60 %, 0.85 at 40 %) and the droplet fit at 10 and 20 um. */
export const CLOUD_PHASES: ReadonlyArray<[string, PhaseDesc]> = [
  ['hg2', { type: 'hg2', g1: 0.2, g2: 0.85, w2: 0.4 }],
  ['mie10', { type: 'mie', diameter: 10 }],
  ['mie20', { type: 'mie', diameter: 20 }],
];
const singlePhase = (tag: string, phase: PhaseDesc, cosine: number): SceneDesc => {
  const sin = Math.sqrt(1 - cosine * cosine);
  const lobes = ptPhaseLobes(phase);
  const expected = slabSingleScatterPhase(SINGLE.sigmaT, SINGLE.albedo, (c) => phaseEval(lobes, c), SINGLE.thickness, cosine, 1, 1024, 128);
  return {
    version: 1,
    name: `vol-cl-single-${tag}${cosine === 1 ? '' : '-oblique'}`,
    purpose: `Thin scattering slab (tau 0.4, albedo 0.8, phase ${JSON.stringify(phase)}) under a unit sky seen at cos ${cosine}, max depth 1: pixel = ${expected.toFixed(6)} (peak-resolved quadrature of the mixture, P5.v14). Run with pt_nee=0 too: the image is then the phase sampler alone.`,
    tier: 1,
    camera: { type: 'perspective', position: [0, 5 * sin, 5 * cosine], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 1 },
    film: { width: 32, height: 32, filter: 'box' },
    integrator: { maxDepth: 1 },
    materials: { boundary: { type: 'interface' } },
    media: { haze: { ...grey(SINGLE.sigmaT, SINGLE.albedo, 0), phase } },
    meshes: [{ name: 'slab', geometry: { type: 'box', size: [40, 40, SINGLE.thickness] }, material: 'boundary', medium: { interior: 'haze' } }],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
    expected: { type: 'mean', value: [expected, expected, expected], tolerance: 0.002, pixelwise: false },
  };
};
export const cloudPhaseSingleScenes: readonly SceneDesc[] = CLOUD_PHASES.flatMap(([tag, p]) => [singlePhase(tag, p, 1), singlePhase(tag, p, 0.6)]);

const sphere = (g: number, albedo: number, tau: number): SceneDesc => ({
  version: 1,
  name: `vol-cl-sph-g${pct(g)}-a${albedo === 1 ? '100' : pct(albedo)}-t${tau}`,
  purpose: `Sphere of grey medium (optical diameter ${tau}, albedo ${albedo}, HG g ${g}), sun partly behind it plus a sky of 0.4, maxDepth 512, no RR: the cloud regime against pbrt and Mitsuba (P5.v13).`,
  tier: 2,
  camera: { type: 'perspective', position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 24 },
  film: { width: 64, height: 64, filter: 'box' },
  integrator: { maxDepth: 512 },
  materials: { boundary: { type: 'interface' } },
  media: { cloud: grey(tau, albedo, g) },
  meshes: [ball('cloud')],
  lights: [SUN_SPHERE],
  environment: SKY_04,
});
export const cloudSphereScenes: readonly SceneDesc[] = [0.8, 0.85, 0.9].flatMap((g) => [0.99, 1].flatMap((a) => CLOUD_TAU.map((t) => sphere(g, a, t))));
/** The sphere at tau 50, albedo 0.99 with the P5.v14 phases: `hg2` has a Mitsuba reference (blendphase), the Mie ones need Cycles. */
export const cloudPhaseSphereScenes: readonly SceneDesc[] = CLOUD_PHASES.map(([tag, phase]) => {
  const base = sphere(0, 0.99, 50);
  return { ...base, name: `vol-cl-sph-${tag}-a99-t50`, purpose: `${base.purpose.replace('HG g 0', `phase ${JSON.stringify(phase)}`)} (P5.v14)`, media: { cloud: { ...grey(50, 0.99, 0), phase } } };
});

const slab = (view: 'top' | 'base', g: number, albedo: number, tau: number): SceneDesc => ({
  version: 1,
  name: `vol-cl-slab-${view}-g${pct(g)}-a${albedo === 1 ? '100' : pct(albedo)}-t${tau}`,
  purpose: `1 m slab of grey medium (optical thickness ${tau}, albedo ${albedo}, HG g ${g}) under the sun (25 degrees off the normal) and a sky of 0.4, seen from ${view === 'top' ? 'above (reflection)' : 'below (transmission)'}, maxDepth 512 (P5.v13).`,
  tier: 2,
  camera: { type: 'perspective', position: [0, view === 'top' ? 2.5 : -2.5, 1.8], target: [0, view === 'top' ? 0.5 : -0.5, 0], up: [0, 1, 0], yFovDeg: 40 },
  film: { width: 64, height: 64, filter: 'box' },
  integrator: { maxDepth: 512 },
  materials: { boundary: { type: 'interface' } },
  media: { deck: grey(tau, albedo, g) },
  meshes: [{ name: 'slab', geometry: { type: 'box', size: [40, 1, 40] }, material: 'boundary', medium: { interior: 'deck' } }],
  lights: [{ type: 'directional', direction: [-0.3, -1, -0.3], irradiance: [3, 3, 3] }],
  environment: SKY_04,
});
export const cloudSlabScenes: readonly SceneDesc[] = [
  ...(['top', 'base'] as const).flatMap((v) => CLOUD_TAU.map((t) => slab(v, 0.85, 0.99, t))),
  slab('base', 0.8, 0.99, 50),
  slab('base', 0.9, 0.99, 50),
  slab('top', 0.85, 1, 200),
];

/** The density file of `vol-cl-cloud256` (bench/assets/generated, git-ignored): `node bench/scenes/volumes/cloud-grid.ts`. */
export const CLOUD256 = { file: 'generated/volumes/cloud256.f32', n: 256, half: 256 } as const;
export const volClCloud256: SceneDesc = {
  version: 1,
  name: 'vol-cl-cloud256',
  purpose: 'Procedural cloud on a 256^3 grid (2 m voxels, 512 m box), peak sigma_t 0.3/m, albedo 0.99, HG g 0.85, sun behind-left plus sky, maxDepth 512: the clouds page\'s scale (P5.v13; needs `node bench/scenes/volumes/cloud-grid.ts`).',
  tier: 2,
  camera: { type: 'perspective', position: [0, 150, 1400], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 24 },
  film: { width: 128, height: 128, filter: 'box' },
  integrator: { maxDepth: 512 },
  materials: { boundary: { type: 'interface' } },
  media: { cloud: { type: 'grid', resolution: [CLOUD256.n, CLOUD256.n, CLOUD256.n], densityFile: CLOUD256.file, bounds: [[-CLOUD256.half, -CLOUD256.half, -CLOUD256.half], [CLOUD256.half, CLOUD256.half, CLOUD256.half]], sigmaA: [0.003, 0.003, 0.003], sigmaS: [0.297, 0.297, 0.297], g: 0.85 } },
  meshes: [{ name: 'cloud-box', geometry: { type: 'box', size: [2 * CLOUD256.half, 2 * CLOUD256.half, 2 * CLOUD256.half] }, material: 'boundary', medium: { interior: 'cloud' } }],
  lights: [SUN_SPHERE],
  environment: SKY_04,
};

export const cloudScenes: readonly SceneDesc[] = [...cloudFurnaceScenes, ...cloudSingleScenes, ...cloudPhaseSingleScenes, ...cloudSphereScenes, ...cloudPhaseSphereScenes, ...cloudSlabScenes, volClCloud256];
