/**
 * Fog (experimental): sunlit thin fog with light shafts, and the estimators that make it converge.
 *
 * A room with slits in its back wall, a few pillars and a small lamp is filled with a thin homogeneous fog (one closed box with a medium-boundary material,
 * optical depth about 0.5 across the room). A low sun shines through the slits, so the fog in the beams scatters light towards the camera: god rays.
 *
 * In thin fog most camera rays never collide with a particle, so the plain estimator (pbrt's volpath: light is sampled only where a real collision happens)
 * sees a shaft only through the rare rays that do collide: a very noisy image. `forcedNee` adds one light sample per free-flight segment, at a distance drawn
 * on the segment, and `equiangular` draws that distance around the lamp so that a lamp inside the fog stops being a 1/r^2 noise source. Both are unbiased:
 * the three estimators converge to the same image, only the noise differs. The estimator is compiled into the kernel (a change rebuilds it).
 */
import { buildGpuScene, type GpuScene, type SceneCopyInput, type RtMaterialInput } from '@pmndrs/raytracer';
import { ptMediumFrom, type PtIntegratorOptions } from '@pmndrs/pathtracer';
import { box, mesh } from '../../shared/cornell.ts';
import { defineExample } from '../../shared/example.ts';
import { sunFrom } from '../../shared/models.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

type V3 = [number, number, number];
type Estimator = 'plain' | 'forced' | 'equiangular';

const query = new URLSearchParams(location.search);
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** The room's inside is x, z in [-3, 3], y in [0, 3]; the fog box is inset by 1 cm so no surface is shared with the walls. */
const FOG_MIN: V3 = [-2.99, 0.01, -2.99];
const FOG_MAX: V3 = [2.99, 2.99, 2.99];

/** Boxes (centre, half size) as one mesh. */
const boxes = (list: [V3, V3][]) => mesh(list.flatMap(([c, h]) => box(c, h, 0)));

/** The static room: floor, ceiling, side walls, a back wall with four slits, two pillars. One grey material. */
function room() {
  const back: [V3, V3][] = [];
  const slitX = [-2.1, -0.7, 0.7, 2.1], w = 0.25;
  // Columns between the slits (full height), then a sill and a lintel over each slit.
  const edges = [-3.4, ...slitX.flatMap((x) => [x - w, x + w]), 3.4];
  for (let i = 0; i < edges.length; i += 2) back.push([[(edges[i]! + edges[i + 1]!) / 2, 1.5, -3.2], [(edges[i + 1]! - edges[i]!) / 2, 1.6, 0.2]]);
  for (const x of slitX) {
    back.push([[x, 0.15, -3.2], [w, 0.25, 0.2]]);
    back.push([[x, 2.85, -3.2], [w, 0.35, 0.2]]);
  }
  return boxes([
    [[0, -0.2, 0], [3.4, 0.2, 3.4]], // floor
    [[0, 3.2, 0], [3.4, 0.2, 3.4]], // ceiling
    [[-3.2, 1.5, 0], [0.2, 1.6, 3.4]], // left wall
    [[3.2, 1.5, 0], [0.2, 1.6, 3.4]], // right wall
    [[0, 1.5, 3.2], [3.4, 1.6, 0.2]], // front wall (behind the camera)
    ...back,
    [[-1.2, 1.5, -0.6], [0.28, 1.5, 0.28]], // pillars in the beams
    [[1.5, 1.0, 0.5], [0.28, 1.0, 0.28]],
  ]);
}

defineExample(async (ctx) => {
  const { device, pane, canvas } = ctx;
  const scale = Number(query.get('scale') ?? 0);
  if (scale > 0) ctx.setRenderScale(scale);
  const est = query.get('estimator');
  const params = {
    density: 0.2, g: 0.6, albedo: 0.9, sunElevation: 22, sunAzimuth: 165, sun: 25,
    lamp: true, lampPower: 30, estimator: (est === 'plain' || est === 'forced' || est === 'equiangular' ? est : 'equiangular') as Estimator,
    maxDepth: 6,
  };

  const orbit = new OrbitCamera(canvas, { position: [0.3, 1.4, 2.75], target: [0, 1.3, -1.5], up: [0, 1, 0], yFovDeg: 62 });
  const roomMesh = room();
  // A small lamp hanging in the fog, facing down (it emits from its front face).
  const lampMesh = mesh([[[-0.15, 2.2, -0.15], [0.15, 2.2, -0.15], [0.15, 2.2, 0.15], [-0.15, 2.2, 0.15]]]);
  const fogMesh = mesh(box([0, 1.5, 0], [(FOG_MAX[0] - FOG_MIN[0]) / 2, (FOG_MAX[1] - FOG_MIN[1]) / 2, (FOG_MAX[2] - FOG_MIN[2]) / 2], 0));
  const input = (): SceneCopyInput => {
    const lamp: V3 = [params.lampPower, params.lampPower * 0.85, params.lampPower * 0.55];
    const materials: RtMaterialInput[] = [
      { baseColor: [0.2, 0.19, 0.18], specularWeight: 0 },
      { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0, emissionColor: lamp },
      { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 }, // the fog boundary scatters nothing itself
    ];
    const meshes = [roomMesh, fogMesh, ...(params.lamp ? [lampMesh] : [])];
    return {
      meshes,
      instances: meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: IDENTITY, material: [0, 2, 1][i]! })),
      materials,
      sun: sunFrom(params.sunElevation, params.sunAzimuth, params.sun),
      environment: { radiance: [0.04, 0.05, 0.07] },
    };
  };

  let gpuScene: GpuScene | null = null;
  let renderer: ProgressiveRenderer | null = null;

  const volumes = (): PtIntegratorOptions['volumes'] => {
    const sigmaS = params.density * params.albedo, sigmaA = params.density * (1 - params.albedo);
    return {
      media: [ptMediumFrom([sigmaA, sigmaA, sigmaA], [sigmaS, sigmaS, sigmaS], params.g)],
      // Scene materials: 0 walls, 1 lamp, 2 the fog boundary (medium 1 inside).
      materials: [{ interior: 0, exterior: 0, boundary: false }, { interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }],
      ...(params.estimator !== 'plain' ? { forcedNee: true } : {}),
      ...(params.estimator === 'equiangular' ? { equiangular: true } : {}),
    };
  };
  const updateStats = () => ctx.stats.set('optical depth across the room', (params.density * 6).toFixed(2));

  /** Rebuilds the kernel (the medium and the estimator are compiled in). */
  const rebuild = async () => {
    ctx.progress('Compiling…');
    try {
      gpuScene ??= await buildGpuScene(device, input(), { label: 'fog' });
      const next = await ProgressiveRenderer.create(ctx, gpuScene, orbit.camera, { integrator: { volumes: volumes(), sunVisible: false } });
      next.settings.maxDepth = params.maxDepth;
      const old = renderer;
      renderer = next;
      updateStats();
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('Could not build the fog', error);
    }
  };
  /** The sun and the lamp only change the scene copy: rebuild it (cheap) and rebind. */
  const rebuildScene = async () => {
    const old = gpuScene;
    gpuScene = await buildGpuScene(device, input(), { label: 'fog' });
    renderer?.setScene(gpuScene);
    if (old) void device.queue.onSubmittedWorkDone().then(() => old.destroy());
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);
  const scene = (e: { last: boolean }) => e.last && void rebuildScene();
  const recompile = (e: { last: boolean }) => e.last && void rebuild();

  const estimator = pane.addFolder({ title: 'Estimator (rebuilds the kernel)' });
  estimator.addBinding(params, 'estimator', { label: 'light in fog', options: { 'Plain (collisions only)': 'plain', 'Forced in-scatter NEE': 'forced', 'Forced + equiangular lamp': 'equiangular' } }).on('change', () => void rebuild());

  const fog = pane.addFolder({ title: 'Fog (rebuilds the kernel)' });
  fog.addBinding(params, 'density', { label: 'extinction (1/m)', min: 0.01, max: 0.4, step: 0.005 }).on('change', recompile);
  fog.addBinding(params, 'g', { label: 'anisotropy g', min: 0, max: 0.95, step: 0.01 }).on('change', recompile);
  fog.addBinding(params, 'albedo', { label: 'albedo', min: 0.5, max: 1, step: 0.01 }).on('change', recompile);

  const sun = pane.addFolder({ title: 'Sun' });
  sun.addBinding(params, 'sunElevation', { label: 'elevation (°)', min: 3, max: 80, step: 1 }).on('change', scene);
  sun.addBinding(params, 'sunAzimuth', { label: 'azimuth (°)', min: -180, max: 180, step: 1 }).on('change', scene);
  sun.addBinding(params, 'sun', { label: 'irradiance', min: 0, max: 20, step: 0.5 }).on('change', scene);

  const lamp = pane.addFolder({ title: 'Lamp' });
  lamp.addBinding(params, 'lamp', { label: 'on' }).on('change', () => void rebuildScene());
  lamp.addBinding(params, 'lampPower', { label: 'radiance', min: 1, max: 200, step: 1 }).on('change', scene);

  const path = pane.addFolder({ title: 'Path' });
  path.addBinding(params, 'maxDepth', { label: 'max scatters', min: 1, max: 16, step: 1 }).on('change', (e) => {
    if (!renderer) return;
    renderer.settings.maxDepth = params.maxDepth;
    if (e.last) renderer.restart();
  });

  await rebuild();
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
