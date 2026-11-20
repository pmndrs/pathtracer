/**
 * Participating media (experimental): fog and a cloud in the Cornell box.
 *
 * A medium lives inside a closed mesh whose material is marked as a medium boundary: rays pass through the boundary and only switch the
 * medium they travel in. `integrator.volumes` takes the media (`ptMediumFrom(sigmaA, sigmaS, g, scale, grid?)`: absorption and scattering
 * per metre, the Henyey-Greenstein anisotropy g) and, per scene-copy material, the medium on each side. A grid medium multiplies the
 * coefficients by a density read from a 3D texture (`volumeGrid`), filled here with procedural noise.
 */
import { buildGpuScene, type GpuScene, type SceneCopyInput } from '@pmndrs/raytracer';
import { gridWorldToIndex, ptMediumFrom, type PtIntegratorOptions } from '@pmndrs/pathtracer';
import { box, cornellBox, CORNELL_CAMERA, mesh } from '../../shared/cornell.ts';
import { defineExample } from '../../shared/example.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

const RES = 48;
const CLOUD_BOUNDS = [[-0.7, 0.4, -0.7], [0.7, 1.7, 0.7]] as const;

/** A soft ball of value noise, x fastest then y then z, zero on the border (the grid reads zero outside). */
function cloudDensity(): Float32Array {
  const d = new Float32Array(RES * RES * RES);
  const hash = (x: number, y: number, z: number) => {
    const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
    return h - Math.floor(h);
  };
  const noise = (x: number, y: number, z: number) => {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
    const fx = x - xi, fy = y - yi, fz = z - zi;
    const s = (t: number) => t * t * (3 - 2 * t);
    let v = 0;
    for (let c = 0; c < 8; c++) {
      const ox = c & 1, oy = (c >> 1) & 1, oz = (c >> 2) & 1;
      v += hash(xi + ox, yi + oy, zi + oz) * (ox ? s(fx) : 1 - s(fx)) * (oy ? s(fy) : 1 - s(fy)) * (oz ? s(fz) : 1 - s(fz));
    }
    return v;
  };
  for (let z = 1; z < RES - 1; z++) {
    for (let y = 1; y < RES - 1; y++) {
      for (let x = 1; x < RES - 1; x++) {
        const p = [x / RES - 0.5, y / RES - 0.5, z / RES - 0.5];
        const r = Math.hypot(p[0]!, p[1]!, p[2]!) * 2;
        const n = noise(x * 0.18, y * 0.18, z * 0.18) * 0.65 + noise(x * 0.4, y * 0.4, z * 0.4) * 0.35;
        d[(z * RES + y) * RES + x] = Math.max(0, 1 - r * r * 1.3 - (1 - n) * 0.6) * 1.6;
      }
    }
  }
  return d;
}

/** The Cornell box plus a box-shaped medium boundary (material 4), in the whole room (fog) or around the cloud's grid. */
function scene(kind: 'fog' | 'cloud'): SceneCopyInput {
  const input = cornellBox();
  const b = kind === 'fog' ? { min: [-0.99, 0.01, -0.99], max: [0.99, 1.98, 0.99] } : { min: CLOUD_BOUNDS[0], max: CLOUD_BOUNDS[1] };
  const c = [0, 1, 2].map((i) => (b.min[i]! + b.max[i]!) / 2) as [number, number, number];
  const h = [0, 1, 2].map((i) => (b.max[i]! - b.min[i]!) / 2) as [number, number, number];
  const meshes = [...input.meshes, mesh(box(c, h, 0))];
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  return {
    ...input,
    meshes,
    instances: [...input.instances, { meshIndex: meshes.length - 1, instanceIndex: meshes.length - 1, objectToWorld: identity, material: 4 }],
    // The boundary scatters nothing itself: black, no specular.
    materials: [...input.materials, { baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 }],
  };
}

defineExample(async (ctx) => {
  const { device, pane, canvas } = ctx;
  const params = { kind: (new URLSearchParams(location.search).get('kind') ?? 'fog') as 'fog' | 'cloud', density: 0.6, albedo: 0.9, g: 0.3 };
  const orbit = new OrbitCamera(canvas, CORNELL_CAMERA);
  const density = cloudDensity();
  let gpuScene: GpuScene | null = null;
  let renderer: ProgressiveRenderer | null = null;

  const volumes = (): PtIntegratorOptions['volumes'] => {
    const sigmaS = params.density * params.albedo, sigmaA = params.density * (1 - params.albedo);
    const grid = params.kind === 'cloud' ? { resolution: [RES, RES, RES] as const, worldToIndex: gridWorldToIndex([RES, RES, RES], CLOUD_BOUNDS), densityMax: density.reduce((a, b) => Math.max(a, b), 0) } : undefined;
    const scale = params.kind === 'cloud' ? 6 : 1;
    return {
      media: [ptMediumFrom([sigmaA, sigmaA, sigmaA], [sigmaS, sigmaS, sigmaS], params.g, scale, grid)],
      // Per material: the Cornell box's four have no medium; the boundary has medium 1 (index 0 + 1) inside.
      materials: [0, 1, 2, 3].map(() => ({ interior: 0, exterior: 0, boundary: false })).concat([{ interior: 1, exterior: 0, boundary: true }]),
    };
  };

  const rebuild = async (newScene: boolean) => {
    ctx.progress('Compiling…');
    try {
      if (newScene || !gpuScene) {
        const old = gpuScene;
        gpuScene = await buildGpuScene(device, scene(params.kind), { label: 'media' });
        if (old) void device.queue.onSubmittedWorkDone().then(() => old.destroy());
      }
      const next = await ProgressiveRenderer.create(ctx, gpuScene, orbit.camera, {
        integrator: { volumes: volumes() },
        ...(params.kind === 'cloud' ? { volumeGrid: { resolution: [RES, RES, RES] as const, density } } : {}),
      });
      if (renderer) Object.assign(next.settings, renderer.settings);
      const old = renderer;
      renderer = next;
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('Could not build the media', error);
    }
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  const media = pane.addFolder({ title: 'Medium' });
  media.addBinding(params, 'kind', { label: 'medium', options: { 'Homogeneous fog': 'fog', 'Cloud (density grid)': 'cloud' } }).on('change', () => void rebuild(true));
  // The coefficients are compiled into the kernel's medium table: a change rebuilds the kernel.
  const recompile = (e: { last: boolean }) => e.last && void rebuild(false);
  media.addBinding(params, 'density', { label: 'extinction (1/m)', min: 0.05, max: 4, step: 0.05 }).on('change', recompile);
  media.addBinding(params, 'albedo', { label: 'scattering albedo', min: 0, max: 1, step: 0.01 }).on('change', recompile);
  media.addBinding(params, 'g', { label: 'anisotropy g', min: -0.9, max: 0.9, step: 0.05 }).on('change', recompile);

  await rebuild(true);
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
