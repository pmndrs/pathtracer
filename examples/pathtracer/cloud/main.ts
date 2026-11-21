/**
 * A cloud (experimental): a heterogeneous medium, path traced under a sun and a sky.
 *
 * The density is a 128^3 grid made on the page (a few soft blobs times fractal value noise, remapped so the cloud has an edge and most of the box
 * is empty). One closed box with a medium-boundary material bounds the grid: rays enter it, switch medium, and are tracked through the density by
 * delta tracking against 8^3 brick majorants (`majorant: 'brick'`). The medium is grey and almost non-absorbing (albedo 0.99), so light bounces
 * many times; the phase function is strongly forward, either one Henyey-Greenstein lobe or the water-droplet Mie fit (`ptPhaseLobes`).
 *
 * The medium table is a uniform buffer (`volumes.runtimeParams`): extinction, albedo, g and droplet size are written with `setMedia` and restart the accumulation, no recompile.
 * Switching the phase between Mie (two lobes) and Henyey-Greenstein changes the kernel's structure and rebuilds it (a few seconds); the sun only rewrites the scene copy.
 */
import { buildGpuScene, type GpuScene, type SceneCopyInput } from '@pmndrs/raytracer';
import { brickMajorants, gridWorldToIndex, ptPhaseLobes, type PtIntegratorOptions, type PtMedium } from '@pmndrs/pathtracer';
import { box, mesh } from '../../shared/cornell.ts';
import { defineExample } from '../../shared/example.ts';
import { sunFrom } from '../../shared/models.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

type V3 = [number, number, number];

/** The cloud's box in metres (the density grid fills it). */
const MIN: V3 = [-2.4, 0, -2.4];
const MAX: V3 = [2.4, 2.6, 2.4];
const query = new URLSearchParams(location.search);
const RES = Math.max(16, Math.min(192, Number(query.get('res') ?? 128)));

/** Integer hash to [0, 1). */
function hash(x: number, y: number, z: number, seed: number): number {
  let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 2147483647) ^ Math.imul(seed, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/** Smooth value noise in [0, 1]. */
function valueNoise(x: number, y: number, z: number, seed: number): number {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const fx = x - xi, fy = y - yi, fz = z - zi;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy), sz = fz * fz * (3 - 2 * fz);
  const l = (a: number, b: number, t: number) => a + (b - a) * t;
  const h = (dx: number, dy: number, dz: number) => hash(xi + dx, yi + dy, zi + dz, seed);
  return l(l(l(h(0, 0, 0), h(1, 0, 0), sx), l(h(0, 1, 0), h(1, 1, 0), sx), sy), l(l(h(0, 0, 1), h(1, 0, 1), sx), l(h(0, 1, 1), h(1, 1, 1), sx), sy), sz);
}

/** Blobs of the cumulus in unit-box coordinates: centre x, y, z and radius. A flat-ish base, billowing top. */
const BLOBS: [number, number, number, number][] = [
  [0.5, 0.36, 0.5, 0.42], [0.3, 0.32, 0.46, 0.3], [0.7, 0.33, 0.54, 0.32], [0.44, 0.55, 0.52, 0.3],
  [0.62, 0.6, 0.44, 0.24], [0.26, 0.5, 0.6, 0.2], [0.76, 0.5, 0.38, 0.2], [0.5, 0.74, 0.5, 0.2],
];

/** Density in [0, 1], x fastest, then y, then z; zero on the border (the grid reads zero outside). */
function cloudDensity(res: number): Float32Array {
  const d = new Float32Array(res * res * res);
  const sx = (MAX[0] - MIN[0]) / (MAX[1] - MIN[1]);
  for (let z = 0; z < res; z++) {
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        const u = (x + 0.5) / res, v = (y + 0.5) / res, w = (z + 0.5) / res;
        // The box is wider than tall: distances in the blobs' space are taken in metres / box height.
        let s = 0;
        for (const [cx, cy, cz, r] of BLOBS) {
          const q = (((u - cx) * sx) ** 2 + (v - cy) ** 2 + ((w - cz) * sx) ** 2) / (r * r);
          if (q < 1) s += (1 - q) * (1 - q);
        }
        if (s <= 0) continue;
        // Three octaves of value noise (period 1.6 m, 0.8 m, 0.4 m).
        const f = 3.2 * sx;
        const n = valueNoise(u * f, v * f, w * f, 1) * 0.55 + valueNoise(u * f * 2.1, v * f * 2.1, w * f * 2.1, 2) * 0.3 + valueNoise(u * f * 4.3, v * f * 4.3, w * f * 4.3, 3) * 0.15;
        // Edge: the blobs carve the shape, the noise eats into it, the remap gives the cloud a soft but definite edge.
        const base = Math.min(1, (v - 0.12) / 0.14);
        const edge = Math.min(1, Math.min(u, 1 - u, v, 1 - v, w, 1 - w) / 0.04);
        const t = Math.max(0, (s * (0.8 + 0.9 * n) - 0.2) / 0.5);
        d[(z * res + y) * res + x] = Math.min(1, t ** 1.15) * Math.max(0, base) * edge;
      }
    }
  }
  return d;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A 128 x 64 equirect sky: blue zenith to a pale horizon, a dim grey-green ground below it (row 0 = up). */
function skyMap(): { width: number; height: number; channels: number; data: Float32Array } {
  const width = 128, height = 64, data = new Float32Array(width * height * 3);
  for (let j = 0; j < height; j++) {
    const el = (0.5 - (j + 0.5) / height) * Math.PI; // +pi/2 at the top
    const up = Math.max(0, Math.sin(el));
    const sky: V3 = [0.75 - 0.6 * Math.sqrt(up), 0.85 - 0.45 * Math.sqrt(up), 1.0 - 0.15 * Math.sqrt(up)];
    const ground: V3 = [0.22, 0.24, 0.2];
    const k = el >= 0 ? 0 : Math.min(1, -el * 6);
    for (let i = 0; i < width; i++) for (let c = 0; c < 3; c++) data[(j * width + i) * 3 + c] = (sky[c]! * (1 - k) + ground[c]! * k) * 0.9;
  }
  return { width, height, channels: 3, data };
}

defineExample(async (ctx) => {
  const { device, pane, canvas } = ctx;
  const scale = Number(query.get('scale') ?? 0);
  if (scale > 0) ctx.setRenderScale(scale);
  const params = { sunElevation: 42, sunAzimuth: 60, extinction: 22, albedo: 0.99, phase: 'mie' as 'mie' | 'hg', g: 0.8, diameter: 20, maxDepth: 128 };

  const t0 = performance.now();
  const density = cloudDensity(RES);
  let densityMax = 0, core = 0;
  for (const v of density) { if (v > densityMax) densityMax = v; }
  // Core optical depth: the largest sum along y of one column (a vertical ray through the cloud), in units of sigma_t.
  for (let z = 0; z < RES; z += 4) for (let x = 0; x < RES; x += 4) { let s = 0; for (let y = 0; y < RES; y++) s += density[(z * RES + y) * RES + x]!; core = Math.max(core, s * ((MAX[1] - MIN[1]) / RES)); }
  const majorant = brickMajorants([RES, RES, RES], density);
  const buildMs = performance.now() - t0;
  ctx.stats.set('density', `${RES}³ built in ${buildMs.toFixed(0)} ms`);

  const orbit = new OrbitCamera(canvas, { position: [1.4, 0.6, 6.6], target: [0, 1.25, 0], up: [0, 1, 0], yFovDeg: 36 });
  const sky = skyMap();
  const input = (): SceneCopyInput => {
    const m = mesh(box([0, (MIN[1] + MAX[1]) / 2, 0], [(MAX[0] - MIN[0]) / 2, (MAX[1] - MIN[1]) / 2, (MAX[2] - MIN[2]) / 2], 0));
    return {
      meshes: [m],
      instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY, material: 0 }],
      // The boundary scatters nothing itself.
      materials: [{ baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 }],
      sun: sunFrom(params.sunElevation, params.sunAzimuth, 6),
      environment: { map: sky },
    };
  };

  let gpuScene: GpuScene | null = null;
  let renderer: ProgressiveRenderer | null = null;

  const volumes = (): PtIntegratorOptions['volumes'] => {
    const lobes = params.phase === 'mie' ? ptPhaseLobes({ type: 'mie', diameter: params.diameter }) : undefined;
    const medium: PtMedium = {
      sigmaT: [params.extinction, params.extinction, params.extinction],
      albedo: [params.albedo, params.albedo, params.albedo],
      g: lobes ? lobes.gA : params.g,
      ...(lobes ? { lobes } : {}),
      grid: { resolution: [RES, RES, RES], worldToIndex: gridWorldToIndex([RES, RES, RES], [MIN, MAX]), densityMax },
    };
    return { media: [medium], materials: [{ interior: 1, exterior: 0, boundary: true }], majorant: 'brick', runtimeParams: true };
  };

  const updateStats = () => ctx.stats.set('core optical depth', (params.extinction * core).toFixed(0));

  /** Rebuilds the kernel (extinction, albedo and phase are compiled in). */
  const rebuild = async () => {
    ctx.progress('Compiling…');
    try {
      gpuScene ??= await buildGpuScene(device, input(), { label: 'cloud' });
      const next = await ProgressiveRenderer.create(ctx, gpuScene, orbit.camera, { integrator: { volumes: volumes(), sunVisible: false }, volumeGrid: { resolution: [RES, RES, RES], density, majorant } });
      next.settings.maxDepth = renderer?.settings.maxDepth ?? params.maxDepth;
      const old = renderer;
      renderer = next;
      updateStats();
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('Could not build the cloud', error);
    }
  };
  /** Extinction, albedo, g and droplet size: rewrite the medium table in place and restart the accumulation (no recompile). */
  const updateMedia = () => {
    const t = performance.now();
    (renderer?.kernel as { setMedia(v: NonNullable<PtIntegratorOptions['volumes']>): void } | undefined)?.setMedia(volumes()!);
    renderer?.restart();
    updateStats();
    ctx.stats.set('medium update', `${(performance.now() - t).toFixed(2)} ms (no recompile)`);
  };
  /** The sun only changes the scene copy: rebuild the (cheap) scene and rebind it. */
  const moveSun = async () => {
    const old = gpuScene;
    gpuScene = await buildGpuScene(device, input(), { label: 'cloud' });
    renderer?.setScene(gpuScene);
    if (old) void device.queue.onSubmittedWorkDone().then(() => old.destroy());
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  const sun = pane.addFolder({ title: 'Sun' });
  sun.addBinding(params, 'sunElevation', { label: 'elevation (°)', min: 2, max: 88, step: 1 }).on('change', (e) => e.last && void moveSun());
  sun.addBinding(params, 'sunAzimuth', { label: 'azimuth (°)', min: -180, max: 180, step: 1 }).on('change', (e) => e.last && void moveSun());

  const medium = pane.addFolder({ title: 'Medium' });
  const live = () => updateMedia();
  medium.addBinding(params, 'extinction', { label: 'density scale (1/m)', min: 2, max: 80, step: 1 }).on('change', live);
  medium.addBinding(params, 'albedo', { label: 'albedo', min: 0.8, max: 1, step: 0.005 }).on('change', live);
  medium.addBinding(params, 'phase', { label: 'phase', options: { 'Mie droplets (Jendersie-d’Eon)': 'mie', 'Henyey-Greenstein': 'hg' } }).on('change', () => { diameter.hidden = params.phase !== 'mie'; g.hidden = params.phase !== 'hg'; void rebuild(); });
  const diameter = medium.addBinding(params, 'diameter', { label: 'droplet size (µm)', min: 5, max: 50, step: 1 });
  diameter.on('change', live);
  const g = medium.addBinding(params, 'g', { label: 'HG g', min: 0, max: 0.95, step: 0.01 });
  g.on('change', live);
  g.hidden = true;

  const path = pane.addFolder({ title: 'Path' });
  path.addBinding(params, 'maxDepth', { label: 'max scatters', min: 1, max: 256, step: 1 }).on('change', (e) => {
    if (!renderer) return;
    renderer.settings.maxDepth = params.maxDepth;
    if (e.last) renderer.restart();
  });

  await rebuild();
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
