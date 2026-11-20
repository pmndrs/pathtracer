/**
 * Many lights: 64 small coloured lamps over a floor with a few boxes, under a dim sky and a low sun.
 *
 * Next-event estimation picks ONE light per vertex and traces one shadow ray to it, so which light it picks decides the noise. The scene
 * copy's light table holds a pick probability per light (`lightSelection`, a scene-copy build option):
 *   pbrt (default)  pbrt-v4's rule: infinite lights (sun, sky) by power share against the rest, emitters by power within
 *   power           every light by its power
 *   kind            the same chance for each kind (emitters, sun, sky), by power within a kind
 *   capped 25%      by power, no light kind above a quarter
 * ReSTIR DI (experimental) instead draws several candidates per pixel, keeps one by its contribution, and reuses good picks from the
 * previous frame and from neighbours: far less noise per sample with many lights, at the cost of extra passes.
 */
import { buildGpuScene, type GpuScene, type LightSelection, type RtMaterialInput, type SceneCopyInput } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { sunFrom } from '../../shared/models.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

const STRATEGIES: Record<string, LightSelection> = { 'pbrt (default)': 'pbrt', power: 'power', kind: 'kind', 'capped 25%': { capped: 0.25 } };

/** An axis-aligned box as quads facing outward. */
function box(cx: number, cz: number, w: number, h: number, d: number) {
  const x0 = cx - w / 2, x1 = cx + w / 2, z0 = cz - d / 2, z1 = cz + d / 2;
  const faces = [
    [[x0, h, z1], [x1, h, z1], [x1, h, z0], [x0, h, z0]],
    [[x0, 0, z1], [x1, 0, z1], [x1, h, z1], [x0, h, z1]],
    [[x1, 0, z0], [x0, 0, z0], [x0, h, z0], [x1, h, z0]],
    [[x1, 0, z1], [x1, 0, z0], [x1, h, z0], [x1, h, z1]],
    [[x0, 0, z0], [x0, 0, z1], [x0, h, z1], [x0, h, z0]],
  ];
  return { positions: new Float32Array(faces.flat(2)), indices: faces.flatMap((_, f) => [0, 1, 2, 0, 2, 3].map((k) => k + f * 4)) };
}

function lampsScene(selection: LightSelection): SceneCopyInput {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const meshes = [
    { positions: new Float32Array([-6, 0, 6, 6, 0, 6, 6, 0, -6, -6, 0, -6]), indices: [0, 1, 2, 0, 2, 3] },
    box(-1.5, -0.5, 1, 1.2, 1), box(1.2, 0.8, 0.8, 0.6, 1.4), box(0.3, -2, 2.5, 0.4, 0.6),
  ];
  const materials: RtMaterialInput[] = [{ baseColor: [0.7, 0.7, 0.7], specularRoughness: 0.5 }];
  const instances = meshes.map((_, i) => ({ meshIndex: i, instanceIndex: i, objectToWorld: identity, material: 0 }));
  // 64 lamps: small downward-facing quads, each with its own colour (one material per lamp).
  let seed = 3;
  const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  for (let i = 0; i < 64; i++) {
    const x = (rand() - 0.5) * 9, z = (rand() - 0.5) * 9, y = 0.4 + rand() * 1.4, s = 0.08;
    meshes.push({ positions: new Float32Array([x - s, y, z - s, x + s, y, z - s, x + s, y, z + s, x - s, y, z + s]), indices: [0, 1, 2, 0, 2, 3] });
    const h = rand() * 6;
    const rgb = [0, 2, 4].map((k) => Math.max(0, Math.min(1, Math.abs(((h + k) % 6) - 3) - 1))) as [number, number, number];
    materials.push({ baseColor: [0, 0, 0], specularWeight: 0, emissionColor: rgb.map((c) => 40 * (0.15 + c)) as [number, number, number] });
    instances.push({ meshIndex: meshes.length - 1, instanceIndex: meshes.length - 1, objectToWorld: identity, material: materials.length - 1 });
  }
  return { meshes, instances, materials, sun: sunFrom(12, -60, 0.6), environment: { radiance: [0.02, 0.025, 0.04] }, options: { lightSelection: selection } };
}

defineExample(async (ctx) => {
  const { device, pane, canvas, stats } = ctx;
  const params = { strategy: 'pbrt (default)', restir: new URLSearchParams(location.search).has('restir'), maxDepth: 4 };
  const camera = { position: [6.5, 4.5, 7] as [number, number, number], target: [0, 0.3, 0] as [number, number, number], up: [0, 1, 0] as [number, number, number], yFovDeg: 45 };
  const orbit = new OrbitCamera(canvas, camera);
  let scene: GpuScene = await buildGpuScene(device, lampsScene(STRATEGIES[params.strategy]!), { label: 'lamps' });
  let renderer: ProgressiveRenderer | null = null;
  stats.set('lights', '64 lamps, sun, sky');

  const rebuild = async () => {
    ctx.progress('Compiling…');
    try {
      const next = await ProgressiveRenderer.create(ctx, scene, orbit.camera, { integrator: { restir: params.restir ? {} : null } });
      next.settings.maxDepth = params.maxDepth;
      next.settings.samplesPerFrame = 1;
      const old = renderer;
      renderer = next;
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
    } catch (error) {
      ctx.error('This combination does not compile', error);
    }
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  const sampling = pane.addFolder({ title: 'Light sampling' });
  sampling.addBinding(params, 'strategy', { label: 'light selection', options: Object.fromEntries(Object.keys(STRATEGIES).map((k) => [k, k])) }).on('change', async () => {
    // The selection lives in the scene copy's light table: rebuild the scene, keep the kernel.
    const old = scene;
    scene = await buildGpuScene(device, lampsScene(STRATEGIES[params.strategy]!), { label: 'lamps' });
    renderer?.setScene(scene);
    await device.queue.onSubmittedWorkDone();
    old.destroy();
  });
  sampling.addBinding(params, 'restir', { label: 'ReSTIR DI (experimental)' }).on('change', () => void rebuild());
  pane.addFolder({ title: 'Render' }).addBinding(params, 'maxDepth', { label: 'max bounces', min: 1, max: 8, step: 1 }).on('change', () => {
    if (!renderer) return;
    renderer.settings.maxDepth = params.maxDepth;
    renderer.restart();
  });

  await rebuild();
  ctx.loop((encoder) => renderer?.frame(encoder) ?? false);
});
