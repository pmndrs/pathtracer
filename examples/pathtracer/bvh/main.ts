/**
 * BVH formats: the same scene built into each of @pmndrs/bvh's node formats, path traced by a kernel compiled for that format.
 *
 *   quantized (default)  binary tree, child boxes quantised to 8 bits against the parent: small nodes
 *   f32                  binary tree, full-precision boxes
 *   bvh4                 4-wide tree (experimental)
 *   cwbvh8               8-wide compressed tree (experimental, the layout GPU ray tracers favour)
 *
 * `buildGpuScene(device, input, { nodeFormat })` builds the tree; `ptSceneOptions(scene)` passes `scene.format` to the kernel, which
 * compiles the matching walk. Images agree between formats (up to which of two coincident triangles a ray reports); speed and memory differ
 * by GPU. "Benchmark" times 64 samples per pixel on this machine: compare formats on one GPU, never across machines.
 */
import type { NodeFormat } from '@pmndrs/bvh';
import { buildGpuScene, type GpuScene } from '@pmndrs/raytracer';
import { defineExample } from '../../shared/example.ts';
import { loadScene, SCENES, type LoadedScene } from '../../shared/models.ts';
import { OrbitCamera } from '../../shared/orbit.ts';
import { ProgressiveRenderer } from '../../shared/progressive.ts';

const FORMATS: Record<string, NodeFormat> = { 'Quantized (default)': 'quantized', f32: 'f32', 'BVH4 (experimental)': 'bvh4', 'CWBVH8 (experimental)': 'cwbvh8' };

defineExample(async (ctx) => {
  const { device, pane, canvas, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const params = {
    scene: SCENES[search.get('scene') ?? 'Toy Car'] ?? SCENES['Toy Car']!,
    format: (search.get('format') ?? 'quantized') as NodeFormat,
    results: '',
  };
  const orbit = new OrbitCamera(canvas, { position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 40 });
  let loaded: LoadedScene | null = null;
  let scene: GpuScene | null = null;
  let renderer: ProgressiveRenderer | null = null;
  let busy = false;
  const timings = new Map<string, string>();

  const build = async () => {
    if (!loaded) return;
    busy = true;
    try {
      ctx.progress(`Building the ${params.format} tree…`);
      const t = performance.now();
      const next = await buildGpuScene(device, loaded.input, { label: params.format, nodeFormat: params.format });
      const buildMs = performance.now() - t;
      ctx.progress('Compiling…');
      const r = await ProgressiveRenderer.create(ctx, next, orbit.camera);
      r.settings.maxDepth = 6;
      const old = renderer, oldScene = scene;
      renderer = r;
      scene = next;
      stats.set('tree', `${params.format}, built in ${buildMs.toFixed(0)} ms`);
      stats.set('memory', `${(next.trace.byteLength / 1048576).toFixed(2)} MB on the GPU`);
      stats.set('triangles', loaded.input.instances.reduce((n, i) => { const m = loaded!.input.meshes[i.meshIndex]!; return n + (m.indices ? m.indices.length : m.positions.length / 3) / 3; }, 0).toLocaleString());
      ctx.progress(null);
      await device.queue.onSubmittedWorkDone();
      old?.destroy();
      oldScene?.destroy();
    } catch (error) {
      ctx.error(`Could not build the ${params.format} scene`, error);
    } finally {
      busy = false;
    }
  };

  const load = async () => {
    const name = Object.keys(SCENES).find((k) => SCENES[k] === params.scene) ?? 'scene';
    ctx.progress(`Loading ${name}…`);
    try {
      loaded = await loadScene(params.scene, name);
      orbit.set(loaded.camera, false, loaded.bounds);
      timings.clear();
      params.results = '';
      pane.refresh();
      await build();
    } catch (error) {
      ctx.error(`Could not load ${name}`, error);
    }
  };
  orbit.onChange = () => renderer?.setCamera(orbit.camera);

  // 64 samples per pixel from a fresh image, submitted in batches and timed on the wall clock (the queue drained before and after).
  const benchmark = async () => {
    if (!renderer || busy) return;
    busy = true;
    ctx.progress('Timing 64 samples per pixel…');
    try {
      const pt = renderer.kernel;
      await device.queue.onSubmittedWorkDone();
      let e = device.createCommandEncoder();
      pt.reset(e);
      pt.encode(e, 1); // warm-up
      device.queue.submit([e.finish()]);
      await device.queue.onSubmittedWorkDone();
      const t = performance.now();
      for (let done = 0; done < 64; done += 8) {
        e = device.createCommandEncoder();
        pt.encode(e, 8);
        device.queue.submit([e.finish()]);
        await device.queue.onSubmittedWorkDone();
      }
      const ms = (performance.now() - t) / 64;
      timings.set(params.format, `${ms.toFixed(2)} ms/spp`);
      params.results = [...timings].map(([f, v]) => `${f}: ${v}`).join('\n');
      pane.refresh();
      renderer.restart();
    } finally {
      ctx.progress(null);
      busy = false;
    }
  };

  const f = pane.addFolder({ title: 'BVH' });
  f.addBinding(params, 'scene', { options: Object.fromEntries(Object.entries(SCENES).filter(([, v]) => v !== 'cornell').concat([['Cornell box', 'cornell']])) }).on('change', () => void load());
  f.addBinding(params, 'format', { label: 'node format', options: FORMATS }).on('change', () => void build());
  f.addButton({ title: 'Benchmark (64 spp)' }).on('click', () => void benchmark());
  f.addBinding(params, 'results', { label: 'this GPU', readonly: true, multiline: true, rows: 4 });

  await load();
  ctx.loop((encoder) => (busy ? false : (renderer?.frame(encoder) ?? false)));
});
