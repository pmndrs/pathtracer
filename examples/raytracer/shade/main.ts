/**
 * One-bounce lighting with `rtShade`: what a reflection or GI effect needs at the point its ray hit. `rtShade(hit, ray, mode, u, mask)`
 * returns emission plus direct light from one light sample (with one shadow ray), and the environment for a miss. It is not a path tracer:
 * there is no indirect light unless you provide it (`shade: { cache: 'external' }` and your own `rtShadeIndirect`).
 *
 * Modes: FLAT (no rays: emission, plus the base colour here), LAMBERT (diffuse only) and FULL (diffuse + GGX).
 */
import type { GpuScene } from '@pmndrs/raytracer';
import { Accumulation, exposeForChecks } from '../../shared/accumulation.ts';
import { defineExample } from '../../shared/example.ts';
import { scenePicker } from '../../shared/scene-picker.ts';
import { ioEntries, TraceKernel } from '../../shared/trace-kernel.ts';
import { VIEW_BYTES, VIEW_WGSL, writeView } from '../../shared/view.ts';

const MODES = { 'Full (diffuse + GGX)': 2, 'Lambert (diffuse only)': 1, 'Flat (albedo, no rays)': 0 };

const WGSL = /* wgsl */ `
${VIEW_WGSL}
@group(2) @binding(0) var<uniform> view: ExView;
@group(2) @binding(1) var<storage, read_write> accum: array<vec4f>;

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= view.width || id.y >= view.height) { return; }
  var seed = exHash(id.x + id.y * view.width) ^ exHash(view.frame * 9781u + 29u);
  let ray = exRay(view, id.x, id.y, vec2f(exRand(&seed), exRand(&seed)));
  let hit = rtClosest(ray, RT_MASK_ALL);
  let mode = select(select(RT_SHADE_FLAT, RT_SHADE_LAMBERT, view.mode == 1u), RT_SHADE_FULL, view.mode == 2u);
  // u = (which light, where on it): one light sample and one shadow ray per call.
  let s = rtShade(hit, ray, mode, vec3f(exRand(&seed), exRand(&seed), exRand(&seed)), RT_MASK_ALL);
  let i = id.y * view.width + id.x;
  accum[i] = accum[i] + vec4f(rtShadeTotal(s), 1.0);
}
`;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const params = { mode: 2 };
  const accumulation = new Accumulation(ctx);
  accumulation.maxFrames = 1024;
  exposeForChecks(accumulation);
  const viewBuffer = device.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let kernel: TraceKernel | null = null;
  let group: GPUBindGroup | null = null;
  let bound: GPUBuffer | null = null;

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  const picker = await scenePicker(ctx, sceneFolder, 'Toy Car', async (scene: GpuScene) => {
    // The module with rtShade is about 50 KB of WGSL: minify keeps it in the size class every backend compiles quickly.
    kernel = await TraceKernel.create(device, caps, scene, { label: 'shade', wgsl: WGSL, entryPoints: ['shade'], io: ioEntries(['uniform', 'write']), module: { shade: { flat: 'albedo' }, minify: true } });
    bound = null;
    stats.set('kernel', `${(kernel.bytes / 1024).toFixed(1)} KB`);
    accumulation.reset();
  }, () => accumulation.reset());

  const shading = pane.addFolder({ title: 'Shading' });
  shading.addBinding(params, 'mode', { options: MODES }).on('change', () => accumulation.reset());
  const view = pane.addFolder({ title: 'View' });
  view.addBinding(accumulation.view, 'exposure', { min: 0, max: 4, step: 0.01 });
  view.addBinding(accumulation.view, 'toneMapping', { label: 'tone mapping', options: { ACES: 'aces', Reinhard: 'reinhard', 'None (clamp)': 'clamp' } });

  ctx.loop((encoder) => {
    if (!kernel) return false;
    const k = kernel;
    return accumulation.frameWith(encoder, (enc, frame) => {
      if (bound !== accumulation.buffer) {
        group = device.createBindGroup({ layout: k.ioLayout, entries: [{ binding: 0, resource: { buffer: viewBuffer } }, { binding: 1, resource: { buffer: accumulation.buffer! } }] });
        bound = accumulation.buffer;
      }
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, params.mode);
      k.dispatch(enc, 'shade', group!, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
    });
  });
});
