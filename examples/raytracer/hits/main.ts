/**
 * Hit inspector: `@pmndrs/raytracer` inside your own compute shader.
 *
 * `rtModuleWgsl` gives the shader `rtClosest` (the first hit along a ray) and `rtSurface` (what is at that hit: position, normals, uv,
 * material id). This kernel shoots one jittered camera ray per pixel per frame and writes one property of the hit as a colour, so the
 * image converges to an anti-aliased view of it. No path tracer is involved.
 */
import type { GpuScene } from '@pmndrs/raytracer';
import { Accumulation, exposeForChecks } from '../../shared/accumulation.ts';
import { defineExample } from '../../shared/example.ts';
import { scenePicker } from '../../shared/scene-picker.ts';
import { ioEntries, TraceKernel } from '../../shared/trace-kernel.ts';
import { VIEW_BYTES, VIEW_WGSL, writeView } from '../../shared/view.ts';

const MODES = { 'Shading normal': 0, 'Geometric normal': 1, 'Front / back face': 2, Distance: 3, 'Texture uv': 4, Barycentrics: 5, 'Material id': 6, 'Instance id': 7, 'Base colour': 8 };

/** `textured`: the scene has textures, so the base colour can apply them (`rtApplyMaterialTextures` exists only then). */
const kernelWgsl = (textured: boolean) => /* wgsl */ `
${VIEW_WGSL}
@group(2) @binding(0) var<uniform> view: ExView;
@group(2) @binding(1) var<storage, read_write> accum: array<vec4f>;

fn idColor(i: u32) -> vec3f {
  let h = exHash(i + 1u);
  return vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0;
}

@compute @workgroup_size(8, 8)
fn inspect(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= view.width || id.y >= view.height) { return; }
  var seed = exHash(id.x + id.y * view.width) ^ exHash(view.frame * 9781u + 7u);
  let ray = exRay(view, id.x, id.y, vec2f(exRand(&seed), exRand(&seed)));
  let hit = rtClosest(ray, RT_MASK_ALL);
  var c = vec3f(0.03);
  if (rtIsHit(hit)) {
    let s = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_MATERIAL);
    switch view.mode {
      case 0u: { c = s.shadingNormal * 0.5 + 0.5; }
      case 1u: { c = s.geometricNormal * 0.5 + 0.5; }
      case 2u: { c = select(vec3f(0.9, 0.25, 0.2), vec3f(0.3, 0.8, 0.35), s.frontFace); }
      case 3u: { c = vec3f(fract(hit.t / max(view.origin.w, 1e-6))); }
      case 4u: { c = vec3f(fract(s.uv), 0.0); }
      case 5u: { c = vec3f(1.0 - hit.bary.x - hit.bary.y, hit.bary.x, hit.bary.y); }
      case 6u: { c = idColor(s.material); }
      case 7u: { c = idColor(hit.instance); }
      default: {
        var m = rtMaterial(s.material);
        ${textured ? 'm = rtApplyMaterialTextures(m, s.uv, 0.0);' : ''}
        c = m.baseColor + m.emissionColor;
      }
    }
  }
  let i = id.y * view.width + id.x;
  accum[i] = accum[i] + vec4f(c, 1.0);
}
`;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const params = { mode: Number(new URLSearchParams(location.search).get('mode') ?? 0), distanceBand: 1 };
  const accumulation = new Accumulation(ctx);
  accumulation.maxFrames = 256;
  accumulation.view.toneMapping = 'clamp';
  exposeForChecks(accumulation);
  const viewBuffer = device.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let kernel: TraceKernel | null = null;
  let group: GPUBindGroup | null = null;
  let bound: GPUBuffer | null = null;

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  const picker = await scenePicker(ctx, sceneFolder, 'Damaged Helmet', async (scene: GpuScene) => {
    kernel = await TraceKernel.create(device, caps, scene, { label: 'hits', wgsl: kernelWgsl(!!scene.copy.textureLayout), entryPoints: ['inspect'], io: ioEntries(['uniform', 'write']) });
    bound = null;
    stats.set('kernel', `${(kernel.bytes / 1024).toFixed(1)} KB`);
    accumulation.reset();
  }, () => accumulation.reset());

  const show = pane.addFolder({ title: 'Show' });
  show.addBinding(params, 'mode', { label: 'property', options: MODES }).on('change', () => accumulation.reset());
  show.addBinding(params, 'distanceBand', { label: 'distance band (m)', min: 0.05, max: 10, step: 0.05 }).on('change', () => accumulation.reset());

  ctx.loop((encoder) => {
    if (!kernel) return false;
    const k = kernel;
    return accumulation.frameWith(encoder, (enc, frame) => {
      if (bound !== accumulation.buffer) {
        group = device.createBindGroup({ layout: k.ioLayout, entries: [{ binding: 0, resource: { buffer: viewBuffer } }, { binding: 1, resource: { buffer: accumulation.buffer! } }] });
        bound = accumulation.buffer;
      }
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, params.mode, [params.distanceBand, 0]);
      k.dispatch(enc, 'inspect', group!, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
    });
  });
});
