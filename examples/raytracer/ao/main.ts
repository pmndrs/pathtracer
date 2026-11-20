/**
 * Ambient occlusion with `rtVisible`: the cheapest query of the trace core. For each pixel's first hit, cosine-distributed rays over the
 * hemisphere ask only "is anything within `radius`?" (an any-hit walk, which stops at the first blocker). The fraction that escape is the
 * pixel's ambient occlusion; frames accumulate into a converged image.
 */
import type { GpuScene } from '@pmndrs/raytracer';
import { Accumulation, exposeForChecks } from '../../shared/accumulation.ts';
import { defineExample } from '../../shared/example.ts';
import { scenePicker } from '../../shared/scene-picker.ts';
import { ioEntries, TraceKernel } from '../../shared/trace-kernel.ts';
import { VIEW_BYTES, VIEW_WGSL, writeView } from '../../shared/view.ts';

const WGSL = /* wgsl */ `
${VIEW_WGSL}
@group(2) @binding(0) var<uniform> view: ExView;
@group(2) @binding(1) var<storage, read_write> accum: array<vec4f>;

// A cosine-distributed direction around n (Malley: a uniform disc point lifted to the hemisphere).
fn exCosineDirection(n: vec3f, u: vec2f) -> vec3f {
  let r = sqrt(u.x);
  let phi = 6.283185307 * u.y;
  let t = normalize(select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9) - n * dot(n, select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9)));
  let b = cross(n, t);
  return normalize(r * cos(phi) * t + r * sin(phi) * b + sqrt(max(0.0, 1.0 - u.x)) * n);
}

// A ray origin just off the surface that does not hit it again (Waechter and Binder, Ray Tracing Gems ch. 6): integer ulp steps far from
// the origin, a float step near it. rtShade's module has the same as rtOffsetOrigin; this kernel does not include rtShade.
fn exOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {
  let ofs = vec3i(256.0 * n);
  let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
  return select(pi, p + (1.0 / 65536.0) * n, abs(p) < vec3f(1.0 / 32.0));
}

fn exSide(n: vec3f, w: vec3f) -> vec3f {
  return select(-n, n, dot(n, w) > 0.0);
}

@compute @workgroup_size(8, 8)
fn ao(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= view.width || id.y >= view.height) { return; }
  var seed = exHash(id.x + id.y * view.width) ^ exHash(view.frame * 9781u + 13u);
  let ray = exRay(view, id.x, id.y, vec2f(exRand(&seed), exRand(&seed)));
  let hit = rtClosest(ray, RT_MASK_ALL);
  var open = 1.0;  // a miss sees the whole sky
  if (rtIsHit(hit)) {
    let s = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL);
    // The side of the surface the camera sees, and an origin just off it (no self-hits).
    let ng = exSide(s.geometricNormal, -ray.direction);
    let n = exSide(s.shadingNormal, -ray.direction);
    let origin = exOffsetOrigin(s.position, ng);
    let rays = max(view.mode, 1u);
    var escaped = 0.0;
    for (var k = 0u; k < rays; k++) {
      var d = exCosineDirection(n, vec2f(exRand(&seed), exRand(&seed)));
      if (dot(d, ng) <= 0.0) { d = reflect(d, ng); }
      // radius in origin.w: anything nearer than that occludes.
      if (rtVisible(RtRay(origin, 0.0, d, view.origin.w), RT_MASK_ALL)) { escaped += 1.0; }
    }
    open = escaped / f32(rays);
  }
  let i = id.y * view.width + id.x;
  accum[i] = accum[i] + vec4f(vec3f(open), 1.0);
}
`;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const params = { radius: 0.5, raysPerFrame: 2 };
  const accumulation = new Accumulation(ctx);
  accumulation.maxFrames = 512;
  accumulation.view.toneMapping = 'clamp';
  exposeForChecks(accumulation);
  const viewBuffer = device.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let kernel: TraceKernel | null = null;
  let group: GPUBindGroup | null = null;
  let bound: GPUBuffer | null = null;

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  const picker = await scenePicker(ctx, sceneFolder, 'Sheen Chair', async (scene: GpuScene, loaded) => {
    kernel = await TraceKernel.create(device, caps, scene, { label: 'ao', wgsl: WGSL, entryPoints: ['ao'], io: ioEntries(['uniform', 'write']), module: { scene: { slot: { group: 1, firstBinding: 0 }, layout: scene.copy.layout, textures: scene.copy.textureLayout, packing: scene.packing, materials: false, lights: false } } });
    bound = null;
    // A radius of a tenth of the scene's size is a good start.
    params.radius = Number((Math.hypot(...loaded.bounds.max.map((v, i) => v - loaded.bounds.min[i]!)) * 0.1).toPrecision(2));
    pane.refresh();
    stats.set('kernel', `${(kernel.bytes / 1024).toFixed(1)} KB`);
    accumulation.reset();
  }, () => accumulation.reset());

  const aoFolder = pane.addFolder({ title: 'Ambient occlusion' });
  aoFolder.addBinding(params, 'radius', { min: 0.001, max: 10, step: 0.001 }).on('change', () => accumulation.reset());
  aoFolder.addBinding(params, 'raysPerFrame', { label: 'rays / frame', min: 1, max: 16, step: 1 }).on('change', () => accumulation.reset());

  ctx.loop((encoder) => {
    if (!kernel) return false;
    const k = kernel;
    return accumulation.frameWith(encoder, (enc, frame) => {
      if (bound !== accumulation.buffer) {
        group = device.createBindGroup({ layout: k.ioLayout, entries: [{ binding: 0, resource: { buffer: viewBuffer } }, { binding: 1, resource: { buffer: accumulation.buffer! } }] });
        bound = accumulation.buffer;
      }
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, params.raysPerFrame, [params.radius, 0]);
      k.dispatch(enc, 'ao', group!, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
    });
  });
});
