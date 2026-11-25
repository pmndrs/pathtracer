/**
 * Ambient occlusion with `rtVisible`: the cheapest query of the trace core. For each pixel's first hit, cosine-distributed rays over the
 * hemisphere ask only "is anything within `radius`?" (an any-hit walk, which stops at the first blocker). The fraction that escape is the
 * pixel's ambient occlusion; frames accumulate into a converged image.
 *
 * The **Query** control swaps the occlusion test for `rtClosest` (closest hit) over the same rays: the image is the same, but the walk must
 * keep going until it has proved which hit is nearest, where `rtVisible` may stop at the first blocker it meets. Compare the frame times.
 * `?query=closest` in the URL starts on it.
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
    // view.mode: bits 0-7 the rays per frame, bit 8 the query (0 = rtVisible, 1 = rtClosest). A uniform, so the branch below is the same
    // for every thread and both arms share one compiled kernel.
    let rays = max(view.mode & 0xffu, 1u);
    let closest = (view.mode & 0x100u) != 0u;
    var escaped = 0.0;
    for (var k = 0u; k < rays; k++) {
      var d = exCosineDirection(n, vec2f(exRand(&seed), exRand(&seed)));
      if (dot(d, ng) <= 0.0) { d = reflect(d, ng); }
      // radius in origin.w: anything nearer than that occludes.
      let probe = RtRay(origin, 0.0, d, view.origin.w);
      var escapes = false;
      if (closest) {
        escapes = !rtIsHit(rtClosest(probe, RT_MASK_ALL));  // the nearest hit, found and then thrown away
      } else {
        escapes = rtVisible(probe, RT_MASK_ALL);  // any hit: stops at the first blocker
      }
      if (escapes) { escaped += 1.0; }
    }
    open = escaped / f32(rays);
  }
  let i = id.y * view.width + id.x;
  accum[i] = accum[i] + vec4f(vec3f(open), 1.0);
}
`;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const QUERIES = { 'rtVisible (any hit)': 'visible', 'rtClosest (closest hit)': 'closest' } as const;
  const params = { radius: 0.5, raysPerFrame: 2, query: (new URLSearchParams(location.search).get('query') === 'closest' ? 'closest' : 'visible') as 'visible' | 'closest' };
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
    kernel = await TraceKernel.create(device, caps, scene, { label: 'ao', wgsl: WGSL, entryPoints: ['ao'], io: ioEntries(['uniform', 'write']), module: { scene: { slot: { group: 1, firstBinding: 0 }, layout: scene.copy.layout, packing: scene.packing, materials: false, lights: false } } });
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
  // The baseline: the same rays through the closest-hit walk. Same image; watch the frame time.
  aoFolder.addBinding(params, 'query', { label: 'Query', options: QUERIES }).on('change', () => accumulation.reset());

  ctx.loop((encoder) => {
    if (!kernel) return false;
    const k = kernel;
    return accumulation.frameWith(encoder, (enc, frame) => {
      if (bound !== accumulation.buffer) {
        group = device.createBindGroup({ layout: k.ioLayout, entries: [{ binding: 0, resource: { buffer: viewBuffer } }, { binding: 1, resource: { buffer: accumulation.buffer! } }] });
        bound = accumulation.buffer;
      }
      stats.set('query', params.query === 'closest' ? 'rtClosest (closest hit)' : 'rtVisible (any hit)');
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, params.raysPerFrame | (params.query === 'closest' ? 0x100 : 0), [params.radius, 0]);
      k.dispatch(enc, 'ao', group!, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
    });
  });
});
