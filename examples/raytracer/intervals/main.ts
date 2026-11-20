/**
 * Interval queries (experimental): `rtClosestInterval(origin, direction, t0, t1, mask)` finds the closest hit with t in (t0, t1], exact at
 * both ends in f32, so consecutive intervals split a ray with no gap and no overlap. Radiance cascades rely on that: cascade n owns the
 * distances (t0 * growth^(n-1), t0 * growth^n] (`rtCascadeStart` / `rtCascadeEnd`) and every hit belongs to exactly one cascade.
 *
 * From each pixel's first hit, cosine rays go out over the hemisphere:
 *   cascade colours   each ray walks the cascades in order with rtClosestInterval and takes the colour of the one its first hit is in
 *                     (white: escaped to the sky). The colours add up to the full occlusion, cascade by cascade.
 *   one cascade       what cascade n alone sees: the fraction of rays with something inside its interval
 *   camera ray        the cascade that holds the camera ray's own hit distance from the camera, as bands
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

fn exCosineDirection(n: vec3f, u: vec2f) -> vec3f {
  let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(a - n * dot(n, a));
  let b = cross(n, t);
  let r = sqrt(u.x);
  let phi = 6.283185307 * u.y;
  return normalize(r * cos(phi) * t + r * sin(phi) * b + sqrt(max(0.0, 1.0 - u.x)) * n);
}

// The rtShade module has these as rtOffsetOrigin / rtSideNormal; this kernel does not include rtShade.
fn exOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {
  let ofs = vec3i(256.0 * n);
  let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
  return select(pi, p + (1.0 / 65536.0) * n, abs(p) < vec3f(1.0 / 32.0));
}
fn exSide(n: vec3f, w: vec3f) -> vec3f { return select(-n, n, dot(n, w) > 0.0); }

// One colour per cascade (up to 8).
fn exCascadeColour(n: u32) -> vec3f {
  switch n {
    case 0u: { return vec3f(0.95, 0.25, 0.2); }
    case 1u: { return vec3f(0.95, 0.65, 0.15); }
    case 2u: { return vec3f(0.35, 0.8, 0.3); }
    case 3u: { return vec3f(0.2, 0.6, 0.95); }
    case 4u: { return vec3f(0.6, 0.35, 0.9); }
    case 5u: { return vec3f(0.9, 0.35, 0.7); }
    case 6u: { return vec3f(0.3, 0.85, 0.85); }
    default: { return vec3f(0.6, 0.6, 0.6); }
  }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= view.width || id.y >= view.height) { return; }
  let mode = view.mode & 15u;
  let chosen = (view.mode >> 4u) & 15u;
  let count = (view.mode >> 8u) & 15u;
  let rays = max((view.mode >> 12u) & 255u, 1u);
  let t0 = view.origin.w;
  let growth = view.right.w;
  var seed = exHash(id.x + id.y * view.width) ^ exHash(view.frame * 9781u + 13u);
  let ray = exRay(view, id.x, id.y, vec2f(exRand(&seed), exRand(&seed)));
  let hit = rtClosest(ray, RT_MASK_ALL);
  var c = vec3f(1.0);
  if (!rtIsHit(hit)) {
    c = vec3f(0.85, 0.9, 1.0);
  } else if (mode == 2u) {
    // The camera ray's hit distance, by cascade: bands that grow by \`growth\` each.
    let n = rtCascadeIndex(hit.t, t0, growth, count);
    c = select(vec3f(0.15), exCascadeColour(n), n < count);
  } else {
    let s = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL);
    let ng = exSide(s.geometricNormal, -ray.direction);
    let nrm = exSide(s.shadingNormal, -ray.direction);
    let origin = exOffsetOrigin(s.position, ng);
    var sum = vec3f(0.0);
    for (var k = 0u; k < rays; k++) {
      var d = exCosineDirection(nrm, vec2f(exRand(&seed), exRand(&seed)));
      if (dot(d, ng) <= 0.0) { d = reflect(d, ng); }
      if (mode == 1u) {
        // One cascade on its own: is anything inside (start, end]?
        let h = rtClosestInterval(origin, d, rtCascadeStart(chosen, t0, growth), rtCascadeEnd(chosen, t0, growth), RT_MASK_ALL);
        sum += select(vec3f(1.0), vec3f(0.0), rtIsHit(h));
      } else {
        // Walk the cascades in order: the first interval with a hit owns this ray. Beyond the last one, the ray escapes.
        var colour = vec3f(1.0);
        for (var n = 0u; n < count; n++) {
          let h = rtClosestInterval(origin, d, rtCascadeStart(n, t0, growth), rtCascadeEnd(n, t0, growth), RT_MASK_ALL);
          if (rtIsHit(h)) { colour = exCascadeColour(n); break; }
        }
        sum += colour;
      }
    }
    c = sum / f32(rays);
  }
  let i = id.y * view.width + id.x;
  accum[i] = accum[i] + vec4f(c, 1.0);
}
`;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const params = { mode: Number(search.get('mode') ?? 0), first: 0.05, growth: 4, count: Number(search.get('count') ?? 4), cascade: Number(search.get('cascade') ?? 1), raysPerFrame: 2 };
  const accumulation = new Accumulation(ctx);
  accumulation.maxFrames = 512;
  accumulation.view.toneMapping = 'clamp';
  exposeForChecks(accumulation);
  const viewBuffer = device.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let kernel: TraceKernel | null = null;
  let group: GPUBindGroup | null = null;
  let bound: GPUBuffer | null = null;
  const legend = () => {
    const ends = Array.from({ length: params.count }, (_, n) => params.first * params.growth ** n);
    stats.set('cascades', ends.map((e, n) => `${n}: ≤ ${e.toPrecision(2)}`).join(' · '));
  };

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  const picker = await scenePicker(ctx, sceneFolder, 'Sheen Chair', async (scene: GpuScene, loaded) => {
    kernel = await TraceKernel.create(device, caps, scene, {
      label: 'intervals', wgsl: WGSL, entryPoints: ['main'], io: ioEntries(['uniform', 'write']),
      module: { intervals: true, scene: { slot: { group: 1, firstBinding: 0 }, layout: scene.copy.layout, packing: scene.packing, materials: false, lights: false } },
    });
    bound = null;
    // The first cascade ends at a hundredth of the scene's size: four cascades growing by 4 then reach about two thirds of it.
    params.first = Number(search.get('first') ?? (Math.hypot(...loaded.bounds.max.map((v, i) => v - loaded.bounds.min[i]!)) * 0.01).toPrecision(2));
    pane.refresh();
    legend();
    stats.set('kernel', `${(kernel.bytes / 1024).toFixed(1)} KB`);
    accumulation.reset();
  }, () => accumulation.reset());

  const f = pane.addFolder({ title: 'Cascades (experimental)' });
  const changed = () => {
    legend();
    accumulation.reset();
  };
  f.addBinding(params, 'mode', { label: 'show', options: { 'Which cascade each AO ray hits in': 0, 'One cascade on its own': 1, 'Camera hit distance': 2 } }).on('change', changed);
  f.addBinding(params, 'first', { label: 'first cascade end', min: 0.001, max: 5, step: 0.001 }).on('change', changed);
  f.addBinding(params, 'growth', { min: 1.5, max: 8, step: 0.5 }).on('change', changed);
  f.addBinding(params, 'count', { label: 'cascades', min: 1, max: 8, step: 1 }).on('change', changed);
  f.addBinding(params, 'cascade', { label: 'cascade (one)', min: 0, max: 7, step: 1 }).on('change', changed);
  f.addBinding(params, 'raysPerFrame', { label: 'rays / frame', min: 1, max: 16, step: 1 }).on('change', changed);

  ctx.loop((encoder) => {
    if (!kernel) return false;
    const k = kernel;
    return accumulation.frameWith(encoder, (enc, frame) => {
      if (bound !== accumulation.buffer) {
        group = device.createBindGroup({ layout: k.ioLayout, entries: [{ binding: 0, resource: { buffer: viewBuffer } }, { binding: 1, resource: { buffer: accumulation.buffer! } }] });
        bound = accumulation.buffer;
      }
      const mode = params.mode | (Math.min(params.cascade, params.count - 1) << 4) | (params.count << 8) | (params.raysPerFrame << 12);
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, mode, [params.first, params.growth]);
      k.dispatch(enc, 'main', group!, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
    });
  });
});
