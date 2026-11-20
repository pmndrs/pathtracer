/**
 * Batch tracing: rays in a GPU buffer in, hits in a GPU buffer out, with no WGSL of the trace core in your own shaders' walk.
 *
 * One frame is five passes, all on the GPU and none read back:
 *   1. `primary`  (ours)            camera rays -> `rays`          (32 B `Ray` records: origin, tMin, direction, tMax)
 *   2. `closest`  (RtBatchTracer)   `rays` -> `hits`               (40 B hit records: t, instance, triangle, front face, barycentrics, material, normal)
 *   3. `shadow`   (ours)            `hits` -> `shadowRays`         (towards the sun, from just off each hit)
 *   4. `visible`  (RtBatchTracer)   `shadowRays` -> `visibility`   (one u32 per ray, 1 = nothing in the way)
 *   5. `compose`  (ours)            base colour x (sun x visibility + sky), accumulated
 * Our three kernels still bind the scene copy for the material colours (`rtMaterial`, `rtSurface` for textures), not the walk.
 */
import { HIT_BYTES, RAY_BYTES, RtBatchTracer, type GpuScene } from '@pmndrs/raytracer';
import { Accumulation, exposeForChecks } from '../../shared/accumulation.ts';
import { defineExample } from '../../shared/example.ts';
import { scenePicker } from '../../shared/scene-picker.ts';
import { ioEntries, TraceKernel } from '../../shared/trace-kernel.ts';
import { VIEW_BYTES, VIEW_WGSL, writeView } from '../../shared/view.ts';

const wgsl = (textured: boolean) => /* wgsl */ `
${VIEW_WGSL}
struct ExRayRecord { origin: vec3f, tMin: f32, direction: vec3f, tMax: f32 }
struct ExLight { toSun: vec3f, sun: f32, sky: vec3f, sunRadius: f32 }  // sunRadius: the sun disc's angular radius (radians); 0 = a point sun, hard shadows

@group(2) @binding(0) var<uniform> view: ExView;
@group(2) @binding(1) var<uniform> light: ExLight;
@group(2) @binding(2) var<storage, read_write> rays: array<ExRayRecord>;
@group(2) @binding(3) var<storage, read> hits: array<u32>;          // ${HIT_BYTES / 4} words per hit
@group(2) @binding(4) var<storage, read_write> shadowRays: array<ExRayRecord>;
@group(2) @binding(5) var<storage, read> visibility: array<u32>;
@group(2) @binding(6) var<storage, read_write> accum: array<vec4f>;

const HIT_WORDS: u32 = ${HIT_BYTES / 4}u;

// The 40 B record of the batch tracer as the module's RtHit, so rtSurface can decode it.
fn exHit(i: u32) -> RtHit {
  let b = i * HIT_WORDS;
  return RtHit(bitcast<f32>(hits[b]), hits[b + 1u], hits[b + 2u], hits[b + 3u] != 0u, vec2f(bitcast<f32>(hits[b + 4u]), bitcast<f32>(hits[b + 5u])),
               hits[b + 6u], vec3f(bitcast<f32>(hits[b + 7u]), bitcast<f32>(hits[b + 8u]), bitcast<f32>(hits[b + 9u])), 0.0);
}

fn exRecordToRay(r: ExRayRecord) -> RtRay { return RtRay(r.origin, r.tMin, r.direction, r.tMax); }

fn exOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {
  let ofs = vec3i(256.0 * n);
  let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
  return select(pi, p + (1.0 / 65536.0) * n, abs(p) < vec3f(1.0 / 32.0));
}

@compute @workgroup_size(8, 8)
fn primary(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= view.width || id.y >= view.height) { return; }
  var seed = exHash(id.x + id.y * view.width) ^ exHash(view.frame * 9781u + 41u);
  // A random point inside the pixel each frame (anti-aliasing by accumulation); mode bit 0 = the pixel centre every frame (jagged edges).
  let jitter = select(vec2f(exRand(&seed), exRand(&seed)), vec2f(0.5), (view.mode & 1u) != 0u);
  let r = exRay(view, id.x, id.y, jitter);
  rays[id.y * view.width + id.x] = ExRayRecord(r.origin, r.tMin, r.direction, r.tMax);
}

@compute @workgroup_size(64)
fn shadow(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= view.width * view.height) { return; }
  let hit = exHit(i);
  if (!rtIsHit(hit)) {
    shadowRays[i] = ExRayRecord(vec3f(0.0), 0.0, light.toSun, 0.0);  // an empty segment: reads visible, compose ignores it
    return;
  }
  let ray = exRecordToRay(rays[i]);
  let n = select(-hit.normal, hit.normal, dot(hit.normal, ray.direction) < 0.0);
  let p = ray.origin + hit.t * ray.direction;
  // A random direction inside the sun's disc (uniform in the cone): partly blocked discs give soft shadows, which is where the noise is.
  var seed = exHash(i * 7919u + 3u) ^ exHash(view.frame * 6151u + 17u);
  let cosMax = cos(light.sunRadius);
  let ct = 1.0 - exRand(&seed) * (1.0 - cosMax);
  let st = sqrt(max(0.0, 1.0 - ct * ct));
  let phi = 6.283185307 * exRand(&seed);
  let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(light.toSun.x) > 0.9);
  let tx = normalize(cross(a, light.toSun));
  let ty = cross(light.toSun, tx);
  let d = normalize(ct * light.toSun + st * (cos(phi) * tx + sin(phi) * ty));
  shadowRays[i] = ExRayRecord(exOffsetOrigin(p, n), 0.0, d, select(0.0, 1e30, dot(n, d) > 0.0));
}

@compute @workgroup_size(64)
fn compose(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= view.width * view.height) { return; }
  let hit = exHit(i);
  var c = light.sky;
  if (rtIsHit(hit)) {
    let ray = exRecordToRay(rays[i]);
    let s = rtSurface(hit, ray, RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_MATERIAL | RT_SURF_POSITION);
    var m = rtMaterial(s.material);
    ${textured ? 'm = rtApplyMaterialTextures(m, s.uv, 0.0);' : ''}
    let n = select(-s.shadingNormal, s.shadingNormal, dot(s.shadingNormal, ray.direction) < 0.0);
    let sun = light.sun * max(dot(n, light.toSun), 0.0) * f32(visibility[i]);
    // Lambert under the sun plus a uniform sky (the sky unshadowed: AO would darken it).
    c = m.baseColor * m.baseWeight * (sun + light.sky * 3.14159265) / 3.14159265 + m.emissionColor;
  }
  accum[i] = accum[i] + vec4f(c, 1.0);
}
`;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const params = { elevation: 45, azimuth: 35, sun: 3, sky: 0.25, sunSize: 0.25, jitter: true }; // sunSize: the real sun's angular radius
  const accumulation = new Accumulation(ctx);
  accumulation.maxFrames = 512;
  exposeForChecks(accumulation);
  const viewBuffer = device.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const lightBuffer = device.createBuffer({ label: 'light', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let state: { scene: GpuScene; kernel: TraceKernel; closest: RtBatchTracer; visible: RtBatchTracer } | null = null;
  let buffers: { rays: GPUBuffer; hits: GPUBuffer; shadowRays: GPUBuffer; visibility: GPUBuffer; group: GPUBindGroup; accum: GPUBuffer; pixels: number } | null = null;

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  const picker = await scenePicker(ctx, sceneFolder, 'Damaged Helmet', async (scene) => {
    const kernel = await TraceKernel.create(device, caps, scene, { label: 'batch', wgsl: wgsl(!!scene.copy.textureLayout), entryPoints: ['primary', 'shadow', 'compose'], walk: false, io: ioEntries(['uniform', 'uniform', 'write', 'read', 'write', 'read', 'write']), module: { scene: { slot: { group: 1, firstBinding: 0 }, layout: scene.copy.layout, textures: scene.copy.textureLayout, packing: scene.packing, lights: false } } });
    // The tracers are compiled for the scene's BVH format, like the module.
    const closest = await RtBatchTracer.create(device, caps, { ...scene.format, mode: 'closest', hitRecord: 'full', label: 'closest' });
    const visible = await RtBatchTracer.create(device, caps, { ...scene.format, mode: 'visible', label: 'visible' });
    state = { scene, kernel, closest, visible };
    buffers = null;
    stats.set('kernel', `${(kernel.bytes / 1024).toFixed(1)} KB`);
    accumulation.reset();
  }, () => accumulation.reset());

  const lightFolder = pane.addFolder({ title: 'Light' });
  for (const [key, opts] of [['elevation', { min: 1, max: 90, step: 1 }], ['azimuth', { min: -180, max: 180, step: 1 }], ['sun', { min: 0, max: 10, step: 0.1 }], ['sky', { min: 0, max: 2, step: 0.01 }]] as const) {
    lightFolder.addBinding(params, key, opts).on('change', () => accumulation.reset());
  }
  // A point sun and a constant sky leave nothing random but the pixel jitter: the image is clean after a few frames and can look rasterised.
  // A sun disc makes every shadow ray a random sample: watch the penumbrae start grainy and converge (stop at 1 sample to see one frame).
  lightFolder.addBinding(params, 'sunSize', { label: 'sun radius (deg)', min: 0, max: 15, step: 0.25 }).on('change', () => accumulation.reset());
  lightFolder.addBinding(params, 'jitter', { label: 'pixel jitter (AA)' }).on('change', () => accumulation.reset());
  const view = pane.addFolder({ title: 'View' });
  view.addBinding(accumulation.view, 'exposure', { min: 0, max: 4, step: 0.01 });

  ctx.loop((encoder) => {
    if (!state) return false;
    const { scene, kernel, closest, visible } = state;
    return accumulation.frameWith(encoder, (enc, frame) => {
      const pixels = accumulation.width * accumulation.height;
      if (!buffers || buffers.accum !== accumulation.buffer || buffers.pixels !== pixels) {
        for (const b of [buffers?.rays, buffers?.hits, buffers?.shadowRays, buffers?.visibility]) b?.destroy();
        const make = (bytes: number) => device.createBuffer({ size: pixels * bytes, usage: GPUBufferUsage.STORAGE });
        const rays = make(RAY_BYTES), hits = make(HIT_BYTES), shadowRays = make(RAY_BYTES), visibility = make(4);
        const group = device.createBindGroup({
          layout: kernel.ioLayout,
          entries: [viewBuffer, lightBuffer, rays, hits, shadowRays, visibility, accumulation.buffer!].map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        buffers = { rays, hits, shadowRays, visibility, group, accum: accumulation.buffer!, pixels };
        stats.set('rays / frame', (2 * pixels).toLocaleString());
      }
      const el = (params.elevation * Math.PI) / 180, az = (params.azimuth * Math.PI) / 180;
      device.queue.writeBuffer(lightBuffer, 0, new Float32Array([Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az), params.sun, params.sky * 0.8, params.sky * 0.9, params.sky, (params.sunSize * Math.PI) / 180]));
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, params.jitter ? 0 : 1);
      const b = buffers;
      kernel.dispatch(enc, 'primary', b.group, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
      closest.encode(enc, scene.trace, { rays: b.rays, results: b.hits, count: pixels });
      kernel.dispatch(enc, 'shadow', b.group, Math.ceil(pixels / 64));
      visible.encode(enc, scene.trace, { rays: b.shadowRays, results: b.visibility, count: pixels });
      kernel.dispatch(enc, 'compose', b.group, Math.ceil(pixels / 64));
    });
  });
});
