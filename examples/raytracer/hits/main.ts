/**
 * Hit inspector: `@pmndrs/raytracer` inside your own compute shader.
 *
 * `rtModuleWgsl` gives the shader `rtClosest` (the first hit along a ray) and `rtSurface` (what is at that hit: position, normals, uv,
 * material id). This kernel shoots one jittered camera ray per pixel per frame and writes one property of the hit as a colour: a G-buffer
 * made by rays instead of rasterisation, for your own shading. Click a pixel: a second entry point traces the ray through it and the
 * panel shows its hit record as numbers. No path tracer is involved.
 */
import type { GpuScene } from '@pmndrs/raytracer';
import { Accumulation, exposeForChecks } from '../../shared/accumulation.ts';
import { defineExample } from '../../shared/example.ts';
import { scenePicker } from '../../shared/scene-picker.ts';
import { ioEntries, TraceKernel } from '../../shared/trace-kernel.ts';
import { VIEW_BYTES, VIEW_WGSL, writeView } from '../../shared/view.ts';

const MODES = { 'Base colour (material + texture)': 8, 'Shading normal (as RGB)': 0, 'Geometric normal (as RGB)': 1, 'Front / back face': 2, 'Distance bands': 3, 'Texture uv': 4, 'Barycentrics': 5, 'Material id': 6, 'Instance id': 7 };
const EXPLAIN: Record<number, string> = {
  8: 'the hit material\'s base colour with its texture, unlit',
  0: 'interpolated normal, xyz mapped to RGB (x right = red, y up = green, z = blue): looks like a normal map because it is one',
  1: 'the triangle\'s own face normal as RGB: facets show',
  2: 'green: the ray hit the triangle\'s front side, red: its back',
  3: 'the hit distance t in bands of the chosen width (each band = one cycle black to white)',
  4: 'the hit\'s texture coordinates: u red, v green (repeats every 1)',
  5: 'barycentric weights of the three vertices of the hit triangle',
  6: 'one random colour per material id',
  7: 'one random colour per instance',
};

/** `textured`: the scene has textures, so the base colour can apply them (`rtApplyMaterialTextures` exists only then). */
const kernelWgsl = (textured: boolean) => /* wgsl */ `
${VIEW_WGSL}
@group(2) @binding(0) var<uniform> view: ExView;
@group(2) @binding(1) var<storage, read_write> accum: array<vec4f>;
// Words 0, 1: the clicked pixel (written by the page); 2..: the hit record the probe entry point writes back.
@group(2) @binding(2) var<storage, read_write> probe: array<u32, 24>;

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

// One ray through the centre of the clicked pixel; the full hit record, bit for bit.
@compute @workgroup_size(1)
fn probeHit() {
  let ray = exRay(view, probe[0], probe[1], vec2f(0.5));
  let hit = rtClosest(ray, RT_MASK_ALL);
  probe[2] = bitcast<u32>(hit.t);
  if (!rtIsHit(hit)) { return; }
  let s = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_MATERIAL);
  probe[3] = hit.instance;
  probe[4] = hit.triangle;
  probe[5] = s.material;
  probe[6] = select(0u, 1u, hit.frontFace);
  probe[7] = bitcast<u32>(hit.bary.x);
  probe[8] = bitcast<u32>(hit.bary.y);
  probe[9] = bitcast<u32>(s.uv.x);
  probe[10] = bitcast<u32>(s.uv.y);
  probe[11] = bitcast<u32>(s.position.x);
  probe[12] = bitcast<u32>(s.position.y);
  probe[13] = bitcast<u32>(s.position.z);
  probe[14] = bitcast<u32>(s.shadingNormal.x);
  probe[15] = bitcast<u32>(s.shadingNormal.y);
  probe[16] = bitcast<u32>(s.shadingNormal.z);
  probe[17] = bitcast<u32>(s.geometricNormal.x);
  probe[18] = bitcast<u32>(s.geometricNormal.y);
  probe[19] = bitcast<u32>(s.geometricNormal.z);
}
`;

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const params = { mode: Number(new URLSearchParams(location.search).get('mode') ?? 8), distanceBand: 1 };
  const accumulation = new Accumulation(ctx);
  accumulation.maxFrames = 256;
  accumulation.view.toneMapping = 'clamp';
  exposeForChecks(accumulation);
  const viewBuffer = device.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let kernel: TraceKernel | null = null;
  let group: GPUBindGroup | null = null;
  let bound: GPUBuffer | null = null;
  const probeBuffer = device.createBuffer({ label: 'probe', size: 96, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const probeRead = device.createBuffer({ label: 'probe read', size: 96, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const sceneFolder = pane.addFolder({ title: 'Scene' });
  const picker = await scenePicker(ctx, sceneFolder, 'Damaged Helmet', async (scene: GpuScene) => {
    kernel = await TraceKernel.create(device, caps, scene, { label: 'hits', wgsl: kernelWgsl(!!scene.copy.textureLayout), entryPoints: ['inspect', 'probeHit'], io: ioEntries(['uniform', 'write', 'write']) });
    bound = null;
    stats.set('kernel', `${(kernel.bytes / 1024).toFixed(1)} KB`);
    accumulation.reset();
  }, () => accumulation.reset());

  const show = pane.addFolder({ title: 'Show' });
  const explain = () => stats.set('showing', EXPLAIN[params.mode] ?? '');
  explain();
  show.addBinding(params, 'mode', { label: 'property', options: MODES }).on('change', () => {
    explain();
    accumulation.reset();
  });
  show.addBinding(params, 'distanceBand', { label: 'distance band (m)', min: 0.05, max: 10, step: 0.05 }).on('change', () => accumulation.reset());

  //* Click to inspect: one ray through the pixel's centre, its hit record read back (a readback on demand, never in the frame loop).
  const panel = document.createElement('div');
  panel.className = 'ex-hit';
  panel.innerHTML = '<p>Click a pixel to read its hit record.</p>';
  const marker = document.createElement('div');
  marker.className = 'ex-hit-marker';
  marker.hidden = true;
  const style = document.createElement('style');
  style.textContent = `
    .ex-hit { position: fixed; left: 16px; bottom: 16px; padding: 10px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); font-size: 12px; max-width: calc(100vw - 32px); }
    .ex-hit p { margin: 0; color: var(--muted); } .ex-hit table { border-collapse: collapse; font-family: var(--mono); }
    .ex-hit td { padding: 1px 10px 1px 0; } .ex-hit td:first-child { color: var(--muted); font-family: var(--font); }
    .ex-hit-marker { position: fixed; width: 13px; height: 13px; margin: -7px 0 0 -7px; border: 2px solid var(--accent); border-radius: 50%; pointer-events: none; box-shadow: 0 0 0 1px #000; }`;
  document.head.append(style);
  document.body.append(panel, marker);
  const fmt = (v: number) => (Math.abs(v) >= 1000 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : v.toFixed(4));
  const v3 = (a: number, b: number, c: number) => `(${fmt(a)}, ${fmt(b)}, ${fmt(c)})`;
  let probing = false;
  const inspectAt = async (clientX: number, clientY: number) => {
    if (!kernel || !group || probing) return;
    const rect = ctx.canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * accumulation.width), y = Math.floor(((clientY - rect.top) / rect.height) * accumulation.height);
    if (x < 0 || y < 0 || x >= accumulation.width || y >= accumulation.height) return;
    probing = true;
    try {
      device.queue.writeBuffer(probeBuffer, 0, new Uint32Array([x, y]));
      const enc = device.createCommandEncoder({ label: 'probe' });
      kernel.dispatch(enc, 'probeHit', group, 1);
      enc.copyBufferToBuffer(probeBuffer, 0, probeRead, 0, 96);
      device.queue.submit([enc.finish()]);
      await probeRead.mapAsync(GPUMapMode.READ);
      const u = new Uint32Array(probeRead.getMappedRange().slice(0));
      probeRead.unmap();
      const f = new Float32Array(u.buffer);
      marker.style.left = `${clientX}px`;
      marker.style.top = `${clientY}px`;
      marker.hidden = false;
      if (!(f[2]! >= 0)) {
        panel.innerHTML = `<p>Pixel (${x}, ${y}): the ray hit nothing (t = ${fmt(f[2]!)}, the sky).</p>`;
        return;
      }
      const m = picker.current()?.loaded.input.materials[u[5]!];
      const rows: [string, string][] = [
        ['pixel', `(${x}, ${y})`],
        ['distance t', `${fmt(f[2]!)} m`],
        ['instance · triangle', `${u[3]} · ${u[4]}`],
        ['material', `${u[5]}${m?.baseColor ? ` · base colour ${v3(...(m.baseColor as [number, number, number]))}` : ''}${m?.metalness !== undefined ? ` · metalness ${fmt(m.metalness)}` : ''}`],
        ['side', u[6] ? 'front face' : 'back face'],
        ['barycentrics', v3(1 - f[7]! - f[8]!, f[7]!, f[8]!)],
        ['uv', `(${fmt(f[9]!)}, ${fmt(f[10]!)})`],
        ['position', v3(f[11]!, f[12]!, f[13]!)],
        ['shading normal', v3(f[14]!, f[15]!, f[16]!)],
        ['geometric normal', v3(f[17]!, f[18]!, f[19]!)],
      ];
      panel.innerHTML = `<table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('')}</table>`;
      (globalThis as { __hit?: unknown }).__hit = Object.fromEntries(rows);
    } finally {
      probing = false;
    }
  };
  // A click is a press and release without dragging (a drag orbits the camera).
  let down: [number, number] | null = null;
  ctx.canvas.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
  ctx.canvas.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) void inspectAt(e.clientX, e.clientY);
    down = null;
  });
  const orbitChanged = picker.orbit.onChange;
  picker.orbit.onChange = () => {
    marker.hidden = true;
    orbitChanged();
  };
  (globalThis as { __inspect?: unknown }).__inspect = inspectAt;

  ctx.loop((encoder) => {
    if (!kernel) return false;
    const k = kernel;
    return accumulation.frameWith(encoder, (enc, frame) => {
      if (bound !== accumulation.buffer) {
        group = device.createBindGroup({ layout: k.ioLayout, entries: [{ binding: 0, resource: { buffer: viewBuffer } }, { binding: 1, resource: { buffer: accumulation.buffer! } }, { binding: 2, resource: { buffer: probeBuffer } }] });
        bound = accumulation.buffer;
      }
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, params.mode, [params.distanceBand, 0]);
      k.dispatch(enc, 'inspect', group!, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
    });
  });
});
