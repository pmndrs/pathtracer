/**
 * Rays from your own shader: what ray tracing adds to a raster pass.
 *
 * `rtModuleWgsl` gives your compute shader `rtClosest` (the first hit along any ray), `rtVisible` (is anything in the way?) and `rtSurface`
 * (position, normals, uv, material at a hit). Each pixel here shoots one camera ray; its first hit is exactly the surface a rasteriser
 * would draw there, so "primary hits only" looks like a raster pass with its G-buffer. The point is what comes next: from that hit the
 * shader traces new rays in any direction, a shadow ray to the sun, a mirror reflection, ambient-occlusion rays, against the whole scene,
 * including what is off screen or hidden. A raster pass needs shadow maps, screen-space reflections or SSAO for those, with their limits.
 *
 * Click a pixel: the `probeRays` entry point traces that pixel's rays once more and the page draws them over the image with each hit's
 * numbers. No path tracer is involved; frames accumulate (jittered camera rays, random AO directions) into an anti-aliased image.
 */
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import type { GpuScene } from '@pmndrs/raytracer';
import { Accumulation, exposeForChecks } from '../../shared/accumulation.ts';
import { defineExample } from '../../shared/example.ts';
import { scenePicker } from '../../shared/scene-picker.ts';
import { ioEntries, TraceKernel } from '../../shared/trace-kernel.ts';
import { VIEW_BYTES, VIEW_WGSL, writeView } from '../../shared/view.ts';

const VIEWS = {
  'Lit by traced rays': 9,
  'G-buffer: base colour': 8,
  'G-buffer: shading normal (as RGB)': 0,
  'G-buffer: geometric normal (as RGB)': 1,
  'G-buffer: front / back face': 2,
  'G-buffer: distance bands': 3,
  'G-buffer: texture uv': 4,
  'G-buffer: barycentrics': 5,
  'G-buffer: material id': 6,
  'G-buffer: instance id': 7,
};
const EXPLAIN: Record<number, string> = {
  9: 'base colour lit by the sun and the sky; each switched-on ray type is traced from every hit',
  8: 'the hit material\'s base colour with its texture, unlit: the first layer of a G-buffer',
  0: 'interpolated normal, xyz as RGB (x red, y green, z blue): what a normal buffer stores',
  1: 'the triangle\'s own face normal as RGB: facets show',
  2: 'green: the ray hit the triangle\'s front side, red: its back',
  3: 'the hit distance t in bands of the chosen width',
  4: 'the hit\'s texture coordinates: u red, v green (repeats every 1)',
  5: 'barycentric weights of the three vertices of the hit triangle',
  6: 'one random colour per material id',
  7: 'one random colour per instance',
};
const SHADOW = 1 << 8, REFLECT = 1 << 9, AO = 1 << 10;
const AO_RAYS = 6;

/** `textured`: the scene has textures, so the base colour can apply them (`rtApplyMaterialTextures` exists only then). */
const kernelWgsl = (textured: boolean) => /* wgsl */ `
${VIEW_WGSL}
struct ExLight { toSun: vec3f, sun: f32, sky: vec3f, mirror: f32 }
@group(2) @binding(0) var<uniform> view: ExView;
@group(2) @binding(1) var<storage, read_write> accum: array<vec4f>;
// Words 0, 1: the clicked pixel (written by the page); then the probe's rays (see probeRays).
@group(2) @binding(2) var<storage, read_write> probe: array<f32, 64>;
@group(2) @binding(3) var<uniform> light: ExLight;

fn idColor(i: u32) -> vec3f {
  let h = exHash(i + 1u);
  return vec3f(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0;
}

// rtShade's module has these as rtOffsetOrigin / rtSideNormal; this kernel does not include rtShade.
fn exOffsetOrigin(p: vec3f, n: vec3f) -> vec3f {
  let ofs = vec3i(256.0 * n);
  let pi = bitcast<vec3f>(bitcast<vec3i>(p) + select(ofs, -ofs, p < vec3f(0.0)));
  return select(pi, p + (1.0 / 65536.0) * n, abs(p) < vec3f(1.0 / 32.0));
}
fn exSide(n: vec3f, w: vec3f) -> vec3f { return select(-n, n, dot(n, w) > 0.0); }
fn exCosineDirection(n: vec3f, u: vec2f) -> vec3f {
  let a = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
  let t = normalize(a - n * dot(n, a));
  let b = cross(n, t);
  let r = sqrt(u.x);
  let phi = 6.283185307 * u.y;
  return normalize(r * cos(phi) * t + r * sin(phi) * b + sqrt(max(0.0, 1.0 - u.x)) * n);
}

struct ExShade { base: vec3f, f0: vec3f, emission: vec3f, gloss: f32 }
fn exMaterial(s: RtSurface) -> ExShade {
  var m = rtMaterial(s.material);
  ${textured ? 'm = rtApplyMaterialTextures(m, s.uv, 0.0);' : ''}
  let base = m.baseColor * m.baseWeight;
  // A mirror ray stands for the specular lobe only on smooth surfaces: fade it out with roughness (a rough cloth reflects nothing sharp).
  let gloss = (1.0 - m.specularRoughness) * (1.0 - m.specularRoughness);
  return ExShade(base, mix(vec3f(0.04), base, m.metalness) * m.specularWeight, m.emissionColor, gloss);
}

// Base colour under the sun and the sky. shadow: the sun's visibility (1 without a shadow ray), ao: the sky's (1 without AO rays).
fn exLit(sh: ExShade, n: vec3f, shadow: f32, ao: f32) -> vec3f {
  return sh.base * (light.sun * max(dot(n, light.toSun), 0.0) * shadow + light.sky * ao) + sh.emission;
}

struct ExTraced { colour: vec3f, sunT: f32, reflDir: vec3f, reflT: f32 }

// One pixel's lit colour: the camera ray's hit, plus the rays the flags switch on. Writes the secondary rays' answers for the probe.
fn exTrace(ray: RtRay, hit: RtHit, seed: ptr<function, u32>, flags: u32, aoRadius: f32, aoT: ptr<function, array<f32, ${AO_RAYS}>>, aoDir: ptr<function, array<vec3f, ${AO_RAYS}>>) -> ExTraced {
  var out = ExTraced(light.sky, -1.0, vec3f(0.0), -1.0);
  if (!rtIsHit(hit)) { return out; }
  let s = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_MATERIAL);
  let ng = exSide(s.geometricNormal, -ray.direction);
  let n = exSide(s.shadingNormal, -ray.direction);
  let origin = exOffsetOrigin(s.position, ng);
  let sh = exMaterial(s);
  // Shadow ray: is anything between this point and the sun? (Off: the sun reaches every surface facing it, as without shadow maps.)
  var shadow = 1.0;
  if ((flags & ${SHADOW}u) != 0u && dot(ng, light.toSun) > 0.0) {
    let blocker = rtClosest(RtRay(origin, 0.0, light.toSun, 1e30), RT_MASK_ALL);
    if (rtIsHit(blocker)) { shadow = 0.0; out.sunT = blocker.t; }
  }
  // Ambient occlusion: how much of the sky the hemisphere above the point sees within the radius.
  var ao = 1.0;
  if ((flags & ${AO}u) != 0u) {
    var open = 0.0;
    for (var k = 0u; k < ${AO_RAYS}u; k++) {
      var d = exCosineDirection(n, vec2f(exRand(seed), exRand(seed)));
      if (dot(d, ng) <= 0.0) { d = reflect(d, ng); }
      let h = rtClosest(RtRay(origin, 0.0, d, aoRadius), RT_MASK_ALL);
      (*aoDir)[k] = d;
      (*aoT)[k] = select(-1.0, h.t, rtIsHit(h));
      open += select(1.0, 0.0, rtIsHit(h));
    }
    ao = open / f32(${AO_RAYS}u);
  }
  var c = exLit(sh, n, shadow, ao);
  // Mirror reflection: the surface seen in the reflected direction, lit the same way (its own shadow ray), weighted by Fresnel.
  if ((flags & ${REFLECT}u) != 0u && light.mirror > 0.0) {
    let r = reflect(ray.direction, n);
    out.reflDir = r;
    let rh = rtClosest(RtRay(origin, 0.0, r, 1e30), RT_MASK_ALL);
    var seen = light.sky;
    if (rtIsHit(rh)) {
      out.reflT = rh.t;
      let rr = RtRay(origin, 0.0, r, 1e30);
      let rs = rtSurface(rh, rr, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_MATERIAL);
      let rng = exSide(rs.geometricNormal, -r);
      var rshadow = 1.0;
      if ((flags & ${SHADOW}u) != 0u && dot(rng, light.toSun) > 0.0) {
        rshadow = select(1.0, 0.0, !rtVisible(RtRay(exOffsetOrigin(rs.position, rng), 0.0, light.toSun, 1e30), RT_MASK_ALL));
      }
      seen = exLit(exMaterial(rs), exSide(rs.shadingNormal, -r), rshadow, 1.0);
    }
    let cosv = max(dot(n, -ray.direction), 0.0);
    let fresnel = sh.f0 + (vec3f(1.0) - sh.f0) * pow(1.0 - cosv, 5.0);
    c = mix(c, seen, fresnel * light.mirror * sh.gloss);
  }
  out.colour = c;
  return out;
}

@compute @workgroup_size(8, 8)
fn inspect(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= view.width || id.y >= view.height) { return; }
  var seed = exHash(id.x + id.y * view.width) ^ exHash(view.frame * 9781u + 7u);
  let ray = exRay(view, id.x, id.y, vec2f(exRand(&seed), exRand(&seed)));
  let hit = rtClosest(ray, RT_MASK_ALL);
  let mode = view.mode & 255u;
  var c = vec3f(0.03);
  if (mode == 9u) {
    var aoT: array<f32, ${AO_RAYS}>;
    var aoDir: array<vec3f, ${AO_RAYS}>;
    c = exTrace(ray, hit, &seed, view.mode, view.origin.w, &aoT, &aoDir).colour;
  } else if (rtIsHit(hit)) {
    let s = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_MATERIAL);
    switch mode {
      case 0u: { c = s.shadingNormal * 0.5 + 0.5; }
      case 1u: { c = s.geometricNormal * 0.5 + 0.5; }
      case 2u: { c = select(vec3f(0.9, 0.25, 0.2), vec3f(0.3, 0.8, 0.35), s.frontFace); }
      case 3u: { c = vec3f(fract(hit.t / max(view.right.w, 1e-6))); }
      case 4u: { c = vec3f(fract(s.uv), 0.0); }
      case 5u: { c = vec3f(1.0 - hit.bary.x - hit.bary.y, hit.bary.x, hit.bary.y); }
      case 6u: { c = idColor(s.material); }
      case 7u: { c = idColor(hit.instance); }
      default: { let sh = exMaterial(s); c = sh.base + sh.emission; }
    }
  }
  let i = id.y * view.width + id.x;
  accum[i] = accum[i] + vec4f(c, 1.0);
}

// One ray through the centre of the clicked pixel and every ray traced from its hit, for the overlay. Layout (f32 words):
//  2 t, 3 material, 4..6 hit position, 7..9 shading normal, 10 sun t (-1: reached the sun, -2: not traced), 11..13 reflection direction,
//  14 reflection t (-1: sky, -2: not traced), 15 AO ray count, 16.. per AO ray: direction xyz and t (-1: open)
@compute @workgroup_size(1)
fn probeRays() {
  let ray = exRay(view, u32(probe[0]), u32(probe[1]), vec2f(0.5));
  let hit = rtClosest(ray, RT_MASK_ALL);
  probe[2] = hit.t;
  if (!rtIsHit(hit)) { return; }
  let s = rtSurface(hit, ray, RT_SURF_POSITION | RT_SURF_GEOMETRIC_NORMAL | RT_SURF_SHADING_NORMAL | RT_SURF_UV | RT_SURF_MATERIAL);
  probe[3] = f32(s.material);
  let n = exSide(s.shadingNormal, -ray.direction);
  probe[4] = s.position.x; probe[5] = s.position.y; probe[6] = s.position.z;
  probe[7] = n.x; probe[8] = n.y; probe[9] = n.z;
  var seed = 12345u;
  var aoT: array<f32, ${AO_RAYS}>;
  var aoDir: array<vec3f, ${AO_RAYS}>;
  let flags = view.mode | ${SHADOW | REFLECT | AO}u;
  let t = exTrace(ray, hit, &seed, flags, view.origin.w, &aoT, &aoDir);
  probe[10] = t.sunT;
  probe[11] = t.reflDir.x; probe[12] = t.reflDir.y; probe[13] = t.reflDir.z;
  probe[14] = t.reflT;
  probe[15] = f32(${AO_RAYS}u);
  for (var k = 0u; k < ${AO_RAYS}u; k++) {
    probe[16u + k * 4u] = aoDir[k].x; probe[17u + k * 4u] = aoDir[k].y; probe[18u + k * 4u] = aoDir[k].z; probe[19u + k * 4u] = aoT[k];
  }
}
`;

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3, k = 1): V3 => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => {
  const l = Math.hypot(...a);
  return [a[0] / l, a[1] / l, a[2] / l];
};

/** World point to CSS pixels on the canvas (the same camera model as `exRay`), or null behind the camera. */
function project(camera: PtLookAtCamera, rect: DOMRect, p: V3): [number, number] | null {
  const fw = norm(sub(camera.target as V3, camera.position as V3));
  const right = norm(cross(fw, camera.up as V3));
  const up = cross(right, fw);
  const d = sub(p, camera.position as V3);
  const z = dot(d, fw);
  if (z <= 1e-4) return null;
  const t = Math.tan((camera.yFovDeg * Math.PI) / 360);
  const aspect = rect.width / rect.height;
  const x = dot(d, right) / z / (t * aspect), y = dot(d, up) / z / t;
  return [rect.left + ((x + 1) / 2) * rect.width, rect.top + ((1 - y) / 2) * rect.height];
}

defineExample(async (ctx) => {
  const { device, caps, pane, stats } = ctx;
  const search = new URLSearchParams(location.search);
  const params = {
    view: Number(search.get('view') ?? 9),
    shadows: search.get('shadows') !== '0',
    reflections: search.get('reflections') !== '0',
    ao: search.get('ao') !== '0',
    reflect: 0.6,
    aoRadius: 0.5,
    elevation: 40,
    azimuth: 30,
    sun: 2.5,
    sky: 0.35,
    distanceBand: 1,
  };
  const accumulation = new Accumulation(ctx);
  accumulation.maxFrames = 256;
  exposeForChecks(accumulation);
  const viewBuffer = device.createBuffer({ label: 'view', size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const lightBuffer = device.createBuffer({ label: 'light', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const probeBuffer = device.createBuffer({ label: 'probe', size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const probeRead = device.createBuffer({ label: 'probe read', size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  let kernel: TraceKernel | null = null;
  let group: GPUBindGroup | null = null;
  let bound: GPUBuffer | null = null;
  const toSun = (): V3 => {
    const el = (params.elevation * Math.PI) / 180, az = (params.azimuth * Math.PI) / 180;
    return [Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)];
  };
  const flags = () => (params.shadows ? SHADOW : 0) | (params.reflections ? REFLECT : 0) | (params.ao ? AO : 0);

  // scenePicker calls its callbacks before it returns: anything they use must exist first.
  let hideOverlay = () => {};
  const sceneFolder = pane.addFolder({ title: 'Scene' });
  const picker = await scenePicker(ctx, sceneFolder, 'Damaged Helmet', async (scene: GpuScene, loaded) => {
    kernel = await TraceKernel.create(device, caps, scene, { label: 'rays', wgsl: kernelWgsl(!!scene.copy.textureLayout), entryPoints: ['inspect', 'probeRays'], io: ioEntries(['uniform', 'write', 'write', 'uniform']) });
    bound = null;
    // An AO radius of a tenth of the scene's size shows contact shadows without darkening everything.
    params.aoRadius = Number((Math.hypot(...loaded.bounds.max.map((v, i) => v - loaded.bounds.min[i]!)) * 0.1).toPrecision(2));
    pane.refresh();
    stats.set('kernel', `${(kernel.bytes / 1024).toFixed(1)} KB`);
    accumulation.reset();
  }, () => {
    hideOverlay();
    accumulation.reset();
  });

  const describe = () => {
    stats.set('showing', EXPLAIN[params.view] ?? '');
    const on = [params.shadows && 'shadow', params.reflections && 'reflection', params.ao && `${AO_RAYS} AO`].filter(Boolean);
    stats.set('rays / pixel', params.view === 9 ? `1 camera${on.length ? ` + ${on.join(' + ')}` : ' only: what a raster pass gives'}` : '1 camera');
  };
  const restart = () => {
    describe();
    accumulation.reset();
  };
  const rays = pane.addFolder({ title: 'Rays from each hit' });
  rays.addBinding(params, 'shadows', { label: 'shadow rays (sun)' }).on('change', restart);
  rays.addBinding(params, 'reflections', { label: 'reflection rays' }).on('change', restart);
  rays.addBinding(params, 'ao', { label: `AO rays (${AO_RAYS})` }).on('change', restart);
  rays.addBinding(params, 'reflect', { label: 'reflection strength', min: 0, max: 1, step: 0.05 }).on('change', restart);
  rays.addBinding(params, 'aoRadius', { label: 'AO radius', min: 0.001, max: 10, step: 0.001 }).on('change', restart);
  const light = pane.addFolder({ title: 'Light', expanded: false });
  for (const [key, opts] of [['elevation', { min: 1, max: 90, step: 1 }], ['azimuth', { min: -180, max: 180, step: 1 }], ['sun', { min: 0, max: 8, step: 0.1 }], ['sky', { min: 0, max: 2, step: 0.01 }]] as const) {
    light.addBinding(params, key, opts).on('change', restart);
  }
  const show = pane.addFolder({ title: 'Show' });
  show.addBinding(params, 'view', { options: VIEWS }).on('change', restart);
  show.addBinding(params, 'distanceBand', { label: 'distance band (m)', min: 0.05, max: 10, step: 0.05 }).on('change', restart);
  describe();

  //* Click a pixel: its rays, traced once more, drawn over the image (a readback on demand, never in the frame loop).
  const style = document.createElement('style');
  style.textContent = `
    .ex-rays { position: fixed; inset: 0; pointer-events: none; z-index: 5; }
    .ex-ray-info { position: fixed; left: 16px; bottom: 16px; padding: 10px 12px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); font-size: 12px; max-width: calc(100vw - 32px); }
    .ex-ray-info p { margin: 0; color: var(--muted); } .ex-ray-info table { border-collapse: collapse; font-family: var(--mono); }
    .ex-ray-info td { padding: 1px 10px 1px 0; } .ex-ray-info td:first-child { font-family: var(--font); }
    .ex-ray-info i { display: inline-block; width: 10px; height: 3px; margin-right: 6px; vertical-align: middle; }`;
  document.head.append(style);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('ex-rays');
  const info = document.createElement('div');
  info.className = 'ex-ray-info';
  info.innerHTML = '<p>Click a pixel to draw its rays.</p>';
  document.body.append(svg, info);
  hideOverlay = () => svg.replaceChildren();
  const fmt = (v: number) => (Math.abs(v) >= 1000 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(3));
  const line = (a: [number, number], b: [number, number], colour: string, dashed = false) => {
    const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l.setAttribute('x1', String(a[0])); l.setAttribute('y1', String(a[1])); l.setAttribute('x2', String(b[0])); l.setAttribute('y2', String(b[1]));
    l.setAttribute('stroke', colour); l.setAttribute('stroke-width', '2'); l.setAttribute('stroke-linecap', 'round');
    if (dashed) l.setAttribute('stroke-dasharray', '5 4');
    svg.append(l);
  };
  const dotAt = (p: [number, number], colour: string, r = 4) => {
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', String(p[0])); c.setAttribute('cy', String(p[1])); c.setAttribute('r', String(r));
    c.setAttribute('fill', colour); c.setAttribute('stroke', '#000'); c.setAttribute('stroke-width', '1');
    svg.append(c);
  };
  let probing = false;
  const inspectAt = async (clientX: number, clientY: number) => {
    if (!kernel || !group || probing) return;
    const rect = ctx.canvas.getBoundingClientRect();
    const x = Math.floor(((clientX - rect.left) / rect.width) * accumulation.width), y = Math.floor(((clientY - rect.top) / rect.height) * accumulation.height);
    if (x < 0 || y < 0 || x >= accumulation.width || y >= accumulation.height) return;
    probing = true;
    try {
      device.queue.writeBuffer(probeBuffer, 0, new Float32Array([x, y]));
      const enc = device.createCommandEncoder({ label: 'probe' });
      kernel.dispatch(enc, 'probeRays', group, 1);
      enc.copyBufferToBuffer(probeBuffer, 0, probeRead, 0, 256);
      device.queue.submit([enc.finish()]);
      await probeRead.mapAsync(GPUMapMode.READ);
      const f = new Float32Array(probeRead.getMappedRange().slice(0));
      probeRead.unmap();
      hideOverlay();
      const camera = picker.orbit.camera;
      const hitT = f[2]!;
      const screenHit: [number, number] = [clientX, clientY];
      if (!(hitT >= 0)) {
        dotAt(screenHit, '#888');
        info.innerHTML = `<p>Pixel (${x}, ${y}): the camera ray hit nothing (the sky).</p>`;
        return;
      }
      const P: V3 = [f[4]!, f[5]!, f[6]!];
      // Long rays are drawn to a length the view can hold: a tenth of the camera distance per unit, at most.
      const far = Math.hypot(...sub(camera.position as V3, P)) * 0.6;
      const draw = (dir: V3, t: number, colour: string) => {
        const end = add(P, dir, t >= 0 ? t : far);
        const a = project(camera, rect, P), b = project(camera, rect, end);
        if (a && b) {
          line(a, b, colour, !(t >= 0));
          if (t >= 0) dotAt(b, colour, 3.5);
        }
      };
      const rows: string[] = [`<tr><td>camera ray</td><td>hit at t = ${fmt(hitT)}, material ${f[3]}</td></tr>`];
      const sunT = f[10]!;
      draw(toSun(), sunT, sunT >= 0 ? '#ff5d5d' : '#ffd166');
      rows.push(`<tr><td><i style="background:#ffd166"></i>shadow ray</td><td>${sunT >= 0 ? `blocked at t = ${fmt(sunT)}: in shadow` : 'reaches the sun: lit'}</td></tr>`);
      const reflDir: V3 = [f[11]!, f[12]!, f[13]!], reflT = f[14]!;
      draw(reflDir, reflT, '#5ec8ff');
      rows.push(`<tr><td><i style="background:#5ec8ff"></i>reflection ray</td><td>${reflT >= 0 ? `sees a surface at t = ${fmt(reflT)}` : 'sees the sky'}</td></tr>`);
      let blocked = 0;
      for (let k = 0; k < f[15]!; k++) {
        const t = f[19 + k * 4]!;
        if (t >= 0) blocked++;
        draw([f[16 + k * 4]!, f[17 + k * 4]!, f[18 + k * 4]!], t >= 0 ? t : params.aoRadius, t >= 0 ? '#c9a7ff' : 'rgba(201,167,255,0.45)');
      }
      rows.push(`<tr><td><i style="background:#c9a7ff"></i>AO rays</td><td>${blocked} of ${f[15]} blocked within ${fmt(params.aoRadius)}</td></tr>`);
      dotAt(screenHit, '#caf543', 5);
      info.innerHTML = `<table>${rows.join('')}</table><p>Solid lines end at the surface they hit; dashed ones leave the scene.</p>`;
      (globalThis as { __rays?: unknown }).__rays = { hitT, sunT, reflT, aoBlocked: blocked };
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
  ctx.onResize(hideOverlay);
  (globalThis as { __inspect?: unknown }).__inspect = inspectAt;

  ctx.loop((encoder) => {
    if (!kernel) return false;
    const k = kernel;
    return accumulation.frameWith(encoder, (enc, frame) => {
      if (bound !== accumulation.buffer) {
        group = device.createBindGroup({ layout: k.ioLayout, entries: [viewBuffer, accumulation.buffer!, probeBuffer, lightBuffer].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        bound = accumulation.buffer;
      }
      const s = toSun();
      device.queue.writeBuffer(lightBuffer, 0, new Float32Array([...s, params.sun, params.sky * 0.8, params.sky * 0.9, params.sky, params.reflect]));
      writeView(device, viewBuffer, picker.orbit.camera, accumulation.width, accumulation.height, frame, params.view | flags(), [params.aoRadius, params.distanceBand]);
      k.dispatch(enc, 'inspect', group!, Math.ceil(accumulation.width / 8), Math.ceil(accumulation.height / 8));
    });
  });
});
