/**
 * nrc-live (track N, N6b): the user-testable NRC demo. Plain path tracing and path tracing + the neural radiance cache
 * render the same view with the same kernel (the cache side only adds the hook), shown split, one side each.
 *
 *   pnpm --filter @pathtracer/bench nrc:live        (Chrome with --enable-unsafe-webgpu adds the subgroup-matrix network)
 *
 * Every frame: the cache trains (training paths, targets, steps) and renders one sample per pixel with the cache; the
 * plain side renders `plain samples / frame`. Benchmark times both with GPU timestamps (one readback at the end) and
 * sets the plain side's samples to equal time. Save A/B uploads both images as PFM next to the run (bench viewer).
 * No readback in the frame loop; at most one frame in flight.
 * Scenes: Sponza (sun + sky), Sponza with VkNRC lighting (`scene=sponza-white-vknrc`; export its glb first:
 * `node bench/adapters/export-scenes.ts sponza-white-vknrc`), Cornell.
 */
import { DEFAULT_SHAPE, type Precision } from '@pathtracer/mlp';
import { PtMegakernel, type PtLookAtCamera } from '@pmndrs/pathtracer';
import { PtMegakernelAdapter } from '../adapters/pt-megakernel/index.ts';
import { uploadImage } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { cornell } from '../scenes/tier0/index.ts';
import { sponzaWhite, sponzaWhiteVknrc } from '../scenes/tier2/sponza-white.ts';
import { NrcCache } from '../nrc/cache.ts';
import { hasSubgroupMatrix, inferArmOptions } from './mlp-common.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('canvas');
const statusEl = $<HTMLDivElement>('status');
const num = (id: string) => Number($<HTMLInputElement>(id).value);
const checked = (id: string) => $<HTMLInputElement>(id).checked;
const value = (id: string) => $<HTMLSelectElement>(id).value;

// sponza-white-vknrc: VkNRC's lighting (sky 10, no sun, 7 scattering vertices), for side-by-side views with the native baseline (N6.2).
const SCENES: Record<string, SceneDesc> = { 'sponza-white': sponzaWhite, 'sponza-white-vknrc': sponzaWhiteVknrc, cornell };

const DISPLAY_WGSL = /* wgsl */ `
struct D { size: vec2u, canvas: vec2f, split: f32, exposure: f32, mode: u32, _p: u32 }
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var<storage, read> plainAccum: array<vec4f>;
@group(0) @binding(2) var<storage, read> nrcAccum: array<vec4f>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
fn mean(v: vec4f) -> vec3f { return select(vec3f(0.0), v.xyz / max(v.w, 1.0), v.w > 0.0); }
// ACES fit (Narkowicz), then a 2.2 display curve.
fn tone(c: vec3f) -> vec3f {
  let x = max(c * d.exposure, vec3f(0.0));
  let t = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(t, vec3f(1.0 / 2.2));
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / d.canvas;
  let px = min(vec2u(uv * vec2f(d.size)), d.size - 1u);
  let i = px.y * d.size.x + px.x;
  let a = mean(plainAccum[i]);
  let b = mean(nrcAccum[i]);
  var c: vec3f;
  if (d.mode == 0u) {
    c = tone(select(b, a, uv.x < d.split));
    if (abs(uv.x - d.split) * d.canvas.x < 1.0) { c = vec3f(1.0); }
  } else if (d.mode == 1u) { c = tone(a); }
  else if (d.mode == 2u) { c = tone(b); }
  else { c = tone(abs(a - b) * 4.0); }
  return vec4f(c, 1.0);
}
`;

interface Live {
  adapter: PtMegakernelAdapter;
  device: GPUDevice;
  plain: PtMegakernel;
  cache: NrcCache;
  scene: SceneDesc;
  width: number;
  height: number;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  display: { pipeline: GPURenderPipeline; params: GPUBuffer; group: GPUBindGroup };
  context: GPUCanvasContext;
}

let live: Live | null = null;
let busy = false;
let building = false;
let needsReset = true;
let split = 0.5;
let lastFrameMs = 0;
let lastFrameEnd = 0;
let paused = false;
/** A new seed on every clear: clearing restarts the sample index, so without it every restarted image would show the same noise. */
let seed = 1;
let fps = 0;
const keys = new Set<string>();
const cam = { position: [0, 0, 0] as [number, number, number], yaw: 0, pitch: 0, yFovDeg: 60, speed: 1 };

function setStatus(text: string, error = false): void {
  statusEl.textContent = text;
  statusEl.classList.toggle('err', error);
}

function sceneBounds(adapter: PtMegakernelAdapter): Live['bounds'] {
  const { input } = adapter.debug;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const inst of input.instances) {
    const m = inst.objectToWorld;
    const p = input.meshes[inst.meshIndex]!.positions;
    for (let i = 0; i < p.length; i += 3) {
      const x = p[i]!, y = p[i + 1]!, z = p[i + 2]!;
      const w = [m[0]! * x + m[4]! * y + m[8]! * z + m[12]!, m[1]! * x + m[5]! * y + m[9]! * z + m[13]!, m[2]! * x + m[6]! * y + m[10]! * z + m[14]!];
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a]!, w[a]!);
        hi[a] = Math.max(hi[a]!, w[a]!);
      }
    }
  }
  return { min: lo as [number, number, number], max: hi as [number, number, number] };
}

function camera(): PtLookAtCamera {
  const d = [Math.cos(cam.pitch) * Math.cos(cam.yaw), Math.sin(cam.pitch), Math.cos(cam.pitch) * Math.sin(cam.yaw)];
  return { position: [...cam.position], target: [cam.position[0] + d[0]!, cam.position[1] + d[1]!, cam.position[2] + d[2]!], up: [0, 1, 0], yFovDeg: cam.yFovDeg };
}

function resetView(scene: SceneDesc, bounds: Live['bounds']): void {
  const c = scene.camera as PtLookAtCamera;
  cam.position = [...c.position] as [number, number, number];
  const d = [c.target[0] - c.position[0], c.target[1] - c.position[1], c.target[2] - c.position[2]];
  const l = Math.hypot(d[0]!, d[1]!, d[2]!);
  cam.yaw = Math.atan2(d[2]!, d[0]!);
  cam.pitch = Math.asin(d[1]! / l);
  cam.yFovDeg = c.yFovDeg;
  cam.speed = 0.15 * Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]);
}

function backendOptions(adapter: GPUAdapter, device: GPUDevice): void {
  const sel = $<HTMLSelectElement>('backend');
  const keep = sel.value;
  const options: [string, string][] = [];
  if (device.features.has('shader-f16')) options.push(['tiled-f16', 'Tiled f16 (no flag needed)']);
  options.push(['tiled-f32', 'Tiled f32 (no flag needed)']);
  if (device.features.has('shader-f16')) options.push(['portable-f16', 'Per-record f16 (naive)']);
  options.push(['portable-f32', 'Per-record f32 (naive)']);
  if (hasSubgroupMatrix(device)) {
    options.push(['sgm-4', 'Subgroup matrix, 4 subgroups']);
    options.push(['sgm-1', 'Subgroup matrix, 1 subgroup']);
  }
  // M7: the non-neural competitor in the network's place (same paths and training records, a world-space hash grid as the store).
  options.push(['hash', 'Hash grid (no network)']);
  sel.replaceChildren(...options.map(([v, t]) => new Option(t, v)));
  sel.value = options.some(([v]) => v === keep) ? keep : (options[0]?.[0] ?? 'portable-f32');
  void adapter;
}

async function buildCache(l: Pick<Live, 'adapter' | 'device' | 'plain' | 'width' | 'height' | 'bounds' | 'scene'>): Promise<NrcCache> {
  const backend = value('backend') || 'portable-f32';
  const gpuAdapter = l.adapter.gpuContext!.adapter;
  let precision: Precision = backend.endsWith('f32') ? 'f32' : 'f16';
  let sgm: ReturnType<typeof inferArmOptions>['sgm'];
  if (backend.startsWith('sgm')) {
    const arm = inferArmOptions('sgm', gpuAdapter, l.device, DEFAULT_SHAPE, 'f16', { records: 64, subgroups: backend === 'sgm-4' ? 4 : 1 });
    precision = arm.precision;
    sgm = arm.sgm;
  }
  const infer = backend.startsWith('tiled') ? 'tiled' : 'perRecord';
  const trainHeight = 72;
  const trainWidth = Math.max(4, Math.round((trainHeight * l.width) / l.height / 4) * 4);
  const { caps, trace, sceneBuffers } = l.adapter.debug;
  const cache = await NrcCache.create(l.device, caps, l.plain, [trace, sceneBuffers], { width: l.width, height: l.height, camera: camera(), ...limits(l.scene) }, l.bounds, {
    ...(backend === 'hash' ? { model: 'hash' as const } : {}),
    precision,
    ...(sgm ? { sgm } : {}),
    infer,
    queryDepth: Number(value('queryDepth')),
    trainDepth: num('trainDepth'),
    trainWidth,
    trainHeight,
    training: value('training') === 'replay' ? 'replay' : 'inline',
    trainSelect: value('trainSelect') === 'pixel' ? 'pixel' : 'tile',
    stepsPerFrame: num('steps'),
    learningRate: num('lr'),
  });
  return cache;
}

function displayGroup(l: Pick<Live, 'device' | 'plain' | 'cache'>, pipeline: GPURenderPipeline, params: GPUBuffer): GPUBindGroup {
  return l.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: l.plain.accumulator as GPUBuffer } }, { binding: 2, resource: { buffer: l.cache.accumulator } }],
  });
}

async function build(full: boolean): Promise<void> {
  if (building) return;
  building = true;
  try {
    const sceneId = value('scene');
    const [w, h] = value('res').split('x').map(Number) as [number, number];
    if (full || !live) {
      setStatus(`Loading ${sceneId} at ${w} × ${h}…`);
      live?.cache.destroy();
      live?.adapter.dispose();
      live = null;
      // Loaded unchanged (the exported glTF is keyed to the whole scene definition); film and depth are set on the kernels.
      const scene = SCENES[sceneId] as SceneDesc;
      const adapter = new PtMegakernelAdapter();
      await adapter.load(scene);
      const device = adapter.gpuContext!.device;
      device.addEventListener('uncapturederror', (e) => setStatus(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`, true));
      backendOptions(adapter.gpuContext!.adapter, device);
      const plain = adapter.renderer as PtMegakernel;
      plain.setFilm(w, h);
      plain.setLimits(limits(scene));
      const bounds = sceneBounds(adapter);
      resetView(scene, bounds);
      plain.setCamera(camera());
      setStatus('Compiling the cache kernels…');
      const cache = await buildCache({ adapter, device, plain, width: w, height: h, bounds, scene });
      canvas.width = w;
      canvas.height = h;
      const context = canvas.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: 'opaque' });
      const module = device.createShaderModule({ label: 'nrc display', code: DISPLAY_WGSL });
      const pipeline = await device.createRenderPipelineAsync({ label: 'nrc display', layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
      const params = device.createBuffer({ label: 'nrc display params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      live = { adapter, device, plain, cache, scene, width: w, height: h, bounds, display: { pipeline, params, group: displayGroup({ device, plain, cache }, pipeline, params) }, context };
    } else {
      setStatus('Rebuilding the cache…');
      const old = live.cache;
      const cache = await buildCache(live);
      old.destroy();
      live.cache = cache;
      live.display.group = displayGroup(live, live.display.pipeline, live.display.params);
    }
    needsReset = true;
    setStatus('');
  } catch (error) {
    setStatus(error instanceof Error ? `${error.message}` : String(error), true);
    console.error(error);
  } finally {
    building = false;
  }
}

function limits(scene: SceneDesc): { maxDepth: number; rrStartDepth?: number } {
  const rr = scene.integrator.rrStartDepth;
  return { maxDepth: num('maxDepth'), ...(rr ? { rrStartDepth: rr } : {}) };
}

function plainSpp(): number {
  return Math.max(1, Math.min(64, Math.round(num('plainSpp'))));
}

function encodeDisplay(l: Live, encoder: GPUCommandEncoder): void {
  const mode = { split: 0, plain: 1, nrc: 2, diff: 3 }[value('view')] ?? 0;
  const p = new ArrayBuffer(32);
  new Uint32Array(p, 0, 2).set([l.width, l.height]);
  new Float32Array(p, 8, 4).set([l.width, l.height, split, 2 ** num('exposure')]);
  new Uint32Array(p, 24, 2).set([mode, 0]);
  l.device.queue.writeBuffer(l.display.params, 0, p);
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view: l.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
  pass.setPipeline(l.display.pipeline);
  pass.setBindGroup(0, l.display.group);
  pass.draw(3);
  pass.end();
  $('tagLeft').hidden = mode !== 0 && mode !== 1;
  $('tagRight').hidden = mode !== 0 && mode !== 2;
  $('tagLeft').textContent = mode === 3 ? '' : `Path tracing · ${plainSpp()} spp/frame`;
}

function moveCamera(dt: number): boolean {
  if (keys.size === 0) return false;
  const f = [Math.cos(cam.yaw), 0, Math.sin(cam.yaw)];
  const r = [-Math.sin(cam.yaw), 0, Math.cos(cam.yaw)];
  const s = cam.speed * dt * (keys.has('shift') ? 3 : 1);
  const mv = [0, 0, 0];
  const add = (v: number[], k: number) => v.forEach((c, i) => (mv[i]! += c * k));
  if (keys.has('w')) add(f, s);
  if (keys.has('s')) add(f, -s);
  if (keys.has('d')) add(r, s);
  if (keys.has('a')) add(r, -s);
  if (keys.has('e')) mv[1]! += s;
  if (keys.has('q')) mv[1]! -= s;
  if (mv.every((v) => v === 0)) return false;
  cam.position = cam.position.map((v, i) => v + mv[i]!) as [number, number, number];
  return true;
}

function frame(now: number): void {
  requestAnimationFrame(frame);
  const l = live;
  if (!l || busy || building || paused) return;
  const dt = lastFrameEnd ? Math.min(0.1, (now - lastFrameEnd) / 1000) : 0;
  if (moveCamera(dt)) needsReset = true;
  const view = value('view');
  const encoder = l.device.createCommandEncoder({ label: 'nrc live frame' });
  if (needsReset || !checked('accumulate')) {
    l.plain.setCamera(camera());
    l.cache.setCamera(camera());
    seed++;
    l.plain.setSeed(seed);
    l.cache.render.setSeed(seed);
    l.plain.reset(encoder);
    l.cache.reset(encoder);
    needsReset = false;
  }
  // Only the side on screen renders, so "With NRC only" and "Path tracing only" show each side's own frame time.
  if (view !== 'plain') {
    l.cache.settings.stepsPerFrame = Math.max(0, Math.round(num('steps')));
    l.cache.encodeFrame(encoder, { train: checked('train') && l.cache.settings.stepsPerFrame > 0 });
  }
  if (view !== 'nrc') l.plain.encode(encoder, plainSpp());
  encodeDisplay(l, encoder);
  l.device.queue.submit([encoder.finish()]);
  busy = true;
  const t0 = performance.now();
  void l.device.queue.onSubmittedWorkDone().then(() => {
    busy = false;
    const end = performance.now();
    lastFrameMs = end - t0;
    if (lastFrameEnd) fps = fps ? 0.9 * fps + 0.1 * (1000 / (end - lastFrameEnd)) : 1000 / (end - lastFrameEnd);
    lastFrameEnd = end;
    if (!statusEl.classList.contains('err')) setStatus(`${l.scene.name} · ${l.width} × ${l.height} · ${fps.toFixed(0)} fps (GPU ${lastFrameMs.toFixed(1)} ms) · plain ${l.plain.samples} spp · NRC ${l.cache.samples} spp · ${l.cache.description}`);
  });
}

/**
 * Times each stage by submitting it alone and waiting for it (wall clock per submit, about ±0.3 ms; GPU timestamps on
 * ~20 passes per frame lost the Metal device). Median of 20 frames after 3 warm-up frames.
 */
async function benchmark(): Promise<void> {
  const l = live;
  if (!l) return;
  const table = $('timings').querySelector('tbody') as HTMLTableSectionElement;
  building = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    const train = checked('train') && l.cache.settings.stepsPerFrame > 0;
    const stages = l.cache.stages.filter((st) => train || (st !== 'trainingPaths' && st !== 'trainSteps'));
    const times: Record<string, number[]> = { plain: [] };
    for (const st of stages) times[st] = [];
    const run = async (encode: (e: GPUCommandEncoder) => void): Promise<number> => {
      const e = l.device.createCommandEncoder({ label: 'nrc benchmark' });
      encode(e);
      const t0 = performance.now();
      l.device.queue.submit([e.finish()]);
      await l.device.queue.onSubmittedWorkDone();
      return performance.now() - t0;
    };
    for (let f = 0; f < 23; f++) {
      l.cache.nextFrame();
      for (const st of stages) {
        const ms = await run((e) => l.cache.encodeStage(e, st));
        if (f >= 3) times[st]!.push(ms);
      }
      const ms = await run((e) => l.plain.encode(e, plainSpp()));
      if (f >= 3) times.plain!.push(ms);
    }
    const med = (v: number[] | undefined) => {
      if (!v || !v.length) return 0;
      const s = [...v].sort((x, y) => x - y);
      return s[Math.floor(s.length / 2)] as number;
    };
    const trainerMs = med(times.trainingPaths);
    const trainMs = med(times.trainSteps);
    const renderMs = med(times.renderPass);
    const inferMs = med(times.inference);
    const plainMs = med(times.plain);
    const nrcMs = trainerMs + trainMs + renderMs + inferMs;
    const perSpp = plainMs / plainSpp();
    const equal = Math.max(1, Math.round(nrcMs / perSpp));
    (window as unknown as { __nrcBench: unknown }).__nrcBench = { trainerMs, trainMs, renderMs, inferMs, nrcMs, plainMs, perSpp, equal, backend: `${l.cache.description}` };
    table.replaceChildren(
      row('Training paths + targets', trainerMs),
      row('Training steps', trainMs),
      row('Render pass (cache side)', renderMs),
      row('Inference + resolve', inferMs),
      row('Cache side, total', nrcMs),
      row(`Plain, ${plainSpp()} spp`, plainMs),
      row('Plain, per sample', perSpp),
      row('Equal-time plain spp', String(equal)),
    );
    if (checked('equalTime')) {
      $<HTMLInputElement>('plainSpp').value = String(equal);
      needsReset = true;
    }
  } finally {
    building = false;
  }
}

function row(label: string, v: number | string): HTMLTableRowElement {
  const tr = document.createElement('tr');
  const a = document.createElement('td');
  a.textContent = label;
  const b = document.createElement('td');
  b.className = 'v';
  b.textContent = typeof v === 'number' ? `${v.toFixed(2)} ms` : v;
  tr.append(a, b);
  return tr;
}

async function saveAB(): Promise<void> {
  const l = live;
  if (!l) return;
  building = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    const [a, b] = await Promise.all([l.plain.readImage(), l.cache.render.readImage()]);
    const tag = `${l.scene.name}-${l.width}x${l.height}`;
    await uploadImage(`${tag}-plain-${a.samples}spp`, { width: a.width, height: a.height, channels: 4, data: a.data });
    await uploadImage(`${tag}-nrc-${b.samples}spp`, { width: b.width, height: b.height, channels: 4, data: b.data });
    $('saveNote').textContent = `Saved ${tag} plain (${a.samples} spp) and NRC (${b.samples} spp) as PFM in bench/results/incoming/.`;
  } catch (error) {
    $('saveNote').textContent = `Save failed: ${error instanceof Error ? error.message : String(error)}. Saving needs the page served by nrc:live.`;
  } finally {
    building = false;
  }
}

// Input.
canvas.addEventListener('pointerdown', (e) => canvas.setPointerCapture(e.pointerId));
canvas.addEventListener('pointermove', (e) => {
  if (!e.buttons) return;
  if (e.shiftKey) {
    const r = canvas.getBoundingClientRect();
    split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    return;
  }
  cam.yaw += e.movementX * 0.004;
  cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch - e.movementY * 0.004));
  needsReset = true;
});
window.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).closest('input, select')) return;
  keys.add(e.key.toLowerCase());
});
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener('blur', () => keys.clear());

// A scene brings its own path length (sponza-white 8, the VkNRC-lit variant 7); the field can still be changed after.
const sceneDepth = (): void => {
  $<HTMLInputElement>('maxDepth').value = String((SCENES[$<HTMLSelectElement>('scene').value] as SceneDesc).integrator.maxDepth);
};
$('scene').addEventListener('change', sceneDepth);
for (const id of ['scene', 'res', 'maxDepth']) $(id).addEventListener('change', () => void build(true));
for (const id of ['backend', 'lr', 'trainDepth', 'training', 'trainSelect']) $(id).addEventListener('change', () => void build(false));
$('queryDepth').addEventListener('change', () => {
  live?.cache.setQueryDepth(Number(value('queryDepth')));
  needsReset = true;
});
$('plainSpp').addEventListener('change', () => (needsReset = true));
$('accumulate').addEventListener('change', () => (needsReset = true));
$('resetImage').addEventListener('click', () => (needsReset = true));
$('resetCache').addEventListener('click', () => {
  live?.cache.resetNetwork();
  needsReset = true;
});
$('resetView').addEventListener('click', () => {
  if (live) resetView(live.scene, live.bounds);
  needsReset = true;
});
$('bench').addEventListener('click', () => void benchmark());
$('pause').addEventListener('click', () => {
  paused = !paused;
  $('pause').textContent = paused ? 'Resume' : 'Pause';
  lastFrameEnd = 0;
  fps = 0;
});
/** A setup that runs at interactive rates on the M5 Pro: 640 × 360, the cache's view only, 2 training steps per frame. */
$('realtime').addEventListener('click', () => {
  const res = $<HTMLSelectElement>('res');
  const rebuild = res.value !== '640x360';
  res.value = '640x360';
  $<HTMLSelectElement>('view').value = 'nrc';
  $<HTMLInputElement>('steps').value = '2';
  const backend = $<HTMLSelectElement>('backend');
  if ([...backend.options].some((o) => o.value === 'sgm-4')) backend.value = 'sgm-4';
  if (rebuild) void build(true);
  else void build(false);
});
$('save').addEventListener('click', () => void saveAB());

const args = new URLSearchParams(location.search);
if (args.get('scene') && SCENES[args.get('scene') as string]) {
  $<HTMLSelectElement>('scene').value = args.get('scene') as string;
  sceneDepth();
}
await build(true);
requestAnimationFrame(frame);
// Headless checks and the runner: frame count and a benchmark hook.
/** Debug and headless use: renders n frames of the current view without requestAnimationFrame (a hidden window pauses rAF). */
async function step(n: number): Promise<void> {
  const l = live;
  if (!l) return;
  building = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    const view = value('view');
    for (let i = 0; i < n; i++) {
      const encoder = l.device.createCommandEncoder({ label: 'nrc step' });
      if (needsReset || !checked('accumulate')) {
        l.plain.setCamera(camera());
        l.cache.setCamera(camera());
        l.plain.reset(encoder);
        l.cache.reset(encoder);
        needsReset = false;
      }
      if (view !== 'plain') l.cache.encodeFrame(encoder, { train: checked('train') && l.cache.settings.stepsPerFrame > 0 });
      if (view !== 'nrc') l.plain.encode(encoder, plainSpp());
      l.device.queue.submit([encoder.finish()]);
      await l.device.queue.onSubmittedWorkDone();
    }
  } finally {
    building = false;
  }
}

/** Debug only (pauses the loop, one readback): mean radiance of both images and the cache's record counts. */
async function means(): Promise<Record<string, number> | null> {
  const l = live;
  if (!l) return null;
  building = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    const [a, b] = await Promise.all([l.plain.readImage(), l.cache.render.readImage()]);
    const mean = (d: Float32Array) => {
      let s = 0;
      for (let i = 0; i < d.length; i += 4) s += 0.2126 * d[i]! + 0.7152 * d[i + 1]! + 0.0722 * d[i + 2]!;
      return s / (d.length / 4);
    };
    const counts = await l.cache.readTrainCount();
    return { plain: mean(a.data), nrc: mean(b.data), plainSpp: a.samples, nrcSpp: b.samples, trainRecords: counts.records, queries: await l.cache.readQueryCount() };
  } finally {
    building = false;
  }
}
/** Debug: point the camera (position, target) and clear both images. */
function setView(position: [number, number, number], target: [number, number, number]): void {
  cam.position = [...position];
  const d = [target[0] - position[0], target[1] - position[1], target[2] - position[2]];
  cam.yaw = Math.atan2(d[2]!, d[0]!);
  cam.pitch = Math.asin(d[1]! / Math.hypot(d[0]!, d[1]!, d[2]!));
  needsReset = true;
}
(window as unknown as { __nrcLive: unknown }).__nrcLive = { benchmark, means, step, setView, reset: () => { needsReset = true; }, pause: (p: boolean) => { paused = p; }, get fps() { return fps; }, get samples() { return live ? { plain: live.plain.samples, nrc: live.cache.samples } : null; }, status: () => statusEl.textContent };
