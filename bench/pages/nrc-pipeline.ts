/**
 * nrc-pipeline (track N, M9; docs/design/nrc-pipeline.md): the neural radiance cache as the low-noise input of a real-time chain.
 *
 *   node bench/live.ts nrc-pipeline          (then open the printed URL in Chrome)
 *
 * Both arms render at 960 x 540 in frame mode and go through the IDENTICAL chain: pack -> Denoiser `DiffuseSpecularDenoiser` (SVGF,
 * remodulated) -> composite -> pmndrs/upscaler temporal path (jitter: false; the path tracer's sampler owns the jitter) -> 1920 x 1080.
 * Left: plain path tracing at N spp per frame (Benchmark sets N for equal time when "Equal time" is on); right: path tracing + cache,
 * one sample per pixel per frame. The stage selector shows the raw frame, the denoised frame (render resolution) or the final output.
 * Denoiser: SVGF (temporal), or OIDN from the Denoiser package (mutracer-style: the small network every frame while the camera moves, the
 * standard network once after `stillSamples` still frames, then the frame is held), OIDN fast / standard every frame, or none.
 * No readback in the frame loop; one frame in flight. Headless twin: nrc-pipeline-study.
 */
import { setupDaylight, updateDaylight } from '../nrc/daylight-controls.ts';
import { isInterior, loadInterior } from '../scenes/interiors.ts';
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { uploadImage } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { cornell } from '../scenes/tier0/index.ts';
import { nrcMaterialsShadow } from '../scenes/tier2/nrc-materials.ts';
import { phase10Settings } from '../nrc/phase10.ts';
import { sponzaWhite } from '../scenes/tier2/sponza-white.ts';
import { NrcPipelineSession, type ArmName } from '../nrc/pipeline-session.ts';
import { readTextureRgba16f, type PipelineStage } from '../realtime/chain.ts';
import { materialArgs } from '../nrc/cache.ts';
import { hasSubgroupMatrix, networkArg } from './mlp-common.ts';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('canvas');
const statusEl = $<HTMLDivElement>('status');
const num = (id: string) => Number($<HTMLInputElement>(id).value);
const checked = (id: string) => $<HTMLInputElement>(id).checked;
const value = (id: string) => $<HTMLSelectElement>(id).value;

const SCENES: Record<string, SceneDesc> = { 'sponza-white': sponzaWhite, [nrcMaterialsShadow.name]: nrcMaterialsShadow, cornell };

const DISPLAY_WGSL = /* wgsl */ `
struct D { canvas: vec2f, split: f32, exposure: f32, mode: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> d: D;
@group(0) @binding(1) var left: texture_2d<f32>;
@group(0) @binding(2) var right: texture_2d<f32>;
@group(0) @binding(3) var smp: sampler;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
// ACES fit (Narkowicz), then a 2.2 display curve (the same as nrc-live).
fn tone(c: vec3f) -> vec3f {
  let x = max(c * d.exposure, vec3f(0.0));
  let t = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(t, vec3f(1.0 / 2.2));
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let uv = pos.xy / d.canvas;
  let a = textureSampleLevel(left, smp, uv, 0.0).xyz;
  let b = textureSampleLevel(right, smp, uv, 0.0).xyz;
  var c: vec3f;
  if (d.mode == 0u) {
    c = tone(select(b, a, uv.x < d.split));
    if (abs(uv.x - d.split) * d.canvas.x < 1.5) { c = vec3f(1.0); }
  } else if (d.mode == 1u) { c = tone(a); } else { c = tone(b); }
  return vec4f(c, 1.0);
}
`;

interface Live {
  session: NrcPipelineSession;
  scene: SceneDesc;
  context: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  params: GPUBuffer;
  sampler: GPUSampler;
  groups: Map<PipelineStage, { left: GPUTexture; right: GPUTexture; group: GPUBindGroup }>;
}

let live: Live | null = null;
let busy = false;
let building = false;
let paused = false;
let split = 0.5;
let fps = 0;
let lastFrameEnd = 0;
let lastFrameMs = 0;
const keys = new Set<string>();
const cam = { position: [0, 0, 0] as [number, number, number], yaw: 0, pitch: 0, yFovDeg: 60, speed: 1 };

function setStatus(text: string, error = false): void {
  statusEl.textContent = text;
  statusEl.classList.toggle('err', error);
}

function camera(): PtLookAtCamera {
  const d = [Math.cos(cam.pitch) * Math.cos(cam.yaw), Math.sin(cam.pitch), Math.cos(cam.pitch) * Math.sin(cam.yaw)];
  return { position: [...cam.position], target: [cam.position[0] + d[0]!, cam.position[1] + d[1]!, cam.position[2] + d[2]!], up: [0, 1, 0], yFovDeg: cam.yFovDeg };
}

function resetView(scene: SceneDesc, bounds: NrcPipelineSession['bounds']): void {
  const c = scene.camera as PtLookAtCamera;
  cam.position = [...c.position] as [number, number, number];
  const d = [c.target[0] - c.position[0], c.target[1] - c.position[1], c.target[2] - c.position[2]];
  cam.yaw = Math.atan2(d[2]!, d[0]!);
  cam.pitch = Math.asin(d[1]! / Math.hypot(d[0]!, d[1]!, d[2]!));
  cam.yFovDeg = c.yFovDeg;
  cam.speed = 0.15 * Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]);
}

function backendOptions(device: GPUDevice): void {
  const sel = $<HTMLSelectElement>('backend');
  const keep = sel.value;
  const options: [string, string][] = [];
  if (device.features.has('shader-f16')) options.push(['tiled-f16', 'Tiled f16'], ['portable-f16', 'Per-record f16']);
  options.push(['tiled-f32', 'Tiled f32'], ['portable-f32', 'Per-record f32']);
  if (hasSubgroupMatrix(device)) options.push(['sgm-f16', 'Subgroup matrix, 4 subgroups']);
  // M7 / N7c: the hash grid in the network's place (same paths and training records).
  options.push(['hash', 'Hash grid (no network)']);
  sel.replaceChildren(...options.map(([v, t]) => new Option(t, v)));
  sel.value = options.some(([v]) => v === keep) ? keep : (options[0]?.[0] as string);
}

async function build(): Promise<void> {
  if (building) return;
  building = true;
  const rebuildControls = ['scene', 'res', 'backend', 'trainLayout', 'maxDepth', 'displayScale'];
  for (const id of rebuildControls) $<HTMLInputElement>(id).disabled = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    const sceneId = value('scene');
    const [w, h] = value('res').split('x').map(Number) as [number, number];
    setStatus(`Loading ${sceneId} at ${w} × ${h}, compiling the kernels…`);
    const sameScene = live?.scene.name === sceneId;
    live?.params.destroy();
    live?.session.destroy();
    live = null;
    const scene = isInterior(sceneId) ? await loadInterior(sceneId) : SCENES[sceneId] as SceneDesc;
    const backend = value('backend') || 'tiled-f16';
    const [kind, prec] = backend.split('-') as [string, string];
    const session = await NrcPipelineSession.create({
      scene,
      width: w,
      height: h,
      displayWidth: num('displayScale') * w,
      displayHeight: num('displayScale') * h,
      ...(num('maxDepth') ? { maxDepth: Math.max(1, Math.min(32, Math.round(num('maxDepth')))) } : {}),
      infer: kind === 'sgm' ? 'sgm' : kind === 'portable' ? 'portable' : 'tiled',
      precision: prec === 'f32' ? 'f32' : 'f16',
      cache: {
        network,
        ...(backend === 'hash' ? { model: 'hash' as const } : preset ?? materialArgs(args.get('factorization'), args.get('materialInputs'), args.get('directionInput'))),
        trainSelect: value('trainLayout') === 'pixel' ? 'pixel' : 'tile',
        stepsPerFrame: num('steps'),
      },
    });
    const device = session.device;
    device.addEventListener('uncapturederror', (e) => setStatus(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`, true));
    backendOptions(device);
    session.plainSpp = Math.max(1, Math.round(num('plainSpp')));
    if (!sameScene) resetView(scene, session.bounds);
    canvas.width = session.options.displayWidth;
    canvas.height = session.options.displayHeight;
    $<HTMLInputElement>('maxDepth').placeholder = String(scene.integrator.maxDepth);
    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    const module = device.createShaderModule({ label: 'nrc pipeline display', code: DISPLAY_WGSL });
    const pipeline = await device.createRenderPipelineAsync({ label: 'nrc pipeline display', layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    const params = device.createBuffer({ label: 'nrc pipeline display params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    live = { session, scene, context, pipeline, params, sampler, groups: new Map() };
    applyDenoiser();
    applyReconstruction();
    $('daylight').hidden = !isInterior(sceneId);
    if (isInterior(sceneId)) {
      if (sameScene) updateDaylight(session);
      else setupDaylight(session);
    }
    session.cutNext();
    setStatus('');
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
    console.error(error);
  } finally {
    for (const id of rebuildControls) $<HTMLInputElement>(id).disabled = false;
    building = false;
  }
}

function displayGroup(l: Live, stage: PipelineStage): GPUBindGroup {
  // The upscaler's output and the denoiser-fed colour are fixed textures; the group is rebuilt only if the texture object changes.
  const left = l.session.arms.plain.stage(stage);
  const right = l.session.arms.nrc.stage(stage);
  const cached = l.groups.get(stage);
  if (cached?.left === left && cached.right === right) return cached.group;
  const group = l.session.device.createBindGroup({ layout: l.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: l.params } }, { binding: 1, resource: left.createView() }, { binding: 2, resource: right.createView() }, { binding: 3, resource: l.sampler }] });
  l.groups.set(stage, { left, right, group });
  return group;
}

function encodeDisplay(l: Live): void {
  const mode = { split: 0, plain: 1, nrc: 2 }[value('view')] ?? 0;
  const p = new ArrayBuffer(32);
  new Float32Array(p, 0, 4).set([canvas.width, canvas.height, split, 2 ** num('exposure')]);
  new Uint32Array(p, 16, 1).set([mode]);
  l.session.device.queue.writeBuffer(l.params, 0, p);
  const encoder = l.session.device.createCommandEncoder({ label: 'nrc pipeline display' });
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view: l.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
  pass.setPipeline(l.pipeline);
  pass.setBindGroup(0, displayGroup(l, value('stageSel') as PipelineStage));
  pass.draw(3);
  pass.end();
  l.session.device.queue.submit([encoder.finish()]);
  $('tagLeft').hidden = mode === 2;
  $('tagRight').hidden = mode === 1;
  const samples = l.session.samples;
  $('tagLeft').textContent = `Plain PT · ${samples.plain} spp`;
  const c = l.session.cache;
  $('tagRight').textContent = c.warming ? `NRC warming · plain PT · ${samples.nrc} spp` : `Path tracing + NRC · ${samples.nrc} spp`;
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

function armsOnScreen(): ArmName[] {
  const v = value('view');
  return v === 'plain' ? ['plain'] : v === 'nrc' ? ['nrc'] : ['plain', 'nrc'];
}

/**
 * Warm-up gate for the current denoiser (N11.7, D30): a training-step budget, the point where the cache stops being worse than plain
 * through that denoiser on any metric (NRC-PIPELINE.md "Warm-up gate"): SVGF 120 steps, OIDN 600, no denoiser 12. In frames: the budget over
 * the steps per gated frame.
 */
function autoGateFrames(denoiser: string, stepsPerGatedFrame: number): number {
  const steps = denoiser === 'svgf' ? 120 : denoiser.startsWith('oidn') ? 600 : 12;
  return Math.ceil(steps / Math.max(1, stepsPerGatedFrame));
}

function gateFrames(): number {
  const g = value('gate');
  const perFrame = num('gateSteps') > 0 ? num('gateSteps') : num('steps');
  if (g === 'off') return 0;
  if (g === 'custom') return Math.max(0, Math.round(num('gateFrames')));
  // The hash grid updates once per frame and is ahead of plain after ~10 frames under both denoisers (N7c, NRC-PIPELINE.md).
  if (value('backend') === 'hash') return 10;
  return autoGateFrames(value('denoiserSel'), perFrame);
}

/** Training steps and the warm-up gate from the controls (every frame: rAF loop and `step`). */
function applyCacheSettings(session: NrcPipelineSession): void {
  session.cache.settings.stepsPerFrame = Math.max(0, Math.round(num('steps')));
  session.accumulateStill = checked('accumulateStill');
  session.svgfHistoryLimit = Math.max(0, Math.min(4096, Math.round(num('svgfHistoryLimit'))));
  session.nrcSpp = Math.max(1, Math.min(16, Math.round(num('nrcSpp')) || 1));
  session.setGate(gateFrames(), num('gateSteps'));
}

function frame(now: number): void {
  requestAnimationFrame(frame);
  const l = live;
  if (!l || busy || building || paused) return;
  const dt = lastFrameEnd ? Math.min(0.1, (now - lastFrameEnd) / 1000) : 0;
  moveCamera(dt);
  applyCacheSettings(l.session);
  // Only the arms on screen render, so a single side shows its own frame rate.
  busy = true;
  const t0 = performance.now();
  // OIDN is awaited inside frame(); one frame in flight.
  void l.session
    .frame(camera(), { arms: armsOnScreen(), train: checked('train') })
    .then(() => {
      encodeDisplay(l);
      return l.session.device.queue.onSubmittedWorkDone();
    })
    .then(() => {
      busy = false;
      const end = performance.now();
      lastFrameMs = end - t0;
      if (lastFrameEnd) fps = fps ? 0.9 * fps + 0.1 * (1000 / (end - lastFrameEnd)) : 1000 / (end - lastFrameEnd);
      lastFrameEnd = end;
      const den = l.session.arms.nrc.lastDenoiser || l.session.arms.plain.lastDenoiser;
      const gate = l.session.cache.warming ? ` · NRC gate closed: plain while training (${l.session.cache.trained.frames}/${l.session.cache.settings.warmupFrames})` : '';
      if (hero && !statusEl.classList.contains('err')) {
        setStatus(`${armsOnScreen().includes('nrc') && checked('train') && num('steps') > 0 ? 'Training live' : 'Training idle'} · ${l.session.cache.trained.steps} updates · ${l.session.options.width} × ${l.session.options.height} → ${canvas.width} × ${canvas.height} · ${fps.toFixed(0)} fps`);
      } else if (!statusEl.classList.contains('err')) setStatus(`${l.scene.name} · ${l.session.options.width} × ${l.session.options.height} → ${canvas.width} × ${canvas.height} · ${fps.toFixed(0)} fps (wait ${lastFrameMs.toFixed(1)} ms) · plain ${l.session.plainSpp} spp · ${l.session.inferDescription} · ${den}${gate}`);
    })
    .catch((error: unknown) => {
      busy = false;
      setStatus(error instanceof Error ? error.message : String(error), true);
    });
}

async function benchmark(): Promise<void> {
  const l = live;
  if (!l) return;
  const table = $('timings').querySelector('tbody') as HTMLTableSectionElement;
  building = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    const t = await l.session.timeStages(20);
    (window as unknown as { __nrcPipelineBench: unknown }).__nrcPipelineBench = t;
    table.replaceChildren(
      row('Plain, 1 spp', t.plainSpp!),
      row('Cache frame (render + train + infer)', t.cacheFrame!),
      row('Pack + depth', t.plainPack!),
      row('Denoise + composite', t.denoiseComposite!),
      row('Upscale', t.upscale!),
      row('Equal-time plain spp', String(t.equalSpp)),
      row('Plain arm, whole chain', t.plainArm!),
      row('NRC arm, whole chain', t.nrcArm!),
    );
    if (checked('equalTime')) {
      $<HTMLInputElement>('plainSpp').value = String(t.equalSpp);
      l.session.plainSpp = t.equalSpp as number;
    }
    l.session.cutNext();
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
    const tag = `${l.scene.name}-${l.session.options.width}x${l.session.options.height}`;
    for (const arm of ['plain', 'nrc'] as const)
      for (const stage of ['raw', 'denoised', 'final'] as const) {
        const img = await readTextureRgba16f(l.session.device, l.session.arms[arm].stage(stage));
        await uploadImage(`${tag}-${arm}-${stage}`, img);
      }
    $('saveNote').textContent = `Saved raw, denoised and final of both arms (${tag}) as PFM in bench/results/incoming/.`;
  } catch (error) {
    $('saveNote').textContent = `Save failed: ${error instanceof Error ? error.message : String(error)}. Saving needs the page served by bench/live.ts.`;
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
});
window.addEventListener('keydown', (e) => {
  if ((e.target as HTMLElement).closest('input, select')) return;
  keys.add(e.key.toLowerCase());
});
window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
window.addEventListener('blur', () => keys.clear());

$('view').addEventListener('change', () => live?.session.cutNext());
for (const id of ['scene', 'res', 'backend', 'trainLayout', 'maxDepth', 'displayScale']) $(id).addEventListener('change', () => void build());
/** The denoiser selector: both arms switch together (same chain). */
function applyDenoiser(): void {
  const v = value('denoiserSel');
  const stillSamples = Math.max(1, Math.round(num('stillSamples')));
  const aux = value('oidnAux') === 'albedo' ? 'albedo' : 'albedo+normal';
  if (v === 'oidn') live?.session.setDenoiser('oidn', { aux, stillSamples, hold: !hero });
  else if (v === 'oidn-fast') live?.session.setDenoiser('oidn', { aux, moving: 'fast', stillSamples: Infinity, hold: false });
  else if (v === 'oidn-std') live?.session.setDenoiser('oidn', { aux, moving: 'balanced', stillSamples: Infinity, hold: false });
  else live?.session.setDenoiser(v === 'none' ? 'none' : 'svgf');
}
for (const id of ['denoiserSel', 'oidnAux', 'stillSamples']) $(id).addEventListener('change', applyDenoiser);
$('plainSpp').addEventListener('change', () => {
  if (live) live.session.plainSpp = Math.max(1, Math.min(16, Math.round(num('plainSpp')) || 1));
});
$('resetCache').addEventListener('click', () => { live?.session.cache.resetNetwork(); live?.session.cutNext(); });
$('fireflyClamp').addEventListener('change', () => live?.session.cutNext());
for (const id of ['clearImage', 'accumulateStill', 'svgfHistoryLimit']) $(id).addEventListener(id === 'clearImage' ? 'click' : 'change', () => live?.session.cutNext());
function applyReconstruction(): void {
  const path = value('upscalePath') as 'spatial' | 'bilinear' | 'temporal';
  for (const id of ['sharpness', 'rcasDenoise']) $<HTMLInputElement>(id).disabled = path === 'bilinear';
  $<HTMLInputElement>('upHistory').disabled = path !== 'temporal';
  $<HTMLInputElement>('svgfHistoryLimit').disabled = value('denoiserSel') !== 'svgf';
  $<HTMLInputElement>('svgfAlpha').disabled = value('denoiserSel') !== 'svgf';
  $<HTMLSelectElement>('oidnAux').disabled = !value('denoiserSel').startsWith('oidn');
  $<HTMLInputElement>('stillSamples').disabled = value('denoiserSel') !== 'oidn';
  if (!live) return;
  for (const arm of Object.values(live.session.arms)) {
    arm.setFireflyClamp(Math.max(0, num('fireflyClamp')));
    arm.setSvgfAlpha(Math.max(0.01, Math.min(1, num('svgfAlpha') || 0.1)));
    arm.upscalePath = path;
    Object.assign(arm.upscaler.settings, { sharpness: Math.min(1, Math.max(0, num('sharpness'))), maxAccumulation: Math.max(1, Math.min(64, num('upHistory'))), rcasDenoise: checked('rcasDenoise') });
    arm.resetHistory();
  }
}
for (const id of ['denoiserSel', 'upscalePath', 'sharpness', 'upHistory', 'rcasDenoise', 'svgfAlpha', 'fireflyClamp']) $(id).addEventListener('change', applyReconstruction);
$('resetView').addEventListener('click', () => {
  if (live) resetView(live.scene, live.session.bounds);
  live?.session.cutNext();
});
$('bench').addEventListener('click', () => void benchmark());
$('save').addEventListener('click', () => void saveAB());
$('pause').addEventListener('click', () => {
  paused = !paused;
  $('pause').textContent = paused ? 'Resume' : 'Pause';
  lastFrameEnd = 0;
  fps = 0;
});

const args = new URLSearchParams(location.search);
const hero = args.get('demo') === 'hero' ? await import('../hero/showcase.ts') : undefined;
hero?.setupHero();
for (const id of ['sunStrength', 'sunAzimuth', 'sunElevation', 'sunRadius']) $(id).addEventListener('input', () => {
  if (live && isInterior(live.scene.name)) updateDaylight(live.session);
});
const preset = hero || args.get('phase10') === 'mlp' ? phase10Settings('mlp') : undefined;
const network = preset?.network ?? networkArg(args.get('network'));
if (preset) $<HTMLInputElement>('steps').value = String(preset.stepsPerFrame);
if (args.has('svgfHistory')) $<HTMLInputElement>('svgfHistoryLimit').value = args.get('svgfHistory')!;
const stage = args.get('stage');
if (stage === 'denoised' || stage === 'raw' || stage === 'final') $<HTMLSelectElement>('stageSel').value = stage;
if (!hero && args.get('scene') && [...$<HTMLSelectElement>('scene').options].some(option => option.value === args.get('scene'))) $<HTMLSelectElement>('scene').value = args.get('scene') as string;
if (args.get('res')) $<HTMLSelectElement>('res').value = args.get('res') as string;
// ?denoiser=svgf|oidn|oidn-fast|oidn-std|none  ?oidnAux=albedo  ?stillSamples=8
if (args.get('denoiser')) $<HTMLSelectElement>('denoiserSel').value = args.get('denoiser') as string;
if (args.get('oidnAux')) $<HTMLSelectElement>('oidnAux').value = args.get('oidnAux') as string;
if (args.get('stillSamples')) $<HTMLInputElement>('stillSamples').value = args.get('stillSamples') as string;
// ?gate=auto|off|<frames>  ?gateSteps=<n>
const gateArg = args.get('gate');
if (gateArg === 'auto' || gateArg === 'off') $<HTMLSelectElement>('gate').value = gateArg;
else if (gateArg) {
  $<HTMLSelectElement>('gate').value = 'custom';
  $<HTMLInputElement>('gateFrames').value = gateArg;
}
if (args.get('gateSteps')) $<HTMLInputElement>('gateSteps').value = args.get('gateSteps') as string;
await build();
requestAnimationFrame(frame);

/** Headless use (a hidden window pauses rAF): n frames of the current view, waited for. */
async function step(n: number): Promise<void> {
  const l = live;
  if (!l) return;
  building = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    for (let i = 0; i < n; i++) {
      applyCacheSettings(l.session);
      await l.session.frame(camera(), { arms: armsOnScreen(), train: checked('train') });
      encodeDisplay(l);
      await l.session.device.queue.onSubmittedWorkDone();
    }
  } finally {
    building = false;
  }
}
(window as unknown as { __nrcPipeline: unknown }).__nrcPipeline = { step, benchmark, saveAB, pause: (p: boolean) => { paused = p; }, get fps() { return fps; }, status: () => statusEl.textContent, get live() { return live; } };
