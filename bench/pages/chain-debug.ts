/**
 * chain-debug: the Classroom real-time chain as a bench, with a view of every pass and a control for every stage.
 *
 *   node bench/live.ts chain-debug          (then open the printed URL in Chrome, not the app's browser pane: it draws ~1 frame a second)
 *
 * Same session as the showcase (`NrcPipelineSession`, `bench/realtime/chain.ts`): path tracing (plain, or + NRC) -> pack -> SVGF / OIDN ->
 * composite -> pmndrs/upscaler. One arm renders. Any four passes show side by side (frame planes, the denoiser's guides and its internal
 * history, its outputs, the OIDN inputs, the reactive mask, the upscaled frame, and a per-pixel temporal-change map), with a nearest-texel zoom
 * around the clicked pixel and a readout of every pass there. Scripted camera motion (strafe, pan, dolly; one step per rendered frame) makes
 * motion-only artefacts repeatable, and "Flicker over 60 frames" turns them into a number per stage.
 *
 * The camera-jitter switch is the experiment behind the page (2026-11-26): with the sampler's per-pixel jitter a thin edge flips between
 * foreground and background in the 1-spp guides (normal, viewZ, albedo) every frame, which looks like a disocclusion to SVGF; `centre` and
 * `upscaler` hands the upscaler core's per-frame sequence to the kernels (`integrator.cameraJitter: 'engine'`) and the upscaler reconstructs with it; `centre` has no jitter.
 */
import { pagesBase } from '../site-paths.ts';
import { LIGHTING_FIELDS, lightingState, setupDaylight, updateDaylight } from '../nrc/daylight-controls.ts';
import { forgetNetwork, loadNetwork, networkKey, saveNetwork, sceneHash, usableNetwork } from '../nrc/network-store.ts';
import { cameraKey } from '../realtime/chain-math.ts';
import { isInterior, loadInterior } from '../scenes/interiors.ts';
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { uploadImage } from '../runner/page-client.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { cornell } from '../scenes/tier0/index.ts';
import { phase10Settings } from '../nrc/phase10.ts';
import { sponzaWhite } from '../scenes/tier2/sponza-white.ts';
import { NrcPipelineSession, type ArmName } from '../nrc/pipeline-session.ts';
import { readBuffer, readTextureRgba16f, type PipelineArm } from '../realtime/chain.ts';
import { FlyCamera, flyInputFromCodes } from '../realtime/fly-camera.ts';

void pagesBase;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>('canvas');
const statusEl = $<HTMLDivElement>('status');
const num = (id: string) => Number($<HTMLInputElement>(id).value);
const checked = (id: string) => $<HTMLInputElement>(id).checked;
const value = (id: string) => $<HTMLSelectElement>(id).value;

const SCENES: Record<string, SceneDesc> = { 'sponza-white': sponzaWhite, cornell };
const PRESET = phase10Settings('mlp');

// ---------------------------------------------------------------- pass catalogue

/** How a pass is drawn: tonemapped radiance, linear [0, 1], normal, a scalar on a colour ramp (channel `ch`), or motion (xy, signed). */
type Mode = 'color' | 'linear' | 'normal' | 'scalar' | 'motion';
const MODE_ID: Record<Mode, number> = { color: 0, linear: 1, normal: 2, scalar: 3, motion: 4 };

interface PassView {
  id: string;
  label: string;
  group: string;
  mode: Mode;
  ch?: number;
  lo?: number;
  hi?: number;
  log?: boolean;
  /** Negate before mapping (viewZ is negative in front). */
  negate?: boolean;
  get: (arm: PipelineArm) => GPUTexture | undefined;
}

const VIEWS: PassView[] = [
  { id: 'final', label: 'Final (upscaled, display res)', group: 'Output', mode: 'color', get: (a) => a.stage('final') },
  { id: 'denoised', label: 'Denoised + composite (upscaler input)', group: 'Output', mode: 'color', get: (a) => a.stage('denoised') },
  { id: 'raw', label: 'Raw beauty (path tracer, frame mean)', group: 'Output', mode: 'color', get: (a) => a.tex.raw },
  { id: 'diffuse', label: 'Diffuse radiance / albedo (SVGF input)', group: 'Denoiser inputs', mode: 'color', get: (a) => a.tex.diffuse },
  { id: 'specular', label: 'Specular radiance / spec factor (SVGF input)', group: 'Denoiser inputs', mode: 'color', get: (a) => a.tex.specular },
  { id: 'emission', label: 'Emission plane (bypasses the filter)', group: 'Denoiser inputs', mode: 'color', get: (a) => a.tex.emission },
  { id: 'albedo', label: 'Albedo factor (diffuse demodulation)', group: 'Denoiser inputs', mode: 'linear', get: (a) => a.tex.albedo },
  { id: 'specF', label: 'Specular factor (specular demodulation)', group: 'Denoiser inputs', mode: 'linear', get: (a) => a.tex.specF },
  { id: 'specShare', label: 'Share of specular-class samples', group: 'Denoiser inputs', mode: 'scalar', ch: 3, lo: 0, hi: 1, get: (a) => a.tex.albedo },
  { id: 'hitShare', label: 'Share of samples whose primary ray hit', group: 'Denoiser inputs', mode: 'scalar', ch: 3, lo: 0, hi: 1, get: (a) => a.tex.specF },
  { id: 'normal', label: 'Normal (world)', group: 'Guides', mode: 'normal', get: (a) => a.tex.normal },
  { id: 'viewZ', label: 'View depth −viewZ (m)', group: 'Guides', mode: 'scalar', ch: 0, lo: 0.3, hi: 30, log: true, negate: true, get: (a) => a.tex.viewZ },
  { id: 'motion', label: 'Motion (NDC delta × 50)', group: 'Guides', mode: 'motion', lo: 50, get: (a) => a.tex.motion },
  { id: 'motionZ', label: 'Motion z (view z change, m)', group: 'Guides', mode: 'scalar', ch: 0, lo: -0.05, hi: 0.05, get: (a) => a.tex.motionZ },
  { id: 'roughness', label: 'Roughness', group: 'Guides', mode: 'scalar', ch: 0, lo: 0, hi: 1, get: (a) => a.tex.roughness },
  { id: 'hitD', label: 'Diffuse hit distance (m)', group: 'Guides', mode: 'scalar', ch: 0, lo: 0.01, hi: 10, log: true, get: (a) => a.tex.hitD },
  { id: 'hitS', label: 'Specular hit distance (m)', group: 'Guides', mode: 'scalar', ch: 0, lo: 0.01, hi: 10, log: true, get: (a) => a.tex.hitS },
  { id: 'history', label: 'SVGF history length (frames, diffuse)', group: 'SVGF internals', mode: 'scalar', ch: 3, lo: 1, hi: 32, log: true, get: (a) => a.denoiserDebug('accum') },
  { id: 'accum', label: 'SVGF temporal accumulation (diffuse)', group: 'SVGF internals', mode: 'color', get: (a) => a.denoiserDebug('accum') },
  { id: 'lambda', label: 'SVGF anti-lag λ (1 = history dropped)', group: 'SVGF internals', mode: 'scalar', ch: 0, lo: 0, hi: 1, get: (a) => a.denoiserDebug('gradient') },
  { id: 'svgfDiffuse', label: 'SVGF output: diffuse (remodulated)', group: 'SVGF internals', mode: 'color', get: (a) => a.lastSvgf?.diffuse },
  { id: 'svgfSpecular', label: 'SVGF output: specular (remodulated)', group: 'SVGF internals', mode: 'color', get: (a) => a.lastSvgf?.specular },
  { id: 'oidnColor', label: 'OIDN input: colour', group: 'OIDN inputs', mode: 'color', get: (a) => a.oidnInputs.color },
  { id: 'oidnAlbedo', label: 'OIDN input: albedo', group: 'OIDN inputs', mode: 'linear', get: (a) => a.oidnInputs.albedo },
  { id: 'oidnNormal', label: 'OIDN input: normal (view space)', group: 'OIDN inputs', mode: 'normal', get: (a) => a.oidnInputs.normal },
  { id: 'reactive', label: 'Reactive mask (upscaler)', group: 'Upscaler', mode: 'scalar', ch: 0, lo: 0, hi: 1, get: (a) => a.reactiveMask },
  { id: 'flicker', label: 'Temporal change (flicker map)', group: 'Measure', mode: 'scalar', ch: 0, lo: 0, hi: 0.5, get: () => flicker?.out },
];
const viewById = new Map(VIEWS.map((v) => [v.id, v]));
const MEASURE_STAGES = ['raw', 'diffuse', 'denoised', 'final'];
const FLICKER_SOURCES = ['final', 'denoised', 'raw', 'diffuse', 'specular', 'albedo', 'normal', 'viewZ', 'accum', 'svgfDiffuse', 'svgfSpecular'];

// ---------------------------------------------------------------- WGSL

const VIEW_WGSL = /* wgsl */ `
struct V { rect: vec4f, crop: vec4f, mode: u32, ch: u32, logScale: u32, negate: u32, lo: f32, hi: f32, exposure: f32, probeOn: f32, probe: vec2f, _q: vec2f }
@group(0) @binding(0) var<uniform> v: V;
@group(0) @binding(1) var src: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
fn tone(c: vec3f) -> vec3f {
  let x = max(c * v.exposure, vec3f(0.0));
  let t = clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0));
  return pow(t, vec3f(1.0 / 2.2));
}
// Turbo colour ramp (polynomial fit, A. Mikhailov 2019, Apache-2.0).
fn ramp(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  let v4 = vec4f(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return clamp(vec3f(
    dot(v4, vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2f(-152.94239396, 59.28637943)),
    dot(v4, vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2f(4.27729857, 2.82956604)),
    dot(v4, vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2f(-89.90310912, 27.34824973))), vec3f(0.0), vec3f(1.0));
}
fn bad(x: f32) -> bool { return x != x || abs(x) > 3.0e38; }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let local = (pos.xy - v.rect.xy) / v.rect.zw;
  let uv = v.crop.xy + local * v.crop.zw;
  let size = vec2f(textureDimensions(src));
  let f = uv * size;
  let t = vec2i(clamp(floor(f), vec2f(0.0), size - 1.0));
  let c = textureLoad(src, t, 0);
  var out = vec3f(0.0);
  if (bad(c.x) || bad(c.y) || bad(c.z) || bad(c.w)) {
    out = vec3f(1.0, 0.0, 1.0);
  } else if (v.mode == 0u) {
    out = tone(c.xyz);
  } else if (v.mode == 1u) {
    out = pow(clamp(c.xyz, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
  } else if (v.mode == 2u) {
    out = select(c.xyz * 0.5 + 0.5, vec3f(0.0), dot(c.xyz, c.xyz) < 1e-8);
  } else if (v.mode == 3u) {
    var s = c[v.ch];
    if (v.negate == 1u) { s = -s; }
    var k = (s - v.lo) / (v.hi - v.lo);
    if (v.logScale == 1u) { k = (log2(max(s, 1e-12)) - log2(max(v.lo, 1e-12))) / (log2(max(v.hi, 1e-12)) - log2(max(v.lo, 1e-12))); }
    out = ramp(k);
  } else {
    out = vec3f(clamp(c.xy * v.lo * 0.5 + 0.5, vec2f(0.0), vec2f(1.0)), 0.5);
  }
  // The clicked texel: a one-pixel yellow frame.
  if (v.probeOn > 0.5) {
    let pt = floor(v.probe * size);
    let texelPerPixel = v.crop.zw * size / v.rect.zw;
    let fr = f - pt;
    let edge = (fr.x < texelPerPixel.x || fr.y < texelPerPixel.y || fr.x > 1.0 - texelPerPixel.x || fr.y > 1.0 - texelPerPixel.y);
    let inside = fr.x >= -texelPerPixel.x && fr.y >= -texelPerPixel.y && fr.x <= 1.0 + texelPerPixel.x && fr.y <= 1.0 + texelPerPixel.y;
    if (inside && edge) { out = vec3f(1.0, 0.9, 0.0); }
  }
  return vec4f(out, 1.0);
}
`;

/**
 * Per-pixel mean of the relative frame-to-frame luminance change of one pass: |L(t) - L'(t-1)| / mean(L, L'), where L' is the previous frame's
 * luminance at the point this pixel showed then (reprojected through the jitter-free motion plane, nearest texel; NDC motion works at any
 * resolution), so camera motion itself does not count, only boiling and flicker. A pixel whose previous point was off screen is not counted.
 * Running mean with weight max(alpha, 1 / n) (alpha 0 = the plain mean over the frames, for measurements).
 */
const FLICKER_WGSL = /* wgsl */ `
struct P { alpha: f32, reset: u32, scalar: u32, reproject: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> prevL: array<f32>;
@group(0) @binding(3) var<storage, read_write> curL: array<f32>;
@group(0) @binding(4) var<storage, read_write> stat: array<vec2f>;
@group(0) @binding(5) var outT: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var motion: texture_2d<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(src);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let c = textureLoad(src, gid.xy, 0);
  var l = dot(c.xyz, vec3f(0.2126, 0.7152, 0.0722));
  if (p.scalar == 1u) { l = c.x; }
  let i = gid.y * size.x + gid.x;
  curL[i] = l;
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
  var puv = uv;
  if (p.reproject == 1u) {
    let msize = vec2f(textureDimensions(motion));
    let mo = textureLoad(motion, vec2u(min(uv * msize, msize - 1.0)), 0).xy;
    let pn = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) - mo;
    puv = vec2f(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
  }
  var st = stat[i];
  var d = 0.0;
  let valid = p.reset == 0u && all(puv >= vec2f(0.0)) && all(puv < vec2f(1.0));
  if (valid) {
    let j = vec2u(puv * vec2f(size));
    let lp = prevL[j.y * size.x + j.x];
    d = abs(l - lp) / (0.5 * (abs(l) + abs(lp)) + 1e-3);
    let n = st.y + 1.0;
    st = vec2f(mix(st.x, d, max(p.alpha, 1.0 / n)), n);
  }
  if (p.reset == 1u) { st = vec2f(0.0); }
  stat[i] = st;
  textureStore(outT, gid.xy, vec4f(st.x, d, l, 1.0));
}
`;

/** The flicker map of one pass (ping-pong luminance, the running mean, a texture to view). */
class FlickerMap {
  readonly source: GPUTexture;
  readonly out: GPUTexture;
  private readonly lum: [GPUBuffer, GPUBuffer];
  readonly stat: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly groups: [GPUBindGroup, GPUBindGroup];
  frames = 0;
  constructor(l: Live, source: GPUTexture, motion: GPUTexture) {
    const device = l.session.device;
    const n = source.width * source.height;
    this.source = source;
    this.lum = [0, 1].map((k) => device.createBuffer({ label: `chain debug flicker lum ${k}`, size: n * 4, usage: GPUBufferUsage.STORAGE })) as [GPUBuffer, GPUBuffer];
    this.stat = device.createBuffer({ label: 'chain debug flicker stat', size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.params = device.createBuffer({ label: 'chain debug flicker params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.out = device.createTexture({ label: 'chain debug flicker map', size: { width: source.width, height: source.height }, format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const group = (k: number) => device.createBindGroup({ layout: l.flickerLayout, entries: [{ buffer: this.params }, source.createView(), { buffer: this.lum[1 - k] as GPUBuffer }, { buffer: this.lum[k] as GPUBuffer }, { buffer: this.stat }, this.out.createView(), motion.createView()].map((resource, binding) => ({ binding, resource })) });
    this.groups = [group(0), group(1)];
  }
  step(l: Live, alpha: number, reproject: boolean): void {
    const device = l.session.device;
    const fp = new ArrayBuffer(16);
    new Float32Array(fp, 0, 1).set([alpha]);
    new Uint32Array(fp, 4, 3).set([this.frames === 0 ? 1 : 0, this.source.format === 'r32float' ? 1 : 0, reproject ? 1 : 0]);
    device.queue.writeBuffer(this.params, 0, fp);
    const encoder = device.createCommandEncoder({ label: 'chain debug flicker' });
    const pass = encoder.beginComputePass({ label: 'chain debug flicker' });
    pass.setPipeline(l.flickerPipe);
    pass.setBindGroup(0, this.groups[this.frames & 1] as GPUBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.source.width / 8), Math.ceil(this.source.height / 8));
    pass.end();
    device.queue.submit([encoder.finish()]);
    this.frames++;
  }
  /** Mean over the counted pixels, the whole image and a window of +-16 render pixels around `at` (readback: not in the frame loop). */
  async mean(l: Live, at: [number, number] | null): Promise<{ image: number; window: number | null; counted: number }> {
    const { width: w, height: h } = this.source;
    const data = new Float32Array(await readBuffer(l.session.device, this.stat, w * h * 8));
    let s = 0;
    let n = 0;
    for (let i = 0; i < w * h; i++) if ((data[2 * i + 1] as number) > 0) {
      s += data[2 * i] as number;
      n++;
    }
    let window: number | null = null;
    if (at) {
      const r = Math.max(1, Math.round((16 * w) / l.session.options.width));
      const cx = Math.floor(at[0] * w);
      const cy = Math.floor(at[1] * h);
      let ws = 0;
      let wn = 0;
      for (let y = Math.max(0, cy - r); y < Math.min(h, cy + r); y++)
        for (let x = Math.max(0, cx - r); x < Math.min(w, cx + r); x++) {
          const k = y * w + x;
          if ((data[2 * k + 1] as number) > 0) {
            ws += data[2 * k] as number;
            wn++;
          }
        }
      window = wn ? ws / wn : null;
    }
    return { image: n ? s / n : 0, window, counted: n / (w * h) };
  }
  destroy(): void {
    for (const b of [...this.lum, this.stat, this.params]) b.destroy();
    this.out.destroy();
  }
}

// ---------------------------------------------------------------- state

interface Live {
  session: NrcPipelineSession;
  scene: SceneDesc;
  jitter: Jitter;
  networkKey: string;
  context: GPUCanvasContext;
  pipeline: GPURenderPipeline;
  layout: GPUBindGroupLayout;
  params: GPUBuffer[];
  flickerPipe: GPUComputePipeline;
  flickerLayout: GPUBindGroupLayout;
}

let live: Live | null = null;
let flicker: FlickerMap | null = null;
let busy = false;
let building = false;
let paused = false;
let stepRequests = 0;
let fps = 0;
let lastFrameEnd = 0;
let lastTick = 0;
let frameIndex = 0;
let motionFrame = 0;
/** Holds the scripted motion at `motionFrame` (the lag measurement renders one pose of the path for many frames). */
let motionHold = false;
/** The clicked texel, as uv in [0, 1] (the zoom centre too). */
let probe: [number, number] | null = null;
const keys = new Set<string>();
const fly = new FlyCamera();
const PIN_STORE = 'chain-debug-pin';

function setStatus(text: string, error = false): void {
  statusEl.textContent = text;
  statusEl.classList.toggle('err', error);
}

const arm = (): ArmName => (value('arm') === 'plain' ? 'plain' : 'nrc');
type Jitter = 'sampler' | 'upscaler' | 'centre';
const jitterOwner = (): Jitter => (value('jitter') === 'upscaler' ? 'upscaler' : value('jitter') === 'centre' ? 'centre' : 'sampler');

function networkKeyParts(scene: SceneDesc, session: NrcPipelineSession): Promise<string> {
  return sceneHash(scene).then((h) => networkKey({ sceneHash: h, network: JSON.stringify([PRESET.network, PRESET]), maxDepth: NrcPipelineSession.limits(session.options).maxDepth }));
}

async function build(): Promise<void> {
  if (building) return;
  building = true;
  const rebuild = ['scene', 'res', 'displayScale', 'maxDepth', 'jitter', 'textureLod'];
  for (const id of rebuild) $<HTMLInputElement>(id).disabled = true;
  try {
    while (busy) await new Promise((r) => setTimeout(r, 5));
    const sceneId = value('scene');
    const [w, h] = value('res').split('x').map(Number) as [number, number];
    setStatus(`Loading ${sceneId} at ${w} × ${h}, compiling the kernels…`);
    const sameScene = live?.scene.name === sceneId;
    const carried = sameScene && live ? await live.session.cache.exportNetwork().catch(() => undefined) : undefined;
    for (const b of live?.params ?? []) b.destroy();
    dropFlicker();
    live?.session.destroy();
    live = null;
    const scene = isInterior(sceneId) ? await loadInterior(sceneId) : (SCENES[sceneId] as SceneDesc);
    const jk = jitterOwner();
    const session = await NrcPipelineSession.create({
      scene,
      width: w,
      height: h,
      displayWidth: num('displayScale') * w,
      displayHeight: num('displayScale') * h,
      ...(num('maxDepth') ? { maxDepth: Math.max(1, Math.min(32, Math.round(num('maxDepth')))) } : {}),
      infer: 'tiled',
      precision: 'f16',
      cache: { network: PRESET.network, ...PRESET, trainSelect: 'tile', stepsPerFrame: num('steps') },
      reactive: { cap: num('reactiveCap') || 8, strength: 1 },
      jitter: jk,
      ...(value('textureLod') !== 'mip0' ? { textureLod: 'cone' as const, textureConeScale: Number(value('textureLod').slice(4)) || 1 } : {}),
    });
    const device = session.device;
    device.addEventListener('uncapturederror', (e) => setStatus(`GPU error: ${(e as GPUUncapturedErrorEvent).error.message}`, true));
    if (!sameScene) resetView(scene, session.bounds);
    canvas.width = session.options.displayWidth;
    canvas.height = session.options.displayHeight;
    $<HTMLInputElement>('maxDepth').placeholder = String(scene.integrator.maxDepth);
    const context = canvas.getContext('webgpu') as GPUCanvasContext;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });
    const layout = device.createBindGroupLayout({ label: 'chain debug view', entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }, { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } }] });
    const module = device.createShaderModule({ label: 'chain debug view', code: VIEW_WGSL });
    const pipeline = await device.createRenderPipelineAsync({ label: 'chain debug view', layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    const params = [0, 1, 2, 3].map((k) => device.createBuffer({ label: `chain debug view ${k}`, size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const flickerLayout = device.createBindGroupLayout({
      label: 'chain debug flicker',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const flickerPipe = await device.createComputePipelineAsync({ label: 'chain debug flicker', layout: device.createPipelineLayout({ bindGroupLayouts: [flickerLayout] }), compute: { module: device.createShaderModule({ label: 'chain debug flicker', code: FLICKER_WGSL }), entryPoint: 'main' } });
    const key = session.cache.canExportNetwork ? await networkKeyParts(scene, session) : '';
    live = { session, scene, jitter: jk, networkKey: key, context, pipeline, layout, params, flickerPipe, flickerLayout };
    $('daylight').hidden = !isInterior(sceneId);
    if (isInterior(sceneId)) {
      if (sameScene) updateDaylight(session);
      else setupDaylight(session);
    }
    // Start warm: a rebuild of the same scene keeps the network; otherwise the one this page saved for this scene and lighting.
    const kept = carried?.weights && carried.trained && carried.weights.length === session.cache.weightCount ? carried : undefined;
    const stored = !kept && key && checked('loadSaved') ? await loadNetwork(key) : undefined;
    const start = kept ?? (usableNetwork(stored, key, lightingState(), session.cache.weightCount) ? stored : undefined);
    if (start?.weights && start.trained) session.cache.loadWeights(start.weights, start.trained);
    applyAll();
    session.cutNext();
    frameIndex = 0;
    setStatus('');
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), true);
    console.error(error);
  } finally {
    for (const id of rebuild) $<HTMLInputElement>(id).disabled = false;
    building = false;
  }
}

function resetView(scene: SceneDesc, bounds: NrcPipelineSession['bounds']): void {
  fly.setPose(scene.camera as PtLookAtCamera);
  fly.speed = 0.15 * Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]);
}

// ---------------------------------------------------------------- settings -> session

function svgfOptions(): Record<string, unknown> {
  const o: Record<string, unknown> = {
    temporalAlpha: Math.max(0.01, Math.min(1, num('svgfAlpha') || 0.1)),
    maxHistory: Math.max(1, Math.round(num('maxHistory')) || 64),
    planeDistance: Math.max(0, num('planeDistance')),
    normalThreshold: Math.max(-1, Math.min(1, num('normalThreshold'))),
    normalTest: value('normalTest'),
    historyFallback: value('historyFallback'),
    historyFilter: value('historyFilter'),
    atrousIterations: Math.max(0, Math.min(8, Math.round(num('atrousIterations')))),
    kernel: value('kernel'),
    sigmaDepth: num('sigmaDepth'),
    sigmaNormal: num('sigmaNormal'),
    sigmaLuminance: num('sigmaLuminance'),
  };
  if (value('antiLag') !== 'default') o.antiLag = value('antiLag');
  if (checked('antiLagHistoryScale')) o.antiLagHistoryScale = true;
  if (value('historyFootprint') !== 'off') {
    o.historyFootprint = value('historyFootprint');
    o.historyFootprintDwell = Math.max(0, num('footprintDwell'));
  }
  // Specular-only options go to the specular signal (the joint pipeline splits when they differ from the diffuse one's).
  const spec: Record<string, unknown> = {};
  if (value('reflectionReprojection') !== 'surface') spec.reflectionReprojection = value('reflectionReprojection');
  if (checked('reflectionHitTest')) spec.reflectionHitTest = true;
  if (Object.keys(spec).length) o.specular = spec;
  return o;
}

function applyDenoiser(): void {
  const l = live;
  if (!l) return;
  const v = value('denoiserSel');
  const stillSamples = Math.max(1, Math.round(num('stillSamples')));
  const aux = value('oidnAux') === 'albedo' ? 'albedo' : 'albedo+normal';
  if (v === 'oidn') l.session.setDenoiser('oidn', { aux, stillSamples, hold: false });
  else if (v === 'oidn-fast') l.session.setDenoiser('oidn', { aux, moving: 'fast', stillSamples: Infinity, hold: false });
  else if (v === 'oidn-std') l.session.setDenoiser('oidn', { aux, moving: 'balanced', stillSamples: Infinity, hold: false });
  else l.session.setDenoiser(v === 'none' ? 'none' : 'svgf');
}

function applyChain(): void {
  const l = live;
  if (!l) return;
  const path = value('upscalePath') as 'spatial' | 'bilinear' | 'temporal';
  const svgf = svgfOptions();
  for (const a of Object.values(l.session.arms)) {
    a.setFireflyClamp(Math.max(0, num('fireflyClamp')));
    a.setDenoiserOptions(svgf);
    a.upscalePath = path;
    a.reactiveEnabled = checked('reactive');
    Object.assign(a.upscaler.settings, { sharpness: Math.min(1, Math.max(0, num('sharpness'))), maxAccumulation: Math.max(1, Math.min(64, num('upHistory'))), rcasDenoise: checked('rcasDenoise'), autoExposure: checked('upAutoExposure'), exposure: 1 });
    a.resetHistory();
  }
  dropFlicker();
}

function applyAll(): void {
  applyDenoiser();
  applyChain();
}

/** Per frame (rAF and steps): samples, training, history limits, the gate. */
function applyFrameSettings(session: NrcPipelineSession): void {
  session.plainSpp = Math.max(1, Math.min(16, Math.round(num('spp')) || 1));
  session.nrcSpp = session.plainSpp;
  session.accumulateStill = checked('accumulateStill');
  session.svgfHistoryLimit = Math.max(0, Math.min(4096, Math.round(num('svgfHistoryLimit'))));
  session.cache.settings.stepsPerFrame = Math.max(0, Math.round(num('steps')));
  const gate = Math.max(0, Math.round(num('gateSteps')));
  session.setGate(Math.ceil(gate / 16), 16);
}

// ---------------------------------------------------------------- camera: scripted motion, pin

/** The camera of this frame: the fly camera, moved by the scripted motion (one step per rendered frame, so runs repeat). */
function frameCamera(): PtLookAtCamera {
  const c = fly.camera;
  const m = value('motion');
  if (m === 'off') return c;
  const amp = Math.max(0, num('motionAmp'));
  const s = Math.sin((2 * Math.PI * motionFrame) / Math.max(4, num('motionPeriod')));
  const p = c.position as readonly number[];
  const t = c.target as readonly number[];
  const d = [t[0]! - p[0]!, t[1]! - p[1]!, t[2]! - p[2]!];
  const len = Math.hypot(d[0]!, d[1]!, d[2]!) || 1;
  const f = d.map((x) => x / len);
  const r = [-f[2]!, 0, f[0]!];
  const rl = Math.hypot(r[0]!, r[2]!) || 1;
  if (m === 'strafe' || m === 'dolly') {
    const dir = m === 'strafe' ? [r[0]! / rl, 0, r[2]! / rl] : f;
    const o = dir.map((x) => x * amp * s);
    return { ...c, position: [p[0]! + o[0]!, p[1]! + o[1]!, p[2]! + o[2]!], target: [t[0]! + o[0]!, t[1]! + o[1]!, t[2]! + o[2]!] };
  }
  // pan: yaw by amp radians
  const a = amp * s;
  const ca = Math.cos(a);
  const sa = Math.sin(a);
  const nd = [d[0]! * ca - d[2]! * sa, d[1]!, d[0]! * sa + d[2]! * ca];
  return { ...c, target: [p[0]! + nd[0]!, p[1]! + nd[1]!, p[2]! + nd[2]!] };
}

interface Pin { scene: string; name?: string; position: number[]; yaw: number; pitch: number }

function currentPin(): Pin {
  return { scene: value('scene'), position: [...fly.position], yaw: fly.yaw, pitch: fly.pitch };
}

function applyPin(p: Pin): void {
  fly.stop();
  fly.position = [p.position[0]!, p.position[1]!, p.position[2]!];
  fly.yaw = p.yaw;
  fly.pitch = p.pitch;
  live?.session.cutNext();
}

function pinToHash(p: Pin): string {
  return `#cam=${[...p.position, p.yaw, p.pitch].map((x) => x.toFixed(4)).join(',')}`;
}

function pinFromHash(): Pin | null {
  const m = /cam=([-\d.,e]+)/.exec(location.hash);
  if (!m) return null;
  const v = (m[1] as string).split(',').map(Number);
  if (v.length !== 5 || v.some((x) => !Number.isFinite(x))) return null;
  return { scene: value('scene'), position: v.slice(0, 3), yaw: v[3]!, pitch: v[4]! };
}

/** Pinned views of every scene (localStorage), newest last. */
function storedPins(): Pin[] {
  try {
    const v = JSON.parse(localStorage.getItem(PIN_STORE) ?? '[]') as Pin[] | Pin | null;
    return Array.isArray(v) ? v : v ? [v] : [];
  } catch {
    return [];
  }
}

function scenePins(): Pin[] {
  return storedPins().filter((p) => p.scene === value('scene'));
}

function storedPin(): Pin | null {
  return scenePins().at(-1) ?? null;
}

function fillPins(): void {
  const pins = scenePins();
  const sel = $<HTMLSelectElement>('pinSel');
  sel.replaceChildren(...(pins.length ? pins.map((p, i) => new Option(p.name || `View ${i + 1}`, String(i))) : [new Option('No pinned views', '')]));
  sel.value = pins.length ? String(pins.length - 1) : '';
}

// ---------------------------------------------------------------- drawing

function viewports(): { x: number; y: number; w: number; h: number }[] {
  const W = canvas.width;
  const H = canvas.height;
  if (value('layout') === 'single') return [{ x: 0, y: 0, w: W, h: H }];
  const w = W / 2;
  const h = H / 2;
  return [{ x: 0, y: 0, w, h }, { x: w, y: 0, w, h }, { x: 0, y: h, w, h }, { x: w, y: h, w, h }];
}

function crop(): [number, number, number, number] {
  const z = Math.max(1, num('zoom'));
  if (z === 1 || !probe) return [0, 0, 1, 1];
  const s = 1 / z;
  const cx = Math.min(1 - s / 2, Math.max(s / 2, probe[0]));
  const cy = Math.min(1 - s / 2, Math.max(s / 2, probe[1]));
  return [cx - s / 2, cy - s / 2, s, s];
}

const groupCache = new WeakMap<GPUTexture, Map<GPUBuffer, GPUBindGroup>>();
function viewGroup(l: Live, tex: GPUTexture, params: GPUBuffer): GPUBindGroup {
  let m = groupCache.get(tex);
  if (!m) groupCache.set(tex, (m = new Map()));
  let g = m.get(params);
  if (!g) {
    g = l.session.device.createBindGroup({ layout: l.layout, entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: tex.createView() }] });
    m.set(params, g);
  }
  return g;
}

function viewSettings(k: number): PassView {
  return viewById.get(value(`view${k}`)) ?? (VIEWS[0] as PassView);
}

function draw(l: Live): void {
  const a = l.session.arms[arm()];
  const encoder = l.session.device.createCommandEncoder({ label: 'chain debug view' });
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view: l.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.08, g: 0.08, b: 0.08, a: 1 } }] });
  pass.setPipeline(l.pipeline);
  const vps = viewports();
  const cr = crop();
  const tags = $('tags').querySelectorAll('span');
  vps.forEach((vp, k) => {
    const v = viewSettings(k);
    const tex = v.get(a);
    const tag = tags[k] as HTMLSpanElement;
    tag.textContent = tex ? v.label : `${v.label}: not produced (check the denoiser)`;
    if (!tex) return;
    const first = k === 0;
    const lo = first && $<HTMLInputElement>('lo').value !== '' ? num('lo') : v.lo ?? 0;
    const hi = first && $<HTMLInputElement>('hi').value !== '' ? num('hi') : v.hi ?? 1;
    const log = first ? checked('logScale') : !!v.log;
    const buf = new ArrayBuffer(80);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f.set([vp.x, vp.y, vp.w, vp.h, ...cr], 0);
    u.set([MODE_ID[v.mode], v.ch ?? 0, log ? 1 : 0, v.negate ? 1 : 0], 8);
    f.set([lo, hi, 2 ** num('exposure'), probe ? 1 : 0, probe?.[0] ?? 0, probe?.[1] ?? 0], 12);
    const params = l.params[k] as GPUBuffer;
    l.session.device.queue.writeBuffer(params, 0, buf);
    pass.setViewport(vp.x, vp.y, vp.w, vp.h, 0, 1);
    pass.setBindGroup(0, viewGroup(l, tex, params));
    pass.draw(3);
  });
  pass.end();
  l.session.device.queue.submit([encoder.finish()]);
}

// ---------------------------------------------------------------- flicker map

function dropFlicker(): void {
  flicker?.destroy();
  flicker = null;
}

function stepFlickerMap(l: Live): void {
  const a = l.session.arms[arm()];
  const src = viewById.get(value('flickerSrc'))?.get(a);
  if (!src) return;
  if (flicker?.source !== src) {
    dropFlicker();
    flicker = new FlickerMap(l, src, a.tex.motion);
  }
  flicker.step(l, 0.05, checked('reproject'));
}

// ---------------------------------------------------------------- frame loop

async function renderOne(l: Live): Promise<void> {
  applyFrameSettings(l.session);
  const camera = frameCamera();
  await l.session.frame(camera, { arms: [arm()], train: checked('train') });
  stepFlickerMap(l);
  draw(l);
  frameIndex++;
  if (value('motion') !== 'off' && !motionHold) motionFrame++;
}

function frame(now: number): void {
  requestAnimationFrame(frame);
  const l = live;
  const dt = lastTick ? (now - lastTick) / 1000 : 0;
  lastTick = now;
  if (!l || building) return;
  if (paused && stepRequests === 0) return;
  if (!paused) fly.advance(dt, flyInputFromCodes(keys));
  if (busy) return;
  if (paused) stepRequests--;
  busy = true;
  const t0 = performance.now();
  void renderOne(l)
    .then(() => l.session.device.queue.onSubmittedWorkDone())
    .then(() => {
      busy = false;
      const end = performance.now();
      if (lastFrameEnd && !paused) fps = fps ? 0.9 * fps + 0.1 * (1000 / (end - lastFrameEnd)) : 1000 / (end - lastFrameEnd);
      lastFrameEnd = end;
      if (statusEl.classList.contains('err')) return;
      const c = l.session.cache;
      const a = l.session.arms[arm()];
      const gate = arm() === 'nrc' && c.warming ? ` · NRC gate closed (plain while training, ${c.trained.steps} steps)` : '';
      setStatus(`${l.scene.name} · ${arm()} · ${l.session.options.width} × ${l.session.options.height} → ${canvas.width} × ${canvas.height} · jitter ${value('jitter')} · ${a.lastDenoiser || '…'} · frame ${frameIndex} · ${paused ? 'paused' : `${fps.toFixed(0)} fps (${(end - t0).toFixed(1)} ms)`} · ${c.trained.steps} training steps${gate}`);
    })
    .catch((error: unknown) => {
      busy = false;
      setStatus(error instanceof Error ? error.message : String(error), true);
    });
}

async function idle(): Promise<void> {
  while (busy || building) await new Promise((r) => setTimeout(r, 5));
}

// ---------------------------------------------------------------- probe (readback on click; never in the frame loop)

const PROBE_VIEWS = ['raw', 'denoised', 'diffuse', 'specular', 'emission', 'albedo', 'specF', 'normal', 'viewZ', 'motion', 'motionZ', 'roughness', 'hitD', 'hitS', 'history', 'lambda', 'svgfDiffuse', 'svgfSpecular', 'reactive', 'flicker'];

/** A whole float texture as vec4 per texel (any format, also textures that are not COPY_SRC). */
const DUMP_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outV: array<vec4f>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(src);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  outV[gid.y * size.x + gid.x] = textureLoad(src, gid.xy, 0);
}
`;
let dumpPipe: { device: GPUDevice; pipe: GPUComputePipeline; layout: GPUBindGroupLayout } | null = null;

async function dumpTexture(device: GPUDevice, tex: GPUTexture): Promise<Float32Array> {
  if (dumpPipe?.device !== device) {
    const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    dumpPipe = { device, layout, pipe: device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module: device.createShaderModule({ code: DUMP_WGSL }), entryPoint: 'main' } }) };
  }
  const size = tex.width * tex.height * 16;
  const out = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(dumpPipe.pipe);
  pass.setBindGroup(0, device.createBindGroup({ layout: dumpPipe.layout, entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: { buffer: out } }] }));
  pass.dispatchWorkgroups(Math.ceil(tex.width / 8), Math.ceil(tex.height / 8));
  pass.end();
  device.queue.submit([enc.finish()]);
  const data = new Float32Array(await readBuffer(device, out, size));
  out.destroy();
  return data;
}

/**
 * Statistics of one pass's channel over the image (the scalar the view shows; luminance for colour passes): mean and percentiles over the
 * pixels where `mask` (default: every pixel) holds, and the share of pixels at or below `low`. Readback: not in the frame loop.
 */
async function passStats(id: string, o: { low?: number; maskHit?: boolean } = {}): Promise<Record<string, number> | string> {
  const l = live;
  if (!l) return 'no session';
  const v = viewById.get(id);
  const a = l.session.arms[arm()];
  const tex = v?.get(a);
  if (!v || !tex) return `${id}: not produced`;
  await idle();
  building = true;
  try {
    const data = await dumpTexture(l.session.device, tex);
    // The hit mask is at render resolution: only for passes of that size (not the final image, not the downsampled anti-lag λ).
    const sameSize = tex.width === a.tex.specF.width && tex.height === a.tex.specF.height;
    const hit = o.maskHit && sameSize ? await dumpTexture(l.session.device, a.tex.specF) : null;
    const vals: number[] = [];
    const n = tex.width * tex.height;
    for (let i = 0; i < n; i++) {
      if (hit && (hit[4 * i + 3] as number) <= 0.5) continue;
      const c = [data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]] as number[];
      let x = v.mode === 'scalar' ? (c[v.ch ?? 0] as number) * (v.negate ? -1 : 1) : 0.2126 * c[0]! + 0.7152 * c[1]! + 0.0722 * c[2]!;
      if (!Number.isFinite(x)) x = NaN;
      vals.push(x);
    }
    const finite = vals.filter((x) => !Number.isNaN(x)).sort((x, y) => x - y);
    const q = (f: number) => finite[Math.min(finite.length - 1, Math.floor(f * finite.length))] ?? NaN;
    const low = o.low ?? NaN;
    return { pixels: vals.length, nonFinite: vals.length - finite.length, mean: finite.reduce((s2, x) => s2 + x, 0) / Math.max(1, finite.length), p10: q(0.1), p25: q(0.25), p50: q(0.5), p75: q(0.75), p90: q(0.9), ...(Number.isFinite(low) ? { atOrBelowLow: finite.filter((x) => x <= low).length / Math.max(1, finite.length) } : {}) };
  } finally {
    building = false;
  }
}

const PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<uniform> at: vec4u;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outV: array<vec4f>;
@compute @workgroup_size(1) fn main() { outV[0] = textureLoad(src, at.xy, 0); }
`;
let probePipe: { device: GPUDevice; pipe: GPUComputePipeline; layout: GPUBindGroupLayout; at: GPUBuffer; out: GPUBuffer } | null = null;

/** One texel of any float texture (also the denoiser's, which are not COPY_SRC): a 1-thread compute load into a buffer, then a 16-byte read. */
async function readTexel(device: GPUDevice, tex: GPUTexture, uv: [number, number]): Promise<number[] | string> {
  if (!(tex.usage & GPUTextureUsage.TEXTURE_BINDING) || tex.format.startsWith('depth')) return tex.format;
  if (probePipe?.device !== device) {
    const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } }, { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    const pipe = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module: device.createShaderModule({ code: PROBE_WGSL }), entryPoint: 'main' } });
    probePipe = { device, pipe, layout, at: device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }), out: device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }) };
  }
  const pp = probePipe;
  const x = Math.min(tex.width - 1, Math.floor(uv[0] * tex.width));
  const y = Math.min(tex.height - 1, Math.floor(uv[1] * tex.height));
  device.queue.writeBuffer(pp.at, 0, new Uint32Array([x, y, 0, 0]));
  const group = device.createBindGroup({ layout: pp.layout, entries: [{ binding: 0, resource: { buffer: pp.at } }, { binding: 1, resource: tex.createView() }, { binding: 2, resource: { buffer: pp.out } }] });
  const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pp.pipe);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(1);
  pass.end();
  enc.copyBufferToBuffer(pp.out, 0, staging, 0, 16);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const v = [...new Float32Array(staging.getMappedRange().slice(0))];
  staging.unmap();
  staging.destroy();
  return v;
}

const fmt = (v: number) => (Number.isNaN(v) ? 'NaN' : !Number.isFinite(v) ? (v > 0 ? '+inf' : '-inf') : Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : v.toFixed(4));

async function readProbe(): Promise<void> {
  const l = live;
  if (!l || !probe) return;
  await idle();
  building = true;
  try {
    const a = l.session.arms[arm()];
    const rows: HTMLTableRowElement[] = [];
    const tex0 = a.tex.raw;
    rows.push(row('texel', `${Math.floor(probe[0] * tex0.width)}, ${Math.floor(probe[1] * tex0.height)}`));
    for (const id of PROBE_VIEWS) {
      const v = viewById.get(id) as PassView;
      const tex = v.get(a);
      if (!tex) continue;
      const r = await readTexel(l.session.device, tex, probe);
      const vals = typeof r === 'string' ? r : v.mode === 'scalar' ? [(v.negate ? -1 : 1) * (r[v.ch ?? 0] as number)] : r;
      rows.push(row(v.label, typeof vals === 'string' ? vals : vals.map(fmt).join('  ')));
    }
    ($('probeTable').querySelector('tbody') as HTMLTableSectionElement).replaceChildren(...rows);
  } finally {
    building = false;
  }
}

function row(label: string, v: string): HTMLTableRowElement {
  const tr = document.createElement('tr');
  const a = document.createElement('td');
  a.textContent = label;
  const b = document.createElement('td');
  b.className = 'v';
  b.textContent = v;
  tr.append(a, b);
  return tr;
}

// ---------------------------------------------------------------- measurements

async function measureFlicker(frames = 60, stages: readonly string[] = MEASURE_STAGES): Promise<Record<string, number>> {
  const l = live;
  if (!l) return {};
  await idle();
  building = true;
  const table = $('flickerTable').querySelector('tbody') as HTMLTableSectionElement;
  const result: Record<string, number> = {};
  try {
    const a = l.session.arms[arm()];
    const reproject = checked('reproject');
    // Built on the first frame (the stages' textures exist once a frame went through), then stepped together.
    let maps: [string, FlickerMap][] | null = null;
    for (let i = 0; i < frames + 1; i++) {
      setStatus(`Measuring flicker: frame ${i} of ${frames + 1}…`);
      await renderOne(l);
      maps ??= stages.flatMap((id) => {
        const t = viewById.get(id)?.get(a);
        return t ? [[id, new FlickerMap(l, t, a.tex.motion)] as [string, FlickerMap]] : [];
      });
      for (const [, m] of maps) m.step(l, 0, reproject);
      await l.session.device.queue.onSubmittedWorkDone();
    }
    const rows = [row('stage', 'image · window')];
    for (const [id, m] of maps ?? []) {
      const r = await m.mean(l, probe);
      result[id] = r.image;
      if (r.window !== null) result[`${id}_window`] = r.window;
      rows.push(row(viewById.get(id)?.label.split(' (')[0] ?? id, `${fmt(r.image)} · ${r.window === null ? '—' : fmt(r.window)}`));
      m.destroy();
    }
    rows.push(row('settings', `${value('motion')} motion · jitter ${value('jitter')} · ${reproject ? 'reprojected' : 'not reprojected'} · ${a.lastDenoiser}`));
    table.replaceChildren(...rows);
    console.info('chain-debug flicker', result);
  } finally {
    building = false;
    setStatus('');
  }
  return result;
}

/** Luminance of the final image, every 2nd texel in x and y (a readback: measurements only). */
async function finalLuma(l: Live): Promise<Float32Array> {
  const tex = l.session.arms[arm()].stage('final');
  const d = await dumpTexture(l.session.device, tex);
  const w = Math.floor(tex.width / 2);
  const h = Math.floor(tex.height / 2);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = 4 * (2 * y * tex.width + 2 * x);
      const v = 0.2126 * d[i]! + 0.7152 * d[i + 1]! + 0.0722 * d[i + 2]!;
      out[y * w + x] = Number.isFinite(v) ? v : 0;
    }
  }
  return out;
}

async function framesThen<T>(l: Live, n: number, label: string, after?: (i: number) => Promise<T>): Promise<T[]> {
  const out: T[] = [];
  for (let i = 0; i < n; i++) {
    if (i % 8 === 0) setStatus(`${label}: frame ${i} of ${n}…`);
    await renderOne(l);
    await l.session.device.queue.onSubmittedWorkDone();
    if (after) out.push(await after(i));
  }
  return out;
}

/** Mean of the final image's luminance over the last `k` of `n` frames (a converged image at a still camera). */
async function settledLuma(l: Live, n: number, k: number, label: string): Promise<Float32Array> {
  await framesThen(l, n - k, label);
  const imgs = await framesThen(l, k, label, () => finalLuma(l));
  const mean = new Float32Array(imgs[0]!.length);
  for (const img of imgs) for (let i = 0; i < mean.length; i++) mean[i]! += img[i]! / k;
  return mean;
}

/**
 * What a long history costs (N11.16), on the final image with the current settings. Readbacks every recorded frame: a measurement, never
 * the frame loop.
 * - Lighting step (`light`: 'lamps' switches the interior lights off, 'sun' halves the sun, 'none' skips): the camera is still, the image
 *   settles, the lighting changes WITHOUT a history cut (as a game's light switch does), and per frame t after the change the residual
 *   r(t) = Σ|L(t) − L_end| / Σ|L_start − L_end| over the pixels the change moves by more than 20 % (L_start, L_end: means of 16 settled frames).
 *   `floor` is the same ratio for single settled frames (noise alone). The lighting is restored afterwards.
 * - Motion (with the scripted motion on): the view settles at phase 0, the path runs to frame `motionFrames`, and at frames `keep` the final
 *   image is kept; then the camera is held at each kept pose, the history cut, and the settled image there is the reference. Per pose: relMSE
 *   Σ(m − r)² / (r² + 1e-2) per pixel (mean), the share of pixels off by more than 20 % + 0.01 (trails and smears; noise after the chain rarely
 *   is), and the relative mean bias.
 */
async function measureLag(o: { light?: 'lamps' | 'sun' | 'none'; settle?: number; after?: number; keep?: readonly number[] } = {}): Promise<Record<string, number>> {
  const l = live;
  if (!l) return {};
  await idle();
  building = true;
  const result: Record<string, number> = {};
  const settle = o.settle ?? 128;
  const after = o.after ?? 32;
  const motion = value('motion');
  const keepAt = o.keep ?? [8, 20, 40, 59];
  try {
    // --- lighting step
    const light = o.light ?? (isInterior(l.scene.name) ? (!$('interiorLights').closest('label')!.hidden ? 'lamps' : 'sun') : 'none');
    if (light !== 'none') {
      motionHold = true;
      motionFrame = 0;
      l.session.cutNext();
      const field = light === 'lamps' ? $<HTMLInputElement>('interiorLights') : $<HTMLInputElement>('sunStrength');
      const before = light === 'lamps' ? field.checked : field.valueAsNumber;
      const set = (v: boolean | number) => {
        if (typeof v === 'boolean') field.checked = v;
        else field.value = String(v);
        // The controls cut the history on a lighting change (the showcase wants an instant switch); the lag test is about the chain that keeps it.
        const session = l.session as { cutNext: () => void };
        session.cutNext = () => {};
        try {
          updateDaylight(l.session);
        } finally {
          delete (session as Partial<typeof session>).cutNext;
        }
      };
      try {
        const start = await settledLuma(l, settle, 16, 'Lag: settling');
        set(light === 'lamps' ? false : (before as number) * 0.5);
        const run = await framesThen(l, after, 'Lag: after the lighting change', () => finalLuma(l));
        await framesThen(l, settle - 16, 'Lag: settling on the new lighting');
        const lastFrames = await framesThen(l, 16, 'Lag: settled', () => finalLuma(l));
        const end = new Float32Array(start.length);
        for (const img of lastFrames) for (let i = 0; i < end.length; i++) end[i]! += img[i]! / 16;
        const mask: number[] = [];
        let step = 0;
        for (let i = 0; i < end.length; i++) {
          const d = Math.abs(start[i]! - end[i]!);
          if (d > 0.2 * Math.max(start[i]!, end[i]!) && d > 1e-3) {
            mask.push(i);
            step += d;
          }
        }
        const resid = (img: Float32Array) => mask.reduce((s, i) => s + Math.abs(img[i]! - end[i]!), 0) / Math.max(step, 1e-12);
        const r = run.map(resid);
        result.light_pixels = mask.length / end.length;
        for (const t of [1, 2, 4, 8, 16, 32]) if (t <= r.length) result[`light_r${t}`] = r[t - 1]!;
        result.light_floor = lastFrames.map(resid).reduce((s, x) => s + x, 0) / lastFrames.length;
        const firstBelow = (x: number) => r.findIndex((v) => v < x) + 1 || Infinity;
        result.light_frames_to_half = firstBelow(0.5);
        result.light_frames_to_tenth = firstBelow(Math.max(0.1, 1.5 * result.light_floor));
      } finally {
        set(before);
        motionHold = false;
      }
    }
    // --- motion
    if (motion !== 'off') {
      motionHold = true;
      motionFrame = 0;
      l.session.cutNext();
      await framesThen(l, settle, 'Ghosting: settling at the start pose');
      motionHold = false;
      const last = Math.max(...keepAt);
      const kept = new Map<number, Float32Array>();
      await framesThen(l, last + 1, 'Ghosting: moving', async (i) => {
        // renderOne draws pose `motionFrame` and then advances it: the pose just drawn is one behind.
        const pose = motionFrame - 1;
        if (keepAt.includes(pose)) kept.set(pose, await finalLuma(l));
        return i;
      });
      motionHold = true;
      for (const [pose, m] of kept) {
        motionFrame = pose;
        l.session.cutNext();
        const ref = await settledLuma(l, settle, 16, `Ghosting: reference at pose ${pose}`);
        let rel = 0;
        let off = 0;
        let sm = 0;
        let sr = 0;
        for (let i = 0; i < ref.length; i++) {
          const d = m[i]! - ref[i]!;
          rel += (d * d) / (ref[i]! * ref[i]! + 1e-2);
          if (Math.abs(d) > 0.2 * ref[i]! + 0.01) off++;
          sm += m[i]!;
          sr += ref[i]!;
        }
        result[`motion_relmse_f${pose}`] = rel / ref.length;
        result[`motion_off20_f${pose}`] = off / ref.length;
        result[`motion_bias_f${pose}`] = sm / Math.max(sr, 1e-12) - 1;
      }
      const poses = [...kept.keys()];
      result.motion_relmse = poses.reduce((s, p) => s + result[`motion_relmse_f${p}`]!, 0) / poses.length;
      result.motion_off20 = poses.reduce((s, p) => s + result[`motion_off20_f${p}`]!, 0) / poses.length;
      motionFrame = 0;
    }
    const table = $('flickerTable').querySelector('tbody') as HTMLTableSectionElement;
    table.replaceChildren(row('lag / ghosting', ''), ...Object.entries(result).map(([k, v]) => row(k, fmt(v))));
    console.info('chain-debug lag', result);
  } finally {
    motionHold = false;
    building = false;
    setStatus('');
  }
  return result;
}

async function savePasses(): Promise<void> {
  const l = live;
  if (!l) return;
  await idle();
  building = true;
  try {
    const a = l.session.arms[arm()];
    const tag = `chain-debug-${l.scene.name}-${arm()}-${value('jitter')}-f${frameIndex}`;
    const ids = ['final', 'denoised', 'raw', 'diffuse', 'specular', 'albedo', 'specF', 'normal', 'motion', 'emission', 'accum', 'svgfDiffuse', 'svgfSpecular'];
    let saved = 0;
    for (const id of ids) {
      const tex = viewById.get(id)?.get(a);
      if (!tex || tex.format !== 'rgba16float' || !(tex.usage & GPUTextureUsage.COPY_SRC)) continue;
      await uploadImage(`${tag}-${id}`, await readTextureRgba16f(l.session.device, tex));
      saved++;
    }
    $('saveNote').textContent = `Saved ${saved} passes as ${tag}-*.pfm in bench/results/incoming/.`;
  } catch (error) {
    $('saveNote').textContent = `Save failed: ${error instanceof Error ? error.message : String(error)}. Saving needs the page served by bench/live.ts.`;
  } finally {
    building = false;
  }
}

// ---------------------------------------------------------------- controls

function fillViewSelects(): void {
  const groups = [...new Set(VIEWS.map((v) => v.group))];
  const mk = () =>
    groups.map((g) => {
      const og = document.createElement('optgroup');
      og.label = g;
      og.append(...VIEWS.filter((v) => v.group === g).map((v) => new Option(v.label, v.id)));
      return og;
    });
  for (const k of [0, 1, 2, 3]) $<HTMLSelectElement>(`view${k}`).replaceChildren(...mk());
  $<HTMLSelectElement>('flickerSrc').replaceChildren(...FLICKER_SOURCES.map((id) => new Option(viewById.get(id)?.label ?? id, id)));
}

function syncRange(): void {
  const v = viewSettings(0);
  $<HTMLInputElement>('lo').value = v.mode === 'scalar' ? String(v.lo ?? 0) : '';
  $<HTMLInputElement>('hi').value = v.mode === 'scalar' ? String(v.hi ?? 1) : '';
  $<HTMLInputElement>('logScale').checked = !!v.log;
}

function syncLayout(): void {
  const single = value('layout') === 'single';
  $('tags').className = `tags ${single ? 'single' : 'grid'}`;
  for (const k of [1, 2, 3]) {
    ($<HTMLSelectElement>(`view${k}`).closest('label') as HTMLElement).hidden = single;
    (($('tags').querySelectorAll('span')[k]) as HTMLElement).hidden = single;
  }
}

function canvasUv(e: PointerEvent): { k: number; uv: [number, number] } | null {
  const r = canvas.getBoundingClientRect();
  const px = ((e.clientX - r.left) / r.width) * canvas.width;
  const py = ((e.clientY - r.top) / r.height) * canvas.height;
  const vps = viewports();
  const k = vps.findIndex((vp) => px >= vp.x && px < vp.x + vp.w && py >= vp.y && py < vp.y + vp.h);
  if (k < 0) return null;
  const vp = vps[k]!;
  const cr = crop();
  return { k, uv: [cr[0] + ((px - vp.x) / vp.w) * cr[2], cr[1] + ((py - vp.y) / vp.h) * cr[3]] };
}

let dragged = false;
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  dragged = false;
});
canvas.addEventListener('pointermove', (e) => {
  if (!e.buttons) return;
  if (Math.abs(e.movementX) + Math.abs(e.movementY) > 0) dragged = true;
  fly.look(e.movementX * 0.004, -e.movementY * 0.004);
});
canvas.addEventListener('pointerup', (e) => {
  if (dragged) return;
  const hit = canvasUv(e);
  if (!hit) return;
  probe = hit.uv;
  if (paused && live) draw(live);
  void readProbe();
});
window.addEventListener('keydown', (e) => {
  if ((e.target instanceof Element && e.target.closest('input, select, textarea')) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === 'Space') {
    e.preventDefault();
    togglePause();
    return;
  }
  if (e.code === 'Period') {
    stepRequests++;
    return;
  }
  keys.add(e.code);
  if (e.code.startsWith('Arrow')) e.preventDefault();
});
window.addEventListener('keyup', (e) => {
  keys.delete(e.code);
  if (e.key === 'Meta') keys.clear();
});
window.addEventListener('blur', () => keys.clear());
document.addEventListener('visibilitychange', () => {
  keys.clear();
  lastTick = 0;
});

function togglePause(): void {
  paused = !paused;
  stepRequests = 0;
  $('pause').textContent = paused ? 'Resume' : 'Pause';
  lastFrameEnd = 0;
  fps = 0;
  fly.stop();
}

for (const id of ['scene', 'res', 'displayScale', 'maxDepth', 'jitter', 'textureLod']) $(id).addEventListener('change', () => {
  if (id === 'scene') {
    history.replaceState(null, '', location.pathname + location.search);
    fillPins();
  }
  void build();
});
for (const id of ['denoiserSel', 'oidnAux', 'stillSamples']) $(id).addEventListener('change', () => {
  applyDenoiser();
  dropFlicker();
});
for (const id of ['svgfAlpha', 'maxHistory', 'planeDistance', 'normalThreshold', 'normalTest', 'historyFallback', 'historyFootprint', 'footprintDwell', 'historyFilter', 'antiLag', 'atrousIterations', 'kernel', 'sigmaDepth', 'sigmaNormal', 'sigmaLuminance', 'reflectionReprojection', 'reflectionHitTest', 'antiLagHistoryScale', 'upscalePath', 'sharpness', 'upHistory', 'reactive', 'rcasDenoise', 'upAutoExposure', 'fireflyClamp']) $(id).addEventListener('change', applyChain);
$('reactiveCap').addEventListener('change', () => void build());
for (const id of ['arm', 'accumulateStill', 'svgfHistoryLimit', 'spp']) $(id).addEventListener('change', () => {
  live?.session.cutNext();
  dropFlicker();
});
$('flickerSrc').addEventListener('change', dropFlicker);
$('reproject').addEventListener('change', dropFlicker);
$('view0').addEventListener('change', syncRange);
$('layout').addEventListener('change', syncLayout);
$('motion').addEventListener('change', () => {
  motionFrame = 0;
  dropFlicker();
});
$('pause').addEventListener('click', togglePause);
$('stepOne').addEventListener('click', () => {
  if (!paused) togglePause();
  stepRequests++;
});
$('resetView').addEventListener('click', () => {
  if (live) resetView(live.scene, live.session.bounds);
  live?.session.cutNext();
});
$('pinView').addEventListener('click', () => {
  const name = $<HTMLInputElement>('pinName').value.trim();
  const p = { ...currentPin(), ...(name ? { name } : {}) };
  try {
    localStorage.setItem(PIN_STORE, JSON.stringify([...storedPins(), p]));
  } catch {
    // no storage: the link still has it
  }
  history.replaceState(null, '', pinToHash(p));
  $<HTMLInputElement>('pinName').value = '';
  fillPins();
  $('pinNote').textContent = `Pinned ${name || 'this view'}: ${pinToHash(p).slice(5)}`;
});
$('pinSel').addEventListener('change', () => {
  const p = scenePins()[Number(value('pinSel'))];
  if (p) {
    applyPin(p);
    history.replaceState(null, '', pinToHash(p));
  }
});
$('gotoPin').addEventListener('click', () => {
  const p = scenePins()[Number(value('pinSel'))] ?? pinFromHash();
  if (p) applyPin(p);
  else $('pinNote').textContent = 'No pin for this scene yet.';
});
$('forgetPin').addEventListener('click', () => {
  const pins = scenePins();
  const gone = pins[Number(value('pinSel'))];
  if (!gone) return;
  try {
    localStorage.setItem(PIN_STORE, JSON.stringify(storedPins().filter((p) => JSON.stringify(p) !== JSON.stringify(gone))));
  } catch {
    // no storage
  }
  fillPins();
});
$('copyLink').addEventListener('click', () => {
  const url = `${location.origin}${location.pathname}?${new URLSearchParams({ scene: value('scene'), res: value('res'), jitter: value('jitter'), denoiser: value('denoiserSel'), layout: value('layout'), views: [0, 1, 2, 3].map((k) => value(`view${k}`)).join(',') }).toString()}${pinToHash(currentPin())}`;
  void navigator.clipboard?.writeText(url).then(() => ($('pinNote').textContent = 'Link copied.'), () => ($('pinNote').textContent = url));
});
$('resetCache').addEventListener('click', () => {
  if (!live) return;
  live.session.cache.resetNetwork();
  live.session.cutNext();
  if (live.networkKey) void forgetNetwork(live.networkKey);
});
$('saveNet').addEventListener('click', () => {
  const l = live;
  if (!l?.networkKey) return;
  const lighting = lightingState();
  void l.session.cache.exportNetwork().then(({ weights, trained }) => saveNetwork({ key: l.networkKey, lighting, weights, trained, savedAt: Date.now() })).then((ok) => ($('saveNote').textContent = ok ? 'Network saved for this scene and lighting.' : 'Saving the network failed.'));
});
$('measure').addEventListener('click', () => void measureFlicker());
$('measureLag').addEventListener('click', () => void measureLag());
$('stats').addEventListener('click', () => {
  const id = value('view0');
  void passStats(id, { maskHit: id !== 'final' }).then((r) => {
    $('statsNote').textContent = typeof r === 'string' ? r : `${viewById.get(id)?.label}${id !== 'final' ? ' (pixels the primary ray hit)' : ''}: mean ${fmt(r.mean as number)} · p10 ${fmt(r.p10 as number)} · p50 ${fmt(r.p50 as number)} · p90 ${fmt(r.p90 as number)}${r.nonFinite ? ` · ${r.nonFinite} non-finite` : ''}`;
  });
});
$('savePasses').addEventListener('click', () => void savePasses());
for (const id of LIGHTING_FIELDS) $(id).addEventListener('input', () => {
  if (live && isInterior(live.scene.name)) updateDaylight(live.session);
});

// ---------------------------------------------------------------- start

fillViewSelects();
const args = new URLSearchParams(location.search);
const pick = (id: string, v: string | null) => {
  const el = $<HTMLSelectElement>(id);
  if (v && [...el.options].some((o) => o.value === v)) el.value = v;
};
pick('scene', args.get('scene'));
pick('res', args.get('res'));
pick('jitter', args.get('jitter'));
pick('textureLod', args.get('textureLod'));
pick('denoiserSel', args.get('denoiser'));
pick('layout', args.get('layout'));
pick('arm', args.get('arm'));
pick('motion', args.get('motion'));
const defaultViews = (args.get('views') ?? 'final,history,denoised,raw').split(',');
for (const k of [0, 1, 2, 3]) pick(`view${k}`, defaultViews[k] ?? null);
pick('flickerSrc', args.get('flickerSrc') ?? 'final');
syncLayout();
syncRange();
fillPins();
await build();
const startPin = pinFromHash() ?? storedPin();
if (startPin) applyPin(startPin);
requestAnimationFrame(frame);

/** Headless use (a hidden window pauses rAF): n frames, waited for. */
async function step(n: number): Promise<void> {
  const l = live;
  if (!l) return;
  await idle();
  building = true;
  try {
    for (let i = 0; i < n; i++) {
      await renderOne(l);
      await l.session.device.queue.onSubmittedWorkDone();
    }
  } finally {
    building = false;
  }
}
/**
 * Edge history (N11.17): over `frames` rendered frames, per render pixel, how often the primary-vertex guides changed surface between
 * consecutive frames (normal cosine below `cos`, depth off by more than `dz` relative, or hit / miss changed) and what SVGF's diffuse
 * history length is at the end. Splits the pixels whose history is below `cap` into "guides flipped" and "guides stable", and returns
 * the bounding boxes of the largest stable-but-short clusters (16 px cells) so they can be found on screen. Readbacks between frames.
 */
async function edgeHistory(o: { frames?: number; cos?: number; dz?: number; cap?: number; trace?: [number, number][] } = {}): Promise<Record<string, unknown> | string> {
  const l = live;
  if (!l) return 'no session';
  const frames = o.frames ?? 32;
  const cosT = o.cos ?? 0.95;
  const dzT = o.dz ?? 0.02;
  const a = l.session.arms[arm()];
  const dev = l.session.device;
  const w = a.tex.normal.width;
  const h = a.tex.normal.height;
  const n = w * h;
  const flips = new Uint16Array(n);
  const minHist = new Float32Array(n).fill(1e9);
  let pn: Float32Array | null = null;
  let pz: Float32Array | null = null;
  let hist: Float32Array | null = null;
  let hit: Float32Array | null = null;
  const traces: Record<string, unknown[]> = {};
  for (let f = 0; f < frames; f++) {
    await step(1);
    await idle();
    building = true;
    try {
      const cn = await dumpTexture(dev, a.tex.normal);
      const cz = await dumpTexture(dev, a.tex.viewZ);
      const acc = a.denoiserDebug('accum');
      hist = acc ? await dumpTexture(dev, acc) : null;
      hit = await dumpTexture(dev, a.tex.specF);
      for (let i = 0; i < n; i++) {
        if (hist) minHist[i] = Math.min(minHist[i] as number, hist[4 * i + 3] as number);
        if (!pn || !pz) continue;
        const z0 = -(pz[4 * i] as number);
        const z1 = -(cz[4 * i] as number);
        const l0 = Math.hypot(pn[4 * i] as number, pn[4 * i + 1] as number, pn[4 * i + 2] as number);
        const l1 = Math.hypot(cn[4 * i] as number, cn[4 * i + 1] as number, cn[4 * i + 2] as number);
        const s0 = l0 > 0.1 && z0 > 0;
        const s1 = l1 > 0.1 && z1 > 0;
        if (!s0 && !s1) continue;
        let flip = s0 !== s1;
        if (!flip) {
          const d = ((pn[4 * i] as number) * (cn[4 * i] as number) + (pn[4 * i + 1] as number) * (cn[4 * i + 1] as number) + (pn[4 * i + 2] as number) * (cn[4 * i + 2] as number)) / (l0 * l1);
          flip = d < cosT || Math.abs(z1 - z0) > dzT * Math.max(z0, z1);
        }
        if (flip) flips[i] = (flips[i] as number) + 1;
      }
      for (const [x, y] of o.trace ?? []) {
        const i = y * w + x;
        (traces[`${x},${y}`] ??= []).push({ f, j: [Number(a.jitter.x.toFixed(3)), Number(a.jitter.y.toFixed(3))], n: [cn[4 * i], cn[4 * i + 1], cn[4 * i + 2]].map((v) => Number((v as number).toFixed(3))), z: Number((-(cz[4 * i] as number)).toFixed(4)), hist: hist ? Number((hist[4 * i + 3] as number).toFixed(2)) : null });
      }
      pn = cn;
      pz = cz;
    } finally {
      building = false;
    }
  }
  if (!hist || !hit) return 'no SVGF history (denoiser not SVGF?)';
  const cap = o.cap ?? 63.5;
  let hitPx = 0, short = 0, low8 = 0, flipped = 0, flippedShort = 0, flippedLow8 = 0, stableShort = 0, stableLow8 = 0;
  const cell = 16;
  const cw = Math.ceil(w / cell);
  const cells = new Uint32Array(cw * Math.ceil(h / cell));
  const flipHist: number[] = [];
  const stableShortPx: [number, number, number][] = [];
  for (let i = 0; i < n; i++) {
    if ((hit[4 * i + 3] as number) <= 0.5) continue;
    hitPx++;
    const hl = hist[4 * i + 3] as number;
    const fl = (flips[i] as number) > 0;
    if (fl) { flipped++; flipHist.push(hl); }
    if (hl < cap) {
      short++;
      if (fl) flippedShort++;
      else {
        stableShort++;
        cells[Math.floor(Math.floor(i / w) / cell) * cw + Math.floor((i % w) / cell)]!++;
        if (stableShortPx.length < 4000) stableShortPx.push([i % w, Math.floor(i / w), hl]);
      }
    }
    if (hl <= 8) { low8++; if (fl) flippedLow8++; else stableLow8++; }
  }
  flipHist.sort((x, y) => x - y);
  const q = (f: number) => flipHist[Math.min(flipHist.length - 1, Math.floor(f * flipHist.length))] ?? NaN;
  const top = [...cells].map((c, k) => ({ c, x: (k % cw) * cell, y: Math.floor(k / cw) * cell })).filter((c) => c.c > 0).sort((p, r) => r.c - p.c).slice(0, 12);
  const out = {
    frames, width: w, height: h, hitPixels: hitPx,
    belowCap: short / hitPx, atOrBelow8: low8 / hitPx,
    flippedShare: flipped / hitPx,
    belowCapFlipped: flippedShort / Math.max(1, short), belowCapStable: stableShort / Math.max(1, short),
    low8Flipped: flippedLow8 / Math.max(1, low8), low8Stable: stableLow8 / Math.max(1, low8),
    flippedHistory: { p10: q(0.1), p50: q(0.5), p90: q(0.9), belowCap: flipHist.filter((x) => x < cap).length / Math.max(1, flipHist.length) },
    stableShortCells: top, traces,
  };
  (window as unknown as { __edgeHistory: unknown }).__edgeHistory = { ...out, flips, minHist, stableShortPx };
  return out;
}
/**
 * Guide stability (N11.17): the frame-to-frame change of every guide plane at a still camera over `frames` frames, split into edge pixels
 * (a depth step above `edge` (relative) or a normal cosine below 0.95 against a 4-neighbour in ANY of the frames, dilated by one pixel:
 * geometric coverage under jitter) and interior pixels (texture aliasing, Monte Carlo noise). The mask is built from all frames first
 * (two passes over the kept dumps), so an edge the jitter only sometimes shows still counts. `bitDiff_<plane>` = the share of float
 * values that differ at all between consecutive frames (0 = bit-identical planes). Per plane and region: the mean change
 * (normal: 1 - cosine; depth: relative; albedo / specular factor: relative luminance; roughness: absolute) and the share of pixel-frames
 * whose change is above the plane's threshold (normal cosine < 0.95, depth 2 %, albedo / specular factor 5 %, roughness 0.05).
 */
async function guideChange(o: { frames?: number; edge?: number; worst?: 'albedo' | 'normal' | 'viewZ' | 'specF' | 'roughness' } = {}): Promise<Record<string, unknown> | string> {
  const l = live;
  if (!l) return 'no session';
  const frames = o.frames ?? 16;
  const a = l.session.arms[arm()];
  const dev = l.session.device;
  const w = a.tex.normal.width;
  const h = a.tex.normal.height;
  const n = w * h;
  const planes = ['normal', 'viewZ', 'albedo', 'specF', 'roughness'] as const;
  const tex = { normal: a.tex.normal, viewZ: a.tex.viewZ, albedo: a.tex.albedo, specF: a.tex.specF, roughness: a.tex.roughness };
  const thr = { normal: 0.05, viewZ: 0.02, albedo: 0.05, specF: 0.05, roughness: 0.05 };
  const sum = { edge: Object.fromEntries(planes.map((k) => [k, 0])), interior: Object.fromEntries(planes.map((k) => [k, 0])) } as Record<string, Record<string, number>>;
  const over = { edge: Object.fromEntries(planes.map((k) => [k, 0])), interior: Object.fromEntries(planes.map((k) => [k, 0])) } as Record<string, Record<string, number>>;
  const count = { edge: 0, interior: 0 } as Record<string, number>;
  const lum = (d: Float32Array, i: number) => 0.2126 * (d[4 * i] as number) + 0.7152 * (d[4 * i + 1] as number) + 0.0722 * (d[4 * i + 2] as number);
  const dumps: Record<string, Float32Array>[] = [];
  for (let f = 0; f <= frames; f++) {
    await step(1);
    await idle();
    building = true;
    try {
      const cur: Record<string, Float32Array> = {};
      for (const k of planes) cur[k] = await dumpTexture(dev, tex[k]);
      dumps.push(cur);
    } finally {
      building = false;
    }
  }
  const raw = new Uint8Array(n);
  const edge = o.edge ?? 0.03;
  for (const cur of dumps) {
    const cz = cur.viewZ as Float32Array, cn = cur.normal as Float32Array;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const z0 = -(cz[4 * i] as number);
      for (const [dx, dy] of [[1, 0], [0, 1]] as const) {
        if (x + dx >= w || y + dy >= h) continue;
        const j = (y + dy) * w + x + dx;
        const z1 = -(cz[4 * j] as number);
        const dn = (cn[4 * i] as number) * (cn[4 * j] as number) + (cn[4 * i + 1] as number) * (cn[4 * j + 1] as number) + (cn[4 * i + 2] as number) * (cn[4 * j + 2] as number);
        if ((z0 > 0) !== (z1 > 0) || Math.abs(z1 - z0) > edge * Math.max(z0, z1) || (z0 > 0 && dn < 0.95)) { raw[i] = 1; raw[j] = 1; }
      }
    }
  }
  const edgeMask = new Uint8Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let e = 0;
    for (let dy = -1; dy <= 1 && !e; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && raw[yy * w + xx]) { e = 1; break; }
    }
    edgeMask[y * w + x] = e;
  }
  const bits = Object.fromEntries(planes.map((k) => [k, 0])) as Record<string, number>;
  const changed = { edge: Object.fromEntries(planes.map((k) => [k, 0])), interior: Object.fromEntries(planes.map((k) => [k, 0])) } as Record<string, Record<string, number>>;
  const worst: { x: number; y: number; d: number }[] = [];
  // interior albedo change: pixel-frames above 5 % / 20 % / 50 %, the summed change of those above 20 % (sharp albedo edges inside a
  // surface: decals, print, panel joints), and how many pixel-frames changed the class flag (albedo alpha = share of specular-class samples)
  const alb = { o5: 0, o20: 0, o50: 0, sum20: 0 };
  let classFlips = 0;
  for (let f = 1; f < dumps.length; f++) {
    const prev = dumps[f - 1]!, cur = dumps[f]!;
    const pn = prev.normal as Float32Array, cn = cur.normal as Float32Array, pz = prev.viewZ as Float32Array, cz = cur.viewZ as Float32Array;
    for (const k of planes) {
      const a0 = new Uint32Array((prev[k] as Float32Array).buffer), a1 = new Uint32Array((cur[k] as Float32Array).buffer);
      let c = 0;
      for (let i = 0; i < a0.length; i++) if (a0[i] !== a1[i]) c++;
      bits[k] = (bits[k] as number) + c / a0.length / frames;
    }
    for (let i = 0; i < n; i++) {
      const z0 = -(pz[4 * i] as number), z1 = -(cz[4 * i] as number);
      if (!(z0 > 0) || !(z1 > 0)) continue;
      const r = edgeMask[i] ? 'edge' : 'interior';
      count[r]!++;
      const l0 = Math.hypot(pn[4 * i] as number, pn[4 * i + 1] as number, pn[4 * i + 2] as number) || 1;
      const l1 = Math.hypot(cn[4 * i] as number, cn[4 * i + 1] as number, cn[4 * i + 2] as number) || 1;
      const d: Record<string, number> = {
        normal: Math.max(0, 1 - ((pn[4 * i] as number) * (cn[4 * i] as number) + (pn[4 * i + 1] as number) * (cn[4 * i + 1] as number) + (pn[4 * i + 2] as number) * (cn[4 * i + 2] as number)) / (l0 * l1)),
        viewZ: Math.abs(z1 - z0) / Math.max(z0, z1),
        roughness: Math.abs(((cur.roughness as Float32Array)[4 * i] as number) - ((prev.roughness as Float32Array)[4 * i] as number)),
      };
      for (const k of ['albedo', 'specF'] as const) {
        const p0 = lum(prev[k] as Float32Array, i), p1 = lum(cur[k] as Float32Array, i);
        d[k] = Math.abs(p1 - p0) / Math.max(1e-4, 0.5 * (p0 + p1));
      }
      for (const k of planes) {
        sum[r]![k]! += d[k] as number;
        if ((d[k] as number) > thr[k]) over[r]![k]!++;
        if ((d[k] as number) > 1e-6) changed[r]![k]!++;
      }
      if (Math.abs(((cur.albedo as Float32Array)[4 * i + 3] as number) - ((prev.albedo as Float32Array)[4 * i + 3] as number)) > 1e-6) classFlips++;
      if (r === 'interior') {
        const da = d.albedo as number;
        if (da > 0.05) alb.o5++;
        if (da > 0.2) { alb.o20++; alb.sum20 += da; }
        if (da > 0.5) alb.o50++;
      }
      if (r === 'interior' && f === 1 && (d[o.worst ?? 'albedo'] as number) > 0.2 && worst.length < 400) worst.push({ x: i % w, y: Math.floor(i / w), d: Number((d[o.worst ?? 'albedo'] as number).toFixed(3)) });
    }
  }
  const out: Record<string, unknown> = { frames, edgeShare: count.edge! / Math.max(1, count.edge! + count.interior!) };
  for (const k of planes) out[`bitDiff_${k}`] = Number((bits[k] as number).toExponential(2));
  // per region and plane: [mean change, share of pixel-frames over the threshold, share that changed at all]
  for (const r of ['edge', 'interior']) for (const k of planes) out[`${r}_${k}`] = [Number((sum[r]![k]! / Math.max(1, count[r]!)).toExponential(2)), Number((over[r]![k]! / Math.max(1, count[r]!)).toFixed(4)), Number((changed[r]![k]! / Math.max(1, count[r]!)).toFixed(4))];
  const ni = Math.max(1, count.interior!);
  out.interiorAlbedo = { over5: Number((alb.o5 / ni).toFixed(4)), over20: Number((alb.o20 / ni).toFixed(4)), over50: Number((alb.o50 / ni).toFixed(4)), meanWithoutOver20: Number(((sum.interior!.albedo! - alb.sum20) / ni).toExponential(2)) };
  out.classFlipShare = Number((classFlips / Math.max(1, count.edge! + count.interior!)).toFixed(4));
  out.worstInterior = worst.slice(0, 40);
  return out;
}
/** Mean linear RGB of a stage over `frames` frames at the current (still) pose, and per pixel the mean of |L(t) - L(t-1)| (luminance). */
async function stageMean(l: Live, get: (a: PipelineArm) => GPUTexture | undefined, frames: number): Promise<{ mean: Float32Array; change: Float32Array; width: number; height: number }> {
  const a = l.session.arms[arm()];
  let mean: Float32Array | null = null;
  let change: Float32Array | null = null;
  let prevL: Float32Array | null = null;
  let width = 0, height = 0;
  for (let f = 0; f < frames; f++) {
    await step(1);
    await idle();
    building = true;
    try {
      const tex = get(a);
      if (!tex) throw new Error('stage not produced');
      const d = await dumpTexture(l.session.device, tex);
      width = tex.width; height = tex.height;
      const n = width * height;
      mean ??= new Float32Array(3 * n);
      change ??= new Float32Array(n);
      const cl = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const r = d[4 * i] as number, g = d[4 * i + 1] as number, b = d[4 * i + 2] as number;
        mean[3 * i] = (mean[3 * i] as number) + r / frames; mean[3 * i + 1] = (mean[3 * i + 1] as number) + g / frames; mean[3 * i + 2] = (mean[3 * i + 2] as number) + b / frames;
        cl[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if (prevL) change[i] = (change[i] as number) + Math.abs((cl[i] as number) - (prevL[i] as number)) / (frames - 1);
      }
      prevL = cl;
    } finally {
      building = false;
    }
  }
  return { mean: mean as Float32Array, change: change as Float32Array, width, height };
}

let edgeRef: { key: string; mean: Float32Array; width: number; height: number; cls: Uint8Array } | null = null;

/**
 * Edge quality at a still camera (N11.17). Reference: the mean of the raw beauty over `refFrames` frames at `refSpp` samples per frame
 * (kept per camera pose, jitter owner and texture LOD; the pixel-area average under a per-frame jitter). Pixel classes, from a 24-frame
 * `edgeHistory` window run when the reference is made (so make the first call with anti-lag off and footprint history off: the classes
 * then say what the plain reprojection test does): `lost` = guides changed surface and the SVGF history is below the cap, `kept` =
 * guides changed surface and the history is at the cap, `rest` = guides stable. Then over `frames` frames, per class: the mean absolute
 * luminance error and the mean signed error of the mean denoised image against the reference, over the mean reference luminance (the
 * filter's bias), and the still flicker (mean |dL| / mean L per frame) of the denoised and final stages. Many readbacks: a measurement,
 * not a frame loop.
 */
async function edgeQuality(o: { frames?: number; refFrames?: number; refSpp?: number } = {}): Promise<Record<string, number> | string> {
  const l = live;
  if (!l) return 'no session';
  const a0 = l.session.arms[arm()];
  const key = `${cameraKey(frameCamera())}|${l.jitter}|${arm()}|${a0.tex.raw.width}|${value('textureLod')}`;
  if (edgeRef?.key !== key) {
    const e = await edgeHistory({ frames: 24 });
    if (typeof e === 'string') return e;
    const flips = (window as unknown as { __edgeHistory: { flips: Uint16Array } }).__edgeHistory.flips;
    const acc = a0.denoiserDebug('accum');
    const hist = acc ? await dumpTexture(l.session.device, acc) : null;
    const cls = new Uint8Array(flips.length);
    for (let i = 0; i < flips.length; i++) cls[i] = (flips[i] as number) > 0 ? (hist && (hist[4 * i + 3] as number) < 63.5 ? 2 : 1) : 0;
    const spp = $<HTMLInputElement>('spp').value;
    $<HTMLInputElement>('spp').value = String(o.refSpp ?? 16);
    const r = await stageMean(l, (a) => a.tex.raw, o.refFrames ?? 128);
    $<HTMLInputElement>('spp').value = spp;
    edgeRef = { key, mean: r.mean, width: r.width, height: r.height, cls };
    await step(80);
  }
  const ref = edgeRef;
  const frames = o.frames ?? 64;
  const den = await stageMean(l, (a) => a.stage('denoised'), frames);
  const fin = await stageMean(l, (a) => a.stage('final'), Math.max(8, frames >> 1));
  const hit = await dumpTexture(l.session.device, a0.tex.specF);
  const lum = (m: Float32Array, i: number) => 0.2126 * (m[3 * i] as number) + 0.7152 * (m[3 * i + 1] as number) + 0.0722 * (m[3 * i + 2] as number);
  const names = ['rest', 'kept', 'lost'] as const;
  const acc = names.map(() => ({ n: 0, ch: 0, l: 0, bias: 0, refL: 0, abs: 0, fch: 0, fl: 0 }));
  const n = ref.width * ref.height;
  let hitN = 0;
  for (let i = 0; i < n; i++) {
    if ((hit[4 * i + 3] as number) <= 0.5) continue;
    hitN++;
    const s = acc[ref.cls[i] as number]!;
    const d = lum(den.mean, i) - lum(ref.mean, i);
    s.n++; s.ch += den.change[i] as number; s.l += lum(den.mean, i);
    s.bias += d; s.abs += Math.abs(d); s.refL += lum(ref.mean, i);
  }
  const sx = fin.width / ref.width, sy = fin.height / ref.height;
  for (let y = 0; y < fin.height; y++) for (let x = 0; x < fin.width; x++) {
    const ri = Math.min(ref.height - 1, Math.floor(y / sy)) * ref.width + Math.min(ref.width - 1, Math.floor(x / sx));
    if ((hit[4 * ri + 3] as number) <= 0.5) continue;
    const s = acc[ref.cls[ri] as number]!;
    const i = y * fin.width + x;
    s.fch += fin.change[i] as number; s.fl += lum(fin.mean, i);
  }
  const r4 = (v: number) => Number(v.toPrecision(4));
  const out: Record<string, number> = {};
  names.forEach((k, c) => {
    const s = acc[c]!;
    out[`${k}_share`] = r4(s.n / Math.max(1, hitN));
    out[`${k}_absErr`] = r4(s.abs / s.refL);
    out[`${k}_bias`] = r4(s.bias / s.refL);
    out[`${k}_denFlicker`] = r4(s.ch / s.l);
    out[`${k}_finFlicker`] = r4(s.fch / s.fl);
  });
  return out;
}
(window as unknown as { __chainDebug: unknown }).__chainDebug = { step, measureFlicker, measureLag, readProbe, passStats, edgeHistory, guideChange, edgeQuality, setProbe: (u: number, v: number) => { probe = [u, v]; }, pause: (p: boolean) => { if (p !== paused) togglePause(); }, get live() { return live; }, fly, cameraKey: () => cameraKey(frameCamera()) };
