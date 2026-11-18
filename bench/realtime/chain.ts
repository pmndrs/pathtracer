/**
 * NRC pipeline (track N, M9; docs/design/nrc-pipeline.md): the real-time chain both demo arms go through, identically.
 *
 *   path tracer (integrator.frame, N samples)  ->  pack (planes -> denoiser / upscaler textures)  ->  Denoiser `DiffuseSpecularDenoiser`
 *   (SVGF, demodulated diffuse + specular, remodulated)  ->  composite (+ sky)  ->  pmndrs/upscaler temporal path (render res -> display res)
 *
 * One `PipelineArm` per arm. Jitter: the path tracer's sampler (pair 0, per pixel) is the pipeline's only jitter owner; the denoiser gets
 * `jitter` [0, 0] and the upscaler runs with `jitter: false` (it reprojects, clips and accumulates, but adds no camera offset). Motion is
 * jitter-free by construction (the planes project world points through the two unjittered cameras).
 *
 * Shared by `bench/nrc/pipeline-session.ts` (the cache arms) and `bench/realtime/session.ts` (the plain real-time kernel, P4.5 / P4.6, no cache):
 * the chain does not care what filled the frame planes.
 *
 * Per frame:  arm.beginFrame(encoder); { pt.encode(encoder, 1); arm.accumulate(encoder, pt.frameBuffers) } x N; arm.pack(encoder, N);
 *             submit; arm.post(camera, reset)  (the denoiser and the upscaler record and submit their own command buffers; no readback)
 */
import { DiffuseSpecularDenoiser } from 'denoiser/realtime';
import { ScalarReduction } from '@pmndrs/downsampler/webgpu';
import { Upscaler } from '@pmndrs/upscaler';
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { AUTO_EXPOSURE, cameraKey, cross, norm, PIPE_FAR, PIPE_NEAR, projectionMatrix, sub, viewMatrix, worldMatrix } from './chain-math.ts';

export { PIPE_FAR, PIPE_NEAR, projectionMatrix, resample, viewMatrix, worldMatrix } from './chain-math.ts';


const TEX = GPUTextureUsage.TEXTURE_BINDING;
const COPY_DST_T = GPUTextureUsage.COPY_DST;
const COPY_SRC_T = GPUTextureUsage.COPY_SRC;

/**
 * The camera object the upscaler reads (it is written for three's cameras: `near`, `far`, `isPerspectiveCamera`, `projectionMatrix`,
 * `matrixWorld`, `matrixWorldInverse`, and it calls `updateProjectionMatrix` / `updateMatrixWorld`). Only `.elements` of the matrices is read.
 */
class UpscalerCamera {
  readonly isPerspectiveCamera = true;
  readonly near = PIPE_NEAR;
  readonly far = PIPE_FAR;
  readonly projectionMatrix = { elements: new Float32Array(16) };
  readonly matrixWorld = { elements: new Float32Array(16) };
  readonly matrixWorldInverse = { elements: new Float32Array(16) };
  set(c: PtLookAtCamera, aspect: number): void {
    this.projectionMatrix.elements.set(projectionMatrix(c.yFovDeg, aspect));
    this.matrixWorld.elements.set(worldMatrix(c));
    this.matrixWorldInverse.elements.set(viewMatrix(c));
  }
  updateProjectionMatrix(): void {}
  updateMatrixWorld(): void {}
}

/**
 * The renderer the upscaler expects is three's `WebGPURenderer`; it only uses `backend.device`, `backend.get(texture).texture` and
 * `initTexture(texture)` for its own StorageTextures (three r186 shapes, `upscaler/src/internal/threeWebGPU.ts`). This stand-in serves them
 * from our device, so no three renderer runs. Inputs are registered as plain key objects (`texture()`).
 */
class UpscalerHost {
  private readonly map = new WeakMap<object, { texture: GPUTexture }>();
  readonly device: GPUDevice;
  readonly backend: { device: GPUDevice; get: (o: object) => { texture: GPUTexture } | undefined };
  constructor(device: GPUDevice) {
    this.device = device;
    this.backend = { device, get: (o) => this.map.get(o) };
  }
  /** three StorageTexture -> GPUTexture: (format, type) RGBA + HalfFloat = rgba16float, Red + Float = r32float, RGBA + UnsignedByte = rgba8unorm. */
  initTexture(tex: { image: { width: number; height: number }; format: number; type: number; name?: string }): void {
    if (this.map.has(tex)) return;
    const RGBA = 1023;
    const RED = 1028;
    const FLOAT = 1015;
    const HALF = 1016;
    const UBYTE = 1009;
    const format: GPUTextureFormat | null = tex.format === RGBA && tex.type === HALF ? 'rgba16float' : tex.format === RED && tex.type === FLOAT ? 'r32float' : tex.format === RGBA && tex.type === UBYTE ? 'rgba8unorm' : null;
    if (!format) throw new Error(`UpscalerHost: no format for three texture "${tex.name}" (format ${tex.format}, type ${tex.type})`);
    const texture = this.device.createTexture({ label: tex.name ?? 'upscaler', size: { width: tex.image.width, height: tex.image.height }, format, usage: TEX | GPUTextureUsage.STORAGE_BINDING | COPY_SRC_T | COPY_DST_T });
    this.map.set(tex, { texture });
  }
  /** A key object the upscaler accepts as a three texture, backed by `texture`. */
  texture(texture: GPUTexture, name: string): object {
    const key = { name, uuid: name };
    this.map.set(key, { texture });
    return key;
  }
  gpu(key: object): GPUTexture {
    const t = this.map.get(key)?.texture;
    if (!t) throw new Error('UpscalerHost: unknown texture');
    return t;
  }
}

// ---------------------------------------------------------------- WGSL

/** Per-pixel sums over the N samples of a frame (6 vec4 per pixel): beauty, diffuse-class radiance, specular-class radiance, albedo factor, specular factor, emission (P4.5a). */
const SUM_VEC4 = 6;

const ACCUM_WGSL = /* wgsl */ `
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read> planes: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> sums: array<vec4f>;

@compute @workgroup_size(64)
fn pipeAccumulate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = p.n;
  if (i >= n) { return; }
  let aF = planes[4u * n + i];
  let sF = planes[5u * n + i];
  sums[i] += vec4f(accum[i].xyz, 1.0);
  sums[n + i] += vec4f(planes[i].xyz * aF.xyz, 0.0);       // diffuse-class radiance (re-modulated)
  sums[2u * n + i] += vec4f(planes[n + i].xyz * sF.xyz, 0.0); // specular-class radiance
  sums[3u * n + i] += vec4f(aF.xyz, aF.w);
  sums[4u * n + i] += vec4f(sF.xyz, sF.w);
  sums[5u * n + i] += vec4f(planes[6u * n + i].xyz, 0.0);      // emission: primary-visible emitters / sky, never demodulated
}
`;

/**
 * Pack: sums / N and the last sample's guides -> one buffer of texture rows (copied to the textures after). Sections, each `rows` rows of a
 * 256-byte aligned stride: 0..7 rgba16float (raw beauty, diffuse demod, specular demod, albedo factor, specular factor, normal, motion, emission), then
 * 8..12 r32float (viewZ, motionZ, roughness, diffuse hit distance, specular hit distance).
 */
const PACK_WGSL = /* wgsl */ `
struct P { w: u32, h: u32, spp: f32, stride8: u32, stride4: u32, base4: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> sums: array<vec4f>;
@group(0) @binding(2) var<storage, read> planes: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outWords: array<u32>;

fn put16(section: u32, x: u32, y: u32, v: vec4f) {
  let i = (section * p.h * p.stride8 + y * p.stride8 + x) * 2u;
  outWords[i] = pack2x16float(v.xy);
  outWords[i + 1u] = pack2x16float(v.zw);
}
fn put32(section: u32, x: u32, y: u32, v: f32) {
  outWords[p.base4 + section * p.h * p.stride4 + y * p.stride4 + x] = bitcast<u32>(v);
}
fn clampHalf(v: vec4f) -> vec4f { return clamp(v, vec4f(-65000.0), vec4f(65000.0)); }

@compute @workgroup_size(8, 8)
fn pipePack(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.w || y >= p.h) { return; }
  let n = p.w * p.h;
  let i = y * p.w + x;
  let inv = 1.0 / p.spp;
  let beauty = sums[i].xyz * inv;
  let aF = max(sums[3u * n + i].xyz, vec3f(1e-6));
  let sF = max(sums[4u * n + i].xyz, vec3f(1e-6));
  let diffuse = sums[n + i].xyz / aF;   // sum of re-modulated radiance / sum of factors = radiance / mean factor
  let specular = sums[2u * n + i].xyz / sF;
  let nz = planes[2u * n + i];
  let mo = planes[3u * n + i];
  put16(0u, x, y, clampHalf(vec4f(beauty, 1.0)));
  put16(1u, x, y, clampHalf(vec4f(diffuse, 1.0)));
  put16(2u, x, y, clampHalf(vec4f(specular, 1.0)));
  put16(3u, x, y, vec4f(aF * inv, sums[3u * n + i].w * inv));   // w: share of specular-class samples
  put16(4u, x, y, vec4f(sF * inv, sums[4u * n + i].w * inv));   // w: share of samples whose primary ray hit
  put16(5u, x, y, vec4f(nz.xyz, 0.0));
  put16(6u, x, y, vec4f(mo.xy, 0.0, 0.0));
  put16(7u, x, y, clampHalf(vec4f(sums[5u * n + i].xyz * inv, 1.0)));   // emission: mean over the samples, added back after the filter
  put32(0u, x, y, nz.w);
  put32(1u, x, y, mo.w);
  put32(2u, x, y, mo.z);
  put32(3u, x, y, planes[i].w);
  put32(4u, x, y, planes[n + i].w);
}
`;

/** Device depth from viewZ (three's WebGPU projection; a miss, viewZ 0, is the far plane). */
const DEPTH_WGSL = /* wgsl */ `
struct P { near: f32, far: f32, _a: f32, _b: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var viewZ: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(q * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {
  let z = textureLoad(viewZ, vec2u(pos.xy), 0).r;
  if (z >= 0.0) { return 1.0; }
  let d = -z;
  return clamp(p.far * (d - p.near) / (d * (p.far - p.near)), 0.0, 1.0);
}
`;

/**
 * Denoised diffuse + specular (remodulated) + the emission plane (primary-visible emitters, the sky behind a primary miss, mirror-seen emitters: noise-free, never filtered, P4.5a);
 * the raw beauty where the primary ray missed (the denoiser outputs 0 there; it holds the sky already).
 */
const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var diffuse: texture_2d<f32>;
@group(0) @binding(1) var specular: texture_2d<f32>;
@group(0) @binding(2) var raw: texture_2d<f32>;
@group(0) @binding(3) var specF: texture_2d<f32>;
@group(0) @binding(4) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var emission: texture_2d<f32>;
@compute @workgroup_size(8, 8)
fn pipeComposite(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(outColor);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let q = gid.xy;
  let seen = textureLoad(specF, q, 0).w > 0.5;
  // Pack wrote the mean of "seen" flags into specF.w: a pixel whose samples all missed keeps its raw (sky) value.
  var c = textureLoad(raw, q, 0).xyz;
  if (seen) { c = textureLoad(diffuse, q, 0).xyz + textureLoad(specular, q, 0).xyz + textureLoad(emission, q, 0).xyz; }
  textureStore(outColor, q, vec4f(max(c, vec3f(0.0)), 1.0));
}
`;

/** Per-pixel sum over frames of |L_t - L_(t-1)| (luminance), for the stability measure; read once at the end. */
const FLICKER_WGSL = /* wgsl */ `
struct P { w: u32, h: u32, measure: u32, _a: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var img: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<vec2f>; // (previous luminance, sum of |change|)
@compute @workgroup_size(8, 8)
fn pipeFlicker(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let i = gid.y * p.w + gid.x;
  let l = dot(textureLoad(img, gid.xy, 0).xyz, vec3f(0.2126, 0.7152, 0.0722));
  var s = state[i];
  if (p.measure == 1u) { s.y += abs(l - s.x); }
  state[i] = vec2f(l, s.y);
}
`;

/**
 * Reactive mask for the upscaler (P4.6, R7): `r = strength * clamp(1 - (n - 1) / cap, 0, 1)` from the SVGF history length n (alpha of its
 * accumulation texture, 1 = fresh or disoccluded). Fresh pixels favour the current frame in the upscaler's accumulation, long histories do not.
 */
const REACTIVE_WGSL = /* wgsl */ `
struct P { cap: f32, strength: f32, _a: f32, _b: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var accum: texture_2d<f32>;
@group(0) @binding(2) var outMask: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn pipeReactive(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(outMask);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let n = textureLoad(accum, gid.xy, 0).w;
  let r = p.strength * clamp(1.0 - (n - 1.0) / p.cap, 0.0, 1.0);
  textureStore(outMask, gid.xy, vec4f(r, 0.0, 0.0, 1.0));
}
`;

/** Display transform (the last stage of the chain): exposure, ACES fit (Narkowicz), sRGB encode, rgba8unorm. */
const TONEMAP_WGSL = /* wgsl */ `
struct P { exposure: f32, _a: f32, _b: f32, _c: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
fn aces(x: vec3f) -> vec3f { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0)); }
fn encode(c: vec3f) -> vec3f { return select(12.92 * c, 1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c > vec3f(0.0031308)); }
@compute @workgroup_size(8, 8)
fn pipeTonemap(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(dst);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let c = max(textureLoad(src, gid.xy, 0).xyz, vec3f(0.0)) * p.exposure;
  textureStore(dst, gid.xy, vec4f(encode(aces(c)), 1.0));
}
`;

/**
 * OIDN input (the `oidn` denoiser): progressive mean of the raw beauty, albedo factor and normal while the camera stays still (reset on
 * motion or a cut), normal rotated to view space (OIDN's aux convention: x right, y up, z toward the camera; 0 where nothing was hit).
 */
const OIDN_PREP_WGSL = /* wgsl */ `
struct P { reset: u32, w: u32, h: u32, count: f32, right: vec4f, up: vec4f, back: vec4f }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var raw: texture_2d<f32>;
@group(0) @binding(2) var albedoIn: texture_2d<f32>;
@group(0) @binding(3) var normalIn: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> prog: array<vec4f>;
@group(0) @binding(5) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var outAlbedo: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var outNormal: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn pipeOidnPrep(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let q = gid.xy;
  let i = (gid.y * p.w + gid.x) * 3u;
  var sc = vec4f(textureLoad(raw, q, 0).xyz, 0.0);
  var sa = vec4f(textureLoad(albedoIn, q, 0).xyz, 0.0);
  var sn = vec4f(textureLoad(normalIn, q, 0).xyz, 0.0);
  if (p.reset == 0u) {
    sc += prog[i];
    sa += prog[i + 1u];
    sn += prog[i + 2u];
  }
  prog[i] = sc;
  prog[i + 1u] = sa;
  prog[i + 2u] = sn;
  let inv = 1.0 / p.count;
  let l = length(sn.xyz);
  var nv = vec3f(0.0);
  if (l > 1e-6) {
    let u = sn.xyz / l;
    nv = vec3f(dot(u, p.right.xyz), dot(u, p.up.xyz), dot(u, p.back.xyz));
  }
  textureStore(outColor, q, vec4f(max(sc.xyz * inv, vec3f(0.0)), 1.0));
  textureStore(outAlbedo, q, vec4f(clamp(sa.xyz * inv, vec3f(0.0), vec3f(1.0)), 1.0));
  textureStore(outNormal, q, vec4f(nv, 1.0));
}
`;

/** OIDN network size (the Denoiser package's quality): fast = `*_small`, balanced = the base network, high = `*_large` (clean aux only). */
export type OidnQuality = 'fast' | 'balanced' | 'high';

export interface OidnArmSettings {
  /** Guides: albedo + normal (`rt_hdr_calb_cnrm*`; our 1-spp guides are nearly noise-free) or albedo only (`rt_hdr_alb*`). */
  aux: 'albedo+normal' | 'albedo';
  /** Network while the camera moves and until `stillSamples` still frames have accumulated. Default fast (the small network, every frame). */
  moving: OidnQuality;
  /** Network once `stillSamples` frames of a still camera have accumulated. Default balanced (the standard network). */
  still: OidnQuality;
  /** Still frames accumulated before the `still` network runs (mutracer-style: a few samples are enough for the standard net). Default 8. */
  stillSamples: number;
  /** After the `still` network ran, hold the frame: the arm stops tracing and denoising until the camera moves. Default true. */
  hold: boolean;
  precision: 'fp16' | 'fp32';
}

export const DEFAULT_OIDN_ARM: OidnArmSettings = { aux: 'albedo+normal', moving: 'fast', still: 'balanced', stillSamples: 8, hold: true, precision: 'fp16' };

type OidnDenoiser = { denoiseTextures(o: unknown): Promise<GPUTexture | undefined> };
declare const __BENCH_DENOISER_TZAS__: string | null | undefined;
/** One Denoiser per (device, quality, aux, precision), shared by both arms: a network switch reloads weights; calls are awaited in turn. */
const oidnInstances = new Map<string, { device: GPUDevice; denoiser: Promise<OidnDenoiser> }>();
function oidnFor(device: GPUDevice, quality: OidnQuality, aux: OidnArmSettings['aux'], precision: OidnArmSettings['precision']): Promise<OidnDenoiser> {
  const prec = precision === 'fp16' && device.features.has('shader-f16') ? 'fp16' : 'fp32';
  const key = `${quality}|${aux}|${prec}`;
  const hit = oidnInstances.get(key);
  if (hit && hit.device === device) return hit.denoiser;
  const tzaUrl = typeof __BENCH_DENOISER_TZAS__ === 'string' ? __BENCH_DENOISER_TZAS__ : undefined;
  if (!tzaUrl) throw new Error('OIDN: no weights (link the Denoiser checkout as the bench dev dependency `denoiser`)');
  const denoiser = import('denoiser').then(({ Denoiser }) => (Denoiser as unknown as { create(o: unknown): Promise<OidnDenoiser> }).create({ device, precision: prec, quality, tzaUrl }));
  oidnInstances.set(key, { device, denoiser });
  return denoiser;
}

// ---------------------------------------------------------------- arm

export interface PipelineSettings {
  /** Render (input) film. */
  width: number;
  height: number;
  /** Display (output) size; 2x the render film by default (960x540 -> 1920x1080). */
  displayWidth: number;
  displayHeight: number;
  /** SVGF options (`DiffuseSpecularDenoiser`); `remodulate` is always on. Default: the Denoiser's defaults (SVGF, anti-lag auto). */
  denoiser?: Record<string, unknown>;
  /** Which denoiser the arm runs (switchable with `setDenoiser`): 'svgf' (default), 'oidn', or 'none' (raw to the upscaler). */
  denoise?: DenoiserKind;
  oidn?: Partial<OidnArmSettings>;
  /** Upscaler runtime settings overrides (`Upscaler.settings`). */
  upscaler?: Record<string, unknown>;
  /**
   * Reactive mask from the SVGF history length (P4.6; `svgf` only; default off: the track N chain is unchanged). `cap`: frames of history after which a
   * pixel is no longer reactive (default 8); `strength`: mask value of a fresh pixel (default 1). Reads the denoiser's `accum` debug texture (internal API).
   */
  reactive?: { cap?: number; strength?: number } | false;
}

const align = (n: number, a: number) => Math.ceil(n / a) * a;

export type PipelineStage = 'raw' | 'denoised' | 'final';
export type DenoiserKind = 'svgf' | 'oidn' | 'none';

export interface PipelineFrameBuffers {
  accum: GPUBuffer;
  planes: GPUBuffer;
  /** Byte offset of plane 0 in `planes` (the wavefront keeps the planes in the accumulator's tail, P4.1b; the megakernel's own buffer starts at 0). Must be a valid storage binding offset (256-byte aligned). */
  planesOffset?: number;
}

export class PipelineArm {
  readonly device!: GPUDevice;
  readonly s!: PipelineSettings;
  readonly denoiser!: DiffuseSpecularDenoiser;
  readonly upscaler!: Upscaler;
  private readonly host!: UpscalerHost;
  private readonly upCamera = new UpscalerCamera();
  private readonly sums!: GPUBuffer;
  private readonly packBuf!: GPUBuffer;
  private readonly packParams!: GPUBuffer;
  private readonly accumParams!: GPUBuffer;
  private readonly accumPipe!: GPUComputePipeline;
  private readonly packPipe!: GPUComputePipeline;
  private readonly compositePipe!: GPUComputePipeline;
  private readonly depthPipe!: GPURenderPipeline;
  private readonly depthGroup!: GPUBindGroup;
  private readonly flickerPipe!: GPUComputePipeline;
  private packGroup: GPUBindGroup | null = null;
  private accumGroup: GPUBindGroup | null = null;
  private accumKey: GPUBuffer | null = null;
  private accumOffsetKey = '';
  private readonly stride8!: number;
  private readonly stride4!: number;
  private readonly base4!: number;
  /** Render-res input textures (rgba16float sections 0..7, r32float 8..12). */
  readonly tex!: Record<'raw' | 'diffuse' | 'specular' | 'albedo' | 'specF' | 'normal' | 'motion' | 'emission' | 'viewZ' | 'motionZ' | 'roughness' | 'hitD' | 'hitS', GPUTexture>;
  readonly depth!: GPUTexture;
  /** Denoised + composited colour (render res): the upscaler's input. */
  readonly color!: GPUTexture;
  private readonly keys!: { color: object; depth: object; velocity: object; reactive?: object };
  /** Reactive mask (rgba8unorm, `.r`, render res; only with `settings.reactive`) and the display-referred output (rgba8unorm sRGB, display res). */
  readonly reactiveMask?: GPUTexture;
  readonly display!: GPUTexture;
  private readonly reactivePipe?: GPUComputePipeline;
  private readonly reactiveParams?: GPUBuffer;
  private readonly tonemapPipe!: GPUComputePipeline;
  private readonly tonemapParams!: GPUBuffer;
  private tonemapGroup: GPUBindGroup | null = null;
  private prevView: Float32Array | null = null;
  private prevProj: Float32Array | null = null;
  private resetNext = true;
  private flicker: { state: GPUBuffer; params: GPUBuffer; group: GPUBindGroup; w: number; h: number; frames: number }[] = [];
  /** The denoiser this arm runs, and the OIDN settings (`setDenoiser`). */
  kind: DenoiserKind = 'svgf';
  oidn: OidnArmSettings = { ...DEFAULT_OIDN_ARM };
  private readonly oidnPrepPipe!: GPUComputePipeline;
  private readonly oidnParams!: GPUBuffer;
  private readonly oidnProg!: GPUBuffer;
  private readonly oidnTex!: { color: GPUTexture; albedo: GPUTexture; normal: GPUTexture };
  private oidnGroup: GPUBindGroup | null = null;
  /** OIDN: frames in the progressive mean, the camera they share, and whether the still frame is held. */
  private oidnCount = 0;
  private oidnCamera = '';
  private oidnHeld = false;
  /** With `settings.reactive`: whether the mask is passed to the upscaler (runtime A/B switch; the mask pass itself always runs). */
  reactiveEnabled = true;
  /** What the last `post` ran (the OIDN network and spp, or the denoiser kind). */
  lastDenoiser = '';

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  static async create(device: GPUDevice, s: PipelineSettings, label: string): Promise<PipelineArm> {
    const { width: w, height: h } = s;
    const n = w * h;
    const sums = device.createBuffer({ label: `${label} sums`, size: n * SUM_VEC4 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const stride8 = align(w * 8, 256) / 8;
    const stride4 = align(w * 4, 256) / 4;
    const base4 = 8 * h * stride8 * 2;
    const packWords = base4 + 5 * h * stride4;
    const packBuf = device.createBuffer({ label: `${label} pack`, size: packWords * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const packParams = device.createBuffer({ label: `${label} pack params`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const accumParams = device.createBuffer({ label: `${label} accum params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(accumParams, 0, new Uint32Array([n, 0, 0, 0]));
    const mk = (name: string, format: GPUTextureFormat, usage = TEX | COPY_DST_T | COPY_SRC_T, size = { width: w, height: h }) => device.createTexture({ label: `${label} ${name}`, size, format, usage });
    const tex = {
      raw: mk('raw', 'rgba16float'),
      diffuse: mk('diffuse', 'rgba16float'),
      specular: mk('specular', 'rgba16float'),
      albedo: mk('albedo', 'rgba16float'),
      specF: mk('specF', 'rgba16float'),
      normal: mk('normal', 'rgba16float'),
      motion: mk('motion', 'rgba16float'),
      emission: mk('emission', 'rgba16float'),
      viewZ: mk('viewZ', 'r32float'),
      motionZ: mk('motionZ', 'r32float'),
      roughness: mk('roughness', 'r32float'),
      hitD: mk('hitD', 'r32float'),
      hitS: mk('hitS', 'r32float'),
    };
    const depth = mk('depth', 'depth32float', TEX | GPUTextureUsage.RENDER_ATTACHMENT);
    const color = mk('color', 'rgba16float', TEX | GPUTextureUsage.STORAGE_BINDING | COPY_SRC_T | COPY_DST_T);
    const module = (code: string, name: string) => device.createShaderModule({ label: `${label} ${name}`, code });
    const oidnTex = { color: mk('oidn color', 'rgba16float', TEX | GPUTextureUsage.STORAGE_BINDING), albedo: mk('oidn albedo', 'rgba16float', TEX | GPUTextureUsage.STORAGE_BINDING), normal: mk('oidn normal', 'rgba16float', TEX | GPUTextureUsage.STORAGE_BINDING) };
    const oidnProg = device.createBuffer({ label: `${label} oidn progressive`, size: n * 3 * 16, usage: GPUBufferUsage.STORAGE });
    const oidnParams = device.createBuffer({ label: `${label} oidn params`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const [tonemapPipe, reactivePipe, accumPipe, packPipe, compositePipe, flickerPipe, oidnPrepPipe, depthPipe] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} tonemap`, layout: 'auto', compute: { module: module(TONEMAP_WGSL, 'tonemap'), entryPoint: 'pipeTonemap' } }),
      s.reactive ? device.createComputePipelineAsync({ label: `${label} reactive`, layout: 'auto', compute: { module: module(REACTIVE_WGSL, 'reactive'), entryPoint: 'pipeReactive' } }) : Promise.resolve(undefined),
      device.createComputePipelineAsync({ label: `${label} accumulate`, layout: 'auto', compute: { module: module(ACCUM_WGSL, 'accumulate'), entryPoint: 'pipeAccumulate' } }),
      device.createComputePipelineAsync({ label: `${label} pack`, layout: 'auto', compute: { module: module(PACK_WGSL, 'pack'), entryPoint: 'pipePack' } }),
      device.createComputePipelineAsync({ label: `${label} composite`, layout: 'auto', compute: { module: module(COMPOSITE_WGSL, 'composite'), entryPoint: 'pipeComposite' } }),
      device.createComputePipelineAsync({ label: `${label} flicker`, layout: 'auto', compute: { module: module(FLICKER_WGSL, 'flicker'), entryPoint: 'pipeFlicker' } }),
      device.createComputePipelineAsync({ label: `${label} oidn prep`, layout: 'auto', compute: { module: module(OIDN_PREP_WGSL, 'oidn prep'), entryPoint: 'pipeOidnPrep' } }),
      device.createRenderPipelineAsync({
        label: `${label} depth`,
        layout: 'auto',
        vertex: { module: module(DEPTH_WGSL, 'depth'), entryPoint: 'vs' },
        fragment: { module: module(DEPTH_WGSL, 'depth'), entryPoint: 'fs', targets: [] },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'always' },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
    const depthParams = device.createBuffer({ label: `${label} depth params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(depthParams, 0, new Float32Array([PIPE_NEAR, PIPE_FAR, 0, 0]));
    const depthGroup = device.createBindGroup({ layout: depthPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: depthParams } }, { binding: 1, resource: tex.viewZ.createView() }] });

    const denoiser = new DiffuseSpecularDenoiser(device, { ...(s.denoiser ?? {}), remodulate: true } as never);
    denoiser.configure({ width: w, height: h });
    const tonemapParams = device.createBuffer({ label: `${label} tonemap params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(tonemapParams, 0, new Float32Array([1, 0, 0, 0]));
    const display = device.createTexture({ label: `${label} display`, size: { width: s.displayWidth, height: s.displayHeight }, format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | TEX | COPY_SRC_T });
    const reactiveMask = s.reactive ? mk('reactive', 'rgba8unorm', TEX | GPUTextureUsage.STORAGE_BINDING) : undefined;
    const reactiveParams = s.reactive ? device.createBuffer({ label: `${label} reactive params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }) : undefined;
    if (reactiveParams && s.reactive) device.queue.writeBuffer(reactiveParams, 0, new Float32Array([s.reactive.cap ?? 8, s.reactive.strength ?? 1, 0, 0]));
    const host = new UpscalerHost(device);
    const upscaler = new Upscaler({ renderer: host as never });
    upscaler.configure({ displayWidth: s.displayWidth, displayHeight: s.displayHeight, renderWidth: w, renderHeight: h, path: 'temporal', jitter: false });
    await upscaler.init();
    Object.assign(upscaler.settings, s.upscaler ?? {});
    const keys = { color: host.texture(color, `${label} color`), depth: host.texture(depth, `${label} depth`), velocity: host.texture(tex.motion, `${label} motion`), ...(reactiveMask ? { reactive: host.texture(reactiveMask, `${label} reactive`) } : {}) };
    // The accumulate and pack groups need the kernel's planes buffer: built on the first `accumulate`.
    const arm = new PipelineArm({ device, s, denoiser, upscaler, host, sums, packBuf, packParams, accumParams, accumPipe, packPipe, compositePipe, depthPipe, depthGroup, flickerPipe, stride8, stride4, base4, tex, depth, color, keys, oidnPrepPipe, oidnParams, oidnProg, oidnTex, display, reactiveMask, reactivePipe, reactiveParams, tonemapPipe, tonemapParams });
    arm.setDenoiser(s.denoise ?? 'svgf', s.oidn ?? {});
    return arm;
  }

  /** Switches the denoiser (compare arms with the same one). Drops the progressive OIDN mean, the held frame and the histories. */
  setDenoiser(kind: DenoiserKind, oidn: Partial<OidnArmSettings> = {}): void {
    this.kind = kind;
    this.oidn = { ...DEFAULT_OIDN_ARM, ...oidn };
    this.oidnCount = 0;
    this.oidnCamera = '';
    this.oidnHeld = false;
    this.resetNext = true;
    if (kind === 'oidn') {
      // Load the networks now, so the first frames do not wait for weights.
      void oidnFor(this.device, this.oidn.moving, this.oidn.aux, this.oidn.precision);
      void oidnFor(this.device, this.oidn.still, this.oidn.aux, this.oidn.precision);
    }
  }

  /** Loads the OIDN networks of the current settings (await before timing). */
  async ready(): Promise<void> {
    if (this.kind !== 'oidn') return;
    await Promise.all([oidnFor(this.device, this.oidn.moving, this.oidn.aux, this.oidn.precision), oidnFor(this.device, this.oidn.still, this.oidn.aux, this.oidn.precision)]);
  }

  /** Timing helper: the next `denoiseOidn(camera, false)` runs the `still` network (as if `stillSamples` - 1 frames had accumulated). */
  primeStill(camera: PtLookAtCamera): void {
    this.oidnCamera = cameraKey(camera);
    this.oidnCount = Math.max(0, this.oidn.stillSamples - 1);
  }

  /** OIDN with `hold`: true while this camera's still frame is held (the session then skips tracing and post for the arm). */
  heldFor(camera: PtLookAtCamera): boolean {
    return this.kind === 'oidn' && this.oidnHeld && !this.resetNext && cameraKey(camera) === this.oidnCamera;
  }

  /** Start of a frame: clears the sample sums. */
  beginFrame(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.sums);
  }

  /** After each path-tracing encode of the frame (`pt.frameBuffers`): adds the sample to the sums. */
  accumulate(encoder: GPUCommandEncoder, fb: PipelineFrameBuffers): void {
    const key = `${fb.planesOffset ?? 0}`;
    if (this.accumKey !== fb.planes || this.accumOffsetKey !== key) {
      this.accumKey = fb.planes;
      this.accumOffsetKey = key;
      const planes: GPUBufferBinding = { buffer: fb.planes, offset: fb.planesOffset ?? 0 };
      const whole = (buffer: GPUBuffer): GPUBufferBinding => ({ buffer });
      this.accumGroup = this.device.createBindGroup({ layout: this.accumPipe.getBindGroupLayout(0), entries: [whole(this.accumParams), whole(fb.accum), planes, whole(this.sums)].map((resource, binding) => ({ binding, resource })) });
      this.packGroup = this.device.createBindGroup({ layout: this.packPipe.getBindGroupLayout(0), entries: [whole(this.packParams), whole(this.sums), planes, whole(this.packBuf)].map((resource, binding) => ({ binding, resource })) });
    }
    const pass = encoder.beginComputePass({ label: 'pipe accumulate' });
    pass.setPipeline(this.accumPipe);
    pass.setBindGroup(0, this.accumGroup as GPUBindGroup);
    pass.dispatchWorkgroups(Math.ceil((this.s.width * this.s.height) / 64));
    pass.end();
  }

  /** End of the path-tracing part: sums / spp -> the input textures, and the depth guide. */
  pack(encoder: GPUCommandEncoder, spp: number): void {
    const { width: w, height: h } = this.s;
    const p = new ArrayBuffer(32);
    new Uint32Array(p, 0, 2).set([w, h]);
    new Float32Array(p, 8, 1).set([spp]);
    new Uint32Array(p, 12, 3).set([this.stride8, this.stride4, this.base4]);
    this.device.queue.writeBuffer(this.packParams, 0, p);
    const pass = encoder.beginComputePass({ label: 'pipe pack' });
    pass.setPipeline(this.packPipe);
    if (!this.packGroup) throw new Error('PipelineArm.pack: call accumulate() first');
    pass.setBindGroup(0, this.packGroup);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    pass.end();
    const t = this.tex;
    const half = [t.raw, t.diffuse, t.specular, t.albedo, t.specF, t.normal, t.motion, t.emission];
    half.forEach((texture, k) => encoder.copyBufferToTexture({ buffer: this.packBuf, offset: k * h * this.stride8 * 8, bytesPerRow: this.stride8 * 8, rowsPerImage: h }, { texture }, { width: w, height: h }));
    [t.viewZ, t.motionZ, t.roughness, t.hitD, t.hitS].forEach((texture, k) => encoder.copyBufferToTexture({ buffer: this.packBuf, offset: (this.base4 + k * h * this.stride4) * 4, bytesPerRow: this.stride4 * 4, rowsPerImage: h }, { texture }, { width: w, height: h }));
    const dp = encoder.beginRenderPass({ label: 'pipe depth', colorAttachments: [], depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    dp.setPipeline(this.depthPipe);
    dp.setBindGroup(0, this.depthGroup);
    dp.draw(3);
    dp.end();
  }

  /** Drop the denoiser's and the upscaler's history on the next `post` (camera cut, new image). */
  resetHistory(): void {
    this.resetNext = true;
  }

  /**
   * Denoise, composite, upscale. Call after the frame's path tracing + `pack` were submitted. `camera` = this frame's (unjittered) camera;
   * the previous one is remembered. Records and submits its own work (the denoiser and the upscaler submit theirs).
   */
  async post(camera: PtLookAtCamera, options: { denoise?: boolean; deltaTime?: number; tonemap?: boolean } = {}): Promise<void> {
    const { width: w, height: h } = this.s;
    const aspect = w / h;
    const view = viewMatrix(camera);
    const proj = projectionMatrix(camera.yFovDeg, aspect);
    const reset = this.resetNext;
    this.resetNext = false;
    if (reset) {
      this.denoiser.resetHistory();
      this.prevView = view;
      this.prevProj = proj;
    }
    const t = this.tex;
    const kind: DenoiserKind = options.denoise === false ? 'none' : this.kind;
    if (kind === 'svgf') {
      const dOut = this.denoiser.dispatchBoth(
        { diffuse: { radiance: t.diffuse, hitDistance: t.hitD }, specular: { radiance: t.specular, hitDistance: t.hitS, demodulation: t.specF } } as never,
        { depth: this.depth, normal: t.normal, motion: t.motion, roughness: t.roughness, albedo: t.albedo, viewZ: t.viewZ, motionZ: t.motionZ },
        { view, projection: proj, prevView: this.prevView as Float32Array, prevProjection: this.prevProj as Float32Array, near: PIPE_NEAR, far: PIPE_FAR, jitter: [0, 0], position: [...camera.position] as [number, number, number] },
      ) as unknown as { diffuse: GPUTexture; specular: GPUTexture };
      const encoder = this.device.createCommandEncoder({ label: 'pipe composite' });
      const group = this.device.createBindGroup({ layout: this.compositePipe.getBindGroupLayout(0), entries: [dOut.diffuse.createView(), dOut.specular.createView(), t.raw.createView(), t.specF.createView(), this.color.createView(), t.emission.createView()].map((resource, binding) => ({ binding, resource })) });
      const pass = encoder.beginComputePass({ label: 'pipe composite' });
      pass.setPipeline(this.compositePipe);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      pass.end();
      if (this.reactivePipe && this.reactiveMask && this.reactiveParams) {
        // History length (alpha of the diffuse signal's accumulation texture) -> the upscaler's reactive mask.
        const accum = (this.denoiser as unknown as { pipes: { debugTexture(n: string): GPUTexture | undefined }[] }).pipes[0]?.debugTexture('accum');
        if (!accum) throw new Error('reactive mask: the denoiser has no accum debug texture');
        const rg = this.device.createBindGroup({ layout: this.reactivePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.reactiveParams } }, { binding: 1, resource: accum.createView() }, { binding: 2, resource: this.reactiveMask.createView() }] });
        const rp = encoder.beginComputePass({ label: 'pipe reactive' });
        rp.setPipeline(this.reactivePipe);
        rp.setBindGroup(0, rg);
        rp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
        rp.end();
      }
      this.device.queue.submit([encoder.finish()]);
      this.lastDenoiser = 'svgf';
    } else if (kind === 'oidn') {
      await this.denoiseOidn(camera, reset);
    } else {
      const encoder = this.device.createCommandEncoder({ label: 'pipe raw' });
      encoder.copyTextureToTexture({ texture: t.raw }, { texture: this.color }, { width: w, height: h });
      this.device.queue.submit([encoder.finish()]);
      this.lastDenoiser = 'none';
    }
    this.upscale(camera, reset, options.deltaTime ?? 1 / 60, kind === 'svgf' && this.reactiveEnabled);
    if (options.tonemap) this.tonemap();
    this.prevView = view;
    this.prevProj = proj;
  }

  /** The upscaler on `color` (render res -> display res). `reactive`: pass the reactive mask (only when `settings.reactive` and the SVGF branch wrote it this frame). */
  upscale(camera: PtLookAtCamera, reset: boolean, deltaTime = 1 / 60, reactive = false): void {
    this.upCamera.set(camera, this.s.width / this.s.height);
    this.upscaler.beginFrame(this.upCamera as never);
    this.upscaler.endFrame(this.upCamera as never);
    this.upscaler.dispatch({ color: this.keys.color, depth: this.keys.depth, velocity: this.keys.velocity, ...(reactive && this.keys.reactive ? { reactive: this.keys.reactive } : {}), reset, deltaTime } as never, this.upCamera as never);
  }

  /**
   * The display stage: exposure, ACES, sRGB -> `display` (rgba8unorm, display res) from the upscaler's output. `exposure` is a number, or `'auto'`:
   * the geometric-mean luminance of the `final` stage is reduced on the GPU (`@pmndrs/downsampler` `ScalarReduction`, two dispatches, no readback)
   * straight into the tonemap parameters, exposure = `AUTO_EXPOSURE.key` / mean, clamped (chain-math.ts has the CPU twin). The result is not
   * smoothed over time: a caller that wants adaptation blends it itself.
   */
  tonemap(exposure: number | 'auto' = 1): void {
    if (exposure !== 'auto') this.device.queue.writeBuffer(this.tonemapParams, 0, new Float32Array([exposure, 0, 0, 0]));
    this.tonemapGroup ??= this.device.createBindGroup({ layout: this.tonemapPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.tonemapParams } }, { binding: 1, resource: this.stage('final').createView() }, { binding: 2, resource: this.display.createView() }] });
    const encoder = this.device.createCommandEncoder({ label: 'pipe tonemap' });
    if (exposure === 'auto') {
      const auto = this.autoExposure();
      auto.reduction.encode(encoder, { width: this.s.displayWidth, height: this.s.displayHeight, bindGroup: auto.group });
    }
    const pass = encoder.beginComputePass({ label: 'pipe tonemap' });
    pass.setPipeline(this.tonemapPipe);
    pass.setBindGroup(0, this.tonemapGroup);
    pass.dispatchWorkgroups(Math.ceil(this.s.displayWidth / 8), Math.ceil(this.s.displayHeight / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private autoExposureState: { reduction: ScalarReduction; group: GPUBindGroup } | null = null;

  /** The auto-exposure reduction (made on first use; reads the `final` stage, writes `tonemapParams[0]`). */
  private autoExposure(): { reduction: ScalarReduction; group: GPUBindGroup } {
    if (this.autoExposureState) return this.autoExposureState;
    const { key, min, max, lumaFloor } = AUTO_EXPOSURE;
    const reduction = new ScalarReduction(this.device, {
      bindings: {
        wgsl: '@group(1) @binding(0) var colour: texture_2d<f32>;\n@group(1) @binding(1) var<storage, read_write> exposureOut: array<f32>;',
        entries: [
          { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
          { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
      },
      load: `fn load(texel: vec2i, layer: u32) -> vec4f {
  let c = max(textureLoad(colour, texel, 0).rgb, vec3f(0.0));
  return vec4f(log(max(dot(c, vec3f(0.2126, 0.7152, 0.0722)), ${lumaFloor.toExponential()})), 0.0, 0.0, 0.0);
}`,
      finish: `fn finish(value: vec4f, weight: f32) {
  exposureOut[0] = clamp(${key} / exp(value.x), ${min}, ${max});
}`,
    });
    const layout = reduction.bindGroupLayouts[1] as GPUBindGroupLayout;
    const group = this.device.createBindGroup({ layout, entries: [{ binding: 0, resource: this.stage('final').createView() }, { binding: 1, resource: { buffer: this.tonemapParams, size: 4 } }] });
    return (this.autoExposureState = { reduction, group });
  }

  /** The exposure the last `tonemap('auto')` (or `tonemap(x)`) used (a readback: not for a frame loop). */
  async readExposure(): Promise<number> {
    return new Float32Array(await readBuffer(this.device, this.tonemapParams, 16))[0] as number;
  }

  /**
   * OIDN on this frame: the progressive mean (restarted when the camera changed or after a cut), the `moving` network until `stillSamples`
   * frames have accumulated, then the `still` network (and the frame is held with `hold`). Awaits the Denoiser package (it waits for its GPU
   * work; no readback). Writes `color`.
   */
  async denoiseOidn(camera: PtLookAtCamera, cut: boolean): Promise<void> {
    const { width: w, height: h } = this.s;
    const key = cameraKey(camera);
    const moved = cut || key !== this.oidnCamera;
    this.oidnCamera = key;
    this.oidnCount = moved ? 1 : this.oidnCount + 1;
    this.oidnHeld = false;
    const o = this.oidn;
    const f = norm(sub(camera.target, camera.position));
    const r = norm(cross(f, camera.up));
    const u = cross(r, f);
    const p = new ArrayBuffer(64);
    new Uint32Array(p, 0, 3).set([moved ? 1 : 0, w, h]);
    new Float32Array(p, 12, 1).set([this.oidnCount]);
    new Float32Array(p, 16, 12).set([r[0], r[1], r[2], 0, u[0], u[1], u[2], 0, -f[0], -f[1], -f[2], 0]);
    this.device.queue.writeBuffer(this.oidnParams, 0, p);
    this.oidnGroup ??= this.device.createBindGroup({
      layout: this.oidnPrepPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.oidnParams } },
        { binding: 1, resource: this.tex.raw.createView() },
        { binding: 2, resource: this.tex.albedo.createView() },
        { binding: 3, resource: this.tex.normal.createView() },
        { binding: 4, resource: { buffer: this.oidnProg } },
        { binding: 5, resource: this.oidnTex.color.createView() },
        { binding: 6, resource: this.oidnTex.albedo.createView() },
        { binding: 7, resource: this.oidnTex.normal.createView() },
      ],
    });
    const encoder = this.device.createCommandEncoder({ label: 'pipe oidn prep' });
    const pass = encoder.beginComputePass({ label: 'pipe oidn prep' });
    pass.setPipeline(this.oidnPrepPipe);
    pass.setBindGroup(0, this.oidnGroup);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    const still = this.oidnCount >= o.stillSamples;
    const quality = still ? o.still : o.moving;
    const net = await oidnFor(this.device, quality, o.aux, o.precision);
    const out = await net.denoiseTextures({ color: this.oidnTex.color, albedo: this.oidnTex.albedo, ...(o.aux === 'albedo+normal' ? { normal: this.oidnTex.normal } : {}), hdr: true, output: this.color, transfer: 'linear' });
    if (!out) throw new Error('OIDN: the run was aborted');
    this.lastDenoiser = `oidn ${quality} ${o.aux} (${this.oidnCount} spp)`;
    if (still && o.hold) this.oidnHeld = true;
  }

  /** The texture of a stage: raw beauty and denoised colour at render resolution, the upscaled output at display resolution. */
  stage(s: PipelineStage): GPUTexture {
    if (s === 'raw') return this.tex.raw;
    if (s === 'denoised') return this.color;
    return this.host.gpu(this.upscaler.outputTexture as unknown as object);
  }

  // ---- stability meter (headless twin): per-pixel sum of |L_t - L_(t-1)| per stage, read once at the end.

  /** Starts measuring frame-to-frame change of the three stages (clears the meters). */
  startFlicker(): void {
    for (const f of this.flicker) {
      f.state.destroy();
      f.params.destroy();
    }
    this.flicker = (['raw', 'denoised', 'final'] as const).map((st) => {
      const tex = this.stage(st);
      const fw = tex.width;
      const fh = tex.height;
      const state = this.device.createBuffer({ label: `flicker ${st}`, size: fw * fh * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const params = this.device.createBuffer({ label: `flicker ${st} params`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const group = this.device.createBindGroup({ layout: this.flickerPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: params } }, { binding: 1, resource: tex.createView() }, { binding: 2, resource: { buffer: state } }] });
      return { state, params, group, w: fw, h: fh, frames: 0 };
    });
  }

  /** One meter step per stage (after `post`); the first call only records the frame. */
  stepFlicker(): void {
    const encoder = this.device.createCommandEncoder({ label: 'pipe flicker' });
    for (const f of this.flicker) {
      this.device.queue.writeBuffer(f.params, 0, new Uint32Array([f.w, f.h, f.frames > 0 ? 1 : 0, 0]));
      const pass = encoder.beginComputePass({ label: 'pipe flicker' });
      pass.setPipeline(this.flickerPipe);
      pass.setBindGroup(0, f.group);
      pass.dispatchWorkgroups(Math.ceil(f.w / 8), Math.ceil(f.h / 8));
      pass.end();
      f.frames++;
    }
    this.device.queue.submit([encoder.finish()]);
  }

  /** Per stage: mean |change| per frame per pixel (luminance), as a w x h map (read back; not for a frame loop). */
  async readFlicker(): Promise<Record<PipelineStage, { width: number; height: number; changes: number; map: Float32Array }>> {
    const out: Record<string, { width: number; height: number; changes: number; map: Float32Array }> = {};
    const names: PipelineStage[] = ['raw', 'denoised', 'final'];
    for (let k = 0; k < this.flicker.length; k++) {
      const f = this.flicker[k]!;
      const data = new Float32Array(await readBuffer(this.device, f.state, f.w * f.h * 8));
      const changes = Math.max(1, f.frames - 1);
      const map = new Float32Array(f.w * f.h);
      for (let i = 0; i < map.length; i++) map[i] = (data[i * 2 + 1] as number) / changes;
      out[names[k]!] = { width: f.w, height: f.h, changes: f.frames - 1, map };
    }
    return out as Record<PipelineStage, { width: number; height: number; changes: number; map: Float32Array }>;
  }

  destroy(): void {
    this.denoiser.dispose();
    this.upscaler.dispose();
    for (const b of [this.sums, this.packBuf, this.packParams, this.accumParams, this.oidnParams, this.oidnProg, this.tonemapParams, ...(this.reactiveParams ? [this.reactiveParams] : [])]) b.destroy();
    for (const t of [...Object.values(this.tex), ...Object.values(this.oidnTex), this.depth, this.color, this.display, ...(this.reactiveMask ? [this.reactiveMask] : [])]) t.destroy();
    for (const f of this.flicker) {
      f.state.destroy();
      f.params.destroy();
    }
  }
}

// ---------------------------------------------------------------- readback helpers (never in a frame loop)

export async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** An rgba16float texture as linear RGBA f32 (row 0 = top). */
export async function readTextureRgba16f(device: GPUDevice, texture: GPUTexture): Promise<{ width: number; height: number; channels: 4; data: Float32Array }> {
  const { width: w, height: h } = texture;
  const stride = align(w * 8, 256);
  const staging = device.createBuffer({ size: stride * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: stride, rowsPerImage: h }, { width: w, height: h });
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const raw = new Uint16Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  const data = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w * 4; x++) data[y * w * 4 + x] = halfToFloat(raw[(y * stride) / 2 + x] as number);
  return { width: w, height: h, channels: 4, data };
}
