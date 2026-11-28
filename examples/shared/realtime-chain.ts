/**
 * The real-time chain behind a path tracer that renders one sample per pixel per frame:
 *
 *   PtMegakernel { mode: 'realtime' } (render size)  ->  pack: frame planes -> textures (and a depth guide from viewZ)
 *     ->  denoiser: SVGF (`denoiser/realtime` `DiffuseSpecularDenoiser`, demodulated diffuse + specular, remodulated) + composite (+ emission plane, sky)
 *                   or OIDN (`denoiser` `Denoiser`, a neural network on the raw frame with albedo and normal guides)
 *     ->  `@pmndrs/upscaler` temporal path (render size -> display size)
 *
 * A trimmed copy of the bench's `bench/realtime/chain.ts` (the page that measured it: `realtime-pipeline`, docs/internal/design/realtime.md
 * section 5), without its meters and A/B knobs. One jitter owner: the path tracer's sampler (pair 0); the denoiser gets jitter [0, 0] and
 * the upscaler runs with `jitter: false`. Motion is jitter-free by construction (the planes project world points through both cameras).
 *
 * Per frame:  chain.beginFrame(encoder); pt.encode(encoder, 1); chain.accumulate(encoder, pt.frameBuffers); chain.pack(encoder);
 *             submit; await chain.post(camera)  (the denoiser and the upscaler record and submit their own command buffers; no readback).
 * `post` returns nothing for SVGF and a promise for OIDN, which waits for its network. With OIDN and a still camera, `heldFor(camera)` says
 * the finished still frame is being held: skip tracing and `post` until the camera moves.
 *
 * OIDN's weights load from `denoiser`'s default URL (jsDelivr, pmndrs/denoiser-weights) the first time OIDN runs; `tzaUrl` overrides it.
 */
import { DiffuseSpecularDenoiser } from 'denoiser/realtime';
import type { Denoiser } from 'denoiser';
import { UpscalerCore } from '@pmndrs/upscaler/core';
import type { PtLookAtCamera } from '@pmndrs/pathtracer';

const TEX = GPUTextureUsage.TEXTURE_BINDING;
const COPY_DST_T = GPUTextureUsage.COPY_DST;
const COPY_SRC_T = GPUTextureUsage.COPY_SRC;

// ---------------------------------------------------------------- camera matrices (three.js conventions: column-major, view down -Z, depth [0, 1])

type V3 = [number, number, number];
const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
const norm = (a: readonly number[]): V3 => {
  const l = Math.hypot(a[0]!, a[1]!, a[2]!);
  return [a[0]! / l, a[1]! / l, a[2]! / l];
};

/** Exact identity of a camera (a moved camera restarts OIDN's progressive mean). */
function cameraKey(c: PtLookAtCamera): string {
  return [...c.position, ...c.target, ...c.up, c.yFovDeg].join(',');
}

/** World -> view, the same basis as the path tracer's camera (right = forward x up). */
export function viewMatrix(c: PtLookAtCamera): Float32Array {
  const f = norm(sub(c.target, c.position));
  const r = norm(cross(f, c.up));
  const u = cross(r, f);
  const p = c.position;
  return new Float32Array([r[0], u[0], -f[0], 0, r[1], u[1], -f[1], 0, r[2], u[2], -f[2], 0, -dot(r, p), -dot(u, p), dot(f, p), 1]);
}

/** three's `makePerspective` for WebGPU (depth 0 at near, 1 at far). */
export function projectionMatrix(yFovDeg: number, aspect: number, near: number, far: number): Float32Array {
  const t = 1 / Math.tan((yFovDeg * Math.PI) / 360);
  const m = new Float32Array(16);
  m[0] = t / aspect;
  m[5] = t;
  m[10] = -far / (far - near);
  m[11] = -1;
  m[14] = (-far * near) / (far - near);
  return m;
}

// ---------------------------------------------------------------- WGSL

/** Per-pixel sums over the frame's samples (6 vec4): beauty, diffuse-class radiance, specular-class radiance, albedo factor, specular factor, emission. */
const SUM_VEC4 = 6;

const ACCUM_WGSL = /* wgsl */ `
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;
@group(0) @binding(2) var<storage, read> planes: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> sums: array<vec4f>;

@compute @workgroup_size(64)
fn chainAccumulate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let n = p.n;
  if (i >= n) { return; }
  let aF = planes[4u * n + i];
  let sF = planes[5u * n + i];
  sums[i] += vec4f(accum[i].xyz, 1.0);
  sums[n + i] += vec4f(planes[i].xyz * aF.xyz, 1.0);
  sums[2u * n + i] += vec4f(planes[n + i].xyz * sF.xyz, 0.0);
  sums[3u * n + i] += aF;
  sums[4u * n + i] += sF;
  sums[5u * n + i] += vec4f(planes[6u * n + i].xyz, 0.0);
}
`;

/**
 * Pack: sums -> one buffer of texture rows (copied into the textures after; rg16float and r16float are not storage formats in core WebGPU).
 * Sections of 256-byte aligned rows: 0..7 rgba16float (beauty, diffuse demod, specular demod, albedo factor, specular factor, normal, motion,
 * emission), then 8..12 r32float (viewZ, motionZ, roughness, diffuse hit distance, specular hit distance).
 */
const PACK_WGSL = /* wgsl */ `
struct P { w: u32, h: u32, stride8: u32, stride4: u32, base4: u32, _a: u32, _b: u32, _c: u32 }
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
fn chainPack(@builtin(global_invocation_id) gid: vec3u) {
  let x = gid.x;
  let y = gid.y;
  if (x >= p.w || y >= p.h) { return; }
  let n = p.w * p.h;
  let i = y * p.w + x;
  let spp = max(1.0, sums[n + i].w);
  let inv = 1.0 / spp;
  let aF = max(sums[3u * n + i].xyz, vec3f(1e-6));
  let sF = max(sums[4u * n + i].xyz, vec3f(1e-6));
  let nz = planes[2u * n + i];
  let mo = planes[3u * n + i];
  put16(0u, x, y, clampHalf(vec4f(sums[i].xyz * inv, 1.0)));
  put16(1u, x, y, clampHalf(vec4f(sums[n + i].xyz / aF, 1.0)));   // sum of remodulated radiance / sum of factors
  put16(2u, x, y, clampHalf(vec4f(sums[2u * n + i].xyz / sF, 1.0)));
  put16(3u, x, y, vec4f(aF * inv, sums[3u * n + i].w * inv));
  put16(4u, x, y, vec4f(sF * inv, sums[4u * n + i].w * inv));      // w: share of samples whose primary ray hit
  put16(5u, x, y, vec4f(nz.xyz, 0.0));
  put16(6u, x, y, vec4f(mo.xy, 0.0, 0.0));
  put16(7u, x, y, clampHalf(vec4f(sums[5u * n + i].xyz * inv, 1.0)));
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

/** Denoised diffuse + specular (remodulated) + the emission plane (never filtered); the raw beauty where every primary ray missed (the sky). */
const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var diffuse: texture_2d<f32>;
@group(0) @binding(1) var specular: texture_2d<f32>;
@group(0) @binding(2) var raw: texture_2d<f32>;
@group(0) @binding(3) var specF: texture_2d<f32>;
@group(0) @binding(4) var outColor: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var emission: texture_2d<f32>;
@compute @workgroup_size(8, 8)
fn chainComposite(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(outColor);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let q = gid.xy;
  var c = textureLoad(raw, q, 0).xyz;
  if (textureLoad(specF, q, 0).w > 0.5) { c = textureLoad(diffuse, q, 0).xyz + textureLoad(specular, q, 0).xyz + textureLoad(emission, q, 0).xyz; }
  textureStore(outColor, q, vec4f(max(c, vec3f(0.0)), 1.0));
}
`;

/**
 * OIDN input: the progressive mean of the raw beauty, albedo factor and normal while the camera stays still (restarted on motion or a cut),
 * the normal rotated to view space (OIDN's convention: x right, y up, z toward the camera; 0 where nothing was hit).
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
fn chainOidnPrep(@builtin(global_invocation_id) gid: vec3u) {
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

// ---------------------------------------------------------------- the chain

export interface RealtimeChainSettings {
  /** Render size (the path tracer's film). */
  width: number;
  height: number;
  /** Display size (the upscaler's output). */
  displayWidth: number;
  displayHeight: number;
  /** Clip planes of the chain's camera in metres. The denoiser reads viewZ beyond `far` as sky, so `far` must reach past the scene. */
  near: number;
  far: number;
  /** OIDN weights base URL (default: `denoiser`'s own, jsDelivr). */
  tzaUrl?: string;
}

/** OIDN network size: fast = the `_small` network, balanced = the standard one. */
export type OidnQuality = 'fast' | 'balanced';

export interface OidnSettings {
  /** Network while the camera moves and until `stillFrames` still frames have been averaged. */
  moving: OidnQuality;
  /** Network once `stillFrames` frames of a still camera have been averaged. */
  still: OidnQuality;
  /** Still frames averaged before the `still` network runs (a few samples are enough for the standard network). */
  stillFrames: number;
  /** After the `still` network ran, keep that frame: no tracing or denoising until the camera moves. */
  hold: boolean;
}

export const DEFAULT_OIDN: OidnSettings = { moving: 'fast', still: 'balanced', stillFrames: 8, hold: true };

export type DenoiserKind = 'svgf' | 'oidn' | 'none';
export type ChainStage = 'raw' | 'denoised' | 'final';
export type UpscalePath = 'temporal' | 'spatial' | 'bilinear';

const align = (n: number, a: number) => Math.ceil(n / a) * a;

export class RealtimeChain {
  readonly device: GPUDevice;
  readonly s: RealtimeChainSettings;
  /** The denoiser: SVGF (temporal, demodulated), OIDN (neural, per frame) or none (the raw frame goes to the upscaler). Switch with `setDenoiser`. */
  private kindValue: DenoiserKind = 'svgf';
  oidn: OidnSettings = { ...DEFAULT_OIDN };
  /** What the last `post` ran (the OIDN network and frame count, or the denoiser kind). */
  lastDenoiser = '';
  private path: UpscalePath = 'temporal';
  private readonly svgf: DiffuseSpecularDenoiser;
  private readonly oidnPrepPipe: GPUComputePipeline;
  private readonly oidnParams: GPUBuffer;
  private readonly oidnProg: GPUBuffer;
  private readonly oidnTex: { color: GPUTexture; albedo: GPUTexture; normal: GPUTexture };
  private oidnGroup: GPUBindGroup | null = null;
  private readonly oidnNets = new Map<OidnQuality, Promise<Denoiser>>();
  private oidnCount = 0;
  private oidnCamera = '';
  private oidnHeld = false;
  /** The upscaler's engine-free core (`@pmndrs/upscaler/core`): our textures go in as they are. */
  private readonly upscaler: UpscalerCore;
  private readonly sums: GPUBuffer;
  private readonly packBuf: GPUBuffer;
  private readonly packParams: GPUBuffer;
  private readonly accumParams: GPUBuffer;
  private readonly accumPipe: GPUComputePipeline;
  private readonly packPipe: GPUComputePipeline;
  private readonly compositePipe: GPUComputePipeline;
  private readonly depthPipe: GPURenderPipeline;
  private readonly depthGroup: GPUBindGroup;
  private accumGroup: GPUBindGroup | null = null;
  private packGroup: GPUBindGroup | null = null;
  private boundPlanes: GPUBuffer | null = null;
  private readonly stride8: number;
  private readonly stride4: number;
  private readonly base4: number;
  /** Render-size inputs of the denoiser and the upscaler. */
  readonly tex: Record<'raw' | 'diffuse' | 'specular' | 'albedo' | 'specF' | 'normal' | 'motion' | 'emission' | 'viewZ' | 'motionZ' | 'roughness' | 'hitD' | 'hitS', GPUTexture>;
  private readonly depth: GPUTexture;
  /** Denoised and composited colour (render size): the upscaler's input. */
  private readonly color: GPUTexture;
  private prevView: Float32Array | null = null;
  private prevProj: Float32Array | null = null;
  private resetNext = true;

  private constructor(device: GPUDevice, s: RealtimeChainSettings, parts: {
    svgf: DiffuseSpecularDenoiser; oidnPrepPipe: GPUComputePipeline; oidnParams: GPUBuffer; oidnProg: GPUBuffer; oidnTex: RealtimeChain['oidnTex']; upscaler: UpscalerCore; sums: GPUBuffer; packBuf: GPUBuffer; packParams: GPUBuffer; accumParams: GPUBuffer;
    accumPipe: GPUComputePipeline; packPipe: GPUComputePipeline; compositePipe: GPUComputePipeline; depthPipe: GPURenderPipeline; depthGroup: GPUBindGroup;
    stride8: number; stride4: number; base4: number; tex: RealtimeChain['tex']; depth: GPUTexture; color: GPUTexture;
  }) {
    this.device = device;
    this.s = s;
    this.svgf = parts.svgf;
    this.oidnPrepPipe = parts.oidnPrepPipe;
    this.oidnParams = parts.oidnParams;
    this.oidnProg = parts.oidnProg;
    this.oidnTex = parts.oidnTex;
    this.upscaler = parts.upscaler;
    this.sums = parts.sums;
    this.packBuf = parts.packBuf;
    this.packParams = parts.packParams;
    this.accumParams = parts.accumParams;
    this.accumPipe = parts.accumPipe;
    this.packPipe = parts.packPipe;
    this.compositePipe = parts.compositePipe;
    this.depthPipe = parts.depthPipe;
    this.depthGroup = parts.depthGroup;
    this.stride8 = parts.stride8;
    this.stride4 = parts.stride4;
    this.base4 = parts.base4;
    this.tex = parts.tex;
    this.depth = parts.depth;
    this.color = parts.color;
  }

  static async create(device: GPUDevice, s: RealtimeChainSettings): Promise<RealtimeChain> {
    const { width: w, height: h } = s;
    const n = w * h;
    const sums = device.createBuffer({ label: 'chain sums', size: n * SUM_VEC4 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const stride8 = align(w * 8, 256) / 8;
    const stride4 = align(w * 4, 256) / 4;
    const base4 = 8 * h * stride8 * 2;
    const packBuf = device.createBuffer({ label: 'chain pack', size: (base4 + 5 * h * stride4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const packParams = device.createBuffer({ label: 'chain pack params', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(packParams, 0, new Uint32Array([w, h, stride8, stride4, base4, 0, 0, 0]));
    const accumParams = device.createBuffer({ label: 'chain accum params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(accumParams, 0, new Uint32Array([n, 0, 0, 0]));
    const mk = (name: string, format: GPUTextureFormat, usage = TEX | COPY_DST_T | COPY_SRC_T) => device.createTexture({ label: `chain ${name}`, size: { width: w, height: h }, format, usage });
    const tex = {
      raw: mk('raw', 'rgba16float'), diffuse: mk('diffuse', 'rgba16float'), specular: mk('specular', 'rgba16float'), albedo: mk('albedo', 'rgba16float'),
      specF: mk('specF', 'rgba16float'), normal: mk('normal', 'rgba16float'), motion: mk('motion', 'rgba16float'), emission: mk('emission', 'rgba16float'),
      viewZ: mk('viewZ', 'r32float'), motionZ: mk('motionZ', 'r32float'), roughness: mk('roughness', 'r32float'), hitD: mk('hitD', 'r32float'), hitS: mk('hitS', 'r32float'),
    };
    const depth = mk('depth', 'depth32float', TEX | GPUTextureUsage.RENDER_ATTACHMENT);
    const color = mk('color', 'rgba16float', TEX | GPUTextureUsage.STORAGE_BINDING | COPY_SRC_T | COPY_DST_T);
    const module = (code: string, name: string) => device.createShaderModule({ label: `chain ${name}`, code });
    const depthModule = module(DEPTH_WGSL, 'depth');
    const [accumPipe, packPipe, compositePipe, depthPipe, oidnPrepPipe] = await Promise.all([
      device.createComputePipelineAsync({ label: 'chain accumulate', layout: 'auto', compute: { module: module(ACCUM_WGSL, 'accumulate'), entryPoint: 'chainAccumulate' } }),
      device.createComputePipelineAsync({ label: 'chain pack', layout: 'auto', compute: { module: module(PACK_WGSL, 'pack'), entryPoint: 'chainPack' } }),
      device.createComputePipelineAsync({ label: 'chain composite', layout: 'auto', compute: { module: module(COMPOSITE_WGSL, 'composite'), entryPoint: 'chainComposite' } }),
      device.createRenderPipelineAsync({
        label: 'chain depth', layout: 'auto',
        vertex: { module: depthModule, entryPoint: 'vs' },
        fragment: { module: depthModule, entryPoint: 'fs', targets: [] },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'always' },
      }),
      device.createComputePipelineAsync({ label: 'chain oidn prep', layout: 'auto', compute: { module: module(OIDN_PREP_WGSL, 'oidn prep'), entryPoint: 'chainOidnPrep' } }),
    ]);
    const storageTex = TEX | GPUTextureUsage.STORAGE_BINDING;
    const oidnTex = { color: mk('oidn color', 'rgba16float', storageTex), albedo: mk('oidn albedo', 'rgba16float', storageTex), normal: mk('oidn normal', 'rgba16float', storageTex) };
    const oidnProg = device.createBuffer({ label: 'chain oidn mean', size: n * 3 * 16, usage: GPUBufferUsage.STORAGE });
    const oidnParams = device.createBuffer({ label: 'chain oidn params', size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const depthParams = device.createBuffer({ label: 'chain depth params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(depthParams, 0, new Float32Array([s.near, s.far, 0, 0]));
    const depthGroup = device.createBindGroup({ layout: depthPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: depthParams } }, { binding: 1, resource: tex.viewZ.createView() }] });

    const svgf = new DiffuseSpecularDenoiser(device, { remodulate: true } as never);
    svgf.configure({ width: w, height: h });
    const upscaler = new UpscalerCore({ device });
    upscaler.configure({ displayWidth: s.displayWidth, displayHeight: s.displayHeight, renderWidth: w, renderHeight: h, path: 'temporal', jitter: false });
    await upscaler.prepare();
    return new RealtimeChain(device, s, { svgf, oidnPrepPipe, oidnParams, oidnProg, oidnTex, upscaler, sums, packBuf, packParams, accumParams, accumPipe, packPipe, compositePipe, depthPipe, depthGroup, stride8, stride4, base4, tex, depth, color });
  }

  get denoiser(): DenoiserKind {
    return this.kindValue;
  }

  /** Switches the denoiser; the histories and OIDN's average restart. Picking OIDN starts loading its networks. */
  setDenoiser(kind: DenoiserKind): void {
    this.kindValue = kind;
    this.oidnCount = 0;
    this.oidnCamera = '';
    this.oidnHeld = false;
    this.resetNext = true;
    if (kind === 'oidn') void this.ready();
  }

  /** Loads the OIDN networks of the current settings (the first `post` waits for them anyway). */
  async ready(): Promise<void> {
    if (this.kindValue !== 'oidn') return;
    await Promise.all([this.net(this.oidn.moving), this.net(this.oidn.still)]);
  }

  /** One `Denoiser` per network size: a size change in one instance would reload its weights. */
  private net(quality: OidnQuality): Promise<Denoiser> {
    let net = this.oidnNets.get(quality);
    if (!net) {
      const precision = this.device.features.has('shader-f16') ? 'fp16' : 'fp32';
      net = import('denoiser').then(({ Denoiser }) => Denoiser.create({ device: this.device, precision, quality, ...(this.s.tzaUrl ? { tzaUrl: this.s.tzaUrl } : {}) }));
      this.oidnNets.set(quality, net);
    }
    return net;
  }

  /** OIDN with `hold`: true while this camera's finished still frame is held (skip tracing and `post`). */
  heldFor(camera: PtLookAtCamera): boolean {
    return this.kindValue === 'oidn' && this.oidnHeld && !this.resetNext && cameraKey(camera) === this.oidnCamera;
  }

  /** Drop the denoiser's and the upscaler's history on the next `post` (a camera cut). */
  resetHistory(): void {
    this.resetNext = true;
  }

  /** Start of a frame: clears the sample sums. */
  beginFrame(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.sums);
  }

  /** After each path-tracing encode (`pt.frameBuffers`): adds the sample to the sums. */
  accumulate(encoder: GPUCommandEncoder, fb: { accum: GPUBuffer; planes: GPUBuffer }): void {
    if (this.boundPlanes !== fb.planes) {
      this.boundPlanes = fb.planes;
      const entries = (list: GPUBuffer[]) => list.map((buffer, binding) => ({ binding, resource: { buffer } }));
      this.accumGroup = this.device.createBindGroup({ layout: this.accumPipe.getBindGroupLayout(0), entries: entries([this.accumParams, fb.accum, fb.planes, this.sums]) });
      this.packGroup = this.device.createBindGroup({ layout: this.packPipe.getBindGroupLayout(0), entries: entries([this.packParams, this.sums, fb.planes, this.packBuf]) });
    }
    const pass = encoder.beginComputePass({ label: 'chain accumulate' });
    pass.setPipeline(this.accumPipe);
    pass.setBindGroup(0, this.accumGroup!);
    pass.dispatchWorkgroups(Math.ceil((this.s.width * this.s.height) / 64));
    pass.end();
  }

  /** End of the path tracing: sums -> the input textures, and the depth guide. */
  pack(encoder: GPUCommandEncoder): void {
    const { width: w, height: h } = this.s;
    if (!this.packGroup) throw new Error('RealtimeChain.pack: call accumulate() first');
    const pass = encoder.beginComputePass({ label: 'chain pack' });
    pass.setPipeline(this.packPipe);
    pass.setBindGroup(0, this.packGroup);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    pass.end();
    const t = this.tex;
    [t.raw, t.diffuse, t.specular, t.albedo, t.specF, t.normal, t.motion, t.emission].forEach((texture, k) =>
      encoder.copyBufferToTexture({ buffer: this.packBuf, offset: k * h * this.stride8 * 8, bytesPerRow: this.stride8 * 8, rowsPerImage: h }, { texture }, { width: w, height: h }));
    [t.viewZ, t.motionZ, t.roughness, t.hitD, t.hitS].forEach((texture, k) =>
      encoder.copyBufferToTexture({ buffer: this.packBuf, offset: (this.base4 + k * h * this.stride4) * 4, bytesPerRow: this.stride4 * 4, rowsPerImage: h }, { texture }, { width: w, height: h }));
    const dp = encoder.beginRenderPass({ label: 'chain depth', colorAttachments: [], depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    dp.setPipeline(this.depthPipe);
    dp.setBindGroup(0, this.depthGroup);
    dp.draw(3);
    dp.end();
  }

  /** The upscaler's path: temporal (reprojects and accumulates frames), spatial (FSR1 on one frame) or bilinear. */
  get upscalePath(): UpscalePath {
    return this.path;
  }

  /** Switches the upscaler's path (compiles its passes; the histories restart). */
  async setUpscalePath(path: UpscalePath): Promise<void> {
    if (path === this.path) return;
    this.path = path;
    this.upscaler.configure({ displayWidth: this.s.displayWidth, displayHeight: this.s.displayHeight, renderWidth: this.s.width, renderHeight: this.s.height, path, jitter: false });
    await this.upscaler.prepare();
    this.resetNext = true;
  }

  /**
   * Denoise, composite, upscale; call after the frame's tracing and `pack` were submitted. `camera` is this frame's (unjittered) camera; the
   * previous one is remembered. Records and submits its own work (the denoiser and the upscaler submit theirs).
   */
  post(camera: PtLookAtCamera, deltaTime = 1 / 60): Promise<void> | void {
    const { width: w, height: h, near, far } = this.s;
    const view = viewMatrix(camera);
    const proj = projectionMatrix(camera.yFovDeg, w / h, near, far);
    const reset = this.resetNext;
    this.resetNext = false;
    if (reset) {
      this.svgf.resetHistory();
      this.prevView = view;
      this.prevProj = proj;
    }
    if (this.kindValue === 'oidn') return this.postOidn(camera, reset, deltaTime, view, proj);
    const t = this.tex;
    const encoder = this.device.createCommandEncoder({ label: 'chain composite' });
    if (this.kindValue === 'svgf') {
      const out = this.svgf.dispatchBoth(
        { diffuse: { radiance: t.diffuse, hitDistance: t.hitD }, specular: { radiance: t.specular, hitDistance: t.hitS, demodulation: t.specF } } as never,
        { depth: this.depth, normal: t.normal, motion: t.motion, roughness: t.roughness, albedo: t.albedo, viewZ: t.viewZ, motionZ: t.motionZ },
        { view, projection: proj, prevView: this.prevView!, prevProjection: this.prevProj!, near, far, jitter: [0, 0], position: [...camera.position] as [number, number, number] },
      ) as unknown as { diffuse: GPUTexture; specular: GPUTexture };
      const group = this.device.createBindGroup({ layout: this.compositePipe.getBindGroupLayout(0), entries: [out.diffuse.createView(), out.specular.createView(), t.raw.createView(), t.specF.createView(), this.color.createView(), t.emission.createView()].map((resource, binding) => ({ binding, resource })) });
      const pass = encoder.beginComputePass({ label: 'chain composite' });
      pass.setPipeline(this.compositePipe);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      pass.end();
    } else {
      encoder.copyTextureToTexture({ texture: t.raw }, { texture: this.color }, { width: w, height: h });
    }
    this.device.queue.submit([encoder.finish()]);
    this.lastDenoiser = this.kindValue;
    this.upscale(camera, reset, deltaTime, view, proj);
  }

  /**
   * OIDN on this frame: the progressive mean of a still camera (restarted when it moves or after a cut), the `moving` network until
   * `stillFrames` frames are averaged, then the `still` network (and with `hold` the frame is kept). Waits for the network; writes `color`.
   */
  private async postOidn(camera: PtLookAtCamera, cut: boolean, deltaTime: number, view: Float32Array, proj: Float32Array): Promise<void> {
    const { width: w, height: h } = this.s;
    const key = cameraKey(camera);
    const moved = cut || key !== this.oidnCamera;
    this.oidnCamera = key;
    this.oidnCount = moved ? 1 : this.oidnCount + 1;
    this.oidnHeld = false;
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
    const encoder = this.device.createCommandEncoder({ label: 'chain oidn prep' });
    const pass = encoder.beginComputePass({ label: 'chain oidn prep' });
    pass.setPipeline(this.oidnPrepPipe);
    pass.setBindGroup(0, this.oidnGroup);
    pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    const o = this.oidn;
    const still = this.oidnCount >= o.stillFrames;
    const quality = still ? o.still : o.moving;
    const net = await this.net(quality);
    const out = await net.denoiseTextures({ color: this.oidnTex.color, albedo: this.oidnTex.albedo, normal: this.oidnTex.normal, hdr: true, output: this.color, transfer: 'linear' });
    if (!out) throw new Error('OIDN: the run was aborted');
    this.lastDenoiser = `OIDN ${quality === 'fast' ? 'small' : 'standard'} network, ${this.oidnCount} frame${this.oidnCount === 1 ? '' : 's'} averaged`;
    if (still && o.hold) this.oidnHeld = true;
    this.upscale(camera, cut, deltaTime, view, proj);
  }

  private upscale(camera: PtLookAtCamera, reset: boolean, deltaTime: number, view: Float32Array, proj: Float32Array): void {
    const { near, far } = this.s;
    this.upscaler.dispatch({ color: this.color, depth: this.depth, velocity: this.tex.motion, reset, deltaTime }, { near, far, perspective: true });
    this.prevView = view;
    this.prevProj = proj;
  }

  /** A stage's texture: the raw frame and the denoised colour at render size, the upscaler's output at display size (all linear HDR rgba16float). */
  stage(s: ChainStage): GPUTexture {
    if (s === 'raw') return this.tex.raw;
    if (s === 'denoised') return this.color;
    return this.upscaler.output;
  }

  /** A stage as linear RGBA f32, row 0 = top (a readback for checks: never in a frame loop). */
  async read(stage: ChainStage): Promise<Float32Array> {
    const texture = this.stage(stage);
    const { width: w, height: h } = texture;
    const stride = align(w * 8, 256);
    const staging = this.device.createBuffer({ size: stride * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow: stride, rowsPerImage: h }, { width: w, height: h });
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const raw = new Uint16Array(staging.getMappedRange().slice(0));
    staging.destroy();
    const out = new Float32Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w * 4; x++) out[y * w * 4 + x] = halfToFloat(raw[(y * stride) / 2 + x]!);
    return out;
  }

  destroy(): void {
    this.svgf.dispose();
    this.upscaler.dispose();
    // dispose() keeps the device (destroyDevice() would take ours down with it).
    for (const net of this.oidnNets.values()) void net.then((d) => d.dispose(), () => {});
    for (const b of [this.sums, this.packBuf, this.packParams, this.accumParams, this.oidnParams, this.oidnProg]) b.destroy();
    for (const t of [...Object.values(this.tex), ...Object.values(this.oidnTex), this.depth, this.color]) t.destroy();
  }
}

function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return sign * f * 2 ** -24;
  if (e === 31) return f ? NaN : sign * Infinity;
  return sign * (1 + f / 1024) * 2 ** (e - 15);
}

// ---------------------------------------------------------------- display

const SHOW_WGSL = /* wgsl */ `
struct P { size: vec2f, exposure: f32, divider: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let q = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(q * 2.0 - 1.0, 0.0, 1.0);
}
// Krzysztof Narkowicz's 2016 fit of the ACES RRT + ODT (THIRD_PARTY.md).
fn aces(x: vec3f) -> vec3f { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3f(0.0), vec3f(1.0)); }
fn srgb(c: vec3f) -> vec3f { return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, 12.92 * c, c <= vec3f(0.0031308)); }
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  if (p.divider >= 0.0 && pos.x >= p.divider && pos.x < p.divider + 2.0) { return vec4f(0.85, 0.85, 0.8, 1.0); }
  // A render-size stage is stretched with bilinear filtering, as a plain upscale would.
  let c = textureSampleLevel(src, smp, pos.xy / p.size, 0.0).rgb;
  return vec4f(srgb(aces(max(c, vec3f(0.0)) * p.exposure)), 1.0);
}
`;

/** Draws a stage (linear HDR) into the canvas: stretched to the target, exposure, ACES, sRGB; `columns` limits it to a column range (a split view). */
export class StageView {
  exposure = 1;
  private readonly device: GPUDevice;
  private readonly pipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  /** One uniform slot per draw of a frame (two draws in a split view write different dividers before one submit). */
  private readonly uniforms: GPUBuffer[] = [];
  private readonly groups = new Map<string, { texture: GPUTexture; group: GPUBindGroup }>();

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    const module = device.createShaderModule({ label: 'stage view', code: SHOW_WGSL });
    this.pipeline = device.createRenderPipeline({ label: 'stage view', layout: 'auto', vertex: { module, entryPoint: 'vs' }, fragment: { module, entryPoint: 'fs', targets: [{ format }] } });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    for (let i = 0; i < 2; i++) this.uniforms.push(device.createBuffer({ label: `stage view ${i}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
  }

  /** `slot` 0 or 1: draws of one frame use different slots. `columns` [x0, x1) of the target; `load` keeps an earlier draw; `divider` draws a line at x0. */
  encode(encoder: GPUCommandEncoder, src: GPUTexture, target: GPUTextureView, size: [number, number], o: { slot?: number; columns?: [number, number]; load?: boolean; divider?: boolean } = {}): void {
    const slot = o.slot ?? 0;
    const uniform = this.uniforms[slot]!;
    const key = `${slot}`;
    let cached = this.groups.get(key);
    if (!cached || cached.texture !== src) {
      cached = { texture: src, group: this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: src.createView() }, { binding: 2, resource: this.sampler }] }) };
      this.groups.set(key, cached);
    }
    this.device.queue.writeBuffer(uniform, 0, new Float32Array([size[0], size[1], this.exposure, o.divider && o.columns ? o.columns[0] : -1]));
    const pass = encoder.beginRenderPass({ label: 'stage view', colorAttachments: [{ view: target, loadOp: o.load ? 'load' : 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, cached.group);
    if (o.columns) {
      const x0 = Math.max(0, Math.min(size[0], Math.floor(o.columns[0])));
      const x1 = Math.max(x0, Math.min(size[0], Math.floor(o.columns[1])));
      pass.setScissorRect(x0, 0, x1 - x0, size[1]);
    }
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    for (const u of this.uniforms) u.destroy();
  }
}
