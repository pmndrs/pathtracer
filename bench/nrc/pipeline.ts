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
 * Per frame:  arm.beginFrame(encoder); { pt.encode(encoder, 1); arm.accumulate(encoder, pt.frameBuffers) } x N; arm.pack(encoder, N);
 *             submit; arm.post(camera, reset)  (the denoiser and the upscaler record and submit their own command buffers; no readback)
 */
import { DiffuseSpecularDenoiser } from 'denoiser/realtime';
import { Upscaler } from '@pmndrs/upscaler';
import type { PtLookAtCamera } from '@pmndrs/pathtracer';
import { PIPE_FAR, PIPE_NEAR, projectionMatrix, viewMatrix, worldMatrix } from './pipeline-math.ts';

export { PIPE_FAR, PIPE_NEAR, projectionMatrix, resample, viewMatrix, worldMatrix } from './pipeline-math.ts';


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

/** Per-pixel sums over the N samples of a frame (5 vec4 per pixel): beauty, diffuse-class radiance, specular-class radiance, albedo factor, specular factor. */
const SUM_VEC4 = 5;

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
}
`;

/**
 * Pack: sums / N and the last sample's guides -> one buffer of texture rows (copied to the textures after). Sections, each `rows` rows of a
 * 256-byte aligned stride: 0..6 rgba16float (raw beauty, diffuse demod, specular demod, albedo factor, specular factor, normal, motion), then
 * 7..11 r32float (viewZ, motionZ, roughness, diffuse hit distance, specular hit distance).
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

/** Denoised diffuse + specular (remodulated), and the raw sky where the primary ray missed (the denoiser outputs 0 there). */
const COMPOSITE_WGSL = /* wgsl */ `
@group(0) @binding(0) var diffuse: texture_2d<f32>;
@group(0) @binding(1) var specular: texture_2d<f32>;
@group(0) @binding(2) var raw: texture_2d<f32>;
@group(0) @binding(3) var specF: texture_2d<f32>;
@group(0) @binding(4) var outColor: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn pipeComposite(@builtin(global_invocation_id) gid: vec3u) {
  let size = textureDimensions(outColor);
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let q = gid.xy;
  let seen = textureLoad(specF, q, 0).w > 0.5;
  // Pack wrote the mean of "seen" flags into specF.w: a pixel whose samples all missed keeps its raw (sky) value.
  var c = textureLoad(raw, q, 0).xyz;
  if (seen) { c = textureLoad(diffuse, q, 0).xyz + textureLoad(specular, q, 0).xyz; }
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

// ---------------------------------------------------------------- arm

export interface PipelineSettings {
  /** Render (input) film. */
  width: number;
  height: number;
  /** Display (output) size; 2x the render film by default (960x540 -> 1920x1080). */
  displayWidth: number;
  displayHeight: number;
  /** Denoiser options (`DiffuseSpecularDenoiser`); `remodulate` is always on. Default: the Denoiser's defaults (SVGF, anti-lag auto). */
  denoiser?: Record<string, unknown>;
  /** Upscaler runtime settings overrides (`Upscaler.settings`). */
  upscaler?: Record<string, unknown>;
}

const align = (n: number, a: number) => Math.ceil(n / a) * a;

export type PipelineStage = 'raw' | 'denoised' | 'final';

export interface PipelineFrameBuffers {
  accum: GPUBuffer;
  planes: GPUBuffer;
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
  private readonly stride8!: number;
  private readonly stride4!: number;
  private readonly base4!: number;
  /** Render-res input textures (rgba16float sections 0..6, r32float 7..11). */
  readonly tex!: Record<'raw' | 'diffuse' | 'specular' | 'albedo' | 'specF' | 'normal' | 'motion' | 'viewZ' | 'motionZ' | 'roughness' | 'hitD' | 'hitS', GPUTexture>;
  readonly depth!: GPUTexture;
  /** Denoised + composited colour (render res): the upscaler's input. */
  readonly color!: GPUTexture;
  private readonly keys!: { color: object; depth: object; velocity: object };
  private prevView: Float32Array | null = null;
  private prevProj: Float32Array | null = null;
  private resetNext = true;
  private flicker: { state: GPUBuffer; params: GPUBuffer; group: GPUBindGroup; w: number; h: number; frames: number }[] = [];

  private constructor(init: Record<string, unknown>) {
    Object.assign(this, init);
  }

  static async create(device: GPUDevice, s: PipelineSettings, label: string): Promise<PipelineArm> {
    const { width: w, height: h } = s;
    const n = w * h;
    const sums = device.createBuffer({ label: `${label} sums`, size: n * SUM_VEC4 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const stride8 = align(w * 8, 256) / 8;
    const stride4 = align(w * 4, 256) / 4;
    const base4 = 7 * h * stride8 * 2;
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
      viewZ: mk('viewZ', 'r32float'),
      motionZ: mk('motionZ', 'r32float'),
      roughness: mk('roughness', 'r32float'),
      hitD: mk('hitD', 'r32float'),
      hitS: mk('hitS', 'r32float'),
    };
    const depth = mk('depth', 'depth32float', TEX | GPUTextureUsage.RENDER_ATTACHMENT);
    const color = mk('color', 'rgba16float', TEX | GPUTextureUsage.STORAGE_BINDING | COPY_SRC_T | COPY_DST_T);
    const module = (code: string, name: string) => device.createShaderModule({ label: `${label} ${name}`, code });
    const [accumPipe, packPipe, compositePipe, flickerPipe, depthPipe] = await Promise.all([
      device.createComputePipelineAsync({ label: `${label} accumulate`, layout: 'auto', compute: { module: module(ACCUM_WGSL, 'accumulate'), entryPoint: 'pipeAccumulate' } }),
      device.createComputePipelineAsync({ label: `${label} pack`, layout: 'auto', compute: { module: module(PACK_WGSL, 'pack'), entryPoint: 'pipePack' } }),
      device.createComputePipelineAsync({ label: `${label} composite`, layout: 'auto', compute: { module: module(COMPOSITE_WGSL, 'composite'), entryPoint: 'pipeComposite' } }),
      device.createComputePipelineAsync({ label: `${label} flicker`, layout: 'auto', compute: { module: module(FLICKER_WGSL, 'flicker'), entryPoint: 'pipeFlicker' } }),
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
    const host = new UpscalerHost(device);
    const upscaler = new Upscaler({ renderer: host as never });
    upscaler.configure({ displayWidth: s.displayWidth, displayHeight: s.displayHeight, renderWidth: w, renderHeight: h, path: 'temporal', jitter: false });
    await upscaler.init();
    Object.assign(upscaler.settings, s.upscaler ?? {});
    const keys = { color: host.texture(color, `${label} color`), depth: host.texture(depth, `${label} depth`), velocity: host.texture(tex.motion, `${label} motion`) };
    // The accumulate and pack groups need the kernel's planes buffer: built on the first `accumulate`.
    return new PipelineArm({ device, s, denoiser, upscaler, host, sums, packBuf, packParams, accumParams, accumPipe, packPipe, compositePipe, depthPipe, depthGroup, flickerPipe, stride8, stride4, base4, tex, depth, color, keys });
  }

  /** Start of a frame: clears the sample sums. */
  beginFrame(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.sums);
  }

  /** After each path-tracing encode of the frame (`pt.frameBuffers`): adds the sample to the sums. */
  accumulate(encoder: GPUCommandEncoder, fb: PipelineFrameBuffers): void {
    if (this.accumKey !== fb.planes) {
      this.accumKey = fb.planes;
      this.accumGroup = this.device.createBindGroup({ layout: this.accumPipe.getBindGroupLayout(0), entries: [this.accumParams, fb.accum, fb.planes, this.sums].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      this.packGroup = this.device.createBindGroup({ layout: this.packPipe.getBindGroupLayout(0), entries: [this.packParams, this.sums, fb.planes, this.packBuf].map((buffer, binding) => ({ binding, resource: { buffer } })) });
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
    const half = [t.raw, t.diffuse, t.specular, t.albedo, t.specF, t.normal, t.motion];
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
  post(camera: PtLookAtCamera, options: { denoise?: boolean; deltaTime?: number } = {}): void {
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
    let dOut: { diffuse: GPUTexture; specular: GPUTexture } | null = null;
    if (options.denoise !== false) {
      dOut = this.denoiser.dispatchBoth(
        { diffuse: { radiance: t.diffuse, hitDistance: t.hitD }, specular: { radiance: t.specular, hitDistance: t.hitS, demodulation: t.specF } } as never,
        { depth: this.depth, normal: t.normal, motion: t.motion, roughness: t.roughness, albedo: t.albedo, viewZ: t.viewZ, motionZ: t.motionZ },
        { view, projection: proj, prevView: this.prevView as Float32Array, prevProjection: this.prevProj as Float32Array, near: PIPE_NEAR, far: PIPE_FAR, jitter: [0, 0], position: [...camera.position] as [number, number, number] },
      ) as unknown as { diffuse: GPUTexture; specular: GPUTexture };
    }
    const encoder = this.device.createCommandEncoder({ label: 'pipe composite' });
    if (dOut) {
      const group = this.device.createBindGroup({ layout: this.compositePipe.getBindGroupLayout(0), entries: [dOut.diffuse.createView(), dOut.specular.createView(), t.raw.createView(), t.specF.createView(), this.color.createView()].map((resource, binding) => ({ binding, resource })) });
      const pass = encoder.beginComputePass({ label: 'pipe composite' });
      pass.setPipeline(this.compositePipe);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
      pass.end();
    } else {
      encoder.copyTextureToTexture({ texture: t.raw }, { texture: this.color }, { width: w, height: h });
    }
    this.device.queue.submit([encoder.finish()]);
    this.upCamera.set(camera, aspect);
    this.upscaler.beginFrame(this.upCamera as never);
    this.upscaler.endFrame(this.upCamera as never);
    this.upscaler.dispatch({ color: this.keys.color, depth: this.keys.depth, velocity: this.keys.velocity, reset, deltaTime: options.deltaTime ?? 1 / 60 } as never, this.upCamera as never);
    this.prevView = view;
    this.prevProj = proj;
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
    for (const b of [this.sums, this.packBuf, this.packParams, this.accumParams]) b.destroy();
    for (const t of [...Object.values(this.tex), this.depth, this.color]) t.destroy();
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
