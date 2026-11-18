/**
 * Host side of ReSTIR DI (P4): the four pre-passes (G-buffer, initial candidates, temporal, spatial) that `PtMegakernel` runs before each
 * shade dispatch when `integrator.restir` is set. One `encodeFrame` = one frame = one sample of the progressive accumulation.
 *
 * Previous-frame reprojection (P4.2e) uses the previous camera the MEGAKERNEL owns: the one `setPrevCamera` / `nextFrame` set (the real-time mode's, the one the frame
 * planes' motion vectors project through), else the camera of the last encoded frame (a static scene, a moving camera). There is no copy of it in here
 * (DIVERGES V130 -> V170): `encodeFrame` takes it as an argument.
 *
 * Inputs the passes share with the shade kernel (P4.2c, P4.2e): the V-buffer texture (`primary: 'vbuffer'`: the G-buffer pass resolves it instead of tracing the camera
 * ray), the blue-noise texture (`sampler: 'bluenoise'`: the G-buffer pass draws the same jitter pair as the shade pass) and the layered-albedo correction texture.
 * `setInputs` hands the views over; the megakernel does it before every encode.
 */
import { DEFAULT_SCENE_PACKING, rtCreateComputePipelineAsync, rtModuleWgsl, rtPackSceneAccess, rtTransformWgsl, type Caps, type RtWgslBuild } from '@pmndrs/raytracer';
import { ptCameraBasis, type PtCameraBasis, type PtLookAtCamera } from '../integrator/camera.ts';
import { PT_BLUENOISE_BINDING } from '../integrator/dimensions.ts';
import { PT_WORKGROUP, resolveIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { ptRaytracerModuleOptions, ptShadingModulesWgsl, type PtMegakernelOptions } from '../integrator/megakernel.ts';
import { PT_VBUFFER_BINDING, ptVbufferWgsl } from '../integrator/vbuffer.ts';
import { PT_LAYER_CORR_BINDING, ptLayerCorrLayoutEntry, ptLayerCorrNeedsTexture } from '../bsdf/layered-correction.ts';
import { PT_RESTIR_BINDING, PT_RESTIR_ENTRIES, PT_RESTIR_FLAG, PT_RESTIR_PARAMS_BYTES, PT_RESTIR_REGION, PT_RESTIR_REGIONS, ptRestirGbufBytes, ptRestirWgsl, resolveRestirOptions, type ResolvedRestirOptions } from './restir-wgsl.ts';

const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const UNIFORM = 0x40;
const COMPUTE = 0x4;

/** Records in the params ring (4 passes per frame): enough for 256 frames outstanding. */
export const PT_RESTIR_PARAM_SLOTS = 1024;

type BuildOptions = Pick<PtMegakernelOptions, 'format' | 'layout' | 'integrator' | 'textures' | 'scenePacking' | 'minify' | 'prune' | 'shortenNames'>;

/**
 * Why this kernel cannot run ReSTIR DI (null = it can). Supported since P4.2b: textured scenes (material lookups at the stored footprint, normal maps, alpha MASK primaries),
 * OpenPBR mode with the deterministic lobes (legacy and OpenPBR `thin`, `mix`, `layered` with `layering: 'albedo'`), `primary: 'vbuffer'` and `sampler: 'bluenoise'`.
 */
export function ptRestirUnsupported(options: Pick<PtMegakernelOptions, 'integrator' | 'textures'> & Partial<Pick<PtMegakernelOptions, 'layout'>>): string | null {
  const o = resolveIntegratorOptions({ ...options.integrator, ...(options.textures ? { textures: true } : {}) });
  if (!o.restir) return 'integrator.restir is not set';
  if (!o.nee) return 'ReSTIR DI replaces next-event estimation at the first vertex: integrator.nee must be on';
  if (o.hooks) return 'path hooks are not supported with ReSTIR yet';
  const m = o.materials;
  if (m) {
    if (m.features.aniso) return 'anisotropic materials are not supported with ReSTIR yet (the target would need the tangent frame of the first vertex)';
    if (m.features.fuzz) return 'fuzz materials are not supported with ReSTIR yet (the target would need the fuzz lobe state)';
    if (m.features.thinFilm) return 'thin-film materials are not supported with ReSTIR yet';
    if (m.features.layered && (m.layered?.layering ?? 'albedo') === 'stochastic') return "stochastic layering is not supported with ReSTIR (its f is a random walk: the target must be deterministic); use layering: 'albedo'";
    if (o.lod !== 'off' && o.lodDepth === 0) return 'materialLodDepth 0 reduces the first vertex: not supported with ReSTIR';
  }
  if (o.restirShared && options.layout && !options.layout.positions) return 'restir.sharedHit resolves the primary hit from barycentrics: the scene copy needs surface positions (layout.positions)';
  return null;
}

/** The module of the four passes (what `PtRestirPasses.create` compiles). */
export function ptRestirBuild(caps: Caps, options: BuildOptions): RtWgslBuild {
  const why = ptRestirUnsupported(options);
  if (why) throw new Error(`ptRestirBuild: ${why}`);
  const { integratorOptions, parts } = ptShadingModulesWgsl(options);
  const resolved = resolveIntegratorOptions(integratorOptions);
  const raytracer = rtModuleWgsl(caps, ptRaytracerModuleOptions(options));
  const ours = [...parts, ...(resolved.primary === 'vbuffer' ? [ptVbufferWgsl()] : []), ptRestirWgsl(integratorOptions)].join('\n');
  const packed = (options.scenePacking ?? DEFAULT_SCENE_PACKING) === 'packed';
  const { code, map } = rtTransformWgsl(`${raytracer}\n${packed ? rtPackSceneAccess(ours) : ours}`, { prune: options.prune, shorten: options.shortenNames, minify: options.minify, keep: [...PT_RESTIR_ENTRIES] });
  return { code, map };
}

export interface PtRestirStats {
  /** Bytes of G-buffer + reservoirs (+ the shared hit buffer) at the current film size. */
  bytes: number;
  /** Of which the two G-buffer frames. */
  gbufBytes: number;
}

/** What the megakernel hands the passes for one frame. */
export interface PtRestirFrame {
  /** Current camera. */
  camera: PtLookAtCamera;
  /** Previous camera, null = no history for this frame. */
  prev: PtLookAtCamera | null;
  seed: number;
  /** Frame (sample) index, the sampler's. */
  frame: number;
  /** `PtLimits.textureConeScale` (textured scenes). */
  coneScale: number;
  /** `primary: 'vbuffer'`: the frame's sub-pixel offset in pixels, 0 = centre. */
  jitter: readonly [number, number];
}

/** Textures the passes bind next to their own buffers. */
export interface PtRestirInputs {
  blueNoise?: GPUTextureView | null;
  vbuffer?: GPUTextureView | null;
  layerCorrection?: GPUTextureView | null;
}

export class PtRestirPasses {
  readonly settings: ResolvedRestirOptions;
  readonly wgslBytes: number;
  private readonly device: GPUDevice;
  private readonly pipelines: Record<(typeof PT_RESTIR_ENTRIES)[number], GPUComputePipeline>;
  private readonly ioLayout: GPUBindGroupLayout;
  private readonly paramsBuffer: GPUBuffer;
  private readonly slotBytes: number;
  private readonly integrator: NonNullable<PtMegakernelOptions['integrator']>;
  private readonly shared: boolean;
  private readonly wantBlue: boolean;
  private readonly wantVbuffer: boolean;
  private readonly wantCorr: boolean;
  private slot = 0;
  private ioGroup: GPUBindGroup | null = null;
  private inputs: PtRestirInputs = {};
  private gbuf: GPUBuffer | null = null;
  /** The reservoirs (4 regions, `PT_RESTIR_REGION`). The shade pass binds the whole buffer and picks its region through `ptParams.limits.w`. */
  reservoirs: GPUBuffer | null = null;
  /** `sharedHit`: one V-buffer texel (`rgba32uint` words, `vbuffer.ts`) per pixel, the primary hit of the frame the G-buffer pass traced; the shade pass resolves it. */
  hitBuffer: GPUBuffer | null = null;
  private width = 0;
  private height = 0;
  /** Frames encoded since `reset` (parity picks the ping-pong halves). */
  frame = 0;
  /** Per-pass switches the check pages flip without rebuilding (all runtime). */
  overrides: Partial<Pick<ResolvedRestirOptions, 'temporal' | 'spatial' | 'candidates'>> = {};

  private constructor(device: GPUDevice, settings: ResolvedRestirOptions, pipelines: PtRestirPasses['pipelines'], ioLayout: GPUBindGroupLayout, wgslBytes: number, integrator: NonNullable<PtMegakernelOptions['integrator']>, flags: { shared: boolean; blue: boolean; vbuffer: boolean; corr: boolean }) {
    this.device = device;
    this.settings = settings;
    this.pipelines = pipelines;
    this.ioLayout = ioLayout;
    this.wgslBytes = wgslBytes;
    this.integrator = integrator;
    this.shared = flags.shared;
    this.wantBlue = flags.blue;
    this.wantVbuffer = flags.vbuffer;
    this.wantCorr = flags.corr;
    this.slotBytes = Math.max(PT_RESTIR_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
    this.paramsBuffer = device.createBuffer({ label: 'pt restir params', size: this.slotBytes * PT_RESTIR_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
  }

  static async create(device: GPUDevice, caps: Caps, options: BuildOptions & { label?: string }, walkLayout: GPUBindGroupLayout, sceneLayout: GPUBindGroupLayout): Promise<PtRestirPasses> {
    const label = `${options.label ?? 'pt'} restir`;
    const { code, map } = ptRestirBuild(caps, options);
    const resolved = resolveIntegratorOptions({ ...options.integrator, ...(options.textures ? { textures: true } : {}) });
    const flags = {
      shared: resolved.restirShared,
      blue: resolved.sampler === 'bluenoise',
      vbuffer: resolved.primary === 'vbuffer',
      corr: !!resolved.materials?.features.layered && ptLayerCorrNeedsTexture(resolved.materials.layered),
    };
    const ioLayout = device.createBindGroupLayout({
      label: `${label} io`,
      entries: [
        { binding: PT_RESTIR_BINDING.params, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_RESTIR_PARAMS_BYTES } },
        { binding: PT_RESTIR_BINDING.gbuf, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: PT_RESTIR_BINDING.reservoirs, visibility: COMPUTE, buffer: { type: 'storage' } },
        ...(flags.shared ? [{ binding: PT_RESTIR_BINDING.hit, visibility: COMPUTE, buffer: { type: 'storage' as const } }] : []),
        ...(flags.vbuffer ? [{ binding: PT_VBUFFER_BINDING, visibility: COMPUTE, texture: { sampleType: 'uint' as const, viewDimension: '2d' as const } }] : []),
        ...(flags.corr ? [ptLayerCorrLayoutEntry()] : []),
        ...(flags.blue ? [{ binding: PT_BLUENOISE_BINDING, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float' as const, viewDimension: '2d' as const } }] : []),
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [walkLayout, sceneLayout, ioLayout] });
    const make = (entryPoint: (typeof PT_RESTIR_ENTRIES)[number]) => rtCreateComputePipelineAsync(device, { label: `${label} ${entryPoint}`, layout, code, map, entryPoint });
    const [gbuf, init, temporal, spatial] = await Promise.all(PT_RESTIR_ENTRIES.map(make));
    const pipelines = { ptRestirGbuf: gbuf!, ptRestirInit: init!, ptRestirTemporal: temporal!, ptRestirSpatial: spatial! };
    return new PtRestirPasses(device, resolveRestirOptions(options.integrator?.restir ?? {}), pipelines, ioLayout, code.length, { ...options.integrator, ...(options.textures ? { textures: true } : {}) }, flags);
  }

  /** The textures the passes read next to the shade kernel (call before `encodeFrame` whenever they change; the megakernel does). */
  setInputs(inputs: PtRestirInputs): void {
    if (inputs.blueNoise !== this.inputs.blueNoise || inputs.vbuffer !== this.inputs.vbuffer || inputs.layerCorrection !== this.inputs.layerCorrection) this.ioGroup = null;
    this.inputs = inputs;
  }

  /** (Re)allocates the G-buffer and reservoirs for a film. */
  setFilm(width: number, height: number): void {
    if (width === this.width && height === this.height && this.gbuf) return;
    this.width = width;
    this.height = height;
    this.gbuf?.destroy();
    this.reservoirs?.destroy();
    this.hitBuffer?.destroy();
    this.hitBuffer = null;
    const npx = width * height;
    this.gbuf = this.device.createBuffer({ label: 'pt restir gbuf', size: ptRestirGbufBytes(this.integrator, npx), usage: STORAGE | COPY_DST });
    this.reservoirs = this.device.createBuffer({ label: 'pt restir reservoirs', size: PT_RESTIR_REGIONS * npx * 16, usage: STORAGE | COPY_SRC | COPY_DST });
    if (this.shared) this.hitBuffer = this.device.createBuffer({ label: 'pt restir hit', size: npx * 16, usage: STORAGE | COPY_SRC | COPY_DST });
    this.ioGroup = null;
    this.frame = 0;
  }

  get stats(): PtRestirStats {
    const npx = this.width * this.height;
    const gbufBytes = ptRestirGbufBytes(this.integrator, npx);
    return { bytes: gbufBytes + PT_RESTIR_REGIONS * npx * 16 + (this.shared ? npx * 16 : 0), gbufBytes };
  }

  /** Forget the history (the next frame has no temporal neighbour) and empty the reservoirs. */
  reset(encoder: GPUCommandEncoder): void {
    if (this.reservoirs) encoder.clearBuffer(this.reservoirs);
    this.frame = 0;
  }

  /** Region index of the final reservoirs of the frame encoded last (what the shade pass reads); the next `encodeFrame` flips it. */
  finalRegion(frame: number): number {
    return (frame & 1) === 0 ? PT_RESTIR_REGION.final0 : PT_RESTIR_REGION.final1;
  }

  private record(passParams: { frame: number; seed: number; camera: PtCameraBasis; prev: PtCameraBasis | null; history: boolean; gbufCur: number; gbufPrev: number; regIn: number; regOut: number; regPrev: number; coneScale: number; jitter: readonly [number, number] }): ArrayBuffer {
    const s = this.settings;
    const o = this.overrides;
    const cand = Math.max(1, o.candidates ?? s.candidates);
    const buffer = new ArrayBuffer(PT_RESTIR_PARAMS_BYTES);
    const u = new Uint32Array(buffer);
    const f = new Float32Array(buffer);
    const flags = (passParams.history ? PT_RESTIR_FLAG.history : 0) | (s.bias === 'unbiased' ? PT_RESTIR_FLAG.unbiased : 0) | (s.initialVisibility ? PT_RESTIR_FLAG.initialVisibility : 0) | (s.neighbours === 'random' ? PT_RESTIR_FLAG.randomNeighbours : 0) | (s.temporalPick === 'random' ? PT_RESTIR_FLAG.randomTemporal : 0);
    u.set([this.width, this.height, passParams.frame >>> 0, passParams.seed >>> 0], 0);
    u.set([cand, o.spatial ?? s.spatial, flags, 0], 4);
    f.set([s.spatialRadius, s.normalCos, s.depthRel, s.historyCap * cand], 8);
    u.set([passParams.gbufCur, passParams.gbufPrev, passParams.regIn, passParams.regOut], 12);
    u.set([passParams.regPrev, 0, 0, 0], 16);
    new Float32Array(buffer, 68, 1).set([passParams.coneScale]); // reg2.y: textureConeScale (bit pattern read with bitcast)
    const c = passParams.camera;
    const p = passParams.prev ?? c;
    // The w lanes of curOrigin / curRight carry the V-buffer frame's sub-pixel position (0.5 + offset), as in the megakernel's params.
    f.set([...c.origin, 0.5 + passParams.jitter[0], ...c.forward, 0, ...c.right, 0.5 + passParams.jitter[1], ...c.up, 0, ...p.origin, 0, ...p.forward, 0, ...p.right, 0, ...p.up, 0], 20);
    return buffer;
  }

  /**
   * Records one frame's passes into `pass` (a compute pass; the caller sets its own pipeline and bind groups again afterwards). Returns the
   * region of the final reservoirs for the shade pass. `walk` / `scene` are the megakernel's bind groups.
   */
  encodeFrame(pass: GPUComputePassEncoder, walk: GPUBindGroup, scene: GPUBindGroup, frameInfo: PtRestirFrame): number {
    if (!this.gbuf || !this.reservoirs) throw new Error('PtRestirPasses: setFilm() first');
    if (this.wantBlue && !this.inputs.blueNoise) throw new Error('PtRestirPasses: sampler bluenoise needs the blue-noise texture view (setInputs)');
    if (this.wantVbuffer && !this.inputs.vbuffer) throw new Error("PtRestirPasses: primary 'vbuffer' needs the V-buffer view (setInputs)");
    if (this.wantCorr && !this.inputs.layerCorrection) throw new Error('PtRestirPasses: the layered correction texture view is missing (setInputs)');
    this.ioGroup ??= this.device.createBindGroup({
      label: 'pt restir io',
      layout: this.ioLayout,
      entries: [
        { binding: PT_RESTIR_BINDING.params, resource: { buffer: this.paramsBuffer, offset: 0, size: PT_RESTIR_PARAMS_BYTES } },
        { binding: PT_RESTIR_BINDING.gbuf, resource: { buffer: this.gbuf } },
        { binding: PT_RESTIR_BINDING.reservoirs, resource: { buffer: this.reservoirs } },
        ...(this.hitBuffer ? [{ binding: PT_RESTIR_BINDING.hit, resource: { buffer: this.hitBuffer } }] : []),
        ...(this.wantVbuffer && this.inputs.vbuffer ? [{ binding: PT_VBUFFER_BINDING, resource: this.inputs.vbuffer }] : []),
        ...(this.wantCorr && this.inputs.layerCorrection ? [{ binding: PT_LAYER_CORR_BINDING, resource: this.inputs.layerCorrection }] : []),
        ...(this.wantBlue && this.inputs.blueNoise ? [{ binding: PT_BLUENOISE_BINDING, resource: this.inputs.blueNoise }] : []),
      ],
    });
    const s = this.settings;
    const temporal = this.overrides.temporal ?? s.temporal;
    const spatial = this.overrides.spatial ?? s.spatial;
    const basis = ptCameraBasis(frameInfo.camera, this.width, this.height);
    const prevBasis = frameInfo.prev ? ptCameraBasis(frameInfo.prev, this.width, this.height) : null;
    const parity = this.frame & 1;
    const history = temporal && prevBasis !== null && this.frame > 0;
    const finalCur = parity === 0 ? PT_RESTIR_REGION.final0 : PT_RESTIR_REGION.final1;
    const finalPrev = parity === 0 ? PT_RESTIR_REGION.final1 : PT_RESTIR_REGION.final0;
    const common = { frame: frameInfo.frame, seed: frameInfo.seed, camera: basis, prev: prevBasis, history, gbufCur: parity, gbufPrev: parity ^ 1, regPrev: finalPrev, coneScale: frameInfo.coneScale, jitter: frameInfo.jitter };
    // Wiring: init -> A; temporal A (+ previous final) -> B or final; spatial (A or B) -> final.
    const afterInit: number = temporal || spatial > 0 ? PT_RESTIR_REGION.init : finalCur;
    const afterTemporal: number = spatial > 0 ? PT_RESTIR_REGION.temporal : finalCur;
    const steps: { entry: (typeof PT_RESTIR_ENTRIES)[number]; regIn: number; regOut: number }[] = [
      { entry: 'ptRestirGbuf', regIn: 0, regOut: 0 },
      { entry: 'ptRestirInit', regIn: 0, regOut: afterInit },
    ];
    let cur = afterInit;
    if (temporal) {
      steps.push({ entry: 'ptRestirTemporal', regIn: cur, regOut: afterTemporal });
      cur = afterTemporal;
    }
    if (spatial > 0) steps.push({ entry: 'ptRestirSpatial', regIn: cur, regOut: finalCur });
    const gx = Math.ceil(this.width / PT_WORKGROUP[0]);
    const gy = Math.ceil(this.height / PT_WORKGROUP[1]);
    for (const step of steps) {
      const offset = (this.slot % PT_RESTIR_PARAM_SLOTS) * this.slotBytes;
      this.slot++;
      this.device.queue.writeBuffer(this.paramsBuffer, offset, this.record({ ...common, regIn: step.regIn, regOut: step.regOut }));
      pass.setPipeline(this.pipelines[step.entry]);
      pass.setBindGroup(0, walk);
      pass.setBindGroup(1, scene);
      pass.setBindGroup(2, this.ioGroup, [offset]);
      pass.dispatchWorkgroups(gx, gy);
    }
    this.frame++;
    return finalCur;
  }

  /** `sharedHit`: the frame's primary hits as V-buffer texels (4 words per pixel), read back. Debug / check pages only, never in a timed region. */
  async readHits(): Promise<Uint32Array> {
    if (!this.hitBuffer) throw new Error('PtRestirPasses.readHits: the kernel was built without restir.sharedHit');
    const size = this.width * this.height * 16;
    const staging = this.device.createBuffer({ label: 'pt restir hit readback', size, usage: 0x1 | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'pt restir hit readback' });
    encoder.copyBufferToBuffer(this.hitBuffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1);
    const words = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return words;
  }

  destroy(): void {
    this.gbuf?.destroy();
    this.reservoirs?.destroy();
    this.hitBuffer?.destroy();
    this.paramsBuffer.destroy();
    this.gbuf = null;
    this.reservoirs = null;
    this.hitBuffer = null;
  }
}

export { PT_RESTIR_REGION };
