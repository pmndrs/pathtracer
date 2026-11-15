/**
 * Host side of ReSTIR DI (P4): the four pre-passes (G-buffer, initial candidates, temporal, spatial) that `PtMegakernel` runs before each
 * shade dispatch when `integrator.restir` is set. One `encodeFrame` = one frame = one sample of the progressive accumulation.
 *
 * Previous-frame reprojection uses the camera of the last encoded frame (analytic: a static scene, a moving camera). The real-time mode
 * (P4.1) will bring motion vectors and a previous view-projection of its own; `PtRestirPasses.setPreviousCamera` is the hook to replace
 * this locally tracked state with that one (DIVERGES V130).
 */
import { DEFAULT_SCENE_PACKING, rtCreateComputePipelineAsync, rtModuleWgsl, rtPackSceneAccess, rtTransformWgsl, type Caps, type RtWgslBuild } from '@pmndrs/raytracer';
import { ptCameraBasis, type PtCameraBasis, type PtLookAtCamera } from '../integrator/camera.ts';
import { PT_WORKGROUP, resolveIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { ptRaytracerModuleOptions, ptShadingModulesWgsl, type PtMegakernelOptions } from '../integrator/megakernel.ts';
import { PT_RESTIR_BINDING, PT_RESTIR_ENTRIES, PT_RESTIR_FLAG, PT_RESTIR_GBUF_VEC4, PT_RESTIR_PARAMS_BYTES, PT_RESTIR_REGION, PT_RESTIR_REGIONS, ptRestirWgsl, resolveRestirOptions, type ResolvedRestirOptions } from './restir-wgsl.ts';

const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const UNIFORM = 0x40;
const COMPUTE = 0x4;

/** Records in the params ring (4 passes per frame): enough for 256 frames outstanding. */
export const PT_RESTIR_PARAM_SLOTS = 1024;

type BuildOptions = Pick<PtMegakernelOptions, 'format' | 'layout' | 'integrator' | 'textures' | 'scenePacking' | 'minify' | 'prune' | 'shortenNames'>;

/** Why this kernel cannot run ReSTIR DI (null = it can). */
export function ptRestirUnsupported(options: Pick<PtMegakernelOptions, 'integrator' | 'textures'>): string | null {
  const o = resolveIntegratorOptions({ ...options.integrator, ...(options.textures ? { textures: true } : {}) });
  if (!o.restir) return 'integrator.restir is not set';
  if (!o.nee) return 'ReSTIR DI replaces next-event estimation at the first vertex: integrator.nee must be on';
  if (o.textures) return 'textured scenes are not supported yet (the passes decode the material without uvs)';
  if (o.materials) return 'OpenPBR mode (integrator.materials) is not supported yet: the target function uses the legacy material adapter';
  if (o.primary === 'vbuffer') return "primary 'vbuffer' is not supported yet (the G-buffer pass traces its own camera ray)";
  if (o.hooks) return 'path hooks are not supported with ReSTIR yet';
  return null;
}

/** The module of the four passes (what `PtRestirPasses.create` compiles). */
export function ptRestirBuild(caps: Caps, options: BuildOptions): RtWgslBuild {
  const why = ptRestirUnsupported(options);
  if (why) throw new Error(`ptRestirBuild: ${why}`);
  const { integratorOptions, parts } = ptShadingModulesWgsl(options);
  const raytracer = rtModuleWgsl(caps, ptRaytracerModuleOptions(options));
  const ours = [...parts, ptRestirWgsl(integratorOptions)].join('\n');
  const packed = (options.scenePacking ?? DEFAULT_SCENE_PACKING) === 'packed';
  const { code, map } = rtTransformWgsl(`${raytracer}\n${packed ? rtPackSceneAccess(ours) : ours}`, { prune: options.prune, shorten: options.shortenNames, minify: options.minify, keep: [...PT_RESTIR_ENTRIES] });
  return { code, map };
}

export interface PtRestirStats {
  /** Bytes of G-buffer + reservoirs at the current film size. */
  bytes: number;
}

export class PtRestirPasses {
  readonly settings: ResolvedRestirOptions;
  readonly wgslBytes: number;
  private readonly device: GPUDevice;
  private readonly pipelines: Record<(typeof PT_RESTIR_ENTRIES)[number], GPUComputePipeline>;
  private readonly ioLayout: GPUBindGroupLayout;
  private readonly paramsBuffer: GPUBuffer;
  private readonly slotBytes: number;
  private slot = 0;
  private ioGroup: GPUBindGroup | null = null;
  private gbuf: GPUBuffer | null = null;
  /** The reservoirs (4 regions, `PT_RESTIR_REGION`). The shade pass binds the whole buffer and picks its region through `ptParams.limits.w`. */
  reservoirs: GPUBuffer | null = null;
  private width = 0;
  private height = 0;
  private prev: PtCameraBasis | null = null;
  /** Frames encoded since `reset` (parity picks the ping-pong halves). */
  frame = 0;
  /** Per-pass switches the check pages flip without rebuilding (all runtime). */
  overrides: Partial<Pick<ResolvedRestirOptions, 'temporal' | 'spatial' | 'candidates'>> = {};

  private constructor(device: GPUDevice, settings: ResolvedRestirOptions, pipelines: PtRestirPasses['pipelines'], ioLayout: GPUBindGroupLayout, wgslBytes: number) {
    this.device = device;
    this.settings = settings;
    this.pipelines = pipelines;
    this.ioLayout = ioLayout;
    this.wgslBytes = wgslBytes;
    this.slotBytes = Math.max(PT_RESTIR_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
    this.paramsBuffer = device.createBuffer({ label: 'pt restir params', size: this.slotBytes * PT_RESTIR_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
  }

  static async create(device: GPUDevice, caps: Caps, options: BuildOptions & { label?: string }, walkLayout: GPUBindGroupLayout, sceneLayout: GPUBindGroupLayout): Promise<PtRestirPasses> {
    const label = `${options.label ?? 'pt'} restir`;
    const { code, map } = ptRestirBuild(caps, options);
    const ioLayout = device.createBindGroupLayout({
      label: `${label} io`,
      entries: [
        { binding: PT_RESTIR_BINDING.params, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_RESTIR_PARAMS_BYTES } },
        { binding: PT_RESTIR_BINDING.gbuf, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: PT_RESTIR_BINDING.reservoirs, visibility: COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [walkLayout, sceneLayout, ioLayout] });
    const make = (entryPoint: (typeof PT_RESTIR_ENTRIES)[number]) => rtCreateComputePipelineAsync(device, { label: `${label} ${entryPoint}`, layout, code, map, entryPoint });
    const [gbuf, init, temporal, spatial] = await Promise.all(PT_RESTIR_ENTRIES.map(make));
    const pipelines = { ptRestirGbuf: gbuf!, ptRestirInit: init!, ptRestirTemporal: temporal!, ptRestirSpatial: spatial! };
    return new PtRestirPasses(device, resolveRestirOptions(options.integrator?.restir ?? {}), pipelines, ioLayout, code.length);
  }

  /** (Re)allocates the G-buffer and reservoirs for a film. */
  setFilm(width: number, height: number): void {
    if (width === this.width && height === this.height && this.gbuf) return;
    this.width = width;
    this.height = height;
    this.gbuf?.destroy();
    this.reservoirs?.destroy();
    const npx = width * height;
    this.gbuf = this.device.createBuffer({ label: 'pt restir gbuf', size: 2 * npx * PT_RESTIR_GBUF_VEC4 * 16, usage: STORAGE | COPY_DST });
    this.reservoirs = this.device.createBuffer({ label: 'pt restir reservoirs', size: PT_RESTIR_REGIONS * npx * 16, usage: STORAGE | COPY_SRC | COPY_DST });
    this.ioGroup = null;
    this.frame = 0;
    this.prev = null;
  }

  get stats(): PtRestirStats {
    const npx = this.width * this.height;
    return { bytes: 2 * npx * PT_RESTIR_GBUF_VEC4 * 16 + PT_RESTIR_REGIONS * npx * 16 };
  }

  /** Forget the history (the next frame has no temporal neighbour) and empty the reservoirs. */
  reset(encoder: GPUCommandEncoder): void {
    if (this.reservoirs) encoder.clearBuffer(this.reservoirs);
    this.frame = 0;
    this.prev = null;
  }

  /** Replaces the locally tracked previous camera (P4.1 hook: the real-time mode's previous view-projection). `null`: no history for the next frame. */
  setPreviousCamera(basis: PtCameraBasis | null): void {
    this.prev = basis;
  }

  /** Region index of the final reservoirs of the frame encoded last (what the shade pass reads); the next `encodeFrame` flips it. */
  finalRegion(frame: number): number {
    return (frame & 1) === 0 ? PT_RESTIR_REGION.final0 : PT_RESTIR_REGION.final1;
  }

  private record(passParams: { frame: number; seed: number; camera: PtCameraBasis; prev: PtCameraBasis | null; history: boolean; gbufCur: number; gbufPrev: number; regIn: number; regOut: number; regPrev: number }): ArrayBuffer {
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
    const c = passParams.camera;
    const p = passParams.prev ?? c;
    f.set([...c.origin, 0, ...c.forward, 0, ...c.right, 0, ...c.up, 0, ...p.origin, 0, ...p.forward, 0, ...p.right, 0, ...p.up, 0], 20);
    return buffer;
  }

  /**
   * Records one frame's passes into `pass` (a compute pass; the caller sets its own pipeline and bind groups again afterwards). Returns the
   * region of the final reservoirs for the shade pass. `walk` / `scene` are the megakernel's bind groups.
   */
  encodeFrame(pass: GPUComputePassEncoder, walk: GPUBindGroup, scene: GPUBindGroup, camera: PtLookAtCamera, seed: number, frame: number): number {
    if (!this.gbuf || !this.reservoirs) throw new Error('PtRestirPasses: setFilm() first');
    this.ioGroup ??= this.device.createBindGroup({
      label: 'pt restir io',
      layout: this.ioLayout,
      entries: [
        { binding: PT_RESTIR_BINDING.params, resource: { buffer: this.paramsBuffer, offset: 0, size: PT_RESTIR_PARAMS_BYTES } },
        { binding: PT_RESTIR_BINDING.gbuf, resource: { buffer: this.gbuf } },
        { binding: PT_RESTIR_BINDING.reservoirs, resource: { buffer: this.reservoirs } },
      ],
    });
    const s = this.settings;
    const temporal = this.overrides.temporal ?? s.temporal;
    const spatial = this.overrides.spatial ?? s.spatial;
    const basis = ptCameraBasis(camera, this.width, this.height);
    const parity = this.frame & 1;
    const history = temporal && this.prev !== null && this.frame > 0;
    const finalCur = parity === 0 ? PT_RESTIR_REGION.final0 : PT_RESTIR_REGION.final1;
    const finalPrev = parity === 0 ? PT_RESTIR_REGION.final1 : PT_RESTIR_REGION.final0;
    const common = { frame, seed, camera: basis, prev: this.prev, history, gbufCur: parity, gbufPrev: parity ^ 1, regPrev: finalPrev };
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
    this.prev = basis;
    this.frame++;
    return finalCur;
  }

  destroy(): void {
    this.gbuf?.destroy();
    this.reservoirs?.destroy();
    this.paramsBuffer.destroy();
    this.gbuf = null;
    this.reservoirs = null;
  }
}

export { PT_RESTIR_REGION };
