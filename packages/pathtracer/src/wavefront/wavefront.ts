/**
 * Host side of the wavefront path tracer (P3.1), the second scheduler beside `PtMegakernel` (D11: build both, then decide).
 *
 *     const pt = await PtWavefront.create(device, caps, { format, layout: copy.layout, integrator: { sampler: 'owen-sobol' } });
 *     pt.setScene(traceBuffers, sceneCopyBuffers);  pt.setFilm(w, h);  pt.setCamera(cam);  pt.setLimits({ maxDepth: 8, rrStartDepth: 4 });
 *     pt.encode(encoder, 4);                        // 4 sample passes, every kernel of every bounce in one compute pass
 *     const image = await pt.readImage();           // outside any timed region
 *
 * Same surface as the megakernel (create / setScene / setFilm / setCamera / setLimits / setSeed / reset / encode / readImage / destroy).
 * One sample pass = camera + args(INIT) + for each depth `intersect, shade, args, shadow`, with GPU-written indirect dispatch args:
 * no readback and no CPU decision inside the frame. The queue protocol and buffer layout are in `layout.ts`. Slice 1: one sample in
 * flight, one path per pixel, no refill / sorting / compaction (P3.2+), no alpha MASK (`ptwfUnsupported`).
 *
 * The params uniform is a ring of dynamic-offset slots like the megakernel's, with one record per depth iteration
 * (`PTWF_PARAM_SLOTS`; `maxPassesPerEncode()` says how many sample passes fit in one `encode`).
 */
import { rtCreateComputePipelineAsync, rtModuleWgsl, rtSceneBindGroupEntries, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtTransformWgsl, rtWalkEntries, withBindings } from '@pmndrs/raytracer';
import type { Caps, RtWgslBuild, SceneCopyBuffers } from '@pmndrs/raytracer';
import { ptFuzzTableF32 } from '../bsdf/fuzz.ts';
import { ptCameraBasis, type PtLookAtCamera } from '../integrator/camera.ts';
import { PT_PARAMS_BYTES, resolveIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { PT_AOV_VEC4, PT_COUNTER_WORDS, ptDecodeAovs, ptOutputsWgsl, ptSummarizeCounters, type PtAovs, type PtCounters } from '../integrator/outputs-wgsl.ts';
import { ptRaytracerModuleOptions, ptShadingModulesWgsl, type PtImage, type PtLimits, type PtMegakernelOptions } from '../integrator/megakernel.ts';
import { PTWF_ARGS_BYTES, PTWF_ARGS_MAIN, PTWF_ARGS_SHADOW, PTWF_HIT_VEC4, PTWF_INIT, PTWF_PARAM_SLOTS, PTWF_SHADOW_VEC4, ptwfBufferBytes, ptwfRecordsPerSample, ptwfStateBytes } from './layout.ts';
import { PTWF_BINDING, ptwfArgsWgsl, ptwfCameraWgsl, ptwfIntersectWgsl, ptwfShadeWgsl, ptwfShadowWgsl, ptwfUnsupported } from './wavefront-wgsl.ts';

const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const UNIFORM = 0x40;
const INDIRECT = 0x100;
const MAP_READ = 0x1;
const COMPUTE = 0x4;

/** Options are the megakernel's (`format`, `layout`, `integrator`, `textures`, text transforms), so one adapter setting switches schedulers. */
export type PtWavefrontOptions = PtMegakernelOptions;

export const PTWF_KERNELS = ['camera', 'intersect', 'shade', 'shadow', 'args'] as const;
/** Group 2 bindings each kernel declares (`PTWF_BINDING` numbers; 2 = fuzz table, only with `features.fuzz`). Must match the `bindings([...])` of the WGSL. */
const IO_BINDINGS: Record<(typeof PTWF_KERNELS)[number], readonly number[]> = { camera: [0, 1, 3, 6], intersect: [0, 3, 4, 5, 6], shade: [0, 1, 2, 3, 4, 5, 6, 7], shadow: [0, 1, 5, 6, 7], args: [0, 5] };
const USES_WALK: Record<(typeof PTWF_KERNELS)[number], boolean> = { camera: false, intersect: true, shade: false, shadow: true, args: false };
const USES_SCENE: Record<(typeof PTWF_KERNELS)[number], boolean> = { camera: false, intersect: false, shade: true, shadow: false, args: false };
export type PtwfKernel = (typeof PTWF_KERNELS)[number];
export type PtWavefrontBuild = Record<PtwfKernel, RtWgslBuild>;

/**
 * The shade kernel has no ray traversal, so it drops the walk's group 0 declarations (a declared binding must be in its pipeline layout, and
 * walk + scene + wavefront buffers would pass the 10 storage buffers per stage of Apple GPUs). The surface, material and light code does not read them.
 */
function stripWalkBindings(code: string): string {
  return code.replace(/^@group\(0\) @binding\(\d+\)[^\n]*\n/gm, '');
}

/** The five kernels' source and source maps. Throws for options the wavefront does not support (`ptwfUnsupported`). */
export function ptWavefrontBuild(caps: Caps, options: Pick<PtWavefrontOptions, 'format' | 'layout' | 'integrator' | 'textures' | 'minify' | 'prune' | 'shortenNames'>): PtWavefrontBuild {
  const { integratorOptions, parts } = ptShadingModulesWgsl(options);
  const why = ptwfUnsupported(integratorOptions);
  if (why) throw new Error(`PtWavefront: unsupported: ${why}`);
  const t = { prune: options.prune, shorten: options.shortenNames, minify: options.minify };
  const walk = rtModuleWgsl(caps, { ...options.format, rayMasks: false, traceSlot: { group: 0, firstBinding: 0 } });
  const full = rtModuleWgsl(caps, ptRaytracerModuleOptions(options));
  // Optional outputs (hooks, counters, AOVs; `integrator/outputs-wgsl.ts`) join the shade kernel only: it is the one that sees every vertex.
  const outputs = ptOutputsWgsl(resolveIntegratorOptions(integratorOptions), PTWF_BINDING.counters);
  const sampler = parts.slice(1, 2).concat(parts.slice(-1)).join('\n'); // Owen-Sobol + the sampler's ptSample2D
  return {
    camera: rtTransformWgsl(`${sampler}\n${ptwfCameraWgsl(integratorOptions)}`, { ...t, keep: ['ptwfCamera'] }),
    intersect: rtTransformWgsl(`${walk}\n${ptwfIntersectWgsl(integratorOptions)}`, { ...t, keep: ['ptwfIntersect'] }),
    shade: rtTransformWgsl(`${stripWalkBindings(full)}\n${parts.join('\n')}\n${outputs ? `${outputs}\n` : ''}${ptwfShadeWgsl(integratorOptions)}`, { ...t, keep: ['ptwfShade'] }),
    shadow: rtTransformWgsl(`${walk}\n${ptwfShadowWgsl(integratorOptions)}`, { ...t, keep: ['ptwfShadow'] }),
    args: rtTransformWgsl(ptwfArgsWgsl(), { ...t, keep: ['ptwfArgsMain'] }),
  };
}

/** Bytes of each kernel's WGSL. */
export function ptWavefrontKernelBytes(build: PtWavefrontBuild): Record<PtwfKernel, number> {
  return Object.fromEntries(PTWF_KERNELS.map((k) => [k, build[k].code.length])) as Record<PtwfKernel, number>;
}

interface PtwfLayouts {
  hooks: GPUBindGroupLayout | null;
  empty: GPUBindGroupLayout;
  walk: GPUBindGroupLayout;
  scene: GPUBindGroupLayout;
  io: Record<PtwfKernel, GPUBindGroupLayout>;
  args: GPUBindGroupLayout;
}

/** Group 2 bindings of a kernel for this build: the fuzz table (2) only with fuzz, the debug counters (8) only with counters (shade kernel). */
function ioBindings(kernel: PtwfKernel, fuzz: boolean, counters: boolean): number[] {
  return [...IO_BINDINGS[kernel], ...(kernel === 'shade' && counters ? [PTWF_BINDING.counters] : [])].filter((binding) => binding !== 2 || fuzz);
}

type TraceScene = Parameters<typeof rtSceneBindGroupEntries>[0];

export class PtWavefront {
  readonly device: GPUDevice;
  readonly options: PtWavefrontOptions;
  /** Largest kernel's WGSL bytes (the D26 budget applies per kernel). */
  readonly wgslBytes: number;
  readonly kernelBytes: Record<PtwfKernel, number>;
  private readonly pipelines: Record<PtwfKernel, GPUComputePipeline>;
  private readonly layouts: PtwfLayouts;
  private readonly paramsBuffer: GPUBuffer;
  private readonly argsBuffer: GPUBuffer;
  private readonly fuzzTable: GPUBuffer | null;
  private readonly slotBytes: number;
  private readonly medium: boolean;
  private hookGroup: GPUBindGroup | null = null;
  private counterBuffer: GPUBuffer | null = null;
  private slot = 0;
  private walkGroup: GPUBindGroup | null = null;
  private sceneGroup: GPUBindGroup | null = null;
  private ioGroups: Record<PtwfKernel, GPUBindGroup> | null = null;
  private emptyGroup: GPUBindGroup | null = null;
  private argsGroup: GPUBindGroup | null = null;
  private buffers: { accum: GPUBuffer; state: GPUBuffer; hit: GPUBuffer; count: GPUBuffer; queue: GPUBuffer; shadow: GPUBuffer } | null = null;
  private width = 0;
  private height = 0;
  private camera: PtLookAtCamera | null = null;
  private limits: PtLimits = { maxDepth: 5 };
  private seed = 0;
  /** Sample passes encoded since the last `reset`. */
  samples = 0;

  private constructor(device: GPUDevice, options: PtWavefrontOptions, pipelines: Record<PtwfKernel, GPUComputePipeline>, layouts: PtwfLayouts, bytes: Record<PtwfKernel, number>, fuzzTable: GPUBuffer | null, medium: boolean) {
    this.device = device;
    this.options = options;
    this.pipelines = pipelines;
    this.layouts = layouts;
    this.kernelBytes = bytes;
    this.wgslBytes = Math.max(...Object.values(bytes));
    this.fuzzTable = fuzzTable;
    this.medium = medium;
    this.slotBytes = Math.max(PT_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
    this.paramsBuffer = device.createBuffer({ label: 'ptwf params', size: this.slotBytes * PTWF_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
    this.argsBuffer = device.createBuffer({ label: 'ptwf args', size: PTWF_ARGS_BYTES, usage: STORAGE | INDIRECT });
  }

  static async create(device: GPUDevice, caps: Caps, options: PtWavefrontOptions): Promise<PtWavefront> {
    const label = options.label ?? 'pt wavefront';
    const build = ptWavefrontBuild(caps, options);
    const features = options.integrator?.materials?.features;
    device.pushErrorScope('validation');
    // One layout per kernel with only the buffers it declares: a shared layout would bind 3 (walk) + 2 (scene) + 7 buffers and pass the
    // 10-storage-buffers-per-stage limit of Apple GPUs (Chrome/Metal). A kernel that has no walk (or no scene) gets an empty group there.
    const empty = device.createBindGroupLayout({ label: `${label} empty`, entries: [] });
    const walk = device.createBindGroupLayout({ label: `${label} walk`, entries: withBindings(rtWalkEntries(options.format)) });
    const scene = device.createBindGroupLayout({ label: `${label} scene`, entries: rtSceneCopyLayoutEntries(0, options.textures) });
    const ioEntries = (kernel: PtwfKernel): GPUBindGroupLayoutEntry[] =>
      ioBindings(kernel, !!features?.fuzz, !!options.integrator?.counters).map((binding) =>
        binding === 0 ? { binding, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_PARAMS_BYTES } } : { binding, visibility: COMPUTE, buffer: { type: binding === 2 ? 'read-only-storage' : 'storage' } },
      ) as GPUBindGroupLayoutEntry[];
    const io = Object.fromEntries(PTWF_KERNELS.map((k) => [k, device.createBindGroupLayout({ label: `${label} io ${k}`, entries: ioEntries(k) })])) as Record<PtwfKernel, GPUBindGroupLayout>;
    const argsLayout = device.createBindGroupLayout({ label: `${label} args`, entries: [{ binding: 0, visibility: COMPUTE, buffer: { type: 'storage' } }] });
    const hooks = options.integrator?.hooks ? device.createBindGroupLayout({ label: `${label} hooks`, entries: options.integrator.hooks.bindings ?? [] }) : null;
    const layouts = { hooks, empty, walk, scene, io, args: argsLayout };
    const pipelineLayout = (k: PtwfKernel): GPUPipelineLayout =>
      device.createPipelineLayout({ label: `${label} ${k}`, bindGroupLayouts: [USES_WALK[k] ? walk : empty, USES_SCENE[k] ? scene : empty, io[k], ...(k === 'args' ? [argsLayout] : []), ...(k === 'shade' && hooks ? [hooks] : [])] });
    const pipelineLayouts = Object.fromEntries(PTWF_KERNELS.map((k) => [k, pipelineLayout(k)])) as Record<PtwfKernel, GPUPipelineLayout>;
    const layoutError = await device.popErrorScope();
    if (layoutError) throw new Error(`PtWavefront: invalid bind group / pipeline layout: ${layoutError.message}`);
    const entry: Record<PtwfKernel, string> = { camera: 'ptwfCamera', intersect: 'ptwfIntersect', shade: 'ptwfShade', shadow: 'ptwfShadow', args: 'ptwfArgsMain' };
    const made = await Promise.all(PTWF_KERNELS.map((k) => rtCreateComputePipelineAsync(device, { label: `${label} ${k}`, layout: pipelineLayouts[k], code: build[k].code, map: build[k].map, entryPoint: entry[k], sourceMap: options.sourceMap })));
    const pipelines = Object.fromEntries(PTWF_KERNELS.map((k, i) => [k, made[i]])) as Record<PtwfKernel, GPUComputePipeline>;
    let fuzzTable: GPUBuffer | null = null;
    if (features?.fuzz) {
      const table = ptFuzzTableF32();
      fuzzTable = device.createBuffer({ label: `${label} fuzz ltc`, size: table.byteLength, usage: STORAGE | COPY_DST });
      device.queue.writeBuffer(fuzzTable, 0, table);
    }
    return new PtWavefront(device, options, pipelines, layouts, ptWavefrontKernelBytes(build), fuzzTable, !!features?.medium);
  }

  setScene(trace: TraceScene, scene: SceneCopyBuffers): void {
    if (!!this.options.textures !== !!scene.textures) throw new Error(`PtWavefront.setScene: the kernel was built ${this.options.textures ? 'with' : 'without'} textures but the scene copy has ${scene.textures ? 'some' : 'none'}`);
    this.walkGroup = this.device.createBindGroup({ label: 'ptwf walk', layout: this.layouts.walk, entries: rtSceneBindGroupEntries(trace) });
    this.sceneGroup = this.device.createBindGroup({ label: 'ptwf scene', layout: this.layouts.scene, entries: rtSceneCopyBindGroupEntries(scene) });
  }

  /** Bytes of path state per path, and of all per-path buffers, for the current film (the spec budget is 96 B of state). */
  memory(): { stateBytesPerPath: number; totalBytes: number } {
    const o = { medium: this.medium, textures: !!this.options.textures };
    return { stateBytesPerPath: ptwfStateBytes(o), totalBytes: ptwfBufferBytes(this.width * this.height, o).total + this.width * this.height * 16 };
  }

  setFilm(width: number, height: number): void {
    if (width === this.width && height === this.height && this.buffers) return;
    const paths = width * height;
    const o = { medium: this.medium, textures: !!this.options.textures };
    const bytes = ptwfBufferBytes(paths, o);
    const max = this.device.limits.maxStorageBufferBindingSize;
    if (bytes.state > max || paths * PTWF_HIT_VEC4 * 16 > max) throw new RangeError(`PtWavefront.setFilm: ${width}x${height} needs ${(bytes.state / 2 ** 20).toFixed(0)} MiB of path state, the device binds at most ${(max / 2 ** 20).toFixed(0)} MiB per storage buffer`);
    this.width = width;
    this.height = height;
    this.destroyBuffers();
    const make = (label: string, size: number, usage = STORAGE) => this.device.createBuffer({ label: `ptwf ${label}`, size, usage });
    this.buffers = {
      accum: make('accum', paths * 16 * (this.options.integrator?.aovs ? 1 + PT_AOV_VEC4 : 1), STORAGE | COPY_SRC | COPY_DST),
      state: make('state', bytes.state),
      hit: make('hit', bytes.hit),
      count: make('counters', bytes.counters),
      queue: make('queue', bytes.queue),
      shadow: make('shadow rays', paths * PTWF_SHADOW_VEC4 * 16),
    };
    this.counterBuffer = this.options.integrator?.counters ? make('debug counters', paths * PT_COUNTER_WORDS * 4, STORAGE | COPY_SRC | COPY_DST) : null;
    this.ioGroups = null;
    this.samples = 0;
  }

  /** Layout of the hooks' bind group (`@group(3)` of the shade kernel), or null without hooks. */
  get hookBindGroupLayout(): GPUBindGroupLayout | null {
    return this.layouts.hooks;
  }

  /** The hooks' bind group (group 3 of the shade kernel). Not needed for a hook without bindings. */
  setHookBindGroup(group: GPUBindGroup): void {
    if (!this.layouts.hooks) throw new Error('PtWavefront.setHookBindGroup: the kernels were built without hooks');
    this.hookGroup = group;
  }

  /** True when the kernels write debug counters: timings are not results. */
  get instrumented(): boolean {
    return !!this.options.integrator?.counters;
  }

  setCamera(camera: PtLookAtCamera): void {
    this.camera = camera;
  }

  setLimits(limits: PtLimits): void {
    this.limits = limits;
  }

  setSeed(seed: number): void {
    this.seed = seed >>> 0;
  }

  reset(encoder: GPUCommandEncoder): void {
    if (this.buffers) encoder.clearBuffer(this.buffers.accum);
    if (this.counterBuffer) encoder.clearBuffer(this.counterBuffer);
    this.samples = 0;
  }

  /** How many sample passes one `encode` call can take with the current `maxDepth` (the params ring must not wrap inside one encoder). */
  maxPassesPerEncode(): number {
    return Math.max(1, Math.floor(PTWF_PARAM_SLOTS / ptwfRecordsPerSample(this.limits.maxDepth)));
  }

  private paramsRecord(sampleIndex: number, iter: number): ArrayBuffer {
    if (!this.camera) throw new Error('PtWavefront: setCamera() first');
    const basis = ptCameraBasis(this.camera, this.width, this.height);
    const buffer = new ArrayBuffer(PT_PARAMS_BYTES);
    const u = new Uint32Array(buffer);
    const f = new Float32Array(buffer);
    u.set([this.width, this.height, sampleIndex >>> 0, this.seed, this.limits.maxDepth >>> 0, (this.limits.rrStartDepth ?? 0) >>> 0, 0, iter >>> 0]);
    f[6] = this.limits.textureConeScale ?? 1;
    f.set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0], 8);
    return buffer;
  }

  /** Records `passes` sample passes into one compute pass of `encoder`. At most `maxPassesPerEncode()`. */
  encode(encoder: GPUCommandEncoder, passes: number, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.walkGroup || !this.sceneGroup) throw new Error('PtWavefront: setScene() first');
    const b = this.buffers;
    if (!b) throw new Error('PtWavefront: setFilm() first');
    if (passes > this.maxPassesPerEncode()) throw new RangeError(`PtWavefront.encode: at most ${this.maxPassesPerEncode()} passes per call at maxDepth ${this.limits.maxDepth}`);
    this.ioGroups ??= this.makeIoGroups(b);
    this.argsGroup ??= this.device.createBindGroup({ label: 'ptwf args', layout: this.layouts.args, entries: [{ binding: 0, resource: { buffer: this.argsBuffer } }] });
    this.emptyGroup ??= this.device.createBindGroup({ label: 'ptwf empty', layout: this.layouts.empty, entries: [] });
    const walkGroup = this.walkGroup;
    const sceneGroup = this.sceneGroup;
    const ioGroups = this.ioGroups;
    const argsGroup = this.argsGroup;
    const emptyGroup = this.emptyGroup;
    if (this.layouts.hooks && !this.hookGroup) {
      if ((this.options.integrator?.hooks?.bindings ?? []).length > 0) throw new Error('PtWavefront: setHookBindGroup() first (the hooks declare bindings)');
      this.hookGroup = this.device.createBindGroup({ label: 'ptwf hooks (empty)', layout: this.layouts.hooks, entries: [] });
    }
    const hookGroup = this.hookGroup;
    const maxDepth = this.limits.maxDepth;
    const pass = encoder.beginComputePass({ label: 'ptwf samples', timestampWrites });
    const gx = Math.ceil(this.width / 8);
    const gy = Math.ceil(this.height / 8);
    let offset = 0;
    const record = (sampleIndex: number, iter: number): void => {
      offset = (this.slot % PTWF_PARAM_SLOTS) * this.slotBytes;
      this.slot++;
      this.device.queue.writeBuffer(this.paramsBuffer, offset, this.paramsRecord(sampleIndex, iter));
    };
    // Every pipeline has its own layout, so a pipeline switch invalidates the groups: set them all again.
    const use = (kernel: PtwfKernel): void => {
      pass.setPipeline(this.pipelines[kernel]);
      pass.setBindGroup(0, USES_WALK[kernel] ? walkGroup : emptyGroup);
      pass.setBindGroup(1, USES_SCENE[kernel] ? sceneGroup : emptyGroup);
      pass.setBindGroup(2, ioGroups[kernel], [offset]);
      if (kernel === 'args') pass.setBindGroup(3, argsGroup);
      else if (kernel === 'shade' && hookGroup) pass.setBindGroup(3, hookGroup);
    };
    for (let i = 0; i < passes; i++) {
      record(this.samples, PTWF_INIT);
      use('camera');
      pass.dispatchWorkgroups(gx, gy);
      use('args');
      pass.dispatchWorkgroups(1);
      for (let d = 0; d <= maxDepth; d++) {
        record(this.samples, d);
        use('intersect');
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        use('shade');
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_MAIN);
        if (d === maxDepth) break; // depth maxDepth ends every path before NEE and the bounce: nothing left to trace
        use('args');
        pass.dispatchWorkgroups(1);
        use('shadow');
        pass.dispatchWorkgroupsIndirect(this.argsBuffer, PTWF_ARGS_SHADOW);
      }
      this.samples++;
    }
    pass.end();
  }

  private makeIoGroups(b: NonNullable<PtWavefront['buffers']>): Record<PtwfKernel, GPUBindGroup> {
    const resources: Record<number, GPUBindingResource | undefined> = {
      0: { buffer: this.paramsBuffer, offset: 0, size: PT_PARAMS_BYTES },
      1: { buffer: b.accum },
      2: this.fuzzTable ? { buffer: this.fuzzTable } : undefined,
      3: { buffer: b.state },
      4: { buffer: b.hit },
      5: { buffer: b.count },
      6: { buffer: b.queue },
      7: { buffer: b.shadow },
      8: this.counterBuffer ? { buffer: this.counterBuffer } : undefined,
    };
    return Object.fromEntries(
      PTWF_KERNELS.map((k) => [k, this.device.createBindGroup({ label: `ptwf io ${k}`, layout: this.layouts.io[k], entries: ioBindings(k, !!this.fuzzTable, !!this.counterBuffer).filter((binding) => resources[binding]).map((binding) => ({ binding, resource: resources[binding] as GPUBindingResource })) })]),
    ) as Record<PtwfKernel, GPUBindGroup>;
  }

  async readImage(): Promise<PtImage> {
    if (!this.buffers) throw new Error('PtWavefront: nothing rendered');
    const size = this.width * this.height * 16;
    const staging = this.device.createBuffer({ label: 'ptwf readback', size, usage: MAP_READ | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'ptwf readback' });
    encoder.copyBufferToBuffer(this.buffers.accum, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1 /* GPUMapMode.READ */);
    const sums = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const n = Math.max(1, this.samples);
    const data = new Float32Array(sums.length);
    for (let i = 0; i < sums.length; i += 4) {
      data[i] = (sums[i] as number) / n;
      data[i + 1] = (sums[i + 1] as number) / n;
      data[i + 2] = (sums[i + 2] as number) / n;
      data[i + 3] = 1;
    }
    return { width: this.width, height: this.height, data, samples: this.samples };
  }

  private async readBuffer(buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const staging = this.device.createBuffer({ label: 'ptwf readback', size, usage: MAP_READ | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'ptwf readback' });
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1 /* GPUMapMode.READ */);
    const data = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return data;
  }

  /** The AOVs (`integrator.aovs`), averaged over the samples so far. Outside any timed region. */
  async readAovs(): Promise<PtAovs> {
    if (!this.buffers) throw new Error('PtWavefront: nothing rendered');
    if (!this.options.integrator?.aovs) throw new Error('PtWavefront.readAovs: the kernels were built without integrator.aovs');
    const sums = new Float32Array(await this.readBuffer(this.buffers.accum, this.width * this.height * 16 * (1 + PT_AOV_VEC4)));
    return ptDecodeAovs(sums, this.width, this.height, this.samples);
  }

  /** The debug counters (`integrator.counters`) summed over the samples so far. Outside any timed region. */
  async readCounters(): Promise<PtCounters> {
    if (!this.counterBuffer) throw new Error('PtWavefront.readCounters: the kernels were built without integrator.counters');
    const raw = new Uint32Array(await this.readBuffer(this.counterBuffer, this.width * this.height * PT_COUNTER_WORDS * 4));
    return { width: this.width, height: this.height, samples: this.samples, raw, summary: ptSummarizeCounters(raw, this.width, this.height, this.samples) };
  }

  private destroyBuffers(): void {
    this.counterBuffer?.destroy();
    this.counterBuffer = null;
    if (!this.buffers) return;
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
    this.buffers = null;
  }

  destroy(): void {
    this.paramsBuffer.destroy();
    this.argsBuffer.destroy();
    this.fuzzTable?.destroy();
    this.destroyBuffers();
  }
}
