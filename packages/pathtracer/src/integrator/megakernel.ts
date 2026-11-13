/**
 * Host side of the megakernel path tracer (P2.4): kernel generation, pipeline, accumulation buffer, per-pass dispatch.
 *
 *     const pt = await PtMegakernel.create(device, caps, { format, layout: copy.layout, integrator: { sampler: 'owen-sobol' } });
 *     pt.setScene(traceBuffers, sceneCopyBuffers);                          // bvh walk buffers + the scene copy
 *     pt.setFilm(width, height); pt.setCamera(cameraDesc);
 *     pt.setLimits({ maxDepth: 8, rrStartDepth: 4 });                       // pbrt semantics; rrStartDepth omitted = no RR
 *     pt.encode(encoder, 16);                                               // 16 sample passes, one dispatch each
 *     const image = await pt.readImage();                                   // accum / samples, outside any timed region
 *
 * One dispatch per sample pass, so no single dispatch outlives a GPU watchdog and a pass is the unit of progress. Each pass
 * gets its own record in a ring of uniform slots (dynamic offsets): `queue.writeBuffer` is ordered with `submit`, so a slot
 * is never rewritten before the pass that read it ran, as long as no more than `PT_PARAM_SLOTS` passes are outstanding.
 * Accumulation is f32 `array<vec4f>` (sum of radiance, w = samples); `readImage` divides. Nothing here reads back unless you call it.
 */
import { rtLightSampleWgsl, rtModuleWgsl, rtOwenSobolWgsl, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtSceneBindGroupEntries, rtWalkEntries, stripWgslComments, withBindings } from '@pmndrs/raytracer';
import type { Caps, SceneCopyBuffers, SceneFormat, SurfaceLayout } from '@pmndrs/raytracer';
import { ptBsdfAdapterWgsl } from '../bsdf/adapter.ts';
import { ptBsdfWgsl } from '../bsdf/bsdf-wgsl.ts';
import { ptCameraBasis, type PtLookAtCamera } from './camera.ts';
import { ptSamplerWgsl } from './dimensions.ts';
import { PT_PARAMS_BYTES, PT_WORKGROUP, ptIntegratorWgsl, resolveIntegratorOptions, type PtIntegratorOptions } from './integrator-wgsl.ts';

// GPUBufferUsage bits spelled out so this file loads in Node (tests).
const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const UNIFORM = 0x40;
const MAP_READ = 0x1;
const COMPUTE = 0x4;

/** Uniform slots in the ring (one per outstanding sample pass). */
export const PT_PARAM_SLOTS = 1024;

type TraceScene = Parameters<typeof rtSceneBindGroupEntries>[0];

export interface PtLimits {
  /** Scattering events per path, pbrt `maxdepth` semantics (1 = direct lighting only; the emission of a camera hit is free). */
  maxDepth: number;
  /** Russian roulette from this scattering vertex (1 = first surface hit); omitted or 0 = off. */
  rrStartDepth?: number;
}

export interface PtMegakernelOptions {
  /** bvh formats of the scene the kernel will run on (`sceneFormatOf(traceBuffers)`). */
  format: SceneFormat;
  /** Surface layout of the scene copy (`copy.layout`). */
  layout: SurfaceLayout;
  integrator?: PtIntegratorOptions;
  /**
   * Drop the indentation of every line and the spaces around `{ } ( ) , ;` (every line break stays, so compiler line numbers
   * are kept). Default true: it is 17% of the text (63.3 KB -> 52.7 KB). Comments must already be gone (they are).
   */
  minify?: boolean;
  label?: string;
}

export interface PtImage {
  width: number;
  height: number;
  /** RGBA float32, row 0 = top, radiance averaged over `samples`. alpha = 1. */
  data: Float32Array;
  samples: number;
}

/** Indentation and the spaces around `{ } ( ) , ;` removed; line breaks kept. Only safe on comment-free source. */
export function minifyWgsl(code: string): string {
  return code.replace(/^[ \t]+/gm, '').replace(/[ \t]*([{}(),;])[ \t]*/g, '$1');
}

/** The kernel's source for these options (what `create` compiles). Exported for size checks and the tests. */
export function ptMegakernelWgsl(caps: Caps, options: Pick<PtMegakernelOptions, 'format' | 'layout' | 'integrator' | 'minify'>): string {
  const integrator = resolveIntegratorOptions(options.integrator);
  const raytracer = rtModuleWgsl(caps, {
    ...options.format,
    rayMasks: false,
    traceSlot: { group: 0, firstBinding: 0 },
    scene: { slot: { group: 1, firstBinding: 0 }, layout: options.layout },
  });
  const ours = [rtLightSampleWgsl(), rtOwenSobolWgsl(), stripWgslComments(ptBsdfWgsl(options.integrator ?? {})), ptBsdfAdapterWgsl(), ptSamplerWgsl(integrator.sampler), ptIntegratorWgsl(options.integrator ?? {})].join('\n');
  // `raytracer` already went through the caps preprocessor (its `enable` lines are first); ours has no `#if`.
  const code = `${raytracer}\n${ours}`;
  return (options.minify ?? true) ? minifyWgsl(code) : code;
}

export class PtMegakernel {
  readonly device: GPUDevice;
  readonly options: PtMegakernelOptions;
  readonly wgslBytes: number;
  private readonly pipeline: GPUComputePipeline;
  private readonly walkLayout: GPUBindGroupLayout;
  private readonly sceneLayout: GPUBindGroupLayout;
  private readonly ioLayout: GPUBindGroupLayout;
  private readonly paramsBuffer: GPUBuffer;
  private readonly slotBytes: number;
  private slot = 0;
  private walkGroup: GPUBindGroup | null = null;
  private sceneGroup: GPUBindGroup | null = null;
  private ioGroup: GPUBindGroup | null = null;
  private accum: GPUBuffer | null = null;
  private width = 0;
  private height = 0;
  private camera: PtLookAtCamera | null = null;
  private limits: PtLimits = { maxDepth: 5 };
  private seed = 0;
  /** Sample passes encoded since the last `reset`. */
  samples = 0;

  private constructor(device: GPUDevice, options: PtMegakernelOptions, pipeline: GPUComputePipeline, layouts: GPUBindGroupLayout[], wgslBytes: number) {
    this.device = device;
    this.options = options;
    this.pipeline = pipeline;
    this.walkLayout = layouts[0] as GPUBindGroupLayout;
    this.sceneLayout = layouts[1] as GPUBindGroupLayout;
    this.ioLayout = layouts[2] as GPUBindGroupLayout;
    this.wgslBytes = wgslBytes;
    this.slotBytes = Math.max(PT_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
    this.paramsBuffer = device.createBuffer({ label: 'pt params', size: this.slotBytes * PT_PARAM_SLOTS, usage: UNIFORM | COPY_DST });
  }

  static async create(device: GPUDevice, caps: Caps, options: PtMegakernelOptions): Promise<PtMegakernel> {
    const label = options.label ?? 'pt megakernel';
    const code = ptMegakernelWgsl(caps, options);
    const layouts = [
      device.createBindGroupLayout({ label: `${label} walk`, entries: withBindings(rtWalkEntries(options.format)) }),
      device.createBindGroupLayout({ label: `${label} scene`, entries: rtSceneCopyLayoutEntries() }),
      device.createBindGroupLayout({
        label: `${label} io`,
        entries: [
          { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PT_PARAMS_BYTES } },
          { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
        ],
      }),
    ];
    const pipeline = await device.createComputePipelineAsync({
      label,
      layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
      compute: { module: device.createShaderModule({ label, code }), entryPoint: 'ptMain' },
    });
    return new PtMegakernel(device, options, pipeline, layouts, code.length);
  }

  setScene(trace: TraceScene, scene: SceneCopyBuffers): void {
    this.walkGroup = this.device.createBindGroup({ label: 'pt walk', layout: this.walkLayout, entries: rtSceneBindGroupEntries(trace) });
    this.sceneGroup = this.device.createBindGroup({ label: 'pt scene', layout: this.sceneLayout, entries: rtSceneCopyBindGroupEntries(scene) });
  }

  setFilm(width: number, height: number): void {
    if (width === this.width && height === this.height && this.accum) return;
    this.width = width;
    this.height = height;
    this.accum?.destroy();
    this.accum = this.device.createBuffer({ label: 'pt accum', size: width * height * 16, usage: STORAGE | COPY_SRC | COPY_DST });
    this.ioGroup = null;
    this.samples = 0;
  }

  setCamera(camera: PtLookAtCamera): void {
    this.camera = camera;
  }

  setLimits(limits: PtLimits): void {
    this.limits = limits;
  }

  /** Scrambling seed base (the per-pixel seeds derive from it). Changing it gives an independent realisation. */
  setSeed(seed: number): void {
    this.seed = seed >>> 0;
  }

  /** Zeroes the accumulation buffer and the sample counter. Encodes into `encoder`. */
  reset(encoder: GPUCommandEncoder): void {
    if (this.accum) encoder.clearBuffer(this.accum);
    this.samples = 0;
  }

  private paramsRecord(sampleIndex: number): ArrayBuffer {
    if (!this.camera) throw new Error('PtMegakernel: setCamera() first');
    const basis = ptCameraBasis(this.camera, this.width, this.height);
    const buffer = new ArrayBuffer(PT_PARAMS_BYTES);
    const u = new Uint32Array(buffer);
    const f = new Float32Array(buffer);
    u.set([this.width, this.height, sampleIndex >>> 0, this.seed, this.limits.maxDepth >>> 0, (this.limits.rrStartDepth ?? 0) >>> 0, 0, 0]);
    f.set([...basis.origin, 0, ...basis.right, 0, ...basis.up, 0, ...basis.forward, 0], 8);
    return buffer;
  }

  /** Records `passes` sample passes (one dispatch each) into `encoder`. At most 256 per call; at most `PT_PARAM_SLOTS` outstanding. */
  encode(encoder: GPUCommandEncoder, passes: number, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.walkGroup || !this.sceneGroup) throw new Error('PtMegakernel: setScene() first');
    if (!this.accum) throw new Error('PtMegakernel: setFilm() first');
    if (passes > 256) throw new RangeError('PtMegakernel.encode: at most 256 passes per call');
    this.ioGroup ??= this.device.createBindGroup({
      label: 'pt io',
      layout: this.ioLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer, offset: 0, size: PT_PARAMS_BYTES } },
        { binding: 1, resource: { buffer: this.accum } },
      ],
    });
    const pass = encoder.beginComputePass({ label: 'pt samples', timestampWrites });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.walkGroup);
    pass.setBindGroup(1, this.sceneGroup);
    const gx = Math.ceil(this.width / PT_WORKGROUP[0]);
    const gy = Math.ceil(this.height / PT_WORKGROUP[1]);
    for (let i = 0; i < passes; i++) {
      const offset = (this.slot % PT_PARAM_SLOTS) * this.slotBytes;
      this.slot++;
      this.device.queue.writeBuffer(this.paramsBuffer, offset, this.paramsRecord(this.samples));
      pass.setBindGroup(2, this.ioGroup, [offset]);
      pass.dispatchWorkgroups(gx, gy);
      this.samples++;
    }
    pass.end();
  }

  /** Copies the accumulator, maps it and averages. The caller keeps this out of timed regions. */
  async readImage(): Promise<PtImage> {
    if (!this.accum) throw new Error('PtMegakernel: nothing rendered');
    const size = this.width * this.height * 16;
    const staging = this.device.createBuffer({ label: 'pt readback', size, usage: MAP_READ | COPY_DST });
    const encoder = this.device.createCommandEncoder({ label: 'pt readback' });
    encoder.copyBufferToBuffer(this.accum, 0, staging, 0, size);
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

  destroy(): void {
    this.paramsBuffer.destroy();
    this.accum?.destroy();
    this.accum = null;
  }
}
