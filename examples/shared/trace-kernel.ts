/**
 * Your own compute shader with the trace core pasted in: the pattern of `@pmndrs/raytracer`'s inline use.
 *
 *   group 0: the bvh walk (`rtWalkEntries`)        -> rtVisible, rtClosest
 *   group 1: the scene copy (`rtSceneCopyLayoutEntries`) -> rtSurface, rtMaterial, the lights (and rtShade with `shade`)
 *   group 2: your own buffers (`io`)
 *
 * `rtModuleWgsl` generates the module for the scene's formats; your WGSL is appended (through `rtPackSceneAccess` when the scene copy is
 * packed, so code that reads the copy directly still works). The examples' kernels write an `array<vec4f>` accumulator (rgb sum, w = samples)
 * that `AccumulationView` draws.
 */
import { rtModuleWgsl, rtPackSceneAccess, rtSceneBindGroupEntries, rtSceneCopyBindGroupEntries, rtSceneCopyLayoutEntries, rtWalkEntries, withBindings, type Caps, type GpuScene, type RtModuleOptions } from '@pmndrs/raytracer';

export interface TraceKernelOptions {
  label: string;
  /** Your WGSL: entry points and helpers, reading group 2. */
  wgsl: string;
  entryPoints: readonly string[];
  /** Layout entries of group 2. */
  io: GPUBindGroupLayoutEntry[];
  /** Extra module options (e.g. `shade: true`). */
  module?: Partial<RtModuleOptions>;
}

export class TraceKernel {
  readonly pipelines: Record<string, GPUComputePipeline>;
  readonly ioLayout: GPUBindGroupLayout;
  readonly bytes: number;
  private readonly walkGroup: GPUBindGroup;
  private readonly sceneGroup: GPUBindGroup;

  private constructor(pipelines: Record<string, GPUComputePipeline>, ioLayout: GPUBindGroupLayout, walkGroup: GPUBindGroup, sceneGroup: GPUBindGroup, bytes: number) {
    this.pipelines = pipelines;
    this.ioLayout = ioLayout;
    this.walkGroup = walkGroup;
    this.sceneGroup = sceneGroup;
    this.bytes = bytes;
  }

  static async create(device: GPUDevice, caps: Caps, scene: GpuScene, o: TraceKernelOptions): Promise<TraceKernel> {
    const module = rtModuleWgsl(caps, {
      ...scene.format,
      traceSlot: { group: 0, firstBinding: 0 },
      scene: { slot: { group: 1, firstBinding: 0 }, layout: scene.copy.layout, textures: scene.copy.textureLayout, packing: scene.packing },
      ...o.module,
    });
    const code = `${module}\n${scene.packing === 'packed' ? rtPackSceneAccess(o.wgsl) : o.wgsl}`;
    const layouts = [
      device.createBindGroupLayout({ label: `${o.label} walk`, entries: withBindings(rtWalkEntries(scene.format)) }),
      device.createBindGroupLayout({ label: `${o.label} scene`, entries: rtSceneCopyLayoutEntries(0, scene.copy.textureLayout, scene.packing) }),
      device.createBindGroupLayout({ label: `${o.label} io`, entries: o.io }),
    ] as const;
    const shader = device.createShaderModule({ label: o.label, code });
    const errors = (await shader.getCompilationInfo()).messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error(`${o.label}: WGSL does not compile:\n${errors.map((m) => `line ${m.lineNum}: ${m.message}\n  ${code.split('\n')[m.lineNum - 1]?.trim() ?? ''}`).join('\n')}`);
    const layout = device.createPipelineLayout({ bindGroupLayouts: layouts });
    const pipelines: Record<string, GPUComputePipeline> = {};
    for (const entryPoint of o.entryPoints) pipelines[entryPoint] = await device.createComputePipelineAsync({ label: `${o.label} ${entryPoint}`, layout, compute: { module: shader, entryPoint } });
    const walkGroup = device.createBindGroup({ layout: layouts[0], entries: rtSceneBindGroupEntries(scene.trace) });
    const sceneGroup = device.createBindGroup({ layout: layouts[1], entries: rtSceneCopyBindGroupEntries(scene.buffers) });
    return new TraceKernel(pipelines, layouts[2], walkGroup, sceneGroup, code.length);
  }

  /** Records one dispatch of `entryPoint` with your group 2. */
  dispatch(encoder: GPUCommandEncoder, entryPoint: string, io: GPUBindGroup, x: number, y = 1): void {
    const pass = encoder.beginComputePass({ label: entryPoint });
    pass.setPipeline(this.pipelines[entryPoint]!);
    pass.setBindGroup(0, this.walkGroup);
    pass.setBindGroup(1, this.sceneGroup);
    pass.setBindGroup(2, io);
    pass.dispatchWorkgroups(x, y);
    pass.end();
  }
}

const COMPUTE = 0x4;
/** Group 2 layout entries: `uniform` / `read` / `write` per binding, in order. */
export function ioEntries(kinds: readonly ('uniform' | 'read' | 'write')[]): GPUBindGroupLayoutEntry[] {
  return kinds.map((kind, binding) => ({ binding, visibility: COMPUTE, buffer: { type: kind === 'uniform' ? 'uniform' : kind === 'read' ? 'read-only-storage' : 'storage' } }));
}
