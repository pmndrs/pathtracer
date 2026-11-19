/** Bench-only loader for the historical Reflections TraceProvider seam. */
import * as bvh from '@pmndrs/bvh';
import * as rt from '@pmndrs/raytracer';
import type { DescSceneCopyInput } from '../scenes/to-scene-copy.ts';

interface TraceProvider {
  readonly entries: readonly Omit<GPUBindGroupLayoutEntry, 'binding'>[];
  readonly traceBuffers: bvh.TraceSceneBuffers;
  readonly sceneCopy: rt.SceneCopy;
  readonly sceneCopyBuffers: rt.SceneCopyBuffers;
  code(slot: { group: number; firstBinding: number }): string;
  resources(): readonly GPUBindingResource[];
  setTransforms(matrices: readonly ArrayLike<number>[]): void;
  destroy(): void;
}

interface EnvironmentHitShading {
  wgsl: string;
  layout: GPUBindGroupLayout;
  createBindGroup(view: GPUTextureView): GPUBindGroup;
  setLighting(intensity: number, lights: object): void;
  destroy(): void;
}

interface ReflectionsModules {
  HIT_WGSL: string;
  createRaytracerTraceProvider(device: GPUDevice, scene: object, options: object): Promise<TraceProvider>;
  environmentHitShading(device: GPUDevice, mapping: 'equirect'): EnvironmentHitShading;
}

export async function reflectionProviderModules(checkout: string): Promise<ReflectionsModules> {
  const at = (file: string) => import(/* @vite-ignore */ `/@fs${checkout}/packages/core/src/trace/${file}`);
  const [provider, raytracer, shading] = await Promise.all([at('provider.ts'), at('raytracer.ts'), at('hitShading.ts')]);
  return { ...provider, ...raytracer, ...shading } as ReflectionsModules;
}

export async function makeReflectionProvider(modules: ReflectionsModules, device: GPUDevice, input: DescSceneCopyInput) {
  const module: typeof rt = {
    ...rt,
    uploadSceneCopy: (d, copy, options) => rt.uploadSceneCopy(d, copy, { ...options, packing: 'split' }),
    rtSceneEntries: () => rt.rtSceneEntries('split'),
    rtModuleWgsl: (caps, options = {}) => rt.rtModuleWgsl(caps, options.scene
      ? { ...options, scene: { ...options.scene, packing: 'split' } } : options),
  };
  return modules.createRaytracerTraceProvider(device, {
    meshes: input.meshes,
    instances: input.instances.map(i => ({ mesh: i.meshIndex, matrixWorld: i.objectToWorld, material: i.material })),
    materials: input.materials.map(openpbr => ({ albedo: openpbr.baseColor ?? [0.5, 0.5, 0.5], openpbr })),
    lights: { sun: input.sun, environment: input.environment },
  }, { module, bvhModule: bvh, shade: { envSampling: 'cosine' }, minify: false });
}
