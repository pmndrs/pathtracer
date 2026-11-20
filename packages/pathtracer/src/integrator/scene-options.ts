/**
 * `ptSceneOptions`: the kernel options a scene decides (bvh formats, surface layout, textures, scene packing, OpenPBR material features,
 * alpha mask, environment map), so a kernel is compiled for exactly the scene it will run on:
 *
 *     const scene = await buildGpuScene(device, await loadGltf(url));
 *     const pt = await PtMegakernel.create(device, caps, ptSceneOptions(scene, { integrator: { nee: true } }));
 *     pt.setScene(scene.trace, scene.buffers);
 *
 * Materials run in OpenPBR mode (`integrator.materials`) with the library's layering default; a metallic-roughness texture decides
 * metalness per texel, so a textured metal compiles both lobes and the stochastic mix whatever its factor says.
 */
import { ALPHA_MODE, createMaterial } from '@pmndrs/raytracer';
import type { GpuScene } from '@pmndrs/raytracer';
import { ptMaterialFeatures } from '../bsdf/adapter.ts';
import type { PtLayeredOptions } from '../bsdf/layered.ts';
import type { PtIntegratorOptions } from './integrator-wgsl.ts';
import type { PtMegakernelOptions } from './megakernel.ts';

export interface PtSceneOptionsOverrides extends Partial<Omit<PtMegakernelOptions, 'format' | 'layout' | 'textures' | 'scenePacking'>> {
  /** Merged over the scene's integrator settings (yours win; `materials` is merged one level deep). */
  integrator?: PtIntegratorOptions;
  /** Layered BSDF settings of OpenPBR mode (default: the library default, `layering: 'albedo'`). */
  layered?: PtLayeredOptions;
}

/** Kernel options for `PtMegakernel.create` / `PtWavefront.create` that match `scene`, with `overrides` merged in. */
export function ptSceneOptions(scene: GpuScene, overrides: PtSceneOptionsOverrides = {}): PtMegakernelOptions {
  const { integrator: extra = {}, layered, ...rest } = overrides;
  const inputs = scene.input.materials;
  const records = inputs.map((m) => createMaterial(m.roughnessMetalTexture !== undefined && (m.metalness ?? 0) > 0 ? { ...m, metalness: 0.5 } : m));
  const features = ptMaterialFeatures(records);
  const alphaMask = !!scene.copy.textureLayout && records.some((m) => m.alphaMode === ALPHA_MODE.mask);
  const envMap = !!scene.input.environment?.map;
  const integrator: PtIntegratorOptions = {
    envMap,
    envSampling: envMap ? 'importance' : 'uniform',
    ...(alphaMask ? { alphaMask } : {}),
    ...extra,
    materials: extra.materials === null ? null : { features, ...(layered ? { layered } : {}), ...(extra.materials ?? {}) },
  };
  return { ...rest, format: scene.format, layout: scene.copy.layout, textures: scene.copy.textureLayout ?? null, scenePacking: scene.packing, integrator };
}
