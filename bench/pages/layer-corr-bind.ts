/**
 * Check pages that compile layered WGSL with `layout: 'auto'`: since D32 the albedo layering defaults to `albedoCorrection: 'texture'`, which declares
 * the correction table at `@group(2) @binding(PT_LAYER_CORR_BINDING)`. A page that only binds its own group 0 then fails at dispatch ("No bind group set
 * at group index 2") and reads garbage. `layerCorrectionGroup` returns the group to set at index 2 when the pipeline's entry point uses the table, else null.
 * Resolve it before encoding (it waits on an error scope), never inside a timed region.
 */
import { PT_LAYER_CORR_BINDING, ptLayerCorrCreateTexture } from '@pmndrs/pathtracer';

const textures = new WeakMap<GPUDevice, GPUTexture>();

export async function layerCorrectionGroup(device: GPUDevice, pipeline: GPUComputePipeline, code: string): Promise<GPUBindGroup | null> {
  if (!new RegExp(`@group\\(2\\)\\s*@binding\\(${PT_LAYER_CORR_BINDING}\\)`).test(code)) return null;
  let texture = textures.get(device);
  if (!texture) textures.set(device, (texture = ptLayerCorrCreateTexture(device, 'check page')));
  // An entry point that never reads the table has no binding 10 in its auto layout: creating the group then fails validation, and nothing is bound.
  device.pushErrorScope('validation');
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(2), entries: [{ binding: PT_LAYER_CORR_BINDING, resource: texture.createView() }] });
  return (await device.popErrorScope()) ? null : group;
}
