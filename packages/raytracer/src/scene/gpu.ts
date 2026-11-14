/**
 * GPU upload of the scene copy: two storage buffers (table, data) and the bind group helpers.
 * Nothing here runs in the frame loop: a scene change is a rebuild + upload, never a readback.
 */
import type { SceneCopy } from './scene-copy.ts';
import type { SceneTextures, TextureLayout, TextureWrap } from './texture.ts';

// GPUBufferUsage bits, spelled out so this module loads in Node (tests) where the WebGPU globals do not exist.
const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const COMPUTE = 0x4; // GPUShaderStage.COMPUTE
const TEXTURE_BINDING = 0x4;
const TEXTURE_COPY_DST = 0x2;

const ADDRESS_MODE: Record<TextureWrap, GPUAddressMode> = { repeat: 'repeat', clamp: 'clamp-to-edge', mirror: 'mirror-repeat' };

/** The texture arrays of a scene on the GPU: one `texture_2d_array` + one trilinear sampler per bucket. */
export interface SceneTextureBuffers {
  readonly textures: readonly GPUTexture[];
  readonly views: readonly GPUTextureView[];
  readonly samplers: readonly GPUSampler[];
  readonly layout: TextureLayout;
  readonly byteLength: number;
}

export interface SceneCopyBuffers {
  readonly table: GPUBuffer;
  readonly data: GPUBuffer;
  readonly layout: SceneCopy['layout'];
  readonly byteLength: number;
  /** Present when the scene copy has textures (bindings `firstBinding + 2` onward, see `rtSceneCopyLayoutEntries`). */
  readonly textures?: SceneTextureBuffers;
  destroy(): void;
}

export interface UploadSceneCopyOptions {
  label?: string;
  /** Add COPY_SRC (tests that read the buffers back). Default false. */
  copySrc?: boolean;
}

/** Writes the scene copy to two new storage buffers. Fails clearly when a buffer would exceed the device's binding size. */
export function uploadSceneCopy(device: GPUDevice, copy: SceneCopy, options: UploadSceneCopyOptions = {}): SceneCopyBuffers {
  const label = options.label ?? 'rt scene';
  const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const make = (name: string, data: Uint32Array) => {
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    if (size > limit) throw new RangeError(`${label} ${name}: ${size} bytes exceeds the device's storage binding limit of ${limit} (chunking is a later task)`);
    const buffer = device.createBuffer({ label: `${label} ${name}`, size, usage: STORAGE | COPY_DST | (options.copySrc ? COPY_SRC : 0) });
    device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    return buffer;
  };
  const table = make('table', copy.table);
  const data = make('data', copy.data);
  const textures = copy.textures ? uploadSceneTextures(device, copy.textures, label) : undefined;
  return {
    table,
    data,
    layout: copy.layout,
    byteLength: copy.byteLength + (copy.textures?.byteLength ?? 0),
    ...(textures ? { textures } : {}),
    destroy() {
      table.destroy();
      data.destroy();
      textures?.textures.forEach((t) => t.destroy());
    },
  };
}

/**
 * Creates one texture array per bucket (`rgba8unorm-srgb` for colour, `rgba8unorm` for data, every mip level written from the CPU
 * chain) with a trilinear sampler whose address mode is the bucket's wrap. Fails clearly when a limit is exceeded.
 */
export function uploadSceneTextures(device: GPUDevice, textures: SceneTextures, label = 'rt scene'): SceneTextureBuffers {
  const { limits } = device;
  const count = textures.buckets.length;
  if (count > limits.maxSampledTexturesPerShaderStage || count > limits.maxSamplersPerShaderStage) throw new RangeError(`${label}: ${count} texture arrays exceed the device's limits (${limits.maxSampledTexturesPerShaderStage} sampled textures, ${limits.maxSamplersPerShaderStage} samplers per stage)`);
  const gpuTextures: GPUTexture[] = [];
  const views: GPUTextureView[] = [];
  const samplers: GPUSampler[] = [];
  textures.buckets.forEach((bucket, index) => {
    if (bucket.width > limits.maxTextureDimension2D || bucket.height > limits.maxTextureDimension2D || bucket.layers > limits.maxTextureArrayLayers) {
      throw new RangeError(`${label}: texture array ${index} (${bucket.width}x${bucket.height}x${bucket.layers}) exceeds the device's texture limits (${limits.maxTextureDimension2D}, ${limits.maxTextureArrayLayers} layers)`);
    }
    const texture = device.createTexture({
      label: `${label} textures ${index} ${bucket.width}x${bucket.height}x${bucket.layers} ${bucket.colorSpace} ${bucket.wrap}`,
      size: [bucket.width, bucket.height, bucket.layers],
      mipLevelCount: bucket.mipLevels,
      format: bucket.colorSpace === 'srgb' ? 'rgba8unorm-srgb' : 'rgba8unorm',
      usage: TEXTURE_BINDING | TEXTURE_COPY_DST,
    });
    bucket.mips.forEach((mip, level) => {
      const w = Math.max(1, bucket.width >> level);
      const h = Math.max(1, bucket.height >> level);
      device.queue.writeTexture({ texture, mipLevel: level }, mip as Uint8Array<ArrayBuffer>, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h, bucket.layers]);
    });
    gpuTextures.push(texture);
    views.push(texture.createView({ dimension: '2d-array' }));
    const address = ADDRESS_MODE[bucket.wrap];
    samplers.push(device.createSampler({ label: `${label} sampler ${index}`, magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: address, addressModeV: address, lodMinClamp: 0, lodMaxClamp: Math.max(0, bucket.mipLevels - 1) }));
  });
  const layout: TextureLayout = { buckets: textures.buckets.map(({ width, height, colorSpace, wrap, layers, mipLevels }) => ({ width, height, colorSpace, wrap, layers, mipLevels })) };
  return { textures: gpuTextures, views, samplers, layout, byteLength: textures.byteLength };
}

export function rtSceneCopyBindGroupEntries(buffers: SceneCopyBuffers, firstBinding = 0): GPUBindGroupEntry[] {
  const entries: GPUBindGroupEntry[] = [buffers.table, buffers.data].map((buffer, i) => ({ binding: firstBinding + i, resource: { buffer } }));
  const t = buffers.textures;
  if (t) {
    // Textures at firstBinding + 2 ..., then one sampler per texture (the same order as the WGSL declarations).
    t.views.forEach((view, i) => entries.push({ binding: firstBinding + 2 + i, resource: view }));
    t.samplers.forEach((sampler, i) => entries.push({ binding: firstBinding + 2 + t.views.length + i, resource: sampler }));
  }
  return entries;
}

/** Layout entries of the scene copy's bindings: table, data, then (when `textures` is given) the texture arrays and their samplers. */
export function rtSceneCopyLayoutEntries(firstBinding = 0, textures?: TextureLayout | null): GPUBindGroupLayoutEntry[] {
  const entries: GPUBindGroupLayoutEntry[] = [0, 1].map((i) => ({ binding: firstBinding + i, visibility: COMPUTE, buffer: { type: 'read-only-storage' as const } }));
  if (textures) {
    const n = textures.buckets.length;
    for (let i = 0; i < n; i++) entries.push({ binding: firstBinding + 2 + i, visibility: COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } });
    for (let i = 0; i < n; i++) entries.push({ binding: firstBinding + 2 + n + i, visibility: COMPUTE, sampler: { type: 'filtering' } });
  }
  return entries;
}
