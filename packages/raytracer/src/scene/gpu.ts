/**
 * GPU upload of the scene copy: two storage buffers (table, data; `packing: 'split'`) or one (`'packed'`, see `SceneCopyPacking`) and the bind group helpers.
 * Nothing here runs in the frame loop: a scene change is a rebuild + upload, never a readback.
 */
import { Downsampler } from '@pmndrs/downsampler/webgpu';
import { DEFAULT_SCENE_PACKING, type SceneCopyPacking } from './constants.ts';
import { packSceneCopy, packedSceneCopyBytes, type SceneCopy } from './scene-copy.ts';
import type { SceneTextures, TextureLayout, TextureWrap } from './texture.ts';

// GPUBufferUsage bits, spelled out so this module loads in Node (tests) where the WebGPU globals do not exist.
const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const COMPUTE = 0x4; // GPUShaderStage.COMPUTE
const TEXTURE_BINDING = 0x4;
const TEXTURE_COPY_DST = 0x2;
const TEXTURE_STORAGE = 0x8;
const TEXTURE_RENDER = 0x10;

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
  readonly packing: SceneCopyPacking;
  /** `split` only: the table (`rtSceneTable`) and the data (`rtSceneData`). */
  readonly table?: GPUBuffer;
  readonly data?: GPUBuffer;
  /** `packed` only: header + table + data in one buffer (`rtScene`). */
  readonly packed?: GPUBuffer;
  /** Bytes of the scene copy's storage buffer(s) (the packed one includes its header and the data's padding). */
  readonly storageBytes: number;
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
  /** `split` (two buffers) or `packed` (one: frees a storage binding in every kernel, needs the whole scene copy under one binding limit). Default `DEFAULT_SCENE_PACKING`. The kernels must be built with the same packing. */
  packing?: SceneCopyPacking;
  /**
   * Fills mip levels 1+ of the texture arrays on the GPU. Default (a scene copy built with `textures: { mips: 'gpu' }`, the default):
   * `rtMipGenerator(device)`, a `@pmndrs/downsampler` `Downsampler` created once per device. Pass your own (a `Downsampler` with
   * options, or any `RtMipGenerator`) to override it; with a CPU chain (`mips: 'cpu'`) none is used unless given (levels 1+ are then overwritten).
   */
  mipGenerator?: RtMipGenerator;
}

/** The part of `@pmndrs/downsampler`'s `Downsampler` the upload uses: record the mip chain of `texture` (level 0 written) into `encoder`. */
export interface RtMipGenerator {
  encode(encoder: GPUCommandEncoder, texture: GPUTexture): unknown;
  /** Drops per-texture state (a `Downsampler` has it): called once the mip work is submitted. */
  release?(texture: GPUTexture): unknown;
}

const mipGenerators = new WeakMap<GPUDevice, Downsampler>();

/**
 * The library's default mip generator for `device`: a `@pmndrs/downsampler` `Downsampler`, created on first use and cached per device
 * (pipelines compile once, repeated uploads reuse them). It holds no GPU resources of its own that outlive the textures it was asked to fill
 * (per-texture state is released after each upload), so there is nothing to dispose: it is garbage collected with the device. The device
 * works with its default limits (fewer levels per dispatch than with `Downsampler.requiredLimits`, same result).
 */
export function rtMipGenerator(device: GPUDevice): RtMipGenerator {
  let generator = mipGenerators.get(device);
  if (!generator) {
    generator = new Downsampler(device);
    mipGenerators.set(device, generator);
  }
  return generator;
}

/** The device's storage binding size cap for one buffer. */
export function sceneCopyBindingLimit(device: GPUDevice): number {
  return Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
}

/** Writes the scene copy to two new storage buffers (or one, packed). Fails clearly when a buffer would exceed the device's binding size. */
export function uploadSceneCopy(device: GPUDevice, copy: SceneCopy, options: UploadSceneCopyOptions = {}): SceneCopyBuffers {
  const label = options.label ?? 'rt scene';
  const limit = sceneCopyBindingLimit(device);
  const packing = options.packing ?? DEFAULT_SCENE_PACKING;
  const byteLength = copy.byteLength + (copy.textures?.byteLength ?? 0);
  if (packing === 'packed') {
    const bytes = packedSceneCopyBytes(copy);
    if (bytes > limit) {
      const mib = (n: number) => `${(n / 2 ** 20).toFixed(1)} MiB`;
      throw new RangeError(`${label}: the packed scene copy is ${mib(bytes)} (table ${mib(copy.table.byteLength)} + data ${mib(copy.data.byteLength)}), over the device's storage binding limit of ${mib(limit)}; packing: 'split' binds them separately (each under the limit: table ${copy.table.byteLength <= limit ? 'fits' : 'does not fit'}, data ${copy.data.byteLength <= limit ? 'fits' : 'does not fit'}; chunking is a later task)`);
    }
    const packedCopy = packSceneCopy(copy);
    const packed = device.createBuffer({ label: `${label} packed`, size: Math.max(16, packedCopy.byteLength), usage: STORAGE | COPY_DST | (options.copySrc ? COPY_SRC : 0) });
    device.queue.writeBuffer(packed, 0, packedCopy.words.buffer as ArrayBuffer, packedCopy.words.byteOffset, packedCopy.words.byteLength);
    const textures = copy.textures ? uploadSceneTextures(device, copy.textures, label, options.mipGenerator, options.copySrc) : undefined;
    return {
      packing,
      packed,
      storageBytes: packedCopy.byteLength,
      layout: copy.layout,
      byteLength,
      ...(textures ? { textures } : {}),
      destroy() {
        packed.destroy();
        textures?.textures.forEach((t) => t.destroy());
      },
    };
  }
  const make = (name: string, data: Uint32Array) => {
    const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
    if (size > limit) throw new RangeError(`${label} ${name}: ${size} bytes exceeds the device's storage binding limit of ${limit} (chunking is a later task)`);
    const buffer = device.createBuffer({ label: `${label} ${name}`, size, usage: STORAGE | COPY_DST | (options.copySrc ? COPY_SRC : 0) });
    device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
    return buffer;
  };
  const table = make('table', copy.table);
  const data = make('data', copy.data);
  const textures = copy.textures ? uploadSceneTextures(device, copy.textures, label, options.mipGenerator, options.copySrc) : undefined;
  return {
    packing,
    table,
    data,
    storageBytes: table.size + data.size,
    layout: copy.layout,
    byteLength,
    ...(textures ? { textures } : {}),
    destroy() {
      table.destroy();
      data.destroy();
      textures?.textures.forEach((t) => t.destroy());
    },
  };
}

/**
 * Rewrites a scene copy into the buffers of an earlier upload when it fits (a rebuilt copy after instances moved or the lights changed: P1.4b has no
 * incremental update yet). Returns false, writing nothing, when the new copy needs bigger buffers or the texture layout changed: upload again then.
 * `split` writes the table and the data in place; `packed` rewrites the whole buffer (the data region moves when the table's length changes, and the
 * header's `dataBase` says where it is). The texture arrays are not touched. Needs `COPY_DST` (always set by `uploadSceneCopy`).
 */
export function writeSceneCopy(device: GPUDevice, buffers: SceneCopyBuffers, copy: SceneCopy): boolean {
  if (buffers.packing === 'packed') {
    const packed = buffers.packed as GPUBuffer;
    if (packed.size < Math.max(16, packedSceneCopyBytes(copy))) return false;
    const words = packSceneCopy(copy).words;
    device.queue.writeBuffer(packed, 0, words.buffer as ArrayBuffer, words.byteOffset, words.byteLength);
    return true;
  }
  const table = buffers.table as GPUBuffer;
  const data = buffers.data as GPUBuffer;
  if (table.size < Math.max(16, copy.table.byteLength) || data.size < Math.max(16, copy.data.byteLength)) return false;
  device.queue.writeBuffer(table, 0, copy.table.buffer as ArrayBuffer, copy.table.byteOffset, copy.table.byteLength);
  if (copy.data.byteLength) device.queue.writeBuffer(data, 0, copy.data.buffer as ArrayBuffer, copy.data.byteOffset, copy.data.byteLength);
  return true;
}

/**
 * Creates one texture array per bucket (`rgba8unorm-srgb` for colour, `rgba8unorm` for data, every mip level written from the CPU
 * chain, or levels 1+ generated here by the default `rtMipGenerator` when the textures were built with `mips: 'gpu'`) with a trilinear sampler whose address mode is the bucket's wrap. Fails clearly when a limit is exceeded.
 */
export function uploadSceneTextures(device: GPUDevice, textures: SceneTextures, label = 'rt scene', mipGenerator?: RtMipGenerator, copySrc = false): SceneTextureBuffers {
  const generator = mipGenerator ?? (textures.gpuMips ? rtMipGenerator(device) : undefined);
  const { limits } = device;
  const encoder = generator ? device.createCommandEncoder({ label: `${label} mips` }) : null;
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
      // A mip generator writes levels 1+ with compute (storage texture, unorm only) or render passes (sRGB cannot be a storage texture).
      usage: TEXTURE_BINDING | TEXTURE_COPY_DST | (copySrc ? 0x1 : 0) | (generator && bucket.mipLevels > 1 ? (bucket.colorSpace === 'srgb' ? TEXTURE_RENDER : TEXTURE_STORAGE | TEXTURE_RENDER) : 0),
    });
    bucket.mips.forEach((mip, level) => {
      const w = Math.max(1, bucket.width >> level);
      const h = Math.max(1, bucket.height >> level);
      device.queue.writeTexture({ texture, mipLevel: level }, mip as Uint8Array<ArrayBuffer>, { bytesPerRow: w * 4, rowsPerImage: h }, [w, h, bucket.layers]);
    });
    if (encoder && bucket.mipLevels > 1) (generator as RtMipGenerator).encode(encoder, texture);
    gpuTextures.push(texture);
    views.push(texture.createView({ dimension: '2d-array' }));
    const address = ADDRESS_MODE[bucket.wrap];
    samplers.push(device.createSampler({ label: `${label} sampler ${index}`, magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', addressModeU: address, addressModeV: address, lodMinClamp: 0, lodMaxClamp: Math.max(0, bucket.mipLevels - 1) }));
  });
  if (encoder) {
    device.queue.submit([encoder.finish()]);
    if (generator?.release) for (const texture of gpuTextures) generator.release(texture);
  }
  const layout: TextureLayout = { buckets: textures.buckets.map(({ width, height, colorSpace, wrap, layers, mipLevels }) => ({ width, height, colorSpace, wrap, layers, mipLevels })) };
  return { textures: gpuTextures, views, samplers, layout, byteLength: textures.byteLength };
}

export function rtSceneCopyBindGroupEntries(buffers: SceneCopyBuffers, firstBinding = 0): GPUBindGroupEntry[] {
  const list = buffers.packing === 'packed' ? [buffers.packed as GPUBuffer] : [buffers.table as GPUBuffer, buffers.data as GPUBuffer];
  const entries: GPUBindGroupEntry[] = list.map((buffer, i) => ({ binding: firstBinding + i, resource: { buffer } }));
  const t = buffers.textures;
  if (t) {
    // Textures after the buffer bindings (2, or 1 packed), then one sampler per texture (the same order as the WGSL declarations).
    const next = firstBinding + list.length;
    t.views.forEach((view, i) => entries.push({ binding: next + i, resource: view }));
    t.samplers.forEach((sampler, i) => entries.push({ binding: next + t.views.length + i, resource: sampler }));
  }
  return entries;
}

/** Layout entries of the scene copy's bindings: table, data (one binding when `packing` is 'packed'), then (when `textures` is given) the texture arrays and their samplers. */
export function rtSceneCopyLayoutEntries(firstBinding = 0, textures?: TextureLayout | null, packing: SceneCopyPacking = DEFAULT_SCENE_PACKING): GPUBindGroupLayoutEntry[] {
  const count = packing === 'packed' ? 1 : 2;
  const entries: GPUBindGroupLayoutEntry[] = Array.from({ length: count }, (_, i) => ({ binding: firstBinding + i, visibility: COMPUTE, buffer: { type: 'read-only-storage' as const } }));
  if (textures) {
    const n = textures.buckets.length;
    for (let i = 0; i < n; i++) entries.push({ binding: firstBinding + count + i, visibility: COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } });
    for (let i = 0; i < n; i++) entries.push({ binding: firstBinding + count + n + i, visibility: COMPUTE, sampler: { type: 'filtering' } });
  }
  return entries;
}
