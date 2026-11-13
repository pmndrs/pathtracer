/**
 * GPU upload of the scene copy: two storage buffers (table, data) and the bind group helpers.
 * Nothing here runs in the frame loop: a scene change is a rebuild + upload, never a readback.
 */
import type { SceneCopy } from './scene-copy.ts';

// GPUBufferUsage bits, spelled out so this module loads in Node (tests) where the WebGPU globals do not exist.
const STORAGE = 0x80;
const COPY_DST = 0x8;
const COPY_SRC = 0x4;
const COMPUTE = 0x4; // GPUShaderStage.COMPUTE

export interface SceneCopyBuffers {
  readonly table: GPUBuffer;
  readonly data: GPUBuffer;
  readonly layout: SceneCopy['layout'];
  readonly byteLength: number;
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
  return {
    table,
    data,
    layout: copy.layout,
    byteLength: copy.byteLength,
    destroy() {
      table.destroy();
      data.destroy();
    },
  };
}

export function rtSceneCopyBindGroupEntries(buffers: SceneCopyBuffers, firstBinding = 0): GPUBindGroupEntry[] {
  return [buffers.table, buffers.data].map((buffer, i) => ({ binding: firstBinding + i, resource: { buffer } }));
}

export function rtSceneCopyLayoutEntries(firstBinding = 0): GPUBindGroupLayoutEntry[] {
  return [0, 1].map((i) => ({ binding: firstBinding + i, visibility: COMPUTE, buffer: { type: 'read-only-storage' as const } }));
}
