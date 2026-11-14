/** Shared helpers for the research-track-N pages (mlp-check, mlp-fit). */

/** A device with every adapter feature and the adapter's max limits (the MLP wants 16-32 KiB of workgroup memory). */
export async function requestMlpDevice(): Promise<{ adapter: GPUAdapter; device: GPUDevice }> {
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  const requiredLimits: Record<string, number> = {};
  for (const name of ['maxComputeWorkgroupStorageSize', 'maxStorageBufferBindingSize', 'maxBufferSize', 'maxStorageBuffersPerShaderStage', 'maxComputeInvocationsPerWorkgroup']) {
    const v = (adapter.limits as unknown as Record<string, number>)[name];
    if (typeof v === 'number') requiredLimits[name] = v;
  }
  const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features] as GPUFeatureName[], requiredLimits, label: 'mlp' });
  return { adapter, device };
}

/** Copies a buffer out and maps it. Only after the measured work, never inside a loop. */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, bytes = src.size): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(src, 0, staging, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

export function storageBuffer(device: GPUDevice, data: ArrayBufferView | number, label: string): GPUBuffer {
  const size = Math.max(16, Math.ceil((typeof data === 'number' ? data : data.byteLength) / 16) * 16);
  const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  return buffer;
}

/** f16 bits → number. */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
