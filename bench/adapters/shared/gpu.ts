import { summarizeAdapter } from '../../runner/page-client.ts';
import type { AdapterSummary } from '../../runner/contract.ts';

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  summary: AdapterSummary;
}

/** A device with every adapter feature and the adapter's maximum limits (the web path tracers want big storage buffers). */
export async function createMaxDevice(): Promise<GpuContext> {
  if (!navigator.gpu) throw new Error('navigator.gpu is undefined: no WebGPU in this browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('requestAdapter() returned null: no usable GPU.');
  const requiredLimits: Record<string, number> = {};
  for (const key in adapter.limits) {
    const value = (adapter.limits as unknown as Record<string, unknown>)[key];
    if (typeof value === 'number') requiredLimits[key] = value;
  }
  const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features] as GPUFeatureName[], requiredLimits });
  return { adapter, device, summary: summarizeAdapter(adapter) };
}

export const nowMs = (): number => performance.now();
