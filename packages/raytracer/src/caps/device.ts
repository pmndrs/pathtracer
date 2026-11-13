import { buildCaps, defaultWgslLanguageFeatures, FEATURE_NAMES, LIMIT_KEYS } from './detect.ts';
import type { Caps, CapsLimits, FeatureFlag, LimitKey } from './detect.ts';

const GiB = 1024 * 1024 * 1024;

/** Sanity caps on requested limits: asking for the adapter maximum of everything wastes memory and hides portability bugs. */
export const DEFAULT_LIMIT_CAPS: CapsLimits = {
  maxStorageBufferBindingSize: 2 * GiB,
  maxBufferSize: 4 * GiB,
  maxStorageBuffersPerShaderStage: 16,
  maxSampledTexturesPerShaderStage: 32,
  maxTextureArrayLayers: 2048,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeWorkgroupsPerDimension: 65535,
};

export interface RequestDeviceOptions {
  label?: string;
  /** Flags to leave unrequested even though the adapter has them (A/B forcing a lower path). */
  disable?: FeatureFlag[];
  /** Override the sanity caps per limit. */
  limitCaps?: Partial<CapsLimits>;
  /** Extra raw feature strings to request when the adapter has them. */
  extraFeatures?: string[];
}

export interface DeviceForCaps {
  device: GPUDevice;
  /** Caps as the device actually got them (features and limits from the device, not the adapter). */
  caps: Caps;
}

/** Features to request: only those the adapter has, with the exact string it exposes. */
export function featuresToRequest(caps: Caps, options: RequestDeviceOptions = {}): string[] {
  const off = new Set(options.disable ?? []);
  const out: string[] = [];
  for (const flag of Object.keys(FEATURE_NAMES) as FeatureFlag[]) {
    const name = caps.featureNames[flag];
    // The heterogeneous table is requested together with the sampling table when both exist.
    if (name && !off.has(flag)) out.push(name);
  }
  for (const f of options.extraFeatures ?? []) if (caps.features.includes(f) && !out.includes(f)) out.push(f);
  return out;
}

/** Limits to request: adapter maximum, clamped by the sanity caps. Never below what the adapter default gives. */
export function limitsToRequest(caps: Caps, options: RequestDeviceOptions = {}): Partial<Record<LimitKey, number>> {
  const out: Partial<Record<LimitKey, number>> = {};
  for (const k of LIMIT_KEYS) {
    const max = caps.limits[k];
    if (max > 0) out[k] = Math.min(max, options.limitCaps?.[k] ?? DEFAULT_LIMIT_CAPS[k]);
  }
  return out;
}

export async function requestDeviceForCaps(
  adapter: GPUAdapter,
  caps: Caps,
  options: RequestDeviceOptions = {},
): Promise<DeviceForCaps> {
  const device = await adapter.requestDevice({
    label: options.label,
    requiredFeatures: featuresToRequest(caps, options) as GPUFeatureName[],
    requiredLimits: limitsToRequest(caps, options) as Record<string, number>,
  });
  const got = buildCaps(
    {
      features: device.features as ReadonlySet<string>,
      limits: device.limits as unknown as Record<string, unknown>,
    },
    defaultWgslLanguageFeatures(),
    adapter.info as unknown as Partial<Record<string, unknown>> | undefined,
  );
  return { device, caps: got };
}
