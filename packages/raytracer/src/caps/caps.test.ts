import { describe, expect, it } from 'vitest';
import { buildCaps, capsHash, detectCaps, deriveTier, featuresToRequest, limitsToRequest, requestDeviceForCaps } from './index.ts';

const MB = 1024 * 1024;
const GB = 1024 * MB;

const baseLimits = {
  maxStorageBufferBindingSize: 128 * MB,
  maxBufferSize: 256 * MB,
  maxStorageBuffersPerShaderStage: 8,
  maxSampledTexturesPerShaderStage: 16,
  maxTextureArrayLayers: 256,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeWorkgroupsPerDimension: 65535,
};

function mockAdapter(features: string[], limits: Record<string, number> = {}, info: Record<string, unknown> = {}) {
  const requests: { requiredFeatures?: string[]; requiredLimits?: Record<string, number> }[] = [];
  const adapter = {
    features: new Set(features),
    limits: { ...baseLimits, ...limits },
    info: { vendor: 'apple', architecture: 'metal-3', device: '', description: 'Apple M', ...info },
    requestDevice: async (d: { requiredFeatures?: string[]; requiredLimits?: Record<string, number> }) => {
      requests.push(d);
      return {
        features: new Set(d.requiredFeatures ?? []),
        limits: { ...baseLimits, ...limits, ...d.requiredLimits },
      };
    },
  };
  return { adapter: adapter as unknown as GPUAdapter, requests };
}
const lang = new Set(['immediate_address_space', 'pointer_composite_access']);
const noLang = new Set<string>();

const T1 = ['shader-f16', 'timestamp-query', 'texture-compression-bc'];
const T1_LIMITS = { maxStorageBufferBindingSize: 2 * GB };

describe('detectCaps', () => {
  it('core-only adapter is tier 0 with everything off', () => {
    const c = detectCaps(mockAdapter([]).adapter, noLang);
    expect(c.tier).toBe(0);
    expect(c.meetsBaseline).toBe(true);
    expect(c.subgroups || c.shaderF16 || c.timestampQuery || c.immediates || c.subgroupMatrix).toBe(false);
    expect(c.resourceTable).toBe('none');
    expect(c.adapter.vendor).toBe('apple');
    expect(c.subgroupMinSize).toBe(0);
  });

  it('below baseline limits clears meetsBaseline, tier stays 0', () => {
    const c = detectCaps(mockAdapter([], { maxStorageBuffersPerShaderStage: 4 }).adapter, noLang);
    expect(c.meetsBaseline).toBe(false);
    expect(c.tier).toBe(0);
  });

  it('tier 1 needs every rule', () => {
    const t = (features: string[], limits: Record<string, number> = T1_LIMITS, l = lang) => detectCaps(mockAdapter(features, limits).adapter, l).tier;
    expect(t(T1)).toBe(1);
    expect(t(['shader-f16', 'timestamp-query', 'texture-compression-astc'])).toBe(1);
    expect(t(T1.filter((f) => f !== 'shader-f16'))).toBe(0);
    expect(t(T1.filter((f) => f !== 'timestamp-query'))).toBe(0);
    expect(t(T1.filter((f) => f !== 'texture-compression-bc'))).toBe(0);
    expect(t(['shader-f16', 'timestamp-query', 'texture-compression-etc2'])).toBe(0);
    expect(t(T1, T1_LIMITS, noLang)).toBe(0); // no immediates
    expect(t(T1, { maxStorageBufferBindingSize: 1 * GB - 1 })).toBe(0);
    expect(t(T1, { ...T1_LIMITS, maxComputeInvocationsPerWorkgroup: 128 })).toBe(0);
  });

  it('tier 2 adds subgroups, tier 3 adds a resource table', () => {
    const sg = [...T1, 'subgroups'];
    expect(detectCaps(mockAdapter(sg, T1_LIMITS).adapter, lang).tier).toBe(2);
    expect(detectCaps(mockAdapter([...T1, 'sampling-resource-table'], T1_LIMITS).adapter, lang).tier).toBe(1);
    const t3 = detectCaps(mockAdapter([...sg, 'chromium-experimental-sampling-resource-table'], T1_LIMITS).adapter, lang);
    expect(t3.tier).toBe(3);
    expect(t3.resourceTable).toBe('sampling');
  });

  it('matches prefixed and unprefixed names, prefers heterogeneous', () => {
    const c = detectCaps(
      mockAdapter([
        'chromium-experimental-sampling-resource-table',
        'heterogeneous-resource-table',
        'chromium-experimental-subgroup-matrix',
        'chromium-experimental-subgroups',
      ]).adapter,
      noLang,
    );
    expect(c.resourceTable).toBe('heterogeneous');
    expect(c.subgroupMatrix).toBe(true);
    expect(c.subgroups).toBe(true);
    expect(c.featureNames.subgroups).toBe('chromium-experimental-subgroups');
    expect(c.featureNames.samplingResourceTable).toBe('chromium-experimental-sampling-resource-table');
  });

  it('reports compressed families, misc flags, subgroup sizes', () => {
    const c = detectCaps(
      mockAdapter(
        ['texture-compression-bc', 'texture-compression-bc-sliced-3d', 'texture-compression-etc2', 'float32-filterable', 'texture-formats-tier1', 'acceleration-structure'],
        {},
        { subgroupMinSize: 4, subgroupMaxSize: 64 },
      ).adapter,
      noLang,
    );
    expect(c.compressed).toEqual({ bc: true, bcSliced3d: true, astc: false, etc2: true });
    expect(c.float32Filterable && c.textureFormatsTier1 && c.accelerationStructure).toBe(true);
    expect([c.subgroupMinSize, c.subgroupMaxSize]).toEqual([4, 64]);
    expect(c.features).toEqual([...c.features].sort());
  });

  it('tolerates a missing info object', () => {
    const { adapter } = mockAdapter([]);
    (adapter as unknown as { info: undefined }).info = undefined;
    expect(detectCaps(adapter, noLang).adapter.vendor).toBe('');
  });
});

describe('requestDeviceForCaps', () => {
  it('requests only present features with their exact names and capped limits', async () => {
    const { adapter, requests } = mockAdapter(
      [...T1, 'chromium-experimental-subgroups'],
      { maxStorageBufferBindingSize: 8 * GB, maxBufferSize: 16 * GB, maxStorageBuffersPerShaderStage: 64 },
    );
    const caps = detectCaps(adapter, lang);
    const { device, caps: got } = await requestDeviceForCaps(adapter, caps);
    void device;
    const r = requests[0]!;
    expect(r.requiredFeatures).toContain('chromium-experimental-subgroups');
    expect(r.requiredFeatures).not.toContain('subgroups');
    expect(r.requiredFeatures).not.toContain('float32-filterable');
    expect(r.requiredLimits!.maxStorageBufferBindingSize).toBe(2 * GB);
    expect(r.requiredLimits!.maxBufferSize).toBe(4 * GB);
    expect(r.requiredLimits!.maxStorageBuffersPerShaderStage).toBe(16);
    expect(got.limits.maxStorageBufferBindingSize).toBe(2 * GB);
    expect(got.subgroups).toBe(true);
  });

  it('disable forces a flag off and caps reflect the device', async () => {
    const { adapter } = mockAdapter([...T1, 'subgroups'], T1_LIMITS);
    const caps = detectCaps(adapter, lang);
    expect(featuresToRequest(caps, { disable: ['subgroups'] })).not.toContain('subgroups');
    const { caps: got } = await requestDeviceForCaps(adapter, caps, { disable: ['subgroups', 'shaderF16'] });
    expect(got.subgroups).toBe(false);
    expect(got.shaderF16).toBe(false);
    expect(got.tier).toBe(0);
    expect(got.adapter.vendor).toBe('apple');
  });

  it('limitCaps override and skip of unreported limits', () => {
    const caps = detectCaps(mockAdapter([], { maxBufferSize: 16 * GB }).adapter, noLang);
    expect(limitsToRequest(caps, { limitCaps: { maxBufferSize: 8 * GB } }).maxBufferSize).toBe(8 * GB);
    const empty = buildCaps({ features: new Set(), limits: {} }, noLang);
    expect(limitsToRequest(empty)).toEqual({});
  });
});

describe('capsHash', () => {
  it('is stable, short hex, and sensitive to flags and limits but not adapter strings', () => {
    const a = detectCaps(mockAdapter(T1, T1_LIMITS).adapter, lang);
    const b = detectCaps(mockAdapter(T1, T1_LIMITS, { vendor: 'nvidia' }).adapter, lang);
    expect(capsHash(a)).toMatch(/^[0-9a-f]{8}$/);
    expect(capsHash(a)).toBe(capsHash(b));
    expect(capsHash(a)).toBe(capsHash(structuredClone(a)));
    const c = detectCaps(mockAdapter([...T1, 'subgroups'], T1_LIMITS).adapter, lang);
    const d = detectCaps(mockAdapter(T1, { maxStorageBufferBindingSize: 3 * GB }).adapter, lang);
    expect(new Set([capsHash(a), capsHash(c), capsHash(d)]).size).toBe(3);
  });
});

describe('deriveTier', () => {
  it('is pure over the flags', () => {
    const c = buildCaps({ features: new Set(T1), limits: { ...baseLimits, ...T1_LIMITS } }, lang);
    expect(deriveTier(c)).toBe(1);
    expect(deriveTier({ ...c, subgroups: true })).toBe(2);
  });
});
