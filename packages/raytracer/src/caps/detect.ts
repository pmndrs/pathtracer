/**
 * Capability detection. `detectCaps` returns individual flags; `tier` is only a label (spec §4, note 06 §4).
 * Pipeline selection must read the flags, never the tier.
 *
 * Feature names are tried in the prefixed and unprefixed form so the code survives Chrome's renames.
 */

/** Limits we care about (the only ones we request raised). */
export const LIMIT_KEYS = [
  'maxStorageBufferBindingSize',
  'maxBufferSize',
  'maxStorageBuffersPerShaderStage',
  'maxSampledTexturesPerShaderStage',
  'maxTextureArrayLayers',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupStorageSize',
  'maxComputeWorkgroupsPerDimension',
] as const;
export type LimitKey = (typeof LIMIT_KEYS)[number];
export type CapsLimits = Record<LimitKey, number>;

/** Candidate feature strings per flag, preferred (current) name first. */
export const FEATURE_NAMES = {
  subgroups: ['subgroups', 'chromium-experimental-subgroups'],
  subgroupSizeControl: ['subgroup-size-control', 'chromium-experimental-subgroup-size-control'],
  shaderF16: ['shader-f16'],
  timestampQuery: ['timestamp-query'],
  samplingResourceTable: ['sampling-resource-table', 'chromium-experimental-sampling-resource-table'],
  heterogeneousResourceTable: ['heterogeneous-resource-table', 'chromium-experimental-heterogeneous-resource-table'],
  subgroupMatrix: ['subgroup-matrix', 'chromium-experimental-subgroup-matrix'],
  float32Filterable: ['float32-filterable'],
  textureFormatsTier1: ['texture-formats-tier1'],
  bc: ['texture-compression-bc'],
  bcSliced3d: ['texture-compression-bc-sliced-3d'],
  astc: ['texture-compression-astc'],
  etc2: ['texture-compression-etc2'],
  // Future ray tracing proposal; the name is not final, we only detect.
  accelerationStructure: ['acceleration-structure', 'chromium-experimental-acceleration-structure'],
} as const;
export type FeatureFlag = keyof typeof FEATURE_NAMES;

export const IMMEDIATES_LANGUAGE_FEATURE = 'immediate_address_space';

export type ResourceTableKind = 'none' | 'sampling' | 'heterogeneous';
export type Tier = 0 | 1 | 2 | 3;

export interface AdapterSummary {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
}

/** One entry of `GPUAdapterInfo.subgroupMatrixConfigs` (subgroup-matrix proposal): operand and result types and the tile shape. */
export interface CapsSubgroupMatrixConfig {
  componentType: string;
  resultComponentType: string;
  M: number;
  N: number;
  K: number;
}

export interface Caps {
  /** Label only (0-3). Derived with the exact rules of note 06 §4; see {@link deriveTier}. */
  tier: Tier;
  /** Tier 0 minimums met (128 MB storage binding, 16 KB workgroup storage, 8 storage buffers). */
  meetsBaseline: boolean;
  subgroups: boolean;
  subgroupSizeControl: boolean;
  shaderF16: boolean;
  timestampQuery: boolean;
  /** WGSL language feature `immediate_address_space`. */
  immediates: boolean;
  resourceTable: ResourceTableKind;
  subgroupMatrix: boolean;
  /**
   * The tile configurations the adapter offers for subgroup matrices (empty without the feature or when the browser does not report
   * them). Not part of `capsHash`. M5 Pro / Chrome 157: f32→f32 and f16→f16 at 8×8×8; Dawn Vulkan on an RTX 3060: f16 at 16×16×16.
   */
  subgroupMatrixConfigs: CapsSubgroupMatrixConfig[];
  float32Filterable: boolean;
  textureFormatsTier1: boolean;
  compressed: { bc: boolean; bcSliced3d: boolean; astc: boolean; etc2: boolean };
  accelerationStructure: boolean;
  limits: CapsLimits;
  /** 0 when the adapter does not report it. */
  subgroupMinSize: number;
  subgroupMaxSize: number;
  adapter: AdapterSummary;
  /** Sorted raw feature strings of the adapter (or device, once built from a device). */
  features: string[];
  /** Sorted WGSL language features of the browser. */
  wgslLanguageFeatures: string[];
  /** The exact feature string matched for each flag that is present; `requestDeviceForCaps` requests these. */
  featureNames: Partial<Record<FeatureFlag, string>>;
}

/** What detection reads; both GPUAdapter and GPUDevice satisfy it. */
export interface CapsSource {
  features: ReadonlySet<string>;
  limits: Record<string, unknown>;
  info?: Partial<Record<string, unknown>>;
}

const BASELINE = { storageBinding: 128 * 1024 * 1024, workgroupStorage: 16 * 1024, storageBuffers: 8 };
const TIER1 = { storageBinding: 1024 * 1024 * 1024, invocations: 256 };

export function defaultWgslLanguageFeatures(): ReadonlySet<string> {
  const gpu = (globalThis as { navigator?: { gpu?: { wgslLanguageFeatures?: ReadonlySet<string> } } }).navigator?.gpu;
  return gpu?.wgslLanguageFeatures ?? new Set<string>();
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** Build caps from any adapter-like or device-like source. `info` supplies adapter info when the source is a device. */
export function buildCaps(
  source: CapsSource,
  wgslLanguageFeatures: ReadonlySet<string> = defaultWgslLanguageFeatures(),
  info: Partial<Record<string, unknown>> | undefined = source.info,
): Caps {
  const featureNames: Partial<Record<FeatureFlag, string>> = {};
  for (const flag of Object.keys(FEATURE_NAMES) as FeatureFlag[]) {
    const hit = FEATURE_NAMES[flag].find((n) => source.features.has(n));
    if (hit) featureNames[flag] = hit;
  }
  const has = (f: FeatureFlag) => featureNames[f] !== undefined;

  const limits = {} as CapsLimits;
  for (const k of LIMIT_KEYS) limits[k] = num(source.limits[k]);

  const str = (k: string) => (typeof info?.[k] === 'string' ? (info[k] as string) : '');
  const resourceTable: ResourceTableKind = has('heterogeneousResourceTable')
    ? 'heterogeneous'
    : has('samplingResourceTable')
      ? 'sampling'
      : 'none';

  const partial = {
    meetsBaseline:
      limits.maxStorageBufferBindingSize >= BASELINE.storageBinding &&
      limits.maxComputeWorkgroupStorageSize >= BASELINE.workgroupStorage &&
      limits.maxStorageBuffersPerShaderStage >= BASELINE.storageBuffers,
    subgroups: has('subgroups'),
    subgroupSizeControl: has('subgroupSizeControl'),
    shaderF16: has('shaderF16'),
    timestampQuery: has('timestampQuery'),
    immediates: wgslLanguageFeatures.has(IMMEDIATES_LANGUAGE_FEATURE),
    resourceTable,
    subgroupMatrix: has('subgroupMatrix'),
    float32Filterable: has('float32Filterable'),
    textureFormatsTier1: has('textureFormatsTier1'),
    compressed: { bc: has('bc'), bcSliced3d: has('bcSliced3d'), astc: has('astc'), etc2: has('etc2') },
    accelerationStructure: has('accelerationStructure'),
    limits,
  };
  return {
    tier: deriveTier(partial),
    ...partial,
    subgroupMinSize: num(info?.subgroupMinSize),
    subgroupMaxSize: num(info?.subgroupMaxSize),
    subgroupMatrixConfigs: has('subgroupMatrix') ? readSubgroupMatrixConfigs(info?.subgroupMatrixConfigs) : [],
    adapter: {
      vendor: str('vendor'),
      architecture: str('architecture'),
      device: str('device'),
      description: str('description'),
    },
    features: [...source.features].sort(),
    wgslLanguageFeatures: [...wgslLanguageFeatures].sort(),
    featureNames,
  };
}

/**
 * Tier rules, exactly note 06 §4 (cumulative):
 * - T0: baseline (meetsBaseline). A device below baseline is still labelled 0; check `meetsBaseline`.
 * - T1: T0 + shader-f16 + timestamp-query + immediates + (bc | astc) + storage binding >= 1 GB + workgroup invocations >= 256.
 * - T2: T1 + subgroups.
 * - T3: T2 + a resource table (sampling or heterogeneous).
 */
export function deriveTier(
  c: Pick<Caps, 'meetsBaseline' | 'shaderF16' | 'timestampQuery' | 'immediates' | 'compressed' | 'limits' | 'subgroups' | 'resourceTable'>,
): Tier {
  const t1 =
    c.meetsBaseline &&
    c.shaderF16 &&
    c.timestampQuery &&
    c.immediates &&
    (c.compressed.bc || c.compressed.astc) &&
    c.limits.maxStorageBufferBindingSize >= TIER1.storageBinding &&
    c.limits.maxComputeInvocationsPerWorkgroup >= TIER1.invocations;
  if (!t1) return 0;
  if (!c.subgroups) return 1;
  if (c.resourceTable === 'none') return 2;
  return 3;
}

/** Detect capabilities of an adapter. `wgslLanguageFeatures` defaults to `navigator.gpu.wgslLanguageFeatures`. */
function readSubgroupMatrixConfigs(raw: unknown): CapsSubgroupMatrixConfig[] {
  if (!raw || typeof (raw as { length?: unknown }).length !== 'number') return [];
  return Array.from(raw as ArrayLike<Record<string, unknown>>, (c) => ({
    componentType: String(c.componentType),
    resultComponentType: String(c.resultComponentType),
    M: num(c.M),
    N: num(c.N),
    K: num(c.K),
  }));
}

export function detectCaps(adapter: GPUAdapter, wgslLanguageFeatures?: ReadonlySet<string>): Caps {
  return buildCaps(
    {
      features: adapter.features as ReadonlySet<string>,
      limits: adapter.limits as unknown as Record<string, unknown>,
      info: adapter.info as unknown as Partial<Record<string, unknown>> | undefined,
    },
    wgslLanguageFeatures,
  );
}
