import type { Caps } from './detect.ts';

/**
 * Stable 8-hex-digit hash (FNV-1a 32) of everything that changes generated code or pipeline behaviour:
 * flags, resource table kind, limits, subgroup sizes. Adapter strings and the tier label are excluded
 * (the tier is derived; bench rows carry adapter info separately).
 */
export function capsHash(caps: Caps): string {
  const c = caps.compressed;
  const key = [
    +caps.subgroups, +caps.subgroupSizeControl, +caps.shaderF16, +caps.timestampQuery, +caps.immediates,
    caps.resourceTable, +caps.subgroupMatrix, +caps.float32Filterable, +caps.textureFormatsTier1,
    +c.bc, +c.bcSliced3d, +c.astc, +c.etc2, +caps.accelerationStructure,
    caps.subgroupMinSize, caps.subgroupMaxSize,
    ...Object.keys(caps.limits).sort().map((k) => `${k}=${caps.limits[k as keyof typeof caps.limits]}`),
  ].join('|');
  let h = 0x811c9dc5;
  for (let i = 0; i < key.length; i++) {
    h ^= key.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}
