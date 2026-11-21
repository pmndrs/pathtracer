/**
 * V1.4: host-side estimate of how close a medium comes to the hard loop caps of the tracking loops (see the table at the top of `volume-wgsl.ts`).
 * Expected tentative collisions along the longest segment = majorant x diagonal, where the majorant is `densityMax x max sigma_t` (`majorant: 'global'`) or
 * `max brick majorant x max sigma_t` (`'brick'`). A grid's diagonal comes from `worldToIndex` and the resolution; a homogeneous medium needs `sceneDiagonal`
 * (the boundary mesh size; skipped without it). The count is Poisson, so once expected / cap passes ~0.1 the tail matters.
 */
import type { PtVolumeOptions } from './volume.ts';

/** The free-flight / ratio-tracking collision cap of `volume-wgsl.ts` (`ptVolSample`, `ptVolTr*`: `for (i < 4096u)`). */
export const PT_VOLUME_COLLISION_CAP = 4096;
/** Warn when expected collisions exceed this fraction of the cap. */
export const PT_VOLUME_CAP_RISK_WARN = 0.1;

export interface PtVolumeCapRisk {
  /** Index into `volumes.media`. */
  medium: number;
  expectedCollisions: number;
  cap: number;
  /** expectedCollisions / cap. */
  risk: number;
}

/** World-space length of the index-space box diagonal of a grid (row-major 3x4 `worldToIndex`, resolution voxels). */
export function ptGridWorldDiagonal(resolution: readonly [number, number, number], worldToIndex: readonly number[]): number {
  const m = worldToIndex;
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]] as [number, number, number, number, number, number, number, number, number];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!(Math.abs(det) > 0)) return 0;
  // inverse of the 3x3 linear part, applied to the index extent (nx, ny, nz)
  const inv = [e * i - f * h, c * h - b * i, b * f - c * e, f * g - d * i, a * i - c * g, c * d - a * f, d * h - e * g, b * g - a * h, a * e - b * d].map((v) => v / det);
  const r = resolution as readonly number[];
  const w = [0, 1, 2].map((row) => (inv[row * 3] as number) * (r[0] as number) + (inv[row * 3 + 1] as number) * (r[1] as number) + (inv[row * 3 + 2] as number) * (r[2] as number));
  return Math.hypot(w[0] as number, w[1] as number, w[2] as number);
}

/** @param brickDensityMax the largest value of `volumeGrid.majorant.values` (density units) when `majorant: 'brick'`; falls back to the grid's `densityMax`. */
export function ptVolumeCapRisk(volumes: PtVolumeOptions, sceneDiagonal?: number, brickDensityMax?: number): PtVolumeCapRisk[] {
  const out: PtVolumeCapRisk[] = [];
  volumes.media.forEach((m, medium) => {
    const sigma = Math.max(m.sigmaT[0], m.sigmaT[1], m.sigmaT[2]);
    let diag: number | undefined;
    let dmax = 1;
    if (m.grid) {
      diag = ptGridWorldDiagonal(m.grid.resolution, m.grid.worldToIndex);
      dmax = volumes.majorant === 'brick' && brickDensityMax !== undefined ? brickDensityMax : m.grid.densityMax;
    } else diag = sceneDiagonal;
    if (diag === undefined || !(diag > 0)) return;
    const expectedCollisions = dmax * sigma * diag;
    out.push({ medium, expectedCollisions, cap: PT_VOLUME_COLLISION_CAP, risk: expectedCollisions / PT_VOLUME_COLLISION_CAP });
  });
  return out;
}

const warned = new Set<string>();

/** `console.warn` once per message when any medium's risk exceeds `PT_VOLUME_CAP_RISK_WARN`. Returns the risks. Never throws. */
export function ptWarnVolumeCapRisk(label: string, volumes: PtVolumeOptions | null | undefined, sceneDiagonal?: number, brickValues?: Float32Array): PtVolumeCapRisk[] {
  if (!volumes) return [];
  let bmax: number | undefined;
  if (brickValues) { bmax = 0; for (let i = 0; i < brickValues.length; i++) if ((brickValues[i] as number) > bmax) bmax = brickValues[i] as number; }
  const risks = ptVolumeCapRisk(volumes, sceneDiagonal, bmax);
  for (const r of risks) {
    if (r.risk <= PT_VOLUME_CAP_RISK_WARN) continue;
    const msg = `${label}: medium ${r.medium} expects ~${r.expectedCollisions.toFixed(0)} tentative collisions on its longest segment (${(r.risk * 100).toFixed(0)}% of the ${r.cap} loop cap): the cap can silently bias the image (free flight darkens, ratio tracking brightens). Use majorant: 'brick' or lower the density scale.`;
    if (warned.has(msg)) continue;
    warned.add(msg);
    console.warn(msg);
  }
  return risks;
}
