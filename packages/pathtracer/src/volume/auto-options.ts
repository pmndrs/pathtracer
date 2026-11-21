// V6.5: a recommended grid setup (format, layout, filter, majorant) for a density grid, from its size, sparsity, the device limits and a memory budget.
// Pure TypeScript, no GPU: it reads `adapter.limits` / `adapter.features` values the caller passes in.
import { ATLAS_CELL, brickAtlasMemory, brickDims, brickMajorants, buildBrickAtlas, ptF16Bits, ptQuantizeGridR8, ptR8Bytes, ptRoundGridF16, type BrickAtlas } from './volume.ts';

export type PtGridFormat = 'r32float' | 'r16float' | 'r8unorm';
export type PtGridLayout = 'dense' | 'atlas';

export interface PtVolumeAutoGridInput {
  resolution: readonly [number, number, number];
  /** Dense density, x fastest, `resolution[0] * resolution[1] * resolution[2]` finite non-negative values. */
  density: Float32Array;
  limits: { maxTextureDimension3D: number; maxBufferSize?: number; [key: string]: number | undefined };
  /** Adapter / device features (`float32-filterable` decides hardware filtering of r32float). */
  features: ReadonlySet<string>;
  /** Bytes the grid may take (texture + majorant + indirection). Default: min(256 MiB, maxBufferSize). */
  budgetBytes?: number;
  /** `reference` (default): exact, manual filtering, the widest format that fits. `preview`: r16 / r8, atlas when sparse, hardware filtering. */
  accuracy?: 'reference' | 'preview';
  /** Brick occupancy below which the atlas is used (default 0.5; the atlas is worse than dense above ~0.6). */
  atlasOccupancy?: number;
  /** Also return the brick minima (for `shadowTracking: 'residual'`). */
  minima?: boolean;
  /** false: decide only, no prepared data (cheap for huge grids; `prepared` is then undefined). Default true. */
  prepare?: boolean;
}

export interface PtGridCandidate {
  format: PtGridFormat;
  layout: PtGridLayout;
  /** Texture dimensions of the density texture (the atlas size for `atlas`). */
  dims: [number, number, number];
  /** Density texture + indirection + majorant bytes. */
  bytes: number;
  fits: boolean;
  /** Why it was skipped, when it was. */
  note: string;
}

export interface PtGridPrepared {
  /** The values the GPU reads (f16-rounded / r8-dequantised, dense): compute `densityMax` from this. */
  density: Float32Array;
  densityMax: number;
  /** Texels to upload: Float32Array (r32float), Uint16Array half bits (r16float), Uint8Array (r8unorm); the atlas texels for `atlas`. */
  texels: Float32Array | Uint16Array | Uint8Array;
  /** r8unorm only: `PtMediumGrid.densityScale` and `volumeGrid.scale`. */
  scale?: number;
  atlas?: BrickAtlas;
  /** `majorant: 'brick'` data (`values` f32 per 8^3 cell, optional `minima`), absent for 'global'. */
  brick?: { resolution: [number, number, number]; values: Float32Array; minima?: Float32Array };
}

export interface PtVolumeAutoGrid {
  format: PtGridFormat;
  layout: PtGridLayout;
  densityFilter: 'manual' | 'hardware';
  majorant: 'brick' | 'global';
  /** Fraction of 8^3 bricks that hold a non-zero density. */
  occupancy: number;
  /** Density texture dimensions, and the bytes of the chosen setup. */
  dims: [number, number, number];
  bytes: number;
  budgetBytes: number;
  /** Every candidate considered, in the order tried. */
  candidates: PtGridCandidate[];
  reasons: string[];
  prepared?: PtGridPrepared;
}

const BYTES: Record<PtGridFormat, number> = { r32float: 4, r16float: 2, r8unorm: 1 };
const MIB = 1024 * 1024;

/** Atlas texture size in texels for a brick count (the `buildBrickAtlas` rule), or null when it does not fit the 10-bit coordinates. */
function atlasDims(occupied: number): [number, number, number] | null {
  const ax = Math.max(1, Math.ceil(Math.cbrt(occupied)));
  const az = Math.max(1, Math.ceil(occupied / (ax * ax)));
  if (ax > 1023 || az > 1023) return null;
  return [ax * ATLAS_CELL, ax * ATLAS_CELL, az * ATLAS_CELL];
}

export function ptVolumeAutoGrid(input: PtVolumeAutoGridInput): PtVolumeAutoGrid {
  const { resolution, density, limits, features } = input;
  const accuracy = input.accuracy ?? 'reference';
  const [nx, ny, nz] = resolution;
  if (!(nx > 0 && ny > 0 && nz > 0)) throw new RangeError('ptVolumeAutoGrid: resolution must be positive');
  if (density.length !== nx * ny * nz) throw new RangeError(`ptVolumeAutoGrid: density has ${density.length} values, resolution needs ${nx * ny * nz}`);
  const maxDim = limits.maxTextureDimension3D;
  const budget = input.budgetBytes ?? Math.min(256 * MIB, limits.maxBufferSize ?? Infinity);
  const atlasBelow = input.atlasOccupancy ?? 0.5;
  const reasons: string[] = [];

  const cells = brickDims(resolution);
  const total = cells[0] * cells[1] * cells[2];
  const mj = brickMajorants(resolution, density).values;
  let occupied = 0;
  for (let i = 0; i < total; i++) if ((mj[i] as number) > 0) occupied++;
  const occupancy = occupied / total;
  const sparse = occupancy < atlasBelow;
  const majorant: 'brick' | 'global' = nx * ny * nz >= 16 ** 3 ? 'brick' : 'global';
  const majorantBytes = majorant === 'brick' ? total * (input.minima ? 8 : 4) : 0;
  reasons.push(`${nx}x${ny}x${nz} grid, ${(occupancy * 100).toFixed(1)}% of ${total} bricks occupied (${sparse ? 'sparse: atlas allowed' : `atlas only below ${atlasBelow * 100}%`}), budget ${(budget / MIB).toFixed(0)} MiB`);

  const denseFits = nx <= maxDim && ny <= maxDim && nz <= maxDim;
  const aDims = atlasDims(occupied);
  const atlasFitsDims = aDims !== null && aDims.every((d) => d <= maxDim);
  const formats: PtGridFormat[] = accuracy === 'reference' ? ['r32float', 'r16float', 'r8unorm'] : ['r16float', 'r8unorm'];

  const candidates: PtGridCandidate[] = [];
  const make = (format: PtGridFormat, layout: PtGridLayout): PtGridCandidate => {
    const dims: [number, number, number] = layout === 'dense' ? [nx, ny, nz] : (aDims ?? [0, 0, 0]);
    const dimsOk = layout === 'dense' ? denseFits : atlasFitsDims;
    const bytes = dims[0] * dims[1] * dims[2] * BYTES[format] + (layout === 'atlas' ? total * 4 : 0) + majorantBytes;
    let note = '';
    if (!dimsOk) note = layout === 'dense' ? `a dimension exceeds maxTextureDimension3D ${maxDim}` : aDims === null ? 'brick count does not fit the 10-bit atlas coordinates' : `atlas ${aDims.join('x')} exceeds maxTextureDimension3D ${maxDim}`;
    else if (bytes > budget) note = `${(bytes / MIB).toFixed(1)} MiB over the ${(budget / MIB).toFixed(0)} MiB budget`;
    else if (layout === 'atlas' && !sparse && denseFits) note = `occupancy ${(occupancy * 100).toFixed(0)}% too high for the atlas`;
    return { format, layout, dims, bytes, fits: note === '', note };
  };

  // Order: reference = widest format first, dense before atlas; preview = atlas (when sparse) before dense, r16 before r8.
  for (const f of formats) {
    const order: PtGridLayout[] = accuracy === 'preview' && sparse ? ['atlas', 'dense'] : ['dense', 'atlas'];
    for (const l of order) candidates.push(make(f, l));
  }
  // The atlas is also the escape when the dense grid cannot be a texture at all.
  let pick = candidates.find((c) => c.fits);
  if (!pick && !denseFits) {
    pick = candidates.filter((c) => c.layout === 'atlas' && (c.layout === 'atlas' ? atlasFitsDims : true)).sort((a, b) => a.bytes - b.bytes)[0];
    if (pick) reasons.push(`no candidate fits the budget; the smallest that fits maxTextureDimension3D is ${pick.format} ${pick.layout}, ${(pick.bytes / MIB).toFixed(0)} MiB over budget`);
  }
  if (!pick) {
    // Over budget but dims fine: the smallest candidate that the device can create, with a loud reason.
    pick = candidates.filter((c) => c.layout === 'dense' ? denseFits : atlasFitsDims).sort((a, b) => a.bytes - b.bytes)[0];
    if (!pick) throw new RangeError(`ptVolumeAutoGrid: ${nx}x${ny}x${nz} exceeds maxTextureDimension3D ${maxDim} and its ${occupied} occupied bricks do not fit an atlas under that limit either (${aDims ? aDims.join('x') : 'too many bricks'}); resample the grid or split the medium`);
    reasons.push(`nothing fits the ${(budget / MIB).toFixed(0)} MiB budget: the smallest option is ${pick.format} ${pick.layout}, ${(pick.bytes / MIB).toFixed(0)} MiB (raise budgetBytes or lower the resolution)`);
  }

  const { format, layout } = pick;
  if (layout === 'atlas' && !denseFits) reasons.push(`dense grid exceeds maxTextureDimension3D ${maxDim}: using the atlas (${pick.dims.join('x')})`);
  else if (layout === 'atlas') reasons.push(`atlas: ${(pick.bytes / MIB).toFixed(1)} MiB against ${((BYTES[format] * nx * ny * nz + majorantBytes) / MIB).toFixed(1)} MiB dense at ${(occupancy * 100).toFixed(0)}% occupancy`);
  else reasons.push(`dense ${format}${accuracy === 'reference' && format === 'r32float' ? ' (exact)' : ''}: ${(pick.bytes / MIB).toFixed(1)} MiB`);
  const skipped = candidates.slice(0, candidates.indexOf(pick)).filter((c) => !c.fits);
  for (const c of skipped) reasons.push(`skipped ${c.format} ${c.layout}: ${c.note}`);
  if (format === 'r16float') reasons.push('r16float rounds densities to half precision (relative error 2^-11); the majorant is computed from the rounded grid, so it stays exact for what the GPU reads');
  if (format === 'r8unorm') reasons.push('r8unorm quantises to 255 levels of the grid maximum: a visible bias on smooth, high-dynamic-range density (dark wisps lose their tail); use only when memory forces it');

  let densityFilter: 'manual' | 'hardware' = 'manual';
  if (accuracy === 'preview') {
    if (format === 'r32float' && !features.has('float32-filterable')) reasons.push('manual filtering: r32float needs float32-filterable, which the device lacks');
    else { densityFilter = 'hardware'; reasons.push('hardware filtering (one textureSampleLevel; 8-bit filter weights, a small bias that is fine for a preview)'); }
  } else reasons.push('manual filtering (exact f32 trilinear)' + (format === 'r32float' && !features.has('float32-filterable') ? '; r32float hardware filtering is unavailable without float32-filterable anyway' : ''));
  if (majorant === 'global') reasons.push('global majorant: the grid is too small for a brick walk to pay');

  const out: PtVolumeAutoGrid = { format, layout, densityFilter, majorant, occupancy, dims: pick.dims, bytes: pick.bytes, budgetBytes: budget, candidates, reasons };
  if (input.prepare === false) return out;

  // Prepared data: the grid as the GPU reads it, then the majorants / atlas from THAT grid.
  let read: Float32Array = density;
  let scale: number | undefined;
  let bytes8: Uint8Array | undefined;
  if (format === 'r16float') read = ptRoundGridF16(density);
  else if (format === 'r8unorm') { const q = ptQuantizeGridR8(density); read = q.dequantized; scale = q.scale; bytes8 = q.bytes; }
  let densityMax = 0;
  for (let i = 0; i < read.length; i++) if ((read[i] as number) > densityMax) densityMax = read[i] as number;
  const prepared: PtGridPrepared = { density: read, densityMax, texels: read };
  if (scale !== undefined) prepared.scale = scale;
  let atlas: BrickAtlas | undefined;
  if (layout === 'atlas') {
    atlas = buildBrickAtlas(resolution, read);
    prepared.atlas = atlas;
    const mem = brickAtlasMemory(atlas, BYTES[format]);
    out.bytes = mem.atlasBytes + mem.indirectionBytes + majorantBytes;
  }
  const src = atlas ? atlas.data : read;
  if (format === 'r16float') prepared.texels = ptF16Bits(src, true);
  else if (format === 'r8unorm') prepared.texels = atlas ? ptR8Bytes(src, scale as number) : (bytes8 as Uint8Array);
  else prepared.texels = src;
  if (majorant === 'brick') prepared.brick = brickMajorants(resolution, read, { minima: input.minima === true });
  out.prepared = prepared;
  return out;
}

export interface PtAdapterInfoLike { vendor?: string; architecture?: string; device?: string; description?: string; backend?: string; platform?: string }

/** Bytes above which a media kernel risks multi-minute DXC compiles on Windows (the ~50 KB class of CLAUDE.md). */
export const PT_DXC_KERNEL_BYTES = 50_000;

/** True when the adapter looks like Windows / D3D12 (Chrome hides `description`, so also the optional `backend` / `platform` the caller can fill from `navigator.userAgent`). */
export function ptAdapterLooksD3d12(info: PtAdapterInfoLike | null | undefined): boolean {
  if (!info) return false;
  const text = [info.backend, info.platform, info.description, info.device].filter(Boolean).join(' ');
  return /d3d ?12|direct3d|dxgi|\bwin(dows|32|64)?\b/i.test(text);
}

/** A warning string when a media kernel is over ~50 KB on a D3D12 / Windows-like adapter (suggests the wavefront scheduler), else null. */
export function ptVolumeKernelBudgetNote(kernelBytes: number, adapterInfo?: PtAdapterInfoLike | null): string | null {
  if (kernelBytes <= PT_DXC_KERNEL_BYTES || !ptAdapterLooksD3d12(adapterInfo)) return null;
  return `media kernel is ${(kernelBytes / 1000).toFixed(1)} KB, over the ~50 KB class where Windows DXC (D3D12) compile time blows up; use scheduler: 'wavefront' (ptwfAutoPolicy does) or drop options (frame planes, outputs, alpha MASK) to shrink it`;
}
