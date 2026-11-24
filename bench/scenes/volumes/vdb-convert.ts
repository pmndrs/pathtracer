/**
 * OpenVDB float grid -> dense density (V3.4). Browser safe: no node:fs, no WebVDB import. The caller parses the file with `parseVdb` of WebVDB's `tools` entry (Node, or a Web
 * Worker: `vdb-worker.ts`) and hands the parsed file here; `vdb-grid.ts` (Node) and `vdb-browser.ts` (browser) both call `convertVdbFile`, so they produce the same Float32 grid.
 *
 * `downsample`^3 voxel blocks are averaged (inactive voxels are the background, 0), `border` zero cells are padded (pbrt reads zero outside a grid, Mitsuba clamps: a zero border makes
 * them agree) and the density is scaled to a peak of 1. Index space: OpenVDB voxel centres sit at integer indices, world = index * voxel size, so a cell block starting at index `i0`
 * spans `[(i0 - 0.5) vs, (i0 + d - 0.5) vs]`.
 */
export interface VdbDense {
  resolution: [number, number, number];
  /** Medium-space box (metres at the VDB's own scale, x right, y up as authored). */
  bounds: [[number, number, number], [number, number, number]];
  density: Float32Array;
  /** The largest value before normalisation. */
  peak: number;
  source: { file: string; sha256: string; grid: string; voxelSize: number; activeVoxels: number };
}

/** The part of WebVDB's parsed grid the conversion reads. */
export interface VdbGridLike {
  name: string;
  gridType: string;
  indexBBox: { min: number[]; max: number[] } | null;
  activeVoxelCount: bigint;
  transform: { voxelSize: number[] };
  iterLeaves(): Iterable<{ origin: number[]; values: Float32Array; valueMask: Uint32Array }>;
}
export interface VdbFileLike {
  grids: VdbGridLike[];
}
export interface VdbConvertOptions {
  downsample?: number;
  border?: number;
  /** Grid name; default the first float tree. */
  grid?: string;
}

/** `file` and `sha256` only label `source`. */
export function convertVdbFile(vdb: VdbFileLike, file: string, sha256: string, options: VdbConvertOptions = {}): VdbDense {
  const down = options.downsample ?? 2;
  const border = options.border ?? 2;
  const grid = vdb.grids.find((g) => (options.grid ? g.name === options.grid : g.gridType.startsWith('Tree_float')));
  if (!grid || !grid.indexBBox) throw new Error(`${file}: no float grid${options.grid ? ` named ${options.grid}` : ''} with an index bounding box`);
  const { min, max } = grid.indexBBox;
  const dim = [0, 1, 2].map((a) => (max[a] as number) - (min[a] as number) + 1);
  const cells = dim.map((d) => Math.ceil(d / down));
  const res = cells.map((c) => c + 2 * border) as [number, number, number];
  const sums = new Float64Array(res[0] * res[1] * res[2]);
  // leaves are 8^3 blocks: value at (x, y, z) = values[(x & 7) * 64 + (y & 7) * 8 + (z & 7)] in OpenVDB's leaf order (x slowest); inactive voxels hold the background and are skipped by the mask
  for (const leaf of grid.iterLeaves()) {
    for (let lx = 0; lx < 8; lx++) for (let ly = 0; ly < 8; ly++) for (let lz = 0; lz < 8; lz++) {
      const li = lx * 64 + ly * 8 + lz;
      if (!(((leaf.valueMask[li >> 5] as number) >>> (li & 31)) & 1)) continue;
      const v = leaf.values[li] as number;
      if (!(v > 0)) continue;
      const i = Math.floor(((leaf.origin[0] as number) + lx - (min[0] as number)) / down) + border;
      const j = Math.floor(((leaf.origin[1] as number) + ly - (min[1] as number)) / down) + border;
      const k = Math.floor(((leaf.origin[2] as number) + lz - (min[2] as number)) / down) + border;
      if (i < border || j < border || k < border || i >= res[0] - border || j >= res[1] - border || k >= res[2] - border) continue;
      sums[i + res[0] * (j + res[1] * k)]! += v;
    }
  }
  let peak = 0;
  const density = new Float32Array(sums.length);
  for (let i = 0; i < sums.length; i++) {
    density[i] = (sums[i] as number) / down ** 3;
    peak = Math.max(peak, density[i] as number);
  }
  for (let i = 0; i < density.length; i++) density[i] = (density[i] as number) / peak;
  const vs = grid.transform.voxelSize[0] as number;
  const lo = [0, 1, 2].map((a) => ((min[a] as number) - 0.5 - border * down) * vs);
  const hi = [0, 1, 2].map((a, idx) => (lo[idx] as number) + (res[idx] as number) * down * vs);
  return { resolution: res, bounds: [lo as [number, number, number], hi as [number, number, number]], density, peak, source: { file, sha256, grid: grid.name, voxelSize: vs, activeVoxels: Number(grid.activeVoxelCount) } };
}
