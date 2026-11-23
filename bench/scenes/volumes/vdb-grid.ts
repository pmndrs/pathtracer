/**
 * OpenVDB float grid -> a dense density file for a `grid` medium (P5, Node only). Uses WebVDB's `packages/vdb` tools (`parseVdb`, pure TypeScript; the sibling checkout's `dist/`,
 * default `~/Developer/WebVDB`, override with `WEBVDB_DIR`), averages `downsample`^3 voxel blocks (inactive voxels are the background, 0), pads `border` zero cells (pbrt reads zero
 * outside a grid, Mitsuba clamps: a zero border makes them agree) and scales the density to a peak of 1.
 *
 *   node bench/scenes/volumes/vdb-grid.ts [<file.vdb> <out-name> [<downsample>]]     default: WebVDB's fixtures/vdb-samples/smoke.vdb -> bench/assets/generated/volumes/vdb-smoke.{f32,json}
 *
 * The output is a derivative of the OpenVDB sample model (CC BY-SA 4.0, THIRD_PARTY): generated, git-ignored, never committed.
 * Index space: OpenVDB voxel centres sit at integer indices, world = index * voxel size, so a cell block starting at index `i0` spans `[(i0 - 0.5) vs, (i0 + d - 0.5) vs]`.
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export interface VdbDense {
  resolution: [number, number, number];
  /** Medium-space box (metres at the VDB's own scale, x right, y up as authored). */
  bounds: [[number, number, number], [number, number, number]];
  density: Float32Array;
  /** The largest value before normalisation. */
  peak: number;
  source: { file: string; sha256: string; grid: string; voxelSize: number; activeVoxels: number };
}

const HERE = dirname(fileURLToPath(import.meta.url));
/** A plain Node `import()` (Vitest's module graph would try to resolve a file outside the repository). */
const nodeImport = new Function('p', 'return import(p)') as (p: string) => Promise<unknown>;

export const GENERATED_DIR = resolve(HERE, '..', '..', 'assets', 'generated', 'volumes');

export async function convertVdb(file: string, options: { downsample?: number; border?: number; grid?: string } = {}): Promise<VdbDense> {
  const down = options.downsample ?? 2;
  const border = options.border ?? 2;
  const webvdb = process.env.WEBVDB_DIR ?? join(homedir(), 'Developer', 'WebVDB');
  const tools = (await nodeImport(pathToFileURL(join(webvdb, 'packages', 'vdb', 'dist', 'tools', 'index.js')).href)) as { parseVdb(b: ArrayBuffer): { grids: Array<{ name: string; gridType: string; indexBBox: { min: number[]; max: number[] } | null; activeVoxelCount: bigint; transform: { voxelSize: number[] }; iterLeaves(): Iterable<{ origin: number[]; values: Float32Array; valueMask: Uint32Array }> }> } };
  const bytes = readFileSync(file);
  const vdb = tools.parseVdb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  const grid = vdb.grids.find((g) => (options.grid ? g.name === options.grid : g.gridType.startsWith('Tree_float'))) ;
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
  return { resolution: res, bounds: [lo as [number, number, number], hi as [number, number, number]], density, peak, source: { file, sha256: createHash('sha256').update(bytes).digest('hex'), grid: grid.name, voxelSize: vs, activeVoxels: Number(grid.activeVoxelCount) } };
}

async function main(): Promise<void> {
  const file = process.argv[2] ?? join(process.env.WEBVDB_DIR ?? join(homedir(), 'Developer', 'WebVDB'), 'fixtures', 'vdb-samples', 'smoke.vdb');
  const name = process.argv[3] ?? 'vdb-smoke';
  const down = process.argv[4] ? Number(process.argv[4]) : 2;
  const out = await convertVdb(file, { downsample: down });
  mkdirSync(GENERATED_DIR, { recursive: true });
  writeFileSync(join(GENERATED_DIR, `${name}.f32`), Buffer.from(out.density.buffer, out.density.byteOffset, out.density.byteLength));
  const meta = { name, resolution: out.resolution, bounds: out.bounds, peakBeforeNormalisation: out.peak, source: out.source };
  writeFileSync(join(GENERATED_DIR, `${name}.json`), `${JSON.stringify(meta, null, 2)}\n`);
  console.log(JSON.stringify(meta));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) await main();
