/**
 * OpenVDB float grid -> a dense density file for a `grid` medium (P5, Node only). Uses WebVDB's `@pmndrs/vdb/tools` (`parseVdb`, pure TypeScript; the sibling checkout's `packages/vdb/dist/tools`,
 * default `~/Developer/WebVDB`, override with `WEBVDB_DIR`); the conversion itself is `vdb-convert.ts` (browser safe, shared with `vdb-browser.ts`).
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
import { convertVdbFile, type VdbConvertOptions, type VdbDense, type VdbFileLike } from './vdb-convert.ts';

export type { VdbDense } from './vdb-convert.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
/** A plain Node `import()` (Vitest's module graph would try to resolve a file outside the repository). */
const nodeImport = new Function('p', 'return import(p)') as (p: string) => Promise<unknown>;

export const GENERATED_DIR = resolve(HERE, '..', '..', 'assets', 'generated', 'volumes');

export async function convertVdb(file: string, options: VdbConvertOptions = {}): Promise<VdbDense> {
  const webvdb = process.env.WEBVDB_DIR ?? join(homedir(), 'Developer', 'WebVDB');
  const tools = (await nodeImport(pathToFileURL(join(webvdb, 'packages', 'vdb', 'dist', 'tools', 'index.js')).href)) as { parseVdb(b: ArrayBuffer): VdbFileLike };
  const bytes = readFileSync(file);
  const vdb = tools.parseVdb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
  return convertVdbFile(vdb, file, createHash('sha256').update(bytes).digest('hex'), options);
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
