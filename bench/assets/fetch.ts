/**
 * Download the bench's external assets listed in MANIFEST.json into bench/assets/downloads/ (git-ignored) and verify
 * every file's sha256 and size. Existing files that already verify are kept.
 *
 *   node bench/assets/fetch.ts [--only <asset-id>] [--force] [--check]
 *
 * `--check` verifies what is on disk without downloading (exit code 1 on missing/mismatched files).
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

export interface ManifestFile { path: string; url: string; sha256: string; bytes: number }
export interface ManifestAsset { id: string; name: string; files: ManifestFile[]; usedBy?: string[]; licence?: unknown }
export interface Manifest { version: 1; assets: ManifestAsset[] }

export function loadManifest(): Manifest {
  return JSON.parse(readFileSync(join(HERE, 'MANIFEST.json'), 'utf8')) as Manifest;
}

const sha256 = (b: Uint8Array): string => createHash('sha256').update(b).digest('hex');

/** 'ok' when the file exists and matches, otherwise why not. */
export function verifyFile(f: ManifestFile): 'ok' | 'missing' | 'mismatch' {
  const p = join(HERE, f.path);
  if (!existsSync(p)) return 'missing';
  const b = readFileSync(p);
  return b.length === f.bytes && sha256(b) === f.sha256 ? 'ok' : 'mismatch';
}

async function download(f: ManifestFile): Promise<void> {
  const res = await fetch(f.url);
  if (!res.ok) throw new Error(`${f.url}: HTTP ${res.status}`);
  const buf = new Uint8Array(await res.arrayBuffer());
  const got = sha256(buf);
  if (got !== f.sha256 || buf.length !== f.bytes) {
    throw new Error(`${f.path}: downloaded ${buf.length} bytes sha256 ${got}, manifest says ${f.bytes} bytes sha256 ${f.sha256}`);
  }
  const dest = join(HERE, f.path);
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest + '.part', buf);
  renameSync(dest + '.part', dest);
}

export async function main(argv: string[]): Promise<number> {
  const only = argv.includes('--only') ? argv[argv.indexOf('--only') + 1] : undefined;
  const force = argv.includes('--force');
  const checkOnly = argv.includes('--check');
  let bad = 0;
  for (const asset of loadManifest().assets) {
    if (only && asset.id !== only) continue;
    for (const f of asset.files) {
      const state = verifyFile(f);
      if (state === 'ok' && !force) { console.log(`ok       ${f.path}`); continue; }
      if (checkOnly) { console.log(`${state.padEnd(8)} ${f.path}`); bad++; continue; }
      process.stdout.write(`fetch    ${f.path} (${(f.bytes / 1e6).toFixed(2)} MB) ... `);
      await download(f);
      console.log('verified');
    }
  }
  return bad ? 1 : 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  main(process.argv.slice(2)).then((c) => process.exit(c), (e) => { console.error(String(e)); process.exit(1); });
}
