// Citation-only provenance scan of a third-party clone (docs/internal/CLEAN-ROOM.md §3).
//
//   node research/tools/provenance-scan.ts <clone-dir> [--git] [--json]
//
// Looks for signs that a source is derived from NVIDIA NRD or another NVIDIA RTX SDK, WITHOUT showing the code: it prints
// file paths, token counts, matched identifiers and (with --git) the commits that introduced them (hash, date, subject).
// It never prints the text around a match, so running it does not expose anyone to the code it flags.
// Read only: it never runs, builds or writes anything in the clone. Exit code 0 = no hits, 1 = hits (review them), 2 = usage.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, extname } from 'node:path';
import { execFileSync } from 'node:child_process';

type Hit = { file: string; kind: string; token: string; count: number };

// Concept names: public in papers and talks (ReBLUR in Ray Tracing Gems II ch. 49, GTC 2020). A hit alone proves nothing.
const CONCEPT: Array<[string, RegExp]> = [
  ['NRD', /\bNRD\b/g],
  ['ReBLUR', /\breblur\b/gi],
  ['ReLAX', /\bReLAX\b/g],
  ['SIGMA', /\bSIGMA\b/g],
  ['RTXDI', /\bRTXDI\b/gi],
  ['RTXGI', /\bRTXGI\b/gi],
  ['RTXPT', /\bRTXPT\b/gi],
  ['SHaRC', /\bSHaRC\b|\bSHARC\b/g],
];
// Source pointers: identifiers, file names or URLs that only someone looking at the SDK's code or tree would write.
const POINTER: RegExp[] = [
  /\b(?:REBLUR|RELAX|SIGMA|NRD|RTXDI|RTXGI|SHARC|RTXPT)_[A-Za-z0-9_]+/g,
  /github\.com\/NVIDIA(?:-RTX|GameWorks)?\/(?:NRD|RTXDI|RTXGI|RTXPT|SHARC)[\w./-]*/gi,
  /\b[\w-]+\.hlsli?\b/g,
];
// Licence or copyright text that would come with copied files.
const HEADER: RegExp[] = [/NVIDIA CORPORATION/g, /NVIDIA Corporation/g, /RTX SDKs? Licen[cs]e/gi, /SPDX-FileCopyrightText:[^\n]*NVIDIA/g];

const SKIP_DIRS = new Set(['.git', 'node_modules', 'target', 'dist', 'build', 'out', '.cache']);
const BINARY = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.glb', '.bin', '.ktx2', '.exr', '.hdr', '.pfm', '.wasm', '.zip',
  '.7z', '.gz', '.pdf', '.mp4', '.webm', '.ttf', '.otf', '.woff', '.woff2', '.ico', '.dds', '.lock']);
const MAX_BYTES = 4 * 1024 * 1024;

function* walk(dir: string): Generator<string> {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.isSymbolicLink()) continue;
    const p = join(dir, e.name);
    if (e.isDirectory()) { if (!SKIP_DIRS.has(e.name)) yield* walk(p); }
    else if (e.isFile() && !BINARY.has(extname(e.name).toLowerCase()) && statSync(p).size <= MAX_BYTES) yield p;
  }
}

function count(text: string, re: RegExp): Map<string, number> {
  const out = new Map<string, number>();
  for (const m of text.matchAll(re)) out.set(m[0], (out.get(m[0]) ?? 0) + 1);
  return out;
}

const args = process.argv.slice(2);
const root = args.find((a) => !a.startsWith('--'));
if (!root) { console.error('usage: node research/tools/provenance-scan.ts <clone-dir> [--git] [--json]'); process.exit(2); }
const useGit = args.includes('--git');
const asJson = args.includes('--json');

const hits: Hit[] = [];
const hlslFiles: string[] = [];
for (const abs of walk(root)) {
  const file = relative(root, abs);
  const ext = extname(file).toLowerCase();
  if (ext === '.hlsl' || ext === '.hlsli') hlslFiles.push(file);
  let text: string;
  try { text = readFileSync(abs, 'utf8'); } catch { continue; }
  if (text.includes('\u0000')) continue;
  for (const [name, re] of CONCEPT) { const n = [...text.matchAll(re)].length; if (n) hits.push({ file, kind: 'concept', token: name, count: n }); }
  for (const re of POINTER) for (const [tok, n] of count(text, re)) {
    // A .hlsl file name only matters next to an SDK name in the same file.
    if (/\.hlsli?$/i.test(tok) && !hits.some((h) => h.file === file && h.kind === 'concept')) continue;
    hits.push({ file, kind: 'pointer', token: tok, count: n });
  }
  for (const re of HEADER) for (const [tok, n] of count(text, re)) hits.push({ file, kind: 'header', token: tok, count: n });
}

const pointers = [...new Set(hits.filter((h) => h.kind === 'pointer').map((h) => h.token))];
const introduced = new Map<string, string[]>();
if (useGit) {
  for (const tok of pointers) {
    try {
      const log = execFileSync('git', ['-C', root, 'log', '-S', tok, '--format=%h %ad %s', '--date=short'], { encoding: 'utf8' });
      introduced.set(tok, log.trim().split('\n').filter(Boolean).map((l) => l.slice(0, 160)));
    } catch { introduced.set(tok, ['(git log failed; shallow clone? try git fetch --unshallow)']); }
  }
}

if (asJson) {
  console.log(JSON.stringify({ root, hits, hlslFiles, pointers, introduced: Object.fromEntries(introduced) }, null, 2));
} else {
  const byKind = (k: string) => hits.filter((h) => h.kind === k);
  console.log(`Provenance scan of ${root} (citation only: no surrounding text is printed)\n`);
  const sec = (title: string, rows: Hit[]) => {
    console.log(`## ${title}: ${rows.length ? '' : 'none'}`);
    const files = [...new Set(rows.map((r) => r.file))].sort();
    for (const f of files) console.log(`  ${f}: ${rows.filter((r) => r.file === f).map((r) => `${r.token} x${r.count}`).join(', ')}`);
    console.log('');
  };
  sec('Licence / copyright headers (copied files?)', byKind('header'));
  sec('Source pointers (identifiers, SDK URLs, .hlsl names next to an SDK name)', byKind('pointer'));
  sec('Concept names (public; context needed)', byKind('concept'));
  console.log(`## HLSL files in the tree: ${hlslFiles.length ? '\n  ' + hlslFiles.join('\n  ') : 'none'}\n`);
  if (useGit) {
    console.log('## Commits that introduced each source pointer (git log -S, subject only)');
    for (const [tok, lines] of introduced) console.log(`  ${tok}:\n    ${lines.join('\n    ') || '(not found in history)'}`);
    console.log('');
  }
  console.log('Classify per docs/internal/CLEAN-ROOM.md §3: header or HLSL from an SDK = off limits; pointer = fence the area it touches;');
  console.log('concept only = usable, note it. Never open a flagged file to "check" it.');
}
process.exit(hits.length || hlslFiles.length ? 1 : 0);
