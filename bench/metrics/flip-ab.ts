/**
 * P0.14 A/B: LDR-FLIP (default, D18) vs HDR-FLIP on firefly / bias / noise variants of the Cornell scene.
 *
 *   node bench/metrics/flip-ab.ts [--name flip-ab] [--ref bench/references/cornell/mitsuba3-llvm_ad_rgb-4096spp.pfm]
 *
 * Inputs are existing renders in bench/results/incoming/ (git-ignored, this machine only): the latest tgpt-wavefront
 * and mutracer Cornell images at several spp. Synthetic variants are derived from the converged tgpt-wavefront 256 spp
 * image C: sparse additive fireflies (log-uniform 20..2000 radiance on 0.005 % / 0.02 % of the pixels, seeded) and
 * global gains (+3 %, +10 %). Writes bench/results/reports/<name>/{report.json, *.pfm} (open in the viewer with
 * ?report=/results/reports/<name>/report.json; pick the "flip" or "flipHdr" metric), prints the table and every pair of
 * variants the two FLIPs rank differently, and prints timings.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Image } from '../src/adapter.ts';
import { flipLdr } from './flip.ts';
import { flipHdrParallel } from './flip-parallel.ts';
import { compareImages } from './metrics.ts';
import { readPFM, writePFM } from './pfm.ts';

const REPO = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const arg = (n: string, d: string) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1]! : d; };
const name = arg('name', 'flip-ab');
const refPath = resolve(REPO, arg('ref', 'bench/references/cornell/mitsuba3-llvm_ad_rgb-4096spp.pfm'));
const incoming = join(REPO, 'bench/results/incoming');

/** Latest upload per image name (run ids sort by time). */
function latest(): Map<string, string> {
  const found = new Map<string, string>();
  for (const run of readdirSync(incoming).sort()) {
    const base = join(incoming, run);
    for (const sub of readdirSync(base, { withFileTypes: true })) {
      if (!sub.isDirectory()) continue;
      for (const f of readdirSync(join(base, sub.name))) if (f.endsWith('.pfm')) found.set(f.replace(/\.pfm$/, ''), join(base, sub.name, f));
    }
  }
  return found;
}

function rng(seed: number) { let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }

function withFireflies(img: Image, fraction: number, seed: number): Image {
  const data = img.data.slice();
  const r = rng(seed);
  const n = Math.round(img.width * img.height * fraction);
  for (let k = 0; k < n; k++) {
    const i = Math.floor(r() * img.width * img.height);
    const add = Math.exp(Math.log(20) + r() * (Math.log(2000) - Math.log(20)));
    for (let c = 0; c < 3; c++) data[i * img.channels + c] = data[i * img.channels + c]! + add;
  }
  return { ...img, data };
}
const withGain = (img: Image, g: number): Image => ({ ...img, data: img.data.map((v) => v * g) });

interface Variant { id: string; label: string; image: Image; kind: 'real' | 'synthetic' }

async function main() {
  const ref = readPFM(readFileSync(refPath));
  const files = latest();
  const load = (n: string): Image | undefined => { const p = files.get(n); if (!p) return undefined; const i = readPFM(readFileSync(p)); return i.width === ref.width && i.height === ref.height ? i : undefined; };
  const variants: Variant[] = [];
  const addReal = (n: string, label: string) => { const image = load(n); if (image) variants.push({ id: n.replace('cornell__', '').replace('__cornell', ''), label, image, kind: 'real' }); };
  for (const spp of [4, 16, 64, 256]) addReal(`tgpt-wavefront__cornell__spp${spp}`, `tgpt wavefront ${spp} spp`);
  for (const spp of [16, 256, 1024]) addReal(`mutracer__cornell__spp${spp}`, `mutracer ${spp} spp`);
  const clean = load('tgpt-wavefront__cornell__spp256');
  if (!clean) throw new Error('need a tgpt-wavefront__cornell__spp256 upload in bench/results/incoming');
  variants.push({ id: 'C+fireflies-0.005pct', label: 'tgpt 256 spp + 13 fireflies (0.005 %)', image: withFireflies(clean, 5e-5, 1), kind: 'synthetic' });
  variants.push({ id: 'C+fireflies-0.02pct', label: 'tgpt 256 spp + 52 fireflies (0.02 %)', image: withFireflies(clean, 2e-4, 2), kind: 'synthetic' });
  variants.push({ id: 'C+gain-3pct', label: 'tgpt 256 spp x 1.03 (biased)', image: withGain(clean, 1.03), kind: 'synthetic' });
  variants.push({ id: 'C+gain-10pct', label: 'tgpt 256 spp x 1.10 (biased)', image: withGain(clean, 1.1), kind: 'synthetic' });

  const outDir = join(REPO, 'bench/results/reports', name);
  mkdirSync(outDir, { recursive: true });
  const refName = 'reference.pfm';
  writeFileSync(join(outDir, refName), readFileSync(refPath));
  let tLdr = 0, tHdr = 0;
  const rows: Array<{ v: Variant; relMse: number; relBias: number; ff: number; ldr: number; hdr: number; maxLdr: number; maxHdr: number }> = [];
  const cells = [];
  for (const v of variants) {
    const m = compareImages(v.image, ref);
    let t = performance.now();
    const l = flipLdr(v.image, ref);
    tLdr += performance.now() - t;
    t = performance.now();
    const h = await flipHdrParallel(v.image, ref);
    tHdr += performance.now() - t;
    const mx = (a: Float32Array) => a.reduce((p, q) => (q > p ? q : p), 0);
    rows.push({ v, relMse: m.relMse, relBias: m.relBias, ff: m.fireflyPct, ldr: l.mean, hdr: h.mean, maxLdr: mx(l.map), maxHdr: mx(h.map) });
    writeFileSync(join(outDir, `${v.id}.pfm`), writePFM(v.image));
    cells.push({ scene: 'cornell', renderer: v.id, image: `${v.id}.pfm`, metrics: { relMse: m.relMse, relBias: m.relBias, psnr: m.psnr, fireflyPct: m.fireflyPct, flip: l.mean, flipHdr: h.mean } });
  }
  writeFileSync(join(outDir, 'report.json'), `${JSON.stringify({
    run: `${name}: LDR-FLIP vs HDR-FLIP on Cornell variants (reference: ${refPath.split('/').pop()})`,
    scenes: [{ id: 'cornell', name: 'cornell', reference: refName }],
    renderers: variants.map((v) => ({ id: v.id, name: v.label })),
    cells,
  }, null, 2)}\n`);

  const f = (x: number, d = 4) => x.toFixed(d);
  console.log(`reference ${refPath.split('/').pop()} ${ref.width}x${ref.height}`);
  console.log(['variant'.padEnd(26), 'kind'.padEnd(9), 'relMSE'.padStart(9), 'relBias'.padStart(8), 'ff %'.padStart(7), 'LDR-FLIP'.padStart(9), 'HDR-FLIP'.padStart(9), 'max LDR'.padStart(8), 'max HDR'.padStart(8)].join(' '));
  for (const r of rows) console.log([r.v.id.padEnd(26), r.v.kind.padEnd(9), r.relMse.toExponential(2).padStart(9), f(r.relBias).padStart(8), f(r.ff, 3).padStart(7), f(r.ldr).padStart(9), f(r.hdr).padStart(9), f(r.maxLdr, 3).padStart(8), f(r.maxHdr, 3).padStart(8)].join(' '));
  console.log('\npairs ranked differently (A is better by LDR-FLIP, B is better by HDR-FLIP):');
  let n = 0;
  for (const a of rows) for (const b of rows) {
    if (a === b || !(a.ldr < b.ldr) || !(b.hdr < a.hdr)) continue;
    n++;
    console.log(`  ${a.v.id} (LDR ${f(a.ldr)}, HDR ${f(a.hdr)}, relMSE ${a.relMse.toExponential(1)})  vs  ${b.v.id} (LDR ${f(b.ldr)}, HDR ${f(b.hdr)}, relMSE ${b.relMse.toExponential(1)})`);
  }
  if (!n) console.log('  none');
  console.log(`\ntime: LDR-FLIP ${(tLdr / rows.length / 1000).toFixed(2)} s/pair, HDR-FLIP ${(tHdr / rows.length / 1000).toFixed(2)} s/pair (worker_threads) at ${ref.width}x${ref.height}`);
  console.log(`report: ${join(outDir, 'report.json')}  (viewer ?report=/results/reports/${name}/report.json)`);
}

if (!existsSync(incoming)) throw new Error('bench/results/incoming is missing (git-ignored, machine local)');
await main();
