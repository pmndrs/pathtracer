/**
 * CLI: node bench/metrics/compare.ts a.pfm b.pfm [--ref r.pfm] [--out dir] [--tonemap aces] [--ev 0] [--no-flip]
 *
 * With --ref: prints a-vs-ref, b-vs-ref and a-vs-b. Without: a is the test and b the reference (metrics are
 * asymmetric). With --out: writes, per pair, abs diff, signed bias map, relative error (sqrt of the per-pixel relMSE
 * term) and the FLIP map as PFM, plus PNG previews (abs diff and previews tonemapped; relerr turbo log 0.1 %..100 %;
 * FLIP magma 0..1).
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import type { Image } from '../src/adapter.ts';
import { relErrToT } from './colormap.ts';
import { flipLdr } from './flip.ts';
import { absDiffMap, biasMap, compareImages, relMseMap, type PairMetrics } from './metrics.ts';
import { readPFM, scalarToImage, writePFM } from './pfm.ts';
import { imageToPNG, scalarToPNG } from './png.ts';
import { TONEMAPS, type TonemapName } from './tonemap.ts';

interface Args { files: string[]; ref?: string; out?: string; tonemap: TonemapName; ev: number; flip: boolean }

function parseArgs(argv: string[]): Args {
  const a: Args = { files: [], tonemap: 'aces', ev: 0, flip: true };
  for (let i = 0; i < argv.length; i++) {
    const s = argv[i]!;
    if (s === '--ref') a.ref = argv[++i];
    else if (s === '--out') a.out = argv[++i];
    else if (s === '--tonemap') {
      const t = argv[++i] as TonemapName;
      if (!TONEMAPS.includes(t)) throw new Error(`--tonemap must be one of ${TONEMAPS.join(', ')}`);
      a.tonemap = t;
    } else if (s === '--ev') a.ev = Number(argv[++i]);
    else if (s === '--no-flip') a.flip = false;
    else if (s === '-h' || s === '--help') { console.log('usage: node bench/metrics/compare.ts a.pfm b.pfm [--ref r.pfm] [--out dir] [--tonemap clamp|reinhard|aces|agx] [--ev N] [--no-flip]'); process.exit(0); }
    else if (s.startsWith('--')) throw new Error(`unknown option ${s}`);
    else a.files.push(s);
  }
  if (a.files.length !== 2) throw new Error('need exactly two images: a.pfm b.pfm (see --help)');
  return a;
}

const load = (p: string): Image => readPFM(readFileSync(p));
const f = (v: number, d = 5) => (Number.isFinite(v) ? (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(d)) : String(v));

function table(rows: Array<{ label: string; m: PairMetrics }>): string {
  const head = ['pair', 'MSE', 'relMSE', 'SMAPE', 'PSNR dB', 'bias', 'relBias', 'firefly %', 'FLIP', 'nonfinite'];
  const body = rows.map(({ label, m }) => [
    label, f(m.mse), f(m.relMse), f(m.smape), f(m.psnr, 2), f(m.bias), f(m.relBias), f(m.fireflyPct, 3),
    m.flip === undefined ? '-' : f(m.flip), `${m.nonFiniteTest}/${m.nonFiniteRef}`,
  ]);
  const w = head.map((h, i) => Math.max(h.length, ...body.map((r) => r[i]!.length)));
  const line = (r: string[]) => r.map((c, i) => (i === 0 ? c.padEnd(w[i]!) : c.padStart(w[i]!))).join('  ');
  return [line(head), line(w.map((n) => '-'.repeat(n))), ...body.map(line)].join('\n');
}

export function main(argv: string[]): void {
  const args = parseArgs(argv);
  const [pa, pb] = args.files as [string, string];
  const A = load(pa), B = load(pb);
  const R = args.ref ? load(args.ref) : undefined;
  const pairs: Array<{ label: string; test: Image; ref: Image }> = R
    ? [
        { label: 'a vs ref', test: A, ref: R },
        { label: 'b vs ref', test: B, ref: R },
        { label: 'a vs b', test: A, ref: B },
      ]
    : [{ label: 'a vs b(ref)', test: A, ref: B }];
  if (args.out) mkdirSync(args.out, { recursive: true });
  const rows: Array<{ label: string; m: PairMetrics }> = [];
  for (const p of pairs) {
    const m = compareImages(p.test, p.ref, { psnrTonemap: args.tonemap, psnrEv: args.ev });
    let flipMap: Float32Array | undefined;
    if (args.flip) {
      const r = flipLdr(p.test, p.ref, { tonemap: args.tonemap, ev: args.ev });
      m.flip = r.mean; flipMap = r.map;
    }
    rows.push({ label: p.label, m });
    if (args.out) {
      const tag = p.label.replace(/[^a-z0-9]+/gi, '_').replace(/_+$/, '');
      const { width: w, height: h } = p.test;
      const abs = absDiffMap(p.test, p.ref);
      const rel = relMseMap(p.test, p.ref).map(Math.sqrt);
      const write = (name: string, bytes: Uint8Array) => writeFileSync(join(args.out!, `${tag}_${name}`), bytes);
      write('absdiff.pfm', writePFM(abs));
      write('absdiff.png', imageToPNG(abs, args.tonemap, args.ev));
      write('bias.pfm', writePFM(biasMap(p.test, p.ref)));
      write('relerr.pfm', writePFM(scalarToImage(w, h, rel)));
      write('relerr.png', scalarToPNG(w, h, rel, 'turbo', relErrToT));
      if (flipMap) {
        write('flip.pfm', writePFM(scalarToImage(w, h, flipMap)));
        write('flip.png', scalarToPNG(w, h, flipMap, 'magma', (x) => x));
      }
    }
  }
  console.log(`a=${basename(pa)} b=${basename(pb)}${args.ref ? ` ref=${basename(args.ref)}` : ''}  ${A.width}x${A.height}`);
  console.log(`PSNR/FLIP display: tonemap=${args.tonemap} ev=${args.ev}; relMSE eps=1e-2; firefly = lum > 10x max(ref lum, 1e-2)`);
  console.log(table(rows));
  if (args.out) console.log(`maps written to ${args.out}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { main(process.argv.slice(2)); } catch (e) { console.error((e as Error).message); process.exit(1); }
}
