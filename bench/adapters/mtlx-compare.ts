/**
 * P5 MaterialX checks 2 and 3 on rendered images: pairs of cells of one `matrix.ts --renderers pt-megakernel` report (default `mtlx-p5`).
 *
 *   node bench/adapters/mtlx-compare.ts [--report mtlx-p5] [--out file.md]
 *
 * Pairs: `mtlx-const-*` against its native `openpbr` twin (must be identical per pixel: same record, same sampler dimensions), then the
 * bake against codegen of one uv graph (bake 1024^2 and 256^2, A = codegen). Same seed per pixel, so the difference is the material only;
 * columns as in pt-wavefront-agreement.ts. Judge the bake rows by the mean and the fraction of pixels past 1e-2 / 5e-2, not by the max (an
 * edge pixel flips a texel).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { pixelAgreement } from './pt-wavefront-agreement.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const { values } = parseArgs({ options: { report: { type: 'string', default: 'mtlx-p5' }, out: { type: 'string' } } });
const runs = JSON.parse(readFileSync(join(REPO, 'bench/results/reports', values.report!, 'runs.json'), 'utf8')) as RunLogEntry[];
const load = (scene: string) => {
  const e = runs.find((r) => r.scene === scene);
  if (!e) return null;
  return readPFM(readFileSync(join(REPO, e.runDir, '1-run', `${e.renderer}__${e.scene}.pfm`)));
};

/** Fraction of pixels (rgb) whose relative difference exceeds `t` (floor 1e-3 of the image mean). */
function overFraction(a: ReturnType<typeof readPFM>, b: ReturnType<typeof readPFM>, t: number): number {
  const n = a.width * a.height;
  let mean = 0;
  for (let i = 0; i < n * 3; i++) mean += a.data[Math.floor(i / 3) * a.channels + (i % 3)]! / (n * 3);
  const floor = 1e-3 * Math.max(1e-6, mean);
  let over = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const va = a.data[i * a.channels + c]!;
      const vb = b.data[i * b.channels + c]!;
      if (Math.abs(vb - va) / Math.max(Math.abs(va), floor) > t) over++;
    }
  }
  return over / (n * 3);
}

const PAIRS: [string, string, string][] = [
  ['mtlx-const-open-pbr-ref', 'mtlx-const-open-pbr', 'constant open_pbr_surface vs native openpbr'],
  ['mtlx-const-std-ref', 'mtlx-const-std', 'constant standard_surface vs hand-mapped openpbr'],
  ['mtlx-uv-codegen', 'mtlx-uv-bake', 'uv graph: bake 1024^2 vs codegen'],
  ['mtlx-uv-codegen', 'mtlx-uv-bake-256', 'uv graph: bake 256^2 vs codegen'],
];
const lines = ['| A (reference side) | B | what | mean rel diff (r, g, b) | relMSE(B, A) | rms rel pixel | pixels > 1e-3 | > 1e-2 | > 5e-2 | max rel pixel | non-finite |', '|---|---|---|---|---|---|---|---|---|---|---|'];
for (const [a, b, what] of PAIRS) {
  const ia = load(a);
  const ib = load(b);
  if (!ia || !ib) {
    lines.push(`| ${a} | ${b} | ${what} | not in the report | | | | | | | |`);
    continue;
  }
  const r = pixelAgreement(ia, ib);
  lines.push(`| ${a} | ${b} | ${what} | ${r.meanRelDiff.map(sci).join(', ')} | ${sci(r.relMse)} | ${sci(r.rmsRelPixel)} | ${(r.over1e3 * 100).toFixed(2)}% | ${(overFraction(ia, ib, 1e-2) * 100).toFixed(2)}% | ${(overFraction(ia, ib, 5e-2) * 100).toFixed(2)}% | ${sci(r.maxRelPixel)} | ${r.nonFinite} |`);
}
console.log(lines.join('\n'));
if (values.out) writeFileSync(values.out, `${lines.join('\n')}\n`);
