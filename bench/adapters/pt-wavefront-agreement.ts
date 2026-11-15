/**
 * P3.1: wavefront vs megakernel, image against image. Same sampler dimensions and the same BSDF / light code, so the two estimators are
 * identical in distribution and (up to float summation order and queue-order independent maths) per pixel.
 *
 *   node bench/adapters/pt-wavefront-agreement.ts --a <report of pt-megakernel> --b <report of pt-wavefront> [--out file.md]
 *
 * For every scene both reports hold: mean per channel (relative difference), relMSE(b, a), the largest and the rms per-pixel difference
 * relative to the pixel (floor 1e-3 of the image mean), how many pixels differ by more than 1e-3 relative, non-finite pixels.
 * Reports come from `node bench/adapters/matrix.ts --renderers pt-megakernel` / `pt-wavefront` (runs.json lists the final PFMs).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const load = (entry: RunLogEntry): Image => readPFM(readFileSync(join(REPO, entry.runDir, '1-run', `${entry.renderer}__${entry.scene}.pfm`)));
const runs = (report: string): RunLogEntry[] => JSON.parse(readFileSync(join(REPO, 'bench/results/reports', report, 'runs.json'), 'utf8')) as RunLogEntry[];

export interface PixelAgreement {
  meanRelDiff: number[];
  relMse: number;
  maxRelPixel: number;
  rmsRelPixel: number;
  over1e3: number;
  pixels: number;
  nonFinite: number;
}

/** Per-pixel comparison of `b` (wavefront) against `a` (megakernel). */
export function pixelAgreement(a: Image, b: Image): PixelAgreement {
  const n = a.width * a.height;
  const mean = [0, 0, 0];
  const meanB = [0, 0, 0];
  let nonFinite = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const va = a.data[i * a.channels + c]!;
      const vb = b.data[i * b.channels + c]!;
      if (!Number.isFinite(va) || !Number.isFinite(vb)) { nonFinite++; continue; }
      mean[c]! += va / n;
      meanB[c]! += vb / n;
    }
  }
  const floor = 1e-3 * Math.max(1e-6, (mean[0]! + mean[1]! + mean[2]!) / 3);
  let max = 0, sq = 0, over = 0, used = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const va = a.data[i * a.channels + c]!;
      const vb = b.data[i * b.channels + c]!;
      if (!Number.isFinite(va) || !Number.isFinite(vb)) continue;
      const d = Math.abs(vb - va) / Math.max(Math.abs(va), floor);
      if (d > max) max = d;
      sq += d * d;
      if (d > 1e-3) over++;
      used++;
    }
  }
  return { meanRelDiff: mean.map((m, c) => (m > 0 ? meanB[c]! / m - 1 : 0)), relMse: compareImages(b, a).relMse, maxRelPixel: max, rmsRelPixel: Math.sqrt(sq / Math.max(1, used)), over1e3: over / Math.max(1, used), pixels: n, nonFinite };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { a: { type: 'string' }, b: { type: 'string' }, out: { type: 'string' } } });
  if (!values.a || !values.b) throw new Error('usage: pt-wavefront-agreement.ts --a <megakernel report> --b <wavefront report> [--out file.md]');
  const rb = runs(values.b);
  const lines = ['| scene | spp | mean rel diff (r, g, b) | relMSE(wf, mk) | max rel pixel | rms rel pixel | pixels > 1e-3 | non-finite |', '|---|---|---|---|---|---|---|---|'];
  for (const ea of runs(values.a)) {
    const eb = rb.find((e) => e.scene === ea.scene);
    if (!eb) continue;
    const r = pixelAgreement(load(ea), load(eb));
    lines.push(`| ${ea.scene} | ${ea.spp} | ${r.meanRelDiff.map(sci).join(', ')} | ${sci(r.relMse)} | ${sci(r.maxRelPixel)} | ${sci(r.rmsRelPixel)} | ${(r.over1e3 * 100).toFixed(3)}% | ${r.nonFinite} |`);
  }
  console.log(lines.join('\n'));
  if (values.out) writeFileSync(values.out, `${lines.join('\n')}\n`);
}
