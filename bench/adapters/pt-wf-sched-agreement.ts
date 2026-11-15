/**
 * P3.3 / P3.4 / P3.10 / R1b: every wavefront scheduling variant against the megakernel (or any `--ref` renderer), per pixel, in ONE report.
 *
 *   node bench/adapters/pt-wf-sched-agreement.ts --report <name> [--ref pt-megakernel] [--tol 1e-4] [--out file.md]
 *
 * The report comes from `matrix.ts --renderers pt-megakernel,pt-wf-sg,pt-wf-sort-both,...`. All variants run the same sampler dimensions per (pixel, sample), so
 * the images must agree to float summation order (the bar of `pt-wavefront-agreement.ts`: max relative pixel difference <= 7e-5, mostly 1e-6); a
 * sample run twice or never shows as a large per-pixel difference. Exit code 1 when a row exceeds `--tol` (max relative pixel difference) or has non-finite pixels.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { readPFM } from '../metrics/pfm.ts';
import { pixelAgreement } from './pt-wavefront-agreement.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const load = (entry: RunLogEntry): Image => readPFM(readFileSync(join(REPO, entry.runDir, '1-run', `${entry.renderer}__${entry.scene}.pfm`)));

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { report: { type: 'string' }, ref: { type: 'string' }, tol: { type: 'string' }, out: { type: 'string' } } });
  if (!values.report) throw new Error('usage: pt-wf-sched-agreement.ts --report <name> [--ref pt-megakernel] [--tol 1e-4] [--out file.md]');
  const ref = values.ref ?? 'pt-megakernel';
  const tol = Number(values.tol ?? '1e-4');
  const runs = JSON.parse(readFileSync(join(REPO, 'bench/results/reports', values.report, 'runs.json'), 'utf8')) as RunLogEntry[];
  const last = new Map<string, RunLogEntry>();
  for (const e of runs) last.set(`${e.scene}\0${e.renderer}`, e);
  const lines = [`| scene | renderer (vs ${ref}) | mean rel diff (r, g, b) | relMSE | max rel pixel | rms rel pixel | pixels > 1e-3 | non-finite | verdict |`, '|---|---|---|---|---|---|---|---|---|'];
  let bad = 0;
  for (const [key, e] of last) {
    if (e.renderer === ref) continue;
    const a = last.get(`${e.scene}\0${ref}`);
    if (!a) continue;
    const r = pixelAgreement(load(a), load(e));
    const ok = r.maxRelPixel <= tol && r.nonFinite === 0;
    if (!ok) bad++;
    void key;
    lines.push(`| ${e.scene} | ${e.renderer} | ${r.meanRelDiff.map(sci).join(', ')} | ${sci(r.relMse)} | ${sci(r.maxRelPixel)} | ${sci(r.rmsRelPixel)} | ${(r.over1e3 * 100).toFixed(3)}% | ${r.nonFinite} | ${ok ? 'ok' : 'DIFFERS'} |`);
  }
  console.log(lines.join('\n'));
  if (values.out) writeFileSync(values.out, `${lines.join('\n')}\n`);
  if (bad > 0) process.exitCode = 1;
}
