/**
 * P5 colour round trip: reads a `matrix.ts` report of the `spectral-swatches-*` scenes (scenes/ab/spectral-swatches.ts) and prints, per cell, the expected RGB, the
 * rendered mean over the cell's central 40 x 40 pixels and the error. Exit code 1 when a cell is off by more than `--tol` (default 0.012 absolute + 1.2% relative).
 *   node bench/adapters/spectral-swatches.ts --report <name> [--tol 0.012]
 */
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { readPFM } from '../metrics/pfm.ts';
import { SWATCH_COLORS, SWATCH_EMIT_SCALE, SWATCH_GRID } from '../scenes/ab/spectral-swatches.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const { values } = parseArgs({ options: { report: { type: 'string' }, tol: { type: 'string' } } });
if (!values.report) throw new Error('usage: spectral-swatches.ts --report <name>');
const tol = Number(values.tol ?? 0.012);
const runs = JSON.parse(readFileSync(join(REPO, 'bench/results/reports', values.report, 'runs.json'), 'utf8')) as RunLogEntry[];
let worst = 0;
let failed = 0;
for (const e of runs) {
  if (!e.scene.startsWith('spectral-swatches')) continue;
  const img = readPFM(readFileSync(join(REPO, e.runDir, '1-run', `${e.renderer}__${e.scene}.pfm`)));
  console.log(`\n${e.scene} (${e.spp} spp)`);
  if (e.scene.endsWith('-env')) {
    const m = [0, 0, 0];
    for (let i = 0; i < img.width * img.height; i++) for (let c = 0; c < 3; c++) m[c]! += (img.data[i * img.channels + c] as number) / (img.width * img.height);
    console.log(`  sky (2.4, 0.8, 0.05): mean ${m.map((v) => v.toFixed(4)).join(' ')}`);
    continue;
  }
  const scale = e.scene.endsWith('-emit') ? SWATCH_EMIT_SCALE : 1;
  const { cols, cell } = SWATCH_GRID;
  SWATCH_COLORS.forEach((c, i) => {
    const x0 = (i % cols) * cell + 12;
    const y0 = Math.floor(i / cols) * cell + 12;
    const m = [0, 0, 0];
    for (let y = 0; y < 40; y++) for (let x = 0; x < 40; x++) for (let k = 0; k < 3; k++) m[k]! += (img.data[((y0 + y) * img.width + x0 + x) * img.channels + k] as number) / 1600;
    const exp = c.map((v) => v * scale);
    const err = m.map((v, k) => v - (exp[k] as number));
    const bad = err.some((d, k) => Math.abs(d) > tol * Math.max(1, scale) + 0.012 * (exp[k] as number));
    worst = Math.max(worst, ...err.map((d, k) => Math.abs(d) / Math.max(1e-3, exp[k] as number >= 0.1 ? (exp[k] as number) : 1)));
    if (bad) failed++;
    console.log(`  ${String(i).padStart(2)} exp ${exp.map((v) => v.toFixed(3)).join(' ')} got ${m.map((v) => v.toFixed(3)).join(' ')} err ${err.map((d) => (d >= 0 ? '+' : '') + d.toFixed(4)).join(' ')}${bad ? '  <-- OFF' : ''}`);
  });
}
console.log(`\nworst relative error (channels >= 0.1) or absolute: ${worst.toFixed(4)}; cells off: ${failed}`);
process.exit(failed > 0 ? 1 : 0);
