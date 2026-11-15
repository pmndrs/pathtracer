/**
 * Paired ms/spp ratios of an interleaved matrix report (QUIET-PASS reports).
 *
 *   node bench/adapters/paired-ratios.ts --report <name> [--base pt-megakernel] [--base-map pt-mk-vbuffer=pt-megakernel,...] [--md]
 *
 * Reads `bench/results/reports/<name>/runs.json`, and for every (scene, renderer) cell the per-round `ms_per_spp`, the runner's load
 * (`load1`) and `loaded` flag from `<runDir>/run.json`. A ratio is cell / base cell of the same round (round i against round i),
 * printed as min..max over the rounds. `--base-map` overrides the base per renderer (the same scheduler's exact policy, say).
 * Prints ms/spp min..max, the paired ratio, rounds marked loaded / invalid, and the load range. Never mixes reports.
 */
import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

interface Round { ms: number; load: number; loaded: boolean; valid: boolean }
export interface Cell { scene: string; renderer: string; rounds: Round[] }

export function readCells(report: string): Cell[] {
  const runs = JSON.parse(readFileSync(join(REPO, 'bench/results/reports', report, 'runs.json'), 'utf8')) as { scene: string; renderer: string; runDir: string }[];
  return runs.map((e) => {
    const rj = JSON.parse(readFileSync(join(REPO, e.runDir, 'run.json'), 'utf8')) as { runs: { result?: { valid?: boolean; metrics?: Record<string, number> }; load?: { load1?: number; loaded?: boolean } }[] };
    const rounds = rj.runs.map((r) => ({ ms: r.result?.metrics?.ms_per_spp ?? NaN, load: r.load?.load1 ?? NaN, loaded: !!r.load?.loaded, valid: r.result?.valid !== false }));
    return { scene: e.scene, renderer: e.renderer, rounds };
  });
}

export const range = (a: number[], d = 3): string => (a.length ? `${Math.min(...a).toFixed(d)}..${Math.max(...a).toFixed(d)}` : '-');

export function ratios(cell: Cell, base: Cell): number[] {
  return cell.rounds.flatMap((r, i) => (base.rounds[i] && Number.isFinite(r.ms) && Number.isFinite(base.rounds[i]!.ms) ? [r.ms / base.rounds[i]!.ms] : []));
}

function main(): void {
  const { values } = parseArgs({ options: { report: { type: 'string' }, base: { type: 'string', default: 'pt-megakernel' }, 'base-map': { type: 'string', default: '' }, md: { type: 'boolean', default: false } } });
  if (!values.report) throw new Error('usage: paired-ratios.ts --report <name> [--base r] [--base-map a=b,...] [--md]');
  const map = new Map((values['base-map'] ?? '').split(',').filter(Boolean).map((kv) => kv.split('=') as [string, string]));
  const cells = readCells(values.report);
  console.log('| scene | renderer | base | ms/spp min..max | ratio to base (paired) min..max | rounds loaded/invalid | load1 |');
  console.log('|---|---|---|---|---|---|---|');
  for (const c of cells) {
    const baseId = map.get(c.renderer) ?? values.base!;
    const base = cells.find((x) => x.scene === c.scene && x.renderer === baseId);
    const rs = base && base !== c ? ratios(c, base) : [];
    const loads = c.rounds.map((r) => r.load);
    console.log(`| ${c.scene} | ${c.renderer} | ${c === base ? '-' : baseId} | ${range(c.rounds.map((r) => r.ms), 4)} | ${rs.length ? range(rs) : '-'} | ${c.rounds.filter((r) => r.loaded).length}/${c.rounds.filter((r) => !r.valid).length} of ${c.rounds.length} | ${range(loads, 1)} |`);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
