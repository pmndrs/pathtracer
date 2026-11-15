/**
 * R2: A/B of the per-ray-class alpha policy (`alphaPolicy`: exact | opaqueShadow | opaqueIndirect | opaqueSecondary), quality and (provisional) timing.
 *
 *   node bench/adapters/alpha-policy-ab.ts --report <name> [--ref <scene>=<pfm>[,<scene>=<pfm>...]] [--out file.md]
 *
 * The report comes from `node bench/adapters/matrix.ts --renderers pt-megakernel,pt-mk-shadow,pt-mk-indirect,pt-mk-secondary,pt-wavefront,pt-wf-shadow,
 * pt-wf-indirect,pt-wf-secondary --scenes tex-mask,foliage-mask --spp 1024 --snapshots 1024 --repeat 5 --interleave --name <name>`. For every (scene, renderer): the image against
 * each reference (relMSE, relBias = mean(t) / mean(r) - 1), the exact policy's own relMSE as the noise floor, and ms/spp from
 * bench/results/competitor.csv (min..max over the valid interleaved runs, ratio to the exact policy of the same scheduler, paired by run). A biased policy shows as
 * relMSE well above the exact one; timing rows carry the load the runner recorded and are provisional until the quiet pass (docs/QUIET-PASS.md).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const pct = (v: number): string => `${(v * 100).toFixed(2)}%`;

interface CsvRow { scene: string; renderer: string; metric: string; value: number; valid: boolean; run: string; note: string }
function csvRows(): CsvRow[] {
  const rows: CsvRow[] = [];
  for (const line of readFileSync(join(REPO, 'bench/results/competitor.csv'), 'utf8').split('\n').slice(1)) {
    const m = /^([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),(.*)$/.exec(line);
    if (m) rows.push({ scene: m[5]!, renderer: m[6]!, metric: m[8]!, value: Number(m[9]), valid: m[10] === '1', run: m[11]!, note: m[12]! });
  }
  return rows;
}

function main(): void {
  const { values } = parseArgs({ options: { report: { type: 'string' }, ref: { type: 'string' }, out: { type: 'string' } } });
  if (!values.report) throw new Error('usage: alpha-policy-ab.ts --report <name> [--ref scene=file.pfm,...] [--out file.md]');
  const runs = JSON.parse(readFileSync(join(REPO, 'bench/results/reports', values.report, 'runs.json'), 'utf8')) as RunLogEntry[];
  const refs = new Map<string, string>((values.ref ?? '').split(',').filter(Boolean).map((kv) => kv.split('=') as [string, string]));
  const csv = csvRows().filter((r) => r.metric === 'ms_per_spp' && runs.some((e) => e.renderer === r.renderer && e.scene === r.scene));
  const lines = ['| scene | renderer | policy | relMSE vs ref | relBias vs ref | ms/spp min..max (valid runs) | ratio to exact (paired) | load note |', '|---|---|---|---|---|---|---|---|'];
  const exactOf = (renderer: string): string => (renderer.startsWith('pt-wf') || renderer === 'pt-wavefront' ? 'pt-wavefront' : 'pt-megakernel');
  for (const scene of new Set(runs.map((e) => e.scene))) {
    const ref = refs.get(scene);
    const refImage = ref ? readPFM(readFileSync(resolve(REPO, ref))) : null;
    for (const e of runs.filter((x) => x.scene === scene)) {
      const img = readPFM(readFileSync(join(REPO, e.runDir, '1-run', `${e.renderer}__${e.scene}.pfm`)));
      const cmp = refImage ? compareImages(img, refImage) : null;
      // the CSV is append-only: take the last `repeat` interleaved rows of the cell (this report's run)
      const last = (renderer: string): CsvRow[] => csv.filter((r) => r.scene === scene && r.renderer === renderer && r.run !== 'median' && r.note.includes('interleaved')).slice(-e.repeat);
      const mine = last(e.renderer).filter((r) => r.valid);
      const base = last(exactOf(e.renderer)).filter((r) => r.valid);
      const ms = mine.map((r) => r.value);
      const ratios = mine.flatMap((r) => { const b = base.find((x) => x.run === r.run); return b ? [r.value / b.value] : []; });
      const range = (a: number[]): string => (a.length ? `${Math.min(...a).toFixed(3)}..${Math.max(...a).toFixed(3)}` : '-');
      const policy = e.renderer.endsWith('shadow') ? 'opaqueShadow' : e.renderer.endsWith('indirect') ? 'opaqueIndirect' : e.renderer.endsWith('secondary') ? 'opaqueSecondary' : 'exact';
      lines.push(`| ${scene} | ${e.renderer} | ${policy} | ${cmp ? sci(cmp.relMse) : '-'} | ${cmp ? pct(cmp.relBias) : '-'} | ${range(ms)} (${ms.length}) | ${range(ratios)} | ${[...new Set(mine.map((r) => /load [\d.]+/.exec(r.note)?.[0] ?? ''))].join(', ')} |`);
    }
  }
  console.log(lines.join('\n'));
  if (values.out) writeFileSync(values.out, `${lines.join('\n')}\n`);
}

main();
