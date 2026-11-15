/**
 * R2b: A/B of the cutout shadow modes (`alphaShadow`: roulette | coverage | stochastic | firstK | classified, against the exact loop and the solid walk).
 *
 *   node bench/runner/run.ts --page alpha-shadow-ab --no-csv --args scene=<s> renderers=pt-megakernel,pt-mk-shadow,pt-mk-roulette,... spp=1024 repeat=5   (one per scene and scheduler)
 *   node bench/adapters/alpha-shadow-ab.ts --runs <incoming dir>,... --ref <scene>=<pfm>,... [--ref2 <scene>=<pfm>,...] [--solid pt-mk-shadow] [--out file.md]
 *
 * Per (scene, renderer): relBias and relMSE of the image against the reference (our own exact 16384 spp render; `--ref2` adds Mitsuba for tex-mask), the paired
 * ms/spp ratio to the exact policy of the same page (round i against round i; rounds alternate order, the first round is an untimed warm-up), and the two numbers R2b is judged by:
 *   (ratio = best round of the mode / best round of exact: bursts of other jobs only ever add time)
 *   lost% = (ratio - ratioSolid) / (1 - ratioSolid)   the share of the solid walk's saving this mode gives back (target < 10%)
 *   eqTime = ratio x relMSE / relMSE(exact)           time to the exact policy's noise with an unbiased mode's variance inflation priced in (< 1 beats exact);
 *                                                     meaningful only where relBias is ~0 (a biased mode's error is inside relMSE as bias^2)
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const pct = (v: number): string => `${(v * 100).toFixed(2)}%`;
const range = (a: number[], d = 3): string => (a.length ? `${Math.min(...a).toFixed(d)}..${Math.max(...a).toFixed(d)}` : '-');
const median = (a: number[]): number => (a.length ? [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]! : NaN);
const wavefront = (id: string): boolean => id.startsWith('pt-wf') || id === 'pt-wavefront';

function parseRefs(s: string | undefined): Map<string, string> {
  return new Map((s ?? '').split(',').filter(Boolean).map((kv) => kv.split('=') as [string, string]));
}

function main(): void {
  const { values } = parseArgs({ options: { runs: { type: 'string' }, ref: { type: 'string' }, ref2: { type: 'string' }, solid: { type: 'string', default: 'pt-mk-shadow' }, out: { type: 'string' } } });
  if (!values.runs) throw new Error('usage: alpha-shadow-ab.ts --runs dir,dir --ref scene=file.pfm[,...] [--ref2 ...] [--solid pt-mk-shadow] [--out file.md]');
  const refs = parseRefs(values.ref);
  const refs2 = parseRefs(values.ref2);
  const lines = ['| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |', '|---|---|---|---|---|---|---|---|---|---|---|'];
  for (const dir of values.runs.split(',')) {
    const d = resolve(REPO, dir);
    const run = JSON.parse(readFileSync(join(d, 'run.json'), 'utf8')) as { runs: { result: { scene: string; notes: { rounds: Record<string, number[]> } }; load?: { load1?: number } }[] };
    const { scene, notes } = run.runs[0]!.result;
    const load = run.runs[0]!.load?.load1;
    const ids = Object.keys(notes.rounds);
    const exactId = ids.find((id) => id === 'pt-megakernel' || id === 'pt-wavefront') as string;
    const solidId = ids.find((id) => id === (wavefront(exactId) ? (values.solid as string).replace('pt-mk-', 'pt-wf-') : values.solid)) as string | undefined;
    const ref = refs.has(scene) ? readPFM(readFileSync(resolve(REPO, refs.get(scene) as string))) : null;
    const ref2 = refs2.has(scene) ? readPFM(readFileSync(resolve(REPO, refs2.get(scene) as string))) : null;
    const image = (id: string) => readPFM(readFileSync(join(d, '1-run', `${id.replace("@", "_b")}__${scene}.pfm`)));
    const mse = new Map<string, number>();
    if (ref) for (const id of ids) mse.set(id, compareImages(image(id), ref).relMse);
    const ratio = (id: string, base: string): number[] => (notes.rounds[id] as number[]).map((v, i) => v / (notes.rounds[base] as number[])[i]!);
    const minOf = (id: string): number => Math.min(...(notes.rounds[id] as number[]));
    const rSolid = solidId ? minOf(solidId) / minOf(exactId) : NaN;
    for (const id of ids) {
      const cmp = ref ? compareImages(image(id), ref) : null;
      const cmp2 = ref2 ? compareImages(image(id), ref2) : null;
      const rs = id === exactId ? [] : ratio(id, exactId);
      const rMin = minOf(id) / minOf(exactId);
      const r = rMin; // the contention of a shared Mac comes in bursts: the best round of each variant is the robust number, the paired median is printed beside it
      const lost = rs.length && id !== solidId ? (r - rSolid) / (1 - rSolid) : NaN;
      const eq = rs.length && mse.get(exactId) ? r * ((mse.get(id) as number) / (mse.get(exactId) as number)) : NaN;
      lines.push(`| ${scene} | ${id} | ${cmp ? pct(cmp.relBias) : '-'} | ${cmp ? sci(cmp.relMse) : '-'} | ${cmp2 ? `${sci(cmp2.relMse)} / ${pct(cmp2.relBias)}` : '-'} | ${range(notes.rounds[id] as number[], 4)} | ${rs.length ? `${range(rs)} (med ${median(rs).toFixed(3)})` : '-'} | ${rs.length ? rMin.toFixed(3) : '-'} | ${Number.isFinite(lost) ? pct(lost) : '-'} | ${Number.isFinite(eq) ? eq.toFixed(2) : '-'} | ${load ?? '-'} |`);
    }
  }
  console.log(lines.join('\n'));
  if (values.out) writeFileSync(values.out, `${lines.join('\n')}\n`);
}

main();
