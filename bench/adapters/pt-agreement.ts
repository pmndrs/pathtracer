/**
 * P2.4 agreement table for our path tracer against the Tier 0 references (and the analytic expectations).
 *
 *   node bench/adapters/pt-agreement.ts --report pt-tier0 [--out bench/results/reports/pt-tier0/PT-AGREEMENT.md]
 *
 * Reads `bench/results/reports/<report>/runs.json` (matrix.ts) and, for every cell:
 *  - relMSE (eps 1e-2, bench/metrics) against Mitsuba llvm_ad_rgb 4096 spp and pbrt-v4 4096 spp at every snapshot, next to
 *    the pbrt-vs-Mitsuba floor of the scene (their own relMSE, the noise of the references);
 *  - relBias (mean test / mean reference - 1) and the max / rms relative difference of 40x40 block means (structure; noise averages out);
 *  - the scene's analytic expectation when it has one: furnaces (every pixel within the tolerance, mean within tolerance/10, as
 *    `references/compare.ts`) and the quad-light closed form (mean of the plane pixels);
 *  - the image mean per channel, non-finite pixels, wall time (best effort, no precision claim).
 * Images are the runner's PFMs, so the A/B viewer opens the same files (report `?report=/results/reports/<name>/report.json`).
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { checkMaskedImage, checkMean, checkUniform, quadLightAnalyticImage } from '../metrics/analytic.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import { getScene } from '../scenes/index.ts';
import { analyseCell, blockMeanDiff, type CellAnalysis } from './baseline.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const pct = (v: number): string => `${(v * 100).toFixed(2)}%`;
const load = (path: string): Image => readPFM(readFileSync(path));

function imageMean(image: Image): number[] {
  const sum = [0, 0, 0];
  const n = image.width * image.height;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) sum[c]! += image.data[i * image.channels + c]!;
  return sum.map((v) => v / n);
}

function finalImagePath(entry: RunLogEntry): string {
  return join(REPO, entry.runDir, '1-run', `${entry.renderer}__${entry.scene}.pfm`);
}

/** Furnace / analytic verdict for the final image of a cell. */
export function expectationOf(entry: RunLogEntry, image: Image): string {
  const scene = getScene(entry.scene);
  const e = scene.expected;
  if (e) {
    const uniform = checkUniform(image, e.value, e.tolerance, { relative: true });
    const mean = checkMean(image, e.value, e.tolerance / 10, { relative: true });
    return `expected ${e.value[0]} (tol ${e.tolerance}): mean ${uniform.mean[0].toFixed(5)}, mean rel dev ${sci(Math.abs(mean.maxDeviation))}, max pixel rel dev ${sci(uniform.maxDeviation)} -> ${uniform.ok && mean.ok ? 'PASS' : `FAIL (pixels ${uniform.ok ? 'ok' : 'out of tolerance'}, mean ${mean.ok ? 'ok' : 'out of tolerance'})`}`;
  }
  if (entry.scene === 'quad-light') {
    const r = checkMaskedImage(image, quadLightAnalyticImage(scene), 0.005);
    return `closed form over ${r.n} plane pixels: mean ${r.meanTest.toFixed(5)} vs ${r.meanExpected.toFixed(5)}, mean rel dev ${sci(r.meanRelDev)} -> ${r.ok ? 'PASS' : 'FAIL'}`;
  }
  return 'no analytic expectation';
}

export interface AgreementRow {
  cell: CellAnalysis;
  entry: RunLogEntry;
  mean: number[];
  expectation: string;
  vsPbrt: { relBiasPct: number; block: { maxRel: number; rmsRel: number } } | null;
  /** The references against each other (Mitsuba as test, pbrt as reference): what a perfect renderer would also show. */
  refPair: { maxRel: number; rmsRel: number } | null;
}

export function analyse(entry: RunLogEntry): AgreementRow {
  const cell = analyseCell(entry, [1e-2, 1e-3]);
  const image = load(finalImagePath(entry));
  const pbrtPath = join(REPO, 'bench', 'references', entry.scene, 'pbrt-v4-cpu-4096spp.pfm');
  let vsPbrt: AgreementRow['vsPbrt'] = null;
  let refPair: AgreementRow['refPair'] = null;
  try {
    const pbrt = load(pbrtPath);
    refPair = blockMeanDiff(load(join(REPO, 'bench', 'references', entry.scene, 'mitsuba3-llvm_ad_rgb-4096spp.pfm')), pbrt, 40);
    vsPbrt = { relBiasPct: compareImages(image, pbrt).relBias * 100, block: blockMeanDiff(image, pbrt, 40) };
  } catch {
    vsPbrt = null;
  }
  return { cell, entry, mean: imageMean(image), expectation: expectationOf(entry, image), vsPbrt, refPair };
}

export function toMarkdown(rows: readonly AgreementRow[], title: string): string {
  const out: string[] = [`# ${title}`, ''];
  out.push('| scene | spp | relMSE vs Mitsuba (RGB) | relMSE vs pbrt-v4 | pbrt-vs-Mitsuba floor | relBias vs Mitsuba / pbrt | block-mean max / rms vs Mitsuba | image mean RGB | non-finite |');
  out.push('|---|---|---|---|---|---|---|---|---|');
  for (const r of rows) {
    const last = r.cell.rows[r.cell.rows.length - 1]!;
    out.push(
      `| ${r.entry.scene} | ${last.spp} | ${sci(last.relMseMitsuba)} | ${sci(last.relMsePbrt)} | ${sci(r.cell.floor)} | ${r.cell.relBiasPct.toFixed(2)}% / ${r.vsPbrt ? `${r.vsPbrt.relBiasPct.toFixed(2)}%` : '-'} | ${pct(r.cell.blockMean.maxRel)} / ${pct(r.cell.blockMean.rmsRel)} | ${r.mean.map((v) => v.toFixed(5)).join(' ')} | ${r.cell.nonFinite} |`,
    );
  }
  out.push('', '## Convergence (relMSE vs Mitsuba / vs pbrt) and wall time (best effort)', '');
  for (const r of rows) {
    out.push(`**${r.entry.scene}**: ${r.cell.rows.map((x) => `${x.spp} spp ${sci(x.relMseMitsuba)} / ${sci(x.relMsePbrt)} (${x.wallMs.toFixed(0)} ms)`).join('; ')}`, '');
    out.push(`- ${r.expectation}`);
    if (r.vsPbrt) out.push(`- block means (40 px) vs pbrt: max ${pct(r.vsPbrt.block.maxRel)}, rms ${pct(r.vsPbrt.block.rmsRel)}${r.refPair ? `; the references against each other: max ${pct(r.refPair.maxRel)}, rms ${pct(r.refPair.rmsRel)}` : ''}`);
    if (r.cell.plateau.mitsuba || r.cell.plateau.pbrt) out.push(`- plateau (bias or reference floor): vs Mitsuba ${r.cell.plateau.mitsuba}, vs pbrt ${r.cell.plateau.pbrt}`);
    out.push('');
  }
  return out.join('\n');
}

function main(): void {
  const { values } = parseArgs({ options: { report: { type: 'string' }, out: { type: 'string', default: '' }, renderer: { type: 'string', default: 'pt-megakernel' } } });
  if (!values.report) throw new Error('--report <name> is required');
  const dir = join(REPO, 'bench', 'results', 'reports', values.report);
  const log = JSON.parse(readFileSync(join(dir, 'runs.json'), 'utf8')) as RunLogEntry[];
  const rows = log.filter((e) => e.renderer === values.renderer).map(analyse);
  const text = toMarkdown(rows, `Agreement of ${values.renderer} with the Tier 0 references (${values.report})`);
  console.log(text);
  writeFileSync(values.out || join(dir, 'PT-AGREEMENT.md'), `${text}\n`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
