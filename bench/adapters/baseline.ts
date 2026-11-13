/**
 * P0.9 baseline analysis: error-vs-time for the web competitors, from the runner directories a matrix report lists.
 *
 *   node bench/adapters/baseline.ts --report p0-baseline [--targets 1e-2,1e-3] [--csv] [--json out.json]
 *
 * Reads `bench/results/reports/<report>/runs.json` (written by matrix.ts), and for every cell and repeat loads the snapshot
 * images (`<renderer>__<scene>__spp<N>` plus the final image), computes relMSE (eps 1e-2, bench/metrics) against the Mitsuba
 * llvm_ad_rgb 4096 spp reference (headline, D15) and the pbrt-v4 one, pairs it with the wall time of that snapshot, fits
 * error = A * t^-k on log-log, and reports the time to reach each target: interpolated between snapshots when bracketed,
 * extrapolated from the fit otherwise (marked), or "not measurable" when the target is under 2x the reference noise floor
 * (the pbrt-vs-Mitsuba relMSE of the scene). Times are medians over the repeats with the min..max spread.
 * --csv appends summary rows (run = `summary`) to bench/results/competitor.csv.
 */
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import { xToReach, type SeriesPoint } from '../metrics/convergence.ts';
import { csvText, CSV_HEADER, type CsvRow } from '../runner/csv.ts';
import { gitCommit } from '../runner/measured-commit.ts';
import { median } from '../runner/stats.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

export interface FitResult {
  /** error = A * t^-k (t in ms). */
  k: number;
  A: number;
  r2: number;
  points: number;
}

/** Least squares of log(error) on log(t). Needs at least 2 points with positive t and error; otherwise null. */
export function fitPowerLaw(points: ReadonlyArray<{ t: number; e: number }>): FitResult | null {
  const p = points.filter((q) => q.t > 0 && q.e > 0 && Number.isFinite(q.t) && Number.isFinite(q.e));
  if (p.length < 2) return null;
  const xs = p.map((q) => Math.log(q.t));
  const ys = p.map((q) => Math.log(q.e));
  const n = p.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i]! - mx) ** 2;
    sxy += (xs[i]! - mx) * (ys[i]! - my);
    syy += (ys[i]! - my) ** 2;
  }
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  return { k: -slope, A: Math.exp(my - slope * mx), r2: syy === 0 ? 1 : (sxy * sxy) / (sxx * syy), points: n };
}

export type TargetKind = 'interp' | 'extrap' | 'not-reached' | 'not-measurable';
export interface TargetResult {
  target: number;
  kind: TargetKind;
  /** Milliseconds (absent for not-measurable / not-reached without a usable fit). */
  ms?: number;
}

/**
 * Time to reach `target`. Under 2x the reference floor nothing is claimed. Otherwise: interpolated (log-log) between the
 * bracketing snapshots when the series gets there; else extrapolated from the fit (needs k > 0.3, else not reached).
 */
export function timeToTarget(series: readonly SeriesPoint[], target: number, floor: number, fit: FitResult | null): TargetResult {
  if (target < 2 * floor) return { target, kind: 'not-measurable' };
  const hit = xToReach(series, target, 'wallMs');
  if (hit !== null) return { target, kind: 'interp', ms: hit };
  if (fit && fit.k > 0.3) return { target, kind: 'extrap', ms: Math.pow(fit.A / target, 1 / fit.k) };
  return { target, kind: 'not-reached' };
}

/** True when the last step of the error series fell by under half of the 1/spp prediction (error ratio > 2x the spp ratio). */
export function isPlateau(series: ReadonlyArray<{ spp: number; error: number }>): boolean {
  if (series.length < 2) return false;
  const a = series[series.length - 2]!;
  const b = series[series.length - 1]!;
  if (!(a.error > 0) || !(b.spp > a.spp)) return false;
  return b.error / a.error > 2 * (a.spp / b.spp);
}

export function fmtMs(ms: number): string {
  if (ms >= 60000) return `${(ms / 60000).toFixed(1)} min`;
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${ms.toFixed(0)} ms`;
}

/** Block-mean comparison (structure check, noise averages out): max and RMS relative difference of NxN block means. */
export function blockMeanDiff(test: Image, ref: Image, block: number): { maxRel: number; rmsRel: number; blocks: number } {
  const bx = Math.floor(test.width / block);
  const by = Math.floor(test.height / block);
  let max = 0;
  let sq = 0;
  let count = 0;
  for (let j = 0; j < by; j++) {
    for (let i = 0; i < bx; i++) {
      let a = 0;
      let b = 0;
      for (let y = 0; y < block; y++) {
        for (let x = 0; x < block; x++) {
          const ti = ((j * block + y) * test.width + (i * block + x)) * test.channels;
          const ri = ((j * block + y) * ref.width + (i * block + x)) * ref.channels;
          a += (test.data[ti]! + test.data[ti + 1]! + test.data[ti + 2]!) / 3;
          b += (ref.data[ri]! + ref.data[ri + 1]! + ref.data[ri + 2]!) / 3;
        }
      }
      const rel = (a - b) / Math.max(b, 1e-3 * block * block);
      max = Math.max(max, Math.abs(rel));
      sq += rel * rel;
      count++;
    }
  }
  return { maxRel: max, rmsRel: Math.sqrt(sq / Math.max(count, 1)), blocks: count };
}

interface RunJson {
  commit: string;
  browser: string;
  adapter: { description?: string; vendor: string; architecture: string } | null;
  timingVerdict?: { kind: string; reason: string };
  runs: Array<{ index: number; variant: string; outcome: string; result?: { valid?: boolean; metrics: Record<string, number>; notes?: Record<string, unknown> } }>;
  startedAt: string;
}

export interface CellAnalysis {
  scene: string;
  renderer: string;
  emitters: string;
  runDir: string;
  repeats: number;
  spps: number[];
  /** Per spp: median over repeats of the snapshot wall time, with spread, and the error (median over repeats). */
  rows: Array<{ spp: number; wallMs: number; wallMin: number; wallMax: number; relMseMitsuba: number; relMsePbrt: number }>;
  msPerSpp: { median: number; min: number; max: number };
  fitMitsuba: FitResult | null;
  fitPbrt: FitResult | null;
  targetsMitsuba: Array<TargetResult & { msMin?: number; msMax?: number }>;
  targetsPbrt: TargetResult[];
  meanRatio: number[];
  relBiasPct: number;
  blockMean: { maxRel: number; rmsRel: number };
  floor: number;
  /** The last 2 snapshots improved by less than half of what 1/spp predicts: the curve is flattening (bias floor), extrapolation is not trustworthy. */
  plateau: { mitsuba: boolean; pbrt: boolean };
  valid: boolean;
  verdict: string;
  device: string;
  browser: string;
  commit: string;
  loadMs: number;
  compileMs: number;
  nonFinite: number;
}

const readImage = (path: string): Image => readPFM(readFileSync(path));

function refPath(scene: string, prefix: string): string | null {
  const dir = join(REPO, 'bench', 'references', scene);
  if (!existsSync(dir)) return null;
  const file = `${prefix}-4096spp.pfm`;
  return existsSync(join(dir, file)) ? join(dir, file) : null;
}

export function analyseCell(entry: RunLogEntry, targets: readonly number[], blockPx = 40): CellAnalysis {
  const dir = join(REPO, entry.runDir);
  const run = JSON.parse(readFileSync(join(dir, 'run.json'), 'utf8')) as RunJson;
  const mitsubaPath = refPath(entry.scene, 'mitsuba3-llvm_ad_rgb');
  const pbrtPath = refPath(entry.scene, 'pbrt-v4-cpu');
  if (!mitsubaPath) throw new Error(`no Mitsuba reference for ${entry.scene}`);
  const mitsuba = readImage(mitsubaPath);
  const pbrt = pbrtPath ? readImage(pbrtPath) : null;
  const floor = pbrt ? compareImages(mitsuba, pbrt).relMse : 0;
  const passRuns = run.runs.filter((r) => r.outcome === 'pass' && r.result);
  const spps = [...entry.snapshots.filter((s) => s < entry.spp), entry.spp];
  const perSpp = new Map<number, { wall: number[]; mit: number[]; pbrt: number[] }>();
  for (const s of spps) perSpp.set(s, { wall: [], mit: [], pbrt: [] });
  const perRunSeries: SeriesPoint[][] = [];
  const perRunSeriesPbrt: SeriesPoint[][] = [];
  const msPerSpp: number[] = [];
  let meanRatio = [0, 0, 0];
  let relBias = 0;
  let block = { maxRel: 0, rmsRel: 0 };
  let nonFinite = 0;
  let loadMs = 0;
  let compileMs = 0;
  let valid = true;
  passRuns.forEach((r, idx) => {
    const m = r.result!.metrics;
    const slot = `${r.index}-${r.variant === '-' ? 'run' : r.variant}`;
    const base = `${entry.renderer}__${entry.scene}`;
    const series: SeriesPoint[] = [];
    const seriesP: SeriesPoint[] = [];
    for (const s of spps) {
      const final = s === entry.spp;
      const wall = final ? m['wall_ms'] : m[`wall_ms_at_${s}`];
      const file = join(dir, slot, final ? `${base}.pfm` : `${base}__spp${s}.pfm`);
      if (wall === undefined || !existsSync(file)) continue;
      const image = readImage(file);
      const a = compareImages(image, mitsuba);
      const b = pbrt ? compareImages(image, pbrt) : null;
      series.push({ spp: s, wallMs: wall, error: a.relMse });
      if (b) seriesP.push({ spp: s, wallMs: wall, error: b.relMse });
      const slotData = perSpp.get(s)!;
      slotData.wall.push(wall);
      slotData.mit.push(a.relMse);
      if (b) slotData.pbrt.push(b.relMse);
      if (final && idx === 0) {
        const mean = (im: Image): number[] => {
          const sum = [0, 0, 0];
          const n = im.width * im.height;
          for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) sum[c]! += im.data[i * im.channels + c]!;
          return sum.map((v) => v / n);
        };
        const mt = mean(image);
        const mr = mean(mitsuba);
        meanRatio = mt.map((v, c) => v / mr[c]!);
        relBias = a.relBias;
        block = blockMeanDiff(image, mitsuba, blockPx);
      }
    }
    perRunSeries.push(series);
    perRunSeriesPbrt.push(seriesP);
    if (m['ms_per_spp'] !== undefined) msPerSpp.push(m['ms_per_spp']);
    nonFinite += m['non_finite_pixels'] ?? 0;
    loadMs = m['load_ms'] ?? loadMs;
    compileMs = m['compile_ms'] ?? compileMs;
    if (r.result!.valid === false) valid = false;
  });
  if (run.timingVerdict && run.timingVerdict.kind === 'invalid') valid = false;
  const rows = spps
    .map((s) => ({ s, d: perSpp.get(s)! }))
    .filter(({ d }) => d.wall.length > 0)
    .map(({ s, d }) => ({ spp: s, wallMs: median(d.wall), wallMin: Math.min(...d.wall), wallMax: Math.max(...d.wall), relMseMitsuba: median(d.mit), relMsePbrt: d.pbrt.length ? median(d.pbrt) : NaN }));
  const fitPoints = (key: 'relMseMitsuba' | 'relMsePbrt') => rows.filter((r) => r[key] >= 2 * floor).map((r) => ({ t: r.wallMs, e: r[key] }));
  const fitMitsuba = fitPowerLaw(fitPoints('relMseMitsuba'));
  const fitPbrt = fitPowerLaw(fitPoints('relMsePbrt'));
  const medianSeries = rows.map((r) => ({ spp: r.spp, wallMs: r.wallMs, error: r.relMseMitsuba }));
  const medianSeriesP = rows.map((r) => ({ spp: r.spp, wallMs: r.wallMs, error: r.relMsePbrt }));
  const targetsMitsuba = targets.map((t) => {
    const mid = timeToTarget(medianSeries, t, floor, fitMitsuba);
    const each = perRunSeries.map((s) => timeToTarget(s, t, floor, fitPowerLaw(s.filter((p) => p.error >= 2 * floor).map((p) => ({ t: p.wallMs, e: p.error }))))).filter((x) => x.ms !== undefined).map((x) => x.ms!);
    return { ...mid, ...(each.length > 1 ? { msMin: Math.min(...each), msMax: Math.max(...each) } : {}) };
  });
  const targetsPbrt = targets.map((t) => timeToTarget(medianSeriesP, t, floor, fitPbrt));
  const verdict = run.timingVerdict ? `${run.timingVerdict.kind}: ${run.timingVerdict.reason}` : 'unknown';
  return {
    scene: entry.scene,
    renderer: entry.renderer,
    emitters: entry.emitters,
    runDir: entry.runDir,
    repeats: passRuns.length,
    spps,
    rows,
    msPerSpp: { median: median(msPerSpp), min: Math.min(...msPerSpp), max: Math.max(...msPerSpp) },
    fitMitsuba,
    fitPbrt,
    targetsMitsuba,
    targetsPbrt,
    meanRatio,
    relBiasPct: relBias * 100,
    blockMean: block,
    floor,
    plateau: { mitsuba: isPlateau(rows.map((r) => ({ spp: r.spp, error: r.relMseMitsuba }))), pbrt: isPlateau(rows.map((r) => ({ spp: r.spp, error: r.relMsePbrt }))) },
    valid,
    verdict,
    device: run.adapter?.description || `${run.adapter?.vendor} ${run.adapter?.architecture}`,
    browser: run.browser,
    commit: run.commit,
    loadMs,
    compileMs,
    nonFinite,
  };
}

const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
function fmtTarget(t: TargetResult & { msMin?: number; msMax?: number }): string {
  if (t.kind === 'not-measurable') return 'not measurable (below 2x ref floor)';
  if (t.ms === undefined) return 'not reached';
  const spread = t.msMin !== undefined && t.msMax !== undefined ? ` (${fmtMs(t.msMin)}..${fmtMs(t.msMax)})` : '';
  return `${t.kind === 'extrap' ? '~' : ''}${fmtMs(t.ms)}${t.kind === 'extrap' ? ' (extrapolated)' : ''}${spread}`;
}

export function toMarkdown(cells: readonly CellAnalysis[], targets: readonly number[]): string {
  const out: string[] = [];
  const head = ['scene', 'renderer', 'ms/spp (min..max)', 'fit k', ...targets.map((t) => `time to relMSE ${sci(t)}`), 'relMSE@final (Mitsuba / pbrt)'];
  out.push(`| ${head.join(' | ')} |`, `|${head.map(() => '---').join('|')}|`);
  for (const c of cells) {
    const last = c.rows[c.rows.length - 1]!;
    out.push(
      `| ${c.scene} | ${c.renderer}${c.emitters !== 'default' ? ` (${c.emitters})` : ''} | ${c.msPerSpp.median.toFixed(c.msPerSpp.median < 10 ? 2 : 1)} (${c.msPerSpp.min.toFixed(1)}..${c.msPerSpp.max.toFixed(1)}) | ${c.fitMitsuba ? c.fitMitsuba.k.toFixed(2) : '-'} | ${c.targetsMitsuba.map(fmtTarget).join(' | ')} | ${sci(last.relMseMitsuba)} / ${sci(last.relMsePbrt)} @${last.spp} spp |`,
    );
  }
  out.push('');
  for (const c of cells) {
    out.push(`${c.scene} / ${c.renderer}${c.emitters !== 'default' ? ` (${c.emitters})` : ''}: floor ${sci(c.floor)}, mean ratio vs Mitsuba ${c.meanRatio.map((v) => v.toFixed(4)).join('/')}, relBias ${c.relBiasPct.toFixed(2)}%, block-mean max/rms ${(c.blockMean.maxRel * 100).toFixed(1)}%/${(c.blockMean.rmsRel * 100).toFixed(1)}%, fit r2 ${c.fitMitsuba ? c.fitMitsuba.r2.toFixed(3) : '-'} (${c.fitMitsuba?.points ?? 0} pts), repeats ${c.repeats}, valid ${c.valid}${c.plateau.mitsuba ? ', PLATEAU vs Mitsuba (bias floor, extrapolation unreliable)' : ''}${c.plateau.pbrt ? ', PLATEAU vs pbrt' : ''}`);
    out.push('  spp | wall median (min..max) | relMSE Mitsuba | relMSE pbrt');
    for (const r of c.rows) out.push(`  ${String(r.spp).padStart(5)} | ${fmtMs(r.wallMs)} (${fmtMs(r.wallMin)}..${fmtMs(r.wallMax)}) | ${sci(r.relMseMitsuba)} | ${sci(r.relMsePbrt)}`);
    out.push(`  targets vs pbrt: ${c.targetsPbrt.map(fmtTarget).join(' | ')}`);
  }
  return out.join('\n');
}

export function csvRows(cells: readonly CellAnalysis[], date: string): CsvRow[] {
  const rows: CsvRow[] = [];
  for (const c of cells) {
    const base = { date, commit: gitCommit(), device: c.device, browser: c.browser, scene: c.scene, renderer: c.renderer, config_hash: `p0-baseline${c.emitters !== 'default' ? `-${c.emitters}` : ''}`, valid: (c.valid ? 1 : 0) as 0 | 1, run: 'summary' };
    const inexact = c.emitters === 'mesh' ? '; INEXACT cell (D19/D20): tgpt emissive-mesh lights, no NEE' : '';
    const note = `P0.9 baseline, median of ${c.repeats} runs; relMSE eps 1e-2 vs Mitsuba llvm_ad_rgb 4096 spp unless named pbrt${inexact}`;
    const add = (metric: string, value: number): void => {
      if (Number.isFinite(value)) rows.push({ ...base, metric, value, note });
    };
    for (const r of c.rows) {
      add(`relmse_mitsuba_at_${r.spp}`, r.relMseMitsuba);
      add(`relmse_pbrt_at_${r.spp}`, r.relMsePbrt);
      add(`wall_ms_median_at_${r.spp}`, r.wallMs);
    }
    add('ms_per_spp_median', c.msPerSpp.median);
    if (c.fitMitsuba) add('fit_k', c.fitMitsuba.k);
    for (const t of c.targetsMitsuba) {
      if (t.ms !== undefined) rows.push({ ...base, metric: `ms_to_relmse_${t.target.toExponential(0).replace('e-', 'em')}`, value: t.ms, note: `${note}; ${t.kind}` });
    }
  }
  return rows;
}

function main(): void {
  const { values } = parseArgs({ options: { report: { type: 'string', default: 'p0-baseline' }, targets: { type: 'string', default: '1e-2,1e-3' }, csv: { type: 'boolean', default: false }, json: { type: 'string', default: '' } } });
  const targets = values.targets!.split(',').map(Number);
  const runs = JSON.parse(readFileSync(join(REPO, 'bench', 'results', 'reports', values.report!, 'runs.json'), 'utf8')) as RunLogEntry[];
  const cells = runs.map((e) => analyseCell(e, targets));
  console.log(toMarkdown(cells, targets));
  if (values.json) writeFileSync(values.json, `${JSON.stringify(cells, null, 2)}\n`);
  if (values.csv) {
    const path = join(REPO, 'bench', 'results', 'competitor.csv');
    const text = csvText(csvRows(cells, new Date().toISOString().slice(0, 10)), !existsSync(path));
    appendFileSync(path, text);
    console.log(`appended ${text.split('\n').filter(Boolean).length} summary rows to ${path} (${CSV_HEADER.split(',').length} columns)`);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
