/**
 * Runs web adapters over bench scenes (default Tier 0; `--scenes sponza-white` for Tier 2; `--scenes all` for every scene) through the bench runner and merges the per-run report manifests into one grid
 * (renderer x scene) for the viewer.
 *
 *   node bench/adapters/matrix.ts [--renderers tgpt-wavefront,tgpt-megakernel,mutracer] [--scenes a,b] [--spp 256]
 *                                 [--snapshots 16,64] [--name baseline] [--timeout 600000] [--force] [--budget N] [--emitters rect|mesh]
 *                                 [--repeat 3] [--interleave] [--append] [--extra k=v,k=v]
 *
 * --extra k=v,k=v: more page args for every cell (e.g. the pt-megakernel settings `pt_sampler=pcg,pt_nee=0`, see bench/pages/competitor.ts).
 * --repeat N: the runner repeats every cell N times (CSV rows run 1..N plus median; uploads go to slots 1-run..N-run; the report
 * shows run 1). --append: add the new cells to an existing report of the same name instead of replacing it (lets one report mix
 * different spp per cell). --interleave (needs --repeat N > 1): per scene, N rounds; each round runs every renderer once (`--repeat 1`,
 * order reversed every round: A B C, C B A, A B C ...) so slow drift (thermals, background load) hits all renderers alike. The rounds
 * are then merged per cell into one directory laid out like a single `--repeat N` run (run.json runs 1..N, slots 1-run..N-run, report.json
 * from round 1, CSV rows run 1..N + median appended by matrix.ts), so baseline.ts and the viewer see no difference. The per-round runner
 * directories stay next to it (they keep only run.json and the report). Rounds run with --no-csv; one round's invalid timing verdict
 * marks the whole cell's rows valid=0. `runs.json` next to report.json lists every cell's runner directory (for bench/adapters/baseline.ts).
 *
 * One runner invocation per (renderer, scene), so a crash in one cell cannot take the others down. Output:
 * bench/results/reports/<name>/report.json (open with ?report=/results/reports/<name>/report.json in the viewer).
 * Each cell keeps the CSV rows the runner appends to bench/results/competitor.csv.
 */
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { appendCsv } from '../runner/csv.ts';
import type { RunSummary } from '../runner/contract.ts';
import { mergeRounds, retargetManifest, roundOrder, type RoundResult } from './interleave.ts';
import type { ReportManifest } from '../viewer/src/report.ts';
import { exportScenes } from './export-scenes.ts';
import { ADAPTER_IDS } from './registry.ts';
import { allScenes, tier0Scenes } from '../scenes/index.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

/** One matrix cell as run: where the runner left its repeats (repo-relative directory holding run.json). */
export interface RunLogEntry { scene: string; renderer: string; emitters: string; spp: number; snapshots: number[]; repeat: number; runDir: string }

export function mergeManifests(parts: Array<{ manifest: ReportManifest; dir: string }>, outDir: string, run: string): ReportManifest {
  const toOut = (dir: string, rel: string | undefined): string | undefined => (rel ? relative(outDir, resolve(dir, rel)).split('\\').join('/') : undefined);
  const scenes = new Map<string, ReportManifest['scenes'][number]>();
  const renderers = new Map<string, ReportManifest['renderers'][number]>();
  const cells: ReportManifest['cells'] = [];
  for (const { manifest, dir } of parts) {
    for (const s of manifest.scenes) if (!scenes.has(s.id)) scenes.set(s.id, { ...s, reference: toOut(dir, s.reference) });
    for (const r of manifest.renderers) if (!renderers.has(r.id)) renderers.set(r.id, r);
    for (const c of manifest.cells) {
      cells.push({ ...c, image: toOut(dir, c.image) ?? '', ...(c.thumb ? { thumb: toOut(dir, c.thumb) } : {}), ...(c.reference ? { reference: toOut(dir, c.reference) } : {}) });
    }
  }
  return { run, scenes: [...scenes.values()], renderers: [...renderers.values()], cells };
}

function main(): void {
  const { values } = parseArgs({
    options: {
      renderers: { type: 'string', default: ADAPTER_IDS.join(',') },
      scenes: { type: 'string', default: tier0Scenes.map((s) => s.name).join(',') },
      spp: { type: 'string', default: '256' },
      snapshots: { type: 'string', default: '' },
      name: { type: 'string', default: 'competitors' },
      timeout: { type: 'string', default: '600000' },
      budget: { type: 'string', default: '' },
      emitters: { type: 'string', default: '' },
      extra: { type: 'string', default: '' },
      force: { type: 'boolean', default: false },
      repeat: { type: 'string', default: '1' },
      append: { type: 'boolean', default: false },
      interleave: { type: 'boolean', default: false },
    },
  });
  const renderers = values.renderers!.split(',').filter(Boolean);
  const sceneNames = values.scenes === 'all' ? allScenes.map((s) => s.name) : values.scenes!.split(',').filter(Boolean);
  exportScenes(sceneNames);
  const outDir = join(REPO, 'bench', 'results', 'reports', values.name!);
  const parts: Array<{ manifest: ReportManifest; dir: string }> = [];
  const runsFile = join(outDir, 'runs.json');
  const runLog: RunLogEntry[] = [];
  if (values.append && existsSync(join(outDir, 'report.json'))) {
    parts.push({ manifest: JSON.parse(readFileSync(join(outDir, 'report.json'), 'utf8')) as ReportManifest, dir: outDir });
    if (existsSync(runsFile)) runLog.push(...(JSON.parse(readFileSync(runsFile, 'utf8')) as RunLogEntry[]));
  }
  const repeat = Number(values.repeat);
  if (values.interleave && !(repeat > 1)) throw new Error('--interleave needs --repeat N with N > 1');
  const emitters = values.emitters || 'default';
  const cellArgs = (scene: string, renderer: string): string[] => {
    const args = [`renderer=${renderer}`, `scene=${scene}`, `spp=${values.spp}`];
    if (values.snapshots) args.push(`snapshots=${values.snapshots}`);
    if (values.budget) args.push(`budget=${values.budget}`);
    if (values.emitters) args.push(`emitters=${values.emitters}`);
    if (values.force) args.push('force=1');
    if (values.extra) args.push(...values.extra.split(',').filter(Boolean));
    return args;
  };
  const runRunner = (repeatArg: string, extra: string[], args: string[]): string => {
    const run = spawnSync(process.execPath, [join(REPO, 'bench/runner/run.ts'), '--page', 'competitor', '--suite', 'competitor', '--timeout', values.timeout!, '--repeat', repeatArg, ...extra, '--args', ...args], { cwd: REPO, encoding: 'utf8' });
    const out = `${run.stdout}${run.stderr}`;
    const lines = out.split('\n').filter((l) => /PASS|FAIL|metrics:|FAILED|The page reported|Timed out/.test(l));
    console.log(lines.map((l) => l.slice(0, 400)).join('\n'));
    return out;
  };
  const record = (scene: string, renderer: string, dir: string): void => {
    const entry: RunLogEntry = { scene, renderer, emitters, spp: Number(values.spp), snapshots: values.snapshots ? values.snapshots.split(',').map(Number) : [], repeat, runDir: relative(REPO, dir) };
    runLog.splice(0, runLog.length, ...runLog.filter((e) => !(e.scene === scene && e.renderer === renderer && e.emitters === entry.emitters)), entry);
  };
  for (const scene of sceneNames) {
    if (values.interleave) {
      const rounds = new Map<string, Array<RoundResult & { manifest: ReportManifest | null }>>(renderers.map((r) => [r, []]));
      for (let round = 0; round < repeat; round++) {
        for (const renderer of roundOrder(renderers, round)) {
          console.log(`\n== ${renderer} on ${scene}  (round ${round + 1}/${repeat})`);
          const out = runRunner('1', ['--no-csv'], cellArgs(scene, renderer));
          const resultsDir = /results: (\S+)\s*$/m.exec(out)?.[1];
          if (!resultsDir || !existsSync(join(resultsDir, 'run.json'))) {
            console.log('   no run.json written (runner failed to start); round skipped');
            continue;
          }
          const reportPath = /^report: (\S+)/m.exec(out)?.[1];
          rounds.get(renderer)!.push({
            runDir: relative(REPO, resultsDir),
            summary: JSON.parse(readFileSync(join(resultsDir, 'run.json'), 'utf8')) as RunSummary,
            manifest: reportPath ? (JSON.parse(readFileSync(reportPath, 'utf8')) as ReportManifest) : null,
          });
        }
      }
      for (const renderer of renderers) {
        const done = rounds.get(renderer)!;
        if (done.length === 0) {
          console.log(`\n== ${renderer} on ${scene}: no rounds finished`);
          continue;
        }
        // Run k = k-th finished round (a skipped round leaves no gap, as a --repeat N run has none).
        const mergedRunId = `${done[0]!.summary.runId}-x${repeat}`;
        const mergedDir = join('bench', 'results', 'incoming', mergedRunId);
        const merged = mergeRounds(done, mergedRunId, mergedDir);
        mkdirSync(join(REPO, mergedDir), { recursive: true });
        for (const c of merged.copies) if (existsSync(join(REPO, c.from))) cpSync(join(REPO, c.from), join(REPO, c.to), { recursive: true });
        writeFileSync(join(REPO, mergedDir, 'run.json'), `${JSON.stringify(merged.summary, null, 2)}\n`);
        const withReport = done.findIndex((d) => d.manifest);
        if (withReport >= 0) {
          const manifest = retargetManifest(done[withReport]!.manifest!, withReport + 1, mergedRunId);
          writeFileSync(join(REPO, mergedDir, 'report.json'), `${JSON.stringify(manifest, null, 2)}\n`);
          parts.push({ manifest, dir: join(REPO, mergedDir) });
        }
        const written = appendCsv(join(REPO, 'bench', 'results', 'competitor.csv'), merged.rows);
        console.log(`\n== ${renderer} on ${scene}: merged ${done.length} rounds -> ${mergedDir}  (csv: ${written} rows)`);
        record(scene, renderer, join(REPO, mergedDir));
      }
      continue;
    }
    for (const renderer of renderers) {
      console.log(`\n== ${renderer} on ${scene}`);
      const out = runRunner(values.repeat!, [], cellArgs(scene, renderer));
      const reportPath = /^report: (\S+)/m.exec(out)?.[1];
      if (!reportPath) {
        console.log('   no report written (run failed or nothing uploaded)');
        continue;
      }
      parts.push({ manifest: JSON.parse(readFileSync(reportPath, 'utf8')) as ReportManifest, dir: dirname(reportPath) });
      record(scene, renderer, dirname(reportPath));
    }
  }
  mkdirSync(outDir, { recursive: true });
  writeFileSync(runsFile, `${JSON.stringify(runLog, null, 2)}\n`);
  const merged = mergeManifests(parts, outDir, values.name!);
  // A re-run cell replaces the earlier one (--append): the last cell per (scene, renderer) wins.
  const lastIndex = new Map(merged.cells.map((c, i) => [`${c.scene}\u0000${c.renderer}`, i] as const));
  merged.cells = merged.cells.filter((c, i) => lastIndex.get(`${c.scene}\u0000${c.renderer}`) === i);
  writeFileSync(join(outDir, 'report.json'), `${JSON.stringify(merged, null, 2)}\n`);
  console.log(`\nreport: ${join(outDir, 'report.json')}  (viewer ?report=/results/reports/${values.name}/report.json)`);
  console.log(posix.join('cells', String(merged.cells.length)));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
