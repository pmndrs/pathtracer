/**
 * Runs web adapters over bench scenes (default Tier 0; `--scenes sponza-white` for Tier 2; `--scenes all` for every scene) through the bench runner and merges the per-run report manifests into one grid
 * (renderer x scene) for the viewer.
 *
 *   node bench/adapters/matrix.ts [--renderers tgpt-wavefront,tgpt-megakernel,mutracer] [--scenes a,b] [--spp 256]
 *                                 [--snapshots 16,64] [--name baseline] [--timeout 600000] [--force] [--budget N] [--emitters rect|mesh]
 *                                 [--repeat 3] [--append]
 *
 * --repeat N: the runner repeats every cell N times (CSV rows run 1..N plus median; uploads go to slots 1-run..N-run; the report
 * shows run 1). --append: add the new cells to an existing report of the same name instead of replacing it (lets one report mix
 * different spp per cell). `runs.json` next to report.json lists every cell's runner directory (for bench/adapters/baseline.ts).
 *
 * One runner invocation per (renderer, scene), so a crash in one cell cannot take the others down. Output:
 * bench/results/reports/<name>/report.json (open with ?report=/results/reports/<name>/report.json in the viewer).
 * Each cell keeps the CSV rows the runner appends to bench/results/competitor.csv.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
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
      force: { type: 'boolean', default: false },
      repeat: { type: 'string', default: '1' },
      append: { type: 'boolean', default: false },
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
  for (const scene of sceneNames) {
    for (const renderer of renderers) {
      const args = [`renderer=${renderer}`, `scene=${scene}`, `spp=${values.spp}`];
      if (values.snapshots) args.push(`snapshots=${values.snapshots}`);
      if (values.budget) args.push(`budget=${values.budget}`);
      if (values.emitters) args.push(`emitters=${values.emitters}`);
      if (values.force) args.push('force=1');
      console.log(`\n== ${renderer} on ${scene}`);
      const run = spawnSync(process.execPath, [join(REPO, 'bench/runner/run.ts'), '--page', 'competitor', '--suite', 'competitor', '--timeout', values.timeout!, '--repeat', values.repeat!, '--args', ...args], { cwd: REPO, encoding: 'utf8' });
      const out = `${run.stdout}${run.stderr}`;
      const lines = out.split('\n').filter((l) => /PASS|FAIL|metrics:|FAILED|The page reported|Timed out/.test(l));
      console.log(lines.map((l) => l.slice(0, 400)).join('\n'));
      const reportPath = /^report: (\S+)/m.exec(out)?.[1];
      if (!reportPath) {
        console.log('   no report written (run failed or nothing uploaded)');
        continue;
      }
      parts.push({ manifest: JSON.parse(readFileSync(reportPath, 'utf8')) as ReportManifest, dir: dirname(reportPath) });
      const entry: RunLogEntry = { scene, renderer, emitters: values.emitters || 'default', spp: Number(values.spp), snapshots: values.snapshots ? values.snapshots.split(',').map(Number) : [], repeat: Number(values.repeat), runDir: relative(REPO, dirname(reportPath)) };
      runLog.splice(0, runLog.length, ...runLog.filter((e) => !(e.scene === scene && e.renderer === renderer && e.emitters === entry.emitters)), entry);
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
