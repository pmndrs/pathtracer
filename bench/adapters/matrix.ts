/**
 * Runs web adapters over Tier 0 scenes through the bench runner and merges the per-run report manifests into one grid
 * (renderer x scene) for the viewer.
 *
 *   node bench/adapters/matrix.ts [--renderers tgpt-wavefront,tgpt-megakernel,mutracer] [--scenes a,b] [--spp 256]
 *                                 [--snapshots 16,64] [--name baseline] [--timeout 600000] [--force] [--budget N] [--emitters rect|mesh]
 *
 * One runner invocation per (renderer, scene), so a crash in one cell cannot take the others down. Output:
 * bench/results/reports/<name>/report.json (open with ?report=/results/reports/<name>/report.json in the viewer).
 * Each cell keeps the CSV rows the runner appends to bench/results/competitor.csv.
 */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { ReportManifest } from '../viewer/src/report.ts';
import { exportScenes } from './export-scenes.ts';
import { ADAPTER_IDS } from './registry.ts';
import { tier0Scenes } from '../scenes/tier0/index.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');

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
    },
  });
  const renderers = values.renderers!.split(',').filter(Boolean);
  const sceneNames = values.scenes!.split(',').filter(Boolean);
  exportScenes(sceneNames);
  const parts: Array<{ manifest: ReportManifest; dir: string }> = [];
  for (const scene of sceneNames) {
    for (const renderer of renderers) {
      const args = [`renderer=${renderer}`, `scene=${scene}`, `spp=${values.spp}`];
      if (values.snapshots) args.push(`snapshots=${values.snapshots}`);
      if (values.budget) args.push(`budget=${values.budget}`);
      if (values.emitters) args.push(`emitters=${values.emitters}`);
      if (values.force) args.push('force=1');
      console.log(`\n== ${renderer} on ${scene}`);
      const run = spawnSync(process.execPath, [join(REPO, 'bench/runner/run.ts'), '--page', 'competitor', '--suite', 'competitor', '--timeout', values.timeout!, '--args', ...args], { cwd: REPO, encoding: 'utf8' });
      const out = `${run.stdout}${run.stderr}`;
      const lines = out.split('\n').filter((l) => /PASS|FAIL|metrics:|FAILED|The page reported|Timed out/.test(l));
      console.log(lines.map((l) => l.slice(0, 400)).join('\n'));
      const reportPath = /^report: (\S+)/m.exec(out)?.[1];
      if (!reportPath) {
        console.log('   no report written (run failed or nothing uploaded)');
        continue;
      }
      parts.push({ manifest: JSON.parse(readFileSync(reportPath, 'utf8')) as ReportManifest, dir: dirname(reportPath) });
    }
  }
  const outDir = join(REPO, 'bench', 'results', 'reports', values.name!);
  mkdirSync(outDir, { recursive: true });
  const merged = mergeManifests(parts, outDir, values.name!);
  writeFileSync(join(outDir, 'report.json'), `${JSON.stringify(merged, null, 2)}\n`);
  console.log(`\nreport: ${join(outDir, 'report.json')}  (viewer ?report=/results/reports/${values.name}/report.json)`);
  console.log(posix.join('cells', String(merged.cells.length)));
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
