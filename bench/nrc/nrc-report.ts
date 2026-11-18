/**
 * Track N write-up (M8 / N8): builds a bench-viewer report of the A/B images behind bench/results/NRC.md.
 *
 *   node bench/nrc/nrc-report.ts            then  pnpm --filter @pathtracer/bench viewer  and open  ?report=/results/reports/nrc/report.json
 *
 * Rows are views (scene, stage, frame count), columns plain path tracing / NRC (network) / hash grid. Each cell is a PFM the study pages
 * uploaded (`nrc-study`, `nrc-pipeline-study`), copied from bench/results/incoming/<run>/1-run/ into bench/results/reports/nrc/ (both
 * git-ignored, so this only works on the machine that ran them: the RTX 3060 runs of 2026-11-17/18 listed in RUNS). A row's reference is
 * the run's own (our 4096 spp path tracer at the same view) where the study uploaded it; the pipeline rows have none (their stages are
 * 960 × 540 and 1920 × 1080, the reference 1280 × 720), so pick two cells to compare them. Metrics are the pages' own numbers (relMSE).
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const bench = join(dirname(fileURLToPath(import.meta.url)), '..');
const incoming = join(bench, 'results', 'incoming');
const out = join(bench, 'results', 'reports', 'nrc');

/** The runs (RTX 3060 Laptop, Chrome 157). Network and plain come from the network run; plain is identical in every run of a pair (same seeds). */
const RUNS = {
  white: { network: '20261117T154737Z-nrc-study', hash: '20261117T214922Z-nrc-study' },
  glossy: { network: '20261117T233402Z-nrc-study', hash: '20261117T234918Z-nrc-study' },
  svgf: { network: '20261117T223018Z-nrc-pipeline-study', hash: '20261117T222843Z-nrc-pipeline-study' },
  oidn: { network: '20261117T223329Z-nrc-pipeline-study', hash: '20261117T223154Z-nrc-pipeline-study' },
} as const;

type Group = keyof typeof RUNS;
interface Row {
  id: string;
  name: string;
  group: Group;
  /** Upload name with `{side}` for plain / nrc, and the metric key likewise. */
  image: string;
  metric?: string;
  reference?: string;
}

const ROWS: Row[] = [
  { id: 'white-warm-f16', name: 'sponza-white, warm cache, 16 frames (720p)', group: 'white', image: 'warm-{side}-f16', metric: 'warm_{side}_relmse_f16', reference: 'static-reference' },
  { id: 'white-warm-f256', name: 'sponza-white, warm cache, 256 frames', group: 'white', image: 'warm-{side}-f256', metric: 'warm_{side}_relmse_f256', reference: 'static-reference' },
  { id: 'white-unseen', name: 'sponza-white, unseen corridor, frame 64 after the cut (single frame)', group: 'white', image: 'unseen-cut-{side}-t64', reference: 'unseen-reference' },
  { id: 'glossy-warm-f16', name: 'sponza-glossy, warm cache, 16 frames', group: 'glossy', image: 'warm-{side}-f16', metric: 'warm_{side}_relmse_f16', reference: 'static-reference' },
  { id: 'glossy-warm-f256', name: 'sponza-glossy, warm cache, 256 frames', group: 'glossy', image: 'warm-{side}-f256', metric: 'warm_{side}_relmse_f256', reference: 'static-reference' },
  { id: 'svgf-f1-denoised', name: 'M9 chain, SVGF, frame 1 denoised (960 × 540)', group: 'svgf', image: 'static-f1-{side}-denoised', metric: 'static_f1_{side}_denoised_relmse' },
  { id: 'svgf-f1-final', name: 'M9 chain, SVGF, frame 1 final (1920 × 1080)', group: 'svgf', image: 'static-f1-{side}-final', metric: 'static_f1_{side}_final_relmse' },
  { id: 'oidn-f1-denoised', name: 'M9 chain, OIDN small, frame 1 denoised', group: 'oidn', image: 'static-f1-{side}-denoised', metric: 'static_f1_{side}_denoised_relmse' },
  { id: 'oidn-f16-denoised', name: 'M9 chain, OIDN small, still frame 16 denoised', group: 'oidn', image: 'static-f16-{side}-denoised', metric: 'static_f16_{side}_denoised_relmse' },
  { id: 'oidn-f16-final', name: 'M9 chain, OIDN small, still frame 16 final', group: 'oidn', image: 'static-f16-{side}-final', metric: 'static_f16_{side}_final_relmse' },
];

const RENDERERS = [
  { id: 'plain', name: 'plain path tracing (1 spp per frame)' },
  { id: 'network', name: 'NRC: network' },
  { id: 'hash', name: 'NRC: hash grid' },
] as const;

function metrics(run: string): Record<string, number> {
  const r = JSON.parse(readFileSync(join(incoming, run, 'run.json'), 'utf8')) as { runs: Array<{ result?: { metrics?: Record<string, number> } }> };
  return r.runs[0]?.result?.metrics ?? {};
}

mkdirSync(out, { recursive: true });
const missing: string[] = [];
const copy = (run: string, name: string): string | null => {
  const src = join(incoming, run, '1-run', `${name}.pfm`);
  if (!existsSync(src)) {
    missing.push(`${run}/${name}.pfm`);
    return null;
  }
  const rel = `${run.slice(0, 16)}-${name}.pfm`;
  copyFileSync(src, join(out, rel));
  return rel;
};

const scenes: Array<{ id: string; name: string; reference?: string }> = [];
const cells: Array<{ scene: string; renderer: string; image: string; reference?: string; metrics?: Record<string, number> }> = [];
for (const row of ROWS) {
  const runs = RUNS[row.group];
  const ref = row.reference ? copy(runs.network, row.reference) : null;
  scenes.push({ id: row.id, name: row.name, ...(ref ? { reference: ref } : {}) });
  for (const r of RENDERERS) {
    const run = r.id === 'hash' ? runs.hash : runs.network;
    const side = r.id === 'plain' ? 'plain' : 'nrc';
    const image = copy(run, row.image.replace('{side}', side));
    if (!image) continue;
    const m = row.metric ? metrics(run)[row.metric.replace('{side}', side)] : undefined;
    cells.push({ scene: row.id, renderer: r.id, image, ...(m !== undefined ? { metrics: { relMse: m } } : {}) });
  }
}
writeFileSync(join(out, 'report.json'), JSON.stringify({ run: 'Track N: plain vs NRC network vs hash grid (RTX 3060 Laptop, 2026-11-17/18)', scenes, renderers: RENDERERS, cells }, null, 1));
console.log(`wrote ${join(out, 'report.json')}: ${scenes.length} views, ${cells.length} cells${missing.length ? `; missing: ${missing.join(', ')}` : ''}`);
console.log('open: pnpm --filter @pathtracer/bench viewer, then ?report=/results/reports/nrc/report.json');
