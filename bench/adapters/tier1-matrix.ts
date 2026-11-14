/**
 * Tier 1 feature-coverage matrix (P2.31): rows = extensions / features, columns = ours, three-gpu-pathtracer, mutracer, pbrt-v4, Mitsuba 3.
 *
 *   node bench/adapters/matrix.ts --renderers tgpt-wavefront,mutracer,pt-megakernel --scenes <full and --no-<tag> scenes> --spp 1024 --snapshots 512 --force \
 *       --name t1cov-1024 --load-wait 0 --extra pt_tint=0,pt_coverage=1          (the scene list: ROWS in tier1-coverage.ts, `--list-scenes` prints it)
 *   node bench/adapters/tier1-matrix.ts --report t1cov-1024
 *
 * Writes bench/results/reports/tier1/coverage.json and bench/results/TIER1-COVERAGE.md. The measured columns come from the ablation runs of the
 * report (rules in tier1-coverage.ts); the pbrt / Mitsuba columns come from running both exporters on a probe material of each row.
 * Page-free: no page-client import, so the test can import the pure part. References: `bench/references/<scene>/{pbrt-v4-cpu,mitsuba3-llvm_ad_rgb}-<spp>spp.pfm`.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { toMitsuba } from '../exporters/mitsuba.ts';
import { toPbrt } from '../exporters/pbrt.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import { ablationSceneName } from '../scenes/tier1/ablations.ts';
import { classifyOpenPbr } from '../scenes/openpbr.ts';
import type { OpenPbrParams, SceneDesc } from '../scenes/schema.ts';
import type { RunLogEntry } from './matrix.ts';
import {
  ADAPTER_OF, AGREE_A, AGREE_B, RENDERER_IDS, ROWS, SENSITIVITY, SYMBOL, judgeCell, nativeOutcome, renderMarkdown,
  type Cell, type CoverageJson, type CoverageRow, type CoverageRowDef, type NativeProbeResult, type RefClass, type RendererColumn,
} from './tier1-coverage.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const RESULTS = join(REPO, 'bench', 'results');
const ASSETS = join(REPO, 'bench', 'assets');

const load = (path: string): Image => readPFM(readFileSync(path));

/** The highest-spp reference PFM of a scene for a renderer prefix, or null. */
export function referenceFile(scene: string, prefix: 'pbrt-v4-cpu' | 'mitsuba3-llvm_ad_rgb'): { path: string; spp: number } | null {
  const dir = join(REPO, 'bench', 'references', scene);
  if (!existsSync(dir)) return null;
  const best = readdirSync(dir)
    .filter((n) => n.startsWith(`${prefix}-`) && n.endsWith('spp.pfm'))
    .map((n) => ({ path: join(dir, n), spp: Number(/-(\d+)spp\.pfm$/.exec(n)?.[1]) }))
    .sort((a, b) => b.spp - a.spp)[0];
  return best ?? null;
}

const meanSum = (image: Image): number => {
  let s = 0;
  const n = image.width * image.height;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
    const v = image.data[i * image.channels + c]!;
    if (Number.isFinite(v)) s += v;
  }
  return s;
};
const maxValue = (image: Image): number => {
  let m = 0;
  const n = image.width * image.height;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
    const v = image.data[i * image.channels + c]!;
    if (Number.isFinite(v) && v > m) m = v;
  }
  return m;
};
/**
 * Copy with every pixel that has a channel above `cap` set to NaN, and how many that was. compareImages leaves non-finite pixels out (on either side), so a
 * firefly of 25000 on a dark pixel (relMSE has eps 1e-2: it would be 6e10 / pixels) cannot decide a cell; the count is reported with the cell.
 */
export function maskFireflies(image: Image, cap: number): { image: Image; masked: number } {
  const data = new Float32Array(image.data);
  let masked = 0;
  const n = image.width * image.height;
  for (let i = 0; i < n; i++) {
    let hit = false;
    for (let c = 0; c < 3; c++) if (data[i * image.channels + c]! > cap) hit = true;
    if (!hit) continue;
    masked++;
    for (let c = 0; c < 3; c++) data[i * image.channels + c] = Number.NaN;
  }
  return { image: { ...image, data }, masked };
}

const viewerUrl = (abs: string): string => `/${abs.startsWith(RESULTS) ? 'results' : 'references'}/${relative(abs.startsWith(RESULTS) ? RESULTS : join(REPO, 'bench', 'references'), abs).split('\\').join('/')}`;
const refQuery = (a: string, b: string, r?: string): string => `a=${viewerUrl(a)}&b=${viewerUrl(b)}${r ? `&r=${viewerUrl(r)}` : ''}`;

function probeScene(p: Record<string, unknown>): SceneDesc {
  return {
    version: 1, name: 'probe', purpose: 'coverage probe', tier: 1,
    camera: { type: 'perspective', position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], yFovDeg: 30 },
    film: { width: 16, height: 16, filter: 'box' }, integrator: { maxDepth: 8 },
    materials: { m: { type: 'openpbr', ...(p as OpenPbrParams) } },
    meshes: [{ name: 's', geometry: { type: 'sphere', radius: 1, segments: 8, rings: 4 }, material: 'm' }],
    environment: { type: 'uniform', radiance: [1, 1, 1] },
  };
}

/** Runs one exporter on the probe, strict first, then with `coverage`. */
export function probeNative(renderer: 'pbrt' | 'mitsuba', params: Record<string, unknown>): NativeProbeResult {
  const fn = renderer === 'pbrt' ? toPbrt : toMitsuba;
  const cls = classifyOpenPbr(params as OpenPbrParams).class as RefClass;
  const scene = probeScene(params);
  try {
    return { ranStrict: true, ranCoverage: false, cls, notes: fn(scene, { assetsDir: ASSETS, spp: 1 }).approximations };
  } catch (strict) {
    try {
      return { ranStrict: false, ranCoverage: true, cls, notes: fn(scene, { assetsDir: ASSETS, spp: 1, coverage: true }).approximations, error: String((strict as Error).message).slice(0, 200) };
    } catch (e) {
      return { ranStrict: false, ranCoverage: false, cls, notes: [], error: String((e as Error).message).slice(0, 200) };
    }
  }
}

const NO_SCENE: Cell = { status: 'no-scene', symbol: SYMBOL['no-scene'], notes: ['no wave-1 scene yet'], reportRef: null };

function main(): void {
  const { values } = parseArgs({ options: { report: { type: 'string', default: 't1cov-1024' }, out: { type: 'string', default: join(RESULTS, 'reports', 'tier1') }, md: { type: 'string', default: join(RESULTS, 'TIER1-COVERAGE.md') }, 'list-scenes': { type: 'boolean', default: false } } });
  if (values['list-scenes']) {
    const names = new Set<string>();
    for (const r of ROWS) if (r.scene) {
      names.add(r.scene);
      if (r.ablation) names.add(ablationSceneName(r.scene, r.ablation));
    }
    console.log([...names].join(','));
    return;
  }
  const reportDir = join(RESULTS, 'reports', values.report!);
  const runs = JSON.parse(readFileSync(join(reportDir, 'runs.json'), 'utf8')) as RunLogEntry[];
  const find = (scene: string, adapter: string): RunLogEntry | undefined => runs.find((e) => e.scene === scene && e.renderer === adapter);
  const spp = runs[0]?.spp ?? 0;
  const halfSpp = Math.max(...(runs[0]?.snapshots ?? [0]));
  const pfm = (e: RunLogEntry, snapshot?: number): string => join(REPO, e.runDir, '1-run', `${e.renderer}__${e.scene}${snapshot !== undefined ? `__spp${snapshot}` : ''}.pfm`);

  // Run conditions from the runner's own load samples.
  const loads: { load1: number; gpuProcs: number | null; loaded: boolean }[] = [];
  const versions: Record<string, string> = {};
  for (const e of runs) {
    const rj = join(REPO, e.runDir, 'run.json');
    if (!existsSync(rj)) continue;
    const j = JSON.parse(readFileSync(rj, 'utf8')) as { commit?: string; runs?: { load?: { load1: number; gpuProcs: number | null; loaded: boolean } }[]; browser?: string | null };
    for (const r of j.runs ?? []) if (r.load) loads.push(r.load);
    if (j.commit) versions['repo commit'] = j.commit.slice(0, 10);
    if (j.browser) versions['browser'] = j.browser;
  }
  const benchPkg = JSON.parse(readFileSync(join(REPO, 'bench', 'package.json'), 'utf8')) as { devDependencies?: Record<string, string>; dependencies?: Record<string, string> };
  for (const k of ['three-gpu-pathtracer', '@meimu/mutracer']) versions[k] = (benchPkg.devDependencies?.[k] ?? benchPkg.dependencies?.[k] ?? '?').replace(/^[\^~]/, '');
  const range = (xs: number[]): { min: number; max: number } => ({ min: Math.min(...xs), max: Math.max(...xs) });
  const gpuProcs = loads.map((l) => l.gpuProcs).filter((v): v is number => v !== null);

  const rows: CoverageRow[] = [];
  const floorCache = new Map<string, { floor: number; pbrt: Image; mitsuba: Image } | null>();
  const refsOf = (scene: string): { floor: number; pbrt: Image; mitsuba: Image } | null => {
    if (!floorCache.has(scene)) {
      const p = referenceFile(scene, 'pbrt-v4-cpu');
      const m = referenceFile(scene, 'mitsuba3-llvm_ad_rgb');
      floorCache.set(scene, p && m ? { pbrt: load(p.path), mitsuba: load(m.path), floor: compareImages(load(p.path), load(m.path)).relMse } : null);
    }
    return floorCache.get(scene)!;
  };

  for (const def of ROWS) {
    const cells = {} as Record<RendererColumn, Cell>;
    const stripped = def.scene && def.ablation ? ablationSceneName(def.scene, def.ablation) : null;
    const refs = def.scene && (def.refClass === 'A' || def.refClass === 'B') ? (def.refs ?? ['pbrt', 'mitsuba']) : [];
    for (const col of ['ours', 'tgpt', 'mutracer'] as const) {
      if (!def.scene || !stripped) {
        cells[col] = NO_SCENE;
        continue;
      }
      const adapter = ADAPTER_OF[col];
      const fullE = find(def.scene, adapter);
      const stripE = find(stripped, adapter);
      const fail = (why: string): Cell => ({ status: 'missing', symbol: SYMBOL.missing, notes: [why], reportRef: null });
      if (!fullE || !stripE) {
        cells[col] = fail(`no render in the report (${!fullE ? def.scene : stripped} did not finish or was skipped)`);
        continue;
      }
      const fullPath = pfm(fullE);
      const stripPath = pfm(stripE);
      const halfPath = pfm(fullE, halfSpp);
      if (![fullPath, stripPath, halfPath].every(existsSync)) {
        cells[col] = fail('render images missing on disk');
        continue;
      }
      const R = refsOf(def.scene);
      // Firefly cap: 4 x the reference maximum (ours' own maximum where the scene has no reference).
      const rawFull = load(fullPath);
      const oursE = find(def.scene, ADAPTER_OF.ours);
      const cap = 4 * (R ? Math.max(maxValue(R.pbrt), maxValue(R.mitsuba)) : maxValue(oursE && existsSync(pfm(oursE)) ? load(pfm(oursE)) : rawFull));
      const maskedFull = maskFireflies(rawFull, cap);
      const full = maskedFull.image;
      const noise = compareImages(full, maskFireflies(load(halfPath), cap).image).relMse;
      const ablation = compareImages(full, maskFireflies(load(stripPath), cap).image);
      const judged: { name: string; relMSE: number; floor: number }[] = [];
      const cell: Cell = { status: 'missing', symbol: SYMBOL.missing, notes: [], reportRef: null, ablationRelMSE: ablation.relMse, noise };
      if (R) {
        cell.floor = R.floor;
        cell.relMSE_pbrt = compareImages(full, R.pbrt).relMse;
        cell.relMSE_mitsuba = compareImages(full, R.mitsuba).relMse;
        cell.relBias_pbrt = meanSum(rawFull) / meanSum(R.pbrt) - 1;
        cell.relBias_mitsuba = meanSum(rawFull) / meanSum(R.mitsuba) - 1;
        for (const name of refs) judged.push({ name, relMSE: name === 'pbrt' ? cell.relMSE_pbrt : cell.relMSE_mitsuba, floor: R.floor });
      }
      if (maskedFull.masked > 0) {
        cell.fireflyPixels = maskedFull.masked;
        cell.notes.push(`${maskedFull.masked} firefly pixel(s) above ${cap.toFixed(1)} (4x the reference maximum, brightest ${maxValue(rawFull).toFixed(0)}) left out of every relMSE; the mean bias below includes them`);
      }
      if (ablation.nonFiniteTest + (ablation.nonFiniteRef ?? 0) > 0) cell.notes.push(`non-finite pixels excluded: ${ablation.nonFiniteTest} full, ${ablation.nonFiniteRef ?? 0} stripped`);
      const v = judgeCell({ refClass: def.refClass as RefClass, ablation: ablation.relMse, noise, refs: judged });
      cell.status = v.status;
      cell.symbol = SYMBOL[v.status];
      cell.ratio = v.ratio;
      cell.notes.push(...v.notes);
      const primary = refs[0] ? referenceFile(def.scene, refs[0] === 'pbrt' ? 'pbrt-v4-cpu' : 'mitsuba3-llvm_ad_rgb') : null;
      cell.reportRef = refQuery(fullPath, stripPath, primary?.path);
      cells[col] = cell;
    }
    for (const col of ['pbrt', 'mitsuba'] as const) {
      const res = nativeOutcome(probeNative(col, def.probe));
      const R = def.scene ? refsOf(def.scene) : null;
      const ref = def.scene ? { p: referenceFile(def.scene, 'pbrt-v4-cpu'), m: referenceFile(def.scene, 'mitsuba3-llvm_ad_rgb') } : null;
      cells[col] = {
        status: res.status, symbol: SYMBOL[res.status],
        ...(R ? { floor: R.floor } : {}),
        notes: [`exporter: ${res.outcome}`, ...res.notes.map((n) => n.slice(0, 260))],
        reportRef: ref?.p && ref.m ? refQuery(ref.p.path, ref.m.path) : null,
      };
    }
    rows.push({
      id: def.id, label: def.label, extension: def.extension, refClass: def.refClass, scene: def.scene ?? null, ablation: def.ablation ?? null, stripped,
      refs, noScene: !def.scene, ...(def.refNote || def.strip ? { note: [def.strip ? `stripped: ${def.strip}` : '', def.refNote ?? ''].filter(Boolean).join('. ') } : {}), cells,
    });
  }

  const json: CoverageJson = {
    schema: 1, generated: new Date().toISOString().slice(0, 10), report: values.report!, spp, halfSpp,
    thresholds: { sensitivity: SENSITIVITY, agreeA: AGREE_A, agreeB: AGREE_B },
    conditions: { ...(loads.length ? { load1: range(loads.map((l) => l.load1)) } : {}), ...(gpuProcs.length ? { gpuProcs: range(gpuProcs) } : {}), loadedRuns: loads.filter((l) => l.loaded).length, runs: loads.length, versions },
    legend: { '●': 'supported', '◐': 'approximate / off reference', '○': 'missing', '–': 'no scene yet' },
    rows,
  };
  mkdirSync(values.out!, { recursive: true });
  writeFileSync(join(values.out!, 'coverage.json'), `${JSON.stringify(json, null, 2)}\n`);
  writeFileSync(values.md!, renderMarkdown(json));
  console.log(`wrote ${join(values.out!, 'coverage.json')} and ${values.md}`);
  for (const r of rows) console.log(`${r.id.padEnd(24)} ${RENDERER_IDS.map((id) => `${id}:${r.cells[id].symbol}`).join('  ')}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
