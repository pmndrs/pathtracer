/**
 * Check the cached references: analytic (furnace) values per renderer and pbrt-v4 vs Mitsuba agreement per scene.
 *
 *   node bench/references/compare.ts [--scenes a,b] [--write]
 *
 * Uses bench/metrics (relMSE eps 1e-2, mean signed bias, SMAPE). `--write` stores bench/references/agreement.json
 * (committed: small) next to the git-ignored PFMs. The agreement between two high-spp renders of different renderers
 * is our noise floor: a test image that matches the reference as well as Mitsuba matches pbrt is "as converged as the
 * references allow".
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { REFERENCES_DIR } from './render.ts';
import { tier0Scenes } from '../scenes/tier0/index.ts';
import { readPFM } from '../metrics/pfm.ts';
import { compareImages } from '../metrics/metrics.ts';
import { checkMean, checkUniform } from '../metrics/analytic.ts';
import type { Image } from '../src/adapter.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { quadLightAnalyticImage } from '../scenes/tier0/quad-light-analytic.ts';

interface Found { id: string; spp: number; pfm: string; seconds?: number }

/** Highest-spp file per renderer id for a scene. */
export function findReferences(scene: string): Found[] {
  const dir = join(REFERENCES_DIR, scene);
  if (!existsSync(dir)) return [];
  const best = new Map<string, Found>();
  for (const f of readdirSync(dir)) {
    const m = /^(.+)-(\d+)spp\.pfm$/.exec(f);
    if (!m) continue;
    const spp = Number(m[2]);
    const id = m[1]!;
    if (!best.has(id) || best.get(id)!.spp < spp) {
      const side = join(dir, f.replace(/\.pfm$/, '.json'));
      const seconds = existsSync(side) ? (JSON.parse(readFileSync(side, 'utf8')) as { seconds: number }).seconds : undefined;
      best.set(id, { id, spp, pfm: join(dir, f), seconds });
    }
  }
  return [...best.values()].sort((a, b) => a.id.localeCompare(b.id));
}

const load = (f: Found): Image => readPFM(readFileSync(f.pfm));
const fmt = (v: number, d = 5) => (Number.isFinite(v) ? v.toPrecision(d) : String(v));

export interface AgreementRow {
  scene: string;
  a: string; b: string; sppA: number; sppB: number;
  relMse: number; smape: number; meanAbsDiff: number; relBias: number; biasRGB: number[];
}

function meanAbsDiff(a: Image, b: Image): number {
  let s = 0;
  const n = a.width * a.height;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) s += Math.abs(a.data[i * a.channels + c]! - b.data[i * b.channels + c]!);
  return s / (n * 3);
}

function furnaceReport(scene: SceneDesc, found: Found[], out: string[]): void {
  const e = scene.expected;
  if (!e) return;
  for (const f of found) {
    const img = load(f);
    const uni = checkUniform(img, e.value, e.tolerance, { relative: true });
    const mean = checkMean(img, e.value, e.tolerance / 10, { relative: true });
    out.push(
      `| ${scene.name} | ${f.id} | ${f.spp} | ${e.value[0]} | ${uni.mean.map((v) => fmt(v, 5)).join(' ')} | ${fmt(mean.maxDeviation, 3)} | ${fmt(uni.maxDeviation, 3)} | ${uni.ok && mean.ok ? 'PASS' : 'FAIL'} |`,
    );
  }
}

/** Quad-light has a closed-form image (direct lighting only): masked mean/max relative deviation per renderer. */
function quadLightReport(scene: SceneDesc, found: Found[], out: string[]): void {
  const exp = quadLightAnalyticImage(scene);
  for (const f of found) {
    const img = load(f);
    let n = 0, sumRel = 0, maxRel = 0, sumT = 0, sumE = 0;
    for (let i = 0; i < scene.film.width * scene.film.height; i++) {
      const e = exp[i * 3]!;
      if (Number.isNaN(e)) continue;
      const v = img.data[i * img.channels + 1]!;
      n++; sumT += v; sumE += e;
      const rel = (v - e) / e;
      sumRel += rel; maxRel = Math.max(maxRel, Math.abs(rel));
    }
    out.push(`| ${scene.name} | ${f.id} | ${f.spp} | analytic radiance image (${n} plane px) | mean ${fmt(sumT / n, 5)} vs ${fmt(sumE / n, 5)} | mean rel dev ${fmt(sumRel / n, 3)} | max pixel rel dev ${fmt(maxRel, 3)} | ${Math.abs(sumRel / n) < 0.005 ? 'PASS' : 'FAIL'} |`);
  }
}

export function main(argv: string[]): void {
  const scenesArg = argv.indexOf('--scenes');
  const names = scenesArg >= 0 ? argv[scenesArg + 1]!.split(',') : tier0Scenes.map((s) => s.name);
  const lines: string[] = [];
  const rows: AgreementRow[] = [];
  const furnace: string[] = [];
  for (const name of names) {
    const scene = tier0Scenes.find((s) => s.name === name)!;
    const found = findReferences(name);
    if (found.length === 0) { console.log(`no references for ${name}`); continue; }
    furnaceReport(scene, found, furnace);
    if (name === 'quad-light') quadLightReport(scene, found, furnace);
    const pbrt = found.find((f) => f.id === 'pbrt-v4-cpu');
    if (!pbrt) continue;
    const ref = load(pbrt);
    for (const f of found) {
      if (f === pbrt) continue;
      const img = load(f);
      const m = compareImages(img, ref);
      rows.push({
        scene: name, a: f.id, b: pbrt.id, sppA: f.spp, sppB: pbrt.spp,
        relMse: m.relMse, smape: m.smape, meanAbsDiff: meanAbsDiff(img, ref), relBias: m.relBias, biasRGB: m.biasRGB,
      });
    }
  }
  lines.push('## Image means and render times');
  lines.push('| scene | renderer | spp | seconds | mean RGB |');
  lines.push('|---|---|---|---|---|');
  for (const name of names) {
    for (const f of findReferences(name)) {
      const img = load(f);
      const n = img.width * img.height;
      const mean = [0, 0, 0];
      for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) mean[c]! += img.data[i * img.channels + c]! / n;
      lines.push(`| ${name} | ${f.id} | ${f.spp} | ${f.seconds ?? '?'} | ${mean.map((v) => fmt(v, 5)).join(' ')} |`);
    }
  }
  lines.push('');
  lines.push('## Furnace / analytic checks (relative tolerance; |mean-expected|/expected must be < tol/10, every pixel < tol)');
  lines.push('| scene | renderer | spp | expected | mean RGB | mean rel dev | max pixel rel dev | verdict |');
  lines.push('|---|---|---|---|---|---|---|---|');
  lines.push(...furnace);
  lines.push('', '## Agreement with pbrt-v4 (test = renderer in column 2, reference = pbrt-v4)');
  lines.push('| scene | test renderer | spp | relMSE (eps 1e-2) | SMAPE | mean abs diff | rel bias (mean t / mean r - 1) | bias RGB |');
  lines.push('|---|---|---|---|---|---|---|---|');
  for (const r of rows) {
    lines.push(`| ${r.scene} | ${r.a} | ${r.sppA} (ref ${r.sppB}) | ${fmt(r.relMse, 3)} | ${fmt(r.smape, 3)} | ${fmt(r.meanAbsDiff, 3)} | ${fmt(r.relBias, 3)} | ${r.biasRGB.map((v) => fmt(v, 2)).join(' ')} |`);
  }
  console.log(lines.join('\n'));
  if (argv.includes('--write')) {
    writeFileSync(join(REFERENCES_DIR, 'agreement.json'), JSON.stringify({ date: new Date().toISOString(), agreement: rows }, null, 2) + '\n');
    writeFileSync(join(REFERENCES_DIR, 'agreement.md'), lines.join('\n') + '\n');
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) main(process.argv.slice(2));
