// P2.32b A/B table: the albedo layering with / without the fitted correction against the stochastic walk (same spp) and the pbrt / Mitsuba references.
//   node bench/layered-correction-ab.ts --reports stoch=p232b-stoch,int=p232b-int,lut=p232b-lut,tex=p232b-tex [--scenes a,b]
// Reads bench/results/reports/<name>/runs.json (bench/adapters/matrix.ts). First report = the baseline ("stoch").
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from './src/adapter.ts';
import { compareImages } from './metrics/metrics.ts';
import { readPFM } from './metrics/pfm.ts';
import type { RunLogEntry } from './adapters/matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({ options: { reports: { type: 'string' }, scenes: { type: 'string', default: '' }, block: { type: 'string', default: '32' } } });
const reports = (values.reports as string).split(',').map((s) => s.split('=') as [string, string]);
const block = Number(values.block);
const load = (p: string): Image => readPFM(readFileSync(p));
const pct = (v: number): string => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`;
const sci = (v: number): string => v.toExponential(1);

function runsOf(name: string): Map<string, string> {
  const runs = JSON.parse(readFileSync(join(REPO, 'bench/results/reports', name, 'runs.json'), 'utf8')) as RunLogEntry[];
  return new Map(runs.map((e) => [e.scene, join(REPO, e.runDir, '1-run', `${e.renderer}__${e.scene}.pfm`)]));
}

function lum(img: Image, i: number): number {
  const o = i * img.channels;
  return (img.data[o]! + img.data[o + 1]! + img.data[o + 2]!) / 3;
}

/** ratios of block means test / ref for blocks whose ref mean is above the median block mean (the bright, base-dominated areas) */
function blockRatios(test: Image, ref: Image): { all: number[]; bright: number[] } {
  const bx = Math.floor(test.width / block);
  const by = Math.floor(test.height / block);
  const items: { r: number; m: number }[] = [];
  for (let j = 0; j < by; j++) for (let i = 0; i < bx; i++) {
    let a = 0;
    let b = 0;
    for (let y = 0; y < block; y++) for (let x = 0; x < block; x++) { const idx = (j * block + y) * test.width + i * block + x; a += lum(test, idx); b += lum(ref, idx); }
    if (b > 1e-3 * block * block) items.push({ r: a / b, m: b });
  }
  const med = items.map((x) => x.m).sort((p, q) => p - q)[Math.floor(items.length / 2)] ?? 0;
  return { all: items.map((x) => x.r), bright: items.filter((x) => x.m >= med).map((x) => x.r) };
}
const q = (a: number[], p: number): number => [...a].sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(p * a.length))] ?? NaN;
const maps = reports.map(([tag, name]) => ({ tag, runs: runsOf(name) }));
const scenes = values.scenes ? (values.scenes as string).split(',') : [...(maps[0]!.runs.keys())];
console.log(`| scene | variant | relBias vs stoch | block ${block}px ratio to stoch (median, p10..p90, bright blocks) | relBias vs pbrt | relBias vs Mitsuba | relMSE vs pbrt | relMSE vs Mitsuba | non-finite |`);
console.log('|---|---|---|---|---|---|---|---|---|');
for (const scene of scenes) {
  const pbrtPath = join(REPO, 'bench/references', scene, 'pbrt-v4-cpu-4096spp.pfm');
  const mitPath = join(REPO, 'bench/references', scene, 'mitsuba3-llvm_ad_rgb-4096spp.pfm');
  const pbrt = existsSync(pbrtPath) ? load(pbrtPath) : null;
  const mit = existsSync(mitPath) ? load(mitPath) : null;
  const base = maps[0]!.runs.get(scene);
  if (!base) continue;
  const baseImg = load(base);
  for (const { tag, runs } of maps) {
    const p = runs.get(scene);
    if (!p || !existsSync(p)) continue;
    const img = load(p);
    const vsBase = compareImages(img, baseImg);
    const br = blockRatios(img, baseImg);
    const vp = pbrt ? compareImages(img, pbrt) : null;
    const vm = mit ? compareImages(img, mit) : null;
    console.log(`| ${scene} | ${tag} | ${tag === maps[0]!.tag ? '0 (baseline)' : pct(vsBase.relBias)} | ${q(br.all, 0.5).toFixed(3)}, ${q(br.all, 0.1).toFixed(3)}..${q(br.all, 0.9).toFixed(3)}; bright ${q(br.bright, 0.5).toFixed(3)} (${q(br.bright, 0.1).toFixed(3)}..${q(br.bright, 0.9).toFixed(3)}) | ${vp ? pct(vp.relBias) : '-'} | ${vm ? pct(vm.relBias) : '-'} | ${vp ? sci(vp.relMse) : '-'} | ${vm ? sci(vm.relMse) : '-'} | ${(vsBase as { nonFinite?: number }).nonFinite ?? '-'} |`);
  }
}
