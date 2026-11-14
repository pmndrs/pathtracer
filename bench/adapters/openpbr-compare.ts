/**
 * Our `openpbr` renders against the pbrt-v4 / Mitsuba 3 references of the Tier 1 scenes (P2.21, P2.22, P2.30) at whatever spp the references have.
 *
 *   node bench/adapters/openpbr-compare.ts --report <matrix name> [--scene <name>] [--out file.md]
 *
 * Per cell of `bench/results/reports/<report>/runs.json`: mean RGB of ours and of both references, relBias (mean ours / mean reference - 1) for each,
 * relMSE (eps 1e-2) of ours vs each and of pbrt vs Mitsuba (the floor at the references' spp, which includes THEIR noise: at 64 spp every number is
 * noise-dominated, judge means, relBias and block means), and the rms/max difference of 32x32 block means (structure; noise averages out).
 * References: `bench/references/<scene>/{pbrt-v4-cpu,mitsuba3-llvm_ad_rgb}-<spp>spp.pfm`, the highest spp present. Expectations of analytic scenes
 * (`SceneDesc.expected`) are checked as the image mean.
 */
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import { getScene } from '../scenes/index.ts';
import { blockMeanDiff } from './baseline.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const load = (path: string): Image => readPFM(readFileSync(path));
const mean = (image: Image): number[] => {
  const sum = [0, 0, 0];
  const n = image.width * image.height;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) sum[c]! += image.data[i * image.channels + c]!;
  return sum.map((v) => v / n);
};
const f = (v: number, d = 4): string => v.toFixed(d);
const pct = (v: number): string => `${(v * 100).toFixed(2)}%`;
const sci = (v: number): string => v.toExponential(2);

function reference(scene: string, prefix: string): { image: Image; spp: number } | null {
  const dir = join(REPO, 'bench', 'references', scene);
  if (!existsSync(dir)) return null;
  const files = readdirSync(dir).filter((n) => n.startsWith(`${prefix}-`) && n.endsWith('spp.pfm'));
  const best = files.map((n) => ({ n, spp: Number(/-(\d+)spp\.pfm$/.exec(n)?.[1]) })).sort((a, b) => b.spp - a.spp)[0];
  return best ? { image: load(join(dir, best.n)), spp: best.spp } : null;
}

const { values } = parseArgs({ options: { report: { type: 'string' }, scene: { type: 'string' }, out: { type: 'string' } } });
if (!values.report) throw new Error('--report <name> is required');
const runs = JSON.parse(readFileSync(join(REPO, 'bench', 'results', 'reports', values.report, 'runs.json'), 'utf8')) as RunLogEntry[];
const lines: string[] = [];
const row = (...cells: string[]): void => void lines.push(`| ${cells.join(' | ')} |`);
row('scene', 'ours spp', 'ours mean RGB', 'pbrt mean RGB (spp)', 'Mitsuba mean RGB (spp)', 'relBias vs pbrt', 'relBias vs Mitsuba', 'relMSE vs pbrt', 'relMSE vs Mitsuba', 'pbrt vs Mitsuba', 'block mean rms vs pbrt / Mitsuba', 'expected');
row(...Array(12).fill('---'));
for (const entry of runs) {
  if (values.scene && entry.scene !== values.scene) continue;
  const ours = load(join(REPO, entry.runDir, '1-run', `${entry.renderer}__${entry.scene}.pfm`));
  const scene = getScene(entry.scene);
  const pbrt = reference(entry.scene, 'pbrt-v4-cpu');
  const mit = reference(entry.scene, 'mitsuba3-llvm_ad_rgb');
  const m = mean(ours);
  const bias = (r: { image: Image } | null): string => (r ? pct(mean(ours).reduce((a, b) => a + b, 0) / mean(r.image).reduce((a, b) => a + b, 0) - 1) : '-');
  const rel = (r: { image: Image } | null): string => (r ? sci(compareImages(ours, r.image).relMse) : '-');
  const blocks = (r: { image: Image } | null): string => (r ? pct(blockMeanDiff(ours, r.image, 32).rmsRel) : '-');
  const expected = scene.expected ? `${scene.expected.value.map((v) => f(v)).join(' ')} (ours ${m.map((v) => f(v)).join(' ')}; dev ${m.map((v, i) => pct(v / (scene.expected!.value[i] as number) - 1)).join(' ')})` : '-';
  row(entry.scene, String(entry.spp), m.map((v) => f(v)).join(' '), pbrt ? `${mean(pbrt.image).map((v) => f(v)).join(' ')} (${pbrt.spp})` : '-', mit ? `${mean(mit.image).map((v) => f(v)).join(' ')} (${mit.spp})` : '-', bias(pbrt), bias(mit), rel(pbrt), rel(mit), pbrt && mit ? sci(compareImages(pbrt.image, mit.image).relMse) : '-', `${blocks(pbrt)} / ${blocks(mit)}`, expected);
}
const text = `${lines.join('\n')}\n`;
if (values.out) writeFileSync(values.out, text);
console.log(text);
