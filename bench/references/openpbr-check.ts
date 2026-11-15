/**
 * P2.20: render the synthetic `openpbr` scenes (openpbr-check-scenes.ts) in pbrt-v4 and Mitsuba 3 and compare them (and a closed form where
 * there is one).
 *
 *   node bench/references/openpbr-check.ts [--scenes a,b] [--spp 256] [--coverage] [--furnace] [--layered-depths 10,32,64,128]
 *
 * Writes bench/references/openpbr-check/<scene>/<renderer>-<spp>spp.pfm (git-ignored) and results.json (committed). Noise: pixel-wise
 * differences are independent, so mean(a - b) has a standard error std(a - b) / sqrt(N); z = mean / se (edges inflate std, so |z| < 3 is
 * conservative evidence of agreement).
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toMitsuba, toPbrt, writeExport } from '../exporters/index.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { Image } from '../src/adapter.ts';
import type { SceneDesc } from '../scenes/schema.ts';
import { coatFurnaceScene, openPbrCheckScenes } from './openpbr-check-scenes.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, '..', '..');
const NATIVE = join(REPO, '.cache', 'native');
const PBRT = join(NATIVE, 'pbrt-v4', 'build', 'pbrt');
const PYTHON = join(NATIVE, 'mitsuba-venv', 'bin', 'python');
const MITSUBA_SCRIPT = join(HERE, 'mitsuba_render.py');
const OUT = join(HERE, 'openpbr-check');

interface Stats { mean: [number, number, number]; center: [number, number, number] }
const stats = (img: Image): Stats => {
  const sum = [0, 0, 0], csum = [0, 0, 0];
  let cn = 0;
  const w = img.width, h = img.height, r = Math.floor(Math.min(w, h) / 8);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const inC = Math.abs(x - w / 2 + 0.5) < r && Math.abs(y - h / 2 + 0.5) < r;
      if (inC) cn++;
      for (let c = 0; c < 3; c++) {
        const v = img.data[(y * w + x) * 3 + c]!;
        sum[c]! += v;
        if (inC) csum[c]! += v;
      }
    }
  }
  return { mean: sum.map((v) => v / (w * h)) as Stats['mean'], center: csum.map((v) => v / cn) as Stats['center'] };
};

function diffStats(a: Image, b: Image): { relMean: number[]; z: number[] } {
  const n = a.width * a.height;
  const relMean: number[] = [], z: number[] = [];
  for (let c = 0; c < 3; c++) {
    let m = 0, ma = 0;
    for (let i = 0; i < n; i++) { m += a.data[i * 3 + c]! - b.data[i * 3 + c]!; ma += a.data[i * 3 + c]!; }
    m /= n; ma /= n;
    let v = 0;
    for (let i = 0; i < n; i++) v += (a.data[i * 3 + c]! - b.data[i * 3 + c]! - m) ** 2;
    const se = Math.sqrt(v / (n - 1) / n);
    relMean.push(ma > 0 ? m / ma : 0);
    z.push(se > 0 ? m / se : 0);
  }
  return { relMean, z };
}

export interface RenderOpts { spp: number; coverage?: boolean; layered?: { maxDepth?: number; nSamples?: number }; tag?: string }

export function renderNative(scene: SceneDesc, renderer: 'pbrt' | 'mitsuba', o: RenderOpts): { image: Image; seconds: number; approximations: string[]; file: string } {
  const work = mkdtempSync(join(NATIVE, 'work-'));
  try {
    const exported = (renderer === 'pbrt' ? toPbrt : toMitsuba)(scene, { assetsDir: join(REPO, 'bench', 'assets'), spp: o.spp, coverage: o.coverage, layered: o.layered });
    const sceneFile = writeExport(exported, work);
    const dir = join(OUT, scene.name);
    mkdirSync(dir, { recursive: true });
    const file = join(dir, `${renderer}${o.tag ?? ''}-${o.spp}spp.pfm`);
    const t0 = Date.now();
    const r = renderer === 'pbrt'
      ? spawnSync(PBRT, ['--quiet', '--outfile', file, sceneFile], { cwd: work, encoding: 'utf8', maxBuffer: 1 << 28 })
      : spawnSync(PYTHON, [MITSUBA_SCRIPT, sceneFile, file, 'llvm_ad_rgb', String(o.spp)], { cwd: work, encoding: 'utf8', maxBuffer: 1 << 28 });
    if (r.status !== 0) throw new Error(`${renderer} failed on ${scene.name}:\n${r.stderr}\n${r.stdout}`);
    return { image: readPFM(readFileSync(file)), seconds: (Date.now() - t0) / 1000, approximations: exported.approximations, file };
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

const arg = (name: string): string | undefined => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : undefined;
};

function main(): void {
  for (const [what, p] of [['pbrt', PBRT], ['mitsuba python', PYTHON]] as const) if (!existsSync(p)) throw new Error(`${what} not found at ${p}`);
  const spp = Number(arg('spp') ?? 256);
  const only = arg('scenes')?.split(',');
  const results: Record<string, unknown> = {};
  const scenes = openPbrCheckScenes.filter((s) => !only || only.includes(s.name));
  for (const scene of scenes) {
    const row: Record<string, unknown> = { purpose: scene.purpose };
    const imgs: Partial<Record<'pbrt' | 'mitsuba', Image>> = {};
    for (const renderer of ['pbrt', 'mitsuba'] as const) {
      try {
        const out = renderNative(scene, renderer, { spp, coverage: process.argv.includes('--coverage') });
        imgs[renderer] = out.image;
        row[renderer] = { seconds: out.seconds, ...stats(out.image), approximations: out.approximations };
      } catch (e) {
        row[renderer] = { error: String((e as Error).message).split('\n')[0] };
      }
    }
    if (imgs.pbrt && imgs.mitsuba) row['pbrt_vs_mitsuba'] = diffStats(imgs.pbrt, imgs.mitsuba);
    if (scene.expected) row['expected'] = scene.expected.value;
    results[scene.name] = row;
    console.log(scene.name, JSON.stringify({ expected: scene.expected?.value, pbrt: (row['pbrt'] as Stats | undefined)?.center, mitsuba: (row['mitsuba'] as Stats | undefined)?.center, diff: row['pbrt_vs_mitsuba'] }));
  }
  if (process.argv.includes('--furnace')) {
    const depths = (arg('layered-depths') ?? '10,32,64,128').split(',').map(Number);
    const furnace: Record<string, unknown> = {};
    for (const rough of [0.3, 0]) {
      const scene = coatFurnaceScene(rough);
      const rows: Record<string, unknown> = {};
      for (const d of depths) {
        const o = renderNative(scene, 'pbrt', { spp, layered: { maxDepth: d }, tag: `-maxdepth${d}` });
        rows[`pbrt maxdepth ${d}`] = { mean: stats(o.image).mean, seconds: o.seconds };
        console.log(scene.name, `pbrt maxdepth ${d}`, stats(o.image).mean.map((v) => v.toFixed(4)).join(' '));
      }
      const mi = renderNative(scene, 'mitsuba', { spp });
      rows['mitsuba roughplastic nonlinear'] = { mean: stats(mi.image).mean, seconds: mi.seconds };
      console.log(scene.name, 'mitsuba', stats(mi.image).mean.map((v) => v.toFixed(4)).join(' '));
      furnace[scene.name] = rows;
    }
    results['furnace'] = furnace;
  }
  mkdirSync(OUT, { recursive: true });
  const prev = existsSync(join(OUT, 'results.json')) ? (JSON.parse(readFileSync(join(OUT, 'results.json'), 'utf8')) as Record<string, unknown>) : {};
  writeFileSync(join(OUT, 'results.json'), JSON.stringify({ ...prev, spp, ...results }, null, 2) + '\n');
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) main();
