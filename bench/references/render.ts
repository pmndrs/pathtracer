/**
 * Render cached ground-truth references with native CPU renderers.
 *
 *   node bench/references/render.ts [--scenes a,b] [--renderers pbrt,mitsuba,mitsuba-spectral] [--spp N]
 *                                   [--spp-spectral N] [--force] [--detach]
 *
 * For each scene and renderer: export the scene (bench/exporters), run the tool, and write
 *   bench/references/<scene>/<renderer-id>-<spp>spp.pfm   (linear float32 RGB; git-ignored)
 *   bench/references/<scene>/<renderer-id>-<spp>spp.json  (sidecar: versions, command, spp, seconds, settings)
 * Existing outputs are kept unless --force. `--detach` re-launches this script in the background (log in
 * .cache/native/logs/) so long 4096-spp jobs survive the calling shell. After rendering, run
 * `node bench/references/compare.ts` for furnace checks and the pbrt-vs-Mitsuba agreement table.
 *
 * Tools live in .cache/native (git-ignored): pbrt-v4/build/pbrt (patched, see pbrt-no-rr.patch) and
 * mitsuba-venv (pip mitsuba). Nothing under research/sources/ is ever built or run.
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { cpus, hostname, platform, release } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toMitsuba, toPbrt, writeExport } from '../exporters/index.ts';
import type { ExportResult } from '../exporters/index.ts';
import { tier0Scenes } from '../scenes/tier0/index.ts';
import { generateAssets } from '../scenes/tier0/generate-assets.ts';
import type { SceneDesc } from '../scenes/schema.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO = resolve(HERE, '..', '..');
export const REFERENCES_DIR = HERE;
const NATIVE = join(REPO, '.cache', 'native');
const PBRT = join(NATIVE, 'pbrt-v4', 'build', 'pbrt');
const PYTHON = join(NATIVE, 'mitsuba-venv', 'bin', 'python');
const MITSUBA_SCRIPT = join(HERE, 'mitsuba_render.py');
const ASSETS = join(REPO, 'bench', 'assets');

export interface RendererSpec {
  /** Id used in file names and result rows. */
  id: string;
  kind: 'pbrt' | 'mitsuba';
  variant?: string;
  defaultSpp: number;
}

export const RENDERERS: Record<string, RendererSpec> = {
  pbrt: { id: 'pbrt-v4-cpu', kind: 'pbrt', defaultSpp: 4096 },
  mitsuba: { id: 'mitsuba3-llvm_ad_rgb', kind: 'mitsuba', variant: 'llvm_ad_rgb', defaultSpp: 4096 },
  'mitsuba-scalar': { id: 'mitsuba3-scalar_rgb', kind: 'mitsuba', variant: 'scalar_rgb', defaultSpp: 1024 },
  'mitsuba-spectral': { id: 'mitsuba3-scalar_spectral', kind: 'mitsuba', variant: 'scalar_spectral', defaultSpp: 1024 },
};

export function referencePath(scene: string, rendererId: string, spp: number, ext: 'pfm' | 'json'): string {
  return join(REFERENCES_DIR, scene, `${rendererId}-${spp}spp.${ext}`);
}

function git(args: string[], cwd: string): string {
  const r = spawnSync('git', args, { cwd, encoding: 'utf8' });
  return r.status === 0 ? r.stdout.trim() : 'unknown';
}

function run(cmd: string, args: string[], cwd: string): { seconds: number; stdout: string; stderr: string; status: number } {
  const t0 = process.hrtime.bigint();
  const r = spawnSync(cmd, args, { cwd, encoding: 'utf8', maxBuffer: 1 << 28 });
  const seconds = Number(process.hrtime.bigint() - t0) / 1e9;
  return { seconds, stdout: r.stdout ?? '', stderr: r.stderr ?? '', status: r.status ?? -1 };
}

function renderOne(scene: SceneDesc, spec: RendererSpec, spp: number, force: boolean): void {
  const pfm = referencePath(scene.name, spec.id, spp, 'pfm');
  const sidecar = referencePath(scene.name, spec.id, spp, 'json');
  if (!force && existsSync(pfm) && existsSync(sidecar)) {
    console.log(`skip   ${scene.name} ${spec.id} ${spp}spp (cached)`);
    return;
  }
  mkdirSync(dirname(pfm), { recursive: true });
  const work = mkdtempSync(join(NATIVE, 'work-'));
  try {
    const opts = { assetsDir: ASSETS, spp };
    const exported: ExportResult = spec.kind === 'pbrt' ? toPbrt(scene, opts) : toMitsuba(scene, opts);
    const sceneFile = writeExport(exported, work);
    let command: string[];
    let seconds: number;
    let version: string;
    const settings: Record<string, unknown> = {
      maxDepth: scene.integrator.maxDepth,
      rrStartDepth: scene.integrator.rrStartDepth ?? null,
      film: scene.film,
    };
    console.log(`render ${scene.name} ${spec.id} ${spp}spp ...`);
    if (spec.kind === 'pbrt') {
      command = [PBRT, '--quiet', '--outfile', pfm, sceneFile];
      const r = run(PBRT, ['--quiet', '--outfile', pfm, sceneFile], work);
      if (r.status !== 0) throw new Error(`pbrt failed (${r.status}):\n${r.stderr}\n${r.stdout}`);
      seconds = r.seconds;
      version = `pbrt-v4 ${git(['rev-parse', 'HEAD'], join(NATIVE, 'pbrt-v4'))} (+bench no-RR patch)`;
      Object.assign(settings, { integrator: 'path', rr: scene.integrator.rrStartDepth !== undefined, sampler: 'zsobol', filter: 'box 0.5', sensor: 'cie1931 iso100 (passthrough)', spectral: true });
    } else {
      command = [PYTHON, MITSUBA_SCRIPT, sceneFile, pfm, spec.variant!, String(spp)];
      const r = run(PYTHON, command.slice(1), work);
      if (r.status !== 0) throw new Error(`mitsuba failed (${r.status}):\n${r.stderr}\n${r.stdout}`);
      const info = JSON.parse(r.stdout.split('\n').filter((l) => l.startsWith('{')).pop()!) as { mitsuba: string; seconds: number };
      seconds = r.seconds;
      version = `mitsuba ${info.mitsuba}`;
      Object.assign(settings, {
        integrator: 'path', variant: spec.variant, rr: scene.integrator.rrStartDepth !== undefined,
        max_depth: scene.integrator.maxDepth + 1, sampler: 'multijitter', filter: 'box 0.5',
        renderSeconds: info.seconds, spectral: spec.variant!.includes('spectral'),
      });
    }
    writeFileSync(sidecar, JSON.stringify({
      scene: scene.name, renderer: spec.id, version, spp, seconds: Number(seconds.toFixed(2)),
      command: command.map((c) => c.replace(REPO + '/', '')).join(' '),
      settings, approximations: exported.approximations,
      host: { hostname: hostname(), platform: `${platform()} ${release()}`, cpus: cpus().length, cpuModel: cpus()[0]?.model },
      date: new Date().toISOString(),
    }, null, 2) + '\n');
    console.log(`done   ${scene.name} ${spec.id} ${spp}spp in ${seconds.toFixed(1)}s`);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

function parseArgs(argv: string[]) {
  const o: Record<string, string | true> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a.startsWith('--')) {
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) { o[a.slice(2)] = next; i++; } else o[a.slice(2)] = true;
    }
  }
  return o;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  if (args['detach']) {
    mkdirSync(join(NATIVE, 'logs'), { recursive: true });
    const log = join(NATIVE, 'logs', `render-${new Date().toISOString().replace(/[:.]/g, '-')}.log`);
    const fd = openSync(log, 'a');
    const rest = process.argv.slice(2).filter((a) => a !== '--detach');
    const child = spawn(process.execPath, [fileURLToPath(import.meta.url), ...rest], { detached: true, stdio: ['ignore', fd, fd] });
    child.unref();
    console.log(`detached pid ${child.pid}; log: ${log}`);
    return;
  }
  for (const [what, p] of [['pbrt', PBRT], ['mitsuba python', PYTHON]] as const) {
    if (!existsSync(p)) throw new Error(`${what} not found at ${p}; see bench/references/README.md for the native setup`);
  }
  generateAssets();
  const sceneNames = typeof args['scenes'] === 'string' ? args['scenes'].split(',') : tier0Scenes.map((s) => s.name);
  const rendererKeys = typeof args['renderers'] === 'string' ? args['renderers'].split(',') : ['pbrt', 'mitsuba'];
  const sppOverride = typeof args['spp'] === 'string' ? Number(args['spp']) : undefined;
  const sppSpectral = typeof args['spp-spectral'] === 'string' ? Number(args['spp-spectral']) : undefined;
  for (const name of sceneNames) {
    const scene = tier0Scenes.find((s) => s.name === name);
    if (!scene) throw new Error(`unknown scene '${name}'`);
    for (const key of rendererKeys) {
      const spec = RENDERERS[key];
      if (!spec) throw new Error(`unknown renderer '${key}' (have ${Object.keys(RENDERERS).join(', ')})`);
      const spp = (key === 'mitsuba-spectral' ? sppSpectral : undefined) ?? sppOverride ?? spec.defaultSpp;
      renderOne(scene, spec, spp, args['force'] === true);
    }
  }
  void readFileSync;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) main();
