/**
 * V7.3 split denoising A/B (quality only; no timings). Media renders carry `volumes.outputs` (orders 0-1 plane); the orders 2+ plane (`radiance - order01`) is the noisy one,
 * so it is filtered on its own and the orders 0-1 are added back (they are nearly noise-free at 4+ spp: 0-4% of the error). Arms per cell:
 *   raw            the beauty
 *   oidn_all       OIDN (colour only, HDR) on the beauty                          [page split-denoise]
 *   split_oidn     OIDN on max(beauty - order01, 0), + order01                    [page split-denoise]
 *   blur_all       Gaussian on the beauty, sigma swept; best + the fixed sigma 2
 *   split_gauss    Gaussian on beauty - order01, + order01, sigma swept

 *   oidn_all_g / split_oidn_g   the same two with the orders 0-1 plane (o / (1 + o)) as OIDN's albedo guide
 *   split_oidn_fade  per-pixel lerp(raw, split_oidn, w), w = clamp(se_rel / rho0, 0, 1): se_rel = standard error of the beauty luminance / (mean luminance + 0.05) from the outputs' second moment
 * Metrics against a reference: relMSE (eps 1e-2), relBias, relMSE inside the edge mask (reference pixels within EDGE_R px of a strong log-luminance gradient), mean LDR-FLIP.
 *
 *   node bench/split-denoise.ts render  --spec <spec.json> [--cells cornell,wdas] [--spps 1,4,16,64] [--seeds 1,2]
 *   node bench/split-denoise.ts oidn    --spec <spec.json>
 *   node bench/split-denoise.ts report  --spec <spec.json> [--out table.md]
 *
 * Run `render` and `oidn` on a quiet machine (they queue for the shared GPU lock); `report` is CPU only.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from './src/adapter.ts';
import { flipLdr } from './metrics/flip.ts';
import { compareImages } from './metrics/metrics.ts';
import { readPFM } from './metrics/pfm.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CELLS: Record<string, { scene: string; extra: string[]; refs: Record<string, string> }> = {
  cornell: { scene: 'vol-cornell-fog', extra: ['pt_forced_nee=1', 'pt_equiangular=1'], refs: { pbrt: 'bench/references/vol-cornell-fog/pbrt-v4-cpu-4096spp.pfm' } },
  wdas: { scene: 'vol-vdb-wdas-cloud', extra: [], refs: { pbrt: 'bench/references/vol-vdb-wdas-cloud/pbrt-v4-cpu-1024spp.pfm', mitsuba: 'bench/references/vol-vdb-wdas-cloud/mitsuba3-llvm_ad_rgb-1024spp.pfm' } },
};
const SIGMAS = [0.75, 1, 1.5, 2, 3, 4];
const RHOS = [0.05, 0.1, 0.2, 0.4, 0.8];
const EDGE_R = 3;
const EDGE_GRAD = 0.25;

interface SpecCell { key: string; cell: string; scene: string; spp: number; seed: number; prefix: string }
interface Spec { cells: SpecCell[]; oidnRun?: string; /** per cell: our own 8192 spp render (seed 99) as an extra reference */ refs?: Record<string, string> }

const { positionals, values } = parseArgs({ allowPositionals: true, options: { spec: { type: 'string' }, cells: { type: 'string', default: 'cornell,wdas' }, spps: { type: 'string', default: '1,4,16,64' }, seeds: { type: 'string', default: '1,2' }, refs: { type: 'boolean' }, out: { type: 'string' } } });
const mode = positionals[0];
const specPath = values.spec;
if (!specPath || !['render', 'oidn', 'report'].includes(mode ?? '')) throw new Error('usage: split-denoise.ts render|oidn|report --spec <spec.json>');
const spec: Spec = existsSync(specPath) ? (JSON.parse(readFileSync(specPath, 'utf8')) as Spec) : { cells: [] };
const save = (): void => writeFileSync(specPath, JSON.stringify(spec, null, 1));

const runner = (page: string, args: string[], extra: string[] = []): { out: string; dir: string | undefined } => {
  const run = spawnSync(process.execPath, [join(REPO, 'bench/runner/run.ts'), '--page', page, '--timeout', '280000', '--load-wait', '0', ...extra, '--args', ...args], { cwd: REPO, encoding: 'utf8' });
  const out = `${run.stdout}${run.stderr}`;
  console.log(out.split('\n').filter((l) => /PASS|FAIL|The page reported|Timed out|Error/.test(l)).map((l) => l.slice(0, 300)).join('\n'));
  return { out, dir: /results: (\S+)\s*$/m.exec(out)?.[1] };
};

if (mode === 'render') {
  for (const cell of values.cells!.split(',')) {
    const c = CELLS[cell];
    if (!c) throw new Error(`unknown cell ${cell}`);
    if (values.refs) {
      // our own long render, same estimator (outputs on so the kernel is the one under test), seed 99: isolates denoiser bias + noise from the renderer's own offset to pbrt / Mitsuba
      const { dir } = runner('competitor', ['renderer=pt-megakernel', `scene=${c.scene}`, 'spp=8192', 'pt_volume_outputs=1', 'pt_seed=99', ...c.extra], ['--suite', 'competitor', '--no-csv']);
      if (!dir) throw new Error(`no run dir for ${cell} reference`);
      spec.refs = { ...spec.refs, [cell]: `${dir.replace(`${REPO}/`, '')}/1-run/pt-megakernel__${c.scene}.pfm` };
      save();
      continue;
    }
    for (const spp of values.spps!.split(',').map(Number)) {
      for (const seed of values.seeds!.split(',').map(Number)) {
        const key = `${cell}_spp${spp}_s${seed}`;
        const { dir } = runner('competitor', ['renderer=pt-megakernel', `scene=${c.scene}`, `spp=${spp}`, 'pt_volume_outputs=1', `pt_seed=${seed}`, ...c.extra], ['--suite', 'competitor', '--no-csv']);
        if (!dir) throw new Error(`no run dir for ${key}`);
        const prefix = `${dir.replace(`${REPO}/`, '')}/1-run/pt-megakernel__${c.scene}`;
        spec.cells = spec.cells.filter((x) => x.key !== key);
        spec.cells.push({ key, cell, scene: c.scene, spp, seed, prefix });
        save();
      }
    }
  }
} else if (mode === 'oidn') {
  const { dir } = runner('split-denoise', [`files=${spec.cells.map((x) => `${x.key}@${x.prefix}`).join(',')}`]);
  if (!dir) throw new Error('no run dir');
  spec.oidnRun = dir.replace(`${REPO}/`, '');
  save();
}

// ---- report -------------------------------------------------------------------------------------------------------------------------------------------------------------

const load = (path: string): Image => readPFM(readFileSync(join(REPO, path)));
const lum = (d: Float32Array, p: number): number => 0.2126 * (d[p * 3] as number) + 0.7152 * (d[p * 3 + 1] as number) + 0.0722 * (d[p * 3 + 2] as number);

function gauss(img: Image, sigma: number): Image {
  const { width: W, height: H } = img;
  const r = Math.ceil(3 * sigma);
  const k = new Float32Array(2 * r + 1);
  let sum = 0;
  for (let i = -r; i <= r; i++) sum += k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < k.length; i++) k[i] = (k[i] as number) / sum;
  const pass = (src: Float32Array, horizontal: boolean): Float32Array => {
    const dst = new Float32Array(src.length);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        for (let c = 0; c < 3; c++) {
          let acc = 0;
          for (let i = -r; i <= r; i++) {
            const xx = horizontal ? Math.min(W - 1, Math.max(0, x + i)) : x;
            const yy = horizontal ? y : Math.min(H - 1, Math.max(0, y + i));
            acc += (k[i + r] as number) * (src[(yy * W + xx) * 3 + c] as number);
          }
          dst[(y * W + x) * 3 + c] = acc;
        }
      }
    }
    return dst;
  };
  const clean = new Float32Array(img.data.length);
  for (let i = 0; i < clean.length; i++) clean[i] = Number.isFinite(img.data[i] as number) ? (img.data[i] as number) : 0;
  return { width: W, height: H, channels: 3, data: pass(pass(clean, true), false) };
}

const combine = (a: Image, b: Image, f: (x: number, y: number) => number): Image => ({ ...a, channels: 3, data: a.data.map((v, i) => f(v, b.data[i] as number)) as Float32Array });

/** Reference pixels within EDGE_R px (Chebyshev) of a strong log-luminance gradient. */
function edgeMask(ref: Image): { mask: Uint8Array; fraction: number } {
  const { width: W, height: H } = ref;
  const L = new Float32Array(W * H);
  for (let p = 0; p < W * H; p++) L[p] = Math.log(Math.max(lum(ref.data, p), 0) + 0.05);
  const strong = new Uint8Array(W * H);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const g = Math.hypot((L[y * W + x + 1] as number) - (L[y * W + x - 1] as number), (L[(y + 1) * W + x] as number) - (L[(y - 1) * W + x] as number)) / 2;
      if (g > EDGE_GRAD) strong[y * W + x] = 1;
    }
  }
  const mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (!strong[y * W + x]) continue;
    for (let j = -EDGE_R; j <= EDGE_R; j++) for (let i = -EDGE_R; i <= EDGE_R; i++) {
      const xx = x + i, yy = y + j;
      if (xx >= 0 && yy >= 0 && xx < W && yy < H) mask[yy * W + xx] = 1;
    }
  }
  let n = 0;
  for (const m of mask) n += m;
  return { mask, fraction: n / (W * H) };
}

function maskedRelMse(test: Image, ref: Image, mask: Uint8Array): number {
  let s = 0, n = 0;
  for (let p = 0; p < mask.length; p++) {
    if (!mask[p]) continue;
    let ok = true, e = 0;
    for (let c = 0; c < 3; c++) {
      const t = test.data[p * 3 + c] as number, r = ref.data[p * 3 + c] as number;
      if (!Number.isFinite(t) || !Number.isFinite(r)) { ok = false; break; }
      e += ((t - r) * (t - r)) / (r * r + 1e-2);
    }
    if (ok) { s += e / 3; n++; }
  }
  return s / Math.max(n, 1);
}

interface Row { relMse: number; relBias: number; edge: number; flip: number }
function score(test: Image, ref: Image, mask: Uint8Array): Row {
  const m = compareImages(test, ref);
  return { relMse: m.relMse, relBias: m.relBias, edge: maskedRelMse(test, ref, mask), flip: flipLdr(test, ref).mean };
}

function report(): void {
  if (!spec.oidnRun) throw new Error('run `oidn` first');
  const lines: string[] = [];
  const perCell = new Map<string, Map<string, Row[]>>(); // `${cell}|${refName}|${spp}` -> arm -> rows per seed
  const info = new Map<string, string>();
  for (const sc of spec.cells) {
    const beauty = load(`${sc.prefix}.pfm`);
    const o01 = load(`${sc.prefix}__vol_order01.pfm`);
    const se = load(`${sc.prefix}__vol_stderr.pfm`);
    const all = load(`${spec.oidnRun}/1-run/${sc.key}__oidn_all.pfm`);
    const split = load(`${spec.oidnRun}/1-run/${sc.key}__split_oidn.pfm`);
    const allG = load(`${spec.oidnRun}/1-run/${sc.key}__oidn_all_g.pfm`);
    const splitG = load(`${spec.oidnRun}/1-run/${sc.key}__split_oidn_g.pfm`);
    const ms = combine(beauty, o01, (b, o) => Math.max(Number.isFinite(b) && Number.isFinite(o) ? b - o : 0, 0));
    const arms: Record<string, Image> = { raw: beauty, oidn_all: all, split_oidn: split, oidn_all_g: allG, split_oidn_g: splitG };
    const addBack = (img: Image): Image => combine(img, o01, (m, o) => Math.max(m, 0) + (Number.isFinite(o) ? o : 0));
    for (const s of SIGMAS) {
      arms[`blur_all_s${s}`] = gauss(beauty, s);
      arms[`split_gauss_s${s}`] = addBack(gauss(ms, s));
    }
    // fade by the pixel's own relative standard error (luminance of the beauty's standard error over its mean luminance)
    const n = beauty.width * beauty.height;
    for (const rho0 of RHOS) {
      const out = new Float32Array(n * 3);
      for (let p = 0; p < n; p++) {
        const rho = lum(se.data, p) / (lum(beauty.data, p) + 0.05);
        const w = sc.spp < 4 ? 1 : Math.min(1, Math.max(0, rho / rho0)); // one or two samples give no usable standard error: full filter
        for (let c = 0; c < 3; c++) out[p * 3 + c] = (1 - w) * (beauty.data[p * 3 + c] as number) + w * (split.data[p * 3 + c] as number);
      }
      arms[`split_oidn_fade${rho0}`] = { width: beauty.width, height: beauty.height, channels: 3, data: out };
    }
    let share = 0, tot = 0;
    for (let p = 0; p < n; p++) { share += lum(o01.data, p); tot += lum(beauty.data, p); }
    info.set(`${sc.cell}|${sc.spp}`, `order01 share of mean radiance ${(100 * share / tot).toFixed(0)}%`);
    for (const [refName, refPath] of Object.entries({ ...CELLS[sc.cell]!.refs, ...(spec.refs?.[sc.cell] ? { ours: spec.refs[sc.cell]! } : {}) })) {
      const ref = load(refPath);
      const { mask, fraction } = edgeMask(ref);
      info.set(`${sc.cell}|edge`, `edge mask ${(100 * fraction).toFixed(0)}% of pixels (|grad log(L+0.05)| > ${EDGE_GRAD}, dilated ${EDGE_R} px)`);
      const k = `${sc.cell}|${refName}|${sc.spp}`;
      if (!perCell.has(k)) perCell.set(k, new Map());
      for (const [arm, img] of Object.entries(arms)) {
        const m = perCell.get(k)!;
        if (!m.has(arm)) m.set(arm, []);
        m.get(arm)!.push(score(img, ref, mask));
      }
    }
  }
  const mean = (rows: Row[], f: (r: Row) => number): number => rows.reduce((s, r) => s + f(r), 0) / rows.length;
  const sci = (v: number): string => v.toExponential(2);
  lines.push('| scene | ref | spp | arm | relMSE | relBias | edge relMSE | LDR-FLIP |', '|---|---|---|---|---|---|---|---|');
  for (const [k, arms] of [...perCell].sort()) {
    const [cell, refName, spp] = k.split('|') as [string, string, string];
    const pick = (prefix: string): string => {
      let best = '', bestV = Infinity;
      for (const [a, rows] of arms) if (a.startsWith(prefix) && !a.includes('fade')) { const v = mean(rows, (r) => r.relMse); if (v < bestV) { bestV = v; best = a; } }
      return best;
    };
    const bestFade = ((): string => { let best = '', bv = Infinity; for (const [a, rows] of arms) if (a.includes('fade')) { const v = mean(rows, (r) => r.relMse); if (v < bv) { bv = v; best = a; } } return best; })();
    const shown = ['raw', 'oidn_all', 'split_oidn', 'oidn_all_g', 'split_oidn_g', pick('blur_all'), 'blur_all_s2', pick('split_gauss'), 'split_gauss_s2', bestFade];
    for (const arm of [...new Set(shown)]) {
      const rows = arms.get(arm)!;
      lines.push(`| ${cell} | ${refName} | ${spp} | ${arm} | ${sci(mean(rows, (r) => r.relMse))} | ${(100 * mean(rows, (r) => r.relBias)).toFixed(2)}% | ${sci(mean(rows, (r) => r.edge))} | ${mean(rows, (r) => r.flip).toFixed(4)} |`);
    }
  }
  // fade rule table: relMSE of every rho0 per scene / spp against pbrt
  lines.push('', 'Fade sweep (relMSE vs our 8192 spp render; raw and split_oidn for reference; seeds averaged):', '', `| scene | spp | raw | split_oidn | ${RHOS.map((r) => `rho0 ${r}`).join(' | ')} |`, `|---|---|---|---|${RHOS.map(() => '---').join('|')}|`);
  for (const [k, arms] of [...perCell].sort()) {
    const [cell, refName, spp] = k.split('|') as [string, string, string];
    if (refName !== 'ours') continue;
    const v = (a: string): string => sci(mean(arms.get(a)!, (r) => r.relMse));
    lines.push(`| ${cell} | ${spp} | ${v('raw')} | ${v('split_oidn')} | ${RHOS.map((r) => v(`split_oidn_fade${r}`)).join(' | ')} |`);
  }
  lines.push('', ...[...info].sort().map(([k, v]) => `- ${k}: ${v}`));
  console.log(lines.join('\n'));
  if (values.out) writeFileSync(values.out, `${lines.join('\n')}\n`);
}

if (mode === 'report') report();
