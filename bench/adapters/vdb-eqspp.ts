/**
 * V3.4b equal-spp noise table for the real OpenVDB scenes: relMSE / relBias / firefly % of ours, pbrt and Mitsuba at 16, 64, 256 spp against the MEAN of the pbrt and Mitsuba
 * 1024 spp references (repo metric `compareImages`; non-finite pixels excluded and counted). Ours come from `matrix.ts --name <n>` snapshots (`pt-megakernel__<scene>__spp<N>.pfm`).
 *
 *   node bench/adapters/vdb-eqspp.ts [--scenes a,b] [--ours <incoming run dir glob root>]   (default: newest `*-competitor/1-run` holding each scene)
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { compareImages, readPFM } from '../metrics/index.ts';
import type { Image } from '../metrics/index.ts';

const ROOT = resolve(import.meta.dirname, '..', '..');
const SPPS = [16, 64, 256, 1024];
const args = process.argv.slice(2);
const flag = (n: string): string | undefined => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : undefined; };
const scenes = (flag('--scenes') ?? 'vol-vdb-wdas-cloud,vol-vdb-bunny-cloud,vol-vdb-smoke').split(',');
const load = (p: string): Image => { const b = readFileSync(p); return readPFM(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

function oursPath(scene: string, spp: number): string | undefined {
  const inc = join(ROOT, 'bench', 'results', 'incoming');
  const runs = readdirSync(inc).filter((d) => d.endsWith('-competitor')).sort().reverse();
  for (const r of runs) {
    const f = join(inc, r, '1-run', `pt-megakernel__${scene}${spp === 1024 ? '' : `__spp${spp}`}.pfm`);
    if (existsSync(f)) return f;
  }
  return undefined;
}

for (const scene of scenes) {
  const dir = join(ROOT, 'bench', 'references', scene);
  const ref = (r: string, spp: number): Image | undefined => { const f = join(dir, `${r}-${spp}spp.pfm`); return existsSync(f) ? load(f) : undefined; };
  const P = 'pbrt-v4-cpu', M = 'mitsuba3-llvm_ad_rgb';
  const p1k = ref(P, 1024), m1k = ref(M, 1024);
  if (!p1k || !m1k) { console.log(`${scene}: missing 1024 spp references`); continue; }
  // reference = mean of the two; a pixel non-finite in one takes the other
  const mean: Image = { ...p1k, data: new Float32Array(p1k.data.length) };
  for (let i = 0; i < mean.data.length; i += p1k.channels) {
    const a = p1k.data.slice(i, i + 3), b = m1k.data.slice(i, i + 3);
    const aok = Number.isFinite(a[0]! + a[1]! + a[2]!), bok = Number.isFinite(b[0]! + b[1]! + b[2]!);
    for (let c = 0; c < p1k.channels; c++) mean.data[i + c] = aok && bok ? (p1k.data[i + c]! + m1k.data[i + c]!) / 2 : aok ? p1k.data[i + c]! : bok ? m1k.data[i + c]! : NaN;
  }
  const floor = compareImages(p1k, m1k);
  console.log(`\n## ${scene}  (ref mean pixel: pbrt ${meanOf(p1k).toFixed(4)}, mitsuba ${meanOf(m1k).toFixed(4)}; pbrt-vs-mitsuba 1024 relMSE ${floor.relMse.toExponential(2)}, relBias ${(100 * floor.relBias).toFixed(2)}%; non-finite pbrt ${floor.nonFiniteTest}, mitsuba ${floor.nonFiniteRef})`);
  console.log('| renderer | spp | relMSE | mean / ref | firefly % | non-finite | ours/pbrt | ours/mitsuba |');
  console.log('|---|---|---|---|---|---|---|---|');
  for (const spp of SPPS) {
    const imgs: Array<[string, Image | undefined]> = [['ours', (() => { const f = oursPath(scene, spp); return f ? load(f) : undefined; })()], ['pbrt', ref(P, spp)], ['mitsuba', ref(M, spp)]];
    const res: Record<string, ReturnType<typeof compareImages>> = {};
    for (const [n, im] of imgs) if (im) res[n] = compareImages(im, mean);
    for (const [n] of imgs) {
      const m = res[n];
      if (!m) continue;
      const rp = n === 'ours' && res.pbrt ? (m.relMse / res.pbrt.relMse).toFixed(2) : '';
      const rm = n === 'ours' && res.mitsuba ? (m.relMse / res.mitsuba.relMse).toFixed(2) : '';
      console.log(`| ${n} | ${spp} | ${m.relMse.toExponential(3)} | ${(1 + m.relBias).toFixed(4)} | ${m.fireflyPct.toFixed(3)} | ${m.nonFiniteTest} | ${rp} | ${rm} |`);
    }
  }
}
function meanOf(im: Image): number { let s = 0, n = 0; for (let i = 0; i < im.data.length; i += im.channels) { const v = im.data[i]! + im.data[i + 1]! + im.data[i + 2]!; if (Number.isFinite(v)) { s += v / 3; n++; } } return s / n; }
void statSync;
