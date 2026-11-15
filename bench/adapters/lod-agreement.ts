/**
 * R1 A/B table: our own renders with the full material (`pt_lod=full`) against the reduced secondary material (`pt_lod=secondary`), and both against the
 * pbrt-v4 / Mitsuba 3 references where the scene has them.
 *
 *   node bench/adapters/lod-agreement.ts --full <report, high spp> --lod <report, same spp, same seed> [--fullN <report, N spp, other seed> --lodN <report, N spp, same other seed>] [--out file.md]
 *
 * Reports come from `matrix.ts --renderers pt-megakernel --extra pt_lod=...` (runs.json lists the final PFMs). Per scene:
 *  - relBias of lod vs full (mean ratio - 1, all channels) and the 32 px block-mean max / rms relative difference (structure of the bias; noise averages out);
 *  - relMSE of lod(N spp) vs full(high spp) next to relMSE of full(N spp) vs full(high spp): the excess is the reduced material's bias plus its extra variance
 *    (the lobe pick), the noise of the high spp image is in both;
 *  - against the references: relMSE and relBias of full and of lod, and the pbrt-vs-Mitsuba floor;
 *  - non-finite pixels.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import { blockMeanDiff } from './baseline.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const pct = (v: number): string => (Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%` : '-');
const loadEntry = (e: RunLogEntry): Image => readPFM(readFileSync(join(REPO, e.runDir, '1-run', `${e.renderer}__${e.scene}.pfm`)));
const runs = (report: string): RunLogEntry[] => JSON.parse(readFileSync(join(REPO, 'bench/results/reports', report, 'runs.json'), 'utf8')) as RunLogEntry[];
const refPath = (scene: string, which: 'pbrt-v4-cpu' | 'mitsuba3-llvm_ad_rgb'): string => join(REPO, 'bench', 'references', scene, `${which}-4096spp.pfm`);
const tryRef = (scene: string, which: 'pbrt-v4-cpu' | 'mitsuba3-llvm_ad_rgb'): Image | null => {
  try {
    return readPFM(readFileSync(refPath(scene, which)));
  } catch {
    return null;
  }
};

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { full: { type: 'string' }, lod: { type: 'string' }, fullN: { type: 'string' }, lodN: { type: 'string' }, out: { type: 'string' } } });
  if (!values.full || !values.lod) throw new Error('usage: lod-agreement.ts --full <report> --lod <report> [--fullN <report> --lodN <report>] [--out file.md]');
  const rLod = runs(values.lod);
  const rFullN = values.fullN ? runs(values.fullN) : [];
  const rLodN = values.lodN ? runs(values.lodN) : [];
  const head = ['| scene | spp | lod vs full: relBias | block32 max / rms | relMSE lod(N) vs full(hi) | relMSE full(N) vs full(hi) | non-finite |', '|---|---|---|---|---|---|---|'];
  const refHead = ['| scene | ref | floor (M vs P) | full relMSE | lod relMSE | full relBias | lod relBias |', '|---|---|---|---|---|---|---|'];
  const lines = [...head];
  const refLines = [...refHead];
  for (const full of runs(values.full)) {
    const lod = rLod.find((e) => e.scene === full.scene);
    if (!lod) continue;
    const a = loadEntry(full);
    const b = loadEntry(lod);
    const m = compareImages(b, a);
    const blk = blockMeanDiff(b, a, 32);
    const fn = rFullN.find((e) => e.scene === full.scene);
    const ln = rLodN.find((e) => e.scene === full.scene);
    const lodNm = ln ? compareImages(loadEntry(ln), a) : null;
    const fullNm = fn ? compareImages(loadEntry(fn), a) : null;
    lines.push(`| ${full.scene} | ${full.spp} | ${pct(m.relBias)} | ${pct(blk.maxRel)} / ${pct(blk.rmsRel)} | ${lodNm ? sci(lodNm.relMse) : '-'} | ${fullNm ? sci(fullNm.relMse) : '-'} | ${m.nonFiniteTest + m.nonFiniteRef} |`);
    const pbrt = tryRef(full.scene, 'pbrt-v4-cpu');
    const mit = tryRef(full.scene, 'mitsuba3-llvm_ad_rgb');
    for (const [name, ref, other] of [['pbrt-v4', pbrt, mit], ['Mitsuba 3', mit, pbrt]] as const) {
      if (!ref) continue;
      const floor = other ? sci(compareImages(other, ref).relMse) : '-';
      const mf = compareImages(a, ref);
      const ml = compareImages(b, ref);
      refLines.push(`| ${full.scene} | ${name} | ${floor} | ${sci(mf.relMse)} | ${sci(ml.relMse)} | ${pct(mf.relBias)} | ${pct(ml.relBias)} |`);
    }
  }
  const text = `${lines.join('\n')}\n\n${refLines.join('\n')}\n`;
  console.log(text);
  if (values.out) writeFileSync(values.out, text);
}
