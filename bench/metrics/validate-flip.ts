/**
 * P0.14: compare our FLIP against NVlabs/flip's own outputs (BSD-3-Clause; see DIVERGES V6).
 *
 *   node bench/metrics/validate-flip.ts [--dir .cache/flip-validation/work]
 *
 * Needs the files written by `.cache/flip-validation/work/prep.py` (see README of that folder): NVlabs' images/
 * reference+test as PFM (PNG as encoded [0,1] values, EXR as linear float) and the error maps their pip package
 * `flip-evaluator` produced. Prints mean FLIP and per-pixel agreement for LDR (sRGB decode on / off) and HDR.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { flipHdr, flipLdr } from './flip.ts';
import { readPFM } from './pfm.ts';
import type { Image } from '../src/adapter.ts';

export interface Agreement { mean: number; refMean: number; meanDiff: number; maxAbs: number; rmse: number; p999: number; pctOver1e3: number }

/** Compare our map against theirs (their map is a grey PFM expanded to 3 channels: take channel 0). */
export function agreement(ours: Float32Array, theirs: Image): Agreement {
  const N = ours.length;
  let sa = 0, sb = 0, mx = 0, se = 0, over = 0;
  const d = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const a = ours[i]!, b = theirs.data[3 * i]!;
    sa += a; sb += b;
    const e = Math.abs(a - b);
    d[i] = e; if (e > mx) mx = e; se += e * e; if (e > 1e-3) over++;
  }
  d.sort();
  return { mean: sa / N, refMean: sb / N, meanDiff: sa / N - sb / N, maxAbs: mx, rmse: Math.sqrt(se / N), p999: d[Math.floor(N * 0.999)]!, pctOver1e3: (100 * over) / N };
}

const fmt = (a: Agreement) =>
  `mean ${a.mean.toFixed(6)} (ref ${a.refMean.toFixed(6)}, diff ${a.meanDiff >= 0 ? '+' : ''}${a.meanDiff.toExponential(2)})  per-pixel: rmse ${a.rmse.toExponential(2)}, p99.9 ${a.p999.toExponential(2)}, max ${a.maxAbs.toExponential(2)}, >1e-3 on ${a.pctOver1e3.toFixed(3)} % of pixels`;

export function main(argv: string[]): void {
  const di = argv.indexOf('--dir');
  const dir = di >= 0 ? argv[di + 1]! : '.cache/flip-validation/work';
  const load = (n: string) => readPFM(readFileSync(join(dir, n)));
  const refP = load('reference.png.pfm'), testP = load('test.png.pfm');
  const refE = load('reference.exr.pfm'), testE = load('test.exr.pfm');
  const nvL = load('nv_ldr_map.pfm'), nvH = load('nv_hdr_map.pfm');
  console.log(`${refP.width}x${refP.height}; NVlabs LDR mean ${(nvL.data.reduce((s, v, i) => s + (i % 3 === 0 ? v : 0), 0) / (refP.width * refP.height)).toFixed(6)}`);
  const t0 = performance.now();
  const on = flipLdr(testP, refP, { tonemap: 'clamp', srgbDecode: true });
  const tLdr = performance.now() - t0;
  console.log(`LDR, sRGB decode ON  (what the reference does)  : ${fmt(agreement(on.map, nvL))}   [${(tLdr / 1000).toFixed(1)} s]`);
  const off = flipLdr(testP, refP, { tonemap: 'clamp', srgbDecode: false });
  console.log(`LDR, sRGB decode OFF (our default input path)    : ${fmt(agreement(off.map, nvL))}`);
  const t1 = performance.now();
  const h = flipHdr(testE, refE, { tonemap: 'aces' });
  const tHdr = performance.now() - t1;
  console.log(`HDR (aces, ${h.count} exposures ${h.start.toFixed(4)}..${h.stop.toFixed(4)} stops)        : ${fmt(agreement(h.map, nvH))}   [${(tHdr / 1000).toFixed(1)} s]`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main(process.argv.slice(2));
