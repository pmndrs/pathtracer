/**
 * Quality of an OIDN-denoised frame against the 4096 spp references (P2 exit item), raw next to denoised. Quality only: no timings.
 *
 *   node bench/adapters/oidn-compare.ts --raw <raw.pfm> --denoised <denoised.pfm> --scene cornell [--out file.md] [--no-hdr-flip]
 *
 * Both images go through the same metrics (bench/metrics): relMSE (eps 1e-2), relBias, firefly %, mean LDR-FLIP (ACES, 0 EV) and mean HDR-FLIP, against
 * pbrt-v4 and Mitsuba (llvm_ad_rgb) 4096 spp. Produce the two images with
 *   node bench/runner/run.ts --page competitor --args renderer=pt-megakernel scene=cornell spp=16 pt_denoise=oidn
 * (`<run dir>/1-run/pt-megakernel__cornell.pfm` and `...__denoised.pfm`). OIDN smooths noise but keeps its own bias (blur of fine detail, shifted shadows): relBias and
 * FLIP show that, relMSE alone would reward blur.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { flipLdr } from '../metrics/flip.ts';
import { flipHdrParallel } from '../metrics/flip-parallel.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const load = (path: string): Image => readPFM(readFileSync(path));

const { values } = parseArgs({ options: { raw: { type: 'string' }, denoised: { type: 'string' }, scene: { type: 'string', default: 'cornell' }, out: { type: 'string' }, 'no-hdr-flip': { type: 'boolean' } } });
if (!values.raw || !values.denoised) throw new Error('usage: oidn-compare.ts --raw <raw.pfm> --denoised <denoised.pfm> --scene <name> [--out file.md] [--no-hdr-flip]');
const refs = { pbrt: join(REPO, 'bench/references', values.scene!, 'pbrt-v4-cpu-4096spp.pfm'), mitsuba: join(REPO, 'bench/references', values.scene!, 'mitsuba3-llvm_ad_rgb-4096spp.pfm') };
const images = { raw: load(values.raw), denoised: load(values.denoised) };
const lines = ['| reference | image | relMSE | relBias | firefly % | LDR-FLIP | HDR-FLIP |', '|---|---|---|---|---|---|---|'];
for (const [refName, refPath] of Object.entries(refs)) {
  const ref = load(refPath);
  for (const [name, image] of Object.entries(images)) {
    const m = compareImages(image, ref);
    const ldr = flipLdr(image, ref).mean;
    const hdr = values['no-hdr-flip'] ? NaN : (await flipHdrParallel(image, ref)).mean;
    lines.push(`| ${refName} | ${name} | ${sci(m.relMse)} | ${(m.relBias * 100).toFixed(2)}% | ${m.fireflyPct.toFixed(3)} | ${ldr.toFixed(4)} | ${Number.isFinite(hdr) ? hdr.toFixed(4) : '-'} |`);
  }
}
console.log(lines.join('\n'));
if (values.out) writeFileSync(values.out, `${lines.join('\n')}\n`);
