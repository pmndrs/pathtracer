/**
 * P5 A/B table: our RGB render against our spectral render (`pt_spectral=1`), and both against the pbrt-v4 (spectral) and Mitsuba 3 (RGB) references.
 *
 *   node bench/adapters/spectral-agreement.ts --rgb <report> --spec <report> [--rgb2 <report> --spec2 <report>] [--region scene=x0,y0,x1,y1 ...] [--out file.md]
 *
 * Reports come from `matrix.ts --renderers pt-megakernel [--extra pt_spectral=1]` (runs.json lists the final PFMs). Per scene (whole image, or a `--region` pixel
 * rectangle [x0, x1) x [y0, y1)): the mean radiance per channel of each render and the per-channel relBias (mean ratio - 1) against pbrt and Mitsuba: a saturated colour that
 * pbrt's spectral uplift moves shows up as a per-channel gap between "rgb" and "pbrt", which "spectral" should close; then relMSE of each against pbrt. With a second pair of
 * reports (other seed, same spp) the noise: relMSE(a, a2) of the RGB pair and of the spectral pair (independent renders: 2x the variance) and the chroma noise (variance of
 * r / (r + g + b) per pixel, in 1e-4), the price of hero-wavelength sampling at equal spp.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { Image } from '../src/adapter.ts';
import { compareImages } from '../metrics/metrics.ts';
import { readPFM } from '../metrics/pfm.ts';
import type { RunLogEntry } from './matrix.ts';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const sci = (v: number): string => (Number.isFinite(v) ? v.toExponential(2) : '-');
const pct = (v: number): string => (Number.isFinite(v) ? `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%` : '-');
const loadEntry = (e: RunLogEntry): Image => readPFM(readFileSync(join(REPO, e.runDir, '1-run', `${e.renderer}__${e.scene}.pfm`)));
export const runs = (report: string): RunLogEntry[] => JSON.parse(readFileSync(join(REPO, 'bench/results/reports', report, 'runs.json'), 'utf8')) as RunLogEntry[];
const refPath = (scene: string, which: string): string => join(REPO, 'bench', 'references', scene, `${which}-4096spp.pfm`);
const tryRef = (scene: string, which: string): Image | null => {
  for (const spp of [4096, 1024]) {
    try {
      return readPFM(readFileSync(refPath(scene, which).replace('4096spp', `${spp}spp`)));
    } catch {
      /* next */
    }
  }
  return null;
};

export type Region = readonly [number, number, number, number];

export function crop(img: Image, r: Region | undefined): Image {
  if (!r) return img;
  const [x0, y0, x1, y1] = r;
  const w = x1 - x0;
  const h = y1 - y0;
  const data = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let c = 0; c < 4; c++) data[(y * w + x) * 4 + c] = img.data[((y0 + y) * img.width + x0 + x) * img.channels + (c < img.channels ? c : 0)] as number;
  return { width: w, height: h, channels: 4, data };
}

/** Per-channel mean over finite pixels. */
export function channelMeans(img: Image): [number, number, number] {
  const s = [0, 0, 0];
  let n = 0;
  for (let i = 0; i < img.width * img.height; i++) {
    const p = [0, 1, 2].map((c) => img.data[i * img.channels + c] as number);
    if (!p.every(Number.isFinite)) continue;
    for (let c = 0; c < 3; c++) s[c]! += p[c] as number;
    n++;
  }
  return s.map((v) => v / Math.max(1, n)) as [number, number, number];
}

/** Variance (x 1e4) of the red chromaticity r / (r + g + b) of the difference of two independent renders, / 2: the chroma noise of one render. */
export function chromaNoise(a: Image, b: Image): number {
  let s = 0;
  let n = 0;
  for (let i = 0; i < a.width * a.height; i++) {
    const pa = [0, 1, 2].map((c) => a.data[i * a.channels + c] as number);
    const pb = [0, 1, 2].map((c) => b.data[i * b.channels + c] as number);
    const sa = pa[0]! + pa[1]! + pa[2]!;
    const sb = pb[0]! + pb[1]! + pb[2]!;
    if (!(sa > 0.01) || !(sb > 0.01)) continue;
    const d = pa[0]! / sa - pb[0]! / sb;
    s += d * d;
    n++;
  }
  return (s / Math.max(1, n) / 2) * 1e4;
}

const rel = (m: readonly number[], ref: readonly number[]): string => m.map((v, c) => pct(v / (ref[c] as number) - 1)).join(' / ');

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { rgb: { type: 'string' }, spec: { type: 'string' }, rgb2: { type: 'string' }, spec2: { type: 'string' }, region: { type: 'string', multiple: true }, out: { type: 'string' } } });
  if (!values.rgb || !values.spec) throw new Error('usage: spectral-agreement.ts --rgb <report> --spec <report> [--rgb2 <report> --spec2 <report>] [--region scene=x0,y0,x1,y1] [--out file.md]');
  const regions = new Map<string, Region>();
  for (const r of values.region ?? []) {
    const [scene, box] = r.split('=');
    regions.set(scene as string, (box as string).split(',').map(Number) as unknown as Region);
  }
  const rSpec = runs(values.spec);
  const rRgb2 = values.rgb2 ? runs(values.rgb2) : [];
  const rSpec2 = values.spec2 ? runs(values.spec2) : [];
  const lines = ['| scene | what | R / G / B mean (rgb) | R / G / B mean (spectral) | spectral vs rgb: relBias R / G / B |', '|---|---|---|---|---|'];
  const refs = ['| scene | ref | floor (other vs it) | rgb relBias R / G / B | spectral relBias R / G / B | rgb relMSE | spectral relMSE | non-finite (rgb, spec) |', '|---|---|---|---|---|---|---|---|'];
  const noise = ['| scene | rgb relMSE(a, a2) | spectral relMSE(a, a2) | ratio | rgb chroma noise | spectral chroma noise | ratio |', '|---|---|---|---|---|---|---|'];
  const f3 = (v: readonly number[]): string => v.map((x) => x.toFixed(4)).join(' / ');
  for (const e of runs(values.rgb)) {
    const s = rSpec.find((x) => x.scene === e.scene);
    if (!s) continue;
    const region = regions.get(e.scene);
    const a = crop(loadEntry(e), region);
    const b = crop(loadEntry(s), region);
    const ma = channelMeans(a);
    const mb = channelMeans(b);
    lines.push(`| ${e.scene}${region ? ` [${region.join(',')}]` : ''} | ${e.spp} spp | ${f3(ma)} | ${f3(mb)} | ${rel(mb, ma)} |`);
    const pbrt = tryRef(e.scene, 'pbrt-v4-cpu');
    const mit = tryRef(e.scene, 'mitsuba3-llvm_ad_rgb');
    for (const [name, ref0, other0] of [['pbrt-v4', pbrt, mit], ['Mitsuba 3', mit, pbrt]] as const) {
      if (!ref0) continue;
      const ref = crop(ref0, region);
      const floor = other0 ? sci(compareImages(crop(other0, region), ref).relMse) : '-';
      const mr = channelMeans(ref);
      const ca = compareImages(a, ref);
      const cb = compareImages(b, ref);
      refs.push(`| ${e.scene} | ${name} | ${floor} | ${rel(ma, mr)} | ${rel(mb, mr)} | ${sci(ca.relMse)} | ${sci(cb.relMse)} | ${ca.nonFiniteTest}, ${cb.nonFiniteTest} |`);
    }
    const e2 = rRgb2.find((x) => x.scene === e.scene);
    const s2 = rSpec2.find((x) => x.scene === e.scene);
    if (e2 && s2) {
      const a2 = crop(loadEntry(e2), region);
      const b2 = crop(loadEntry(s2), region);
      const na = compareImages(a, a2).relMse;
      const nb = compareImages(b, b2).relMse;
      const ca = chromaNoise(a, a2);
      const cb = chromaNoise(b, b2);
      noise.push(`| ${e.scene} | ${sci(na)} | ${sci(nb)} | ${(nb / na).toFixed(2)}x | ${ca.toFixed(3)} | ${cb.toFixed(3)} | ${(cb / ca).toFixed(2)}x |`);
    }
  }
  const text = `${lines.join('\n')}\n\n${refs.join('\n')}\n${noise.length > 2 ? `\n${noise.join('\n')}\n` : ''}`;
  console.log(text);
  if (values.out) writeFileSync(values.out, text);
}
