/**
 * Quality numbers for the blue-noise assets (`pnpm bluenoise:report`; run `pnpm bluenoise:generate` first).
 * Prints markdown tables (pasted into REPORT.md) and writes linear PFM images for the bench viewer into
 * `.cache/bluenoise-ab/` (git-ignored): the reference, and per sampler the 1 spp / 4 spp estimates and the
 * 64-frame running mean of the soft-shadow integrand.
 *
 * Samplers compared (all per pixel):
 *  - `white`: independent hash random numbers.
 *  - `owen-sobol`: Owen-scrambled Sobol, per-pixel scramble seed (src/sampling/owen-sobol.ts).
 *  - `vc2d-N`: classic 2D void-and-cluster texture, sample/frame k shifted by k * golden ratio (R1 / R2).
 *  - `gf-N`: jointly optimised vec2 texture (2D integrand only), same golden-ratio shift as vc2d.
 *  - `st-NxD`: spatiotemporal texture, sample/frame k reads slice k mod D; every full cycle adds a golden-ratio
 *    shift (research/notes/04-sampling-reuse.md section 5.2).
 * Each case is averaged over TRIALS random toroidal texture offsets / scramble seeds.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodePfm } from '../../../../bench/runner/pfm.ts';
import { type BlueNoiseMeta, type BlueNoiseTexture, parseBlueNoise } from '../../src/noise/bluenoise.ts';
import { hash32, owenSobol2D } from '../../src/sampling/owen-sobol.ts';
import { powerSpectrum2D, radialAverage, temporalSpectrum, variance } from './analysis.ts';
import { mulberry32 } from './rng.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = join(HERE, '..', '..', 'assets', 'bluenoise');
const PFM_DIR = join(HERE, '..', '..', '..', '..', '.cache', 'bluenoise-ab');

const IMG = 128;
const TRIALS = 16;
const PHI1 = 0.6180339887498949;
const R2: [number, number] = [0.7548776662466927, 0.5698402909980532];
const THETA = 0.4;
const BLUR_SIGMA = 2;

function loadTexture(name: string): BlueNoiseTexture {
  const manifest = JSON.parse(readFileSync(join(ASSETS, 'manifest.json'), 'utf8')) as { textures: BlueNoiseMeta[] };
  const meta = manifest.textures.find((t) => t.name === name);
  if (!meta) throw new Error(`missing ${name}: run pnpm bluenoise:generate`);
  return parseBlueNoise(readFileSync(join(ASSETS, meta.f32)), meta);
}

/** One channel of a texture as a z-major volume. */
function channelOf(tex: BlueNoiseTexture, c: number): Float32Array {
  const { width, height, depth, channels } = tex.meta;
  const n = width * height * depth;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = tex.data[i * channels + c]!;
  return out;
}

function frac(x: number): number {
  return x - Math.floor(x);
}

// ---------------------------------------------------------------- spectra tables

/** RAPS normalised so white noise reads 1, grouped into 10 bands of 0.1 Nyquist. */
function bands(raps: Float64Array): number[] {
  const nyq = raps.length;
  const out: number[] = [];
  for (let b = 0; b < 10; b++) {
    let s = 0;
    let c = 0;
    for (let i = 1; i < nyq; i++) {
      const f = i / nyq;
      if (f >= b / 10 && f < (b + 1) / 10) { s += raps[i]!; c++; }
    }
    out.push(c ? s / c : NaN);
  }
  return out;
}

function sliceRaps(vol: Float32Array, w: number, h: number, d: number): Float64Array {
  const v = variance(vol);
  let acc: Float64Array | null = null;
  for (let z = 0; z < d; z++) {
    const r = radialAverage(powerSpectrum2D(vol.subarray(z * w * h, (z + 1) * w * h), w, h), w, h);
    if (!acc) acc = new Float64Array(r.length);
    for (let i = 0; i < r.length; i++) acc[i]! += r[i]! / d / v;
  }
  return acc!;
}

function whiteVolume(w: number, h: number, d: number, seed: number): Float32Array {
  const r = mulberry32(seed);
  const out = new Float32Array(w * h * d);
  for (let i = 0; i < out.length; i++) out[i] = r();
  return out;
}

/** vc2d animated over `d` frames by the golden-ratio shift (what a renderer without an ST texture does). */
function goldenVolume(slice: Float32Array, d: number): Float32Array {
  const out = new Float32Array(slice.length * d);
  for (let z = 0; z < d; z++) for (let i = 0; i < slice.length; i++) out[z * slice.length + i] = frac(slice[i]! + z * PHI1);
  return out;
}

const fmt = (x: number, p = 2): string => (Number.isFinite(x) ? x.toFixed(p) : '-');

function spectraTables(): string {
  const lines: string[] = [];
  const rows: Array<[string, Float32Array, number, number]> = [];
  for (const s of [64, 128]) {
    rows.push([`white ${s}²`, whiteVolume(s, s, 1, 7), s, 1]);
    rows.push([`vc2d-${s}`, channelOf(loadTexture(`vc2d-scalar-${s}`), 0), s, 1]);
    const vecB = channelOf(loadTexture(`vc2d-vec2-${s}`), 1);
    rows.push([`vc2d-${s} (vec2 .y)`, vecB, s, 1]);
    const gf = loadTexture(`gf-vec2-${s}`);
    rows.push([`gf-${s} .x`, channelOf(gf, 0), s, 1]);
    rows.push([`gf-${s} .y`, channelOf(gf, 1), s, 1]);
    const d = s === 64 ? 32 : 64;
    rows.push([`st-${s}x${d} (avg of slices)`, channelOf(loadTexture(`st-scalar-${s}x${d}`), 0), s, d]);
  }
  lines.push('| texture | ' + Array.from({ length: 10 }, (_, b) => `${b / 10}-${(b + 1) / 10}`).join(' | ') + ' |');
  lines.push('|---|' + '---|'.repeat(10));
  for (const [name, vol, s, d] of rows) lines.push(`| ${name} | ${bands(sliceRaps(vol, s, s, d)).map((x) => fmt(x)).join(' | ')} |`);
  return lines.join('\n');
}

/** Per-pixel temporal power spectrum in 8 bands of 1/8 Nyquist, white = 1. */
function temporalTables(): string {
  const rows: Array<[string, Float32Array, number, number]> = [];
  for (const [s, d] of [[64, 32], [128, 64]] as const) {
    rows.push([`white ${s}²x${d}`, whiteVolume(s, s, d, 11), s, d]);
    rows.push([`vc2d-${s} + golden ratio, ${d} frames`, goldenVolume(channelOf(loadTexture(`vc2d-scalar-${s}`), 0), d), s, d]);
    rows.push([`st-${s}x${d}`, channelOf(loadTexture(`st-scalar-${s}x${d}`), 0), s, d]);
  }
  const lines = ['| texture | ' + Array.from({ length: 8 }, (_, b) => `${b / 8}-${(b + 1) / 8}`).join(' | ') + ' |', '|---|' + '---|'.repeat(8)];
  for (const [name, vol, s, d] of rows) {
    const t = temporalSpectrum(vol, s, s, d);
    const nyq = d / 2;
    const out: number[] = [];
    for (let b = 0; b < 8; b++) {
      let sum = 0;
      let c = 0;
      for (let k = 1; k <= nyq; k++) {
        const f = k / nyq;
        if (f > b / 8 && f <= (b + 1) / 8) { sum += t[k]!; c++; }
      }
      out.push(c ? sum / c : NaN);
    }
    lines.push(`| ${name} | ${out.map((x) => fmt(x)).join(' | ')} |`);
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------- rank uniformity

function histogramDeviation(vals: ArrayLike<number>, bins: number): number {
  const h = new Float64Array(bins);
  for (let i = 0; i < vals.length; i++) h[Math.min(bins - 1, Math.floor(vals[i]! * bins))]! += 1;
  const e = vals.length / bins;
  let m = 0;
  for (let b = 0; b < bins; b++) m = Math.max(m, Math.abs(h[b]! - e) / e);
  return m;
}

function uniformityTable(): string {
  const lines = [
    '| texture | global permutation | worst slice: max histogram deviation (16 bins) | worst pixel over time: max deviation (4 bins) | mean pixel over time: max deviation (4 bins) |',
    '|---|---|---|---|---|',
  ];
  const cases: Array<[string, Float32Array, number, number, boolean]> = [];
  for (const s of [64, 128]) cases.push([`vc2d-${s}`, channelOf(loadTexture(`vc2d-scalar-${s}`), 0), s, 1, true]);
  for (const [s, d] of [[64, 32], [128, 64]] as const) {
    cases.push([`white ${s}²x${d}`, whiteVolume(s, s, d, 13), s, d, false]);
    cases.push([`st-${s}x${d}`, channelOf(loadTexture(`st-scalar-${s}x${d}`), 0), s, d, true]);
  }
  for (const [name, vol, s, d, isRank] of cases) {
    const n = vol.length;
    let perm = '-';
    if (isRank) {
      // 1/(N-1) steps survive float32 at these sizes; check the ranks are a permutation.
      const seen = new Uint8Array(n);
      let ok = true;
      for (let i = 0; i < n; i++) {
        const r = Math.round(vol[i]! * (n - 1));
        if (seen[r]) ok = false;
        seen[r] = 1;
      }
      perm = ok ? 'yes' : 'NO';
    }
    let worstSlice = 0;
    for (let z = 0; z < d; z++) worstSlice = Math.max(worstSlice, histogramDeviation(vol.subarray(z * s * s, (z + 1) * s * s), 16));
    let worstPix = 0;
    let meanPix = 0;
    if (d > 1) {
      const seq = new Float32Array(d);
      for (let p = 0; p < s * s; p++) {
        for (let z = 0; z < d; z++) seq[z] = vol[z * s * s + p]!;
        const dev = histogramDeviation(seq, 4);
        worstPix = Math.max(worstPix, dev);
        meanPix += dev / (s * s);
      }
    }
    lines.push(`| ${name} | ${perm} | ${fmt(worstSlice, 3)} | ${d > 1 ? fmt(worstPix, 3) : '-'} | ${d > 1 ? fmt(meanPix, 3) : '-'} |`);
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------- integration A/B

/** Smooth, toroidal penumbra parameter in roughly [-0.1, 1.1]. */
function penumbra(x: number, y: number): number {
  return 0.5 + 0.6 * Math.sin((2 * Math.PI * (x + 0.5)) / IMG) * Math.cos((2 * Math.PI * (y + 0.5)) / IMG);
}

/** 1D: visibility of a linear light = P(u < c). 2D: area light [0,1]^2 cut by an occluder edge n.(u,v) < c'. */
function truth(dims: 1 | 2): Float64Array {
  const out = new Float64Array(IMG * IMG);
  const cs = Math.cos(THETA);
  const sn = Math.sin(THETA);
  for (let y = 0; y < IMG; y++) {
    for (let x = 0; x < IMG; x++) {
      const c = penumbra(x, y);
      if (dims === 1) out[y * IMG + x] = Math.min(1, Math.max(0, c));
      else {
        const c2 = c * (cs + sn);
        let s = 0;
        const M = 4096;
        for (let i = 0; i < M; i++) {
          const u = (i + 0.5) / M;
          s += Math.min(1, Math.max(0, (c2 - u * cs) / sn));
        }
        out[y * IMG + x] = s / M;
      }
    }
  }
  return out;
}

function integrand(dims: 1 | 2, x: number, y: number, u: number, v: number): number {
  const c = penumbra(x, y);
  if (dims === 1) return u < c ? 1 : 0;
  const cs = Math.cos(THETA);
  const sn = Math.sin(THETA);
  return u * cs + v * sn < c * (cs + sn) ? 1 : 0;
}

type SampleFn = (x: number, y: number, k: number, trial: number, out: number[]) => void;

interface Sampler {
  name: string;
  sample: SampleFn;
}

function makeSamplers(dims: 1 | 2): Sampler[] {
  const list: Sampler[] = [
    {
      name: 'white',
      sample: (x, y, k, t, out) => {
        const h = hash32((y * IMG + x) ^ Math.imul(t + 1, 0x9e3779b9));
        out[0] = hash32(h + 2 * k) / 4294967296;
        out[1] = hash32(h + 2 * k + 1) / 4294967296;
      },
    },
    {
      name: 'owen-sobol',
      sample: (x, y, k, t, out) => {
        owenSobol2D(k, hash32(y * IMG + x + Math.imul(t, 0x85ebca6b)), out);
      },
    },
  ];
  const flat: Array<[string, string]> = [];
  for (const s of [64, 128]) flat.push([`vc2d-${s}`, dims === 1 ? `vc2d-scalar-${s}` : `vc2d-vec2-${s}`]);
  if (dims === 2) for (const s of [64, 128]) flat.push([`gf-${s}`, `gf-vec2-${s}`]);
  for (const [label, texName] of flat) {
    const tex = loadTexture(texName);
    const s = tex.meta.width;
    const ch = tex.meta.channels;
    list.push({
      name: label,
      sample: (x, y, k, t, out) => {
        const r = mulberry32(t + 100);
        const ox = Math.floor(r() * s);
        const oy = Math.floor(r() * s);
        const i = (((y + oy) % s) * s + ((x + ox) % s)) * ch;
        if (dims === 1) out[0] = frac(tex.data[i]! + k * PHI1);
        else {
          out[0] = frac(tex.data[i]! + k * R2[0]);
          out[1] = frac(tex.data[i + 1]! + k * R2[1]);
        }
      },
    });
  }
  for (const [s, d] of [[64, 32], [128, 64]] as const) {
    const tex = loadTexture(dims === 1 ? `st-scalar-${s}x${d}` : `st-vec2-${s}x${d}`);
    const ch = tex.meta.channels;
    list.push({
      name: `st-${s}x${d}`,
      sample: (x, y, k, t, out) => {
        const r = mulberry32(t + 100);
        const ox = Math.floor(r() * s);
        const oy = Math.floor(r() * s);
        const oz = Math.floor(r() * d);
        const z = (k + oz) % d;
        const cycle = Math.floor((k + oz) / d) - Math.floor(oz / d);
        const i = ((z * s + ((y + oy) % s)) * s + ((x + ox) % s)) * ch;
        if (dims === 1) out[0] = frac(tex.data[i]! + cycle * PHI1);
        else {
          out[0] = frac(tex.data[i]! + cycle * R2[0]);
          out[1] = frac(tex.data[i + 1]! + cycle * R2[1]);
        }
      },
    });
  }
  return list;
}

/** Toroidal separable Gaussian blur: the stand-in for "a denoiser follows". */
function blur(img: Float64Array, sigma: number): Float64Array {
  const R = Math.ceil(3 * sigma);
  const k = new Float64Array(2 * R + 1);
  let ks = 0;
  for (let i = -R; i <= R; i++) ks += k[i + R] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < k.length; i++) k[i]! /= ks;
  const tmp = new Float64Array(img.length);
  const out = new Float64Array(img.length);
  for (let y = 0; y < IMG; y++) {
    for (let x = 0; x < IMG; x++) {
      let s = 0;
      for (let i = -R; i <= R; i++) s += k[i + R]! * img[y * IMG + ((x + i + IMG) % IMG)]!;
      tmp[y * IMG + x] = s;
    }
  }
  for (let y = 0; y < IMG; y++) {
    for (let x = 0; x < IMG; x++) {
      let s = 0;
      for (let i = -R; i <= R; i++) s += k[i + R]! * tmp[((y + i + IMG) % IMG) * IMG + x]!;
      out[y * IMG + x] = s;
    }
  }
  return out;
}

const rms = (a: Float64Array): number => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

/** Share of error power below 0.25 Nyquist, relative to white (white reads ~1, blue < 1). */
function lowBand(err: Float64Array): number {
  const v = variance(err);
  if (v === 0) return NaN;
  const raps = radialAverage(powerSpectrum2D(err, IMG, IMG), IMG, IMG);
  const nyq = raps.length;
  let s = 0;
  let c = 0;
  for (let i = 1; i < nyq; i++) if (i / nyq < 0.25) { s += raps[i]!; c++; }
  return s / c / v;
}

interface ErrStats { rmse: number; blurred: number; low: number }

function addStats(acc: ErrStats, err: Float64Array): void {
  acc.rmse += rms(err) / TRIALS;
  acc.blurred += rms(blur(err, BLUR_SIGMA)) / TRIALS;
  acc.low += lowBand(err) / TRIALS;
}

function writePfm(name: string, img: Float64Array): void {
  mkdirSync(PFM_DIR, { recursive: true });
  writeFileSync(join(PFM_DIR, `${name}.pfm`), encodePfm({ width: IMG, height: IMG, channels: 1, data: Float32Array.from(img) }));
}

function integrationTables(): string {
  const out: string[] = [];
  for (const dims of [1, 2] as const) {
    const ref = truth(dims);
    if (dims === 2) writePfm('reference-2d', ref);
    const samplers = makeSamplers(dims);
    out.push(`\n**${dims}D integrand** (${dims === 1 ? 'scalar textures, step u < c' : 'vec2 textures, area light cut by an edge'}); RMSE / RMSE after Gaussian blur σ=${BLUR_SIGMA} px / low-band error power (white = 1)\n`);
    out.push('| sampler | 1 spp | 2 spp | 4 spp |');
    out.push('|---|---|---|---|');
    const tmp = [0, 0];
    for (const smp of samplers) {
      const cells: string[] = [];
      for (const spp of [1, 2, 4]) {
        const acc: ErrStats = { rmse: 0, blurred: 0, low: 0 };
        for (let t = 0; t < TRIALS; t++) {
          const est = new Float64Array(IMG * IMG);
          for (let y = 0; y < IMG; y++) {
            for (let x = 0; x < IMG; x++) {
              let s = 0;
              for (let k = 0; k < spp; k++) {
                smp.sample(x, y, k, t, tmp);
                s += integrand(dims, x, y, tmp[0]!, tmp[1]!);
              }
              est[y * IMG + x] = s / spp;
            }
          }
          if (t === 0 && dims === 2 && (spp === 1 || spp === 4)) writePfm(`${smp.name}-2d-${spp}spp`, est);
          const err = est.map((v, i) => v - ref[i]!);
          addStats(acc, err);
        }
        cells.push(`${fmt(acc.rmse, 4)} / ${fmt(acc.blurred, 4)} / ${fmt(acc.low)}`);
      }
      out.push(`| ${smp.name} | ${cells.join(' | ')} |`);
    }
  }
  return out.join('\n');
}

/** 1 spp per frame, accumulated over frames: running mean and an EMA (alpha 0.1, TAA-like). */
function temporalAccumulationTable(): string {
  const dims = 2 as const;
  const ref = truth(dims);
  const samplers = makeSamplers(dims);
  const checkpoints = [1, 4, 16, 64];
  const F = 64;
  const ALPHA = 0.1;
  const lines = [
    `| sampler | ${checkpoints.map((c) => `mean of ${c}: RMSE / blurred`).join(' | ')} | EMA α=${ALPHA} @64: RMSE / blurred |`,
    '|---|' + '---|'.repeat(checkpoints.length + 1),
  ];
  const tmp = [0, 0];
  for (const smp of samplers) {
    const acc = checkpoints.map(() => ({ rmse: 0, blurred: 0 }));
    const ema = { rmse: 0, blurred: 0 };
    for (let t = 0; t < TRIALS; t++) {
      const sum = new Float64Array(IMG * IMG);
      const e = new Float64Array(IMG * IMG);
      for (let f = 0; f < F; f++) {
        for (let y = 0; y < IMG; y++) {
          for (let x = 0; x < IMG; x++) {
            smp.sample(x, y, f, t, tmp);
            const v = integrand(dims, x, y, tmp[0]!, tmp[1]!);
            const i = y * IMG + x;
            sum[i]! += v;
            e[i] = f === 0 ? v : e[i]! + ALPHA * (v - e[i]!);
          }
        }
        const ci = checkpoints.indexOf(f + 1);
        if (ci >= 0) {
          const err = sum.map((v, i) => v / (f + 1) - ref[i]!);
          acc[ci]!.rmse += rms(err) / TRIALS;
          acc[ci]!.blurred += rms(blur(err, BLUR_SIGMA)) / TRIALS;
          if (t === 0 && f + 1 === F) writePfm(`${smp.name}-2d-mean${F}`, sum.map((v) => v / F));
        }
      }
      const err = e.map((v, i) => v - ref[i]!);
      ema.rmse += rms(err) / TRIALS;
      ema.blurred += rms(blur(err, BLUR_SIGMA)) / TRIALS;
    }
    lines.push(`| ${smp.name} | ${acc.map((a) => `${fmt(a.rmse, 4)} / ${fmt(a.blurred, 4)}`).join(' | ')} | ${fmt(ema.rmse, 4)} / ${fmt(ema.blurred, 4)} |`);
  }
  return lines.join('\n');
}

function main(): void {
  if (!existsSync(join(ASSETS, 'manifest.json'))) throw new Error('run pnpm bluenoise:generate first');
  const t0 = performance.now();
  console.log('## Spatial spectrum (radially averaged, white = 1; columns are bands of the Nyquist frequency)\n');
  console.log(spectraTables());
  console.log('\n## Rank uniformity\n');
  console.log(uniformityTable());
  console.log('\n## Per-pixel temporal spectrum (white = 1; columns are bands of the temporal Nyquist frequency)\n');
  console.log(temporalTables());
  console.log(`\n## Integration A/B (${IMG}² image, ${TRIALS} trials of random texture offsets / seeds)\n`);
  console.log(integrationTables());
  console.log('\n## Temporal accumulation (2D integrand, 1 spp per frame)\n');
  console.log(temporalAccumulationTable());
  console.log(`\n(report took ${((performance.now() - t0) / 1000).toFixed(1)} s; PFMs in ${PFM_DIR})`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();
