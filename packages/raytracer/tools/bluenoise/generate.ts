/**
 * Generates the blue-noise assets into packages/raytracer/assets/bluenoise/ (`pnpm bluenoise:generate`).
 * Deterministic (seeded). One job per (texture family, size, channel) runs in a worker thread.
 *
 * Output per texture: `<name>.png` (16-bit; scalar = grey, vec2 = RGB with B = 0; volumes tiled into an atlas,
 * committed) and `<name>.f32` (raw little-endian float32, git-ignored, rebuilt by this script), plus `manifest.json`.
 * Families:
 *  - `vc2d`: classic 2D void-and-cluster. The scalar texture is channel 0 of the vec2 one; channel 1 is an
 *    independent run with another seed.
 *  - `st`: spatiotemporal void-and-cluster (3D torus, separate spatial / temporal sigma), same channel scheme.
 *  - `gf`: vec2 only, jointly optimised by pixel swaps (Georgiev & Fajardo 2016), see vectorSwap.ts.
 */
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { mkdirSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import type { BlueNoiseMeta } from '../../src/noise/bluenoise.ts';
import { encodePng } from './png.ts';
import { vectorSwapVec2 } from './vectorSwap.ts';
import { voidAndCluster } from './voidAndCluster.ts';

export const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'assets', 'bluenoise');

export const SIGMA_SPATIAL = 1.5;
export const SIGMA_TEMPORAL = 1.5;
export const GF_ITERATIONS_PER_PIXEL = 200;
export const GF_SIGMA_IMAGE = 2.1;
const BASE_SEED = 20261113;

type Kind = BlueNoiseMeta['kind'];

interface Job {
  kind: Kind;
  size: number;
  depth: number;
  /** For `gf` the single job produces both channels, interleaved. */
  channel: 0 | 1;
  seed: number;
}

interface JobResult {
  job: Job;
  values: Float32Array;
  ms: number;
}

/** (family, size, depth): 2D classic and gf at 64 and 128; spatiotemporal at 64x32 slices and 128x64 slices. */
const FAMILIES: Array<{ kind: Kind; size: number; depth: number }> = [
  { kind: 'vc2d', size: 64, depth: 1 },
  { kind: 'vc2d', size: 128, depth: 1 },
  { kind: 'st', size: 64, depth: 32 },
  { kind: 'st', size: 128, depth: 64 },
  { kind: 'gf', size: 64, depth: 1 },
  { kind: 'gf', size: 128, depth: 1 },
];

export function textureName(kind: Kind, channels: 1 | 2, size: number, depth: number): string {
  return `${kind}-${channels === 1 ? 'scalar' : 'vec2'}-${size}${depth > 1 ? `x${depth}` : ''}`;
}

function atlasColumns(depth: number): number {
  return depth === 1 ? 1 : 2 ** Math.ceil(Math.log2(depth) / 2);
}

function runJob(job: Job): JobResult {
  if (job.kind === 'gf') {
    const g = vectorSwapVec2({ width: job.size, height: job.size, sigmaImage: GF_SIGMA_IMAGE, iterationsPerPixel: GF_ITERATIONS_PER_PIXEL, seed: job.seed });
    return { job, values: g.values, ms: g.ms };
  }
  const r = voidAndCluster({
    width: job.size,
    height: job.size,
    depth: job.depth,
    sigmaSpatial: SIGMA_SPATIAL,
    sigmaTemporal: SIGMA_TEMPORAL,
    seed: job.seed,
  });
  return { job, values: r.values, ms: r.ms };
}

function toPng(channels: Float32Array[], size: number, depth: number) {
  const cols = atlasColumns(depth);
  const rows = Math.ceil(depth / cols);
  const outCh = channels.length === 1 ? 1 : 3;
  const width = size * cols;
  const height = size * rows;
  const data = new Uint16Array(width * height * outCh);
  for (let z = 0; z < depth; z++) {
    const ox = (z % cols) * size;
    const oy = Math.floor(z / cols) * size;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const src = (z * size + y) * size + x;
        const dst = ((oy + y) * width + ox + x) * outCh;
        for (let c = 0; c < channels.length; c++) data[dst + c] = Math.round(channels[c]![src]! * 65535);
      }
    }
  }
  return encodePng({ width, height, channels: outCh as 1 | 3, bitDepth: 16, data });
}

async function main(): Promise<void> {
  const t0 = performance.now();
  const jobs: Job[] = [];
  FAMILIES.forEach((f, fi) => {
    const channels = f.kind === 'gf' ? [0 as const] : ([0, 1] as const);
    for (const channel of channels) jobs.push({ ...f, channel, seed: BASE_SEED + fi * 101 + channel * 7919 });
  });
  // Slowest first so the pool stays busy (a gf pixel costs about as much as 4 st cells).
  const cost = (j: Job) => j.size * j.size * j.depth * (j.kind === 'gf' ? 4 : 1);
  jobs.sort((a, b) => cost(b) - cost(a));
  const results: JobResult[] = [];
  let next = 0;
  const poolSize = Math.max(1, Math.min(jobs.length, cpus().length));
  await Promise.all(
    Array.from({ length: poolSize }, async () => {
      while (next < jobs.length) {
        const job = jobs[next++]!;
        results.push(
          await new Promise<JobResult>((resolve, reject) => {
            const w = new Worker(fileURLToPath(import.meta.url), { workerData: job });
            w.once('message', resolve);
            w.once('error', reject);
          }),
        );
      }
    }),
  );
  mkdirSync(OUT_DIR, { recursive: true });
  const textures: BlueNoiseMeta[] = [];
  const timings: string[] = [];
  for (const f of FAMILIES) {
    const n = f.size * f.size * f.depth;
    const mine = results.filter((r) => r.job.kind === f.kind && r.job.size === f.size).sort((a, b) => a.job.channel - b.job.channel);
    // Per-channel planes: gf comes back interleaved from one job.
    const planes =
      f.kind === 'gf'
        ? [0, 1].map((c) => Float32Array.from({ length: n }, (_, i) => mine[0]!.values[2 * i + c]!))
        : mine.map((r) => r.values);
    timings.push(`${f.kind} ${f.size}^2 x${f.depth}: ${mine.map((r) => r.ms.toFixed(0)).join(' + ')} ms (one core per job)`);
    for (const channels of f.kind === 'gf' ? ([2] as const) : ([1, 2] as const)) {
      const used = planes.slice(0, channels);
      const raw = new Float32Array(n * channels);
      for (let i = 0; i < n; i++) for (let c = 0; c < channels; c++) raw[i * channels + c] = used[c]![i]!;
      const name = textureName(f.kind, channels, f.size, f.depth);
      writeFileSync(join(OUT_DIR, `${name}.f32`), new Uint8Array(raw.buffer));
      writeFileSync(join(OUT_DIR, `${name}.png`), toPng(used, f.size, f.depth));
      textures.push({
        name,
        kind: f.kind,
        width: f.size,
        height: f.size,
        depth: f.depth,
        channels,
        f32: `${name}.f32`,
        png: `${name}.png`,
        atlasColumns: atlasColumns(f.depth),
        sigmaSpatial: f.kind === 'gf' ? GF_SIGMA_IMAGE : SIGMA_SPATIAL,
        sigmaTemporal: f.depth > 1 ? SIGMA_TEMPORAL : null,
        seed: mine[0]!.job.seed,
      });
    }
  }
  textures.sort((a, b) => a.name.localeCompare(b.name));
  writeFileSync(join(OUT_DIR, 'manifest.json'), `${JSON.stringify({ version: 1, textures }, null, 2)}\n`);
  console.log(timings.join('\n'));
  console.log(`wrote ${textures.length} textures to ${OUT_DIR} in ${((performance.now() - t0) / 1000).toFixed(1)} s wall`);
}

if (isMainThread) {
  if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) await main();
} else {
  parentPort!.postMessage(runJob(workerData as Job));
}
