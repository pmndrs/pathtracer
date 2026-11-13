/**
 * HDR-FLIP spread over worker_threads (Node only; not in the browser-safe barrel). Exposures are independent, so the
 * work splits perfectly; each worker max-folds its share and the main thread folds the partial maps. The result is
 * bit-identical to `flipHdr` (max is order independent; ties keep the lower exposure index).
 */
import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';
import type { Image } from '../src/adapter.ts';
import { DEFAULT_PPD, hdrFlipPlan, type HdrFlipOptions, type HdrFlipResult } from './flip.ts';
import { assertSameSize } from './metrics.ts';

function shared(img: Image): Float32Array {
  const n = img.width * img.height;
  const out = new Float32Array(new SharedArrayBuffer(n * 12));
  for (let i = 0; i < n; i++) { out[3 * i] = img.data[i * img.channels]!; out[3 * i + 1] = img.data[i * img.channels + 1]!; out[3 * i + 2] = img.data[i * img.channels + 2]!; }
  return out;
}

export async function flipHdrParallel(test: Image, ref: Image, opts: HdrFlipOptions = {}, threads = Math.max(1, availableParallelism() - 2)): Promise<HdrFlipResult> {
  assertSameSize(test, ref);
  const tonemap = opts.tonemap ?? 'aces';
  const ppd = opts.ppd ?? DEFAULT_PPD;
  const plan = hdrFlipPlan(ref, opts);
  const T = Math.min(threads, plan.count);
  const t = shared(test), r = shared(ref);
  const parts = await Promise.all(Array.from({ length: T }, (_, w) => new Promise<{ map: Float32Array; exposureIndex: Uint8Array }>((resolve, reject) => {
    const indices = plan.exposures.map((_, k) => k).filter((k) => k % T === w);
    const worker = new Worker(new URL('./flip-worker.ts', import.meta.url), {
      workerData: { width: ref.width, height: ref.height, test: t, ref: r, exposures: indices.map((k) => plan.exposures[k]!), indices, tonemap, ppd },
    });
    worker.once('message', resolve); worker.once('error', reject);
  })));
  const N = ref.width * ref.height;
  const map = new Float32Array(N), exposureIndex = new Uint8Array(N);
  for (const p of parts) {
    for (let i = 0; i < N; i++) {
      const v = p.map[i]!;
      if (v > map[i]! || (v === map[i]! && p.exposureIndex[i]! < exposureIndex[i]!)) { map[i] = v; exposureIndex[i] = p.exposureIndex[i]!; }
    }
  }
  let sum = 0;
  for (let i = 0; i < N; i++) sum += map[i]!;
  return { map, mean: sum / N, exposureIndex, start: plan.start, stop: plan.stop, count: plan.count, tonemap, ppd };
}
