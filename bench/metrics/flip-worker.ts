/** worker_threads body for flip-parallel.ts: computes the max-fold of a subset of HDR-FLIP exposures. Node only. */
import { parentPort, workerData } from 'node:worker_threads';
import { hdrFlipExposureMap, type HdrTonemapName } from './flip.ts';

interface Job {
  width: number; height: number; test: Float32Array; ref: Float32Array; // SharedArrayBuffer-backed
  exposures: number[]; indices: number[]; tonemap: HdrTonemapName; ppd: number;
}
const j = workerData as Job;
const test = { width: j.width, height: j.height, channels: 3 as const, data: j.test };
const ref = { width: j.width, height: j.height, channels: 3 as const, data: j.ref };
const N = j.width * j.height;
const map = new Float32Array(N), exposureIndex = new Uint8Array(N);
for (let n = 0; n < j.exposures.length; n++) {
  const m = hdrFlipExposureMap(test, ref, j.exposures[n]!, j.tonemap, j.ppd);
  const k = j.indices[n]!;
  for (let i = 0; i < N; i++) if (m[i]! > map[i]! || (m[i]! === map[i]! && k < exposureIndex[i]!)) { map[i] = m[i]!; exposureIndex[i] = k; }
}
parentPort!.postMessage({ map, exposureIndex }, [map.buffer, exposureIndex.buffer]);
