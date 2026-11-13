/** Metrics worker: holds copies of the loaded images and computes pair metrics + LDR-FLIP (and, on request, HDR-FLIP) off the main thread. */
import { compareImages } from '../../metrics/metrics.ts';
import { flipHdr, flipLdr } from '../../metrics/flip.ts';
import type { Image } from '../../src/adapter.ts';

export type ToWorker =
  | { type: 'image'; slot: number; width: number; height: number; data: Float32Array }
  | { type: 'clear'; slot: number }
  | { type: 'pair'; key: string; test: number; ref: number }
  | { type: 'hdr'; key: string; test: number; ref: number };

export type FromWorker =
  | { type: 'metrics'; key: string; metrics: ReturnType<typeof compareImages> }
  | { type: 'flip'; key: string; mean: number; map: Float32Array; width: number; height: number }
  | { type: 'flipHdr'; key: string; mean: number; map: Float32Array; width: number; height: number; start: number; stop: number; count: number }
  | { type: 'hdrProgress'; key: string; done: number; count: number }
  | { type: 'error'; key: string; message: string };

const images = new Map<number, Image>();

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === 'image') images.set(m.slot, { width: m.width, height: m.height, channels: 3, data: m.data });
  else if (m.type === 'clear') images.delete(m.slot);
  else if (m.type === 'hdr') {
    const t = images.get(m.test), r = images.get(m.ref);
    if (!t || !r) return;
    try {
      // single thread: about 15 s at 1280x720 (progress is posted per exposure); the CLI uses worker_threads
      const h = flipHdr(t, r, { onExposure: (k, count) => (self as unknown as Worker).postMessage({ type: 'hdrProgress', key: m.key, done: k + 1, count } satisfies FromWorker) });
      (self as unknown as Worker).postMessage(
        { type: 'flipHdr', key: m.key, mean: h.mean, map: h.map, width: t.width, height: t.height, start: h.start, stop: h.stop, count: h.count } satisfies FromWorker, [h.map.buffer]);
    } catch (err) {
      (self as unknown as Worker).postMessage({ type: 'error', key: m.key, message: String(err) } satisfies FromWorker);
    }
  } else {
    const t = images.get(m.test), r = images.get(m.ref);
    if (!t || !r) return;
    try {
      const metrics = compareImages(t, r);
      (self as unknown as Worker).postMessage({ type: 'metrics', key: m.key, metrics } satisfies FromWorker);
      const f = flipLdr(t, r);
      (self as unknown as Worker).postMessage(
        { type: 'flip', key: m.key, mean: f.mean, map: f.map, width: t.width, height: t.height } satisfies FromWorker, [f.map.buffer]);
    } catch (err) {
      (self as unknown as Worker).postMessage({ type: 'error', key: m.key, message: String(err) } satisfies FromWorker);
    }
  }
};
