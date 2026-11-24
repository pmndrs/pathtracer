/**
 * Browser `.vdb` loader (V3.4): `loadVdbDense(bytes)` parses an OpenVDB file in a Web Worker and returns the same `VdbDense` the Node path (`vdb-grid.ts`) writes to a file.
 * The worker imports WebVDB's `parseVdb` by URL: pass `toolsUrl` (the bench server defines `__BENCH_WEBVDB__`, the `/@fs/` URL of the checkout; null when it is missing).
 * Packaging this into a library (a dependency of `@pmndrs/pathtracer` on `@pmndrs/vdb`) is a maintainer decision and not done here.
 */
import type { VdbConvertOptions, VdbDense } from './vdb-convert.ts';
import type { VdbWorkerReply, VdbWorkerRequest } from './vdb-worker.ts';

declare const __BENCH_WEBVDB__: string | null | undefined;

export const WEBVDB_TOOLS_PATH = 'packages/vdb/dist/tools/index.js';

export function defaultWebVdbToolsUrl(): string {
  if (typeof __BENCH_WEBVDB__ !== 'string') throw new Error('WebVDB checkout not found (WEBVDB_DIR or ~/Developer/WebVDB with packages/vdb/dist built)');
  return `${__BENCH_WEBVDB__}/${WEBVDB_TOOLS_PATH}`;
}

export interface VdbBrowserResult {
  dense: VdbDense;
  workerParseMs: number;
  workerConvertMs: number;
}

/** `bytes` is transferred to the worker (detached afterwards): pass a copy to keep it. */
export function loadVdbDense(bytes: ArrayBuffer, file: string, options: VdbConvertOptions & { toolsUrl?: string } = {}): Promise<VdbBrowserResult> {
  const { toolsUrl, ...convert } = options;
  const worker = new Worker(new URL('./vdb-worker.ts', import.meta.url), { type: 'module' });
  return new Promise((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<VdbWorkerReply>) => {
      worker.terminate();
      const r = event.data;
      if (r.ok) resolve({ dense: r.dense, workerParseMs: r.parseMs, workerConvertMs: r.convertMs });
      else reject(new Error(r.error));
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(`vdb worker failed: ${e.message}`));
    };
    const request: VdbWorkerRequest = { bytes, file, toolsUrl: toolsUrl ?? defaultWebVdbToolsUrl(), options: convert };
    worker.postMessage(request, [bytes]);
  });
}
