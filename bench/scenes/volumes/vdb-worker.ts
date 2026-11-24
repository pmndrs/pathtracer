/**
 * Web Worker for `vdb-browser.ts` (V3.4): parses a `.vdb` ArrayBuffer with WebVDB's `parseVdb` and runs the shared `convertVdbFile`, off the main thread.
 * WebVDB's tools entry is imported by URL (the bench server serves the sibling checkout through `/@fs/`, see `webVdbRoot` in runner/server.ts), so the bench has no build-time dependency on it.
 */
import { convertVdbFile, type VdbConvertOptions, type VdbDense, type VdbFileLike } from './vdb-convert.ts';

export interface VdbWorkerRequest {
  bytes: ArrayBuffer;
  file: string;
  toolsUrl: string;
  options?: VdbConvertOptions;
}
export type VdbWorkerReply = { ok: true; dense: VdbDense; parseMs: number; convertMs: number } | { ok: false; error: string };

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

self.onmessage = async (event: MessageEvent<VdbWorkerRequest>) => {
  try {
    const { bytes, file, toolsUrl, options } = event.data;
    const sha = await sha256Hex(bytes);
    const tools = (await import(/* @vite-ignore */ toolsUrl)) as { parseVdb(b: ArrayBuffer): VdbFileLike };
    const t0 = performance.now();
    const vdb = tools.parseVdb(bytes);
    const t1 = performance.now();
    const dense = convertVdbFile(vdb, file, sha, options);
    const t2 = performance.now();
    const reply: VdbWorkerReply = { ok: true, dense, parseMs: t1 - t0, convertMs: t2 - t1 };
    (self as unknown as Worker).postMessage(reply, [dense.density.buffer]);
  } catch (error) {
    const reply: VdbWorkerReply = { ok: false, error: error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error) };
    (self as unknown as Worker).postMessage(reply);
  }
};
