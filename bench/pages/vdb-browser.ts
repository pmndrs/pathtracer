/**
 * vdb-browser (V3.4): loads a `.vdb` in a Web Worker (WebVDB `parseVdb` + the shared `convertVdbFile`) and reports the dense Float32 grid's sha256, resolution, bounds and peak.
 * Pass `expect=<sha256>` (the Node path's grid: `shasum -a 256 bench/assets/generated/volumes/vdb-smoke.f32`) to fail on a mismatch; the check is bit-identity of the Float32 bytes.
 * Args: file=<path under WebVDB, default fixtures/vdb-samples/smoke.vdb>  down=2  border=2  expect=<hex>. No GPU, no timing (worker times are informational).
 */
import { getArgs, reportError, reportResult } from '../runner/page-client.ts';
import { defaultWebVdbToolsUrl, loadVdbDense } from '../scenes/volumes/vdb-browser.ts';

declare const __BENCH_WEBVDB__: string | null | undefined;

async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

try {
  const args = getArgs();
  const rel = args.file ?? 'fixtures/vdb-samples/smoke.vdb';
  const response = await fetch(`${__BENCH_WEBVDB__}/${rel}`);
  if (!response.ok) throw new Error(`fetch ${rel}: ${response.status}`);
  const bytes = await response.arrayBuffer();
  const t0 = performance.now();
  const { dense, workerParseMs, workerConvertMs } = await loadVdbDense(bytes, rel, { toolsUrl: defaultWebVdbToolsUrl(), downsample: args.down ? Number(args.down) : undefined, border: args.border ? Number(args.border) : undefined });
  const wallMs = performance.now() - t0;
  const gridSha = await sha256Hex(dense.density.buffer as ArrayBuffer);
  if (args.expect && args.expect !== gridSha) throw new Error(`dense grid sha256 ${gridSha} != expected ${args.expect}`);
  reportResult({
    scene: 'vdb-browser',
    renderer: 'vdb-worker',
    config: { file: rel, downsample: args.down ?? 2, border: args.border ?? 2 },
    metrics: { cells: dense.density.length, peak_before_normalisation: dense.peak, worker_parse_ms_informational: workerParseMs, worker_convert_ms_informational: workerConvertMs, wall_ms_informational: wallMs },
    notes: { gridSha256: gridSha, fileSha256: dense.source.sha256, resolution: dense.resolution, bounds: dense.bounds, source: dense.source, matchedExpected: Boolean(args.expect) },
  });
} catch (error) {
  reportError(error);
}
