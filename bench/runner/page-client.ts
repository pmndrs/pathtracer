/**
 * Browser helper that bench pages import:
 *
 *   import { getArgs, uploadImage, reportResult, reportError } from '/bench/runner/page-client.ts';
 *
 * Shape follows ~/Developer/bvh/tests/gpu/client.ts (MIT, same author). Rules for pages:
 *  - small JSON goes through reportResult (the runner reads it over CDP);
 *  - big float images go through uploadImage (HTTP POST to the runner's server, written as PFM), never CDP;
 *  - read back once, after the timed region (no mapAsync in a frame loop).
 * Opening the page by hand in Chrome works too: it draws a green or red summary.
 */

import { QUERY_TOKEN, REPORT_PROPERTY, RESERVED_QUERY, UPLOAD_PATH, type AdapterSummary, type BenchReport, type BenchResult, type UploadRecord } from './contract.ts';

const startTime = performance.now();
const params = new URLSearchParams(location.search);
const token = params.get(QUERY_TOKEN);
const uploaded: string[] = [];
let finished = false;

/** Page arguments from `--args k=v` / `--ab k=A,B`, as strings. The runner's own parameters (token, gpu) are left out. */
export function getArgs(): Record<string, string> {
  const args: Record<string, string> = {};
  for (const [key, value] of params) if (!RESERVED_QUERY.has(key)) args[key] = value;
  return args;
}

/** Number argument with a default; throws on a value that is not a number, so a typo does not become a silent default. */
export function numberArg(args: Record<string, string>, key: string, fallback: number): number {
  const text = args[key];
  if (text === undefined) return fallback;
  const value = Number(text);
  if (!Number.isFinite(value)) throw new Error(`Argument ${key}="${text}" is not a number.`);
  return value;
}

export interface UploadableImage {
  width: number;
  height: number;
  channels: 1 | 3 | 4;
  /** Linear float32, row-major, row 0 = top (bench/src/adapter.ts `Image`). */
  data: Float32Array;
}

/**
 * Sends one image to the runner, which writes it to bench/results/incoming/<run-id>/<slot>/<name>.pfm (flipped to
 * PFM's bottom-to-top; alpha dropped). `name` is a plain file stem: letters, digits, `.`, `_`, `-`.
 */
export async function uploadImage(name: string, image: UploadableImage): Promise<UploadRecord> {
  const query = new URLSearchParams({ name, w: String(image.width), h: String(image.height), c: String(image.channels) });
  if (token) query.set(QUERY_TOKEN, token);
  const body = new Uint8Array(image.data.buffer, image.data.byteOffset, image.data.byteLength);
  const response = await fetch(`${UPLOAD_PATH}?${query}`, { method: 'POST', body: body as unknown as BodyInit, headers: { 'Content-Type': 'application/octet-stream' } });
  const json = (await response.json()) as UploadRecord | { error: string };
  if (!response.ok || 'error' in json) throw new Error(`Upload of "${name}" failed: ${'error' in json ? json.error : response.status}`);
  uploaded.push(name);
  return json;
}

/** `adapter.info` as plain fields (it does not JSON.stringify). */
export function summarizeAdapter(adapter: GPUAdapter): AdapterSummary {
  const info = adapter.info;
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: (info as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? false,
    features: [...adapter.features].sort(),
  };
}

/** Records every `uncapturederror` (addEventListener, never onuncapturederror). Any entry fails the run. */
export function watchDevice(device: GPUDevice): string[] {
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (event) => errors.push((event as GPUUncapturedErrorEvent).error.message));
  void device.lost.then((info) => errors.push(`device lost (${info.reason}): ${info.message}`));
  return errors;
}

/** Waits for the queue and a short grace, so late validation errors show up before the report. */
export async function drainDevice(device: GPUDevice): Promise<void> {
  await device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 50));
}

export interface ReportExtras {
  adapter?: AdapterSummary | null;
  deviceErrors?: string[];
}

function plain(value: unknown): unknown {
  // CDP returnByValue turns typed arrays into index-keyed objects, and JSON cannot hold bigint or NaN: convert first.
  return JSON.parse(
    JSON.stringify(value, (_key, item: unknown) => {
      if (typeof item === 'bigint') return item.toString();
      if (typeof item === 'number' && !Number.isFinite(item)) return null;
      if (ArrayBuffer.isView(item) && !(item instanceof DataView)) return Array.from(item as unknown as ArrayLike<number>);
      return item;
    }),
  );
}

function publish(report: BenchReport): void {
  if (finished) return;
  finished = true;
  (window as unknown as Record<string, unknown>)[REPORT_PROPERTY] = report;
  drawSummary(report);
}

/** Ends the run successfully. `metrics` become CSV rows; pass `adapter` (and `deviceErrors`) so the runner can check them. */
export function reportResult(result: BenchResult, extras: ReportExtras = {}): void {
  const clean = plain(result) as BenchResult;
  publish({
    token,
    status: 'pass',
    result: { ...clean, notes: { ...clean.notes, uploads: uploaded.slice() } },
    adapter: extras.adapter ?? null,
    deviceErrors: extras.deviceErrors ?? [],
    durationMs: performance.now() - startTime,
  });
}

/** Ends the run as failed. */
export function reportError(error: unknown, extras: ReportExtras = {}): void {
  publish({
    token,
    status: 'fail',
    message: error instanceof Error ? (error.stack ?? error.message) : String(error),
    result: null,
    adapter: extras.adapter ?? null,
    deviceErrors: extras.deviceErrors ?? [],
    durationMs: performance.now() - startTime,
  });
}

// An exception nobody caught must fail the run too, instead of leaving the runner to time out.
window.addEventListener('error', (event) => reportError(`Uncaught error: ${event.message}`));
window.addEventListener('unhandledrejection', (event) => reportError(`Unhandled rejection: ${String(event.reason)}`));

function drawSummary(report: BenchReport): void {
  const view = document.createElement('pre');
  view.style.cssText = 'font: 14px monospace; padding: 12px; color: #eee; background: #111; margin: 0; white-space: pre-wrap;';
  const adapter = report.adapter ? `${report.adapter.vendor} / ${report.adapter.architecture}` : 'no adapter reported';
  view.textContent = [
    `${document.title}: ${report.status.toUpperCase()}`,
    `adapter: ${adapter}`,
    report.message ? `message: ${report.message}` : '',
    report.result ? JSON.stringify(report.result, null, 2) : '',
  ]
    .filter((line) => line !== '')
    .join('\n');
  document.body.style.cssText = `margin: 0; background: ${report.status === 'pass' ? '#0b3d17' : '#5a1111'};`;
  document.body.append(view);
}
