/**
 * The contract between a bench page (browser) and the runner (Node). Types and constants only, so both
 * sides can import it. Shape follows ~/Developer/bvh/tests/gpu/contract.ts (MIT, same author), reduced to what the
 * pathtracer bench needs: one JSON result per page run, big images through the upload endpoint.
 */

/** Which GPU a run is asked to use. `apple` is the integrated Apple GPU on macOS. */
export type GpuChoice = 'apple' | 'nvidia' | 'intel' | 'amd';
export const GPU_CHOICES: readonly GpuChoice[] = ['apple', 'nvidia', 'intel', 'amd'];

/** What `GPUAdapter.info` says, copied into plain fields (it does not JSON.stringify). */
export interface AdapterSummary {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  isFallbackAdapter: boolean;
  features: string[];
}

/** What a page hands to `reportResult`. Everything is optional except the metrics. */
export interface BenchResult {
  /** Scene id (bench/scenes). Used for the CSV `scene` column. */
  scene?: string;
  /** Renderer or variant name (`ours`, `mutracer`, `smoke`, ...). CSV `renderer` column. */
  renderer?: string;
  /** The page's own configuration, hashed with the page args into the CSV `config_hash` column. */
  config?: Record<string, unknown>;
  /** Named numbers; the unit goes in the name (`gpu_ms_total`, `mrays_per_s`). Each becomes one CSV row per run. */
  metrics: Record<string, number>;
  /** False when the page itself knows its numbers are not trustworthy (quantized timers, fallback path, ...). */
  valid?: boolean;
  /** Free-form facts; printed and written to the run JSON, not to the CSV. */
  notes?: Record<string, unknown>;
}

/** The object a page leaves on `window.__benchReport` when it is done. */
export interface BenchReport {
  /** Echo of the `token` query parameter, so the runner ignores a stale page's report. */
  token: string | null;
  status: 'pass' | 'fail';
  /** Why the page failed. Absent on pass. */
  message?: string;
  result: BenchResult | null;
  adapter: AdapterSummary | null;
  /** Text of every `uncapturederror` the device raised. Any entry fails the run. */
  deviceErrors: string[];
  durationMs: number;
}

/** Record of one image the page uploaded (written as `<name>.json` next to the PFM). */
export interface UploadRecord {
  name: string;
  width: number;
  height: number;
  /** Channels the page sent (3 or 4). The PFM always has 3: alpha is dropped. */
  channels: 1 | 3 | 4;
  token: string | null;
  slot: string;
  /** Path of the PFM, relative to the repo root. */
  pfm: string;
  bytes: number;
  /** Count of NaN or infinite floats in the data (a renderer bug worth knowing about). */
  nonFinite: number;
  receivedAt: string;
}

/** Name of the window property the page writes its {@link BenchReport} to. */
export const REPORT_PROPERTY = '__benchReport';

/** Query parameters the runner adds to a page URL; they are not page args. */
export const QUERY_TOKEN = 'token';
export const QUERY_GPU = 'gpu';
export const RESERVED_QUERY = new Set([QUERY_TOKEN, QUERY_GPU]);

/** The upload endpoint (POST, raw float32 body). */
export const UPLOAD_PATH = '/__bench/upload';

export const DEFAULT_PAGE_TIMEOUT_MS = 60_000;

/** One `--json` run summary. */
export interface RunSummary {
  schema: 1;
  runId: string;
  startedAt: string;
  commit?: string;
  page: string;
  gpuRequested: GpuChoice | null;
  adapter: AdapterSummary | null;
  browser: string | null;
  ok: boolean;
  totalMs: number;
  error: string | null;
  timingVerdict: { kind: 'valid' | 'invalid' | 'no-verdict'; reason: string } | null;
  runs: RunRecord[];
  uploads: UploadRecord[];
  csvFile: string | null;
}

export interface RunRecord {
  /** 1-based index in execution order. */
  index: number;
  /** Variant label (`A`/`B` for an A/B run, `-` otherwise). */
  variant: string;
  args: Record<string, string>;
  outcome: 'pass' | 'fail';
  reasons: string[];
  durationMs: number;
  result: BenchResult | null;
  /** Machine load measured just before this run (P0.30); absent on records from older runners. */
  load?: {
    load1: number;
    loadSource: 'loadavg' | 'cpu-delta' | 'unavailable';
    gpuProcs: number | null;
    loaded: boolean;
    waitedMs: number;
    reasons: string[];
  };
}
