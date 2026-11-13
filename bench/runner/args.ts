import { parseArgs } from 'node:util';
import { GPU_CHOICES, type GpuChoice } from './contract.ts';
import { DEFAULT_LOAD_WAIT_S, DEFAULT_MAX_GPU_PROCS, defaultMaxLoad } from './load-check.ts';

export interface RunnerOptions {
  page: string | null;
  /** Which GPU to demand; null means "whatever Chrome picks" (still reported). */
  gpu: GpuChoice | null;
  headed: boolean;
  /** Runs per variant (per ABBA round-pair with --ab, so --ab with --repeat 3 makes 6 runs). */
  repeat: number;
  /** Page arguments, passed in the URL query and read by `getArgs()`. */
  args: Record<string, string>;
  /** A/B: one page arg and its two values. Runs are interleaved A B B A. */
  ab: { key: string; values: [string, string] } | null;
  /** CSV suite name (bench/results/<suite>.csv); default is the page name. */
  suite: string | null;
  chromePath: string | null;
  jsonPath: string | null;
  verbose: boolean;
  /** Launch Chrome with the GPU disabled, to see the "WebGPU unavailable" failure. */
  withoutWebGpu: boolean;
  /** Accept a software fallback adapter (SwiftShader) instead of failing. For GPU-less CI only. */
  allowSoftware: boolean;
  /** Skip the cross-agent GPU lock. */
  noLock: boolean;
  /** Do not append CSV rows (images and JSON are still written). */
  noCsv: boolean;
  timeoutMs: number | null;
  /** Machine-load gate (P0.30): loaded when load1 > maxLoad or Chrome GPU processes > maxGpuProcs. */
  maxLoad: number;
  maxGpuProcs: number;
  /** Seconds to wait for the machine to quiet down before running anyway as `loaded`. */
  loadWaitS: number;
  /** Fail instead of running loaded. */
  strictLoad: boolean;
  help: boolean;
}

/** A mistake in the command line. The message is shown to the user as is. */
export class UsageError extends Error {}

export const USAGE = `Usage: node bench/runner/run.ts --page <name> [options]

  --page <name>        Page to run: bench/pages/<name>.html (required).
  --gpu apple|nvidia|intel|amd
                       Demand this GPU. The run fails if Chrome gives another one.
  --headed             Show the Chrome window (default is headless).
  --repeat <n>         Run n times and report the median (default 1). With --ab: n ABBA rounds of 2 runs per round.
  --args k=v [k=v ...] Page arguments (read in the page with getArgs()). Stops at the next --flag.
  --ab key=A,B         A/B: run the page with key=A and key=B, interleaved A B B A per --repeat round.
  --suite <name>       CSV file bench/results/<name>.csv (default: the page name).
  --no-csv             Do not append CSV rows.
  --chrome <path>      Chrome or Edge executable (default: look for an installed one; or CHROME_PATH).
  --json <file>        Also write the whole run as JSON.
  --timeout <ms>       Replace the page timeout (default 60000).
  --no-lock            Do not take the shared GPU lock (RT_GPU_LOCK / RT_GPU_QUEUE, see gpu-lock.ts).
  --allow-software     Accept a software adapter (SwiftShader). Also PT_ALLOW_SOFTWARE_ADAPTER=1. GPU-less CI only.
  --max-load <n>       Machine counts as loaded above this 1-minute load average (default: CPU core count; on Windows the
                       CPU busy fraction over 1 s times the core count).
  --max-gpu-procs <n>  Loaded above this many other Chrome GPU processes (default 2).
  --load-wait <s>      Wait up to this long for the load to drop (default 120), then run anyway, rows marked valid=0 'loaded'.
  --strict-load        Fail the run instead of running loaded.
  --no-webgpu          Start Chrome with the GPU disabled. The run must fail clearly (self-check).
  --verbose            Print all console output, not only for failing runs.
  --help               Show this text.`;

/**
 * `--args` takes any number of k=v words, which node:util parseArgs cannot express, so they are lifted out first.
 * Returns the argv without them, and the parsed pairs.
 */
export function extractPageArgs(argv: string[]): { rest: string[]; args: Record<string, string> } {
  const rest: string[] = [];
  const args: Record<string, string> = {};
  let collecting = false;
  for (const word of argv) {
    if (word === '--args') {
      collecting = true;
      continue;
    }
    if (collecting && word.startsWith('--')) collecting = false;
    if (!collecting) {
      rest.push(word);
      continue;
    }
    const eq = word.indexOf('=');
    if (eq <= 0) throw new UsageError(`--args expects key=value words, got "${word}".`);
    args[word.slice(0, eq)] = word.slice(eq + 1);
  }
  return { rest, args };
}

export function parseRunnerArgs(argv: string[], env: NodeJS.ProcessEnv = process.env): RunnerOptions {
  const { rest, args } = extractPageArgs(argv);
  let values;
  try {
    ({ values } = parseArgs({
      args: rest,
      options: {
        page: { type: 'string' },
        gpu: { type: 'string' },
        headed: { type: 'boolean' },
        repeat: { type: 'string' },
        ab: { type: 'string' },
        suite: { type: 'string' },
        'no-csv': { type: 'boolean' },
        chrome: { type: 'string' },
        json: { type: 'string' },
        timeout: { type: 'string' },
        'no-lock': { type: 'boolean' },
        'allow-software': { type: 'boolean' },
        'max-load': { type: 'string' },
        'max-gpu-procs': { type: 'string' },
        'load-wait': { type: 'string' },
        'strict-load': { type: 'boolean' },
        'no-webgpu': { type: 'boolean' },
        verbose: { type: 'boolean' },
        help: { type: 'boolean' },
      },
      strict: true,
      allowPositionals: false,
    }));
  } catch (error) {
    throw new UsageError(error instanceof Error ? error.message : String(error));
  }

  if (values.gpu !== undefined && !(GPU_CHOICES as readonly string[]).includes(values.gpu)) {
    throw new UsageError(`--gpu must be one of ${GPU_CHOICES.join(', ')}, got "${values.gpu}".`);
  }
  const positive = (name: string, text: string | undefined, fallback: number | null): number | null => {
    if (text === undefined) return fallback;
    const value = Number(text);
    if (!Number.isInteger(value) || value <= 0) throw new UsageError(`--${name} must be a positive integer, got "${text}".`);
    return value;
  };
  const repeat = positive('repeat', values.repeat, 1) as number;
  const timeoutMs = positive('timeout', values.timeout, null);
  const nonNegative = (name: string, text: string | undefined, fallback: number): number => {
    if (text === undefined) return fallback;
    const value = Number(text);
    if (!Number.isFinite(value) || value < 0) throw new UsageError(`--${name} must be a number >= 0, got "${text}".`);
    return value;
  };
  const maxLoad = nonNegative('max-load', values['max-load'], defaultMaxLoad());
  const maxGpuProcs = nonNegative('max-gpu-procs', values['max-gpu-procs'], DEFAULT_MAX_GPU_PROCS);
  const loadWaitS = nonNegative('load-wait', values['load-wait'], DEFAULT_LOAD_WAIT_S);

  let ab: RunnerOptions['ab'] = null;
  if (values.ab !== undefined) {
    const match = /^([^=,\s]+)=([^,]+),([^,]+)$/.exec(values.ab);
    if (!match) throw new UsageError(`--ab expects key=A,B (two values), got "${values.ab}".`);
    ab = { key: match[1] as string, values: [match[2] as string, match[3] as string] };
    if (ab.key in args) throw new UsageError(`--ab key "${ab.key}" is also given in --args.`);
  }
  if (values.page !== undefined && !/^[A-Za-z0-9._-]+$/.test(values.page)) {
    throw new UsageError(`--page must be a plain page name (letters, digits, . _ -), got "${values.page}".`);
  }

  return {
    page: values.page ?? null,
    gpu: (values.gpu as GpuChoice | undefined) ?? null,
    headed: values.headed ?? false,
    repeat,
    args,
    ab,
    suite: values.suite ?? null,
    chromePath: values.chrome ?? null,
    jsonPath: values.json ?? null,
    verbose: values.verbose ?? false,
    withoutWebGpu: values['no-webgpu'] ?? false,
    allowSoftware: values['allow-software'] === true || env.PT_ALLOW_SOFTWARE_ADAPTER === '1',
    noLock: values['no-lock'] ?? false,
    noCsv: values['no-csv'] ?? false,
    timeoutMs,
    maxLoad,
    maxGpuProcs,
    loadWaitS,
    strictLoad: values['strict-load'] ?? false,
    help: values.help ?? false,
  };
}
