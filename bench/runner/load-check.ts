/**
 * Machine-load check before a timed run (P0.30). The Mac is shared and other agents' Chrome jobs do not take our GPU
 * lock, so timings taken on a loaded machine are not trusted (maintainer ruling). Before each run, after the lock is
 * held, we record the 1-minute load average and the number of foreign Chrome GPU processes, wait for them to drop
 * under the thresholds, and otherwise mark the run `loaded` (or fail it with --strict-load).
 *
 * Everything that decides or parses is a pure function (tested without spawning anything); the sampling functions at
 * the bottom are the only ones that touch the OS.
 */

import { execFile } from 'node:child_process';
import { cpus, loadavg } from 'node:os';

export interface LoadSample {
  /** 1-minute load average; on Windows the CPU busy fraction over ~1 s times the core count (os.loadavg() is 0 there). */
  load1: number;
  /** Where load1 came from. */
  loadSource: 'loadavg' | 'cpu-delta' | 'unavailable';
  /** Chrome GPU processes (`--type=gpu-process`) that are not descendants of this process; null when `ps` failed. */
  gpuProcs: number | null;
}

export interface LoadThresholds {
  /** Loaded when load1 is above this (default: the number of CPU cores). */
  maxLoad: number;
  /** Loaded when gpuProcs is above this (default 2). */
  maxGpuProcs: number;
}

export const DEFAULT_MAX_GPU_PROCS = 2;
export const DEFAULT_LOAD_WAIT_S = 120;

export interface LoadVerdict {
  loaded: boolean;
  reasons: string[];
}

/** Pure: is this sample over a threshold? An unavailable measurement never counts as loaded. */
export function judgeLoad(sample: LoadSample, thresholds: LoadThresholds): LoadVerdict {
  const reasons: string[] = [];
  if (sample.loadSource !== 'unavailable' && sample.load1 > thresholds.maxLoad) {
    reasons.push(`load ${sample.load1.toFixed(2)} > ${thresholds.maxLoad}`);
  }
  if (sample.gpuProcs !== null && sample.gpuProcs > thresholds.maxGpuProcs) {
    reasons.push(`${sample.gpuProcs} Chrome GPU processes > ${thresholds.maxGpuProcs}`);
  }
  return { loaded: reasons.length > 0, reasons };
}

export interface ProcessInfo {
  pid: number;
  ppid: number;
  command: string;
}

/** Pure: lines of `<pid> <ppid> <command...>` (what `ps -axo pid=,ppid=,command=` and our PowerShell line print). */
export function parseProcessList(text: string): ProcessInfo[] {
  const out: ProcessInfo[] = [];
  for (const line of text.split(/\r?\n/)) {
    const match = /^\s*(\d+)\s+(\d+)\s+(.*)$/.exec(line);
    if (match) out.push({ pid: Number(match[1]), ppid: Number(match[2]), command: match[3] as string });
  }
  return out;
}

/**
 * Pure: Chrome GPU processes that are not ours. Ours are the descendants of `ownPid` (the bench Chrome we launched is
 * our child); counting them would make every run look loaded.
 */
export function countForeignGpuProcesses(processes: readonly ProcessInfo[], ownPid: number): number {
  const childrenOf = new Map<number, number[]>();
  for (const p of processes) childrenOf.set(p.ppid, [...(childrenOf.get(p.ppid) ?? []), p.pid]);
  const own = new Set<number>();
  const stack = [ownPid];
  while (stack.length > 0) {
    const pid = stack.pop() as number;
    for (const child of childrenOf.get(pid) ?? []) {
      if (!own.has(child)) {
        own.add(child);
        stack.push(child);
      }
    }
  }
  return processes.filter((p) => p.command.includes('--type=gpu-process') && !own.has(p.pid)).length;
}

export interface CpuTimes {
  idle: number;
  total: number;
}

/** Pure: the CPU busy fraction between two snapshots (summed over cores), 0 when no time passed. */
export function cpuBusyFraction(before: CpuTimes, after: CpuTimes): number {
  const total = after.total - before.total;
  if (total <= 0) return 0;
  return Math.min(1, Math.max(0, 1 - (after.idle - before.idle) / total));
}

/** Pure: the one-line note for the CSV `note` column and the log. */
export function formatLoadNote(sample: LoadSample, loaded: boolean): string {
  const load = sample.loadSource === 'unavailable' ? 'load n/a' : `load ${sample.load1.toFixed(2)}${sample.loadSource === 'cpu-delta' ? ' (cpu)' : ''}`;
  const gpu = sample.gpuProcs === null ? 'gpu-procs n/a' : `gpu-procs ${sample.gpuProcs}`;
  return `${load}, ${gpu}${loaded ? '; loaded' : ''}`;
}

export interface LoadOutcome {
  sample: LoadSample;
  loaded: boolean;
  reasons: string[];
  waitedMs: number;
}

export interface WaitDeps {
  sample: () => Promise<LoadSample>;
  sleep: (ms: number) => Promise<void>;
  now: () => number;
  /** Called once when the first sample is over a threshold. */
  onWait?: (reasons: string[]) => void;
}

/**
 * Samples, and while loaded rechecks every `pollMs` until `waitMs` has passed. Returns the last sample; `loaded` is
 * true when it was still over a threshold at the deadline. The deadline is checked after each sample, so a slow
 * sampler can overshoot it by one sample.
 */
export async function waitForQuiet(deps: WaitDeps, thresholds: LoadThresholds, waitMs: number, pollMs = 5000): Promise<LoadOutcome> {
  const start = deps.now();
  let sample = await deps.sample();
  let verdict = judgeLoad(sample, thresholds);
  if (verdict.loaded) deps.onWait?.(verdict.reasons);
  while (verdict.loaded && deps.now() - start < waitMs) {
    await deps.sleep(Math.min(pollMs, Math.max(0, waitMs - (deps.now() - start))));
    sample = await deps.sample();
    verdict = judgeLoad(sample, thresholds);
  }
  return { sample, loaded: verdict.loaded, reasons: verdict.reasons, waitedMs: deps.now() - start };
}

/** CPU cores, for the default --max-load. */
export function defaultMaxLoad(): number {
  return Math.max(1, cpus().length);
}

function run(file: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(file, args, { maxBuffer: 16 * 1024 * 1024, windowsHide: true, timeout: 10_000 }, (error, stdout) => (error ? reject(error) : resolve(stdout)));
  });
}

function cpuTimes(): CpuTimes {
  let idle = 0;
  let total = 0;
  for (const cpu of cpus()) {
    idle += cpu.times.idle;
    total += cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq;
  }
  return { idle, total };
}

/** Real sampler. Never throws: a failing probe becomes `unavailable` / null and is shown as n/a. */
export async function sampleLoad(ownPid: number = process.pid): Promise<LoadSample> {
  let load1 = 0;
  let loadSource: LoadSample['loadSource'] = 'unavailable';
  let sleepAfter: Promise<void> | null = null;
  if (process.platform === 'win32') {
    // os.loadavg() is always 0 on Windows: use the CPU busy fraction over one second instead.
    const before = cpuTimes();
    sleepAfter = new Promise((resolve) => setTimeout(resolve, 1000));
    await sleepAfter;
    load1 = cpuBusyFraction(before, cpuTimes()) * cpus().length;
    loadSource = 'cpu-delta';
  } else {
    load1 = loadavg()[0] ?? 0;
    loadSource = 'loadavg';
  }
  let gpuProcs: number | null = null;
  try {
    const text =
      process.platform === 'win32'
        ? // tasklist has no command lines; CIM does.
          await run('powershell.exe', [
            '-NoProfile',
            '-Command',
            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*--type=gpu-process*' } | ForEach-Object { '{0} {1} {2}' -f $_.ProcessId, $_.ParentProcessId, $_.CommandLine }",
          ])
        : await run('ps', ['-axo', 'pid=,ppid=,command=']);
    gpuProcs = countForeignGpuProcesses(parseProcessList(text), ownPid);
  } catch {
    gpuProcs = null;
  }
  return { load1, loadSource, gpuProcs };
}
