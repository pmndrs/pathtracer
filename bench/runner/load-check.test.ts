import { describe, expect, it } from 'vitest';
import { parseRunnerArgs } from './args.ts';
import type { RunRecord } from './contract.ts';
import { countForeignGpuProcesses, cpuBusyFraction, formatLoadNote, judgeLoad, parseProcessList, waitForQuiet, type LoadSample } from './load-check.ts';
import { buildRows } from './plan.ts';

const t = { maxLoad: 4, maxGpuProcs: 2 };
const sample = (load1: number, gpuProcs: number | null, loadSource: LoadSample['loadSource'] = 'loadavg'): LoadSample => ({ load1, loadSource, gpuProcs });

describe('judgeLoad', () => {
  it('is quiet at and under the thresholds', () => {
    expect(judgeLoad(sample(4, 2), t)).toEqual({ loaded: false, reasons: [] });
  });
  it('names every reason when over', () => {
    const v = judgeLoad(sample(4.5, 3), t);
    expect(v.loaded).toBe(true);
    expect(v.reasons).toEqual(['load 4.50 > 4', '3 Chrome GPU processes > 2']);
  });
  it('never counts an unavailable measurement as loaded', () => {
    expect(judgeLoad(sample(99, null, 'unavailable'), t).loaded).toBe(false);
  });
});

describe('process parsing', () => {
  const ps = [
    '  100     1 /sbin/launchd',
    '  200   100 node bench/runner/run.ts --page smoke',
    '  300   200 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --user-data-dir=/tmp/pt-bench-chrome-a',
    '  301   300 /Applications/Google Chrome.app/.../Google Chrome Helper (GPU) --type=gpu-process --user-data-dir=/tmp/pt-bench-chrome-a',
    '  400   100 /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --remote-debugging-port=0',
    '  401   400 Google Chrome Helper (GPU) --type=gpu-process --use-angle=metal',
    '  501   500 Google Chrome Helper (GPU) --type=gpu-process --headless',
    '  502   500 Google Chrome Helper (Renderer) --type=renderer',
    'garbage line',
  ].join('\n');
  it('parses pid, ppid and command and skips junk', () => {
    const procs = parseProcessList(ps);
    expect(procs).toHaveLength(8);
    expect(procs[1]).toEqual({ pid: 200, ppid: 100, command: 'node bench/runner/run.ts --page smoke' });
  });
  it('counts foreign GPU processes only, not our own descendants', () => {
    expect(countForeignGpuProcesses(parseProcessList(ps), 200)).toBe(2);
    expect(countForeignGpuProcesses(parseProcessList(ps), 999)).toBe(3);
  });
  it('handles CRLF output (Windows)', () => {
    expect(parseProcessList('10 1 chrome.exe --type=gpu-process\r\n').length).toBe(1);
  });
});

describe('cpuBusyFraction', () => {
  it('is 1 - idle share', () => {
    expect(cpuBusyFraction({ idle: 100, total: 200 }, { idle: 150, total: 300 })).toBeCloseTo(0.5);
  });
  it('is 0 when no time passed', () => {
    expect(cpuBusyFraction({ idle: 1, total: 1 }, { idle: 1, total: 1 })).toBe(0);
  });
});

describe('formatLoadNote', () => {
  it('formats and flags loaded', () => {
    expect(formatLoadNote(sample(1.234, 1), false)).toBe('load 1.23, gpu-procs 1');
    expect(formatLoadNote(sample(7, 5), true)).toBe('load 7.00, gpu-procs 5; loaded');
    expect(formatLoadNote(sample(0.5, null, 'cpu-delta'), false)).toBe('load 0.50 (cpu), gpu-procs n/a');
    expect(formatLoadNote(sample(0, null, 'unavailable'), false)).toBe('load n/a, gpu-procs n/a');
  });
});

describe('waitForQuiet', () => {
  function fake(samples: LoadSample[]) {
    let time = 0;
    let i = 0;
    const slept: number[] = [];
    return {
      slept,
      deps: {
        sample: async () => samples[Math.min(i++, samples.length - 1)] as LoadSample,
        sleep: async (ms: number) => {
          slept.push(ms);
          time += ms;
        },
        now: () => time,
      },
    };
  }
  it('returns at once when quiet', async () => {
    const f = fake([sample(1, 0)]);
    const out = await waitForQuiet(f.deps, t, 120_000);
    expect(out).toMatchObject({ loaded: false, waitedMs: 0 });
    expect(f.slept).toEqual([]);
  });
  it('waits until the load drops', async () => {
    const f = fake([sample(9, 0), sample(8, 0), sample(2, 0)]);
    const out = await waitForQuiet(f.deps, t, 120_000, 5000);
    expect(out.loaded).toBe(false);
    expect(f.slept).toEqual([5000, 5000]);
    expect(out.waitedMs).toBe(10_000);
  });
  it('gives up at the deadline and reports loaded', async () => {
    const f = fake([sample(9, 5)]);
    const out = await waitForQuiet(f.deps, t, 12_000, 5000);
    expect(out.loaded).toBe(true);
    expect(f.slept).toEqual([5000, 5000, 2000]);
    expect(out.reasons).toHaveLength(2);
  });
  it('does not wait with a zero wait budget', async () => {
    const f = fake([sample(9, 0)]);
    expect((await waitForQuiet(f.deps, t, 0)).loaded).toBe(true);
    expect(f.slept).toEqual([]);
  });
});

describe('CSV rows for loaded runs', () => {
  const run = (index: number, loaded: boolean): RunRecord => ({
    index,
    variant: '-',
    args: {},
    outcome: 'pass',
    reasons: [],
    durationMs: 1,
    result: { scene: 's', renderer: 'r', metrics: { ms: index } },
    load: { ...sample(loaded ? 9 : 1, loaded ? 4 : 0), loaded, waitedMs: 0, reasons: [] },
  });
  const base = { date: 'd', commit: 'c', device: 'dev', browser: 'b', page: 'p', timingValid: true, verdictNote: 'timing valid' };
  it('marks only loaded runs valid=0 and puts load in the note', () => {
    const rows = buildRows({ ...base, runs: [run(1, false), run(2, true)] });
    expect(rows.map((r) => [r.run, r.valid])).toEqual([['1', 1], ['2', 0], ['median', 0]]);
    expect(rows[0]?.note).toBe('timing valid; load 1.00, gpu-procs 0');
    expect(rows[1]?.note).toContain('loaded');
    expect(rows[2]?.note).toContain('loaded in 1 of 2 runs');
  });
});

describe('load options', () => {
  it('defaults: cores, 2 procs, 120 s, not strict', () => {
    const o = parseRunnerArgs(['--page', 'x']);
    expect(o.maxLoad).toBeGreaterThan(0);
    expect([o.maxGpuProcs, o.loadWaitS, o.strictLoad]).toEqual([2, 120, false]);
  });
  it('parses overrides', () => {
    const o = parseRunnerArgs(['--page', 'x', '--max-load', '3.5', '--max-gpu-procs', '0', '--load-wait', '0', '--strict-load']);
    expect([o.maxLoad, o.maxGpuProcs, o.loadWaitS, o.strictLoad]).toEqual([3.5, 0, 0, true]);
  });
  it('rejects junk', () => {
    expect(() => parseRunnerArgs(['--max-load', 'abc'])).toThrow();
    expect(() => parseRunnerArgs(['--load-wait', '-1'])).toThrow();
  });
});
