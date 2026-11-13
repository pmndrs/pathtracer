/** Pure planning and row-building for run.ts, kept apart so tests can import it without starting anything. */

import { configHash, type CsvRow } from './csv.ts';
import type { RunnerOptions } from './args.ts';
import type { AdapterSummary, RunRecord } from './contract.ts';
import { abbaOrder, median } from './stats.ts';

/** UTC date for the CSV; the run's start time, not the write time. */
export function utcDate(date: Date): string {
  return date.toISOString().slice(0, 10);
}

/** `20261113T101500Z-ab12`: sorts by time, unique enough for one machine. */
export function makeRunId(date: Date, page: string): string {
  const stamp = date.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
  return `${stamp}-${page}`;
}

/** Device label for the CSV: the adapter description when there is one (it names the GPU), else vendor/architecture. */
export function deviceLabel(adapter: AdapterSummary): string {
  return adapter.description || adapter.device || `${adapter.vendor} ${adapter.architecture}`.trim();
}

/** One planned page run. */
export interface PlannedRun {
  variant: string;
  args: Record<string, string>;
}

export function planRuns(options: Pick<RunnerOptions, 'args' | 'ab' | 'repeat'>): PlannedRun[] {
  if (options.ab) {
    const { key, values } = options.ab;
    return abbaOrder<0 | 1>(0, 1, options.repeat).map((side) => ({
      variant: side === 0 ? 'A' : 'B',
      args: { ...options.args, [key]: values[side] },
    }));
  }
  return Array.from({ length: options.repeat }, () => ({ variant: '-', args: { ...options.args } }));
}

/**
 * CSV rows for a set of finished runs: every metric of every run (`run` = its 1-based index within the variant),
 * plus a `median` row per variant and metric when the variant ran more than once.
 */
export function buildRows(input: {
  date: string;
  commit: string;
  device: string;
  browser: string;
  page: string;
  runs: RunRecord[];
  /** Applied to every row: the timing verdict and its reason. */
  timingValid: boolean;
  verdictNote: string;
}): CsvRow[] {
  const rows: CsvRow[] = [];
  const variants = [...new Set(input.runs.map((run) => run.variant))];
  for (const variant of variants) {
    const runs = input.runs.filter((run) => run.variant === variant && run.outcome === 'pass' && run.result);
    if (runs.length === 0) continue;
    const first = runs[0]?.result;
    const scene = first?.scene ?? '-';
    const renderer = first?.renderer ?? input.page;
    const hash = configHash({ page: input.page, args: runs[0]?.args, config: first?.config });
    const valid = (run: RunRecord): 0 | 1 => (input.timingValid && run.result?.valid !== false ? 1 : 0);
    const base = { date: input.date, commit: input.commit, device: input.device, browser: input.browser, scene, renderer, config_hash: hash };
    const note = (extra: string) => [variant === '-' ? '' : `variant ${variant}`, extra, input.verdictNote].filter(Boolean).join('; ');
    const metricNames = [...new Set(runs.flatMap((run) => Object.keys(run.result?.metrics ?? {})))];
    for (const metric of metricNames) {
      const values: number[] = [];
      runs.forEach((run, index) => {
        const value = run.result?.metrics[metric];
        if (typeof value !== 'number') return;
        values.push(value);
        rows.push({ ...base, metric, value, valid: valid(run), run: String(index + 1), note: note('') });
      });
      if (runs.length > 1 && values.length > 0) {
        const allValid = runs.every((run) => valid(run) === 1);
        rows.push({ ...base, metric, value: median(values), valid: allValid ? 1 : 0, run: 'median', note: note(`median of ${values.length}`) });
      }
    }
  }
  return rows;
}

