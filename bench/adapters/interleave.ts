/**
 * Pure logic for `matrix.ts --repeat N --interleave`: round ordering and merging N single-run runner invocations into the
 * layout one `run.ts --repeat N` invocation would have left (run.json with runs 1..N, uploads in slots k-run, CSV rows run 1..N + median).
 * No filesystem or process access here, so it is unit tested (interleave.test.ts).
 */
import { buildRows, deviceLabel, utcDate } from '../runner/plan.ts';
import type { CsvRow } from '../runner/csv.ts';
import type { RunSummary } from '../runner/contract.ts';
import { slotFor } from '../runner/report.ts';
import type { ReportManifest } from '../viewer/src/report.ts';

/** Renderer order of round `round` (0-based): forward on even rounds, reversed on odd ones (A B C, C B A, A B C ...). */
export function roundOrder<T>(renderers: readonly T[], round: number): T[] {
  return round % 2 === 0 ? [...renderers] : [...renderers].reverse();
}

/** One finished single-run invocation (`--repeat 1`) of a cell. `runDir` is its repo-relative runner directory. */
export interface RoundResult { summary: RunSummary; runDir: string }

export interface MergedRounds {
  summary: RunSummary;
  /** Directory copies (repo-relative) that put round k's single `1-run` slot at `k-run` in the merged directory. */
  copies: Array<{ from: string; to: string }>;
  /** CSV rows (run 1..N + median) as the runner would have appended for a `--repeat N` invocation; empty when any round failed. */
  rows: CsvRow[];
}

const posixPath = (p: string): string => p.split('\\').join('/');

/**
 * Merges the rounds of one cell (in round order) into one summary as `run.ts --repeat N` would write it. Round k becomes run k.
 * Rounds must come from `--repeat 1 --no-csv` invocations. `mergedDir` is the repo-relative directory the merged run.json goes to.
 */
export function mergeRounds(rounds: readonly RoundResult[], mergedRunId: string, mergedDir: string): MergedRounds {
  if (rounds.length === 0) throw new Error('mergeRounds: no rounds');
  const first = rounds[0]!.summary;
  const runs: RunSummary['runs'] = [];
  const uploads: RunSummary['uploads'] = [];
  const copies: MergedRounds['copies'] = [];
  rounds.forEach(({ summary, runDir }, k) => {
    const index = k + 1;
    for (const run of summary.runs) runs.push({ ...run, index: index + (run.index - 1), variant: run.variant });
    const from = slotFor(1, '-');
    const to = slotFor(index, '-');
    copies.push({ from: posixPath(`${runDir}/${from}`), to: posixPath(`${mergedDir}/${to}`) });
    for (const upload of summary.uploads) {
      const prefix = `${posixPath(runDir)}/${from}/`;
      const pfm = posixPath(upload.pfm);
      uploads.push({ ...upload, slot: to, pfm: pfm.startsWith(prefix) ? `${posixPath(mergedDir)}/${to}/${pfm.slice(prefix.length)}` : upload.pfm });
    }
  });
  const verdicts = rounds.map((r) => r.summary.timingVerdict);
  const bad = verdicts.find((v) => v && v.kind === 'invalid') ?? verdicts.find((v) => v && v.kind === 'no-verdict') ?? null;
  const timingVerdict = bad ?? verdicts.find((v) => v) ?? null;
  const ok = rounds.every((r) => r.summary.ok);
  const summary: RunSummary = {
    schema: 1,
    runId: mergedRunId,
    startedAt: first.startedAt,
    ...(first.commit !== undefined ? { commit: first.commit } : {}),
    page: first.page,
    gpuRequested: first.gpuRequested,
    adapter: first.adapter,
    browser: first.browser,
    ok,
    totalMs: rounds.reduce((sum, r) => sum + r.summary.totalMs, 0),
    error: rounds.map((r) => r.summary.error).find((e) => e !== null) ?? null,
    timingVerdict: timingVerdict ? { ...timingVerdict } : null,
    runs,
    uploads,
    csvFile: null,
  };
  let rows: CsvRow[] = [];
  if (ok && summary.adapter) {
    const kind = timingVerdict?.kind;
    // One flag for the whole cell, as in a single --repeat N invocation: an invalid round makes every row of the cell valid=0.
    rows = buildRows({
      date: utcDate(new Date(first.startedAt)),
      commit: first.commit ?? 'unknown',
      device: deviceLabel(summary.adapter),
      browser: summary.browser ?? 'Chrome',
      page: summary.page,
      runs,
      timingValid: kind === 'valid',
      verdictNote: kind === 'valid' ? 'timing valid (interleaved)' : `timing ${kind ?? 'no-verdict'}: ${timingVerdict?.reason ?? ''} (interleaved)`.slice(0, 160),
    });
  }
  return { summary, copies, rows };
}

/** Points a round's report manifest at the merged directory: its cell images live in `1-run/`, now `${round}-run/` (round is 1-based). */
export function retargetManifest(manifest: ReportManifest, round: number, mergedRunId: string): ReportManifest {
  const from = `${slotFor(1, '-')}/`;
  const to = `${slotFor(round, '-')}/`;
  const fix = (p: string | undefined): string | undefined => (p !== undefined && p.startsWith(from) ? to + p.slice(from.length) : p);
  return {
    ...manifest,
    run: mergedRunId,
    cells: manifest.cells.map((c) => ({ ...c, image: fix(c.image) ?? '', ...(c.thumb ? { thumb: fix(c.thumb) } : {}) })),
  };
}
