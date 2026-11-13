import { describe, expect, it } from 'vitest';
import type { RunRecord } from './contract.ts';
import { buildRows, deviceLabel, makeRunId, planRuns, utcDate } from './plan.ts';

describe('planRuns', () => {
  it('repeats a single config', () => {
    expect(planRuns({ args: { a: '1' }, ab: null, repeat: 2 })).toEqual([
      { variant: '-', args: { a: '1' } },
      { variant: '-', args: { a: '1' } },
    ]);
  });

  it('interleaves an A/B as ABBA and sets the key', () => {
    const plan = planRuns({ args: { spp: '8' }, ab: { key: 'k', values: ['x', 'y'] }, repeat: 2 });
    expect(plan.map((run) => run.variant)).toEqual(['A', 'B', 'B', 'A']);
    expect(plan.map((run) => run.args.k)).toEqual(['x', 'y', 'y', 'x']);
    expect(plan.every((run) => run.args.spp === '8')).toBe(true);
  });
});

function record(index: number, variant: string, metrics: Record<string, number>, extra: Partial<RunRecord['result'] & object> = {}): RunRecord {
  return { index, variant, args: { k: variant }, outcome: 'pass', reasons: [], durationMs: 1, result: { scene: 's', renderer: 'r', metrics, ...extra } };
}

describe('buildRows', () => {
  const base = { date: '2026-11-13', commit: 'abc', device: 'dev', browser: 'br', page: 'p', timingValid: true, verdictNote: 'timing valid' };

  it('writes each run and a median row per metric and variant', () => {
    const rows = buildRows({ ...base, runs: [record(1, 'A', { ms: 1 }), record(2, 'B', { ms: 5 }), record(3, 'B', { ms: 7 }), record(4, 'A', { ms: 3 })] });
    const a = rows.filter((row) => row.note.includes('variant A'));
    expect(a.map((row) => [row.run, row.value])).toEqual([['1', 1], ['2', 3], ['median', 2]]);
    const b = rows.filter((row) => row.note.includes('variant B'));
    expect(b.map((row) => [row.run, row.value])).toEqual([['1', 5], ['2', 7], ['median', 6]]);
    expect(new Set(rows.map((row) => row.config_hash)).size).toBe(2);
  });

  it('writes no median row for a single run', () => {
    const rows = buildRows({ ...base, runs: [record(1, '-', { ms: 1, n: 2 })] });
    expect(rows.map((row) => row.run)).toEqual(['1', '1']);
    expect(rows[0]?.note).toBe('timing valid');
  });

  it('marks rows invalid for a bad timing verdict or a page that says so', () => {
    expect(buildRows({ ...base, timingValid: false, runs: [record(1, '-', { ms: 1 })] })[0]?.valid).toBe(0);
    const rows = buildRows({ ...base, runs: [record(1, '-', { ms: 1 }, { valid: false }), record(2, '-', { ms: 1 })] });
    expect(rows.map((row) => row.valid)).toEqual([0, 1, 0]); // runs 1, 2, then the median (not all valid)
  });

  it('skips failed runs and falls back to page name and "-" scene', () => {
    const failed: RunRecord = { ...record(1, '-', { ms: 1 }), outcome: 'fail', result: null };
    const ok: RunRecord = { ...record(2, '-', { ms: 2 }), result: { metrics: { ms: 2 } } };
    const rows = buildRows({ ...base, runs: [failed, ok] });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ scene: '-', renderer: 'p', run: '1' });
  });
});

describe('labels', () => {
  it('formats the UTC date, run id and device', () => {
    const date = new Date('2026-11-13T04:17:13.456Z');
    expect(utcDate(date)).toBe('2026-11-13');
    expect(makeRunId(date, 'smoke')).toBe('20261113T041713Z-smoke');
    const adapter = { vendor: 'apple', architecture: 'metal-3', device: '', description: 'Apple M5 Pro', isFallbackAdapter: false, features: [] };
    expect(deviceLabel(adapter)).toBe('Apple M5 Pro');
    expect(deviceLabel({ ...adapter, description: '' })).toBe('apple metal-3');
  });
});
