import { describe, expect, it } from 'vitest';
import type { RunRecord, RunSummary } from '../runner/contract.ts';
import { mergeRounds, retargetManifest, roundOrder } from './interleave.ts';

describe('roundOrder', () => {
  it('reverses every other round', () => {
    const r = ['A', 'B', 'C'];
    expect([0, 1, 2, 3].map((k) => roundOrder(r, k).join(''))).toEqual(['ABC', 'CBA', 'ABC', 'CBA']);
    expect(r).toEqual(['A', 'B', 'C']);
  });
});

const round = (n: number, wall: number, extra: Partial<RunSummary> = {}): { summary: RunSummary; runDir: string } => {
  const run: RunRecord = { index: 1, variant: '-', args: { spp: '8' }, outcome: 'pass', reasons: [], durationMs: 1, result: { scene: 'cornell', renderer: 'r', metrics: { wall_ms: wall }, valid: true } as never };
  return {
    runDir: `bench/results/incoming/round${n}`,
    summary: {
      schema: 1, runId: `round${n}`, startedAt: '2026-11-13T10:00:00.000Z', commit: 'abc', page: 'competitor', gpuRequested: null,
      adapter: { vendor: 'apple', architecture: 'm', device: '', description: 'Apple M', isFallbackAdapter: false } as never,
      browser: 'Chrome', ok: true, totalMs: 10, error: null, timingVerdict: { kind: 'valid', reason: 'ok' }, runs: [run],
      uploads: [{ name: 'r__cornell.pfm', width: 1, height: 1, channels: 3, token: null, slot: '1-run', pfm: `bench/results/incoming/round${n}/1-run/r__cornell.pfm`, bytes: 1, nonFinite: 0, receivedAt: '' }],
      csvFile: null, ...extra,
    },
  };
};

describe('mergeRounds', () => {
  const merged = mergeRounds([round(1, 30), round(2, 10), round(3, 20)], 'round1-x3', 'bench/results/incoming/round1-x3');
  it('numbers runs 1..N and moves slots', () => {
    expect(merged.summary.runs.map((r) => r.index)).toEqual([1, 2, 3]);
    expect(merged.summary.uploads.map((u) => u.slot)).toEqual(['1-run', '2-run', '3-run']);
    expect(merged.summary.uploads[1]!.pfm).toBe('bench/results/incoming/round1-x3/2-run/r__cornell.pfm');
    expect(merged.copies[2]).toEqual({ from: 'bench/results/incoming/round3/1-run', to: 'bench/results/incoming/round1-x3/3-run' });
    expect(merged.summary.runId).toBe('round1-x3');
  });
  it('writes CSV rows run 1..N plus a median', () => {
    const wall = merged.rows.filter((r) => r.metric === 'wall_ms');
    expect(wall.map((r) => r.run)).toEqual(['1', '2', '3', 'median']);
    expect(wall.map((r) => r.value)).toEqual([30, 10, 20, 20]);
    expect(wall.every((r) => r.valid === 1)).toBe(true);
  });
  it('marks every row invalid when one round timing is invalid, and writes none when a round failed', () => {
    const bad = mergeRounds([round(1, 1), round(2, 2, { timingVerdict: { kind: 'invalid', reason: 'throttled' } })], 'x', 'd');
    expect(bad.rows.every((r) => r.valid === 0)).toBe(true);
    const failed = mergeRounds([round(1, 1), round(2, 2, { ok: false })], 'x', 'd');
    expect(failed.rows).toEqual([]);
  });
});

describe('retargetManifest', () => {
  it('moves 1-run images to the round slot', () => {
    const m = retargetManifest({ run: 'a', scenes: [], renderers: [], cells: [{ scene: 's', renderer: 'r', image: '1-run/x.pfm', thumb: '1-run/x.png' }] }, 3, 'b');
    expect(m.run).toBe('b');
    expect(m.cells[0]).toMatchObject({ image: '3-run/x.pfm', thumb: '3-run/x.png' });
  });
});
