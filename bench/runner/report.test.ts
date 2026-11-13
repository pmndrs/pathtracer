import { describe, expect, it } from 'vitest';
import type { Image } from '../src/adapter.ts';
import type { RunRecord, UploadRecord } from './contract.ts';
import { buildReportManifest, pickReference, slotFor, type ReportIo } from './report.ts';

const flat = (value: number, width = 4, height = 4): Image => ({ width, height, channels: 3, data: new Float32Array(width * height * 3).fill(value) });

function upload(slot: string, name: string): UploadRecord {
  return { name, width: 4, height: 4, channels: 3, token: null, slot, pfm: `bench/results/incoming/RUN/${slot}/${name}.pfm`, bytes: 0, nonFinite: 0, receivedAt: '' };
}

function run(index: number, variant: string, result: RunRecord['result'], outcome: RunRecord['outcome'] = 'pass'): RunRecord {
  return { index, variant, args: {}, outcome, reasons: [], durationMs: 1, result };
}

const base = { runId: 'RUN', page: 'competitor', manifestDir: 'bench/results/incoming/RUN', referencesDir: 'bench/references', flip: false };

function io(refs: Record<string, string[]>, images: Record<string, Image>): ReportIo {
  return { listReferences: (scene) => refs[scene] ?? [], readImage: (path) => images[path] ?? (() => { throw new Error(`no ${path}`); })() };
}

describe('slotFor', () => {
  it('matches the runner slot names', () => {
    expect(slotFor(1, '-')).toBe('1-run');
    expect(slotFor(3, 'B')).toBe('3-B');
  });
});

describe('pickReference', () => {
  it('prefers Mitsuba llvm_ad_rgb at the highest spp, then pbrt, then anything', () => {
    expect(pickReference(['pbrt-v4-cpu-4096spp.pfm', 'mitsuba3-llvm_ad_rgb-1024spp.pfm', 'mitsuba3-llvm_ad_rgb-4096spp.pfm', 'mitsuba3-llvm_ad_rgb-4096spp.json'])).toBe('mitsuba3-llvm_ad_rgb-4096spp.pfm');
    expect(pickReference(['pbrt-v4-cpu-4096spp.pfm', 'mitsuba3-scalar_spectral-512spp.pfm'])).toBe('pbrt-v4-cpu-4096spp.pfm');
    expect(pickReference(['other-8spp.pfm', 'other-64spp.pfm'])).toBe('other-64spp.pfm');
    expect(pickReference(['readme.json'])).toBeNull();
  });
});

describe('buildReportManifest', () => {
  const refName = 'mitsuba3-llvm_ad_rgb-4096spp.pfm';

  it('is null without any cell', () => {
    expect(buildReportManifest({ ...base, runs: [], uploads: [] }, io({}, {}))).toBeNull();
    expect(buildReportManifest({ ...base, runs: [run(1, '-', null, 'fail')], uploads: [] }, io({}, {}))).toBeNull();
  });

  it('builds a scene x renderer grid with paths relative to the manifest and metrics against the reference', () => {
    const uploads = [upload('1-A', 'x__cornell'), upload('1-A', 'x__cornell__spp4'), upload('2-B', 'y__cornell')];
    const runs = [
      run(1, 'A', { scene: 'cornell', renderer: 'x', metrics: { spp: 8, wall_ms: 12, ms_per_spp: 1.5 }, notes: { image: 'x__cornell' } }),
      run(2, 'B', { scene: 'cornell', renderer: 'y', metrics: { spp: 8 } }),
    ];
    const manifest = buildReportManifest(
      { ...base, runs, uploads },
      io({ cornell: [refName, 'pbrt-v4-cpu-4096spp.pfm'] }, { [`bench/results/incoming/RUN/1-A/x__cornell.pfm`]: flat(0.5), [`bench/results/incoming/RUN/2-B/y__cornell.pfm`]: flat(0.5), [`bench/references/cornell/${refName}`]: flat(0.5) }),
    )!;
    expect(manifest.scenes).toEqual([{ id: 'cornell', name: 'cornell', reference: `../../../references/cornell/${refName}` }]);
    expect(manifest.renderers.map((r) => r.id)).toEqual(['x', 'y']);
    expect(manifest.cells[0]).toMatchObject({ scene: 'cornell', renderer: 'x', image: '1-A/x__cornell.pfm', metrics: { spp: 8, wallMs: 12, msPerSpp: 1.5, relMse: 0 } });
    expect(manifest.cells[1]?.image).toBe('2-B/y__cornell.pfm');
  });

  it('drops metrics when the reference size differs, keeps the cell', () => {
    const uploads = [upload('1-run', 'x__s')];
    const manifest = buildReportManifest(
      { ...base, runs: [run(1, '-', { scene: 's', renderer: 'x', metrics: { spp: 1 } })], uploads },
      io({ s: [refName] }, { 'bench/results/incoming/RUN/1-run/x__s.pfm': flat(1, 4, 4), [`bench/references/s/${refName}`]: flat(1, 8, 8) }),
    )!;
    expect(manifest.cells[0]?.metrics).toEqual({ spp: 1 });
  });

  it('turns a skipped run into a greyed cell on the reference image', () => {
    const manifest = buildReportManifest(
      { ...base, runs: [run(1, '-', { scene: 's', renderer: 'x', metrics: { skipped: 1 }, notes: { skipped: 'no conductors' } })], uploads: [] },
      io({ s: [refName] }, {}),
    )!;
    expect(manifest.cells[0]).toMatchObject({ scene: 's', renderer: 'x', skipped: 'no conductors', image: `../../../references/s/${refName}` });
  });

  it('keeps the first of a repeated cell, and gives an A/B variant of the same renderer its own column', () => {
    const uploads = [upload('1-run', 'a'), upload('2-run', 'a'), upload('3-A', 'a'), upload('4-B', 'a')];
    const result = { scene: 's', renderer: 'x', metrics: {} };
    const manifest = buildReportManifest({ ...base, runs: [run(1, '-', result), run(2, '-', result), run(3, 'A', result), run(4, 'B', result)], uploads }, io({}, {}))!;
    expect(manifest.cells.map((c) => [c.renderer, c.image])).toEqual([['x', '1-run/a.pfm'], ['x (A)', '3-A/a.pfm'], ['x (B)', '4-B/a.pfm']]);
  });

  it('shows a page that reports no scene or renderer as scene "-" and renderer = page', () => {
    const manifest = buildReportManifest({ ...base, page: 'smoke', runs: [run(1, '-', { metrics: {} })], uploads: [upload('1-run', 'gradient')] }, io({}, {}))!;
    expect(manifest.cells[0]).toMatchObject({ scene: '-', renderer: 'smoke', image: '1-run/gradient.pfm' });
  });
});
