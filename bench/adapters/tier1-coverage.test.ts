import { describe, expect, it } from 'vitest';
import { AGREE_A, ROWS, SENSITIVITY, judgeCell, nativeOutcome, renderMarkdown } from './tier1-coverage.ts';
import type { CoverageJson, NativeProbeResult } from './tier1-coverage.ts';
import { probeNative } from './tier1-matrix.ts';

describe('judgeCell', () => {
  it('missing when the ablation is within the noise', () => {
    const v = judgeCell({ refClass: 'A', ablation: 5e-4, noise: 1e-4, refs: [] });
    expect(v.status).toBe('missing');
    expect(v.ratio).toBeCloseTo(5);
  });
  it('full at exactly the sensitivity ratio without a reference (class C)', () => {
    expect(judgeCell({ refClass: 'C', ablation: SENSITIVITY * 1e-4, noise: 1e-4, refs: [{ name: 'pbrt', relMSE: 1, floor: 1e-3 }] }).status).toBe('full');
  });
  it('class A: judged on the excess over own noise, not the raw relMSE', () => {
    const floor = 1e-3;
    const noise = 5e-3;
    // relMSE 6e-3 = noise 5e-3 + 1e-3 excess <= 1.5 x floor: full.
    expect(judgeCell({ refClass: 'A', ablation: 1, noise, refs: [{ name: 'pbrt', relMSE: noise + floor, floor }] }).status).toBe('full');
    // excess just above 1.5 x floor: approx.
    expect(judgeCell({ refClass: 'A', ablation: 1, noise, refs: [{ name: 'pbrt', relMSE: noise + AGREE_A * floor * 1.01, floor }] }).status).toBe('approx');
  });
  it('class B has the wider bound; every judging reference must agree', () => {
    const refs = [{ name: 'pbrt', relMSE: 3e-3, floor: 1e-3 }, { name: 'mitsuba', relMSE: 9e-3, floor: 1e-3 }];
    expect(judgeCell({ refClass: 'B', ablation: 1, noise: 0, refs: refs.slice(0, 1) }).status).toBe('full');
    expect(judgeCell({ refClass: 'B', ablation: 1, noise: 0, refs }).status).toBe('approx');
  });
  it('zero noise and zero ablation is missing, not NaN', () => {
    expect(judgeCell({ refClass: 'A', ablation: 0, noise: 0, refs: [] }).status).toBe('missing');
  });
});

describe('nativeOutcome', () => {
  const run = (over: Partial<NativeProbeResult>): NativeProbeResult => ({ ranStrict: true, ranCoverage: false, cls: 'A', notes: [], ...over });
  it('class A with no note is exact', () => expect(nativeOutcome(run({})).status).toBe('full'));
  it('class B is approximate', () => expect(nativeOutcome(run({ cls: 'B' })).status).toBe('approx'));
  it('an unverified-tangent note downgrades class A', () => expect(nativeOutcome(run({ notes: ['x: the GGX tangent axis is dp/du (unverified, P2.26)'] })).status).toBe('approx'));
  it('a coverage-only run that approximates through the IOR is coverage', () => {
    const r = nativeOutcome(run({ ranStrict: false, ranCoverage: true, cls: 'C', notes: ['coverage: pbrt: material m: specular_weight/specular_color scale F0 only through an IOR of 1.2 (x)'] }));
    expect(r.outcome).toBe('coverage');
  });
  it('a coverage-only run that drops the feature is none', () => {
    expect(nativeOutcome(run({ ranStrict: false, ranCoverage: true, cls: 'C', notes: ['coverage: pbrt: material m: class C: fuzz (sheen) has no reference; dropped or approximated'] })).status).toBe('missing');
  });
  it('an exporter that cannot even do coverage is none', () => expect(nativeOutcome(run({ ranStrict: false, ranCoverage: false })).outcome).toBe('none'));
});

describe('rows and exporters', () => {
  it('every row names a scene with an ablation, or none of them', () => {
    for (const r of ROWS) expect(!!r.scene).toBe(!!r.ablation);
    expect(new Set(ROWS.map((r) => r.id)).size).toBe(ROWS.length);
  });
  it('the exporters give the expected native outcomes on the wave-1 probes', () => {
    const probe = (id: string): Record<string, unknown> => ROWS.find((r) => r.id === id)!.probe;
    expect(nativeOutcome(probeNative('pbrt', probe('core-metal'))).status).toBe('full');
    expect(nativeOutcome(probeNative('mitsuba', probe('transmission-thin'))).status).toBe('full');
    expect(nativeOutcome(probeNative('pbrt', probe('sheen'))).status).toBe('missing');
    expect(nativeOutcome(probeNative('mitsuba', probe('specular-lt1'))).status).toBe('approx');
    expect(nativeOutcome(probeNative('mitsuba', probe('dispersion'))).status).toBe('missing');
    expect(nativeOutcome(probeNative('pbrt', probe('dispersion'))).status).toBe('approx');
  });
});

describe('renderMarkdown', () => {
  it('prints a table with every renderer column', () => {
    const cell = { status: 'full' as const, symbol: '●', notes: ['n'], reportRef: null, ratio: 42 };
    const j: CoverageJson = {
      schema: 1, generated: '2026-11-14', report: 'x', spp: 1024, halfSpp: 512, thresholds: { sensitivity: 10, agreeA: 1.5, agreeB: 5 },
      conditions: { loadedRuns: 0, runs: 0, versions: {} }, legend: {},
      rows: [{ id: 'r', label: 'row', extension: 'core', refClass: 'A', scene: 's', ablation: 'a', stripped: 's--no-a', refs: ['pbrt'], noScene: false, cells: { ours: cell, tgpt: cell, mutracer: cell, pbrt: cell, mitsuba: cell } }],
    };
    expect(renderMarkdown(j)).toContain('| row | A | ● 42x | ● 42x | ● 42x | ● 42x | ● 42x |');
  });
});
