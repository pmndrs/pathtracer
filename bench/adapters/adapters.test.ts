import { describe, expect, it } from 'vitest';
import type { ReportManifest } from '../viewer/src/report.ts';
import { getScene } from '../scenes/index.ts';
import { getTier0Scene } from '../scenes/tier0/index.ts';
import { mergeManifests } from './matrix.ts';
import { ADAPTER_IDS, unsupportedReason } from './registry.ts';

const exact = { materials: ['diffuse', 'dielectric', 'black'] as const, emissiveMeshes: true, directionalLight: true, environment: ['uniform', 'hdri'] as const, rawOutput: true };

describe('unsupportedReason', () => {
  it('lists the conductors a renderer cannot do exactly and passes the rest', () => {
    expect(unsupportedReason(exact, getTier0Scene('furnace-diffuse'))).toBeNull();
    expect(unsupportedReason(exact, getTier0Scene('cornell'))).toBeNull();
    expect(unsupportedReason(exact, getTier0Scene('furnace-conductor'))).toMatch(/conductor/);
    expect(unsupportedReason(exact, getTier0Scene('env-sphere'))).toMatch(/copper/);
  });

  it('accepts the Tier 2 Sponza scene (sun + uniform sky, one diffuse) and skips it where directional lights are missing', () => {
    expect(unsupportedReason(exact, getScene('sponza-white'))).toBeNull();
    expect(unsupportedReason({ ...exact, directionalLight: false }, getScene('sponza-white'))).toMatch(/directional/);
  });

  it('skips environments and emitters the renderer does not list', () => {
    expect(unsupportedReason({ ...exact, environment: [] }, getTier0Scene('furnace-diffuse'))).toMatch(/uniform environment/);
    expect(unsupportedReason({ ...exact, emissiveMeshes: false }, getTier0Scene('cornell'))).toMatch(/emissive/);
  });
});

describe('mergeManifests', () => {
  it('rewrites paths relative to the merged manifest and dedupes scenes and renderers', () => {
    const part = (renderer: string, dir: string): { manifest: ReportManifest; dir: string } => ({
      dir,
      manifest: {
        scenes: [{ id: 's', reference: '../../../references/s/ref.pfm' }],
        renderers: [{ id: renderer }],
        cells: [{ scene: 's', renderer, image: '1-run/a.pfm', metrics: { relMse: 1 } }],
      },
    });
    const merged = mergeManifests([part('a', '/r/bench/results/incoming/A'), part('b', '/r/bench/results/incoming/B')], '/r/bench/results/reports/x', 'x');
    expect(merged.scenes).toEqual([{ id: 's', reference: '../../../references/s/ref.pfm' }]);
    expect(merged.renderers.map((r) => r.id)).toEqual(['a', 'b']);
    expect(merged.cells.map((c) => c.image)).toEqual(['../../incoming/A/1-run/a.pfm', '../../incoming/B/1-run/a.pfm']);
  });

  it('knows the three adapters', () => {
    expect([...ADAPTER_IDS]).toEqual(['tgpt-wavefront', 'tgpt-megakernel', 'mutracer']);
  });
});
