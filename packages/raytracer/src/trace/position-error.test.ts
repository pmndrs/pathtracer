// V302 / P5.v19: RtSurface.positionError, the pbrt-v4 style conservative bound of the decoded world position (zero without decoded positions).
import { describe, expect, it } from 'vitest';
import { buildCaps } from '../caps/index.ts';
import { rtGamma } from './wgsl-source.ts';
import { rtModuleWgsl } from './modules.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());

describe('surface position error bound', () => {
  it('gamma(n) = n eps / (1 - n eps), eps = 2^-24', () => {
    expect(rtGamma(1)).toBeCloseTo(2 ** -24, 12);
    expect(rtGamma(7)).toBeCloseTo((7 * 2 ** -24) / (1 - 7 * 2 ** -24), 15);
    expect(Math.fround(rtGamma(7))).toBeGreaterThan(4.17e-7);
  });
  it('is filled from the vertex magnitudes only in the position block, as one isotropic value', () => {
    const text = rtModuleWgsl(caps, { scene: { slot: { group: 2, firstBinding: 0 }, layout: { normals: true, uvs: false, positions: true } } });
    expect(text).toContain('positionError: vec3f');
    expect(text).toContain('s.positionError = vec3f(0.0);');
    expect(text).toContain('let eo = RT_GAMMA7 * (abs(c0) + abs(c1) + abs(c2));');
    expect(text).toContain('s.positionError = vec3f(max(pe.x, max(pe.y, pe.z)));');
  });
  it("positions: 'ray' (no vertex decode) reports the scene-wide bound 4 gamma(7) R from the scene header (V1.7); decoded layouts do not carry it", () => {
    const ray = rtModuleWgsl(caps, { scene: { slot: { group: 2, firstBinding: 0 }, layout: { normals: true, uvs: false, positions: false } } });
    expect(ray).toContain('const RT_SCENE_ERR_SCALE: f32 = ');
    expect(ray).toContain('s.positionError = vec3f(RT_SCENE_ERR_SCALE * bitcast<f32>(rtSTable(3u).y));');
    const decoded = rtModuleWgsl(caps, { scene: { slot: { group: 2, firstBinding: 0 }, layout: { normals: true, uvs: false, positions: true } } });
    expect(decoded).not.toContain('RT_SCENE_ERR_SCALE');
  });
});
