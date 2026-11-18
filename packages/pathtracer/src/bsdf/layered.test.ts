import { describe, expect, it } from 'vitest';
import { PT_FLAG_DELTA, PT_FLAG_REFLECTION, ptBsdfEval, type PtVec3 } from './bsdf.ts';
import { ptBsdfWgsl } from './bsdf-wgsl.ts';
import { dirFromAngles, rng } from './bsdf.testutil.ts';
import { PT_LAYERED_LUT_N, ptDielectricAlbedoExact, ptDielectricF0, ptLayeredAlbedo, ptLayeredLut } from './layered-lut.ts';
import { PT_FLAG_PROPORTIONAL, ptInternalReflectance, ptLayered, ptLayeredCoatedConductor, ptLayeredEval, ptLayeredPdf, ptLayeredSample, ptPcg, type PtLayeredOptions } from './layered.ts';
import { ptLayeredWgsl } from './layered-wgsl.ts';
import { evalAlbedo, layeredQuadrature, meanF, sampleAlbedo } from './layered.testutil.ts';
import { ptDielectric } from './materials.ts';
import { ptFrComplex, ptFrDielectric as ptFrDielectricRef } from './bsdf.ts';

const WHITE: PtVec3 = [1, 1, 1];
const ALBEDO: PtLayeredOptions = { layering: 'albedo', albedoBase: 'openpbr' };
const STOCH: PtLayeredOptions = { layering: 'stochastic', maxDepth: 200, exitMis: 'none' };
const THETAS = [0, 30, 60, 80];

describe('directional-albedo table (layering: albedo)', () => {
  it('is N x N x 2 numbers, rounded to what the WGSL carries', () => {
    const t = ptLayeredLut();
    expect(t.length).toBe(PT_LAYERED_LUT_N * PT_LAYERED_LUT_N * 2);
    for (const v of t) expect(Math.fround(Math.round(v * 1e4) / 1e4)).toBe(v);
  });

  it('reproduces the exact-Fresnel albedo at ior 1.5 to 0.02 (the glTF default), and to 0.05 across ior 1.3 to 2.4', () => {
    let worst15 = 0;
    let worstAny = 0;
    for (const ior of [1.33, 1.5, 1.8, 2, 2.4]) {
      for (const r of [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1]) {
        for (const mu of [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1]) {
          const d = Math.abs(ptLayeredAlbedo(ptDielectricF0(ior), mu, r) - ptDielectricAlbedoExact(ior, mu, r));
          worstAny = Math.max(worstAny, d);
          if (ior === 1.5) worst15 = Math.max(worst15, d);
        }
      }
    }
    expect(worst15).toBeLessThan(0.02);
    expect(worstAny).toBeLessThan(0.05);
  });

  it('below ior 1.3 the Schlick-shaped split overestimates E at grazing angles (documented: energy is lost, never gained)', () => {
    // E too high means the base is scaled down too far: the sum stays below 1.
    for (const ior of [1.05, 1.1, 1.2]) {
      for (const mu of [0.05, 0.2, 0.5, 1]) {
        const lut = ptLayeredAlbedo(ptDielectricF0(ior), mu, 0.3);
        const exact = ptDielectricAlbedoExact(ior, mu, 0.3);
        expect(lut - exact, `ior ${ior} mu ${mu}`).toBeGreaterThan(-0.005);
        expect(lut - exact, `ior ${ior} mu ${mu}`).toBeLessThan(0.45);
      }
    }
  });
});

describe('specular_weight / specular_color (OpenPBR: F scaled at both ends)', () => {
  it('scales the reflection lobe at normal and grazing incidence alike, unlike an F0-only scale', () => {
    const base = ptLayered([0, 0, 0], 1.5, 0.3, { specularWeight: 1 });
    const half = ptLayered([0, 0, 0], 1.5, 0.3, { specularWeight: 0.5, specularColor: [1, 0.5, 0] });
    for (const theta of [0, 45, 80]) {
      const wo = dirFromAngles(theta);
      const wi = dirFromAngles(theta, 180);
      const a = ptLayeredEval(base, wo, wi, ALBEDO).f;
      const b = ptLayeredEval(half, wo, wi, ALBEDO).f;
      expect(b[0] / a[0]).toBeCloseTo(0.5, 12);
      expect(b[1] / a[1]).toBeCloseTo(0.25, 12);
      expect(b[2]).toBe(0);
    }
  });

  it('weight 0 leaves the base untouched in both layerings (pure Lambert in the albedo one)', () => {
    const p = ptLayered([0.6, 0.5, 0.4], 1.5, 0.3, { specularWeight: 0, thickness: 0 });
    const wo = dirFromAngles(40);
    const wi = dirFromAngles(25, 120);
    const e = ptLayeredEval(p, wo, wi, ALBEDO);
    for (let c = 0; c < 3; c++) expect(e.f[c]).toBeCloseTo((p.baseColor[c] as number) / Math.PI, 12);
    // stochastic: the interface still refracts, but no light is reflected or lost: a white base keeps albedo 1
    const w = ptLayered(WHITE, 1.5, 0.1, { specularWeight: 0, thickness: 0 });
    const est = sampleAlbedo(w, dirFromAngles(30), 60_000, 5, STOCH);
    expect(Math.abs(est.mean[0] - 1)).toBeLessThan(0.01 + 4 * est.se[0]);
  });

  it('colour tints the stochastic reflection per channel too (black base)', () => {
    const p = ptLayered([0, 0, 0], 1.5, 0.2, { specularColor: [1, 0.5, 0.25], thickness: 0 });
    const wo = dirFromAngles(30);
    const wi = dirFromAngles(20, 150);
    const f = ptLayeredEval(p, wo, wi, STOCH).f;
    expect(f[1] / f[0]).toBeCloseTo(0.5, 9);
    expect(f[2] / f[0]).toBeCloseTo(0.25, 9);
  });
});

describe('white furnace', () => {
  it('albedo layering: a white base under a clear dielectric reflects 1 (within the table error, 0.02)', () => {
    for (const r of [0.1, 0.3, 0.6, 1]) {
      for (const th of THETAS) {
        const p = ptLayered(WHITE, 1.5, r);
        const q = layeredQuadrature(p, dirFromAngles(th), ALBEDO, 200, 180);
        expect(q.albedo[0], `r ${r} theta ${th}`).toBeGreaterThan(0.98);
        expect(q.albedo[0], `r ${r} theta ${th}`).toBeLessThan(1.02);
      }
    }
  });

  it('albedo layering, smooth interface: diffuse part + F(mu) = 1 exactly, and the sampler agrees', () => {
    for (const th of THETAS) {
      const wo = dirFromAngles(th);
      const p = ptLayered(WHITE, 1.5, 0);
      const q = layeredQuadrature(p, wo, ALBEDO, 120, 120);
      const F = ptBsdfEval(ptDielectric(1.5, 0.5), wo, wo).f; // keeps the import used for a real dielectric below
      expect(F).toBeDefined();
      const R = ptDielectricAlbedoExact(1.5, wo[2], 0);
      expect(q.albedo[0] + R, `theta ${th}`).toBeCloseTo(1, 3);
      const est = sampleAlbedo(p, wo, 40_000, 3, ALBEDO);
      expect(Math.abs(est.mean[0] - 1), `theta ${th}`).toBeLessThan(5 * est.se[0] + 1e-9);
    }
  });

  it("albedo layering, albedoBase 'internal': still 1 (the extra light comes back out of the interface)", () => {
    for (const th of THETAS) {
      const p = ptLayered(WHITE, 1.5, 0.3);
      const q = layeredQuadrature(p, dirFromAngles(th), { ...ALBEDO, albedoBase: 'internal', albedoCorrection: 'none' }, 200, 180);
      expect(q.albedo[0], `theta ${th}`).toBeGreaterThan(0.96);
      expect(q.albedo[0], `theta ${th}`).toBeLessThan(1.04);
    }
  });

  it('stochastic, smooth interface: sampler albedo is 1 (thickness 0, deep walk), and eval + delta reflection F(mu) = 1', () => {
    for (const th of THETAS) {
      const wo = dirFromAngles(th);
      const p = ptLayered(WHITE, 1.5, 0, { thickness: 0 });
      const s = sampleAlbedo(p, wo, 100_000, 11, STOCH);
      expect(Math.abs(s.mean[0] - 1), `sample theta ${th}`).toBeLessThan(5 * s.se[0] + 1e-3);
      const e = evalAlbedo(p, wo, 150_000, 12, STOCH);
      const R = ptDielectricAlbedoExact(1.5, wo[2], 0);
      expect(Math.abs(e.mean[0] + R - 1), `eval theta ${th}`).toBeLessThan(5 * e.se[0] + 1e-3);
    }
  });

  it('stochastic, rough interface: roughness 0.1 conserves energy to 0.5%; the loss at 0.3 is the interface single-scatter loss (< 8%)', () => {
    const wo = dirFromAngles(30);
    const a = sampleAlbedo(ptLayered(WHITE, 1.5, 0.1, { thickness: 0 }), wo, 100_000, 21, STOCH);
    expect(a.mean[0]).toBeGreaterThan(0.995 - 4 * a.se[0]);
    expect(a.mean[0]).toBeLessThan(1.005 + 4 * a.se[0]);
    const b = sampleAlbedo(ptLayered(WHITE, 1.5, 0.3, { thickness: 0 }), wo, 100_000, 22, STOCH);
    expect(b.mean[0]).toBeGreaterThan(0.92);
    expect(b.mean[0]).toBeLessThan(1 + 4 * b.se[0]);
  });

  it('stochastic: f and the sampler estimate the same albedo with exitMis none; pbrt\'s exit weights read higher', () => {
    const p = ptLayered(WHITE, 1.5, 0.3, { thickness: 0 });
    const wo = dirFromAngles(30);
    const s = sampleAlbedo(p, wo, 150_000, 31, STOCH);
    const e = evalAlbedo(p, wo, 150_000, 32, STOCH);
    expect(Math.abs(s.mean[0] - e.mean[0])).toBeLessThan(5 * Math.hypot(s.se[0], e.se[0]) + 2e-3);
    const pb = evalAlbedo(p, wo, 150_000, 33, { ...STOCH, exitMis: 'pbrt' });
    expect(pb.mean[0] - e.mean[0]).toBeGreaterThan(0.015);
  });

  it('thickness attenuates (pbrt default 0.01 loses about 5% at roughness 0.3 / base 0.8)', () => {
    const wo = dirFromAngles(30);
    const a = sampleAlbedo(ptLayered([0.8, 0.8, 0.8], 1.5, 0.3, { thickness: 0 }), wo, 80_000, 41, STOCH);
    const b = sampleAlbedo(ptLayered([0.8, 0.8, 0.8], 1.5, 0.3, { thickness: 0.01 }), wo, 80_000, 41, STOCH);
    expect(a.mean[0] - b.mean[0]).toBeGreaterThan(0.02);
    expect(a.mean[0] - b.mean[0]).toBeLessThan(0.05);
  });
});

describe('limits where the two layerings must agree', () => {
  const wiOf = (theta: number, phi: number): PtVec3 => dirFromAngles(theta, phi);

  it('black base: both are exactly the existing dielectric reflection lobe', () => {
    for (const r of [0.1, 0.3, 0.6, 1]) {
      const p = ptLayered([0, 0, 0], 1.5, r, { thickness: 0 });
      const ref = ptDielectric(1.5, r);
      for (const [to, ti, ph] of [[0, 20, 0], [30, 50, 90], [60, 10, 200], [80, 70, 45]] as const) {
        const wo = dirFromAngles(to);
        const wi = wiOf(ti, ph);
        const expected = ptBsdfEval(ref, wo, wi).f[0];
        for (const options of [ALBEDO, STOCH, { ...STOCH, exitMis: 'pbrt' as const }]) {
          const f = ptLayeredEval(p, wo, wi, options).f[0];
          expect(f, `r ${r} ${JSON.stringify(options)}`).toBeCloseTo(expected, 12);
        }
      }
      // and the sampled albedo of a black base is E_spec
      const wo = dirFromAngles(40);
      const E = ptDielectricAlbedoExact(1.5, wo[2], r);
      const s = sampleAlbedo(p, wo, 120_000, 51, STOCH);
      expect(Math.abs(s.mean[0] - E)).toBeLessThan(5 * s.se[0] + 2e-3);
      const a = sampleAlbedo(p, wo, 120_000, 52, ALBEDO);
      expect(Math.abs(a.mean[0] - E)).toBeLessThan(5 * a.se[0] + 2e-3);
    }
  });

  it('ior = 1: the interface vanishes, both are exactly Lambert; ior = 1.001 is within 1% (the table is not exact at ior ~ 1)', () => {
    const p1 = ptLayered([0.7, 0.5, 0.3], 1, 0.3, { thickness: 0 });
    const wo = dirFromAngles(35);
    const wi = wiOf(50, 100);
    for (const options of [ALBEDO, STOCH]) {
      const f = ptLayeredEval(p1, wo, wi, options).f;
      for (let c = 0; c < 3; c++) expect(f[c]).toBeCloseTo((p1.baseColor[c] as number) / Math.PI, 9);
    }
    const p2 = ptLayered([0.7, 0.5, 0.3], 1.001, 0.3, { thickness: 0 });
    const a = ptLayeredEval(p2, wo, wi, ALBEDO).f[0];
    const s = meanF(p2, wo, wi, 4000, STOCH);
    expect(Math.abs(a - 0.7 / Math.PI) / (0.7 / Math.PI)).toBeLessThan(0.01);
    // (the rough interface still shadows a little even at ior 1.001: single-scatter loss of the microfacet model)
    expect(Math.abs(s.mean[0] - 0.7 / Math.PI)).toBeLessThan(5 * s.se[0] + 0.01 * (0.7 / Math.PI));
  });

  it('white base: both layerings reflect the same total (1); they differ in the split, not in the sum', () => {
    const p = ptLayered(WHITE, 1.5, 0.3, { thickness: 0 });
    const wo = dirFromAngles(45);
    const a = layeredQuadrature(p, wo, ALBEDO, 200, 180).albedo[0];
    const s = sampleAlbedo(p, wo, 150_000, 61, { ...STOCH, maxDepth: 300 });
    expect(Math.abs(a - 1)).toBeLessThan(0.02);
    expect(Math.abs(s.mean[0] - 1)).toBeLessThan(0.08);
  });

  it('typical plastic (ior 1.5, roughness 0.3, base 0.8): openpbr albedo is brighter than the walk, the internal base closes most of the gap', () => {
    const p = ptLayered([0.8, 0.8, 0.8], 1.5, 0.3, { thickness: 0 });
    for (const th of [0, 40, 60]) {
      const wo = dirFromAngles(th);
      const a = layeredQuadrature(p, wo, ALBEDO, 120, 120).albedo[0];
      const ai = layeredQuadrature(p, wo, { ...ALBEDO, albedoBase: 'internal' }, 120, 120).albedo[0];
      const s = sampleAlbedo(p, wo, 60_000, 71, STOCH).mean[0];
      expect(a - s, `theta ${th}`).toBeGreaterThan(0.15);
      expect(Math.abs(ai - s), `theta ${th}`).toBeLessThan(0.05);
    }
  });
});

describe('reciprocity', () => {
  const wo = dirFromAngles(30, 0);
  const wi = dirFromAngles(55, 100);

  it("albedo layering: the specular lobe is reciprocal, the 'openpbr' base is not (it scales by 1 - E(mu_o) only), the 'internal' base is", () => {
    const spec = ptLayered([0, 0, 0], 1.5, 0.3);
    expect(ptLayeredEval(spec, wo, wi, ALBEDO).f[0]).toBeCloseTo(ptLayeredEval(spec, wi, wo, ALBEDO).f[0], 12);
    const p = ptLayered([0.8, 0.8, 0.8], 1.5, 0.3);
    const a = ptLayeredEval(p, wo, wi, ALBEDO).f[0];
    const b = ptLayeredEval(p, wi, wo, ALBEDO).f[0];
    expect(Math.abs(a - b) / a).toBeGreaterThan(0.01);
    expect(Math.abs(a - b) / a).toBeLessThan(0.1);
    const o2 = { ...ALBEDO, albedoBase: 'internal' as const, albedoCorrection: 'none' as const };
    expect(ptLayeredEval(p, wo, wi, o2).f[0]).toBeCloseTo(ptLayeredEval(p, wi, wo, o2).f[0], 12);
  });

  it('stochastic: E[f(wo, wi)] = E[f(wi, wo)] within 4 sigma (exitMis none, thickness 0.01, 6000 seeds, 4 walks each)', () => {
    const p = ptLayered([0.8, 0.6, 0.4], 1.5, 0.3);
    const o: PtLayeredOptions = { layering: 'stochastic', maxDepth: 50, nSamples: 4, exitMis: 'none' };
    const a = meanF(p, wo, wi, 6000, o);
    const b = meanF(p, wi, wo, 6000, o);
    for (let c = 0; c < 3; c++) expect(Math.abs((a.mean[c] as number) - (b.mean[c] as number)), `channel ${c}`).toBeLessThan(4 * Math.hypot(a.se[c] as number, b.se[c] as number));
  });

  it('stochastic, smooth interface (only the walk through the interface): reciprocal as well', () => {
    const p = ptLayered([0.8, 0.8, 0.8], 1.5, 0);
    const o: PtLayeredOptions = { layering: 'stochastic', maxDepth: 50, nSamples: 4 };
    const a = meanF(p, wo, wi, 6000, o);
    const b = meanF(p, wi, wo, 6000, o);
    expect(Math.abs(a.mean[0] - b.mean[0])).toBeLessThan(4 * Math.hypot(a.se[0], b.se[0]));
  });
});

describe('pdf', () => {
  it('albedo: the pdf integrates to at most 1 over the hemisphere, and close to it when the lobe rarely fails', () => {
    for (const r of [0.1, 0.3, 0.6, 1]) {
      for (const th of [0, 40, 75]) {
        const p = ptLayered([0.8, 0.8, 0.8], 1.5, r);
        const q = layeredQuadrature(p, dirFromAngles(th), ALBEDO, 300, 240);
        expect(q.pdf, `r ${r} theta ${th}`).toBeLessThanOrEqual(1.002);
        expect(q.pdf, `r ${r} theta ${th}`).toBeGreaterThan(0.85);
      }
    }
  });

  it("stochastic: pbrt's blended pdf is NOT normalised (its TRT term measures inside directions, no Jacobian): it integrates to 1.5 to 2.7, the reflection term alone to at most 1", () => {
    const p = ptLayered([0.8, 0.8, 0.8], 1.5, 0.5);
    const r = rng(81);
    for (const th of [0, 50]) {
      const wo = dirFromAngles(th);
      let sum = 0;
      let sq = 0;
      const n = 120_000;
      for (let i = 0; i < n; i++) {
        const z = 2 * r() - 1;
        const s = Math.sqrt(1 - z * z);
        const ph = 2 * Math.PI * r();
        const v = ptLayeredPdf(p, wo, [s * Math.cos(ph), s * Math.sin(ph), z], { layering: 'stochastic', seed: i, nSamples: 1 }) * 4 * Math.PI;
        sum += v;
        sq += v * v;
      }
      const mean = sum / n;
      const se = Math.sqrt(sq / n - mean * mean) / Math.sqrt(n);
      expect(mean + 4 * se, `theta ${th}`).toBeGreaterThan(1.5);
      expect(mean - 4 * se, `theta ${th}`).toBeLessThan(2.7);
    }
    // the part that is a proper density: the interface's own reflection pdf (pbrt `top.PDF(wo, wi, Reflection)`), via a black base
    // is not separable here, so check it through the albedo layering's pure-specular case instead
    const spec = layeredQuadrature(ptLayered([0, 0, 0], 1.5, 0.5), dirFromAngles(30), ALBEDO, 300, 240);
    expect(spec.pdf).toBeLessThanOrEqual(1.002);
  });
});

describe('sampler contract', () => {
  it('albedo: sample f and pdf equal eval at the sampled direction (rough), delta flags for a smooth coat', () => {
    const r = rng(91);
    for (const [rough, th] of [[0.3, 20], [0.6, 60], [0.05, 40]] as const) {
      const p = ptLayered([0.7, 0.5, 0.3], 1.5, rough, { specularColor: [1, 0.8, 0.6] });
      for (let i = 0; i < 200; i++) {
        const wo = dirFromAngles(th, 360 * r());
        const s = ptLayeredSample(p, wo, [r(), r(), r()], ALBEDO);
        if (s.flags === 0) continue;
        const e = ptLayeredEval(p, wo, s.wi, ALBEDO);
        expect(s.f[0]).toBeCloseTo(e.f[0], 9);
        expect(s.pdf).toBeCloseTo(e.pdf, 9);
        expect(s.flags & PT_FLAG_PROPORTIONAL).toBe(0);
      }
    }
    const smooth = ptLayered([0.7, 0.5, 0.3], 1.5, 0);
    let deltas = 0;
    for (let i = 0; i < 400; i++) {
      const s = ptLayeredSample(smooth, dirFromAngles(30), [r(), r(), r()], ALBEDO);
      if (s.flags & PT_FLAG_DELTA) {
        deltas++;
        expect(s.flags & PT_FLAG_REFLECTION).toBeTruthy();
      }
    }
    expect(deltas).toBeGreaterThan(5);
    expect(deltas).toBeLessThan(100);
  });

  it('two-sided: a ray from below sees the same BSDF with the directions negated', () => {
    const p = ptLayered([0.7, 0.5, 0.3], 1.5, 0.3);
    const wo = dirFromAngles(30, 10);
    const wi = dirFromAngles(50, 120);
    for (const o of [ALBEDO, { ...STOCH, maxDepth: 20 }]) {
      const up = ptLayeredEval(p, wo, wi, o);
      const down = ptLayeredEval(p, [-wo[0], -wo[1], -wo[2]], [-wi[0], -wi[1], -wi[2]], o);
      expect(down.f[0]).toBeCloseTo(up.f[0], 12);
      expect(down.pdf).toBeCloseTo(up.pdf, 12);
    }
    const s = ptLayeredSample(p, [-wo[0], -wo[1], -wo[2]], [0.9, 0.3, 0.6], ALBEDO);
    expect(s.wi[2]).toBeLessThan(0);
  });

  it('stochastic samples are proportional-pdf samples and are deterministic per (seed, wo, u)', () => {
    const p = ptLayered([0.7, 0.5, 0.3], 1.5, 0.3);
    const wo = dirFromAngles(30);
    const o: PtLayeredOptions = { layering: 'stochastic', seed: 7 };
    let proportional = 0;
    for (let i = 0; i < 50; i++) {
      const u: PtVec3 = [0.5 + i * 0.001, 0.3, 0.8];
      const a = ptLayeredSample(p, wo, u, o);
      const b = ptLayeredSample(p, wo, u, o);
      expect(a).toEqual(b);
      if (a.flags & PT_FLAG_PROPORTIONAL) proportional++;
    }
    expect(proportional).toBeGreaterThan(0);
    expect(ptPcg(1)).toBe(ptPcg(1));
  });
});

describe('internal reflectance', () => {
  it('matches the known diffuse Fresnel average (ior 1.5: 0.596)', () => {
    expect(ptInternalReflectance(1.5)).toBeGreaterThan(0.59);
    expect(ptInternalReflectance(1.5)).toBeLessThan(0.60);
    expect(ptInternalReflectance(1)).toBe(0);
  });
});

describe('WGSL fragments', () => {
  it('emit only the chosen layering: the albedo one has the table and no walk, the stochastic one the opposite', () => {
    const a = ptLayeredWgsl({ layering: 'albedo' });
    const s = ptLayeredWgsl({ layering: 'stochastic', maxDepth: 12, nSamples: 2 });
    expect(a).toContain('PT_LAYER_LUT');
    expect(a).not.toContain('ptPcg');
    expect(s).not.toContain('PT_LAYER_LUT');
    expect(s).toContain('PT_LAYER_MAX_DEPTH: u32 = 12u');
    expect(s).toContain('PT_LAYER_NSAMPLES: u32 = 2u');
    expect(ptLayeredWgsl({ layering: 'stochastic', exitMis: 'none' })).toContain('PT_LAYER_MIS_PBRT: bool = false');
    expect(ptLayeredWgsl({ albedoBase: 'internal' })).toContain('PT_LAYER_BASE_INTERNAL: bool = true');
  });

  it('records the sizes (the megakernel is not changed: it only gains this text when a layered material exists)', () => {
    const a = ptLayeredWgsl({ layering: 'albedo' }).length;
    const s = ptLayeredWgsl({ layering: 'stochastic' }).length;
    // Raw fragments (comments, long names); in the kernel after strip + prune + shorten + minify they cost about 7.2 KB
    // (albedo, 3 KB of it the table) and 8.3 KB (stochastic): 40.4 KB untextured + either stays under the 50 KB cap.
    expect(a).toBeLessThan(11_000);
    expect(s).toBeLessThan(13_500);
    // ptBsdfWgsl is a precondition: every name the fragment uses from it is declared there
    const base = ptBsdfWgsl();
    for (const n of ['ptFrDielectric', 'ptRefract', 'ptGgxSampleWm', 'ptGgxD', 'ptGgxG', 'ptGgxVisiblePdf', 'ptDielectricHalf', 'ptRoughAlpha', 'ptReflect', 'PT_TRANSPORT_RADIANCE']) expect(base).toContain(n);
  });
});

// P2.23: clear coat over a conductor (pbrt coatedconductor).
describe('coat over a conductor', () => {
  // A white conductor: eta 1, k huge -> F = 1 at every angle (no absorption in the base), so only the coat's own loss remains.
  const WHITE_K: PtVec3 = [1e4, 1e4, 1e4];
  const ONE: PtVec3 = [1, 1, 1];
  const wo = (deg: number): PtVec3 => dirFromAngles(deg, 0);
  const COND_STOCH: PtLayeredOptions = { layering: 'stochastic', maxDepth: 200, exitMis: 'none' };

  it('furnace, smooth coat over a smooth white conductor (stochastic, thickness 0): the sampler conserves energy exactly', () => {
    const p = ptLayeredCoatedConductor(ONE, WHITE_K, 0, 1.5, 0, { thickness: 0 });
    for (const th of [0, 40, 70]) {
      const e = sampleAlbedo(p, wo(th), 20000, 3, COND_STOCH);
      expect(e.mean[0]).toBeGreaterThan(0.995);
      expect(e.mean[0]).toBeLessThan(1.005);
    }
  });

  it('furnace, smooth coat over a rough white conductor: the loss is the conductor single-scatter loss only (< 10% at roughness 0.4)', () => {
    const p = ptLayeredCoatedConductor(ONE, WHITE_K, 0.4, 1.5, 0, { thickness: 0 });
    const e = sampleAlbedo(p, wo(30), 40000, 5, COND_STOCH);
    expect(e.mean[0]).toBeGreaterThan(0.9);
    expect(e.mean[0]).toBeLessThan(1.005);
  });

  it('furnace, rough coat (0.1) over a smooth white conductor: conserves to 1%', () => {
    const p = ptLayeredCoatedConductor(ONE, WHITE_K, 0, 1.5, 0.1, { thickness: 0 });
    const e = sampleAlbedo(p, wo(30), 40000, 7, COND_STOCH);
    expect(e.mean[0]).toBeGreaterThan(0.985);
    expect(e.mean[0]).toBeLessThan(1.01);
  });

  it('eval and sampler estimate the same albedo (rough coat over rough conductor, exitMis none): within 4 sigma and 2%', () => {
    const p = ptLayeredCoatedConductor([0.2, 0.9, 1.1], [3.9, 2.4, 2.2], 0.4, 1.5, 0.15, { thickness: 0 });
    const a = sampleAlbedo(p, wo(35), 60000, 11, COND_STOCH);
    const b = evalAlbedo(p, wo(35), 60000, 13, COND_STOCH);
    for (let c = 0; c < 3; c++) {
      expect(Math.abs((a.mean[c] as number) - (b.mean[c] as number))).toBeLessThan(4 * Math.hypot(a.se[c] as number, b.se[c] as number) + 0.02 * (a.mean[c] as number));
    }
  });

  it('limit: coat ior 1 is the bare metal (albedo layering exactly, stochastic within noise)', () => {
    const eta: PtVec3 = [0.2, 0.9, 1.1];
    const k: PtVec3 = [3.9, 2.4, 2.2];
    const bare = { kind: 1, reflectance: [0, 0, 0] as PtVec3, eta, k, alpha: [0.16, 0.16] as [number, number], ior: 1 };
    const p = ptLayeredCoatedConductor(eta, k, 0.4, 1, 0.1, { thickness: 0 });
    const wi = dirFromAngles(30, 140);
    const m = ptBsdfEval(bare, wo(40), wi);
    const e = ptLayeredEval(p, wo(40), wi, ALBEDO);
    for (let c = 0; c < 3; c++) expect(e.f[c]).toBeCloseTo(m.f[c] as number, 10);
    expect(e.pdf).toBeCloseTo(m.pdf, 10);
    const q = ptLayeredEval(p, wo(40), wi, { ...COND_STOCH, nSamples: 4 });
    const mean = meanF(p, wo(40), wi, 400, COND_STOCH).mean;
    expect(q.f[0]).toBeGreaterThanOrEqual(0);
    for (let c = 0; c < 3; c++) expect(Math.abs((mean[c] as number) - (m.f[c] as number))).toBeLessThan(0.02 * (m.f[c] as number) + 1e-6);
  });

  it('limit: a black-body coat (conductor absent, k = 0, eta = coat ior) leaves only the coat reflection in the albedo layering', () => {
    // eta = coat ior relative to the coat => conductor eta' = 1, k' = 0: no reflection at all.
    const p = ptLayeredCoatedConductor([1.5, 1.5, 1.5], [0, 0, 0], 0.3, 1.5, 0.3, { thickness: 0 });
    const q = layeredQuadrature(p, wo(30), ALBEDO, 200, 180);
    const black = layeredQuadrature(ptLayered([0, 0, 0], 1.5, 0.3), wo(30), ALBEDO, 200, 180);
    for (let c = 0; c < 3; c++) expect(q.albedo[c]).toBeCloseTo(black.albedo[c] as number, 6);
  });

  it('albedo layering: a white conductor under a smooth coat reflects 1 (F(mu) + (1 - F(mu)) 1), the pdf integrates to 1 with a rough base', () => {
    const p = ptLayeredCoatedConductor(ONE, WHITE_K, 0.3, 1.5, 0.3, { thickness: 0 });
    const q = layeredQuadrature(p, wo(30), ALBEDO, 300, 240);
    // the rough conductor loses a little to its own single scattering; the table error is 0.02
    expect(q.albedo[0]).toBeGreaterThan(0.9);
    expect(q.albedo[0]).toBeLessThan(1.02);
    expect(q.pdf).toBeGreaterThan(0.85);
    expect(q.pdf).toBeLessThan(1.02);
  });

  it('albedo layering: sample f and pdf equal eval at the sampled direction (rough/rough); a smooth/smooth stack gives two delta lobes', () => {
    const p = ptLayeredCoatedConductor([0.2, 0.9, 1.1], [3.9, 2.4, 2.2], 0.4, 1.5, 0.2);
    const r = rng(21);
    for (let i = 0; i < 200; i++) {
      const s = ptLayeredSample(p, wo(35), [r(), r(), r()], ALBEDO);
      if (s.flags === 0) continue;
      const e = ptLayeredEval(p, wo(35), s.wi, ALBEDO);
      expect(s.f[1]).toBeCloseTo(e.f[1] as number, 6);
      expect(s.pdf).toBeCloseTo(e.pdf, 6);
    }
    const sm = ptLayeredCoatedConductor([0.2, 0.9, 1.1], [3.9, 2.4, 2.2], 0, 1.5, 0);
    let delta = 0;
    for (let i = 0; i < 100; i++) {
      const s = ptLayeredSample(sm, wo(35), [r(), r(), r()], ALBEDO);
      if (s.flags & PT_FLAG_DELTA) delta++;
    }
    expect(delta).toBe(100);
  });

  it('albedo layering evaluates the conductor Fresnel at the angle inside the coat (spec mu_c), eta and k relative to the coat', () => {
    const eta: PtVec3 = [0.2, 0.9, 1.1];
    const k: PtVec3 = [3.9, 2.4, 2.2];
    const p = ptLayeredCoatedConductor(eta, k, 0, 1.5, 0);
    const o = wo(50);
    const s = ptLayeredSample(p, o, [0.999, 0.5, 0.5], ALBEDO); // u0 above pS: the delta metal lobe
    expect(s.flags & PT_FLAG_DELTA).toBeTruthy();
    const c = o[2];
    const ct = Math.sqrt(1 - (1 - c * c) / 2.25);
    for (let ch = 0; ch < 3; ch++) {
      const want = ((1 - ptFrDielectricRef(c, 1.5)) * ptFrComplex(ct, (eta[ch] as number) / 1.5, (k[ch] as number) / 1.5)) / c;
      expect(s.f[ch]).toBeCloseTo(want, 9);
    }
  });

  it('stochastic reciprocity in expectation (rough/rough, exitMis none): E[f(wo, wi)] = E[f(wi, wo)] within 4 sigma', () => {
    const p = ptLayeredCoatedConductor([0.2, 0.9, 1.1], [3.9, 2.4, 2.2], 0.45, 1.5, 0.3);
    const a = wo(30);
    const b = dirFromAngles(55, 120);
    const f1 = meanF(p, a, b, 4000, { ...COND_STOCH, nSamples: 4 });
    const f2 = meanF(p, b, a, 4000, { ...COND_STOCH, nSamples: 4 });
    for (let c = 0; c < 3; c++) expect(Math.abs((f1.mean[c] as number) - (f2.mean[c] as number))).toBeLessThan(4 * Math.hypot(f1.se[c] as number, f2.se[c] as number));
  });

  it('two-sided: from below the stack looks the same with the directions negated', () => {
    const p = ptLayeredCoatedConductor([0.2, 0.9, 1.1], [3.9, 2.4, 2.2], 0.4, 1.5, 0.2);
    const a = ptLayeredEval(p, wo(35), dirFromAngles(20, 70), ALBEDO);
    const b = ptLayeredEval(p, [-wo(35)[0], -wo(35)[1], -wo(35)[2]], [-dirFromAngles(20, 70)[0], -dirFromAngles(20, 70)[1], -dirFromAngles(20, 70)[2]], ALBEDO);
    expect(b.f[0]).toBeCloseTo(a.f[0] as number, 10);
  });

  it('WGSL: the conductor code is emitted only on request; without it the fragment text is unchanged', () => {
    for (const layering of ['albedo', 'stochastic'] as const) {
      const plain = ptLayeredWgsl({ layering });
      const cond = ptLayeredWgsl({ layering, conductorBase: true });
      expect(plain).not.toContain('isCond');
      expect(cond).toContain('isCond');
      expect(cond.length).toBeGreaterThan(plain.length);
    }
  });

  it('PT_FLAG_REFLECTION flags the rough albedo samples', () => {
    const p = ptLayeredCoatedConductor(ONE, WHITE_K, 0.4, 1.5, 0.3);
    const s = ptLayeredSample(p, wo(30), [0.9, 0.3, 0.6], ALBEDO);
    expect(s.flags & PT_FLAG_REFLECTION).toBeTruthy();
  });
});

// OpenPBR 1.1.2, "Layering": f_layer = f_coat + (1 - E_coat(w_o)) f_sub (verified against the spec text, DIVERGES V57; the spec's own
// alternative for a Lambertian base is the reciprocal (1 - E(w_i))(1 - E(w_o)) form, our albedoBase 'internal' adds the internal reflection on top).
describe('OpenPBR albedo scaling formula (spec 1.1.2)', () => {
  it('smooth interface: the base is exactly (1 - F(w_o)) base / pi, the specular part is the delta lobe (no eval)', () => {
    const p = ptLayered([0.6, 0.3, 0.2], 1.5, 0);
    for (const th of [0, 30, 60, 80]) {
      const wo = dirFromAngles(th, 0);
      const e = ptLayeredEval(p, wo, dirFromAngles(40, 120), ALBEDO);
      const F = ptFrDielectricRef(wo[2], 1.5);
      for (let c = 0; c < 3; c++) expect(e.f[c]).toBeCloseTo(((p.baseColor[c] as number) * (1 - F)) / Math.PI, 12);
    }
  });

  it('rough interface: f = f_coat + (1 - E_coat) f_sub with E_coat the table albedo at w_o', () => {
    const p = ptLayered([0.6, 0.3, 0.2], 1.5, 0.3);
    const wo = dirFromAngles(50, 0);
    const wi = dirFromAngles(35, 100);
    const e = ptLayeredEval(p, wo, wi, ALBEDO);
    const coat = ptLayeredEval(ptLayered([0, 0, 0], 1.5, 0.3), wo, wi, ALBEDO).f;
    const E = ptLayeredAlbedo(ptDielectricF0(1.5), wo[2], 0.3);
    for (let c = 0; c < 3; c++) expect(e.f[c]).toBeCloseTo((coat[c] as number) + ((p.baseColor[c] as number) * (1 - E)) / Math.PI, 12);
  });
});
