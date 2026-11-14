// P2.25: OpenPBR fuzz (LTC sheen). Checks: the LTC lobe is a normalised density and its sampler matches it; furnace over a white Lambert base
// (albedo scaling: w F R + (1 - w R) = 1); fuzz_weight 0 is the base exactly; the mixture pdf integrates to at most 1; reciprocity (approximate, LTC
// has the cosine baked in); the table is the one of the reference; and the WGSL gating.
import { describe, expect, it } from 'vitest';
import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIFFUSE, PT_FLAG_DELTA, ptBsdfEval, ptBsdfSample, type PtBsdfParams, type PtVec3 } from './bsdf.ts';
import { ptConductor, ptDiffuse } from './materials.ts';
import { dirFromAngles, rng } from './bsdf.testutil.ts';
import { PT_FUZZ_RES, PT_FUZZ_TABLE, ptFuzzActive, ptFuzzAlbedo, ptFuzzBaseOf, ptFuzzCoeffs, ptFuzzLobeEval, ptFuzzLobePdf, ptFuzzLobeSample, ptFuzzedEval, ptFuzzedSample, ptFuzzTableF32, type PtFuzzParams } from './fuzz.ts';
import { ptFuzzWgsl } from './fuzz-wgsl.ts';
import { ptMaterialFeatures } from './adapter.ts';
import { createMaterial } from '@pmndrs/raytracer';

const WHITE: PtVec3 = [1, 1, 1];
const fz = (weight: number, roughness: number, color: PtVec3 = WHITE): PtFuzzParams => ({ weight, color, roughness });
const lambert = (r: number) => ptFuzzBaseOf(ptDiffuse([r, r, r]), ptBsdfEval, ptBsdfSample);

/** Integral over the +z hemisphere of g(wi) by a midpoint rule in (theta, phi). */
function hemi(g: (wi: PtVec3) => number, nt = 240, np = 240): number {
  let s = 0;
  for (let i = 0; i < nt; i++) {
    const th = ((i + 0.5) / nt) * (Math.PI / 2);
    for (let j = 0; j < np; j++) {
      const ph = ((j + 0.5) / np) * 2 * Math.PI;
      s += g([Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)]) * Math.sin(th);
    }
  }
  return (s * (Math.PI / 2) * 2 * Math.PI) / (nt * np);
}

describe('fuzz table', () => {
  it('is the 32 x 32 table of ltc-sheen (1024 cells of (aInv, bInv, R)), R in [0, 1)', () => {
    expect(PT_FUZZ_RES).toBe(32);
    expect(PT_FUZZ_TABLE.length).toBe(32 * 32 * 3);
    expect(PT_FUZZ_TABLE.slice(0, 3)).toEqual([0.01415, 0.0006, 0.00001]);
    for (let i = 0; i < 1024; i++) {
      expect(PT_FUZZ_TABLE[i * 3 + 2]).toBeGreaterThanOrEqual(0);
      expect(PT_FUZZ_TABLE[i * 3 + 2]).toBeLessThan(1);
    }
    expect(ptFuzzTableF32().length).toBe(4096);
  });
  it('bilinear lookup reproduces a node and interpolates between nodes', () => {
    const node = ptFuzzCoeffs(16 / 31, 8 / 31);
    expect(node[2]).toBeCloseTo(PT_FUZZ_TABLE[(8 * 32 + 16) * 3 + 2] as number, 12);
    const mid = ptFuzzCoeffs(16.5 / 31, 8 / 31);
    expect(mid[2]).toBeCloseTo(((PT_FUZZ_TABLE[(8 * 32 + 16) * 3 + 2] as number) + (PT_FUZZ_TABLE[(8 * 32 + 17) * 3 + 2] as number)) / 2, 12);
  });
});

describe('LTC lobe', () => {
  const cases: [number, number, number][] = [[0.9, 0.5, 0], [0.6, 0.5, 70], [0.3, 0.25, 130], [0.8, 0.75, 20], [0.15, 0.9, 200], [0.5, 1, 10]];
  it('is a normalised density over the hemisphere (integral 1 within 3e-3)', () => {
    for (const [cos, alpha, phi] of cases) {
      const sin = Math.sqrt(1 - cos * cos);
      const wo: PtVec3 = [sin * Math.cos((phi * Math.PI) / 180), sin * Math.sin((phi * Math.PI) / 180), cos];
      const c = ptFuzzCoeffs(cos, alpha);
      expect(c[0]).toBeGreaterThan(1e-4);
      expect(hemi((wi) => ptFuzzLobePdf(c, wo, wi))).toBeCloseTo(1, 2);
    }
  });
  it('the sampler matches the pdf: E[g / pdf] = integral of g (g = wi.z^2 and a lobe-aligned g)', () => {
    const r = rng(7);
    for (const [cos, alpha, phi] of cases) {
      const sin = Math.sqrt(1 - cos * cos);
      const wo: PtVec3 = [sin * Math.cos((phi * Math.PI) / 180), sin * Math.sin((phi * Math.PI) / 180), cos];
      const c = ptFuzzCoeffs(cos, alpha);
      const n = 200000;
      let a = 0;
      let b = 0;
      for (let i = 0; i < n; i++) {
        const wi = ptFuzzLobeSample(c, wo, r(), r());
        const p = ptFuzzLobePdf(c, wo, wi);
        a += (wi[2] * wi[2]) / p;
        b += Math.max(0, wi[0] * wo[0] + wi[1] * wo[1]) / p; // projection on wo's horizontal direction
      }
      expect(a / n).toBeCloseTo((2 * Math.PI) / 3, 1);
      const want = hemi((wi) => Math.max(0, wi[0] * wo[0] + wi[1] * wo[1]));
      expect(Math.abs(b / n - want)).toBeLessThan(0.02 * Math.max(1, want));
    }
  });
});

describe('layering (albedo scaling)', () => {
  const wos = [5, 30, 55, 75, 85].map((t) => dirFromAngles(t, 33));
  it('furnace: white fuzz over a white Lambert base conserves energy: integral f cos = 1 for any weight and roughness', () => {
    for (const w of [1, 0.6, 0.25]) {
      for (const rough of [0.1, 0.5, 0.9]) {
        for (const wo of wos) {
          const f = fz(w, rough);
          if (!ptFuzzActive(f, wo[2])) continue;
          const albedo = hemi((wi) => ptFuzzedEval(f, lambert(1), wo, wi).f[1] * wi[2], 300, 300);
          expect(albedo).toBeCloseTo(1, 2);
        }
      }
    }
  });
  it('albedo of the fuzz alone is w F E_fuzz', () => {
    const f = fz(0.8, 0.5, [0.9, 0.5, 0.1]);
    const wo = dirFromAngles(40, 10);
    const a = hemi((wi) => ptFuzzLobeEval(f, wo, wi).f[0] * wi[2], 300, 300);
    expect(a).toBeCloseTo(0.9 * ptFuzzAlbedo(wo[2], 0.5), 2);
  });
  it('a coloured fuzz does not tint the base: base term is (1 - w E) times the base, grey', () => {
    const f = fz(1, 0.5, [1, 0, 0]);
    const wo = dirFromAngles(30, 0);
    const wi = dirFromAngles(50, 100);
    const e = ptFuzzedEval(f, lambert(0.5), wo, wi);
    const keep = 1 - ptFuzzAlbedo(wo[2], 0.5);
    // green and blue only carry the base
    expect(e.f[1]).toBeCloseTo((keep * 0.5) / Math.PI, 12);
    expect(e.f[2]).toBeCloseTo(e.f[1], 15);
    expect(e.f[0]).toBeGreaterThan(e.f[1]);
  });
  it('weight 0 (or a black colour) is the base exactly: eval and sample bitwise', () => {
    const base = lambert(0.7);
    const metal = ptFuzzBaseOf(ptConductor([0.2, 0.9, 1.1], [3.9, 2.4, 2.1], 0.4), ptBsdfEval, ptBsdfSample);
    for (const b of [base, metal]) {
      for (const f of [fz(0, 0.5), fz(1, 0.5, [0, 0, 0])]) {
        for (const wo of wos) {
          const wi = dirFromAngles(40, 200);
          expect(ptFuzzedEval(f, b, wo, wi)).toEqual(b.eval(wo, wi));
          expect(ptFuzzedSample(f, b, wo, [0.3, 0.6, 0.2])).toEqual(b.sample(wo, [0.3, 0.6, 0.2]));
        }
      }
    }
  });
  it('the mixture pdf integrates to at most 1 (Lambert and rough metal bases; black base: exactly the lobe)', () => {
    const bases = [lambert(0.8), ptFuzzBaseOf(ptConductor([0.2, 0.9, 1.1], [3.9, 2.4, 2.1], 0.5), ptBsdfEval, ptBsdfSample)];
    for (const b of bases) {
      for (const wo of wos) {
        const total = hemi((wi) => ptFuzzedEval(fz(0.7, 0.4), b, wo, wi).pdf);
        expect(total).toBeLessThanOrEqual(1.004);
        expect(total).toBeGreaterThan(0.9);
      }
    }
    const black = ptFuzzBaseOf({ kind: PT_BSDF_BLACK, reflectance: [0, 0, 0], eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior: 1 } as PtBsdfParams, ptBsdfEval, ptBsdfSample);
    expect(hemi((wi) => ptFuzzedEval(fz(1, 0.5), black, wos[1] as PtVec3, wi).pdf)).toBeCloseTo(1, 2);
  });
  it('sample and eval agree: the sample returns the total f and the mixture pdf at its direction', () => {
    const r = rng(3);
    const b = lambert(0.6);
    for (let i = 0; i < 400; i++) {
      const wo = dirFromAngles(10 + 70 * r(), 360 * r());
      const f = fz(0.5 + 0.5 * r(), r(), [r(), r(), r()]);
      const s = ptFuzzedSample(f, b, wo, [r(), r(), r()]);
      if (s.flags === 0) continue;
      const e = ptFuzzedEval(f, b, wo, s.wi);
      expect(s.pdf).toBeCloseTo(e.pdf, 9);
      expect(s.f[0]).toBeCloseTo(e.f[0], 9);
    }
  });
  it('Monte Carlo of f cos / pdf over the sampler gives the albedo (1 for a white Lambert base), also with a delta base lobe', () => {
    const r = rng(11);
    const wo = dirFromAngles(50, 20);
    const f = fz(1, 0.45);
    let sum = 0;
    const n = 300000;
    for (let i = 0; i < n; i++) {
      const s = ptFuzzedSample(f, lambert(1), wo, [r(), r(), r()]);
      if (s.flags) sum += (s.f[1] * Math.abs(s.wi[2])) / s.pdf;
    }
    expect(sum / n).toBeCloseTo(1, 1);
    // smooth conductor with a perfect reflectance (eta 1, k huge is ~1): the delta lobe keeps its flag and is scaled by 1 - w E
    const mirror = ptFuzzBaseOf({ ...ptConductor([1, 1, 1], [1000, 1000, 1000], 0), alpha: [0, 0] }, ptBsdfEval, ptBsdfSample);
    let delta = 0;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const s = ptFuzzedSample(f, mirror, wo, [r(), r(), r()]);
      if (s.flags) {
        const v = (s.f[1] * Math.abs(s.wi[2])) / s.pdf;
        total += v;
        if (s.flags & PT_FLAG_DELTA) delta += v;
      }
    }
    expect(total / n).toBeCloseTo(1, 1);
    expect(delta / n).toBeCloseTo(1 - ptFuzzAlbedo(wo[2], 0.45), 1);
  });
  it('reciprocity holds only approximately (cosine baked into the LTC): median relative asymmetry below 0.5, recorded', () => {
    const r = rng(5);
    const rel: number[] = [];
    for (let i = 0; i < 2000; i++) {
      const wo = dirFromAngles(5 + 80 * r(), 360 * r());
      const wi = dirFromAngles(5 + 80 * r(), 360 * r());
      const f = fz(1, 0.2 + 0.7 * r());
      const a = ptFuzzLobeEval(f, wo, wi).f[0];
      const b = ptFuzzLobeEval(f, wi, wo).f[0];
      if (a + b > 1e-4) rel.push(Math.abs(a - b) / (0.5 * (a + b)));
    }
    rel.sort((x, y) => x - y);
    expect(rel.length).toBeGreaterThan(500);
    expect(rel[Math.floor(rel.length / 2)] as number).toBeLessThan(0.5);
  });
});

describe('adapter features and WGSL gating', () => {
  it('ptMaterialFeatures sets fuzz only for a material with fuzz weight and colour', () => {
    expect(ptMaterialFeatures([createMaterial({})]).fuzz).toBeUndefined();
    expect(ptMaterialFeatures([createMaterial({ sheenWeight: 1, sheenColor: [0, 0, 0] })]).fuzz).toBeUndefined();
    expect(ptMaterialFeatures([createMaterial({ sheenWeight: 1, sheenColor: [0.5, 0.2, 0.1] })]).fuzz).toBe(true);
  });
  it('the fuzz WGSL binds the table and names the base evaluators it wraps', () => {
    const w = ptFuzzWgsl({ layered: false });
    expect(w).toContain('ptFuzzTable');
    expect(w).toContain('ptBsdfEval(');
    expect(ptFuzzWgsl({ layered: true })).toContain('ptEval(');
  });
  it('unused PT_BSDF_* kinds stay referenced (sanity)', () => {
    expect([PT_BSDF_DIFFUSE, PT_BSDF_CONDUCTOR]).toEqual([0, 1]);
    expect(ptFuzzLobePdf([0, 0, 0], [0, 0, 1], [0, 0, 1])).toBe(0);
  });
});
