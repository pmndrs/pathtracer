import { describe, expect, it } from 'vitest';
import {
  PT_FLAG_DELTA,
  PT_FLAG_REFLECTION,
  PT_FLAG_TRANSMISSION,
  ptBsdfEval,
  ptBsdfIsDelta,
  ptBsdfPdf,
  ptBsdfSample,
  ptFrComplex,
  ptFrDielectric,
  ptFrameFromNormal,
  type PtBsdfOptions,
  type PtVec3,
} from './bsdf.ts';
import { ptBsdfWgsl } from './bsdf-wgsl.ts';
import { ptBlack, ptConductor, ptDielectric, ptDiffuse } from './materials.ts';
import { bruteForceAlbedo, chiSquareTest, dirFromAngles, mcAlbedo, rng } from './bsdf.testutil.ts';

const GOLD_ETA: PtVec3 = [0.143, 0.375, 1.442];
const GOLD_K: PtVec3 = [3.983, 2.386, 1.603];
const MIRROR_ETA: PtVec3 = [1e-4, 1e-4, 1e-4];
const MIRROR_K: PtVec3 = [1e4, 1e4, 1e4];

const ROUGHNESS = [0.05, 0.1, 0.25, 0.5, 1];
/** Lower bounds on R+T (importance transport, ior 1.5, all wo tested) with a little margin below the measured values. */
const MIN_RT: Record<number, number> = { 0.05: 0.995, 0.1: 0.995, 0.25: 0.95, 0.5: 0.84, 1: 0.42 };
const WO_THETA = [0, 30, 60, 80];
const VNDFS: PtBsdfOptions[] = [{ vndf: 'standard' }, { vndf: 'cap' }];

function sum3(a: readonly number[]): number {
  return (a[0] as number) + (a[1] as number) + (a[2] as number);
}

describe('Fresnel', () => {
  it('conductor with k = 0 equals the dielectric formula, normal incidence is analytic', () => {
    for (const c of [0.05, 0.3, 0.7, 1]) expect(ptFrComplex(c, 1.5, 0)).toBeCloseTo(ptFrDielectric(c, 1.5), 12);
    expect(ptFrDielectric(1, 1.5)).toBeCloseTo(0.04, 12);
    expect(ptFrComplex(1, 0.143, 3.983)).toBeCloseTo(((0.143 - 1) ** 2 + 3.983 ** 2) / ((0.143 + 1) ** 2 + 3.983 ** 2), 12);
  });
  it('dielectric from inside reflects totally past the critical angle', () => {
    expect(ptFrDielectric(-0.3, 1.5)).toBe(1);
    expect(ptFrDielectric(-0.9, 1.5)).toBeLessThan(0.1);
  });
});

describe('Lambert', () => {
  for (const options of VNDFS) {
    it('furnace: albedo = reflectance, pdf = cos / pi, both hemispheres', () => {
      const p = ptDiffuse([0.8, 0.5, 0.2]);
      for (const th of [0, 45, 80, 120, 170]) {
        const wo = dirFromAngles(th, 20);
        const mc = mcAlbedo(p, wo, 50_000, 1, options);
        for (let c = 0; c < 3; c++) expect(Math.abs((mc.refl[c] as number) - (p.reflectance[c] as number))).toBeLessThan(1e-12 + 5 * mc.se[c]!);
        const bf = bruteForceAlbedo(p, wo, options);
        for (let c = 0; c < 3; c++) expect(bf.refl[c]).toBeCloseTo(p.reflectance[c] as number, 3);
        expect(bf.pdfRefl).toBeCloseTo(1, 3);
      }
    });
  }
});

describe('smooth (delta) lobes', () => {
  it('smooth conductor is the Fresnel reflectance exactly', () => {
    const p = ptConductor(GOLD_ETA, GOLD_K, 0);
    expect(ptBsdfIsDelta(p)).toBe(true);
    for (const th of [0, 30, 70]) {
      const wo = dirFromAngles(th, 40);
      const s = ptBsdfSample(p, wo, [0.3, 0.3, 0.3]);
      expect(s.flags).toBe(PT_FLAG_DELTA | PT_FLAG_REFLECTION);
      const w = (s.f[1] * Math.abs(s.wi[2])) / s.pdf;
      expect(w).toBeCloseTo(ptFrComplex(Math.cos((th * Math.PI) / 180), GOLD_ETA[1], GOLD_K[1]), 12);
      expect(ptBsdfEval(p, wo, s.wi).pdf).toBe(0);
    }
  });
  it('smooth dielectric: R + T = 1 in importance transport, R + T / etap^2 in radiance transport', () => {
    for (const ior of [1.33, 1.5, 2.4]) {
      const p = ptDielectric(ior, 0);
      for (const th of [0, 30, 60, 120, 150, 175]) {
        const wo = dirFromAngles(th);
        const cos = wo[2];
        const R = ptFrDielectric(cos, ior);
        // Sample R and T by choosing uc on either side of R: weight f |cos| / pdf per branch.
        const sr = ptBsdfSample(p, wo, [R * 0.5, 0, 0], { transport: 'importance' });
        expect(sr.flags).toBe(PT_FLAG_DELTA | PT_FLAG_REFLECTION);
        const wr = (sr.f[0] * Math.abs(sr.wi[2])) / sr.pdf;
        expect(wr).toBeCloseTo(1, 12); // f|cos|/pdf = R / R
        for (const transport of ['importance', 'radiance'] as const) {
          const st = ptBsdfSample(p, wo, [Math.min(0.999999, R + (1 - R) * 0.5), 0, 0], { transport });
          if (R >= 1) {
            expect(st.flags).toBe(PT_FLAG_DELTA | PT_FLAG_REFLECTION);
            continue;
          }
          expect(st.flags).toBe(PT_FLAG_DELTA | PT_FLAG_TRANSMISSION);
          const wt = (st.f[0] * Math.abs(st.wi[2])) / st.pdf;
          expect(wt).toBeCloseTo(transport === 'importance' ? 1 : 1 / (st.etap * st.etap), 12);
          // Snell: sin_o = etap sin_i... check the bend: etap sin(theta_i) = sin(theta_o)
          expect(st.etap * Math.hypot(st.wi[0], st.wi[1])).toBeCloseTo(Math.hypot(wo[0], wo[1]), 12);
        }
      }
    }
  });
  it('dielectric eta = 1 passes straight through', () => {
    const p = ptDielectric(1, 0.5);
    expect(ptBsdfIsDelta(p)).toBe(true);
    const wo = dirFromAngles(40, 10);
    const s = ptBsdfSample(p, wo, [0.5, 0.2, 0.2]);
    expect(s.flags).toBe(PT_FLAG_DELTA | PT_FLAG_TRANSMISSION);
    for (let i = 0; i < 3; i++) expect(s.wi[i]).toBeCloseTo(-(wo[i] as number), 12);
  });
});

describe('black', () => {
  it('never scatters', () => {
    const p = ptBlack();
    expect(ptBsdfSample(p, [0, 0, 1], [0.5, 0.5, 0.5]).flags).toBe(0);
    expect(ptBsdfEval(p, [0, 0, 1], [0, 0.6, 0.8])).toEqual({ f: [0, 0, 0], pdf: 0 });
  });
});

/** Furnace table: brute-force integral of eval (reference) against the Monte Carlo estimate from sampling. */
describe('white furnace: sampling vs brute-force integration of eval', () => {
  const N = 60_000;

  it('conductor (single scattering loses energy as roughness grows)', () => {
    const mirror: Record<number, number[]> = {};
    for (const rough of ROUGHNESS) {
      const p = ptConductor(GOLD_ETA, GOLD_K, rough);
      for (const th of WO_THETA) {
        const wo = dirFromAngles(th, 25);
        const bf = bruteForceAlbedo(p, wo);
        for (const options of VNDFS) {
          const mc = mcAlbedo(p, wo, N, 7 + th, options);
          for (let c = 0; c < 3; c++) {
            const tol = 5 * (mc.se[c] as number) + 4e-3;
            expect(Math.abs((mc.refl[c] as number) - (bf.refl[c] as number)), `rough ${rough} th ${th} vndf ${options.vndf} c ${c}`).toBeLessThan(tol);
            expect(mc.trans[c]).toBe(0);
          }
          // The integral of the pdf is the probability that a sample stays above the horizon.
          expect(Math.abs(mc.success - bf.pdfRefl), `pdf mass rough ${rough} th ${th}`).toBeLessThan(5 * mc.successSe + 4e-3);
        }
      }
      // White conductor: energy lost at normal incidence grows with roughness.
      const white = ptConductor(MIRROR_ETA, MIRROR_K, rough);
      expect(ptFrComplex(1, 1e-4, 1e4)).toBeGreaterThan(0.999);
      const wo = dirFromAngles(60, 0);
      const bf = bruteForceAlbedo(white, wo);
      (mirror[60] ??= []).push(bf.refl[0]);
      expect(bf.refl[0]).toBeLessThanOrEqual(1 + 3e-3);
    }
    const series = mirror[60] as number[];
    expect(series[series.length - 1] as number).toBeLessThan(0.95); // roughness 1 loses at least 5 %
    expect(series[0] as number).toBeGreaterThan(0.99); // roughness 0.05 loses under 1 %
    for (let i = 1; i < series.length; i++) expect(series[i] as number).toBeLessThanOrEqual((series[i - 1] as number) + 3e-3);
  });

  it('rough dielectric: R + T conserved up to single-scatter loss (importance); radiance rescales T by 1/etap^2', () => {
    for (const rough of ROUGHNESS) {
      const p = ptDielectric(1.5, rough);
      for (const th of [0, 30, 60, 80, 130, 160]) {
        const wo = dirFromAngles(th, 25);
        for (const transport of ['importance', 'radiance'] as const) {
          const bf = bruteForceAlbedo(p, wo, { transport });
          for (const vndf of ['standard', 'cap'] as const) {
            const mc = mcAlbedo(p, wo, N, 11 + th, { transport, vndf });
            const tol = 5 * (mc.se[0] as number) + 5e-3;
            expect(Math.abs((mc.refl[0] as number) + (mc.trans[0] as number) - (bf.refl[0] as number) - (bf.trans[0] as number)), `rough ${rough} th ${th} ${transport} ${vndf}`).toBeLessThan(tol);
            expect(Math.abs((mc.refl[0] as number) - (bf.refl[0] as number)), `R rough ${rough} th ${th} ${transport} ${vndf}`).toBeLessThan(tol);
            expect(Math.abs(mc.success - bf.pdfRefl - bf.pdfTrans), `pdf mass rough ${rough} th ${th}`).toBeLessThan(5 * mc.successSe + 5e-3);
          }
          if (transport === 'importance') {
            const total = (bf.refl[0] as number) + (bf.trans[0] as number);
            expect(total, `energy created? rough ${rough} th ${th}`).toBeLessThanOrEqual(1 + 3e-3);
            // The single-scatter loss grows with roughness but stays modest.
            expect(total, `energy lost? rough ${rough} th ${th}`).toBeGreaterThan(MIN_RT[rough] as number);
          }
        }
      }
    }
  }, 120_000);
});

describe('pdf consistency', () => {
  const cases: [string, ReturnType<typeof ptDiffuse>, PtVec3][] = [
    ['diffuse', ptDiffuse([0.5, 0.5, 0.5]), dirFromAngles(50, 10)],
    ['conductor', ptConductor(GOLD_ETA, GOLD_K, 0.4), dirFromAngles(50, 10)],
    ['aniso conductor', { ...ptConductor(GOLD_ETA, GOLD_K, 0.4), alpha: [0.1, 0.5] as const }, dirFromAngles(50, 10)],
    ['dielectric out', ptDielectric(1.5, 0.4), dirFromAngles(50, 10)],
    ['dielectric in', ptDielectric(1.5, 0.4), dirFromAngles(140, 10)],
    ['dielectric in 60', ptDielectric(1.5, 0.3), dirFromAngles(120, 10)],
  ];
  for (const [name, p, wo] of cases) {
    for (const options of VNDFS) {
      it(`${name} (${options.vndf}): sample pdf and f equal ptBsdfPdf / ptBsdfEval at the sampled direction`, () => {
        const rand = rng(99);
        let checked = 0;
        for (let i = 0; i < 4000; i++) {
          const s = ptBsdfSample(p, wo, [rand(), rand(), rand()], options);
          if (s.flags === 0) continue;
          const e = ptBsdfEval(p, wo, s.wi, options);
          expect(Math.abs(s.pdf - e.pdf)).toBeLessThanOrEqual(1e-6 * Math.max(1, s.pdf));
          expect(ptBsdfPdf(p, wo, s.wi)).toBeCloseTo(e.pdf, 12);
          for (let c = 0; c < 3; c++) expect(Math.abs((s.f[c] as number) - (e.f[c] as number))).toBeLessThanOrEqual(1e-6 * Math.max(1, e.f[c] as number));
          checked++;
        }
        expect(checked).toBeGreaterThan(1000);
      });
    }
  }
});

describe('reciprocity of reflection lobes', () => {
  it('f(wo, wi) = f(wi, wo) for diffuse, conductor and dielectric reflection', () => {
    const rand = rng(5);
    const params = [ptDiffuse([0.3, 0.6, 0.9]), ptConductor(GOLD_ETA, GOLD_K, 0.35), { ...ptConductor(GOLD_ETA, GOLD_K, 0.5), alpha: [0.1, 0.5] as const }, ptDielectric(1.5, 0.35)];
    for (const p of params) {
      for (let i = 0; i < 500; i++) {
        const side = rand() < 0.5 ? 1 : -1;
        const mk = (): PtVec3 => {
          const z = 0.05 + 0.95 * rand();
          const s = Math.sqrt(1 - z * z);
          const phi = 2 * Math.PI * rand();
          return [s * Math.cos(phi), s * Math.sin(phi), side * z];
        };
        const a = mk();
        const b = mk();
        if (p.kind !== 2 && side < 0) continue;
        const f1 = ptBsdfEval(p, a, b).f;
        const f2 = ptBsdfEval(p, b, a).f;
        for (let c = 0; c < 3; c++) expect(Math.abs((f1[c] as number) - (f2[c] as number))).toBeLessThanOrEqual(1e-9 * Math.max(1, f1[c] as number));
      }
    }
  });
  it('rough dielectric transmission is reciprocal after undoing the radiance scale: f_I(wo,wi) etap(wo)^-2 ... = f_I(wi,wo) etap(wi)^-2', () => {
    // pbrt importance transport has f_I(wo,wi) / eta_i^2 symmetric... verified numerically: f_R(wo,wi) / etap_o^-2 form.
    const p = ptDielectric(1.5, 0.4);
    const rand = rng(8);
    let n = 0;
    for (let i = 0; i < 2000 && n < 300; i++) {
      const wo = dirFromAngles(10 + 70 * rand(), 360 * rand());
      const s = ptBsdfSample(p, wo, [0.99, rand(), rand()], { transport: 'importance' });
      if (!(s.flags & PT_FLAG_TRANSMISSION)) continue;
      const back = ptBsdfEval(p, s.wi, wo, { transport: 'importance' }).f[0];
      // Generalised reciprocity: f(wo,wi) / eta_i^2 = f(wi,wo) / eta_o^2 with eta_o = 1, eta_i = 1.5 (importance values).
      expect(Math.abs((s.f[0] as number) / (1.5 * 1.5) - back / 1)).toBeLessThanOrEqual(1e-6 * Math.max(1e-3, back));
      n++;
    }
    expect(n).toBeGreaterThan(100);
  });
});

describe('chi-square goodness of fit of sampled directions vs pdf', () => {
  const cases: [string, ReturnType<typeof ptDiffuse>, PtVec3][] = [
    ['diffuse', ptDiffuse([0.5, 0.5, 0.5]), dirFromAngles(40, 30)],
    ['conductor r0.5', ptConductor(GOLD_ETA, GOLD_K, 0.5), dirFromAngles(40, 30)],
    ['conductor r0.8 grazing', ptConductor(GOLD_ETA, GOLD_K, 0.8), dirFromAngles(75, 200)],
    ['conductor aniso', { ...ptConductor(GOLD_ETA, GOLD_K, 0.5), alpha: [0.25, 0.6] as const }, dirFromAngles(50, 100)],
    ['dielectric out r0.5', ptDielectric(1.5, 0.5), dirFromAngles(30, 30)],
    ['dielectric in r0.5', ptDielectric(1.5, 0.5), dirFromAngles(150, 30)],
    ['dielectric in 60deg r0.7', ptDielectric(1.5, 0.7), dirFromAngles(120, 30)],
  ];
  for (const [name, p, wo] of cases) {
    for (const options of VNDFS) {
      it(`${name} (${options.vndf})`, () => {
        const r = chiSquareTest(p, wo, 400_000, 1234, options);
        // 1e-3 significance: with a fixed seed this is a deterministic check; a wrong pdf or sampler gives p ~ 0.
        expect(r.p, `chi2 ${r.stat.toFixed(1)} dof ${r.dof} p ${r.p}`).toBeGreaterThan(1e-3);
      }, 60_000);
    }
  }
});

describe('chi-square power', () => {
  it('rejects a pdf from different parameters (the test can fail)', () => {
    const wo = dirFromAngles(40, 30);
    const a = chiSquareTest(ptConductor(GOLD_ETA, GOLD_K, 0.5), wo, 400_000, 1, {}, ptConductor(GOLD_ETA, GOLD_K, 0.45));
    expect(a.p).toBeLessThan(1e-6);
    const b = chiSquareTest(ptDielectric(1.5, 0.5), dirFromAngles(150, 30), 400_000, 1, {}, ptDielectric(1.4, 0.5));
    expect(b.p).toBeLessThan(1e-6);
  });
});

describe('frame and WGSL text', () => {
  it('ptFrameFromNormal is orthonormal and right-handed', () => {
    const rand = rng(3);
    for (let i = 0; i < 100; i++) {
      const z = 2 * rand() - 1;
      const s = Math.sqrt(1 - z * z);
      const phi = 2 * Math.PI * rand();
      const n: PtVec3 = [s * Math.cos(phi), s * Math.sin(phi), z];
      const f = ptFrameFromNormal(n);
      const d = (a: PtVec3, b: PtVec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
      expect(d(f.t, f.t)).toBeCloseTo(1, 12);
      expect(d(f.b, f.b)).toBeCloseTo(1, 12);
      expect(d(f.t, f.b)).toBeCloseTo(0, 12);
      expect(d(f.t, n)).toBeCloseTo(0, 12);
      expect(d(f.b, n)).toBeCloseTo(0, 12);
    }
  });
  it('generates the module with the settings baked in', () => {
    const a = ptBsdfWgsl();
    expect(a).toContain('fn ptBsdfSample');
    expect(a).toContain('const PT_VNDF_CAP: bool = false;');
    expect(ptBsdfWgsl({ vndf: 'cap', transport: 'importance' })).toContain('const PT_TRANSPORT_RADIANCE: bool = false;');
    expect(sum3([1, 2, 3])).toBe(6);
  });
});
