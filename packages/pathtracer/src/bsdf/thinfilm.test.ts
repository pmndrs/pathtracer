// P2.27: OpenPBR thin film. The Fourier-space RGB model against the exact spectral Airy sum of bench/metrics/thinfilm-airy.ts (CIE 1931 + D65);
// weight 0 and thickness 0 are the plain material bit for bit; interference colours; energy; the film in the conductor and layered lobes; gating.
import { createMaterial } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { airyRgb } from '../../../../bench/metrics/thinfilm-airy.ts';
import { PT_BSDF_LAYERED, ptBsdfEval, ptBsdfSample, ptFrDielectric, type PtVec3 } from './bsdf.ts';
import { ptBsdfWgsl } from './bsdf-wgsl.ts';
import { ptBsdfFromOpenPbr, ptFilmOfMaterial, ptLayeredFromBsdf, ptMaterialFeatures, type PtMaterialOptions } from './adapter.ts';
import { ptLayeredEval } from './layered.ts';
import { ptLayeredWgsl } from './layered-wgsl.ts';
import { ptConductor } from './materials.ts';
import { PT_FILM_WHITE, ptConductorF0, ptFilmActive, ptFilmFresnel, ptFilmSensitivity, ptThinFilmAiry, type PtThinFilm } from './thinfilm.ts';
import { ptThinFilmWgsl } from './thinfilm-wgsl.ts';
import { dirFromAngles } from './bsdf.testutil.ts';

const F0_GLASS = ((1.5 - 1) / 2.5) ** 2;
const glass: PtVec3 = [F0_GLASS, F0_GLASS, F0_GLASS];
const maxAbs = (a: readonly number[], b: readonly number[]): number => Math.max(...a.map((v, i) => Math.abs(v - (b[i] as number))));

describe('Fourier-space thin film against the exact spectral Airy sum', () => {
  it('dielectric base: within 0.012 absolute for cos >= 0.6 (film 1.4, 1.8; 0.05 to 2 um), 0.065 at cos 0.25', () => {
    let worstNear = 0;
    let worstFar = 0;
    for (const n of [1.4, 1.8]) {
      for (const d of [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2]) {
        for (const cos of [1, 0.8, 0.6]) worstNear = Math.max(worstNear, maxAbs(ptThinFilmAiry(cos, n, d, glass), airyRgb({ filmIor: n, thicknessUm: d, baseN: 1.5, baseK: 0 }, cos)));
        worstFar = Math.max(worstFar, maxAbs(ptThinFilmAiry(0.25, n, d, glass), airyRgb({ filmIor: n, thicknessUm: d, baseN: 1.5, baseK: 0 }, 0.25)));
      }
    }
    expect(worstNear).toBeLessThan(0.012);
    expect(worstFar).toBeLessThan(0.065);
  });
  it('grey metal base (eta 1, k 6, F0 0.9; real effective IOR, no phase from k): within 0.04 near normal, 0.07 at cos 0.4', () => {
    const f0 = 36 / 40;
    for (const d of [0.1, 0.3, 0.5, 1]) {
      expect(maxAbs(ptThinFilmAiry(1, 1.4, d, [f0, f0, f0]), airyRgb({ filmIor: 1.4, thicknessUm: d, baseN: 1, baseK: 6 }, 1))).toBeLessThan(0.04);
      expect(maxAbs(ptThinFilmAiry(0.4, 1.4, d, [f0, f0, f0]), airyRgb({ filmIor: 1.4, thicknessUm: d, baseN: 1, baseK: 6 }, 0.4))).toBeLessThan(0.07);
    }
  });
  it('a constant spectrum is a grey RGB: the exact integral gives (c, c, c) and the balanced Fourier curves sum to (1, 1, 1) at zero path difference', () => {
    const s = ptFilmSensitivity(0, [0, 0, 0]);
    for (const v of s) expect(v).toBeCloseTo(1, 2);
    expect(PT_FILM_WHITE[0]).toBeCloseTo(1.2048, 4);
    const thick = airyRgb({ filmIor: 1.4, thicknessUm: 0.0001, baseN: 1.5, baseK: 0 }, 1);
    expect(thick[0]).toBeCloseTo(thick[2], 3);
  });
});

describe('weight 0 and thickness 0 are the plain material', () => {
  it('ptFilmFresnel returns the plain Fresnel object itself; ptFilmActive is false', () => {
    const plain: PtVec3 = [0.1, 0.2, 0.3];
    for (const film of [undefined, { weight: 0, thickness: 0.5, ior: 1.4 }, { weight: 1, thickness: 0, ior: 1.4 }] as (PtThinFilm | undefined)[]) {
      expect(ptFilmActive(film)).toBe(false);
      expect(ptFilmFresnel(film, 0.7, plain, glass)).toBe(plain);
    }
  });
  it('conductor eval and sample are bitwise the film-free results', () => {
    const p = ptConductor([1, 1, 1], [6, 6, 6], 0.35);
    const wo = dirFromAngles(35, 20);
    const wi = dirFromAngles(50, 200);
    for (const film of [{ weight: 0, thickness: 0.4, ior: 1.4 }, { weight: 1, thickness: 0, ior: 1.4 }] as PtThinFilm[]) {
      expect(ptBsdfEval(p, wo, wi, { film })).toEqual(ptBsdfEval(p, wo, wi));
      expect(ptBsdfSample(p, wo, [0.3, 0.6, 0.2], { film })).toEqual(ptBsdfSample(p, wo, [0.3, 0.6, 0.2]));
    }
  });
  it('layered eval is bitwise the film-free result', () => {
    const lp = ptLayeredFromBsdf(ptBsdfFromOpenPbr(createMaterial({ baseColor: [0.5, 0.4, 0.3], specularRoughness: 0.3 }), { features: { mix: false, thin: false, layered: true, medium: false }, layered: { layering: 'albedo' } }, 0));
    const wo = dirFromAngles(35, 20);
    const wi = dirFromAngles(50, 200);
    const plain = ptLayeredEval(lp, wo, wi);
    expect(ptLayeredEval({ ...lp, film: { weight: 0, thickness: 0.4, ior: 1.4 } }, wo, wi)).toEqual(plain);
    expect(ptLayeredEval({ ...lp, film: { weight: 1, thickness: 0, ior: 1.4 } }, wo, wi)).toEqual(plain);
  });
  it('a vanishing film (thickness 1e-4 um) is within 0.004 of the plain Fresnel (continuity)', () => {
    for (const cos of [1, 0.7, 0.4]) {
      const F = ptFrDielectric(cos, 1.5);
      expect(maxAbs(ptThinFilmAiry(cos, 1.4, 1e-4, glass), [F, F, F])).toBeLessThan(0.004);
    }
  });
});

describe('interference colours', () => {
  it('a quarter-wave MgF2-like film (n 1.38, 0.0996 um on glass 1.5) reflects less green than red and blue: the classic magenta coating, and far less than bare glass', () => {
    const [r, g, b] = airyRgb({ filmIor: 1.38, thicknessUm: 0.0996, baseN: 1.5, baseK: 0 }, 1);
    expect(g).toBeLessThan(r);
    expect(g).toBeLessThan(b);
    expect(g).toBeLessThan(F0_GLASS * 0.5);
    const m = ptThinFilmAiry(1, 1.38, 0.0996, glass);
    expect(m[1]).toBeLessThan(m[0]);
    expect(m[1]).toBeLessThan(m[2]);
  });
  it('the colour changes with thickness (hue circles) and washes out for a thick film', () => {
    const thin = ptThinFilmAiry(1, 1.4, 0.3, glass);
    const thicker = ptThinFilmAiry(1, 1.4, 0.6, glass);
    expect(maxAbs(thin, thicker)).toBeGreaterThan(0.008);
    const thick = ptThinFilmAiry(1, 1.4, 20, glass);
    expect(Math.max(...thick) - Math.min(...thick)).toBeLessThan(0.002);
    // the incoherent limit of two interfaces in series (R12 + (1 - R12)^2 R23 / (1 - R12 R23)) with R12 = 0.0278 (1 -> 1.4), R23 = 0.0023 (1.4 -> 1.5)
    const R12 = ((1.4 - 1) / 2.4) ** 2;
    const R23 = ((1.5 - 1.4) / 2.9) ** 2;
    expect(thick[1]).toBeCloseTo(R12 + ((1 - R12) ** 2 * R23) / (1 - R12 * R23), 2);
  });
  it('reflectance is within [0, 1] everywhere (clamped, no negative colours)', () => {
    for (const d of [0.01, 0.2, 0.7]) for (const cos of [0.05, 0.5, 1]) for (const v of ptThinFilmAiry(cos, 1.9, d, [0.04, 0.5, 0.95])) expect(v >= 0 && v <= 1.0001).toBe(true);
  });
});

describe('film in the lobes', () => {
  it('a conductor with a film changes its colour and keeps its pdf (only the Fresnel changes)', () => {
    const p = ptConductor([1, 1, 1], [6, 6, 6], 0.35);
    const wo = dirFromAngles(35, 20);
    const wi = dirFromAngles(50, 200);
    const film: PtThinFilm = { weight: 1, thickness: 0.3, ior: 1.4 };
    const a = ptBsdfEval(p, wo, wi);
    const b = ptBsdfEval(p, wo, wi, { film });
    expect(b.pdf).toBe(a.pdf);
    expect(maxAbs(a.f, b.f)).toBeGreaterThan(1e-3 * a.f[0]);
    const half = ptBsdfEval(p, wo, wi, { film: { ...film, weight: 0.5 } });
    for (let i = 0; i < 3; i++) expect(half.f[i]).toBeCloseTo(0.5 * ((a.f[i] as number) + (b.f[i] as number)), 12); // linear in the weight
  });
  it('layered: the film replaces the interface Fresnel (smooth top: the sampled delta carries the film Fresnel), the base keeps 1 - s E per channel', () => {
    const options: PtMaterialOptions = { features: { mix: false, thin: false, layered: true, medium: false }, layered: { layering: 'albedo' } };
    const m = createMaterial({ baseColor: [0.5, 0.5, 0.5], specularRoughness: 0 });
    const p = ptBsdfFromOpenPbr(m, options, 0);
    expect(p.kind).toBe(PT_BSDF_LAYERED);
    const film: PtThinFilm = { weight: 1, thickness: 0.3, ior: 1.4 };
    const lp = ptLayeredFromBsdf(p, film);
    const wo = dirFromAngles(40, 0);
    const plainLp = ptLayeredFromBsdf(p);
    // eval of the Lambert part scales with the per-channel keep: differs per channel with a film, equal channels without
    const wi = dirFromAngles(30, 100);
    const a = ptLayeredEval(plainLp, wo, wi).f;
    const b = ptLayeredEval(lp, wo, wi).f;
    expect(a[0]).toBeCloseTo(a[2] as number, 12);
    expect(Math.abs((b[0] as number) - (b[2] as number))).toBeGreaterThan(1e-5);
  });
  it('features and adapter: the film is dropped under a coat and on glass', () => {
    expect(ptMaterialFeatures([createMaterial({ thinFilmWeight: 1, thinFilmThickness: 0.3 })]).thinFilm).toBe(true);
    expect(ptMaterialFeatures([createMaterial({ thinFilmWeight: 0 })]).thinFilm).toBeUndefined();
    expect(ptMaterialFeatures([createMaterial({ thinFilmWeight: 1, thinFilmThickness: 0.3, coatWeight: 1 })]).thinFilm).toBeUndefined();
    expect(ptMaterialFeatures([createMaterial({ thinFilmWeight: 1, thinFilmThickness: 0.3, transmissionWeight: 1 })]).thinFilm).toBeUndefined();
    expect(ptFilmOfMaterial(createMaterial({ thinFilmWeight: 0.5, thinFilmThickness: 0.3, thinFilmIor: 1.7 }))).toEqual({ weight: 0.5, thickness: 0.3, ior: 1.7 });
    expect(ptConductorF0([1, 1, 1], [6, 6, 6])[0]).toBeCloseTo(0.9, 12);
  });
});

describe('WGSL gating', () => {
  it('the default kernel text has no film code; thinFilm routes the conductor Fresnel and patches the albedo layering', () => {
    expect(ptBsdfWgsl()).not.toContain('ptFrConductor');
    expect(ptLayeredWgsl({ layering: 'albedo' })).not.toContain('ptFilm');
    const b = ptBsdfWgsl({ thinFilm: true });
    expect(b).toContain('ptFrConductor(dwo, p.eta, p.k)');
    expect(b).toContain('ptFrConductor(c, p.eta, p.k)');
    expect(b).not.toContain('ptFrComplex(dwo, p.eta, p.k)');
    const l = ptLayeredWgsl({ layering: 'albedo', thinFilm: true });
    expect(l).toContain('ptFilmLayerR(t, dwo)');
    expect(l).toContain('ptFilmTopAlbedo(p, mu)');
    expect(l).toContain('ptFilmF(wo.z, p.ior)');
    expect(() => ptLayeredWgsl({ layering: 'stochastic', thinFilm: true })).toThrow(/albedo/);
    expect(ptThinFilmWgsl()).toContain('fn ptMaterialFilm');
  });
});
