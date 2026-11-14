import { createMaterial, type RtMaterial } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import {
  PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, PT_BSDF_LAYERED, PT_BSDF_THIN,
  PT_FLAG_DELTA, PT_FLAG_REFLECTION, PT_FLAG_TRANSMISSION, ptBsdfEval, ptBsdfIsDelta, ptBsdfSample, ptFrDielectric, ptThinReflectance, type PtBsdfParams, type PtVec3,
} from './bsdf.ts';
import { PT_NO_FEATURES, ptBsdfFromMaterial, ptBsdfOpenPbrWgsl, ptInternalReflectanceFit, ptLayeredFromBsdf, ptMaterialFeatures, ptMediumAfterTransmission, ptMediumSigma, type PtMaterialFeatures, type PtMaterialOptions } from './adapter.ts';
import { ptInternalReflectance, ptLayeredEval, ptLayeredSample } from './layered.ts';
import { rng } from './bsdf.testutil.ts';

const ALL: PtMaterialFeatures = { mix: true, thin: true, layered: true, medium: true };
const opts = (features: Partial<PtMaterialFeatures> = ALL, extra: Partial<PtMaterialOptions> = {}): PtMaterialOptions => ({ features: { ...PT_NO_FEATURES, ...features }, ...extra });
const thin = (ior: number, extra: Partial<RtMaterial> = {}): RtMaterial => createMaterial({ transmissionWeight: 1, thinWalled: true, ior, baseColor: [1, 1, 1], specularRoughness: 0, ...extra });

describe('thin-walled dielectric (PT_BSDF_THIN, pbrt ThinDielectricBxDF)', () => {
  const p = (ior: number, tint: PtVec3 = [1, 1, 1]): PtBsdfParams => ({ kind: PT_BSDF_THIN, reflectance: tint, eta: [0, 0, 0], k: [0, 0, 0], alpha: [0, 0], ior });

  it('normal incidence closed form: R\' = R + T^2 R / (1 - R^2) = 2R / (1 + R), R = ((n-1)/(n+1))^2', () => {
    for (const n of [1, 1.33, 1.5, 1.76, 2.42]) {
      const R = ((n - 1) / (n + 1)) ** 2;
      expect(ptThinReflectance(1, n)).toBeCloseTo((2 * R) / (1 + R), 14);
    }
    // 1.5: 0.0769 reflected, 0.9231 transmitted
    expect(ptThinReflectance(1, 1.5)).toBeCloseTo(0.0769230769, 9);
  });

  it('R + T = 1 at every angle with no absorption: the sample weights f |cos| / pdf sum, over the two events, to R\' + T\'', () => {
    for (const n of [1.2, 1.5, 2.42]) {
      for (const cos of [1, 0.8, 0.5, 0.2, 0.05]) {
        const wo: PtVec3 = [Math.sqrt(1 - cos * cos), 0, cos];
        const R = ptThinReflectance(cos, n);
        let sum = 0;
        let pdfSum = 0;
        for (const uc of [R / 2, R + (1 - R) / 2]) {
          const s = ptBsdfSample(p(n), wo, [uc, 0.3, 0.3]);
          expect(s.flags & PT_FLAG_DELTA).toBe(PT_FLAG_DELTA);
          sum += (s.f[0] * Math.abs(s.wi[2])) / s.pdf * s.pdf; // weight times probability of choosing it
          pdfSum += s.pdf;
        }
        expect(sum).toBeCloseTo(1, 12);
        expect(pdfSum).toBeCloseTo(1, 12);
      }
    }
  });

  it('transmission goes straight through (wi = -wo, etap 1, delta, no radiance scale); reflection mirrors', () => {
    const wo: PtVec3 = [0.6, 0, 0.8];
    const R = ptThinReflectance(0.8, 1.5);
    const t = ptBsdfSample(p(1.5), wo, [0.99, 0.5, 0.5]);
    expect(t.flags).toBe(PT_FLAG_TRANSMISSION | PT_FLAG_DELTA);
    expect(t.wi).toEqual([-0.6, -0, -0.8]);
    expect(t.etap).toBe(1);
    expect(t.pdf).toBeCloseTo(1 - R, 14);
    const r = ptBsdfSample(p(1.5), wo, [R / 2, 0.5, 0.5]);
    expect(r.flags).toBe(PT_FLAG_REFLECTION | PT_FLAG_DELTA);
    expect(r.wi).toEqual([-0.6, -0, 0.8]);
    // symmetric about the sheet: seen from behind it is the same
    const back = ptBsdfSample(p(1.5), [0.6, 0, -0.8], [0.99, 0.5, 0.5]);
    expect(back.wi[2]).toBeCloseTo(0.8, 14);
    expect(back.pdf).toBeCloseTo(1 - R, 14);
  });

  it('is a delta lobe: eval and pdf are 0, the integrator never does NEE on it', () => {
    expect(ptBsdfEval(p(1.5), [0, 0, 1], [0, 0, 1])).toEqual({ f: [0, 0, 0], pdf: 0 });
    expect(ptBsdfIsDelta(p(1.5))).toBe(true);
  });

  it('ior 1 is invisible: R = 0, the ray passes with weight 1', () => {
    const s = ptBsdfSample(p(1), [0.6, 0, 0.8], [0.5, 0.5, 0.5]);
    expect(s.pdf).toBe(1);
    expect(s.f[0] * Math.abs(s.wi[2])).toBeCloseTo(1, 14);
  });

  it('thin R\' is above a single interface R and the sheet at 80 deg is far brighter than at 0', () => {
    expect(ptThinReflectance(1, 1.5)).toBeGreaterThan(ptFrDielectric(1, 1.5));
    expect(ptThinReflectance(0.17, 1.5)).toBeGreaterThan(0.3);
  });

  it('a tint multiplies only the transmitted lobe', () => {
    const s = ptBsdfSample(p(1.5, [0.2, 0.5, 0.8]), [0, 0, 1], [0.999, 0, 0]);
    const T = 1 - ptThinReflectance(1, 1.5);
    expect(s.f[0] / (T / 1)).toBeCloseTo(0.2, 12);
    expect(s.f[2] / (T / 1)).toBeCloseTo(0.8, 12);
  });
});

describe('OpenPBR adapter (P2.21 / P2.22 / P2.30)', () => {
  it('without options it is the legacy adapter: a non-metal with a specular lobe is pure Lambert', () => {
    const m = createMaterial({ baseColor: [0.5, 0.5, 0.5], specularWeight: 1, ior: 1.5, specularRoughness: 0.3 });
    expect(ptBsdfFromMaterial(m).kind).toBe(PT_BSDF_DIFFUSE);
    expect(ptBsdfFromMaterial(m, opts(PT_NO_FEATURES)).kind).toBe(PT_BSDF_DIFFUSE);
  });

  it('pure metal: complex Fresnel from the base colour (eta 1, Gulbrandsen k), alpha = roughness^2; metal wins over transmission', () => {
    const m = createMaterial({ baseColor: [0.9, 0.6, 0.2], metalness: 1, specularRoughness: 0.5, transmissionWeight: 1 });
    const p = ptBsdfFromMaterial(m, opts());
    expect(p.kind).toBe(PT_BSDF_CONDUCTOR);
    expect(p.eta).toEqual([1, 1, 1]);
    expect(p.alpha[0]).toBeCloseTo(0.25, 12);
    // normal-incidence reflectance of the conductor is the base colour: FrComplex(1, 1, k) = ((1-1)^2 + k^2)/((1+1)^2 + k^2) = k^2/(4+k^2) = r
    for (let i = 0; i < 3; i++) expect((p.k[i]! ** 2) / (4 + p.k[i]! ** 2)).toBeCloseTo(m.baseColor[i]!, 5);
  });

  it('fractional metalness is a stochastic mix with probability metalness: the expectation is the exact linear blend', () => {
    const m = createMaterial({ baseColor: [0.6, 0.6, 0.6], metalness: 0.3, specularWeight: 0, specularRoughness: 0.4 });
    const f = opts({ mix: true });
    let metals = 0;
    const n = 10000;
    for (let i = 0; i < n; i++) if (ptBsdfFromMaterial(m, f, (i + 0.5) / n).kind === PT_BSDF_CONDUCTOR) metals++;
    expect(metals / n).toBeCloseTo(0.3, 3);
    // ends are exact for every u
    for (const u of [0, 0.5, 0.999999]) {
      expect(ptBsdfFromMaterial(createMaterial({ metalness: 1, specularWeight: 0 }), f, u).kind).toBe(PT_BSDF_CONDUCTOR);
      expect(ptBsdfFromMaterial(createMaterial({ metalness: 0, specularWeight: 0 }), f, u).kind).toBe(PT_BSDF_DIFFUSE);
    }
    // without `mix` the legacy 0.5 threshold holds
    expect(ptBsdfFromMaterial(m, opts({ mix: false }), 0.1).kind).toBe(PT_BSDF_DIFFUSE);
  });

  it('thick transmission is a dielectric; thin-walled (and the feature) is the thin lobe with the transmission colour as tint', () => {
    const glass = createMaterial({ transmissionWeight: 1, ior: 1.33, specularRoughness: 0.2, transmissionColor: [0.5, 0.6, 0.7] });
    const g = ptBsdfFromMaterial(glass, opts());
    expect(g.kind).toBe(PT_BSDF_DIELECTRIC);
    expect(g.ior).toBe(1.33);
    const t = ptBsdfFromMaterial(thin(1.5, { transmissionColor: [0.5, 0.6, 0.7] }), opts());
    expect(t.kind).toBe(PT_BSDF_THIN);
    expect(t.reflectance[0]).toBeCloseTo(0.5, 4);
    expect(ptBsdfFromMaterial(thin(1.5), opts({ thin: false })).kind).toBe(PT_BSDF_DIELECTRIC);
    expect(ptBsdfFromMaterial(thin(1.5, { transmissionColor: [0.5, 0.5, 0.5] }), opts(ALL, { transmissionTint: false })).reflectance).toEqual([1, 1, 1]);
  });

  it('dielectric over diffuse is the layered lobe: interface (ior, roughness^2, Fresnel scale specularColor * specularWeight) over the base', () => {
    const m = createMaterial({ baseColor: [0.5, 0.02, 0.01], baseWeight: 1, specularRoughness: 0.44, ior: 1.5, specularWeight: 0.25, specularColor: [1, 0.5, 1] });
    const p = ptBsdfFromMaterial(m, opts({ layered: true }, { layerThickness: 1e-6 }));
    expect(p.kind).toBe(PT_BSDF_LAYERED);
    const l = ptLayeredFromBsdf(p);
    expect(l.baseColor).toEqual([0.5, 0.02, 0.01]);
    expect(l.ior).toBe(1.5);
    expect(l.alpha[0]).toBeCloseTo(0.44 * 0.44, 12);
    expect(l.specularColor).toEqual([0.25, 0.125, 0.25]);
    expect(l.thickness).toBe(1e-6);
    // ior 1 and specular 0 are plain Lambert / black
    expect(ptBsdfFromMaterial(createMaterial({ ior: 1, specularWeight: 1 }), opts({ layered: true })).kind).toBe(PT_BSDF_DIFFUSE);
    expect(ptBsdfFromMaterial(createMaterial({ specularWeight: 0 }), opts({ layered: true })).kind).toBe(PT_BSDF_DIFFUSE);
    expect(ptBsdfFromMaterial(createMaterial({ baseColor: [0, 0, 0], specularWeight: 0 }), opts({ layered: true })).kind).toBe(PT_BSDF_BLACK);
    // a black base with a specular lobe is still a lobe (IORTestGrid black spheres)
    expect(ptBsdfFromMaterial(createMaterial({ baseColor: [0, 0, 0], ior: 1.76, specularRoughness: 0 }), opts({ layered: true })).kind).toBe(PT_BSDF_LAYERED);
  });

  it('a coat (weight > 0.5) replaces the interface by the coat and drops the base specular lobe (what the pbrt exporter writes)', () => {
    const m = createMaterial({ baseColor: [0.5, 0.02, 0.01], specularRoughness: 0.44, coatWeight: 1, coatRoughness: 0.03, coatIor: 1.5, specularWeight: 0.3 });
    const l = ptLayeredFromBsdf(ptBsdfFromMaterial(m, opts({ layered: true })));
    expect(l.alpha[0]).toBeCloseTo(0.03 * 0.03, 12);
    expect(l.specularColor).toEqual([1, 1, 1]);
  });

  it('albedoBase internal fills the internal reflectance from the closed-form fit; openpbr leaves it 0', () => {
    const m = createMaterial({ ior: 1.5 });
    expect(ptBsdfFromMaterial(m, opts({ layered: true }, { layered: { layering: 'albedo', albedoBase: 'internal' } })).k[1]).toBeCloseTo(ptInternalReflectanceFit(1.5), 12);
    expect(ptBsdfFromMaterial(m, opts({ layered: true })).k[1]).toBe(0);
  });

  it('the internal-reflectance fit is within 0.012 of the numeric average for ior 1.05 to 2.6', () => {
    let worst = 0;
    for (let ior = 1.05; ior <= 2.6; ior += 0.05) worst = Math.max(worst, Math.abs(ptInternalReflectanceFit(ior) - ptInternalReflectance(ior)));
    expect(worst).toBeLessThan(0.012);
  });

  it('the layered mapping evaluates like the layered BSDF it names (black base: exactly the dielectric reflection lobe)', () => {
    const m = createMaterial({ baseColor: [0, 0, 0], ior: 1.5, specularRoughness: 0.4 });
    const p = ptBsdfFromMaterial(m, opts({ layered: true }));
    const e = ptLayeredEval(ptLayeredFromBsdf(p), [0.3, 0, 0.954], [-0.2, 0.1, 0.973], { layering: 'albedo' });
    expect(e.f[0]).toBeGreaterThan(0);
  });
});

describe('stochastic layered sample keeps f and pdf inside f32 range (P2.21 wiring)', () => {
  it('a near-smooth rough interface multiplies f and pdf by ~1e4 per bounce: the sample rescales them, so the GPU does not overflow to inf / inf = NaN', () => {
    const random = rng(99);
    const l = { baseColor: [0.9, 0.9, 0.9] as PtVec3, ior: 1.97, alpha: [0.002, 0.002] as const, specularWeight: 1, specularColor: [1, 1, 1] as PtVec3, thickness: 1e-6, internalReflectance: 0 };
    let worst = 1;
    let samples = 0;
    for (let i = 0; i < 20000; i++) {
      const wo: PtVec3 = [0.3, 0.1, Math.sqrt(1 - 0.1)];
      const s = ptLayeredSample(l, wo, [random(), random(), random()], { layering: 'stochastic', maxDepth: 64, exitMis: 'pbrt', seed: i });
      if (s.flags === 0) continue;
      samples++;
      expect(Number.isFinite(s.pdf) && s.pdf > 0 && s.f.every(Number.isFinite)).toBe(true);
      worst = Math.max(worst, s.pdf, 1 / s.pdf);
    }
    expect(samples).toBeGreaterThan(1000);
    expect(worst).toBeLessThan(1e22);
  });
});

describe('material features of a table', () => {
  it('names exactly the lobes the table needs', () => {
    expect(ptMaterialFeatures([createMaterial({ specularWeight: 0 })])).toEqual(PT_NO_FEATURES);
    expect(ptMaterialFeatures([createMaterial({ metalness: 1 })])).toEqual(PT_NO_FEATURES);
    expect(ptMaterialFeatures([createMaterial({ metalness: 0.4 })])).toEqual({ ...PT_NO_FEATURES, mix: true, layered: true });
    expect(ptMaterialFeatures([thin(1.5)])).toEqual({ ...PT_NO_FEATURES, thin: true });
    expect(ptMaterialFeatures([createMaterial({ transmissionWeight: 1 })])).toEqual(PT_NO_FEATURES);
    expect(ptMaterialFeatures([createMaterial({ transmissionWeight: 1, transmissionDepth: 0.5, transmissionColor: [0.5, 0.5, 0.5] })])).toEqual({ ...PT_NO_FEATURES, medium: true });
    expect(ptMaterialFeatures([createMaterial({ baseColor: [0.8, 0.8, 0.8] })])).toEqual({ ...PT_NO_FEATURES, layered: true });
    expect(ptMaterialFeatures([createMaterial({ coatWeight: 1 }), thin(1.5)])).toEqual({ ...PT_NO_FEATURES, layered: true, thin: true });
  });

  it('the WGSL text carries only the requested lobes', () => {
    const none = ptBsdfOpenPbrWgsl(opts(PT_NO_FEATURES));
    expect(none).not.toMatch(/PT_BSDF_THIN|PT_BSDF_LAYERED|ptMediumSigma|ptInternalReflectanceFit/);
    expect(ptBsdfOpenPbrWgsl(opts({ thin: true }))).toContain('PT_BSDF_THIN');
    expect(ptBsdfOpenPbrWgsl(opts({ layered: true }))).toContain('PT_BSDF_LAYERED');
    expect(ptBsdfOpenPbrWgsl(opts({ medium: true }))).toContain('ptMediumSigma');
    expect(ptBsdfOpenPbrWgsl(opts({ mix: true }))).toContain('u < m.metalness');
  });
});

describe('absorbing interior medium (P2.22): Beer-Lambert', () => {
  it('sigma = -ln(transmissionColor) / transmissionDepth: the colour is the transmittance at that depth', () => {
    const m = createMaterial({ transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 0.5 });
    const s = ptMediumSigma(m);
    for (let i = 0; i < 3; i++) expect(Math.exp(-s[i]! * 0.5)).toBeCloseTo(m.transmissionColor[i]!, 12);
    expect(ptMediumSigma(createMaterial({ transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 0 }))).toEqual([0, 0, 0]);
    expect(ptMediumSigma(thin(1.5, { transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 1 }))).toEqual([0, 0, 0]);
    expect(ptMediumSigma(createMaterial({ transmissionWeight: 1, transmissionColor: [0, 1, 1], transmissionDepth: 1 }))[0]).toBeCloseTo(-Math.log(1e-6), 9);
  });

  it('a ray that enters (it came from outside, wo.z > 0) takes the material\'s absorber; one that exits (wo.z < 0) leaves it', () => {
    const m = createMaterial({ transmissionWeight: 1, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 0.5 });
    expect(ptMediumAfterTransmission(m, 0.7)).toEqual(ptMediumSigma(m));
    expect(ptMediumAfterTransmission(m, -0.7)).toEqual([0, 0, 0]);
  });

  /**
   * A face-on slab (two smooth dielectric interfaces, thickness d), unit radiance behind it and none in front, camera ray at normal incidence.
   * Path simulation with the same BSDF sampler and the same medium rule as the integrator: attenuate by exp(-sigma t) at the next hit, switch
   * the medium on a transmission. Closed form of the pixel: T^2 e^(-s d) / (1 - R^2 e^(-2 s d)) with R = ((n-1)/(n+1))^2 and T = 1 - R.
   */
  it('through a slab the simulated radiance matches T^2 e^(-sigma d) / (1 - R^2 e^(-2 sigma d)) per channel (Beer ordering: more depth is darker)', () => {
    const n = 1.5;
    const R = ((n - 1) / (n + 1)) ** 2;
    const T = 1 - R;
    const d = 0.5;
    const glass = createMaterial({ transmissionWeight: 1, ior: n, specularRoughness: 0, transmissionColor: [0.1, 0.5, 0.9], transmissionDepth: 0.25 });
    const p = ptBsdfFromMaterial(glass, opts());
    const sigma = ptMediumSigma(glass);
    const random = rng(7);
    const paths = 300000;
    const sum = [0, 0, 0];
    for (let k = 0; k < paths; k++) {
      const beta = [1, 1, 1];
      let top = true; // the face being hit: the top (outward +z, first hit) or the bottom (outward -z)
      let dz = -1; // heading toward the back
      let t = 0; // distance travelled since the last hit (the first ray starts outside, in vacuum)
      let med: PtVec3 = [0, 0, 0];
      for (let depth = 0; depth < 200; depth++) {
        for (let c = 0; c < 3; c++) beta[c] = beta[c]! * Math.exp(-med[c]! * t);
        const outward = top ? 1 : -1;
        const wo: PtVec3 = [0, 0, -dz * outward];
        const s = ptBsdfSample(p, wo, [random(), random(), random()]);
        if (s.flags === 0) break;
        const transmit = (s.flags & PT_FLAG_TRANSMISSION) !== 0;
        const w = ((s.f[0] * Math.abs(s.wi[2])) / s.pdf) * (transmit ? s.etap * s.etap : 1); // radiance transport: undo 1/etap^2 over entry and exit
        for (let c = 0; c < 3; c++) beta[c] = beta[c]! * w;
        if (transmit) med = ptMediumAfterTransmission(glass, wo[2]);
        dz = s.wi[2] * outward > 0 ? 1 : -1;
        const inside = (top && dz < 0) || (!top && dz > 0);
        if (!inside) {
          if (!top) for (let c = 0; c < 3; c++) sum[c]! += beta[c]!; // left through the back: unit radiance
          break; // left through the front (or reflected off the outside of the front): no radiance
        }
        top = !top;
        t = d;
      }
    }
    for (let c = 0; c < 3; c++) {
      const x = Math.exp(-sigma[c]! * d);
      const expected = (T * T * x) / (1 - R * R * x * x);
      if (process.env.PT_VERBOSE) console.log(`slab channel ${c}: simulated ${(sum[c]! / paths).toFixed(5)} closed form ${expected.toFixed(5)}`);
      expect(sum[c]! / paths).toBeCloseTo(expected, 2);
    }
    // ordering: channels with a smaller transmittance are darker
    expect(sum[0]!).toBeLessThan(sum[1]!);
    expect(sum[1]!).toBeLessThan(sum[2]!);
  });
});
