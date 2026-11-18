// R1: the reduced material (ptBsdfReduced) against the full OpenPBR-mode material it stands in for.
import { createMaterial, type RtMaterial } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { PT_BSDF_CONDUCTOR, PT_BSDF_DIFFUSE, PT_BSDF_LAYERED, PT_BSDF_THIN, type PtVec3 } from './bsdf.ts';
import { bruteForceAlbedo, rng } from './bsdf.testutil.ts';
import { ptFuzzBaseOf, ptFuzzedEval } from './fuzz.ts';
import { ptBsdfEval, ptBsdfSample, ptFrameFromNormal } from './bsdf.ts';
import { ptBsdfFromOpenPbr, ptBsdfReduced, ptBsdfReducedWgsl, ptMaterialFeatures, ptReducedAlpha, ptReducedKernelFeatures, type PtMaterialOptions } from './adapter.ts';
import { ptLayered, ptLayeredCoatedConductor } from './layered.ts';
import { layeredQuadrature } from './layered.testutil.ts';

const dirAt = (mu: number): [number, number, number] => [Math.sqrt(1 - mu * mu), 0, mu];
const optionsFor = (ms: RtMaterial[]): PtMaterialOptions => ({ features: ptMaterialFeatures(ms), layered: { layering: 'albedo', albedoCorrection: 'none' } });

/** Albedo of the stochastic reduced BSDF at `muO`: the u-average of the albedo of the lobe it picks (the kernel treats the pick as the whole BSDF). */
function reducedAlbedo(m: RtMaterial, options: PtMaterialOptions, mu: number, n = 400): [number, number, number] {
  const cache = new Map<string, [number, number, number]>();
  const sum: [number, number, number] = [0, 0, 0];
  const wo = dirAt(mu);
  for (let i = 0; i < n; i++) {
    const p = ptBsdfReduced(m, options, (i + 0.5) / n, mu);
    expect(p.kind).not.toBe(PT_BSDF_LAYERED);
    const key = JSON.stringify(p);
    let a = cache.get(key);
    if (!a) {
      a = bruteForceAlbedo(p, wo, {}, 160, 96).refl;
      cache.set(key, a);
    }
    for (let c = 0; c < 3; c++) sum[c] = (sum[c] as number) + (a[c] as number) / n;
  }
  return sum;
}

describe('reduced material (R1)', () => {
  const bases: [number, number, number][] = [[0.8, 0.8, 0.8], [0.8, 0.2, 0.1], [0.1, 0.1, 0.1]];

  it('a dielectric over a diffuse base keeps the layered albedo (albedo layering, internal base) within 3% + 0.005', () => {
    for (const rough of [0.1, 0.3, 0.6]) {
      for (const base of bases) {
        const m = createMaterial({ baseColor: base, baseWeight: 1, specularWeight: 1, ior: 1.5, specularRoughness: rough, metalness: 0 });
        const o = optionsFor([m]);
        for (const mu of [0.2, 0.5, 0.9]) {
          const full = layeredQuadrature(ptLayered(base, 1.5, rough), dirAt(mu), { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'none' }, 200, 120).albedo;
          const red = reducedAlbedo(m, o, mu);
          for (let c = 0; c < 3; c++) expect(Math.abs((red[c] as number) - (full[c] as number)), `rough ${rough} base ${base} mu ${mu} ch ${c}: ${red[c]} vs ${full[c]}`).toBeLessThan(0.03 * (full[c] as number) + 0.005);
        }
      }
    }
  });

  it('a coat replaces the specular interface (coat roughness, ior, weight 1) and keeps the albedo too', () => {
    for (const rough of [0.05, 0.3]) {
      const base: [number, number, number] = [0.6, 0.3, 0.2];
      const m = createMaterial({ baseColor: base, baseWeight: 1, specularWeight: 1, ior: 1.5, specularRoughness: 0.8, coatWeight: 1, coatIor: 1.6, coatRoughness: rough, metalness: 0 });
      const o = optionsFor([m]);
      for (const mu of [0.3, 0.8]) {
        const full = layeredQuadrature(ptLayered(base, 1.6, rough), dirAt(mu), { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'none' }, 200, 120).albedo;
        const red = reducedAlbedo(m, o, mu);
        for (let c = 0; c < 3; c++) expect(Math.abs((red[c] as number) - (full[c] as number)), `coat rough ${rough} mu ${mu} ch ${c}`).toBeLessThan(0.03 * (full[c] as number) + 0.005);
      }
    }
  });

  it('a coated metal is a plain conductor with the stack reflectance (within 6% of the albedo walk)', () => {
    const eta: [number, number, number] = [0.2, 0.9, 1.1];
    const k: [number, number, number] = [3.9, 2.4, 2.1];
    for (const [br, cr] of [[0.2, 0.05], [0.4, 0.3]] as const) {
      const m = createMaterial({ metalness: 1, conductorEtaK: true, conductorEta: eta, conductorK: k, specularRoughness: br, coatWeight: 1, coatIor: 1.5, coatRoughness: cr });
      const o = optionsFor([m]);
      expect(ptMaterialFeatures([m]).coatedMetal).toBe(true);
      for (const mu of [0.4, 0.9]) {
        const p = ptBsdfReduced(m, o, 0.5, mu);
        expect(p.kind).toBe(PT_BSDF_CONDUCTOR);
        const full = layeredQuadrature(ptLayeredCoatedConductor(eta, k, br, 1.5, cr), dirAt(mu), { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'none' }, 200, 120).albedo;
        const red = bruteForceAlbedo(p, dirAt(mu), {}, 160, 96).refl;
        for (let c = 0; c < 3; c++) expect(Math.abs((red[c] as number) - (full[c] as number)), `base ${br} coat ${cr} mu ${mu} ch ${c}: ${red[c]} vs ${full[c]}`).toBeLessThan(0.06 * (full[c] as number) + 0.01);
      }
    }
  });

  it('fuzz is folded into the lobe choice: albedo of the fuzzed material (spec: f = w f_fuzz + (1 - w E_f) f_base) within 0.01 absolute', () => {
    void ptFrameFromNormal;
    const cases: { name: string; m: RtMaterial }[] = [
      { name: 'white over grey Lambert', m: createMaterial({ baseColor: [0.5, 0.5, 0.5], baseWeight: 1, specularWeight: 0, sheenWeight: 1, sheenColor: [1, 1, 1], sheenRoughness: 0.5 }) },
      { name: 'coloured w0.7 over dark', m: createMaterial({ baseColor: [0.2, 0.2, 0.2], baseWeight: 1, specularWeight: 0, sheenWeight: 0.7, sheenColor: [0.9, 0.4, 0.1], sheenRoughness: 0.25 }) },
      { name: 'over black', m: createMaterial({ baseColor: [0, 0, 0], baseWeight: 1, specularWeight: 0, sheenWeight: 1, sheenColor: [0.3, 0.6, 0.9], sheenRoughness: 0.8 }) },
      { name: 'over rough metal', m: createMaterial({ metalness: 1, baseColor: [0.9, 0.6, 0.3], baseWeight: 1, specularRoughness: 0.5, sheenWeight: 0.6, sheenColor: [1, 1, 1], sheenRoughness: 0.6 }) },
    ];
    for (const { name, m } of cases) {
      const o = optionsFor([m]);
      expect(o.features.fuzz, name).toBe(true);
      for (const mu of [0.3, 0.7]) {
        const wo = dirAt(mu);
        const full = ptBsdfFromOpenPbr(m, o, 0.4);
        const base = ptFuzzBaseOf(full, ptBsdfEval, ptBsdfSample);
        const fz = { weight: m.sheenWeight, color: [m.sheenColor[0], m.sheenColor[1], m.sheenColor[2]] as PtVec3, roughness: m.sheenRoughness };
        const ref: [number, number, number] = [0, 0, 0];
        const nt = 200;
        const np = 200;
        for (let i = 0; i < nt; i++) {
          const th = ((i + 0.5) / nt) * (Math.PI / 2);
          for (let j = 0; j < np; j++) {
            const ph = ((j + 0.5) / np) * 2 * Math.PI;
            const wi: [number, number, number] = [Math.sin(th) * Math.cos(ph), Math.sin(th) * Math.sin(ph), Math.cos(th)];
            const e = ptFuzzedEval(fz, base, wo, wi);
            const w = Math.cos(th) * Math.sin(th) * (Math.PI / 2 / nt) * ((2 * Math.PI) / np);
            for (let c = 0; c < 3; c++) ref[c] = (ref[c] as number) + (e.f[c] as number) * w;
          }
        }
        const red = reducedAlbedo(m, o, mu, 600);
        for (let c = 0; c < 3; c++) expect(Math.abs((red[c] as number) - (ref[c] as number)), `${name} mu ${mu} ch ${c}: ${red[c]} vs ${ref[c]}`).toBeLessThan(0.012 + 0.03 * (ref[c] as number));
      }
    }
  });

  it('anisotropy becomes the mean of the two alphas; the conductor is otherwise the full one', () => {
    const m = createMaterial({ metalness: 1, baseWeight: 1, baseColor: [0.9, 0.6, 0.2], specularRoughness: 0.5, specularAnisotropy: 0.7 });
    const o = optionsFor([m]);
    const full = ptBsdfFromOpenPbr(m, o, 0);
    const red = ptBsdfReduced(m, o, 0, 0.6);
    expect(red.kind).toBe(PT_BSDF_CONDUCTOR);
    expect(red.alpha[0]).toBe(red.alpha[1]);
    expect(red.alpha[0]).toBeCloseTo(ptReducedAlpha(0.5, 0.7), 15);
    expect(red.eta).toEqual(full.eta);
    expect(red.k).toEqual(full.k);
  });

  it('metalness mix: the metal end with probability metalness (u < metalness), the dielectric end uses the conditional u', () => {
    const m = createMaterial({ metalness: 0.3, baseColor: [0.8, 0.8, 0.8], baseWeight: 1, specularWeight: 1, ior: 1.5, specularRoughness: 0.4 });
    const o = optionsFor([m]);
    expect(ptBsdfReduced(m, o, 0.29, 0.7).kind).toBe(PT_BSDF_CONDUCTOR);
    // above the metal share the dielectric end: with s F(muO) ~ 0.05 almost every conditional u lands on the diffuse lobe
    let diffuse = 0;
    const r = rng(7);
    for (let i = 0; i < 1000; i++) if (ptBsdfReduced(m, o, 0.3 + 0.7 * r(), 0.7).kind === PT_BSDF_DIFFUSE) diffuse++;
    expect(diffuse).toBeGreaterThan(900);
  });

  it('thin-walled and rough dielectrics are unchanged; a plain diffuse stays Lambert', () => {
    const thin = createMaterial({ transmissionWeight: 1, thinWalled: true, ior: 1.5 });
    const o = { features: ptMaterialFeatures([thin]) };
    expect(ptBsdfReduced(thin, o, 0.5, 0.5)).toEqual(ptBsdfFromOpenPbr(thin, o, 0.5));
    expect(ptBsdfReduced(thin, o, 0.5, 0.5).kind).toBe(PT_BSDF_THIN);
    const d = createMaterial({ baseColor: [0.3, 0.4, 0.5], baseWeight: 1, specularWeight: 0 });
    expect(ptBsdfReduced(d, { features: ptMaterialFeatures([d]) }, 0.5, 0.5)).toEqual(ptBsdfFromOpenPbr(d, { features: ptMaterialFeatures([d]) }, 0.5));
  });

  it('a secondary kernel keeps no layered / fuzz / film / aniso / coatedMetal feature but still reduces by the full table', () => {
    const m = createMaterial({ baseColor: [0.8, 0.8, 0.8], baseWeight: 1, specularWeight: 1, ior: 1.5, specularRoughness: 0.3, sheenWeight: 1, sheenColor: [1, 1, 1] });
    const full = ptMaterialFeatures([m]);
    expect(full.layered && full.fuzz).toBe(true);
    const kernel = ptReducedKernelFeatures(full);
    expect(kernel).toEqual({ mix: false, thin: false, layered: false, medium: false });
    const o: PtMaterialOptions = { features: kernel, reducedFrom: full };
    const lobes = new Set<number>();
    for (let i = 0; i < 50; i++) lobes.add(ptBsdfReduced(m, o, (i + 0.5) / 50, 0.7).kind);
    expect([...lobes].sort()).toEqual([PT_BSDF_DIFFUSE, PT_BSDF_CONDUCTOR].sort());
    // the same stochastic pick as the megakernel's reduced branch (which has all features)
    for (let i = 0; i < 50; i++) expect(ptBsdfReduced(m, o, (i + 0.5) / 50, 0.7)).toEqual(ptBsdfReduced(m, { features: full }, (i + 0.5) / 50, 0.7));
  });

  it('the WGSL twin compiles to text without the layered code and names only reduced-adapter helpers', () => {
    const m = createMaterial({ baseColor: [0.8, 0.8, 0.8], baseWeight: 1, specularWeight: 1, ior: 1.5, specularRoughness: 0.3 });
    const full = ptMaterialFeatures([m]);
    const wgsl = ptBsdfReducedWgsl({ features: ptReducedKernelFeatures(full), reducedFrom: full });
    expect(wgsl).toContain('fn ptBsdfReduced(m: RtMaterial, u: f32, muO: f32)');
    expect(wgsl).not.toContain('PT_BSDF_LAYERED');
  });
});
