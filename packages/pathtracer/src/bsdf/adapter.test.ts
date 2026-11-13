import { createMaterial, defaultMaterial, roughnessToAlpha, type RtMaterial } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { PT_BSDF_BLACK, PT_BSDF_CONDUCTOR, PT_BSDF_DIELECTRIC, PT_BSDF_DIFFUSE, ptBsdfEval, ptFrComplex } from './bsdf.ts';
import { conductorKFromReflectance, ptBsdfAdapterWgsl, ptBsdfFromMaterial } from './adapter.ts';

// The materials `bench/scenes/to-scene-copy.ts` makes from the Tier 0 bench types (kept in sync by bench/adapters/pt-megakernel.test.ts).
const conductor = (roughness: number): RtMaterial => createMaterial({ conductorEtaK: true, conductorEta: [0.2, 0.9, 1.1], conductorK: [3.9, 2.4, 2.1], metalness: 1, baseWeight: 1, specularRoughness: roughness, specularWeight: 1 });
const dielectric = (ior: number, roughness: number): RtMaterial => createMaterial({ baseWeight: 0, baseColor: [0, 0, 0], transmissionWeight: 1, transmissionColor: [1, 1, 1], transmissionDepth: 0, specularWeight: 1, ior, specularRoughness: roughness });
const black = (): RtMaterial => createMaterial({ baseColor: [0, 0, 0], baseWeight: 0, specularWeight: 0 });

describe('RtMaterial -> PtBsdfParams (P2.2)', () => {
  it('diffuse: Lambert with baseColor * baseWeight', () => {
    const p = ptBsdfFromMaterial(createMaterial({ baseColor: [0.5, 0.25, 0.75], baseWeight: 0.5, specularWeight: 0 }));
    expect(p.kind).toBe(PT_BSDF_DIFFUSE);
    expect(p.reflectance).toEqual([0.25, 0.125, 0.375]);
    expect(ptBsdfEval(p, [0, 0, 1], [0, 0, 1]).f[0]).toBeCloseTo(0.25 / Math.PI, 12);
  });

  it('conductor: eta and k from the record, alpha = roughness^2 (isotropic)', () => {
    const m = conductor(0.7);
    const p = ptBsdfFromMaterial(m);
    expect(p.kind).toBe(PT_BSDF_CONDUCTOR);
    expect(p.eta).toEqual(m.conductorEta);
    expect(p.k).toEqual(m.conductorK);
    expect(p.alpha[0]).toBeCloseTo(roughnessToAlpha(0.7), 15);
    expect(p.alpha[1]).toBe(p.alpha[0]);
  });

  it('a roughness 0 conductor is smooth (delta) and a roughness 0 dielectric too', () => {
    expect(ptBsdfFromMaterial(conductor(0)).alpha).toEqual([0, 0]);
    expect(ptBsdfFromMaterial(dielectric(1.5, 0)).alpha).toEqual([0, 0]);
  });

  it('dielectric: ior and alpha from the record; wins over a conductor flag and over the base layer', () => {
    const p = ptBsdfFromMaterial(dielectric(1.5, 0.3));
    expect(p.kind).toBe(PT_BSDF_DIELECTRIC);
    expect(p.ior).toBe(1.5);
    expect(p.alpha[0]).toBeCloseTo(0.09, 15);
    expect(ptBsdfFromMaterial(createMaterial({ transmissionWeight: 1, metalness: 1, conductorEtaK: true, baseWeight: 1 })).kind).toBe(PT_BSDF_DIELECTRIC);
  });

  it('black: no base layer, nothing else', () => {
    expect(ptBsdfFromMaterial(black()).kind).toBe(PT_BSDF_BLACK);
    expect(ptBsdfFromMaterial(createMaterial({ baseColor: [0.9, 0.9, 0.9], baseWeight: 0 })).kind).toBe(PT_BSDF_BLACK);
  });

  it('thresholds: blend weights below 0.5 do not switch the lobe', () => {
    expect(ptBsdfFromMaterial(createMaterial({ transmissionWeight: 0.4, baseWeight: 1 })).kind).toBe(PT_BSDF_DIFFUSE);
    expect(ptBsdfFromMaterial(createMaterial({ metalness: 0.4, conductorEtaK: true, baseWeight: 1 })).kind).toBe(PT_BSDF_DIFFUSE);
    expect(ptBsdfFromMaterial(createMaterial({ metalness: 0.6, conductorEtaK: true })).kind).toBe(PT_BSDF_CONDUCTOR);
  });

  it('a metal without eta/k reflects its base colour at normal incidence (Gulbrandsen inverse)', () => {
    const p = ptBsdfFromMaterial(createMaterial({ metalness: 1, baseColor: [0.9, 0.5, 0.1], baseWeight: 1, conductorEtaK: false }));
    expect(p.kind).toBe(PT_BSDF_CONDUCTOR);
    [0.9, 0.5, 0.1].forEach((r, i) => expect(ptFrComplex(1, 1, p.k[i] as number)).toBeCloseTo(r, 9));
    expect(conductorKFromReflectance(0.5)).toBeCloseTo(2, 12);
    expect(Number.isFinite(conductorKFromReflectance(1))).toBe(true);
    expect(conductorKFromReflectance(0)).toBe(0);
  });

  it('the default OpenPBR material (0.8 grey dielectric) maps to a Lambertian for now', () => {
    expect(ptBsdfFromMaterial(defaultMaterial()).kind).toBe(PT_BSDF_DIFFUSE);
  });
});

describe('WGSL adapter', () => {
  const wgsl = ptBsdfAdapterWgsl();
  it('mirrors the TypeScript rules', () => {
    expect(wgsl).toContain('fn ptBsdfFromMaterial(m: RtMaterial) -> PtBsdfParams');
    expect(wgsl).toContain('m.transmissionWeight > 0.5');
    expect(wgsl).toContain('m.metalness > 0.5');
    expect(wgsl).toContain('m.conductorEtaK');
    expect(wgsl).toContain('m.specularRoughness * m.specularRoughness');
    expect(wgsl).toContain('0.9999');
    for (const kind of ['PT_BSDF_BLACK', 'PT_BSDF_DIELECTRIC', 'PT_BSDF_CONDUCTOR', 'PT_BSDF_DIFFUSE']) expect(wgsl).toContain(kind);
  });
  it('has balanced braces and no comments', () => {
    expect(wgsl.split('{').length).toBe(wgsl.split('}').length);
    expect(wgsl).not.toContain('//');
  });
});
