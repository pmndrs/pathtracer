// P2.26: anisotropic GGX conductors. The BSDF itself was always anisotropic in its local frame (alpha is a pair, the VNDF sampler stretches
// by it); what is new is the alpha pair from OpenPBR's anisotropy, the tangent frame (Gram-Schmidt, rotation), and the material adapter.
import { createMaterial } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { PT_BSDF_CONDUCTOR, ptBsdfEval, ptBsdfSample, ptFrameFromNormal, ptFrameFromTangent, type PtBsdfParams, type PtVec3 } from './bsdf.ts';
import { ptBsdfFromMaterial, ptBsdfAdapterWgsl, ptMaterialFeatures, ptOpenPbrAlpha, type PtMaterialOptions } from './adapter.ts';
import { ptBsdfWgsl } from './bsdf-wgsl.ts';
import { ptConductor } from './materials.ts';
import { bruteForceAlbedo, chiSquareTest, dirFromAngles, mcAlbedo, rng } from './bsdf.testutil.ts';

const ETA: PtVec3 = [0.143, 0.375, 1.442];
const K: PtVec3 = [3.983, 2.386, 1.603];
const metal = (roughness: number, anisotropy: number, extra: object = {}) =>
  createMaterial({ conductorEtaK: true, conductorEta: [...ETA] as [number, number, number], conductorK: [...K] as [number, number, number], metalness: 1, baseWeight: 1, specularRoughness: roughness, specularAnisotropy: anisotropy, specularWeight: 1, ...extra });
const aniso = (alpha: readonly [number, number]): PtBsdfParams => ({ ...ptConductor(ETA, K, 0.5), alpha });
const dot = (a: PtVec3, b: PtVec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const unit = (v: number[]): PtVec3 => { const l = Math.hypot(v[0]!, v[1]!, v[2]!); return [v[0]! / l, v[1]! / l, v[2]! / l]; };
const toLocal = (f: { t: PtVec3; b: PtVec3; n: PtVec3 }, v: PtVec3): PtVec3 => [dot(v, f.t), dot(v, f.b), dot(v, f.n)];

describe('OpenPBR alpha pair (ptOpenPbrAlpha)', () => {
  it('anisotropy 0 is r^2 on both axes, exactly', () => {
    for (const r of [0, 0.05, 0.3, 0.7, 1]) expect(ptOpenPbrAlpha(r, 0)).toEqual([r * r, r * r]);
  });
  it('alpha_x = r^2 sqrt(2 / (1 + (1 - a)^2)), alpha_y = (1 - a) alpha_x; a = 1 gives alpha_y 0; alpha_x^2 + alpha_y^2 = 2 r^4', () => {
    const [ax, ay] = ptOpenPbrAlpha(0.5, 0.6);
    expect(ax).toBeCloseTo(0.25 * Math.sqrt(2 / (1 + 0.16)), 14);
    expect(ay).toBeCloseTo(0.4 * ax, 14);
    expect(ax * ax + ay * ay).toBeCloseTo(2 * 0.25 * 0.25, 14);
    expect(ptOpenPbrAlpha(0.5, 1)[1]).toBe(0);
  });
});

describe('ptFrameFromTangent', () => {
  const normals: PtVec3[] = [unit([0, 0, 1]), unit([0.3, -0.5, 0.8]), unit([-0.7, 0.1, -0.4]), unit([1, 1, 0.2])];
  it('is orthonormal and right-handed with local +X along the tangent made orthogonal to n', () => {
    for (const n of normals) {
      const tangent: [number, number, number, number] = [0.4, 0.9, -0.2, 1];
      const f = ptFrameFromTangent(n, tangent);
      expect(dot(f.t, f.t)).toBeCloseTo(1, 12);
      expect(dot(f.b, f.b)).toBeCloseTo(1, 12);
      expect(dot(f.t, f.b)).toBeCloseTo(0, 12);
      expect(dot(f.t, n)).toBeCloseTo(0, 12);
      expect(dot(f.b, n)).toBeCloseTo(0, 12);
      const cross: PtVec3 = [f.t[1] * f.b[2] - f.t[2] * f.b[1], f.t[2] * f.b[0] - f.t[0] * f.b[2], f.t[0] * f.b[1] - f.t[1] * f.b[0]];
      expect(dot(cross, n)).toBeCloseTo(1, 12);
      // t is the Gram-Schmidt of the tangent: coplanar with (tangent, n), same side as the tangent
      const gs = unit([tangent[0] - n[0] * dot(n, [tangent[0], tangent[1], tangent[2]]), tangent[1] - n[1] * dot(n, [tangent[0], tangent[1], tangent[2]]), tangent[2] - n[2] * dot(n, [tangent[0], tangent[1], tangent[2]])]);
      expect(dot(f.t, gs)).toBeCloseTo(1, 12);
    }
  });
  it('a tangent that is zero or parallel to the normal falls back to the normal-only frame', () => {
    const n = unit([0.2, 0.3, 0.9]);
    expect(ptFrameFromTangent(n, [0, 0, 0, 1])).toEqual(ptFrameFromNormal(n));
    expect(ptFrameFromTangent(n, [n[0] * 2, n[1] * 2, n[2] * 2, 1])).toEqual(ptFrameFromNormal(n));
  });
  it('rotation turns the axis counter-clockwise from the tangent toward the bitangent cross(n, t) * w (glTF)', () => {
    const n: PtVec3 = [0, 0, 1];
    const f0 = ptFrameFromTangent(n, [1, 0, 0, 1], 0);
    expect(f0.t).toEqual([1, 0, 0]);
    const f90 = ptFrameFromTangent(n, [1, 0, 0, 1], Math.PI / 2);
    expect(f90.t[0]).toBeCloseTo(0, 12);
    expect(f90.t[1]).toBeCloseTo(1, 12);
    const f90m = ptFrameFromTangent(n, [1, 0, 0, -1], Math.PI / 2); // handedness -1: the bitangent is -y
    expect(f90m.t[1]).toBeCloseTo(-1, 12);
    const f45 = ptFrameFromTangent(n, [1, 0, 0, 1], Math.PI / 4);
    expect(f45.t[0]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(f45.t[1]).toBeCloseTo(Math.SQRT1_2, 12);
    expect(ptFrameFromTangent(n, [1, 0, 0, 1], 2 * Math.PI).t[0]).toBeCloseTo(1, 12);
  });
});

describe('anisotropic conductor (alpha pair in the tangent frame)', () => {
  it('anisotropy 0 through the adapter is the isotropic conductor, bit for bit', () => {
    const options: PtMaterialOptions = { features: { mix: false, thin: false, layered: false, medium: false, aniso: true } };
    const plain: PtMaterialOptions = { features: { mix: false, thin: false, layered: false, medium: false } };
    for (const r of [0.05, 0.3, 0.8]) {
      const m = metal(r, 0);
      const a = ptBsdfFromMaterial(m, options, 0);
      const b = ptBsdfFromMaterial(m, plain, 0);
      expect(a).toEqual(b);
      expect(a.alpha[0]).toBe(a.alpha[1]);
      const wo = dirFromAngles(40, 30);
      const wi = dirFromAngles(25, 200);
      expect(ptBsdfEval(a, wo, wi)).toEqual(ptBsdfEval(b, wo, wi));
      expect(ptBsdfSample(a, wo, [0.3, 0.6, 0.2])).toEqual(ptBsdfSample(b, wo, [0.3, 0.6, 0.2]));
    }
  });
  it('adapter: metal with anisotropy > 0 splits alpha (OpenPBR) only when features.aniso; materialFeatures sets the key only then', () => {
    const m = metal(0.5, 0.6);
    expect(ptBsdfFromMaterial(m, { features: { mix: false, thin: false, layered: false, medium: false, aniso: true } }, 0).alpha).toEqual(ptOpenPbrAlpha(m.specularRoughness, m.specularAnisotropy));
    expect(ptBsdfFromMaterial(m, { features: { mix: false, thin: false, layered: false, medium: false } }, 0).alpha[0]).toBe(ptBsdfFromMaterial(m, { features: { mix: false, thin: false, layered: false, medium: false } }, 0).alpha[1]);
    expect(ptMaterialFeatures([metal(0.5, 0)])).toEqual({ mix: false, thin: false, layered: false, medium: false });
    expect(ptMaterialFeatures([metal(0.5, 0.6)]).aniso).toBe(true);
    expect('aniso' in ptMaterialFeatures([createMaterial({ baseColor: [1, 1, 1], specularAnisotropy: 0.6 })])).toBe(false); // a dielectric has no anisotropic lobe
    expect(ptBsdfFromMaterial(m, { features: { mix: false, thin: false, layered: false, medium: false, aniso: true } }, 0).kind).toBe(PT_BSDF_CONDUCTOR);
  });
  it('swapping alpha_x and alpha_y is the same BSDF seen through a frame rotated by 90 degrees', () => {
    const n = unit([0.2, -0.3, 0.9]);
    const t: [number, number, number, number] = [0.8, 0.4, -0.1, 1];
    const f0 = ptFrameFromTangent(n, t, 0);
    const f90 = ptFrameFromTangent(n, t, Math.PI / 2);
    const A = aniso([0.12, 0.5]);
    const B = aniso([0.5, 0.12]);
    const r = rng(5);
    for (let i = 0; i < 200; i++) {
      const wo = unit([r() * 2 - 1, r() * 2 - 1, 0.1 + r()]);
      const wi = unit([r() * 2 - 1, r() * 2 - 1, 0.1 + r()]);
      // world directions: A in the unrotated frame equals B in the 90-degree frame? (x -> y): frame f90 has t = b0, so alpha_x of B lies along b0 = alpha_y of A.
      const a = ptBsdfEval(A, toLocal(f0, wo), toLocal(f0, wi));
      const b = ptBsdfEval(B, toLocal(f90, wo), toLocal(f90, wi));
      expect(b.f[0]).toBeCloseTo(a.f[0], 10);
      expect(b.pdf).toBeCloseTo(a.pdf, 10);
    }
  });
  it('pdf integrates to at most 1, and the sampler albedo equals the brute-force integral of eval', () => {
    for (const alpha of [[0.05, 0.4], [0.3, 0.3], [0.02, 0.2], [0.5, 0.1]] as const) {
      for (const th of [0, 40, 75]) {
        const p = aniso(alpha);
        const wo = dirFromAngles(th, 35);
        const bf = bruteForceAlbedo(p, wo);
        expect(bf.pdfRefl).toBeLessThanOrEqual(1 + 2e-3);
        const mc = mcAlbedo(p, wo, 60_000, 11);
        for (let c = 0; c < 3; c++) expect(Math.abs(mc.refl[c]! - bf.refl[c]!)).toBeLessThan(5 * mc.se[c]! + 2e-3);
      }
    }
  }, 120_000);
  it('reciprocity: f(wo, wi) = f(wi, wo)', () => {
    const p = aniso([0.06, 0.35]);
    const r = rng(8);
    for (let i = 0; i < 300; i++) {
      const wo = unit([r() * 2 - 1, r() * 2 - 1, 0.05 + r()]);
      const wi = unit([r() * 2 - 1, r() * 2 - 1, 0.05 + r()]);
      const ab = ptBsdfEval(p, wo, wi), ba = ptBsdfEval(p, wi, wo);
      for (let c = 0; c < 3; c++) expect(ab.f[c]).toBeCloseTo(ba.f[c]!, 9);
    }
  });
  it('the sampled directions follow the pdf (chi-square), standard and cap VNDF', () => {
    for (const vndf of ['standard', 'cap'] as const) {
      const res = chiSquareTest(aniso([0.08, 0.4]), dirFromAngles(50, 70), 200_000, 3, { vndf });
      expect(res.p).toBeGreaterThan(1e-4);
    }
  }, 120_000);
});

describe('WGSL gating', () => {
  it('ptFrameFromTangent and ptOpenPbrAlpha are compiled only with aniso', () => {
    expect(ptBsdfWgsl({})).not.toContain('ptFrameFromTangent');
    expect(ptBsdfWgsl({ aniso: true })).toContain('fn ptFrameFromTangent(');
    const off: PtMaterialOptions = { features: { mix: false, thin: false, layered: false, medium: false } };
    expect(ptBsdfAdapterWgsl(off)).not.toContain('ptOpenPbrAlpha');
    const on = ptBsdfAdapterWgsl({ features: { ...off.features, aniso: true } });
    expect(on).toContain('fn ptOpenPbrAlpha(');
    expect(on).toContain('p.alpha = ptOpenPbrAlpha(m.specularRoughness, m.specularAnisotropy);');
  });
});
