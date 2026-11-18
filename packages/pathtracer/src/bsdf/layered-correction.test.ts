import { describe, expect, it } from 'vitest';
import type { PtVec3 } from './bsdf.ts';
import { ptLayered, ptLayeredEval, ptLayeredSample, ptInternalReflectance, type PtLayeredOptions } from './layered.ts';
import { PT_LAYER_CORR_NM, PT_LAYER_CORR_NR, ptLayerCorrNeedsTexture, ptLayerCorrCoefs, ptLayerCorrection, ptLayerCorrLut, ptLayerCorrTextureHalf } from './layered-correction.ts';
import { PT_LAYER_CORR_DATA } from './layered-correction-data.ts';
import { ptLayeredWgsl } from './layered-wgsl.ts';
import { evalAlbedo, layeredQuadrature, sampleAlbedo } from './layered.testutil.ts';
import { f16BitsToF32 } from '@pmndrs/raytracer';

const INTERNAL: PtLayeredOptions = { layering: 'albedo', albedoBase: 'internal', albedoCorrection: 'none' };
/** D32: with nothing set, an albedo layering has the `texture` correction. */
const DEFAULTED: PtLayeredOptions = { layering: 'albedo' };
const LUT: PtLayeredOptions = { ...INTERNAL, albedoCorrection: 'lut' };
const TEX: PtLayeredOptions = { ...INTERNAL, albedoCorrection: 'texture' };
const woAt = (mu: number): PtVec3 => [Math.sqrt(1 - mu * mu), 0, mu];

describe('albedo correction table (P2.32b)', () => {
  it('has 8 x 8 nodes of (a, b, c); smooth rows are the identity and every node is plausible', () => {
    expect(PT_LAYER_CORR_DATA.length).toBe(PT_LAYER_CORR_NR * PT_LAYER_CORR_NM * 3);
    for (let ir = 0; ir < 2; ir++) for (let im = 0; im < PT_LAYER_CORR_NM; im++) expect(PT_LAYER_CORR_DATA.slice((ir * PT_LAYER_CORR_NM + im) * 3, (ir * PT_LAYER_CORR_NM + im) * 3 + 3)).toEqual([1, 0, 0]);
    for (let n = 0; n < PT_LAYER_CORR_NR * PT_LAYER_CORR_NM; n++) {
      const [a, b, c] = [PT_LAYER_CORR_DATA[n * 3] as number, PT_LAYER_CORR_DATA[n * 3 + 1] as number, PT_LAYER_CORR_DATA[n * 3 + 2] as number];
      expect(a).toBeGreaterThan(0.4);
      expect(a).toBeLessThan(1.1);
      expect(b).toBeLessThanOrEqual(0);
      expect(c).toBeLessThanOrEqual(0.001);
    }
  });

  it('the factor is at most 1, falls with the base and with roughness, and is 1 for a smooth or black-ish base at low roughness', () => {
    const ri = ptInternalReflectance(1.5);
    for (const mu of [0.1, 0.5, 1]) for (const r of [0.2, 0.5, 1]) for (const base of [0, 0.3, 0.6, 1]) expect(ptLayerCorrection('lut', base, ri, mu, r)).toBeLessThanOrEqual(1);
    expect(ptLayerCorrection('lut', 1, ri, 0.5, 0.8)).toBeLessThan(ptLayerCorrection('lut', 0.3, ri, 0.5, 0.8));
    expect(ptLayerCorrection('lut', 1, ri, 0.5, 1)).toBeLessThan(ptLayerCorrection('lut', 1, ri, 0.5, 0.5));
    expect(ptLayerCorrection('lut', 1, ri, 0.5, 0.02)).toBe(1);
    expect(ptLayerCorrection('lut', 0.8, ri, 0.7, 0)).toBe(1);
  });

  it('texture encoding = the table rounded to f16 (error < 1e-3 per number)', () => {
    const half = ptLayerCorrTextureHalf();
    const lut = ptLayerCorrLut();
    for (let n = 0; n < PT_LAYER_CORR_NR * PT_LAYER_CORR_NM; n++) {
      for (let k = 0; k < 3; k++) expect(Math.abs(f16BitsToF32(half[n * 4 + k] as number) - (lut[n * 3 + k] as number))).toBeLessThan(1e-3);
      expect(half[n * 4 + 3]).toBe(0);
    }
    for (const [mu, r] of [[0.3, 0.6], [0.9, 0.95], [0.05, 0.4]] as const) {
      const a = ptLayerCorrCoefs('lut', mu, r);
      const b = ptLayerCorrCoefs('texture', mu, r);
      for (let k = 0; k < 3; k++) expect(Math.abs((a[k] as number) - (b[k] as number))).toBeLessThan(1e-3);
    }
  });
});

describe("albedoCorrection against the stochastic walk (the fit's ground truth, held-out points)", () => {
  // Points between the nodes (r, mu, ior, base): the corrected albedo path must land within 6% of the fit target (r >= 0.33: the mean of the walk's sampler albedo and of its
  // pbrt-exit-MIS f albedo; below: the sampler albedo), where the closed form is off by far more.
  const points: [number, number, number, number][] = [
    [0.6, 0.45, 1.5, 1],
    [0.9, 0.2, 1.5, 0.8],
    [0.4, 0.75, 1.5, 0.9],
    [0.7, 0.6, 1.33, 1],
    [0.55, 0.35, 1.7, 0.7],
    [0.8, 0.85, 2, 0.9],
  ];
  it('total albedo within 6% (and 4 sigma) of the fit target, better than the plain internal base', () => {
    let errPlain = 0;
    let errCorr = 0;
    for (const [r, mu, ior, base] of points) {
      const p = ptLayered([base, base, base], ior, r, { thickness: 1e-6 });
      const walk = sampleAlbedo(p, woAt(mu), 60000, 11, { layering: 'stochastic', maxDepth: 64 });
      const ev = r >= 0.33 ? evalAlbedo(p, woAt(mu), 60000, 12, { layering: 'stochastic', maxDepth: 64, exitMis: 'pbrt' }) : walk;
      const w = 0.5 * (walk.mean[0] + ev.mean[0]);
      const se = 0.5 * (walk.se[0] + ev.se[0]);
      const plain = layeredQuadrature(p, woAt(mu), INTERNAL, 160, 144).albedo[0];
      const corr = layeredQuadrature(p, woAt(mu), LUT, 160, 144).albedo[0];
      expect(Math.abs(corr - w)).toBeLessThan(0.06 * w + 4 * se);
      errPlain += Math.abs(plain - w) / w;
      errCorr += Math.abs(corr - w) / w;
    }
    expect(errCorr).toBeLessThan(0.5 * errPlain);
  });

  it('texture encoding gives the same albedo as the lut one to 2e-3', () => {
    const p = ptLayered([0.9, 0.9, 0.9], 1.5, 0.7, { thickness: 1e-6 });
    const a = layeredQuadrature(p, woAt(0.5), LUT, 120, 108).albedo[0];
    const b = layeredQuadrature(p, woAt(0.5), TEX, 120, 108).albedo[0];
    expect(Math.abs(a - b)).toBeLessThan(2e-3);
  });
});

describe('albedoCorrection default (D32)', () => {
  it("an albedo layering with nothing set is the 'texture' correction; 'none', the stochastic walk and the openpbr base opt out", () => {
    const p = ptLayered([0.8, 0.5, 0.2], 1.5, 0.6, { thickness: 1e-6 });
    const wi: PtVec3 = [0.3, 0.2, Math.sqrt(1 - 0.13)];
    expect(ptLayeredEval(p, woAt(0.6), wi, DEFAULTED)).toEqual(ptLayeredEval(p, woAt(0.6), wi, TEX));
    expect(ptLayeredEval(p, woAt(0.6), wi, DEFAULTED).f[0]).toBeLessThan(ptLayeredEval(p, woAt(0.6), wi, INTERNAL).f[0]);
    expect(ptLayeredWgsl(DEFAULTED)).toBe(ptLayeredWgsl(TEX));
    expect(ptLayeredWgsl(DEFAULTED)).not.toBe(ptLayeredWgsl(INTERNAL));
    expect(ptLayeredWgsl({ layering: 'albedo', albedoBase: 'openpbr' })).not.toContain('ptLayerCorr');
    expect(ptLayeredWgsl({ layering: 'stochastic' })).not.toContain('ptLayerCorr');
    expect(ptLayerCorrNeedsTexture(undefined)).toBe(true);
    expect(ptLayerCorrNeedsTexture({ albedoCorrection: 'none' })).toBe(false);
    expect(ptLayerCorrNeedsTexture({ albedoCorrection: 'lut' })).toBe(false);
    expect(ptLayerCorrNeedsTexture({ layering: 'stochastic' })).toBe(false);
  });
});

describe('albedoCorrection keeps the layered BSDF well behaved', () => {
  it("'none' is exactly the plain internal base", () => {
    const p = ptLayered([0.8, 0.5, 0.2], 1.5, 0.6, { thickness: 1e-6 });
    const wi: PtVec3 = [0.3, 0.2, Math.sqrt(1 - 0.13)];
    expect(ptLayeredEval(p, woAt(0.6), wi, { ...INTERNAL, albedoCorrection: 'none' })).toEqual(ptLayeredEval(p, woAt(0.6), wi, INTERNAL));
  });

  it('white furnace: a white base under any rough clear coat reflects at most 1 (+0.01) with the correction, per cosine', () => {
    for (const ior of [1.33, 1.5, 2]) {
      for (const r of [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1]) {
        for (const mu of [0.1, 0.5, 0.9]) {
          const p = ptLayered([1, 1, 1], ior, r, { thickness: 1e-6 });
          expect(layeredQuadrature(p, woAt(mu), LUT, 120, 108).albedo[0]).toBeLessThan(1.01);
        }
      }
    }
  });

  it('the pdf integrates to at most 1, and the sampler returns eval at the sampled direction', () => {
    const p = ptLayered([0.9, 0.7, 0.5], 1.5, 0.6, { thickness: 1e-6 });
    const wo = woAt(0.55);
    expect(layeredQuadrature(p, wo, LUT, 160, 144).pdf).toBeLessThanOrEqual(1.002);
    for (let i = 0; i < 40; i++) {
      const s = ptLayeredSample(p, wo, [(i + 0.5) / 40, ((i * 7) % 40) / 40 + 0.01, ((i * 13) % 40) / 40 + 0.01], LUT);
      if (s.flags === 0) continue;
      const e = ptLayeredEval(p, wo, s.wi, LUT);
      expect(s.f[0]).toBeCloseTo(e.f[0], 9);
      expect(s.pdf).toBeCloseTo(e.pdf, 9);
    }
  });

  it('only the base lobe changes: a black base is the plain specular lobe', () => {
    const p = ptLayered([0, 0, 0], 1.5, 0.6, { thickness: 1e-6 });
    const wi: PtVec3 = [0.3, 0.2, Math.sqrt(1 - 0.13)];
    expect(ptLayeredEval(p, woAt(0.6), wi, LUT)).toEqual(ptLayeredEval(p, woAt(0.6), wi, INTERNAL));
  });
});

describe('albedoCorrection WGSL', () => {
  it('none emits nothing new; lut carries the table; texture the binding', () => {
    const plain = ptLayeredWgsl(INTERNAL);
    expect(plain).not.toContain('ptLayerCorr');
    expect(ptLayeredWgsl({ ...INTERNAL, albedoCorrection: 'none' })).toBe(plain);
    const lut = ptLayeredWgsl(LUT);
    expect(lut).toContain('PT_LAYER_CORR: array<f32, 192>');
    expect(lut).toContain('ptLayerCorr(p, mu)');
    expect(lut).toContain('ptLayerCorr(p, wo.z)');
    expect(lut).not.toContain('textureLoad');
    const tex = ptLayeredWgsl(TEX);
    expect(tex).toContain('@group(2) @binding(10) var ptLayerCorrTex: texture_2d<f32>;');
    expect(tex).toContain('textureLoad(ptLayerCorrTex');
    expect(tex).not.toContain('PT_LAYER_CORR:');
  });

  it("the correction needs the internal base: with 'openpbr' (or the stochastic walk) the text is unchanged", () => {
    expect(ptLayeredWgsl({ layering: 'albedo', albedoBase: 'openpbr', albedoCorrection: 'lut' })).toBe(ptLayeredWgsl({ layering: 'albedo', albedoBase: 'openpbr' }));
    expect(ptLayeredWgsl({ layering: 'stochastic', albedoCorrection: 'lut' })).toBe(ptLayeredWgsl({ layering: 'stochastic' }));
  });

  it('the const array carries the table (4 decimals at most)', () => {
    const lut = ptLayeredWgsl(LUT);
    const m = /PT_LAYER_CORR: array<f32, 192> = array<f32, 192>\(([^)]*)\)/.exec(lut);
    expect(m).not.toBeNull();
    const nums = (m?.[1] as string).split(',').map(Number);
    expect(nums.length).toBe(192);
    for (let i = 0; i < 192; i++) expect(nums[i]).toBeCloseTo(PT_LAYER_CORR_DATA[i] as number, 4);
  });
});
