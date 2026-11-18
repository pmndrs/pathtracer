import { describe, expect, it } from 'vitest';
import { buildRgb2SpecTable, reflectanceToRgb, rgb2SpecTexture, rgbToCoeffs, sigmoidSpectrum, SPEC_WHITE_XYZ, SPEC_XYZ_TO_RGB, xyzToRgb, cieAt, SPEC_YD, type Rgb2SpecTable, type Vec3 } from './rgb2spec.ts';
import { cauchyCoefficients, iorAt, laneData, laneRadianceToRgb, pathLambdas, visibleLambda, visiblePdf } from './wavelengths.ts';
import { ptSpectralWgsl, resolveSpectralOptions } from './spectral-wgsl.ts';

// res 32 fits in ~0.5 s; the kernel default is 64 (same method)
let table: Rgb2SpecTable;
const T = (): Rgb2SpecTable => (table ??= buildRgb2SpecTable(32));

const rng = (seed: number) => () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);

describe('rgb -> spectrum table', () => {
  it('maps the D65 white to exactly (1, 1, 1) and keeps Y = 1', () => {
    const rgb = xyzToRgb(SPEC_WHITE_XYZ);
    for (const v of rgb) expect(v).toBeCloseTo(1, 12);
    expect(SPEC_WHITE_XYZ[1]).toBe(1);
    expect(SPEC_YD).toBeGreaterThan(100);
    expect(SPEC_XYZ_TO_RGB).toHaveLength(9);
  });

  it('a grey is a constant spectrum', () => {
    for (const g of [0.02, 0.18, 0.5, 0.9]) {
      const c = rgbToCoeffs(T(), [g, g, g]);
      for (const l of [400, 500, 600, 700]) expect(sigmoidSpectrum(c, l)).toBeCloseTo(g, 6);
      for (const v of reflectanceToRgb((l) => sigmoidSpectrum(c, l))) expect(v).toBeCloseTo(g, 4);
    }
  });

  it('round-trips random colours of the cube (res 32: mean 1e-3, max 1e-2)', () => {
    const r = rng(7);
    let worst = 0;
    let sum = 0;
    const n = 600;
    for (let i = 0; i < n; i++) {
      const rgb: Vec3 = [r(), r(), r()];
      const back = reflectanceToRgb((l) => sigmoidSpectrum(rgbToCoeffs(T(), rgb), l));
      const e = Math.max(...rgb.map((v, k) => Math.abs(v - (back[k] as number))));
      worst = Math.max(worst, e);
      sum += e;
    }
    expect(worst).toBeLessThan(0.012);
    expect(sum / n).toBeLessThan(0.0015);
  });

  it('spectra stay in [0, 1] (energy conserving reflectances)', () => {
    const r = rng(11);
    for (let i = 0; i < 200; i++) {
      const c = rgbToCoeffs(T(), [r(), r(), r()]);
      for (let l = 360; l <= 830; l += 10) {
        const s = sigmoidSpectrum(c, l);
        expect(s).toBeGreaterThanOrEqual(0);
        expect(s).toBeLessThanOrEqual(1);
      }
    }
  });

  it('the kernel texture has the layout the WGSL reads (coefficient texel, CMF row)', () => {
    const tex = rgb2SpecTexture(T());
    expect(tex.width).toBe(32 * 32);
    expect(tex.height).toBe(3 * 32 + 1);
    const o = ((1 * 32 + 5) * tex.width + 3 + 7 * 32) * 4; // m = 1, k = 5, x = 3, y = 7
    const src = (((1 * 32 + 5) * 32 + 7) * 32 + 3) * 3;
    expect(tex.data[o]).toBe(T().data[src]);
    expect(tex.data[o + 2]).toBe(T().data[src + 2]);
    const row = 3 * 32 * tex.width * 4;
    expect(tex.data[row + 4 * 20 + 1]).toBeCloseTo(cieAt(460)[1], 6); // ybar at 460 nm (index 20)
    expect(tex.data[row + 4 * 20 + 3]).toBeCloseTo(cieAt(460)[3] / SPEC_YD, 8);
  });

  it('the table needs res >= 10 (the CMF row)', () => {
    expect(() => buildRgb2SpecTable(9)).toThrow(/res/);
  });
});

describe('hero wavelengths', () => {
  it('the density integrates to 1 and the inverse CDF matches it', () => {
    let s = 0;
    for (let l = 360.05; l < 830; l += 0.1) s += visiblePdf(l) * 0.1;
    expect(s).toBeCloseTo(1, 3);
    // F(F^-1(u)) = u by a numeric CDF
    const cdf = (x: number): number => {
      let a = 0;
      for (let l = 360.05; l < x; l += 0.1) a += visiblePdf(l) * 0.1;
      return a;
    };
    for (const u of [0.1, 0.4, 0.75, 0.95]) expect(cdf(visibleLambda(u))).toBeCloseTo(u, 2);
  });

  it('the three lanes are one number rotated by thirds', () => {
    const [a, b, c] = pathLambdas(0.2);
    expect(a).toBeCloseTo(visibleLambda(0.2), 9);
    expect(b).toBeCloseTo(visibleLambda(0.2 + 1 / 3), 6);
    expect(c).toBeCloseTo(visibleLambda(0.2 + 2 / 3), 6);
  });

  it('the estimator is unbiased: a white emitter averages to (1, 1, 1), a coloured one to its RGB', () => {
    const tab = T();
    const emitters: Vec3[] = [[1, 1, 1], [0.8, 0.2, 0.05], [0.1, 0.6, 0.9]];
    const n = 20000;
    for (const rgb of emitters) {
      const m = Math.max(...rgb) * 2;
      const c = rgbToCoeffs(tab, rgb.map((v) => v / m));
      const sum = [0, 0, 0];
      for (let i = 0; i < n; i++) {
        const d = laneData((i + 0.5) / n); // stratified u: low noise
        const L = [0, 1, 2].map((k) => m * sigmoidSpectrum(c, d.lambda[k] as number) * (d.d65[k] as number)) as Vec3;
        const o = laneRadianceToRgb(L, d);
        sum[0]! += o[0] / n; sum[1]! += o[1] / n; sum[2]! += o[2] / n;
      }
      for (let k = 0; k < 3; k++) expect(sum[k]).toBeCloseTo(rgb[k] as number, 2);
    }
  });

  it('terminating the secondary lanes (hero x 3) leaves the mean unchanged', () => {
    const n = 20000;
    const sum = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const d = laneData((i + 0.5) / n);
      const L: Vec3 = [3 * (d.d65[0] as number), 0, 0];
      const o = laneRadianceToRgb(L, d);
      sum[0]! += o[0] / n; sum[1]! += o[1] / n; sum[2]! += o[2] / n;
    }
    for (const v of sum) expect(v).toBeCloseTo(1, 2);
  });
});

describe('dispersion law', () => {
  it('passes through n_d and reproduces the Abbe number', () => {
    for (const [ior, abbe] of [[1.5168, 64.17], [1.62, 36.4], [1.92, 20.9]] as const) {
      const d = 20 / abbe;
      expect(iorAt(ior, d, 587.5618)).toBeCloseTo(ior, 8);
      const v = (iorAt(ior, d, 587.5618) - 1) / (iorAt(ior, d, 486.1327) - iorAt(ior, d, 656.2725));
      expect(v).toBeCloseTo(abbe, 6);
    }
  });
  it('no dispersion = constant, blue bends more than red', () => {
    expect(iorAt(1.5, 0, 450)).toBe(1.5);
    expect(cauchyCoefficients(1.5, 0)).toEqual([1.5, 0]);
    expect(iorAt(1.5, 0.5, 450)).toBeGreaterThan(iorAt(1.5, 0.5, 650));
  });
});

describe('spectral WGSL', () => {
  it('is generated for the options and names its binding / constants', () => {
    const o = resolveSpectralOptions(true);
    expect(o).toEqual({ res: 64, dispersion: true });
    const text = ptSpectralWgsl(o!);
    expect(text).toContain('@group(2) @binding(18)');
    expect(text).toContain('const PT_SPEC_RES: u32 = 64u;');
    expect(text).toContain('fn ptSpecMaterial');
    expect(ptSpectralWgsl({ res: 32, dispersion: false })).not.toContain('ptSpecIor');
    expect(resolveSpectralOptions(false)).toBeNull();
    expect(() => resolveSpectralOptions({ res: 5 })).toThrow(/res/);
  });
});
