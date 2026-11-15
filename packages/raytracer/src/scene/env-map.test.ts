import { describe, expect, it } from 'vitest';
import { ENV_TEXEL_WORDS, buildEnvMapTable, envMapLookup, envMapMean, envMapPdf, envMapToWorld, envMapViewOfTable, envUvToDir, envWorldToMap, sampleEnvMap, type EnvMapInput } from './env-map.ts';
import { rtLightSampleWgsl } from './light-sample.ts';
import { bitsF32 } from './material.ts';

const Y = (c: readonly number[]) => 0.2126 * (c[0] as number) + 0.7152 * (c[1] as number) + 0.0722 * (c[2] as number);

/** A smooth sky with a bright hot spot, like env-sphere's, but small. */
function sky(w = 64, h = 32): EnvMapInput {
  const data = new Float32Array(w * h * 3);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const d = envUvToDir((i + 0.5) / w, (j + 0.5) / h);
      const sun = 30 * Math.exp(-0.5 * ((Math.acos(Math.min(1, Math.max(-1, 0.5 * d[0] + 0.7 * d[1] - 0.5 * d[2]))) / 0.12) ** 2));
      const base = d[1] >= 0 ? 0.3 + 0.7 * (1 - d[1]) : 0.2;
      data.set([base + sun, base * 0.9 + sun * 0.8, 1.0 * base + sun * 0.5], (j * w + i) * 3);
    }
  }
  return { width: w, height: h, channels: 3, data };
}

function constant(c: number, w = 32, h = 16): EnvMapInput {
  return { width: w, height: h, channels: 3, data: new Float32Array(w * h * 3).fill(c) };
}

/** Midpoint rule over (u, v): sum f(d) dw with dw = 2 pi^2 sin(theta) du dv, in map space. */
function integrate(f: (d: [number, number, number]) => number, n = 512): number {
  let sum = 0;
  for (let j = 0; j < n; j++) {
    const v = (j + 0.5) / n;
    for (let i = 0; i < n; i++) {
      const u = (i + 0.5) / n;
      sum += f(envUvToDir(u, v)) * 2 * Math.PI * Math.PI * Math.sin(Math.PI * v);
    }
  }
  return sum / (n * n);
}

describe('environment map table', () => {
  it('has monotone CDFs ending in exactly 1 and the layout the WGSL reads', () => {
    const img = sky();
    const t = buildEnvMapTable(img);
    expect(t.words.length).toBe(img.height + img.width * img.height * ENV_TEXEL_WORDS);
    let prev = 0;
    for (let j = 0; j < img.height; j++) {
      const c = bitsF32(t.words[j] as number);
      expect(c).toBeGreaterThanOrEqual(prev);
      prev = c;
    }
    expect(prev).toBe(1);
    for (let j = 0; j < img.height; j++) {
      let p = 0;
      for (let i = 0; i < img.width; i++) {
        const c = bitsF32(t.words[img.height + (j * img.width + i) * ENV_TEXEL_WORDS + 3] as number);
        expect(c).toBeGreaterThanOrEqual(p);
        p = c;
      }
      expect(p).toBe(1);
    }
    // texel colour is the file's, bit for bit
    const at = img.height + (5 * img.width + 7) * ENV_TEXEL_WORDS;
    expect(bitsF32(t.words[at] as number)).toBe((img.data as Float32Array)[(5 * img.width + 7) * 3]);
  });

  it('mean of a constant map is the constant; sin-weighted mean differs from the plain mean on a sky', () => {
    envMapMean(constant(0.7)).forEach((v) => expect(v).toBeCloseTo(0.7, 6));
    const img = sky();
    const m = envMapMean(img);
    const plain = [0, 0, 0];
    for (let p = 0; p < img.width * img.height; p++) for (let k = 0; k < 3; k++) plain[k] = (plain[k] as number) + (img.data[p * 3 + k] as number) / (img.width * img.height);
    expect(Math.abs(m[0] / (plain[0] as number) - 1)).toBeGreaterThan(0.01);
  });
});

describe('environment map pdf and sampling (TypeScript twin)', () => {
  for (const rotation of [0, 0.7]) {
    const img = sky();
    const view = envMapViewOfTable(buildEnvMapTable(img), rotation);

    it(`integrates the pdf to 1 over the sphere (rotation ${rotation})`, () => {
      const total = integrate((m) => envMapPdf(view, envMapToWorld(rotation, m)));
      expect(Math.abs(total - 1)).toBeLessThan(2e-3);
    });

    it(`sample -> pdf round trip: the pdf a sample reports is the pdf of its direction (rotation ${rotation})`, () => {
      let checked = 0;
      for (let a = 0; a < 64; a++) {
        for (let b = 0; b < 64; b++) {
          const s = sampleEnvMap(view, (a + 0.5) / 64, (b + 0.5) / 64);
          expect(Math.hypot(...s.wi)).toBeCloseTo(1, 12);
          const p = envMapPdf(view, s.wi);
          if (!(s.pdf > 0)) continue;
          expect(Math.abs(p / s.pdf - 1)).toBeLessThan(1e-9);
          checked++;
        }
      }
      expect(checked).toBeGreaterThan(3000);
    });

    it(`importance sampling is unbiased: mean of L/pdf equals the integral of L, and beats uniform sampling (rotation ${rotation})`, () => {
      const exact = integrate((m) => Y(envMapLookup(view, envMapToWorld(rotation, m))));
      const n = 128;
      let sum = 0;
      let sum2 = 0;
      let uni = 0;
      let uni2 = 0;
      for (let a = 0; a < n; a++) {
        for (let b = 0; b < n; b++) {
          const u0 = (a + 0.5) / n;
          const u1 = (b + 0.5) / n;
          const s = sampleEnvMap(view, u0, u1);
          const est = Y(s.radiance) / s.pdf;
          sum += est;
          sum2 += est * est;
          const z = 1 - 2 * u0;
          const r = Math.sqrt(1 - z * z);
          const w: [number, number, number] = [r * Math.cos(2 * Math.PI * u1), z, r * Math.sin(2 * Math.PI * u1)];
          const e2 = Y(envMapLookup(view, w)) * 4 * Math.PI;
          uni += e2;
          uni2 += e2 * e2;
        }
      }
      const N = n * n;
      expect(Math.abs(sum / N / exact - 1)).toBeLessThan(0.01);
      expect(Math.abs(uni / N / exact - 1)).toBeLessThan(0.05);
      expect(Math.sqrt(sum2 / N - (sum / N) ** 2)).toBeLessThan(0.5 * Math.sqrt(uni2 / N - (uni / N) ** 2));
    });
  }

  it('lookup is bilinear: texel centres return the texel, and the seam at u = 0 wraps', () => {
    const img = sky(16, 8);
    const view = envMapViewOfTable(buildEnvMapTable(img));
    for (const [i, j] of [[3, 2], [0, 4], [15, 1]] as const) {
      const d = envMapToWorld(0, envUvToDir((i + 0.5) / 16, (j + 0.5) / 8));
      const c = envMapLookup(view, d);
      for (let k = 0; k < 3; k++) expect(c[k]).toBeCloseTo((img.data as Float32Array)[(j * 16 + i) * 3 + k] as number, 5);
    }
    const a = envMapLookup(view, envMapToWorld(0, envUvToDir(0.0001, 0.5)));
    const b = envMapLookup(view, envMapToWorld(0, envUvToDir(0.9999, 0.5)));
    for (let k = 0; k < 3; k++) expect(Math.abs((a[k] as number) / (b[k] as number) - 1)).toBeLessThan(0.01);
  });

  it('rotation turns the environment about +Y exactly like the bench sampleEquirect (world -> map is a rotation by -a)', () => {
    const d: [number, number, number] = [0.3, 0.4, Math.sqrt(1 - 0.25)];
    const m = envWorldToMap(0.9, d);
    const back = envMapToWorld(0.9, m);
    d.forEach((v, k) => expect(back[k]).toBeCloseTo(v, 12));
    // a +X direction with a quarter-turn rotation (counter-clockwise from above) lands on the map's... -Z axis side
    const q = envWorldToMap(Math.PI / 2, [1, 0, 0]);
    expect(q[0]).toBeCloseTo(0, 12);
    expect(q[2]).toBeCloseTo(1, 12);
  });

  it('furnace: a constant map has pdf ~ 1/(4 pi), and the irradiance estimate L/pdf * cos is pi L to 1e-3', () => {
    const c = 0.8;
    const view = envMapViewOfTable(buildEnvMapTable(constant(c, 64, 32)));
    // Rows are picked by u1 with probability ~ sin(theta): many strata in u1 keep the quadrature error of the rows (1/n1) small.
    const n0 = 128;
    const n1 = 4096;
    let e = 0;
    let pdfSum = 0;
    for (let a = 0; a < n0; a++) {
      for (let b = 0; b < n1; b++) {
        const s = sampleEnvMap(view, (a + 0.5) / n0, (b + 0.5) / n1);
        pdfSum += s.pdf;
        e += (Math.max(0, s.wi[1]) * Y(s.radiance)) / s.pdf;
      }
    }
    const N = n0 * n1;
    expect(Math.abs(e / N / (Math.PI * c) - 1)).toBeLessThan(1e-3);
    expect(Math.abs((pdfSum / N) * 4 * Math.PI - 1)).toBeLessThan(0.02);
    expect(integrate((m) => envMapPdf(view, m))).toBeCloseTo(1, 2);
  });

  it('a black map is never sampled (pdf 0) and has pdfNorm 0; negative and NaN texels count as black', () => {
    const t = buildEnvMapTable({ width: 4, height: 4, channels: 3, data: Float32Array.from({ length: 48 }, (_, i) => (i % 5 === 0 ? -1 : Number.NaN)) });
    expect(t.pdfNorm).toBe(0);
    expect(sampleEnvMap(envMapViewOfTable(t), 0.3, 0.6).pdf).toBe(0);
  });

  it('never picks a zero-weight pixel next to bright ones', () => {
    const data = new Float32Array(8 * 4 * 3);
    data.set([5, 5, 5], (2 * 8 + 3) * 3);
    const view = envMapViewOfTable(buildEnvMapTable({ width: 8, height: 4, channels: 3, data }));
    for (let a = 0; a < 40; a++) {
      const s = sampleEnvMap(view, (a + 0.5) / 40, ((a * 7) % 40 + 0.5) / 40);
      expect(envMapPdf(view, s.wi)).toBeGreaterThan(0);
      const m = envWorldToMap(0, s.wi);
      const u = Math.atan2(m[0], -m[2]) / (2 * Math.PI);
      expect(Math.floor((u - Math.floor(u)) * 8)).toBe(3);
      expect(Math.floor((Math.acos(m[1]) / Math.PI) * 4)).toBe(2);
    }
  });
});

describe('environment map WGSL', () => {
  it('is only emitted on request, adds under 4 KB, and is a drop-in for the constant functions', () => {
    const base = rtLightSampleWgsl();
    const map = rtLightSampleWgsl({ envMap: true });
    expect(base).not.toContain('rtEnvMapLookup');
    for (const name of ['rtEnvMapLookup', 'rtEnvMapPdf', 'rtSampleEnvMap', 'rtEnvironmentLe', 'rtEnvironmentPdf', 'rtSampleEnvironmentLi']) expect(map, name).toContain(`fn ${name}(`);
    for (const name of ['rtEnvironmentLe', 'rtEnvironmentPdf', 'rtSampleEnvironmentLi']) expect(base, name).toContain(`fn ${name}(`);
    expect(map.length - base.length).toBeLessThan(4000);
    expect(rtLightSampleWgsl({ envMap: true, envImportance: false })).toContain('RT_ENV_IMPORTANCE: bool = false');
    expect(map).toContain('RT_ENV_IMPORTANCE: bool = true');
  });
});
