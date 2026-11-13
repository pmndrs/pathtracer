import { describe, expect, it } from 'vitest';
import { owenSobol2D } from '../sampling/owen-sobol.ts';
import { buildCaps } from '../caps/index.ts';
import { rtModuleWgsl } from '../trace/index.ts';
import type { RtEnvironmentLight, RtSunLight, RtTriangleLight, Vec3Tuple } from './lights.ts';
import {
  ENV_UNIFORM_PDF,
  lightPdfLi,
  orthonormalBasis,
  rtLightSampleWgsl,
  sampleEnvironmentLi,
  sampleLightLi,
  sampleSunLi,
  sampleTriangleLi,
  sampleUniformTriangle,
  sunContains,
  triangleSolidAnglePdf,
} from './light-sample.ts';

function triangleLight(p0: Vec3Tuple, p1: Vec3Tuple, p2: Vec3Tuple, radiance = 10, twoSided = false, flip = false): RtTriangleLight {
  const e1: Vec3Tuple = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
  const e2: Vec3Tuple = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
  const c: Vec3Tuple = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
  const len = Math.hypot(...c);
  const s = flip ? -1 : 1;
  return {
    kind: 'triangle',
    instance: 0,
    triangle: 0,
    material: 0,
    p0,
    edge1: e1,
    edge2: e2,
    normal: [(s * c[0]) / len, (s * c[1]) / len, (s * c[2]) / len],
    area: len / 2,
    radiance: [radiance, radiance, radiance],
    twoSided,
    power: 0,
  };
}

/** The quad-light scene's emitter: 1 m x 1 m at height 1, facing down (-Y), as two triangles. */
function quadLights(twoSided = false): RtTriangleLight[] {
  const a: Vec3Tuple = [-0.5, 1, -0.5];
  const b: Vec3Tuple = [0.5, 1, -0.5];
  const c: Vec3Tuple = [0.5, 1, 0.5];
  const d: Vec3Tuple = [-0.5, 1, 0.5];
  // Counter-clockwise seen from -Y (clockwise seen from +Y) gives a downward normal.
  return [triangleLight(a, b, c, 10, twoSided), triangleLight(a, c, d, 10, twoSided)];
}

/** Irradiance at the origin facing +Y from a centred parallel rectangle of half sides a, b at height h, radiance L (closed form, Lambert's formula). */
function rectangleIrradianceClosedForm(a: number, b: number, h: number, L: number): number {
  const x = a / h;
  const y = b / h;
  const quarter = (1 / (2 * Math.PI)) * ((x / Math.sqrt(1 + x * x)) * Math.atan(y / Math.sqrt(1 + x * x)) + (y / Math.sqrt(1 + y * y)) * Math.atan(x / Math.sqrt(1 + y * y)));
  return Math.PI * L * 4 * quarter;
}

/** Independent numeric integral over the rectangle (midpoint rule): the closed form's own check. */
function rectangleIrradianceNumeric(a: number, b: number, h: number, L: number, n = 600): number {
  let sum = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = -a + ((i + 0.5) / n) * 2 * a;
      const z = -b + ((j + 0.5) / n) * 2 * b;
      const r2 = x * x + h * h + z * z;
      sum += (h * h) / (r2 * r2);
    }
  }
  return L * sum * ((2 * a) / n) * ((2 * b) / n);
}

/** Solid angle of the rectangle seen from the origin: 4 x the corner formula's atan. */
function rectangleSolidAngle(a: number, b: number, h: number): number {
  return 4 * Math.atan((a * b) / (h * Math.sqrt(a * a + b * b + h * h)));
}

function stratified(n: number, seed: number): [number, number][] {
  const out: [number, number][] = [];
  const pair = [0, 0];
  for (let i = 0; i < n; i++) {
    owenSobol2D(i, seed, pair);
    out.push([pair[0] as number, pair[1] as number]);
  }
  return out;
}

describe('closed forms used by the tests', () => {
  it('Lambert rectangle irradiance: closed form equals a numeric integral', () => {
    const closed = rectangleIrradianceClosedForm(0.5, 0.5, 1, 10);
    expect(rectangleIrradianceNumeric(0.5, 0.5, 1, 10)).toBeCloseTo(closed, 5);
    // The unit-quad / unit-height / L=10 case of the quad-light scene (bench/metrics/analytic.ts agrees: about 7.5227 W/m^2).
    expect(closed).toBeCloseTo(7.5227, 3);
  });
});

describe('sampleUniformTriangle', () => {
  it('stays inside the triangle and has the barycentric means of a uniform warp', () => {
    let m1 = 0;
    let m2 = 0;
    let m0 = 0;
    const pts = stratified(1 << 14, 7);
    for (const [u0, u1] of pts) {
      const [b1, b2] = sampleUniformTriangle(u0, u1);
      const b0 = 1 - b1 - b2;
      expect(b0).toBeGreaterThanOrEqual(-1e-12);
      expect(b1).toBeGreaterThanOrEqual(0);
      expect(b2).toBeGreaterThanOrEqual(-1e-12);
      m0 += b0;
      m1 += b1;
      m2 += b2;
    }
    for (const m of [m0, m1, m2]) expect(m / pts.length).toBeCloseTo(1 / 3, 2);
  });

  it('is area-uniform: each of the four sub-triangles of the medial subdivision gets a quarter', () => {
    const counts = [0, 0, 0, 0];
    const pts = stratified(1 << 14, 3);
    for (const [u0, u1] of pts) {
      const [b1, b2] = sampleUniformTriangle(u0, u1);
      const b0 = 1 - b1 - b2;
      // Medial triangle: all barycentrics <= 1/2; the three corner triangles: one barycentric > 1/2.
      if (b0 > 0.5) counts[0]!++;
      else if (b1 > 0.5) counts[1]!++;
      else if (b2 > 0.5) counts[2]!++;
      else counts[3]!++;
    }
    for (const c of counts) expect(c / pts.length).toBeCloseTo(0.25, 2);
  });
});

describe('triangle light sampling', () => {
  const lights = quadLights();
  const p: Vec3Tuple = [0, 0, 0];

  it('pdf equals the solid-angle conversion dist^2 / (|cos| area), and lightPdfLi agrees', () => {
    for (const light of lights) {
      for (const [u0, u1] of stratified(256, 11)) {
        const s = sampleTriangleLi(light, p, u0, u1);
        expect(s.valid).toBe(true);
        const cosLight = -(light.normal[0] * s.wi[0] + light.normal[1] * s.wi[1] + light.normal[2] * s.wi[2]);
        expect(s.pdf).toBeCloseTo(triangleSolidAnglePdf(light.area, s.distance, cosLight), 12);
        expect(lightPdfLi(light, s.wi, s.distance, cosLight)).toBeCloseTo(s.pdf, 12);
        expect(s.distance).toBeCloseTo(Math.hypot(s.point[0] - p[0], s.point[1] - p[1], s.point[2] - p[2]), 12);
        expect(Math.hypot(...s.wi)).toBeCloseTo(1, 12);
      }
    }
  });

  it('E[1/pdf] over the lights equals the solid angle of the quad (analytic)', () => {
    let sum = 0;
    for (const light of lights) {
      const pts = stratified(1 << 15, 5);
      let acc = 0;
      for (const [u0, u1] of pts) {
        const s = sampleTriangleLi(light, p, u0, u1);
        if (s.valid) acc += 1 / s.pdf;
      }
      sum += acc / pts.length;
    }
    expect(sum).toBeCloseTo(rectangleSolidAngle(0.5, 0.5, 1), 3);
  });

  it('Monte Carlo irradiance of the quad light matches the closed form (cosine at the receiver, point under the light)', () => {
    let e = 0;
    for (const light of lights) {
      const pts = stratified(1 << 15, 9);
      let acc = 0;
      for (const [u0, u1] of pts) {
        const s = sampleTriangleLi(light, p, u0, u1);
        if (s.valid) acc += (s.radiance[0] * Math.max(0, s.wi[1])) / s.pdf;
      }
      e += acc / pts.length;
    }
    const analytic = rectangleIrradianceClosedForm(0.5, 0.5, 1, 10);
    expect(Math.abs(e - analytic) / analytic).toBeLessThan(2e-4);
  });

  it('works at a tilted, off-centre receiver too (E from a light pair vs a numeric integral)', () => {
    // Receiver at (0.3, 0, -0.2) facing +Y: the rectangle relative to it is not centred; integrate numerically over the quad.
    const q: Vec3Tuple = [0.3, 0, -0.2];
    let e = 0;
    for (const light of lights) {
      const pts = stratified(1 << 15, 21);
      let acc = 0;
      for (const [u0, u1] of pts) {
        const s = sampleTriangleLi(light, q, u0, u1);
        if (s.valid) acc += (s.radiance[0] * Math.max(0, s.wi[1])) / s.pdf;
      }
      e += acc / pts.length;
    }
    let numeric = 0;
    const n = 500;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = -0.5 + (i + 0.5) / n - q[0];
        const z = -0.5 + (j + 0.5) / n - q[2];
        const r2 = x * x + 1 + z * z;
        numeric += 10 / (r2 * r2);
      }
    }
    numeric /= n * n;
    expect(Math.abs(e - numeric) / numeric).toBeLessThan(5e-4);
  });

  it('a one-sided light gives nothing from behind; a two-sided light lights both sides', () => {
    const below: Vec3Tuple = [0, 2, 0]; // above the quad, behind its downward-facing emitting side
    for (const light of lights) {
      expect(sampleTriangleLi(light, below, 0.3, 0.4).valid).toBe(false);
    }
    for (const light of quadLights(true)) {
      const s = sampleTriangleLi(light, below, 0.3, 0.4);
      expect(s.valid).toBe(true);
      expect(s.pdf).toBeGreaterThan(0);
    }
  });

  it('zero distance and a degenerate hit return invalid', () => {
    const light = lights[0] as RtTriangleLight;
    const onIt = triangleLightPointAt(light);
    expect(sampleTriangleLi(light, onIt, 0, 0).valid).toBe(false);
  });
});

function triangleLightPointAt(light: RtTriangleLight): Vec3Tuple {
  // u = (0, 0): b0 = 0, b1 = 0, b2 = 1 -> vertex 2 of the triangle.
  return [light.p0[0] + light.edge2[0], light.p0[1] + light.edge2[1], light.p0[2] + light.edge2[2]];
}

function sun(radius: number, direction: Vec3Tuple = [0.3, 0.8, 0.2]): RtSunLight {
  const len = Math.hypot(...direction);
  return {
    kind: 'sun',
    direction: [direction[0] / len, direction[1] / len, direction[2] / len],
    angularRadius: radius,
    solidAngle: 2 * Math.PI * (1 - Math.cos(radius)),
    irradiance: [50, 46, 40],
    power: 0,
  };
}

describe('sun cone sampling', () => {
  for (const radius of [0.00465, 0.05, 0.4]) {
    const s = sun(radius);
    it(`radius ${radius}: every sample is inside the disc, pdf = 1/solid angle, radiance = irradiance/solid angle`, () => {
      for (const [u0, u1] of stratified(512, 2)) {
        const l = sampleSunLi(s, u0, u1);
        expect(l.valid).toBe(true);
        expect(l.delta).toBe(false);
        expect(Math.hypot(...l.wi)).toBeCloseTo(1, 12);
        expect(sunContains(s, l.wi)).toBe(true);
        expect(l.pdf).toBeCloseTo(1 / s.solidAngle, 6);
        expect(lightPdfLi(s, l.wi)).toBeCloseTo(l.pdf, 6);
        expect(l.radiance[0] * s.solidAngle).toBeCloseTo(50, 9);
        expect(l.distance).toBe(Infinity);
      }
    });

    it(`radius ${radius}: uniform in solid angle (mean cos = 1 - omc/2) and irradiance recovered`, () => {
      const pts = stratified(1 << 14, 13);
      let meanCos = 0;
      let e = 0;
      for (const [u0, u1] of pts) {
        const l = sampleSunLi(s, u0, u1);
        meanCos += l.wi[0] * s.direction[0] + l.wi[1] * s.direction[1] + l.wi[2] * s.direction[2];
        e += (l.radiance[0] * (l.wi[0] * s.direction[0] + l.wi[1] * s.direction[1] + l.wi[2] * s.direction[2])) / l.pdf;
      }
      meanCos /= pts.length;
      const expected = 1 - (1 - Math.cos(radius)) / 2;
      expect(Math.abs(1 - meanCos - (1 - expected))).toBeLessThan(1e-6 + 2e-3 * (1 - expected));
      // E over the cone of L cos / pdf = irradiance * mean cos (a receiver facing the sun).
      expect(e / pts.length / (50 * expected)).toBeCloseTo(1, 4);
    });
  }

  it('directions just outside the disc are rejected', () => {
    const s = sun(0.05);
    const [b0] = orthonormalBasis(s.direction);
    const at = (angle: number): Vec3Tuple => [s.direction[0] * Math.cos(angle) + b0[0] * Math.sin(angle), s.direction[1] * Math.cos(angle) + b0[1] * Math.sin(angle), s.direction[2] * Math.cos(angle) + b0[2] * Math.sin(angle)];
    expect(sunContains(s, at(0.049))).toBe(true);
    expect(sunContains(s, at(0.051))).toBe(false);
    expect(lightPdfLi(s, at(0.1))).toBe(0);
  });

  it('a delta sun (radius 0) returns its direction, the irradiance as radiance, pdf 1 and delta = true; it cannot be hit', () => {
    const s = sun(0);
    const l = sampleLightLi(s, [0, 0, 0], 0.4, 0.6);
    expect(l.delta).toBe(true);
    expect(l.pdf).toBe(1);
    expect(l.wi).toEqual(s.direction);
    expect(l.radiance).toEqual([50, 46, 40]);
    expect(sunContains(s, s.direction)).toBe(false);
    expect(lightPdfLi(s, s.direction)).toBe(0);
  });

  it('keeps the angular radius precise where 1 - cos in f32 would not (0.265 deg sun)', () => {
    const s = sun(0.00465);
    const l = sampleSunLi(s, 0.999, 0.25);
    const angle = Math.acos(Math.min(1, l.wi[0] * s.direction[0] + l.wi[1] * s.direction[1] + l.wi[2] * s.direction[2]));
    expect(angle).toBeLessThanOrEqual(0.00465 + 1e-9);
    expect(angle).toBeGreaterThan(0.0046);
  });
});

describe('environment sampling (uniform sphere)', () => {
  const env: RtEnvironmentLight = { kind: 'environment', radiance: [1, 0.5, 0.25], scale: 2, rotationY: 0, mapSlot: 0xffffffff, power: 0 };

  it('has pdf 1/(4 pi), radiance scaled, and integrates the radiance over the sphere to 4 pi L', () => {
    const pts = stratified(1 << 14, 17);
    let acc = 0;
    let meanZ = 0;
    for (const [u0, u1] of pts) {
      const l = sampleEnvironmentLi(env, u0, u1);
      expect(l.pdf).toBeCloseTo(ENV_UNIFORM_PDF, 15);
      expect(l.radiance).toEqual([2, 1, 0.5]);
      expect(Math.hypot(...l.wi)).toBeCloseTo(1, 12);
      acc += l.radiance[0] / l.pdf;
      meanZ += l.wi[2];
    }
    expect(acc / pts.length).toBeCloseTo(2 * 4 * Math.PI, 2);
    expect(Math.abs(meanZ / pts.length)).toBeLessThan(1e-3);
    expect(lightPdfLi(env, [0, 0, 1])).toBeCloseTo(ENV_UNIFORM_PDF, 15);
  });

  it('irradiance on a diffuse receiver under a uniform sky: E = pi L (cosine-weighted, estimated with the uniform sphere)', () => {
    const pts = stratified(1 << 15, 19);
    let acc = 0;
    for (const [u0, u1] of pts) {
      const l = sampleEnvironmentLi(env, u0, u1);
      acc += (l.radiance[0] * Math.max(0, l.wi[2])) / l.pdf;
    }
    expect(acc / pts.length / (Math.PI * 2)).toBeCloseTo(1, 3);
  });
});

describe('sampleLightLi dispatch', () => {
  it('routes by kind', () => {
    const t = sampleLightLi(quadLights()[0] as RtTriangleLight, [0, 0, 0], 0.2, 0.3);
    expect(t.distance).toBeLessThan(Infinity);
    expect(sampleLightLi(sun(0.1), [0, 0, 0], 0.2, 0.3).distance).toBe(Infinity);
  });
});

describe('WGSL', () => {
  const wgsl = rtLightSampleWgsl();
  it('declares the public names', () => {
    for (const name of ['RtLightLi', 'rtSampleLightLi', 'rtSampleTriangleLi', 'rtSampleSunLi', 'rtSampleEnvironmentLi', 'rtTriangleSolidAnglePdf', 'rtSunContains', 'rtSunPdf', 'rtSunRadiance', 'rtEnvironmentRadiance', 'RT_ENV_UNIFORM_PDF']) {
      expect(wgsl).toContain(name);
    }
  });

  it('carries no comments and balanced braces', () => {
    expect(wgsl).not.toMatch(/\/\//);
    expect(wgsl.split('{').length).toBe(wgsl.split('}').length);
  });

  it('uses the same constants as the twin', () => {
    expect(wgsl).toContain(`RT_ENV_UNIFORM_PDF: f32 = ${Math.fround(1 / (4 * Math.PI))}`);
  });

  it('is a small addition to the scene module (the whole file stays under 6 KB)', () => {
    expect(wgsl.length).toBeLessThan(6000);
  });

  it('is reachable through the module generator when the scene module asks for it (scene `sampling`)', () => {
    const caps = buildCaps({ features: new Set(), limits: {} }, new Set());
    // modules.ts does not pass `sampling` through yet (another agent owns trace/): the text must at least compose after it.
    const code = rtModuleWgsl(caps, { scene: { slot: { group: 2, firstBinding: 0 }, layout: { normals: true, uvs: false, positions: true } } }) + wgsl;
    expect(code).toContain('fn rtSampleLightLi');
    expect(code).toContain('fn rtLight(');
    expect(code.length).toBeLessThan(50_000);
  });
});
