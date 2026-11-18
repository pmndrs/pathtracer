import { describe, expect, it } from 'vitest';
import { ptIntegratorWgsl, resolveIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { balanced, declaredIdentifiers, RESERVED } from '../../../raytracer/src/wgsl-names.testutil.ts';
import { oracleInit, oracleMerge, oracleRng, ptRsOctDecode, ptRsOctEncode, type OracleDomain } from './restir-oracle.ts';
import { PT_RESTIR_MAX_SPATIAL, PT_RESTIR_PARAMS_BYTES, ptRestirGbufBytes, ptRestirGbufLayout, ptRestirWgsl, resolveRestirOptions } from './restir-wgsl.ts';
import { ptRestirUnsupported } from './restir.ts';

// Discrete world: 12 light samples, three pixels whose targets differ and partly vanish (occlusion, backfacing), the exact answer for pixel 0 is sum f0.
const N = 12;
const q = Array.from({ length: N }, (_, i) => 0.02 + 0.01 * ((i * 5) % 7));
const qsum = q.reduce((a, b) => a + b, 0);
const qn = q.map((v) => v / qsum);
const cdf = qn.map((_, i) => qn.slice(0, i + 1).reduce((a, b) => a + b, 0));
const f0 = (y: number) => (y % 4 === 3 ? 0 : 1 + ((y * 3) % 5));
const f1 = (y: number) => (y % 3 === 0 ? 0 : 0.5 + ((y * 7) % 6));
const f2 = (y: number) => (y < 3 ? 0 : 2 + ((y * 5) % 4));
const domains: OracleDomain[] = [{ phat: f0 }, { phat: f1 }, { phat: f2 }];
const truth = Array.from({ length: N }, (_, y) => f0(y)).reduce((a, b) => a + b, 0);

function estimate(unbiased: boolean, trials: number, counts: [number, number, number]): number {
  const rand = oracleRng(12345);
  let sum = 0;
  for (let t = 0; t < trials; t++) {
    const rs = domains.map((d, j) => oracleInit(counts[j] as number, cdf, qn, d.phat, rand));
    const out = oracleMerge(domains, rs, unbiased, rand);
    if (out.y >= 0) sum += f0(out.y) * out.w;
  }
  return sum / trials;
}

describe('ReSTIR DI reservoir arithmetic (CPU oracle of ptRsMerge)', () => {
  it('RIS alone (M candidates, one domain) is unbiased', () => {
    const rand = oracleRng(7);
    let sum = 0;
    const trials = 200000;
    for (let t = 0; t < trials; t++) {
      const r = oracleInit(4, cdf, qn, f0, rand);
      if (r.y >= 0) sum += f0(r.y) * r.w;
    }
    expect(Math.abs(sum / trials / truth - 1)).toBeLessThan(0.01);
  });

  it('the Talbot (balance heuristic) combine is unbiased with three overlapping domains of different targets and confidences', () => {
    const e = estimate(true, 300000, [4, 6, 3]);
    expect(Math.abs(e / truth - 1)).toBeLessThan(0.01);
  });

  it('the 1/sum(M) combine is biased when the domains disagree on the target (the price of the cheap variant)', () => {
    const e = estimate(false, 300000, [4, 6, 3]);
    expect(Math.abs(e / truth - 1)).toBeGreaterThan(0.03);
  });

  it('with identical targets the biased combine is unbiased too', () => {
    const same: OracleDomain[] = [{ phat: f0 }, { phat: f0 }, { phat: f0 }];
    const rand = oracleRng(99);
    let sum = 0;
    const trials = 200000;
    for (let t = 0; t < trials; t++) {
      const rs = same.map((d) => oracleInit(3, cdf, qn, d.phat, rand));
      const out = oracleMerge(same, rs, false, rand);
      if (out.y >= 0) sum += f0(out.y) * out.w;
    }
    expect(Math.abs(sum / trials / truth - 1)).toBeLessThan(0.01);
  });

  it('M = 1 with no other domain gives W = 1 / q (plain NEE without MIS)', () => {
    const rand = oracleRng(3);
    for (let t = 0; t < 50; t++) {
      const r = oracleInit(1, cdf, qn, (y) => f0(y) + 1, rand);
      expect(r.w).toBeCloseTo(1 / (qn[r.y] as number), 9);
    }
  });
});

describe('ReSTIR DI WGSL', () => {
  const wgsl = ptRestirWgsl({});
  it('is balanced, declares no reserved word and fits the uniform record', () => {
    expect(balanced(wgsl)).toBe(true);
    expect([...declaredIdentifiers(wgsl)].filter((n) => RESERVED.has(n))).toEqual([]);
    // PtRsParams: 13 vec4
    expect(PT_RESTIR_PARAMS_BYTES).toBe(13 * 16);
    expect(wgsl).toContain(`array<PtRsGeom, ${PT_RESTIR_MAX_SPATIAL + 1}>`);
  });
  it('defaults: 8 candidates, temporal, 1 spatial neighbour, biased with visibility, random reuse offsets', () => {
    expect(resolveRestirOptions()).toMatchObject({ candidates: 8, temporal: true, spatial: 1, bias: 'biased', initialVisibility: true, neighbours: 'random', temporalPick: 'random', historyCap: 20 });
    expect(resolveRestirOptions({ bias: 'unbiased' }).initialVisibility).toBe(false);
    expect(resolveRestirOptions({ spatial: 99 }).spatial).toBe(PT_RESTIR_MAX_SPATIAL);
  });
  it('the integrator text has no ReSTIR code unless asked, and the shade side with it', () => {
    expect(ptIntegratorWgsl({})).not.toMatch(/ptRestir/);
    expect(resolveIntegratorOptions({}).restir).toBe(false);
    const on = ptIntegratorWgsl({ restir: {} });
    expect(on).toMatch(/fn ptRestirLd/);
    expect(on).toMatch(/select\(s\.pdf, 0\.0, vertex == 0u\)/);
  });
});

describe('ReSTIR DI G-buffer (P4.2c)', () => {
  it('the octahedral normal codec round-trips a unit vector to 1e-4 (poles, axes, the lower hemisphere)', () => {
    const rand = oracleRng(5);
    const probes: [number, number, number][] = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]];
    for (let i = 0; i < 20000; i++) {
      const z = 2 * rand() - 1;
      const a = 2 * Math.PI * rand();
      const r = Math.sqrt(1 - z * z);
      probes.push([r * Math.cos(a), r * Math.sin(a), z]);
    }
    let worst = 0;
    for (const n of probes) {
      const d = ptRsOctDecode(ptRsOctEncode(n));
      worst = Math.max(worst, Math.hypot(d[0] - n[0], d[1] - n[1], d[2] - n[2]));
    }
    expect(worst).toBeLessThan(1e-4);
  });
  it('a record is 20 bytes for an untextured scene (was 32), 32 with textures, +4 with a stochastic material pick', () => {
    expect(ptRestirGbufLayout({}).words * 4).toBe(20);
    expect(ptRestirGbufLayout({ textures: true }).words * 4).toBe(32);
    const mats = { features: { mix: true, thin: false, layered: false, medium: false } };
    expect(ptRestirGbufLayout({ materials: mats }).words * 4).toBe(24);
    expect(ptRestirGbufLayout({ textures: true, materials: mats }).words * 4).toBe(36);
    expect(ptRestirGbufBytes({}, 1920 * 1080)).toBe(2 * 1920 * 1080 * 20);
  });
  it('the passes module is balanced and names no reserved word in every supported flavour', () => {
    const flavours = [{ sampler: 'bluenoise' as const }, { primary: 'vbuffer' as const }, { textures: true }, { textures: true, alphaMask: true }];
    for (const f of flavours) {
      const text = ptRestirWgsl({ ...f, restir: {} });
      expect(balanced(text)).toBe(true);
      expect([...declaredIdentifiers(text)].filter((n) => RESERVED.has(n))).toEqual([]);
    }
    const shared = ptRestirWgsl({ restir: { sharedHit: true } });
    expect(balanced(shared)).toBe(true);
    expect([...declaredIdentifiers(shared)].filter((n) => RESERVED.has(n))).toEqual([]);
  });
  it('supports textures, OpenPBR deterministic lobes, V-buffer and blue noise; refuses what the target cannot reproduce', () => {
    expect(ptRestirUnsupported({ integrator: { restir: {}, sampler: 'bluenoise', primary: 'vbuffer' } })).toBeNull();
    const base = { mix: true, thin: true, layered: true, medium: true };
    expect(ptRestirUnsupported({ integrator: { restir: {}, materials: { features: base, layered: { layering: 'albedo' } } } })).toBeNull();
    expect(ptRestirUnsupported({ integrator: { restir: {}, materials: { features: base, layered: { layering: 'stochastic' } } } })).toMatch(/stochastic/);
    expect(ptRestirUnsupported({ integrator: { restir: {}, materials: { features: { ...base, fuzz: true } } } })).toMatch(/fuzz/);
    expect(ptRestirUnsupported({ integrator: { restir: {}, materials: { features: { ...base, aniso: true } } } })).toMatch(/anisotropic/);
    expect(ptRestirUnsupported({ integrator: { restir: { sharedHit: true } }, layout: { normals: true, uvs: false, positions: false } })).toMatch(/positions/);
  });
  it('the shade kernel reads the shared hit only with sharedHit and trace primaries', () => {
    expect(ptIntegratorWgsl({ restir: {} })).not.toMatch(/ptRestirHit/);
    expect(ptIntegratorWgsl({ restir: { sharedHit: true } })).toMatch(/ptRestirHit/);
    expect(resolveIntegratorOptions({ restir: { sharedHit: true }, primary: 'vbuffer' }).restirShared).toBe(false);
  });
});
