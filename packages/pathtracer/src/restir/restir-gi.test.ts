import { describe, expect, it } from 'vitest';
import { balanced, declaredIdentifiers, RESERVED } from '../../../raytracer/src/wgsl-names.testutil.ts';
import { ptIntegratorWgsl, resolveIntegratorOptions } from '../integrator/integrator-wgsl.ts';
import { giInit, giMerge, giTruth, oracleRng, type GiPixel, type GiSample, type GiWorld } from './restir-gi-oracle.ts';
import { PT_RESTIR_GI_DOMAINS, PT_RESTIR_GI_KEY, PT_RESTIR_GI_MAX_SPATIAL, ptRestirGiUnsupported, resolveRestirGiOptions } from './restir-gi-wgsl.ts';
import { PT_RESTIR_KEY, ptRestirWgsl, resolveRestirOptions } from './restir-wgsl.ts';
import { ptRestirUnsupported } from './restir.ts';

const world: GiWorld = {
  half: 2,
  diffuseLimit: 0.9,
  rmin: 0.3,
  lo: (a, b) => 1 + 0.8 * Math.sin(3 * a) * Math.cos(2 * b),
  vis: (p, _a, b) => (b > p.y - 0.4 ? 1 : 0),
};
const pixels: GiPixel[] = [
  { x: 0, y: 0, h: 0.6 },
  { x: 0.5, y: -0.2, h: 0.7 },
  { x: -0.4, y: 0.3, h: 0.55 },
];
const truth = giTruth(world, pixels[0] as GiPixel);

function estimate(unbiased: boolean, trials: number, counts: number): number {
  const rand = oracleRng(4242);
  let sum = 0;
  for (let t = 0; t < trials; t++) {
    // each domain makes `counts` independent candidates and keeps one (RIS), as a pixel that has merged `counts` times would; M = 1 candidates are the real case.
    const ys: GiSample[] = pixels.map((_, i) => giInit(world, pixels, i, rand));
    const out = giMerge(world, pixels, ys, unbiased, rand);
    if (out.w > 0) sum += giTargetOf(out) * out.w;
  }
  void counts;
  return sum / trials;
}

const giTargetOf = (y: GiSample): number => {
  const p = pixels[0] as GiPixel;
  const r = Math.hypot(y.a - p.x, y.b - p.y, p.h);
  const c = p.h / r;
  return (1 / Math.PI) * c * world.lo(y.a, y.b) * (c / (r * r)) * world.vis(p, y.a, y.b);
};

describe('ReSTIR GI reconnection arithmetic (CPU oracle of ptRestirGiInit / ptGiMerge on a toy plane)', () => {
  it('a fresh candidate is the path-tracing estimate: F W has the quadrature as its mean (the area-measure Jacobian is right)', () => {
    const rand = oracleRng(1);
    let sum = 0;
    const trials = 400000;
    for (let t = 0; t < trials; t++) {
      const y = giInit(world, pixels, 0, rand);
      if (y.w > 0) sum += giTargetOf(y) * y.w;
    }
    expect(Math.abs(sum / trials / truth - 1)).toBeLessThan(0.01);
  });

  it('the Talbot combine of three pixels (visibility, a non-shiftable region, different heights) is unbiased', () => {
    const e = estimate(true, 400000, 1);
    expect(Math.abs(e / truth - 1)).toBeLessThan(0.012);
  });

  it('the 1/sum(M) combine is biased on the same world (pixels disagree on visibility and support)', () => {
    const e = estimate(false, 400000, 1);
    expect(Math.abs(e / truth - 1)).toBeGreaterThan(0.02);
  });
});

describe('ReSTIR GI options and text', () => {
  it('resolves defaults and clamps', () => {
    const r = resolveRestirGiOptions({ spatial: 99, historyCap: 1000 });
    expect(r.spatial).toBe(PT_RESTIR_GI_MAX_SPATIAL);
    expect(r.historyCap).toBe(255);
    expect(resolveRestirGiOptions().bias).toBe('biased');
    expect(PT_RESTIR_GI_DOMAINS).toBe(PT_RESTIR_GI_MAX_SPATIAL + 1);
  });

  it('the DI-only resolved options (and so the adapter config hash) do not mention gi', () => {
    expect('gi' in resolveRestirOptions({})).toBe(false);
    expect(resolveRestirOptions({ gi: {} }).gi?.temporal).toBe(true);
  });

  it('GI streams are disjoint from the DI ones', () => {
    const di = new Set<number>();
    for (let c = 0; c < 128; c++) di.add(c);
    di.add(PT_RESTIR_KEY.shift);
    for (let i = 0; i < 9; i++) {
      di.add(PT_RESTIR_KEY.temporalSelect + i);
      di.add(PT_RESTIR_KEY.spatialOffset + i);
      di.add(PT_RESTIR_KEY.spatialSelect + i);
    }
    const k = PT_RESTIR_GI_KEY;
    const gi: number[] = [k.dir, k.comp, k.light, k.point, k.mix, k.shift];
    for (let i = 0; i < PT_RESTIR_GI_DOMAINS; i++) gi.push(k.temporalSelect + i, k.spatialOffset + i, k.spatialSelect + i);
    expect(new Set(gi).size).toBe(gi.length);
    for (const v of gi) expect(di.has(v)).toBe(false);
  });

  it('refuses what it cannot do', () => {
    const base = { restir: {}, nee: true };
    const why = (o: object) => ptRestirUnsupported({ integrator: { restir: { gi: {} }, ...o } });
    expect(why({})).toBeNull();
    expect(why({ textures: true })).toMatch(/textured/);
    expect(why({ envSampling: 'none' })).toMatch(/light-sampled/);
    expect(ptRestirGiUnsupported({ ...resolveIntegratorOptions({ restir: {} }), restirGi: false })).toMatch(/not set/);
    void base;
  });

  it('shade and passes text carry the GI code only when asked, with balanced braces and no reserved identifiers', () => {
    const off = ptRestirWgsl({ restir: {} });
    const on = ptRestirWgsl({ restir: { gi: {} } });
    expect(off).not.toContain('ptRestirGi');
    for (const entry of ['ptRestirGiInit', 'ptRestirGiTemporal', 'ptRestirGiSpatial']) expect(on).toContain(`fn ${entry}(`);
    expect(balanced(on)).toBe(true);
    for (const name of declaredIdentifiers(on)) expect(RESERVED.has(name), name).toBe(false);
    const shade = ptIntegratorWgsl({ restir: { gi: {} } });
    expect(shade).toContain('fn ptRestirGiLi(');
    expect(ptIntegratorWgsl({ restir: {} })).not.toContain('ptRestirGi');
    expect(balanced(shade)).toBe(true);
    for (const name of declaredIdentifiers(shade)) expect(RESERVED.has(name), name).toBe(false);
  });
});
