import { describe, expect, it } from 'vitest';
import { resolveIntegratorOptions } from './integrator-wgsl.ts';
import { ptwfUnsupported } from '../wavefront/wavefront-wgsl.ts';

// P5 merge: the opt-in features were built on separate branches; every pair that was never run together must throw clearly.
const volumes = { media: [{ sigmaA: [0.1, 0.1, 0.1], sigmaS: [0.5, 0.5, 0.5], g: 0 }], materials: [] } as never;
const restir = { candidates: 8, spatial: 1, bias: 'unbiased' } as never;

describe('P5 feature combinations', () => {
  it('spectral + volumes throws (media sigma is RGB)', () => {
    expect(() => resolveIntegratorOptions({ spectral: true, volumes })).toThrow(/spectral.*volumes/);
  });
  it('spectral + restir (DI or GI) throws', () => {
    expect(() => resolveIntegratorOptions({ spectral: true, restir })).toThrow(/spectral/);
    expect(() => resolveIntegratorOptions({ spectral: true, restir: { ...(restir as object), gi: {} } as never })).toThrow(/spectral/);
  });
  it('volumes + restir DI runs for homogeneous media on the megakernel only (V5.4); GI, sharedHit, visibility and non-homogeneous media throw', () => {
    expect(resolveIntegratorOptions({ volumes, restir }).restir).toBe(true);
    expect(ptwfUnsupported({ volumes, restir })).toMatch(/volumes/);
    expect(() => resolveIntegratorOptions({ volumes, restir: { ...(restir as object), gi: {} } as never })).toThrow(/GI/);
    expect(() => resolveIntegratorOptions({ volumes, restir: { ...(restir as object), sharedHit: true } as never })).toThrow(/sharedHit/);
    expect(() => resolveIntegratorOptions({ volumes, restir: { candidates: 8, bias: 'biased' } })).toThrow(/initialVisibility/);
    expect(resolveIntegratorOptions({ volumes, restir: { candidates: 8, bias: 'biased', initialVisibility: false } }).restir).toBe(true);
    const v = volumes as { media: object[]; materials: [] };
    expect(() => resolveIntegratorOptions({ volumes: { ...v, media: [{ ...v.media[0], height: { H: 1, y0: 0 } }] } as never, restir })).toThrow(/homogeneous/);
    expect(() => resolveIntegratorOptions({ volumes: { ...v, media: [{ ...v.media[0], emission: [1, 1, 1] }] } as never, restir })).toThrow(/homogeneous/);
    expect(() => resolveIntegratorOptions({ volumes: { ...v, media: [{ ...v.media[0], grid: { resolution: [2, 2, 2] } }] } as never, restir })).toThrow(/homogeneous/);
    expect(() => resolveIntegratorOptions({ volumes: { ...v, mediumStack: true } as never, restir })).toThrow(/mediumStack/);
  });
  it('volumes, spectral (V270, V272) and restir.gi (P5) all run on the wavefront', () => {
    expect(ptwfUnsupported({ volumes })).toBeNull();
    expect(ptwfUnsupported({ spectral: true })).toBeNull();
    expect(ptwfUnsupported({ restir: { ...(restir as object), gi: {} } as never })).toBeNull();
  });
  it('materialx composes with spectral (MaterialX first, then the uplift) and with volumes on the options level', () => {
    const o = resolveIntegratorOptions({ spectral: true, textures: true, materialx: { wgsl: 'fn ptMtlxApply() {}' } });
    expect(o.mtlx).toBe(true);
    expect(o.spectral).not.toBeNull();
  });
});
