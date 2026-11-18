// P5 at the kernel-text level: no media = the default text, media add the volume code only, names are legal WGSL, the wavefront takes the same media (V270).
import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { RESERVED, balanced, declaredIdentifiers } from '../../../raytracer/src/wgsl-names.testutil.ts';
import { ptMegakernelBuild } from '../integrator/megakernel.ts';
import { ptwfUnsupported } from '../wavefront/wavefront-wgsl.ts';
import { PT_VOLUME_ALL_PARTS, ptVolumeRatioShadows, ptVolumeValidate, ptVolumeWgsl } from './volume-wgsl.ts';
import type { PtVolumeOptions } from './volume.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const grid = { resolution: [8, 8, 8] as const, worldToIndex: [8, 0, 0, 3.5, 0, 8, 0, 3.5, 0, 0, 8, 3.5], densityMax: 1 };
const homog: PtVolumeOptions = { media: [{ sigmaT: [1, 1, 1], albedo: [0.9, 0.9, 0.9], g: 0.3 }], materials: [{ interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }] };
const gridded: PtVolumeOptions = { ...homog, media: [{ ...homog.media[0]!, grid }] };
const bricked: PtVolumeOptions = { ...gridded, majorant: 'brick' };
const ratio: PtVolumeOptions = { ...homog, shadowTracking: 'ratio' };
const build = (volumes?: PtVolumeOptions) => ptMegakernelBuild(caps, { format: {}, layout, ...(volumes ? { integrator: { volumes } } : {}), shortenNames: false, minify: false }).code;

describe('integrator.volumes (P5)', () => {
  it('without media the kernel text is the default one (`volumes: null` too)', () => {
    const none = ptMegakernelBuild(caps, { format: {}, layout }).code;
    expect(ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: null } }).code).toBe(none);
    expect(none).not.toContain('ptVol');
  });

  it('homogeneous media add the tables, the free-flight loop and the shadow transmittance; no 3D texture', () => {
    const text = build(homog);
    for (const needle of ['ptVolSample', 'ptVolTr(', 'ptHgSample', 'ptSampleLdMedium', 'var<private> ptVolMedia', 'curMed']) expect(text).toContain(needle);
    expect(text).not.toContain('texture_3d');
    expect(balanced(text)).toBe(true);
  });

  it('a grid medium binds the density as a 3D texture at group 2 binding 16', () => {
    const text = build(gridded);
    expect(text).toMatch(/@group\(2\) @binding\(16\) var ptVolGrid: texture_3d<f32>/);
    expect(text).toContain('textureLoad(ptVolGrid');
  });

  it('majorant brick adds the coarse texture and the DDA; global and the analytic shadow do not', () => {
    const text = build(bricked);
    expect(text).toMatch(/@group\(2\) @binding\(17\) var ptVolMaj: texture_3d<f32>/);
    for (const needle of ['ptDdaInit', 'ptDdaNext', 'ptVolSampleBrick', 'ptVolTrBrick']) expect(text).toContain(needle);
    expect(build(gridded)).not.toContain('ptDda');
    expect(build(homog)).not.toContain('ptDda');
    // ratio tracking of a homogeneous medium compiles the tracking loop without a texture
    expect(build(ratio).length).toBeGreaterThan(build(homog).length);
    expect(build(ratio)).not.toContain('texture_3d');
  });

  it('pieces: every part alone is legal text, the default is all of them; estimated shadows fork their own stream (V270)', () => {
    expect(ptVolumeWgsl(gridded)).toBe(ptVolumeWgsl(gridded, PT_VOLUME_ALL_PARTS));
    for (const part of PT_VOLUME_ALL_PARTS) expect(balanced(ptVolumeWgsl(bricked, [part]))).toBe(true);
    expect(ptVolumeWgsl(homog, ['seedOf'])).not.toContain('ptVolMedia');
    expect(ptVolumeWgsl(homog, ['seedOf', 'tables', 'hg'])).not.toContain('ptVolSample');
    expect(ptVolumeWgsl(homog)).not.toContain('ptVolFork');
    expect(ptVolumeWgsl(homog)).toContain('fn ptVolTr(');
    for (const v of [ratio, gridded, bricked]) {
      const text = ptVolumeWgsl(v);
      expect(text).toContain('fn ptVolFork()');
      expect(text).toContain('fn ptVolTrCore(');
      expect(text).toContain('fn ptVolTr(');
      expect(ptVolumeRatioShadows(v)).toBe(true);
    }
    expect(ptVolumeRatioShadows(homog)).toBe(false);
  });

  it('declares no reserved WGSL word', () => {
    for (const volumes of [homog, gridded, bricked, ratio]) {
      const bad = [...declaredIdentifiers(ptVolumeWgsl(volumes))].filter((n) => RESERVED.has(n));
      expect(bad).toEqual([]);
    }
  });

  it('shortened and minified kernels still build and stay below 50 KB', () => {
    for (const volumes of [homog, gridded, bricked]) {
      const { code } = ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes } });
      expect(code.length).toBeLessThan(50_000);
    }
  });

  it('validates its input and refuses combinations it does not implement', () => {
    expect(() => ptVolumeValidate({ media: [], materials: [] })).toThrow(/no media/);
    expect(() => ptVolumeValidate({ ...homog, media: [{ ...homog.media[0]!, g: 1 }] })).toThrow(/outside/);
    expect(() => ptVolumeValidate({ ...homog, materials: [{ interior: 2, exterior: 0, boundary: false }] })).toThrow(/does not exist/);
    expect(() => ptVolumeValidate({ ...gridded, cameraMedium: 1 })).toThrow(/homogeneous/);
    expect(() => build({ ...homog, cameraMedium: 1 })).not.toThrow();
    expect(() => ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: homog, restir: {} as never } })).toThrow(/cannot be combined/);
    expect(() => ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: homog, counters: true } })).toThrow(/cannot be combined/);
    expect(ptwfUnsupported({ volumes: homog })).toBeNull();
  });
});
