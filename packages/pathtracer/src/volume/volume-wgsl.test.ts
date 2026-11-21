// P5 at the kernel-text level: no media = the default text, media add the volume code only, names are legal WGSL, the wavefront takes the same media (V270).
import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { RESERVED, balanced, declaredIdentifiers } from '../../../raytracer/src/wgsl-names.testutil.ts';
import { ptMegakernelBuild } from '../integrator/megakernel.ts';
import { ptwfUnsupported } from '../wavefront/wavefront-wgsl.ts';
import { PT_VOLUME_ALL_PARTS, ptVolumeMediaCompatible, ptVolumeMediaData, ptVolumeRatioShadows, ptVolumeValidate, ptVolumeWgsl } from './volume-wgsl.ts';
import type { PtVolumeOptions } from './volume.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const grid = { resolution: [8, 8, 8] as const, worldToIndex: [8, 0, 0, 3.5, 0, 8, 0, 3.5, 0, 0, 8, 3.5], densityMax: 1 };
const homog: PtVolumeOptions = { media: [{ sigmaT: [1, 1, 1], albedo: [0.9, 0.9, 0.9], g: 0.3 }], materials: [{ interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }] };
const gridded: PtVolumeOptions = { ...homog, media: [{ ...homog.media[0]!, grid }] };
const hardware: PtVolumeOptions = { ...gridded, densityFilter: 'hardware' };
const atlased: PtVolumeOptions = { ...gridded, gridLayout: 'atlas' };
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

  it("gridLayout 'atlas' adds the indirection binding 21 and reads through it; 'dense' text has no trace of it", () => {
    const text = build(atlased);
    expect(text).toContain('@group(2) @binding(21) var ptVolInd: texture_3d<u32>');
    expect(text).toContain('ptVolInd');
    expect(build(gridded)).not.toContain('ptVolInd');
    const hwAtlas = build({ ...atlased, densityFilter: 'hardware' });
    expect(hwAtlas).toContain('textureSampleLevel(ptVolGrid');
    expect(hwAtlas).not.toContain('bf.x');
  });

  it("densityFilter 'hardware' samples once through a sampler at binding 20; 'manual' text has neither", () => {
    const text = build(hardware);
    expect(text).toMatch(/@group\(2\) @binding\(20\) var ptVolSampler: sampler/);
    expect(text).toContain('textureSampleLevel(ptVolGrid');
    expect(text).not.toContain('textureLoad(ptVolGrid');
    expect(balanced(text)).toBe(true);
    expect(build(gridded)).not.toContain('ptVolSampler');
    expect(build({ ...gridded, densityFilter: 'manual' })).toBe(build(gridded));
  });

  it("V2.2 shadowTracking 'residual' reads the minorant (vec4 DDA cell), needs a brick grid, and leaves the brick text unchanged without it", () => {
    const residual: PtVolumeOptions = { ...bricked, shadowTracking: 'residual' };
    const text = build(residual);
    expect(text).toContain('mm.y');
    expect(text).toContain('exp(-m.sigmaT * (sg.w * (sg.y - sg.x)))');
    expect(build(bricked)).not.toContain('mm.y');
    expect(() => ptVolumeValidate({ ...gridded, shadowTracking: 'residual' })).toThrow(/brick/);
    expect(() => ptVolumeValidate({ ...homog, shadowTracking: 'residual' })).toThrow(/brick/);
    expect(balanced(ptVolumeWgsl(residual))).toBe(true);
    expect(ptVolumeRatioShadows(residual)).toBe(true);
  });

  it("V2.3 shadowTracking 'telescoping' needs a brick grid, reads the minorant, carries the sample index, and leaves other texts unchanged", () => {
    const tele: PtVolumeOptions = { ...bricked, shadowTracking: 'telescoping' };
    const text = build(tele);
    expect(text).toContain('fn ptVolTele(');
    expect(text).toContain('ptVolSampleIdx');
    expect(text).toContain('ptMax3(abs(tr)) > 0.0');
    expect(build(bricked)).not.toContain('ptVolTele');
    expect(build({ ...bricked, shadowTracking: 'residual' })).not.toContain('ptVolSampleIdx');
    expect(() => ptVolumeValidate({ ...gridded, shadowTracking: 'telescoping' })).toThrow(/brick/);
    expect(balanced(ptVolumeWgsl(tele))).toBe(true);
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
      expect(code.length).toBeLessThan(50_500);
    }
  });

  it('validates its input and refuses combinations it does not implement', () => {
    expect(() => ptVolumeValidate({ media: [], materials: [] })).toThrow(/no media/);
    expect(() => ptVolumeValidate({ ...homog, media: [{ ...homog.media[0]!, g: 1 }] })).toThrow(/outside/);
    expect(() => ptVolumeValidate({ ...homog, materials: [{ interior: 2, exterior: 0, boundary: false }] })).toThrow(/does not exist/);
    expect(() => ptVolumeValidate({ ...gridded, cameraMedium: 1 })).not.toThrow(); // V3.3: a camera inside a grid medium
    expect(() => build({ ...homog, cameraMedium: 1 })).not.toThrow();
    expect(() => ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: homog, restir: {} as never } })).toThrow(/cannot be combined/);
    expect(() => ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: homog, counters: true } })).toThrow(/cannot be combined/);
    expect(ptwfUnsupported({ volumes: homog })).toBeNull();
  });
});

describe('P5.v14 phase lobes and P5.v15 / P5.v16 outputs at the kernel-text level', () => {
  const lobes = { gA: 0.2, gB: 0.85, w: 0.4, alpha: 0 };
  const twoLobe: PtVolumeOptions = { ...gridded, media: [{ ...gridded.media[0]!, g: 0.2, lobes }] };
  const withOut: PtVolumeOptions = { ...gridded, outputs: true };

  it('a lobe with weight 0 is one HG lobe: the text of the plain kernel', () => {
    expect(build({ ...gridded, media: [{ ...gridded.media[0]!, lobes: { gA: 0.3, gB: 0.9, w: 0, alpha: 0 } }] })).toBe(build(gridded));
  });
  it('two lobes add a 7th table row per medium and route the phase sample and the NEE density through the mixture', () => {
    const text = build(twoLobe);
    for (const needle of ['fn ptVolPhaseSample(', 'fn ptVolDraineOneMinusU(', 'ptVolPhaseSample(curMed, woM,', 'prevPdf = ptVolPhase(curMed,', 'let ph = ptVolPhase(id, dot(wo, li.wi));', 'array<vec4f, 7>', '(id - 1u) * 7u']) expect(text).toContain(needle);
    expect(text).not.toContain('let gM = ');
    expect(balanced(text)).toBe(true);
    expect([...declaredIdentifiers(ptVolumeWgsl(twoLobe))].filter((n) => RESERVED.has(n))).toEqual([]);
    const { code } = ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: { ...twoLobe, majorant: 'brick' } } });
    expect(code.length).toBeLessThan(52_000);
  });
  it('the wavefront shade kernel takes the lobes too', () => {
    expect(ptwfUnsupported({ volumes: twoLobe })).toBeNull();
  });
  it('lobe A must be the medium g, lobes are validated', () => {
    expect(() => ptVolumeValidate({ ...gridded, media: [{ ...gridded.media[0]!, g: 0.3, lobes }] })).toThrow(/lobe A/);
    expect(() => ptVolumeValidate({ ...gridded, media: [{ ...gridded.media[0]!, g: 0.2, lobes: { ...lobes, w: 1.5 } }] })).toThrow(/weight/);
  });
  it('outputs add the order capture, the primary transmittance march and the tail writes, megakernel only', () => {
    const text = build(withOut);
    for (const needle of ['ptVoBegin(ray, curMed, pixelSeed, sampleIndex);', 'ptVoCapture(depth, radiance);', 'ptVoEnd(radiance);', 'ptVoAccumulate(index, dims.x * dims.y, l.xyz);']) expect(text).toContain(needle);
    // both depth increments are preceded by the capture
    expect(text.split('ptVoCapture(depth, radiance);').length - 1).toBe(2);
    expect(balanced(text)).toBe(true);
    expect(ptVolumeWgsl(withOut, ['seedOf', 'tables', 'hg'])).not.toContain('ptVoBegin');
    expect(ptwfUnsupported({ volumes: withOut })).toBeNull(); // V5.2: the wavefront (plain scheduler) has them too
    expect(() => ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: withOut, accumulate: false } })).toThrow(/needs accumulate/);
    // grid + brick + outputs is 50.5 KB, with the two-lobe phase 51.5 KB: just over the ~50 KB DXC class (a Windows compile-time concern; these are reference kernels)
    expect(ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: { ...withOut, majorant: 'brick' } } }).code.length).toBeLessThan(52_500);
    expect(ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes: { ...withOut, majorant: 'brick', media: twoLobe.media } } }).code.length).toBeLessThan(54_000);
  });
});

describe('height fog (V4.5)', () => {
  const heightOpts: PtVolumeOptions = { ...homog, media: [{ ...homog.media[0]!, height: { y0: 0, H: 1.5 } }] };
  it('is emitted only when a height medium exists', () => {
    expect(build(homog)).not.toContain('ptVolHeight');
    expect(build(heightOpts)).toContain('ptVolHeightG');
    expect(ptVolumeWgsl(heightOpts)).toContain('ptVolIsHeight');
  });
  it('is refused with a grid, a bad H, forced NEE', () => {
    expect(() => ptVolumeValidate({ ...heightOpts, media: [{ ...heightOpts.media[0]!, grid }] })).toThrow(/grid and a height/);
    expect(() => ptVolumeValidate({ ...heightOpts, media: [{ ...heightOpts.media[0]!, height: { y0: 0, H: 0 } }] })).toThrow(/height/);
    expect(() => ptVolumeValidate({ ...heightOpts, forcedNee: true })).toThrow(/forcedNee/);
  });
  it('declares legal names', () => {
    const names = declaredIdentifiers(ptVolumeWgsl(heightOpts));
    for (const n of names) expect(RESERVED.has(n)).toBe(false);
  });
});

describe('volumes.runtimeParams (V6.6)', () => {
  const rt: PtVolumeOptions = { ...homog, runtimeParams: true };
  it('declares the table as a uniform at binding 22 and has no private initializer; the default text is unchanged', () => {
    const text = build(rt);
    expect(text).toMatch(/@group\(2\) @binding\(22\) var<uniform> ptVolMedia: array<vec4f, 6>;/);
    expect(text).not.toContain('var<private> ptVolMedia');
    expect(balanced(text)).toBe(true);
    expect(build({ ...homog, runtimeParams: false })).toBe(build(homog));
  });
  it('the data holds the compiled rows (majorant, r8 scale fold)', () => {
    const g = { ...grid, densityMax: 2, densityScale: 0.5 };
    const m: PtVolumeOptions = { ...homog, media: [{ sigmaT: [1, 2, 3], albedo: [0.5, 0.6, 0.7], g: 0.25, grid: g }] };
    const d = ptVolumeMediaData(m);
    expect(d.length).toBe(24);
    expect(Array.from(d.slice(0, 8))).toEqual([0.5, 1, 1.5, 6, 0.5, Math.fround(0.6), Math.fround(0.7), 0.25]);
    expect(d[8]).toBe(1.5); // max sigma_t x scale
  });
  it('setMedia validation accepts new numbers and refuses structure changes', () => {
    const next = { ...rt, media: [{ sigmaT: [3, 3, 3], albedo: [0.2, 0.3, 0.4], g: -0.5 }] } as PtVolumeOptions;
    expect(() => ptVolumeMediaCompatible(rt, next)).not.toThrow();
    expect(() => ptVolumeMediaCompatible(rt, { ...rt, media: [...rt.media, ...rt.media] })).toThrow(/number of media/);
    expect(() => ptVolumeMediaCompatible(rt, { ...rt, media: [{ ...rt.media[0]!, grid }] })).toThrow(/grid/);
    expect(() => ptVolumeMediaCompatible(rt, { ...rt, media: [{ ...rt.media[0]!, lobes: { gA: 0.3, gB: -0.5, w: 0.2, alpha: 0 } }] })).toThrow(/phase lobes/);
    expect(() => ptVolumeMediaCompatible(rt, { ...rt, shadowTracking: 'ratio' })).toThrow(/option/);
    expect(() => ptVolumeMediaCompatible(rt, { ...rt, media: [{ ...rt.media[0]!, g: 1.5 }] })).toThrow(/g 1.5/);
  });
});

describe('V4.6 emissive media', () => {
  const emissive: PtVolumeOptions = { ...homog, media: [{ ...homog.media[0]!, emission: [1, 2, 3] }] };
  it('without emission the volume text and the kernel are unchanged; with it only the emission code appears', () => {
    expect(build({ ...homog, media: [{ ...homog.media[0]!, emission: [0, 0, 0] }] })).toBe(build(homog));
    const t = build(emissive);
    for (const n of ['ptVolEmit', 'ptVolEmitTake']) expect(t).toContain(n);
    expect(build(homog)).not.toContain('ptVolEmit');
    expect(ptVolumeWgsl(emissive)).toContain('(vec3f(1.0) - m.albedo)');
  });
  it('grid media add the per-collision term; height / brick / outputs refuse', () => {
    expect(ptVolumeWgsl({ ...gridded, media: [{ ...gridded.media[0]!, emission: [1, 1, 1] }] })).toContain('(st - ss) * le / m.mu');
    expect(() => ptVolumeValidate({ ...bricked, media: [{ ...bricked.media[0]!, emission: [1, 1, 1] }] })).toThrow(/brick/);
    expect(() => ptVolumeValidate({ ...emissive, media: [{ ...emissive.media[0]!, height: { y0: 0, H: 1 } }] })).toThrow(/height/);
    expect(() => ptVolumeValidate({ ...emissive, outputs: true })).toThrow(/outputs/);
  });
  it('runtime table carries the emission row last; setMedia cannot switch emission on', () => {
    const rt = { ...emissive, runtimeParams: true };
    const data = ptVolumeMediaData(rt);
    expect(Array.from(data.slice(data.length - 4))).toEqual([1, 2, 3, 0]);
    expect(() => ptVolumeMediaCompatible({ ...homog, runtimeParams: true }, rt)).toThrow(/emits/);
  });
});

