import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptWavefrontBuild } from '../wavefront/wavefront.ts';
import { ptwfUnsupported } from '../wavefront/wavefront-wgsl.ts';
import type { PtVolumeOptions } from '../volume/volume.ts';
import { resolveIntegratorOptions } from './integrator-wgsl.ts';
import { ptMegakernelBuild } from './megakernel.ts';
import { ptApplyMode, ptModePreset } from './modes.ts';

// V5.3: what `mode: 'preview'` switches on for a scene with media, and that it never yields a refused combination.
const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const grid = { resolution: [8, 8, 8] as const, worldToIndex: [8, 0, 0, 3.5, 0, 8, 0, 3.5, 0, 0, 8, 3.5], densityMax: 1 };
const mats = [{ interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }];
const med = { sigmaT: [1, 1, 1] as [number, number, number], albedo: [0.9, 0.9, 0.9] as [number, number, number], g: 0.3 };
const homog: PtVolumeOptions = { media: [med], materials: mats };
const gridded: PtVolumeOptions = { media: [{ ...med, grid }], materials: mats };
const height: PtVolumeOptions = { media: [{ ...med, height: { y0: 0, H: 1.5 } }], materials: mats };
const preview = (volumes: PtVolumeOptions, extra: object = {}, scheduler: 'megakernel' | 'wavefront' = 'megakernel') =>
  ptApplyMode({ mode: 'preview' as const, integrator: { volumes }, ...extra }, scheduler).integrator as { volumes: PtVolumeOptions; [k: string]: unknown };

describe('preview preset for media (V5.3)', () => {
  it('names the volume options and the scatter cap', () => {
    const p = ptModePreset('preview');
    expect(p.volumes).toMatchObject({ trRoulette: true, forcedNee: true, equiangular: true, separateDepth: true });
    expect(p.limits?.maxVolumeDepth).toBe(256);
    expect(ptModePreset('reference').volumes).toBeNull();
    expect(ptModePreset('reference').limits).toBeNull();
  });
  it('homogeneous medium: forced + equiangular + separate depth, no grid-only options', () => {
    const v = preview(homog).volumes;
    expect(v).toMatchObject({ forcedNee: true, equiangular: true, separateDepth: true });
    expect(v.trRoulette).toBeUndefined();
    expect(v.densityFilter).toBeUndefined();
    expect(() => resolveIntegratorOptions(preview(homog) as never)).not.toThrow();
    expect(ptMegakernelBuild(caps, { format: {}, layout, integrator: preview(homog) as never }).code.length).toBeGreaterThan(0);
  });
  it('grid medium: trRoulette, hardware filter only for r16 / r8 grids', () => {
    expect(preview(gridded).volumes).toMatchObject({ trRoulette: true, separateDepth: true });
    expect(preview(gridded).volumes.densityFilter).toBeUndefined();
    expect(preview(gridded, { volumeGrid: { format: 'r16float' } }).volumes.densityFilter).toBe('hardware');
    expect(preview(gridded, { volumeGrid: { format: 'r8unorm' } }).volumes.densityFilter).toBe('hardware');
    expect(preview(gridded, { volumeGrid: { format: 'r32float' } }).volumes.densityFilter).toBeUndefined();
    expect(preview(gridded).volumes.forcedNee).toBeUndefined();
  });
  it('never makes a refused combination', () => {
    // height fog refuses forced / equiangular
    expect(preview(height).volumes.forcedNee).toBeUndefined();
    expect(() => ptMegakernelBuild(caps, { format: {}, layout, integrator: preview(height) as never })).not.toThrow();
    // outputs refuse forced / equiangular and AOVs
    const out = preview({ ...homog, outputs: true });
    expect(out.volumes.forcedNee).toBeUndefined();
    expect(out.aovs).toBe(false);
    expect(() => ptMegakernelBuild(caps, { format: {}, layout, integrator: out as never })).not.toThrow();
    // nee off refuses forced
    const noNee = ptApplyMode({ mode: 'preview' as const, integrator: { volumes: homog, nee: false } }).integrator as { volumes: PtVolumeOptions };
    expect(noNee.volumes.forcedNee).toBeUndefined();
    // wavefront: forced / equiangular are megakernel only, the reduced material kernel has no scatter vertex
    const wf = preview(homog, {}, 'wavefront');
    expect(wf.volumes.forcedNee).toBeUndefined();
    expect(wf.materialLod).toBe('full');
    expect(ptwfUnsupported(wf as never)).toBeNull();
    expect(() => ptWavefrontBuild(caps, { format: {}, layout, integrator: wf as never } as never)).not.toThrow();
  });
  it('explicit caller values win, a scene without media and other modes are untouched', () => {
    expect(preview({ ...homog, forcedNee: false }).volumes.forcedNee).toBe(false);
    expect(preview({ ...homog, equiangular: false }).volumes.equiangular).toBe(false);
    expect(preview({ ...gridded, trRoulette: false }).volumes.trRoulette).toBe(false);
    const ref = ptApplyMode({ mode: 'reference' as const, integrator: { volumes: homog } }).integrator as { volumes: PtVolumeOptions };
    expect(ref.volumes).toEqual(homog);
    const rt = ptApplyMode({ mode: 'realtime' as const, integrator: { volumes: homog } }).integrator as { volumes: PtVolumeOptions };
    expect(rt.volumes).toEqual(homog);
    expect(ptApplyMode({ mode: 'preview' as const, integrator: {} }).integrator).not.toHaveProperty('volumes');
  });
});

// V7.5: real-time frames of a media scene split the camera ray's single scattering off the denoised planes; the guides come from a guide ray.
describe('realtime preset frameMedia (V7.5)', () => {
  const rt = (integrator: object, scheduler: 'megakernel' | 'wavefront' = 'megakernel') => ptApplyMode({ mode: 'realtime' as const, integrator }, scheduler).integrator as Record<string, unknown>;
  it('is on for a media scene with frame outputs, and only then', () => {
    expect(rt({ volumes: homog }).frameMedia).toBe(true);
    expect(rt({}).frameMedia).toBeUndefined();
    expect(rt({ volumes: homog }, 'wavefront').frameMedia).toBeUndefined();
    expect(rt({ volumes: { ...homog, outputs: true } }).frameMedia).toBeUndefined();
    expect(rt({ volumes: homog, frameEmission: 'none' }).frameMedia).toBeUndefined();
    expect(ptApplyMode({ mode: 'preview' as const, integrator: { volumes: homog } }).integrator).not.toHaveProperty('frameMedia');
  });
  it('an explicit caller value wins', () => {
    expect(rt({ volumes: homog, frameMedia: false }).frameMedia).toBe(false);
  });
  it('the kernel compiles; the guide ray exists only with frameMedia, and the scatter point no longer sets the guides', () => {
    const build = (frameMedia: boolean) => ptMegakernelBuild(caps, { format: {}, layout, integrator: { ...(rt({ volumes: homog, frameMedia }) as object), shortenNames: false } as never, shortenNames: false, minify: false } as never).code;
    const on = build(true);
    const off = build(false);
    expect(on).toContain('gRay');
    expect(off).not.toContain('gRay');
    expect(on).toContain('gTrav');
    expect(off).toContain('fAlbedo = ptVolMedium(curMed).albedo');
    expect(on).not.toContain('fAlbedo = ptVolMedium(curMed).albedo');
    expect(on.match(/fSeen = true/g)?.length).toBe(2);
  });
});
