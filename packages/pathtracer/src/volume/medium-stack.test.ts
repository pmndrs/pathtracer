// V6.2 medium stack: the TypeScript twin of the WGSL push / pop rules, and the kernel text of both schedulers with and without the option.
import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { RESERVED, balanced, declaredIdentifiers } from '../../../raytracer/src/wgsl-names.testutil.ts';
import { ptMegakernelBuild } from '../integrator/megakernel.ts';
import { ptwfShadeWgsl } from '../wavefront/wavefront-wgsl.ts';
import { ptDetectCameraMediumStack } from './camera-medium.ts';
import { ptMediumStackIds, ptMediumStackPack, ptMediumStackPop, ptMediumStackPush, ptMediumStackTop, type PtVolumeOptions } from './volume.ts';

const ids = (s: number) => ptMediumStackIds(s);

describe('medium stack twin (V6.2)', () => {
  it('nested: enter outer, enter inner, leave inner returns to outer, leave outer returns to vacuum', () => {
    let s = ptMediumStackPush(0, 1).stack;
    expect(ptMediumStackTop(s)).toBe(1);
    s = ptMediumStackPush(s, 2).stack;
    expect(ids(s)).toEqual([2, 1]);
    s = ptMediumStackPop(s, 2);
    expect(ptMediumStackTop(s)).toBe(1);
    s = ptMediumStackPop(s, 1);
    expect(s).toBe(0);
  });
  it('overlapping: leaving the OUTER-entered medium first removes it from under the top', () => {
    let s = ptMediumStackPack([2, 1]); // entered 2, then 1
    s = ptMediumStackPop(s, 2);
    expect(ids(s)).toEqual([1]);
    expect(ptMediumStackTop(s)).toBe(1);
  });
  it('the same medium twice pops its most recent entry; an id not on the stack and id 0 change nothing', () => {
    const s = ptMediumStackPack([1, 2, 1]);
    expect(ids(ptMediumStackPop(s, 1))).toEqual([2, 1]);
    expect(ptMediumStackPop(s, 5)).toBe(s);
    expect(ptMediumStackPop(s, 0)).toBe(s);
    expect(ptMediumStackPush(s, 0)).toEqual({ stack: s, overflow: false });
  });
  it('overflow drops the OUTERMOST entry and reports it', () => {
    let s = ptMediumStackPack([1, 2, 3, 4]);
    expect(ids(s)).toEqual([4, 3, 2, 1]);
    const r = ptMediumStackPush(s, 5);
    expect(r.overflow).toBe(true);
    s = r.stack;
    expect(ids(s)).toEqual([5, 4, 3, 2]);
    expect(ptMediumStackPush(ptMediumStackPack([1, 2, 3]), 4).overflow).toBe(false);
  });
  it('ids up to 63 fit, the stack stays in 24 bits', () => {
    const s = ptMediumStackPack([63, 62, 61, 60]);
    expect(s).toBeLessThan(1 << 24);
    expect(ids(s)).toEqual([60, 61, 62, 63]);
  });
  it('camera stack: all containing media, outermost first', () => {
    const box = (h: number, medium: number) => ({ medium, positions: [-h, -h, -h, h, -h, -h, h, h, -h, -h, h, -h, -h, -h, h, h, -h, h, h, h, h, -h, h, h], indices: [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5] });
    expect(ptDetectCameraMediumStack([0, 0, 0], [box(1, 2), box(3, 1), box(5, 3)])).toEqual([3, 1, 2]);
    expect(ptDetectCameraMediumStack([2, 0, 0], [box(1, 2), box(3, 1), box(5, 3)])).toEqual([3, 1]);
    expect(ptDetectCameraMediumStack([9, 0, 0], [box(1, 2)])).toEqual([]);
  });
});

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const two: PtVolumeOptions = { media: [{ sigmaT: [1, 1, 1], albedo: [0.9, 0.9, 0.9], g: 0.3 }, { sigmaT: [2, 2, 2], albedo: [0.5, 0.5, 0.5], g: 0 }], materials: [{ interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }, { interior: 2, exterior: 0, boundary: true }] };
const stacked: PtVolumeOptions = { ...two, mediumStack: true, cameraStack: [1] };
const build = (volumes: PtVolumeOptions) => ptMegakernelBuild(caps, { format: {}, layout, integrator: { volumes }, shortenNames: false, minify: false }).code;

describe('medium stack kernels (V6.2)', () => {
  it('off: no stack code; on: helpers, camera stack word and the stack in the interface branch', () => {
    const off = build(two);
    expect(off).not.toContain('ptVolPush');
    expect(off).not.toContain('curStack');
    const on = build(stacked);
    for (const n of ['ptVolPush', 'ptVolPop', 'ptVolCross', 'var curStack = 1u', 'ptVolCurSt']) expect(on).toContain(n);
    expect(balanced(on)).toBe(true);
    for (const id of declaredIdentifiers(on)) expect(RESERVED.has(id)).toBe(false);
  });
  it('the wavefront keeps the skip count in the top byte', () => {
    const wf = ptwfShadeWgsl({ volumes: stacked });
    expect(wf).toContain('p.med | (p.skips << 24u)');
    expect(ptwfShadeWgsl({ volumes: two })).toContain('p.med | (p.skips << 16u)');
  });
  it('rejects cameraStack without the stack and more than 63 media', () => {
    expect(() => build({ ...two, cameraStack: [1] })).toThrow(/cameraStack needs mediumStack/);
    expect(() => build({ ...stacked, media: Array.from({ length: 64 }, () => two.media[0]!) })).toThrow(/63 media/);
  });
});
