import { buildCaps, buildSceneCopy } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptWavefrontBuild } from '../wavefront/wavefront.ts';
import { ptMegakernelWgsl } from './megakernel.ts';
import { PT_VBUFFER_BINDING, PT_VBUFFER_NO_MATERIAL, ptVbufferPack, ptVbufferResolve, ptVbufferUnpack, ptVbufferWgsl } from './vbuffer.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };

describe('V-buffer texel format', () => {
  it('a miss is four zero words, and so is the cleared target', () => {
    expect(ptVbufferPack({ hit: false, instance: 5, triangle: 9, material: 1, bary: [0.2, 0.3] })).toEqual([0, 0, 0, 0]);
    expect(ptVbufferUnpack([0, 0, 0, 0]).hit).toBe(false);
  });

  it('round-trips ids exactly and barycentrics to half a unorm16 step', () => {
    for (const [b1, b2] of [[0, 0], [1, 0], [0, 1], [0.25, 0.5], [1 / 3, 1 / 3], [0.999, 0.0004]] as const) {
      const words = ptVbufferPack({ hit: true, instance: 4_000_000, triangle: 0xfffffffe, material: PT_VBUFFER_NO_MATERIAL, bary: [b1, b2] });
      expect(words[0]).toBe(4_000_001);
      const t = ptVbufferUnpack(words);
      expect(t).toMatchObject({ hit: true, instance: 4_000_000, triangle: 0xfffffffe, material: PT_VBUFFER_NO_MATERIAL });
      expect(Math.abs(t.bary[0] - b1)).toBeLessThanOrEqual(0.5 / 65535 + 1e-9);
      expect(Math.abs(t.bary[1] - b2)).toBeLessThanOrEqual(0.5 / 65535 + 1e-9);
    }
  });

  it('instance 0 is not a miss (the word holds instance + 1)', () => {
    expect(ptVbufferUnpack(ptVbufferPack({ hit: true, instance: 0, triangle: 0, material: 0, bary: [0, 0] })).hit).toBe(true);
  });
});

describe('CPU resolve (twin of ptVbufferHit)', () => {
  const triangle = { positions: new Float32Array([0, 0, 0, 2, 0, 0, 0, 3, 0]), indices: new Uint32Array([0, 1, 2]), normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]) };
  // column-major: x mirrored (scale -1), translated by (10, 20, 30)
  const mirrored = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1];
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const copy = buildSceneCopy({
    meshes: [triangle],
    instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: identity }, { meshIndex: 0, instanceIndex: 3, objectToWorld: mirrored }],
    materials: [{ baseColor: [1, 1, 1] }],
    options: { surface: { positions: true } },
  });

  it('position is the barycentric combination through the instance matrix', () => {
    const r = ptVbufferResolve(copy, { instance: 0, triangle: 0, bary: [0.25, 0.5] });
    expect(r.position[0]).toBeCloseTo(0.5, 6);
    expect(r.position[1]).toBeCloseTo(1.5, 6);
    expect(r.position[2]).toBeCloseTo(0, 6);
    const m = ptVbufferResolve(copy, { instance: 3, triangle: 0, bary: [0.25, 0.5] });
    expect(m.position[0]).toBeCloseTo(10 - 0.5, 6);
    expect(m.position[1]).toBeCloseTo(20 + 1.5, 6);
    expect(m.position[2]).toBeCloseTo(30, 6);
  });

  it('the normal is the authored front side, also for a mirrored instance (bvh: the winding flips, the normal does not)', () => {
    expect(ptVbufferResolve(copy, { instance: 0, triangle: 0, bary: [0.2, 0.2] }).normal).toEqual([0, 0, 1]);
    const n = ptVbufferResolve(copy, { instance: 3, triangle: 0, bary: [0.2, 0.2] }).normal;
    expect(n[0]).toBeCloseTo(0, 6);
    expect(n[1]).toBeCloseTo(0, 6);
    expect(n[2]).toBeCloseTo(1, 6);
  });

  it('throws for an instance or triangle that is not in the scene copy', () => {
    expect(() => ptVbufferResolve(copy, { instance: 50, triangle: 0, bary: [0, 0] })).toThrow(RangeError);
    expect(() => ptVbufferResolve(copy, { instance: 0, triangle: 1, bary: [0, 0] })).toThrow(RangeError);
  });
});

describe("integrator.primary: 'vbuffer' kernels", () => {
  const plain = ptMegakernelWgsl(caps, { format: {}, layout });
  const vb = ptMegakernelWgsl(caps, { format: {}, layout, integrator: { primary: 'vbuffer' } });

  it("'trace' (the default) has no V-buffer code and spelling it out changes nothing", () => {
    expect(plain).not.toMatch(/ptVbuffer|PT_VBUFFER/);
    expect(ptMegakernelWgsl(caps, { format: {}, layout, integrator: { primary: 'trace' } })).toBe(plain);
  });

  it('the V-buffer kernel resolves the first hit from the texture, takes the jitter from the params and does not read sampler pair 0', () => {
    const readable = ptMegakernelWgsl(caps, { format: {}, layout, integrator: { primary: 'vbuffer' }, minify: false, shortenNames: false });
    expect(readable).toContain(`@group(2) @binding(${PT_VBUFFER_BINDING}) var ptVbuffer: texture_2d<u32>`);
    expect(readable).toContain('ptVbufferHit(px, ray)');
    expect(readable).toContain('vec2f(ptParams.origin.w, ptParams.right.w)');
    expect(readable).not.toContain('PT_PAIR_JITTER)');
    expect(vb.length).toBeGreaterThan(plain.length);
    expect(ptVbufferWgsl()).toContain('textureLoad(ptVbuffer');
  });

  it('needs positions in the surface layout (the hit position comes from the barycentrics)', () => {
    expect(() => ptMegakernelWgsl(caps, { format: {}, layout: { normals: true, uvs: false, positions: false }, integrator: { primary: 'vbuffer' } })).toThrow(/positions/);
  });

  it('the wavefront: camera takes the params jitter, intersect resolves depth-0 paths from the V-buffer; refill is refused', () => {
    const base = { format: {}, layout, minify: false, shortenNames: false, prune: false } as const;
    const traced = ptWavefrontBuild(caps, base);
    const built = ptWavefrontBuild(caps, { ...base, integrator: { primary: 'vbuffer' } });
    expect(traced.intersect.code).not.toContain('ptVbuffer');
    expect(built.intersect.code).toContain('ptVbufferHit(vec2u(');
    expect(built.camera.code).toContain('vec2f(ptParams.origin.w, ptParams.right.w)');
    expect(built.shade.code).toBe(ptWavefrontBuild(caps, { ...base, integrator: { primary: 'vbuffer', nee: true } }).shade.code);
    expect(() => ptWavefrontBuild(caps, { ...base, integrator: { primary: 'vbuffer' }, refill: true })).toThrow(/refill/);
  });
});
