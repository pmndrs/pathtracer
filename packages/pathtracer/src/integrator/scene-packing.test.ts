/**
 * Scene copy packing (P3.11) in the integrator kernels: a packed kernel binds one scene buffer, reaches every table / data word through the
 * rewritten accessors (nothing of the split names is left, in the appended integrator text too), and the split kernels are the pre-D32 default text (`scenePacking: 'split'`); the library default is `packed` since D32.
 * The images of the two are compared on the GPU (`matrix.ts --renderers pt-megakernel,pt-mk-packed,pt-wavefront,pt-wf-packed`).
 */
import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptMegakernelBuild } from './megakernel.ts';
import { ptWavefrontBuild, PTWF_KERNELS } from '../wavefront/wavefront.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const texLayout = { normals: true, uvs: true, positions: true };
const bucket = (srgb: boolean) => ({ width: 8, height: 8, colorSpace: srgb ? ('srgb' as const) : ('linear' as const), wrap: 'repeat' as const, layers: 1, mipLevels: 4 });
const textures = { buckets: [bucket(true), bucket(false)] };
const feat = { mix: true, thin: true, layered: true, medium: true, thinFilm: true, fuzz: true };

const variants: Array<[string, object]> = [
  ['plain', { format: {}, layout }],
  ['env', { format: {}, layout, integrator: { envMap: true, envSampling: 'importance' } }],
  ['tex-mask', { format: {}, layout: texLayout, textures, integrator: { alphaMask: true } }],
  ['pbr', { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } } } }],
  ['vbuffer', { format: {}, layout, integrator: { primary: 'vbuffer' } }],
];

const groupBindings = (code: string, group: number): number[] => [...code.matchAll(new RegExp(`@group\\(${group}\\)\\s*@binding\\((\\d+)\\)`, 'g'))].map((m) => Number(m[1]));

describe('packed megakernel / wavefront text (P3.11)', () => {
  for (const [name, options] of variants) {
    it(`${name}: megakernel binds one scene buffer and no split name is left; the default is packed`, () => {
      const split = ptMegakernelBuild(caps, { ...options, scenePacking: 'split' } as never).code;
      const packed = ptMegakernelBuild(caps, { ...options, scenePacking: 'packed' } as never).code;
      expect(ptMegakernelBuild(caps, options as never).code).toBe(packed);
      expect(split).toMatch(/rtSceneTable/);
      expect(packed).not.toMatch(/rtSceneTable|rtSceneData/);
      expect(packed).toMatch(/@group\(1\)\s*@binding\(0\)var<storage,\s*read>\s*rtScene:\s*array<vec4u>/);
      // Bindings 0.. (buffers, then textures / samplers): one fewer, the rest shifted down by one.
      const slots = groupBindings(packed, 1);
      expect(slots).toEqual(groupBindings(split, 1).slice(0, -1));
    });

    it(`${name}: every wavefront kernel is rewritten the same way`, () => {
      const split = ptWavefrontBuild(caps, { ...options, scenePacking: 'split' } as never);
      const packed = ptWavefrontBuild(caps, { ...options, scenePacking: 'packed' } as never);
      for (const kernel of PTWF_KERNELS) {
        expect(packed[kernel].code, kernel).not.toMatch(/rtSceneTable|rtSceneData/);
        const had = /rtSceneTable/.test(split[kernel].code);
        expect(/rtScene\b/.test(packed[kernel].code), kernel).toBe(had);
      }
    });
  }

  it('the packed kernel stays within a hundred bytes of the split one (the data accessor replaces a binding)', () => {
    const split = ptMegakernelBuild(caps, { format: {}, layout, scenePacking: 'split' } as never).code.length;
    const packed = ptMegakernelBuild(caps, { format: {}, layout, scenePacking: 'packed' } as never).code.length;
    expect(Math.abs(packed - split)).toBeLessThan(400);
  });
});
