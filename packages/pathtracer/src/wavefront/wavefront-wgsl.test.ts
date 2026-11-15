import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptMegakernelBuild } from '../integrator/megakernel.ts';
import { PTWF_KERNELS, ptWavefrontBuild, ptWavefrontKernelBytes } from './wavefront.ts';
import { ptwfUnsupported } from './wavefront-wgsl.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const texLayout = { normals: true, uvs: true, positions: true };
const bucket = { width: 256, height: 256, colorSpace: 'srgb' as const, wrap: 'repeat' as const, layers: 2, mipLevels: 9 };
const feat = { mix: true, thin: true, layered: true, medium: true };

const variants: Record<string, Parameters<typeof ptWavefrontBuild>[1]> = {
  plain: { format: {}, layout },
  env: { format: {}, layout, integrator: { envMap: true, envSampling: 'importance' } },
  pcg: { format: {}, layout, integrator: { sampler: 'pcg', nee: false } },
  tex: { format: {}, layout: texLayout, textures: { buckets: [bucket] } },
  'pbr-albedo': { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'albedo' } } } } as never,
  'pbr-stochastic': { format: {}, layout, integrator: { materials: { features: feat, layered: { layering: 'stochastic', maxDepth: 64 } } } } as never,
  'pbr-fuzz': { format: {}, layout, integrator: { materials: { features: { ...feat, fuzz: true }, layered: { layering: 'albedo' } } } } as never,
};

describe('wavefront WGSL (P3.1)', () => {
  for (const [name, options] of Object.entries(variants)) {
    it(`${name}: every kernel is under the 50 KB budget (D26), names its entry point and has no leftover directives`, () => {
      const build = ptWavefrontBuild(caps, options);
      const bytes = ptWavefrontKernelBytes(build);
      const entries = { camera: 'ptwfCamera', intersect: 'ptwfIntersect', shade: 'ptwfShade', shadow: 'ptwfShadow', args: 'ptwfArgsMain' } as const;
      for (const k of PTWF_KERNELS) {
        expect(bytes[k], `${name} ${k}`).toBeLessThan(50_000);
        expect(build[k].code).toContain(entries[k]);
        expect(build[k].code).not.toMatch(/^#/m);
      }
    });
  }

  it('the shade kernel is smaller than the megakernel (no walk code), intersect and shadow are walk only', () => {
    const wf = ptWavefrontKernelBytes(ptWavefrontBuild(caps, variants.plain!));
    const mk = ptMegakernelBuild(caps, variants.plain!).code.length;
    expect(wf.shade).toBeLessThan(mk);
    expect(wf.intersect).toBeLessThan(wf.shade);
    expect(wf.args).toBeLessThan(2000);
  });

  it('only the kernels that use a binding declare it (args: params + counters + args)', () => {
    const b = ptWavefrontBuild(caps, { ...variants.plain!, prune: false, shortenNames: false, minify: false });
    expect(b.args.code).toContain('@group(3) @binding(0)');
    expect(b.args.code).not.toContain('@group(0)');
    expect(b.args.code).not.toContain('ptwfHit');
    expect(b.camera.code).not.toContain('@group(0)');
    expect(b.shade.code).toContain('@group(1)');
    expect(b.shade.code).not.toContain('@group(0)'); // no walk bindings: a declared binding must be in the kernel's layout
    expect(b.intersect.code).toContain('@group(0)');
    expect(b.shadow.code).toContain('@group(0)');
    expect(b.intersect.code).not.toContain('@group(1)');
  });

  it('alpha MASK throws a clear error', () => {
    expect(ptwfUnsupported({ alphaMask: true, textures: true })).toMatch(/alpha MASK/);
    expect(() => ptWavefrontBuild(caps, { format: {}, layout: texLayout, textures: { buckets: [bucket] }, integrator: { alphaMask: true } })).toThrow(/alpha MASK/);
    expect(ptwfUnsupported({})).toBeNull();
  });

  it('the shade kernel carries the same estimator constants and sampler pairs as the megakernel', () => {
    const wf = ptWavefrontBuild(caps, { ...variants.plain!, prune: false, shortenNames: false, minify: false }).shade.code;
    const mk = ptMegakernelBuild(caps, { ...variants.plain!, prune: false, shortenNames: false, minify: false }).code;
    for (const line of ['const PT_FLOAT_SCALE: f32 = 0.0000152587890625;', 'const PT_NEE: bool = true;', 'fn ptMis(pThis: f32, pOther: f32) -> f32 {', 'fn ptEscaped(dir: vec3f, direct: bool, prevPdf: f32) -> vec3f {']) {
      expect(mk).toContain(line);
      expect(wf).toContain(line);
    }
    for (const call of ['ptVertexPair(vertex, PT_SLOT_BSDF_DIRECTION)', 'ptVertexPair(vertex, PT_SLOT_SELECT_COMPONENT)', 'ptVertexPair(vertex, PT_SLOT_RR)', 'ptVertexPair(vertex, PT_SLOT_LIGHT_POINT)']) {
      expect(mk).toContain(call);
      expect(wf).toContain(call);
    }
    // the NEE sampler's light-sampling body up to the visibility test is the megakernel's text, line for line
    const body = (code: string, start: string, end: string) => code.slice(code.indexOf(start), code.indexOf(end)).split('\n').slice(1).join('\n');
    const a = body(mk, 'fn ptSampleLd(', 'let pl = pick.pmf');
    const b = body(wf, 'fn ptwfSampleLd(', 'let pl = pick.pmf');
    const norm = (t: string) => t.replace(/if \(!(ptVisibleMasked|rtVisible)\(ray, RT_MASK_ALL\)\) \{\s*return vec3f\(0\.0\);\s*\}\s*/, '').replace(/return vec3f\(0\.0\);/g, 'return none;').replace(/let none = [^\n]*\n/, '').split('\n').map((l) => l.trim()).filter(Boolean).join('\n');
    expect(norm(b)).toBe(norm(a));
  });
});
