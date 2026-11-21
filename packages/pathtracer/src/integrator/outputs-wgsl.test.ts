import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptWavefrontBuild } from '../wavefront/wavefront.ts';
import { ptwfUnsupported } from '../wavefront/wavefront-wgsl.ts';
import { PT_AOV_VEC4, PT_COUNTER, PT_COUNTER_WORDS, PT_TERM_NAMES, ptCounterImage, ptDecodeAovs, ptHookIsUpdate, ptHooksWgsl, ptOutputsWgsl, ptSummarizeCounters } from './outputs-wgsl.ts';
import { ptMegakernelWgsl } from './megakernel.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const hook = { wgsl: 'fn ptOnVertex(v: PtPathVertex) -> u32 { return select(PT_HOOK_CONTINUE, PT_HOOK_TERMINATE, v.depth >= 3u); }\nfn ptTerminate(v: PtPathVertex) -> vec3f { return vec3f(0.5); }' };
const mk = (integrator?: object, minify = true) => ptMegakernelWgsl(caps, { format: {}, layout, integrator, minify, shortenNames: false } as never);

describe('optional integrator outputs (R10 hooks, R4 counters, AOVs)', () => {
  it('all off: the megakernel text equals the default one (and nothing of them is in it)', () => {
    const plain = mk();
    expect(mk({ hooks: null, counters: false, aovs: false })).toBe(plain);
    for (const name of ['PtPathVertex', 'ptOnVertex', 'ptCounters', 'ptCountRay', 'ptAovGuide', 'ptAovAlbedo']) expect(plain).not.toContain(name);
  });

  it('volumes + aovs (V5.1a): the first-vertex scatter writes the guides; with volumes.outputs it is refused; the wavefront writes them too (V5.1b)', () => {
    const volumes = { media: [{ sigmaT: [1, 1, 1], albedo: [0.9, 0.9, 0.9], g: 0.3 }], materials: [{ interior: 0, exterior: 0, boundary: false }, { interior: 1, exterior: 0, boundary: true }] };
    const k = mk({ volumes, aovs: true }, false);
    expect(k).toContain('ptVolMedium(curMed).albedo, -ray.direction');
    expect(k).toContain('ptAovGuide');
    expect(mk({ volumes }, false)).not.toContain('ptAovGuide');
    expect(() => mk({ volumes: { ...volumes, outputs: true }, aovs: true })).toThrow(/aovs.*volumes.outputs/);
    expect(ptwfUnsupported({ volumes, aovs: true } as never)).toBeNull(); // V5.1b: the wavefront writes the same guides
  });

  it('hooks: both functions are called after NEE and before the BSDF sample, in both schedulers', () => {
    const mega = mk({ hooks: hook }, false);
    expect(mega).toContain('ptOnVertex(hv)');
    expect(mega.indexOf('ptSampleLd(surf.position')).toBeLessThan(mega.indexOf('ptOnVertex(hv)'));
    expect(mega.indexOf('ptOnVertex(hv)')).toBeLessThan(mega.indexOf('let dirU'));
    const wf = ptWavefrontBuild(caps, { format: {}, layout, integrator: { hooks: hook }, minify: false, shortenNames: false } as never).shade.code;
    expect(wf).toContain('ptOnVertex(hv)');
    expect(wf.indexOf('ptwfSampleLd(surf.position')).toBeLessThan(wf.indexOf('ptOnVertex(hv)'));
    expect(wf.indexOf('ptOnVertex(hv)')).toBeLessThan(wf.indexOf('let dirU'));
    // the other four wavefront kernels do not carry the hook
    const build = ptWavefrontBuild(caps, { format: {}, layout, integrator: { hooks: hook } } as never);
    for (const k of ['camera', 'intersect', 'shadow', 'args'] as const) expect(build[k].code).not.toContain('PtPathVertex');
  });

  it('a missing hook function gets a default (continue / black); a user one is not duplicated', () => {
    const onlyTerminate = ptHooksWgsl({ wgsl: 'fn ptTerminate(v: PtPathVertex) -> vec3f { return vec3f(1.0); }' });
    expect(onlyTerminate.match(/fn ptOnVertex/g)).toHaveLength(1);
    expect(onlyTerminate.match(/fn ptTerminate/g)).toHaveLength(1);
    const both = ptHooksWgsl(hook);
    expect(both.match(/fn ptOnVertex/g)).toHaveLength(1);
    expect(both.match(/fn ptTerminate/g)).toHaveLength(1);
    expect(ptHooksWgsl(null)).toBe('');
  });

  it('provides evaluated material and path-segment inputs to both shade schedulers', () => {
    const hooks = {
      ...hook,
      evaluatedMaterial: true,
      pathSegment: true,
      wgsl: hook.wgsl + '\nfn ptOnRadiance(pixel: u32, sampleIndex: u32, index: u32, flags: u32, value: vec3f) {}',
    };
    const wave = ptWavefrontBuild(caps, { format: {}, layout, integrator: { hooks }, minify: false, shortenNames: false } as never);
    for (const shader of [mk({ hooks }, false), wave.shade.code]) {
      expect(shader).toContain('frame.n, ptHookRoughness(bp), ptHookDiffuseReflectance(bp), ptHookSpecularReflectance(bp)');
      expect(shader).toContain('distance(surf.position, prevP), prevPdf, specularBounce, ptBsdfIsDelta(bp)');
    }
    expect(wave.shadow.code).toContain('fn ptOnRadiance');
    expect(wave.shadow.code).not.toContain('fn ptHookRoughness');
    expect(mk({ hooks: hook }, false)).not.toContain('fn ptHookRoughness');
  });

  it('counters and AOVs add their bindings only when on; counters live at group 2 binding 3 (fuzz is 2), AOVs use no binding', () => {
    expect(mk({ counters: true })).toMatch(/@group\(2\)\s*@binding\(3\)\s*var<storage,\s*read_write>\s*ptCounters/);
    expect(mk({ aovs: true })).not.toContain('ptCounters');
    expect(mk({ aovs: true })).toContain('ptAovGuide');
    const wf = ptWavefrontBuild(caps, { format: {}, layout, integrator: { counters: true } } as never);
    expect(wf.shade.code).toMatch(/@group\(2\)\s*@binding\(8\)\s*var<storage,\s*read_write>\s*ptCounters/);
    expect(wf.intersect.code).not.toContain('ptCounters');
  });

  it('the AOV tail: radiance sums first, then PT_AOV_VEC4 vec4 per pixel; decoding averages and renormalises', () => {
    expect(PT_AOV_VEC4).toBe(2);
    const w = 2;
    const h = 1;
    const n = w * h;
    const sums = new Float32Array(n * 4 * (1 + PT_AOV_VEC4));
    // pixel 0: 4 samples, 3 hit: albedo (0.3 x 4 summed over 3 hits = 0.9 -> /4), normal (0, 2, 0) summed (+y twice), viewZ sum -6 over 3 hits
    sums.set([0.9, 0.9, 0.9, -6, 0, 2, 0, 3], n * 4);
    const a = ptDecodeAovs(sums, w, h, 4);
    expect(a.albedo[0]).toBeCloseTo(0.225, 6);
    expect([...a.normal.slice(0, 3)]).toEqual([0, 1, 0]);
    expect(a.viewZ[0]).toBeCloseTo(-2, 6);
    expect(a.coverage[0]).toBeCloseTo(0.75, 6);
    expect(a.viewZ[1]).toBe(0);
    expect([...a.normal.slice(3, 6)]).toEqual([0, 0, 0]);
  });

  it('counter summary: means over pixels and samples; reasons sum to 1', () => {
    const raw = new Uint32Array(2 * PT_COUNTER_WORDS);
    // 2 pixels, 2 samples: pixel 0 escaped twice at depth 1; pixel 1 hit maxDepth (3) and RR (1)
    raw.set([2, 4, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    raw.set([4, 7, 3, 0, 0, 0, 1, 1, 0, 0, 0, 0], PT_COUNTER_WORDS);
    const s = ptSummarizeCounters(raw, 2, 1, 2);
    expect(s.depth).toBeCloseTo(6 / 4, 9);
    expect(s.extensionRays).toBeCloseTo(11 / 4, 9);
    expect(s.shadowRays).toBeCloseTo(3 / 4, 9);
    expect(s.reasons.escaped).toBeCloseTo(0.5, 9);
    expect(s.reasons.maxDepth).toBeCloseTo(0.25, 9);
    expect(s.reasons.rr).toBeCloseTo(0.25, 9);
    expect(Object.values(s.reasons).reduce((x, y) => x + y, 0)).toBeCloseTo(1, 9);
    expect(PT_TERM_NAMES).toHaveLength(7);
    expect([...ptCounterImage(raw, 2, 1, 2, PT_COUNTER.extensionRays)]).toEqual([2, 3.5]);
  });

  it('update paths: one pixel per BxB block per sample, rotating over B^2 samples (TS twin of the WGSL flag)', () => {
    const B = 4;
    for (let sample = 0; sample < 32; sample++) {
      let updates = 0;
      for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) if (ptHookIsUpdate(x, y, sample, B)) updates++;
      expect(updates).toBe(16);
    }
    // each pixel is an update path exactly once in B^2 consecutive samples
    for (const [x, y] of [[0, 0], [5, 7], [15, 15]] as const) {
      let count = 0;
      for (let sample = 0; sample < B * B; sample++) if (ptHookIsUpdate(x, y, sample, B)) count++;
      expect(count).toBe(1);
    }
    expect(ptHookIsUpdate(3, 3, 0, 0)).toBe(false);
    expect(ptOutputsWgsl({ hooks: null, counters: false, aovs: false }, 3)).toBe('');
  });
});
