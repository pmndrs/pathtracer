import { buildCaps } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptWavefrontBuild } from '../wavefront/wavefront.ts';
import { ptwfUnsupported } from '../wavefront/wavefront-wgsl.ts';
import { PT_BLUENOISE_BINDING, ptBlueNoiseAlpha, ptBlueNoiseOffset, ptBlueNoiseShift, ptSampleBlueNoise, ptSampleOwenSobol, ptSamplerWgsl, type PtBlueNoiseData } from './dimensions.ts';
import { resolveIntegratorOptions } from './integrator-wgsl.ts';
import { ptMegakernelWgsl, validateBlueNoise } from './megakernel.ts';
import { PT_MODE_NOTES, PT_MODES, ptApplyMode, ptModeNotes, ptModePreset } from './modes.ts';
import { PT_FRAME_BINDING, PT_FRAME_PARAMS_BINDING, PT_FRAME_PLANES, ptDecodeFrame, ptOutputsWgsl } from './outputs-wgsl.ts';
import { ptCameraBasis } from './camera.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const mk = (integrator?: object, minify = true) => ptMegakernelWgsl(caps, { format: {}, layout, integrator, minify, shortenNames: false } as never);

/** A white-ish vec2 texture: not blue, but the sampler's structure (toroidal lookup, shift) does not depend on that. */
function fakeTexture(size: number): PtBlueNoiseData {
  const data = new Float32Array(size * size * 2);
  let a = 12345;
  for (let i = 0; i < data.length; i++) {
    a = (Math.imul(a, 1664525) + 1013904223) >>> 0;
    data[i] = (a >>> 8) / 2 ** 24;
  }
  return { width: size, height: size, data };
}

describe('blue-noise sampler (P4.1)', () => {
  const tex = fakeTexture(64);

  it('the default sampler text does not mention blue noise, the bluenoise text binds the texture', () => {
    const plain = ptSamplerWgsl('owen-sobol');
    expect(plain).not.toContain('ptBlueNoise');
    expect(ptSamplerWgsl('owen-sobol', { pairs: 3 })).toBe(plain);
    expect(ptSamplerWgsl('bluenoise')).toContain('PT_BN_OWEN: bool = true');
    const bn = ptSamplerWgsl('bluenoise', { pairs: 4, shift: 'kronecker' });
    expect(bn).toContain(`@group(2) @binding(${PT_BLUENOISE_BINDING}) var ptBlueNoise`);
    expect(bn).toContain('PT_BN_PAIRS: u32 = 4u');
    expect(bn).toContain('PT_BN_OWEN: bool = false');
    expect(ptSamplerWgsl('bluenoise', { shift: 'owen' })).toContain('PT_BN_OWEN: bool = true');
  });

  it('values are in [0, 1) and the shift is the same for every pixel: pixel differences are texture differences (mod 1)', () => {
    for (const shift of ['kronecker', 'owen'] as const) {
      for (let frame = 0; frame < 40; frame += 7) {
        const a = ptSampleBlueNoise(tex, 5, 9, frame, 3, { shift });
        const b = ptSampleBlueNoise(tex, 6, 9, frame, 3, { shift });
        for (const v of [...a, ...b]) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThan(1);
        }
        const [ox, oy] = ptBlueNoiseOffset(3);
        const ta = tex.data.slice(((9 + oy) % 64) * 128 + ((5 + ox) % 64) * 2, ((9 + oy) % 64) * 128 + ((5 + ox) % 64) * 2 + 2);
        const tb = tex.data.slice(((9 + oy) % 64) * 128 + ((6 + ox) % 64) * 2, ((9 + oy) % 64) * 128 + ((6 + ox) % 64) * 2 + 2);
        for (let c = 0; c < 2; c++) {
          const want = (((tb[c] as number) - (ta[c] as number)) % 1 + 1) % 1;
          const got = (((b[c] as number) - (a[c] as number)) % 1 + 1) % 1;
          expect(Math.min(Math.abs(got - want), 1 - Math.abs(got - want))).toBeLessThan(1e-6);
        }
      }
    }
  });

  it('unbiased in time: the mean of a pixel over many frames is 1/2 (both shifts, every pair), and its 2D distribution is flat', () => {
    for (const shift of ['kronecker', 'owen'] as const) {
      for (const pair of [0, 2, 3, 5]) {
        const n = 4096;
        let sx = 0;
        let sy = 0;
        const cells = new Uint16Array(16);
        for (let f = 0; f < n; f++) {
          const [x, y] = ptSampleBlueNoise(tex, 11, 30, f, pair, { shift });
          sx += x;
          sy += y;
          cells[Math.floor(y * 4) * 4 + Math.floor(x * 4)]!++;
        }
        expect(Math.abs(sx / n - 0.5)).toBeLessThan(0.003);
        expect(Math.abs(sy / n - 0.5)).toBeLessThan(0.003);
        for (const c of cells) expect(Math.abs(c - n / 16)).toBeLessThan(n / 16 / 8);
      }
    }
  });

  it('the owen shift is a (0, m, 2)-net over 2^m frames, shifted: points stay on the sample lattice of the net', () => {
    // shift alone (texture value 0 is not available, so test the shift function directly)
    const n = 64;
    const seen = new Set<number>();
    for (let f = 0; f < n; f++) {
      const [x, y] = ptBlueNoiseShift('owen', 2, f);
      seen.add(Math.floor(y * 8) * 8 + Math.floor(x * 8));
    }
    expect(seen.size).toBe(n);
  });

  it('kronecker steps are distinct and not related across pairs (no diagonal): all 2 x 15 components differ', () => {
    const alphas = Array.from({ length: 15 }, (_, k) => ptBlueNoiseAlpha(k)).flat();
    expect(new Set(alphas).size).toBe(alphas.length);
    expect(ptBlueNoiseAlpha(0)[0] / 2 ** 32).toBeCloseTo(0.7548776662, 8);
  });

  it('pairs at or past the limit are the Owen-Sobol pair of the pixel seed', () => {
    for (const pair of [6, 7, 20]) expect(ptSampleBlueNoise(tex, 3, 4, 9, pair, { pairs: 6 }, 777)).toEqual(ptSampleOwenSobol(777, 9, pair));
  });

  it('different pairs read different texels (their values at one pixel are not a function of each other)', () => {
    const n = 64;
    let s = 0;
    for (let x = 0; x < n; x++) for (let y = 0; y < n; y++) s += (ptSampleBlueNoise(tex, x, y, 0, 2)[0] - 0.5) * (ptSampleBlueNoise(tex, x, y, 0, 3)[0] - 0.5);
    expect(Math.abs(s / (n * n) / (1 / 12))).toBeLessThan(0.1);
  });

  it('the kernel: bluenoise binds the texture and sets the pixel; the wavefront refuses it', () => {
    const code = mk({ sampler: 'bluenoise' }, false);
    expect(code).toContain('ptBnPixel = px;');
    expect(code).toContain('textureLoad(ptBlueNoise');
    expect(mk({})).not.toContain('ptBlueNoise');
    expect(ptwfUnsupported({ sampler: 'bluenoise' })).toMatch(/bluenoise/);
    expect(() => ptWavefrontBuild(caps, { format: {}, layout, integrator: { sampler: 'bluenoise' } } as never)).toThrow(/bluenoise/);
  });

  it('validateBlueNoise wants a vec2, power-of-two texture', () => {
    const meta = { name: 'x', kind: 'gf', width: 64, height: 64, depth: 1, channels: 2, f32: '', png: '', atlasColumns: 1, sigmaSpatial: 1, sigmaTemporal: null, seed: 0 } as const;
    expect(() => validateBlueNoise(undefined)).toThrow(/blueNoise/);
    expect(() => validateBlueNoise({ meta: { ...meta, channels: 1 }, data: new Float32Array(1) })).toThrow(/vec2/);
    expect(() => validateBlueNoise({ meta: { ...meta, width: 48 }, data: new Float32Array(1) })).toThrow(/power-of-two/);
    expect(() => validateBlueNoise({ meta, data: new Float32Array(1) })).not.toThrow();
  });
});

describe('per-frame outputs (P4.1)', () => {
  it('off: the kernel text is the default one and carries nothing of them', () => {
    const plain = mk();
    expect(mk({ frame: false, accumulate: true })).toBe(plain);
    for (const name of ['ptFramePlanes', 'ptFrameParams', 'ptFrameMotion', 'ptAovSpecular', 'fDirect']) expect(plain).not.toContain(name);
    expect(plain).toMatch(/ptAccum\[index\]\s*=\s*ptAccum\[index\]\s*\+\s*vec4f\(l\.xyz,\s*1\.0\)/);
  });

  it('on: planes, the previous-camera uniform, overwrite instead of accumulate', () => {
    const code = mk({ frame: true }, false);
    expect(code).toContain(`@group(2) @binding(${PT_FRAME_PARAMS_BINDING}) var<uniform> ptFrameParams`);
    expect(code).toContain(`@group(2) @binding(${PT_FRAME_BINDING}) var<storage, read_write> ptFrame`);
    expect(code).toContain('ptAccum[index] = vec4f(l.xyz, 1.0)');
    expect(code).toContain('ptFramePlanes(pxIndex');
    expect(resolveIntegratorOptions({ frame: true }).accumulate).toBe(false);
    expect(resolveIntegratorOptions({ frame: true, accumulate: true }).accumulate).toBe(true);
    expect(resolveIntegratorOptions({}).accumulate).toBe(true);
  });

  it('the megakernel with everything the real-time mode adds stays under the 50 KB DXC guideline (plain Cornell-class scene)', () => {
    const size = (integrator: object) => ptMegakernelWgsl(caps, { format: {}, layout, integrator } as never).length;
    expect(size({ frame: true, sampler: 'bluenoise' })).toBeLessThan(50_000);
    expect(size({ frame: true, sampler: 'bluenoise' })).toBeGreaterThan(size({}));
  });

  it('the wavefront refuses frame outputs; outputs text gates on the flag', () => {
    expect(ptwfUnsupported({ frame: true })).toMatch(/frame/);
    expect(ptOutputsWgsl({ hooks: null, counters: false, aovs: false }, null)).toBe('');
    expect(ptOutputsWgsl({ hooks: null, counters: false, aovs: false, frame: true }, null)).toContain('fn ptFramePlanes');
  });

  it('decode splits the planes in order', () => {
    const w = 2;
    const h = 1;
    const n = w * h * 4;
    const planes = new Float32Array(PT_FRAME_PLANES * n).map((_, i) => i);
    const radiance = new Float32Array(n).fill(7);
    const f = ptDecodeFrame(radiance, planes, w, h);
    expect(f.diffuse[0]).toBe(0);
    expect(f.specular[0]).toBe(n);
    expect(f.normalViewZ[0]).toBe(2 * n);
    expect(f.motionRoughness[0]).toBe(3 * n);
    expect(f.albedo[0]).toBe(4 * n);
    expect(f.specularFactor[0]).toBe(5 * n);
    expect(f.emission[0]).toBe(6 * n);
    expect(PT_FRAME_PLANES).toBe(7);
    expect(f.radiance).toBe(radiance);
  });

  it('frameEmission (P4.5a): off without frame, mirror by default with it; the kernel text carries the emission plane exactly as asked', () => {
    expect(resolveIntegratorOptions({}).frameEmission).toBe('none');
    expect(resolveIntegratorOptions({ frameEmission: 'mirror' }).frameEmission).toBe('none'); // no frame, no plane
    expect(resolveIntegratorOptions({ frame: true }).frameEmission).toBe('mirror');
    expect(resolveIntegratorOptions({ frame: true, frameEmission: 'none' }).frameEmission).toBe('none');
    expect(mk({ frameEmission: 'primary' })).toBe(mk()); // not a frame kernel: the text is the plain one
    const mirror = mk({ frame: true }, false);
    expect(mirror).toContain('var fEmit = vec3f(0.0);');
    expect(mirror).toContain('var fPure = true;');
    expect(mirror).toContain('fPure = ptFramePureNext(fPure, specularBounce, bp.kind);');
    expect(mirror).toContain('PtRadiance(radiance + fEmit, 0.0)');
    expect(mirror).not.toContain('fDirect');
    const primary = mk({ frame: true, frameEmission: 'primary' }, false);
    expect(primary).toContain('var fEmit = vec3f(0.0);');
    expect(primary).not.toContain('fPure');
    expect(primary).toContain('if (depth == 0u) {\n                fEmit +=');
    const none = mk({ frame: true, frameEmission: 'none' }, false);
    expect(none).not.toContain('fEmit');
    expect(none).toContain('fDirect');
    expect(none).toContain('PtRadiance(radiance, 0.0)');
  });

  it('the WGSL projection is the closed form of the camera basis (right, up scaled by tan(fov/2)): NDC +y up, viewZ negative in front', () => {
    const b = ptCameraBasis({ position: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], yFovDeg: 90 }, 2, 1);
    // a point straight ahead at distance 2, one unit up: ndc.y = 0.5 for a 90 degree vertical fov (tan = 1) at z = 2? (1 / 2) / 1
    const p = [0, 1, -2];
    const dot = (a: number[], c: number[]) => a[0]! * c[0]! + a[1]! * c[1]! + a[2]! * c[2]!;
    const z = dot(p, b.forward);
    expect(dot(p, b.up) / (dot(b.up, b.up) * z)).toBeCloseTo(0.5, 12);
    expect(dot(p, b.right) / (dot(b.right, b.right) * z)).toBeCloseTo(0, 12);
    expect(-z).toBe(-2);
  });
});

describe('mode presets (P4.1)', () => {
  it('reference changes nothing; no mode returns the options untouched', () => {
    const o = { integrator: { nee: false } };
    expect(ptApplyMode(o)).toBe(o);
    expect(ptApplyMode({ ...o, mode: 'reference' as const }).integrator).toEqual({ nee: false });
    expect(ptModePreset('reference').limits).toBeNull();
  });

  it('explicit options win over the preset; undefined does not', () => {
    const r = ptApplyMode({ mode: 'realtime' as const, integrator: { alphaPolicy: 'exact' as const, materialLod: undefined } });
    expect(r.integrator.alphaPolicy).toBe('exact');
    expect(r.integrator.materialLod).toBe('secondary');
    expect(r.integrator.textureLod).toBe('cone');
    expect(r.integrator.frame).toBe(true);
    expect(ptApplyMode({ mode: 'realtime' as const, integrator: { frame: false } }).integrator.frame).toBe(false);
  });

  it('realtime uses the blue-noise sampler only when the texture is given, and not on the wavefront', () => {
    expect(ptApplyMode({ mode: 'realtime' as const, integrator: {} }).integrator.sampler).toBe('owen-sobol');
    expect(ptApplyMode({ mode: 'realtime' as const, blueNoise: {}, integrator: {} }).integrator.sampler).toBe('bluenoise');
    expect(ptApplyMode({ mode: 'realtime' as const, blueNoise: {}, integrator: {} }, 'wavefront').integrator.sampler).toBe('owen-sobol');
    expect(ptApplyMode({ mode: 'realtime' as const, integrator: {} }, 'wavefront').integrator.frame).toBeUndefined();
    expect(ptApplyMode({ mode: 'realtime' as const, integrator: { sampler: 'pcg' as const }, blueNoise: {} }).integrator.sampler).toBe('pcg');
    expect(ptApplyMode({ mode: 'realtime' as const, integrator: { restir: true } as object, blueNoise: {} }).integrator?.sampler).toBe('bluenoise'); // P4.2e: ReSTIR binds the texture too
  });

  it('preview: D31 cutouts (exact + transmit), reduced secondary materials, AOVs for OIDN, no frame outputs', () => {
    const p = ptApplyMode({ mode: 'preview' as const, integrator: {} }).integrator;
    expect(p.alphaPolicy).toBe('exact');
    expect(p.alphaShadow).toBe('transmit');
    expect(p.alphaTransmit).toBe(0.45);
    expect(p.materialLod).toBe('secondary');
    expect(p.aovs).toBe(true);
    expect(p.frame).toBeUndefined();
  });

  it('D31 max-speed cutouts: asking for a solid policy drops the preset\'s transmit shadows', () => {
    for (const mode of ['preview', 'realtime'] as const) {
      const fast = ptApplyMode({ mode, integrator: { alphaPolicy: 'opaqueShadow' as const } }).integrator;
      expect(fast.alphaPolicy).toBe('opaqueShadow');
      expect(fast.alphaShadow).toBeUndefined();
      expect(fast.alphaTransmit).toBeUndefined();
      const own = ptApplyMode({ mode, integrator: { alphaPolicy: 'opaqueShadow' as const, alphaShadow: 'coverage' as const } }).integrator;
      expect(own.alphaShadow).toBe('coverage');
      expect(ptApplyMode({ mode, integrator: {} }).integrator.alphaShadow).toBe('transmit');
    }
  });

  it('the D24 layering merges into layered materials where the caller left a field undefined', () => {
    const materials = { features: { layered: true }, layered: { layering: 'stochastic' as const, albedoBase: undefined } };
    const r = ptApplyMode({ mode: 'realtime' as const, integrator: { materials: materials as never } }).integrator.materials as unknown as { layered: Record<string, unknown> };
    expect(r.layered.layering).toBe('stochastic');
    expect(r.layered.albedoBase).toBe('internal');
    expect(r.layered.albedoCorrection).toBe('texture');
    const none = { features: { layered: false } };
    expect(ptApplyMode({ mode: 'realtime' as const, integrator: { materials: none as never } }).integrator.materials).toBe(none);
  });

  it('every mode entry is documented (a measured number or "pending"), and unknown modes throw', () => {
    for (const mode of PT_MODES) {
      const notes = ptModeNotes(mode);
      expect(notes.length).toBeGreaterThan(0);
      const preset = ptModePreset(mode);
      for (const key of Object.keys(preset.integrator)) expect(PT_MODE_NOTES[`${mode}.${key}`] ?? PT_MODE_NOTES[`${mode}.*`], `${mode}.${key}`).toBeTruthy();
      if (preset.limits) expect(PT_MODE_NOTES[`${mode}.limits`]).toBeTruthy();
      if (preset.layered) expect(PT_MODE_NOTES[`${mode}.layered`]).toBeTruthy();
      if (preset.scenePacking) expect(PT_MODE_NOTES[`${mode}.scenePacking`]).toBeTruthy();
    }
    expect(() => ptModePreset('turbo' as never)).toThrow(/unknown mode/);
  });

  it('realtime resolves to a kernel the megakernel can build (frame + blue noise) and to a wavefront without them', () => {
    const o = ptApplyMode({ mode: 'realtime' as const, blueNoise: {}, integrator: {} });
    expect(() => mk(o.integrator)).not.toThrow();
    const wf = ptApplyMode({ mode: 'realtime' as const, blueNoise: {}, integrator: {} }, 'wavefront');
    expect(() => ptWavefrontBuild(caps, { format: {}, layout, integrator: wf.integrator } as never)).not.toThrow();
  });
});
