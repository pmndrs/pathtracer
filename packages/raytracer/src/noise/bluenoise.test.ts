import { describe, expect, it } from 'vitest';
import { type BlueNoiseMeta, blueNoiseAt, parseBlueNoise } from './bluenoise.ts';

const meta: BlueNoiseMeta = {
  name: 't', kind: 'st', width: 2, height: 2, depth: 2, channels: 2, f32: 't.f32', png: 't.png',
  atlasColumns: 2, sigmaSpatial: 1.5, sigmaTemporal: 1.5, seed: 0,
};

describe('blue-noise runtime loader', () => {
  const values = Float32Array.from({ length: 16 }, (_, i) => i / 16);

  it('parses raw float32 from unaligned bytes and checks the size', () => {
    const bytes = new Uint8Array(1 + 64);
    bytes.set(new Uint8Array(values.buffer), 1);
    expect(parseBlueNoise(bytes.subarray(1), meta).data).toEqual(values);
    expect(() => parseBlueNoise(bytes.subarray(2), meta)).toThrow(/expected 64 bytes/);
  });

  it('reads texels with toroidal wrap in x, y and z', () => {
    const tex = parseBlueNoise(values.buffer, meta);
    // index = ((z * h + y) * w + x) * channels + c
    expect(blueNoiseAt(tex, 1, 0, 1, 1)).toBe(values[((1 * 2 + 0) * 2 + 1) * 2 + 1]);
    expect(blueNoiseAt(tex, -1, 2, 3, 0)).toBe(blueNoiseAt(tex, 1, 0, 1, 0));
  });
});
