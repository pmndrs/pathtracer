import { describe, expect, it } from 'vitest';
import { BLOB_BINS, DEFAULT_IMAGE_ENCODING, FREQUENCIES, NRC_ENCODED, type NrcInput, encodeNrc, encodeImage, frequency, imageEncodingWgsl, imageEncodingWidth, nrcEncodingWgsl, oneBlob, triangleWave } from './encoding.ts';

describe('image encoding layouts (N1.3)', () => {
  it('the default layout is the original one', () => {
    const e = encodeImage(0.3, 0.7);
    const legacy = new Float32Array(64);
    frequency(0.3, legacy, 0);
    frequency(0.7, legacy, FREQUENCIES);
    oneBlob(0.3, legacy, 2 * FREQUENCIES);
    oneBlob(0.7, legacy, 2 * FREQUENCIES + BLOB_BINS);
    legacy[2 * FREQUENCIES + 2 * BLOB_BINS] = 1;
    expect(Array.from(e)).toEqual(Array.from(legacy));
    expect(imageEncodingWidth(DEFAULT_IMAGE_ENCODING)).toBe(33);
  });

  it('two phases interleave the shifted wave per octave', () => {
    const e = encodeImage(0.3, 0.7, 64, undefined, { octaves: 8, phases: 2 });
    expect(e[0]).toBeCloseTo(triangleWave(0.3), 6);
    expect(e[1]).toBeCloseTo(triangleWave(0.3 + 0.5), 6);
    expect(e[2]).toBeCloseTo(triangleWave(0.6), 6);
    expect(e[3]).toBeCloseTo(triangleWave(0.6 + 0.5), 6);
    expect(e[16]).toBeCloseTo(triangleWave(0.7), 6); // v starts after 8 octaves × 2 phases
    expect(e[32 + 2 * BLOB_BINS]).toBe(1);
    expect(e[32 + 2 * BLOB_BINS + 1]).toBe(0);
    expect(imageEncodingWidth({ octaves: 8, phases: 2 })).toBe(41);
  });

  it('rejects layouts wider than the net', () => {
    expect(() => encodeImage(0, 0, 64, undefined, { octaves: 16, phases: 2 })).toThrow(/needs 73 inputs/);
  });

  it('WGSL twin writes the same slots', () => {
    const w = imageEncodingWgsl({ octaves: 8, phases: 2 }, 'imgEncode');
    expect(w).toContain('fn imgEncode(uv: vec2f) -> array<f32, NN_IN>');
    expect(w).toContain('e[a * 16u + f * 2u + 1u] = nnTriangle(x * scale + 0.5);');
    expect(w).toContain('nnOneBlob(uv.x, &e, 32u);');
    expect(w).toContain('e[40u] = 1.0;');
    expect(imageEncodingWgsl()).not.toContain('+ 0.5);');
  });
});

describe('NRC encoding WGSL twin', () => {
  it('fills the same 64 slots as encodeNrc', () => {
    const w = nrcEncodingWgsl();
    expect(3 * FREQUENCIES + 5 * BLOB_BINS + 8).toBe(NRC_ENCODED);
    expect(w).toContain('nnOneBlob(1.0 - exp(-roughness), &e, o + 4u * NN_BLOB_BINS);');
    expect(w).toContain('e[c + 6u] = 1.0; e[c + 7u] = 1.0;');
  });
});

it('reduces position bandwidth without shifting any material or angular features', () => {
  const input: NrcInput = { position: [0.1, -0.39, 0.62], direction: [0.2, 0.8], normal: [0.4, 0.7], roughness: 0.2, diffuse: [0.1, 0.2, 0.3], specular: [0.4, 0.3, 0.2] };
  const full = encodeNrc(input);
  const reduced = encodeNrc(input, new Float32Array(64).fill(1), 4);
  for (let i = 0; i < 64; i++) expect(reduced[i]).toBe(i < 36 && i % 12 >= 4 ? 0 : full[i]);
  for (const bands of [0, 4.5, 13]) {
    expect(() => nrcEncodingWgsl(bands)).toThrow(/positionBands/);
    expect(() => encodeNrc(input, undefined, bands)).toThrow(/positionBands/);
  }
});
