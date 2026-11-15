import { describe, expect, it } from 'vitest';
import {
  SRGB_TO_LINEAR,
  applyNormalMap,
  bucketSize,
  buildSceneTextures,
  coneUvFootprint,
  downsampleBytes,
  linearToSrgb8,
  mipLevelCount,
  packTextureSlot,
  resampleTexture,
  sampleSceneTexture,
  srgbToLinear,
  textureLod,
  triangleTextureFrame,
  unpackTextureSlot,
  wrapIndex,
  type TextureInput,
} from './texture.ts';

/** A w x h RGBA8 texture whose texel (x, y) is `f(x, y)`. */
function make(w: number, h: number, f: (x: number, y: number) => [number, number, number, number], extra: Partial<TextureInput> = {}): TextureInput {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(f(x, y), (y * w + x) * 4);
  return { width: w, height: h, data, colorSpace: 'linear', ...extra };
}

describe('sRGB', () => {
  it('decodes the end points and the 0.5 mid-grey exactly as IEC 61966-2-1 says', () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(255)).toBeCloseTo(1, 7);
    expect(srgbToLinear(128)).toBeCloseTo(0.21586, 5);
    expect(srgbToLinear(10)).toBeCloseTo(10 / 255 / 12.92, 7); // linear segment below 0.04045
  });

  it('encode(decode(c)) is the identity for all 256 codes', () => {
    for (let c = 0; c < 256; c++) expect(linearToSrgb8(SRGB_TO_LINEAR[c] as number)).toBe(c);
  });

  it('clamps out-of-range input', () => {
    expect(linearToSrgb8(-1)).toBe(0);
    expect(linearToSrgb8(7)).toBe(255);
  });
});

describe('bucket sizes', () => {
  it('keeps powers of two, rounds others to the nearest in log space, and caps', () => {
    expect(bucketSize(256, 'nearest', 4096)).toBe(256);
    expect(bucketSize(300, 'nearest', 4096)).toBe(256);
    expect(bucketSize(400, 'nearest', 4096)).toBe(512); // sqrt(256*512) = 362
    expect(bucketSize(362, 'nearest', 4096)).toBe(256);
    expect(bucketSize(363, 'nearest', 4096)).toBe(512);
    expect(bucketSize(300, 'up', 4096)).toBe(512);
    expect(bucketSize(500, 'down', 4096)).toBe(256);
    expect(bucketSize(8192, 'nearest', 4096)).toBe(4096);
    expect(() => bucketSize(0, 'nearest', 4096)).toThrow();
  });

  it('counts mip levels down to 1x1', () => {
    expect(mipLevelCount(1, 1)).toBe(1);
    expect(mipLevelCount(256, 256)).toBe(9);
    expect(mipLevelCount(1024, 64)).toBe(11);
  });
});

describe('resampling and mips', () => {
  it('copies the bytes unchanged when the size already fits', () => {
    const t = make(4, 4, (x, y) => [x * 40, y * 40, 7, 255]);
    expect(Array.from(resampleTexture(t, 4, 4))).toEqual(Array.from(t.data as Uint8Array));
  });

  it('a constant texture stays constant under any resize, in both colour spaces', () => {
    for (const colorSpace of ['linear', 'srgb'] as const) {
      const t = make(6, 10, () => [100, 150, 200, 255], { colorSpace });
      for (const [w, h] of [[8, 8], [4, 16], [2, 2]] as const) {
        const out = resampleTexture(t, w, h);
        for (let i = 0; i < out.length; i += 4) expect(Array.from(out.subarray(i, i + 4))).toEqual([100, 150, 200, 255]);
      }
    }
  });

  it('a 2x2 box mip of a linear texture is the rounded mean; of an sRGB texture the mean in linear light', () => {
    const lin = new Uint8Array([0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255]);
    expect(Array.from(downsampleBytes(lin, 2, 2, false))).toEqual([128, 128, 128, 255]);
    // sRGB black/white checker: the mean in linear light is 0.5, which is sRGB code 188, not 128.
    expect(Array.from(downsampleBytes(lin, 2, 2, true))).toEqual([188, 188, 188, 255]);
  });

  it('builds a full chain whose top is one texel holding the (linear-light) mean', () => {
    const t = make(8, 8, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255, 255] : [0, 0, 0, 255]), { colorSpace: 'srgb' });
    const s = buildSceneTextures([t]);
    const bucket = s.buckets[0]!;
    expect(bucket.mipLevels).toBe(4);
    expect(bucket.mips.map((m) => m.length)).toEqual([8 * 8 * 4, 4 * 4 * 4, 2 * 2 * 4, 4]);
    expect(Array.from(bucket.mips[3]!.subarray(0, 3))).toEqual([188, 188, 188]);
    expect(s.byteLength).toBe(256 + 64 + 16 + 4);
  });

  it('rows stay rows: resampling a vertical gradient keeps it monotone and keeps row 0 on top', () => {
    const t = make(5, 20, (_x, y) => [Math.round((y / 19) * 255), 0, 0, 255]);
    const out = resampleTexture(t, 4, 16);
    let last = -1;
    for (let y = 0; y < 16; y++) {
      const v = out[(y * 4) * 4]!;
      expect(v).toBeGreaterThanOrEqual(last);
      last = v;
    }
    expect(out[0]!).toBeLessThan(20);
    expect(out[15 * 16]!).toBeGreaterThan(235);
  });
});

describe('bucket packing', () => {
  const sq = (n: number, colorSpace: 'srgb' | 'linear' = 'linear', wrap?: 'repeat' | 'clamp' | 'mirror') => make(n, n, () => [1, 2, 3, 255], { colorSpace, ...(wrap ? { wrap } : {}) });

  it('puts same-size same-space same-wrap textures in one array, one layer each, in input order', () => {
    const s = buildSceneTextures([sq(16), sq(32), sq(16), sq(16)]);
    expect(s.buckets.map((b) => [b.width, b.layers])).toEqual([[16, 3], [32, 1]]);
    expect(s.slots.map((x) => [x.bucket, x.layer])).toEqual([[0, 0], [1, 0], [0, 1], [0, 2]]);
  });

  it('separates colour spaces and wrap modes (one format and one sampler per array)', () => {
    const s = buildSceneTextures([sq(16, 'srgb'), sq(16, 'linear'), sq(16, 'linear', 'clamp'), sq(16, 'linear', 'mirror')]);
    expect(s.buckets.length).toBe(4);
    expect(new Set(s.buckets.map((b) => `${b.colorSpace}/${b.wrap}`)).size).toBe(4);
  });

  it('rounds non-power-of-two sizes to buckets and records the source size', () => {
    const s = buildSceneTextures([make(300, 100, () => [9, 9, 9, 255])]);
    expect([s.buckets[0]!.width, s.buckets[0]!.height]).toEqual([256, 128]);
    expect([s.slots[0]!.sourceWidth, s.slots[0]!.sourceHeight]).toEqual([300, 100]);
  });

  it('splits an array past maxLayers into two arrays', () => {
    const s = buildSceneTextures([sq(4), sq(4), sq(4)], { maxLayers: 2 });
    expect(s.buckets.map((b) => b.layers)).toEqual([2, 1]);
    expect(s.slots.map((x) => [x.bucket, x.layer])).toEqual([[0, 0], [0, 1], [1, 0]]);
  });

  it('merges the cheapest same-space pair when there are more arrays than maxBuckets, and fails when merging cannot help', () => {
    const inputs = [sq(16), sq(32), sq(256)];
    const s = buildSceneTextures(inputs, { maxBuckets: 2 });
    expect(s.buckets.length).toBe(2);
    expect(s.buckets.map((b) => b.width)).toEqual([32, 256]); // 16 and 32 merge: cheapest
    expect(s.slots[0]!.width).toBe(32);
    expect(() => buildSceneTextures([sq(16, 'srgb'), sq(16, 'linear')], { maxBuckets: 1 })).toThrow(/at most 1/);
  });

  it('checks sizes and data length', () => {
    expect(() => buildSceneTextures([{ width: 2, height: 2, data: new Uint8Array(15), colorSpace: 'linear' }])).toThrow(/expected 16/);
  });

  it('slot records round-trip', () => {
    const s = buildSceneTextures([sq(16), sq(64)]);
    const words = new Uint32Array(8);
    packTextureSlot(s.slots[1]!, words, 4);
    expect(unpackTextureSlot(words, 4)).toEqual({ bucket: 1, layer: 0, width: 64, height: 64, mipLevels: 7 });
  });
});

describe('texture lookup twin', () => {
  // 2x2: (0,0) red, (1,0) green, (0,1) blue, (1,1) white. Row 0 = top.
  const quad = make(2, 2, (x, y) => (y === 0 ? (x === 0 ? [255, 0, 0, 255] : [0, 255, 0, 255]) : x === 0 ? [0, 0, 255, 255] : [255, 255, 255, 255]));
  const set = (wrap: 'repeat' | 'clamp' | 'mirror' = 'repeat', colorSpace: 'srgb' | 'linear' = 'linear') => buildSceneTextures([{ ...quad, wrap, colorSpace }], { mips: true });

  it('hits texel centres exactly: uv (0, 0) is the top-left texel (glTF), v grows downward', () => {
    const s = set();
    expect(sampleSceneTexture(s, 0, 0.25, 0.25)).toEqual([1, 0, 0, 1]);
    expect(sampleSceneTexture(s, 0, 0.75, 0.25)).toEqual([0, 1, 0, 1]);
    expect(sampleSceneTexture(s, 0, 0.25, 0.75)).toEqual([0, 0, 1, 1]);
    expect(sampleSceneTexture(s, 0, 0.75, 0.75)).toEqual([1, 1, 1, 1]);
  });

  it('interpolates bilinearly between centres', () => {
    const s = set();
    const mid = sampleSceneTexture(s, 0, 0.5, 0.25); // between red and green
    expect(mid[0]).toBeCloseTo(0.5, 6);
    expect(mid[1]).toBeCloseTo(0.5, 6);
    const centre = sampleSceneTexture(s, 0, 0.5, 0.5);
    expect(centre[0]).toBeCloseTo(0.5, 6);
    expect(centre[2]).toBeCloseTo(0.5, 6);
  });

  it('wrap modes differ at the edge: repeat blends with the opposite edge, clamp with itself, mirror with the nearer neighbour', () => {
    const at = (wrap: 'repeat' | 'clamp' | 'mirror') => sampleSceneTexture(set(wrap), 0, 0, 0.25); // x on the left edge, between texel -1 and 0
    expect(at('clamp')).toEqual([1, 0, 0, 1]);
    expect(at('repeat')[0]).toBeCloseTo(0.5, 6); // red and green
    expect(at('repeat')[1]).toBeCloseTo(0.5, 6);
    expect(at('mirror')).toEqual([1, 0, 0, 1]); // texel -1 mirrors to texel 0
    expect(wrapIndex(-1, 4, 'mirror')).toBe(0);
    expect(wrapIndex(4, 4, 'mirror')).toBe(3);
    expect(wrapIndex(8, 4, 'mirror')).toBe(0);
    expect(wrapIndex(-1, 4, 'repeat')).toBe(3);
    expect(wrapIndex(9, 4, 'clamp')).toBe(3);
  });

  it('repeat is periodic in uv', () => {
    const s = set('repeat');
    for (const [u, v] of [[0.1, 0.2], [0.9, 0.4], [0.33, 0.77]] as const) {
      const a = sampleSceneTexture(s, 0, u, v);
      const b = sampleSceneTexture(s, 0, u + 3, v - 2);
      for (let k = 0; k < 4; k++) expect(b[k]).toBeCloseTo(a[k] as number, 5);
    }
  });

  it('sRGB textures filter in linear light: the midpoint of black and white is 0.5, not 0.214', () => {
    const bw = make(2, 1, (x) => (x === 0 ? [0, 0, 0, 255] : [255, 255, 255, 255]), { colorSpace: 'srgb' });
    const s = buildSceneTextures([bw]);
    expect(sampleSceneTexture(s, 0, 0.5, 0.5)[0]).toBeCloseTo(0.5, 6);
    expect(sampleSceneTexture(s, 0, 0.75, 0.5)[0]).toBeCloseTo(1, 6);
    // Alpha is never decoded.
    const a = make(1, 1, () => [255, 255, 255, 128], { colorSpace: 'srgb' });
    expect(sampleSceneTexture(buildSceneTextures([a]), 0, 0.5, 0.5)[3]).toBeCloseTo(128 / 255, 7);
  });

  it('trilinear: lod blends levels and clamps to the chain', () => {
    const checker = make(8, 8, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255, 255] : [0, 0, 0, 255]));
    const s = buildSceneTextures([checker]);
    expect(sampleSceneTexture(s, 0, 0.5 / 8, 0.5 / 8)[0]).toBe(1); // level 0 texel centre
    expect(sampleSceneTexture(s, 0, 0.5 / 8, 0.5 / 8, 1)[0]).toBeCloseTo(0.5, 2); // level 1 is the mean of the 2x2 box
    expect(sampleSceneTexture(s, 0, 0.1, 0.1, 99)[0]).toBeCloseTo(sampleSceneTexture(s, 0, 0.1, 0.1, 3)[0], 9);
    expect(sampleSceneTexture(s, 0, 0.5 / 8, 0.5 / 8, -4)[0]).toBe(1);
    const half = sampleSceneTexture(s, 0, 0.5 / 8, 0.5 / 8, 0.5)[0];
    expect(half).toBeGreaterThan(0.5);
    expect(half).toBeLessThan(1);
  });

  it('throws for a slot out of range', () => {
    expect(() => sampleSceneTexture(set(), 3, 0, 0)).toThrow(/slot 3/);
  });
});

describe('level of detail', () => {
  it('lod is log2 of the footprint in stored texels, never negative', () => {
    expect(textureLod(256, 256, 4 / 256)).toBeCloseTo(2, 9);
    expect(textureLod(1024, 256, 1 / 512)).toBeCloseTo(0, 9); // sqrt(1024*256) = 512 texels per uv unit
    expect(textureLod(256, 256, 0)).toBe(0);
    expect(textureLod(256, 256, 0.1 / 256)).toBe(0);
  });

  it('cone footprint: width times uv density over the cosine, floored at 1e-3', () => {
    expect(coneUvFootprint(0.02, 1, 0.5)).toBeCloseTo(0.01, 12);
    expect(coneUvFootprint(0.02, 0.5, 0.5)).toBeCloseTo(0.02, 12);
    expect(coneUvFootprint(0.02, -0.5, 0.5)).toBeCloseTo(0.02, 12);
    expect(coneUvFootprint(0.02, 0, 0.5)).toBeCloseTo(0.02 * 0.5 * 1000, 9);
  });

  it('a plane viewed straight on: one texel per pixel at the distance where the cone is one texel wide', () => {
    // A 4 m quad with uv 0..1 (0.25 uv per m), 1024 texels across: 256 texels per metre. A cone 1/256 m wide covers 1 texel.
    const frame = triangleTextureFrame([0, 0, 0], [4, 0, 0], [0, 0, -4], [0, 0], [1, 0], [0, 1]);
    expect(frame.uvDensity).toBeCloseTo(0.25, 12);
    expect(textureLod(1024, 1024, coneUvFootprint(1 / 256, 1, frame.uvDensity))).toBeCloseTo(0, 9);
    expect(textureLod(1024, 1024, coneUvFootprint(8 / 256, 1, frame.uvDensity))).toBeCloseTo(3, 9);
  });
});

describe('texture frame and normal maps', () => {
  it('a ground quad with u along +x and v along +z (glTF v down): tangent +x, bitangent up-in-image = -z, handedness checked against the normal', () => {
    // p0 = origin, p1 = +x, p2 = +z; uv0 = (0,0), uv1 = (1,0), uv2 = (0,1). Winding p0,p1,p2 gives normal x cross z = -y.
    const f = triangleTextureFrame([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0], [1, 0], [0, 1]);
    expect(f.tangent).toEqual([1, 0, 0]);
    expect(f.uvDensity).toBeCloseTo(1, 12);
    // n = (0, -1, 0); B_up = -dp/dv = (0, 0, -1); cross(n, T) = (0,-1,0) x (1,0,0) = (0, 0, 1): opposite, so handedness -1.
    expect(f.handedness).toBe(-1);
  });

  it('uvs scaled up: density counts uv units per metre and the tangent stays dp/du', () => {
    const f = triangleTextureFrame([0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0], [8, 0], [0, 8]);
    expect(f.uvDensity).toBeCloseTo(Math.sqrt(32 / 3), 9);
    expect(f.tangent).toEqual([1, 0, 0]);
  });

  it('degenerate uvs give density 0 (lod 0) and a default tangent', () => {
    const f = triangleTextureFrame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]);
    expect(f.uvDensity).toBe(0);
    expect(f.tangent).toEqual([1, 0, 0]);
  });

  it('a flat texel (0.5, 0.5, 1) leaves the normal alone; a +x tilt bends it toward the tangent; +y toward the image-up bitangent', () => {
    const n = [0, 0, 1];
    const t = [1, 0, 0];
    const E = 0.5 + 0.5 * Math.SQRT1_2; // texel channel that encodes sin 45 degrees = cos 45 degrees
    expect(applyNormalMap(n, t, 1, [0.5, 0.5, 1])).toEqual([0, 0, 1]);
    const tilted = applyNormalMap(n, t, 1, [E, 0.5, E]);
    expect(tilted[0]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(tilted[2]).toBeCloseTo(Math.SQRT1_2, 6);
    const up = applyNormalMap(n, t, 1, [0.5, E, E]); // bitangent = cross(n, t) = +y
    expect(up[1]).toBeCloseTo(Math.SQRT1_2, 6);
    const mirrored = applyNormalMap(n, t, -1, [0.5, E, E]);
    expect(mirrored[1]).toBeCloseTo(-Math.SQRT1_2, 6);
  });

  it('orthogonalises the tangent against the normal and returns unit vectors', () => {
    const n = [0, 1, 0];
    const out = applyNormalMap(n, [1, 0.7, 0], 1, [0.9, 0.2, 0.8]);
    expect(Math.hypot(out[0], out[1], out[2])).toBeCloseTo(1, 9);
    // A tangent along the normal has no frame: the shading normal is unchanged order of magnitude (no NaN).
    expect(Number.isFinite(applyNormalMap(n, [0, 1, 0], 1, [0.9, 0.2, 0.8])[0])).toBe(true);
  });
});
