import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE } from './config.ts';
import { DEFAULT_GRADIENT, gradientPartialBytes, gradientReduceWgsl, gradientTileWgsl, validateGradientOptions } from './wgsl-gradient.ts';

const f32 = { shape: DEFAULT_SHAPE, precision: 'f32' } as const;
const f16 = { shape: DEFAULT_SHAPE, precision: 'f16' } as const;

describe('tiled weight gradient generator', () => {
  it('emits both kernels, small, with the layer offsets of the shape', () => {
    for (const o of [f32, f16]) {
      const tile = gradientTileWgsl(o, DEFAULT_GRADIENT);
      const reduce = gradientReduceWgsl(o, DEFAULT_GRADIENT);
      expect(tile).toContain('fn nnGradientTile');
      expect(reduce).toContain('fn nnGradientReduce');
      expect(tile.length).toBeLessThan(50_000);
      expect(tile).toContain('array<u32, 6>(0u, 4096u, 8192u, 12288u, 16384u, 20480u)');
      expect(tile).toContain('workgroupUniformLoad');
    }
    expect(gradientTileWgsl(f16, DEFAULT_GRADIENT)).toMatch(/^enable f16;/);
  });

  it('sizes workgroup staging from the chunk and the partial buffer from the splits', () => {
    const tile = gradientTileWgsl(f32, { splits: 32, chunk: 16 });
    expect(tile).toContain('nnSDelta: array<vec4f, 256>');
    expect(tile).toContain('const NN_SPLITS = 32u');
    expect(gradientPartialBytes(f32, { splits: 32, chunk: 16 })).toBe(32 * 20672 * 4);
  });

  it('keeps staging within the default 16 KiB-plus workgroup budget at the default chunk', () => {
    const bytes = 2 * DEFAULT_GRADIENT.chunk * 16 * 16;
    expect(bytes).toBeLessThanOrEqual(32768);
  });

  it('rejects options and shapes the tile cannot cover', () => {
    expect(() => validateGradientOptions(f32, { splits: 0, chunk: 32 })).toThrow(/splits/);
    expect(() => validateGradientOptions(f32, { splits: 8, chunk: 24 })).toThrow(/chunk/);
    expect(() => validateGradientOptions({ shape: { ...DEFAULT_SHAPE, width: 128 }, precision: 'f32' }, DEFAULT_GRADIENT)).toThrow(/tile/);
  });
});
