import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE, layerOffset } from './config.ts';
import { DEFAULT_GRADIENT } from './wgsl-gradient.ts';
import { gradientSgmWgsl, sgmGradientSubgroups, validateSgmGradient } from './wgsl-gradient-sgm.ts';
import type { SgmConfig } from './wgsl-sgm.ts';

const NV: SgmConfig = { componentType: 'f16', resultComponentType: 'f32', M: 16, N: 16, K: 16, minSubgroupSize: 32, maxSubgroupSize: 32 };
const o = { shape: DEFAULT_SHAPE, precision: 'f16' as const };

describe("sgm weight gradient (N2r-d)", () => {
  it('δᵀ as a column-major left view, a as a row-major right view, f32 sums, one subgroup per output row block', () => {
    const code = gradientSgmWgsl(o, DEFAULT_GRADIENT, { config: NV });
    expect(sgmGradientSubgroups(o, { config: NV })).toBe(4);
    expect(code).toContain('@workgroup_size(128)');
    expect(code).toContain('enable subgroups;');
    expect(code).toContain('let d = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&nnDeltas, row + sg * 16u, true, 64u);');
    expect(code).toContain('let a3 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&nnActs, row + 48u, false, 64u);');
    expect(code).toContain('var z3 = subgroup_matrix_result<f32, 16, 16>();');
    // Constant-bounded record loop (no Tint loop counter), whole K-record steps up to roundup(count, K).
    expect(code).toContain('for (var s = 0u; s < 0xffffffffu; s++) {');
    expect(code).toContain('let padded = (count + 15u) / 16u * 16u;');
    // The output layer's 3 real rows are copied out of the padded tile.
    expect(code).toContain(`nnPartials[split * NN_WEIGHTS + ${layerOffset(DEFAULT_SHAPE, 5)}u + i] = nnOutTile[i];`);
    expect(code).toContain('for (var j = 0u; j < 2u; j++) {'); // 192 copies over 128 lanes
    expect(code).toContain('fn nnGradientSgm(');
    expect(code.length).toBeLessThan(50_000);
  });

  it('validates the config and shape', () => {
    expect(() => validateSgmGradient(o, DEFAULT_GRADIENT, { config: { ...NV, resultComponentType: 'f16' } })).toThrow(/f32 results/);
    expect(() => validateSgmGradient({ ...o, precision: 'f32' }, DEFAULT_GRADIENT, { config: NV })).toThrow(/precision/);
    expect(() => validateSgmGradient({ ...o, shape: { ...DEFAULT_SHAPE, inputs: 32 } }, DEFAULT_GRADIENT, { config: NV })).toThrow(/inputs must equal width/);
  });
});
