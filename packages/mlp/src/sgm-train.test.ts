import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE } from './config.ts';
import { BUFFER_HOOKS_TRAIN } from './wgsl.ts';
import { sgmTrainWorkgroupBytes, trainSgmWgsl, validateSgmTrain } from './wgsl-sgm-train.ts';
import type { SgmConfig, SgmOptions } from './wgsl-sgm.ts';

const APPLE: SgmConfig = { componentType: 'f16', resultComponentType: 'f16', M: 8, N: 8, K: 8, minSubgroupSize: 32, maxSubgroupSize: 32 };
const F32: SgmConfig = { ...APPLE, componentType: 'f32', resultComponentType: 'f32' };
const opts = (o: Partial<SgmOptions> = {}): SgmOptions => ({ shape: DEFAULT_SHAPE, config: APPLE, records: 64, subgroups: 1, ...o });

describe('sgm train generator', () => {
  it('native: forward uses the transposed weight view, backward the row-major one', () => {
    const code = trainSgmWgsl(opts(), BUFFER_HOOKS_TRAIN);
    expect(code.startsWith('enable f16;\nenable chromium_experimental_subgroup_matrix;')).toBe(true);
    expect(code).toMatch(/subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>\(&nnSgmW, [^;]*, true, 64u\)/);
    expect(code).toMatch(/subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>\(&nnSgmW, [^;]*, false, 64u\)/);
    expect(code).toContain('fn nnForwardBackward(');
    expect(code).toContain('@workgroup_size(32)');
    expect(code).not.toContain('subgroup_id');
    expect(code.length).toBeLessThan(50_000);
  });

  it('emits one forward product and one ReLU pass per hidden layer, one backward product per layer 1..hidden', () => {
    const code = trainSgmWgsl(opts(), BUFFER_HOOKS_TRAIN);
    expect(code.match(/\/\/ forward layer \d/g)).toHaveLength(6);
    expect(code.match(/\/\/ a_\d = relu/g)).toHaveLength(5);
    expect(code.match(/\/\/ backward layer \d/g)).toHaveLength(5);
    // The output layer's backward K is the padded output width (8), layer 5 forward writes padded rows.
    expect(code).toContain('(K = 8 layer outputs, N = 64 inputs)');
    expect(code).toContain('(padded to 8)');
  });

  it('4 subgroups split row blocks by subgroup_id', () => {
    const code = trainSgmWgsl(opts({ subgroups: 4 }), BUFFER_HOOKS_TRAIN);
    expect(code).toContain('@builtin(subgroup_id) sgid: u32');
    expect(code).toContain('@workgroup_size(128)');
  });

  it('emulated form: plain WGSL, no matrix built-ins, both multiply stand-ins, any tile size', () => {
    for (const t of [8, 16]) {
      const code = trainSgmWgsl(opts({ config: { ...F32, M: t, N: t, K: t }, emulate: true, subgroups: 2, records: 64 }), BUFFER_HOOKS_TRAIN);
      expect(code).not.toMatch(/subgroup_matrix|subgroupMatrix|^enable/m);
      expect(code).toContain('emuMma(');
      expect(code).toContain('emuMmaN(');
      expect(code).toContain('@workgroup_size(64)');
    }
  });

  it('keeps the tiled arm bindings and entry point', () => {
    const code = trainSgmWgsl(opts(), BUFFER_HOOKS_TRAIN);
    for (const b of ['@binding(0) var<storage, read> nnSgmW', '@binding(1) var<storage, read> nnState', '@binding(2) var<uniform> params', '@binding(3) var<storage, read_write> nnActs', '@binding(4) var<storage, read_write> nnDeltas', '@binding(5) var<storage, read_write> nnLoss']) expect(code).toContain(b);
  });

  it('sizes workgroup memory and validates', () => {
    expect(sgmTrainWorkgroupBytes(opts())).toBe(64 * 64 * 4 + 64 * 16);
    expect(sgmTrainWorkgroupBytes(opts({ config: F32 }))).toBe(64 * 64 * 8 + 64 * 16);
    expect(() => validateSgmTrain(opts({ records: 60 }))).toThrow();
    expect(() => validateSgmTrain(opts({ config: { ...APPLE, K: 16, N: 8 } }))).toThrow(/multiple of K/); // padded out 8 is not a multiple of 16
  });
});
