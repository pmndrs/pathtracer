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

  it("'blocked' (N2r-d): ping-pong f16 buffers, held tiles, in-place ReLU and mask, constant-trip lane loops", () => {
    const NV: SgmConfig = { ...APPLE, M: 16, N: 16, K: 16 };
    const code = trainSgmWgsl(opts({ config: NV, layout: 'blocked' }), BUFFER_HOOKS_TRAIN);
    expect(code).toContain('var<workgroup> nnA0: array<T, 4608>;');
    expect(code).toContain('var<workgroup> nnA1: array<T, 4608>;');
    expect(code).not.toMatch(/\bnnS\b/);
    expect(code).not.toMatch(/for \(var \w+ = lid/);
    // forward: transposed right tiles, all 16 of a hidden layer held; backward: row-major view.
    expect(code).toContain('let w3_3 = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&nnSgmW, ');
    expect(code).toMatch(/subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>\(&nnSgmW, \d+u, false, 64u\)/);
    expect(code.match(/\/\/ forward layer \d/g)).toHaveLength(6);
    expect(code.match(/\/\/ backward layer \d/g)).toHaveLength(5);
    expect(code).toContain('(K = 16 layer outputs, N = 64 inputs)');
    // a_1 lives in nnA1 (parity 1): ReLU in place, 4 halves per item; δ_5 in nnA0 (parity 6).
    expect(code).toContain('let v3 = max(nnA1[at + 3u], T(0.0));');
    expect(code).toContain('nnA0[at + 15u] = T(0.0);');
    // ReLU mask as bits in workgroup memory (default): a_1 row rl at word rl * 2 + col / 32.
    expect(code).toContain('var<workgroup> nnMask: array<atomic<u32>, 640>;');
    expect(code).toContain('atomicOr(&nnMask[(0u + rl) * 2u + col / 32u], m << (col % 32u));');
    expect(code).toContain('let m = atomicLoad(&nnMask[(256u + rl) * 2u + col / 32u]) >> (col % 32u);');
    expect(code).toContain('v3 = select(T(0.0), T(nnA1[at + 3u]), (m & 8u) != 0u);');
    expect(sgmTrainWorkgroupBytes(opts({ config: NV, layout: 'blocked' }))).toBe(64 * 72 * 2 * 2 + 64 * 16 + 640 * 4);
    const global = trainSgmWgsl(opts({ config: NV, layout: 'blocked', trainMask: 'global' }), BUFFER_HOOKS_TRAIN);
    expect(global).not.toContain('nnMask');
    expect(global).toMatch(/v0 = select\(T\(0\.0\), T\(nnA1\[at \+ 0u\]\), nnActs\[g \+ 0u\] > T\(0\.0\)\);/);
    expect(sgmTrainWorkgroupBytes(opts({ config: NV, layout: 'blocked', trainMask: 'global' }))).toBe(64 * 72 * 2 * 2 + 64 * 16);
    expect(code.length).toBeLessThan(50_000);
  });

  it("'blocked' with f32 results keeps nnA + nnS and converts in the passes; no emulated form", () => {
    const NV32: SgmConfig = { ...APPLE, resultComponentType: 'f32', M: 16, N: 16, K: 16 };
    const code = trainSgmWgsl(opts({ config: NV32, layout: 'blocked' }), BUFFER_HOOKS_TRAIN);
    expect(code).toContain('var<workgroup> nnS: array<TR, 4608>;');
    expect(code).toContain('let v0 = T(max(f32(nnS[at + 0u]), 0.0));');
    expect(sgmTrainWorkgroupBytes(opts({ config: NV32, layout: 'blocked' }))).toBe(64 * 72 * 6 + 64 * 16 + 640 * 4);
    expect(() => trainSgmWgsl(opts({ config: { ...F32, M: 16, N: 16, K: 16 }, layout: 'blocked', emulate: true }), BUFFER_HOOKS_TRAIN)).toThrow(/no emulated form/);
  });

  it('sizes workgroup memory and validates', () => {
    expect(sgmTrainWorkgroupBytes(opts())).toBe(64 * 64 * 4 + 64 * 16);
    expect(sgmTrainWorkgroupBytes(opts({ config: F32 }))).toBe(64 * 64 * 8 + 64 * 16);
    expect(() => validateSgmTrain(opts({ records: 60 }))).toThrow();
    expect(() => validateSgmTrain(opts({ config: { ...APPLE, K: 16, N: 8 } }))).toThrow(/multiple of K/); // padded out 8 is not a multiple of 16
  });
});
