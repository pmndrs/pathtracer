import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE, weightCount } from './config.ts';
import { BUFFER_HOOKS_INFER } from './wgsl.ts';
import { inferSgmWgsl, packSgmWgsl, pickSubgroupMatrixConfig, sgmLayerOut, sgmWeightCount, sgmWorkgroupBytes, validateSgm, type SgmConfig, type SgmOptions } from './wgsl-sgm.ts';

const APPLE: SgmConfig = { componentType: 'f16', resultComponentType: 'f16', M: 8, N: 8, K: 8, minSubgroupSize: 32, maxSubgroupSize: 32 };
const opts = (o: Partial<SgmOptions> = {}): SgmOptions => ({ shape: DEFAULT_SHAPE, config: APPLE, records: 64, subgroups: 1, ...o });

describe('sgm layout', () => {
  it('pads the 3-wide output layer to N rows', () => {
    expect(sgmLayerOut(DEFAULT_SHAPE, { N: 8 }, 5)).toBe(8);
    expect(sgmWeightCount(DEFAULT_SHAPE, { N: 8 })).toBe(weightCount(DEFAULT_SHAPE) - 3 * 64 + 8 * 64);
    expect(sgmWeightCount(DEFAULT_SHAPE, { N: 16 })).toBe(5 * 64 * 64 + 16 * 64);
  });

  it('sizes workgroup memory for activations (T) plus results (TR)', () => {
    expect(sgmWorkgroupBytes(opts())).toBe(64 * 64 * (2 + 2));
    expect(sgmWorkgroupBytes(opts({ config: { ...APPLE, componentType: 'f32', resultComponentType: 'f32' } }))).toBe(64 * 64 * 8);
  });

  it('rejects record counts that do not split into M-row blocks per subgroup', () => {
    expect(() => validateSgm(opts({ records: 60 }))).toThrow();
    expect(() => validateSgm(opts({ records: 64, subgroups: 16 }))).toThrow();
    expect(() => validateSgm(opts({ emulate: true }))).toThrow(/f32 only/);
  });
});

describe('sgm generator', () => {
  it('native form: enable first, transposed weight view, zero-initialised accumulators', () => {
    const code = inferSgmWgsl(opts(), BUFFER_HOOKS_INFER);
    expect(code.startsWith('enable f16;\nenable chromium_experimental_subgroup_matrix;')).toBe(true);
    expect(code).toContain('subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>(&nnSgmW');
    expect(code).toMatch(/subgroupMatrixLoad<subgroup_matrix_right<f16, 8, 8>>\(&nnSgmW, [^;]*, true, 64u\)/);
    expect(code).toContain('subgroup_matrix_result<f16, 8, 8>()');
    expect(code).not.toContain('subgroup_id');
    expect(code.length).toBeLessThan(50_000);
  });

  it('several subgroups split rows by subgroup_id, or by local index for the control case', () => {
    const byId = inferSgmWgsl(opts({ subgroups: 4 }), BUFFER_HOOKS_INFER);
    expect(byId).toContain('enable subgroups;');
    expect(byId).toContain('@builtin(subgroup_id) sgid: u32');
    expect(byId).toContain('@workgroup_size(128)');
    const byIndex = inferSgmWgsl(opts({ subgroups: 4, offsetsFromLocalIndex: true }), BUFFER_HOOKS_INFER);
    expect(byIndex).not.toContain('subgroup_id');
    expect(byIndex).toContain('lid / 32u');
  });

  it('places a diagnostic directive after the enables', () => {
    const code = inferSgmWgsl(opts({ subgroups: 4, offsetsFromLocalIndex: true, uniformityDiagnosticOff: 'subgroup_matrix_uniformity' }), BUFFER_HOOKS_INFER);
    const lines = code.split('\n');
    const diag = lines.findIndex((l) => l.startsWith('diagnostic('));
    const lastEnable = lines.map((l, i) => (l.startsWith('enable ') ? i : -1)).reduce((a, b) => Math.max(a, b), -1);
    expect(diag).toBeGreaterThan(lastEnable);
  });

  it('emulated form uses no matrix built-ins and no enables', () => {
    const code = inferSgmWgsl(opts({ config: { ...APPLE, componentType: 'f32', resultComponentType: 'f32' }, emulate: true, subgroups: 2 }), BUFFER_HOOKS_INFER);
    expect(code).not.toMatch(/subgroup_matrix|subgroupMatrix|^enable/m);
    expect(code).toContain('emuMma(');
    // Lanes share the tile (8 × 8 = 2 elements per lane); no lane holds a whole tile.
    expect(code).toContain('alias EmuResult = array<f32, 2>;');
    expect(code).toContain('nnEmuLane = lid % 32u;');
    expect(code).not.toContain('continue;');
    const t4 = inferSgmWgsl(opts({ config: { ...APPLE, componentType: 'f32', resultComponentType: 'f32', M: 4, N: 4, K: 4 }, emulate: true }), BUFFER_HOOKS_INFER);
    expect(t4).toContain('alias EmuResult = array<f32, 1>;');
    expect(t4).toContain('if (e >= 16u) { continue; }');
  });

  it('pack kernel zero-fills padded output rows', () => {
    const code = packSgmWgsl(opts());
    expect(code).toContain('row < 3u');
  });
});

describe('pickSubgroupMatrixConfig', () => {
  const info = (configs: Array<Partial<SgmConfig>>) => ({ subgroupMatrixConfigs: configs, subgroupMinSize: 32, subgroupMaxSize: 32 }) as unknown as GPUAdapterInfo;

  it('prefers the requested operand type, then f32 results', () => {
    const c = pickSubgroupMatrixConfig(info([
      { componentType: 'f32', resultComponentType: 'f32', M: 8, N: 8, K: 8 },
      { componentType: 'f16', resultComponentType: 'f16', M: 8, N: 8, K: 8 },
      { componentType: 'f16', resultComponentType: 'f32', M: 16, N: 16, K: 16 },
    ]), DEFAULT_SHAPE, true);
    expect(c).toMatchObject({ componentType: 'f16', resultComponentType: 'f32', M: 16 });
    expect(pickSubgroupMatrixConfig(info([{ componentType: 'f16', resultComponentType: 'f16', M: 8, N: 8, K: 8 }]), DEFAULT_SHAPE, false)?.componentType).toBe('f16');
  });

  it('takes the requested accumulator type and the biggest tile among ties (NVIDIA list)', () => {
    const nvidia = info(['f16', 'f32'].flatMap((r) => [[16, 8, 8], [16, 8, 16], [16, 16, 16]].map(([M, N, K]) => ({ componentType: 'f16' as const, resultComponentType: r as 'f16' | 'f32', M, N, K }))));
    expect(pickSubgroupMatrixConfig(nvidia, DEFAULT_SHAPE, true)).toMatchObject({ resultComponentType: 'f32', M: 16, N: 16, K: 16 });
    expect(pickSubgroupMatrixConfig(nvidia, DEFAULT_SHAPE, true, 'f16')).toMatchObject({ resultComponentType: 'f16', M: 16, N: 16, K: 16 });
  });

  it('skips integer configs and tiles that do not divide the network', () => {
    expect(pickSubgroupMatrixConfig(info([{ componentType: 'u8' as never, resultComponentType: 'u32' as never, M: 8, N: 8, K: 32 }, { componentType: 'f16', resultComponentType: 'f16', M: 8, N: 24, K: 8 }]), DEFAULT_SHAPE, true)).toBeNull();
    expect(pickSubgroupMatrixConfig(undefined, DEFAULT_SHAPE, true)).toBeNull();
  });
});
