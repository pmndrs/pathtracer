import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE, MLP_SHAPES } from './config.ts';
import { inferTiledWgsl, inferTiledWorkgroupBytes } from './wgsl-forward.ts';

const HOOKS = 'fn nnInferLoad(r: u32) -> array<f32, NN_IN> { var x: array<f32, NN_IN>; return x; }\nfn nnInferStore(r: u32, y: vec3f) {}';

describe('tiled inference (N1.4)', () => {
  it('keeps the inference entry point and bindings, stores through the hook, no activation stores', () => {
    const w = inferTiledWgsl({ shape: DEFAULT_SHAPE, precision: 'f16' }, HOOKS, { records: 64 });
    expect(w).toContain('fn nnInfer(');
    expect(w).toContain('@group(0) @binding(0) var<storage, read> nnWeights: array<T4>;');
    expect(w).toContain('@group(0) @binding(1) var<storage, read> nnState: NnState;');
    expect(w).not.toContain('@binding(2)');
    expect(w).not.toContain('nnActs');
    expect(w).toContain('nnInferStore(r, acc0.xyz)');
    expect(w.length).toBeLessThan(50_000);
  });
  it('fits 32 KiB of workgroup memory at 64 records in f16 and f32', () => {
    expect(inferTiledWorkgroupBytes({ shape: DEFAULT_SHAPE, precision: 'f16' }, { records: 64 })).toBe(64 * 64 * 4 + 4096 * 2);
    expect(inferTiledWorkgroupBytes({ shape: DEFAULT_SHAPE, precision: 'f32' }, { records: 64 })).toBe(32768);
  });
  it('loads 64 inputs and stages only real RGB rows for the compact network', () => {
    const code = inferTiledWgsl({ shape: MLP_SHAPES.compact, precision: 'f16' }, HOOKS, { records: 64 });
    expect(code).toContain('// layer 0: 32 × 64');
    const [, output] = code.split('// layer 3: 3 × 32');
    expect(output).toContain('let w3 = vec4f(0.0)');
    expect(output).not.toContain('nnLayer[(4u * tx + 3u)');
  });
});
