import { describe, expect, it } from 'vitest';
import { DEFAULT_SHAPE, MLP_SHAPES } from './config.ts';
import { DEFAULT_FORWARD, forwardWorkgroupBytes, trainTiledWgsl, validateForwardOptions } from './wgsl-forward.ts';
import { BUFFER_HOOKS_TRAIN, trainWgsl } from './wgsl.ts';

const f32 = { shape: DEFAULT_SHAPE, precision: 'f32' } as const;
const f16 = { shape: DEFAULT_SHAPE, precision: 'f16' } as const;

/** `@group(0) @binding(n) var<...> name` lines of a kernel, so the two arms can be compared. */
function bindings(code: string): string[] {
  return code.split('\n').filter((l) => l.startsWith('@group(0)'));
}

describe('tiled forward-backward generator', () => {
  it('keeps the entry point and every group-0 binding of the per-record kernel', () => {
    for (const o of [f32, f16]) {
      const tiled = trainTiledWgsl(o, BUFFER_HOOKS_TRAIN);
      expect(tiled).toContain('fn nnForwardBackward(');
      expect(tiled).toContain('@workgroup_size(256)');
      expect(bindings(tiled)).toEqual(bindings(trainWgsl(o, BUFFER_HOOKS_TRAIN)));
    }
    expect(trainTiledWgsl(f16, BUFFER_HOOKS_TRAIN)).toMatch(/^enable f16;/);
  });

  it('stays under the ~50 KB kernel size trap for 16..64 records (128 unrolls past it)', () => {
    for (const records of [16, 32, 64]) {
      for (const o of [f32, f16]) expect(trainTiledWgsl(o, BUFFER_HOOKS_TRAIN, { records }).length).toBeLessThan(50_000);
    }
  });

  it('writes every layer once to each of activations and deltas, and calls the hook once per record slot', () => {
    const code = trainTiledWgsl(f32, BUFFER_HOOKS_TRAIN, { records: 32 });
    expect(code.match(/nnTrainLoad\(/g)?.length).toBe(2); // the hook definition and the one call
    for (const l of [1, 2, 3, 4, 5]) expect(code).toContain(`nnActs[(${l}u * params.capacity`);
    for (const l of [0, 1, 2, 3, 4, 5]) expect(code).toContain(`nnDeltas[(${l}u * params.capacity`);
    expect(code).not.toMatch(/ptr<storage/);
  });

  it('sizes the register block and workgroup memory from the records', () => {
    expect(trainTiledWgsl(f32, BUFFER_HOOKS_TRAIN, { records: 64 })).toContain('var acc3 = vec4f(0.0)');
    expect(trainTiledWgsl(f32, BUFFER_HOOKS_TRAIN, { records: 32 })).not.toContain('var acc2 = vec4f(0.0)');
    expect(trainTiledWgsl(f32, BUFFER_HOOKS_TRAIN, { records: 64 })).toContain('var<workgroup> nnTile: array<vec4f, 1024>');
    // 64 records: tile 16 KiB + staged layer 16 KiB (f32) or 8 KiB (f16) + targets 1 KiB
    expect(forwardWorkgroupBytes(f32, { records: 64 })).toBe(16384 + 16384 + 1024);
    expect(forwardWorkgroupBytes(f16, { records: 64 })).toBe(16384 + 8192 + 1024);
    expect(forwardWorkgroupBytes(f32, { records: 32 })).toBeLessThanOrEqual(32768);
  });

  it('stages only real RGB rows with 64 inputs and 32-wide hidden layers', () => {
    const code = trainTiledWgsl({ shape: MLP_SHAPES.compact, precision: 'f32' }, BUFFER_HOOKS_TRAIN, { records: 16 });
    expect(code).toContain('// forward layer 0: 32 × 64');
    const [, finalLayers] = code.split('// forward layer 3: 3 × 32');
    const [outputForward, backwardLayers] = finalLayers!.split('// backward layer 3');
    const [outputBackward, hiddenBackward] = backwardLayers!.split('// backward layer 2');
    expect(outputForward).toContain('let w3 = vec4f(0.0)');
    expect(outputBackward).not.toContain('4u * o4 + 3u');
    expect(hiddenBackward).toContain('4u * o4 + 3u');
  });

  it('rejects options and shapes the tile cannot cover', () => {
    expect(() => validateForwardOptions(f32, { records: 0 })).toThrow(/records/);
    expect(() => validateForwardOptions(f32, { records: 24 })).toThrow(/records/);
    expect(() => validateForwardOptions(f32, { records: 512 })).toThrow(/records/);
    expect(() => validateForwardOptions({ shape: { ...DEFAULT_SHAPE, width: 128 }, precision: 'f32' }, DEFAULT_FORWARD)).toThrow(/tile/);
  });
});
