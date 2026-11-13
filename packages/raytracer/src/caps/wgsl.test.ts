import { describe, expect, it } from 'vitest';
import { buildCaps, buildWgsl, capsDefines, preprocess, wgslEnableLines } from './index.ts';

const caps = (features: string[]) => buildCaps({ features: new Set(features), limits: {} }, new Set());

describe('preprocess', () => {
  const src = ['a', '#if X', 'x', '#elif Y', 'y', '#else', 'z', '#endif', 'b'].join('\n');
  it('picks the right branch and keeps line count', () => {
    expect(preprocess(src, { X: true }).split('\n')).toEqual(['a', '', 'x', '', '', '', '', '', 'b']);
    expect(preprocess(src, { Y: true }).split('\n')).toEqual(['a', '', '', '', 'y', '', '', '', 'b']);
    expect(preprocess(src, {}).split('\n')).toEqual(['a', '', '', '', '', '', 'z', '', 'b']);
    expect(preprocess(src, { X: true, Y: true }).split('\n')[2]).toBe('x');
    expect(preprocess(src, { X: true, Y: true }).split('\n')[4]).toBe('');
  });
  it('supports negation, numbers and nesting', () => {
    const s = '#if !A\n1\n#if B\n2\n#else\n3\n#endif\n#endif\n4';
    expect(preprocess(s, { B: 1 }).split('\n').filter(Boolean)).toEqual(['1', '2', '4']);
    expect(preprocess(s, { B: 0 }).split('\n').filter(Boolean)).toEqual(['1', '3', '4']);
    expect(preprocess(s, { A: true, B: 1 }).split('\n').filter(Boolean)).toEqual(['4']);
  });
  it('does not evaluate branches of an inactive parent', () => {
    const s = '#if A\n#if B\nb\n#else\nnotb\n#endif\n#endif';
    expect(preprocess(s, { B: true }).trim()).toBe('');
    expect(preprocess(s, {}).trim()).toBe('');
  });
  it('rejects malformed input', () => {
    expect(() => preprocess('#if X\nx', {})).toThrow(/missing #endif/);
    expect(() => preprocess('#endif', {})).toThrow(/without #if/);
    expect(() => preprocess('#if X\n#else\n#else\n#endif', {})).toThrow(/duplicate #else/);
    expect(() => preprocess('#if X\n#else\n#elif Y\n#endif', {})).toThrow(/after #else/);
    expect(() => preprocess('#if X && Y\n#endif', {})).toThrow(/bad condition/);
  });
  it('leaves WGSL attributes and other # text alone', () => {
    expect(preprocess('@group(0) var<storage> a: u32;\n// #if no', {})).toContain('@group(0)');
  });
});

describe('caps helpers', () => {
  it('derives defines', () => {
    const d = capsDefines(caps(['subgroups', 'shader-f16', 'sampling-resource-table']));
    expect(d).toMatchObject({ HAS_SUBGROUPS: true, HAS_F16: true, BINDLESS: true, BINDLESS_HETEROGENEOUS: false, HAS_TIMESTAMP: false });
  });
  it('emits enable lines only for present features', () => {
    expect(wgslEnableLines(caps([]))).toBe('');
    expect(wgslEnableLines(caps(['shader-f16', 'subgroups']))).toBe('enable f16;\nenable subgroups;');
    expect(wgslEnableLines(caps(['chromium-experimental-subgroup-matrix']))).toBe('enable chromium_experimental_subgroup_matrix;');
    expect(wgslEnableLines(caps(['subgroup-matrix']))).toBe('enable subgroup_matrix;');
  });
  it('buildWgsl combines both', () => {
    const out = buildWgsl('#if HAS_SUBGROUPS\nfast\n#else\nslow\n#endif', caps(['subgroups']));
    expect(out.startsWith('enable subgroups;\n')).toBe(true);
    expect(out).toContain('fast');
    expect(out).not.toContain('slow');
  });
});
