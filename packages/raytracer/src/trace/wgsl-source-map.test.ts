import { describe, expect, it } from 'vitest';
import { rtMinifyWgsl, rtTransformWgsl } from './modules.ts';
import {
  rtComposeLineMaps,
  rtFormatCompilationMessages,
  rtFormatSourceContext,
  rtMapCompilationInfo,
  rtMapCompilationMessage,
  rtRenameWgslNames,
  rtWgslIdentityMap,
  rtWgslSourceMapV3,
} from './wgsl-source-map.ts';

const src = [
  'enable f16;', // 1
  'struct Used { a: f32, }', // 2
  'struct Dead { b: f32, }', // 3
  'const KEEP: f32 = 1.0;', // 4
  'const DEAD: f32 = 2.0;', // 5
  'fn helperUsed(x: f32) -> f32 {', // 6
  '  let longLocalName = x * KEEP;', // 7
  '  return longLocalName;', // 8
  '}', // 9
  'fn helperDead() -> f32 { return DEAD; }', // 10
  'fn onlyFromDead() -> f32 { return helperDead(); }', // 11
  '@group(0) @binding(0) var<storage, read_write> outBuf: array<f32>;', // 12
  '@compute @workgroup_size(1)', // 13
  'fn mainEntry() {', // 14
  '  var u: Used;', // 15
  '  u.a = helperUsed(1.0);', // 16
  '  outBuf[0] = u.a;', // 17
  '}', // 18
].join('\n');

const msg = (lineNum: number, message: string, type: 'error' | 'warning' = 'error') => ({ type, message, lineNum, linePos: 3 });

describe('rtTransformWgsl source map', () => {
  const built = rtTransformWgsl(src, { keep: ['mainEntry'] });
  const outLines = built.code.split('\n');

  it('prune shifts lines and the map undoes it', () => {
    expect(built.dropped.sort()).toEqual(['DEAD', 'Dead', 'helperDead', 'onlyFromDead']);
    expect(outLines.length).toBeLessThan(18);
    const at = outLines.findIndex((l) => l.startsWith('fn mainEntry'));
    expect(at).toBeGreaterThanOrEqual(0);
    // `fn mainEntry` is source line 14; the lone attribute line travels with it (13).
    expect(built.map.lines[at]).toBe(13);
    expect(built.map.lines[at - 1]).toBe(12);
    expect(built.map.sources?.[13]).toBe('fn mainEntry() {');
    const m = rtMapCompilationMessage(msg(at + 1, 'bad'), built.map);
    expect(m.lineNum).toBe(14);
    expect(m.sourceLine).toBe('fn mainEntry() {');
    expect(m.outputLineNum).toBe(at + 1);
  });

  it('the map is strictly increasing and every output line reads as its source line after minify + renames', () => {
    const minSrc = rtMinifyWgsl(src).split('\n');
    for (let i = 0; i < outLines.length; i++) {
      const s = built.map.lines[i] as number;
      expect(s).toBeGreaterThanOrEqual(0);
      if (i > 0) expect(s).toBeGreaterThan(built.map.lines[i - 1] as number);
      expect(rtRenameWgslNames(outLines[i] as string, built.map)).toBe(minSrc[s]);
    }
  });

  it('names map short -> original, only for what was shortened; entries, bindings, locals and members stay', () => {
    const n = built.map.names;
    expect(Object.values(n)).toEqual(expect.arrayContaining(['helperUsed', 'Used', 'KEEP']));
    expect(Object.values(n)).not.toContain('mainEntry');
    expect(Object.values(n)).not.toContain('outBuf');
    expect(Object.values(n)).not.toContain('longLocalName');
    expect(built.code).toContain('longLocalName');
  });

  it('enable lines: counted, first, and shift only the template line, not the source line', () => {
    expect(built.map.enableLines).toBe(1);
    expect(outLines[0]).toBe('enable f16;');
    expect(built.map.lines[0]).toBe(0);
    const two = rtTransformWgsl(`enable f16;\nenable subgroups;\n${src.split('\n').slice(1).join('\n')}`, { keep: ['mainEntry'] });
    expect(two.map.enableLines).toBe(2);
    const at = (b: typeof two) => b.code.split('\n').findIndex((l) => l.startsWith('fn mainEntry'));
    const tpl = (b: typeof two) => (b.map.lines[at(b)] as number) - (b.map.enableLines as number);
    expect(tpl(two)).toBe(tpl(built));
  });

  it('options off give the identity map', () => {
    const plain = rtTransformWgsl(src, { prune: false, shorten: false, minify: false });
    expect(plain.code).toBe(src);
    expect([...plain.map.lines]).toEqual([...Array(18).keys()]);
    expect(plain.map.names).toEqual({});
  });

  it('composition: stage maps compose, -1 stays -1', () => {
    expect([...rtComposeLineMaps(Int32Array.from([0, 2, 3, -1]), Int32Array.from([5, 6, 7, 8]))]).toEqual([5, 7, 8, -1]);
    expect([...rtComposeLineMaps(Int32Array.from([9]), Int32Array.from([5]))]).toEqual([-1]);
  });
});

describe('message translation', () => {
  const names = { C5: 'rtSampleEnvironment', C: 'rtSample', C50: 'rtSampleEnvironmentLi', Ab: 'ptBsdf' };
  const map = { lines: Int32Array.from([0, 4, 5]), names, sources: ['l1', 'l2', 'l3', 'l4', 'fn rtSampleEnvironment() {', 'return 1;'] };

  it('rewrites several short names in one message, whole word only, longest first, never after a dot', () => {
    const text = "cannot call 'C5' from C50; C is not Ab or C5x, or s.C5 (A C)";
    expect(rtRenameWgslNames(text, map)).toBe("cannot call 'rtSampleEnvironment' from rtSampleEnvironmentLi; rtSample is not ptBsdf or C5x, or s.C5 (A rtSample)");
  });

  it('names that are substrings of other names do not corrupt each other', () => {
    expect(rtRenameWgslNames('C C5 C50 C500 xC5 C5_', map)).toBe('rtSample rtSampleEnvironment rtSampleEnvironmentLi C500 xC5 C5_');
  });

  it('maps a message and the whole info, keeps the compiled position, flags lines without a source', () => {
    const [a, b, c] = rtMapCompilationInfo({ messages: [msg(2, "no 'C5'"), msg(3, 'warn C', 'warning'), msg(0, 'global')] }, map);
    expect(a).toMatchObject({ lineNum: 5, sourceLine: 'fn rtSampleEnvironment() {', outputLineNum: 2, outputLinePos: 3, message: "no 'rtSampleEnvironment'" });
    expect(b).toMatchObject({ lineNum: 6, type: 'warning', message: 'warn rtSample' });
    expect(c).toMatchObject({ lineNum: 0, outputLineNum: 0 });
    expect(rtMapCompilationMessage(msg(99, 'x'), map).lineNum).toBe(0);
    expect(rtMapCompilationMessage(msg(1, 'x'), { ...map, lines: Int32Array.from([-1]) }).lineNum).toBe(0);
  });

  it('formats source context and a full error text', () => {
    expect(rtFormatSourceContext(map, 5, 1)).toBe('  4 | l4\n> 5 | fn rtSampleEnvironment() {\n  6 | return 1;');
    expect(rtFormatSourceContext({ ...map, sources: undefined }, 5)).toBe('');
    const text = rtFormatCompilationMessages(rtMapCompilationInfo({ messages: [msg(2, "bad 'C5'")] }, map), map, 'kernel');
    expect(text).toContain('WGSL (kernel) failed to compile');
    expect(text).toContain('error at source line 5 (compiled line 2)');
    expect(text).toContain("bad 'rtSampleEnvironment'");
    expect(text).toContain('> 5 | fn rtSampleEnvironment() {');
  });

  it('identity map has the code as sources and counts enable lines', () => {
    const m = rtWgslIdentityMap('enable f16;\nfn a() {}');
    expect(m.enableLines).toBe(1);
    expect(m.sources).toEqual(['enable f16;', 'fn a() {}']);
  });
});

describe('Source Map v3', () => {
  it('encodes line deltas as VLQ', () => {
    const v3 = rtWgslSourceMapV3({ lines: Int32Array.from([0, 1, 10, -1, 3]), names: { A: 'foo' } });
    expect(v3.version).toBe(3);
    expect(v3.names).toEqual(['foo']);
    // lines: +0, +1, +9 (S = 18 -> 'S'), none, -7 (P = 15)
    expect(v3.mappings).toBe('AAAA;AACA;AASA;;AAPA');
  });
});
