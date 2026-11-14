import { buildCaps, rtMapCompilationMessage, rtMinifyWgsl, rtRenameWgslNames } from '@pmndrs/raytracer';
import { describe, expect, it } from 'vitest';
import { ptMegakernelBuild } from './megakernel.ts';

const caps = buildCaps({ features: new Set(['subgroups', 'shader-f16', 'timestamp-query']), limits: {} }, new Set());
const bare = buildCaps({ features: new Set(), limits: {} }, new Set());
const layout = { normals: true, uvs: false, positions: true };
const opts = { format: {}, layout, integrator: {} };

/** Seeded PRNG so a failure reproduces. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('megakernel source map (P2.10b)', () => {
  const full = ptMegakernelBuild(caps, opts);
  const readable = ptMegakernelBuild(caps, { ...opts, prune: false, shortenNames: false, minify: false });
  const out = full.code.split('\n');

  it('the source of the map is exactly the readable kernel', () => {
    expect(full.map.sources?.join('\n')).toBe(readable.code);
    expect(readable.map.names).toEqual({});
  });

  it('shortens and prunes for real (the map has work to do)', () => {
    expect(Object.keys(full.map.names).length).toBeGreaterThan(100);
    expect(out.length).toBeLessThan((full.map.sources as string[]).length);
  });

  it('every output line maps to the readable line it came from, text and all', () => {
    const minSrc = rtMinifyWgsl(readable.code).split('\n');
    let last = -1;
    for (let i = 0; i < out.length; i++) {
      const s = full.map.lines[i] as number;
      expect(s).toBeGreaterThan(last);
      last = s;
      if (rtRenameWgslNames(out[i] as string, full.map) !== minSrc[s]) throw new Error(`line ${i + 1} -> ${s + 1}: ${out[i]} vs ${minSrc[s]}`);
    }
  });

  it('random output lines and random identifiers give back the original line and name', () => {
    const rand = rng(1234);
    const shorts = Object.keys(full.map.names);
    for (let k = 0; k < 300; k++) {
      const short = shorts[Math.floor(rand() * shorts.length)] as string;
      const original = full.map.names[short] as string;
      const re = new RegExp(`(?<![\\w.])${short}(?!\\w)`);
      const line = out.findIndex((l) => re.test(l));
      expect(line).toBeGreaterThanOrEqual(0);
      const m = rtMapCompilationMessage({ type: 'error', message: `unknown '${short}' near ${short}`, lineNum: line + 1, linePos: 1 }, full.map);
      expect(m.message).toBe(`unknown '${original}' near ${original}`);
      expect(m.sourceLine).toContain(original);
      expect(m.lineNum).toBe((full.map.lines[line] as number) + 1);
      expect(readable.code.split('\n')[m.lineNum - 1]).toBe(m.sourceLine);
    }
    for (let k = 0; k < 300; k++) {
      const line = Math.floor(rand() * out.length);
      const m = rtMapCompilationMessage({ type: 'error', message: 'x', lineNum: line + 1, linePos: 1 }, full.map);
      expect(m.sourceLine).toBe((full.map.sources as string[])[full.map.lines[line] as number]);
    }
  });

  it('the entry point keeps its name and its source line', () => {
    const at = out.findIndex((l) => l.startsWith('fn ptMain'));
    const m = rtMapCompilationMessage({ type: 'error', message: 'x', lineNum: at + 1, linePos: 1 }, full.map);
    expect(m.sourceLine).toMatch(/^fn ptMain/);
  });

  it('enable lines: the offset is carried, and the template line does not depend on the device features', () => {
    const b = ptMegakernelBuild(bare, opts);
    expect(full.map.enableLines).toBe(2);
    expect(b.map.enableLines).toBe(0);
    const line = (x: typeof full) => {
      const at = x.code.split('\n').findIndex((l) => l.startsWith('fn ptMain'));
      return (x.map.lines[at] as number) - (x.map.enableLines as number);
    };
    expect(line(full)).toBe(line(b));
    expect(out[0]).toMatch(/^enable /);
    expect(full.map.lines[0]).toBe(0);
  });
});
