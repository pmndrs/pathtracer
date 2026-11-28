// P-W1: Tint gives every loop it cannot prove finite a 64-bit iteration counter, and on Chrome D3D12 (DXC SM 6.6+, RTX 3060) those
// counters on the walk's descent and pop loops made the megakernel 3.7x slower. bvh 0.1.1 (bvh#3) writes those loops with a literal
// bound itself (we used to rewrite them, `rtBoundWalkLoops`, removed with V294). This test fails if a bvh release brings an
// unbounded loop back into any walk variant the raytracer can compile.
import { describe, expect, it } from 'vitest';
import { traceModuleWgsl, type AnyHitOrder } from '@pmndrs/bvh';
import { stripWgslComments } from './modules.ts';

const variants: Array<Parameters<typeof traceModuleWgsl>[0]> = [];
for (const rayMasks of [true, false])
  for (const triangleLeaves of ['records', 'indexed'] as const)
    for (const nodeFormat of ['f32', 'quantized'] as const)
      for (const anyHitOrder of [undefined, 'far', 'near', 'larger'] as (AnyHitOrder | undefined)[])
        for (const positionLayout of triangleLeaves === 'indexed' ? (['vec4', 'packed'] as const) : ([undefined] as const))
          variants.push({ traceSlot: { group: 0, firstBinding: 0 }, rayMasks, triangleLeaves, nodeFormat, ...(anyHitOrder ? { anyHitOrder } : {}), ...(positionLayout ? { positionLayout } : {}) });
for (const nodeFormat of ['bvh4', 'cwbvh8'] as const)
  for (const wideDecode of ['exact', 'folded'] as const)
    for (const wideWalk of nodeFormat === 'bvh4' ? (['nested', 'single'] as const) : ([undefined] as const))
      for (const rayMasks of [true, false]) variants.push({ traceSlot: { group: 0, firstBinding: 0 }, rayMasks, nodeFormat, wideDecode, ...(wideWalk ? { wideWalk } : {}) });

/** The one loop shape Tint leaves without a counter: index declared in the init, compared with a literal first, `++`. */
const LITERAL_FOR = /^for \(var (\w+) = \d+u?; \1 < (?:0x[0-9a-f]+|\d+)u?; \1\+\+\)$/;
/** Runtime-bounded loops that remain: a leaf's triangle loop (P-W1 measured no cost for these). */
const ALLOWED = new Set(['for (var i = 0u; i < count; i += 1u)']);

describe('bvh walk loops (P-W1, bvh#3)', () => {
  it('has no `loop {}` or `while`, and every `for` either has a literal bound or is a leaf triangle loop, in every walk variant', () => {
    for (const options of variants) {
      const walk = stripWgslComments(traceModuleWgsl(options));
      const label = JSON.stringify(options);
      expect(walk, label).not.toMatch(/\bloop\s*\{/);
      expect(walk, label).not.toMatch(/\bwhile\b/);
      const fors = walk.match(/\bfor \([^)]*\)/g) ?? [];
      expect(fors.length, label).toBeGreaterThan(0);
      for (const f of fors) expect(LITERAL_FOR.test(f) || ALLOWED.has(f), `${label}: ${f}`).toBe(true);
    }
  });
});
