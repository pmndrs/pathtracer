// P-W1: bvh's walk loops are rewritten as constant-bounded `for` loops (no Tint loop counter). The rewrite must apply to every walk
// variant the installed bvh emits, or a bvh change silently brings back the 3.7x D3D12 slowdown on NVIDIA.
import { describe, expect, it } from 'vitest';
import { traceModuleWgsl, type AnyHitOrder } from '@pmndrs/bvh';
import { rtBoundWalkLoops, rtBoundWalkLoopsApplies } from './modules.ts';

const variants: Array<Parameters<typeof traceModuleWgsl>[0]> = [];
for (const rayMasks of [true, false])
  for (const triangleLeaves of ['records', 'indexed'] as const)
    for (const nodeFormat of ['f32', 'quantized'] as const)
      for (const anyHitOrder of [undefined, 'far', 'near', 'larger'] as (AnyHitOrder | undefined)[])
        for (const positionLayout of triangleLeaves === 'indexed' ? (['vec4', 'packed'] as const) : ([undefined] as const))
          variants.push({ traceSlot: { group: 0, firstBinding: 0 }, rayMasks, triangleLeaves, nodeFormat, ...(anyHitOrder ? { anyHitOrder } : {}), ...(positionLayout ? { positionLayout } : {}) });

// Wide walks (X1.i): six loops, bounded the same way, the two walk functions get their final return.
const wideVariants: Array<Parameters<typeof traceModuleWgsl>[0]> = [];
for (const nodeFormat of ['bvh4', 'cwbvh8'] as const)
  for (const wideDecode of ['exact', 'folded'] as const)
    for (const wideWalk of nodeFormat === 'bvh4' ? (['nested', 'single'] as const) : ([undefined] as const))
      for (const rayMasks of [true, false]) wideVariants.push({ traceSlot: { group: 0, firstBinding: 0 }, rayMasks, nodeFormat, wideDecode, ...(wideWalk ? { wideWalk } : {}) });

describe('rtBoundWalkLoops on wide walks (X1.i)', () => {
  it('applies to every wide variant, leaves no bare loop and ends both walks with a return', () => {
    for (const options of wideVariants) {
      const walk = traceModuleWgsl(options);
      expect(rtBoundWalkLoopsApplies(walk), JSON.stringify(options)).toBe(true);
      const loops = (walk.match(/\bloop \{/g) ?? []).length;
      const out = rtBoundWalkLoops(walk);
      expect(out, JSON.stringify(options)).not.toMatch(/\bloop \{/);
      for (let i = 0; i < loops; i++) expect(out).toContain(`for (var rtWalkStep${i} = 0u; rtWalkStep${i} < 0xffffffffu; rtWalkStep${i}++) {`);
      expect(out).not.toContain(`rtWalkStep${loops}`);
      const closest = out.slice(out.indexOf('fn bvhWalkClosest'), out.indexOf('fn bvhWalkAny'));
      expect(closest.trimEnd()).toMatch(/return BvhHit\(-1\.0, 0u, 0u, vec2f\(0\.0\), true, BVH_NO_MATERIAL, vec3f\(0\.0\)\);\s*\}$/);
      const any = out.slice(out.indexOf('fn bvhWalkAny'));
      expect(any.slice(0, any.indexOf('\n}\n') + 2)).toMatch(/return false;\s*\}$/);
    }
  });
});

describe('rtBoundWalkLoops (P-W1)', () => {
  it('applies to every walk variant of the installed bvh', () => {
    for (const options of variants) expect(rtBoundWalkLoopsApplies(traceModuleWgsl(options)), JSON.stringify(options)).toBe(true);
  });

  it('leaves no bare loop in the walk, bounds four loops with unique constant-bounded indices, and gives both walks a final return', () => {
    for (const options of variants) {
      const out = rtBoundWalkLoops(traceModuleWgsl(options));
      expect(out).not.toMatch(/\bloop \{/);
      for (let i = 0; i < 4; i++) expect(out).toContain(`for (var rtWalkStep${i} = 0u; rtWalkStep${i} < 0xffffffffu; rtWalkStep${i}++) {`);
      expect(out).not.toContain('rtWalkStep4');
      const closest = out.slice(out.indexOf('fn bvhWalkClosest'), out.indexOf('fn bvhWalkAny'));
      expect(closest.trimEnd()).toMatch(/return bvhFinishClosest\(result, hitRecord, hitSlot\);\s*\}$/);
      const any = out.slice(out.indexOf('fn bvhWalkAny'));
      expect(any.slice(0, any.indexOf('\n}\n') + 2)).toMatch(/return false;\s*\}$/);
    }
  });

  it('returns text it does not recognise unchanged', () => {
    expect(rtBoundWalkLoops('fn f() { loop { break; } }')).toBe('fn f() { loop { break; } }');
  });
});
