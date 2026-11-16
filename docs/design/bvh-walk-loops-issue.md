# Draft issue for pmndrs/bvh: walk loops cost 3.7x on D3D12 / NVIDIA (Tint loop counters)

Filed as https://github.com/pmndrs/bvh/issues/3 (2026-11-16). Text as posted below.

---

**Title:** Walk `loop {}`s get Tint's 64-bit loop counter: 3.7x slower when the walk is inlined into a big kernel on Chrome D3D12 / NVIDIA

**Body:**

### Symptom

The pathtracer's megakernel inlines `bvhWalkClosest` and `bvhWalkAny` into its bounce loop. On an RTX 3060 Laptop in Chrome 157 (Dawn D3D12, DXC), it rendered Sponza at 191-202 ms/spp. On Dawn Vulkan on the same GPU it ran at 59-64, and on an M5 Pro at 56-63. The trace core alone is fine there: `trace-timing` reads 0.75-0.86x the M5 Pro. Standalone trace kernels do not show the slowdown; only the walk nested inside a large kernel does.

### Cause

Tint's `PreventInfiniteLoops` transform runs on every backend as part of robustness. It adds a 64-bit iteration counter (`vec2u` with a borrow) to every loop it cannot prove finite. In the walk, that is the outer descent `loop {}` and the inner pop `loop {}` of both functions (`src/trace/wgsl.ts`, around lines 741 and 827).

With those counters, DXC at Shader Model 6.6+ produces code that runs 3.7x slower on NVIDIA. dawn.node D3D12 toggle bisection, megakernel ms/spp at depth 1:

| Setting | ms/spp |
| --- | --- |
| Default | 68.1 |
| `disable_robustness` | 15.7 |
| `d3d12_dont_use_shader_model_66_or_higher` (same HLSL, different DXC target) | 18.9 |

The stack-index clamps are not the cause; removing them changed nothing.

Tint leaves a loop alone only in this form: `for (var i = <int>; i <op> <literal constant>; i++)`, with the bound test as the first statement. A runtime bound, `while` and `loop {}` all get the counter.

### Suggested change

Write the four walk loops as constant-bounded `for` loops:

```wgsl
for (var step = 0u; step < 0xffffffffu; step++) {   // was: loop {
```

Because WGSL can no longer prove the end of the function unreachable, both walk functions then need a final return:

- `return bvhFinishClosest(result, hitRecord, hitSlot);` in `bvhWalkClosest`
- `return false;` in `bvhWalkAny`

That return replaces the comment "No return here: the loop above only exits through 'return'." The traversal is unchanged; each kernel grows by about 130 bytes.

The pathtracer applies this rewrite as a post-process today (`rtBoundWalkLoops` in `packages/raytracer/src/trace/modules.ts`). With it, Chrome D3D12 runs Sponza at 58-61 ms/spp, a 3.3x speedup, with an identical image (max relative difference 4.5e-7). Dawn Vulkan is unchanged, and the trace-core CPU oracle shows 0 mismatches. Once bvh emits this shape itself, the post-process can be deleted.

Related places that would need the same care:

- `src/trace/walk-counters.ts` anchors on the text `'    loop {\n        let count = ...'` and would need its anchor updated.
- `src/trace/query-wgsl.ts` has the same walk structure (lines 145 and 231).
- The H-PLOC build loop (`gpu-hploc-wgsl.ts:933`) runs once per build, so it matters less.
- The remaining counters (triangle loops with a runtime `count`) cost nothing measurable.

The original rationale and numbers are in the pathtracer repo: TASKS.md row P-W1 and DIVERGES.md row V143.
