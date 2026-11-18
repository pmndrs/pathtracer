# Tracing support for gi's Split Radiance Cascades (P5.rc1)

What `@pmndrs/raytracer` offers a 3D radiance-cascades technique, how it behaves at interval boundaries, what the cascade-shaped workload
costs, and how gi calls it. The technique itself (probe hash, ray generation, deposit, merge, shading) is gi's (D3, spec section 8:
"3D Split Radiance Cascades over a triangle BVH in WebGPU"); this repo provides **tracing**. Code: `packages/raytracer/src/trace/interval.ts`
(WGSL + TypeScript twins), `interval-batch.ts` (the dispatcher). Checks: `bench/pages/rc-interval-check.ts` (correctness),
`bench/pages/rc-cascades.ts` (the Sponza workload and a naive merge). DIVERGES V290-V293, TASKS P5.rc1-P5.rc6.

## 1. What Split RC asks of a tracer

From the paper (Freeman, Sannikov, Pitt-Francis, arXiv 2607.20384, section 3.2 / 5, `research/notes/08-realtime-gi.md`):

- A cascade n owns the interval `(t_{n-1}, t_n]`, `t_{-1} = 0`. A ray is "split": a ray from x hitting at t, `t_{n-1} < t <= t_n`, tells cascade n
  "occluded, radiance J" and every cascade k < n "transparent". So the tracer must answer **closest hit within an interval, with exact
  half-open boundaries**, and the caller needs `t` to find the cascade (`rtCascadeIndex`).
- Two ray sources: classic RC traces from each probe (many rays, one origin, neighbouring directions, per cascade), Split RC from
  visible surfaces (about one ray per pixel, incoherent, 1 spp). Both are covered (`probes` and `rays` sources).
- Radiance at the hit: albedo / emission / direct light, or a cache lookup (their "secondary cache"): `rtShade` with the indirect hook.
- Short rays: the interval ends bound the work, and cascade 0's rays are a few cells long.
- Miss in the last cascade samples the sky: that is the caller's (a miss is `t < 0`, the interval ended).

## 2. API

```ts
import { RtIntervalTracer, rtModuleWgsl, cascadeEnd, cascadeIndex, intervalBounds, bundleDirection } from '@pmndrs/raytracer';

const tracer = await RtIntervalTracer.create(device, caps, {
  ...sceneFormatOf(traceBuffers),                  // the bvh scene's formats, as RtBatchTracer
  source: 'probes',                                // 'probes' | 'rays'
  output: 'shaded',                                // 'hit' | 'distance' | 'visible' | 'shaded'
  convention: 'open-low',                          // (t0, t1] (default, the paper) | 'closed-low' [t0, t1)
  scene: { layout: copy.layout, packing },         // shaded only: the scene copy
  shadeMode: 'lambert',                            // shaded only: 'flat' | 'lambert' | 'full'
  shade: { cache: 'external' },                    // shaded only: rtShade options; hooksWgsl: 'fn rtShadeIndirect(p: RtShadePoint) -> vec3f { ... }'
  indirect: false,                                 // the probe / ray count lives in a GPU buffer (RtIndirectCount)
});

tracer.beginFrame();                               // resets the uniform ring (up to 256 jobs per frame)
for (const c of cascades)                          // one dispatch per cascade, one command encoder, no CPU work beyond a 48 B writeBuffer
  tracer.dispatch(pass, trace, {
    source: probeBuffer,                           // array<vec4f>: xyz position, w = interval scale (0 = skip the probe)
    results: results[c.n],                         // intervalResultBytes(output) per ray, probe-major, cell-minor
    count: c.probeCount,                           // or { buffer, word, capacity } for an indirect tracer
    interval: { t0: c.start, t1: c.end },          // scaled per probe by probe.w (Split RC LODs: 1, 2, 4 ...)
    directions: c.dirsPerAxis,                     // D = directions^2 equal-area octahedral cells per probe
    jitter: [jx, jy],                              // position inside the cell, per frame (0.5 = centre)
    seed: frame,                                   // shaded: the light-sample hash
    sceneCopy,                                     // shaded only
  });
```

- **`rays` source**: a `RtRay` AoS buffer (32 B); the ray's own `tMin` / `tMax` ARE the interval. For Split RC's surface rays: gi writes the rays,
  (`0, t_top]`, then reads `t` and picks cascades with `rtCascadeIndex(t, t0, growth, count)`. No `job.interval` / `directions`.
- **`probes` source**: ray `i` is probe `i / D`, cell `i % D` (cell = `(c % N, c / N)`, N = `directions`, D = N^2). Directions are the equal-area octahedral map
  (`rtOctEqualAreaDecode`, TS twin `bundleDirection(x, y, N, jitter)`): every cell has the same solid angle and a 2 x 2 split of a cell stays inside it,
  so cascade n + 1 (2N per axis) refines cascade n exactly (merge reads the four children `(2x + dx, 2y + dy)`). A workgroup of 64 is one probe's
  directions when D >= 64: one origin, neighbouring directions, nothing read from a ray buffer.
- **Outputs**: `hit` = the 40 B (or 24 B `hitRecord: 'compact'`) closest-hit record of `RtBatchTracer`; `distance` = one f32 per ray (`t`, -1 = nothing in the interval);
  `visible` = one u32 (1 = nothing blocks; any-hit, see the limit in section 3); `shaded` = one `vec4f` per ray, `rgb` = `rtShade` radiance at the hit (emission + one
  next-event sample + the `cache` hook), `w` = `t` (`w = -1` and black on no hit). A miss never reads the environment: the interval ended, the caller decides.
- **Inline module** (your own kernel): `rtModuleWgsl(caps, { intervals: true })` adds `rtClosestInterval(origin, direction, t0, t1, mask) -> RtHit` (`(t0, t1]`), `rtClosestIntervalLo`
  (`[t0, t1)`), `rtVisibleInterval` / `Lo`, `rtIntervalBounds`, `rtCascadeEnd / Start / Index`, `rtOctEqualAreaDecode`. Without `intervals` the module text is unchanged.
- **Per-ray masks, cones**: not here (one mask per dispatch; cone widths per cascade are `cone.width + spread * t` on the returned `t`, `rtConeAt`).
- Cost of the calls: params go to a dynamic-offset uniform ring (`INTERVAL_RING_SLOTS` = 256 per frame); `reuseParams: true` repeats the previous job without a slot (timing, several
  frames of one batch). Bind groups are cached per (source, results, counts) buffers. Nothing is read back.

## 3. Semantics at the boundaries (the part to rely on)

Intervals are half-open in f32, on the `t` the walk computes for the hit (a function of the ray and the triangle only, not of the interval):

| convention | interval | a hit at exactly t0 | a hit at exactly t1 |
|---|---|---|---|
| `open-low` (default) | `(t0, t1]` | belongs to the interval below | belongs here |
| `closed-low` | `[t0, t1)` | belongs here | belongs to the interval above |

`(a, b]` and `(b, c]` partition `(a, c]`: `closest(a, c)` is the nearer of the parts' answers and every hit lands in exactly one cascade. `t0 = 0` with `open-low` never reports a surface at `t = 0` (a ray
that starts on a surface does not hit it). `t1` is clamped to `RT_T_INF` (3.4028235e38); an empty interval (`t1 <= t0`) is answered without walking; `t0 >= 0`; NaN not supported.

Two things found while testing it (`rc-interval-check`), both handled and both worth knowing:

1. **bvh's box test is not exact at `tMax`** (V290). A triangle hit at exactly `tMax` can lose its box: the slab entry `(box - origin) * inverseDirection` rounds a few ulp above `tMax`, and the walk
   culls with `tEnter <= tExit` where `tExit = min(far * (1 + 4e-7), tMax)`. Measured on the plain inclusive walk (`RtBatchTracer`) over the four scenes: of 36,666 rays whose `tMax` is set to their own unbounded hit `t`, **4,740 (12.9%) miss it**; with `tMin` set to it
   139 (0.4%) miss it (the far side's relative padding). `rtClosestInterval*` therefore walks a range
   widened by 2^-19 of the coordinates' magnitude (`o + t`) and cuts the answer back to the exact bounds on the computed `t` (above `tMax`: a miss, nothing nearer exists; below `tMin`: re-walk from the next
   float, at most 3 times, needs a hit inside the slack zone). Cost: nothing measurable (section 5).
2. **The any-hit forms cannot do that** (no `t` to cut on): `rtVisibleInterval*` and the `visible` output use the exact bounds, so a blocker within ~1e-6 relative of an interval end can be missed.
   Use `distance` / `hit` where a boundary hit matters (a cascade's transmittance is `distance < 0`).

Ties: two triangles at the same `t` (a ray through a shared edge) may swap with the range (the walk's child order depends on it); `rc-interval-check` counts them (`owner_ties`) and accepts them only when both
sit on an edge / vertex or are coplanar / share a corner.

## 4. Correctness (`node bench/runner/run.ts --page rc-interval-check --args rays=2048`)

All four trace-core scenes (edges, cornell, quad-light, sponza-white), both scene formats (quantized + records + merged; f32 + indexed), ray sets primary / incoherent / edges / inside:

1. **Cascade partition vs the CPU oracle**: every ray cut into six intervals (`e_n = diagonal / 64 * 4^n`, then to infinity), both conventions, against bvh's CPU walk (`createCpuTracer`) fed the f32 bounds of
   the TypeScript twin, with the V27 classification (ties, grazing, tPosition, watertight slips; `bench/pages/trace-compare.ts`, moved out of trace-core). A hit within 1e-5 of an interval end is counted
   `boundary`. Exact GPU invariants: every reported `t` lies inside the bounds, the owner of the unbounded closest hit reports it (bit for bit, or a counted tie), every interval before it misses.
2. **Boundary exactness** at the GPU's own `t_h` (the unbounded closest hit): `(0, t_h]`, `(0, next(t_h)]`, `[0, next(t_h))`, `[t_h, inf)`, `(prev(t_h), inf]` return it; `(0, prev(t_h)]`, `[0, t_h)`, `[0, prev(t_h))` miss;
   `(t_h, inf]`, `(next(t_h), inf]`, `[next(t_h), inf)` report only something farther.
3. **Outputs**: `visible` and `distance` against `hit`, an indirect tracer (count in a GPU buffer, clamped to its capacity) bit for bit against a direct one.
4. **Probe bundles** against the `rays` source fed the twin's directions; `w = 0` probes skipped; indirect probes bit for bit.
5. **WGSL helpers** (`rtCascadeIndex / End / Start` exact on every boundary, growths 2 / 3.3 / 1.7 / 4; `rtOctEqualAreaDecode` within 3.3e-5, GPU sin / cos).
6. **Shaded output** against an inline reference kernel (`rtClosestInterval` + `rtShade`, the same hash) on Cornell and quad-light, flat / lambert / full, two seeds, split and packed scene copy; and its `w` against `distance`.

Result (M5 Pro, Chrome 157, 2026-11-18, load 16-21: correctness is not load-sensitive): **pass**. 49,152 source rays, 589,824 interval rays, 439,992 exact-boundary queries: 0 mismatches, 0 watertight slips, 0 reported
`t` outside its interval, 0 partition errors, 0 output mismatches. Counted, not failed: 8,094 ties, 4,020 grazing, 208 tPosition, 97 owner ties, 12 `walk_level` (disagreements with the CPU that the plain
inclusive walk over the same bounds shares: far hits at 1-5 degree incidence on Sponza's mirrored instance, ray aimed at a shared edge: bvh / V27 territory, not the interval logic). Probe bundles: 187 of 196,608 rays
(0.095%) hit a different triangle or t than the `rays` source (GPU `sin` / `cos` are ~3e-5 off the doubles, enough to flip an edge ray). Shaded: 61,440 rays, 0 mismatches.

`pnpm test`: 17 unit tests (`interval.test.ts`: f32 neighbours, partition for both conventions at every boundary, cascade index, equal-area map: unit length, uniform z, 3% cell areas, nesting; generated WGSL for 88 option
combinations x 2 caps: size, binding numbers, reserved words, struct layout). Every existing kernel is byte-identical (`node bench/size-snapshot.ts` 252 files `diff -r`; 48 raytracer batch / inline kernels, `scratchpad` dump).

## 5. Workload numbers (`rc-cascades`, Sponza, provisional)

Setup: `sponza-white` (524k triangles), camera primary hits on 1280 x 720 (917k visible pixels), cascade 0 = one probe per 0.25 m cell of the hit pushed 0.125 m along its normal (5879 probes), cascade n the cell of
spacing 0.25 * 2^n holding its child (1623 / 494 / 152 probes), directions per probe 16 / 64 / 256 / 1024, intervals (0, 0.5], (0.5, 2], (2, 8], (8, 32]: spacing x2, directions x4, interval x4 (probes / 8, rays per
probe x 4: about the same ray count per cascade, 94k-156k, 480k per frame). Timing: GPU timestamps around >= 300 ms of back-to-back identical dispatches in one pass, 6 interleaved rounds, median (min .. max).
**Machine loaded (load 17-19 on 18 cores, 2 other GPU processes, the shared Mac: provisional, quiet-pass row P5.rc4).**

| cascade (rays) | probes-hit | rays-hit | plain-clamped | plain-full | probes-shaded |
|---|---|---|---|---|---|
| c0 (0, 0.5], 94k | 223 (192 .. 228) | 223 (201 .. 227) | 226 (210 .. 231) | 109 (101 .. 112) | 135 (128 .. 139) |
| c1 (0.5, 2], 104k | 214 (204 .. 216) | 212 (203 .. 215) | 215 (206 .. 219) | 128 (124 .. 131) | 127 (124 .. 129) |
| c2 (2, 8], 126k | 159 (158 .. 164) | 160 (157 .. 163) | 161 (159 .. 165) | 137 (135 .. 141) | 95 (94 .. 97) |
| c3 (8, 32], 156k | 262 (249 .. 264) | 262 (252 .. 265) | 266 (256 .. 268) | 174 (168 .. 177) | 150 (147 .. 153) |
| Split RC form, 917k rays, 1 / pixel, (0, 32] | 101 (99 .. 107) | 101 (99 .. 107) | 101 (99 .. 104) | 101 (100 .. 103) | 72 (70 .. 75) |

Mrays/s. `probes-hit`: `RtIntervalTracer` probes source, compact hit records. `rays-hit`: same rays from a 32 B ray buffer. `plain-clamped`: `RtBatchTracer` closest on that ray buffer with `tMin` / `tMax` = the interval
(inclusive, no half-open semantics, no slack). `plain-full`: the same origins and directions, `[0, 1e30]`. `probes-shaded`: `rtShade` lambert (sun + sky light sample, shadow ray) at every hit. The last row is the Split RC
form: one uniform-sphere ray (flipped into the surface's hemisphere) per visible pixel.

Reading it:

- **The interval machinery is free**: probes-hit / rays-hit / plain-clamped agree within 1-3% in every cascade. Half-open bounds, the slack widening and re-walk, the probe direction decode, the uniform ring and the 32 B ray
  buffer traffic do not show next to the walk (plain-clamped is the cheapest the walk can be with that interval; it is also the variant that misses boundary hits, section 3).
- **Short intervals pay**: against full-length rays from the same origins the interval trace is 2.0x (c0), 1.7x (c1), 1.2x (c2), 1.5x (c3) faster. Cascade 2 (2 .. 8 m) is the slowest per ray for every variant (cause not investigated); cascade 3 has 152 origins, so the rays of
  a workgroup share one origin and neighbouring directions, and it is the fastest.
- **Whole frame** (4 cascades, 480k rays, median ms per dispatch summed): hit only 2.3 ms (208 Mrays/s average), shaded 3.9 ms, plain full-length 3.5 ms. Shading adds 65-76% to the trace time per cascade (the shadow ray per hit).
- **Split RC form**: 917k incoherent rays take 9.2 ms (101 Mrays/s), 12.9 ms shaded. This is the paper's 1 spp trace (they report 1.3 ms for Sponza with hardware RT on a 3080 Laptop, of 8.6 ms for the whole frame; their resolution and
  ray count are not ours): a software walk is several times slower at the trace stage, which matches research note 08's budget statement ("lean on caches"). Per ray the probe-bundle form is about 2x faster than
  the surface-ray form (origin coherence and short intervals); the two are not an equal-quality comparison.
- A trace of 480k rays is 2-4 ms; Split RC's own frame budget (8.6 ms total on a 3080 Laptop) is therefore reachable only if gi keeps its ray count near 0.5 M per frame (probe-bundle form) or accepts ~10 ms for the 1 spp
  form on this GPU.

## 6. Naive merge (check, not product)

`rc-cascades` merges top-down (nearest parent probe, 2 x 2 child directions averaged, a miss takes the upper cascade's radiance, the top cascade's miss takes the sky) and compares the mean radiance of cascade 0 against
unbounded rays from the same probes through the same shading:

| configuration | probes | mean RC | mean brute force | ratio |
|---|---|---|---|---|
| sparse hierarchy (nearest parent at the cell centre) | 1176 | 0.296 | 0.356 | **0.83** |
| co-located cascades (every cascade's probe p at cascade 0's probe p) | 1176 | 0.316 | 0.356 | **0.89** |
| co-located, directions per axis constant (8), one seed: "first interval with a hit" | 75,264 cells | cell by cell | | **0 differing cells, worst relative 2.1e-8** |

The last row is the one that validates tracing: with the same ray in every cascade the merge must reproduce the unbounded ray's radiance exactly, so a hit lost or counted twice at a cascade seam, or an interval reporting
outside itself, would show; it is bit-level equal. The 0.89 / 0.83 are the naive technique's own bias (a coarse cell's centre ray decides "hit within the interval" for the whole cell; parents on the far side of a wall
see the wall), the penumbra hypothesis failing at close walls; Split RC's surface-spawned rays and trilinear parent interpolation exist for this. It is gi's to fix, but the numbers are the baseline for gi's A/B.

## 7. How gi plugs in

- **Probe-bundle path (classic RC, GI via probes)**: one `RtIntervalTracer` (`probes`, `shaded` or `distance`), one dispatch per cascade per frame with `directions` doubling and `interval` from `cascadeEnd`.
  Hit radiance: `shade: { cache: 'external' }` with `hooksWgsl` defining `rtShadeIndirect` over gi's world hash cache (D3): the tracer already compiles `rtShade` with your hook (kernel 46.9 KB minified,
  under the ~50 KB class; a hook with a big lookup can push it over: keep it a function call into a small text, or build your own kernel with the inline module).
- **Split RC path (rays from surfaces)**: your ray generation writes `RtRay`s with `(0, t_top]` into a buffer (the `inline` kernel with `rtClosestInterval` is the same cost as the batch form),
  use `output: 'distance'` / `hit` and `cascadeIndex(t, t0, growth, count)` for the deposit; `rtVisibleInterval` is NOT boundary exact, use `distance`.
- **Ordering and results**: results come out probe-major, cell-minor (`index = probe * D + y * N + x`); if gi wants Morton order for the merge's contiguous child reads (paper section 6) remap the cell index in
  the merge, or tell us and we add an `order: 'morton'` (the kernel's `c` is the only place).
- **Sparse probes**: keep the probe list compact and use the indirect form (`count: { buffer, word, capacity }`), or set `w = 0` to skip a probe in place (it still launches D threads that exit).
- **Not provided (gi's)**: probe hashing and LOD (`w` carries the scale), ray-count prefix sums (Algorithm 3), R2 sampling (the bundle uses cell centre + a `jitter` you vary per frame; for R2 on the `rays` source write the
  directions yourself), deposit (atomics on your cascade storage), merge, irradiance baking, temporal blend.

Open (not done): `order: 'morton'`; trilinear sparse parent lookup helper; an `any-hit` interval walk that is boundary exact (needs the box slack inside bvh's any-hit walk; a request to bvh: pad `tMax` in `bvhBoxEntry` by a few ulp
of the coordinates, which would also remove the widening here); wavefront / queue ABI integration (this tracer takes buffers; `RtBatchTracer`'s indirect counts already feed it); per-ray masks; shared-memory origin cache for the
probe bundle (the walk already starts from one origin per workgroup; measured equal to the ray-buffer form, so no gain expected).
