# Trace core (P1 slice 1)

What `@pmndrs/raytracer` traces today, how the two entry points relate, the batch ABI, how correctness is checked,
and the choices still open. API reference: [packages/raytracer/README.md](../../../packages/raytracer/README.md).

## Layers in this slice

| Layer | WGSL | Notes |
|---|---|---|
| L0 | `rtVisible(ray, mask) -> bool` | any-hit over `[tMin, tMax]` (inclusive), bvh's any-hit walk (far child first by default, `anyHitOrder` setting). `true` = visible (V25) |
| L1 | `rtClosest(ray, mask) -> RtHit`, `rtClosestCone(ray, mask, cone)` | closest hit over `[tMin, tMax]`; `t = RT_MISS_T` (−1) on a miss. The cone is carried through only: `coneWidth = width + spread·t` (V23) |
| — | `rtSurface(hit, ray, want)` | lazy decode: position, geometric normal + front face (free, from the hit), shading normal, UV, material id. Reads only what `want` asks for (V24) |

Interval queries for gi's Split Radiance Cascades (half-open `(t0, t1]`, the batched `RtIntervalTracer`, probe direction bundles) are a separate layer on top: [split-rc-tracing.md](split-rc-tracing.md).

Underneath is bvh's walk, unchanged: `traceModuleWgsl` with its compile-time formats (quantized/f32 nodes,
record/indexed leaves, vec4/packed positions), `rayMasks`, `anyHitOrder` and the `bvhTrianglePasses` hook (the
`triangleFilter` option). The generator checks bvh's `TRACER_GPU_LAYOUT_VERSION` against our contract (v2) before
emitting a walk. Everything goes through the caps preprocessor (`buildWgsl`): `#if` blocks, caps `enable` lines.

## Inline vs batch

Both run the same WGSL functions; the batch kernel is the inline module plus a 40-line entry point.

| | Inline (`rtModuleWgsl`) | Batch (`RtBatchTracer`) |
|---|---|---|
| Who writes the kernel | you | us |
| Ray source | anything in your shader | a GPU ray buffer (AoS 32 B or SoA 2×16 B) |
| Count | yours | a number (direct) or a u32 in a GPU buffer (indirect) |
| Results | whatever you compute | one 40/24 B hit record or one u32 per ray |
| Cost | no extra pass, no ray/hit traffic | one pass, rays in + hits out through memory |
| Use for | megakernels, shading at the hit, `rtSurface` | wavefront queues, consumers that already have ray buffers (reflections), tests |

The correctness page checks that the inline kernel and every batch variant give bit-identical hits.

## Queue ABI v0 (batch)

Group 0: the scene, bvh's walk bindings (`rtWalkEntries` + `rtSceneBindGroupEntries`). Group 1: the batch.

| Binding | Type | Content |
|---|---|---|
| 0 | storage, read | AoS: `array<RtRay>` (32 B, = contract `Ray` = `BvhRay`). SoA: `array<vec4f>` origin.xyz + tMin |
| 1 | storage, read | SoA only: `array<vec4f>` direction.xyz + tMax |
| 2 | storage, rw | results: `RtHitRecord` (40 B: t, instance, triangle, frontFace u32, bary vec2f, material, normal xyz; or 24 B compact) or `u32` (1 = visible) |
| 3 | uniform | `RtBatchParams { count, mask, capacity, countWord }` (16 B) |
| 4 | storage, read | indirect only: counts `array<u32>`; the batch traces `min(counts[countWord], capacity)` rays |
| 5, 6 | storage | cones only: `array<RtCone>` in (8 B), `array<f32>` cone width out |

- **Dispatch.** Rays map to `(workgroup.y · numWorkgroups.x + workgroup.x) · size + local`, so batches past
  65535 workgroups use a 2D grid (`dispatchGrid`). Indirect batches first run `rtBatchArgs` (one thread) that turns
  the GPU count into `(x, y, 1)` args; both dispatches go into the same pass, WebGPU orders them.
- **No CPU in the hot path.** Params uniforms, args buffers and bind groups are created once per batch shape
  (buffers + count + mask + capacity) and cached; a steady frame records dispatches only. No readback anywhere.
- **Capacity.** Every buffer must hold `capacity` entries; the tracer checks sizes when a shape is first seen.
- **Mask.** One ray mask per batch (bvh instance masks, 8 bits); per-ray masks are not in v0.

## Correctness (bench/pages/trace-core)

Scenes: Cornell and quad-light (Tier 0, plus a mirrored, scaled extra instance each), sponza-white (Tier 2, 262k
triangles, per-triangle materials, plus a mirrored second instance), and `edges` (bvh's tessellated quad under three
transforms including a mirror, and a closed box). Each is built twice: quantized nodes + record leaves + static
instances merged (bvh's defaults), and f32 nodes + indexed leaves, unmerged. Ray sets: primary, incoherent (half with
random `[tMin, tMax]` intervals), edges (aimed at triangle edges and vertices; on the plain quad exactly at the grid
lines, a quarter of them straight down), inside (from inside the closed box: every ray must hit), shadow segments
(to emitters, the sun, other surface points) and the same shadow segments with a ray mask that hides the emitters.

The oracle is bvh's CPU walk over the **same** `TraceSceneData` the GPU traces (`createCpuTracer`; `CpuSceneBvh`
for the list of all hits). The production path (`SceneTrees`, GPU-built trees refined by tinybvh) is compared to a
CPU-built scene of the same format. Checks per ray: closest hit (t, instance, triangle, front face, material,
normal), visibility, L0/L1 consistency on the GPU, every batch variant bit-identical to the base batch, the inline
kernel bit-identical to the batch, cone widths, and `rtSurface` against `referenceSurface` (doubles), including the
barycentric world position against `origin + t·dir`.

**Why not bit-exact.** bvh's CPU walk runs f64 arithmetic on the f32 scene; the GPU runs f32. Near edges they can
disagree for legitimate reasons, so those cases are classified, counted and reported per ray set (V27):

- *tie*: another triangle at the same t, next to an edge/vertex or on a coplanar surface (a block standing on the floor);
- *grazing*: hit vs miss where the ray only **touches** the surface at an edge or vertex (convex edge of a solid seen
  from outside, boundary edge of an open mesh). Decided geometrically: around the edge, do all triangles face the ray
  the same way (the ray crosses the surface) or not (it only touches it)?
- *tPosition*: same triangle, t further than 1e-5 relative, but the gap measured along the normal is within 1e-5 of the
  coordinates' magnitude (f32 rounding of the object-space ray at grazing angles).

A miss where the ray **crosses** a shared edge or vertex is a watertightness slip and always fails, as does any
other disagreement.

Results on the M5 Pro (Chrome 157, 2026-11-13; the machine was loaded, which does not matter for correctness):
**pass**. 589,784 rays over four scenes and four scene builds (quantized + records + merged, f32 + indexed,
`SceneTrees` GPU trees, `SceneTrees` refined trees): 0 mismatches, 0 watertight slips, 0 escapes from the closed box;
every batch variant and the inline kernel bit-identical to the base batch; `rtSurface` position within 2.3e-7 of the
scene size. Counted (rerun 2026-11-14 with bvh build workers on, bvh's `sharedEdgeRays` and strict primary rays):
15,414 ties, 9,228 grazing, 3,444 tPosition, and 8 `primary_tie`. By ray set: primary 8 primary ties in 98k rays
(Sponza's duplicate triangles, same 3 vertices: both hit, tree order picks one); incoherent 129 ties and 56
tPosition in 131k; edges (aimed at edges on purpose) 12,517 ties, 5,333 grazing and 368 tPosition in 131k; inside
2,736 ties in 33k; shadow 32 ties and 3,008 tPosition in 131k (segments that start 1e-4 × scene size off a surface
and graze it). **Primary rays are strict**: any other instance/triangle difference or a hit-vs-miss fails. The page takes ~4 s; the largest generated kernel is 49.8 KB (the inline module with `rtSurface` plus the
page's kernel), just under the 50 KB cap.

## Open choices (D11: both are built; A/B on a quiet machine decides)

Every choice below is already a setting, and the shipped default is the suggestion. Nothing needs deciding before
P1.6 (the quiet-machine A/B, M5 Pro + RTX 3060); the maintainer's call is whether the suggestion and the test plan
are right. `trace-timing` takes each setting as a page argument so the runner alternates them ABBA
(`--ab workgroupSize=64,128`). The only timings so far were taken on a loaded machine (P1.2, provisional): sponza-white,
1M rays, wg 64 / AoS / full: L1 primary 116, L1 bounce 78, L0 shadow 83, L0 AO 234 Mrays/s. No A/B numbers yet.

### 1. Workgroup size: 64 or 128 (32 and 256 also built)

**What it is.** How many rays one GPU workgroup traces together in the batch kernel. Each thread walks its own ray,
so this only changes scheduling, never results (the page checks every size gives bit-identical hits).

**Suggestion: 64.** bvh's batch kernel uses 64, and it is the safe middle across vendors.

| | 64 | 128 |
|---|---|---|
| Pros | More, smaller workgroups: the GPU can swap in another when one stalls on memory; a workgroup finishes when its slowest ray does, and smaller groups waste less on one long ray | Fewer workgroups to launch; on NVIDIA (32-wide warps) it can raise occupancy if registers allow |
| Cons | More launch overhead on very large batches | Divergent rays (bounces) keep a bigger group waiting on its slowest ray |

- **Speed:** expected within ±10% either way; the winner may differ between Apple and NVIDIA. **Quality:** none.
- **Setting:** yes (`workgroupSize`). If Apple and NVIDIA disagree we pick per adapter from the caps tier.
- **Settle it:** `trace-timing --ab workgroupSize=64,128` on sponza-white, primary and bounce rays, M5 and 3060, paired
  ratios. Under 3% difference: keep 64.

### 2. Ray layout: AoS or SoA

**What it is.** How a ray buffer is laid out. AoS (array of structs): each ray is one 32 B record (origin, tMin,
direction, tMax), the contract layout that bvh also uses. SoA (struct of arrays): two buffers, one of origins + tMin
and one of directions + tMax, 16 B each.

**Suggestion: AoS.** It is the contract and bvh layout, needs one buffer, and one 32 B load per ray is already well
aligned.

| | AoS | SoA |
|---|---|---|
| Pros | One buffer and binding; same layout as bvh and the `Ray` contract; simplest for consumers | Wavefront code that rewrites only origins (or only directions) touches half the memory; neighbouring threads read neighbouring 16 B values |
| Cons | Rewriting one half still moves the whole 32 B record | Two buffers and two bindings; consumers must convert; one more layout to keep tested |

- **Speed:** for the trace itself, likely equal (each ray is read once, then the walk dominates). SoA could win in P2
  wavefront queues when a stage writes only half the ray. **Quality:** none (bit-identical hits).
- **Setting:** yes (`rayLayout`).
- **Settle it:** `--ab rayLayout=aos,soa` now for the trace cost; repeat in P2 with the real queue stages, where the
  difference would show.

### 3. Hit record: full (40 B) or compact (24 B)

**What it is.** What the batch writes per ray. Full: t, instance, triangle, front face, barycentrics, material id
and geometric normal. Compact drops material and normal (the consumer looks them up later with `rtSurface`).

**Suggestion: full by default; compact for consumers that call `rtSurface` anyway.**

| | Full 40 B | Compact 24 B |
|---|---|---|
| Pros | Shading can start without another lookup; simplest to use | 40% less write bandwidth, and the reading stage loads less too |
| Cons | More memory traffic per ray | Normal and material cost a lookup later (vertex/triangle reads) |

- **Speed:** compact helps when the trace is bandwidth-bound (coherent primary rays at high resolution); probably no
  difference on incoherent bounces, where the walk dominates. **Quality:** none.
- **Setting:** yes (`hitRecord`).
- **Settle it:** `--ab hitRecord=full,compact` for trace cost; the real answer needs the end-to-end cost (trace plus
  the later lookup), so repeat once L2 shading exists (P1.3).

### 4. Hit position in `rtSurface`: from the ray or from the triangle

**What it is.** Where the world position of a hit comes from. From the ray: `origin + t·direction` (no extra memory
reads, but f32 error grows with distance, so the point can land slightly in front of or behind the surface). From
the triangle: interpolate its three vertices with the barycentrics (exactly on the surface, but needs the vertex
positions: +12 B per vertex in the surface data, plus three reads per lookup).

**Suggestion: from the ray by default, triangle opt-in (`positions: true`).** Measured error so far is small (within
2.3e-7 of scene size), and every new ray already starts with an offset.

| | From the ray | From the triangle |
|---|---|---|
| Pros | Free; no extra scene memory | On the surface, so the offset for the next ray can be smaller and more reliable |
| Cons | Error grows with t; at grazing angles a continuation ray can re-hit its own triangle ("shadow acne", light leaks) | +12 B/vertex of memory and three reads per lookup |

- **Speed:** triangle costs a few % of decode time (unmeasured). **Quality:** this is the one choice that can change
  images: self-intersection shows as dark speckles or leaks on large or distant surfaces.
- **Setting:** yes (`positions`).
- **Settle it (P1.8):** count continuation rays that hit their own triangle at t < 1e-4 on Sponza and a far-away
  large plane, both modes, and show the two renders side by side in the bench viewer. Switch the default if the ray
  mode shows visible artefacts.

### 5. Indirect dispatch arguments: our kernel or the producer's

**What it is.** In a wavefront renderer, the number of rays in a queue is only known on the GPU. Something must turn
that count into dispatch arguments. Today a tiny one-thread kernel (`rtBatchArgs`) does it: the producer only writes
a count. The alternative: the producer writes the dispatch arguments itself.

**Suggestion: keep our args kernel.** Producers stay simple, and the grid maths (2D grid past 65,535 workgroups)
stays in one place.

| | Our args kernel | Producer writes the args |
|---|---|---|
| Pros | Producer writes one number; no grid maths in other packages; capacity clamp in one place | One dispatch fewer per batch |
| Cons | One extra tiny dispatch per batch | Every producer must get the 2D grid and the clamp right |

- **Speed:** one tiny dispatch costs microseconds; only matters with many small batches per frame. **Quality:** none.
- **Setting:** could become one (accept a ready args buffer); not built yet.
- **Settle it:** time an empty-queue frame with and without the args dispatch. (bvh will not add an indirect batch
  kernel, pmndrs/bvh#1 item 5, so ours stays.)

### 6. Ray mask: one per batch or one per ray

**What it is.** Masks pick which instances a ray can hit (for example: shadow rays that ignore emitters). v0 has one
mask for the whole batch. Per-ray masks would need a 4 B field, and the 32 B ray has no room (SoA has a spare lane).

**Suggestion: one per batch until a consumer needs to mix masks in one queue (P1.9).** No current consumer does.

| | Per batch (v0) | Per ray |
|---|---|---|
| Pros | Free (a uniform); the ray stays 32 B | Mixed ray kinds in one queue |
| Cons | Mixed masks need separate batches | Ray grows to 48 B in AoS (extra bandwidth on every ray), or SoA only |

- **Speed:** per ray costs bandwidth on every ray, even when unused. **Quality:** none.
- **Setting:** would be a queue ABI version (v1), not a runtime switch.
- **Settle it:** decide when a consumer needs it; then A/B 32 B + per-batch against 48 B per-ray on the same queue.

### 7. `rtVisible` returns true for visible (bvh returns true for blocked)

**What it is.** bvh's any-hit trace answers "is something in the way?" (true = blocked). Our `rtVisible` answers
"can the ray get through?" (true = visible), as the spec's L0 reads (DIVERGES V25).

**Suggestion: keep true = visible.** The name says what it returns, and shading code multiplies by visibility.

- **Pros:** reads naturally (`if (rtVisible(...)) { add light }`). **Cons:** the opposite of bvh; anyone moving
  between the two can trip.
- **Speed/quality:** none (one negation). **Setting:** no; an API name, decided once.
- **Settle it:** naming only. If it confuses, rename to `rtUnoccluded` rather than flip the polarity.

## Not in this slice

L2 `rtShade`, the material/light scene copy (D4) replacing the surface table, L3, reflections moving onto the core
(D5), per-ray masks, subgroup ops (none are used yet, so the uniform-flow rule holds trivially).
