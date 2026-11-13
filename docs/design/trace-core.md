# Trace core (P1 slice 1)

What `@pmndrs/raytracer` traces today, how the two entry points relate, the batch ABI, how correctness is checked,
and the choices still open. API reference: [packages/raytracer/README.md](../../packages/raytracer/README.md).

## Layers in this slice

| Layer | WGSL | Notes |
|---|---|---|
| L0 | `rtVisible(ray, mask) -> bool` | any-hit over `[tMin, tMax]` (inclusive), bvh's any-hit walk (far child first by default, `anyHitOrder` setting). `true` = visible (V25) |
| L1 | `rtClosest(ray, mask) -> RtHit`, `rtClosestCone(ray, mask, cone)` | closest hit over `[tMin, tMax]`; `t = RT_MISS_T` (−1) on a miss. The cone is carried through only: `coneWidth = width + spread·t` (V23) |
| — | `rtSurface(hit, ray, want)` | lazy decode: position, geometric normal + front face (free, from the hit), shading normal, UV, material id. Reads only what `want` asks for (V24) |

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
scene size. Counted: 14,122 ties, 9,228 grazing, 3,563 tPosition. By ray set: primary 8 ties in 98k rays; incoherent
129 ties and 56 tPosition in 131k; edges (aimed at edges on purpose) 11,241 ties and 5,333 grazing in 131k; inside
2,712 ties in 33k; shadow 3,127 tPosition in 131k (segments that start 1e-4 × scene size off a surface and graze
it). The page takes ~4 s; the largest generated kernel is 49.8 KB (the inline module with `rtSurface` plus the
page's kernel), just under the 50 KB cap.

## Open choices (D11: both are built; A/B on a quiet machine decides)

Each is a setting already; `trace-timing` takes it as a page argument so the runner can alternate them
(`--ab workgroupSize=64,128`).

1. **Workgroup size 64 vs 128** (32 and 256 also built). Suggest 64 (bvh's batch default).
   Pro 64: more workgroups in flight to hide divergence, matches bvh. Pro 128: fewer workgroups, maybe better on
   NVIDIA. A/B: `trace-timing --ab workgroupSize=64,128` on sponza-white, M5 and RTX 3060, paired ratios.
2. **AoS vs SoA rays.** Suggest AoS (the contract layout, bvh-compatible, one 32 B load).
   Pro SoA: wavefront queues compact origin and direction separately; lanes load 16 B each. Con SoA: two buffers and
   bindings. A/B: `--ab rayLayout=aos,soa`.
3. **Full (40 B) vs compact (24 B) hit records.** Suggest full by default, compact for consumers that call
   `rtSurface` anyway. A/B: `--ab hitRecord=full,compact` (write bandwidth).
4. **Position in `rtSurface`: from the ray vs from barycentrics.** Default ray (no vertex reads; error grows with
   t); `positions: true` gives an on-surface point for +12 B/vertex. A/B: self-intersection rate of continuation rays
   in P2 (count hits with t < 1e-4 at the origin's own triangle) and decode cost.
5. **Indirect args: our one-thread kernel vs producer-written args.** Suggest keep the args kernel (the producer only
   has to write a count). Pro producer-written: one dispatch fewer. A/B: dispatch overhead in an empty-queue frame.
6. **Ray mask per batch vs per ray.** v0 has one mask per batch (a uniform). Per-ray masks need a 4 B field the
   32 B ray has no room for (or SoA's spare lane). Decide when a consumer mixes masks inside one queue.

## Not in this slice

L2 `rtShade`, the material/light scene copy (D4) replacing the surface table, L3, reflections moving onto the core
(D5), per-ray masks, subgroup ops (none are used yet, so the uniform-flow rule holds trivially).
