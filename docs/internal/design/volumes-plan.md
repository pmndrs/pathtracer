# Volumes plan: the best web path tracer for media (2026-11-21)

Goal: not parity with offline renderers, but the best participating-media path tracer on the web, judged on **speed, quality and accessibility** (runs everywhere a
browser runs WebGPU: Apple, Windows / D3D12, Intel iGPUs; sensible defaults; demos that converge in seconds). Inputs: the clouds handoff
([CLOUDS-HANDOFF.md](../CLOUDS-HANDOFF.md), items 1-6 done or open), the survey ([research/notes/14](../../../research/notes/14-volumes-survey.md)) and the comparison with our code
([research/notes/15](../../../research/notes/15-volumes-vs-ours.md)). Where we stand: the estimator agrees with pbrt / Mitsuba at their noise floor in the cloud regime
(bench/results/VOLUMES.md); what is missing is speed per sample, noise per time, memory, real-time modes and web-facing features.

## How this track is run (the orchestration protocol)

- **Tasks are small.** Every task below is S (one agent round, <= ~60 min / ~50 tool calls) or is split into S steps. An agent that reaches its budget, or fails the
  same thing three times, stops and reports (done, failing output, its theory). It never widens its scope.
- **The orchestrator reviews every report before the next round**: is the agent in a logic loop, polishing something trivial, or testing too heavily? Then it either
  re-briefs, splits the task, or drops it.
- **Checks are the cheapest thing that can fail**, in this order: unit test / TS twin -> closed form at <= 1024 spp -> same-seed A/B of an existing `vol-*` scene at
  <= 256 spp -> (gates only) a reference comparison. Unbiasedness checks use the second-moment standard error (`volumes.outputs`) instead of extra seeds. No 4096 spp
  sweeps or full agreement tables for a local change; say so before running one.
- **Timing never blocks a task**: correctness lands, the timing goes to the quiet-pass queue (docs/internal/QUIET-PASS.md); defaults change after it.
- **Kernel bytes are a budget**: volume megakernels are 50.5-54 KB already; a task that adds WGSL names the bytes it adds, and V1.2 (hardware filtering) pays some back.
  Every default kernel stays byte-identical unless the task's point is to change it.
- Docs per task as usual (TASKS row, DIVERGES for any departure, Traps for anything that cost time).

## Phases

Status: ⬜ not started · 🟨 in progress · ✅ done. ★ = the clouds handoff needs it or it attacks its grain / memory problem.

### ✅ V0 — Validation and outputs (P5.v13-v19, 2026-11-21)

Cloud regime at the pbrt / Mitsuba floor (27 cells + a 256^3 cloud), droplet phases, volume outputs (order split, transmittance, standard error), r16float grids, ground
scenes, per-hit origin offset bound + free flight from the true crossing. Open from it: Cycles comparison of the Mie / Draine phases (waits on the clouds side's script).

### 🟨 V1 — Housekeeping and cheap speed (all S)

Gate: `vol-cl-cloud256` and `vol-cl-sph-g85-a99-t200` at 256 spp keep their mean within 2 standard errors of today's, with fewer density lookups per shadow ray (counter); quiet-pass ms/spp queued.

| id | task | check (minutes) |
|---|---|---|
| ✅ V1.1 ★ | Roulette in ratio tracking (T < 0.05, survive 0.75, weight 1/0.75), shadow rays and the outputs' transmittance, both schedulers | TS twin mean at 1e5 trials vs line integral; the two scenes at 256 spp within the SE; lookup counter |
| V1.2 ★ | Hardware-filtered density (`textureSampleLevel` on r16float / r8unorm, r32float with `float32-filterable`) as an option, manual 8-tap stays the default until the A/B | `vol-grid-slab` closed form; same-seed relMSE vs manual on `vol-cl-cloud256` (~1e-5); bytes saved |
| ✅ V1.3 ★ | Separate `maxVolumeDepth` (medium scatters vs surface bounces) | equal to today with both caps equal; `vol-cl-ground-sph` with surface 8 / medium 2048 |
| V1.4 | Loop caps visible: count truncations (4096 collisions, brick cell cap, ratio-tracking cap); warn at upload when densityMax x sigma_t x diagonal > ~1e3 | synthetic dense slab near the cap |
| ✅ V1.5 | Doc fixes: volumes.md sections 3 / 6.1 and DIVERGES V243 still describe the global majorant only | read-through |
| ✅ V1.6 ★ | Per-pixel wavefront check of lobe scenes | `wf-p5-check cases=vol-cl-single-mie10,vol-cl-sph-hg2-a99-t50 spp=16` |
| V1.7 | Offset-bound follow-ups: ReSTIR G-buffer positions, light-sample end points, medium interface skips, `positions: 'ray'` | `vol-cl-ground-occluder` with each path on |
| V1.8 ★ | Windows DXC compile of the 50-54 KB volume megakernels (and the wavefront split as fallback) | `kernel-compile` page on the 3060, cold / warm |

### ⬜ V2 — Noise per time (S steps of M items)

Gate: equal-time relMSE on `vol-cl-cloud256` (and one dense homogeneous sphere) at least 2x better than V1, mean within 2 SE (provisional until the quiet pass).

| id | task | check |
|---|---|---|
| V2.1 ★ | Minorant per brick (brick texture rg = min, max) | TS twin: min <= every density sample in the cell |
| V2.2 ★ | Residual ratio tracking for shadow rays against the minorant (verify the residual-tracking patent status first) | TS twin unbiased; `vol-grid-slab`; 256 spp within SE |
| V2.3 ★ | Misso 2022 telescoping transmittance, port of Cycles `volume_transmittance` (Apache-2.0, attribution), A/B vs ratio tracking | TS twin unbiased; equal-spp variance on `vol-cl-cloud256` |
| V2.4 | Decomposition tracking for free flight (control part analytic, residual majorant) | `vol-grid-slab`; lookup counter |
| V2.5 ★ | Adaptive sampling per tile from the second moment (stop when SE < target) | spp saved at equal SE on `vol-cl-cloud256` |
| V2.6 | Finite sun disc A/B against the delta sun for peaked phases (agree with the clouds side first) | `vol-cl-sph-mie20` 64 spp firefly count |

### ⬜ V3 — Memory and scale

Gate: the clouds' 504x352x616 grid fits in <= 128 MB with an image same-seed equal (relMSE <= 1e-5) to the dense r32 one.

| id | task | check |
|---|---|---|
| V3.1 ★ | r8unorm grids with a per-medium scale | same-seed relMSE vs r16 |
| V3.2 ★ | Sparse brick atlas: indirection texture at brick resolution + atlas of occupied 8^3 bricks with a 1-voxel apron (one texture, no storage buffer); gives several grids per kernel | same-seed image vs dense on `vol-cl-cloud256`; memory printout |
| V3.3 | Camera inside a grid medium; auto-detect the camera medium | `vol-camera-medium` grid twin |
| V3.4 | Browser `.vdb` loader (WebVDB `parseVdb` in a worker -> dense / atlas; licence check of WebVDB / NanoVDB first) | `vol-vdb-smoke` from the browser = from Node |

### ⬜ V4 — Lights in media and outdoor fog

Gate: lamp-lit fog and thin-fog god rays at least 3x better equal-time relMSE than V1 (own 4096 spp reference, once).

| id | task | check |
|---|---|---|
| V4.1 | Equiangular + distance MIS for emissive triangles in homogeneous media | new `vol-lamp-fog` (maxDepth 1) vs a 1D quadrature |
| V4.2 | Forced in-scatter NEE for thin homogeneous media | thin-fog closed form (single scatter) |
| V4.3 | Volume scattering probability guiding (VSPG, Cycles) | thin-fog scene equal time |
| V4.4 | Equiangular in grid media (Cycles' scheme) | `vol-cornell-smoke` grid twin |
| V4.5 | Height fog (exponential in y, closed-form optical depth and exact distance sampling, no majorant) | closed-form T and single-scatter quadrature |
| V4.6 | Emissive media (Le or temperature grid + blackbody) | closed form Le (1 - exp(-sigma_a d)) |

### ⬜ V5 — Real time and preview with media

Gate: `vol-cornell-smoke` and a cloud scene through the real-time chain (SVGF / OIDN / upscaler) with the plane identity exact (1e-6) and a preview that reaches the
reference mean within 1% in under 2 s on the M5.

| id | task | check |
|---|---|---|
| V5.1 | Lift `frame` / `aovs` refusal for media: medium primary-vertex semantics (albedo sigma_s / sigma_t, normal 0 / -wo, viewZ and motion from the first scatter), sky x T in the emission plane | `realtime-check` plane identity on `vol-cornell-smoke` |
| V5.2 ★ | `volumes.outputs` on the wavefront; `ptwfAutoPolicy` for media after the quiet-pass A/B (inline refill read 0.72x per spp, provisional) | `wf-p5-check` outputs per pixel |
| V5.3 | `preview` media preset (ray-marched shadows N = 1, lower medium depth, hardware filtering), A/B vs reference mode | mean within 1% on the V0 scenes |
| V5.4 | ReSTIR DI on surfaces behind homogeneous media (final shadow ray through `ptVolTr`) | `restir-check` bias scan on a fog scene |
| V5.5 | Research: volumetric ReSTIR; NRC at medium vertices (NRC track) | design notes first |

### ⬜ V6 — Web parity and demos

Gate: three examples in the catalog (cloud, Sponza god rays, lamp fog) that converge in seconds with OIDN on the M5 and run on the 3060 and an Intel iGPU; glTF volume
materials render as media.

| id | task | check |
|---|---|---|
| V6.1 | glTF `KHR_materials_volume` attenuation as an absorbing medium in the volume path (and the `volume_scatter` proposal as a scattering one) | the Beer-slab closed form in `openpbr-check` |
| V6.2 | Medium stack (nested / overlapping media, camera inside any medium; three-gpu-pathtracer has one) | nested-sphere closed form (absorbers) |
| ✅ V6.3 | Cloud example (procedural 256^3 grid, no download, progressive + OIDN) | examples smoke |
| V6.4 | Sponza thin-fog god-ray example (needs V4.2) and lamp-fog example (needs V4.1) | examples smoke |
| V6.6 | API ergonomics found by the cloud example (V6.3): sigma_t / albedo / phase as runtime uniforms (sliders without a recompile, today every change rebuilds the kernel), a `ptMediumFrom` overload taking a phase, `volumeGrid` building its own brick majorants when `majorant: 'brick'`, `rrStartDepth` exposed in `ProgressiveRenderer` | same image before / after; a slider change takes < 1 frame |
| V6.5 | Adapter-aware defaults (grid format / size from limits, wavefront when the megakernel passes the DXC class) | caps page on the three machines |

## Suggested order

V1.1, V1.2, V1.5, V1.6 first (S, minutes to check, direct win on the clouds' grain), then V2.1 + V2.2 + V2.3 as one A/B family, then V3.2. V1.8 and the quiet-pass items run
when the Windows machine / a quiet Mac are free. V4-V6 overlap with V2-V3 as agents free up.
