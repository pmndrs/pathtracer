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

**Closed out 2026-11-25.** V1-V6 are merged with main; the remaining work (V1.8, V5.4b, V4.3c, V1.7 NEE offsets, the brick / roulette / preview
default decisions, V4.4 and wavefront follow-ups, several grids, the V6 examples) is tracked as GitHub issues #9-#23 (list in TASKS `V.close`).

Status: ⬜ not started · 🟨 in progress · ✅ done. ★ = the clouds handoff needs it or it attacks its grain / memory problem.

### ✅ V0 — Validation and outputs (P5.v13-v19, 2026-11-21)

Cloud regime at the pbrt / Mitsuba floor (27 cells + a 256^3 cloud), droplet phases, volume outputs (order split, transmittance, standard error), r16float grids, ground
scenes, per-hit origin offset bound + free flight from the true crossing. Open from it: Cycles comparison of the Mie / Draine phases (waits on the clouds side's script).

### 🟨 V1 — Housekeeping and cheap speed (all S; only V1.8 left: Windows / 3060)

Gate: `vol-cl-cloud256` and `vol-cl-sph-g85-a99-t200` at 256 spp keep their mean within 2 standard errors of today's, with fewer density lookups per shadow ray (counter); quiet-pass ms/spp queued.

| id | task | check (minutes) |
|---|---|---|
| ✅ V1.1 ★ | Roulette in ratio tracking (T < 0.05, survive 0.75, weight 1/0.75), shadow rays and the outputs' transmittance, both schedulers | TS twin mean at 1e5 trials vs line integral; the two scenes at 256 spp within the SE; lookup counter |
| ✅ V1.2 ★ | Hardware-filtered density (`textureSampleLevel` on r16float / r8unorm, r32float with `float32-filterable`) as an option, manual 8-tap stays the default until the A/B | `vol-grid-slab` closed form; same-seed relMSE vs manual on `vol-cl-cloud256` (~1e-5); bytes saved |
| ✅ V1.3 ★ | Separate `maxVolumeDepth` (medium scatters vs surface bounces) | equal to today with both caps equal; `vol-cl-ground-sph` with surface 8 / medium 2048 |
| ✅ V1.4 | Loop caps visible: count truncations (4096 collisions, brick cell cap, ratio-tracking cap); warn at upload when densityMax x sigma_t x diagonal > ~1e3 | synthetic dense slab near the cap |
| ✅ V1.5 | Doc fixes: volumes.md sections 3 / 6.1 and DIVERGES V243 still describe the global majorant only | read-through |
| ✅ V1.6 ★ | Per-pixel wavefront check of lobe scenes | `wf-p5-check cases=vol-cl-single-mie10,vol-cl-sph-hg2-a99-t50 spp=16` |
| ✅ V1.7 (scene-wide bound for ReSTIR / 'ray' positions; interface skips need nothing; NEE light ends: maintainer's call) | Offset-bound follow-ups: ReSTIR G-buffer positions, light-sample end points, medium interface skips, `positions: 'ray'` | `vol-cl-ground-occluder` with each path on |
| V1.8 ★ | Windows DXC compile of the 50-54 KB volume megakernels (and the wavefront split as fallback) | `kernel-compile` page on the 3060, cold / warm: handed off as [WINDOWS-HANDOFF W18](../WINDOWS-HANDOFF.md#w18-volume-kernels-on-dxc-v18) |

### ✅ V2 — Noise per time (all built, 2026-11-24; gate = quiet pass)

Gate: equal-time relMSE on `vol-cl-cloud256` (and one dense homogeneous sphere) at least 2x better than V1, mean within 2 SE (provisional until the quiet pass).

| id | task | check |
|---|---|---|
| ✅ V2.1 ★ | Minorant per brick (brick texture rg = min, max) | TS twin: min <= every density sample in the cell |
| ✅ V2.2 (not a win) ★ | Residual ratio tracking for shadow rays against the minorant (verify the residual-tracking patent status first) | TS twin unbiased; `vol-grid-slab`; 256 spp within SE |
| ✅ V2.3 (not a win) ★ | Misso 2022 telescoping transmittance, port of Cycles `volume_transmittance` (Apache-2.0, attribution), A/B vs ratio tracking | TS twin unbiased; equal-spp variance on `vol-cl-cloud256` |
| ✅ V2.4 (opt-in: lookups x0.59-0.88, noise-neutral, ~8% faster provisional; patent US10269172B2 open) | Decomposition tracking for free flight (control part analytic, residual majorant) | `vol-grid-slab`; lookup counter |
| ✅ V2.5 ★ (closed, opt-in: allocated sampling is unbiased and saves 15-36% of samples, but loses 1.75-2.4x wall time today; stop-on-estimate = biased preview; tile retirement open) | Adaptive sampling per tile from the second moment (stop when SE < target) | spp saved at equal SE on `vol-cl-cloud256` |
| ✅ V2.6 | Finite sun disc A/B against the delta sun for peaked phases (agree with the clouds side first) | `vol-cl-sph-mie20` 64 spp firefly count |

### ✅ V3 — Memory and scale (2026-11-24)

Gate: the clouds' 504x352x616 grid fits in <= 128 MB with an image same-seed equal (relMSE <= 1e-5) to the dense r32 one.

| id | task | check |
|---|---|---|
| ✅ V3.1 ★ | r8unorm grids with a per-medium scale | same-seed relMSE vs r16 |
| ✅ V3.2 ★ | Sparse brick atlas: indirection texture at brick resolution + atlas of occupied 8^3 bricks with a 1-voxel apron (one texture, no storage buffer); gives several grids per kernel | same-seed image vs dense on `vol-cl-cloud256`; memory printout |
| ✅ V3.3 | Camera inside a grid medium; auto-detect the camera medium | `vol-camera-medium` grid twin |
| ✅ V3.4 | Browser `.vdb` loader (WebVDB `parseVdb` in a worker -> dense / atlas; licence check of WebVDB / NanoVDB first) | `vol-vdb-smoke` from the browser = from Node |

### ✅ V4 — Lights in media and outdoor fog (all built, 2026-11-24; gate = quiet pass)

Gate: lamp-lit fog and thin-fog god rays at least 3x better equal-time relMSE than V1 (own 4096 spp reference, once).

| id | task | check |
|---|---|---|
| ✅ V4.1 | Equiangular + distance MIS for emissive triangles in homogeneous media | new `vol-lamp-fog` (maxDepth 1) vs a 1D quadrature |
| ✅ V4.2 | Forced in-scatter NEE for thin homogeneous media | thin-fog closed form (single scatter) |
| ✅ V4.3 (closed, off by default: 2.3-4.2x on thin / lamp fog, none or a loss on thick / grid media; height and brick refused; thin-fog-sun z -2.45 flag open) | Volume scattering probability guiding (VSPG, Cycles) | thin-fog scene equal time |
| ✅ V4.4 (off by default: `forcedGrid`, closed-form proposals + ratio-tracked T; 13x lower pixel variance on `vol-lamp-fog-grid`; megakernel only) | Equiangular in grid media (Cycles' scheme) | `vol-cornell-smoke` grid twin |
| ✅ V4.5 | Height fog (exponential in y, closed-form optical depth and exact distance sampling, no majorant) | closed-form T and single-scatter quadrature |
| ✅ V4.6 | Emissive media (Le or temperature grid + blackbody) | closed form Le (1 - exp(-sigma_a d)) |

### ✅ V5 — Real time and preview with media (2026-11-24; V5.4b frame-mode check and the V5.3 default open)

Gate: `vol-cornell-smoke` and a cloud scene through the real-time chain (SVGF / OIDN / upscaler) with the plane identity exact (1e-6) and a preview that reaches the
reference mean within 1% in under 2 s on the M5.

| id | task | check |
|---|---|---|
| ✅ V5.1 | (V5.1a ✅ aovs + OIDN megakernel, V5.1b ✅ wavefront, V5.1c ✅ frame + media megakernel, V5.1d ✅ wavefront frame + media; the upscaler bias is the visible-lamp trap, rest of image -0.8%; residual: `vol-cloud` 2.2e-5 on 5 px, upscaler control +2.7% unsplit) Lift `frame` / `aovs` refusal for media: medium primary-vertex semantics (albedo sigma_s / sigma_t, normal 0 / -wo, viewZ and motion from the first scatter), sky x T in the emission plane | `realtime-check` plane identity on `vol-cornell-smoke` |
| ✅ V5.2 ★ (plain scheduler) | `volumes.outputs` on the wavefront; `ptwfAutoPolicy` for media after the quiet-pass A/B (inline refill read 0.72x per spp, provisional) | `wf-p5-check` outputs per pixel |
| ✅ V5.3 (maintainer to confirm) | `preview` media preset (ray-marched shadows N = 1, lower medium depth, hardware filtering), A/B vs reference mode | mean within 1% on the V0 scenes |
| ✅ V5.4 (homogeneous media, megakernel; media add no bias; frame-mode GPU check and a pre-existing 128^2 ReSTIR offset open) | ReSTIR DI on surfaces behind homogeneous media (final shadow ray through `ptVolTr`) | `restir-check` bias scan on a fog scene |
| ✅ V5.5 (notes: volumes-restir-nrc.md) | Research: volumetric ReSTIR; NRC at medium vertices (NRC track) | design notes first |

### 🟨 V6 — Web parity and demos (open: the Sponza god-ray example of V6.4; 3060 / iGPU runs)

Gate: three examples in the catalog (cloud, Sponza god rays, lamp fog) that converge in seconds with OIDN on the M5 and run on the 3060 and an Intel iGPU; glTF volume
materials render as media.

| id | task | check |
|---|---|---|
| ✅ V6.1 | glTF `KHR_materials_volume` attenuation as an absorbing medium in the volume path (and the `volume_scatter` proposal as a scattering one) | the Beer-slab closed form in `openpbr-check` |
| ✅ V6.2 | Medium stack (nested / overlapping media, camera inside any medium; three-gpu-pathtracer has one) | nested-sphere closed form (absorbers) |
| ✅ V6.3 | Cloud example (procedural 256^3 grid, no download, progressive + OIDN) | examples smoke |
| ✅ V6.4 (fog example; Sponza version open) | (2026-11-22: `fog` example rebuilt with a plain | forced + equiangular split view: plain 3x noisier per pixel in a beam at the same spp; the first version's camera started in vacuum inside the fog box (no `cameraMedium`). Earlier note: `fog` example built — room with slits, sun + lamp, plain / forced / equiangular selector; shafts only moderately visible, estimator difference not yet convincing on screen: tune narrower beams, darker interior, thinner fog, and add a same-seed split view) Sponza thin-fog god-ray example (needs V4.2) and lamp-fog example (needs V4.1) | examples smoke |
| ✅ V6.6 (runtime params, phase overload, auto brick majorants, `rrStartDepth`, fog sliders; run-time Mie / HG switch left as an opt-in follow-up) | API ergonomics found by the cloud example (V6.3): sigma_t / albedo / phase as runtime uniforms (sliders without a recompile, today every change rebuilds the kernel), a `ptMediumFrom` overload taking a phase, `volumeGrid` building its own brick majorants when `majorant: 'brick'`, `rrStartDepth` exposed in `ProgressiveRenderer` | same image before / after; a slider change takes < 1 frame |
| ✅ V6.5 | Adapter-aware defaults (grid format / size from limits, wavefront when the megakernel passes the DXC class) | caps page on the three machines |

## V7 Two-part media estimator

Idea: a raymarcher looks finished in a frame because it takes several light estimates along the view ray (single scattering, deterministic); we keep the unbiased path tracer for multiple scattering and replace only the single-scatter-from-camera part.

| Item | What | Status |
|---|---|---|
| V7.1 | Homogeneous media: `volumes.marchSamples` N stratified distances on the camera segment (see forced-wgsl.ts header and DIVERGES V330). Megakernel only, implies `forcedNee`, composes with `equiangular`. Unbiased on `vol-cornell-fog` (within 1 SE at 1024 spp); error vs pbrt: x0.55-0.62 of the default at 1-64 spp from forced + equiangular, N adds a further gain only where one light sample per segment is the noise (lamp close to the ray: 6-18x at 1 spp, floor from 16 spp up) | done (2026-11-24) |
| V7.2 | Grid media: the same N-strata estimate through the majorant / ratio-tracking transmittance (a grid segment has no closed-form exponential CDF: needs a distance proposal, e.g. equal-optical-depth strata from the brick majorants) | on hold: no headroom (below) |
| V7.3 | Split denoising: filter the orders-2+ (multiple scattering) radiance on its own and add the orders 0-1 back (needs `outputs` to accept `forcedNee` / `equiangular`: the forced estimate belongs to order 1), then the real denoiser (OIDN with guides) on that plane instead of a blur | step 1 done (outputs + forced / equi / march, order-0-1 plane bit-exact vs `maxDepth` 1); step 2 measured (2026-11-24): OIDN on the MS plane beats OIDN on the beauty only for the soft cloud at >= 16 spp, loses on Cornell fog at <= 16 spp, so it is NOT a general default; guides and the real-time chain open (`bench/results/SPLIT-DENOISE.md`) |

**Where the noise is (2026-11-24, `pt_volume_outputs=1`, relMSE against our own 1024 spp render).** At 16 spp the orders 0-1 plane carries 0% of the error on
`vol-vdb-wdas-cloud` (65% of the radiance) and 4% on `vol-cornell-fog` (57% of the radiance): multiple scattering is the noise in both, so improving single
scattering (V7.1 N > 1, V7.2) cannot help dense clouds or room fog; it helps only single-scatter-dominated scenes (thin fog under a sun, a lamp close to the ray).
Crude test of V7.3 (a Gaussian blur, sigma in pixels, stands in for a denoiser): Cornell fog 4 spp raw 0.556, blur everything 0.070 at best (0.37 at sigma 4:
edges smear), blur only orders 2+ 0.027 (sigma 4); 16 spp 0.131 -> 0.010. WDAS cloud (soft everywhere at 128 px) 4 spp 0.351 -> 0.0145 split, 0.0092 blur all;
16 spp 0.089 -> 0.0090 split. A filter is biased, so it is a real-time / preview tool and fades out as samples accumulate; the converged image stays the path tracer's.

**V7.3 measured with OIDN (2026-11-24, `bench/results/SPLIT-DENOISE.md`, vs our own 8192 spp render, seeds 1-2, 128x128).** relMSE raw -> `oidn_all` -> `split_oidn`:
Cornell fog (forced + equiangular) 1 spp 1.01 -> 9.5e-3 -> 7.3e-2, 4 spp 0.31 -> 3.2e-3 -> 1.2e-2, 16 spp 8.7e-2 -> 2.2e-3 -> 3.3e-3, 64 spp 1.9e-2 -> 9.4e-4 -> 1.0e-3, 256 spp 4.9e-3 -> 5.2e-4 -> 4.6e-4, 1024 spp 1.35e-3 -> 3.3e-4 -> 2.8e-4;
WDAS cloud 1 spp 1.81 -> 5.9e-3 -> 1.2e-2, 4 spp 0.32 -> 5.1e-3 -> 5.1e-3, 16 spp 8.1e-2 -> 2.9e-3 -> 2.5e-3, 64 spp 2.1e-2 -> 4.0e-3 -> 1.5e-3, 256 spp 5.3e-3 -> 2.3e-3 -> 8.6e-4, 1024 spp 1.4e-3 -> 5.5e-4 -> 5.0e-4.
The crude blur test above is confirmed for Gaussians (split_gauss beats blur_all on Cornell at 4-16 spp), but OIDN's own network on the whole beauty uses the structure of the noisy colour and is the better filter on Cornell at <= 16 spp:
the MS plane alone is sparse and structure-poor at 1-4 spp. Split pays on the soft cloud from 16 spp up (lower relMSE, edge error and bias: OIDN darkens `oidn_all` by 1.2-2.6% at <= 64 spp, split by 0.65-1.1% from 4 spp). Using the order-0-1 plane as OIDN's albedo guide does not help.
No crossover to raw within 1-1024 spp (the denoised arms are at the 8192 spp reference's own noise floor, ~1.7e-4, at 1024 spp); extrapolating the gain (x2-2.5 less per x4 spp) puts it at roughly 8k-16k spp, so the fade is by spp (w = 1 up to 256, linear in log2(spp) to 0 at 4096), not by the pixel's standard error: a per-pixel rho = se / (mean + 0.05) fade only gives back error (every rho0 tried was worse than w = 1 at every spp) and the standard error needs >= 4 samples anyway.

## Suggested order

V1.1, V1.2, V1.5, V1.6 first (S, minutes to check, direct win on the clouds' grain), then V2.1 + V2.2 + V2.3 as one A/B family, then V3.2. V1.8 and the quiet-pass items run
when the Windows machine / a quiet Mac are free. V4-V6 overlap with V2-V3 as agents free up.
