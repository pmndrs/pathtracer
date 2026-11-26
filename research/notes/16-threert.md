# 12 - ThreeRT (Meshy): hardware ray-traced three.js, audited

Date: 2026-11-14. Clone: `research/sources/web/ThreeRT/` (fast-forwarded to `8964dc4`, 2026-11-14, "kernel variants by WESL
conditional translation"; git-ignored, read only, never built). Three parallel audit passes (core, real-time + browser, bench +
process), spot-checked by the lead. Tags: **[V]** verified in source/docs this pass, **[A]** the authors' own numbers (not
reproduced by us).

## 0. Clean-room status (read this first)

**Update 2026-11-26 (resolved):** the authors answered on [meshy-dev/ThreeRT#78](https://github.com/meshy-dev/ThreeRT/issues/78) (closed 2026-11-24): their session records show no NRD read, but the three ReBLUR-named changes (#33, #39, #55) were written where no records were kept, so they treated them as unverified and replaced them; the same review found code written from Unreal Engine source (the post chain, part of the specular temporal pass, the specular energy terms, part of the atmosphere, a fog density), also replaced. The replacement (#268, `5cb95f6`) was written by fresh sessions in a history-free tree from a behaviour spec, black-box measurements and cited public sources (their `docs/PROVENANCE.md`). They ask clean rooms to exclude everything before `5cb95f6`. So the fence is now all of ThreeRT before `5cb95f6` (docs/internal/CLEAN-ROOM.md §4, D35). Everything in this note comes from our clone at `8964dc4`, which predates it: treat it as measured results and public concepts only, and take any design for the replaced areas (post chain, specular temporal and energy terms, atmosphere, fog) from the public papers, not from this note. A new read of ThreeRT needs a clone at `5cb95f6` or later and a scan first. The 2026-11-14 text below is kept as the record.

**What the evidence shows** (citation-only provenance scan, 2026-11-14: identifier and token matches, git history, licence
headers; the denoiser code itself was not opened):

- One pointer to NRD *source*: ROADMAP §1a says NRD's ReBLUR bounds the specular history by parallax, followed by "(its source,
  `REBLUR_TemporalAccumulation`)". It was added in commit `fa8cb12` (#39, 2026-11-13, "The specular history per pixel: bounded
  by each reflection's own parallax"), which also touched `live.wgsl`. `REBLUR_TemporalAccumulation` follows NRD's shader naming,
  so the author at least looked at NRD's repository for that pass [V].
- About eight other mentions name ReBLUR *concepts* ("ReBLUR's fast history", "Anti-lag (as ReBLUR's)", "ReBLUR's form", "NRD's
  default is 30" frames). Fast history, anti-lag and parallax-driven accumulation are all described in public sources (Ray
  Tracing Gems II ch. 49, GTC 2020 talk; see `~/Developer/Denoiser/docs/research/rt-denoiser/sources/reblur.md`), and the frame
  count is an NRD settings default. These mentions prove nothing about source use [V].
- No NVIDIA copyright header, no NRD licence file, no HLSL and no NRD identifier anywhere except that one citation. The code is
  WGSL in ThreeRT's own style. The "All rights reserved" hits are BSD headers in vendored parry3d [V].
- So "written from NRD's source" (this note's first draft) was too strong. Correct statement: **one pass (the specular history
  bound, #39) was designed with NRD's source consulted; whether any code was translated is unknown, and we will not open it to
  find out.**

**Licence.** NRD ships under the NVIDIA RTX SDKs License (Denoiser `sources/nrd-provenance.md` §1.1): install and use, object-code
distribution inside an application, every source file reserves all rights, and §4(e) bans anything that would subject the SDK to
an open-source licence. A translation of NRD shader code into an MIT repo would therefore not be validly MIT. Reading the code
and writing an independent implementation of its ideas is a different matter (copyright covers expression, not ideas), and is
probably fine for them. We can't tell which happened without reading their code. Not legal advice.

**Our rule is stricter than copyright.** The Denoiser clean room bars reading NRD code, ports of it, and summaries of its logic. A
pass designed with NRD's source open is close enough that we fence it. The fence costs little, since we don't need their denoiser.
Until the provenance is clear, nobody on this project or its siblings opens:

- `renderer/shaders/live.wgsl` (all of it), `renderer/src/denoise.rs`;
- `renderer/src/live/*` beyond pass names, scheduling and binding layout;
- `docs/ROADMAP.md` §2 "Temporal fixes" and the denoiser paragraphs inside §1 / §1a;
- the `override LIVE` signal-splitting tail of `renderer/shaders/pathtrace.wgsl` `main` (S:3276 on).

The audits obeyed this: nothing below describes how their denoiser works, only pass names and measured results. The rest of
the repo (offline path tracer, BVH, light tree, browser port, bench, API) is MIT and usable with attribution. To lift or narrow
the fence, ask the authors (an issue on meshy-dev/ThreeRT) whether any NRD code was translated. That is the maintainer's call,
since it is public.

## 1. TL;DR

- **What it is.** A Rust + wgpu 30 engine behind a three.js-style TypeScript API (`@threert/scene`, its own reimplementation of
  three's class names, not a wrapper). Native clients (macOS, iOS, Steam Deck, Android) trace with **hardware ray queries**
  (Metal, Vulkan; `EXPERIMENTAL_RAY_QUERY`). A game runtime, not just a renderer. MIT, Copyright 2026 Meshy LLC; the store
  clients are closed source [V].
- **Browser:** new this week (#68/#69): the engine compiles to wasm32 and runs its **live** path on WebGPU in Chrome, always on
  its **software BVH** (`merged = cfg!(wasm32)` forces `hwrt = false`, `gpu.rs:1354,1363`) [V]. Only hello-world-class scenes
  were measured in Chrome (6.6 ms GPU, trace 2.1 ms, M5 Pro) [A]. The offline path tracer, OIDN and image compare are not built
  for wasm. "Runs in any browser" in the README means a plain three.js raster fallback, not their path tracer.
- **Headline:** Bistro night 1280x720, ThreeRT live 15.5 ms GPU at render scale 0.55, ΔE76 3.4 vs their own 4096 spp offline
  render; full resolution 40.3 ms, ΔE 3.1; three.js WebGPU + SSGI/SSR/TRAA 34.7 ms, ΔE 10.1 [A, `docs/COMPARE.md`]. Measured on
  an Apple **M5** (the README reel says M3 Max). Live luminance reads 0.916 of the reference, i.e. ~8% dark.
- **It is UE5-matched, not physics-matched.** The BSDF is a UE5 DefaultLit port, defaults clamp indirect light, drop
  multiple-scattering energy and skip NEE on the last vertex. Their "unbiased reference" is their own renderer with clamps off;
  UE 5.8's path tracer is only a milestone audit; no pbrt or Mitsuba anywhere. So their ΔE numbers are not comparable to our
  relMSE/FLIP against pbrt/Mitsuba.
- **For us:** not a reference renderer; a **native-only, internal competitor row** and a rich source of WebGPU engineering
  lessons (8-storage-buffer layouts, loop guards, WGSL zero-init stacks, two-level software BVH for edits, light tree, browser
  compile costs, dynamic resolution).

## 2. Offline path tracer (pathtrace.wgsl, LIVE = false)

| Area | ThreeRT | Ours (plan / state) |
|---|---|---|
| Scheduler | One megakernel `main`, 8x8, one path/pixel/dispatch; 4 dispatches then `poll(wait)` per batch (gpu.rs:3081) | Megakernel v1 done (P2.4); wavefront vs megakernel A/B in P3 |
| Traversal | HW ray query (8 BLAS groups: opaque/cutout x caster/non-caster x static/dynamic, one TLAS, masks for shadow casters). SW fallback: BVH2, 32 B nodes, CPU binned SAH (16 bins, leaves 4..16), stack 48 (overflow silently dropped), Möller-Trumbore | `@pmndrs/bvh` BVH2 (+ BVH4/CWBVH on a branch), watertight edge test, H-PLOC |
| Sampler | PCG hash, white noise, 24-bit floats, box-filter jitter (S:209-219, 2923) | Owen-Sobol (reference), own blue noise (P0.17) |
| BSDF | UE5 DefaultLit: Lambert x (1-E_spec), GGX + UE Smith-joint-approx V, Schlick F with F90 = saturate(50 F0.g), clearcoat lobe; VNDF (Heitz 2018) sampling; pdf uses exact G1 while eval uses the approximation; rough glass = jittered delta lobe, no BTDF weight. No sheen/aniso/thin/SSS | pbrt-v4 ports, exact pdf/eval pairs, OpenPBR 1.1.2 subset target |
| Multi-scatter | UE-style `1 + F0(1-E)/E` from a 32x32 LUT; preset `energyCompensation` 0 (UE parity) / 1 (passes furnace) | Turquin LUT option, off in reference mode (P2.5) |
| Lights | lux (sun), candela (point/spot), nits (rect, emissive, sky); UE range window on point lights; Hillaire 2020 sky baked to equirect | radiance units in the scene copy |
| Light picking | Locals in spatial groups of 16 with bounding spheres (all groups at camera hits, RIS 3-8 candidates elsewhere); emitter-triangle CDF with a 65,536-bucket guide table; **light tree** (≥256 static emitters, median split, sphere bounds, power x cos-bound / max(d², r²), no orientation cones) + a refit subtree for moving emitters; MIS pmf recomputed by walking the same tree path | Vose alias over power (P1.4), A/B pending (P1.4f) |
| MIS | Power heuristic everywhere; env importance sampling (marginal/conditional CDF) | NEE + MIS (balance/power to decide) |
| Path control | maxBounces 8 (cinematic) / 3 (preview); RR from depth 3 with q = min(max(thr), 0.95); origin offset 1e-4 x max(1, max\|p\|) along Ng; non-finite zeroed with integer bit tests (Metal fast-math folds float NaN tests, S:3270) | pbrt RR, robust offsets (P1.8) |
| Volumes | Delta tracking + ratio tracking, up to 32 media, analytic height fog; single scattering on camera rays at half the tiles | P5 |

**Deliberate biases in the defaults** (override all of these to compare against pbrt/Mitsuba): `clampIndirect` 64 x 2^EV100
(luminance clamp, hue kept) on NEE at depth>0 and emission/sky hits at depth>1 (lib.rs:707); `energyCompensation` 0; no NEE at
the last vertex unless the `last_direct` switch (on only for Android); roughness floor 0.02; normal maps only at camera hits and
mirror-like paths; texture filtering before sRGB decode; shadow rays through glass go straight (no caustics); dim-sky env NEE
skipped (<1% of emitter radiance, still correct via BSDF MIS); live-only: cutouts opaque for indirect rays, a radiance cache,
light LOD (authors measured 0.3-1.7% darker).

**Kernel variants.** One WESL module (4 files); `@if(flag)` conditional translation (hwrt, units, vcolor, grid, tint, unlit,
background + device switches), the rest a pipeline-overridable `FEAT` bitmask; scene material functions spliced in as WGSL text.
A naga test validates every variant (gpu.rs:3196); 2,304 combinations proved identical to the old text patching. Source is
164 KB WGSL (heavily commented); generated size per variant not measured. Disk pipeline cache on Vulkan only (SHA-sealed).

**Oracles.** CPU twins of the BVH and two-level walks whose tests grep the WGSL for the walk's statements in order, so a kernel
edit forces the twin to follow (bvh.rs `the_walk_is_the_kernels_loop`, units.rs `the_cpu_kernel_is_the_shaders`); brute force vs
walk under random edits; light-tree invariants; ~190 GPU render tests (§5).

## 3. Real-time frame (structure and numbers only, see §0)

- Megakernel trace (primary, direct with NEE/RIS, GI through a world-space radiance cache of 2^18 entries, reflection rays) at
  render scale 0.375-0.6, 1 spp, 4 bounces (native window drops to 2 at the floor scale over 92% budget); Halton(2,3) jitter.
- ~40 compute passes after it, each with its **own bind-group layout derived from the shader by naga at build time**: reflection
  resolve, diffuse temporal + 5 spatial, specular temporal + 3 spatial, compose, their **own temporal upscaler** (`judge`,
  `reject`, `upscale`; modelled on UE TSR/FSR2, not FSR/MetalFX), DOF, motion blur, auto exposure, bloom, UE filmic tonemap LUT.
  All state lives in storage buffers, not textures.
- Numbers [A, M5 Pro, 2560x1440 output]: Bistro traced 960x540 at 14.4 ms GPU; Kabukichō 736x414 at 14.4 ms; output-grid passes
  2.8 ms (was 9.9 ms before the one-upscaler redesign). Trace cost vs bounces (Kabukichō, scale 0.6): 7.4 / 12.1 / 14.3 ms at
  1 / 2 / 4 bounces. A raster visibility pass for primaries measured -3..-5% and was left behind a flag.
- "Code that never runs" in the megakernel cost +1.6 ms; inlined direct light beat a separate pass on a big scene (latency
  overlap with traversal) [A].
- Dynamic resolution governor (live/mod.rs:722): budget 90% of the frame, next scale = scale x sqrt(budget/gpu), hysteresis,
  urgent path after two frames >1.25x budget, ignore frames rendered at the old scale and the first 2 s; floor scales with display
  density.
- GPU timing without stalls: 4-slot ring of timestamp queries read back several frames late by async map; slots that straddle a
  frame gap are rejected (Metal writes pass timestamps in pairs; a misread made 70-80 ms phantom spans).

## 4. Browser port (the part closest to us)

- wasm32 + wgpu 30 + wasm-bindgen; module ~7.3 MiB (1.5 MiB brotli). Game TS runs in the page's JS, the engine (world, Rapier
  physics, voxels, live view) in wasm with 35 exports; JS↔wasm scalar call ~40 ns, JSON descriptor ~1 µs [A, Chrome 157].
- **8 storage buffers per stage.** Chrome offered 10; they request 8 (the default) [V `merged.rs:29`]. Two answers:
  - *Megakernel:* `merged.rs`, a WGSL token rewriter that folds 7 read-only tables into one struct of power-of-two arrays (the
    kernel recompiles only when a table doubles) and puts 6 writable buffers in one buffer at `@align(256)` ranges.
  - *Live passes:* restructured so every pass binds ≤8; unit test `every_pass_fits_a_browser` (live/layout.rs:244) [V]; cost
    +0.18 ms on Bistro 1440p; the merged layout for passes was deleted.
- **Loop guard.** A hung dispatch took down the macOS WindowServer. Every kernel loop now counts iterations and exits past 65,536,
  with an async-read trip counter in the frame log; on by default in browsers. Cost: +1.2 ms trace (11%) on Kabukichō natively;
  354 of 43,200 floats differ by ≤2e-7. They also CPU-validate every BVH tree before upload.
- **WGSL zero-init trap.** Function-local arrays are zero-initialised on every call; one module-private 64-entry stack shared by
  the walks gave 0.36-1.5 ms back per frame [A, trace_units.wgsl header].
- **Two-level software BVH ("units")** for dynamic scenes: per-mesh model-space trees, a top tree over world boxes, settled units
  refitted and loose units rebuilt, one loop walks both levels (no per-lane stall at the boundary). A moving body went from
  16.5 ms to 0.84 ms of CPU per frame [A].
- Compile: first Chrome compile of the kernel ~1.4 s cold, 26 ms warm (Chrome caches across profiles, so cold runs are hard to
  reproduce). Tint is stricter than naga (`a * b ^ c * d` passed naga, failed Chrome).
- Missing vs native: particles, translucency, OIDN, offline render; 23-24 of ~190 render tests fail under software traversal.
  No Bistro/Kabukichō in a browser yet; no Safari/Firefox; no public demo URL of the wasm build.

## 5. Bench, tests and process

- **Reference:** their own offline render, clamps off, no denoiser, 4096 spp (16k for estimator studies), cached by content hash
  + `TRUTH_VERSION`. They print the reference's own noise as a row (Bistro 1024 spp: ΔE 1.8) [A]. `tools/unbiased-check.ts`
  compares hardware vs software traversal at 16,384 spp (region means): a self-consistency check that caught a real 3.5% bug.
- **Metric:** mean CIE76 ΔE on 8-bit sRGB PNGs after each renderer's own tonemap (compare.rs), plus PSNR, SSIM, p95/p99 ΔE,
  regional ΔE and `trail_px` (p95 over rows of the longest run of ΔE>10, a ghosting measure). HDR compare is luminance ratios
  only: no relMSE, FLIP or firefly counts. No time-to-equal-error.
- **Timing:** GPU timestamps, a GPU lock file, a quiet-GPU check via `ioreg` <15% busy before and after (refuses otherwise),
  quoted repeatability (±0.03-0.05 ms), A/B pairs from one run, first-frame times confirmed by alternating launches.
- **Render test suite** (`tests/render`, ~190): one TS module per test with `{id, features, reference: analytic|converged|ue,
  metric, tolerance, spp}`; 143 analytic, 11 converged, 5 UE; region-mean radiance or region ratios at 160x90, 64 spp, denoiser
  off, linear f32 in nits; tolerance mostly 0.02. One warm `serve` process, ~1 minute. Not in CI (no GPU); merge gate runs it on
  a Mac, hardware traversal only.
- **Process:** `docs/WORKFLOW.md` is an author/supervisor agent protocol with tiered merge gates (docs / tools / feature-gated
  renderer / changes scenes / hot path), scheduled perf suite every ~3 h with automatic bisect, "300 ms rule" for bench loop
  sizing, ROADMAP as a dated decision log with "Known issues" (cause, fix, not done). One Claude skill for scene authoring.

## 6. Running it in our bench (adapter plan, P0.32)

- Native only (macOS, Metal); our headless-Chrome runner cannot host it. Build in `.cache/native/threert/` (never in the audit
  clone); needs Rust 1.100, Node 24, Bun.
- Drive `threert-render serve` (JSON lines on stdin, keeps the GPU and BLAS warm) or `render <bundle.js> --spp N --no-denoise
  --hdr out.raw --stats out.json --camera '{...}' --override '{...}'`. `--hdr` is headerless linear f32 RGB, unexposed: wrap to
  PFM ourselves (row order to check).
- Scenes are JS bundles of their DSL: generate a tiny TS wrapper per scene (`new GLTFModel(path)` + camera + lights) and bundle
  with `tools/bundle-game.ts`. Camera fov is vertical degrees; units map 1:1 radiance → nits (their 1-nit furnace reads 1.0);
  verify on quad-light (closed form) before trusting any scene.
- Overrides for a fair, unbiased row: `clampIndirect` off, `energyCompensation` 1, `last_direct` on, `maxBounces` = our maxDepth,
  RR depth to match. Even so the UE DefaultLit BSDF, UE rect-light convention and straight glass shadows differ from pbrt: report
  it in a separate "inexact" report (D20), never the headline matrix, and internal only (D8).
- Timing: their `--stats` GPU fields, not wall clock (wall clock includes bundling and BLAS builds). Hardware traversal is not
  byte-repeatable on 5 of their scenes.

## 7. Ideas worth A/B-ing here (each → TASKS)

1. Shared module-private traversal stack vs function-local array (WGSL zero-init cost) — bvh's walk and our kernels.
2. Per-entry-point "≤8 storage buffers" unit test for every `rt*`/`pt*` kernel (megakernel today binds how many?).
3. Loop guard + async trip counter + CPU validation of tree data before upload, A/B its cost.
4. Light tree vs power alias (P1.4f) on a many-light scene; guide-table CDF vs alias; their tree-path MIS pmf is the trap to
   get right.
5. Non-finite detection by integer bit tests (Metal fast-math).
6. Bench: `unbiased-check`-style agreement between traversal back ends (CPU oracle vs GPU walk, BVH2 vs BVH4) at high spp by
   region means; `trail_px` ghosting metric and a reference-noise row for P4; render tests as `{features, analytic, region,
   tolerance}` modules with `unsupported` as a status.
7. P4: dynamic-resolution governor, one temporal upscaler at 0.375-0.6 scale (their biggest real-time win), timestamp ring
   with late async reads.
8. DX: three.js conformance snippets run unchanged with `compatNote(unsupported|mapped)` warnings; an `Engine` interface with
   a three.js backend for the same scene.

Not to copy: their default biases (§2), white-noise sampling, the LDR ΔE76 headline metric, a reference made by the renderer
under test.
