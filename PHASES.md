# Phases

Agreed 2026-11-13 (decisions D1–D21 in [DECISIONS.md](DECISIONS.md)); no longer under discussion. Each phase ends with a
number from the bench. Details and rationale: [spec/SPEC-draft.md](spec/SPEC-draft.md) §11. Task-level progress:
[TASKS.md](TASKS.md). Phases overlap: a phase starts as soon as what it needs exists, and its gate is checked when the
numbers are in.
Status: ⬜ not started · 🟨 in progress · ✅ done. _Last updated 2026-11-18._

## 🟨 P0 — Foundations: know the number to beat

- Monorepo scaffold, `detectCaps()`, shared contract types.
- Bench: GPU runner (headless Chrome), `Renderer` adapter interface, metrics, A/B viewer.
- Tier 0 scenes; exporters to pbrt-v4 and Mitsuba 3; cached references rendered on the Mac CPU.
- mutracer and three-gpu-pathtracer (WebGPU) running behind the adapter.
- Literature verification pass.
- `windows-testing` branch + Windows handoff doc.

**Exit gate:** one command prints time-to-equal-error for each competitor on Cornell and Sponza against pbrt-v4, on M5
(and on RTX 3060L once Windows runs it).

**Progress:** everything above is built (P0.1–P0.17, P0.19, P0.26, P0.30). M5 half of the gate passed 2026-11-14: the
quiet-machine table (P0.25, `bench/results/quiet-pass/20261114-1419/REPORT.md`). The 3060L half waits on the Windows
tasks (W1–W10).

## 🟨 P1 — Trace core for utility consumers (@pmndrs/raytracer)

L0 visibility, L1 closest-hit (intervals, ray cones), lazy `rtSurface`, L2 one-bounce shade; inline WGSL + batch
dispatcher; queue ABI; material/light scene copy. Reflections moves onto it.
**Exit gate:** reflections switches by deleting code; traced hits no longer darker than the path tracer; no
regression in reflection rays/s.
**Parallel (in bvh):** BVH4 / CWBVH8 vs BVH2. Gate ≥1.3× incoherent closest-hit on 3060L and M5.

**Progress:** built and checked: L0/L1/`rtSurface` and the batch tracer (P1.1, 590k test rays agree with bvh's CPU
walk), the scene copy with materials, lights and light sampling (P1.4), and L2 `rtShade` (P1.3). Reflections runs on the
core on a branch (P1.5: same hits, trace cost equal within the noise, `rtShade` hits match PT maxDepth 1; left: quiet
timing, the default hit shading P1.5b, the merge). Left: the trace settings A/B (P1.6, quiet pass), incremental scene updates, textures,
environment maps (P1.4b–e). Wide BVH (X1): both formats built and correct on a bvh branch. Quiet M5 numbers (2026-11-14) pass 1.3× on merged
trees per scene, BVH4 on Sponza (1.34–1.43×) and CWBVH8 on Bistro (1.39–1.46×), but no one format passes both; the
3060 runs are still to do.

## 🟨 P2 — Reference path tracer, megakernel (@pmndrs/pathtracer)

Owen-Sobol, OpenPBR-subset uber BSDF, NEE + MIS, area/env/sun lights, RR, robust offsets, vec4 radiance, OIDN final.
**Exit gate:** converges to pbrt-v4 within the pbrt-vs-Mitsuba noise floor on Tier 0–1; replaces three-gpu-pathtracer
as the siblings' reference renderer.

**Progress:** the megakernel runs with Owen-Sobol, NEE + MIS, RR, Wächter-Binder offsets, vec4 radiance and the
pbrt-v4 BSDF subset (diffuse, conductor, smooth/rough dielectric). It matches the references on Tier 0 (furnaces exact,
quad light to its closed form, Cornell within the references' own disagreement) and on sponza-white (means within
0.13%) and on env-sphere with an importance-sampled HDR map (P2.8; +0.01% vs Mitsuba). Textures (P1.4c) and OpenPBR
materials (P2.17–P2.23) are in; Tier 1 wave 1 (7 glTF test models) agrees with whichever reference is valid, and the
coverage matrix exists (P2.31). Left: Tier 1 wave 2 (anisotropy, textured models, sheen, iridescence, dispersion), the
Tier 1 gate decision, OIDN on final frames, kernel size on Windows (P2.10; layered kernels are 51–52 KB), and the
switch-over for the siblings.

## ✅ P3 — Speed (built; gate provisional on the M5, the 3060 half pending Windows)

Wavefront vs megakernel (both kept as settings until measured); wide BVH if it passed; coherence tricks; visibility-
buffer primaries; NVIDIA/AMD timing on Windows; ray-class material LOD, alpha policy, cache hooks and debug views (R1, R2, R4,
R10 from research note 12).
**Progress (2026-11-18):** closed on a provisional basis (D32). Built: wavefront (P3.1/P3.2/P3.10), subgroup appends and sorts (P3.3/P3.4: appends yes, sorts no), alpha MASK + policy (P3.7, R2, D31), V-buffer primaries (P3.6), material LOD
(R1), scene-copy packing (P3.11, now the default), the auto policy as code (P3.9, now the recommended scheduler entry point, with a kernel-size rule), hooks/counters/AOVs (R10, R4). The library defaults follow the best provisional evidence (D32); the quiet pass is an optional
confirmation (docs/QUIET-PASS.md). **Gate, provisional on the M5 (best effort, load recorded; `bench/results/P3-GATE.md`): time to equal quality vs three-gpu-pathtracer and mutracer on `sponza-white` (the Tier 2 scene with native references) and Cornell.** The
3060 half (the same table, and the >= 150 Mrays/s incoherent number: last read 102 bounce / 229 primary Mrays/s on the trace core before any wide BVH, P-W1) is pending Windows, WINDOWS-HANDOFF W13. Still open: the wide BVH integration (waits for the bvh branch merge, maintainer's call, D25), W10 (DXC compile time), P3.5 tile mode, P3.8 per-stage timing.
**Exit gate:** faster time-to-equal-quality than every web competitor on every Tier 2 scene (M5, 3060L); ≥150 Mrays/s
incoherent on 3060L.

## ✅ P4 — Real time (built end to end; gate provisional on the M5, P4.5b pending the 3060)

Blue-noise sampler, demodulated AOVs, SVGF-class filter → upscaler, ReSTIR DI, cache hooks with gi's hash cache,
preview mode; gi moves onto the core. Real-time termination policy, light grid, history confidence and gate scenes
(R3, R5-R9). Real-time quality is judged against our own reference-mode image plus temporal stability (R9).
**Exit gate:** ~30 fps on 3060L-class (half-res → 1080p, 2 bounces + ReSTIR DI; estimate to confirm in P3); steadier
in motion than preview mode on the Denoiser scorecard metrics.

Already done early: our own blue-noise textures (P0.17, D16).
**Progress (2026-11-15):** P4.1 built (megakernel): mode presets, the blue-noise sampler, per-frame denoiser inputs (demodulated diffuse / specular, normal, viewZ, camera motion, roughness), `realtime-check` page
(`docs/design/realtime.md`). Left: wavefront frame mode (P4.1b), ReSTIR defaults and the unification (P4.2d / P4.2g), R3, R6-R9.
**Progress (2026-11-18, closed provisionally, D32):** the real-time path exists on both schedulers (megakernel and wavefront agree per pixel), ReSTIR DI is opt-in with unbiased defaults, and the chain is measured on the M5 Pro; the 30 fps gate on the RTX 3060L waits for P4.5b (Windows). Open: P4.2d (candidates / k / historyCap at equal chain cost), R3, R6-R9, the N / NRC integration. P4.5-P4.7 built and measured on the M5 Pro without a cache (`bench/realtime/`, page `realtime-pipeline`, docs/design/realtime.md section 5, bench/results/REALTIME-PIPELINE.md): kernel -> SVGF -> upscaler, one jitter owner (the sampler), reactive mask, preview chain. Provisional frame 19.6 ms (51 fps) at 960 x 540 -> 1080p, maxDepth 2 (31.0 ms with biased ReSTIR); steadier in motion than the preview pipeline's OIDN (boil 2.5x lower, FLIP 20% lower). The 30 fps gate on the 3060L is estimated at 23-32 fps without ReSTIR and waits for the 3060 run (P4.5b).

## ⬜ P5 — Breadth

Items: bindless Tier 3, ReSTIR GI, spectral + dispersion, volumes (WebVDB), MaterialX -> WGSL, tracing support for gi's Split RC, coop-matrix denoiser.
Each item gated on its own bench row.

**Merged on main (2026-11-18, four branches, `bench/results/VOLUMES.md`, docs/design/{volumes,spectral,restir-gi,materialx}.md):** built = ReSTIR GI (TASKS P5.1-P5.5), volumes (P5.v1-P5.v8), spectral + dispersion (P5.s1-P5.s4, P2.28), MaterialX -> WGSL (P5.m1-P5.m7). Left = bindless Tier 3, Split RC tracing, coop-matrix denoiser, and the wavefront port of ReSTIR GI (volumes and spectral run on the wavefront since 2026-11-18, V270-V272, per pixel equal to the megakernel; ReSTIR GI is megakernel only; MaterialX runs on both). Combinations that were never run together throw (`p5-combinations.test.ts`): spectral x volumes, spectral x restir, volumes x restir; group-2 binding table in CLAUDE.md.

**Progress (2026-11-18):** volumes built (TASKS P5.v1-P5.v8): homogeneous and dense-grid media on the megakernel, delta tracking, NEE through media with MIS, scene schema and pbrt / Mitsuba exporters, real VDB via WebVDB; seven closed forms pass and four scenes agree with pbrt and Mitsuba at the references' own noise floor (`bench/results/VOLUMES.md`, docs/design/volumes.md). Open: brick-majorant timing and defaults (quiet pass), wavefront support, NanoVDB traversal.
