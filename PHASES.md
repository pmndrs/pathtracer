# Phases

Agreed 2026-11-13 (decisions D1–D21 in [DECISIONS.md](DECISIONS.md)); no longer under discussion. Each phase ends with a
number from the bench. Details and rationale: [spec/SPEC-draft.md](spec/SPEC-draft.md) §11. Task-level progress:
[TASKS.md](TASKS.md). Phases overlap: a phase starts as soon as what it needs exists, and its gate is checked when the
numbers are in.
Status: ⬜ not started · 🟨 in progress · ✅ done. _Last updated 2026-11-14._

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
walk), the scene copy with materials, lights and light sampling (P1.4), and L2 `rtShade` (P1.3). Left: reflections
onto the core (P1.5, the gate itself), the trace settings A/B (P1.6, quiet pass), incremental scene updates, textures,
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

## ⬜ P3 — Speed

Wavefront vs megakernel (both kept as settings until measured); wide BVH if it passed; coherence tricks; visibility-
buffer primaries; NVIDIA/AMD timing on Windows.
**Exit gate:** faster time-to-equal-quality than every web competitor on every Tier 2 scene (M5, 3060L); ≥150 Mrays/s
incoherent on 3060L.

## ⬜ P4 — Real time

Blue-noise sampler, demodulated AOVs, SVGF-class filter → upscaler, ReSTIR DI, cache hooks with gi's hash cache,
preview mode; gi moves onto the core.
**Exit gate:** ~30 fps on 3060L-class (half-res → 1080p, 2 bounces + ReSTIR DI; estimate to confirm in P3); steadier
in motion than preview mode on the Denoiser scorecard metrics.

Already done early: our own blue-noise textures (P0.17, D16).

## ⬜ P5 — Breadth

Bindless Tier 3, ReSTIR GI, spectral, volumes (WebVDB), MaterialX → WGSL, tracing support for gi's Split RC,
coop-matrix denoiser. Each item gated on its own bench row.
