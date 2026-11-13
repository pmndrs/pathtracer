# Phases

Each phase ends with a number from the bench. Details and rationale: [spec/SPEC-draft.md](spec/SPEC-draft.md) §11.
Status: ⬜ not started · 🟨 in progress · ✅ done.

## 🟨 P0 — Foundations: know the number to beat

- Monorepo scaffold, `detectCaps()`, shared contract types.
- Bench: GPU runner (headless Chrome), `Renderer` adapter interface, metrics, A/B viewer.
- Tier 0 scenes; exporters to pbrt-v4 and Mitsuba 3; cached references rendered on the Mac CPU.
- mutracer and three-gpu-pathtracer (WebGPU) running behind the adapter.
- Literature verification pass.
- `windows-testing` branch + Windows handoff doc.

**Exit gate:** one command prints time-to-equal-error for each competitor on Cornell and Sponza against pbrt-v4, on M5
(and on RTX 3060L once Windows runs it).

## ⬜ P1 — Trace core for utility consumers (@pmndrs/raytracer)

L0 visibility, L1 closest-hit (intervals, ray cones), lazy `rtSurface`, L2 one-bounce shade; inline WGSL + batch
dispatcher; queue ABI; material/light scene copy. Reflections moves onto it.
**Exit gate:** reflections switches by deleting code; traced hits no longer darker than the path tracer; no
regression in reflection rays/s.
**Parallel (in bvh):** BVH4 / CWBVH8 vs BVH2. Gate ≥1.3× incoherent closest-hit on 3060L and M5.

## ⬜ P2 — Reference path tracer, megakernel (@pmndrs/pathtracer)

Owen-Sobol, OpenPBR-subset uber BSDF, NEE + MIS, area/env/sun lights, RR, robust offsets, vec4 radiance, OIDN final.
**Exit gate:** converges to pbrt-v4 within the pbrt-vs-Mitsuba noise floor on Tier 0–1; replaces three-gpu-pathtracer
as the siblings' reference renderer.

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

## ⬜ P5 — Breadth

Bindless Tier 3, ReSTIR GI, spectral, volumes (WebVDB), MaterialX → WGSL, tracing support for gi's Split RC,
coop-matrix denoiser. Each item gated on its own bench row.
