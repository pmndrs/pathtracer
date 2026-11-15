# 12 · NVIDIA "Real-Time Path Tracing in Games: Best Practices for Performance and Scalability" (June 2026)

Source: https://github.com/NVIDIA-RTX/Whitepapers/tree/main/pt_performance_whitepaper (English README.md, read in full,
2026-11-15). An NVIDIA paper: ideas fine (D10). Its references to RTXPT, RTXDI, RTXGI/SHaRC and NRD were not followed.
Section numbers (§) are the paper's.

## What it is

A qualitative best-practices guide for D3D12/DXR game renderers at 60 fps. Almost no measurements: one table (Table 6,
§14), one BLAS-flattening experiment (§14), and illustrative budget percentages (Figure 1, "not measured"). No image
metrics, no ground-truth comparison, no cache sizes, no bias or leak numbers. Read it as a checklist, not a source of
tunables.

## Is it GI rather than path tracing? Yes, mostly

- Primary visibility is rasterised (§1). Diffuse GI is "one shallow bounce plus radiance-cache fallback" (§1); deep
  paths run only for cache updates at low spatial frequency. Only primary and near-glossy hits get full materials;
  diffuse secondary and cache-update hits get reduced BRDFs (Table 3).
- Termination by depth limits, throughput thresholds and cache termination; advice against naive Russian roulette at
  low rpp (§2, §8). No bounce counts given.
- Quality = the final denoised, upscaled image and its temporal stability. Bias is fine if "low frequency, temporally
  stable, and hard for players to notice" (§2), validated "against a slower reference path" that is never defined.

So it is a GI-plus-specular renderer with controlled bias. Our reference mode (pbrt/Mitsuba equivalence, D15, D23) has
no counterpart in it; our real-time mode (P4) is where it applies.

## The one measured table (Table 6: RTXGI2 PT sample, RTX 5090, 1080p)

| Config | Frame | Rays | SHaRC update | SHaRC query |
|---|---|---|---|---|
| Baseline | 4.30 ms | 3.91 | | |
| + SHaRC | 3.69 (1.17x) | | 0.69 | 2.56 |
| + SER | 3.78 (1.14x) | 3.40 | | |
| + SHaRC + SER | 3.31 (1.30x) | | 0.69 | 2.16 |
| + SHaRC + SER + force-opaque on update rays | 2.95 (1.46x) | | 0.31 | 2.19 |

Update rays were dominated by any-hit (alpha) execution; forcing them opaque doubled warp occupancy. Static BLAS
flattening: 14.7 -> 11.4 ms, 36% fewer traversal samples (GPU and scene not stated).

## Mapping to our plan

Already ours: NEE + MIS, bounded VNDF (P2.1/P2.3), dedicated sun and env samplers (P1.4d/g), emissive light table,
ray-cone texture LOD (P1.4c), ReSTIR DI (P4), caches in gi (D3), wavefront queues as the SER substitute (P3).

Not portable: SER, HitObject, OMM, DXR payload qualifiers, Clustered Geometry / Partitioned TLAS, DLSS-RR, NRD (D10),
shader-clock heatmaps (WGSL has no clock: use node-visit counters).

Missing, worth adding (tasks R1-R8 in TASKS.md):

1. **Ray-class material LOD** (Table 3, §7): a reduced "diffuse-secondary" material class for bounce >= 2 and cache
   updates (no layered walk, fuzz, film; conservative MIP). Biased, real-time setting only. Also the natural answer to
   D26 (51-55 KB kernels): the secondary class is a smaller kernel or wavefront class.
2. **Per-ray-class alpha policy** (§3, §14): indirect / cache rays treat MASK as opaque. Largest single win in the
   paper; our masked shadow rays are closest-hit loops, so the same cost exists in software.
3. **Termination policy** (§2, §8): conflicts with P2.13 / spec §6 (RR after bounce 3 outside reference mode). A/B RR vs
   depth + throughput threshold vs cache termination at 1-4 spp with the denoiser, judged on flicker as well as relMSE.
4. **Per-signal budgets and debug views** (Tables 1, 5, §13): rays per pixel per signal, depth + termination reason,
   alpha-step count, node-visit heatmap, memory, per-pass time split.
5. **World-space light grid + adaptive light selection** (NEE-AT-style feedback with a global fallback) for secondary
   hits (§8). Screen-tile lists and ReSTIR DI do not help indirect NEE. After ReSTIR DI; could share gi's hash table.
6. **Decorrelated ReSTIR reuse** (§10): randomise neighbour offsets and temporal selection. A rule for the ReSTIR task.
7. **Renderer-driven history confidence** (§10): cache invalidation and lighting change feed the denoiser/upscaler
   reactive mask (spec §7 has history length only).
8. **Real-time gate scenes and failure list** (§10, §13): alpha-heavy, dynamic-light, glossy interior, open world;
   cuts, fast rotation, disocclusion, thin geometry, animated emissives, dynamic env.

## What it changes in how we measure

Our reference mode stays judged against pbrt/Mitsuba. For the real-time mode the paper's criterion is bias that is low
frequency and stable. Proposal (R9): judge real-time presets against our own converged reference-mode image of the same
frame (relBias, low-pass relMSE, FLIP) plus a temporal-stability metric over a camera path, and report every biased
setting against its unbiased fallback. That keeps one ground truth for both modes.
