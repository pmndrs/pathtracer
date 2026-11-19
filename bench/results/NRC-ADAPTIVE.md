# Adaptive NRC termination and portable cache experiments

November 19, 2026 · Apple M5 Pro · Chrome 157.0.8103.80, headless Metal. Times are provisional on a shared desktop. Image error is linear HDR relative MSE unless labeled otherwise. Results are paired within a scene, resolution, seed sequence and reference. The neural measurements use portable tiled fp16. Run directories below are relative to `bench/results/incoming`.

## Phase 2. When to end a display path

The `termination=spread` experiment keeps the training tail fixed at M and uses the NRC paper's area-spread criterion on display paths. At the first surface it stores the primary projected area. Each later non-delta segment adds distance divided by the square root of its incoming directional PDF times its arrival cosine. A query becomes eligible once the squared sum exceeds `spreadThreshold` times the primary area (default 0.01), subject to the minimum `queryDepth`. It skips delta surfaces. Each pixel makes at most one display attempt. A hash miss with continuation enabled traces the remaining suffix. A selected training pixel uses the **actual accepted depth** to correct beauty and the denoiser planes. The spread state is private to a megakernel path. The study uses surface paths with conventional BSDF sampling and next-event estimation. [NRC paper, section 3.4](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing) · [Implementation](../nrc/termination.ts)

| Comparison | Fixed depth | Spread | Interpretation |
|---|---:|---:|---|
| Glossy Sponza, hash, warmed 256-frame error | 0.03189 | 0.03121 | Two runs per arm. The fixed floor crop improved from 0.04050 to 0.03809. |
| Same hash workload, median frame time per run | 3.2–3.3 ms | 3.5–3.6 ms | The image improved at equal samples. The extra tracing costs time. |
| Glossy Sponza, portable fp16 MLP, warmed 256-frame error | 0.02379–0.02385 | 0.02154–0.02184 | About 9% lower error. The warmed frame medians were 9.5–11.6 ms at fixed depth and 8.3–9.4 ms with spread across two runs per arm. |
| Diffuse Sponza, hash + SVGF + upscale, final frame-16 error | 0.1105 | 0.1111 | The complete chain measured 12.7 vs 16.5 ms in one pair at 640 × 360 to 1280 × 720. |
| Glossy Sponza, single frame after a camera flight, hash | 1.899 | 1.936 | Camera motion did not reproduce the accumulated still-image gain in this pair. |

The glossy stills use 320 × 180, the renderer's 1,024-spp reference and A/B/B/A order. The floor crop is x = 80…239, y = 120…169 in top-down image coordinates. The reconstructed diffuse comparison uses Mitsuba at 4,096 spp. Separate instrumented frames counted rough/glossy scattering vertices of 70,796/51,189 at fixed depth versus 74,453/53,806 with spread. Rough/glossy display attempts were 32,306/24,187 versus 32,061/24,122. Mean attempted display depth moved from 1.00 to 1.11. These visit counts include training paths. Instrumentation is off in the timed still comparisons. [Study and image output](../pages/nrc-study.ts) · [Reconstruction study](../pages/nrc-pipeline-study.ts)

**Coarse and noisy cells.** Both comparisons below use the same spread rule, a 64-frame ground-floor warm-up and a cut into the upper gallery. Each row shows one matched pair against the gallery's own 1,024-spp reference. The coarse-level and variance controls were run separately.

| First frame after the cut | Continued pixels | Signed brightness bias | Error | Cut frame |
|---|---:|---:|---:|---:|
| Accept up to three coarser levels | 11.6% | +10.2% | 6.23 | 4.7 ms |
| Accept only the finest level | 91.2% | −3.7% | 11.31 | 6.1 ms |
| All levels, variance off | 11.6% | +10.2% | 6.23 | 4.7 ms |
| All levels, variance threshold 0.002 | 36.7% | −2.2% | 8.39 | 5.2 ms |

The variance arm accumulates first and second moments of bounded log luminance per training **sample**, including variation within a single frame. It applies the grid's existing history decay, then tests the selected cell at display time. A rejected query keeps its RGB for bootstrapping if training shares that vertex. The variance arm used 64 MiB vs 40 MiB at 2²⁰ slots, plus small parameter and counter buffers. The cutoff reduced the bright bias with fewer continued paths than rejecting every coarse entry. Its extra tracing raised single-frame noise, and the added moments cost bandwidth and storage. This is a confidence heuristic inspired by variance-aware termination. The published variance-trading estimator uses a different decision rule. [Kandlbinder et al.](https://cg.ivd.kit.edu/publications/2024/variance-trading/2024_variance_trading.pdf) · [Hash implementation](../nrc/hash-grid.ts)

**Correctness.** A black network in a closed emissive room revealed where each pixel stopped. All 1,024 observed query depths agreed with the rendered energy in aggregate and every RGB channel agreed with a valid truncated-path value. A perfect mirror and a rough conductor each reproduced the traced offscreen card to zero measured error at three camera positions. A diffuse → mirror → diffuse fixture visited 1,280 delta vertices, issued zero cache queries there and issued all 1,024 display queries at the second diffuse hit. Every pixel matched the two-bounce traced reference. [GPU regressions](../pages/nrc-check.ts)

**Decision.** Keep fixed depth as the general control. Keep spread and both confidence cutoffs independently selectable. The state of a shared lighting cache matters: an empty region, a noisy region and a region populated by spatially coarse samples behaved differently in these runs.

## Phases 3–5

| Phase and implementation | Measured result | Decision |
|---|---|---|
| 3. Frozen inference, `frozenDisplayOnly: true` with `train: false` | Gated training queries dropped from 256 to zero in a 32 × 32 room. With the cache visible at depth 1, 1,280 queries became 1,024 display queries. Across beauty and seven output planes, maximum fp32 differences were 2.4e−7 on diffuse and 7.2e−7 on mirror paths. At 320 × 180, 30 paired fp16 frozen frames gave diffuse p50/p95 of 2.1/3.4 ms vs 2.9/3.6 ms, and mirror 1.8/2.3 ms vs 2.7/2.9 ms. | Use the opt-in for consumers reading an already trained cache. The test also corrected first-bounce distance and emission accounting in the existing training-pixel fix-up. A live reflections consumer remains a later phase. [Frozen benchmark](../pages/nrc-frozen.ts) |
| 4. Touched-slot hash resolve and bounded age checks, `hgActive=1` | At 2²⁰ slots, only 0.76% of cells were touched in the measured steady gallery frame. The full scan took 2.6–3.5 ms per warmed whole frame and active maintenance took 3.4–3.5 ms. Active bookkeeping added 4.125 MiB plus 16 bytes. Occupancy, age, expiry, camera cuts, refresh and frame-counter wrap passed parity checks. | Keep full-table resolve as the control. Sparse maintenance has no repeatable whole-frame win in this workload. [Parity and sparse-update benchmark](../pages/nrc-hash-maintenance.ts) |
| 5. Evaluated material inputs, `materialInputs=evaluated` | The fp16/fp32 feature and training-target checks passed on normal-mapped metal, coated plastic and diffuse panels. On the roughness-ramp fixture the cold 256-frame glossy crop went from 0.0452 to 0.0443, the subsequent accumulated crop from 0.0244 to 0.0261, and last-view temporal error from 4.59 to 4.64. | Keep legacy inputs as the default. Evaluated normals, roughness, lobe, diffuse reflectance and F0 are available for the phase 6 factorization experiment. Model width and raw RGB targets were held fixed. Evaluated records use one extra vec4 in inline MLP mode. [Material fixture](../scenes/tier2/nrc-materials.ts) · [Feature oracle](../nrc/material-check.ts) |

The isolated two-cell update also passed at 2¹⁰ and 2²² slots. At 2²² the original test lost its Metal device while both comparison grids and readbacks were resident. The revised test runs one arm at a time and drains batches of at most eight frames. At that size the sparse synthetic update took 0.111 ms/full versus 0.585 ms/active per frame including submission and completion. Use the paired real-scene results for the backend decision.

## Reproduction

Run from the pathtracer root. The bench saves configuration, device information, measurement conditions and uploaded PFMs per arm. For matched gallery images use the gallery PFM saved with each run. The Mitsuba reference supplied with the repository depicts the ground floor.

```sh
node bench/runner/run.ts --page nrc-check --gpu apple --ab check=spread,mirror
node bench/runner/run.ts --page nrc-check --gpu apple --args check=delta-chain
node bench/runner/run.ts --page nrc-study --gpu apple --repeat 2 --timeout 600000 --ab termination=fixed,spread --args scene=sponza-glossy model=hash hgContinue=1 res=320x180 ref=ours refSpp=1024 scenario=static frames=1,16,256 warm=64 infer=tiled flip=0
node bench/runner/run.ts --page nrc-study --gpu apple --repeat 2 --timeout 600000 --ab termination=fixed,spread --args scene=sponza-glossy model=mlp res=320x180 ref=ours refSpp=1024 scenario=static frames=1,16,256 warm=64 infer=tiled precision=f16 flip=0
node bench/runner/run.ts --page nrc-study --gpu apple --ab maxDisplayCoarse=3,0 --args model=hash hgContinue=1 termination=spread res=320x180 refSpp=1024 scenario=unseen warm=64 infer=tiled flip=0
node bench/runner/run.ts --page nrc-study --gpu apple --ab hgVariance=-1,0.002 --args model=hash hgContinue=1 termination=spread res=320x180 refSpp=1024 scenario=unseen warm=64 infer=tiled flip=0
node bench/runner/run.ts --page nrc-frozen --gpu apple --args width=320 height=180 precision=f16 infer=tiled repeats=30 --ab material=diffuse,mirror
node bench/runner/run.ts --page nrc-hash-maintenance --gpu apple --args hgLog2=20 timeFrames=128 checkVariance=1
node bench/runner/run.ts --page nrc-study --gpu apple --repeat 2 --ab hgActive=0,1 --args model=hash scenario=unseen res=320x180 refSpp=512 warm=64 flip=0 hgLog2=20
node bench/runner/run.ts --page nrc-study --gpu apple --ab materialInputs=legacy,evaluated --args scene=nrc-materials scenario=static res=160x90 refSpp=512 frames=1,16,256 warm=0 infer=tiled queryDepth=0 flip=0 motionViews=3 motionRefSpp=256
```

The glossy hash and fp16 runs are `20261119T034942Z-nrc-study` and `20261119T035527Z-nrc-study`. The coarse and variance studies are `20261119T035046Z-nrc-study` and `20261119T035454Z-nrc-study`. The reconstruction and instrumented camera-flight results are `20261119T035045Z-nrc-pipeline-study` and `20261119T035141Z-nrc-study`. Frozen timing, active maintenance and material images are in `20261119T035054Z-nrc-frozen`, `20261119T035252Z-nrc-study` and `20261119T035030Z-nrc-study`. Windows measurements remain open.
