# Tasks

Owner = who is working on it (agent label or "maintainer"). Mark done with ✅ and a one-line result. Add follow-ups
as new rows rather than editing old ones. Open questions for the maintainer go in the last section, each with a
suggestion, pros/cons and the A/B that would settle it.

## P0 — Foundations

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| P0.1 | Monorepo scaffold: pnpm workspace, TS, Vitest, packages/raytracer, packages/pathtracer, bench | lead | ✅ | commit 73b95d1 |
| P0.2 | Contract types: conventions, Ray/Hit (bvh layout v2), FrameCamera/AOVs, `RendererAdapter`, scene schema | lead | ✅ | `packages/raytracer/src/contract/`, `bench/src/adapter.ts`, `bench/scenes/schema.ts` |
| P0.3 | `detectCaps()` + `caps.html` diagnostics page | agent:caps | ✅ | `packages/raytracer/src/caps/`; M5 Pro = tier 2, subgroups 32, f16, timestamps, immediates, no resource table |
| P0.4 | GPU runner (port of bvh `tests/gpu`) | agent:runner | ✅ | `node bench/runner/run.ts --page <name>`; PFM upload, CSV, ABBA `--ab`, GPU lock, timer calibration page |
| P0.5 | PFM IO, metrics, compare CLI, A/B viewer | agent:metrics | ✅ | `bench/metrics/`, `pnpm --filter @pathtracer/bench viewer`; LDR-FLIP ported from Denoiser (BSD-3 lineage) |
| P0.6 | Tier 0 scenes, exporters (pbrt/Mitsuba/glTF), native refs | agent:scenes | ✅ | 6 scenes, 4096 spp refs; furnace exact in both; pbrt-vs-Mitsuba relMSE 2e-8…6.4e-4 (`bench/references/agreement.md`) |
| P0.7 | Competitor adapters: mutracer, three-gpu-pathtracer (WebGPU, both schedulers) behind `RendererAdapter` | agent:competitors | ✅ | `bench/adapters/` + `COMPETITORS.md`; npm-pinned; Tier 0 at 256 spp: Cornell relMSE 5e-4…1.1e-3, mutracer 6.3 ms/spp vs tgpt wavefront 13 / megakernel 64 |
| P0.8 | Literature verification pass | agent:verify | ✅ | `research/VERIFY.md`; OIDN 3 timeline, STBN licence, CWBVH 6.3× gap, ReSTIR GI ~8 ms |
| P0.9 | First baseline table: time-to-equal-error, competitors vs reference on Cornell + Sponza (M5) | agent:baseline | 🟨 | **Timings provisional (maintainer, 2026-11-13): taken under high load, not trusted.** Correctness/error-vs-spp parts stand. `bench/results/BASELINE-P0.md`: relMSE 1e-2 vs Mitsuba reached in Cornell 133 ms (mutracer) / 242 ms (tgpt-wavefront) / 1.1 s (megakernel), Sponza 1.2 min / 1.1 min / 2.7 min; 1e-3 on Sponza not measurable (floor 4.1e-3); fits k 0.9-1.13; machine heavily loaded by other agents (spread up to 1.75x) |
| P0.10 | `windows-testing` branch + `docs/WINDOWS-HANDOFF.md` | lead | ✅ | pushed 2026-11-13; Windows tasks W1–W8 in the handoff (W6 = P0.25 on the RTX 3060) |
| P0.11 | Runner emits the viewer's `report.json` manifest (schema in `bench/viewer/src/report.ts`) | agent:competitors | ✅ | `bench/runner/report.ts`; `matrix.ts --name X` → `bench/results/reports/X/report.json` |
| P0.12 | Dedupe PFM IO: `bench/scenes/pfm.ts` → reuse `bench/metrics/pfm.ts` | agent:sponza | ✅ | `bench/scenes/pfm.ts` deleted |
| P0.13 | Tier 2 "sponza-white": Sponza geometry (glTF sample asset, sha256 in manifest) with untextured grey diffuse, so pbrt/Mitsuba/web match exactly | agent:sponza | ✅ | 262,266 tris, 1280×720, sun + sky; refs 4096 spp (Mitsuba 24 min, pbrt 57 min); pbrt-vs-Mitsuba relMSE 4.1e-3 (noise), bias +0.22% |
| P0.14 | FLIP validation against NVlabs reference images; HDR-FLIP as an A/B against LDR-FLIP on firefly scenes | agent:flip | ✅ | LDR mean 0.159693 vs NVlabs 0.159693 (diff 2.6e-8, max pixel 1.6e-5) once sRGB decode is on; HDR-FLIP 0.283547 = theirs; `compare.ts --hdr-flip`, viewer H key, report `flip-ab`; default stays LDR (D18). Follow-ups: D18 decision, in-browser HDR parallelism |
| P0.15 | Move the quad-light analytic check from `bench/references/compare.ts` into `bench/metrics` | agent:sponza | ✅ | `bench/metrics/analytic.ts` (+ `checkMaskedImage`) |
| P0.16 | `THIRD_PARTY.md` (FLIP lineage, bvh runner port, any ported code) | agent:legal | ✅ | `THIRD_PARTY.md` + `licenses/NVlabs-FLIP-BSD-3-Clause.txt`, linked from README; follow-up: pin the NVlabs commit and verify the licence text in P0.14 |
| P0.17 | Blue-noise texture generator of our own (void-and-cluster; FAST generator, EA BSD-3) since STBN assets are non-commercial | agent:bluenoise | ✅ | `packages/raytracer/tools/bluenoise/` (`pnpm bluenoise:generate` / `bluenoise:report`, REPORT.md), assets in `packages/raytracer/assets/bluenoise/` (16-bit PNG committed, .f32 git-ignored), loaders + Owen-Sobol in `src/noise`, `src/sampling`. vc2d 64²/128² (128²: 0.09 s), st 64²×32/128²×64 (18 s), gf vec2 64²/128² (27 s, M5, loaded). A/B on a step integrand after a σ2 blur at 1 spp: blue 4.3x lower error than white/Owen-Sobol (1D), vec2 1.7x independent channels vs 2.5x gf; st does not beat vc2d + golden ratio under accumulation. Follow-ups: P4 A/B through OIDN/TAA; joint vec2 for st; FAST-style filter-adapted loss |
| P0.18 | Resource-table and subgroup-matrix compile probes on the caps page (Windows/Chrome Canary) | — | ⬜ | names are guesses until tested |
| P0.19 | Web adapters load `file` meshes (Sponza) through the glb exporter output; runner/matrix iterate `allScenes` | agent:baseline | ✅ | all three render sponza-white; image mean within -0.2% of Mitsuba and +0.0% of pbrt, no per-renderer fix needed (V19); `directionalLight` now true in the adapters; `matrix.ts` gained `--scenes all`, `--repeat`, `--append`, `runs.json` |
| P0.20 | mutracer: exact conductor eta/k via F0/F82 instead of F0 only | — | ⬜ | glTF export carries F0 only (V7) |
| P0.21 | three-gpu-pathtracer GPU timestamps without a readback in the loop | — | ⬜ | wall clock only today |
| P0.22 | Move glTF export from Node pre-export into the page or runner (exporter edits are not detected as stale) | — | ⬜ | |
| P0.23 | Investigate tgpt megakernel's 2% blue-channel noise at facet edges | — | ⬜ | competitor quirk, low priority |
| P0.24 | `sponza-white-ao` variant (diffuse, no lights, ambient-occlusion-like) | — | ⬜ | skipped in P0.13 |
| P0.25 | Rerun the P0.9 baseline on a quiet machine with ABBA across renderers (`matrix.ts` over the runner's `--ab renderer=A,B`), and quote paired ratios | — | ⬜ | **P0 exit gate now depends on this.** the first table ran at load average 97-779 from other agents' GPU jobs (caveat 1 in BASELINE-P0.md) |
| P0.26 | mutracer on Cornell plateaus at relMSE ~6.8e-4 vs Mitsuba (mean +0.07%, 40 px block means up to 3.6% off): find the bias (bevel off, RIS candidates, sun/NEE, spectral-vs-RGB) | — | ✅ | Our adapter's fault, fixed: `readPixels('hdr')` is straight (un-premultiplied) colour, so the 12 px silhouette rim read fully lit (91% of the relMSE); premultiplying by alpha gives relMSE 5.2e-4 / 1.4e-4 / 4.8e-5 at 256 / 1024 / 4096 spp. See COMPETITORS.md "P0.26", V22. Follow-up: rerun the quiet-machine P0.25 timings/baseline table with the fixed adapter (BASELINE-P0 mutracer Cornell rows and the 1e-3 time are stale) |
| P0.27 | Sponza tgpt-megakernel at 1024 spp (capped at 512 for the 30 min budget) and a longer reference or several reference seeds so 1e-3 becomes measurable | — | ⬜ | reference floor 4.1e-3 (pbrt vs Mitsuba) blocks every target under ~8e-3 on Sponza |
| P0.28 | Competitor adapters: yield to the event loop without a GPU sync inside tgpt's timed loop (runner now tolerates a busy page, V17) | — | ⬜ | minutes-long synchronous loops made `Runtime.evaluate` time out |
| P0.29 | Report an unbiased error estimate that subtracts the reference's own noise (V18), or use two independent reference seeds | — | ⬜ | Sponza "time to 1e-2" currently waits for test noise ~8e-3 |
| P0.30 | Check the shared-machine load before timing and refuse or flag (runner: load average, number of GPU-process Chrome instances) | agent:legal | ✅ | `bench/runner/load-check.ts`; `--max-load/--max-gpu-procs/--load-wait/--strict-load`; load + procs in run JSON and CSV note, loaded rows `valid=0`; Windows path (CPU delta, CIM) untested, add to W-tasks |
| P0.31 | Capsaicin as a native GPU reference on Windows (W9) | maintainer (Windows) | ⬜ | research/notes/11-capsaicin.md §9 has the W9 draft (export Tier 0 + sponza-white, matched EXR→PFM refs `capsaicin-*spp.pfm`, compare vs pbrt/Mitsuba, time ref PT + GI-1.2 on the 3060); suggestion: cross-check and timing baseline only, not a D15 reference |

## P1 — Trace core

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| P1.1 | Slice 1: L0 `rtVisible`, L1 `rtClosest`/`rtClosestCone`, lazy `rtSurface`, batch dispatcher (queue ABI v0) on `@pmndrs/bvh` main `dist/` | agent:trace-core | ✅ | `packages/raytracer/src/trace/`, `docs/design/trace-core.md`; correctness page `trace-core` PASSES on M5 (Chrome 157): 589,784 rays over Cornell, quad-light, `edges`, sponza-white × {quantized+records+merged, f32+indexed, SceneTrees GPU trees, SceneTrees refined}: 0 mismatches, 0 watertight slips, 0 escapes; batch variants and inline kernel bit-identical; `rtSurface` max position error 2.3e-7 × scene size. Counted, not failed (f64 CPU walk vs f32 GPU, V27): 14,122 ties, 9,228 grazing, 3,563 tPosition, almost all in the edge-aimed set |
| P1.2 | Timing page `trace-timing` (rays/s, L0 + L1, A/B by page arg) | agent:trace-core | 🟨 | built; numbers are **provisional** (`valid: false`, loaded machine): sponza-white 1M rays, M5 Pro, wg 64 AoS full: L1 primary 116, L1 bounce 78, L0 shadow 83, L0 AO (0.1 × scene) 234 Mrays/s. Quiet-machine run: `--ab workgroupSize=64,128`, `--ab rayLayout=aos,soa`, `--ab hitRecord=full,compact` |
| P1.3 | L2 `rtShade` (flat / Lambert / full, 1 NEE ray, cache hook) | — | ⬜ | needs P1.4 |
| P1.4 | Material/light scene copy (D4), replacing the slice-1 surface table (`buildSurfaceTable`) | — | ⬜ | start from ddgi's scene copy; tangents and material params land with it (V24) |
| P1.5 | Reflections onto the core (D5): `TraceProvider` over `RtBatchTracer` / inline module; exit gate: delete code, no rays/s regression | — | ⬜ | after P1.3 |
| P1.6 | Decide the slice-1 settings by A/B on a quiet machine (workgroup 64/128, AoS/SoA, full/compact) → DECISIONS | — | ⬜ | M5 + RTX 3060 (Windows W-task), paired ratios |
| P1.7 | Bit-exact oracle: ask bvh for an f32-faithful CPU walk, then tighten trace-core to bit equality | — | ⬜ | see bvh requests in the P1.1 report; V27 |
| P1.8 | `rtSurface` position A/B: ray-based vs barycentric (`positions: true`) self-intersection rate on continuation rays | — | ⬜ | design doc, open choice 4 |
| P1.9 | Per-ray masks in the queue ABI (v1) if a consumer mixes masks in one queue | — | ⬜ | v0: one mask per batch |

## Parallel tracks

| ID | Task | Where | Status |
|---|---|---|---|
| X1 | Wide BVH: BVH4 / CWBVH8 node formats + WGSL traversal + GPU collapse pass; gate ≥1.3× on 3060L and M5 | ~/Developer/bvh, branch `claude/wide-bvh-684631` (not merged) | 🟨 both formats built, all correctness gates pass on M5; Sponza nodes 185→74.5 (BVH4)→40.5 MiB (CWBVH). Provisional (loaded) M5 closest-hit: BVH4 merged 1.28–1.32×, CWBVH ≈1.0×, both slower on instanced incoherent (0.57–0.86×). Needs quiet M5 rerun + 3060 runs (bvh `docs/internals/WINDOWS-RUNBOOK.md` §2d); GPU collapse and single-loop instanced walk are follow-ups |
| X2 | Requests from P1.1 to bvh `main`: (1) an f32-faithful CPU walk so trace-core can require bit-exact hits; (2) build workers fail to load via Vite `/@fs/`, and `'auto'` can't fall back for merged groups; (3) export the shared-edge ray generator from `oracle-fixtures`; (4) document that `whenBuilt()` needs `update()` calls; (5) optional indirect count-in-buffer `TraceBatchKernel` | ~/Developer/bvh | ⬜ not started (needs the maintainer's go) |

## N — Neural radiance cache (research track)

Spec and plan: [docs/design/nrc.md](docs/design/nrc.md). Reference implementation: VkNRC (Vulkan, NV cooperative matrix;
**no licence: we follow its design, own code, D23**; read-only clone `research/sources/native/vknrc`). Backends A/B'd: WGSL f32 / f16 / subgroup-matrix,
ORT-Web, WebNN.

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| N0.1 | Check the NRC paper's numbers (net shape, encodings, records/frame, steps, loss, spread threshold, ms on 3090) against `research/papers/gi/nrc-*.pdf`; fix the [M] items in nrc.md §1 | — | ⬜ | arXiv is not reachable from cloud sessions; do it on the Mac |
| N0.2 | `detectCaps` reads `adapter.info.subgroupMatrixConfigs` (componentType, resultComponentType, M, N, K, subgroup sizes); caps page prints them and compiles (a) a one-tile `subgroupMatrixMultiplyAccumulate`, (b) the same with per-subgroup offsets from `@builtin(subgroup_id)` under `subgroup_uniformity` (4 subgroups/workgroup), (c) with `diagnostic(off, …)` on the matrix uniformity diagnostic under Chrome's actual name | — | ⬜ | folds in P0.18's subgroup-matrix half; M5 Pro first. (b) decides N-K2 (nrc-diverges.md) and would unblock the Denoiser's multi-subgroup item too |
| N0.3 | Which backend reaches matrix hardware: Dawn Vulkan vs D3D12 on Windows/RTX 3060L (flag needed?), Metal `simdgroup_matrix` vs M5 neural accelerators | — | ⬜ | Windows part becomes a W-task |
| N0.4 | Read the Denoiser's WGSL runtime (`denoiser/wgsl`, `NetworkRuntime`/`AutoRuntime`, f16 partial sums, subgroup-matrix conv kernel, parity tests) and note what the MLP engine reuses | agent:nrc | ✅ | read from github.com/pmndrs/denoiser (MIT). Findings in nrc.md §2b: on M5 Pro f16 FMA ~6.6 TFMA/s > subgroup-matrix MMA ~3.8 > f32 FMA ~3.1; f16 matrices only with f16 accumulate; >1 subgroup per workgroup fails uniformity |
| N1 | MLP engine: WGSL generator for forward / backward / Adam (f32, f16), f32 master weights; TS f64 oracle with finite-difference gradient check; `mlp-check` and `mlp-fit` pages | agent:nrc | 🟨 | `packages/mlp` (`@pathtracer/mlp`, private, WGSL prefix `nn`), `bench/pages/mlp-check.ts`, `mlp-fit.ts`. **f32 verified on SwiftShader** (cloud, CPU): vs oracle inference 2.3e-7, gradient 3.9e-7, Adam update p99 6.9e-5 (rel); `mlp-fit oracleSteps=12` replay p99 1.1e-5. f16 not run (SwiftShader has no `shader-f16`). Weight gradient = deterministic per-weight pass (V28). No timings yet |
| N1.1 | Run on the M5 Pro: `mlp-check --ab precision=f32,f16`, `mlp-fit --args oracleSteps=8 batch=1024 precision=f16`, then `mlp-fit --ab precision=f32,f16 --repeat 2` (defaults: 256², batch 16384, 1024 steps); viewer A/B of the fitted PFMs | — | ⬜ | first real timings (ms/step, ms/Mquery); quiet-machine rules |
| N1.2 | Faster portable kernels: several threads per record (the 64-float private arrays spill), tiled weight-gradient pass (record chunks in workgroup memory) | — | ⬜ | before N2's table means anything |
| N2 | `mlp-timing` microbench: ms/1M inferences, ms per 16k training step, every arm, widths 64 × {2,4,5} hidden, batch 2^14–2^21, ABBA, load-checked | — | ⬜ | M5 Pro + RTX 3060L; can end the track early |
| N3 | `wgsl-sgm` arm: subgroup-matrix forward then backward, tile shape from the adapter config, ReLU and its 0/1 derivative mask via workgroup memory (no per-element tile access, no NaN trick as in VkNRC). Generator stages are switches so each gpuweb gap is one deletable path: `relu: roundtrip|element`, `chain: roundtrip|convert`, `subgroupsPerWorkgroup: 1|4`, `accumulate: f16|f32` | — | ⬜ | gate: matches `wgsl-f16` within f16 tolerance; report the round-trip share of each layer (evidence for subgroup-matrix-feedback.md W1/W2) |
| N4 | Encodings: frequency + one-blob vs multiresolution hash grid, A/B on `mlp-fit` | — | ⬜ | both stay settings |
| N5 | ORT-Web and WebNN arms: inference + training-step-as-graph (fwd + bwd + Adam exported as one ONNX / WebNN graph, weights in and out) | — | ⬜ | comparison rows in N2 only |
| N6 | `nrc` demo page: diffuse wavefront PT on sponza-white over trace core, spread termination, sparse training paths (per subgroup, as VkNRC) with self-training tail written by the inference pass, 4 × 16k indirect train steps, at full and ratio-2 resolution | — | ⬜ | VkNRC config first, then one change at a time (DIVERGES). Equal-time relMSE/FLIP vs Mitsuba, bias, still-camera stability |
| N6.1 | VkNRC native on the RTX 3060L as the native timing baseline (W10): build under `.cache/native/`, local patches for per-pass timestamps, fixed camera, PFM dump | maintainer (Windows) | ⬜ | D22. Linux build verified 2026-11-14 (no GPU to run); patches never committed |
| N6.2 | OBJ exporter for sponza-white (VkNRC input), and a "VkNRC lighting" setting on the `nrc` page (white sky 10, no sun) for like-for-like images | — | ⬜ | nrc.md §9 |
| N7 | Non-neural hash cache (SHaRC-like, `@pmndrs/gi`) as the equal-time competitor to N6 | — | ⬜ | needs gi on the trace core (P4) or a bench-local copy |
| N8 | `bench/results/NRC.md`: per-vendor "what subgroup-matrix buys" table + viewer report | — | ⬜ | public use needs the maintainer (D8) |
| N9 | Post the subgroup-matrix feedback (draft in docs/design/subgroup-matrix-feedback.md §4) with N2/N3 numbers attached | maintainer | ⬜ | outward-facing (D8): maintainer's go; candidates #4195, #11077, #11079, the proposal's uniformity TODO |

## Open questions for the maintainer

- **NRC: where does the MLP engine live?** Suggestion: private workspace package `packages/mlp` (WGSL prefix `nn*`),
  demo as bench pages until N6; a production cache follows D3 into `@pmndrs/gi`. Details, pros/cons and the deciding
  test in [docs/design/nrc.md](docs/design/nrc.md) §8 (also: gpuweb evidence, Windows Vulkan flag;
  VkNRC licence and native build settled by D22/D23).

- **Capsaicin: a third reference renderer, or a cross-check only?** Suggestion: cross-check and native timing baseline on
  the RTX 3060, for opaque scenes only. Keep pbrt + Mitsuba as the references (D15).
  - Pro (cross-check): independent GPU implementation, same 3060 as W6, MIT, already built on the laptop.
  - Con (as a reference): no glass, f16 output, environment handling needs a patch, Windows-only with a window, its own
    biases (unclamped RR, dropped NaNs, stochastic MIS light PDF).
  - Settles it: W9 on Cornell + sponza-white at 1024–4096 spp. Promote only if mean bias vs pbrt and Mitsuba is
    < 0.3% on both and the error stays within the pbrt-vs-Mitsuba floor.

- ~~Sponza reference spp~~: settled by P0.9: keep 4096 for both (at 2048 the pbrt-vs-Mitsuba floor rises to 6.2e-3 and even 1e-2 becomes unmeasurable on Sponza).
