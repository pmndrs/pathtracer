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
| P0.14 | FLIP validation against NVlabs reference images; HDR-FLIP as an A/B against LDR-FLIP on firefly scenes | — | ⬜ | |
| P0.15 | Move the quad-light analytic check from `bench/references/compare.ts` into `bench/metrics` | agent:sponza | ✅ | `bench/metrics/analytic.ts` (+ `checkMaskedImage`) |
| P0.16 | `THIRD_PARTY.md` (FLIP lineage, bvh runner port, any ported code) | — | ⬜ | |
| P0.17 | Blue-noise texture generator of our own (void-and-cluster; FAST generator, EA BSD-3) since STBN assets are non-commercial | — | ⬜ | needed by P4, cheap to start |
| P0.18 | Resource-table and subgroup-matrix compile probes on the caps page (Windows/Chrome Canary) | — | ⬜ | names are guesses until tested |
| P0.19 | Web adapters load `file` meshes (Sponza) through the glb exporter output; runner/matrix iterate `allScenes` | agent:baseline | ✅ | all three render sponza-white; image mean within -0.2% of Mitsuba and +0.0% of pbrt, no per-renderer fix needed (V19); `directionalLight` now true in the adapters; `matrix.ts` gained `--scenes all`, `--repeat`, `--append`, `runs.json` |
| P0.20 | mutracer: exact conductor eta/k via F0/F82 instead of F0 only | — | ⬜ | glTF export carries F0 only (V7) |
| P0.21 | three-gpu-pathtracer GPU timestamps without a readback in the loop | — | ⬜ | wall clock only today |
| P0.22 | Move glTF export from Node pre-export into the page or runner (exporter edits are not detected as stale) | — | ⬜ | |
| P0.23 | Investigate tgpt megakernel's 2% blue-channel noise at facet edges | — | ⬜ | competitor quirk, low priority |
| P0.24 | `sponza-white-ao` variant (diffuse, no lights, ambient-occlusion-like) | — | ⬜ | skipped in P0.13 |
| P0.25 | Rerun the P0.9 baseline on a quiet machine with ABBA across renderers (`matrix.ts` over the runner's `--ab renderer=A,B`), and quote paired ratios | — | ⬜ | **P0 exit gate now depends on this.** the first table ran at load average 97-779 from other agents' GPU jobs (caveat 1 in BASELINE-P0.md) |
| P0.26 | mutracer on Cornell plateaus at relMSE ~6.8e-4 vs Mitsuba (mean +0.07%, 40 px block means up to 3.6% off): find the bias (bevel off, RIS candidates, sun/NEE, spectral-vs-RGB) | — | ⬜ | tgpt images are 0.4% max, so it is mutracer-specific |
| P0.27 | Sponza tgpt-megakernel at 1024 spp (capped at 512 for the 30 min budget) and a longer reference or several reference seeds so 1e-3 becomes measurable | — | ⬜ | reference floor 4.1e-3 (pbrt vs Mitsuba) blocks every target under ~8e-3 on Sponza |
| P0.28 | Competitor adapters: yield to the event loop without a GPU sync inside tgpt's timed loop (runner now tolerates a busy page, V17) | — | ⬜ | minutes-long synchronous loops made `Runtime.evaluate` time out |
| P0.29 | Report an unbiased error estimate that subtracts the reference's own noise (V18), or use two independent reference seeds | — | ⬜ | Sponza "time to 1e-2" currently waits for test noise ~8e-3 |
| P0.30 | Check the shared-machine load before timing and refuse or flag (runner: load average, number of GPU-process Chrome instances) | — | ⬜ | all P0.9 spreads were load-driven |

## Parallel tracks

| ID | Task | Where | Status |
|---|---|---|---|
| X1 | Wide BVH: BVH4 / CWBVH8 node formats + WGSL traversal + GPU collapse pass; gate ≥1.3× on 3060L and M5 | ~/Developer/bvh, branch `wide-bvh` | 🟨 running (session `wide-bvh`, started 2026-11-13) |

## Open questions for the maintainer

- ~~Sponza reference spp~~: settled by P0.9: keep 4096 for both (at 2048 the pbrt-vs-Mitsuba floor rises to 6.2e-3 and even 1e-2 becomes unmeasurable on Sponza).
