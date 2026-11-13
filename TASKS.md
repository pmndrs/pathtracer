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
| P0.7 | Competitor adapters: mutracer, three-gpu-pathtracer (WebGPU, both schedulers) behind `RendererAdapter` | agent:competitors | 🟨 | |
| P0.8 | Literature verification pass | agent:verify | ✅ | `research/VERIFY.md`; OIDN 3 timeline, STBN licence, CWBVH 6.3× gap, ReSTIR GI ~8 ms |
| P0.9 | First baseline table: time-to-equal-error, competitors vs reference on Cornell + Sponza (M5) | — | ⬜ | P0 exit gate; needs P0.7 + P0.13 |
| P0.10 | `windows-testing` branch + `docs/WINDOWS-HANDOFF.md` | lead | ⬜ | near end of P0 |
| P0.11 | Runner emits the viewer's `report.json` manifest (schema in `bench/viewer/src/report.ts`) | — | ⬜ | decided: viewer format wins |
| P0.12 | Dedupe PFM IO: `bench/scenes/pfm.ts` → reuse `bench/metrics/pfm.ts` | — | ⬜ | |
| P0.13 | Tier 2 "sponza-white": Sponza geometry (glTF sample asset, sha256 in manifest) with untextured grey diffuse, so pbrt/Mitsuba/web match exactly | — | ⬜ | needed for P0.9 |
| P0.14 | FLIP validation against NVlabs reference images; HDR-FLIP as an A/B against LDR-FLIP on firefly scenes | — | ⬜ | |
| P0.15 | Move the quad-light analytic check from `bench/references/compare.ts` into `bench/metrics` | — | ⬜ | |
| P0.16 | `THIRD_PARTY.md` (FLIP lineage, bvh runner port, any ported code) | — | ⬜ | |
| P0.17 | Blue-noise texture generator of our own (void-and-cluster; FAST generator, EA BSD-3) since STBN assets are non-commercial | — | ⬜ | needed by P4, cheap to start |
| P0.18 | Resource-table and subgroup-matrix compile probes on the caps page (Windows/Chrome Canary) | — | ⬜ | names are guesses until tested |

## Parallel tracks

| ID | Task | Where | Status |
|---|---|---|---|
| X1 | Wide BVH: BVH4 / CWBVH8 node formats + WGSL traversal + GPU collapse pass; gate ≥1.3× on 3060L and M5 | ~/Developer/bvh | ⬜ not started |

## Open questions for the maintainer

_None right now._
