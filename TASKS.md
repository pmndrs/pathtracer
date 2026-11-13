# Tasks

Owner = who is working on it (agent label or "maintainer"). Mark done with ✅ and a one-line result. Add follow-ups
as new rows rather than editing old ones. Open questions for the maintainer go in the last section, each with a
suggestion, pros/cons and the A/B that would settle it.

## P0 — Foundations

| ID | Task | Owner | Status | Notes |
|---|---|---|---|---|
| P0.1 | Monorepo scaffold: pnpm workspace, TS, Vitest, packages/raytracer, packages/pathtracer, bench | lead | 🟨 | |
| P0.2 | Contract types: conventions, Ray/Hit (mirrors bvh layout v2), FrameCamera/AOVs, `Renderer` bench adapter, scene description schema | lead | 🟨 | `packages/raytracer/src/contract/`, `bench/src/adapter.ts`, `bench/scenes/schema.ts` |
| P0.3 | `detectCaps()` + `caps.html` diagnostics page | agent:caps | ⬜ | spec §4, research note 06 §4–5 |
| P0.4 | GPU runner: port bvh `tests/gpu` (Chrome over CDP, static server, pages, `--gpu`, clock note, GPU lock, timer calibration) into `bench/runner` | agent:runner | ⬜ | |
| P0.5 | Image IO (PFM r/w), metrics (relMSE, SMAPE, FLIP, bias map, firefly %), A/B viewer page (swipe, toggle, diff, false colour, exposure) | agent:metrics | ⬜ | |
| P0.6 | Tier 0 scenes in the neutral schema; exporters to pbrt-v4 and Mitsuba 3 (+ glTF for web renderers); build pbrt-v4 and install Mitsuba on Mac; render cached references | agent:scenes | ⬜ | |
| P0.7 | Competitor adapters: mutracer, three-gpu-pathtracer (WebGPU, both schedulers) behind `Renderer` | — | ⬜ | after P0.4 + P0.6 |
| P0.8 | Literature verification pass (INDEX §4 backlog, NVIDIA licence scope, Chrome flags, failed paper downloads) | agent:verify | ⬜ | |
| P0.9 | First baseline table: time-to-equal-error, competitors vs pbrt-v4 on Cornell + Sponza (M5) | — | ⬜ | P0 exit gate |
| P0.10 | `windows-testing` branch + `docs/WINDOWS-HANDOFF.md` | lead | ⬜ | near end of P0 |

## Parallel tracks

| ID | Task | Where | Status |
|---|---|---|---|
| X1 | Wide BVH: BVH4 / CWBVH8 node formats + WGSL traversal + GPU collapse pass; gate ≥1.3× on 3060L and M5 | ~/Developer/bvh | ⬜ not started |

## Open questions for the maintainer

_None right now._
