# Decisions

Settled calls. Each has a date, what was decided, and why. Change one only with a new entry that supersedes it.
Open questions live in TASKS.md, phrased with a suggestion, pros/cons and the A/B that would settle them.

| ID | Date | Decision | Why |
|---|---|---|---|
| D1 | 2026-11-13 | One GitHub repo `pmndrs/pathtracer`, publishing `@pmndrs/raytracer` (prefix `rt`) and `@pmndrs/pathtracer` (prefix `pt`, depends on raytracer). Names confirmed free. | gi and reflections need rays, not the path tracer |
| D2 | 2026-11-13 | `@pmndrs/bvh` owns build, refit, layout and the walk. Wide-BVH (BVH4, CWBVH8) work lands in bvh. | bvh's scope rule: it ends at the hit |
| D3 | 2026-11-13 | Hash radiance cache and probe atlas live in `@pmndrs/gi`. This repo exposes hooks only (L2 cache argument, L3 `onVertex`/`terminate`). | gi already has a world hash cache; keeps dependencies one-way (gi → raytracer) |
| D4 | 2026-11-13 | The core owns hit shading and the material/light scene copy (L2), replacing reflections' and ddgi's own. Start from ddgi's scene copy and extend it. | L2 can't exist otherwise; ends the "traced hits too dark" bias |
| D5 | 2026-11-13 | First sibling to move onto the core: reflections. ddgi follows in P4. | cleanest seam (`TraceProvider`) |
| D6 | 2026-11-13 | Develop on Mac. Windows comes in via a `windows-testing` branch plus a handoff doc, pushed near the end of P0. | Windows needs a foundation to work from |
| D7 | 2026-11-13 | Spectral is v2. Radiance is carried as vec4 from day one. | avoids a rewrite; few effects need it |
| D8 | 2026-11-13 | Competitor comparisons stay internal until a methodology page exists; offer it to the other authors before publishing. | fairness |
| D9 | 2026-11-13 | Literature-only verification pass now; everything else is measured in P0. | measuring beats re-reading |
| D10 | 2026-11-13 | NVIDIA RTX SDK code is off limits; NVIDIA examples, books, talks and papers are fine. NRD code never; native NRD only via the Denoiser Windows harness, consumed as images. | licence; Denoiser clean room |
| D11 | 2026-11-13 | Build both, decide by A/B; competing approaches become settings until measured. | early tuning needs options |
| D12 | 2026-11-13 | Interchange image format is PFM (float32 linear, row 0 = top in memory; files written bottom-up per spec). | pbrt-v4 and Mitsuba both write it; trivial to parse |
| D13 | 2026-11-13 | OIDN 3 releases 2026-11-30 (per maintainer). Reserve its temporal inputs; don't hard-code before release. | |
