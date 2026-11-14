# Quiet pass 2026-11-14 14:19 (M5 Pro, Chrome 157, commit 7056f80)

All other apps closed except the Claude app (Remote Control); every row is marked `best effort (background apps:
Claude)`, none `loaded`. Load (1 min) 2.8 at start, 1.2–2.6 during, 2.0 at the end; the 5-min average reached 5.3 at the
end of the Sponza baseline (tgpt-megakernel's long synchronous replay). No macOS thermal warning in any trace-timing
run. Machine on AC, display allowed to sleep. The log's "browser GPU processes: 1" is the `grep -c` counting itself
(fixed in `bench/quiet-pass.sh`); `pgrep` found none.

Defaults and DECISIONS are unchanged: the calls below are the maintainer's.

## 1. P1.6 trace-core settings (trace-timing, sponza-white, 1 M rays, 5 ABBA rounds)

Paired B/A ratio of Mrays/s per round, min..max (median), and the median p90/p10 of the 30 passes:

| A/B | primary | bounce | shadow (L0) | AO (L0) |
|---|---|---|---|---|
| workgroup 128 / 64 | 0.66..1.01 (0.97) | 0.79..1.11 (1.05) | 0.71..1.05 (0.84) | 0.64..1.01 (1.00) |
| SoA / AoS | 0.95..1.05 (0.99) | 0.93..1.18 (1.09) | 0.78..1.13 (0.81) | 0.97..1.02 (0.99) |
| compact / full hit record | 0.94..1.01 (0.96) | 0.85..1.11 (1.06) | 0.84..1.07 (1.00) | 0.96..1.04 (0.99) |
| p90/p10 (median, all runs) | 2.9–3.1 | 1.4–1.5 | 1.6–1.8 | 2.7–2.8 |

**No call possible.** Every range straddles 1. The control shows it: the L0 shadow set never reads the hit record, yet
compact/full moves it 0.84..1.07. p90/p10 stays at ~3 for the ~4 ms passes even with the machine quiet, so the spread is
not other apps: it is most likely the GPU's own clock ramp between short, separated passes (the pass pattern is bimodal,
~4 ms and ~10 ms). Suggested next step (TASKS P1.6a): time back-to-back batches of the same dispatch inside one command
buffer, long enough (≥0.5 s) to hold the clock up, as bvh's pages do, and keep the p10-of-30 as a second column.

## 2. P0.25 competitor baseline (P0 exit gate, M5 half)

`baseline.md` here; report `bench/results/reports/p0-baseline-quiet-20261114-1419`. Renderers interleaved, 3 rounds,
order reversed each round. Cornell 1024 spp, sponza-white 512 spp.

| scene | renderer | ms/spp (min..max) | time to relMSE 1e-2 | time to 1e-3 | relMSE@final (Mitsuba / pbrt) |
|---|---|---|---|---|---|
| cornell | tgpt-wavefront | 11.5 (11.3..12.4) | 215 ms | 1.8 s | 1.53e-4 / 5.64e-4 |
| cornell | tgpt-megakernel | 66.2 (64.4..66.4) | 1.1 s | 9.2 s | 1.38e-4 / 5.48e-4 |
| cornell | mutracer | 6.63 (6.5..6.7) | 114 ms | 942 ms | 1.38e-4 / 5.44e-4 |
| sponza-white | tgpt-wavefront | 105.7 (104.1..106.8) | ~56.0 s (extrapolated) | not measurable (floor) | 1.08e-2 / 1.10e-2 |
| sponza-white | tgpt-megakernel | 379.6 (347.9..414.2) | 2.8 min | not measurable | 8.72e-3 / 8.88e-3 |
| sponza-white | mutracer | 57.7 (57.3..57.8) | ~48.7 s (extrapolated) | not measurable | 1.72e-2 / 1.74e-2 |

All runs valid, means within 0.2% of Mitsuba, run-to-run spread small (Sobol renderers share one noise realisation, so
only time varies). This is the quiet replacement for the provisional BASELINE-P0.md and includes the fixed mutracer
adapter (P0.26). The 3060L half waits on Windows (W6).

## 3. Our megakernel on the same settings (extra; P2.15)

`pt-baseline.md` here; report `bench/results/reports/pt-baseline-quiet-20261114-1419`. Run right after the
competitors, not interleaved with them (time ratios across the two reports are best effort; the load was the same).

| scene | renderer | ms/spp | time to relMSE 1e-2 | time to 1e-3 | relMSE@final (Mitsuba / pbrt) |
|---|---|---|---|---|---|
| cornell | pt-megakernel | 13.9 (13.9..13.9) | 169 ms | 1.1 s | 7.25e-5 / 4.82e-4 |
| sponza-white | pt-megakernel | 49.3 (49.3..49.3) | ~34.4 s (extrapolated) | not measurable | 1.44e-2 / 1.46e-2 |

- Cornell: mutracer is 1.5x faster to 1e-2 and 1.2x faster to 1e-3. It costs half our time per spp; we have about half
  its variance per spp (7.25e-5 vs 1.38e-4 at 1024 spp).
- Sponza: we are fastest, 1.4x ahead of mutracer and 1.6x ahead of tgpt-wavefront to 1e-2 (all three extrapolated
  from 512 spp along their own 1/spp fits). tgpt-wavefront has the lowest variance per spp (1.08e-2 vs our 1.44e-2 at
  512 spp); worth a look at what it does differently (its light sampling or MIS) before P3.
- Means: Cornell relBias −0.01%, Sponza −0.13% vs Mitsuba (same as P2.9).

## 4. X1 wide BVH (bvh branch `claude/wide-bvh-684631` at 11d19f6, HANDOFF "Quiet pass")

Summary `bvh/wide-summary.md`, raw `bvh/wide-*.json`. 3 runs per scene, all quiet, 0 answers differing from f32. Ratios
are min..max of the three runs' medians, speed over f32 BVH2 (above 1 = faster). Gate: ≥1.3x incoherent closest hit
(surface and air-grid rays).

| closest hit | Sponza surface | Sponza air-grid | Bistro surface | Bistro air-grid |
|---|---|---|---|---|
| BVH4, merged | 1.30..1.34 | 1.27..1.28 | 1.03..1.04 | 1.25 |
| BVH4 folded (L1), merged | **1.36..1.43** | **1.34** | 1.05 | 1.27..1.28 |
| CWBVH8, merged | 0.98..1.02 | 1.01 | **1.40..1.41** | **1.39..1.40** |
| CWBVH8 folded (L1), merged | 1.06..1.09 | 1.10..1.11 | **1.46** | **1.45..1.46** |
| best instanced (BVH4 single loop, L7) | 0.88..0.89 | 0.91 | 0.96 | 0.97 |

- **Gate on M5: met per scene on merged trees, but by a different format on each scene.** BVH4 folded passes Sponza,
  CWBVH8 (folded or not) passes Bistro; neither passes both. Instanced trees stay slower than f32 for incoherent rays
  with every format (0.55–0.97x).
- **L1 (folded plane decode) wins everywhere:** +1–8% for BVH4, +3–13% for CWBVH8, on every scene, tree, ray set and
  mode. A clear candidate for the default.
- **L7 (single-loop walk):** helps instanced trees (1.06–1.48x over the nested BVH4 walk on surface/air-grid) but hurts
  merged trees (0.77–0.96x) and primary rays (0.66–0.77x closest). Not enough to make instanced wide trees beat f32.
- The 3060L half of the gate still waits on Windows (bvh runbook 2d).
