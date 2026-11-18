# P3 gate: time to equal quality vs the web competitors (PROVISIONAL, Apple M5 Pro)

Status: provisional (D32). Taken 2026-11-18 15:00-15:40 on the maintainer's working machine with other agents' jobs running (`--load-wait 0`,
never waiting for quiet). Not a quiet-pass number: quote it as "best effort, load recorded". The 3060 half and the Mrays/s gate are pending
Windows (WINDOWS-HANDOFF W13).

## Conditions

- Apple M5 Pro (Metal 3), Chrome 157 headless, commit 7e06441 + the uncommitted D32 defaults (the same tree that is committed next).
- Report `bench/results/reports/p3-gate-provisional-20261118` (cells: cornell 1024 spp, sponza-white 512 spp, snapshots 4/16/64/256,
  3 rounds, all five renderers interleaved per round, order reversing every round). Tables: `.../baseline.md`, summary rows appended to
  `bench/results/competitor.csv`. Method = the P0 baseline (`bench/adapters/baseline.ts`): relMSE (eps 1e-2) of each snapshot against the
  Mitsuba `llvm_ad_rgb` 4096 spp reference (headline, D15; the pbrt-v4 numbers sit next to it in `baseline.md`), power-law fit of
  error against wall time, time to reach the target interpolated or (marked) extrapolated. Medians over the 3 rounds, min..max in brackets.
- Load: load1 at the start of each round ranged 1.5 to 84 (the high values are CPU jobs of other agents: Cornell round 3 ran at 33-84,
  Sponza rounds 1-2 had 14-46 on some cells; the GPU runner lock serialises GPU pages but not apps). Paired-ratio tool marks 0-2 of 3 rounds per
  cell `loaded`; no round is `invalid` (no thermal warning). The min..max spreads are small (Sponza tgpt-megakernel 358.8-360.0 ms/spp, ours 37.3-42.4),
  so the ranking does not depend on the loaded rounds, but absolute ms are not quiet numbers.
- "pt-auto" = our best scheduler per `ptwfAutoPolicy` (D32): Sponza (1280x720) -> plain wavefront; Cornell (512x512, maxDepth 32) -> wavefront with
  inline refill, one layer. Library defaults of D32 (packed scene copy). `pt-megakernel` is shown for reference. Competitors: three-gpu-pathtracer
  (both schedulers) and mutracer, denoisers off, as in P0.

## Tier 2: sponza-white (the Tier 2 scene with native references) - GATE ROW

| renderer | ms/spp (min..max) | relMSE @512 spp (Mitsuba) | time to relMSE 1e-2 (extrapolated from the 512 spp fit) | ours is faster by |
|---|---|---|---|---|
| tgpt-megakernel | 359.4 (358.8..360.0) | 8.72e-3 | 2.6 min (2.6..2.6) | 5.0x |
| tgpt-wavefront | 102.3 (100.6..106.3) | 1.08e-2 | 54.6 s (53.2..56.7) | 1.76x |
| mutracer | 60.3 (58.4..65.9) | 1.72e-2 | 50.9 s (49.1..60 s) | 1.64x |
| **pt-auto (ours)** | **38.5 (37.3..42.4)** | 1.68e-2 | **31.0 s (30.2..34.5)** | 1 |
| pt-megakernel (ours) | 53.9 (47.9..60.7) | 1.68e-2 | 41.9 s (39.0..49.8) | 0.74 (the auto policy gains 1.35x) |

- Our frame time per sample against each competitor, paired per round (`paired-ratios.ts --base pt-auto`): tgpt-wavefront 2.37-2.85x, tgpt-megakernel 8.5-9.6x,
  mutracer 1.52-1.62x, our megakernel 1.24-1.45x slower than the auto choice.
- Equal quality, not equal spp: mutracer and ours read the same relMSE at the same spp (1.72e-2 and 1.68e-2), tgpt has lower noise per spp (8.7e-3 / 1.1e-2),
  which is why the time ratio against tgpt-wavefront (1.76x) is smaller than its per-spp ratio (2.6x).
- Means agree with the references to 0.13% (ours relBias -0.13%, tgpt -0.18%, mutracer -0.19%), so the error is noise, not bias.
- The 1e-2 target is reached by extrapolation for all (the 512 spp snapshot reads 1.7e-2 for ours); the 1e-3 target is under the Sponza reference noise floor
  (4.1e-3, pbrt vs Mitsuba) and is not measurable for anyone. Against the pbrt reference the numbers are the same within 1% (`baseline.md`).
- Verdict, M5: **faster than every web competitor on sponza-white** (1.64x mutracer, 1.76x tgpt-wavefront, 5.0x tgpt-megakernel; min..max ranges do not overlap).

## Other Tier 2 scenes: no native reference, no exact competitor (D20)

`sponza-white-vknrc` (same geometry, other camera), `sponza-glossy` (rough metal floor and columns), `sponza-lamps` / `sponza-lamps-visible` (64 quad lamps) have no pbrt
or Mitsuba reference and both competitors add an energy-compensation term to rough metals that cannot be switched off (D19 / D20), so a time-to-equal-error table
against them would be inexact or self-referenced and is NOT part of the default matrix. What was measured: our auto choice (plain wavefront on all three) against our megakernel,
interleaved, 128 spp, 3 rounds (`p3-gate-tier2-ours`): vknrc 0.826-0.857, glossy 0.589-0.790 (2 of 3 rounds loaded), lamps 0.782-0.805 of the megakernel time. The
competitors' own relative speed on these scenes follows the sponza-white ranking (same geometry, same tracer cost per ray); a native-reference table for the lamps and glossy
scenes needs the Mitsuba renders (hours of CPU) and is left to the optional quiet pass.

## Context: Cornell (Tier 0, 512x512, maxDepth 32, 1024 spp)

| renderer | ms/spp (min..max) | relMSE @1024 (Mitsuba / pbrt) | time to 1e-2 | time to 1e-3 |
|---|---|---|---|---|
| tgpt-megakernel | 61.3 (58.1..62.6) | 1.38e-4 / 5.48e-4 | 1.1 s | 8.6 s |
| tgpt-wavefront | 11.9 (10.4..12.2) | 1.53e-4 / 5.64e-4 | 205 ms | 1.7 s |
| mutracer | 6.19 (6.17..7.78) | 1.38e-4 / 5.44e-4 | 106 ms (105..108) | 867 ms (864..945) |
| **pt-auto (ours)** | 8.78 (8.70..11.36) | 7.25e-5 / 4.82e-4 | 117 ms (115..127) | 730 ms (719..805) |
| pt-megakernel (ours) | 13.4 (13.4..17.1) | 7.25e-5 / 4.82e-4 | 165 ms | 1.1 s |

mutracer is 1.4x cheaper per sample (paired 0.685-0.712 of ours); ours reads 1.9x lower relMSE at equal spp (Owen-Sobol), so at equal quality: mutracer is about 10%
faster to 1e-2 (the ranges do not overlap) and ours is 1.19x faster to 1e-3 (1.15x against the pbrt reference, 1.3 s vs 1.5 s; 1e-3 is only 2.3x the 4.4e-4 floor).
Cornell is not a Tier 2 scene; it shows the small-film end, where the auto policy's inline refill is the reason ours is within reach (our megakernel needs 165 ms to 1e-2).
Before D32 the quiet baseline of 2026-11-14 had mutracer 1.5x faster on Cornell; the gap closed with the wavefront and refill (P3.1/P3.2) and the sampler.

## Pending

- **Mrays/s on the RTX 3060 Laptop: pending Windows (W-row W13).** The only 3060 numbers so far are the trace-core ones of P-W1 (Chrome 157, D3D12): primary 229 and bounce
  (incoherent) 102 Mrays/s against the M5's 285 / 137; the gate is >= 150 Mrays/s incoherent, so it is NOT met by the plain BVH2 walk there. The wide BVH (1.3-1.46x on merged trees on the M5,
  D25) would put the estimate at about 130-150: it waits for the bvh branch merge (maintainer's call) and the 3060 run. Path tracer on the 3060 (track N / P-W1 fix): megakernel sponza-white
  58.0-60.7 ms/spp, wavefront 49.5-51.8 (was 191-202 megakernel before the loop-counter fix), i.e. within about 10% of the M5 megakernel's 53.9 ms/spp in the table above (different runs, different conditions: not a paired ratio); no competitor numbers there yet.
- The 3060 competitor table (same commands, `--name p3-gate-3060`) and `ptwfAutoPolicy`'s choice on that GPU: W13.
- A quiet rerun of this table is an optional confirmation (docs/QUIET-PASS.md baseline step; add `pt-auto` to `--renderers`).
