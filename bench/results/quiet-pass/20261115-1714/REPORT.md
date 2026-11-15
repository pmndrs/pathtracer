# Quiet pass 20261115-1714: `--only p3` (P3.1, P3.2, R2 / P3.7, P3.6)

## Conditions: NOT a clean quiet pass

- 17:14-17:43, commit eacab97, Apple M5 Pro, Chrome 157 headless. The machine had just rebooted: Spotlight indexing and
  fileproviderd at ~100% CPU each, 1 browser GPU process, background apps Claude, Creative Cloud, Fabric, Notion, Spotify.
  Load average 4.5-5.5 at start, 4.8 at end; per-round load1 3.4-6.4.
- All rows are valid=1 and none is marked `loaded` (the runner gates on load1 > core count). Notes read
  `best effort (background apps: ...)`. Treat absolute ms as provisional. Paired ratios come from rounds run back to back
  (interleaved ABBA, round i against round i), so they are informative, and they match the provisional TASKS numbers almost
  everywhere.
- A clean rerun is still needed before any gate or default is decided. No default or DECISIONS entry is changed by this report.
- Tool: `node bench/adapters/paired-ratios.ts --report <name> [--base pt-megakernel] [--base-map a=b,...]`. Ratio = cell /
  base cell of the same round, min..max over rounds. (`alpha-policy-ab.ts` timing columns are wrong for tex-mask in this
  pass: it takes the last 5 CSV rows of a cell and the later p36 runs appended tex-mask rows; its quality columns are fine.)

## P3.1 / P3.2: schedulers (ratio to megakernel, 5 rounds, min..max)

| scene (film, spp) | mk ms/spp | slice 1 (pt-wavefront) | inline (pt-wf-refill) | batch | inline L2 | inline L4 | load1 |
|---|---|---|---|---|---|---|---|
| cornell 512x512, 256 | 13.90-13.93 | 0.622..0.627 | 0.562..0.566 | 0.646..0.655 | 0.730..0.733 | 0.718..0.720 | 3.7-5.1 |
| furnace-dielectric (maxDepth 1000), 256 | 16.13-16.33 | 5.25..6.74 (85-110 ms) | 0.292..0.561 (1 noisy round) | 0.324..0.335 | 0.182..0.186 | 0.126..0.131 | 3.9-6.3 |
| sponza-white 1280x720, 128 | 49.55-55.21 | 0.766..0.864 | 0.944..1.088 | 0.773..0.931 | 0.870..1.059 | 0.819..1.015 | 3.6-5.8 |
| env-sphere 384x256, 1024 | 1.793-1.817 | 1.566..1.594 | 1.393..1.407 | 1.530..1.551 | 0.928..0.964 | 0.694..0.725 | 3.2-4.5 |
| quad-light 256x256, 1024 | 0.118-0.127 | 1.511..1.592 | 1.694..1.845 | 1.799..1.917 | 1.388..1.487 (best) | 1.578..1.713 | 4.2-5.3 |

All provisional TASKS numbers hold. Sponza is noisier (mk 49.6-55.2 ms across rounds); its minima match provisional (slice 1
0.77, batch 0.77), and inline on Sponza reads about 1.0 (V87 coherence loss). Per-stage timestamps were not collected.

### Layers sweep (env-sphere 384x256, 1024 spp, pt-wf-refill vs mk, 3 rounds, load1 4.2-5.5)

| L | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| ratio | 1.394..1.407 | 0.941..0.948 | 0.705..0.706 | 0.566..0.570 | 0.513..0.514 | 0.500..0.502 | 0.512..0.513 |
| wf ms/spp | 2.51 | 1.70 | 1.26 | 1.02 | 0.92 | 0.90 | 0.92 |

Plateau from L16 (~1M slots for 98k pixels); L64 no better.

### refillThreshold (sponza-white 1280x720, 64 spp, pt-wf-batch vs mk, 3 rounds, load1 4.7-6.4)

0.1: 0.794..0.796; 0.25: 0.795..0.796; 0.5: 0.795..0.797. No measurable effect on Sponza.

## R2 / P3.7: alpha policy and wavefront MASK (320x240, 1024 spp, 5 rounds, load1 3.8-5.9)

Ratio to the same scheduler's exact policy (paired min..max):

| scene | scheduler | shadow | indirect | secondary | exact ms/spp |
|---|---|---|---|---|---|
| tex-mask | mk | 0.845..0.868 | 1.025..1.043 | 0.874..0.881 | 0.322-0.325 |
| tex-mask | wf | 0.939..0.955 | 0.979..1.003 | 0.933..0.947 | 0.400-0.407 |
| foliage-mask | mk | 0.793..0.804 | 1.008..1.020 | 0.770..0.780 | 1.261-1.270 |
| foliage-mask | wf | 0.880..0.886 | 0.953..0.958 | 0.818..0.822 | 1.103-1.108 |

Quality against our own exact 16384 spp render (relMSE / relBias, identical for both schedulers): tex-mask exact 2.11e-5 / 0.00%,
shadow 3.00e-2 / -5.35%, indirect 8.49e-3 / -2.85%, secondary 5.38e-2 / -8.42%; foliage-mask exact 6.89e-5 / 0.00%, shadow
5.95e-2 / -12.67%, indirect 9.32e-3 / -3.78%, secondary 9.45e-2 / -17.28%.

Wavefront MASK vs megakernel (exact): tex-mask 1.242..1.259, foliage-mask 0.869..0.878. Holds.

### How many ms is the 8-23%? (median saved per sample; 1080p = 27x pixels, a linear ESTIMATE)

| scene | sched | policy | exact ms | policy ms | saved ms (min..max) | saved % | darker | 1080p ms (est.) |
|---|---|---|---|---|---|---|---|---|
| tex-mask | mk | shadow | 0.324 | 0.279 | 0.046 (0.042..0.050) | 14.1% | 5.3% | 1.2 |
| tex-mask | mk | indirect | 0.324 | 0.336 | -0.012 (-0.014..-0.008) | -3.7% | 2.9% | -0.3 |
| tex-mask | mk | secondary | 0.324 | 0.284 | 0.039 (0.039..0.041) | 12.2% | 8.4% | 1.1 |
| foliage-mask | mk | shadow | 1.265 | 1.004 | 0.260 (0.249..0.262) | 20.6% | 12.7% | 7.0 |
| foliage-mask | mk | indirect | 1.265 | 1.281 | -0.019 (-0.026..-0.010) | -1.5% | 3.8% | -0.5 |
| foliage-mask | mk | secondary | 1.265 | 0.980 | 0.284 (0.278..0.292) | 22.5% | 17.3% | 7.7 |
| tex-mask | wf | shadow | 0.406 | 0.383 | 0.021 (0.018..0.025) | 5.3% | 5.3% | 0.6 |
| tex-mask | wf | indirect | 0.406 | 0.401 | 0.005 (-0.001..0.009) | 1.3% | 2.9% | 0.1 |
| tex-mask | wf | secondary | 0.406 | 0.380 | 0.023 (0.022..0.027) | 5.6% | 8.4% | 0.6 |
| foliage-mask | wf | shadow | 1.104 | 0.975 | 0.131 (0.126..0.133) | 11.9% | 12.7% | 3.5 |
| foliage-mask | wf | indirect | 1.104 | 1.056 | 0.048 (0.046..0.052) | 4.4% | 3.8% | 1.3 |
| foliage-mask | wf | secondary | 1.104 | 0.906 | 0.198 (0.196..0.201) | 17.9% | 17.3% | 5.3 |

The saving is 5-23% of trace time but only 0.02-0.3 ms per sample at 320x240 (est. 0.6-7.7 ms per sample at 1080p), for
3-17% lost brightness. The 1080p numbers assume the same cutout density.

## P3.6: V-buffer vs traced (raster pass INCLUDED), ratio vbuffer / traced of the same scheduler, 5 rounds

| scene | maxDepth | megakernel | wavefront (slice 1) | provisional |
|---|---|---|---|---|
| cornell 512x512 | default (32) | 0.998..1.003 | 1.001..1.008 | 1.00 / 1.00 |
| cornell | 2 | 0.933..0.973 | 0.984..1.011 | 0.95 / 1.00 |
| sponza-white 1280x720 | default (8, RR 4) | 0.947..0.949 | 0.940..0.942 | 0.95 / 0.94 |
| sponza-white | 2 | 0.899..0.900 | 0.898..0.902 | 0.90 / 0.90 |
| tex-mask 320x240 | default | 1.002..1.069 | 1.041..1.068 | not timed |
| tex-mask | 2 | 0.937..1.070 | 1.010..1.074 | not timed |

Provisional numbers hold. tex-mask is neutral to 4-7% slower (a 0.3 ms film: the bench raster pass is a fixed cost).
MetalRoughSpheres was not in the pass script. Correctness (`--page vbuffer-check`) was not part of this pass.

## Recommendations (no decisions; the maintainer decides after a quiet rerun)

- P3.9 auto policy: `batch` for big films (Sponza 0.77-0.93 of mk; threshold irrelevant there); `inline` + `layers` for small
  films (env-sphere 0.50 at L16-32, L2 already 0.94, L1 1.40); megakernel for trivial shading (quad-light 1.4-1.9x slower in
  every variant); `inline` without layers only for deep uniform paths (Cornell 0.56; furnace 0.29).
- Alpha policy default: keep `exact`; the opaque policies stay A/B / preview settings.
- V-buffer: worth wiring into the P4 real-time mode where the primary is a large share of rays (Sponza 5-6% at default depth,
  10% at depth 2, even with the naive bench raster); not a universal win on tiny films.
