# Quiet pass 20261124: standard pass + volume, real-time and ReSTIR rows

## Conditions: near-quiet, best effort

- 13:09-14:21 (phase 1 `bench/quiet-pass.sh` 13:09-14:00, volume rows 14:00-14:16, single-page rows 14:16-14:21), then the
  short-scene rerun and the V1.8 compile runs. Commit 6426176 (`clouds/volume-reference`), Apple M5 Pro, Chrome 158 headless.
- Load average 3.75 / 3.02 / 3.12 at start, 4.53 / 3.95 / 4.03 at the end of the chain; per-run load1 2.6-8.3 (18 cores,
  so no run is marked `loaded`). One browser GPU process (the maintainer's own Chrome window), `fileproviderd` busy,
  background apps ChatGPT, Claude, Creative Cloud, Cursor, Discord, Fabric, Linear, Slack, Spotify: single-page rows carry the
  `best effort (background apps: ...)` note. No other agent held the GPU lock; no builds or tests ran during the timed rows.
- Every number below is a paired ratio from interleaved rounds unless it says ms; absolute ms are best effort. No default
  and no DECISIONS entry is changed by this report.
- Files: phase 1 in `../20261124-1309/` (baseline.md, trace-timing JSON); this directory: `abba.json` (5-round ABBA of
  competitor-page variants, driver `scripts/abba.ts`, run by `scripts/chain.sh` and `scripts/rerun-short.sh` (they name the session scratchpad as their path; edit `SP` to rerun): each round runs every variant once through
  `bench/runner/run.ts --page competitor`, order reversed every other round, ratio = variant / first variant of the same
  round), `abba-8192.json` (rerun of the short scenes), one `<tag>.json` per page, `chain.log`.

## Phase 1: trace-timing A/Bs (sponza-white, 5 ABBA rounds)

Paired B/A of `*_batch_mrays_per_s` (above 1 = B faster), max `*_batch_spread` over the 10 runs, the old p10 ratio and
max `p90_over_p10`.

| A/B | ray set | batch B/A | batch spread | p10 B/A | p90/p10 |
|---|---|---|---|---|---|
| workgroupSize 64 -> 128 (load1 2.6-3.8) | l1_primary | 0.996..1.008 | 1.009 | 0.995..1.004 | 2.63 |
| | l1_bounce | 0.998..1.012 | 1.008 | 0.951..1.431 | 1.64 |
| | l0_shadow | 0.998..1.006 | 1.003 | 0.976..1.443 | 1.78 |
| | l0_ao | 0.997..1.006 | 1.004 | 0.995..1.004 | 2.56 |
| rayLayout aos -> soa (load1 2.7-3.3) | l1_primary | 0.978..1.181 | 1.129 | 0.966..1.173 | 2.70 |
| | l1_bounce | 0.974..1.158 | 1.116 | 0.925..1.140 | 1.26 |
| | l0_shadow | 0.982..1.128 | 1.100 | 0.950..1.443 | 1.68 |
| | l0_ao | 0.981..1.128 | 1.095 | 0.967..1.182 | 2.63 |
| hitRecord full -> compact (load1 2.6-3.2) | l1_primary | 0.977..1.033 | 1.130 | 0.979..1.038 | 2.67 |
| | l1_bounce | 0.943..1.001 | 1.116 | 0.928..0.991 | 1.18 |
| | l0_shadow | 0.990..1.060 | 1.076 | 0.925..1.013 | 1.26 |
| | l0_ao | 0.995..1.063 | 1.072 | 1.002..1.073 | 2.54 |

workgroupSize ties (spread 1.00x: clean). rayLayout and hitRecord ran with batch spreads of 1.07-1.13 (something
interfered inside those page runs); their ranges straddle 1 and agree with the 20261114 reports (defaults best or tied,
compact bounce slower). No change suggested.

## Phase 1: P0 baseline (competitors)

From `../20261124-1309/baseline.md` (3 repeats each):

| scene | renderer | ms/spp (min..max) | fit k | time to relMSE 1e-2 | time to relMSE 1e-3 | relMSE@final (Mitsuba / pbrt) |
|---|---|---|---|---|---|---|
| cornell | tgpt-wavefront | 12.1 (12.0..12.2) | 1.12 | 206 ms | 1.8 s | 1.53e-4 / 5.64e-4 @1024 |
| cornell | tgpt-megakernel | 63.9 (62.7..65.1) | 1.12 | 1.1 s | 9.0 s | 1.38e-4 / 5.48e-4 @1024 |
| cornell | mutracer | 6.57 (6.5..6.7) | 1.12 | 106 ms | 863 ms | 1.38e-4 / 5.44e-4 @1024 |
| sponza-white | tgpt-wavefront | 118.3 (114.0..118.9) | 0.97 | ~1.0 min (extrapolated) | below 2x ref floor | 1.08e-2 / 1.10e-2 @512 |
| sponza-white | tgpt-megakernel | 426.3 (424.8..429.2) | 0.99 | 3.1 min | below 2x ref floor | 8.72e-3 / 8.88e-3 @512 |
| sponza-white | mutracer | 70.8 (66.6..71.4) | 0.89 | ~1.4 min (extrapolated) | below 2x ref floor | 1.72e-2 / 1.74e-2 @512 |

mutracer on Sponza had one slow repeat (4 spp 856 ms vs 232 ms; 64 spp 12.1 s vs 4.3 s): its min is the number to quote.

## V1 media (megakernel unless noted; 5 rounds)

`vol-cl-cloud256` (256^3 grid), 256 spp, load1 3.3-5.5:

| variant | ms/spp (min..max) | ratio to global (min..max) | luminance |
|---|---|---|---|
| global majorant | 12.76..12.79 | 1 | 0.41575 |
| brick | 9.93..10.01 | 0.777..0.783 | 0.41607 |
| brick + `trRoulette` | 6.32..6.36 | 0.495..0.498 | 0.41612 |
| brick, r16float grid | 9.32..9.37 | 0.729..0.734 | 0.41604 |
| brick, r16float + hardware filter | 8.31..8.36 | 0.650..0.655 | 0.41603 |
| wavefront inline refill (`pt-wf-refill`), brick | 7.93..8.03 | 0.620..0.629 | 0.41607 |

Relative to brick: roulette 0.64, r16float 0.94, r16 + hardware filter 0.84 (8-bit filter weights, see Traps), wavefront
refill 0.80. The r16float kernels compiled in ~500 ms vs ~240 ms (pipeline creation in the page, cache state unknown).

`vol-cl-sph-g85-a99-t200` (`pt_volume_shadow=ratio`), 256 spp, load1 4.7-7.5: roulette 0.989..0.996, wavefront inline refill
0.810..0.816 (ms/spp 19.96 -> 16.24). Luminance identical to 5 digits.

## P5 media (megakernel, 5 rounds)

| scene, spp | A | B | B/A (min..max) | ms/spp A | load1 |
|---|---|---|---|---|---|
| vol-cornell-smoke, 1024 | shadow analytic | ratio | 0.956..0.967 | 2.066 | 4.9-6.0 |
| vol-milk-sphere, 1024 | shadow analytic | ratio | 1.003..1.006 | 1.578 | 4.7-5.1 |
| vol-grid-cloud, 1024 | majorant global | brick | 1.171..1.176 | 0.611 | 5.0-5.9 |
| vol-vdb-smoke, 256 | majorant global | brick | 0.868..0.887 | 0.943 | 5.6-8.3 |

Brick wins on the big and sparse grids (cloud256 0.78, vdb-smoke 0.87) and loses 17% on the small `vol-grid-cloud`. Ratio
tracking costs nothing on homogeneous media (smoke 0.96, milk 1.005). The "cornell without the cube" baseline was not run.

## V2.4 decomposition tracking (brick + `trRoulette`, 256 spp, 5 rounds)

| scene | delta ms/spp | decomposition / delta | luminance delta / decomp. | load1 |
|---|---|---|---|---|
| vol-cl-cloud256 | 6.32 | 0.908..0.920 | 0.41612 / 0.41624 | 6.5-7.9 |
| vol-vdb-bunny-cloud | 2.20 | 1.011..1.022 | 0.51896 / 0.51893 | 5.6-6.2 |

The provisional 0.915 holds on cloud256; on the bunny cloud it is 2% slower. The chromatic grid scene was not run.

## V7 media defaults: `forcedNee` + `equiangular` (+ `marchSamples` 4), megakernel, 5 rounds

| scene, spp | default ms/spp | forced + equi | + march 4 | load1 |
|---|---|---|---|---|
| vol-cornell-fog, 256 | 6.50 | 1.196..1.202 | 1.222..1.228 | 5.6-7.5 |
| vol-cornell-smoke, 256 | 2.11 | 1.128..1.143 | 1.154..1.162 | 5.4-6.6 |
| vol-milk-sphere, 256 | 1.62 | 1.122..1.133 | 1.140..1.147 | 3.7-4.5 |
| vol-lamp-fog, 8192 | 0.050 | 1.036..1.047 | 1.391..1.463 | 4.8-5.4 |
| vol-thin-fog-sun, 8192 | 0.066 | 1.110..1.126 | 1.419..1.434 | 6.8-7.5 |

At 256 spp the lamp-fog and thin-sun renders lasted ~30 ms and their ratios ranged 0.87-1.48 (timing noise), so they were
rerun at 8192 spp (`abba-8192.json`). With the settled error ratios (TASKS V7.1: relMSE x0.55-0.68 on Cornell fog), +20% cost
still gives Cornell fog an equal-time error of x0.66-0.82. On the two thin scenes the forced + equiangular pair costs 4-12%, but march 4
costs +40-43% (the per-sample march dominates when paths are short); against the settled ~20x (lamp fog with march) and ~800x
(thin sun at 1 spp) error gains that is still an equal-time win, but march 4 is not free in thin media.

## V4.3 VSPG (plain vs `pt_vspg=1`, 5 rounds)

| scene, spp | plain ms/spp | vspg / plain | load1 |
|---|---|---|---|
| vol-grid-cloud, 256 | 0.645 | 1.016..1.045 | 3.3-4.0 |
| vol-thin-fog-sun, 8192 | 0.066 | 1.207..1.251 | 7.1-7.5 |
| vol-lamp-fog, 8192 | 0.050 | 1.02..1.06 (3 clean rounds; rounds 0 and 3 had a slow plain run, 0.085 / 0.066 ms) | 5.4-6.1 |

(The 256 spp runs of the two thin scenes were too short to time, as in V7.) VSPG costs +2..6% on lamp fog and grid cloud and
+21..25% on thin sun fog, inside the provisional +1..17% except thin sun.

## V4.4 forced / equiangular on grid media (`vol-forced-grid-check`, 8 seeds, 256 spp)

| case | ms per seed | mean (z, rel vs plain) | pixel variance ratio plain / case (masked) |
|---|---|---|---|
| vol-lamp-fog-grid plain | 15.99 | 0.004682 | 1 |
| vol-lamp-fog-grid forced-grid | 16.04 | z 1.40, -2.1% | 34.5 (35.9) |
| vol-lamp-fog-grid equi-grid | 15.90 | z 0.90, -1.4% | 7.6 (50.9) |
| vol-cornell-smoke-grid plain | 530.7 | 0.1442 | 1 |
| vol-cornell-smoke-grid equi-grid | 621.3 | z 0.63, -0.005% | 1.007 (0.987) |

Near the lamp both are free and cut variance 8-50x; on the smoke grid equiangular costs +17% for no variance gain. The
equal-time relMSE on `vol-grid-cloud` and a `vol-cl-*` scene was not run.

## V2.5 adaptive vs uniform (`render-until-check match=1`, 64x64, 6 seeds)

| scene, p99 pixel target | allocated spp | uniform spp at equal p99 | allocated wall | uniform wall (estimated) |
|---|---|---|---|---|
| vol-cl-sph-g85-a99-t50, 0.05 | 1174 | 1088 | 34.3 s | 16.8 s |
| vol-cl-cloud256, 0.15 | 158 | 203 | 7.9 s | 2.4 s |

Allocation needs about as many samples as uniform for the same p99 (sphere: 8% more; cloud: 22% fewer) and takes 2.0-3.3x
the wall time. The uniform wall is the page's estimate at the matched spp, not a separate measured render. Adaptive still
does not pay here.

## P5 wavefront media / spectral (`wf-p5-check timing=1 rounds=7`, wavefront / megakernel ms per spp, median of 7)

| case (spp) | mk ms/spp | plain | inline | ordered | batch | max relpx |
|---|---|---|---|---|---|---|
| vol-cornell-smoke (128) | 2.03 | 1.171 | 0.733 | 0.747 | 0.777 | 1.6e-6 |
| vol-milk-sphere (128) | 1.54 | 2.360 | 0.991 | 1.022 | 1.049 | 1.1e-6 |
| vol-vdb-smoke-brick (128) | 0.75 | 2.259 | 1.066 | 1.129 | 1.172 | 1.1e-6 |
| spectral cornell (16) | 16.1 | 0.534 | 0.504 | 0.636 | 0.602 | 9.0e-7 |
| spectral swatches-refl (16) | 0.19 | 1.233 | 1.733 | 1.900 | 1.862 | 6.6e-7 |
| dispersion-prism (16) | 0.18 | 2.862 | 2.897 | 3.429 | 3.483 | 6.1e-7 |

All variants agree per pixel (no pixel over tolerance). The provisional numbers hold on the smoke cube (plain 1.19, refill
0.72) and spectral Cornell (0.48-0.58). Inline refill is the best wavefront variant on media. Milk and vdb-smoke tie at
best; small scenes (swatches) and the prism lose 1.2-3.5x.

## V5.1d wavefront frame + media (`wf-frame-check timing=1 rounds=7`, ms per 1 spp frame)

| case | mk kernel bytes | max wf kernel bytes | mk ms | wf / mk (min..max) |
|---|---|---|---|---|
| vol-emit | 51,380 | 42,250 | 0.2465 | 1.503..1.517 |
| vol-smoke-lift | 50,880 | 42,130 | 0.717 | 0.634..0.643 |
| vol-cloud | 52,490 | 42,320 | 0.1935 | 0.938..1.028 |

Agreement: 1 pixel of 98,304 on `vol-smoke-lift` (radiance 1.26e-5, diffuse plane 1.0e-4 on 3 px) and 4 diffuse-plane pixels on
`vol-cloud` (2.4e-5) are just over the 1e-5 gate: the known V5.1d residual (TASKS V5.1d: `vol-cloud` 2.2e-5 on 5 px), now seen on
the lifted smoke too. Geometry planes are bit-identical.

## Real-time and ReSTIR (single pages)

**P4.1** (`realtime-check check=timing`, ratio to the plain 1 spp kernel, min..max of 5 rounds):

| scene | plain ms | frame | frame + blue noise | `mode: 'realtime'` |
|---|---|---|---|---|
| cornell 512x512 | 1.473 | 1.050..1.058 | 1.092..1.104 | 1.096..1.103 |
| sponza-white 1280x720 | 26.38 | 1.015..1.018 | 1.023..1.032 | 1.027..1.034 |

Same as provisional.

**P4.1b / P4.2a** (`wf-frame-check timing=1 rounds=7`, wavefront / megakernel per frame):

| case | mk ms | wf / mk (min..max) |
|---|---|---|
| sponza-white 960x540 NEE | 15.59 | 0.889..0.899 |
| sponza-white 960x540 ReSTIR unbiased | 18.16 | 0.928..0.932 |
| cornell 512x512 | 1.625 | 1.159..1.209 |
| many-lights 256x256 ReSTIR unbiased | 0.5145 | 1.142..1.161 |

Inside the provisional band (0.84-0.95 Sponza, 1.02-1.16 small scenes): consistent with V213's frame rule. 1280x720 was not run.

**P4 ReSTIR DI / P4.2c** (`restir-check check=timing`, 1280x720, ms per frame median of 5, ratio = min / NEE min):

| scene | NEE | RIS | + temporal | + spatial (biased) | unbiased | sharedHit | blue noise |
|---|---|---|---|---|---|---|---|
| sponza-white | 49.0 | 1.068 | 1.110 | 1.186 | 1.104 | 1.120 | 1.194 |
| many-lights-256 | 2.57 | 1.241 | 1.334 | 1.512 | 1.468 | 1.402 | 1.498 |

sharedHit / traced (both RIS + t + s): 0.943 Sponza, 0.929 many-lights (provisional x0.942). Blue noise / Owen: 1.005 / 0.998.
The V-buffer variant is not in the page.

**P4.5 / P4.7 / P5.1 / P5.6** (`realtime-pipeline scenarios=timing`, sponza-white 960x540 -> 1920x1080, maxDepth 2, ms median
(min..max) of 5 rounds; the two schedulers ran as two pages, not interleaved per round):

| trace mode | mk trace | mk frame (fps) | wf trace | wf frame (fps) | wf / mk trace |
|---|---|---|---|---|---|
| NEE | 15.67 (15.5..15.8) | 19.5 (51) | 15.26 (14.1..17.1) | 28.2 (36) | 0.97 |
| ReSTIR DI | 16.47 (16.4..17.2) | 26.3 (38) | 23.79 (22.8..24.2) | 33.5 (30) | 1.44 |
| DI + GI biased | 23.42 (21.2..25.9) | 36.8 (27) | 32.17 (27.6..36.6) | 46.6 (21) | 1.37 |
| DI + GI unbiased | 32.26 (31.1..32.8) | 40.3 (25) | 41.83 (39.6..42.2) | 67.0 (15) | 1.30 |

Megakernel stages (NEE): pack 1.69, SVGF + composite 1.13, upscale 0.76, tonemap 0.04 ms. The wavefront page's whole frame is
7-25 ms more than its stages add up to (NEE: 20.9 summed vs 28.2 queued) and its post is ~2x the megakernel's: queued frames
overlap worse there; worth a look before trusting the wavefront frame column. GI's trace share on the megakernel: +7 ms biased,
+16 ms unbiased over DI. The P5.6 provisional (wf / mk 1.00-1.14 with GI) does not hold here: 1.30-1.37.
OIDN preview chain (P4.7, 640x360, maxDepth 4, OIDN small): frame 32.8 ms (29.7..40.4), 30 fps; trace 16.4, denoise + composite
14.0, pack 0.75 ms.

**P5.rc4** (`rc-cascades`, sponza-white 1280x720, probe spacing 0.25, ms median; Split RC form `split.*`):

| | c0 | c1 | c2 | c3 | split |
|---|---|---|---|---|---|
| probes-hit / plain-clamped | 1.010 | 0.984 | 1.026 | 1.004 | (rays-hit 0.973) |
| plain-full / plain-clamped | 2.06 | 1.72 | 1.19 | 1.55 | 1.00 |
| same, `dirs0=8`: probes-hit / clamped | 1.038 | 1.006 | 1.004 | 0.999 | (0.999) |
| `dirs0=8`: full / clamped | 2.35 | 1.76 | 1.20 | 1.59 | 1.01 |

The interval machinery costs 0-4% over the plain clamped walk; short intervals are 1.2-2.4x faster than full length. Both
claims in split-rc-tracing.md section 5 hold.

**X5** (`downsampler-check`, M5 only; the row asks for the RTX 3060 / Intel UHD): mips within 2 codes everywhere (level 0
exact, `mips_default_vs_explicit_differing_bytes` 0); GPU chain 8.2x (1024^2) / 8.4x (2048^2) cheaper than the CPU chain
(30.5 vs 249 ms, 125 vs 1047 ms). Exposure reduction matches to 6e-8.

## V1.8 Mac reference (`kernel-compile`, 5 repeats, literal nonce)

Megakernel (`prefix=plain,vol-,spectral`), load1 4.5, cold ms per repeat flat (no cache answered), warm 0.7-1.1 ms:

| kernel | KB | cold median (min..max) ms |
|---|---|---|
| plain | 41.6 | 207 (205..245) |
| vol-homogeneous | 45.3 | 281 (280..291) |
| vol-homogeneous-forced / -equi | 46.3 / 47.1 | 331 / 338 |
| vol-grid | 46.9 | 364 (363..366) |
| vol-grid-forced / -equi / -equi-march4 | 47.9 / 48.7 / 48.9 | 431 / 441 / 445 |
| vol-grid-equi-march4-fgrid | 49.4 | 417 (411..422) |
| vol-grid-brick | 50.1 | 443 (442..448) |
| vol-grid-brick-decomp | 50.7 | 451 (448..471) |
| spectral / -env / -tex | 46.7 / 49.4 / 50.3 | 363 / 409 / 457 |
| spectral-pbr-albedo / -stochastic | 56.2 / 56.6 | 520 / 499 |

Wavefront (`set=wavefront prefix=wf-plain,wf-vol-`), load1 7.2, cold flat except the first repeat of a few kernels, warm 0.2-0.8 ms.
The args kernel has no float literal, so it gets the comment nonce and its 0.7 ms is a cache answer (listed in `notes.commentNonce`).

| kernel set | camera | intersect | shade | shadow | sum | largest kernel |
|---|---|---|---|---|---|---|
| wf-plain | 18.7 | 54.6 (12.7 KB) | 140.6 (30.4 KB) | 45.6 | 259 | 141 ms |
| wf-vol-homogeneous | 19.4 | 76.5 (17.6 KB) | 174.7 (35.2 KB) | 65.5 | 336 | 175 ms |
| wf-vol-ratio | 19.6 | 76.7 | 173.7 | 69.0 | 339 | 174 ms |
| wf-vol-grid | 19.3 | 139.8 (18.4 KB) | 172.3 | 130.9 | 462 | 172 ms |
| wf-vol-grid-brick | 19.5 | 164.9 (20.9 KB) | 173.4 | 149.4 | 507 | 173 ms |
| wf-vol-grid-brick-decomp | 19.4 | 171.0 (21.6 KB) | 172.8 | 149.0 | 512 | 173 ms |
| wf-vol-tex | 19.7 | 76.3 | 274.0 (38.8 KB) | 65.5 | 436 | 274 ms |

On Metal the wavefront set costs about as much in total as the matching megakernel (brick-decomp 512 vs 451 ms; homogeneous 336 vs
281 ms), and no single wavefront kernel is above 274 ms. Compile cost here is close to linear in size; the Windows question (W18) is
whether DXC is superlinear past ~45 KB, which this Mac cannot show. Note the grid kernels: intersect and shadow cost 2x their
homogeneous twins at +1-4 KB (the trilinear grid walk), so bytes alone underpredict grid kernels.

## Not run in this pass (still in the Queue)

P1.5 (reflections checkout), P2.3 (not runnable), the long `--only p3` / matrix rows (P3.1, P3.2, R2 / P3.7, R2b, P3.6, P3.11,
P2.32b, R1, P3.3 group, X1.i), the headed NRC rows (N12.4, N7a, N11, N11.7), and the parts named above (P5 media without the
cube, V2.4 chromatic grid, V4.4 equal-time relMSE, P4.1b at 1280x720, P4.2c V-buffer, X5 on Windows).
