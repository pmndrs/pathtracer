# Blue-noise generator: report (TASKS P0.17)

Our own blue-noise textures, because the NVIDIA STBN assets and generator are non-commercial (D16, DIVERGES V3).
Everything here is original code written from papers; no third-party generator code was read or ported
(EA's FAST repo included). Rerun:

```sh
pnpm bluenoise:generate   # writes packages/raytracer/assets/bluenoise/ (PNG + manifest committed, .f32 git-ignored)
pnpm bluenoise:report     # prints the tables below, writes A/B PFMs to .cache/bluenoise-ab/
```

## What is generated

| family | method | sizes | channels | file names |
|---|---|---|---|---|
| `vc2d` | Ulichney 1993 void-and-cluster, Gaussian σ = 1.5, toroidal | 64², 128² | scalar, vec2 (two independent runs) | `vc2d-scalar-64`, `vc2d-vec2-128`, ... |
| `st` | void-and-cluster on a 3D torus; energy = spatial Gaussian (σ = 1.5) inside a slice + temporal Gaussian (σ = 1.5 slices) along a pixel's own time axis (the STBN / FAST "separate" form). Ranks over the whole volume | 64²×32, 128²×64 | scalar, vec2 (two independent runs) | `st-scalar-64x32`, `st-vec2-128x64`, ... |
| `gf` | vec2 by pixel swapping, Georgiev & Fajardo 2016 energy (σᵢ = 2.1, σₛ = 1, d = 2), stratified shuffled start, 200 proposals per pixel, greedy | 64², 128² | vec2 only | `gf-vec2-64`, `gf-vec2-128` |

Formats: 16-bit PNG (grey for scalar, RGB with B = 0 for vec2; volumes as an atlas of `atlasColumns` slices per
row), raw little-endian float32 `.f32`, and `manifest.json`. Runtime loaders in `src/noise/bluenoise.ts`:
`decodeBlueNoisePng` (own PNG decode via `DecompressionStream`, never a canvas, so 16 bits survive) and
`parseBlueNoise` (raw `.f32`), both to the same interleaved `Float32Array`. PNG and `.f32` agree to 0.5/65535.
The scalar texture of `vc2d`/`st` is channel 0 of the matching vec2 texture. Committed size: about 8.3 MB of PNG
(the `.f32` files are another 14 MB and are git-ignored).

## Timings

Apple M5 Pro, Node 22.22.2, one worker thread per job, 2026-11-13, machine shared with other agents (load average
9-13 on 18 cores; numbers are indicative, not a benchmark).

| texture | time per job |
|---|---|
| vc2d 64² | 30-40 ms |
| vc2d 128² | 80-95 ms |
| st 64²×32 | 1.1-1.3 s |
| st 128²×64 | 15-18 s |
| gf 64² | 6.7 s |
| gf 128² | 27 s |
| all 10 textures, wall | 27 s (gf-128 is the critical path) |

128² classic takes well under a second. The 3D volume and the swap optimiser take tens of seconds: fine for an
offline asset build, not for runtime generation. The void-and-cluster inner loop is an incremental arg-max/arg-min
over a 32-wide tournament tree (`extrema.ts`), so each rank costs a window update, not a full scan.

## 1. Spatial spectrum

Radially averaged power spectrum, normalised so white noise reads 1; columns are bands of the Nyquist frequency.
`st` is the average over its slices.

| texture | 0-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| white 64² | 1.16 | 0.95 | 1.11 | 0.91 | 0.92 | 1.05 | 1.06 | 1.05 | 1.01 | 0.97 |
| vc2d-64 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.13 | 0.69 | 1.33 | 1.59 | 1.52 |
| vc2d-64 (vec2 .y) | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.16 | 0.63 | 1.45 | 1.77 | 1.57 |
| gf-64 .x | 0.01 | 0.01 | 0.02 | 0.05 | 0.17 | 0.75 | 1.34 | 1.35 | 1.19 | 1.25 |
| gf-64 .y | 0.01 | 0.01 | 0.02 | 0.05 | 0.20 | 0.72 | 1.10 | 1.30 | 1.41 | 1.37 |
| st-64x32 (avg of slices) | 0.00 | 0.00 | 0.00 | 0.02 | 0.08 | 0.42 | 1.10 | 1.42 | 1.47 | 1.38 |
| white 128² | 0.83 | 1.00 | 1.02 | 0.95 | 1.04 | 1.00 | 0.99 | 1.03 | 1.01 | 1.00 |
| vc2d-128 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.12 | 0.59 | 1.36 | 1.60 | 1.63 |
| vc2d-128 (vec2 .y) | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.12 | 0.57 | 1.31 | 1.64 | 1.61 |
| gf-128 .x | 0.01 | 0.01 | 0.02 | 0.06 | 0.21 | 0.68 | 1.13 | 1.34 | 1.38 | 1.34 |
| gf-128 .y | 0.01 | 0.01 | 0.02 | 0.05 | 0.19 | 0.66 | 1.15 | 1.35 | 1.34 | 1.33 |
| st-128x64 (avg of slices) | 0.00 | 0.00 | 0.00 | 0.02 | 0.08 | 0.38 | 1.03 | 1.40 | 1.45 | 1.41 |

All of them have the low-frequency hole. `vc2d` is the deepest; each `st` slice gives up a little (0.08 at
0.4-0.5 Nyquist) to make room for the temporal constraint; the `gf` channels give up more (about 0.2) for joint 2D
quality.

## 2. Rank uniformity

| texture | global permutation | worst slice: max histogram deviation (16 bins) | worst pixel over time: max deviation (4 bins) | mean pixel over time: max deviation (4 bins) |
|---|---|---|---|---|
| vc2d-64 | yes | 0.000 | - | - |
| vc2d-128 | yes | 0.000 | - | - |
| white 64²x32 | - | 0.195 | 1.375 | 0.423 |
| st-64x32 | yes | 0.082 | 0.625 | 0.161 |
| white 128²x64 | - | 0.107 | 0.938 | 0.298 |
| st-128x64 | yes | 0.045 | 0.438 | 0.117 |

`vc2d` and both `gf` channels are exact permutations of k/(N-1) (unit tests check this, also after PNG decode).
`st` is exact over the volume and about 2.5x more uniform than white noise per slice and per pixel over time, but
not exact: a pixel's 32 or 64 values are not stratified.

## 3. Per-pixel temporal spectrum

Power spectrum of each pixel's sequence over the slices, averaged over pixels, white = 1; columns are bands of the
temporal Nyquist frequency. The `vc2d` rows are what a renderer without an ST texture does: frame k reads
`fract(v + k * 0.618...)`.

| texture | 0-0.125 | 0.125-0.25 | 0.25-0.375 | 0.375-0.5 | 0.5-0.625 | 0.625-0.75 | 0.75-0.875 | 0.875-1 |
|---|---|---|---|---|---|---|---|---|
| white 64²x32 | 1.00 | 1.01 | 1.00 | 0.99 | 1.00 | 1.00 | 1.01 | 0.99 |
| vc2d-64 + golden ratio, 32 frames | 0.18 | 0.39 | 0.58 | 1.11 | 0.37 | 4.39 | 0.57 | 0.44 |
| st-64x32 | 0.03 | 0.07 | 0.26 | 0.84 | 1.47 | 1.76 | 1.96 | 2.14 |
| white 128²x64 | 0.99 | 1.00 | 0.99 | 1.00 | 1.00 | 1.01 | 1.01 | 1.00 |
| vc2d-128 + golden ratio, 64 frames | 0.15 | 0.30 | 0.58 | 1.32 | 0.26 | 2.93 | 2.02 | 0.44 |
| st-128x64 | 0.03 | 0.06 | 0.22 | 0.74 | 1.41 | 1.73 | 1.95 | 2.13 |

`st` is a clean high-pass in time (0.03 in the lowest band). The golden-ratio sequence also has much less
low-frequency power than white, but as spikes, not a smooth high-pass.

## 4. Integration A/B at 1-4 spp

Integrand per pixel: a soft-shadow-like step. 1D: a linear light, visible where `u < c(x, y)` (scalar textures).
2D: a unit-square area light cut by an occluder edge at 0.4 rad, visible where `u cos θ + v sin θ < c'(x, y)`
(vec2 textures). `c` varies smoothly and toroidally over a 128² image, so the image has lit, shadowed and penumbra
regions. Exact per-pixel answers (analytic / 4096-point quadrature). Samples per pixel k = 0..spp-1: white = hash;
Owen-Sobol = `owenSobol2D(k, hash(pixel))` (`src/sampling/owen-sobol.ts`, after Burley 2020); `vc2d`/`gf` = texel
shifted by k times the golden ratio (R1 in 1D, R2 in 2D); `st` = slice k. 16 trials of random texture offsets / seeds.

Each cell: **RMSE / RMSE after a Gaussian blur σ = 2 px (stand-in for a denoiser) / error power below 0.25 Nyquist
(white = 1)**.

**1D integrand, scalar textures**

| sampler | 1 spp | 2 spp | 4 spp |
|---|---|---|---|
| white | 0.4067 / 0.0567 / 0.97 | 0.2874 / 0.0402 / 0.98 | 0.2032 / 0.0285 / 0.98 |
| owen-sobol | 0.4069 / 0.0578 / 1.01 | 0.1860 / 0.0265 / 1.03 | 0.0948 / 0.0134 / 1.01 |
| vc2d-64 | 0.4071 / 0.0134 / 0.05 | 0.2127 / 0.0105 / 0.12 | 0.1309 / 0.0078 / 0.17 |
| vc2d-128 | 0.4072 / 0.0133 / 0.05 | 0.2128 / 0.0105 / 0.12 | 0.1310 / 0.0078 / 0.17 |
| st-64x32 | 0.4072 / 0.0144 / 0.06 | 0.2120 / 0.0101 / 0.11 | 0.1202 / 0.0070 / 0.17 |
| st-128x64 | 0.4071 / 0.0144 / 0.06 | 0.2117 / 0.0101 / 0.11 | 0.1134 / 0.0070 / 0.19 |

**2D integrand, vec2 textures**

| sampler | 1 spp | 2 spp | 4 spp |
|---|---|---|---|
| white | 0.3689 / 0.0515 / 0.97 | 0.2606 / 0.0367 / 1.00 | 0.1842 / 0.0259 / 0.98 |
| owen-sobol | 0.3688 / 0.0519 / 0.99 | 0.1885 / 0.0267 / 1.02 | 0.1109 / 0.0158 / 1.03 |
| vc2d-64 | 0.3692 / 0.0304 / 0.34 | 0.2593 / 0.0236 / 0.42 | 0.1304 / 0.0176 / 0.91 |
| vc2d-128 | 0.3698 / 0.0309 / 0.35 | 0.2579 / 0.0237 / 0.42 | 0.1309 / 0.0173 / 0.88 |
| gf-64 | 0.3701 / 0.0209 / 0.16 | 0.2577 / 0.0200 / 0.29 | 0.1307 / 0.0149 / 0.64 |
| gf-128 | 0.3693 / 0.0204 / 0.15 | 0.2589 / 0.0200 / 0.30 | 0.1311 / 0.0153 / 0.68 |
| st-64x32 | 0.3690 / 0.0298 / 0.33 | 0.2080 / 0.0222 / 0.58 | 0.1322 / 0.0163 / 0.79 |
| st-128x64 | 0.3693 / 0.0304 / 0.34 | 0.2080 / 0.0226 / 0.59 | 0.1281 / 0.0163 / 0.81 |

Reading it:

- **Raw per-pixel RMSE at 1 spp is the same for everything** (0.37-0.41): blue noise moves error to high
  frequencies, it does not remove it. The gain shows after a low-pass: at 1 spp the blurred error is **4.3x lower
  than white / Owen-Sobol in 1D** (0.0133 vs 0.0567) and 1.7x lower in 2D with independent vec2 channels.
- **Two independent scalar channels are a weak vec2.** Each channel is blue, but a 2D integrand thresholds a
  combination of both, and that pattern keeps 34% of white's low-frequency error. The jointly optimised `gf` keeps
  15% and is 2.5x better than white after the blur (0.0204). Mapping a scalar rank to a 2D low-discrepancy point
  (Sobol or R2 indexed by rank) was tried and is worse than white (threshold low band 1.04-1.09), so it is not shipped.
- **At 4 spp Owen-Sobol has the lowest raw RMSE** (0.095 in 1D vs 0.11-0.13), as expected of a (0,2)-sequence,
  and its error is white. Blue noise still wins after the blur (0.0070-0.0078 vs 0.0134 in 1D), but its spectrum
  whitens with spp (low band 0.05 → 0.17 in 1D, 0.15 → 0.68 for `gf` in 2D): the golden-ratio or slice-to-slice
  extension is not blue across samples within a pixel.
- `st` uses its slices as samples; at 2 spp in 2D it has the best raw RMSE of the texture samplers (0.208), because
  consecutive slices are anti-correlated per pixel.
- 64² and 128² are indistinguishable on this test; 128² only matters for visible tiling.

The bench viewer can compare the images: `.cache/bluenoise-ab/reference-2d.pfm` and
`<sampler>-2d-{1,4}spp.pfm`, `<sampler>-2d-mean64.pfm` (linear, single channel, git-ignored).

## 5. Temporal accumulation: `st` vs `vc2d` + golden ratio

2D integrand, 1 spp per frame, 64 frames. Running mean (progressive) and an exponential moving average α = 0.1
(TAA-like history). Cells: RMSE / RMSE after the σ = 2 blur.

| sampler | mean of 1 | mean of 4 | mean of 16 | mean of 64 | EMA α=0.1 @64 |
|---|---|---|---|---|---|
| white | 0.3689 / 0.0515 | 0.1842 / 0.0259 | 0.0924 / 0.0130 | 0.0463 / 0.0065 | 0.0846 / 0.0120 |
| owen-sobol | 0.3688 / 0.0519 | 0.1109 / 0.0158 | 0.0394 / 0.0056 | 0.0142 / 0.0020 | 0.0408 / 0.0057 |
| vc2d-64 | 0.3692 / 0.0304 | 0.1304 / 0.0176 | 0.0602 / 0.0063 | 0.0240 / 0.0029 | 0.0574 / 0.0060 |
| vc2d-128 | 0.3698 / 0.0309 | 0.1309 / 0.0173 | 0.0605 / 0.0063 | 0.0241 / 0.0031 | 0.0573 / 0.0061 |
| gf-64 | 0.3701 / 0.0209 | 0.1307 / 0.0149 | 0.0604 / 0.0078 | 0.0239 / 0.0033 | 0.0571 / 0.0072 |
| gf-128 | 0.3693 / 0.0204 | 0.1311 / 0.0153 | 0.0607 / 0.0082 | 0.0241 / 0.0033 | 0.0572 / 0.0071 |
| st-64x32 | 0.3690 / 0.0298 | 0.1322 / 0.0163 | 0.0629 / 0.0085 | 0.0334 / 0.0045 | 0.0627 / 0.0075 |
| st-128x64 | 0.3693 / 0.0304 | 0.1281 / 0.0163 | 0.0603 / 0.0084 | 0.0307 / 0.0042 | 0.0608 / 0.0076 |

**Our `st` does not beat a 2D texture animated by the golden ratio** on this integrand: level up to ~4 frames,
1.3-1.4x worse at 64 frames of plain accumulation, 6-9% worse under the EMA. The golden-ratio shift makes each
pixel's sequence a 1D low-discrepancy sequence (stratified over time); `st` only makes it high-pass (not stratified,
section 2). `gf` wins the first frames but its blurred error falls behind `vc2d` from ~16 frames on (the R2 shift of
a jointly optimised pair stays less blue than R1/R2 on independent dither arrays). Owen-Sobol converges fastest once
more than a handful of frames are averaged, which matches research note 04: Sobol for progressive/reference, blue
noise only where a spatial filter follows within a few frames.

`st` parameter sweep (64²×32 scalar, 1D integrand, one seed; mean-of-32 and EMA cells are RMSE / blurred):

| temporal weight / σₜ | slice low band (<0.5 Nyq) | temporal low (3 lowest bins) | 1 frame blurred | mean of 32 | EMA |
|---|---|---|---|---|---|
| `vc2d` + golden ratio | (0.00) | - | 0.0171 | 0.0176 / 0.0019 | 0.0385 / 0.0033 |
| 1 / 1.5 (shipped) | 0.020 | 0.10 | 0.0171 | 0.0257 / 0.0027 | 0.0386 / 0.0038 |
| 0.25 / 1.5 | 0.009 | 0.23 | 0.0147 | 0.0330 / 0.0030 | 0.0460 / 0.0036 |
| 4 / 1.5 | 0.052 | 0.06 | 0.0189 | 0.0229 / 0.0025 | 0.0350 / 0.0037 |
| 1 / 1 | 0.014 | 0.18 | 0.0167 | 0.0315 / 0.0028 | 0.0440 / 0.0038 |
| 1 / 2.5 | 0.023 | 0.09 | 0.0175 | 0.0205 / 0.0024 | 0.0352 / 0.0035 |
| 2 / 1 | 0.020 | 0.14 | 0.0166 | 0.0302 / 0.0029 | 0.0422 / 0.0038 |

A wider temporal σ (2.5) helps convergence at a small spatial cost but still trails the golden-ratio baseline on
plain accumulation. Single seed, so differences under ~10% are noise.

## Caveats

- One integrand family (steps). Smooth integrands favour Owen-Sobol more; real shading through OIDN or a temporal
  upscaler is the test that matters (research note 04: "does OIDN benefit from blue noise at all").
- The blur is a fixed Gaussian; FAST (Donnelly et al. 2024) shows noise should be tuned to the actual filter.
- `gf` uses greedy swaps from a random start. Prototype runs at 128² (threshold-pattern low band, lower is better):
  independent channels 0.39, refining the independent pair 0.20-0.25, random start 100 proposals/pixel 0.17,
  300 proposals/pixel 0.145; shipped 200 gives 0.15-0.16 in the A/B. Near its plateau, not a global optimum.
  Toroidal distance in sample space made it worse (0.44-0.53) and is not used.
- No `gf`-style joint vec2 for the spatiotemporal family yet (cost: about 64x the 2D swap run).

## Suggestion (for the maintainer, not a decision)

Ship `vc2d` (scalar) and `gf` (vec2) at 128², animated per frame by the golden-ratio (R1/R2) shift, as the default
real-time blue-noise source; keep `st` generated and selectable; keep Owen-Sobol for reference mode.

- Pros: best or equal on every test here at 1-4 spp after a filter (2D: 2.5x better than white with `gf` vs 1.7x
  with `st`); converges like a low-discrepancy sequence under accumulation; 2D textures are tiny (64 KB scalar,
  128 KB vec2 as float32) against 4-8 MB for `st-128x64`; one texture fetch either way.
- Cons: only tested with a Gaussian blur and a running/EMA mean on step integrands; STBN's published advantage is
  under TAA with reprojection and neighbourhood clamping, which this test does not model; `gf` is not
  spatiotemporal at all and loses to `vc2d` + R2 after ~16 accumulated frames.
- The A/B that settles it: the P4 real-time path at 1 spp through our temporal accumulation + OIDN (and an
  à-trous filter), `vc2d`/`gf` + golden ratio vs `st` vs Owen-Sobol vs white, error against an 8k spp reference
  over 64 frames with camera motion, in the bench viewer.
