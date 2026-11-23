# Participating media (P5): results

Run 2026-11-18 on the M5 Pro (Chrome / Metal, headless), `pt-megakernel` megakernel, 4096 spp, references pbrt-v4 `volpath` and Mitsuba 3 `llvm_ad_rgb` `volpath` at 4096 spp (their sidecars are
`bench/references/vol-*/`). Design and the estimators: [docs/internal/design/volumes.md](../../docs/internal/design/volumes.md); divergences V240-V247; tasks P5.v1-P5.v11.
**No timing here is a result**: the machine ran at load 3-60 with the GPU lock shared by six agents; wall times are provisional and unpaired.

## Scenes against pbrt and Mitsuba (relMSE eps 1e-2; floor = pbrt-vs-Mitsuba relMSE; non-finite pixels 0 everywhere)

| scene | pbrt vs Mitsuba (floor) | ours vs pbrt | ours vs Mitsuba | ours mean RGB |
|---|---|---|---|---|
| `vol-cornell-smoke` (homogeneous smoke cube, Cornell box) | 6.87e-4, relBias -0.97% | 7.16e-4, +1.01% | 1.01e-4, +0.03% | 0.21701 0.15246 0.06319 |
| `vol-milk-sphere` (scattering inside smooth glass, sun + sky) | 6.16e-5, -0.01% | 6.57e-5, -0.00% | 5.18e-5, -0.01% | 0.52371 |
| `vol-grid-cloud` (48^3 procedural grid) | 6.60e-5, 0.00% | 5.90e-5, +0.01% | 5.20e-5, +0.01% | 0.56450 |
| `vol-vdb-smoke` (WebVDB `smoke.vdb`, a real VDB, 60x115x60) | 4.84e-5, 0.00% | 3.82e-5, 0.00% | 3.71e-5, -0.00% | 0.50258 |

Every cell is at or below the pbrt-vs-Mitsuba floor, i.e. the difference is the references' own noise. The 1% of the Cornell scene against pbrt is the colour of the walls (the plain `cornell` references
already differ by pbrt -0.69%: pbrt uplifts the chromatic reflectances spectrally), not the medium; against Mitsuba (RGB, like us) the mean agrees to 0.03%.

## Settings A/B (same images, two settings; V241, V243)

| A / B | scene | relMSE vs Mitsuba A | relMSE vs Mitsuba B | mean A / B |
|---|---|---|---|---|
| `volumeShadow` analytic / ratio (homogeneous shadow transmittance) | `vol-cornell-smoke` | 1.01e-4 | 1.06e-4 | 0.21701 / 0.21700 |
| same | `vol-milk-sphere` | 5.18e-5 | 5.25e-5 | 0.52371 / 0.52373 |
| `volumeMajorant` global / brick (grid) | `vol-grid-cloud` | 5.20e-5 (pbrt 5.90e-5) | 5.30e-5 (pbrt 6.15e-5) | 0.56450 / 0.56448 |
| same | `vol-grid-slab` (closed form 0.49703) | 0.49684 | 0.49666 | |

Ratio tracking through a homogeneous medium costs about 5% noise on the smoke cube (tau 2.2) and nothing measurable on the milk sphere (the shadow rays are blocked by the glass); the closed form stays the default.
The brick majorant gives the same mean and noise as the global one (the same estimator with a different majorant) and, on this dense 48^3 cloud (peak sigma_t 8.4 per metre, path length about 1 m),
was NOT faster in a provisional, unpaired wall-clock look (2.87 s vs 2.45 s for 4096 spp, load 5-55): the DDA pays off for sparse grids with a dense core, not here. Defaults stay `analytic` and `global`
until the quiet pass (QUIET-PASS row P5 media).

### Closed-form scenes: ours against pbrt / Mitsuba and the analytic value (`pt-agreement.ts --report vol-check`)

| scene | spp | relMSE vs Mitsuba (RGB) | relMSE vs pbrt-v4 | pbrt-vs-Mitsuba floor | relBias vs Mitsuba / pbrt | block-mean max / rms vs Mitsuba | image mean RGB | non-finite |
|---|---|---|---|---|---|---|---|---|
| vol-absorb-slab | 4096 | 4.99e-4 | 1.39e-3 | 1.26e-3 | 0.01% / 2.62% | 0.00% / 0.00% | 0.74076 0.44926 0.20182 | 0 |
| vol-furnace-sphere | 4096 | 4.67e-5 | 2.05e-5 | 2.75e-5 | 0.00% / 0.00% | 0.01% / 0.01% | 1.00004 1.00004 1.00004 | 0 |
| vol-furnace-sphere-back | 4096 | 4.98e-5 | 2.06e-5 | 2.88e-5 | 0.03% / 0.01% | 0.03% / 0.03% | 1.00015 1.00015 1.00015 | 0 |
| vol-slab-single | 4096 | 4.94e-5 | 6.82e-5 | 4.70e-5 | 0.02% / -0.01% | 0.00% / 0.00% | 0.84908 0.84908 0.84908 | 0 |
| vol-slab-single-oblique | 4096 | 8.31e-5 | 1.23e-4 | 7.66e-5 | 0.00% / 0.02% | 0.00% / 0.00% | 0.74924 0.74924 0.74924 | 0 |
| vol-grid-slab | 4096 | 3.11e-4 | 4.19e-4 | 3.67e-4 | 0.93% / 0.00% | 0.00% / 0.00% | 0.49684 0.49684 0.49684 | 0 |
| vol-camera-medium | 4096 | 2.33e-4 | 8.01e-3 | 8.09e-3 | -0.04% / 9.61% | 0.00% / 0.00% | 0.74074 0.20323 0.05575 | 0 |

#### Convergence (relMSE vs Mitsuba / vs pbrt) and wall time (best effort, load 3-60, shared GPU: not a timing result)

**vol-absorb-slab**: 64 spp 1.97e-2 / 2.19e-2 (10 ms); 256 spp 5.15e-3 / 6.31e-3 (28 ms); 1024 spp 1.45e-3 / 2.43e-3 (96 ms); 4096 spp 4.99e-4 / 1.39e-3 (396 ms)

- expected 0.7408182206817179 (mean only, tol 0.001): mean 0.74076, mean rel dev 3.62e-4 -> PASS
- block means (40 px) vs pbrt: max 0.00%, rms 0.00%; the references against each other: max 0.00%, rms 0.00%
- plateau (bias or reference floor): vs Mitsuba false, vs pbrt true

**vol-furnace-sphere**: 64 spp 1.39e-3 / 1.37e-3 (43 ms); 256 spp 3.58e-4 / 3.36e-4 (139 ms); 1024 spp 1.09e-4 / 8.08e-5 (543 ms); 4096 spp 4.67e-5 / 2.05e-5 (2095 ms)

- expected 1 (mean only, tol 0.001): mean 1.00004, mean rel dev 3.84e-5 -> PASS
- block means (40 px) vs pbrt: max 0.00%, rms 0.00%; the references against each other: max 0.01%, rms 0.01%

**vol-furnace-sphere-back**: 64 spp 1.44e-3 / 1.40e-3 (53 ms); 256 spp 3.68e-4 / 3.42e-4 (177 ms); 1024 spp 1.13e-4 / 8.56e-5 (673 ms); 4096 spp 4.98e-5 / 2.06e-5 (2660 ms)

- expected 1 (mean only, tol 0.001): mean 1.00015, mean rel dev 1.47e-4 -> PASS
- block means (40 px) vs pbrt: max 0.01%, rms 0.01%; the references against each other: max 0.02%, rms 0.02%

**vol-slab-single**: 64 spp 2.76e-3 / 2.77e-3 (13 ms); 256 spp 6.30e-4 / 6.47e-4 (39 ms); 1024 spp 1.53e-4 / 1.72e-4 (114 ms); 4096 spp 4.94e-5 / 6.82e-5 (361 ms)

- expected 0.8491004504449329 (mean only, tol 0.0002): mean 0.84908, mean rel dev 2.62e-5 -> PASS
- block means (40 px) vs pbrt: max 0.00%, rms 0.00%; the references against each other: max 0.00%, rms 0.00%

**vol-slab-single-oblique**: 64 spp 4.60e-3 / 4.67e-3 (14 ms); 256 spp 1.09e-3 / 1.12e-3 (43 ms); 1024 spp 2.67e-4 / 3.23e-4 (129 ms); 4096 spp 8.31e-5 / 1.23e-4 (406 ms)

- expected 0.7491837681111222 (mean only, tol 0.0002): mean 0.74924, mean rel dev 7.32e-5 -> PASS
- block means (40 px) vs pbrt: max 0.00%, rms 0.00%; the references against each other: max 0.00%, rms 0.00%

**vol-grid-slab**: 64 spp 1.21e-2 / 1.22e-2 (8 ms); 256 spp 3.13e-3 / 3.22e-3 (20 ms); 1024 spp 8.77e-4 / 9.46e-4 (53 ms); 4096 spp 3.11e-4 / 4.19e-4 (171 ms)

- expected 0.4970286437828962 (mean only, tol 0.001): mean 0.49684, mean rel dev 3.84e-4 -> PASS
- block means (40 px) vs pbrt: max 0.00%, rms 0.00%; the references against each other: max 0.00%, rms 0.00%

**vol-camera-medium**: 64 spp 7.99e-3 / 1.63e-2 (9 ms); 256 spp 2.14e-3 / 1.00e-2 (23 ms); 1024 spp 6.03e-4 / 8.46e-3 (49 ms); 4096 spp 2.33e-4 / 8.01e-3 (149 ms)

- expected 0.7408182206817178 (mean only, tol 0.001): mean 0.74074, mean rel dev 6.12e-4 -> PASS
- block means (40 px) vs pbrt: max 0.00%, rms 0.00%; the references against each other: max 0.00%, rms 0.00%
- plateau (bias or reference floor): vs Mitsuba false, vs pbrt true



## What did not match, and why

- A chromatic absorber or smoke in pbrt (`vol-absorb-slab`, `vol-camera-medium`): pbrt reads 0.718 / 9.6% off the closed form (spectral uplift of an RGB sigma, V62). The references of the scenes above use grey sigma.
- `vol-grid-slab` in Mitsuba: 0.49222 against the closed form 0.49703 (-1.0%): `gridvolume` clamps outside the cell centres where pbrt and we read zero, and this analytic grid has no zero border (V246). pbrt `uniformgrid` 0.49702.
- Mitsuba's `heterogeneous` medium emits a rare NaN sample (about 1 pixel per 3000 per 512 spp on the cloud): `mitsuba_render.py` drops that pixel's chunk and reports `nan_pixel_chunks`; the 4096 spp references of the cloud have no non-finite pixel.


## Cloud regime (P5.v13-v18, 2026-11-21, docs/internal/CLOUDS-HANDOFF.md)

Same machine and references (pbrt-v4 `volpath`, Mitsuba 3 `llvm_ad_rgb` `volpath`, 4096 spp each, ours 4096 spp `pt-megakernel`), grey sigma, maxDepth 512, no Russian roulette, sun + sky of 0.4.
relMSE here is on the channel-mean image (eps 1e-2); "at floor" = ours vs Mitsuba at most 1.25x the pbrt-vs-Mitsuba relMSE. Load 27-120 all day: no timing is a result.

### Spheres (diameter 1 m, optical diameter t, sun partly behind) and slabs (1 m, seen from above `top` or below `base`)

| scene | pbrt vs Mitsuba (floor) | ours vs pbrt | ours vs Mitsuba | relBias vs Mitsuba / pbrt | ours mean | verdict |
|---|---|---|---|---|---|---|
| `vol-cl-sph-g80-a99-t10` | 1.00e-03 | 8.39e-04 | 8.10e-04 | +0.07% / -0.07% | 0.55102 | at floor |
| `vol-cl-sph-g80-a99-t50` | 5.48e-04 | 5.36e-04 | 4.27e-04 | +0.07% / -0.02% | 0.40604 | at floor |
| `vol-cl-sph-g80-a99-t200` | 2.77e-04 | 2.14e-04 | 2.47e-04 | +0.27% / -0.04% | 0.35444 | at floor |
| `vol-cl-sph-g80-a100-t10` | 1.00e-03 | 7.77e-04 | 7.92e-04 | +0.07% / -0.09% | 0.57409 | at floor |
| `vol-cl-sph-g80-a100-t50` | 6.17e-04 | 4.98e-04 | 4.83e-04 | +0.06% / +0.02% | 0.46904 | at floor |
| `vol-cl-sph-g80-a100-t200` | 2.03e-04 | 1.23e-04 | 1.78e-04 | +0.18% / -0.01% | 0.41570 | at floor |
| `vol-cl-sph-g85-a99-t10` | 1.80e-03 | 1.51e-03 | 1.41e-03 | +0.15% / +0.06% | 0.56591 | at floor |
| `vol-cl-sph-g85-a99-t50` | 1.10e-03 | 1.07e-03 | 8.52e-04 | +0.08% / +0.04% | 0.41591 | at floor |
| `vol-cl-sph-g85-a99-t200` | 4.85e-04 | 4.14e-04 | 3.95e-04 | +0.17% / -0.08% | 0.34945 | at floor |
| `vol-cl-sph-g85-a100-t10` | 1.74e-03 | 1.41e-03 | 1.38e-03 | +0.17% / +0.08% | 0.59050 | at floor |
| `vol-cl-sph-g85-a100-t50` | 1.13e-03 | 9.39e-04 | 9.03e-04 | +0.09% / +0.04% | 0.48434 | at floor |
| `vol-cl-sph-g85-a100-t200` | 3.98e-04 | 2.70e-04 | 3.16e-04 | +0.08% / -0.03% | 0.41920 | at floor |
| `vol-cl-sph-g90-a99-t10` | 3.98e-03 | 3.24e-03 | 2.93e-03 | +0.04% / -0.03% | 0.56959 | at floor |
| `vol-cl-sph-g90-a99-t50` | 3.03e-03 | 2.90e-03 | 2.26e-03 | +0.06% / -0.10% | 0.43478 | at floor |
| `vol-cl-sph-g90-a99-t200` | 9.72e-04 | 8.78e-04 | 7.63e-04 | +0.08% / -0.05% | 0.34279 | at floor |
| `vol-cl-sph-g90-a100-t10` | 3.88e-03 | 3.11e-03 | 2.94e-03 | +0.03% / -0.02% | 0.59528 | at floor |
| `vol-cl-sph-g90-a100-t50` | 2.96e-03 | 2.35e-03 | 2.45e-03 | +0.08% / +0.02% | 0.51220 | at floor |
| `vol-cl-sph-g90-a100-t200` | 1.07e-03 | 7.70e-04 | 8.14e-04 | -0.02% / -0.09% | 0.42748 | at floor |
| `vol-cl-slab-top-g85-a99-t10` | 3.88e-03 | 3.39e-03 | 3.01e-03 | -0.00% / -0.12% | 0.63667 | at floor |
| `vol-cl-slab-top-g85-a99-t50` | 5.30e-03 | 4.94e-03 | 3.97e-03 | -0.07% / -0.18% | 0.64357 | at floor |
| `vol-cl-slab-top-g85-a99-t200` | 5.32e-03 | 4.94e-03 | 4.02e-03 | -0.11% / -0.13% | 0.63477 | at floor |
| `vol-cl-slab-base-g85-a99-t10` | 4.51e-03 | 3.94e-03 | 3.16e-03 | -0.10% / -0.18% | 0.70668 | at floor |
| `vol-cl-slab-base-g85-a99-t50` | 1.69e-03 | 2.14e-03 | 1.23e-03 | +0.06% / +0.14% | 0.25115 | at floor |
| `vol-cl-slab-base-g85-a99-t200` | 2.74e-04 | 3.94e-04 | 2.93e-04 | +0.30% / +0.68% | 0.21647 | at floor |
| `vol-cl-slab-base-g80-a99-t50` | 6.82e-04 | 9.08e-04 | 5.15e-04 | +0.12% / +0.17% | 0.25412 | at floor |
| `vol-cl-slab-base-g90-a99-t50` | 6.93e-03 | 7.95e-03 | 4.46e-03 | +0.06% / +0.20% | 0.25844 | at floor |
| `vol-cl-slab-top-g85-a100-t200` | 3.91e-03 | 3.06e-03 | 3.04e-03 | -0.27% / -0.47% | 1.05888 | at floor |

Every cell is at or below the floor. relBias against Mitsuba stays within +-0.3% (the albedo-1 tau-200 slab seen from above: -0.27%, pbrt itself -0.2% from Mitsuba there).

Re-check 2026-11-23 (fresh 1024 spp render of all 35 reference cells at d2d7ccc, default settings, run name `vol-visual-cmp`; mean differences with a standard error from 8 x 8
blocks of the difference image): every cell within +-0.4% of Mitsuba. The two tau-200 slabs repeat their offsets at a new seed, so they are systematic, not noise, even
though relMSE calls them at floor: `slab-base-g85-a99-t200` +0.37% vs Mitsuba (7.5 SE) and +0.74% vs pbrt (pbrt -0.37% from Mitsuba, 13 SE); `slab-top-g85-a100-t200`
-0.39% vs Mitsuba, -0.59% vs pbrt. The references disagree by a similar amount there, so the truth is open; the cheap test is a depth sweep of those two cells (the
512 cap is where they differ most: albedo 1, tau 200). Visual page: our render beside both references with difference maps (artifact `Volume Reference Comparison`).

### Ground and phase cells

| scene | pbrt vs Mitsuba (floor) | ours vs pbrt | ours vs Mitsuba | relBias vs Mitsuba / pbrt | ours mean | verdict |
|---|---|---|---|---|---|---|
| `vol-cl-ground-sph-g85-a99-t50` (the tau 50 sphere 1 m above a grey ground) | 5.64e-04 | 5.60e-04 | 4.31e-04 | -0.01% / +0.03% | 0.29977 | at floor |
| `vol-cl-sph-hg2-a99-t50` (two HG lobes 0.2 / 0.85, 40 %; Mitsuba `blendphase`, pbrt has no such phase) | - | - | 1.10e-04 | +0.05% / - | 0.38336 | below the neighbouring cells' floors (~1e-3) |

The Mie spheres (`vol-cl-sph-mie10-a99-t50`, `-mie20-`) have no pbrt or Mitsuba reference (V300): Blender Cycles, open (P5.v14).

### 256^3 cloud (`vol-cl-cloud256`: 2 m voxels, 512 m box, peak sigma_t 0.3/m, albedo 0.99, g 0.85, 20% of the box occupied)

| variant | ours vs pbrt | ours vs Mitsuba | floor | relBias vs Mitsuba / pbrt | provisional wall, 4096 spp |
|---|---|---|---|---|---|
| global majorant, r32float | 4.95e-4 | 3.94e-4 | 5.20e-4 | +0.015% / +0.012% | 56.1 s |
| brick majorant, r32float | 5.29e-4 | 4.49e-4 | 5.20e-4 | +0.002% / -0.001% | 42.6 s |
| brick majorant, r16float | 5.26e-4 | 4.49e-4 | 5.20e-4 | -0.000% / -0.003% | 51.2 s |

Mitsuba's `heterogeneous` medium left 3 NaN pixels of 16,384 (excluded, as every metric does) and took 8125 s; pbrt 69 s. r16float vs r32float with the same seed: relMSE 5.5e-6.

### Closed forms (ours, mean pixel)

| scene | closed form | ours |
|---|---|---|
| `vol-cl-furnace-g{80,85,90,95,99}-t{10,50,200}` (albedo 1, unit sky, depth 4096, 1024 spp) | 1 | 0.99984 .. 1.00072 (all 15) |
| `vol-cl-single-g85` / `-oblique` | 0.874538 / 0.769085 | 0.874527 / 0.769183 |
| `vol-cl-single-g95-oblique` / `g99-oblique` | 0.780641 / 0.785884 | 0.780757 / 0.785934 |
| `vol-cl-single-hg2` / `-oblique` (NEE on; `pt_nee=0` sampler only) | 0.849691 / 0.756589 | 0.849655 (0.849787) / 0.756758 (0.756866) |
| `vol-cl-single-mie10` / `-oblique` | 0.879167 / 0.770928 | 0.879198 (0.879202) / 0.771101 (0.771063) |
| `vol-cl-single-mie20` / `-oblique` | 0.879843 / 0.771719 | 0.879873 (0.879879) / 0.771873 (0.771834) |
| `vol-cl-ground-furnace` (non-absorbing slab over a white ground) | 1 | 0.99974 |
| `vol-cl-ground-absorb` (E_3 closed form; f32 BVH nodes, see V302) | 0.194558 | 0.194576 (pbrt 0.19465, Mitsuba 0.19451) |

The oblique single-scatter cells read +1-2e-4 high at three seeds (a small consistent bias, open; far below every gate).

### Depth caps (mean pixel; how many orders a cloud needs)

| maxDepth | furnace g 0.8 t 200 | furnace g 0.9 t 200 | sphere g 0.85 albedo 0.99 t 200 | sphere g 0.8 albedo 1 t 200 |
|---|---|---|---|---|
| 64 | 0.694 | 0.593 | 0.33552 | 0.36601 |
| 128 | 0.798 | 0.738 | 0.34567 | 0.38751 |
| 256 | 0.875 | 0.852 | 0.34894 | 0.40371 |
| 512 | 0.931 | 0.935 | 0.34942 | 0.41597 |
| 1024 | 0.971 | 0.986 | 0.34944 | 0.42557 |
| 2048 | 0.994 | 1.000 | 0.34944 | 0.43202 |
| 8192 | | | 0.34944 | 0.43387 |

### Volume outputs (`volumes.outputs`)

Primary transmittance: absorber slab 0.493432 vs 0.493435, grid slab 0.496876 vs 0.497029 (ratio tracking, 1024 spp). Orders 0-1 = a maxDepth 1 render per pixel (bit for bit), orders 0-1 + 2+ = beauty;
beauty bit-identical with outputs on or off. Standard error vs the seed-to-seed difference: mean diff^2 / mean (se_a^2 + se_b^2) = 1.005 (PCG), 0.93 / 1.006 (Owen-Sobol, tau 10 / 50).
