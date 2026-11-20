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
