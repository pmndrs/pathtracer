# V7.3 split denoising A/B (2026-11-24, quality only, no timings)

Question: media noise is multiple scattering (orders 2+); orders 0-1 are 57% (Cornell fog) / 65% (WDAS cloud) of the radiance and ~0-4% of the error. Does filtering only
`radiance - order01` (the "MS plane") and adding `order01` back beat filtering the beauty? Arms: `raw`; `oidn_all` (OIDN, colour only, HDR, no guides, on the beauty);
`split_oidn` (OIDN on max(beauty - order01, 0), + order01); `blur_all` / `split_gauss` (Gaussian, sigma in {0.75, 1, 1.5, 2, 3, 4}, the best per cell shown: an ORACLE choice, the fixed sigma 2 rows are in the full table).
Two further arms with `order01 / (1 + order01)` as OIDN's albedo guide (`*_g`) were not better than the unguided ones (full table) and are not shown below.

Setup. `vol-cornell-fog` with `forcedNee` + `equiangular` + `volumes.outputs` (V7.3 step 1), `vol-vdb-wdas-cloud` default estimator, both 128x128, `pt-megakernel`, seeds 1 and 2 averaged (spp 1, 4, 16, 64, 256, 1024).
Reference below: our own 8192 spp render of the same kernel (seed 99; isolates the denoiser from the renderer's offset to pbrt: Cornell fog reads +0.5..+1% vs pbrt, WDAS -0.3%); the full table also has pbrt (Cornell 4096 spp, WDAS 1024 spp)
and Mitsuba (WDAS 1024 spp, NaN pixels excluded and counted by `compareImages`). Metrics: relMSE (eps 1e-2), relBias = mean(test) / mean(ref) - 1, edge relMSE = relMSE over reference pixels within 3 px of
|grad log(L + 0.05)| > 0.25 (17% of Cornell pixels, 9% of WDAS), mean LDR-FLIP (ACES, 0 EV). Machine load was 5-24 (shared Mac, other agents' jobs): correctness numbers, not timings.
Reproduce: `node bench/split-denoise.ts render --spec s.json [--refs]`, `... oidn --spec s.json` (page `split-denoise`, needs the Denoiser sibling), `... report --spec s.json`.

Reference noise floor. The 8192 spp reference has relMSE about (raw relMSE at spp) x spp / 8192 of its own: 1.7e-4 for both scenes against the 1024 spp rows (raw 1.35e-3 / 1.40e-3). Subtract it from the denoised 1024 rows before reading a ratio
(Cornell split 2.8e-4 -> ~1.1e-4, WDAS split 5.0e-4 -> ~3.3e-4).

## Summary (vs our 8192 spp render)

| scene | spp | arm | relMSE | relBias | edge relMSE | LDR-FLIP |
|---|---|---|---|---|---|---|
| cornell | 1 | raw | 1.01e+0 | 2.97% | 8.52e-1 | 0.2807 |
| cornell | 1 | oidn_all | 9.47e-3 | 0.60% | 4.28e-2 | 0.1161 |
| cornell | 1 | split_oidn | 7.28e-2 | 2.26% | 3.34e-1 | 0.1353 |
| cornell | 1 | blur_all_s1 | 1.18e-1 | 2.97% | 3.00e-1 | 0.2446 |
| cornell | 1 | split_gauss_s4 | 7.78e-2 | 2.52% | 3.45e-1 | 0.1702 |
| cornell | 4 | raw | 3.13e-1 | -0.07% | 1.89e-1 | 0.1906 |
| cornell | 4 | oidn_all | 3.23e-3 | -1.66% | 1.19e-2 | 0.0871 |
| cornell | 4 | split_oidn | 1.20e-2 | -0.63% | 5.36e-2 | 0.0863 |
| cornell | 4 | blur_all_s1 | 5.29e-2 | -0.08% | 1.81e-1 | 0.1676 |
| cornell | 4 | split_gauss_s4 | 1.52e-2 | -0.48% | 6.40e-2 | 0.1234 |
| cornell | 16 | raw | 8.67e-2 | 0.71% | 5.57e-2 | 0.1272 |
| cornell | 16 | oidn_all | 2.18e-3 | -0.52% | 8.26e-3 | 0.0706 |
| cornell | 16 | split_oidn | 3.33e-3 | 0.20% | 1.33e-2 | 0.0667 |
| cornell | 16 | blur_all_s0.75 | 2.15e-2 | 0.71% | 6.21e-2 | 0.1208 |
| cornell | 16 | split_gauss_s3 | 5.43e-3 | 0.50% | 2.04e-2 | 0.0979 |
| cornell | 64 | raw | 1.92e-2 | 0.52% | 1.58e-2 | 0.0863 |
| cornell | 64 | oidn_all | 9.43e-4 | 0.11% | 2.96e-3 | 0.0519 |
| cornell | 64 | split_oidn | 1.01e-3 | -0.05% | 3.23e-3 | 0.0476 |
| cornell | 64 | blur_all_s0.75 | 1.19e-2 | 0.52% | 5.64e-2 | 0.0823 |
| cornell | 64 | split_gauss_s1.5 | 2.04e-3 | 0.50% | 5.99e-3 | 0.0695 |
| cornell | 256 | raw | 4.88e-3 | 0.06% | 4.07e-3 | 0.0557 |
| cornell | 256 | oidn_all | 5.18e-4 | -0.17% | 1.24e-3 | 0.0399 |
| cornell | 256 | split_oidn | 4.60e-4 | -0.14% | 1.09e-3 | 0.0362 |
| cornell | 256 | blur_all_s0.75 | 9.71e-3 | 0.06% | 5.35e-2 | 0.0561 |
| cornell | 256 | split_gauss_s1 | 9.91e-4 | 0.06% | 2.45e-3 | 0.0494 |
| cornell | 1024 | raw | 1.35e-3 | 0.17% | 1.13e-3 | 0.0356 |
| cornell | 1024 | oidn_all | 3.28e-4 | -0.07% | 6.06e-4 | 0.0309 |
| cornell | 1024 | split_oidn | 2.81e-4 | 0.13% | 4.68e-4 | 0.0282 |
| cornell | 1024 | blur_all_s0.75 | 9.44e-3 | 0.17% | 5.45e-2 | 0.0412 |
| cornell | 1024 | split_gauss_s0.75 | 5.34e-4 | 0.17% | 1.15e-3 | 0.0344 |
| wdas | 1 | raw | 1.81e+0 | 0.23% | 3.43e+0 | 0.0811 |
| wdas | 1 | oidn_all | 5.92e-3 | -2.09% | 2.94e-2 | 0.0438 |
| wdas | 1 | split_oidn | 1.15e-2 | -2.14% | 4.69e-2 | 0.0379 |
| wdas | 1 | blur_all_s4 | 1.51e-2 | 0.23% | 6.87e-2 | 0.0478 |
| wdas | 1 | split_gauss_s4 | 3.27e-2 | 0.23% | 1.75e-1 | 0.0713 |
| wdas | 4 | raw | 3.18e-1 | -0.74% | 5.59e-1 | 0.0555 |
| wdas | 4 | oidn_all | 5.07e-3 | -2.63% | 2.44e-2 | 0.0379 |
| wdas | 4 | split_oidn | 5.12e-3 | -0.97% | 1.97e-2 | 0.0278 |
| wdas | 4 | blur_all_s3 | 7.34e-3 | -0.74% | 4.40e-2 | 0.0377 |
| wdas | 4 | split_gauss_s2 | 1.35e-2 | -0.74% | 6.41e-2 | 0.0453 |
| wdas | 16 | raw | 8.05e-2 | -0.33% | 1.27e-1 | 0.0344 |
| wdas | 16 | oidn_all | 2.85e-3 | -2.20% | 1.39e-2 | 0.0291 |
| wdas | 16 | split_oidn | 2.54e-3 | -1.09% | 1.09e-2 | 0.0213 |
| wdas | 16 | blur_all_s2 | 3.97e-3 | -0.33% | 2.38e-2 | 0.0257 |
| wdas | 16 | split_gauss_s1.5 | 6.46e-3 | -0.33% | 3.39e-2 | 0.0305 |
| wdas | 64 | raw | 2.05e-2 | -0.22% | 3.39e-2 | 0.0203 |
| wdas | 64 | oidn_all | 4.03e-3 | -1.21% | 1.10e-2 | 0.0237 |
| wdas | 64 | split_oidn | 1.48e-3 | -0.65% | 5.89e-3 | 0.0159 |
| wdas | 64 | blur_all_s1.5 | 2.28e-3 | -0.22% | 1.39e-2 | 0.0185 |
| wdas | 64 | split_gauss_s1 | 3.17e-3 | -0.22% | 1.46e-2 | 0.0201 |
| wdas | 256 | raw | 5.25e-3 | -0.02% | 8.94e-3 | 0.0116 |
| wdas | 256 | oidn_all | 2.30e-3 | -0.31% | 4.47e-3 | 0.0169 |
| wdas | 256 | split_oidn | 8.59e-4 | -0.12% | 3.06e-3 | 0.0118 |
| wdas | 256 | blur_all_s1 | 1.17e-3 | -0.02% | 6.32e-3 | 0.0121 |
| wdas | 256 | split_gauss_s0.75 | 1.56e-3 | -0.02% | 7.22e-3 | 0.0130 |
| wdas | 1024 | raw | 1.40e-3 | 0.02% | 2.51e-3 | 0.0069 |
| wdas | 1024 | oidn_all | 5.49e-4 | -0.12% | 1.50e-3 | 0.0124 |
| wdas | 1024 | split_oidn | 5.04e-4 | -0.03% | 1.81e-3 | 0.0084 |
| wdas | 1024 | blur_all_s0.75 | 6.16e-4 | 0.02% | 3.16e-3 | 0.0081 |
| wdas | 1024 | split_gauss_s0.75 | 9.78e-4 | 0.02% | 6.21e-3 | 0.0100 |


## Full table (all references, all arms, seeds averaged)

| scene | ref | spp | arm | relMSE | relBias | edge relMSE | LDR-FLIP |
|---|---|---|---|---|---|---|---|
| cornell | ours | 1 | raw | 1.01e+0 | 2.97% | 8.52e-1 | 0.2807 |
| cornell | ours | 1 | oidn_all | 9.47e-3 | 0.60% | 4.28e-2 | 0.1161 |
| cornell | ours | 1 | split_oidn | 7.28e-2 | 2.26% | 3.34e-1 | 0.1353 |
| cornell | ours | 1 | oidn_all_g | 9.99e-3 | -1.92% | 4.48e-2 | 0.1201 |
| cornell | ours | 1 | split_oidn_g | 7.34e-2 | 2.53% | 3.35e-1 | 0.1362 |
| cornell | ours | 1 | blur_all_s1 | 1.18e-1 | 2.97% | 3.00e-1 | 0.2446 |
| cornell | ours | 1 | blur_all_s2 | 2.00e-1 | 2.87% | 1.10e+0 | 0.2078 |
| cornell | ours | 1 | split_gauss_s4 | 7.78e-2 | 2.52% | 3.45e-1 | 0.1702 |
| cornell | ours | 1 | split_gauss_s2 | 9.01e-2 | 2.89% | 3.48e-1 | 0.2095 |
| cornell | ours | 1 | split_oidn_fade0.05 | 7.28e-2 | 2.26% | 3.34e-1 | 0.1353 |
| cornell | ours | 1024 | raw | 1.35e-3 | 0.17% | 1.13e-3 | 0.0356 |
| cornell | ours | 1024 | oidn_all | 3.28e-4 | -0.07% | 6.06e-4 | 0.0309 |
| cornell | ours | 1024 | split_oidn | 2.81e-4 | 0.13% | 4.68e-4 | 0.0282 |
| cornell | ours | 1024 | oidn_all_g | 3.97e-4 | -0.26% | 1.08e-3 | 0.0321 |
| cornell | ours | 1024 | split_oidn_g | 2.95e-4 | 0.24% | 5.33e-4 | 0.0302 |
| cornell | ours | 1024 | blur_all_s0.75 | 9.44e-3 | 0.17% | 5.45e-2 | 0.0412 |
| cornell | ours | 1024 | blur_all_s2 | 1.49e-1 | 0.08% | 8.95e-1 | 0.0759 |
| cornell | ours | 1024 | split_gauss_s0.75 | 5.34e-4 | 0.17% | 1.15e-3 | 0.0344 |
| cornell | ours | 1024 | split_gauss_s2 | 1.30e-3 | 0.10% | 4.88e-3 | 0.0490 |
| cornell | ours | 1024 | split_oidn_fade0.05 | 3.70e-4 | -0.33% | 5.09e-4 | 0.0308 |
| cornell | ours | 16 | raw | 8.67e-2 | 0.71% | 5.57e-2 | 0.1272 |
| cornell | ours | 16 | oidn_all | 2.18e-3 | -0.52% | 8.26e-3 | 0.0706 |
| cornell | ours | 16 | split_oidn | 3.33e-3 | 0.20% | 1.33e-2 | 0.0667 |
| cornell | ours | 16 | oidn_all_g | 2.93e-3 | -0.95% | 1.26e-2 | 0.0710 |
| cornell | ours | 16 | split_oidn_g | 3.48e-3 | 0.52% | 1.36e-2 | 0.0674 |
| cornell | ours | 16 | blur_all_s0.75 | 2.15e-2 | 0.71% | 6.21e-2 | 0.1208 |
| cornell | ours | 16 | blur_all_s2 | 1.49e-1 | 0.62% | 8.84e-1 | 0.1180 |
| cornell | ours | 16 | split_gauss_s3 | 5.43e-3 | 0.50% | 2.04e-2 | 0.0979 |
| cornell | ours | 16 | split_gauss_s2 | 5.53e-3 | 0.64% | 1.77e-2 | 0.0985 |
| cornell | ours | 16 | split_oidn_fade0.05 | 3.39e-3 | 0.12% | 1.33e-2 | 0.0675 |
| cornell | ours | 256 | raw | 4.88e-3 | 0.06% | 4.07e-3 | 0.0557 |
| cornell | ours | 256 | oidn_all | 5.18e-4 | -0.17% | 1.24e-3 | 0.0399 |
| cornell | ours | 256 | split_oidn | 4.60e-4 | -0.14% | 1.09e-3 | 0.0362 |
| cornell | ours | 256 | oidn_all_g | 7.04e-4 | -0.73% | 2.47e-3 | 0.0410 |
| cornell | ours | 256 | split_oidn_g | 4.99e-4 | 0.06% | 1.23e-3 | 0.0382 |
| cornell | ours | 256 | blur_all_s0.75 | 9.71e-3 | 0.06% | 5.35e-2 | 0.0561 |
| cornell | ours | 256 | blur_all_s2 | 1.47e-1 | -0.03% | 8.84e-1 | 0.0808 |
| cornell | ours | 256 | split_gauss_s1 | 9.91e-4 | 0.06% | 2.45e-3 | 0.0494 |
| cornell | ours | 256 | split_gauss_s2 | 1.47e-3 | -0.01% | 5.41e-3 | 0.0554 |
| cornell | ours | 256 | split_oidn_fade0.05 | 5.16e-4 | -0.45% | 1.12e-3 | 0.0394 |
| cornell | ours | 4 | raw | 3.13e-1 | -0.07% | 1.89e-1 | 0.1906 |
| cornell | ours | 4 | oidn_all | 3.23e-3 | -1.66% | 1.19e-2 | 0.0871 |
| cornell | ours | 4 | split_oidn | 1.20e-2 | -0.63% | 5.36e-2 | 0.0863 |
| cornell | ours | 4 | oidn_all_g | 4.90e-3 | -0.70% | 2.22e-2 | 0.0854 |
| cornell | ours | 4 | split_oidn_g | 1.22e-2 | -0.46% | 5.41e-2 | 0.0860 |
| cornell | ours | 4 | blur_all_s1 | 5.29e-2 | -0.08% | 1.81e-1 | 0.1676 |
| cornell | ours | 4 | blur_all_s2 | 1.58e-1 | -0.16% | 9.12e-1 | 0.1505 |
| cornell | ours | 4 | split_gauss_s4 | 1.52e-2 | -0.48% | 6.40e-2 | 0.1234 |
| cornell | ours | 4 | split_gauss_s2 | 1.80e-2 | -0.14% | 5.96e-2 | 0.1384 |
| cornell | ours | 4 | split_oidn_fade0.05 | 1.23e-2 | -0.91% | 5.39e-2 | 0.0869 |
| cornell | ours | 64 | raw | 1.92e-2 | 0.52% | 1.58e-2 | 0.0863 |
| cornell | ours | 64 | oidn_all | 9.43e-4 | 0.11% | 2.96e-3 | 0.0519 |
| cornell | ours | 64 | split_oidn | 1.01e-3 | -0.05% | 3.23e-3 | 0.0476 |
| cornell | ours | 64 | oidn_all_g | 1.33e-3 | -0.76% | 5.37e-3 | 0.0521 |
| cornell | ours | 64 | split_oidn_g | 1.09e-3 | 0.30% | 3.56e-3 | 0.0481 |
| cornell | ours | 64 | blur_all_s0.75 | 1.19e-2 | 0.52% | 5.64e-2 | 0.0823 |
| cornell | ours | 64 | blur_all_s2 | 1.51e-1 | 0.44% | 9.01e-1 | 0.0929 |
| cornell | ours | 64 | split_gauss_s1.5 | 2.04e-3 | 0.50% | 5.99e-3 | 0.0695 |
| cornell | ours | 64 | split_gauss_s2 | 2.15e-3 | 0.46% | 7.43e-3 | 0.0693 |
| cornell | ours | 64 | split_oidn_fade0.05 | 1.04e-3 | -0.12% | 3.25e-3 | 0.0490 |
| cornell | pbrt | 1 | raw | 1.01e+0 | 3.56% | 8.54e-1 | 0.2799 |
| cornell | pbrt | 1 | oidn_all | 1.00e-2 | 1.17% | 4.32e-2 | 0.1194 |
| cornell | pbrt | 1 | split_oidn | 7.34e-2 | 2.85% | 3.29e-1 | 0.1415 |
| cornell | pbrt | 1 | oidn_all_g | 1.06e-2 | -1.36% | 4.50e-2 | 0.1236 |
| cornell | pbrt | 1 | split_oidn_g | 7.39e-2 | 3.11% | 3.30e-1 | 0.1423 |
| cornell | pbrt | 1 | blur_all_s1 | 1.18e-1 | 3.55% | 2.97e-1 | 0.2434 |
| cornell | pbrt | 1 | blur_all_s2 | 2.04e-1 | 3.46% | 1.11e+0 | 0.2063 |
| cornell | pbrt | 1 | split_gauss_s4 | 7.79e-2 | 3.11% | 3.40e-1 | 0.1683 |
| cornell | pbrt | 1 | split_gauss_s2 | 9.04e-2 | 3.48% | 3.43e-1 | 0.2084 |
| cornell | pbrt | 1 | split_oidn_fade0.05 | 7.34e-2 | 2.85% | 3.29e-1 | 0.1415 |
| cornell | pbrt | 1024 | raw | 1.88e-3 | 0.74% | 1.40e-3 | 0.0513 |
| cornell | pbrt | 1024 | oidn_all | 8.44e-4 | 0.50% | 8.69e-4 | 0.0480 |
| cornell | pbrt | 1024 | split_oidn | 7.95e-4 | 0.70% | 7.25e-4 | 0.0464 |
| cornell | pbrt | 1024 | oidn_all_g | 9.11e-4 | 0.31% | 1.27e-3 | 0.0495 |
| cornell | pbrt | 1024 | split_oidn_g | 8.21e-4 | 0.81% | 8.12e-4 | 0.0483 |
| cornell | pbrt | 1024 | blur_all_s0.75 | 9.94e-3 | 0.74% | 5.42e-2 | 0.0558 |
| cornell | pbrt | 1024 | blur_all_s2 | 1.53e-1 | 0.65% | 9.04e-1 | 0.0869 |
| cornell | pbrt | 1024 | split_gauss_s0.75 | 1.03e-3 | 0.74% | 1.42e-3 | 0.0497 |
| cornell | pbrt | 1024 | split_gauss_s2 | 1.74e-3 | 0.67% | 5.02e-3 | 0.0619 |
| cornell | pbrt | 1024 | split_oidn_fade0.05 | 8.39e-4 | 0.24% | 7.43e-4 | 0.0444 |
| cornell | pbrt | 16 | raw | 8.75e-2 | 1.28% | 5.67e-2 | 0.1281 |
| cornell | pbrt | 16 | oidn_all | 2.54e-3 | 0.05% | 8.20e-3 | 0.0713 |
| cornell | pbrt | 16 | split_oidn | 3.77e-3 | 0.78% | 1.34e-2 | 0.0692 |
| cornell | pbrt | 16 | oidn_all_g | 3.27e-3 | -0.39% | 1.22e-2 | 0.0735 |
| cornell | pbrt | 16 | split_oidn_g | 3.96e-3 | 1.09% | 1.38e-2 | 0.0717 |
| cornell | pbrt | 16 | blur_all_s0.75 | 2.19e-2 | 1.28% | 6.16e-2 | 0.1205 |
| cornell | pbrt | 16 | blur_all_s2 | 1.52e-1 | 1.20% | 8.92e-1 | 0.1179 |
| cornell | pbrt | 16 | split_gauss_s3 | 5.80e-3 | 1.07% | 2.03e-2 | 0.0978 |
| cornell | pbrt | 16 | split_gauss_s2 | 5.94e-3 | 1.22% | 1.77e-2 | 0.0986 |
| cornell | pbrt | 16 | split_oidn_fade0.05 | 3.82e-3 | 0.69% | 1.35e-2 | 0.0690 |
| cornell | pbrt | 256 | raw | 5.43e-3 | 0.63% | 4.34e-3 | 0.0651 |
| cornell | pbrt | 256 | oidn_all | 1.02e-3 | 0.40% | 1.48e-3 | 0.0522 |
| cornell | pbrt | 256 | split_oidn | 9.61e-4 | 0.43% | 1.35e-3 | 0.0498 |
| cornell | pbrt | 256 | oidn_all_g | 1.19e-3 | -0.16% | 2.59e-3 | 0.0537 |
| cornell | pbrt | 256 | split_oidn_g | 1.03e-3 | 0.63% | 1.52e-3 | 0.0529 |
| cornell | pbrt | 256 | blur_all_s0.75 | 1.02e-2 | 0.63% | 5.32e-2 | 0.0652 |
| cornell | pbrt | 256 | blur_all_s2 | 1.51e-1 | 0.54% | 8.93e-1 | 0.0898 |
| cornell | pbrt | 256 | split_gauss_s1 | 1.48e-3 | 0.63% | 2.69e-3 | 0.0592 |
| cornell | pbrt | 256 | split_gauss_s2 | 1.91e-3 | 0.56% | 5.55e-3 | 0.0655 |
| cornell | pbrt | 256 | split_oidn_fade0.05 | 9.90e-4 | 0.12% | 1.37e-3 | 0.0489 |
| cornell | pbrt | 4 | raw | 3.17e-1 | 0.50% | 1.89e-1 | 0.1910 |
| cornell | pbrt | 4 | oidn_all | 3.62e-3 | -1.10% | 1.18e-2 | 0.0894 |
| cornell | pbrt | 4 | split_oidn | 1.24e-2 | -0.07% | 5.24e-2 | 0.0921 |
| cornell | pbrt | 4 | oidn_all_g | 5.29e-3 | -0.13% | 2.18e-2 | 0.0886 |
| cornell | pbrt | 4 | split_oidn_g | 1.27e-2 | 0.11% | 5.29e-2 | 0.0917 |
| cornell | pbrt | 4 | blur_all_s1 | 5.38e-2 | 0.49% | 1.81e-1 | 0.1668 |
| cornell | pbrt | 4 | blur_all_s2 | 1.61e-1 | 0.41% | 9.20e-1 | 0.1496 |
| cornell | pbrt | 4 | split_gauss_s4 | 1.54e-2 | 0.09% | 6.26e-2 | 0.1226 |
| cornell | pbrt | 4 | split_gauss_s2 | 1.83e-2 | 0.43% | 5.82e-2 | 0.1377 |
| cornell | pbrt | 4 | split_oidn_fade0.05 | 1.27e-2 | -0.34% | 5.26e-2 | 0.0910 |
| cornell | pbrt | 64 | raw | 1.98e-2 | 1.10% | 1.62e-2 | 0.0899 |
| cornell | pbrt | 64 | oidn_all | 1.39e-3 | 0.68% | 3.18e-3 | 0.0581 |
| cornell | pbrt | 64 | split_oidn | 1.47e-3 | 0.52% | 3.52e-3 | 0.0548 |
| cornell | pbrt | 64 | oidn_all_g | 1.77e-3 | -0.20% | 5.40e-3 | 0.0601 |
| cornell | pbrt | 64 | split_oidn_g | 1.60e-3 | 0.88% | 3.86e-3 | 0.0583 |
| cornell | pbrt | 64 | blur_all_s0.75 | 1.24e-2 | 1.10% | 5.60e-2 | 0.0856 |
| cornell | pbrt | 64 | blur_all_s2 | 1.54e-1 | 1.01% | 9.10e-1 | 0.0974 |
| cornell | pbrt | 64 | split_gauss_s1.5 | 2.50e-3 | 1.07% | 6.23e-3 | 0.0743 |
| cornell | pbrt | 64 | split_gauss_s2 | 2.58e-3 | 1.03% | 7.61e-3 | 0.0746 |
| cornell | pbrt | 64 | split_oidn_fade0.05 | 1.49e-3 | 0.45% | 3.54e-3 | 0.0545 |
| wdas | mitsuba | 1 | raw | 1.96e+0 | 0.33% | 3.46e+0 | 0.0805 |
| wdas | mitsuba | 1 | oidn_all | 7.06e-3 | -2.02% | 3.20e-2 | 0.0454 |
| wdas | mitsuba | 1 | split_oidn | 1.26e-2 | -2.07% | 4.67e-2 | 0.0394 |
| wdas | mitsuba | 1 | oidn_all_g | 8.55e-3 | -4.88% | 3.70e-2 | 0.0541 |
| wdas | mitsuba | 1 | split_oidn_g | 1.39e-2 | -4.15% | 5.19e-2 | 0.0437 |
| wdas | mitsuba | 1 | blur_all_s4 | 1.65e-2 | 0.29% | 6.91e-2 | 0.0488 |
| wdas | mitsuba | 1 | blur_all_s2 | 4.00e-2 | 0.29% | 9.49e-2 | 0.0531 |
| wdas | mitsuba | 1 | split_gauss_s4 | 3.42e-2 | 0.29% | 1.59e-1 | 0.0718 |
| wdas | mitsuba | 1 | split_gauss_s2 | 5.02e-2 | 0.29% | 1.42e-1 | 0.0633 |
| wdas | mitsuba | 1 | split_oidn_fade0.05 | 1.26e-2 | -2.07% | 4.67e-2 | 0.0394 |
| wdas | mitsuba | 1024 | raw | 2.54e-3 | 0.09% | 5.21e-3 | 0.0103 |
| wdas | mitsuba | 1024 | oidn_all | 1.66e-3 | -0.05% | 4.04e-3 | 0.0165 |
| wdas | mitsuba | 1024 | split_oidn | 1.64e-3 | 0.04% | 4.21e-3 | 0.0123 |
| wdas | mitsuba | 1024 | oidn_all_g | 1.97e-3 | 0.13% | 4.74e-3 | 0.0177 |
| wdas | mitsuba | 1024 | split_oidn_g | 2.02e-3 | 0.10% | 4.79e-3 | 0.0115 |
| wdas | mitsuba | 1024 | blur_all_s0.75 | 1.76e-3 | 0.09% | 5.63e-3 | 0.0119 |
| wdas | mitsuba | 1024 | blur_all_s2 | 3.78e-3 | 0.09% | 2.33e-2 | 0.0207 |
| wdas | mitsuba | 1024 | split_gauss_s0.75 | 2.12e-3 | 0.09% | 8.14e-3 | 0.0133 |
| wdas | mitsuba | 1024 | split_gauss_s2 | 7.13e-3 | 0.10% | 4.64e-2 | 0.0292 |
| wdas | mitsuba | 1024 | split_oidn_fade0.05 | 1.63e-3 | -0.01% | 4.09e-3 | 0.0118 |
| wdas | mitsuba | 16 | raw | 8.36e-2 | -0.26% | 1.46e-1 | 0.0343 |
| wdas | mitsuba | 16 | oidn_all | 3.87e-3 | -2.13% | 1.69e-2 | 0.0309 |
| wdas | mitsuba | 16 | split_oidn | 3.69e-3 | -1.02% | 1.36e-2 | 0.0234 |
| wdas | mitsuba | 16 | oidn_all_g | 5.69e-3 | -1.17% | 2.39e-2 | 0.0317 |
| wdas | mitsuba | 16 | split_oidn_g | 4.10e-3 | -1.12% | 1.55e-2 | 0.0241 |
| wdas | mitsuba | 16 | blur_all_s2 | 5.17e-3 | -0.26% | 2.63e-2 | 0.0272 |
| wdas | mitsuba | 16 | split_gauss_s1.5 | 7.67e-3 | -0.26% | 3.46e-2 | 0.0314 |
| wdas | mitsuba | 16 | split_gauss_s2 | 8.71e-3 | -0.25% | 4.92e-2 | 0.0347 |
| wdas | mitsuba | 16 | split_oidn_fade0.05 | 3.65e-3 | -1.11% | 1.35e-2 | 0.0232 |
| wdas | mitsuba | 256 | raw | 6.45e-3 | 0.05% | 1.35e-2 | 0.0135 |
| wdas | mitsuba | 256 | oidn_all | 3.41e-3 | -0.23% | 7.99e-3 | 0.0194 |
| wdas | mitsuba | 256 | split_oidn | 2.03e-3 | -0.04% | 5.94e-3 | 0.0150 |
| wdas | mitsuba | 256 | oidn_all_g | 2.71e-3 | -0.02% | 7.98e-3 | 0.0206 |
| wdas | mitsuba | 256 | split_oidn_g | 2.64e-3 | -0.05% | 7.25e-3 | 0.0142 |
| wdas | mitsuba | 256 | blur_all_s1 | 2.34e-3 | 0.05% | 8.86e-3 | 0.0148 |
| wdas | mitsuba | 256 | blur_all_s2 | 3.85e-3 | 0.05% | 2.34e-2 | 0.0212 |
| wdas | mitsuba | 256 | split_gauss_s0.75 | 2.71e-3 | 0.05% | 9.49e-3 | 0.0154 |
| wdas | mitsuba | 256 | split_gauss_s2 | 7.19e-3 | 0.05% | 4.64e-2 | 0.0297 |
| wdas | mitsuba | 256 | split_oidn_fade0.05 | 2.03e-3 | -0.13% | 5.84e-3 | 0.0146 |
| wdas | mitsuba | 4 | raw | 3.34e-1 | -0.70% | 6.32e-1 | 0.0549 |
| wdas | mitsuba | 4 | oidn_all | 6.07e-3 | -2.56% | 2.68e-2 | 0.0392 |
| wdas | mitsuba | 4 | split_oidn | 6.37e-3 | -0.90% | 2.20e-2 | 0.0297 |
| wdas | mitsuba | 4 | oidn_all_g | 8.11e-3 | -2.89% | 3.90e-2 | 0.0426 |
| wdas | mitsuba | 4 | split_oidn_g | 6.91e-3 | -2.77% | 2.62e-2 | 0.0341 |
| wdas | mitsuba | 4 | blur_all_s3 | 8.55e-3 | -0.67% | 4.54e-2 | 0.0390 |
| wdas | mitsuba | 4 | blur_all_s2 | 1.00e-2 | -0.67% | 3.69e-2 | 0.0378 |
| wdas | mitsuba | 4 | split_gauss_s2 | 1.47e-2 | -0.67% | 6.27e-2 | 0.0461 |
| wdas | mitsuba | 4 | split_oidn_fade0.05 | 6.33e-3 | -1.26% | 2.22e-2 | 0.0296 |
| wdas | mitsuba | 64 | raw | 2.18e-2 | -0.15% | 4.48e-2 | 0.0208 |
| wdas | mitsuba | 64 | oidn_all | 5.04e-3 | -1.14% | 1.55e-2 | 0.0249 |
| wdas | mitsuba | 64 | split_oidn | 2.61e-3 | -0.58% | 8.80e-3 | 0.0181 |
| wdas | mitsuba | 64 | oidn_all_g | 3.54e-3 | -0.49% | 1.24e-2 | 0.0253 |
| wdas | mitsuba | 64 | split_oidn_g | 2.99e-3 | -0.50% | 9.31e-3 | 0.0183 |
| wdas | mitsuba | 64 | blur_all_s1.5 | 3.47e-3 | -0.15% | 1.65e-2 | 0.0202 |
| wdas | mitsuba | 64 | blur_all_s2 | 4.12e-3 | -0.15% | 2.40e-2 | 0.0227 |
| wdas | mitsuba | 64 | split_gauss_s1 | 4.34e-3 | -0.15% | 1.68e-2 | 0.0214 |
| wdas | mitsuba | 64 | split_gauss_s2 | 7.52e-3 | -0.15% | 4.69e-2 | 0.0312 |
| wdas | mitsuba | 64 | split_oidn_fade0.05 | 2.60e-3 | -0.67% | 8.66e-3 | 0.0179 |
| wdas | ours | 1 | raw | 1.81e+0 | 0.23% | 3.43e+0 | 0.0811 |
| wdas | ours | 1 | oidn_all | 5.92e-3 | -2.09% | 2.94e-2 | 0.0438 |
| wdas | ours | 1 | split_oidn | 1.15e-2 | -2.14% | 4.69e-2 | 0.0379 |
| wdas | ours | 1 | oidn_all_g | 7.90e-3 | -4.95% | 3.63e-2 | 0.0538 |
| wdas | ours | 1 | split_oidn_g | 1.31e-2 | -4.22% | 5.42e-2 | 0.0435 |
| wdas | ours | 1 | blur_all_s4 | 1.51e-2 | 0.23% | 6.87e-2 | 0.0478 |
| wdas | ours | 1 | blur_all_s2 | 3.81e-2 | 0.23% | 9.84e-2 | 0.0528 |
| wdas | ours | 1 | split_gauss_s4 | 3.27e-2 | 0.23% | 1.75e-1 | 0.0713 |
| wdas | ours | 1 | split_gauss_s2 | 4.83e-2 | 0.23% | 1.52e-1 | 0.0632 |
| wdas | ours | 1 | split_oidn_fade0.05 | 1.15e-2 | -2.14% | 4.69e-2 | 0.0379 |
| wdas | ours | 1024 | raw | 1.40e-3 | 0.02% | 2.51e-3 | 0.0069 |
| wdas | ours | 1024 | oidn_all | 5.49e-4 | -0.12% | 1.50e-3 | 0.0124 |
| wdas | ours | 1024 | split_oidn | 5.04e-4 | -0.03% | 1.81e-3 | 0.0084 |
| wdas | ours | 1024 | oidn_all_g | 8.35e-4 | 0.06% | 2.14e-3 | 0.0138 |
| wdas | ours | 1024 | split_oidn_g | 8.92e-4 | 0.03% | 2.26e-3 | 0.0079 |
| wdas | ours | 1024 | blur_all_s0.75 | 6.16e-4 | 0.02% | 3.16e-3 | 0.0081 |
| wdas | ours | 1024 | blur_all_s2 | 2.56e-3 | 0.02% | 2.19e-2 | 0.0177 |
| wdas | ours | 1024 | split_gauss_s0.75 | 9.78e-4 | 0.02% | 6.21e-3 | 0.0100 |
| wdas | ours | 1024 | split_gauss_s2 | 5.94e-3 | 0.02% | 4.97e-2 | 0.0270 |
| wdas | ours | 1024 | split_oidn_fade0.05 | 5.02e-4 | -0.08% | 1.69e-3 | 0.0081 |
| wdas | ours | 16 | raw | 8.05e-2 | -0.33% | 1.27e-1 | 0.0344 |
| wdas | ours | 16 | oidn_all | 2.85e-3 | -2.20% | 1.39e-2 | 0.0291 |
| wdas | ours | 16 | split_oidn | 2.54e-3 | -1.09% | 1.09e-2 | 0.0213 |
| wdas | ours | 16 | oidn_all_g | 4.69e-3 | -1.23% | 2.26e-2 | 0.0299 |
| wdas | ours | 16 | split_oidn_g | 3.00e-3 | -1.19% | 1.34e-2 | 0.0222 |
| wdas | ours | 16 | blur_all_s2 | 3.97e-3 | -0.33% | 2.38e-2 | 0.0257 |
| wdas | ours | 16 | split_gauss_s1.5 | 6.46e-3 | -0.33% | 3.39e-2 | 0.0305 |
| wdas | ours | 16 | split_gauss_s2 | 7.52e-3 | -0.33% | 5.13e-2 | 0.0337 |
| wdas | ours | 16 | split_oidn_fade0.05 | 2.51e-3 | -1.17% | 1.09e-2 | 0.0212 |
| wdas | ours | 256 | raw | 5.25e-3 | -0.02% | 8.94e-3 | 0.0116 |
| wdas | ours | 256 | oidn_all | 2.30e-3 | -0.31% | 4.47e-3 | 0.0169 |
| wdas | ours | 256 | split_oidn | 8.59e-4 | -0.12% | 3.06e-3 | 0.0118 |
| wdas | ours | 256 | oidn_all_g | 1.58e-3 | -0.09% | 4.88e-3 | 0.0179 |
| wdas | ours | 256 | split_oidn_g | 1.52e-3 | -0.13% | 4.60e-3 | 0.0116 |
| wdas | ours | 256 | blur_all_s1 | 1.17e-3 | -0.02% | 6.32e-3 | 0.0121 |
| wdas | ours | 256 | blur_all_s2 | 2.63e-3 | -0.02% | 2.19e-2 | 0.0185 |
| wdas | ours | 256 | split_gauss_s0.75 | 1.56e-3 | -0.02% | 7.22e-3 | 0.0130 |
| wdas | ours | 256 | split_gauss_s2 | 6.00e-3 | -0.02% | 4.95e-2 | 0.0277 |
| wdas | ours | 256 | split_oidn_fade0.05 | 8.69e-4 | -0.20% | 2.97e-3 | 0.0118 |
| wdas | ours | 4 | raw | 3.18e-1 | -0.74% | 5.59e-1 | 0.0555 |
| wdas | ours | 4 | oidn_all | 5.07e-3 | -2.63% | 2.44e-2 | 0.0379 |
| wdas | ours | 4 | split_oidn | 5.12e-3 | -0.97% | 1.97e-2 | 0.0278 |
| wdas | ours | 4 | oidn_all_g | 7.30e-3 | -2.96% | 4.00e-2 | 0.0416 |
| wdas | ours | 4 | split_oidn_g | 5.98e-3 | -2.84% | 2.60e-2 | 0.0331 |
| wdas | ours | 4 | blur_all_s3 | 7.34e-3 | -0.74% | 4.40e-2 | 0.0377 |
| wdas | ours | 4 | blur_all_s2 | 8.75e-3 | -0.74% | 3.30e-2 | 0.0369 |
| wdas | ours | 4 | split_gauss_s2 | 1.35e-2 | -0.74% | 6.41e-2 | 0.0453 |
| wdas | ours | 4 | split_oidn_fade0.05 | 5.15e-3 | -1.33% | 1.99e-2 | 0.0278 |
| wdas | ours | 64 | raw | 2.05e-2 | -0.22% | 3.39e-2 | 0.0203 |
| wdas | ours | 64 | oidn_all | 4.03e-3 | -1.21% | 1.10e-2 | 0.0237 |
| wdas | ours | 64 | split_oidn | 1.48e-3 | -0.65% | 5.89e-3 | 0.0159 |
| wdas | ours | 64 | oidn_all_g | 2.49e-3 | -0.56% | 9.55e-3 | 0.0235 |
| wdas | ours | 64 | split_oidn_g | 1.91e-3 | -0.57% | 7.24e-3 | 0.0165 |
| wdas | ours | 64 | blur_all_s1.5 | 2.28e-3 | -0.22% | 1.39e-2 | 0.0185 |
| wdas | ours | 64 | blur_all_s2 | 2.92e-3 | -0.22% | 2.21e-2 | 0.0210 |
| wdas | ours | 64 | split_gauss_s1 | 3.17e-3 | -0.22% | 1.46e-2 | 0.0201 |
| wdas | ours | 64 | split_gauss_s2 | 6.34e-3 | -0.22% | 4.96e-2 | 0.0299 |
| wdas | ours | 64 | split_oidn_fade0.05 | 1.48e-3 | -0.74% | 5.71e-3 | 0.0159 |
| wdas | pbrt | 1 | raw | 1.94e+0 | 0.17% | 5.03e+0 | 0.0819 |
| wdas | pbrt | 1 | oidn_all | 7.13e-3 | -2.14% | 3.27e-2 | 0.0440 |
| wdas | pbrt | 1 | split_oidn | 1.27e-2 | -2.19% | 4.82e-2 | 0.0382 |
| wdas | pbrt | 1 | oidn_all_g | 8.75e-3 | -5.00% | 3.70e-2 | 0.0537 |
| wdas | pbrt | 1 | split_oidn_g | 1.40e-2 | -4.27% | 5.34e-2 | 0.0434 |
| wdas | pbrt | 1 | blur_all_s4 | 1.65e-2 | 0.17% | 6.97e-2 | 0.0492 |
| wdas | pbrt | 1 | blur_all_s2 | 4.00e-2 | 0.17% | 1.07e-1 | 0.0541 |
| wdas | pbrt | 1 | split_gauss_s4 | 3.41e-2 | 0.17% | 1.59e-1 | 0.0725 |
| wdas | pbrt | 1 | split_gauss_s2 | 5.01e-2 | 0.17% | 1.53e-1 | 0.0644 |
| wdas | pbrt | 1 | split_oidn_fade0.05 | 1.27e-2 | -2.19% | 4.82e-2 | 0.0382 |
| wdas | pbrt | 1024 | raw | 2.88e-3 | -0.03% | 6.38e-3 | 0.0113 |
| wdas | pbrt | 1024 | oidn_all | 2.00e-3 | -0.18% | 4.89e-3 | 0.0161 |
| wdas | pbrt | 1024 | split_oidn | 1.98e-3 | -0.09% | 5.07e-3 | 0.0116 |
| wdas | pbrt | 1024 | oidn_all_g | 2.31e-3 | 0.01% | 5.76e-3 | 0.0174 |
| wdas | pbrt | 1024 | split_oidn_g | 2.38e-3 | -0.02% | 5.87e-3 | 0.0116 |
| wdas | pbrt | 1024 | blur_all_s0.75 | 2.08e-3 | -0.03% | 6.12e-3 | 0.0125 |
| wdas | pbrt | 1024 | blur_all_s2 | 4.00e-3 | -0.03% | 2.27e-2 | 0.0210 |
| wdas | pbrt | 1024 | split_gauss_s0.75 | 2.42e-3 | -0.03% | 8.53e-3 | 0.0142 |
| wdas | pbrt | 1024 | split_gauss_s2 | 7.32e-3 | -0.03% | 4.51e-2 | 0.0300 |
| wdas | pbrt | 1024 | split_oidn_fade0.05 | 1.97e-3 | -0.14% | 4.96e-3 | 0.0114 |
| wdas | pbrt | 16 | raw | 8.27e-2 | -0.38% | 1.56e-1 | 0.0357 |
| wdas | pbrt | 16 | oidn_all | 4.10e-3 | -2.25% | 1.64e-2 | 0.0301 |
| wdas | pbrt | 16 | split_oidn | 3.90e-3 | -1.15% | 1.36e-2 | 0.0223 |
| wdas | pbrt | 16 | oidn_all_g | 6.05e-3 | -1.29% | 2.44e-2 | 0.0307 |
| wdas | pbrt | 16 | split_oidn_g | 4.35e-3 | -1.25% | 1.58e-2 | 0.0230 |
| wdas | pbrt | 16 | blur_all_s2 | 5.43e-3 | -0.38% | 2.49e-2 | 0.0280 |
| wdas | pbrt | 16 | split_gauss_s1.5 | 7.89e-3 | -0.38% | 3.31e-2 | 0.0324 |
| wdas | pbrt | 16 | split_gauss_s2 | 8.89e-3 | -0.38% | 4.73e-2 | 0.0358 |
| wdas | pbrt | 16 | split_oidn_fade0.05 | 3.87e-3 | -1.23% | 1.36e-2 | 0.0222 |
| wdas | pbrt | 256 | raw | 6.68e-3 | -0.08% | 1.40e-2 | 0.0145 |
| wdas | pbrt | 256 | oidn_all | 3.69e-3 | -0.36% | 8.22e-3 | 0.0191 |
| wdas | pbrt | 256 | split_oidn | 2.30e-3 | -0.17% | 6.12e-3 | 0.0138 |
| wdas | pbrt | 256 | oidn_all_g | 2.98e-3 | -0.15% | 8.17e-3 | 0.0199 |
| wdas | pbrt | 256 | split_oidn_g | 2.97e-3 | -0.18% | 7.66e-3 | 0.0138 |
| wdas | pbrt | 256 | blur_all_s1 | 2.61e-3 | -0.08% | 8.87e-3 | 0.0154 |
| wdas | pbrt | 256 | blur_all_s2 | 4.06e-3 | -0.08% | 2.27e-2 | 0.0216 |
| wdas | pbrt | 256 | split_gauss_s0.75 | 2.98e-3 | -0.08% | 9.54e-3 | 0.0161 |
| wdas | pbrt | 256 | split_gauss_s2 | 7.37e-3 | -0.08% | 4.50e-2 | 0.0305 |
| wdas | pbrt | 256 | split_oidn_fade0.05 | 2.30e-3 | -0.25% | 6.04e-3 | 0.0137 |
| wdas | pbrt | 4 | raw | 3.26e-1 | -0.79% | 7.00e-1 | 0.0565 |
| wdas | pbrt | 4 | oidn_all | 6.24e-3 | -2.68% | 2.71e-2 | 0.0385 |
| wdas | pbrt | 4 | split_oidn | 6.48e-3 | -1.02% | 2.32e-2 | 0.0285 |
| wdas | pbrt | 4 | oidn_all_g | 8.21e-3 | -3.01% | 3.87e-2 | 0.0418 |
| wdas | pbrt | 4 | split_oidn_g | 7.10e-3 | -2.89% | 2.77e-2 | 0.0333 |
| wdas | pbrt | 4 | blur_all_s3 | 8.68e-3 | -0.79% | 4.38e-2 | 0.0395 |
| wdas | pbrt | 4 | blur_all_s2 | 1.01e-2 | -0.79% | 3.57e-2 | 0.0384 |
| wdas | pbrt | 4 | split_gauss_s2 | 1.48e-2 | -0.79% | 6.14e-2 | 0.0467 |
| wdas | pbrt | 4 | split_oidn_fade0.05 | 6.45e-3 | -1.38% | 2.34e-2 | 0.0285 |
| wdas | pbrt | 64 | raw | 2.20e-2 | -0.28% | 4.39e-2 | 0.0221 |
| wdas | pbrt | 64 | oidn_all | 5.30e-3 | -1.27% | 1.42e-2 | 0.0245 |
| wdas | pbrt | 64 | split_oidn | 2.88e-3 | -0.70% | 8.81e-3 | 0.0170 |
| wdas | pbrt | 64 | oidn_all_g | 3.82e-3 | -0.62% | 1.23e-2 | 0.0245 |
| wdas | pbrt | 64 | split_oidn_g | 3.29e-3 | -0.63% | 9.68e-3 | 0.0174 |
| wdas | pbrt | 64 | blur_all_s1.5 | 3.69e-3 | -0.28% | 1.57e-2 | 0.0209 |
| wdas | pbrt | 64 | blur_all_s2 | 4.32e-3 | -0.28% | 2.29e-2 | 0.0234 |
| wdas | pbrt | 64 | split_gauss_s1 | 4.55e-3 | -0.28% | 1.62e-2 | 0.0223 |
| wdas | pbrt | 64 | split_gauss_s2 | 7.68e-3 | -0.28% | 4.52e-2 | 0.0322 |
| wdas | pbrt | 64 | split_oidn_fade0.05 | 2.86e-3 | -0.80% | 8.64e-3 | 0.0170 |

Fade sweep (relMSE vs our 8192 spp render; raw and split_oidn for reference; seeds averaged):

| scene | spp | raw | split_oidn | rho0 0.05 | rho0 0.1 | rho0 0.2 | rho0 0.4 | rho0 0.8 |
|---|---|---|---|---|---|---|---|---|
| cornell | 1 | 1.01e+0 | 7.28e-2 | 7.28e-2 | 7.28e-2 | 7.28e-2 | 7.28e-2 | 7.28e-2 |
| cornell | 1024 | 1.35e-3 | 2.81e-4 | 3.70e-4 | 5.73e-4 | 8.52e-4 | 1.07e-3 | 1.20e-3 |
| cornell | 16 | 8.67e-2 | 3.33e-3 | 3.39e-3 | 3.73e-3 | 5.68e-3 | 1.02e-2 | 1.77e-2 |
| cornell | 256 | 4.88e-3 | 4.60e-4 | 5.16e-4 | 8.80e-4 | 1.50e-3 | 2.45e-3 | 3.45e-3 |
| cornell | 4 | 3.13e-1 | 1.20e-2 | 1.23e-2 | 1.31e-2 | 1.60e-2 | 2.37e-2 | 3.98e-2 |
| cornell | 64 | 1.92e-2 | 1.01e-3 | 1.04e-3 | 1.45e-3 | 2.77e-3 | 4.76e-3 | 8.09e-3 |
| wdas | 1 | 1.81e+0 | 1.15e-2 | 1.15e-2 | 1.15e-2 | 1.15e-2 | 1.15e-2 | 1.15e-2 |
| wdas | 1024 | 1.40e-3 | 5.04e-4 | 5.02e-4 | 6.11e-4 | 9.04e-4 | 1.13e-3 | 1.26e-3 |
| wdas | 16 | 8.05e-2 | 2.54e-3 | 2.51e-3 | 2.85e-3 | 5.17e-3 | 8.65e-3 | 1.52e-2 |
| wdas | 256 | 5.25e-3 | 8.59e-4 | 8.69e-4 | 1.07e-3 | 1.62e-3 | 2.86e-3 | 3.91e-3 |
| wdas | 4 | 3.18e-1 | 5.12e-3 | 5.15e-3 | 5.75e-3 | 7.54e-3 | 1.16e-2 | 2.04e-2 |
| wdas | 64 | 2.05e-2 | 1.48e-3 | 1.48e-3 | 1.93e-3 | 3.21e-3 | 5.03e-3 | 9.60e-3 |

- cornell|1: order01 share of mean radiance 59%
- cornell|1024: order01 share of mean radiance 57%
- cornell|16: order01 share of mean radiance 57%
- cornell|256: order01 share of mean radiance 57%
- cornell|4: order01 share of mean radiance 57%
- cornell|64: order01 share of mean radiance 57%
- cornell|edge: edge mask 17% of pixels (|grad log(L+0.05)| > 0.25, dilated 3 px)
- wdas|1: order01 share of mean radiance 65%
- wdas|1024: order01 share of mean radiance 65%
- wdas|16: order01 share of mean radiance 65%
- wdas|256: order01 share of mean radiance 65%
- wdas|4: order01 share of mean radiance 66%
- wdas|64: order01 share of mean radiance 65%
- wdas|edge: edge mask 9% of pixels (|grad log(L+0.05)| > 0.25, dilated 3 px)
