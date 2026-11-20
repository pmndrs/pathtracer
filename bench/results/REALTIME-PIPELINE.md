# Real-time pipeline, no cache (P4.5 / P4.6 / P4.7)

The library's real-time path through the same chain track N built (`bench/realtime/chain.ts`): `PtMegakernel` in `mode: 'realtime'` (frame planes, blue-noise
sampler, maxDepth 2, optional ReSTIR DI) at 960 x 540 -> Denoiser `DiffuseSpecularDenoiser` (SVGF + anti-lag, remodulated) -> pmndrs/upscaler temporal
path 2x (jitter: false) -> tonemap. Design and wiring: [docs/internal/design/realtime.md section 5](../../docs/internal/design/realtime.md). Page: `realtime-pipeline`
(`node bench/runner/run.ts --page realtime-pipeline --timeout 3600000 --args scenes=... scenarios=static,control,path,timing ...`).

**Machine and conditions.** Apple M5 Pro, Chrome 157 headless, Metal 3, 2026-11-18, runner load average 3-17 at the start of each run (other sessions' Chrome jobs on this
Mac; every run is `--load-wait 0`, so the runner did not wait for quiet). Quality numbers do not depend on load. **Timing is PROVISIONAL** (batched wall clock, no GPU
timestamps, queued for the quiet pass: docs/internal/QUIET-PASS.md "P4.5 / P4.6").

**References** are our own plain path tracer at the chain's path length (NEE, maxDepth 2, no Russian roulette), rendered at the display size (1920 x 1080, 1024 spp
static; 512 x 512, 2048 spp for many-lights-64; 256 spp per pose on the path) and tent-resampled to the render size for the raw / denoised stages. The scene's own pbrt /
Mitsuba references are 8-vertex images (28% brighter on sponza-white: mean luma 0.241 against 0.173) and cannot judge a 2-bounce chain. The reference's own noise floor
is below the chain's errors here (the control row below measures the resolution floor of the 2x reconstruction: relMSE 0.0154). relBias is (mean test - mean reference) / mean reference.

**Metrics.** relMSE / relBias / LDR-FLIP as everywhere (`bench/metrics`), stage by stage (raw = the frame's summed sample at the render size, denoised = after SVGF + composite, final =
after the upscaler at the display size). Flicker = per-pixel mean |dL| between consecutive frames over frames 32..64, divided by (reference luma + 0.01), averaged. Boil = the Denoiser
scorecard's "calm" (second temporal difference along the reprojected history, tonemapped luma, `bench/realtime/boil.ts`, unit-tested: 0 for a noise-free image under exact reprojection, sigma for
independent noise of standard deviation sigma), simplified (V191): overall rather than per region (static / moving / disoccluded), depth-consistency validity (96% of probed pixels valid at k = 1, 88% at k = 4), nearest-tap motion,
k = 1 and 4, no per-frame reference so no reference floor (the albedo control, a noise-free image through the same reprojection, is the floor: 0.0002-0.0031). Compare chains on one path, not scenes.

## Findings

1. **Static camera, sponza-white, maxDepth 2 (SVGF, 64 frames):** raw 1-spp relMSE 4.5 -> denoised 0.119 (frame 1) -> 0.065 (frame 64) -> final 0.173 -> 0.056 (floor
   0.015); final LDR-FLIP 0.170 -> 0.103. Flicker (mean |dL| per frame / reference luma) raw 1.41, denoised 0.104, final 0.0045. The plane identity (diffuse x albedo +
   specular x specular factor = beauty) is exact (max relative difference 0, also with ReSTIR).
2. **The upscaler's darkening (N11.5) depends on the noise it is fed:** noise-free input (the reference, `control`): **+0.2%**; behind SVGF (residual noise): **-1.5% at frame 1 growing to
   -7.3% at frame 64** (denoised stage -1.2%); raw 1-spp input: **-90%** (and +30% at its first frame). So it is the temporal path's treatment of noisy HDR input (Reinhard-space
   accumulation and neighbourhood clipping of highlights), not an offset or the sharpening; the noise-free control rules out a constant bias. Reactive mask: bias -4.8% instead of -7.0% at the last pose, but 2.3x the boil.
3. **Moving camera (64-frame orbit + dolly, boil = Denoiser scorecard "calm" at k = 1 / 4 on the final stage, lower is steadier; the albedo control, a noise-free image through the same reprojection, reads 0.0002):**

   | chain | boil k=1 | boil k=4 | final relMSE at the last pose | final FLIP at the last pose | final bias at the last pose |
   |---|---|---|---|---|---|
   | SVGF (default) | **0.0036** | **0.0056** | **0.0249** | **0.110** | -7.0% |
   | SVGF + reactive mask (cap 8) | 0.0082 | 0.0104 | 0.0293 | 0.107 | -4.8% |
   | OIDN small every frame (the preview-mode denoiser) | 0.0090 | 0.0109 | 0.0300 | 0.137 | -4.1% |
   | no denoiser (raw into the upscaler) | 0.0050 | 0.0069 | 0.273 | 0.500 | -95% |

   The SVGF chain is **2.5x (k=1) / 1.9x (k=4) steadier than the preview pipeline's OIDN small at the same render size and path length, with 20% lower FLIP**; OIDN has the lower relMSE early in the path
   (0.056 against 0.119 at pose 21). P4.7's own configuration (640 x 360 -> 1280 x 720, maxDepth 4): boil 0.0038 against 0.0107 (2.8x / 2.1x), FLIP 0.129 against 0.166, relMSE 0.038 against 0.041 at the last pose (OIDN 0.060 against 0.084 at pose 21).
   The "raw into the upscaler" row looks calm because the upscaler averages the noise away and the image is 95% too dark: boil is meaningless without the error columns.
4. **ReSTIR DI through the chain (biased = default settings; unbiased = balance-heuristic MIS without initial visibility; frame 64, final stage):**

   | scene (depth) | NEE relMSE / FLIP / bias | ReSTIR biased | ReSTIR unbiased | cost (trace ms, provisional) |
   |---|---|---|---|---|
   | many-lights-64 (2) | 5.5e-3 / 0.0636 / -2.9% | 3.9e-3 (1.4x) / 0.0626 / -3.8% | **2.9e-3 (1.9x) / 0.0455 (-28%)** / -1.7% | 0.34 -> 0.52 (biased), frame 0.90 -> 1.05 ms |
   | sponza-lamps, direct light only (1) | 0.095 / 0.0578 / -14.2% | 0.021 (4.5x) / 0.0528 / -16.8% | **0.034 (2.8x) / 0.0415 (-28%)** / -8.4% | 8.4 -> 13.2 (biased), 10.9 (unbiased) |
   | sponza-lamps (2) | 0.121 / 0.202 / -11.4% | 0.087 (1.4x) / 0.212 / -15.9% | 0.111 (1.1x) / 0.2025 / -10.0% | 15.7 -> 20.6 (biased), 19.1 (unbiased) |
   | sponza-white, sun + sky (2) | 0.056 / 0.103 / -7.3% | 0.037 (1.5x) / 0.133 / -15.8% | 0.056 (1.0x) / 0.166 / -9.7% | 15.2 -> 20.6 (biased), 17.5 (unbiased) |

   Raw 1-spp relMSE with ReSTIR: many-lights-64 3.3 -> 0.14, sponza-lamps depth 1 8.0 -> 0.13 (biased) / 0.58 (unbiased). **ReSTIR pays where there are many lights and the first vertex's direct light dominates
   the noise** (unbiased: 1.9-2.8x relMSE and 28% lower FLIP at +13..30% frame time); with a second bounce (sponza-lamps depth 2) the indirect vertex's NEE, which ReSTIR does not touch, is the
   remaining noise (1.1x). On a sun + sky scene there is nothing to resample: the biased mode only darkens (-15.8%, FLIP 30% worse) and the unbiased mode (no MIS with BSDF sampling at the first vertex, V134) has twice the raw noise.
   Biased mode's bias grows with the frames (sponza-lamps depth 1: -0.1% at frame 1, -12.4% raw at frame 64): never read its relMSE without relBias and FLIP.
   With ReSTIR the sampler is Owen-Sobol (modes.ts), so the on/off pair also differs in sampler; blue-noise-vs-Owen-Sobol was +-1.5x after a low-pass (realtime.md section 4).
5. **Cut / lag.** After a cut the SVGF final needs 32 frames to come within 10% of its frame-64 relMSE (the filter's history is 64 frames); frame 1 is already usable (final relMSE 0.17,
   denoised 0.12).

6. **Emitters seen by the primary ray (P4.5a, V200-V202; the emission plane).** Before, an emitter's radiance was demodulated by its albedo floor (1e-3) and reached SVGF as emission / 1e-3 (f16 clamp
   65000 in the pack: 1238-1264 clamped pixels on `sponza-lamps-visible`), which the filter spread over the neighbourhood. Now plane 6 (`emission`) carries the light the primary BSDF does not modulate
   (emitter at the primary hit, the sky behind a miss, perfect-mirror chains) and the composite adds it AFTER the filter, before the upscaler. Same code, `frameEmission` none (the old behaviour; reproduces the
   pre-change run to the last digit) against `mirror` (default; `primary` is identical on these scenes, which have no mirror), static camera, 960 x 540 -> 1080p (Cornell 512 -> 1024), 64 frames, SVGF, own depth-2
   references at 1024 spp; `emit` = reference luminance > 8 dilated by one pixel (lamps: 0.65% of the pixels and about 64% of the image's energy, solved from the three biases; Cornell's light: 0.64% of the pixels, about 61%), `rest` = everything else:

   | scene, frame 64 | metric | none (before) | emission plane (after) |
   |---|---|---|---|
   | sponza-lamps-visible | denoised relMSE (emit / rest) | 228 (1730 / 214) | **0.14 (2.7 / 0.12)** |
   | | final relMSE (emit / rest) | 0.43 (30.8 / 0.23) | **0.062 (2.1 / 0.049)** (sponza-white: 0.056) |
   | | denoised / final bias | -24.3% / -37.6% | -0.8% / -25.6% (rest -4.8%, emit -37.1%: see below) |
   | | final FLIP | 0.128 | 0.126 |
   | | f16 clamp hits (diffuse + specular + raw) | 1264 | 0 |
   | | flicker (mean abs change / ref luma), denoised: emit / rest | 7.77 / 0.26 | 0.43 / 0.115 |
   | | flicker final: emit / rest; mean abs change on emit pixels | 0.068 / 0.0097; 0.34 | 0.043 / 0.0071; 0.53 |
   | cornell, light in view | denoised relMSE (emit / rest) | 1.0e4 (3.0e4 / 9.9e3) | **0.0022 (0.17 / 0.0011)** |
   | | final relMSE (emit / rest) | 857 (449 / 859) | **0.0013 (0.12 / 0.00055)** |
   | | denoised / final bias | +1676% / +182% | +0.05% / -8.1% (rest -1.2%, emit -12.4%) |
   | | final FLIP | 0.085 | 0.043 |
   | | flicker final: emit / rest | 0.21 / 0.082 | 0.0092 / 0.00085 |
   | sponza-white (no emitter in view), static + 64-frame orbit | every metric | raw 4.52, denoised 0.0652, final 0.0563, FLIP 0.1029, flicker 1.407 / 0.1041 / 0.00453, boil 0.0036 (k = 1), 0.0056 (k = 4), pose errors 0.118 / 0.055 / 0.0249 | identical to the last printed digit (and to the numbers in finding 1 and 3: the chain did not move) |

   What it does not fix: (a) the lamps are 1.5 px across at 960 x 540, so their silhouette is a binary 1-spp coverage sample and the emission plane, unfiltered, shows it: the absolute frame-to-frame change on emitter pixels
   is 0.34 -> 0.53 (the lamps are now at full brightness instead of smeared) and the emitter-region relMSE (2.7 denoised, 2.1 final) is edge coverage, as the raw sample's 3.7 (TASKS P4.5c: temporal accumulation of the plane).
   (b) The upscaler loses 37% of the lamps' energy (final bias on emit pixels -37.1% against +0.3% at the denoised stage; Cornell's light -12.4% against +0.1%; the rest of the image -4.8% / -1.2%): small very bright
   features are clipped by its neighbourhood clamp, the same family as the noisy-HDR darkening of finding 2 (TASKS P4.6a). The whole-image relBias of a lamp scene is mostly the lamps' energy.
   Plane identity `diffuse x albedo + specular x specularFactor + emission = radiance`: exact (<= 1.1e-7; 0 on sponza-white). Closed forms (`realtime-check check=emission`: a lamp is its radiance with an empty
   diffuse plane, the sky is the emission of a missed ray, a mirror's lamp reflection is emission in `mirror` and specular-class radiance in `primary`, a mirror's lit floor never emission) pass in all three modes.
   Cost (sponza-white, provisional, load 4-5, 3 interleaved pairs, ms): trace 15.15-15.21 `mirror` against 15.19-15.22 `none` (no measurable cost), accumulate + pack 1.56 against 1.56 (the old chain's 1.35 was another session: the
   extra emission section is about +0.2 ms of traffic by bandwidth), whole frame (median of rounds) 18.9-19.2 against 19.0-21.3 (the first `none` run read 21.3 under a burst; its min was 20.3). Kernel text: `rt-frame` 44.0 KB before, 44.4 `none`, 44.5 `primary`, 45.2 `mirror`.
   Run: `node bench/runner/run.ts --page realtime-pipeline --timeout 3600000 --load-wait 0 --args scenes=sponza-lamps-visible,cornell scenarios=static variants=svgf emitThreshold=8 refSpp=1024 checkpoints=1,4,16,64 [ad=frameEmission:none]`.

## Tables

Frames 1 / 4 / 16 / 64 after a cut at the scene camera; `path` rows: boil and per-pose error of the orbit.

### sponza-white, NEE (the real-time preset, blue noise), 960x540 -> 1920x1080

Run `20261118T023607Z-realtime-pipeline` (`--args scenes=sponza-white scenarios=static,control,path,timing refSpp=1024 pathRefSpp=256`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-white_nee | svgf | 1 | 4.55e+0 | 1.19e-1 | 1.73e-1 | 0.4% / -0.4% / -1.5% | 0.1700 |
| sponza-white_nee | svgf | 4 | 4.54e+0 | 7.43e-2 | 9.25e-2 | 0.6% / -2.9% / -5.6% | 0.1398 |
| sponza-white_nee | svgf | 16 | 4.48e+0 | 6.96e-2 | 6.65e-2 | -0.2% / -1.3% / -6.8% | 0.1203 |
| sponza-white_nee | svgf | 64 | 4.52e+0 | 6.52e-2 | 5.63e-2 | -0.1% / -1.2% / -7.3% | 0.1029 |
| sponza-white_nee | none | 1 | 4.41e+0 | 4.41e+0 | 7.00e+0 | 0.1% / 0.1% / 29.5% | 0.4868 |
| sponza-white_nee | none | 4 | 4.53e+0 | 4.53e+0 | 2.83e-1 | -0.1% / -0.1% / -86.1% | 0.5377 |
| sponza-white_nee | none | 16 | 4.58e+0 | 4.58e+0 | 2.97e-1 | 0.3% / 0.3% / -89.1% | 0.5502 |
| sponza-white_nee | none | 64 | 4.49e+0 | 4.49e+0 | 2.99e-1 | -0.1% / -0.1% / -89.8% | 0.5511 |

| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |
|---|---|---|---|---|---|---|---|
| sponza-white_nee | svgf | 0.0036 | 0.0056 | 0.0125 | 0.0002 | 96% | f21: relMSE 1.19e-1, bias -5.9%, FLIP 0.1291; f42: relMSE 5.51e-2, bias -5.3%, FLIP 0.1204; f63: relMSE 2.49e-2, bias -7.0%, FLIP 0.1104 |
| sponza-white_nee | svgf-reactive | 0.0082 | 0.0104 | 0.0125 | 0.0002 | 96% | f21: relMSE 1.29e-1, bias -3.9%, FLIP 0.1281; f42: relMSE 5.50e-2, bias -4.2%, FLIP 0.1217; f63: relMSE 2.93e-2, bias -4.8%, FLIP 0.1067 |
| sponza-white_nee | oidn-fast | 0.0090 | 0.0109 | 0.0149 | 0.0002 | 96% | f21: relMSE 5.61e-2, bias -4.8%, FLIP 0.1629; f42: relMSE 4.32e-2, bias -3.5%, FLIP 0.1539; f63: relMSE 3.00e-2, bias -4.1%, FLIP 0.1373 |
| sponza-white_nee | none | 0.0050 | 0.0069 | 0.0981 | 0.0002 | 96% | f21: relMSE 3.05e-1, bias -91.7%, FLIP 0.5666; f42: relMSE 3.11e-1, bias -94.8%, FLIP 0.5529; f63: relMSE 2.73e-1, bias -95.4%, FLIP 0.5000 |

Control (N11.5), the upscaler alone on the noise-free reference (resampled to 960x540), 32 static frames: final relBias +0.22%, relMSE 0.0154 (the resolution floor of a 2x reconstruction; plain 2x resample of the same input: 0.0334).

### sponza-white, ReSTIR (biased, 8 candidates, 1 spatial, initial visibility; Owen-Sobol)

Run `20261118T025530Z-realtime-pipeline` (`--args scenes=sponza-white restir=on scenarios=static,path,timing variants=svgf pathVariants=svgf refSpp=1024 pathRefSpp=256`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-white_restir | svgf | 1 | 6.73e+0 | 1.47e-1 | 1.92e-1 | -2.0% / -2.7% / -3.8% | 0.1934 |
| sponza-white_restir | svgf | 4 | 4.09e+0 | 8.11e-2 | 8.23e-2 | -5.4% / -7.3% / -9.3% | 0.1683 |
| sponza-white_restir | svgf | 16 | 4.00e+0 | 5.77e-2 | 4.93e-2 | -7.0% / -8.3% / -12.5% | 0.1478 |
| sponza-white_restir | svgf | 64 | 3.90e+0 | 5.36e-2 | 3.68e-2 | -9.1% / -10.3% / -15.8% | 0.1332 |

| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |
|---|---|---|---|---|---|---|---|
| sponza-white_restir | svgf | 0.0037 | 0.0059 | 0.0128 | 0.0002 | 96% | f21: relMSE 7.29e-2, bias -13.9%, FLIP 0.1521; f42: relMSE 4.61e-2, bias -14.7%, FLIP 0.1448; f63: relMSE 2.66e-2, bias -15.0%, FLIP 0.1267 |

### sponza-white, ReSTIR unbiased (balance-heuristic MIS, no initial visibility)

Run `20261118T030606Z-realtime-pipeline` (`--args scenes=sponza-white restir=on scenarios=static,path,timing variants=svgf pathVariants=svgf refSpp=1024 pathRefSpp=256 bias=unbiased vis=0`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-white_restir | svgf | 1 | 8.60e+0 | 1.54e-1 | 2.10e-1 | -0.1% / -1.1% / -2.1% | 0.2000 |
| sponza-white_restir | svgf | 4 | 8.46e+0 | 1.12e-1 | 9.31e-2 | -0.1% / -5.0% / -7.6% | 0.1803 |
| sponza-white_restir | svgf | 16 | 8.82e+0 | 1.02e-1 | 6.65e-2 | 2.1% / -2.0% / -9.4% | 0.1791 |
| sponza-white_restir | svgf | 64 | 8.93e+0 | 9.09e-2 | 5.58e-2 | -0.4% / -2.8% / -9.7% | 0.1656 |

| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |
|---|---|---|---|---|---|---|---|
| sponza-white_restir | svgf | 0.0041 | 0.0076 | 0.0138 | 0.0002 | 96% | f21: relMSE 1.10e-1, bias -7.6%, FLIP 0.1941; f42: relMSE 6.91e-2, bias -8.0%, FLIP 0.1809; f63: relMSE 3.33e-2, bias -9.8%, FLIP 0.1513 |

### sponza-lamps (64 lamps), maxDepth 2: NEE vs ReSTIR biased

Run `20261118T024255Z-realtime-pipeline` (`--args scenes=sponza-lamps restir=off,on scenarios=static,path,timing variants=svgf pathVariants=svgf refSpp=1024 pathRefSpp=256`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-lamps_nee | svgf | 1 | 5.33e+1 | 6.65e-1 | 5.99e-1 | -0.7% / -5.4% / -5.6% | 0.3091 |
| sponza-lamps_nee | svgf | 4 | 6.36e+1 | 5.41e-1 | 2.18e-1 | 2.4% / -15.2% / -14.2% | 0.2676 |
| sponza-lamps_nee | svgf | 16 | 5.90e+1 | 2.93e-1 | 1.37e-1 | -0.4% / -8.6% / -14.1% | 0.2334 |
| sponza-lamps_nee | svgf | 64 | 5.94e+1 | 2.82e-1 | 1.21e-1 | -0.1% / -8.2% / -11.4% | 0.2020 |
| sponza-lamps_restir | svgf | 1 | 5.58e+1 | 9.40e-1 | 8.54e-1 | 0.3% / -3.8% / -4.0% | 0.2916 |
| sponza-lamps_restir | svgf | 4 | 5.01e+1 | 4.14e-1 | 3.02e-1 | -1.4% / -11.9% / -11.3% | 0.2574 |
| sponza-lamps_restir | svgf | 16 | 5.21e+1 | 2.69e-1 | 1.37e-1 | -5.5% / -11.9% / -14.3% | 0.2328 |
| sponza-lamps_restir | svgf | 64 | 5.96e+1 | 2.39e-1 | 8.72e-2 | -6.8% / -13.4% / -15.9% | 0.2119 |

| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |
|---|---|---|---|---|---|---|---|
| sponza-lamps_nee | svgf | 0.0075 | 0.0132 | 0.0208 | 0.0020 | 96% | f21: relMSE 2.32e-1, bias -10.0%, FLIP 0.2443; f42: relMSE 2.56e-1, bias -9.4%, FLIP 0.2446; f63: relMSE 2.27e-1, bias -11.5%, FLIP 0.2349 |
| sponza-lamps_restir | svgf | 0.0071 | 0.0124 | 0.0194 | 0.0020 | 96% | f21: relMSE 2.02e-1, bias -11.3%, FLIP 0.2433; f42: relMSE 2.07e-1, bias -14.7%, FLIP 0.2475; f63: relMSE 1.84e-1, bias -15.9%, FLIP 0.2337 |

### sponza-lamps, maxDepth 2, ReSTIR unbiased

Run `20261118T030353Z-realtime-pipeline` (`--args scenes=sponza-lamps restir=on scenarios=static,timing variants=svgf refSpp=1024 maxDepth=2 bias=unbiased vis=0`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-lamps_restir | svgf | 1 | 5.59e+1 | 1.80e+0 | 1.72e+0 | 3.3% / -1.6% / -1.9% | 0.2915 |
| sponza-lamps_restir | svgf | 4 | 5.02e+1 | 6.42e-1 | 6.55e-1 | 2.5% / -9.4% / -9.0% | 0.2563 |
| sponza-lamps_restir | svgf | 16 | 5.22e+1 | 4.65e-1 | 2.53e-1 | 1.6% / -5.3% / -10.2% | 0.2290 |
| sponza-lamps_restir | svgf | 64 | 6.00e+1 | 2.76e-1 | 1.11e-1 | 1.1% / -7.0% / -10.0% | 0.2025 |

### sponza-lamps, maxDepth 1 (direct light only): NEE vs ReSTIR biased

Run `20261118T025355Z-realtime-pipeline` (`--args scenes=sponza-lamps restir=off,on scenarios=static,timing variants=svgf refSpp=1024 maxDepth=1`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-lamps_nee | svgf | 1 | 8.12e+0 | 4.10e-1 | 4.16e-1 | 0.1% / -4.1% / -5.1% | 0.0799 |
| sponza-lamps_nee | svgf | 4 | 8.29e+0 | 1.76e-1 | 1.73e-1 | 1.0% / -15.2% / -13.8% | 0.0718 |
| sponza-lamps_nee | svgf | 16 | 7.60e+0 | 1.74e-1 | 1.10e-1 | 0.2% / -6.6% / -14.6% | 0.0639 |
| sponza-lamps_nee | svgf | 64 | 8.01e+0 | 1.63e-1 | 9.50e-2 | -0.3% / -7.2% / -14.2% | 0.0578 |
| sponza-lamps_restir | svgf | 1 | 1.45e+0 | 6.60e-1 | 6.61e-1 | -0.1% / -3.5% / -4.2% | 0.0607 |
| sponza-lamps_restir | svgf | 4 | 3.30e-1 | 2.07e-1 | 2.35e-1 | -3.5% / -9.5% / -9.2% | 0.0545 |
| sponza-lamps_restir | svgf | 16 | 1.41e-1 | 3.19e-2 | 5.60e-2 | -9.0% / -11.3% / -12.6% | 0.0496 |
| sponza-lamps_restir | svgf | 64 | 1.32e-1 | 2.83e-2 | 2.11e-2 | -12.4% / -13.7% / -16.8% | 0.0528 |

### sponza-lamps, maxDepth 1, ReSTIR unbiased

Run `20261118T030258Z-realtime-pipeline` (`--args scenes=sponza-lamps restir=on scenarios=static,timing variants=svgf refSpp=1024 maxDepth=1 bias=unbiased vis=0`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-lamps_restir | svgf | 1 | 1.63e+0 | 1.73e+0 | 1.69e+0 | 4.9% / 0.1% / -0.8% | 0.0618 |
| sponza-lamps_restir | svgf | 4 | 4.65e-1 | 4.89e-1 | 5.96e-1 | 2.9% / -5.3% / -5.7% | 0.0543 |
| sponza-lamps_restir | svgf | 16 | 2.50e-1 | 3.10e-1 | 1.70e-1 | 3.0% / -0.4% / -6.4% | 0.0481 |
| sponza-lamps_restir | svgf | 64 | 5.83e-1 | 1.15e-1 | 3.41e-2 | 0.7% / -3.2% / -8.4% | 0.0415 |

### many-lights-64 (256 x 256 -> 512 x 512), maxDepth 2: NEE vs ReSTIR biased

Run `20261118T025508Z-realtime-pipeline` (`--args scenes=many-lights-64 restir=off,on scenarios=static,path,timing variants=svgf pathVariants=svgf refSpp=2048 pathRefSpp=1024 pathFrames=64`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| many-lights-64_nee | svgf | 1 | 3.33e+0 | 1.54e-2 | 1.58e-2 | 0.1% / 0.1% / 0.0% | 0.0975 |
| many-lights-64_nee | svgf | 4 | 3.37e+0 | 1.35e-2 | 9.30e-3 | 1.0% / -6.7% / -4.0% | 0.0908 |
| many-lights-64_nee | svgf | 16 | 3.38e+0 | 8.27e-3 | 6.52e-3 | 0.1% / -1.3% / -3.4% | 0.0742 |
| many-lights-64_nee | svgf | 64 | 3.31e+0 | 8.46e-3 | 5.54e-3 | -1.1% / -1.2% / -2.9% | 0.0636 |
| many-lights-64_restir | svgf | 1 | 3.86e-1 | 8.31e-3 | 8.44e-3 | -1.1% / -1.3% / -1.4% | 0.0760 |
| many-lights-64_restir | svgf | 4 | 1.41e-1 | 5.88e-3 | 4.58e-3 | -1.6% / -2.3% / -2.2% | 0.0645 |
| many-lights-64_restir | svgf | 16 | 1.23e-1 | 4.84e-3 | 3.52e-3 | -1.8% / -1.9% / -2.7% | 0.0583 |
| many-lights-64_restir | svgf | 64 | 1.39e-1 | 6.02e-3 | 3.87e-3 | -2.5% / -2.6% / -3.8% | 0.0626 |

| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |
|---|---|---|---|---|---|---|---|
| many-lights-64_nee | svgf | 0.0012 | 0.0021 | 0.0043 | 0.0031 | 99% | f21: relMSE 5.87e-3, bias -1.8%, FLIP 0.0648; f42: relMSE 7.09e-3, bias -0.9%, FLIP 0.0662; f63: relMSE 6.63e-3, bias -1.3%, FLIP 0.0641 |
| many-lights-64_restir | svgf | 0.0011 | 0.0018 | 0.0037 | 0.0030 | 99% | f21: relMSE 3.81e-3, bias -2.4%, FLIP 0.0579; f42: relMSE 4.72e-3, bias -2.2%, FLIP 0.0598; f63: relMSE 4.83e-3, bias -2.7%, FLIP 0.0611 |

### many-lights-64, ReSTIR unbiased

Run `20261118T030535Z-realtime-pipeline` (`--args scenes=many-lights-64 restir=on scenarios=static,path,timing variants=svgf pathVariants=svgf refSpp=2048 pathRefSpp=1024 bias=unbiased vis=0`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| many-lights-64_restir | svgf | 1 | 4.02e-1 | 8.72e-3 | 8.87e-3 | -0.7% / -0.9% / -1.0% | 0.0760 |
| many-lights-64_restir | svgf | 4 | 1.61e-1 | 6.16e-3 | 4.87e-3 | -0.7% / -1.5% / -1.6% | 0.0639 |
| many-lights-64_restir | svgf | 16 | 1.36e-1 | 4.88e-3 | 3.67e-3 | -0.4% / -0.3% / -1.6% | 0.0546 |
| many-lights-64_restir | svgf | 64 | 1.48e-1 | 5.06e-3 | 2.89e-3 | 0.5% / 0.3% / -1.7% | 0.0455 |

| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |
|---|---|---|---|---|---|---|---|
| many-lights-64_restir | svgf | 0.0011 | 0.0018 | 0.0037 | 0.0030 | 99% | f21: relMSE 3.57e-3, bias -0.9%, FLIP 0.0487; f42: relMSE 4.24e-3, bias 0.3%, FLIP 0.0483; f63: relMSE 3.86e-3, bias 0.6%, FLIP 0.0466 |

### P4.7 preview pipeline: 640x360 -> 1280x720, maxDepth 4, SVGF vs OIDN small every frame

Run `20261118T025838Z-realtime-pipeline` (`--args scenes=sponza-white scenarios=static,path,timing res=640x360 maxDepth=4 variants=svgf,oidn-fast pathVariants=svgf,oidn-fast refSpp=1024 pathRefSpp=256`).

| config | variant | frame | raw relMSE | denoised relMSE | final relMSE | bias raw / den / final | final FLIP |
|---|---|---|---|---|---|---|---|
| sponza-white_nee | svgf | 1 | 5.02e+0 | 1.23e-1 | 1.46e-1 | 0.3% / -0.0% / -0.8% | 0.2019 |
| sponza-white_nee | svgf | 4 | 5.04e+0 | 8.69e-2 | 7.30e-2 | -0.5% / -2.5% / -4.5% | 0.1703 |
| sponza-white_nee | svgf | 16 | 4.99e+0 | 7.23e-2 | 5.25e-2 | -0.6% / -1.0% / -5.6% | 0.1447 |
| sponza-white_nee | svgf | 64 | 5.03e+0 | 6.90e-2 | 4.41e-2 | 0.0% / -0.9% / -6.0% | 0.1213 |
| sponza-white_nee | oidn-fast | 1 | 4.97e+0 | 3.42e-2 | 6.00e-2 | -0.5% / -1.5% / -1.2% | 0.1949 |
| sponza-white_nee | oidn-fast | 4 | 4.98e+0 | 1.41e-2 | 3.53e-2 | -0.4% / 0.3% / -1.6% | 0.1463 |
| sponza-white_nee | oidn-fast | 16 | 5.00e+0 | 6.94e-3 | 2.42e-2 | 0.1% / -0.0% / -1.9% | 0.1174 |
| sponza-white_nee | oidn-fast | 64 | 5.04e+0 | 3.82e-3 | 1.52e-2 | -0.4% / -0.1% / -2.4% | 0.0884 |

| config | variant | boil final k=1 | k=4 | denoised k=1 | albedo control k=1 | valid | per-pose final error |
|---|---|---|---|---|---|---|---|
| sponza-white_nee | svgf | 0.0038 | 0.0061 | 0.0155 | 0.0003 | 95% | f21: relMSE 8.42e-2, bias -4.9%, FLIP 0.1550; f42: relMSE 6.23e-2, bias -4.1%, FLIP 0.1474; f63: relMSE 3.78e-2, bias -6.0%, FLIP 0.1290 |
| sponza-white_nee | oidn-fast | 0.0107 | 0.0127 | 0.0185 | 0.0003 | 95% | f21: relMSE 6.01e-2, bias -3.2%, FLIP 0.1904; f42: relMSE 5.12e-2, bias -3.3%, FLIP 0.1866; f63: relMSE 4.09e-2, bias -4.0%, FLIP 0.1662 |

## Timing (PROVISIONAL)

Wall clock per iteration of batched work: the kernel pass, accumulate + pack and the tonemap as N iterations in one command buffer, the post chain (the Denoiser and the upscaler submit their
own command buffers) as N queued calls awaited once, N chosen for >= 250 ms per round, 5 rounds, median (`RealtimePipelineSession.timeStages`; no GPU timestamps). "whole frame" = trace + pack + post + tonemap queued back to
back, so it can be below the sum of the stages (overlap) or above it (the stages above run with a warm cache and an idle queue). The OIDN rows await every denoise (the Denoiser package waits for its GPU work), so they are an upper
bound for a frame that could overlap. Load average at the start of each run: 2.1-17 (other sessions' Chrome jobs; the runner records it in each run's `run.json`); rows from different runs differ by the load of that moment, not only by the setting:
compare within a run (NEE and ReSTIR rows of one scene were taken one after the other in the same run, except where the label says otherwise).

| config | trace | pack | denoise + composite | upscale | tonemap | whole frame (queued) | fps | frame min..max |
|---|---|---|---|---|---|---|---|---|
| sponza-white, NEE, maxDepth 2 (960x540 -> 1920x1080) | 15.16 | 1.35 | 1.23 | 0.75 | 0.044 | 19.58 | 51 | 19.20..20.13 |
| sponza-white, ReSTIR biased + visibility | 20.63 | 1.37 | 1.59 | 0.85 | 0.053 | 31.04 | 32 | 29.30..34.15 |
| sponza-white, ReSTIR unbiased | 17.51 | 1.36 | 1.13 | 0.75 | 0.041 | 21.01 | 48 | 20.99..21.03 |
| sponza-lamps, NEE | 15.68 | 1.39 | 1.12 | 0.76 | 0.042 | 19.28 | 52 | 19.22..19.35 |
| sponza-lamps, ReSTIR biased + visibility | 20.64 | 1.37 | 1.13 | 0.76 | 0.039 | 24.15 | 41 | 24.09..24.20 |
| sponza-lamps, ReSTIR unbiased | 19.11 | 1.37 | 1.52 | 0.80 | 0.052 | 29.91 | 33 | 27.64..30.91 |
| sponza-lamps, maxDepth 1, NEE | 8.41 | 1.36 | 1.11 | 0.74 | 0.041 | 11.98 | 83 | 11.93..11.99 |
| sponza-lamps, maxDepth 1, ReSTIR biased + visibility | 13.18 | 1.35 | 1.11 | 0.75 | 0.039 | 16.61 | 60 | 16.56..16.69 |
| sponza-lamps, maxDepth 1, ReSTIR unbiased | 10.93 | 1.35 | 1.10 | 0.73 | 0.039 | 14.45 | 69 | 14.39..14.48 |
| many-lights-64 (512 x 512 display), NEE | 0.34 | 0.14 | 0.24 | 0.15 | 0.020 | 0.90 | 1113 | 0.90..0.92 |
| many-lights-64, ReSTIR biased + visibility | 0.52 | 0.14 | 0.24 | 0.15 | 0.022 | 1.05 | 951 | 1.05..1.06 |
| sponza-white, maxDepth 8, no RR (3060 ratio) | 55.52 | 1.36 | 1.22 | 0.78 | 0.045 | 61.35 | 16 | 60.00..61.75 |
| preview config: 640x360 -> 1280x720, maxDepth 4, SVGF | 13.18 | 0.56 | 0.59 | 0.37 | 0.030 | 14.92 | 67 | 14.90..14.95 |
| preview config: 640x360 -> 1280x720, maxDepth 4, OIDN small every frame (no overlap) | 13.26 | 0.56 | 4.64 | 0.38 | 0.033 | 20.12 | 50 | 19.94..20.43 |
| sponza-white 960x540 -> 1920x1080, maxDepth 2, OIDN small every frame (no overlap) | 15.22 | 1.44 | 11.84 | 0.76 | 0.050 | 33.36 | 30 | 32.34..35.79 |

min..max of the whole-frame rounds is 0.3-12% of the median (the loaded rows: sponza-white ReSTIR biased 29.3..34.2 ms, sponza-lamps unbiased 27.6..30.9 ms).

## Not measured here

- The RTX 3060 Laptop (the gate's GPU) and a quiet machine: QUIET-PASS "P4.5 / P4.6".
- Instance motion, dynamic lights, pbrt / Mitsuba references at depth 2, the upscaler-owned jitter A/B (N11.2).
