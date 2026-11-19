# Prefiltered incident radiance on isotropic conductors

Keep the outgoing-RGB compact MLP as the control. Prefiltering lowered held-out RGB error at oblique angles by 8% and raised it by 144% at grazing angles across two seeds. At grazing angles, the directly integrated split-sum prediction already lost 37% of the blue highlight energy. The learned prediction was close to that approximation and inherited its bias.

## Matched Mac comparison

Apple M5 Pro, headless Chrome 157. Each arm used the same compact tiled fp16 network, reflected-direction input, four position frequency bands, 512 optimizer steps, 4,096 records per step, and 32 traced suffix samples per record. The camera ray hit a planar mirror before reaching the cached surface. On the second hit, both arms used the same positions, views, roughness values, material, seed sequence and optimizer settings. There was no still refinement.

Six held-out 72 × 48 camera views show the same finite emissive cards, small occluder and roughness strips. The roughness strips run from 0.18 to 0.62. Minimum oblique N·V was 0.588. Maximum grazing N·V was 0.196. The reference traces 4,096 GGX samples per pixel against the fixture geometry. Metrics cover eligible conductor pixels. Both arms exactly preserved the 6,912 traced delta-mirror and glass pixels. This is a standalone shader fixture with analytic ray intersections and the renderer's GGX BSDF. It isolates the specular target and reconstruction from the rest of the production renderer. [Fixture and estimators](../nrc/prefiltered-specular.ts) · [Study runner](../pages/nrc-prefiltered-specular.ts)

| View | Arm | Relative RGB RMSE | Luminance bias | Blue peak | Mean blue energy | Blue RMS width |
|---|---|---:|---:|---:|---:|---:|
| Oblique | Traced reference | — | — | 4.89 | 0.611 | 11.01 px |
| Oblique | Integrated split sum | 0.1165 | −6.1% | 4.48 | 0.565 | 10.63 px |
| Oblique | Outgoing RGB MLP | 0.3832 | −4.8% | 2.87 | 0.552 | 13.74 px |
| Oblique | Prefiltered incident MLP | 0.3542 | −10.0% | 3.02 | 0.505 | 13.00 px |
| Grazing | Traced reference | — | — | 2.13 | 0.778 | 19.14 px |
| Grazing | Integrated split sum | 0.4685 | −33.5% | 1.39 | 0.492 | 19.74 px |
| Grazing | Outgoing RGB MLP | 0.1976 | +1.1% | 2.02 | 0.793 | 19.49 px |
| Grazing | Prefiltered incident MLP | 0.4814 | −33.4% | 1.12 | 0.495 | 19.75 px |

The MLP rows are arithmetic means across seeds 1 and 17. Relative RGB RMSE is the square root of summed squared RGB error divided by summed squared reference RGB. Blue excess is max(0, B − 0.55(R + G)) in linear HDR. Its peak and mean report highlight intensity and energy, and its weighted horizontal standard deviation reports spread across the image. The individually scored roughness strips show that the oblique improvement held at 0.18, 0.30, and 0.45. At 0.62, oblique error rose from 0.217 to 0.262. Grazing error rose in all four strips. [Side-by-side HDR comparison](incoming/20261119T101643Z-nrc-prefiltered-specular/report.json) · [Two-seed measurements](incoming/20261119T101410Z-nrc-prefiltered-specular/run.json)

The split sum learns a normalized GGX convolution about the reflected view direction. The prefilter assumes its normal and view both equal that direction. The explicit normalization keeps the training target unbiased for the chosen filter at finite sample counts. Reconstruction multiplies the predicted RGB by a bilinearly sampled 48 × 48 table of exact-Fresnel, correlated-Smith directional albedo for the fixture conductor. The table occupies 36 KiB. The white furnace agreed with traced directional albedo within 0.000892 per channel across 16 view/roughness combinations. The 8,192-spp reference and LUT repeat reduced that bound to 0.000335 and changed the integrated grazing error from 0.4685 to 0.4683. The remaining grazing error comes from applying the reflected-axis approximation to spatially varying illumination and visibility. [Klepikov and Frolov, 2026](https://arxiv.org/abs/2610.11702) · [Karis, 2013](https://de45xmedrsdbp.cloudfront.net/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf)

Provisional GPU timestamps on the shared desktop put the median update at 0.0942 ms for outgoing RGB and 0.0856 ms for prefiltering. The timed update includes target sampling and the optimizer. Single frozen inference times varied from 0.191 to 0.249 ms for outgoing RGB and from 0.177 to 0.327 ms for prefiltering, with the LUT multiplication included in the latter. These measurements cover the fixture kernels. They exclude construction of the reference, LUT initialization, and the separate traced fallback.

## Reproduction

```sh
node bench/runner/run.ts --page nrc-prefiltered-specular --gpu apple --no-csv \
  --ab seed=1,17 --args width=72 height=48 steps=512 batch=4096 trainSpp=32 refSpp=4096
node bench/runner/run.ts --page nrc-prefiltered-specular --gpu apple --no-csv \
  --args width=72 height=48 steps=512 batch=4096 trainSpp=32 refSpp=8192 lutSpp=8192 seed=1 images=0
node bench/runner/run.ts --page nrc-prefiltered-specular --gpu apple --no-csv \
  --ab arm=raw,prefiltered --args seed=1
```

The first command writes four vertically stacked HDR images per seed. Rows run from left, center, and right oblique views to left, center, and right grazing views. Each view shows four roughness strips over the traced mirror and glass strips. The 8,192-spp confirmation was run `20261119T101443Z-nrc-prefiltered-specular`.
