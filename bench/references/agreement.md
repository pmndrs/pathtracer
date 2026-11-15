## Image means and render times
| scene | renderer | spp | seconds | mean RGB |
|---|---|---|---|---|
| furnace-diffuse | mitsuba3-llvm_ad_rgb | 4096 | 35.59 | 0.50000 0.50000 0.50000 |
| furnace-diffuse | mitsuba3-scalar_spectral | 1024 | 15.77 | 0.50000 0.49994 0.49988 |
| furnace-diffuse | pbrt-v4-cpu | 4096 | 74.15 | 0.50000 0.50000 0.50000 |
| furnace-dielectric | mitsuba3-llvm_ad_rgb | 4096 | 20.13 | 0.99994 0.99994 0.99994 |
| furnace-dielectric | mitsuba3-scalar_spectral | 1024 | 15.54 | 0.99993 0.99982 0.99974 |
| furnace-dielectric | pbrt-v4-cpu | 4096 | 50.93 | 0.99989 0.99989 0.99989 |
| furnace-conductor | mitsuba3-llvm_ad_rgb | 4096 | 42.04 | 0.69040 0.69040 0.69040 |
| furnace-conductor | mitsuba3-scalar_spectral | 1024 | 20.33 | 0.69041 0.69031 0.69026 |
| furnace-conductor | pbrt-v4-cpu | 4096 | 54.39 | 0.69285 0.69286 0.69284 |
| cornell | mitsuba3-llvm_ad_rgb | 4096 | 532.9 | 0.21277 0.14881 0.061487 |
| cornell | mitsuba3-scalar_spectral | 512 | 115.67 | 0.21196 0.14795 0.060282 |
| cornell | pbrt-v4-cpu | 4096 | 1076.85 | 0.21194 0.14797 0.060254 |
| quad-light | mitsuba3-llvm_ad_rgb | 4096 | 24.53 | 0.20588 0.20588 0.20588 |
| quad-light | mitsuba3-scalar_spectral | 1024 | 8.46 | 0.20587 0.20585 0.20585 |
| quad-light | pbrt-v4-cpu | 4096 | 72.3 | 0.20588 0.20588 0.20588 |
| env-sphere | mitsuba3-llvm_ad_rgb | 4096 | 48.61 | 0.42964 0.43667 0.57432 |
| env-sphere | mitsuba3-scalar_spectral | 1024 | 19.07 | 0.41994 0.41629 0.55192 |
| env-sphere | pbrt-v4-cpu | 4096 | 265.57 | 0.43357 0.43644 0.57620 |
| sponza-white | mitsuba3-llvm_ad_rgb | 4096 | 1455.89 | 0.24646 0.24077 0.23987 |
| sponza-white | pbrt-v4-cpu | 4096 | 3430.59 | 0.24590 0.24024 0.23939 |

## Furnace / analytic checks (relative tolerance; |mean-expected|/expected must be < tol/10, every pixel < tol)
| scene | renderer | spp | expected | mean RGB | mean rel dev | max pixel rel dev | verdict |
|---|---|---|---|---|---|---|---|
| furnace-diffuse | mitsuba3-llvm_ad_rgb | 4096 | 0.5 | 0.50000 0.50000 0.50000 | 5.29e-7 | 0.000977 | PASS |
| furnace-diffuse | mitsuba3-scalar_spectral | 1024 | 0.5 | 0.50000 0.49994 0.49988 | 0.000234 | 0.0199 | FAIL |
| furnace-diffuse | pbrt-v4-cpu | 4096 | 0.5 | 0.50000 0.50000 0.50000 | 3.73e-8 | 0.000488 | PASS |
| furnace-dielectric | mitsuba3-llvm_ad_rgb | 4096 | 1 | 0.99994 0.99994 0.99994 | 0.0000561 | 0.00947 | PASS |
| furnace-dielectric | mitsuba3-scalar_spectral | 1024 | 1 | 0.99993 0.99982 0.99974 | 0.000258 | 0.0130 | PASS |
| furnace-dielectric | pbrt-v4-cpu | 4096 | 1 | 0.99989 0.99989 0.99989 | 0.000110 | 0.0146 | PASS |
| quad-light | mitsuba3-llvm_ad_rgb | 4096 | analytic radiance image (57662 plane px) | mean 0.23383 vs 0.23383 | mean rel dev -0.0000824 | max pixel rel dev 0.100 | PASS |
| quad-light | mitsuba3-scalar_spectral | 1024 | analytic radiance image (57662 plane px) | mean 0.23380 vs 0.23383 | mean rel dev -0.000209 | max pixel rel dev 0.104 | PASS |
| quad-light | pbrt-v4-cpu | 4096 | analytic radiance image (57662 plane px) | mean 0.23383 vs 0.23383 | mean rel dev -0.0000888 | max pixel rel dev 0.0991 | PASS |

## Agreement with pbrt-v4 (test = renderer in column 2, reference = pbrt-v4)
| scene | test renderer | spp | relMSE (eps 1e-2) | SMAPE | mean abs diff | rel bias (mean t / mean r - 1) | bias RGB |
|---|---|---|---|---|---|---|---|
| furnace-diffuse | mitsuba3-llvm_ad_rgb | 4096 (ref 4096) | 2.44e-8 | 0.0000618 | 0.0000624 | -5.04e-7 | -2.5e-7 -2.6e-7 -2.5e-7 |
| furnace-diffuse | mitsuba3-scalar_spectral | 1024 (ref 4096) | 0.00000735 | 0.000814 | 0.000822 | -0.000120 | -0.0000012 -0.000062 -0.00012 |
| furnace-dielectric | mitsuba3-llvm_ad_rgb | 4096 (ref 4096) | 1.53e-7 | 0.0000303 | 0.0000607 | 0.0000535 | 0.000054 0.000053 0.000054 |
| furnace-dielectric | mitsuba3-scalar_spectral | 1024 (ref 4096) | 2.06e-7 | 0.0000907 | 0.000182 | -0.0000598 | 0.000041 -0.000072 -0.00015 |
| furnace-conductor | mitsuba3-llvm_ad_rgb | 4096 (ref 4096) | 0.0000315 | 0.00209 | 0.00291 | -0.00354 | -0.0025 -0.0025 -0.0024 |
| furnace-conductor | mitsuba3-scalar_spectral | 1024 (ref 4096) | 0.0000691 | 0.00298 | 0.00414 | -0.00364 | -0.0024 -0.0025 -0.0026 |
| cornell | mitsuba3-llvm_ad_rgb | 4096 (ref 4096) | 0.000439 | 0.0251 | 0.00190 | 0.00692 | 0.00083 0.00084 0.0012 |
| cornell | mitsuba3-scalar_spectral | 512 (ref 4096) | 0.000143 | 0.0100 | 0.00118 | 0.0000708 | 0.000025 -0.000023 0.000028 |
| quad-light | mitsuba3-llvm_ad_rgb | 4096 (ref 4096) | 0.00000184 | 0.000539 | 0.000256 | 0.00000778 | 0.0000025 0.0000020 2.9e-7 |
| quad-light | mitsuba3-scalar_spectral | 1024 (ref 4096) | 0.0000188 | 0.00156 | 0.000737 | -0.0000834 | -0.0000014 -0.000024 -0.000027 |
| env-sphere | mitsuba3-llvm_ad_rgb | 4096 (ref 4096) | 0.000652 | 0.00587 | 0.00500 | -0.00386 | -0.0039 0.00023 -0.0019 |
| env-sphere | mitsuba3-scalar_spectral | 1024 (ref 4096) | 0.0204 | 0.0363 | 0.0223 | -0.0401 | -0.014 -0.020 -0.024 |
| sponza-white | mitsuba3-llvm_ad_rgb | 4096 (ref 4096) | 0.00409 | 0.0411 | 0.00945 | 0.00216 | 0.00056 0.00053 0.00048 |
