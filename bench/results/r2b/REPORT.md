# R2b cutout shadow A/B (provisional timings: shared Mac, see load1)

Quality: against our exact 16384 spp render of the same scene (the sampler is deterministic: every mode uses the same random numbers). Timing: ms/spp at 1024 spp from ONE page that holds every variant (rounds alternate order, first round untimed); `ratio of mins` = best round / best round of the same scheduler's exact policy. NOT a quiet-machine result: see docs/QUIET-PASS.md R2b row.


## final (ab4: 10 renderers, 7 rounds)


### foliage-mask (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask | pt-megakernel | -0.00% | 6.89e-5 | - | 1.4694..4.7460 | - | - | - | - | 4.259765625 |
| foliage-mask | pt-mk-shadow | -12.67% | 5.95e-2 | - | 1.1223..4.2137 | 0.607..1.091 (med 0.844) | 0.764 | - | 659.32 | 4.259765625 |
| foliage-mask | pt-mk-transmit@0.455 | -0.00% | 1.03e-1 | - | 1.1566..4.5781 | 0.351..1.395 (med 0.896) | 0.787 | 9.90% | 1171.89 | 4.259765625 |
| foliage-mask | pt-mk-transmit@0.45 | -0.14% | 1.01e-1 | - | 1.1907..4.6144 | 0.349..1.380 (med 0.910) | 0.810 | 19.72% | 1185.55 | 4.259765625 |
| foliage-mask | pt-mk-roulette10 | 0.01% | 1.54e-3 | - | 1.2317..4.6530 | 0.361..1.249 (med 0.924) | 0.838 | 31.53% | 18.69 | 4.259765625 |
| foliage-mask | pt-mk-roulette | -0.00% | 5.51e-4 | - | 1.2900..4.8560 | 0.398..1.023 (med 0.998) | 0.878 | 48.33% | 7.01 | 4.259765625 |
| foliage-mask | pt-mk-coverage | 1.26% | 1.20e-1 | - | 1.1176..4.5463 | 0.328..0.958 (med 0.879) | 0.761 | -1.35% | 1324.31 | 4.259765625 |
| foliage-mask | pt-mk-stochastic | -0.42% | 9.18e-2 | - | 1.2774..4.7189 | 0.375..0.994 (med 0.974) | 0.869 | 44.70% | 1157.69 | 4.259765625 |
| foliage-mask | pt-mk-firstk | -1.73% | 7.82e-3 | - | 1.3231..6.2858 | 0.404..1.324 (med 1.021) | 0.900 | 57.86% | 102.19 | 4.259765625 |
| foliage-mask | pt-mk-classified | -0.00% | 6.89e-5 | - | 1.4437..4.8929 | 0.746..1.057 (med 0.930) | 0.982 | 92.57% | 0.98 | 4.259765625 |

### foliage-mask (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask | pt-wavefront | -0.00% | 6.89e-5 | - | 1.1339..10.1268 | - | - | - | - | 2.92822265625 |
| foliage-mask | pt-wf-shadow | -12.67% | 5.95e-2 | - | 1.0040..8.1434 | 0.804..1.125 (med 0.884) | 0.885 | - | 764.39 | 2.92822265625 |
| foliage-mask | pt-wf-transmit@0.455 | -0.00% | 1.03e-1 | - | 1.0450..6.9685 | 0.131..1.176 (med 0.908) | 0.922 | 31.58% | 1372.12 | 2.92822265625 |
| foliage-mask | pt-wf-transmit@0.45 | -0.14% | 1.01e-1 | - | 1.0465..7.6958 | 0.131..1.639 (med 0.906) | 0.923 | 32.71% | 1350.28 | 2.92822265625 |
| foliage-mask | pt-wf-roulette10 | -0.01% | 1.56e-3 | - | 1.1083..7.4931 | 0.139..2.314 (med 0.965) | 0.977 | 80.30% | 22.10 | 2.92822265625 |
| foliage-mask | pt-wf-roulette | -0.01% | 5.60e-4 | - | 1.1419..7.5342 | 0.143..2.423 (med 1.005) | 1.007 | 106.17% | 8.18 | 2.92822265625 |
| foliage-mask | pt-wf-coverage | 1.26% | 1.20e-1 | - | 1.0137..7.0536 | 0.127..2.064 (med 0.893) | 0.894 | 7.44% | 1556.64 | 2.92822265625 |
| foliage-mask | pt-wf-stochastic | -0.41% | 9.18e-2 | - | 1.1187..6.6701 | 0.140..2.303 (med 0.980) | 0.987 | 88.27% | 1314.28 | 2.92822265625 |
| foliage-mask | pt-wf-firstk | -1.73% | 7.82e-3 | - | 1.1163..6.4874 | 0.197..2.366 (med 0.973) | 0.984 | 86.47% | 111.73 | 2.92822265625 |
| foliage-mask | pt-wf-classified | -0.00% | 6.89e-5 | - | 1.1526..6.8641 | 0.324..2.530 (med 1.006) | 1.017 | 114.44% | 1.02 | 2.92822265625 |

### tex-mask (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tex-mask | pt-megakernel | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.3187..0.3753 | - | - | - | - | 3.09326171875 |
| tex-mask | pt-mk-shadow | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 0.2786..0.3310 | 0.831..0.919 (med 0.865) | 0.874 | - | 1242.59 | 3.09326171875 |
| tex-mask | pt-mk-transmit@0.459 | -0.00% | 5.79e-2 | 5.90e-2 / 0.16% | 0.3254..0.3956 | 0.981..1.087 (med 1.027) | 1.021 | 116.83% | 2795.25 | 3.09326171875 |
| tex-mask | pt-mk-transmit@0.45 | -0.11% | 5.60e-2 | 5.71e-2 / 0.05% | 0.3299..0.4034 | 0.979..1.111 (med 1.035) | 1.035 | 128.05% | 2744.20 | 3.09326171875 |
| tex-mask | pt-mk-roulette10 | 0.01% | 6.22e-4 | 7.58e-4 / 0.17% | 0.3140..0.3841 | 0.907..1.074 (med 0.985) | 0.985 | 88.29% | 28.98 | 3.09326171875 |
| tex-mask | pt-mk-roulette | 0.00% | 2.16e-4 | 3.42e-4 / 0.16% | 0.3322..0.4413 | 0.952..1.188 (med 1.043) | 1.043 | 133.90% | 10.63 | 3.09326171875 |
| tex-mask | pt-mk-coverage | 0.48% | 6.69e-2 | 6.83e-2 / 0.63% | 0.2886..0.3772 | 0.814..1.069 (med 0.906) | 0.906 | 24.88% | 2864.09 | 3.09326171875 |
| tex-mask | pt-mk-stochastic | 0.47% | 6.70e-2 | 6.84e-2 / 0.63% | 0.3071..0.4136 | 0.866..1.172 (med 0.964) | 0.964 | 71.22% | 3056.35 | 3.09326171875 |
| tex-mask | pt-mk-firstk | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.3279..0.4432 | 0.914..1.256 (med 1.029) | 1.029 | 123.17% | 1.03 | 3.09326171875 |
| tex-mask | pt-mk-classified | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.3664..0.4854 | 1.018..1.376 (med 1.153) | 1.150 | 219.27% | 1.15 | 3.09326171875 |

### tex-mask (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tex-mask | pt-wavefront | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.4455..0.5239 | - | - | - | - | 3.0458984375 |
| tex-mask | pt-wf-shadow | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 0.4220..0.4915 | 0.911..0.952 (med 0.938) | 0.947 | - | 1346.07 | 3.0458984375 |
| tex-mask | pt-wf-transmit@0.459 | -0.00% | 5.79e-2 | 5.90e-2 / 0.16% | 0.4338..0.5160 | 0.951..1.022 (med 0.974) | 0.974 | 50.21% | 2665.33 | 3.0458984375 |
| tex-mask | pt-wf-transmit@0.45 | -0.11% | 5.60e-2 | 5.71e-2 / 0.05% | 0.4467..0.5199 | 0.935..1.030 (med 0.991) | 1.003 | 104.98% | 2657.78 | 3.0458984375 |
| tex-mask | pt-wf-roulette10 | 0.01% | 6.43e-4 | 7.87e-4 / 0.16% | 0.4554..0.5206 | 0.946..1.037 (med 1.005) | 1.022 | 141.91% | 31.06 | 3.0458984375 |
| tex-mask | pt-wf-roulette | 0.00% | 2.22e-4 | 3.45e-4 / 0.16% | 0.4578..0.5112 | 0.922..1.048 (med 1.013) | 1.028 | 152.28% | 10.80 | 3.0458984375 |
| tex-mask | pt-wf-coverage | 0.48% | 6.69e-2 | 6.83e-2 / 0.63% | 0.4354..0.4607 | 0.859..1.011 (med 0.961) | 0.977 | 57.26% | 3091.21 | 3.0458984375 |
| tex-mask | pt-wf-stochastic | 0.47% | 6.70e-2 | 6.84e-2 / 0.63% | 0.4301..0.4652 | 0.848..1.013 (med 0.991) | 0.965 | 34.44% | 3061.52 | 3.0458984375 |
| tex-mask | pt-wf-firstk | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.4397..0.4708 | 0.839..1.025 (med 1.012) | 0.987 | 75.52% | 0.99 | 3.0458984375 |
| tex-mask | pt-wf-classified | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.4423..0.4680 | 0.858..1.014 (med 1.004) | 0.993 | 86.31% | 0.99 | 3.0458984375 |

### foliage-mask-dense (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-dense | pt-megakernel | -0.00% | 7.85e-5 | - | 3.1954..3.6113 | - | - | - | - | 4.72216796875 |
| foliage-mask-dense | pt-mk-shadow | -8.84% | 4.07e-2 | - | 2.2275..2.6510 | 0.697..0.750 (med 0.721) | 0.697 | - | 361.04 | 4.72216796875 |
| foliage-mask-dense | pt-mk-transmit@0.323 | 0.01% | 8.03e-2 | - | 2.5124..2.8195 | 0.766..0.815 (med 0.800) | 0.786 | 29.43% | 803.59 | 4.72216796875 |
| foliage-mask-dense | pt-mk-transmit@0.45 | 3.49% | 1.43e-1 | - | 2.5746..2.8463 | 0.765..0.823 (med 0.803) | 0.806 | 35.86% | 1467.15 | 4.72216796875 |
| foliage-mask-dense | pt-mk-roulette10 | -0.01% | 1.18e-3 | - | 2.6975..3.0366 | 0.792..0.916 (med 0.860) | 0.844 | 48.55% | 12.72 | 4.72216796875 |
| foliage-mask-dense | pt-mk-roulette | -0.01% | 4.48e-4 | - | 2.9321..3.5528 | 0.853..1.027 (med 0.920) | 0.918 | 72.80% | 5.24 | 4.72216796875 |
| foliage-mask-dense | pt-mk-coverage | 4.91% | 1.76e-1 | - | 2.5516..3.0399 | 0.760..0.879 (med 0.805) | 0.799 | 33.48% | 1788.32 | 4.72216796875 |
| foliage-mask-dense | pt-mk-stochastic | -0.10% | 5.96e-2 | - | 3.0264..3.8602 | 0.886..1.120 (med 0.966) | 0.947 | 82.53% | 719.05 | 4.72216796875 |
| foliage-mask-dense | pt-mk-firstk | -2.57% | 1.15e-2 | - | 3.0239..3.2547 | 0.881..0.980 (med 0.950) | 0.946 | 82.28% | 138.04 | 4.72216796875 |
| foliage-mask-dense | pt-mk-classified | -0.00% | 7.85e-5 | - | 3.2039..3.6705 | 0.962..1.090 (med 1.008) | 1.003 | 100.88% | 1.00 | 4.72216796875 |

## first pass (ab2: 8 renderers, 5 rounds)


### foliage-mask (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask | pt-megakernel | -0.00% | 6.89e-5 | - | 1.3794..1.4053 | - | - | - | - | 4.078125 |
| foliage-mask | pt-mk-shadow | -12.67% | 5.95e-2 | - | 1.0889..1.1283 | 0.788..0.804 (med 0.800) | 0.789 | - | 681.45 | 4.078125 |
| foliage-mask | pt-mk-roulette10 | 0.01% | 1.54e-3 | - | 1.2636..1.2977 | 0.907..0.925 (med 0.923) | 0.916 | 60.13% | 20.42 | 4.078125 |
| foliage-mask | pt-mk-roulette | -0.00% | 5.51e-4 | - | 1.3785..1.4231 | 0.999..1.028 (med 1.010) | 0.999 | 99.70% | 7.98 | 4.078125 |
| foliage-mask | pt-mk-coverage | 1.26% | 1.20e-1 | - | 1.1540..1.2018 | 0.829..0.865 (med 0.840) | 0.837 | 22.42% | 1456.73 | 4.078125 |
| foliage-mask | pt-mk-stochastic | -0.42% | 9.18e-2 | - | 1.3279..1.4168 | 0.963..1.023 (med 0.995) | 0.963 | 82.29% | 1282.00 | 4.078125 |
| foliage-mask | pt-mk-firstk | -1.73% | 7.82e-3 | - | 1.4351..1.5580 | 1.023..1.125 (med 1.062) | 1.040 | 119.16% | 118.07 | 4.078125 |
| foliage-mask | pt-mk-classified | -0.00% | 6.89e-5 | - | 1.4648..1.5298 | 1.044..1.105 (med 1.081) | 1.062 | 129.41% | 1.06 | 4.078125 |

### tex-mask (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tex-mask | pt-megakernel | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 1.2303..2.5959 | - | - | - | - | 5.24169921875 |
| tex-mask | pt-mk-shadow | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 2.1742..2.5296 | 0.863..2.056 (med 0.976) | 1.767 | - | 2511.57 | 5.24169921875 |
| tex-mask | pt-mk-roulette10 | 0.01% | 6.22e-4 | 7.58e-4 / 0.17% | 1.0201..2.7034 | 0.505..1.645 (med 1.004) | 0.829 | 122.26% | 24.39 | 5.24169921875 |
| tex-mask | pt-mk-roulette | 0.00% | 2.16e-4 | 3.42e-4 / 0.16% | 1.7286..2.6179 | 0.913..1.405 (med 1.071) | 1.405 | 47.21% | 14.32 | 5.24169921875 |
| tex-mask | pt-mk-coverage | 0.48% | 6.69e-2 | 6.83e-2 / 0.63% | 2.2258..2.4420 | 0.886..1.967 (med 0.999) | 1.809 | -5.46% | 5721.72 | 5.24169921875 |
| tex-mask | pt-mk-stochastic | 0.47% | 6.70e-2 | 6.84e-2 / 0.63% | 2.1735..2.5571 | 0.967..1.767 (med 1.023) | 1.767 | 0.07% | 5602.28 | 5.24169921875 |
| tex-mask | pt-mk-firstk | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 2.4526..2.7720 | 0.977..2.044 (med 1.134) | 1.994 | -29.50% | 1.99 | 5.24169921875 |
| tex-mask | pt-mk-classified | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 2.4563..2.7035 | 0.978..2.165 (med 1.096) | 1.997 | -29.89% | 2.00 | 5.24169921875 |

### tex-mask (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tex-mask | pt-wavefront | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 4.2706..4.4041 | - | - | - | - | 4.10400390625 |
| tex-mask | pt-wf-shadow | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 3.8437..4.0024 | 0.882..0.932 (med 0.913) | 0.900 | - | 1279.07 | 4.10400390625 |
| tex-mask | pt-wf-roulette10 | 0.01% | 6.43e-4 | 7.87e-4 / 0.16% | 3.9695..4.2776 | 0.928..0.985 (med 0.968) | 0.930 | 29.48% | 28.25 | 4.10400390625 |
| tex-mask | pt-wf-roulette | 0.00% | 2.22e-4 | 3.45e-4 / 0.16% | 4.1662..4.5238 | 0.970..1.027 (med 0.980) | 0.976 | 75.55% | 10.25 | 4.10400390625 |
| tex-mask | pt-wf-coverage | 0.48% | 6.69e-2 | 6.83e-2 / 0.63% | 3.8756..4.0956 | 0.898..0.959 (med 0.910) | 0.908 | 7.48% | 2870.08 | 4.10400390625 |
| tex-mask | pt-wf-stochastic | 0.47% | 6.70e-2 | 6.84e-2 / 0.63% | 4.1052..4.3315 | 0.941..1.012 (med 0.960) | 0.961 | 61.25% | 3048.50 | 4.10400390625 |
| tex-mask | pt-wf-firstk | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 4.0584..4.3738 | 0.949..1.023 (med 0.974) | 0.950 | 50.30% | 0.95 | 4.10400390625 |
| tex-mask | pt-wf-classified | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 4.0727..4.5337 | 0.925..1.044 (med 0.979) | 0.954 | 53.64% | 0.95 | 4.10400390625 |

### foliage-mask-dense (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-dense | pt-megakernel | -0.00% | 7.85e-5 | - | 8.0720..8.4416 | - | - | - | - | 17.15478515625 |
| foliage-mask-dense | pt-mk-shadow | -8.84% | 4.07e-2 | - | 6.2376..7.0524 | 0.762..0.836 (med 0.773) | 0.773 | - | 400.21 | 17.15478515625 |
| foliage-mask-dense | pt-mk-roulette10 | -0.01% | 1.18e-3 | - | 6.8066..7.7548 | 0.843..0.919 (med 0.853) | 0.843 | 31.02% | 12.71 | 17.15478515625 |
| foliage-mask-dense | pt-mk-roulette | -0.01% | 4.48e-4 | - | 7.4987..8.4447 | 0.921..1.001 (med 0.945) | 0.929 | 68.75% | 5.30 | 17.15478515625 |
| foliage-mask-dense | pt-mk-coverage | 4.91% | 1.76e-1 | - | 6.5502..7.2218 | 0.794..0.856 (med 0.817) | 0.811 | 17.04% | 1817.36 | 17.15478515625 |
| foliage-mask-dense | pt-mk-stochastic | -0.10% | 5.96e-2 | - | 7.8707..8.4526 | 0.967..1.002 (med 0.990) | 0.975 | 89.03% | 740.29 | 17.15478515625 |
| foliage-mask-dense | pt-mk-firstk | -2.57% | 1.15e-2 | - | 7.7439..8.5324 | 0.951..1.012 (med 0.987) | 0.959 | 82.12% | 139.94 | 17.15478515625 |
| foliage-mask-dense | pt-mk-classified | -0.00% | 7.85e-5 | - | 8.2860..8.7380 | 1.010..1.036 (med 1.018) | 1.027 | 111.67% | 1.03 | 17.15478515625 |

### foliage-mask-dense (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-dense | pt-wavefront | -0.00% | 7.85e-5 | - | 16.4427..17.4854 | - | - | - | - | 4.62109375 |
| foliage-mask-dense | pt-wf-shadow | -8.84% | 4.07e-2 | - | 14.6680..15.3143 | 0.855..0.907 (med 0.884) | 0.892 | - | 462.01 | 4.62109375 |
| foliage-mask-dense | pt-wf-roulette10 | 0.00% | 1.17e-3 | - | 15.9341..16.6555 | 0.919..0.986 (med 0.944) | 0.969 | 71.34% | 14.47 | 4.62109375 |
| foliage-mask-dense | pt-wf-roulette | -0.01% | 4.46e-4 | - | 16.2401..17.1936 | 0.938..1.028 (med 0.983) | 0.988 | 88.59% | 5.61 | 4.62109375 |
| foliage-mask-dense | pt-wf-coverage | 4.91% | 1.76e-1 | - | 13.8563..15.5485 | 0.792..0.946 (med 0.887) | 0.843 | -45.73% | 1887.31 | 4.62109375 |
| foliage-mask-dense | pt-wf-stochastic | -0.10% | 5.97e-2 | - | 16.9245..17.1571 | 0.969..1.033 (med 0.999) | 1.029 | 127.15% | 782.24 | 4.62109375 |
| foliage-mask-dense | pt-wf-firstk | -2.57% | 1.15e-2 | - | 15.8562..16.1696 | 0.911..0.975 (med 0.934) | 0.964 | 66.96% | 140.66 | 4.62109375 |
| foliage-mask-dense | pt-wf-classified | -0.00% | 7.85e-5 | - | 16.5749..17.2903 | 0.948..1.029 (med 0.995) | 1.008 | 107.45% | 1.01 | 4.62109375 |

### foliage-mask-grid16 (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid16 | pt-megakernel | -0.00% | 8.79e-5 | - | 6.7753..7.0676 | - | - | - | - | 17.4873046875 |
| foliage-mask-grid16 | pt-mk-shadow | -12.36% | 5.85e-2 | - | 6.0021..6.2583 | 0.856..0.901 (med 0.886) | 0.886 | - | 589.17 | 17.4873046875 |
| foliage-mask-grid16 | pt-mk-roulette10 | -0.00% | 1.67e-3 | - | 6.3242..6.5676 | 0.910..0.946 (med 0.929) | 0.933 | 41.66% | 17.71 | 17.4873046875 |
| foliage-mask-grid16 | pt-mk-roulette | -0.00% | 6.18e-4 | - | 6.6298..6.9934 | 0.941..1.007 (med 0.958) | 0.979 | 81.18% | 6.87 | 17.4873046875 |
| foliage-mask-grid16 | pt-mk-coverage | 1.50% | 1.09e-1 | - | 6.3504..6.5142 | 0.905..0.937 (med 0.926) | 0.937 | 45.05% | 1158.70 | 17.4873046875 |
| foliage-mask-grid16 | pt-mk-stochastic | -0.19% | 8.25e-2 | - | 6.6011..6.8720 | 0.941..0.990 (med 0.954) | 0.974 | 77.47% | 913.86 | 17.4873046875 |
| foliage-mask-grid16 | pt-mk-firstk | -1.65% | 7.51e-3 | - | 4.3363..7.3747 | 0.640..1.062 (med 0.989) | 0.640 | -215.42% | 54.63 | 17.4873046875 |
| foliage-mask-grid16 | pt-mk-classified | -0.01% | 8.82e-5 | - | 1.3218..4.8667 | 0.195..0.691 (med 0.680) | 0.195 | -605.28% | 0.20 | 17.4873046875 |

### foliage-mask-grid16 (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid16 | pt-wavefront | -0.00% | 8.79e-5 | - | 2.4096..2.6014 | - | - | - | - | 3.466796875 |
| foliage-mask-grid16 | pt-wf-shadow | -12.36% | 5.85e-2 | - | 2.0891..2.1890 | 0.803..0.908 (med 0.867) | 0.867 | - | 576.61 | 3.466796875 |
| foliage-mask-grid16 | pt-wf-roulette10 | 0.01% | 1.67e-3 | - | 2.2672..2.3689 | 0.873..0.981 (med 0.942) | 0.941 | 55.58% | 17.90 | 3.466796875 |
| foliage-mask-grid16 | pt-wf-roulette | 0.00% | 6.13e-4 | - | 2.3880..2.6710 | 0.918..1.108 (med 0.992) | 0.991 | 93.27% | 6.90 | 3.466796875 |
| foliage-mask-grid16 | pt-wf-coverage | 1.50% | 1.09e-1 | - | 2.1489..2.3461 | 0.826..0.974 (med 0.893) | 0.892 | 18.68% | 1102.50 | 3.466796875 |
| foliage-mask-grid16 | pt-wf-stochastic | -0.19% | 8.25e-2 | - | 2.3760..2.4748 | 0.915..1.027 (med 0.986) | 0.986 | 89.52% | 925.50 | 3.466796875 |
| foliage-mask-grid16 | pt-wf-firstk | -1.65% | 7.51e-3 | - | 2.3355..2.5295 | 0.934..1.011 (med 0.972) | 0.969 | 76.90% | 82.74 | 3.466796875 |
| foliage-mask-grid16 | pt-wf-classified | -0.01% | 8.82e-5 | - | 1.2397..1.3393 | 0.478..0.556 (med 0.514) | 0.515 | -264.99% | 0.52 | 3.466796875 |

### foliage-mask-grid12 (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid12 | pt-megakernel | -0.00% | 8.79e-5 | - | 2.5249..2.6221 | - | - | - | - | 4.11669921875 |
| foliage-mask-grid12 | pt-mk-shadow | -12.36% | 5.85e-2 | - | 2.0045..2.1072 | 0.766..0.834 (med 0.796) | 0.794 | - | 528.04 | 4.11669921875 |
| foliage-mask-grid12 | pt-mk-roulette10 | -0.00% | 1.67e-3 | - | 2.3525..2.3572 | 0.897..0.932 (med 0.932) | 0.932 | 66.88% | 17.68 | 4.11669921875 |
| foliage-mask-grid12 | pt-mk-roulette | -0.00% | 6.18e-4 | - | 2.4594..2.6481 | 0.940..1.049 (med 0.973) | 0.974 | 87.41% | 6.84 | 4.11669921875 |
| foliage-mask-grid12 | pt-mk-coverage | 1.50% | 1.09e-1 | - | 2.1519..2.2501 | 0.821..0.890 (med 0.852) | 0.852 | 28.32% | 1053.67 | 4.11669921875 |
| foliage-mask-grid12 | pt-mk-stochastic | -0.19% | 8.25e-2 | - | 2.4531..2.5524 | 0.936..1.010 (med 0.972) | 0.972 | 86.21% | 911.39 | 4.11669921875 |
| foliage-mask-grid12 | pt-mk-firstk | -1.65% | 7.51e-3 | - | 2.5622..2.6546 | 0.978..1.050 (med 1.015) | 1.015 | 107.17% | 86.63 | 4.11669921875 |
| foliage-mask-grid12 | pt-mk-classified | -0.00% | 8.80e-5 | - | 2.9122..3.1058 | 1.111..1.230 (med 1.183) | 1.153 | 174.42% | 1.15 | 4.11669921875 |

### foliage-mask-grid12 (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid12 | pt-wavefront | -0.00% | 8.79e-5 | - | 2.2958..2.4918 | - | - | - | - | 3.80419921875 |
| foliage-mask-grid12 | pt-wf-shadow | -12.36% | 5.85e-2 | - | 1.9915..2.0917 | 0.800..0.909 (med 0.867) | 0.867 | - | 576.97 | 3.80419921875 |
| foliage-mask-grid12 | pt-wf-roulette10 | 0.04% | 1.70e-3 | - | 2.1651..2.1670 | 0.869..0.944 (med 0.942) | 0.943 | 57.06% | 18.20 | 3.80419921875 |
| foliage-mask-grid12 | pt-wf-roulette | 0.02% | 6.18e-4 | - | 2.2805..2.3756 | 0.915..1.033 (med 0.993) | 0.993 | 94.96% | 6.98 | 3.80419921875 |
| foliage-mask-grid12 | pt-wf-coverage | 1.50% | 1.09e-1 | - | 2.0448..2.2029 | 0.822..0.960 (med 0.889) | 0.891 | 17.52% | 1101.18 | 3.80419921875 |
| foliage-mask-grid12 | pt-wf-stochastic | -0.19% | 8.26e-2 | - | 2.2605..2.2948 | 0.909..1.000 (med 0.984) | 0.985 | 88.41% | 924.43 | 3.80419921875 |
| foliage-mask-grid12 | pt-wf-firstk | -1.65% | 7.51e-3 | - | 2.2284..2.4255 | 0.894..1.053 (med 1.011) | 0.971 | 77.86% | 82.86 | 3.80419921875 |
| foliage-mask-grid12 | pt-wf-classified | -0.00% | 8.80e-5 | - | 2.5244..2.6219 | 1.052..1.140 (med 1.097) | 1.100 | 175.13% | 1.10 | 3.80419921875 |

## transmit probes (ab3: 5 renderers, 5 rounds)


### foliage-mask (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask | pt-megakernel | -0.00% | 6.89e-5 | - | 1.2745..1.4669 | - | - | - | - | 4.31982421875 |
| foliage-mask | pt-mk-shadow | -12.67% | 5.95e-2 | - | 1.0126..1.1109 | 0.691..0.872 (med 0.739) | 0.794 | - | 685.87 | 4.31982421875 |
| foliage-mask | pt-mk-transmit0 | -12.67% | 5.95e-2 | - | 1.1451..1.2506 | 0.852..0.913 (med 0.908) | 0.898 | 50.60% | 775.63 | 4.31982421875 |
| foliage-mask | pt-mk-transmit1 | 15.17% | 4.70e-1 | - | 1.1436..1.3622 | 0.833..1.055 (med 0.929) | 0.897 | 50.00% | 6112.86 | 4.31982421875 |
| foliage-mask | pt-mk-transmit | 1.25% | 1.20e-1 | - | 1.1562..1.2570 | 0.795..0.986 (med 0.912) | 0.907 | 54.81% | 1578.00 | 4.31982421875 |

### foliage-mask (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask | pt-wavefront | -0.00% | 6.89e-5 | - | 1.1241..1.3207 | - | - | - | - | 3.96630859375 |
| foliage-mask | pt-wf-shadow | -12.67% | 5.95e-2 | - | 0.9955..1.0997 | 0.813..0.974 (med 0.829) | 0.886 | - | 764.51 | 3.96630859375 |
| foliage-mask | pt-wf-transmit0 | -12.67% | 5.95e-2 | - | 1.0359..1.2310 | 0.785..1.005 (med 0.922) | 0.922 | 31.44% | 795.55 | 3.96630859375 |
| foliage-mask | pt-wf-transmit1 | 15.17% | 4.70e-1 | - | 1.0316..1.2299 | 0.842..1.084 (med 1.005) | 0.918 | 28.09% | 6252.42 | 3.96630859375 |
| foliage-mask | pt-wf-transmit | 1.25% | 1.20e-1 | - | 1.0280..1.1283 | 0.779..0.998 (med 0.915) | 0.915 | 25.28% | 1590.84 | 3.96630859375 |

### tex-mask (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tex-mask | pt-megakernel | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.3188..0.4171 | - | - | - | - | 4.3525390625 |
| tex-mask | pt-mk-shadow | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 0.2744..0.2757 | 0.658..0.863 (med 0.858) | 0.861 | - | 1223.11 | 4.3525390625 |
| tex-mask | pt-mk-transmit0 | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 0.3247..0.3267 | 0.779..1.025 (med 1.015) | 1.018 | 113.19% | 1447.28 | 4.3525390625 |
| tex-mask | pt-mk-transmit1 | 6.30% | 2.62e-1 | 2.68e-1 / 6.46% | 0.3301..0.4284 | 0.793..1.336 (med 1.034) | 1.035 | 125.27% | 12845.32 | 4.3525390625 |
| tex-mask | pt-mk-transmit | 0.48% | 6.69e-2 | 6.83e-2 / 0.63% | 0.3236..0.5189 | 0.778..1.628 (med 1.014) | 1.015 | 110.77% | 3210.07 | 4.3525390625 |

### tex-mask (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| tex-mask | pt-wavefront | 0.00% | 2.11e-5 | 1.44e-4 / 0.16% | 0.4037..0.5031 | - | - | - | - | 4.9267578125 |
| tex-mask | pt-wf-shadow | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 0.3790..0.4762 | 0.755..1.179 (med 0.936) | 0.939 | - | 1334.17 | 4.9267578125 |
| tex-mask | pt-wf-transmit0 | -5.35% | 3.00e-2 | 2.95e-2 / -5.20% | 0.3933..0.4835 | 0.791..1.198 (med 0.977) | 0.974 | 57.71% | 1384.37 | 4.9267578125 |
| tex-mask | pt-wf-transmit1 | 6.30% | 2.62e-1 | 2.68e-1 / 6.46% | 0.3924..0.4925 | 0.789..1.216 (med 0.975) | 0.972 | 54.15% | 12060.10 | 4.9267578125 |
| tex-mask | pt-wf-transmit | 0.48% | 6.69e-2 | 6.83e-2 / 0.63% | 0.3891..0.3931 | 0.781..0.967 (med 0.962) | 0.964 | 40.71% | 3047.86 | 4.9267578125 |

### foliage-mask-dense (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-dense | pt-megakernel | -0.00% | 7.85e-5 | - | 3.1550..3.3543 | - | - | - | - | 4.6240234375 |
| foliage-mask-dense | pt-mk-shadow | -8.84% | 4.07e-2 | - | 2.2867..2.3040 | 0.682..0.725 (med 0.683) | 0.725 | - | 375.38 | 4.6240234375 |
| foliage-mask-dense | pt-mk-transmit0 | -8.84% | 4.07e-2 | - | 2.5086..2.6166 | 0.749..0.826 (med 0.777) | 0.795 | 25.55% | 411.80 | 4.6240234375 |
| foliage-mask-dense | pt-mk-transmit1 | 18.56% | 7.22e-1 | - | 2.4991..2.5951 | 0.746..0.792 (med 0.774) | 0.792 | 24.46% | 7282.77 | 4.6240234375 |
| foliage-mask-dense | pt-mk-transmit | 4.86% | 1.75e-1 | - | 2.4986..2.7152 | 0.746..0.838 (med 0.801) | 0.792 | 24.41% | 1764.85 | 4.6240234375 |

### foliage-mask-dense (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-dense | pt-wavefront | -0.00% | 7.85e-5 | - | 2.4948..2.6847 | - | - | - | - | 4.89453125 |
| foliage-mask-dense | pt-wf-shadow | -8.84% | 4.07e-2 | - | 2.0718..2.1714 | 0.772..0.870 (med 0.830) | 0.830 | - | 430.09 | 4.89453125 |
| foliage-mask-dense | pt-wf-transmit0 | -8.84% | 4.07e-2 | - | 2.1205..2.2246 | 0.790..0.892 (med 0.850) | 0.850 | 11.52% | 440.20 | 4.89453125 |
| foliage-mask-dense | pt-wf-transmit1 | 18.56% | 7.22e-1 | - | 2.1229..2.2259 | 0.791..0.892 (med 0.851) | 0.851 | 12.10% | 7823.60 | 4.89453125 |
| foliage-mask-dense | pt-wf-transmit | 4.86% | 1.75e-1 | - | 2.1208..2.2221 | 0.790..0.890 (med 0.854) | 0.850 | 11.59% | 1894.36 | 4.89453125 |

### foliage-mask-grid16 (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid16 | pt-megakernel | -0.00% | 8.79e-5 | - | 2.7207..2.8280 | - | - | - | - | 6.28515625 |
| foliage-mask-grid16 | pt-mk-shadow | -12.36% | 5.85e-2 | - | 2.0955..2.1899 | 0.741..0.802 (med 0.771) | 0.770 | - | 512.24 | 6.28515625 |
| foliage-mask-grid16 | pt-mk-transmit0 | -12.36% | 5.85e-2 | - | 2.2637..2.3598 | 0.829..0.839 (med 0.834) | 0.832 | 26.90% | 553.35 | 6.28515625 |
| foliage-mask-grid16 | pt-mk-transmit1 | 15.32% | 4.34e-1 | - | 2.2777..2.4681 | 0.805..0.902 (med 0.868) | 0.837 | 29.15% | 4127.86 | 6.28515625 |
| foliage-mask-grid16 | pt-mk-transmit | 1.48% | 1.09e-1 | - | 2.2517..2.4604 | 0.796..0.903 (med 0.826) | 0.828 | 24.98% | 1022.14 | 6.28515625 |

### foliage-mask-grid16 (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid16 | pt-wavefront | -0.00% | 8.79e-5 | - | 2.3998..2.5078 | - | - | - | - | 4.82177734375 |
| foliage-mask-grid16 | pt-wf-shadow | -12.36% | 5.85e-2 | - | 2.0802..2.1830 | 0.867..0.870 (med 0.868) | 0.867 | - | 576.49 | 4.82177734375 |
| foliage-mask-grid16 | pt-wf-transmit0 | -12.36% | 5.85e-2 | - | 2.1765..2.2766 | 0.906..0.948 (med 0.908) | 0.907 | 30.13% | 603.18 | 4.82177734375 |
| foliage-mask-grid16 | pt-wf-transmit1 | 15.32% | 4.34e-1 | - | 2.1802..2.1838 | 0.871..0.909 (med 0.908) | 0.908 | 31.29% | 4479.39 | 4.82177734375 |
| foliage-mask-grid16 | pt-wf-transmit | 1.48% | 1.09e-1 | - | 2.1639..2.2576 | 0.900..0.938 (med 0.902) | 0.902 | 26.18% | 1113.64 | 4.82177734375 |

### foliage-mask-grid12 (mk)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid12 | pt-megakernel | -0.00% | 8.79e-5 | - | 2.5172..2.7275 | - | - | - | - | 3.95947265625 |
| foliage-mask-grid12 | pt-mk-shadow | -12.36% | 5.85e-2 | - | 2.0005..2.0162 | 0.733..0.801 (med 0.798) | 0.795 | - | 528.60 | 3.95947265625 |
| foliage-mask-grid12 | pt-mk-transmit0 | -12.36% | 5.85e-2 | - | 2.1747..2.2753 | 0.834..0.899 (med 0.867) | 0.864 | 33.72% | 574.63 | 3.95947265625 |
| foliage-mask-grid12 | pt-mk-transmit1 | 15.32% | 4.34e-1 | - | 2.1576..2.2607 | 0.791..0.896 (med 0.859) | 0.857 | 30.41% | 4226.68 | 3.95947265625 |
| foliage-mask-grid12 | pt-mk-transmit | 1.48% | 1.09e-1 | - | 2.1534..2.3491 | 0.792..0.933 (med 0.858) | 0.855 | 29.60% | 1056.67 | 3.95947265625 |

### foliage-mask-grid12 (wf)

| scene | renderer | relBias | relMSE | vs ref2 (relMSE / relBias) | ms/spp min..max | ratio to exact (paired) | ratio of mins | saving lost vs solid | eqTime vs exact | load1 |
|---|---|---|---|---|---|---|---|---|---|---|
| foliage-mask-grid12 | pt-wavefront | -0.00% | 8.79e-5 | - | 2.2923..2.3934 | - | - | - | - | 3.33056640625 |
| foliage-mask-grid12 | pt-wf-shadow | -12.36% | 5.85e-2 | - | 1.9856..1.9901 | 0.831..0.868 (med 0.866) | 0.866 | - | 576.15 | 3.33056640625 |
| foliage-mask-grid12 | pt-wf-transmit0 | -12.36% | 5.85e-2 | - | 2.0370..2.2183 | 0.853..0.964 (med 0.890) | 0.889 | 16.75% | 591.06 | 3.33056640625 |
| foliage-mask-grid12 | pt-wf-transmit1 | 15.32% | 4.34e-1 | - | 2.0416..2.0603 | 0.853..0.895 (med 0.891) | 0.891 | 18.25% | 4391.81 | 3.33056640625 |
| foliage-mask-grid12 | pt-wf-transmit | 1.48% | 1.09e-1 | - | 2.0290..2.0357 | 0.849..0.887 (med 0.885) | 0.885 | 14.14% | 1093.31 | 3.33056640625 |
