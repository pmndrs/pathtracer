# Learned spatial features for the compact NRC decoder

Keep the four-band position encoder as the default. Learned trilinear features matched the raw-weight control's equal-time error on the card and raised whole-frame grazing error by 19% against both fixed controls. The direct RGB grid had the lowest equal-time error on the card and in the grazing glossy crop. The fixed compact decoder retained the lowest whole-frame grazing error.

## Setup

The opt-in encoder replaces the 12 active frequency inputs with two learned values at each of six grid resolutions (4, 8, 16, 32, 64, and 128 cells per scene-box axis). It interpolates each pair from the eight enclosing vertices. Each level allocates up to 4,096 vertices, with dense indexing where the full lattice fits and separate hash slots at finer levels. The other input lanes and the 64 → 32 → 32 → 32 → RGB decoder are identical between MLP arms. All MLPs use fp16 portable tiled inference, raw RGB targets, and reflected directions. The learned and fixed raw controls both use raw inference weights to isolate the encoding. A second fixed arm uses the selected phase 10 compact preset with warm-up EMA. The RGB grid uses the phase 10 spatial control with outgoing view directions and a 10 MiB payload. All arms query at the second hit, use the same inline training path selection, and disable late path-traced refinement.

The feature gradient pass reads the first-layer preactivation deltas and the matching weights before decoder Adam, multiplies by the transpose of the first layer, and scatters the resulting input gradient using the same trilinear weights as inference. Signed, fixed-point atomic sums are clipped to keep even a hot slot within the integer range. Sparse Adam updates use the accumulated gradients after the scatter dispatch. Loading decoder weights or resetting the cache initializes a fresh feature table and optimizer alongside the decoder. The portable training layout is required.

## Apple M5 Pro results

Chrome 157 headless, 640 × 360, three repetitions per arm and fixture. Each run used a 1,024 spp path-traced reference, a four-frame cold series, 64 warm frames, and a four-frame warm series. The frozen result renders 64 fresh samples with the model held fixed. The equal-time result starts from that same warmed model with a fresh accumulation and runs live training and rendering for 600 ms of summed submit-to-completion time. Budgets overshot by at most one frame and reached 600–605 ms. The runner accepted all 24 runs under its load and thermal checks.

| Fixture | Cache | Frozen relMSE | Frozen glossy relMSE | Equal-time relMSE | Equal-time glossy relMSE | Equal-time frames |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Blue card | Fixed compact MLP | 0.004031 | — | 0.002435 | — | 151 |
| Blue card | Fixed compact MLP with warm-up EMA | 0.004071 | — | 0.002495 | — | 142 |
| Blue card | Learned compact MLP | 0.004203 | — | 0.002440 | — | 145 |
| Blue card | Direct RGB grid | 0.004373 | — | 0.001990 | — | 229 |
| Grazing | Fixed compact MLP | 0.009673 | 0.005586 | 0.007639 | 0.003651 | 158 |
| Grazing | Fixed compact MLP with warm-up EMA | 0.009426 | 0.005761 | 0.007653 | 0.003674 | 158 |
| Grazing | Learned compact MLP | 0.011764 | 0.007526 | 0.009079 | 0.003876 | 152 |
| Grazing | Direct RGB grid | 0.011275 | 0.005512 | 0.008455 | 0.003072 | 261 |

On the card, the frozen learned and fixed reflection errors were 0.1147 and 0.1148 relative RMS. Equal-time wall striping was 0.00576 for learned, 0.00567 for fixed, and 0.00520 for direct RGB. The card equal-time relMSE ranges overlap for the two MLP arms (fixed 0.002386–0.002446, learned 0.002423–0.002532). On the grazing fixture the fixed MLP beat the learned MLP on whole-image equal-time error in all repeats.

| Fixture | Cache | Startup, median / first (ms) | Online frame (ms) | Train stage (ms) | Inference stage (ms) | Frozen frame (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Blue card | Fixed compact MLP | 12.1 / 33.8 | 3.9 | 1.0 | 1.2 | 2.5 |
| Blue card | Fixed compact MLP with warm-up EMA | 12.7 / 33.2 | 4.0 | 1.0 | 1.2 | 2.6 |
| Blue card | Learned compact MLP | 11.4 / 41.0 | 4.1 | 1.2 | 1.2 | 2.5 |
| Blue card | Direct RGB grid | 11.4 / 34.8 | 2.5 | 0.4 | 0.5 | 2.0 |
| Grazing | Fixed compact MLP | 11.5 / 45.6 | 3.8 | 1.0 | 1.4 | 2.4 |
| Grazing | Fixed compact MLP with warm-up EMA | 11.9 / 35.8 | 3.8 | 1.0 | 1.3 | 2.6 |
| Grazing | Learned compact MLP | 14.4 / 53.4 | 3.8 | 1.1 | 1.5 | 2.5 |
| Grazing | Direct RGB grid | 10.0 / 37.8 | 2.3 | 0.3 | 0.5 | 1.8 |

Startup spans cache construction, pipeline creation, and queued initialization. The first repetition starts with a new browser session for that arm and fixture. Later repetitions reuse that session. Stage work includes the scheduler and queue completion overhead. Inference includes render resolve, and the RGB training stage includes grid resolve.

## Interpolation, collisions, and traffic

The production WebGPU encoder reproduced two analytic linear fields in the dense level and constant features at the other five levels with a maximum absolute error of 8.35 × 10⁻⁷. Two probes straddling the same cell boundary differed by 0.0001993 for an expected 0.0002000. Both fixtures updated approximately 10,245 of the 17,238 allocated vertices during cold and warm training, as measured by the per-vertex optimizer counters.

A separate optimizer check sent 16,384 concurrent records through the production scatter and update kernels. Two of the 48 trilinear corners shared a hash slot. CPU addressing, the collision diagnostic, and GPU optimizer readback all identified 47 touched slots. The first and second known Adam updates agreed with the expected values within 4.01 × 10⁻⁷. An intervening batch with equal positive and negative gradients and a subsequent empty batch left the feature values bit-for-bit unchanged.

| Resolution | Card distinct corners / aliases | Grazing distinct corners / aliases |
| --- | ---: | ---: |
| 4 | 24 / 0 | 24 / 0 |
| 8 | 60 / 0 | 42 / 0 |
| 16 | 148 / 3 | 110 / 1 |
| 32 | 496 / 26 | 320 / 18 |
| 64 | 1,514 / 233 | 966 / 109 |
| 128 | 2,918 / 844 | 2,278 / 515 |

Collision counts are medians from the first 8,192 stored training records of the final warm frame. An alias is a distinct lattice corner beyond the first occupant of a table slot. The densest level aliased 29% of the distinct card corners and 23% of the grazing corners in that sample.

The fixed decoder allocated 18,016,672 GPU bytes including its training buffers. Learned features added 620,568 bytes across the FP32 table, gradient sums, Adam moments, and update counters. Inference reads the 137,904-byte feature table and 8,384-byte decoder weights. The direct RGB grid allocated 10,485,844 bytes including keys, values, accumulators, and age metadata. Renderer and packed record buffers are outside these totals.

Each network training step admits up to 16,384 records. The feature gradient emits up to 96 four-byte atomic operands per record, or 6 MiB per full step and 12 MiB per two-step frame, before hardware read-modify-write and cache effects. Zero-quantized operands are skipped. Each step also exchanges 137,904 bytes of gradient slots to reset the accumulator and updates optimizer state for vertices with a nonzero sum. Each interpolation reads up to 48 pairs of FP32 values per query (384 logical bytes). The RGB grid adds four four-byte operands at each of four pyramid levels per admitted record, or up to 64 bytes per record, before key probes and resolve. These are logical payload bounds, since actual memory-controller traffic is not exposed by the test.

## Reproduce

From the pathtracer root, run the same settings for `fixed`, `ema`, `learned`, and `rgb` on both scenes.

```sh
for scene in nrc-materials-card nrc-materials-grazing; do
  for arm in fixed ema learned rgb; do
    node bench/runner/run.ts --page nrc-study --gpu apple --no-csv --load-wait 0 \
      --timeout 120000 --repeat 3 --args "phase11=$arm" "scene=$scene" \
      res=640x360 frames=4 warm=64 refSpp=1024 phase11BudgetMs=600 \
      phase11FrozenSpp=64 flip=0 upload=0 --json "/tmp/nrc-phase11-$scene-$arm.json"
  done
done
```

The measured learned arm used `featureLog2=12 featureLr=0.01`. The run IDs in table order were `20261119T101214Z`, `101738Z`, `101231Z`, `101256Z`, `101313Z`, `101756Z`, `101338Z`, and `101402Z`, each with the `-nrc-study` suffix. A separate grazing capture saved the frozen, equal-time, and reference images for fixed and learned under `20261119T101803Z-nrc-study` and `20261119T101831Z-nrc-study`. The two reference files are byte-identical. The implementation is opt-in through [the NRC study](../pages/nrc-study.ts) and [the feature encoder](../nrc/spatial-features.ts).
