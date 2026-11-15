# NRC equal-time study, Sponza (track N, M5 / TASKS N6c), 2026-11-15

**Question (Q2 in [nrc-plan.md](../../docs/design/nrc-plan.md)):** does the neural radiance cache make a better image than
plain path tracing in the same time?

**Answer on the M5 Pro: yes, once the cache has warmed up.** With a warm cache, the cache side's error (relMSE against the
Mitsuba reference) is **3.0–3.8× lower** than plain path tracing at 1280×720 and **1.3–1.8× lower** at 640×360, for the
same number of frames at about equal time per frame. Its bias is under 0.4% after 4 frames at 1280×720 (−2.8% on the very
first frame), and FLIP is lower too. Frame-to-frame flicker on a still camera drops by about a third. From a cold start the
cache is biased for the first frames while it learns (+64% on frame 1, +12% after 16 frames, +1.3% after 256).

## Setup

- Scene `sponza-white` with its own camera and settings (grey Lambert 0.5 everywhere, sun + sky, maxDepth 8, Russian roulette
  from depth 4, NEE). Both sides run the same megakernel; the cache side only adds the NRC hook.
- Cache: 64 → 5×64 → 3 network, subgroup-matrix inference (f16, 4 subgroups), portable f16 training with the tiled kernels;
  training paths inside the render pass as whole 8×8 tiles (every 16th, rotating), training depth 4, cache queried at the
  second hit, learning rate 0.002 (VkNRC's), batch 16384.
- **Equal time:** median wall clock of one cache frame (render with training paths, inference, resolve, training steps) and of
  one plain sample per pixel, each submitted alone; plain gets k = round(cache / plain) samples per frame.
- Reference: Mitsuba 3, 4096 spp (box-downsampled 2× for 640×360). Floors (pbrt-v4 vs Mitsuba, both 4096 spp): relMSE 4.0e-3 at
  1280×720 and 1.1e-3 at 640×360, LDR-FLIP 0.044 / 0.027. Every number below is above them.
- Cold: fresh network, both images start empty. Warm: 300 more training frames (no accumulation, a new seed each frame), then
  both images restart. Page: `node bench/runner/run.ts --page nrc-study --args res=1280x720 steps=4` (raw rows in
  `bench/results/nrc-study.csv`; the 16- and 256-frame images are uploaded as PFM next to the run for the viewer).
- M5 Pro, Chrome 157 headless (runner flags), load 2.6–2.8, no other GPU job. Quality numbers are deterministic for the seed;
  one run per configuration.

## Results

### 1280×720, 4 training steps per frame (k = 1: cache frame 106.8 ms, plain 88.4 ms per sample; the cache side gets 1.21× the time)

| Frames | Plain relMSE | Cache relMSE (warm) | Ratio | Cache relBias (warm) | Plain FLIP | Cache FLIP (warm) |
|---|---|---|---|---|---|---|
| 1 | 8.97 | 2.63 | 3.4× | −2.8% | 0.487 | 0.347 |
| 4 | 2.14 | 0.599 | 3.6× | −0.3% | 0.318 | 0.230 |
| 16 | 0.504 | 0.134 | 3.8× | +0.35% | 0.216 | 0.143 |
| 64 | 0.125 | 0.0334 | 3.7× | +0.14% | 0.143 | 0.094 |
| 256 | 0.0319 | 0.0105 | 3.0× | +0.20% | 0.089 | 0.070 |

Correcting for the 1.21× time (plain error × 1/1.21, relMSE ∝ 1/spp) the ratios are 2.5–3.1×.

Cold start, same resolution: relMSE 7.59 / 0.297 / 0.0234 at 1 / 16 / 256 frames (plain 8.97 / 0.504 / 0.0319), but biased
(+64% / +12% / +1.3%) and FLIP worse than plain (0.37 vs 0.22 at 16 frames): the early frames show a cache that is still learning.

### 640×360 (k = 2: cache frame 37.9 ms with 2 steps, 46.0 ms with 4; plain 21.8–24.2 ms per sample)

| Frames | Plain relMSE (2 spp/frame) | Cache relMSE, 2 steps | Ratio | relBias | Cache relMSE, 4 steps | relBias |
|---|---|---|---|---|---|---|
| 1 | 4.25 | 2.65 | 1.6× | +3.5% | 2.63 | +1.4% |
| 4 | 1.03 | 0.625 | 1.6× | +5.3% | 0.614 | +4.0% |
| 16 | 0.249 | 0.137 | 1.8× | +1.1% | 0.135 | +1.3% |
| 64 | 0.0609 | 0.0359 | 1.7× | +0.8% | 0.0332 | +0.5% |
| 256 | 0.0155 | 0.0121 | 1.3× | +0.5% | 0.0098 | +0.4% |

At this size the cache's fixed costs (training, inference setup) buy plain path tracing a second sample per frame, so the
advantage is smaller; at 256 frames the cache's bias starts to dominate its error (FLIP 0.080 vs 0.067 for plain).

### Stability (warm cache, still camera, single frames with a new seed each)

Mean relative frame-to-frame change: plain 1.60 (1280×720) / 1.48 (640×360), cache 1.08 / 1.08: about 33% / 27% less flicker.

## Caveats and what is not measured yet

- **Timing is wall clock per submit in the runner's headless Chrome**, and it does not match the windowed Chrome demo: there the
  cache side of a 640×360 frame took 9.0 ms and one plain sample 13 ms, so in the demo the cache costs *less* than one plain
  sample. The ratio decides k; with the demo's numbers, plain would get 1 sample at 640×360 and the cache's lead would be larger.
  Find out why headless is 2–4× slower (and slower for the cache than for plain) before quoting either as a final timing.
- One scene, one still camera, one seed. Not measured: a moving camera (how fast the cache adapts), glossy materials (the
  roughness input is fixed at 1, N-I10), other GPUs (the RTX 3060 is M6).
- A real-time renderer would denoise both sides; this compares raw images.
- **Fixed during the study:** the training tiles rotated with the sample index, which a cleared image restarts, so during
  warm-up the cache trained on the same 1/16 of the tiles and the warm 640×360 runs showed a +41–70% bias spike in the first
  frames. Tiles now rotate with a GPU frame counter (`nrcCount[1]`) and every cleared frame gets a new seed; the numbers above
  are after the fix. The live demo had the same bug while the camera moved.
