# NRC equal-time study, Sponza (track N, M5 / TASKS N6c), 2026-11-15

**Question (Q2 in [nrc-plan.md](../../docs/design/nrc-plan.md)):** does the neural radiance cache make a better image than
plain path tracing in the same time?

**Answer on the M5 Pro: yes, once the cache has warmed up, and it warms up within a few frames.** In the windowed Chrome a
cache frame costs less than one plain sample (0.61× at 1280×720), and with a warm cache its error (relMSE against the Mitsuba
reference) is **3.2–4.3× lower** at the same frame count, **5–7× per unit of time**. In an upper-gallery corridor the cache was
never trained on, it is at its steady-state error from the first frame after a cut. (The first, headless pass below measured
3.0–3.8× at 1280×720 and 1.3–1.8× at 640×360, where plain got 2 samples per frame.) Its bias is under 0.4% after 4 frames at 1280×720 (−2.8% on the very
first frame), and FLIP is lower too. Frame-to-frame flicker on a still camera drops by about a third. From a cold start the
cache is biased for the first frames while it learns (+64% on frame 1, +12% after 16 frames, +1.3% after 256).

## Windowed Chrome (the reference setting from 2026-11-15 on)

The maintainer's call: time interactive work in a windowed Chrome started with `--enable-unsafe-webgpu` (it matches real use
better than the runner's headless Chrome; the difference is backlogged as TASKS N6e). Same page, driven over CDP
(`nrc-study.html?scenario=all&res=…&steps=…`). Load about 2, no other GPU job. These numbers are after the training-pixel fix
(below); they supersede an earlier windowed pass whose cache images still had noisy training tiles.

| | 1280×720, 4 steps | 640×360, 2 steps |
|---|---|---|
| Cache frame (render + training + inference) | **32.1 ms** | **9.2 ms** |
| Plain path tracing, 1 sample per pixel | 52.8 ms | 13.8 ms |
| Cache frame / plain sample | 0.61 | 0.67 |

A cache frame costs less than one plain sample, so plain gets 1 sample per frame and still uses more time.

**Warm cache, error after N frames (relMSE vs Mitsuba; plain 1 spp per frame):**

| Frames | 720p plain | 720p cache | Ratio | 720p cache bias | 360p plain | 360p cache | Ratio | 360p cache bias |
|---|---|---|---|---|---|---|---|---|
| 1 | 8.97 | 2.36 | 3.8× | −2.5% | 8.58 | 2.32 | 3.7× | −1.7% |
| 4 | 2.14 | 0.530 | 4.0× | +0.9% | 2.16 | 0.538 | 4.0× | +2.7% |
| 16 | 0.504 | 0.117 | 4.3× | +0.5% | 0.505 | 0.122 | 4.1× | +2.6% |
| 64 | 0.125 | 0.0297 | 4.2× | +0.3% | 0.122 | 0.0328 | 3.7× | +1.6% |
| 256 | 0.0319 | 0.00983 | 3.2× | +0.4% | 0.0306 | 0.0113 | 2.7× | +1.0% |

Per unit of time (plain error scaled by the time ratio, relMSE ∝ 1/time) the cache's lead is **5.3-7.1× at 1280×720** and
4.1-6.2× at 640×360. LDR-FLIP (720p): 0.34 vs 0.49 at 1 frame, 0.137 vs 0.216 at 16, 0.070 vs 0.089 at 256. Flicker on a
still camera: 1.00 vs 1.60 (−38%, 720p), 1.06 vs 1.61 (360p).

**Adaptation at the reference view** (real-time mode: every frame cleared, a new seed, training on; single-frame relMSE):

| Frames after arriving | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| 720p fresh network | 3.33 | 4.47 | 2.49 | 2.24 | 2.27 | 2.37 |
| 720p cut from the opposite end of the atrium | 2.45 | 2.39 | 2.46 | 2.39 | 2.37 | 2.44 |
| 720p 60-frame flight from the opposite end | 2.36 | 2.38 | 2.43 | 2.41 | 2.40 | 2.43 |

Steady state (cache warmed at the view): 2.43; plain path tracing: 9.01. After a cut or a flight the cache is at its steady
error on the first frame; a fresh network settles in 4-8 frames (0.1-0.25 s at 720p) and overshoots on frames 1-2.

## A region the cache never saw: the upper gallery

View: inside the upper gallery's north corridor (camera (−8, 4.95, −4.5) looking along +x): its vault and back wall are not
visible from any ground-floor view, and it is lit almost only by bounced light, so it is a harder region (plain path tracing's
single-frame error there is 14.6 against 9.0 on the ground floor). There is no Mitsuba render of this view: the reference is
our own path tracer at 4096 spp (it matches Mitsuba on Sponza, P2.9; its own noise is about 1/4096 of a single frame's).

| Frames after arriving (720p, single-frame relMSE) | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| Fresh network | 16.6 (+174% bias) | 6.50 (+49%) | 5.21 | 5.17 | 5.37 | 5.55 |
| Cut from a cache trained only on the ground floor | **5.66 (+6.7%)** | 5.67 (−3.7%) | 5.64 | 5.61 | 5.74 | 5.55 |

Steady state (cache warmed in the corridor): 5.70, bias +1.4%; plain path tracing: 14.6. 640×360 matches (cut: 5.56 at frame 1,
+4.9% bias; steady 5.39; plain 14.5).

**Reading:** the cache trained only on the ground floor predicts the corridor well enough that the first frame after the cut
is already at the steady-state error (2.6× below plain), with a few percent of bias that training removes within 2-4 frames.
Two reasons: the network is a smooth function of position, direction and normal, and the ground-floor training paths bounce
up into the gallery, so its light was never completely unseen. A space with no light path to the trained region (another
building, a closed room) would be a harder test. A **fresh** network is the weak spot: its first frame is worse than plain
path tracing (+174% bias in the corridor). Damp it: EMA weights (N-D7), a learning-rate warm-up, or show plain path tracing
until the loss settles.

## Fixed during this study: training tiles visible in the image

Training paths run past the cache vertex to depth M, so their pixels showed plain path tracing's noise: dark, speckled 8-pixel
columns every 128 pixels at 1280×720 (the tiles lined up in columns because 160 tiles per row is a multiple of 16). Their mean
was right (0.99 of the reference in those columns), so relMSE barely showed it, but it was plainly visible in real time. Now a
training path also adds a display query at the cache vertex, its tail query is training-only, and a fix-up pass removes what
the path gathered beyond the cache vertex (the per-vertex sums `ptOnRadiance` already keeps): every pixel shows the same
estimate. Tiles now rotate diagonally ((tx + 4·ty) mod 16). The query buffer had to grow to pixels + training paths (two
queries per training path overflowed it and dropped tails: the room gate fell to 5% error and, because `nrc-check` reported
before checking, still said PASS; it now fails on a miss).

## Headless runner (first pass)

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
