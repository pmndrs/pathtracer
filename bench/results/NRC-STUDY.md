# NRC equal-time study, Sponza (track N, M5 / TASKS N6c), 2026-11-15

**Question (Q2 in [nrc-plan.md](../../docs/design/nrc-plan.md)):** does the neural radiance cache make a better image than
plain path tracing in the same time?

**Answer on the M5 Pro: yes, once the cache has warmed up, and it warms up within a few frames.** In the windowed Chrome a
cache frame costs less than one plain sample (0.58× at 1280×720), and with a warm cache its error (relMSE against the Mitsuba
reference) is **3.1–3.8× lower** at the same frame count, **5–6× per unit of time**. (The first, headless pass below measured
3.0–3.8× at 1280×720 and 1.3–1.8× at 640×360, where plain got 2 samples per frame.) Its bias is under 0.4% after 4 frames at 1280×720 (−2.8% on the very
first frame), and FLIP is lower too. Frame-to-frame flicker on a still camera drops by about a third. From a cold start the
cache is biased for the first frames while it learns (+64% on frame 1, +12% after 16 frames, +1.3% after 256).

## Windowed Chrome (the reference setting from 2026-11-15 on)

The maintainer's call: time interactive work in a windowed Chrome started with `--enable-unsafe-webgpu` (it matches real use
better than the runner's headless Chrome; the difference is backlogged as TASKS N6e). Same page, driven over CDP
(`nrc-study.html?scenario=all&res=…&steps=…`). Load 2.1-2.4, no other GPU job.

| | 1280×720, 4 steps | 640×360, 2 steps |
|---|---|---|
| Cache frame (render + training + inference) | **34.6 ms** | **8.9 ms** |
| Plain path tracing, 1 sample per pixel | 59.8 ms | 14.0 ms |
| Cache frame / plain sample | 0.58 | 0.64 |

So in the window the cache frame costs less than one plain sample: plain gets 1 sample per frame and still uses more time.

**Warm cache, error after N frames (relMSE vs Mitsuba; plain 1 spp per frame):**

| Frames | 720p plain | 720p cache | Ratio | 720p cache bias | 360p plain | 360p cache | Ratio | 360p cache bias |
|---|---|---|---|---|---|---|---|---|
| 1 | 8.97 | 2.65 | 3.4× | −2.0% | 8.58 | 2.77 | 3.1× | +7.5% |
| 4 | 2.14 | 0.599 | 3.6× | | 2.16 | 0.636 | 3.4× | |
| 16 | 0.504 | 0.133 | 3.8× | −0.2% | 0.505 | 0.137 | 3.7× | +1.7% |
| 64 | 0.125 | 0.0332 | 3.8× | | 0.122 | 0.0359 | 3.4× | |
| 256 | 0.0319 | 0.0104 | 3.1× | +0.4% | 0.0306 | 0.0119 | 2.6× | +0.5% |

Per unit of time (plain error scaled by the time ratio, relMSE ∝ 1/time) the cache's lead is 5.3-6.5× at 1280×720 and
4.0-5.8× at 640×360. LDR-FLIP (720p): 0.34 vs 0.49 at 1 frame, 0.14 vs 0.22 at 16, 0.070 vs 0.089 at 256. Flicker on a
still camera: 1.08 vs 1.60 (720p), 1.10 vs 1.61 (360p).

**Adaptation (real-time mode: every frame cleared, new seed, training on; single-frame relMSE at the reference view):**

| Frames after arriving | 1 | 2 | 4 | 8 | 16 | 64 |
|---|---|---|---|---|---|---|
| 720p cold (fresh network) | 3.52 | 4.81 (+65% bias) | 2.78 | 2.56 | 2.58 | 2.64 |
| 720p cut from the opposite view | 2.70 | 2.65 | 2.66 | 2.69 | 2.70 | 2.65 |
| 720p 60-frame flight from the opposite view | 2.80 | 2.72 | 2.67 | 2.75 | 2.69 | 2.78 |
| 360p cold | 3.64 | 4.54 (+58%) | 4.35 (+58%) | 2.68 | 2.38 | 2.60 |
| 360p cut | 2.68 | 2.72 | 2.71 | 2.72 | 2.62 | 2.63 |

Steady state (cache warmed at the view): 2.79 (720p), 2.74 (360p); plain: 9.01 / 8.67. A cache trained on the opposite end of
the atrium is already at its steady-state error on the first frame after a cut: the two views share the scene's surfaces and
light, and the network generalises across them. A fresh network needs about 8 frames (0.3 s at 720p, 0.07 s at 360p) and
overshoots on frames 2-4 (bias +58-65%) before it settles: worth damping (EMA weights, N-D7, or a smaller learning rate for the
first steps). Not measured: moving into a part of the scene the cache has never seen (the upper gallery, outdoors).

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
