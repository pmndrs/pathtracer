# NRC equal-time study, Sponza (track N, M5 / TASKS N6c), 2026-11-15

**Question (Q2 in [nrc-plan.md](../../docs/internal/design/nrc-plan.md)):** does the neural radiance cache make a better image than
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

## RTX 3060 Laptop, Windows (M6 / WN5, 2026-11-18)

Same page and settings (`nrc-study.html?scenario=all&res=1280x720&steps=4`), windowed Chrome 157 on D3D12 over CDP
(`node bench/runner/run.ts --page nrc-study --gpu nvidia --headed`), per-record f16 network (`infer=portable`: Chrome on Windows has
no subgroup matrices, and per-record beats tiled on NVIDIA). **Reference: our own path tracer at 4096 spp** (`ref=ours`; the Mitsuba
PFMs are not on this machine), so the bias column is against our converged image, not Mitsuba's. Quality numbers are deterministic for
a seed; the run itself was thermally throttled, so its timing rows are not used (the frame times below come from the demo).

**Warm cache, error after N frames (720p, relMSE; plain 1 spp per frame):**

| Frames | Plain | Cache | Ratio (3060) | Ratio (M5 Pro) | Cache bias | FLIP plain / cache |
|---|---|---|---|---|---|---|
| 1 | 8.98 | 2.40 | 3.7× | 3.8× | +0.24% | 0.487 / 0.322 |
| 4 | 2.14 | 0.527 | 4.1× | 4.0× | +0.14% | 0.318 / 0.212 |
| 16 | 0.504 | 0.116 | 4.4× | 4.3× | +0.23% | 0.216 / 0.127 |
| 64 | 0.125 | 0.0282 | 4.4× | 4.2× | +0.20% | 0.143 / 0.080 |
| 256 | 0.0318 | 0.00869 | 3.7× | 3.2× | +0.35% | 0.088 / 0.060 |

Flicker on a still camera 1.00 vs 1.60 (−38%, as on the Mac). Adaptation (single frames): steady 2.42 vs plain 9.00; after a cut 2.38 and
after the flight 2.39 on the first frame; a fresh network 2.33 at frame 1 (−35% bias), 2.17 by frame 4. Unseen corridor: plain 14.6,
steady 5.73, first frame after the cut 5.61 (+1.5% bias), fresh 5.19 (−32%) then 5.10 by frame 4. **The quality answer is the Mac's.**

**Frame times (the demo's Benchmark, `node bench/live-bench.ts --repeat 2 --cool 58`):** windowed Chrome 157, wall clock per submit,
median of 20 frames, the demo's loop paused and the card cooled below 58 °C before each cell; each 60-frame warm-up brings it back to
80-87 °C at 1.7-1.85 GHz, so these are best effort. Two rounds in opposite order agree within 2%.

| ms | 960×540 tiled | 960×540 per-record | 1280×720 tiled | 1280×720 per-record |
|---|---|---|---|---|
| Training steps (4) | 5.0-5.2 | 5.1-5.2 | 5.0-5.1 | 5.3-5.8 |
| Render pass (cache side) | 22.3-22.4 | 22.2-22.6 | 38.5-39.0 | 38.5-38.7 |
| Inference + resolve | 15.7-16.0 | 10.2-10.7 | 27.2 | 17.8-18.0 |
| **Cache frame** | 43.0-43.6 | **37.5-38.5** | 70.7-71.3 | **62.0-62.1** |
| Plain, 1 sample | 32.2-32.6 | 33.2-33.3 | 58.0-59.1 | 59.9-60.2 |
| Cache frame / plain sample | 1.33 | **1.14** | 1.21 | **1.03** |

**Reading:** on the 3060 in Chrome a cache frame costs about one plain sample (1.03× at 720p, 1.14× at 540p), not 0.61× as on the
M5 Pro, so the lead per unit of time equals the lead per frame (3.7-4.4×) instead of 5-7×. The render pass is cheap (0.65 of a plain
sample: paths stop at the cache vertex); the network is what costs: inference 18 ms at 720p in Chrome's portable kernel. Our blocked
subgroup-matrix kernel does the same queries ~4.4× faster on Dawn/Vulkan (3.85 vs 17 ms per million, N2r-c), which would put the
cache frame near 0.8 of a plain sample (estimate: ~1.05 M queries per 720p frame), but Chrome on Windows does not offer subgroup
matrices (D3D12). First load: the cache kernels compile in 2.1-2.4 s on DXC.

## Hash grid vs network (M7 / Q4, RTX 3060 Laptop, 2026-11-18)

Does the network beat a plain hash-grid radiance cache in the same time? The hash arm is `NrcCache` with `model: 'hash'`
([nrc-hash-cache.md](../../docs/internal/design/nrc-hash-cache.md)): the same render paths, training records, targets and resolve as the network
arm, only the store differs (a multi-resolution world-space hash grid, 2^20 cells, 16 px cells, a 16- or 64-sample threshold, 3
fallback levels; tuned at 640×360 first, nrc-hash-cache.md §6). Quality: `nrc-study model=hash scenario=all res=1280x720 steps=4
ref=ours` (headless, deterministic for a seed; same seeds and references as the WN5 run above, so the network columns are copied from
it). Time: `node bench/live-bench.ts --res 960x540,1280x720 --backends portable-f16,hash --repeat 2 --cool 58`, windowed Chrome 157,
two rounds in opposite order (within 3%), card at 79-86 °C once warm: best effort.

**Frame time (windowed, median of 20, ms):**

| | 960×540 hash | 960×540 network | 1280×720 hash | 1280×720 network |
|---|---|---|---|---|
| Update + resolve (hash) / 4 training steps (network) | 2.9-3.0 | 5.0-5.5 | 3.0 | 5.7-5.9 |
| Render pass | 21.9-22.8 | 22.2-22.3 | 38.2-38.4 | 38.3-39.1 |
| Lookup (hash) / inference (network), + targets and resolve | 3.0-3.5 | 10.7-10.9 | 2.9-3.8 | 18.1-18.2 |
| **Cache frame** | **28.4-28.7** | 38.0-38.6 | **44.3-45.0** | 62.1-63.2 |
| Plain, 1 sample | 32.8-33.6 | 33.1-33.5 | 58.2-58.6 | 59.3-60.9 |
| Cache frame / plain sample | **0.85** | 1.15 | **0.76** | 1.04 |

**Quality at 720p (relMSE against our 4096 spp image; single frames unless noted):**

| | Plain | Network | Hash, 64 samples | Hash, 16 samples |
|---|---|---|---|---|
| Steady state (warm, median of 8 frames) | 9.00 | **2.42** | 2.48 | 2.53 |
| Warm, accumulated 256 frames | 0.0318 | 0.00869 | **0.0079** | 0.0080 |
| Bias after 256 frames | +0.01% | +0.35% | **+0.18%** | +0.31% |
| Flicker (still camera, relative frame-to-frame change) | 1.60 | **1.00** | 1.06 | 1.09 |
| First frame after a cut | — | **2.38** | 2.43 | 2.45 |
| First frame after a fly-through | — | **2.39** | 2.44 | 2.59 |
| Fresh cache, frame 4 | — | **2.17** | 2.45 (−6.5%) | 2.77 (−4.1%) |
| Unseen corridor, steady (plain 14.6) | 14.6 | **5.73** | 5.83 | 5.91 |
| Unseen corridor, first frame after the cut | — | **5.61** | 5.81 | 5.99 |
| Unseen corridor, fresh cache frame 4 | — | **5.10** | 5.68 | 6.12 |
| Queries that miss the grid, first frame after the cut | — | — | 6.4% | 1.8% |

**Per unit of time** (steady single frames, efficiency = relMSE × ms of plain over the same for the cache; higher is better): network
**3.5×** plain, hash **4.8×** plain, so the hash grid is **1.37× more efficient than the network** on this machine. Per frame the network
is ahead by 2-4% in steady state, after a cut and in flight, 11-13% on a fresh cache and in the unseen corridor (where its
generalisation shows), and behind on the accumulated image (+0.35% bias vs +0.18%).

**Reading (Q4):** on the RTX 3060 in Chrome, **no**: a plain hash grid fed by the same training paths is as good per frame within a few
percent and costs 0.76 of a plain sample against the network's 1.04, because Chrome on Windows runs the network in its portable kernel
(18 ms of inference at 720p, against 3 ms for the grid lookup). The network's edge is generalisation: a fresh cache and an unseen
region, where it is 11-13% better. Caveats: sponza-white is one white Lambert material, which a view-independent grid cell represents
exactly; the network's direction and albedo inputs would matter on glossy or textured scenes, which this study does not have. On the
M5 Pro (subgroup-matrix inference, cache frame 0.61 of a plain sample) the gap would be smaller; on Windows with Dawn/Vulkan matrices
(3.85 ms per million queries) the network frame would drop to ~0.8 of a plain sample, still above the grid.

## Glossy floor and columns: network vs hash grid (N7b, 2026-11-18)

Does the network's direction and roughness input pay off where the cached radiance depends on the view? `sponza-glossy`
(`bench/scenes/tier2/sponza-glossy.ts`): sponza-white with the floor of both storeys a polished grey metal (roughness 0.15) and the ground-floor
columns satin metal (0.3), everything else the grey Lambertian. `nrc-study scene=sponza-glossy scenario=all res=1280x720 steps=4 ref=ours`
(our own 4096 spp references), RTX 3060, headless; quality is deterministic for a seed. Runs `20261117T233402Z` (network),
`234918Z` (hash grid, defaults), `20261118T000335Z` (hash grid, 4 × 4 direction bins at glossy vertices, `hgDirBins=4`). Since this run the
network gets each vertex's real roughness (it read 1 everywhere before, N-I10; no change on sponza-white, which is all Lambert).

| (relMSE unless noted) | Plain | Network | Hash grid | Hash, direction bins |
|---|---|---|---|---|
| Warm, single frame | 32.0 | 1.72 | **1.70** | 1.74 |
| Warm, accumulated 16 frames | 18.1 (a firefly: 10.7 at 4 frames) | **0.092** | 0.106 | 0.106 |
| Warm, accumulated 256 frames | 0.826 | **0.013** | 0.026 | 0.023 |
| LDR-FLIP, single frame / 256 frames | 0.457 / 0.111 | **0.272 / 0.103** | 0.311 / 0.149 | 0.298 / 0.127 |
| Bias after 256 frames | +0.04% | +0.9% | +0.3% | −0.3% |
| Steady state (median of 8 single frames) | 50.3 | **1.64** | 1.67 | 1.71 |
| First frame after a cut / fly-through | — | **1.74 / 1.61** | 1.74 / 1.63 | 1.89 / 1.71 |
| Fresh cache, frame 4 | — | **1.40** | 1.96 | 2.10 |
| Unseen corridor: steady / first frame after the cut (plain 326) | 326 | **5.49 / 4.52** | 5.59 / 4.72 | 5.65 / 4.75 |
| Flicker (plain 1.61) | 1.61 | **0.90** | 0.94 | 0.97 |
| Cache frame / plain sample (headless, hot card) | 1 | 0.92 | **0.65** | 0.66 |

* Plain path tracing is far worse here than on sponza-white (single frame 32 vs 9, still 0.83 after 256 frames): the polished floor turns
  the sun and sky into caustic fireflies at deep vertices. Both caches end those paths at vertex 1, so they are 20-30× better per frame.
* **Per frame the two caches are still within a few percent** (steady 1.64 vs 1.67); per unit of time the grid stays ahead on single
  frames (0.65 vs 0.92 of a plain sample: about 1.4× the network's efficiency).
* **Where the image converges, the network wins clearly**: accumulated 256 frames it has half the grid's error (0.013 vs 0.026), and in
  FLIP the grid (0.149) is even worse than plain path tracing (0.111) while the network (0.103) is better. That error is a floor, not noise
  (more frames do not remove it): one value per cell averages the reflection seen from all directions and blurs it at cell size. The
  network's direction and roughness inputs represent it. The network also learns faster from a fresh start (frame 4: 1.40 vs 1.96).
* **Direction bins do not fix it** (4 × 4 octahedral bins of wo at glossy vertices): splitting a cell 16 ways leaves each bin with a sixteenth
  of the samples, so single frames get noisier (1.71 vs 1.67) and the converged floor barely moves (0.023 vs 0.026; FLIP 0.127 vs 0.149).
  Kept as a setting (`dirBins`, default off).

**Reading (Q4, refined):** on a view-independent (diffuse) scene the hash grid wins outright; once glossy surfaces carry the cached light
the network's inputs pay: same single-frame error, half the converged error and a visibly sharper image, at 1.4× the grid's frame cost in
Chrome on Windows. Which is better is a property of the scene and of how long the camera rests.

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

## Damping a fresh cache (N6f, 2026-11-15)

A fresh network's first frames were the weak spot. Frame 1 renders before any training (render → inference → training is the
frame order), so it used the randomly initialised network's output directly. Settings A/B'd (`nrc-study.html?scenario=cold`,
640×360, 2 steps, windowed Chrome, 3 network seeds each; single-frame relMSE, mean |bias| in brackets):

| Setting | Ground floor t1 | t4 | t8 | Unseen corridor t1 | t4 | t8 |
|---|---|---|---|---|---|---|
| Random output layer (before) | 4.67 (31%) | 3.37 (25%) | 2.57 (9%) | 11.0 (77%) | 6.47 (32%) | 5.46 (13%) |
| Zero output layer | 2.36 (36%) | 2.61 (29%) | 2.17 (7%) | 5.36 (30%) | 5.46 (20%) | 5.13 (8%) |
| Plain path tracing for the first 4 frames | 9.31 (1%) | 8.86 (1%) | 2.58 (9%) | 14.8 (2%) | 14.2 (2%) | 5.40 (11%) |
| Learning rate ramped over 16 frames | 4.67 (31%) | 3.62 (27%) | 3.14 (23%) | 11.0 (77%) | 8.25 (52%) | 6.13 (31%) |
| Warm-up EMA weights | 4.67 (31%) | 3.25 (26%) | 2.71 (15%) | 11.0 (77%) | 6.33 (31%) | 5.48 (14%) |
| **Zero output layer + warm-up EMA (new default)** | **2.36 (36%)** | **2.45 (22%)** | **2.19 (3%)** | **5.36 (30%)** | **5.45 (19%)** | **5.14 (4%)** |
| Zero output layer + learning-rate ramp | 2.36 (36%) | 2.26 (30%) | 2.33 (12%) | 5.36 (30%) | 5.29 (21%) | 5.29 (21%) |

Steady state for comparison: about 2.4 (ground floor) and 5.4 (corridor); plain path tracing 8.6 / 14.5. With the new default a
fresh cache's first frame is already at the steady error, and the bias that remains for frames 1-4 is a slight darkening (the
cache starts at 0 and grows) instead of the old over-brightening (+77% in the corridor); it is down to 3-4% by frame 8. The cut
into the corridor (a warm cache) is unaffected by every setting (5.3 at frame 1). Showing plain path tracing first is unbiased
but 4× noisier for those frames; it stays a setting (`warmupFrames`). The warm-up EMA uses the decay min(0.99, (1 + t) / (10 + t))
(TensorFlow's num_updates form), so it follows the weights closely early on and needs no bias correction.

The defaults also help the steady state (640×360, warm cache): relMSE 0.116 vs 0.122 at 16 frames and 0.0092 vs 0.0113 at 256,
bias +0.1-0.5% instead of +1-2.6%, LDR-FLIP at 256 frames 0.069 (plain 0.087; before 0.078), flicker 1.01 vs plain 1.61.

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
