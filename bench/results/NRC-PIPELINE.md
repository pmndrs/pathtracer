# NRC through a denoiser + upscaler at equal time (track N, M9 / TASKS N11), 2026-11-17

**Question:** the cache's raw frame is still noisy at one sample per pixel. Does its lower-noise input survive a real-time chain
(SVGF denoiser at 960 × 540, then a temporal upscaler to 1920 × 1080) as a better final image than plain path tracing in the same time?

**Answer on the RTX 3060 Laptop, sponza-white: yes, but the chain eats most of the gap.** (With SVGF; OIDN, added the same day, is far better
than SVGF for both arms and keeps a 1.3-1.6× cache lead on single frames and 2× less flicker: see "OIDN in the same chain".) At equal time (plain gets 1 sample per
pixel per frame; a cache frame cost less than that here, see Timing) the cache arm's error is

| stage | relMSE ratio plain / NRC | FLIP plain → NRC | frame-to-frame flicker plain / NRC |
|---|---|---|---|
| raw (960 × 540) | **3.5-3.8×** | 0.48 → 0.31 | 1.5× |
| denoised (SVGF, 960 × 540) | **1.6-2.2×** | 0.13-0.20 → 0.10-0.14 | 1.8× |
| final (upscaled, 1920 × 1080) | **1.05-1.6×** (1.6× after a cut, 1.05-1.2× once the upscaler's history is full) | 0.12-0.21 → 0.10-0.15 (17-30% lower) | **2.0×** |

The denoised plain frame shows SVGF's typical 1-spp failure (blotches, streaks along edges, smeared contact shadows); the cache
arm's denoised frame is visibly cleaner (`static-f64-*-denoised.pfm`). After the upscaler, whose own temporal accumulation (24 frames)
keeps averaging a static view, both finals converge toward a floor that is mostly common to both (resampling mismatch against the
reference and a shared upscaler bias, below), so relMSE alone undersells the cache; FLIP and flicker keep a clear lead. Both arms lose
2.5-4% of the mean in the upscaler stage (section "Bias in the chain").

## OIDN in the same chain (added 2026-11-17, TASKS N11.3)

Same scene, camera, warm cache (300 frames), reference and upscaler; only the denoiser changes, for both arms. OIDN is the Denoiser package's
(WGSL on our device, fp16), fed the raw beauty, the primary-hit albedo factor and the view-space normal. While the camera is still, every OIDN
mode denoises the **progressive mean** of the frames since it last moved (1 frame while it moves), the idea from mutracer's docs: the small
network while moving, the standard one once a few still samples exist. Modes: `oidn` = small net until 8 still frames, then the standard net
once and the frame is held (no more tracing or denoising until the camera moves); `oidn-fast` / `oidn-std` = that net on every frame, no hold.
Runs `20261117T123742Z` (oidn), `124015Z` (oidn-fast), `124324Z` (oidn-fast, albedo only), `124631Z` (oidn-std); SVGF = the run above.

**Denoised stage (960 × 540), relMSE plain / NRC (ratio):**

| denoiser | static frame 1 (1 spp) | static frame 4 | static frame 16 (oidn: held at 8) | path, arrival frame 1 | path, frame 16 after |
|---|---|---|---|---|---|
| SVGF | 0.107 / 0.060 (1.77×) | 0.082 / 0.042 (1.97×) | 0.057 / 0.036 (1.58×) | 0.061 / 0.038 (1.62×) | 0.059 / 0.037 (1.60×) |
| OIDN small (`oidn-fast`), albedo + normal | 0.027 / 0.020 (1.37×) | 0.012 / 0.0097 (1.27×) | 0.0059 / 0.0061 (0.97×) | 0.017 / 0.012 (1.42×) | 0.0059 / 0.0057 (1.05×) |
| OIDN small, albedo only | 0.038 / 0.028 (1.37×) | 0.016 / 0.012 (1.39×) | 0.0076 / 0.0067 (1.14×) | 0.024 / 0.015 (1.63×) | 0.0074 / 0.0063 (1.16×) |
| OIDN standard (`oidn-std`) | 0.023 / 0.017 (1.37×) | 0.010 / 0.0086 (1.19×) | 0.0053 / 0.0053 (0.99×) | 0.015 / 0.011 (1.38×) | 0.0052 / 0.0048 (1.09×) |
| OIDN progressive (`oidn`, standard at 8, hold) | 0.027 / 0.020 (1.34×) | 0.012 / 0.0099 (1.24×) | 0.0073 / 0.0072 (1.02×) | 0.019 / 0.013 (1.46×) | (frame 8) 0.0073 / 0.0066 (1.12×) |

**Final stage (1920 × 1080), relMSE plain / NRC, and flicker of the final:**

| denoiser | static frame 1 | static frame 16 | path arrival | final flicker plain / NRC (static, path) |
|---|---|---|---|---|
| SVGF | 0.100 / 0.063 | 0.032 / 0.027 | 0.028 / 0.026 | 1.97×, 2.04× (from frame 16; the others from frame 2) |
| OIDN small, albedo + normal | 0.030 / 0.023 | 0.0091 / 0.0089 | 0.020 / 0.018 | 1.89×, 2.02× |
| OIDN small, albedo only | 0.043 / 0.032 | 0.012 / 0.010 | 0.032 / 0.026 | 2.00×, 2.16× |
| OIDN standard | 0.026 / 0.019 | 0.0081 / 0.0079 | 0.018 / 0.014 | 1.93×, 2.15× |
| OIDN progressive | 0.030 / 0.023 | 0.011 / 0.011 (held) | 0.021 / 0.019 | 1.98×, 2.51× |

What it shows:

* **OIDN beats SVGF by a wide margin on our 1-spp input:** at frame 1 the plain arm's denoised relMSE drops 4× (0.107 → 0.027 with the small
  net), the cache arm's 3× (0.060 → 0.020); the final, 3.3× / 2.7×. Plain + OIDN small (0.027) is already better than cache + SVGF (0.060).
  Not using OIDN here would indeed have been a mistake.
* **The cache still helps under OIDN, less than under SVGF:** 1.3-1.4× lower relMSE on single frames (frame 1, arrival after the path), 1.6×
  with albedo-only guides; once a still camera has accumulated 8-16 frames both arms meet (0.97-1.1×), because OIDN on a 16-frame mean is
  already near the floor. Final-stage flicker stays **2× lower** with the cache in every OIDN mode (2.5× on the path with the progressive mode).
* **relMSE flatters OIDN's smoothing.** LDR-FLIP at frame 1: plain 0.202 (SVGF) → 0.178 (OIDN small); cache 0.141 (SVGF) vs 0.147 (OIDN small),
  0.133 (OIDN standard). The small net paints over fine detail (curtain folds, capitals; `static-f1-*-denoised.pfm`) where SVGF keeps edges
  with speckle. FLIP ranks cache + SVGF level with cache + OIDN small at frame 1.
* **Albedo-only guides** cost ~40% relMSE against albedo + normal on single frames, and the cache's lead grows (1.4-1.6×): without normals,
  OIDN leans more on the colour, and the cache's colour is cleaner. That is the configuration where the cache pays most.
* **The upscaler bias shrinks with OIDN:** −0.1 … −1.7% on the static view (SVGF: −1.8 … −6.1%), consistent with the bias depending on residual
  noise (N11.5). On the camera path it stays −2 … −3.5% for both denoisers (motion, not noise).
* **Progressive mode**: the standard net at 8 still frames reaches 0.0073 denoised (plain and cache alike), then holds: no tracing, no
  denoising and zero flicker until the camera moves. The small net on the 16-frame mean ends lower (0.0059) only because it saw twice the samples.
* **A cold cache is worse than none, under both denoisers** (measured in the next section, N11.6): OIDN needs ~150 training frames
  before the cache helps, SVGF ~10. The warm-up gate (N11.7, section "Warm-up gate") shows plain path tracing until then: never worse than plain at start-up.

**Cost (best effort, same throttled 3060, wall clock):** OIDN small 22-25 ms per frame at 960 × 540 (post − post without denoiser), standard
48-66 ms; SVGF 2.8 ms. With the small net the cache arm's whole chain was 76-87 ms vs 72.6 ms with SVGF. The progressive mode spends the
standard net once per still view and nothing after it.

## Cold start (N11.6, 2026-11-17)

How long must a fresh network train before the cache arm beats plain path tracing through the chain? Per warm-up W: fresh weights and
optimizer (`resetNetwork`; zero output layer and warm-up EMA, the N6f defaults), W training frames at the reference view (4 steps of 16k
records each), a cut, then frame 1 and frame 4 (the cache keeps training; OIDN denoises the 1- and 4-frame means). Plain is measured each
time as the control. Runs `20261117T130444Z` (SVGF), `130811Z` + `131553Z` (OIDN small, albedo + normal), `131133Z` (OIDN small, albedo only).

**Denoised relMSE, plain / NRC (ratio > 1: the cache helps), frame 1 after the cut:**

| warm-up frames | SVGF | OIDN small, albedo + normal | OIDN small, albedo only |
|---|---|---|---|
| 0 | 0.107 / 0.247 (0.43×) | 0.027 / 0.221 (0.12×) | 0.038 / 0.220 (0.17×) |
| 3 | 0.124 / 0.168 (0.74×) | 0.027 / 0.102 (0.26×) | 0.039 / 0.142 (0.28×) |
| 10 | 0.133 / 0.105 (1.27×) | 0.026 / 0.058 (0.45×) | 0.037 / 0.072 (0.51×) |
| 30 | 0.107 / 0.082 (1.31×) | 0.026 / 0.039 (0.66×) | 0.036 / 0.045 (0.80×) |
| 100 | 0.124 / 0.072 (1.71×) | 0.026 / 0.028 (0.93×) | 0.038 / 0.036 (1.05×) |
| 150 | | 0.027 / 0.024 (1.13×) | |
| 200 | | 0.027 / 0.020 (1.32×) | |
| 250 | | 0.026 / 0.020 (1.28×) | |
| 300 | 0.120 / 0.064 (1.89×) | 0.026 / 0.019 (1.35×) | 0.038 / 0.028 (1.33×) |

(Frame 4 tells the same story, with the OIDN crossover a little later: 0.72× at 100, 0.91× at 150, 1.09× at 200. Final-stage ratios follow
the denoised ones within ~10%. Plain's spread between rows, 0.107-0.133 for SVGF, is its noise.)

* **The raw frame hides it.** From W = 3 on, the cache arm's raw relMSE (2.2-2.5) and bias (|≤ 2.5%|) look the same as at W = 300. What
  is left after a few hundred training steps is smooth, low-frequency error: invisible under 1-spp noise, intact after a denoiser removes
  that noise. OIDN removes white noise best, so it exposes the cache's error most, and needs the longest training.
* **Crossover:** SVGF ~5-10 frames; OIDN albedo + normal ~120-150 frames (plateau ~1.3× from 200); OIDN albedo only ~100. LDR-FLIP is
  kinder to the cache (OIDN: the cache arm's FLIP is already lower at W = 100, 0.160 vs 0.176), so a perceptual crossover is earlier.
* **W = 0** is −36% dark at frame 1 (the zero output layer predicts no indirect light) and overshoots at frame 4 (+9 … +17% raw bias):
  N6f's damping fixed the raw error of a fresh cache, not the bias a denoiser shows.
* What this means at run time: 150 frames at 4 steps is ~600 training steps of 16k records, ~2.5 s at 60 fps, ~10 s at the ~15 fps of a cache frame on this 3060.
  It is per network, not per view: nrc-study found a cache warmed in one part of Sponza steady from the first frame in an unseen corridor
  (raw error; through OIDN not yet measured).

**Suggestion (maintainer's call): a warm-up gate per denoiser.** The cache arm shows plain path tracing while the network trains, and switches
to the cache after G training frames; `NrcSettings.warmupFrames` already does exactly this (0 today). G ≈ 10 with SVGF, ≈ 150-200 with OIDN.
Pros: never worse than plain at start-up or a scene load; one existing setting. Cons: a fixed count, not a measured readiness (a harder scene
or fewer steps per frame needs more); start-up frames cost a full plain path. Alternative: more training steps per frame (or a higher
learning rate) while the network is young, which shortens the gate instead of hiding it. A/B that settles it: this coldstart scenario with
`warmupFrames` = 0 / 150 / 200 and with 8-16 steps per frame during warm-up, plus an unseen-region cut through OIDN (N11.7). **Built and measured: next section.**

## Warm-up gate (N11.7, 2026-11-17)

**Question:** at start-up, can the cache arm show plain path tracing through the chain while the network trains in the background, and
switch to the cache once it has trained, so that it is never worse than plain? (The maintainer's idea, which is `NrcSettings.warmupFrames`: the
hook runs full paths for the image while the same pass's training tiles keep recording, so training is exactly what it would be with the cache on
screen.)

**Setup:** twin `scenarios=gate` (nrc-pipeline-study.ts §6). Per arm: fresh network (`resetNetwork`), gate G (frames) at s steps per gated frame,
then a 300-frame start-up timeline. The camera looks around near the reference view (yaw ±10°, 0.5 m sway, 120-frame cycle: the moving-camera
denoiser path runs and the network trains on nearby views). At each checkpoint frame there is a cut to the reference view and one measured frame
(both arms; plain runs 1 spp, N = 1 here). After the timeline there is a cut into the upper gallery's corridor (nrc-study's view C, which the
timeline never sees), and frames 1 and 4 are measured against our own 2048 spp reference there. Runs `20261117T162022Z` (SVGF: G = 0, 10, 30) and
`20261117T163133Z` (OIDN small net, albedo + normal: G = 0, 150, 200, and 40 frames × 16 steps, 75 × 8). Quality only: no timing taken
from these runs.

**Denoised relMSE ratio plain / NRC (> 1: the cache arm is better) and FLIP of the cache arm, single frame after a cut:**

| frame (steps if 4/frame) | SVGF G = 0 | SVGF G = 10 | SVGF G = 30 | OIDN G = 0 | OIDN G = 150 | OIDN G = 200 | OIDN 40 × 16 steps | OIDN 75 × 8 steps |
|---|---|---|---|---|---|---|---|---|
| 1 (4) | 0.43×, FLIP 0.54 vs 0.20 | gated 1.00× | gated 1.25× | 0.12×, 0.55 vs 0.18 | gated 1.02× | gated 0.99× | gated 1.03× | gated 0.95× |
| 10 (40) | 0.99×, 0.24 vs 0.19 | gated 1.25× | gated 0.75× | 0.35×, 0.25 vs 0.18 | gated 0.97× | gated 0.95× | gated 1.09× | gated 0.96× |
| 11 / 31 / 41 (first cache frame) | | **1.17×, FLIP 0.225 vs 0.198** | **1.37×, 0.193 vs 0.202** | | | | **1.16×, 0.153 vs 0.174** (644 steps) | |
| 15 (60) | 1.09×, 0.218 vs 0.196 | 1.33×, 0.203 vs 0.198 | gated 1.06× | | | | | |
| 40 (160) | | | | 0.79×, 0.190 vs 0.179 | gated 0.95× | gated 0.96× | gated 0.99× | gated 1.01× |
| 76 (304) | | | | 0.88×, 0.176 vs 0.177 | gated 0.99× | gated 1.02× | 1.25× | **1.14×, 0.153 vs 0.176** (604 steps) |
| 100 (400) | 2.23× | 1.71× | 1.99× | 1.01×, 0.164 vs 0.176 | gated 1.01× | gated 1.04× | 1.24× | 1.18× |
| 151 (604) | | | | 1.12×, 0.158 vs 0.178 | **1.17×, 0.158 vs 0.176** | gated 1.03× | 1.32× | 1.27× |
| 201 (804) | | | | 1.12× | 1.17× | **1.19×, 0.151 vs 0.177** | 1.42× | 1.19× |
| 300 (1200) | 1.78× | 1.89× | 2.02× | 1.31× | 1.34× | 1.46× | 1.51× (1680 steps) | 1.35× (1500 steps) |
| unseen corridor, frame 1 | 1.27× | 1.44× | 1.40× | 0.96×, 0.149 vs 0.151 | 1.02× | 1.08× | 0.85× | 0.99× |
| unseen corridor, frame 4 | 1.44× | 1.45× | 1.36× | 0.76×, 0.122 vs 0.109 | 0.72× | 0.85× | 0.69× | 0.82× |

(SVGF single frames after a cut spread a lot: plain alone reads 0.106-0.157 between checkpoints, so a gated row of 0.75× or 1.25× is plain vs
plain noise. OIDN's spread is ±5%. Final-stage ratios follow the denoised ones within ~10%; tables per frame: `node gate.ts <run>` in the session
scratchpad, metrics `gate-g<G>[s<s>]_f<t>_*` in the run JSON.)

* **The gate does what it should.** While it is closed, the cache arm IS plain path tracing (denoised 0.95-1.09× with OIDN, final 0.93-1.06× with
  SVGF: plain-vs-plain noise; bias as plain). The first cache frame after it opens is as good as an ungated network after the same number of steps (OIDN G = 150 at
  frame 151: 1.17× vs 1.12×; SVGF G = 30 at frame 31: 1.37× vs 1.53×, within SVGF's spread). Training does not notice the gate: per-frame loss,
  step counts and the frame-300 error match the ungated arm.
* **Ungated, the young cache is worse than plain for a long time:** OIDN until ~100 frames by relMSE (0.12× at frame 1, 0.35× at 10, 0.79× at 40)
  and ~75 by FLIP; SVGF until ~10-15 frames by relMSE and ~30 by FLIP (the frame-10 cache is level in relMSE but 0.24 vs 0.19 in FLIP).
  That matches N11.6's offline curve, now on a moving start-up timeline.
* **The gate is a budget of training steps, not frames.** OIDN: 150 frames × 4, 75 × 8 and 40 × 16 steps open at 600-644 steps, and the first cache
  frame reads the same (1.17×, 1.14×, 1.16×; FLIP 0.153-0.158). SVGF: G = 10 (40 steps) still loses on FLIP for 5 frames (0.225 vs 0.198 at frame 11,
  0.203 vs 0.198 at 15), G = 30 (120 steps) never does. **Setting (D30): 120 steps with SVGF, 600 with OIDN** (30 and 150 frames at 4 steps).
* **More steps per gated frame shortened the gate with no sign of instability** (the maintainer's concern). With 16 steps per frame the OIDN gate
  opens after 40 frames instead of 150. The per-frame loss of the 16-step arm stays in the same band as the 4-step arms (30-190, the noise of one
  frame's targets, no drift and no NaN). After the switch it stays ahead of the 4-step arms at equal frames (1.32× vs 1.12-1.17× at frame 151), because
  it has had more steps. One scene, one seed, the default learning rate (no boost). Cost: 12 more steps per gated frame. Training measured
  ~5 ms per 4 steps in M6 (best effort), so about +15 ms on a ~62 ms frame, for 40 frames instead of 150. The maintainer decides whether the
  demo's auto mode uses it (default today: the frame's own steps).
* **A cut into an unseen region is not covered by the gate under OIDN.** With SVGF, a network warmed for 300 frames on the ground floor is
  1.3-1.4× better than plain at frame 1 in the corridor: one gate per network is enough. With OIDN it is only level at frame 1 (0.85-1.08×) and
  worse at frame 4 (0.69-0.85×, FLIP 0.12-0.13 vs 0.11), whatever the steps spent elsewhere (1200-1680). The cache's raw error there is 2.6×
  below plain's and its bias is level (±2%). It is the low-frequency error again, now in a region the network has not trained on, and OIDN's
  4-frame mean exposes it. Caveat: the 2048 spp corridor reference has a floor of about 7e-3 relMSE (raw 1-spp relMSE 14.5 / 2048), the
  same size as the frame-4 values (8-12e-3). So the frame-4 ratios are compressed toward 1, and the true gap is larger, not smaller. FLIP
  agrees on the direction. Follow-up N11.10: how many frames in the new region until the cache wins again, against a 16k spp reference. Then
  either re-gate on a cut into poorly trained space, or use the ramp (N11.8).

## Setup

* Scene `sponza-white` (sun + sky, grey Lambert, maxDepth 8, RR from vertex 4, NEE), the scene camera. Both arms: the frame-mode megakernel
  (`integrator.frame`, blue-noise sampler, scene copy packed); the cache arm adds the hook (inline training, 4 steps of 16k records per frame,
  tiled f16 inference: Chrome on Windows has no subgroup matrices), query at vertex 1. Chain (identical): pack → Denoiser
  `DiffuseSpecularDenoiser` (SVGF, A-SVGF anti-lag auto, the Denoiser's defaults, remodulated) → composite → pmndrs/upscaler temporal path
  (defaults: sharpness 0.8, auto exposure, 24 frames; `jitter: false`, the sampler owns the jitter). [nrc-pipeline.md](../../docs/design/nrc-pipeline.md).
* Reference: **our own** plain megakernel at 4096 spp, 1280 × 720 (the Mitsuba PFM is git-ignored and lives on the Mac; N11.4 reruns against
  it), resampled (tent) to 960 × 540 for raw / denoised and to 1920 × 1080 for the final. Its image mean (0.24612 0.24045 0.23956) lies between
  Mitsuba's (0.24646 0.24077 0.23987, −0.14%) and pbrt's (0.24590 0.24024 0.23939, +0.09%) in `bench/references/agreement.md`, so bias numbers
  below 0.2% are not meaningful and the ratios do not depend on the choice.
* Cache warmed 300 frames at the view first (a user who has been looking around; nrc-study showed a warm cache is steady from frame 1
  after a cut). `static`: a cut to the view, 64 frames, checkpoints 1, 4, 16, 64. `path`: a cut to a pose 2 m back, 0.4 m up, 1.2 m to the
  side, a 48-frame smoothstep dolly + pan into the view (histories reproject along it), then a hold; checkpoints 1, 4, 16 frames after arrival.
* Flicker: per pixel, mean |L_t − L_(t−1)| over the frames (static: frames 16-64; path: the hold), divided by (reference luminance + 0.01),
  averaged (nrc-study's measure; accumulated on the GPU, read once).
* Run `20261117T113629Z-nrc-pipeline-study` (`node bench/runner/run.ts --page nrc-pipeline-study --gpu nvidia --timeout 3600000 --args
  warm=300 frames=64 ref=ours`), Chrome 157 headless, D3D12, RTX 3060 Laptop. The run was marked failed only by the runner's console-error
  rule (the probe for the missing Mitsuba PFM; fixed: `ref=` arg). PFMs of every stage, both arms, `static-f1`, `static-f64`, `path-mid`,
  `path-f1` and the reference are in `bench/results/incoming/<run>/1-run/` (git-ignored; the bench viewer compares them).

## Full table

| scenario | frame | stage | plain relMSE | NRC relMSE | ratio | plain FLIP | NRC FLIP | plain bias | NRC bias |
|---|---|---|---|---|---|---|---|---|---|
| static | 1 | raw | 8.835 | 2.311 | 3.82× | 0.483 | 0.313 | -0.3% | -0.3% |
| static | 1 | denoised | 0.107 | 0.060 | 1.77× | 0.202 | 0.141 | -1.4% | -1.0% |
| static | 1 | final | 0.100 | 0.063 | 1.59× | 0.209 | 0.147 | -1.8% | -1.4% |
| static | 4 | raw | 9.127 | 2.407 | 3.79× | 0.483 | 0.310 | 0.4% | 0.4% |
| static | 4 | denoised | 0.082 | 0.042 | 1.97× | 0.179 | 0.123 | -4.1% | -2.4% |
| static | 4 | final | 0.050 | 0.036 | 1.37× | 0.173 | 0.126 | -5.1% | -3.4% |
| static | 16 | raw | 8.617 | 2.439 | 3.53× | 0.481 | 0.309 | -0.0% | 0.3% |
| static | 16 | denoised | 0.057 | 0.036 | 1.58× | 0.138 | 0.101 | -2.1% | -0.8% |
| static | 16 | final | 0.032 | 0.027 | 1.18× | 0.149 | 0.112 | -6.1% | -3.8% |
| static | 64 | raw | 8.962 | 2.407 | 3.72× | 0.483 | 0.310 | 0.1% | 0.6% |
| static | 64 | denoised | 0.081 | 0.038 | 2.16× | 0.134 | 0.099 | -2.0% | -0.6% |
| static | 64 | final | 0.025 | 0.024 | 1.06× | 0.123 | 0.102 | -5.6% | -4.0% |
| path | 1 | raw | 9.111 | 2.439 | 3.73× | 0.483 | 0.308 | 0.3% | 0.8% |
| path | 1 | denoised | 0.061 | 0.038 | 1.62× | 0.132 | 0.100 | -1.6% | -0.5% |
| path | 1 | final | 0.028 | 0.026 | 1.10× | 0.128 | 0.104 | -4.0% | -2.7% |
| path | 4 | raw | 8.915 | 2.424 | 3.68× | 0.483 | 0.309 | -0.2% | 0.3% |
| path | 4 | denoised | 0.068 | 0.037 | 1.82× | 0.134 | 0.100 | -1.8% | -0.7% |
| path | 4 | final | 0.027 | 0.025 | 1.08× | 0.125 | 0.102 | -4.2% | -2.7% |
| path | 16 | raw | 8.978 | 2.414 | 3.72× | 0.482 | 0.309 | 0.0% | 0.3% |
| path | 16 | denoised | 0.059 | 0.037 | 1.60× | 0.133 | 0.097 | -1.9% | -0.4% |
| path | 16 | final | 0.026 | 0.024 | 1.05× | 0.121 | 0.099 | -4.6% | -3.0% |

| scenario | stage | plain flicker | NRC flicker | ratio |
|---|---|---|---|---|
| static | raw | 1.6289 | 1.0648 | 1.53× |
| static | denoised | 0.1046 | 0.0564 | 1.85× |
| static | final | 4.80e-3 | 2.44e-3 | 1.97× |
| path | raw | 1.6345 | 1.0951 | 1.49× |
| path | denoised | 0.1022 | 0.0566 | 1.81× |
| path | final | 5.67e-3 | 2.78e-3 | 2.04× |

Raw relMSE is ~9 for plain 1 spp (a single sample of 8-vertex paths with a sun); the denoiser removes ~99% of it. The raw cache arm sits at
3.5-3.8× lower relMSE, in line with the windowed nrc-study on the M5 Pro (3.2-4.3× at equal frames).

## Bias in the chain

The cache's light is in the planes where it belongs: `diffuse × albedo + specular × specFactor` equals the beauty bit for bit in both arms
(`plane_identity_*_max_rel` 0, beauty mean 0.242 against the reference's 0.241; exact because sponza-white's albedo 0.5 is a power of two), and
the cache arm's DENOISED frame is unbiased (−0.1% at 16 frames). Two stages lose light:

* SVGF on the plain arm: −1.4 … −4.1% (the 1-spp input has the fireflies and near-black pixels SVGF's variance-guided à-trous pulls toward
  their neighbours); on the cache arm −0.4 … −2.4% in the first frames, −0.1 … −0.8% once the history is full.
* The upscaler: another −2.5 … −4% on BOTH arms, on a static camera, so it is not reprojection. Upscaler A/B at 16 frames (static view, warm
  150, `refFile` = the 4096 spp reference; runs `20261117T114956Z`, `115028Z`, `115102Z`):

| upscaler settings | plain final relMSE / bias | NRC final relMSE / bias | final flicker plain / NRC |
|---|---|---|---|
| defaults (sharpness 0.8, auto exposure) | 0.0323 / −6.1% | 0.0318 / −3.2% | 7.9e-3 / 3.8e-3 |
| sharpness 0 (no RCAS sharpening) | 0.0291 / −6.7% | 0.0306 / −3.4% | 6.6e-3 / 2.9e-3 |
| auto exposure off (exposure 1) | 0.0324 / −5.2% | 0.0318 / −2.8% | 8.0e-3 / 3.9e-3 |

  Neither knob removes it. The temporal path always accumulates in a Reinhard-compressed space (`FLAG_INPUT_REINHARD`), and averaging noisy HDR
  values after `x / (1 + x)` and expanding back is biased low; the noisier plain input loses more, which fits. Not confirmed (the sibling is
  read-only here): the test is the same chain fed a noise-free input (the reference itself), N11.5. The final relMSE floor (~0.025) is mostly common to
  both arms: the bias adds about b² (1e-3 for −3%, 4e-3 for −6%), and the rest is edge and texture mismatch between the upscaler's reconstruction
  of a 960 × 540 image and the tent-upsampled 1280 × 720 reference, plus SVGF's blur. That shared floor is why the final relMSE ratio shrinks
  toward 1 while FLIP and flicker keep the cache's lead; a reference rendered at 1920 × 1080 would lower the floor (N11.4). Sharpness 0 lowers both arms' final relMSE and flicker a little; the plain arm gains more.

## Timing (best effort, NOT a quiet-pass number)

Wall clock per submit + `onSubmittedWorkDone`, median of 20, headless Chrome on the RTX 3060 Laptop at 89 °C with thermal throttling in 97% of
the samples (SM median 900 MHz) and CPU load 16 from other sessions: the runner marked the timing INVALID. Ratios only, queued for the quiet pass
(QUIET-PASS N11).

| piece, 960 × 540 | ms |
|---|---|
| plain path tracing, 1 spp | 74.1 |
| cache frame (render + 4 training steps + inference + resolve) | 61.7 |
| pack + depth | 4.2 |
| denoise + composite | 2.8 |
| upscale to 1920 × 1080 | 3.9 |
| plain arm, whole chain (N = 1) | 85.0 |
| NRC arm, whole chain | 72.6 |

A cache frame costs 0.83× a plain sample here (0.61× windowed on the M5 Pro), so plain at 1 spp per frame already spends more time than the
cache arm: the ratios above are at "equal time or better for plain". The post chain (pack, denoise, upscale) is ~11 ms of wall time per arm at
this size; the trace dominates and neither arm reaches 16.7 ms on this GPU (nrc-pipeline.md §5).
