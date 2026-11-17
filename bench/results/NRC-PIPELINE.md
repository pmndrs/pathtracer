# NRC through a denoiser + upscaler at equal time (track N, M9 / TASKS N11), 2026-11-17

**Question:** the cache's raw frame is still noisy at one sample per pixel. Does its lower-noise input survive a real-time chain
(SVGF denoiser at 960 × 540, then a temporal upscaler to 1920 × 1080) as a better final image than plain path tracing in the same time?

**Answer on the RTX 3060 Laptop, sponza-white: yes, but the chain eats most of the gap.** At equal time (plain gets 1 sample per
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
