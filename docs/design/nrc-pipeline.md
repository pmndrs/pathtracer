# NRC pipeline: cache → denoiser → upscaler (track N, M9)

The cache's strength is not a raw frame (still noisy at one sample per pixel) but a lower-noise input to a real-time chain. M9 puts
both arms through the same chain at equal time and measures the output. Plan row: [nrc-plan.md](nrc-plan.md) M9; tasks: TASKS N11.x;
divergences: [nrc-diverges.md](nrc-diverges.md) N-I13..N-I17.

```
path tracer, frame mode (integrator.frame)          960 x 540, 6 planes + beauty
  plain arm: N samples / frame  ─┐
  NRC arm: 1 sample + cache      ─┤  accumulate (sums of re-modulated class radiance and factors)
                                  ▼
                pack: textures (diffuse / specular demodulated, albedo, specular factor, normal, motion, viewZ, motionZ,
                      roughness, hit distances) + device depth from viewZ (one render pass, frag_depth)
                                  ▼
                Denoiser `DiffuseSpecularDenoiser` (SVGF + A-SVGF anti-lag, remodulate)       960 x 540
                  + composite: diffuse + specular, raw sky where the primary ray missed
                or OIDN (Denoiser package): progressive mean while still, small / standard net, hold
                                  ▼
                pmndrs/upscaler, temporal path, jitter: false                                1920 x 1080
```

Code: [`bench/nrc/pipeline.ts`](../../bench/nrc/pipeline.ts) (`PipelineArm`: the chain), [`bench/nrc/pipeline-session.ts`](../../bench/nrc/pipeline-session.ts)
(scene, both arms, one frame, stage timing), [`bench/nrc/cache.ts`](../../bench/nrc/cache.ts) (frame-mode resolve). Demo:
`node bench/live.ts nrc-pipeline`; headless twin: `node bench/runner/run.ts --page nrc-pipeline-study --gpu nvidia --timeout 3600000 [--args ref=ours]` (the default
`ref=mitsuba` needs the git-ignored Mitsuba PFM, which only the Mac has).

## 1. What already existed

* main's real-time mode (P4.1, [realtime.md](realtime.md)): `integrator.frame` writes one sample per pixel per encode into the beauty
  buffer and 6 planes (demodulated diffuse / specular with first-bounce hit distance, normal + viewZ, motion + roughness + motionZ,
  albedo factor, specular factor). Megakernel only.
* The cache (`NrcCache`, N6a/N6d): a hooked copy of the plain kernel; paths stop at vertex `queryDepth` (default 1, the second vertex)
  after its emission and NEE, append a query, and a resolve pass adds `throughput × prediction` to the pixel after inference.
* `bench/adapters/shared/oidn.ts`: OIDN 2 from the Denoiser sibling (one frame, AOV guides; used for P4.7 previews).
* The Denoiser's `denoiser-realtime` (sibling, MIT, clean room from the SVGF / A-SVGF papers, its own RULES.md): per-signal temporal
  denoisers on raw WebGPU, input contract = CAPTURE-FORMAT (caller-demodulated diffuse and specular, three.js view-space conventions).
* pmndrs/upscaler (sibling, MIT with AMD's FidelityFX attribution): FSR1 spatial and an FSR2/3-style temporal path as raw WGSL compute
  passes, API written for three's `WebGPURenderer`.

## 2. The cache's light in the frame planes

The planes split radiance by the class of the PRIMARY vertex (V122): Lambert and layered = diffuse, conductor / dielectric lobe =
specular; everything after the primary vertex (its NEE, all bounces) belongs to that class. The cache's estimate replaces the rest of
the path at vertex `queryDepth` ≥ 0, which is the primary vertex or later, so it always belongs to the primary vertex's class too.
The integrator never sees it (the hook returns 0), so the planes are short of exactly what the resolve adds to the beauty. In frame mode
the resolve therefore also adds `throughput × prediction / factor` to plane 0 (diffuse, factor = plane 4 rgb) or plane 1 (specular,
factor = plane 5 rgb), chosen by plane 4's alpha (1 = specular-class primary). The training-pixel fix-up (which removes what a training
path gathered beyond the cache vertex) subtracts from the same plane. The stored factors are the floored ones (V126), so
`diffuse × albedo + specular × specFactor = beauty` stays exact with the cache too. The hit-distance alpha is the first bounce ray's
length and is unchanged (the cache vertex is at or beyond it). Text is emitted only in frame mode (`resolveWgsl(frame)`), so the
existing pages' kernels are unchanged.

N samples per frame (plain arm): the kernel overwrites the planes per sample, so the arm sums re-modulated class radiance and the
factors over the frame's samples and divides: `diffuse = Σ(dᵢ·aᵢ) / Σaᵢ` keeps `diffuse × mean(a) = mean diffuse-class radiance` exact.
Guides (normal, viewZ, motion, roughness, hit distance) come from the frame's last sample (N-I15).

## 3. Denoisers: the Denoiser's real-time SVGF, and its OIDN

Both come from the Denoiser sibling and run on our device; `PipelineArm.setDenoiser('svgf' | 'oidn' | 'none')` switches both arms together.

| | `denoiser-realtime` `DiffuseSpecularDenoiser` (`svgf`) | Denoiser package OIDN (`oidn`) |
|---|---|---|
| Kind | temporal (history, reprojection with our motion), à-trous spatial, anti-lag | single image, UNet (small / base / large) |
| Input | demodulated diffuse + specular, our planes map 1:1 (realtime.md §3 already wrote the mapping) | beauty + albedo (+ view-space normal), the progressive mean while the camera is still |
| Cost (RTX 3060, 960 × 540, best effort) | 2.8 ms | small net 22-25 ms, standard 48-66 ms |
| Quality at 1 spp (relMSE, denoised, plain / cache) | 0.107 / 0.060 | small 0.027 / 0.020; standard 0.023 / 0.017 |
| Status | experimental (0.1.0, private), defaults decided by Dennis 2026-10-25 (SVGF, anti-lag auto) | the package's stable path (P2 exit item) |

SVGF is the filter the spec names (§7: demod → SVGF-class filter → upscaler) and costs a tenth of OIDN's small net; OIDN is 3-4× lower in
relMSE on our 1-spp frames (it smooths more: FLIP is level for the cache arm, NRC-PIPELINE.md). Neither is the default yet: the decision
needs the quiet-pass cost and an A/B in motion with the realtime preset (N11.3). Clean room: the Denoiser package is our own code written
from papers and Intel's Apache-2.0 OIDN weights; NRD is never opened, and the Denoiser's native NRD harness is not used here.

SVGF inputs: `diffuse.radiance` = plane 0 rgb, `specular.radiance` = plane 1 rgb with `demodulation` = plane 5 rgb, `guides.albedo` = plane 4
rgb (diffuse remodulation), `hitDistance` = planes 0 / 1 alpha, `normal` = plane 2 rgb, `viewZ` = plane 2 a, `motion` = plane 3 xy,
`roughness` = plane 3 z, `motionZ` = plane 3 a, `depth` = device depth written from viewZ with the pipeline's projection (near 0.05 m,
far 200 m). It outputs 0 where no surface was hit; the composite puts the raw sky back there.

OIDN (N-I17; the schedule follows mutracer's denoising docs, not its code): a prep pass keeps the mean of beauty, albedo factor and normal
over the frames since the camera last moved (exact camera equality; a cut resets it), rotates the normal to view space (OIDN's aux
convention) and writes three rgba16float textures; `denoiseTextures({ color, albedo, normal?, hdr: true, output })` writes the upscaler's
input directly. Settings (`OidnArmSettings`): `aux` albedo + normal (the `calb_cnrm` nets; our 1-spp guides are nearly clean) or albedo only;
`moving` net (default small) until `stillSamples` (default 8) still frames, then the `still` net (default standard) once; `hold` (default
on) then keeps that frame: the arm stops tracing and denoising until the camera moves. One Denoiser instance per network (a switch reloads
weights). `denoiseTextures` waits for its GPU work (no buffer is mapped), so `post` and `session.frame` are async.

## 4. Upscaler: pmndrs/upscaler, temporal path

The only upscaler sibling is pmndrs/upscaler (`GitHub/upscaler/upscaler` on Windows, `~/Developer/upscaler` on the Mac; bench links it as
`@pmndrs/upscaler` via `link:../../upscaler`, a worktree needs the `.claude/worktrees/upscaler` junction). Its API wants three's
`WebGPURenderer`, but it only touches `backend.device`, `backend.get(texture).texture` and `initTexture(texture)` for its own
StorageTextures. `UpscalerHost` (pipeline.ts) provides those three from our device, so no three renderer runs; its camera stand-in gives
`near`, `far`, `projectionMatrix`, `matrixWorld`, `matrixWorldInverse`. Inputs: colour (rgba16float, denoised), depth (depth32float from
viewZ), velocity (our motion plane: NDC current − previous, the same convention as three's `velocity` node). Settings: defaults
(sharpness 0.8, auto exposure, 24-frame accumulation), the same in both arms.

**Jitter: exactly one owner, the path tracer's sampler.** In frame mode the sampler's pair 0 jitters every pixel's primary ray inside the
pixel (blue noise, a different offset per pixel and frame). The upscaler's own jitter is a single camera offset per frame, which the
megakernel has no way to apply (its camera rays take the per-pixel pair). So the upscaler runs with `jitter: false` (it still reprojects,
clips, accumulates and upscales; its doc: "no reconstruction gain, no smear risk") and the denoiser gets `jitter: [0, 0]`. Motion is
jitter-free by construction (realtime.md §3). The alternative, the upscaler as the owner (one Halton offset per frame through a uniform
camera shift in the kernel, pair 0 off), may resolve more detail at 2×; it needs a kernel option and is an A/B for later (N11.2, N-I14).

## 5. Frame budget, 960 × 540 rendered, 1920 × 1080 output

Target 16.7 ms. Measured on the RTX 3060 Laptop, Chrome 157 D3D12 headless, wall clock per submit, median of 20 (run `20261117T113629Z`).
**BEST EFFORT, timing marked INVALID by the runner:** thermal throttling in 97% of samples (89 °C, SM median 900 MHz) and CPU load 16 from
other sessions; NRC headless times also differ from windowed ones (N6e). Ratios only; the real numbers are the quiet pass's (QUIET-PASS N11).

| piece | RTX 3060 (best effort) | M5 Pro (plan §3, not measured here) |
|---|---|---|
| plain path tracing, 1 spp (8 vertices, RR from 4) | 74.1 ms | ~13.5 ms |
| cache frame (render to vertex 1 + 4 training steps of 16k + inference + resolve; tiled f16) | 61.7 ms (0.83× a plain sample) | ~11 ms (sgm f16; 0.61× windowed at 720p) |
| pack + depth | 4.2 ms | — |
| denoise + composite: SVGF / OIDN small / OIDN standard | 2.8 / 22-25 / 48-66 ms | — |
| upscale | 3.9 ms | — |
| **NRC arm, whole chain** (SVGF / OIDN small) | **72.6 / 76-87 ms** | |
| plain arm at equal time (N = 1) | 85.0 ms | |

A cache frame is cheaper than one plain sample, so equal time gives plain N = 1 and the cache arm still spends less. Neither GPU reaches
16.7 ms at 960 × 540 with the scene's 8-vertex paths: the trace (common to both arms) is the bulk. The post chain is ~11 ms of wall time
here (submit overhead included); on a quiet GPU it should be a few ms. Levers if the budget matters: the realtime preset's maxDepth 2 for
both arms (equal time spent on depth instead of samples, §7), sgm inference where the browser offers it, fewer training steps.

## 6. Results (sponza-white, RTX 3060, [bench/results/NRC-PIPELINE.md](../../bench/results/NRC-PIPELINE.md))

At equal time the cache arm wins at every stage, but the chain narrows the gap: relMSE ratio plain / NRC **3.5-3.8×** raw, **1.6-2.2×** after
SVGF, **1.05-1.6×** final (1.6× right after a cut, ~1.1× once the upscaler's 24-frame history is full); LDR-FLIP of the final 17-30% lower;
frame-to-frame flicker **2× lower** at the final stage, on the static view and after the camera path. The denoised plain frame shows SVGF's
1-spp artifacts that the cache arm avoids. The plane identity holds exactly with the cache. Both arms lose 2.5-4% of the mean in the
upscaler (not removed by sharpness 0 or exposure 1; suspected: accumulation in Reinhard space on residual noise, N11.5), and SVGF loses
another 1.4-4% on the plain arm only.

With OIDN in the same chain (same day): OIDN is 3-4× better than SVGF for both arms at 1 spp (plain + OIDN small, 0.027, already beats
cache + SVGF, 0.060). The cache keeps a 1.3-1.4× relMSE lead on single frames (1.6× with albedo-only guides) and 2× less flicker in the
final; once a still camera has 8-16 frames accumulated, both arms meet. A fresh cache is worse than plain through either denoiser until it
has trained: ~5-10 frames with SVGF, ~120-150 with OIDN (N11.6; the raw frame hides it). The progressive mode reaches 0.0073 at 8 still frames with the standard net, then holds the frame at zero cost.

## 7. Open choices (build both, then decide)

* Upscaler-owned jitter vs sampler-owned (§4): needs a uniform camera-offset option in the frame kernel (byte-identical when off).
* Start-up (N11.6 found a young cache looks worse than plain through a denoiser; the training itself is not affected by what is displayed):
  warm-up gate, plain + OIDN on screen while the network trains (`warmupFrames` ≈ 10 SVGF, ≈ 150-200 OIDN; N11.7, the maintainer's
  preferred direction); a ramp instead of a switch (N11.8); a saved network per scene in IndexedDB (N11.9). More steps per frame while
  young only if training stays stable; no learning-rate boost.
* SVGF vs OIDN as the default (§3): built and A/B'd for quality; needs quiet-pass cost and motion with the realtime preset. The Denoiser's
  `recurrent` filter (memory-lean) as a third arm.
* Plain N spp vs plain at the same frame rate with a lower `maxDepth` (the realtime preset uses 2): equal time can be spent either way.
* Guides of the last sample vs the first (N spp): only the plain arm with N > 1 is affected.
* Upscaler sharpness: RCAS 0.8 (default) vs 0: 0 lowered final relMSE and flicker a little in both arms (NRC-PIPELINE.md); one 16-frame run,
  not a decision.
* The upscaler's mean loss (−2.5 … −4%): feed it a noise-free input to separate noise-dependent bias (Reinhard-space accumulation) from a
  plain offset (N11.5); the fix, if any, is the upscaler maintainer's.
