# Handoff: denoiser edges and history under per-frame jitter (N11.14-N11.18)

Branch `claude/neural-demo-refinements-6505a9` (pathtracer, head 0b878cc) and Denoiser branch `fix/jitter-edge-history`
(worktree `~/Developer/Denoiser/.claude/worktrees/chain-edge-history`, head 4f05e6c). Nothing is pushed. TASKS.md N11.14-N11.18
holds every number; the Denoiser write-up is `results/JITTER-EDGE-HISTORY.md` there.

## The problem

Real-time chain: path tracer 1 spp at half display resolution -> SVGF (Denoiser) -> temporal upscaler (pmndrs/upscaler core) -> display.
Edges and thin lines stay noisy and their SVGF history keeps resetting, also with a still camera. Jitter must stay: the temporal
upscaler needs it for anti-aliasing and super-resolution.

## What is settled

- Guides (depth, normal, albedo, roughness) come from the path tracer's primary vertex. With jitter off they are bit-identical
  frame to frame. With jitter on, interiors change only through texture aliasing (fixed: cone texture LOD, start = one display pixel,
  now the session default), and edges change surface because the sample lands on another surface. That last part is inherent.
- With jitter off, SVGF works (still history all red) but the image aliases and the upscaler cannot super-resolve.
- Jitter has one owner per chain (`PipelineSettings.jitter`: 'sampler' per pixel, 'upscaler' per-frame sequence, 'centre').
- The Denoiser's own demos never fed it per-frame jittered input: they use an unjittered raster G-buffer, `jitter: [0, 0]`, and a
  spatial (FSR1) upscaler (`examples/three-pathtracer-webgpu/src/main.ts`, `examples/three-temporal/src/realtime-scenes.ts`).
  Jittered input + SVGF + a temporal upscaler is new ground for that library.

## What was tried (bench options; defaults unchanged)

| Option | Effect |
|---|---|
| `antiLagRadius: 0` (no max over 3x3 strata) | still history p50 36 -> 62, flicker -12% still; lamp lag unchanged; ghosting +10% |
| exact gradient anti-lag (`bench/realtime/gradient-samples.ts`, replays the previous frame) | flicker -19% still, lamp in 2 frames; ghosting +30-60%; costs a full extra frame |
| `historyFootprint` (Denoiser) | -11% flicker, +12% error: borrows the other surface's shading |
| `secondHistory` (Denoiser) | -6-8% error on lost edge pixels, +112 B/px; flicker unchanged |

## Open questions for a fresh look

1. Why does the exact anti-lag ghost more? Probably the 'auto' anti-lag was also shortening histories that were wrong after
   reprojection; with exact gradients they live 64 frames. Measure history correctness under motion separately from lighting change.
2. An edge pixel at 1 spp with jitter is a mixture of two surfaces; SVGF stores one signal per pixel. Candidates: a per-surface
   history keyed on the guide (secondHistory is a start), denoising with a sub-pixel-aware reprojection, or a combined
   denoise + upscale temporal pass (the idea behind DLSS Ray Reconstruction).
3. Is the spatial (a-trous) pass isolating edge pixels from both sides? Check edge weights at the 25% of pixels whose surface flips.

## Tools

- `node bench/live.ts chain-debug` -> http://127.0.0.1:5291/bench/pages/chain-debug.html (next free port if taken).
- `__chainDebug`: `measureLag`, `sharpness`, `guideChange`, `edgeHistory({ trace })`, `edgeQuality`; drive frames yourself, the
  app's browser pane fires rAF about once a second.
- The bench links the main Denoiser checkout again (2026-11-27). To try the branch options (`historyFootprint`, `secondHistory`) point it at
  the worktree: `ln -sfn ~/Developer/Denoiser/.claude/worktrees/chain-edge-history/packages/denoiser bench/node_modules/denoiser`;
  restore with `ln -sfn ../../../Denoiser/packages/denoiser bench/node_modules/denoiser`.
- Status: shipped as a known issue (TASKS N11.19, tracked in pmndrs/denoiser#51); the showcase runs with accumulate while still, `antiLagRadius: 0` and
  `antiLagHistoryScale` on.
