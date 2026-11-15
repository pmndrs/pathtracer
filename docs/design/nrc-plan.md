# Track N plan: a neural radiance cache on WebGPU, start to finish

The one page to track the experiment. Detail lives elsewhere: the design and budget in [nrc.md](nrc.md), every
difference from the paper and VkNRC in [nrc-diverges.md](nrc-diverges.md), task rows in [TASKS.md](../../TASKS.md)
section N. This file says what we are trying to find out, what "done" means, the milestones in order, and where each
one stands. Update the **Progress** column and the log at the bottom at the end of every session.

Shareable progress page (private until shared from its Share menu): https://claude.ai/artifact/DLEmXfvC7ChP8481rioCAx.
Its milestones and log are uploaded copies of §4 and §7: update both at the end of a session.

## 1. The questions

The experiment exists to answer four questions, each with a measured answer:

| # | Question | Answered by | How we will know |
|---|---|---|---|
| Q1 | Can WebGPU run NRC's network fast enough for real time? | M2 | MLP inference + training per frame within the budget in §3, on the M5 Pro and the RTX 3060 |
| Q2 | Does the cache make a better image than plain path tracing in the same time? | M5 | Equal-time relMSE and FLIP vs the Mitsuba reference, plus bias and frame-to-frame stability |
| Q3 | What do subgroup matrices (tensor tiles) buy, per vendor? | M2, M6 | Paired ratios, matrix path vs the best portable kernel, on Apple and NVIDIA |
| Q4 | Does the neural cache beat a plain hash-grid cache (SHaRC-like) in the same time? | M7 | Same equal-time table with a third row |

A "no" is a result too. If Q1 fails at full resolution the demo runs at the upscaler's half resolution; if Q2 or Q4
fails the write-up says so.

## 2. What the user will be able to test (the demo, M4)

`pnpm --filter @pathtracer/bench nrc:live`, then open the printed URL in Chrome. Plain Chrome runs the portable f16
network; Chrome started with `--enable-unsafe-webgpu` adds the subgroup-matrix backend.

- **Split view**: a draggable divider, plain path tracing on one side and path tracing + NRC on the other, at **equal
  time per frame** (the default) or equal samples per pixel.
- **Views**: final image; "cache only" (the network's prediction at the first hit: what the cache has learned);
  relative error against the 4096 spp reference (fixed reference cameras only); training-path coverage.
- **Live numbers**: ms per pass (trace, train, infer, resolve), queries and training records per frame, loss curve,
  relMSE against the reference on demand (button, so no readback in the frame loop).
- **Controls**: scene (sponza-white first, then Cornell and textured Sponza), resolution scale (1, ½), backend,
  network size, training steps per frame, training-path ratio, termination threshold, learning rate, EMA on/off,
  pause training, reset the cache, camera (orbit/fly, a few saved views, a scripted fly-through to watch the cache
  adapt).
- **Save A/B**: writes both halves as linear PFM plus timings, so the bench viewer shows the pair side by side.
- Headless twin: the same page with `?offscreen=1` is a runner page (`nrc`) that produces the M5 tables (not built yet).

**Built (2026-11-15):** everything above except the reference error view, the coverage view and the headless twin.

## 3. Budget (what "fast enough" means)

Target frame: 16.7 ms on the M5 Pro at 960×540 (upscaled to 1080p), path tracing included. VkNRC-shaped net
(64 → 5×64 → 3), one query per pixel, 4 training steps of 16k records per frame.

| Piece | Measured now (M5 Pro, 2026-11-15) | Target |
|---|---|---|
| Inference, 518k queries (960×540) | 4.4 ms (subgroup matrix f16, 4 subgroups); **5.7 ms tiled f16, no flag**; 49 ms naive | ≤ 4 ms |
| Training, 4 × 16k records | 2.0 ms (portable f16, 0.51 ms per step: tiled gradient + tiled forward-backward; was 27.5) | ≤ 4 ms ✅ |
| Path tracing, ~2 short paths per pixel | 640×360: 5.6 ms with training paths (960×540: ~13.5) | rest of the frame |
| **Cache side, 640×360 total** | **9.0 ms** (render 5.6, 2 steps 1.3, inference 2.1; subgroup matrix f16, flagged Chrome) | 16.7 ms ✅ |

Training met its budget on 2026-11-15 (0.51 ms per step); the path tracer's own passes are now the larger part of a frame.

## 4. Milestones

Status: ✅ done · 🟡 in progress · ⬜ not started. Rows in TASKS.md carry the detail.

| # | Milestone | Deliverable | Exit gate | TASKS | Progress |
|---|---|---|---|---|---|
| M0 | Groundwork | Subgroup-matrix probe, adapter caps, paper numbers checked | We know which machines run the matrix path | N0.1–N0.4 | 🟡 probe done (Apple); paper check and Windows open |
| M1 | MLP engine + oracle | `packages/mlp`: forward, backward, Adam, f32/f16, CPU oracle; check, fit, timing and live pages | GPU matches the oracle; image fit converges | N1, N1.1, N1.3, N10 | ✅ engine, correctness, live page, encoding A/B (two-phase waves: fit error 0.26 → 0.16) |
| M2 | Speed | Fast portable kernels (N1.2), subgroup-matrix inference (N3a) and training (N3b), round-trip cost per layer (N3) | §3 budget met on the M5 Pro, or the gap is measured | N1.2, N2, N3, N3a–c | ✅ inference (sgm 8.1, tiled 11.0 ms per million queries, no flag needed for tiled); training ✅ (0.51 ms per step, tiled); sgm forward-backward built (N3b: faster in f32 only on Apple) |
| M3 | NRC on our path tracer | A cache hook for main's integrator (`ptOnVertex` / `ptTerminate`, `updateBlock` training paths): query and training buffers, target back-propagation, inference, resolve into the accumulation | Closed-room check: the learned cache converges to the analytic indirect radiance `ρE/(1−ρ)`; megakernel = wavefront | N6a | ✅ megakernel (room gate: 0.06-0.55% mean error, all backends); wavefront open |
| M4 | Demo page | `nrc-live` (§2), user-testable | The maintainer can run it and see both halves | N6b | 🟡 built and running (`nrc:live`); try it in flagged Chrome |
| M5 | Equal-time study | `nrc` runner page; tables on sponza-white: relMSE, FLIP, bias, stability vs plain PT at equal ms | Q2 answered on the M5 Pro | N6c | ✅ (windowed): warm cache 3.2-4.3× lower relMSE at equal frames, 5-7× per unit time; steady from frame 1 after a cut, even into an unseen corridor ([NRC-STUDY.md](../../bench/results/NRC-STUDY.md)) |
| M6 | Windows / NVIDIA | Matrix path on the RTX 3060 (Dawn Vulkan vs D3D12), N2 table there, VkNRC native baseline | Q3 answered for NVIDIA; native vs web ratio | N0.3, N6.1, N6.2 | ⬜ handed off: [nrc-windows-handoff.md](nrc-windows-handoff.md) (WN1-WN7, branch `windows-nrc`) |
| M7 | Hash-cache competitor | SHaRC-like non-neural cache at equal time | Q4 answered | N7 | ⬜ |
| M8 | Write-up | `bench/results/NRC.md`, viewer report; gpuweb feedback (maintainer decides) | Maintainer review | N8, N9 | ⬜ |

Order: M2 and M3 can run in parallel (M3 starts on the portable backend); M4 needs M3; M5 needs M4's runner twin.
M6 needs the maintainer's Windows laptop and can happen any time after M2.

## 5. How M3 fits main's path tracer

Main's integrator already has the hooks a cache needs (docs/design/integrator.md, "Hooks, counters and AOVs"):

- `ptTerminate(v) -> vec3f` ends the path and adds `throughput × value`; the cache supplies the **indirect**
  radiance (the vertex's own emission and NEE are already in). NRC cannot evaluate the network inside the path, so the
  hook returns 0 and appends a query record (encoded vertex, pixel, throughput); after inference a resolve pass adds
  `throughput × prediction` into the pixel's accumulation. Query-record order follows pixels, so the add is a gather,
  no atomics.
- `updateBlock: B` marks one pixel per B×B block per sample as a training path. Those paths run to their natural end
  and append `(encoded vertex, throughput, emitted)` per vertex; a small pass sums `emitted` backwards along each path
  to build per-vertex targets (main has no path-end callback, V84). The tail vertex queries the cache too
  (self-training, N-I5).
- Storage budget: the megakernel leaves 4 buffers for a hook, the wavefront's shade kernel 2. The cache needs a query
  buffer and a training buffer (counts in their headers), so it fits both schedulers.
- First scene: sponza-white (diffuse, so the reflectance inputs are constants); first correctness check: the closed
  emissive room from `pt-outputs-check`, where the true indirect radiance is known in closed form.

## 6. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Training stays too slow without subgroup-matrix training | Fewer steps per frame, a staler cache | N3b; fewer records (8k) or 2 steps; measure what quality that costs |
| Subgroup matrices stay behind a Chrome flag | Normal users get the portable backend only | The demo works on portable f16 at ½ resolution; the write-up reports both |
| The cache's bias is visible (NRC is biased) | Q2 may be "faster but wrong" | Report bias separately; unbiased training-path fraction is a setting (N-I3) |
| The image fit never learns fine detail | Hint that the encoding is wrong for NRC too | N1.3 encoding A/B now; hash grid (N4) later |
| Shared Mac load | Timings that can't be quoted | Runner load check; quiet pass before quoting; paired ratios only |

## 7. Progress log

Newest first. One line per session: what changed, what it showed.

- **2026-11-15 (Mac, session 10).** Tiled portable inference (N1.4) is the default: 8.6× the naive kernel in f16, within 1.36× of
  subgroup matrices. In plain Chrome (no flag) the 960×540 cache frame went from 64 ms to 21 ms (flagged: 19 ms).
- **2026-11-15 (Mac, session 9).** Subgroup-matrix training (forward-backward) built and verified: 1.55× faster than the tiled kernel
  in f32, 10-25% slower in f16. On Apple, well-tiled plain WGSL is about as fast as the matrix path; NVIDIA is the real test (M6).
  Next: tiled portable inference (N1.4) for users without the flag.
- **2026-11-15 (Mac, session 8).** Fresh-cache damping (N6f): zero output layer + warm-up EMA are the defaults now; a fresh cache's
  first frame is at the steady error (was 2-3× worse with +77% bias), and steady-state bias fell to +0.1-0.5%.
- **2026-11-15 (Mac, session 7).** Unseen-region test: a cache trained only on the ground floor is at its steady error in the
  upper-gallery corridor from the first frame after a cut. Found and fixed: training tiles showed as noisy columns in the image
  (now display query + fix-up), plus a query-buffer overflow. Final windowed numbers: warm error 3.2-4.3× lower than plain at
  equal frames, 5-7× per unit time; a fresh cache overshoots on its first 1-2 frames (N6f).
- **2026-11-15 (Mac, session 6).** Windowed Chrome is the timing reference now (headless differences backlogged, N6e). There a
  cache frame costs less than one plain sample (0.58× at 720p); warm error 3.1-3.8× lower at equal frames, 5-6× per unit time.
  Adaptation: a cache trained elsewhere in the atrium is at steady state on the first frame after a cut; a fresh one in ~8 frames.
- **2026-11-15 (Mac, session 5).** Equal-time study on Sponza (M5): with a warm cache the cache side has 3.0-3.8× lower relMSE than
  plain path tracing at 1280×720 and 1.3-1.8× at 640×360, bias < 0.4% after 4 frames (720p), a third less flicker. Fixed a
  training-tile rotation bug the study exposed. Timing in the headless runner disagrees with the windowed demo: open.
- **2026-11-15 (Mac, session 4).** Training step 6.9 → 0.51 ms (tiled forward-backward). Opt-in `ptOnRadiance` hook in main's
  integrator (other kernels byte-identical); training paths now ride in the render pass as whole 8 × 8 tiles: the cache side of a
  640×360 Sponza frame is 9.0 ms. Demo runs in flagged Chrome.
- **2026-11-15 (Mac, session 3).** The cache runs on main's path tracer (`bench/nrc/cache.ts`): passes the closed-room gate
  on every backend; `nrc-live` demo built (`pnpm --filter @pathtracer/bench nrc:live`). On the portable network inference is
  the big per-frame cost (59 ms for 500k queries), so the subgroup-matrix backend matters for the demo. Progress page published.
- **2026-11-15 (Mac, session 2).** Merged main's 41 local commits (path tracer P1–P3 with hooks); track N IDs
  renumbered (D28/D29, V139–V142, W11). Emulated subgroup-matrix kernel verified on the Apple GPU (N3c). Denoiser
  issue pmndrs/denoiser#31 filed for `subgroup_id` offsets. Quiet timing rerun (matrix path 11.2× portable f16). Image
  encoding A/B: two-phase waves at Nyquist, fit error 0.26 → 0.16. Tiled weight gradient: training step 6.9 → 2.7 ms
  (f16). ReLU round trip measured at 12-24% of matrix-path inference. This plan written.
- **2026-11-15 (Mac, session 1).** Native subgroup-matrix inference compiles in Chrome 157 and matches the oracle;
  4 subgroups via `subgroup_id` compile with no override. Matrix path 7.5–11× faster than our naive portable f16;
  training is now the larger cost. lr sweep on the image fit (0.01 best). The old emulated kernel hung the GPU.
- **2026-11-14 (cloud).** Spec, divergence register, MLP engine with oracle, pages; verified on SwiftShader only.
