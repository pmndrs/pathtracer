# A neural radiance cache on WebGPU: what we found (track N, M8)

Research track N asked whether NRC (Müller et al. 2021: a small network that learns the light in a scene while it renders, so paths can
stop after one or two bounces) can run in the browser, and whether it is worth it. This is the summary; the detail is in the pages it
links. The plan and its four questions: [nrc-plan.md](../../docs/internal/design/nrc-plan.md). Every departure from the paper and from VkNRC (the
native implementation we follow): [nrc-diverges.md](../../docs/internal/design/nrc-diverges.md).

**Images:** `node bench/nrc/nrc-report.ts`, then `pnpm --filter @pathtracer/bench viewer` and open `?report=/results/reports/nrc/report.json`:
plain path tracing, the network and the hash grid side by side on ten views (sponza-white, sponza-glossy, the denoiser chain), with the
references. It copies the runs' PFMs, which are git-ignored: it only works on the machine that ran them (the RTX 3060 laptop).

**Status of the numbers:** quality numbers are deterministic for a seed and final. Timings are best effort (shared Mac, a laptop GPU that
runs at 80-87 °C under load, other apps open); the ones that would decide a default are queued for the maintainer's quiet pass
([docs/internal/QUIET-PASS.md](../../docs/internal/QUIET-PASS.md): N2r-e, N11.1). Ratios are paired on one machine, never across machines.

## The four answers

| | Question | Answer |
|---|---|---|
| Q1 | Can WebGPU run NRC's network fast enough for real time? | **Yes on the M5 Pro**: inference 4.4 ms for 518k queries (960 × 540, subgroup matrix f16; 5.7 ms with the portable tiled kernel, no flag), training 0.51 ms per 16k-record step; the cache side of a 640 × 360 frame is 9.0 ms including path tracing. **On the RTX 3060 in Chrome, the network is affordable but not cheap**: 18 ms of inference + 5 ms of training in a 62 ms 720p frame, because Chrome on Windows has no subgroup matrices (D3D12). The same kernels on Dawn/Vulkan reach the tensor cores: 3.85 ms per million queries, 1.6× VkNRC native. The paper: 1.59 ms query + 1.11 ms training per 1080p frame on an RTX 3090 (CUDA, tensor cores). |
| Q2 | Does the cache make a better image than plain path tracing in the same time? | **Yes.** sponza-white, warm cache: 3.2-4.3× lower relMSE than plain at equal frames on the M5 Pro (5-7× per unit of time, a cache frame costs 0.61 of a plain sample), 3.7-4.4× on the RTX 3060 (where a cache frame costs 1.03 plain samples, so the lead per unit of time is the same). Steady from the first frame after a camera cut, even into a corridor it never saw. Through a real-time chain (denoiser + 2× upscaler, M9) the lead shrinks but stays: 1.6-2.2× after SVGF, 1.3-1.4× after OIDN on single frames, 2× less flicker. |
| Q3 | What do subgroup matrices buy, per vendor? | **Apple (M5 Pro, Chrome 157):** inference 1.3-1.4× over the best portable kernel (8.1 vs 11.0 ms per million queries); training gains only in f32 (1.55×), f16 is 10-25% slower (Apple offers f16 → f16 only, 8×8×8 tiles). **NVIDIA (RTX 3060):** nothing in Chrome (no subgroup matrices on D3D12); on Dawn/Vulkan 4.4× over Chrome's best portable kernel for inference (3.85 vs 17 ms per million) and 1.3× for training (0.61 vs 0.82 ms per step, paired) once the kernels use a tensor-core layout (the first straightforward port: 13.9 ms, 1.67× over the tiled kernel on the same backend). Table below. |
| Q4 | Does the network beat a plain hash-grid cache (SHaRC-like) in the same time? | **On a diffuse scene, no**: a world-space hash grid fed by the same training paths is within 2-4% of the network per frame and costs 0.76 vs 1.04 of a plain sample (1.37× the network's efficiency in Chrome on the 3060); through the denoiser chain it is better than the network after SVGF and OIDN, and warms up in 10 frames instead of ~150. **On a glossy scene, yes once the image converges**: per frame they still tie, but after 256 frames the network has half the grid's error and a sharper image (the grid stores one value per cell and blurs reflections). The network also generalises better: 11-13% ahead on a fresh cache and in an unseen region. |

## What subgroup matrices buy (Q3)

ms per million inference queries and per 16k-record training step, the VkNRC-shaped network (64 → 5 × 64 → 3), f16 unless noted.

| | Apple M5 Pro, Chrome 157 (Metal) | RTX 3060 Laptop, Chrome 157 (D3D12) | RTX 3060 Laptop, Dawn/Vulkan (dawn.node) | VkNRC native (Vulkan, NV cooperative matrix) |
|---|---|---|---|---|
| Subgroup matrices offered | f32→f32, f16→f16, 8×8×8 | none | f16 (f16 and f32 sums) and int8, 16×16×16 | NV extension, 16×16×16 |
| Inference, portable (best) | 11.0 (tiled) | 17 (per record; tiled 23.3) | — | — |
| Inference, subgroup matrix | 8.1 (4 subgroups) | — | **3.85** (blocked layout; first port 13.9) | **2.4** |
| Training step, portable | 0.51 (tiled) | — | 0.82 (tiled, paired) | — |
| Training step, subgroup matrix | f32 1.55× faster, f16 10-25% slower | — | **0.61** (0.54 with the global ReLU mask) | **0.27** |
| Peak reached | — | — | 42 TFLOP/s f16 sums (`sgm-roofline`) | — |

What made the NVIDIA kernels fast was layout, not the API: weights held in registers, f16 sums, padded rows, four subgroups per
workgroup, and loops with a constant trip count (Tint adds a 64-bit counter to every loop it cannot prove finite; on D3D12 those cost 3-4×
inside a big kernel). The remaining 1.6× (inference) and 2× (training) to VkNRC is mostly a ReLU round trip through memory that WGSL forces
(no per-element access to matrix tiles), and f32 gradient sums without float atomics. The draft feedback for the subgroup-matrix proposal is
[docs/internal/design/subgroup-matrix-feedback.md](../../docs/internal/design/subgroup-matrix-feedback.md) (posting it is the maintainer's call, N9).
Running the matrix kernels on Windows: [docs/internal/design/dawn-node-vulkan.md](../../docs/internal/design/dawn-node-vulkan.md).

## Web vs native (RTX 3060, 720p)

| ms per frame | VkNRC (native) | Ours, Chrome (portable network) | Ours, Dawn/Vulkan network (estimate) |
|---|---|---|---|
| Path tracing (cache side) | 1.6 (hardware ray queries, V-buffer) | 38 (software BVH in WGSL) | 38 |
| Inference | 2.2 | 18 | ~4 |
| Training (4 steps) | 1.07 | 5 | ~2.2 |
| Frame | 5.1-5.3 | 62 | ~45 (estimate) |

The network is no longer what separates us from native: software ray tracing is (24× the hardware path). Our path tracer was 3.3× slower
on NVIDIA than it should be until the constant-trip-count fix (P-W1: Sponza 191 → 58-61 ms per sample, level with the M5 Pro).

## The cache in practice

* **Quality** ([NRC-STUDY.md](NRC-STUDY.md)): error 3-4× below plain path tracing per frame, bias 0.1-0.5%, flicker −38%, steady from the
  first frame after a cut. A fresh network predicts nothing at first (the output layer starts at 0, N6f) and needs a few frames to catch up.
* **Through a denoiser** ([NRC-PIPELINE.md](NRC-PIPELINE.md)): the denoiser removes plain path tracing's noise better than it removes the
  cache's smooth error, so the lead shrinks (OIDN: 1.3-1.4× on single frames, about 1× once a still camera has accumulated 16 frames), while
  flicker stays 2× lower. A young network is worse than none through OIDN for about 150 frames; a warm-up gate shows plain path tracing until
  it has trained enough (D30: 16 steps per gated frame, so 38 frames).
* **Against a hash grid** ([nrc-hash-cache.md](../../docs/internal/design/nrc-hash-cache.md)): see Q4. The grid's weakness was blotches from young
  cells holding one noisy sample (fixed by writing every sample into coarser cells too, plus a sample threshold) and view-independent cells
  on glossy surfaces (not fixed by splitting cells by direction). Its strengths are cost, warm-up and low-frequency accuracy where the
  light does not depend on the view.

## Where to start

| | |
|---|---|
| Demo, one cache vs plain | `pnpm --filter @pathtracer/bench nrc:live` (split view, equal time, Cache menu: network backends or hash grid) |
| Demo, through denoiser + upscaler | `nrc-pipeline` page (SVGF / OIDN, warm-up gate, hash grid option) |
| Studies (headless twins) | `node bench/runner/run.ts --page nrc-study` / `nrc-pipeline-study` (`model=hash`, `scene=sponza-glossy`, `ref=ours`) |
| Correctness | `--page nrc-check` (closed room, analytic answer), `mlp-check` / `mlp-fit` (network vs CPU oracle) |
| Code | `packages/mlp` (the network: forward, backward, Adam; portable and subgroup-matrix kernels), `bench/nrc/cache.ts` (the cache on the path tracer's hooks), `bench/nrc/hash-grid.ts`, `bench/nrc/pipeline*.ts` |

## What is not settled

* Timings for the quiet pass (N2r-e, N11.1); the Mac's side of the hash-grid comparison (N7b) and of the denoiser chain against Mitsuba
  (N11.4).
* One scene family (Sponza; diffuse and a glossy variant). Textured and transmissive scenes, and the paper's reflectance factorisation
  (N-D4), are not tested.
* Our render paths stop at a fixed vertex, not by the paper's area-spread rule (N-I4), and no training path is left unbiased (N-I3:
  the paper uses 1 in 16).
