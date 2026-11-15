# NRC divergences: where we differ from the paper and VkNRC, and what to do about it

Research track N's own divergence register ([nrc.md](nrc.md), TASKS section N). Each row says what a source does, what
we do (or plan to), why, and the follow-up that would close or confirm it. Follow-ups are the work list: a row is
closed when its A/B has run and the result is recorded here, or when the difference is accepted for good. The repo-wide
[DIVERGES.md](../../DIVERGES.md) keeps one pointer row (V142) to this file plus the engine-level rows V139–V141.

Sources:
- **Paper:** Müller, Rousselle, Novák, Keller, "Real-time Neural Radiance Caching for Path Tracing", SIGGRAPH 2021
  (arXiv 2106.12372). Paper facts are **[M] from memory** until N0.1 checks them against the PDF on the Mac.
- **VkNRC:** AdamYuan/VkNRC `a667859`, read from its source [V]. No licence: we follow its design, never its code (D29).
- **Denoiser:** pmndrs/denoiser `7014384`, `packages/denoiser-wgsl` + `docs/specs/runtimes.md` [V], MIT, ours.
- **tcnn:** NVlabs/tiny-cuda-nn, BSD-3 [V licence]; behaviour [M].

Why: **platform** (a WebGPU/WGSL limit), **perf**, **licence**, **simplification** (do it properly later), **design**
(a deliberate choice to A/B), **scene** (test-setup difference, not an algorithm one).
Status: **planned** (not built yet), **built** (in `packages/mlp` or a bench page), **closed** (A/B done or accepted).

## A. Network and training

| ID | Topic | Paper [M] | VkNRC [V] | Ours | Why | Status | Follow-up |
|---|---|---|---|---|---|---|---|
| N-D1 | Net shape | ~64 wide, ~5 hidden, ReLU, fully fused | 64 → 5 × 64 → 3, ReLU, no bias | Same as VkNRC (`DEFAULT_SHAPE`) | design | built | N0.1 confirm paper shape; N2 sweep 64 × {2, 4, 5} |
| N-D2 | Position encoding | frequency encoding (sin/cos) | triangle wave, 12 octaves per axis, scene normalised to [−1, 1] | Triangle wave, 12 octaves (`encoding.ts`); scene normalisation planned for N6 | perf (no transcendental) | built | Image fit (N1.3, 2026-11-15): a second, quarter-period-shifted triangle per octave (sin/cos style) cut relMSE 0.26 → 0.15 and octaves past Nyquist cost quality at low lr. A/B the two-phase layout for NRC positions in N6 (3 × 12 × 2 = 72 inputs: needs a wider input layer or fewer octaves); triangle vs sin; hash grid is N4 |
| N-D3 | Other inputs | one-blob for direction, normal, roughness | one-blob 4 bins, quartic kernel; reflectances raw; 2 constant ones | Same layout (`encodeNrc`) | design | built | A/B Gaussian vs quartic kernel only if N6 shows banding |
| N-D4 | Output factorisation | prediction is radiance ÷ (diffuse + specular reflectance), multiplied back at the query | none: radiance predicted directly | VkNRC's (none) first | design | planned | A/B on N6: factorised vs not, relMSE and texture detail on a textured scene |
| N-D5 | Loss | relative L2 | relative L2 on **luminance** of the prediction: `2(p − t)/(lum(p)² + 0.01)`, scale 1 | Same as VkNRC; prediction clamped ≥ 0 inside lum only | design | built | A/B per-channel vs luminance denominator in N6 (colour shifts in saturated light?) |
| N-D6 | Optimiser | Adam [M lr] | Adam lr 0.002, β 0.9/0.999, ε 1e-8, gradient ÷ batch, NaN/inf → 0 | Same | design | built | Image fit (2026-11-15, f16, 1024 steps): relMSE 0.84 at 0.002, 0.32 at 0.005, 0.26 at 0.01, 0.32 at 0.02. Keep VkNRC's 0.002 for NRC until N6 sweeps it (online training is a different regime); the image task wants ~0.01 (N1.3) |
| N-D7 | Inference weights | EMA of weights | EMA optional (off by default), debiased form | Warm-up EMA on by default in the NRC cache (decay min(0.99, (1+t)/(10+t)), TensorFlow's num_updates form, no bias correction needed); plain EMA from the initial weights (V140) and off kept as settings | design | built, A/B'd | N6f (NRC-STUDY.md): with a zero output layer it removed a fresh cache's overshoot, and steady-state bias fell to +0.1-0.5% |
| N-D8 | Batch | ~4 steps × 16k records per frame | 4 batches × 16,384, trained in sequence; record → random batch | Engine supports it; the N6 page will mirror it | design | planned | N6 sweep steps/frame and records/frame |
| N-D9 | Weight init | [M] | He normal σ = √(2/64) for every layer | He normal σ = √(2/fan_in) (equal here, fan_in = 64) | design | built | none |
| N-D10 | f16 loss scaling | tcnn scales the loss (≈128 [M]) for f16 gradients | scale 1, gradients summed then ÷ batch in the optimiser | Deltas stored without ÷ N, gradient pass divides (V139): keeps f16 deltas out of subnormals | platform / design | built | N1.1: check f16 delta underflow counts on the Mac; add a loss scale if any |
| N-D11 | Inference output clamp | [M] | `max(prediction, 0)` before use | Not clamped in the engine; the resolve hook must clamp | simplification | planned | Clamp in the N6 resolve hook; record if it biases dark regions |

## B. Kernels (how the MLP runs)

| ID | Topic | VkNRC / tcnn | Ours | Why | Status | Follow-up |
|---|---|---|---|---|---|---|
| N-K1 | Matrix hardware | NV cooperative matrix, 16×16×16 f16 tiles | Portable arms first (`wgsl-f32`, `wgsl-f16`); `wgsl-sgm` from the adapter's configs (Apple 8×8×8) | platform | built (portable; sgm inference N3a verified natively on M5 Pro 2026-11-15: 8×8×8, f32 and f16) / planned (sgm training N3b) | N3b; Windows 16³ configs (N0.3) |
| N-K2 | Records per workgroup | 128 records, 4 subgroups of 32, weights shared in shared memory | Portable: 64 records, one thread each, one layer staged in workgroup memory. Sgm: 1 or N subgroups, rows split by `subgroup_id` (built; emulated form verified) | platform (Denoiser: >1 subgroup failed uniformity) | built, verified (M5 Pro, Chrome 157: `subgroup_id` offsets compile without a diagnostic override, 4 subgroups 1.5× faster than 1) | try 128 records × 4 subgroups as VkNRC (needs 32 KiB: available on Apple) |
| N-K3 | Activation (ReLU) | per-element on accumulator tiles | Portable: in registers. Sgm: store tile → per-lane ReLU in workgroup memory → reload | platform (no element access yet, gpuweb #11077) | built (sgm inference, verified natively) | Measured (N3, M5 Pro f16): the ReLU pass is 20-24% of inference with 1 subgroup, 12-13% with 4. Drop it if #11077 ships |
| N-K4 | ReLU derivative | forward activations stored as 0/NaN mask, backward MMA accumulates onto it, NaN → 0 | Explicit test `a > 0` (portable); 0/1 mask multiply in workgroup memory (sgm) | platform (WGSL makes no NaN guarantees) | built / planned | none unless #11077 lands |
| N-K5 | Layer chaining | result tile used directly as next layer's operand | Sgm: store result, reload as left operand | platform (no result → left conversion yet, gpuweb #11079) | built (sgm inference) | N3; drop if conversion ships |
| N-K6 | Weight gradients | per-subgroup dW tiles, summed in shared memory, f32 `atomicAdd` to global | Separate per-weight pass over stored activations and deltas: deterministic, no atomics (V139) | platform (no float atomics) | built: `perWeight` and `tiled` (split-K partials + a reduction dispatch, default; 16-22× faster pass, N1.2) | fixed-point i32 `atomicAdd` arm; why tiled fits end slightly worse at 1024 steps (TASKS N1.2) |
| N-K7 | Accumulation precision | f16 accumulate (VkNRC); Denoiser keeps f32 accumulation | Portable f16: vec4 dot in f16, row sum in f32. Sgm: f16 → f16 on Apple (only config), f16 → f32 where offered | platform / design | built / planned | A/B accumulate f16 vs f32 where both exist (Windows) |
| N-K8 | Subgroup size | fixed 32, full subgroups required | `@subgroup_size(32)` where `subgroup-size-control` exists (Chrome 151+); else tiles from the adapter's size | platform | planned (Apple: closed, size is fixed at 32) | N0.2 on Windows/NVIDIA, AMD (32/64) |
| N-K9 | Workgroup memory | 32 KiB for backward | Portable: 16 KiB (f32 layer) / 8 KiB (f16); request the adapter max | platform | built | none |
| N-K10 | Loop structure | GLSL `[[unroll]]` over tile arrays | Host-generated unrolled WGSL (like the Denoiser) | platform (no `@unroll`, gpuweb #11073) | built (portable) | switch to `@unroll` if it ships |
| N-K11 | Runtime shape | standalone kernels | Standalone `MlpEngine`, not behind the Denoiser's `NetworkRuntime` (V141) | simplification | built | N5 adds `ort` / `webnn` arms; revisit a shared interface |

## C. Integrator and scene (N6; the cache runs on main's megakernel since 2026-11-15)

| ID | Topic | Paper [M] | VkNRC [V] | Ours (plan) | Why | Follow-up |
|---|---|---|---|---|---|---|
| N-I1 | Ray tracing | hardware RT | Vulkan ray queries, V-buffer raster for primaries | Software BVH (trace core, `rtClosest`), primary rays traced | platform (no RT in WebGPU) | report trace vs MLP time separately |
| N-I2 | Training-path selection | random pixel per screen tile | per **subgroup**: whole subgroup trains with p = 0.03 | Built: whole 8 × 8 tiles of the render pass (one megakernel workgroup; every 16th tile, rotating), close to VkNRC's per-subgroup choice; per-pixel and the separate trainer film kept as settings. Per-pixel selection cost 2 ms more per frame (lanes wait on the training path) | platform / design | A/B per-tile vs per-subgroup: spatial coverage vs speed |
| N-I3 | Unbiased training paths | a small fraction of training paths are not terminated in the cache [M] | none | VkNRC's (none) first | design | A/B with 1/16 unbiased paths: bias under strong indirect light |
| N-I4 | Termination | area-spread heuristic, c = 0.01 | same c; `a0 = d²/(4π cos)`, accumulates `√(d²/(pdf·cos))` | Same as VkNRC | design | sweep c in N6 |
| N-I5 | Self-training | tail of training path queries the cache | inference pass writes the prediction into training records before training | Same: inference before training each frame | design | none |
| N-I6 | Materials | full BSDFs | Cook-Torrance (diffuse + specular) | Lambert (sponza-white) first | scene | Cook-Torrance once P1.3/P1.4 land |
| N-I7 | Lighting | scenes with emitters and env | constant white sky (radiance 10), emissive materials, no NEE | sponza-white sun + sky with NEE; a "VkNRC lighting" setting for like-for-like images (N6.2) | scene | compare both settings against VkNRC native (N6.1) |
| N-I8 | Resolution | 1080p | window size | full and upscaler ratio 2 | perf (budget, nrc.md §2) | report both |
| N-I9 | Training targets | radiance gathered along the training path [M] | training paths in the main pass, targets back-propagated along the path | **Inline (default):** training paths are part of the render pass; the opt-in `ptOnRadiance` hook sums each path's light per vertex, target_k = (Σ_{j>k} gathered_j + β_M·cache(x_M)) / β_k. **Replay** (kept as a setting): a separate trainer film replayed M + 1 times | none left for the megakernel (the hook was added to main's integrator, opt-in) | Sponza 640×360: inline tiles 9.0 ms vs replays 11.5 ms per frame on the cache side, same room accuracy. The hook exists on both schedulers; the cache runs on the megakernel |
| N-I10 | Roughness input | per-vertex roughness | material roughness | 1.0 for every vertex (the hook vertex has no roughness yet) | simplification | exact for sponza-white (all Lambert); read `rtMaterial(v.material)` roughness before glossy scenes |

## Follow-up queue (by when it can run)

- **Mac, no Windows:** N0.1 (paper check: closes every [M]), N0.2 (subgroup-matrix configs + the `subgroup_id`
  uniformity probe, N-K2), N1.1 (f16 underflow, N-D10), N1.2 (tiled gradient pass, N-K6), N3 (round-trip costs,
  N-K3/K5), N4 (hash grid, N-D2), lr and EMA sweeps (N-D6/D7).
- **Needs N6 (integrator):** N-D4, N-D5, N-D8, N-D11, N-I2, N-I3, N-I4.
- **Windows:** N-K7 accumulate A/B, VkNRC native comparison (N-I7, N6.1).
- **Waiting on gpuweb:** N-K3 (#11077), N-K5 (#11079), N-K10 (#11073); see
  [subgroup-matrix-feedback.md](subgroup-matrix-feedback.md).
