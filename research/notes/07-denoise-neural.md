# 07 - Reconstruction: denoising, temporal accumulation, upscaling, neural rendering (web-feasible)

Date: 2026-11-13. Confidence tags: [V] verified from a fetched page / local code, [U] unverified (search snippet, recollection, or third-party recap).

## 1. TL;DR

- **mutracer's "blurry but stable in motion" is not temporal reprojection.** [V] Its `RenderLoop` swaps to a *separate half-res preview Render* (adaptive scale 0.125-0.5, `docs/progressive-rendering.md` L185-190) during interaction, 1-2 spp, and runs **a spatial denoiser on every preview frame** (JNLM or OIDN; docs say OIDN is "cleaner at one or two samples per pixel, for example for denoising while the camera moves", `docs/denoising.md`). Stable = low-pass of a fresh independent frame + bilinear upscale. The `motion` AOV exists but is a debug/aux output (`docs/api.md` L380), not fed to a history filter. So its stability is spatial blur + resolution drop; it flickers at the edge of what a per-frame denoiser can hide.
- **OIDN 3 is still not released** [V-ish]. Latest GitHub release is 2.5.1 (18 Aug 2026; 2.5.0 = XMX/AMX-FP16 perf, 2.4.0 = AMX-FP16, RDNA3.5). CG Channel (Jan 2026): OIDN 3 = temporal denoising; U-Net now **kernel-prediction** instead of direct prediction; separate real-time and final-quality variants; real-time inputs = previous denoised frame + backward motion vectors + depth; final-quality also uses future frames + forward MVs; kernels reusable across AOVs; supersampling supported; Apache-2.0. **Schedule [V 2026-11-13, GitHub milestones RenderKit/oidn]: milestone "3.0 Alpha" due 2026-11-30 (5 issues, all spatial: viewport/lightmap/final-frame spatial filters #294-296, composable AOV #297, and #298 viewport temporal + upscaler **Intel GPUs only**); "3.0 Release" due 2027-03-31 (#299 viewport temporal + upscaler, all GPUs + AMX-FP16 CPUs); "3.1 Release" due 2027-06-30 (#300 final-frame temporal filter). So 2026-11-30 is an alpha, not the release, and the temporal filter is not in the first alpha for non-Intel GPUs.** A BCON26 recap [U] lists albedo split diffuse/specular + roughness + depth as aux and "final Q1 2027" (consistent with the 2027-03-31 milestone). No weights/sizes/timings public. Our `TemporalDenoiser.BeautyInputs` slot is right; plan inputs as `{color(demod), albedo_diffuse, albedo_spec, roughness, normal, depth, motion, prevOutput}`.
- **Kernel-prediction + recurrent history is the shape of the future** (OIDN 3, KPCN, I3D25 AMD joint denoise-upscale). It is also the *web-friendly* shape: network outputs per-pixel filter weights applied to *demodulated* noisy colour, so f16 error and over-blur are bounded, and the same machinery maps to upscaling.
- **FSR Ray Regeneration (ML, RDNA4, DX12 only, v1.1-1.2 in FSR SDK 2.2/2.3) and DLSS RR (transformer, DLSS4) are closed binaries** [V/U]; Denoiser repo already records Ray Regen's licence as non-portable. Public design info = I3D25 paper (AMD, joint multi-branch multi-scale), NVIDIA RR blog posts. An open joint denoiser-upscaler is feasible as our own network; no weights to borrow except OIDN.
- **Fastest real win for us: a hand-tuned, variance-guided, demodulated diffuse/specular temporal+spatial filter (SVGF/A-SVGF lineage)** feeding the existing upscaler, because it runs in <1 ms/1080p-class and needs no model. OIDN stays the "converged/preview-quality" path. Neural models = later.
- **Neural rendering ratings:** NRC = Prototype (CPU-free, but needs fused MLP; WGSL has subgroups only, no cooperative matrix); NTC = Prototype (decode-on-sample too slow without matrix HW; transcode-to-BCn at load is feasible and fits gputex); neural importance sampling = Reference; neural materials = Reference; Gaussian splat hybrids = Later (reference: only as a scene representation, not a reconstruction tool).
- **WebGPU has no matrix-multiply-accumulate yet.** Subgroups shipped in Chrome 134 (Feb 2025) [V]; I found no evidence of a subgroup_matrix / cooperative matrix origin trial [U - verify in gpuweb/Chrome status before committing]. WebNN is the only route to tensor cores/NPU from the web today.

## 2. Sources

| name | link | year | licence / venue | local |
|---|---|---|---|---|
| SVGF (Schied et al.) | research.nvidia.com SVGF preprint | 2017 | HPG | `research/papers/denoise/svgf-schied2017.pdf` |
| A-SVGF (gradient estimation) | cg.ivd.kit.edu adaptive_temporal_filtering | 2018 | HPG | `.../asvgf-schied2018-gradient-estimation.pdf` |
| Real-time Neural Radiance Caching (Mueller) | arxiv.org/abs/2106.12372 | 2021 | SIGGRAPH | `.../nrc-mueller2021-real-time-neural-radiance-caching.pdf` |
| Neural Importance Sampling (Mueller) | arxiv.org/abs/1808.07840 | 2019 | TOG | `.../neural-importance-sampling-mueller2019.pdf` |
| Random-access neural compression of material textures (NTC) | arxiv.org/abs/2305.17105 | 2023 | SIGGRAPH | `.../ntc-random-access-neural-texture-compression-2023.pdf` |
| Joint Denoising + Upscaling, multi-branch multi-scale (AMD) | gpuopen.com/download/I3D25_Joint_Denoising_Upscaling_Multi-branch_Multi-scale_Feature Network.pdf | 2025 | I3D25 | link only (77 MB, not saved) |
| OIDN | github.com/RenderKit/oidn | 2.5.1, 2026 | Apache-2.0 | Denoiser repo already vendors .tza |
| OIDN 3 preview | cgchannel.com/2026/01/open-image-denoise-3-will-support-temporal-denoising | 2026 | news | - |
| FSR SDK 2.x Ray Regeneration | gpuopen.com/amd-fsr-rayregeneration, .../amd-fsr-sdk-2-2-now-available | 2026 | SDK source MIT per press [U]; ML denoiser ships as signed DX12 DLL, binary-only per our Denoiser docs [V local] | - |
| FSR Redstone dev overview | gpuopen.com/learn/amd-fsr-redstone-developers-neural-rendering | 2025 | - | page gives no architecture |
| RTX NTC SDK | github.com/NVIDIA-RTX/RTXNTC | 2025 | NVIDIA SDK licence [U] | DX12 SM6.10 LinAlg / cooperative vectors only |
| NRD (ReBLUR/ReLAX/SIGMA) | github.com/NVIDIA-RTX/NRD | - | NVIDIA RTX SDK licence: read for ideas, do not port [V per Denoiser spec] | - |
| FidelityFX denoiser v1.1.4 | GPUOpen | - | MIT | already ported in Denoiser `denoiser/ffx` |
| mutracer | github.com/pissang/mutracer | commit 06088ce | see repo LICENSE | `research/sources/web/mutracer` (already cloned by another agent) |
| Our Denoiser | ~/Developer/Denoiser | - | own | docs/specs/{temporal,runtimes}.md, docs/research/rt-denoiser/ |

Not downloaded / not found: KPCN (Bako 2017, no clean OA PDF found), NVIDIA Neural Temporal Adaptive Sampling (Hasselgren 2020; URL returned HTML), Zeltner neural appearance models (wrong arXiv id), DLSS RR (no paper).

## 3. Findings

### 3.1 OIDN 2.x -> 3, and our port
- **OIDN 2.x:** direct-prediction U-Net (`rt_hdr`, `rt_hdr_alb`, `rt_hdr_alb_nrm`, small/large/xl via `--quality`), 3-channel colour + optional albedo/normal (prefilter nets `cleanAux`), input in HDR via PU transfer, weights as `.tza`, loadable via `oidnSetSharedFilterData`. Spatial only, one frame; consecutive frames vary. 2.4/2.5 are hardware work (AMX-FP16, XMX), nothing architectural. [V]
- **OIDN 3 (announced):** temporal; kernel prediction; two variants (real-time: prev denoised + backward MV + depth; final: +future frames, forward MV); supersampling; multi-AOV kernel reuse; Apache-2.0. Unknowns: parameter count, channel widths, per-frame ms, quantisation, whether weights stay `.tza`, tile/receptive-field limits for motion. Watch the OIDN repo `devel` branch/GitHub issues for a beta; BRIEF says do not clone big repos, but OIDN is small - clone `RenderKit/oidn` `--depth 1` once the beta tag appears. [U]
- **Our port vs OIDN native** (from `Denoiser/docs/specs/runtimes.md`): same `.tza`, WGSL fused conv/relu/pool/upsample, WebNN, ORT-web, HF kernels; WGSL 2.5-3x faster than ORT, ~2x kernels. Conceptual diffs vs native: native uses oneDNN/CUTLASS/MIOpen/Metal MPS tensor paths with winograd/fp16 tensor cores; ours is hand conv in WGSL (no MMA), so we are bandwidth/ALU bound at fp16 with `shader-f16`. Kernel prediction (OIDN 3) adds a per-pixel weighted gather after the net: cheap and expressible in WGSL.
- **Quantisation:** f16 is already used (WGSL f16, WebNN fp16; CoreML gpu fp16 error 0.7-1.3e-3, ANE 2.5-3.2e-3 vs fp32 [V local]). int8: not offered by OIDN; would need our own QAT/PTQ; WebGPU has `dot4I8Packed` (WGSL `packed_4x8_integer_dot_product`) which gave 2.3-2.9x over naive in Meet's subgroup trial context [V-ish], so an int8 conv through `dot4I8Packed` is the only MMA-free speedup. **Rating: Prototype** (int8 on one net, measure PSNR drop vs ms; HDR tails likely suffer, kernel-prediction nets tolerate it better).
- **WebNN / ORT-Web would buy:** access to tensor cores/ANE/NPU and vendor-tuned convs that WGSL can't reach (no cooperative matrix). Measured costs in our repo: macOS CoreML cold start 5-7 s first build, 0.4-1.2 s per new geometry (static shapes!), `createContext(gpuDevice)` falls back to CPU (7-8x slower) so no zero-copy with WebGPU [V local]. Verdict: for *real-time* per-frame use, WebNN's per-geometry graph build, interop copies and async dispatch are a poor fit; for *progressive/final* denoising and NPUs (low power) it is attractive. **Keep `AutoRuntime`; WebNN = Prototype only for fp16 large models on non-Apple (needs Windows/DirectML numbers).**

### 3.2 Real-time denoisers
| denoiser | what | open? | take |
|---|---|---|---|
| SVGF (2017) | demodulate, reproject colour+moments, variance from temporal moments (spatial fallback when history < 4 frames), 5-pass edge-aware a-trous guided by luminance variance, depth, normal; first atrous pass output feeds history | paper | **Adopt** as baseline; ~1-2 ms @1080p on 2017 GPUs |
| A-SVGF (2018) | gradient samples (re-use previous-frame random seeds, forward-project to detect shading change) -> per-pixel temporal alpha; handles lighting change without ghosting | paper | **Prototype** later; needs stratified-gradient hooks in the tracer (ties to our sampler: seed reuse per pixel) |
| NRD ReBLUR/ReLAX/SIGMA | hit-distance-aware, specular lobe reprojection, anti-lag, history-fix, SIGMA for shadows | source public, restrictive licence | Reference only - implement from RTG II chapter / SVGF |
| FFX denoiser (shadows, reflections) | tiles, classification, variance-guided | MIT | already ported |
| KPCN (2017) / recurrent (RAE, NTAS) | learn kernels / recurrent latent state | paper | Reference |
| OIDN 3 | see 3.1 | Apache-2.0 | watch |
| DLSS RR / FSR Ray Regen | joint (RR) / decoupled (Ray Regen) neural denoise; take demod diffuse/spec, hit distance, roughness, MV, depth, normals | closed | Reference; confirms the input list |
| I3D25 AMD joint denoise-upscale | multi-branch multi-scale; derives spatial+temporal filter weights; says chaining denoiser + upscaler naively hurts | paper | Reference for design |

Existing in our tree: `denoiser/realtime` (clean-room per-signal, experimental, "visually behind FFX on reflection bench" [V local STOCKTAKE-2026-11-06]).

### 3.3 Temporal behavior: grain vs smear
Design space: a pixel estimate = f(current noisy sample(s), history). Knobs:
1. **Reprojection quality.** Camera-only MVs are cheap but fail on moving objects/reflections. Use per-pixel analytic MVs from the primary hit (previous object transform). For mirror/glossy reflections, reproject using the *virtual hit position* (hit distance along the reflected ray; NRD/ReBLUR style) - the surface MV smears reflections.
2. **Disocclusion:** compare reprojected depth (linear, relative tolerance by grazing angle), normal dot, object/instance ID; optionally A-SVGF gradients for shading changes. On fail -> reset history length N to 0 and *widen the spatial filter* (as SVGF does with its variance fallback) rather than showing raw noise.
3. **History rectification/clamping:** neighbourhood AABB/variance clip in YCoCg (TAA) bounds ghosts but also bounds *noise reduction* - at 1 spp the neighbourhood is itself noisy so clamping tolerance must scale with the estimated variance (variance clipping with gamma proportional to sigma). For path-traced signals prefer *variance-guided alpha* over clamping: alpha = max(1/N, a_min), with a_min tied to sigma/mean.
4. **Reset vs decay:** hard reset on camera cut/scene edit; exponential decay with N capped (e.g. 32-64 for diffuse, ~8-16 for specular) for motion; unbounded 1/N only when static (our upscaler already "settles" stills).
5. **The blur/grain continuum:** grain = few spp, little filtering, history reset. Smear/ghost = long history with bad MV. "Blurry but stable" = spatial-first strategy: reduce variance with a wide edge-aware filter on the *current* frame and treat history as a tiebreaker. mutracer sits at the extreme (resolution drop + denoiser); SVGF/ReBLUR sit mid (history + variance-adaptive blur); stills sit at grain->converged. To control it expose **one temporal-strictness slider that maps to {max N, a_min, atrous iterations, clamp gamma}** and drive it from estimated variance and camera speed, so motion = more spatial blur/short N, rest = long N.
6. **With our upscaler:** `@pmndrs/upscaler` publishes motion + disocclusion guides (README). A path-traced input is currently *spatial only* (tracer random jitter, `HANDOFF` 2026-10-19; upscaler#93 plans the fix). Recommendation: tracer should use the **upscaler's jitter sequence** (shared Halton/R2 with per-frame offset) for the *primary* ray only, and denoiser run on the low-res demodulated signal in jittered space with MVs, then upscaler consumes denoised radiance + the same MV/disocclusion guides. Avoid denoise-then-temporal-upscale double-history: either (a) denoiser history + upscaler history disjoint, with upscaler's reactive mask set from denoiser variance/history-length (reactive/transparency-composition mask analogue), or (b) joint. Order for chain: demodulate -> denoise (temporal) -> remodulate -> upscale -> tonemap. Reactive mask = high where denoiser N is small.
7. **Mismatch risk:** a long-history denoiser followed by FSR-style temporal upscaler lags twice. Cap the upscaler history blend when denoiser N < threshold.

### 3.4 Sample patterns vs denoisers; inputs
- **Blue-noise error distribution** (Heitz et al.): with a per-pixel 2D blue-noise mask, error becomes high-frequency, which any low-pass spatial filter (a-trous, NLM, U-Net) removes much better than white noise at the same spp; benefit is largest at 1-4 spp, shrinks with convergence and with temporal accumulation unless the mask is *rotated per frame* (Cranley-Patterson shift with golden-ratio / R2 sequence) so temporal averaging stays unbiased and doesn't correlate with the spatial pattern. STBN (spatiotemporal blue noise, NVIDIA 2022) is the right default for real-time modes; keep Sobol/Owen scrambled for reference mode. Pair with the sampler note (see other agents' notes); here: **denoiser tests must be run with both white and STBN input** - trained networks (OIDN) were trained on white-ish independent noise and may over-smooth / produce structure on blue noise. [U - verify by running our OIDN on STBN frames]
- **Feed demodulated irradiance:** divide by albedo (first-hit diffuse albedo; for specular divide by specular reflectance F0/environment BRDF approx). Filter irradiance, remodulate after. Textures stay sharp, denoiser sees smooth signal. Needs albedo AOV with *noise-free first hit*; for glass/mirror chains use the albedo of the first non-delta surface or "virtual" albedo.
- **Diffuse/specular split** (needed by OIDN 3 aux per recap [U], NRD, DLSS RR): separate accumulators; specular uses roughness-driven lobe radius and hit-distance reprojection; different alpha/N for each.
- **Hit distance (specular):** store the *primary-ray length* of the reflection ray (or accumulated for multi-bounce) -> virtual-position reprojection + lobe footprint (`dist * tan(lobe angle)`), and for shadows penumbra width (SIGMA). Cost: one float in the tracer's ray payload; effectively free.
- **Aux buffers list for the spec:** colour (diffuse irradiance, specular radiance, separate), albedo (diffuse, spec/F0), world/view normal, roughness, linear depth, motion vector (pixel, pointing to previous), object ID / instance, hit distance(s), (optional) per-pixel sample count / variance for adaptive sampling.

### 3.5 Neural rendering beyond denoising (web feasibility)
| item | what | web verdict | rating |
|---|---|---|---|
| Neural Radiance Caching (Mueller 2021) | tiny MLP (fully-fused, 64 neurons x 5 layers, hash/frequency encoding), trained online on ~4% of paths from short training paths; query at path terminus after ~2-3 bounces; paper: ~2.6 ms @1080p on RTX 3090 for fused net (train+infer) [U - confirm from PDF §] | needs MMA to hit those numbers. Without cooperative matrix, WGSL f16 scalar/`dot4` MLP is maybe 5-20x slower [U estimate] - above our budget. WebNN can't train. Hybrid: hash-grid radiance cache (no MLP; Instant-NGP style) or radiance cascades/probes (sibling gi packages) gets 80% of NRC's benefit | **Prototype** (hash-grid variant only); full NRC **Later** pending subgroup/cooperative matrix |
| tiny-cuda-nn fully fused MLP in WGSL | one workgroup holds weights in shared memory/registers; uses MMA. In WGSL: subgroup ops are in (Chrome 134), `subgroup_matrix` is a proposal [U], f16 in via `shader-f16` | feasible functionally; speed unknown | **Prototype** microbench (64x64 MLP, batch 2^16 on M-series + Windows NVIDIA/AMD); native cooperative-vector/WMMA reference needs Windows |
| Neural importance sampling (Mueller 2019, Dahm, NPG) | normalising-flow/MLP learns guide distribution | per-ray net eval too costly; path-guiding with spatial caches (vMF/SD-tree, ReSTIR) beats it on cost | **Reference only** |
| Neural texture compression (NTC) | 4-8x vs BCn at equal quality; decode via per-texel MLP; RTX NTC SDK needs DX12 SM6.10 LinAlg / cooperative vectors (no Vulkan/WebGPU yet) [V search] | Two modes: inference-on-sample (needs MMA + STF + DLSS-like AA - not for us) and **transcode-to-BCn at load/stream** (compute shader runs MLP once per block; no MMA needed if slow-ish). Fits `gputex` streaming | **Prototype** transcode path only; sample-time **Later** |
| Neural materials (Zeltner 2023 etc.) | MLP replaces layered BSDF | per-hit MLP cost, divergence | **Reference only** |
| Gaussian splat hybrids | splats as scene/probe representation; mixed ray-trace + splat render | Rasterised splats are web-feasible (existing viewers), traced splats need BVH over splats (ours can) - but unrelated to reconstruction | **Later** |
| Neural joint denoise+upscale (own) | I3D25-style small CNN w/ temporal recurrence | needs training pipeline + data; run via our WGSL conv engine / WebNN | **Later** (after OIDN 3 weights and an SVGF-class baseline); research track |

### 3.6 What an open joint denoiser-upscaler looks like
Inputs at render res: demod diffuse + spec radiance (1-2 spp), albedo/F0, normal, roughness, depth, hit distance, jittered MV, previous output (high-res) reprojected + history-length. Output: high-res remodulated colour. Architecture (I3D25-like, scaled for web): feature extraction at render res -> multi-scale branches (a-trous-like dilations) -> predict per-pixel kernels for (a) spatial denoise of current frame, (b) sub-pixel-aware temporal blend weights for upscaled history; apply kernel to *pixel-shuffle* upsampled inputs. Pure-WGSL budget target: <=2 ms at 960x540->1920x1080 on M-series f16. Start with non-learned version: SVGF-style demod filter at render res feeding our FSR2/3 upscaler with reactive mask from variance (zero training), then learn the two weight maps.

## 4. For the spec
1. Define one **Reconstruction guide-buffer contract** (3.4 list) written by the tracer; denoiser/upscaler consume it. Align with `FrameGuides`/`FrameCamera` in `Denoiser/packages/denoiser-core/src/temporal.ts` instead of a new type.
2. Tracer must output **separate demod diffuse/specular, albedo, hit distance, roughness, object ID, MV** (MRT/storage). MV for specular via virtual hit position as an optional second MV.
3. Tracer owns jitter; expose `jitter(frame)` shared with the upscaler; sample sequence keyed per-pixel with per-frame shift (STBN default real-time, Owen-Sobol reference).
4. Modes: `reference` (no filter / OIDN on converged), `preview` (spatial OIDN small on low res during motion = the mutracer look, but with a variance-guided temporal prepass), `realtime` (temporal filter + upscaler). One "temporal strictness" control.
5. Reconstruction runs as passes on the **same GPUDevice** with explicit pre/post chain: demod -> temporal denoise -> remod -> upscaler (reactive mask from denoiser N) -> tonemap.
6. Denoiser backends stay behind `NetworkRuntime`; do not depend on WebNN for per-frame paths.
7. Reserve a slot for OIDN 3 `BeautyInputs` with (diffuse albedo, spec albedo, roughness, depth, MV, prev) but do not hardcode until weights/format are public.

## 5. Bench / comparison ideas
- Look comparison in motion: ours (SVGF-class temporal + OIDN preview) vs mutracer (spatial per-frame) vs three-gpu-pathtracer; measure temporal flicker (frame-to-frame error vs converged ref, like our `flicker 0.008` metric) and lag/ghost (error after a camera step), at fixed 1 and 4 spp.
- STBN vs white vs Owen-Sobol input into: OIDN small/large, JNLM, SVGF-class.
- OIDN WGSL f16 vs int8-dot4 PSNR/ms, 960x540 and 1920x1080, M-series + Windows GPUs.
- WebNN vs WGSL per-frame latency incl. geometry change; NPU power.
- MLP microbench for NRC in WGSL (3.5) vs hash-grid cache, ms/Mqueries.
- Chain ordering test: denoise->upscale vs upscale->denoise vs joint; PSNR/SSIM + temporal flicker.
- Inspect mutracer code: how preview scale is chosen (`RenderLoop`), whether the OIDN call is per frame and at what tile size (`src/RenderLoop.ts`, `src/oidn.ts`, `src/renderer/WGPUJNLMDenoiser.ts`) - not fully read here.

## 6. Needs Windows / native hardware
- OIDN native (CPU/CUDA/HIP/SYCL) golden outputs and ms; OIDN 3 beta once released.
- NVIDIA: DLSS RR, NRD, RTXNTC, OptiX cooperative vectors comparisons; AMD: FSR Ray Regeneration (RDNA4 + Win11 + DX12 SM6.6 only), FFX denoiser native.
- WebNN on DirectML/ONNX Runtime GPU+NPU (Intel/AMD/Qualcomm NPUs); fp16/int8 behaviour.
- Cooperative matrix / subgroup matrix behaviour on D3D12/Vulkan Dawn backends.

## 7. Open questions
- Will WebGPU ship `subgroup_matrix` (and which tile sizes/types) in 2026-27? Check gpuweb proposals + Chrome Status before NRC/MLP work.
- OIDN 3: weight format, model size, per-frame cost, does the real-time variant need *noise-free* aux and a specific noise distribution (blue-noise safe)?
- Does STBN input degrade OIDN (trained on independent noise)? [U]
- How should the denoiser and upscaler share history validity (single disocclusion mask)?
- Can per-pixel gradient samples (A-SVGF) be made cheap with a seed-reuse sampler?
- int8 usefulness vs the HDR dynamic range handled by PU transfer.
- Exact NRC cost numbers from the paper were not re-read here: check PDF before quoting.
