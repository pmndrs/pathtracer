# 03 - Native open-source renderers: quality references and speed tricks

Date: 2026-11-13. Clones in `research/sources/native/` (shallow, MANIFEST updated). Items marked (unverified) come from memory or search snippets, not from code I read.

## 1. TL;DR

- **Ground truth = pbrt-v4** (Apache-2.0, deterministic, spectral, CPU+GPU, simple text format, correct by construction). **Second reference = Mitsuba 3** (BSD-3, spectral/RGB/polarized variants, Python-scriptable, XML, easy batch rendering and per-pixel AOV/diff). Use Mitsuba for the scripted test bench, pbrt-v4 to adjudicate disagreements. Cycles (Apache-2.0) is the third, "artist-facing" reference for Principled BSDF look.
- **pbrt-v4's GPU path is not a speed model**, it is a clean wavefront design: SoA work queues, one atomic counter each, one kernel per material type (that *is* its only coherence sorting; no ray sorting), 1M-pixel scanline slabs, host-driven depth loop. It is 12x CPU on Moana (26.7 s on an A6000 vs 326 s on a 32-core 3970X, Pharr blog) and 27-100x over a 6-core CPU in 2020 (blog/HPG slides).
- **Cycles is the best open source for wavefront engineering ideas** (Apache-2.0, safe to port ideas): global state SoA, per-kernel queues, shader-key counting sort of paths, compaction of main and shadow paths, refill when active paths < 25% of max, schedule the kernel with the most queued paths. Sobol-Burley + tabulated Sobol + blue-noise sampling patterns are in `src/kernel/sample/`.
- **Wavefront is not a free win**: Dr.Jit paper reports wavefront is 2.0-7.7x slower than megakernel on GPU (Mitsuba context); a May 2026 Utah paper (Vulkan, RTX 3060 Ti, inline ray query) measured wavefront 16% faster, attributed to cache locality. For WebGPU (no SER, no callable shaders, no recursion) wavefront with material-sorted queues is the realistic answer, but benchmark both on our own scenes.
- **RTXPT is the best "real-time unified PT" blueprint** (single megakernel, V-buffer start, SER, NEE-AT feedback light sampling, SHaRC cache, ReSTIR as fallback, DLSS-RR). Its tricks that port to WGSL: V-buffer restart, NEE-AT, SHaRC hash grid (pure compute logic), sub-sampling (trace 1/4..3/4 of pixels per frame), path decomposition for denoiser guides. SER, OMM and DLSS-RR do not port.
- **Bevy Solari is the only WGSL/WESL implementation of ReSTIR DI/GI + world radiance cache** in the set (MIT/Apache-2.0). It requires `wgpu_ray_query` (hardware RT), but the ReSTIR, light-tile presampling and world-cache compact/update passes are portable to our software BVH. Read first.
- **tinybvh already ships OpenCL wavefront tracers** (`kernels/wavefront.cl`, `wavefront2.cl`, `traverse_cwbvh.cl`, `tiny_bvh_pt.cpp`, `tiny_bvh_gltf.cpp`): free reference for our own bvh package, same author as our core.
- **Needs Windows**: RTXPT/Falcor/RTXDI (DX12/Vulkan + DXR), OptiX pbrt GPU path, Cycles OptiX/HIP, HIPRT. Everything needed for reference *images* (pbrt CPU, Mitsuba LLVM/CPU, Cycles CPU) runs on the Mac.

## 2. Sources

| name | link | year | license | local | why |
|---|---|---|---|---|---|
| pbrt-v4 | github.com/mmp/pbrt-v4 | 2020-26 | Apache-2.0 | native/pbrt-v4 | ground truth, GPU wavefront, samplers |
| pbr-book 4ed ch.15 | pbr-book.org/4ed/Wavefront_Rendering_on_GPUs | 2023 | CC BY-NC-ND | link | wavefront design text |
| pbrt-gpu HPG20 | cdn.pharr.org/pbrt-gpu-hpg2020.pdf | 2020 | author site | link | speedups |
| Moana GPU blog | pharr.org/matt/blog/2021/07/29/moana-rendered-on-the-gpu | 2021 | - | link | 26.7 s vs 326 s |
| Mitsuba 3 | github.com/mitsuba-renderer/mitsuba3 | 2022-26 | BSD-3 | native/mitsuba3 | spectral variants, scripting |
| Dr.Jit | github.com/mitsuba-renderer/drjit | 2022-26 | BSD-3 | native/drjit | JIT, megakernel vs wavefront |
| Dr.Jit paper | arxiv.org/abs/2202.01284 | 2022 | SIGGRAPH | link | JIT design, 2.0-7.7x wavefront penalty |
| Cycles (standalone) | github.com/blender/cycles | 2011-26 | Apache-2.0 | native/cycles (18 MB, full standalone repo, no sparse needed) | GPU integrator |
| Falcor | github.com/NVIDIAGameWorks/Falcor | 2020-25 | NVIDIA BSD-3-like | native/falcor | research PT framework |
| RTXPT 1.9.0 | github.com/NVIDIA-RTX/RTXPT | 2023-26 | NVIDIA RTX SDKs License | native/rtxpt | real-time PT blueprint |
| RTXDI 3.0 | github.com/NVIDIA-RTX/RTXDI | | NVIDIA RTX SDKs License | native/rtxdi | ReSTIR DI/GI |
| RTXGI | github.com/NVIDIA-RTX/RTXGI | | same | native/rtxgi | DDGI/NRC-style |
| SHARC | github.com/NVIDIA-RTX/SHARC | | same | native/sharc | spatial-hash radiance cache |
| NRD | github.com/NVIDIA-RTX/NRD | | same | native/nrd | ReBLUR/ReLAX/SIGMA |
| Capsaicin (local) | ~/Developer/Capsaicin | | MIT (AMD) | not cloned | `reference_path_tracer`, `gi1` (hash-grid cache, world-space ReSTIR, screen probes) |
| HIPRT | github.com/GPUOpen-LibrariesAndSDKs/HIPRT | | MIT | native/hiprt | AMD RT, BVH on AMD |
| Orochi | github.com/GPUOpen-LibrariesAndSDKs/Orochi | | MIT | native/orochi | CUDA/HIP dynamic loader |
| Embree | github.com/RenderKit/embree | | Apache-2.0 | native/embree | CPU BVH reference, BVH quality yardstick |
| OIDN | github.com/RenderKit/oidn | | Apache-2.0 | native/oidn | denoiser networks (ours ports these) |
| OpenPGL | github.com/RenderKit/openpgl | | Apache-2.0 | native/openpgl | path guiding (vMF/VMM, SDTree) |
| kajiya | github.com/EmbarkStudios/kajiya | 2019-22 | Apache-2.0/MIT | native/kajiya | ircache, rtdgi, rtr, wrc, ReSTIR GI |
| Tungsten | github.com/tunabrain/tungsten | 2014-16 | BSD-like | native/tungsten | JSON scene fmt, Bitterli scene set |
| LuxCoreRender | github.com/LuxCoreRender/LuxCore | | Apache-2.0 | native/luxcore | OpenCL hybrid, BIDIR, OpenCL kernels |
| appleseed | github.com/appleseedhq/appleseed | | MIT | native/appleseed | production CPU renderer (project archived) |
| Lightmetrica v3 | github.com/hi2p-perim/lightmetrica-v3 | | MIT | native/lightmetrica | research, comparison harness |
| tinybvh | github.com/jbikker/tinybvh | 2024-26 | MIT | native/tinybvh (893 MB with assets, read only) | our BVH upstream; wavefront.cl, pt example, gltf example |
| Bevy Solari | github.com/bevyengine/bevy crates/bevy_solari | 2025-26 | MIT/Apache-2.0 | native/bevy-solari (sparse, 1.4 MB) | WGSL ReSTIR + world cache |
| Bitterli rendering-resources | benedikt-bitterli.me/resources | 2016 | per-scene CC | link only (GBs) | reference scene set, pbrt/mitsuba/tungsten exports |
| Filament | github.com/google/filament | | Apache-2.0 | link only | real-time PBR spec (not a PT) |
| Godot SDFGI/HDDGI | godotengine.org / godot repo | | MIT | link only | probe GI, cascades |
| Unreal Lumen / SWRT | dev.epicgames.com docs | | proprietary (UE EULA) | link only | SWRT = SDF + mesh cards + surface cache |
| Megakernel vs wavefront (Utah) | arxiv.org/abs/2605.27323 | 2026 | CC BY 4.0 (paper) | link | 16% wavefront win, RTX 3060 Ti |
| RenderMan XPU | diglib.eg.org (Pixar paper) | 2025 | EG | link | CPU+GPU hybrid, 5-15x |
| WebGPU OpenPBR PT (Stucki) | arxiv.org/abs/2407.19977 | 2024 | arXiv | link | web PT w/ OpenPBR, a competitor reference |
| Jacco Bikker blogs | jacco.ompf2.com, "Wavefront path tracing" series | 2019-24 | - | link | wavefront tutorials, BVH, CWBVH |
| Khronos vk_gltf_renderer | ecosystem.khronos.org (nvpro-samples) | | Apache-2.0 | link | Vulkan glTF PT, good glTF-native reference (unverified details) |

## 3. Findings per renderer

### 3.1 pbrt-v4 (Reference: ground truth; GPU design: Reference only)
- **Wavefront GPU** (`src/pbrt/wavefront/*`, `src/pbrt/gpu/optix/*`): `workqueue.h:42-110` is `SOA<WorkItem>` + one `cuda::atomic<int> size`; push = `fetch_add` (relaxed). `ForAllQueued` (`workqueue.h:~110`) launches `maxQueued` threads and each early-outs on `index >= Size()`, so the launch size is static and the host never reads queue counts (no CPU round trip for dispatch size). A WebGPU port should instead use `dispatchWorkgroupsIndirect` with args written by a tiny "prepare" kernel, which is better than pbrt.
- **Memory**: `integrator.cpp:229-260`: `maxSamples = 1M`, `scanlinesPerPass` computed so queues cover whole scanlines, `maxQueueSize = res.x * scanlinesPerPass`; two ping-pong ray queues, shadow queue, escaped, hit-area-light, medium, subsurface queues, and `MaterialEvalQueue` (multi-queue, one per concrete material type, split `Basic` vs `Universal` texture evaluator). `PixelSampleState` SoA carries L, filter weight, sampler dim, etc.
- **Loop** (`integrator.cpp:~320-440`): for each scanline slab, for each sample index, for each depth: GenerateRaySamples (sampler), IntersectClosest (OptiX), SampleMediumInteraction, HandleEscapedRays, HandleEmissiveIntersection, EvaluateMaterialsAndBSDFs (kernels per material type, produce next ray + shadow ray via NEE), TraceShadowRays, SampleSubsurface. Host launches ~10-20 kernels per bounce.
- **Sorting**: none for rays. Coherence only from per-material kernels. No Morton or direction sort. This is the OptiX/RT-core world where traversal is hardware; for software traversal in WGSL, ray sorting may pay (see 4.2).
- **Samplers** (`samplers.h`): `ZSobolSampler` (`samplers.h:225+`; Morton-index + Owen-scrambled Sobol with `FastOwen`/`PermuteDigits`/`Owen` randomizers; needs pow2 spp), `PaddedSobol`, `Sobol`, `PMJ02BN` (`PMJ02BNSampler`, table-based, blue-noise-shaped error over pixels), `Halton`, `Stratified`, `Independent`. Default for pbrt-v4 is ZSobol. Takeaway: ZSobol is cheap in compute (a few dozen ALU, no tables) and gives blue-noise-like pixel error distribution without textures: **Adopt now** for WGSL (we need u32 ops only; Morton via bit-spread, `reverseBits`, hash for Owen). PMJ02BN needs 4 MB-ish tables (pbrt `pmj02tables`); Later.
- **Speed**: 12x CPU on Moana (26.7 s on RTX A6000 vs 326.5 s on 32-core 3970X); 27-51x per-scene vs 6-core in HPG20 slides; 50-100x quoted in the 2020 blog. Book says 2,669 s CPU 32-core vs ~8x faster GPU on one scene. Spectral rendering costs ~4 wavelengths per path (hero-wavelength style) and is the main reason the numbers are not high.
- **Quality-reference suitability**: excellent. Deterministic with `--seed`, `--spp`, `--pixelbounds`, `--gpu` flag; EXR output (`--outfile x.exr`), scene text format `.pbrt` (documented, `pbrt.org/fileformat-v4`), `--upgrade` converts v3. Integrators: `path`, `volpath`, `bdpt`, `mlt`, `sppm`, `simplepath`. Uses spectral rendering unless `--spectral`... default is spectral (RGB colors upsampled via RGB-to-spectrum tables). Use `film "rgb"` + `integrator "volpath"` for ours. No glTF import.
- **Scene export to pbrt-v4**: Assimp (`assimp export in.gltf out.pbrt`; pbrt README states Assimp has a pbrt-v4 exporter; material conversion is lossy, so expect to hand-map). Better: write our own exporter in the test bench (mesh -> PLY/`trianglemesh`, materials -> `diffuse`/`conductor`/`dielectric`/`coateddiffuse`/`diffusetransmission`, lights -> `diffuse` area lights, `infinite` light for env, camera -> `perspective` with `fov` along the shorter axis; pbrt uses left-handed, glTF right-handed, flip handedness with `Scale -1 1 1`). Keep a mapping table: glTF metallic-roughness -> pbrt `coateddiffuse`/`conductor` approximation is **not** exact; therefore the bench should define a restricted material subset (Lambert, rough conductor GGX with named eta/k, smooth/rough dielectric, emissive) with exact equivalents in all three renderers.

### 3.2 Mitsuba 3 / Dr.Jit (Reference: second ground truth; JIT: Reference only)
- Variants (`mitsuba/variants`): `scalar_rgb`, `scalar_spectral`, `llvm_ad_rgb`, `llvm_ad_spectral`, `cuda_ad_rgb`, `cuda_ad_spectral`, plus `_polarized`. LLVM variants run on the Mac CPU (Apple Silicon works; CUDA needs NVIDIA).
- **Dr.Jit** JIT-traces Python/C++ array ops and compiles into a wavefront-style kernel per traced loop; loops can be recorded (`dr.while_loop` in "symbolic" mode produces a megakernel). Mitsuba's `path` integrator (`src/integrators/path.cpp`) is a *single* loop that Dr.Jit executes either evaluated (wavefront; reorders/compacts) or symbolic (megakernel). Reported: wavefront 2.0-7.7x slower on GPU in their benchmarks (Dr.Jit paper), which is why the default is megakernel with symbolic loops.
- Lessons: (a) JIT specialization per scene (constants folded, unused BSDF branches removed) is a real win; in WGSL we can mimic with **pipeline-override constants + string-templated WGSL per scene material set** (Adopt now). (b) Mitsuba's samplers (`src/samplers`): `independent`, `stratified`, `multijitter`, `orthogonal`, `ldsampler` (Halton/Sobol-style); no ZSobol, so for bit-comparable noise vs pbrt use `independent` and compare converged images only.
- **Reference suitability**: very good. XML scene format with `<shape type="obj|ply|serialized">`, `<bsdf type="diffuse|roughconductor|roughdielectric|principled|twosided|blendbsdf">`, `<emitter>`, `<sensor>`; `mi.load_file`, `mi.render(scene, spp=N, seed=S)` returns a bitmap; trivial to loop over scenes and write EXR. Differentiable path gives us gradient-based parameter checks (e.g., verifying our BSDF sampling pdf via `mi.render` of `bsdf` plugin eval/sample tests in `python_tests`).
- **Scene export to Mitsuba**: Blender add-on (mitsuba-blender, BSD), write XML directly from our scene description (simple), or `mi.load_dict(...)` from Python; no native glTF loader (verified by search, not by code). Convert glTF->PLY/OBJ + material dict ourselves.
- Spectral: `scalar_spectral` uses 4 wavelengths (hero-ish, `sample_rgb_spectrum`, sRGB->spectrum tables via `Spectrum uplifting` Jakob-Hanika 2019). Same as pbrt-v4 up to the uplift method; RGB variants skip spectral for faster/lower-noise comparisons with our RGB mode.

### 3.3 Blender Cycles (Apache-2.0; ideas OK; Reference for GPU wavefront scheduling)
Local `native/cycles/src/integrator`, `src/kernel/integrator`, `src/kernel/device/gpu/kernel.h`.
- **State machine path tracer**: kernels `init_from_camera`, `intersect_closest`, `intersect_shadow`, `intersect_subsurface`, `intersect_volume_stack`, `shade_surface`, `shade_background`, `shade_light`, `shade_volume`, `shade_shadow`, `megakernel` (`kernel/integrator/*.h`). Each path carries `INTEGRATOR_STATE` in a **SoA** (one device buffer per struct member, `path_trace_work_gpu.cpp:~140-160`), and `state_flow.h` pushes a path index onto the next kernel's queue with an atomic counter.
- **Scheduling** (`path_trace_work_gpu.cpp`): `min_num_active_main_paths_ = max_num_paths_/4` (line 148, "1:4 busy:total found to give good performance"); loop at ~392-420: enqueue new work tiles when active < min, else run `get_most_queued_kernel()` (lines 429-452; picks the kernel with most queued paths); `compact_main_paths` (729) and `compact_shadow_paths` (755) keep state arrays dense so refilling works; work tiles use `max_num_path_states/8` to refill greedily; shadow catcher splits path states in halves.
- **Shader sorting**: `shader_sort_key` stored in path state; `sort_key_counter[kernel][key]` is atomically incremented when a path is queued (`state_flow.h:113-199`); kernels `INTEGRATOR_SORT_BUCKET_PASS` and `SORT_WRITE_PASS` (`device/gpu/kernel.h:~496-600`) do a **partitioned local-atomic counting sort** by shader id; `num_sort_partitions` (queue.h:116, Metal `queue.mm:290`) chosen per device; sorted only when shader count is large enough. This is the real-world answer to "sort by material": counting sort by key with prefix sums per partition. **Prototype** (WGSL: workgroup-atomic histograms + subgroup ops).
- **Sampling**: `kernel/sample/pattern.h`: `SOBOL_BURLEY` (Burley 2020 Owen-scrambled Sobol with hashed index shuffle), `TABULATED_SOBOL` (precomputed tabulated, faster), `BLUE_NOISE_PURE` / `BLUE_NOISE_FIRST` (screen-space blue noise first N samples then switch; Cycles 4.2+), scrambling distance, adaptive sampling (`adaptive_sampling.cpp`), `denoiser_oidn(_gpu)`, `denoiser_optix`, `denoiser_dlss`.
- **Light tree** (`kernel/light/tree.h`): Conty/Kulla-style importance LBVH over lights with orientation cones and per-node energy; handles many emissive triangles. **Adopt now** conceptually (or use NEE-AT/ReSTIR instead for real-time; keep light tree for the reference mode).
- **Shadow catcher** (`shadow_catcher.h`): paths are split at the first bounce on a catcher object to produce "with" and "without" passes. Low priority for us (**Later**).
- **Guiding**: `kernel/integrator/guiding.h` integrates **OpenPGL** (RIS between BSDF and guide, `guiding_sampling_prob`, surface+volume guiding). CPU-only in Cycles. Path guiding on web GPU: **Reference only** (maybe Later).
- Cycles BVH: Embree on CPU, OptiX/HIPRT/MetalRT/oneAPI RT on GPU; Cycles' own BVH is software fallback (`kernel/bvh/*`).
- **Reference suitability**: Cycles CPU in headless Blender (`blender -b scene.blend -f 1 -- --cycles-device CPU`) with seeds and fixed samples; **Principled v2** is the definition of what artists expect; scenes through Blender glTF importer give the most faithful glTF look (loads glTF + KHR_materials_*). Blender is GB-sized, so not cloned; the Blender binary is a Windows/Mac download. Good as visual-sanity reference, not mathematically exact vs pbrt.

### 3.4 NVIDIA: Falcor, RTXPT, RTXDI, RTXGI, NRD, SHaRC (all NVIDIA licenses: read for ideas; check redistribution terms before copying anything)
- **Falcor** (`native/falcor`, 78 MB): research framework; `Source/RenderPasses/PathTracer` is the origin of RTXPT's path tracer: single raygen, `TinyUniformSampleGenerator` (pcg-based) and `HaltonSampler`; ReSTIR PT (`ReSTIRPT` fork by Lin et al., not in the NVIDIA repo, unverified), `Slang` shader language. Windows/Vulkan/D3D12 only.
- **RTXPT 1.9.0** (README + `Docs/Design.md` read): Unified forward PT with NEE; V-buffer prepass with **Primary Surface Replacement** (mirror chains) and 3-layer path decomposition; `NEE-AT` (temporally adaptive light sampling from per-pixel feedback, up to 2M lights, emissive triangle LOD clusters); SHaRC traces 1 path per 4x4 pixel block (~6% of pixels) and main path terminates into the cache; sub-sampling 3/4, 1/2, 1/4 pixels per frame; SER via HitObject; inline `RayQuery` with OMM alpha; "Practical Hash-based Owen Scrambling" (Burley 2020) sampler; async compute overlap for DLSS-RR. NVIDIA claims ~20% gain 1.8.1->1.9.0 (README). Requires DXR 1.1 + SM 6.6; DX12 or Vulkan; Windows. Shader files `Rtxpt/Shaders/PathTracer/PathTracer.hlsli`, `PathTracerNEE.hlsli`, `PathTracerSHaRC.hlsli`, `PathState.hlsli` (compact path state: ~ position/dir packed, throughput as f16/rgb9e5-style, worth reading for payload packing).
- **RTXDI 3.0** (`native/rtxdi`): ReSTIR DI/GI SDK, includes `RIS` tile presampling (`PresampleLights`), temporal+spatial resampling with bias-correction modes (Basic, Pairwise MIS, Raytraced), light BVH/importance, reservoir packing (`RTXDI_Reservoir`, 128-bit style). Reservoir layout and pairwise MIS are directly portable to WGSL. **Adopt now** (conceptually) for the real-time tier; code is NVIDIA-licensed, write our own.
- **SHARC**: header-only HLSL/Slang spatial-hash cache: cell keys from position+normal quantized by world-cell size with LOD from camera distance, `RWStructuredBuffer` of 64-bit-ish hash entries + accumulated radiance (fixed-point u32 per channel so `InterlockedAdd` works), stages `Update` (trace short paths, write radiance), `Resolve` (convert accumulated to cached radiance with temporal blending, age-based eviction), `Compaction`, and `Query`. Maps to WGSL because WGSL has u32 atomics (`atomicAdd`, `atomicCompareExchangeWeak`) and no 64-bit atomics: use 2x u32 keys and CAS on the lower word. **Prototype** (also Bevy Solari's `world_cache_*.wesl` is the WGSL precedent, with life/age and compaction).
- **RTXGI** (`native/rtxgi`): DDGI + Neural Radiance Cache (NRC, not open weights; unverified). DDGI is covered by sibling `webgpu-ddgi`.
- **NRD**: ReBLUR, ReLAX, SIGMA shadow; `Source/Shaders` HLSL; already referenced by `rt-denoiser-sources`; other agents cover denoisers.

### 3.5 AMD: Capsaicin (local), HIPRT, Orochi
- **Capsaicin** (`~/Developer/Capsaicin/src/core/src/render_techniques`): `reference_path_tracer` (rt + comp variants), `gi1` (GI-1.0: screen probes, `hash_grid_cache.hlsl`, `world_space_restir.hlsl`, `glossy_reflections.hlsl`, `gi_denoiser.hlsl`), `variance_estimate`, `image_metrics` (useful for comparisons, check it computes MSE/PSNR/FLIP-like). Good example of a hash-grid radiance cache + world-space ReSTIR with no screen-space dependency; MIT.
- **HIPRT** (`native/hiprt`, MIT): AMD RT library: BVH build (LBVH/PLOC/H-PLOC style builders `hiprt/impl/`), "bvh8 / 4-wide with box encoded", traversal in HIP with the `ray_query`-like API and stack handling (short stack + LDS), `hiprtGeomCustomTraversal`; useful as reference for software-traversal tricks AMD uses where HW RT is absent (RDNA2/3 RT only accelerates box/triangle tests, traversal is software on shader). Directly relevant: **AMD's traversal is shader code**, so tuning (stack in LDS, BVH4/8 layout, early-out ordering) transfers to WGSL. Also Orochi (MIT) is only a loader; **Reference only**.
- AMD hardware needs: HIPRT build/run needs Windows/Linux with AMD GPU (or CUDA via Orochi).

### 3.6 Intel: Embree, OIDN, OpenPGL (Apache-2.0)
- **Embree** (`native/embree`): quality yardstick for BVH build (SAH binned, spatial splits, `BVH4/8`, `MBlur`), packet/stream tracing; use it as the CPU BVH reference to compare our GPU LBVH/H-PLOC tree SAH cost and traversal count per ray (not time). Mostly tinybvh territory (sibling).
- **OIDN** (`native/oidn`): 2.x/3.x UNet weights `weights/*.tza` (Apache-2.0). `oidn` CLI/`oidnDenoise` is the native reference for denoiser comparison on Windows/Mac (supports Metal, CUDA, HIP, SYCL). Our Denoiser already outperformed mutracer; for correctness compare PSNR vs OIDN native on identical noisy EXR + albedo/normal AOVs (`oidnDenoise -hdr in.pfm -alb a.pfm -nrm n.pfm`).
- **OpenPGL** (`native/openpgl`): guiding fields (Gaussian/vMF mixture, SDTree-like spatial structure), training on CPU, query sampling `PGLSampler`. Gains in hard indirect scenes (caustics, door-ajar). Cost: per-vertex data capture + CPU training; a GPU/WebGPU port is a research item. **Later / Reference only**.

### 3.7 kajiya (Apache-2.0/MIT; Reference for real-time GI)
`native/kajiya/assets/shaders`: `ircache/` (world-space irradiance cache with pool allocation, `age_ircache_entries.hlsl`, `ircache_compact_entries.hlsl`, `scroll_cascades.hlsl`, `prepare_trace_dispatch_args.hlsl` = indirect dispatch args computed GPU-side), `rtdgi/` (ReSTIR-GI-like with spatial/temporal reuse and bilateral filter), `rtr/` (reflections), `wrc/` (world radiance cache), `lighting/`, `ssgi/`. Notable: a 1-spp budget at 1080p pipeline, hash grid / cascade pool, GPU-driven dispatch. Rust+Vulkan with HW RT. Ideas port; the **ircache entry lifecycle** (age, compact, trace budget by indirect dispatch) is a good template for WGSL **Prototype**.

### 3.8 Tungsten, LuxCoreRender, appleseed, Lightmetrica
- **Tungsten** (`native/tungsten`, Bitterli): JSON scene format (`.json` + meshes `.wo3`/obj), integrators include `path_tracer`, `light_tracer`, `photon_map`, `progressive_photon_map`, `multiplexed_metropolis`, `kelemen_mlt`, `reversible_jump_mlt`; its sibling scene exports (`rendering-resources`) ship each scene for **pbrt-v3/v4(ish), Mitsuba (0.6 XML), Tungsten**. Standard test scenes for sampling/MIS research (Cornell box, veach-mis, veach-bidir, bathroom, kitchen, living-room, staircase, bedroom, dining-room, glass-of-water, volumetric-caustic). Bitterli scenes: link-only because of size. **Best source of pre-made multi-renderer test scenes.** Also `pbrt-v4-scenes` (github.com/mmp/pbrt-v4-scenes, link only, Moana-class and full scenes) exists (unverified size).
- **LuxCoreRender** (`native/luxcore`, Apache-2.0): OpenCL megakernel-ish (`slg/kernels`), PATHOCL, BIDIRCPU, BAKECPU; Blender add-on BlendLuxCore; scenes in its own `.scn/.cfg`. Notably slow startup (OpenCL compile) and heavyweight; the lesson is **kernel compile time and specializing material evaluation per scene** (same web issue: shader compile time). **Reference only**.
- **appleseed** (`native/appleseed`, MIT, project archived): production CPU PT with BDPT/SPPM, OSL shaders; good code for volume/subsurface; not needed. **Reference only**.
- **Lightmetrica v3** (`native/lightmetrica`, MIT): research renderer with Python API, designed for comparing integrators; **Reference only**.

### 3.9 Bevy Solari (MIT/Apache-2.0; **most directly relevant WGSL code**)
`native/bevy-solari/crates/bevy_solari/src` (2757 lines of WESL/WGSL total):
- `pathtracer/pathtracer.wesl` (100 lines): the reference accumulation path tracer, one 8x8 workgroup per tile, PCG-style RNG from pixel index + frame*5782582, jitter via `rand_vec2f`, `accumulation_texture` rgba32float (alpha = sample count), MIS between BSDF and light sampling.
- `realtime/`: `initial_path.wesl` (508 lines: first-bounce path with reservoir creation), `restir.wesl` (365: DI/GI temporal+spatial reuse), `presample_light_tiles.wesl` (RIS light tile presampling), `world_cache_{update,query,compact}.wesl` (spatial hash radiance cache: active-cell list, per-cell update budget `world_cache_cell_updates_soft_target`, life/age, GI rays short-range `world_cache_max_gi_ray_distance`, DI via RIS), `resolve_dlss_rr_textures.wesl`.
- `scene/sampling.wesl` (359), `scene/brdf.wesl` (176): light sampling with emissive triangle sampling, power heuristic, environment sampling, BRDF with F_AB LUT. `scene/binder/*.rs` handles bindless-ish buffer/texture arrays (`binding_array`), TLAS instance setup in compute (`setup_tlas_instances.wesl`).
- Uses `enable wgpu_ray_query;` (native wgpu extension, **not** WebGPU browser); so for the web we replace `trace_ray` with our software BVH walk. The rest ports verbatim in concept. License MIT/Apache so even code reuse is allowed (still prefer re-implementation; attribute).
- Performance claims: Bevy docs show real-time at 1080p-ish on RTX 40 class (unverified numbers; not measured).

### 3.10 tinybvh and Jacco Bikker (MIT)
- `native/tinybvh`: layouts BVH2/4/8, `BVH8_CWBVH` (80-byte compressed wide BVH, Ylitie 2017), `BVH_GPU`, `BVH4_GPU`, SBVH, `PLOC`-ish, `BVH_Verbose`, optimizer (`bvhopt.cpp`, `tiny_bvh_optimizer.cpp`), `tiny_scene.h` (TLAS, instancing), AMD HW RT bridge (`tiny_bvh_amd_hwrt.cpp`, `kernels/traverse_amd_hw*.cl`).
- **Wavefront reference**: `kernels/wavefront.cl`, `wavefront2.cl`, `wavefront_amd_hw.cl`, `raytracer.cl` and `examples/tiny_bvh_pt.cpp`/`tiny_bvh_renderer.cpp`: the "Wavefront path tracing" blog design (generate -> extend -> shade -> connect queues, atomic counters, compact rays). Same author as the BVH we use, so layout assumptions match. `tiny_bvh_gltf.cpp` shows glTF loading via `tinygltf`, useful for test bench loader semantics.
- Bikker's posts (jacco.ompf2.com: "Wavefront Path Tracing", "How to build a BVH" series, "Ray Tracing with the CWBVH") are the best tutorials on queue-based design and SAH tricks; link only.

### 3.11 Others
- **Filament**: not a path tracer, but its `docs/Filament.md` PBR model (GGX, Smith height-correlated visibility, Burley diffuse, multiple-scattering energy compensation, clear coat, anisotropy, IBL split-sum) is the canonical real-time spec that glTF content targets; link only. Their offline "cmgen" IBL prefilter is a useful reference for env importance sampling.
- **Godot SDFGI / HDDGI**: SDFGI = cascaded SDF voxel GI (cheap software "ray tracing" of a distance field; MIT); HDDGI is Godot's DDGI proposal. Relevant for sibling GI libs, **Reference only** here.
- **Unreal Lumen / software ray tracing** (link only): Software RT = per-mesh SDF + global SDF + mesh cards -> surface cache; screen probes + radiance cache; hardware RT path optional. Lesson for us: a "trace + minimal shade" utility core should expose **cheap approximate tracing (SDF / low-res)** as an alternative backend for far-field rays. **Later**.
- **2024-2026 claimants**: RenderMan XPU (Pixar; 5-15x vs CPU, closed), Utah megakernel-vs-wavefront paper (above), Stucki WebGPU/OpenPBR (2024), Rayzee (three.js WebGPU PT, npm `rayzee` 6.x-7.x; unverified claims), Khronos `vk_gltf_renderer`/nvpro-samples. I did not find an open renderer with credible published speed beyond RTXPT, Solari and the Cycles/pbrt GPU backends. mutracer is covered by other agents.
- **Brook / "GPU path tracing in a weekend"** style: Ray Tracing in One Weekend ports (WebGPU/Rust); no useful speed ideas, useful only as a sanity-check smallpt scene (Cornell). **Reference only**.

## 4. Techniques / ratings

| technique | source | WGSL mapping | rating |
|---|---|---|---|
| SoA work queues with atomic counters + GPU-written indirect dispatch args | pbrt-v4, Cycles, tinybvh | `atomicAdd` on counter in storage buffer; a 1-thread "prep" kernel writes `DispatchIndirect` args (pbrt instead launches max size and early-outs) | Adopt now |
| Queue per material class / counting sort by shader key | pbrt (per type), Cycles (partitioned bucket sort) | workgroup-atomic histograms, subgroup ballots; pre-sorted keys | Prototype |
| Refill when active < 25% + compaction | Cycles | indirect dispatch + prefix-sum compaction (subgroup scan) | Prototype |
| "Run kernel with most queued paths" scheduling | Cycles | needs readback or GPU-side selection; GPU-side: single uber "scheduler" pass picks next indirect args | Later |
| Megakernel vs wavefront | Dr.Jit (wavefront 2-7.7x slower), Utah paper (wavefront +16%) | Benchmark both on our scenes; start with a megakernel with per-bounce `workgroup` loop and compile-time material specialization; build the wavefront in parallel | Prototype (both) |
| Scene-specialized shader (override constants / templated WGSL) | Dr.Jit JIT, LuxCore compile pain | pipeline constants and string templating; cache pipelines | Adopt now |
| ZSobol / hash-Owen Sobol sampler | pbrt-v4, RTXPT (Burley 2020), Cycles | pure u32 ALU; Morton from pixel + sample index; Owen via `nested uniform scramble` (Laine-Karras hash) | Adopt now |
| Blue-noise first N samples then switch | Cycles BLUE_NOISE_FIRST | one 128x128 tileable texture or ZSobol (already blue-ish) | Adopt now (if ZSobol alone insufficient) |
| PMJ02BN tables | pbrt-v4 | tables as storage buffer, a few KB-MB | Later |
| Light tree / many-light sampling | Cycles `light/tree.h`, RTXPT NEE-AT | LBVH over emissive tris + cones; NEE-AT feedback for real-time | Prototype |
| ReSTIR DI/GI with reservoirs and RIS light tiles | RTXDI, Bevy Solari | reservoirs as struct in storage buffer (vec4u pack) | Prototype |
| Spatial-hash radiance cache | SHaRC, Capsaicin `hash_grid_cache`, Bevy Solari world cache, kajiya ircache | u32 atomics, fixed-point accumulate, no 64-bit atomics -> two-word keys | Prototype |
| V-buffer start, Primary Surface Replacement, path decomposition | RTXPT | primary raster or compute V-buffer; denoiser guides | Prototype |
| Sub-sampling pixels per frame (checkerboard 1/4..3/4) | RTXPT | trivial, pairs with temporal accumulation | Adopt now |
| Short-stack / LDS stack traversal, BVH4/8 layouts | HIPRT, tinybvh CWBVH | workgroup memory stack; sibling `bvh` covers | Reference only (owned by bvh pkg) |
| Path guiding | OpenPGL/Cycles | CPU training; no | Later |
| SER / OMM / DLSS-RR | RTXPT | not available on web | Reference only |

## 5. For the spec
1. **Reference pair**: pbrt-v4 (`volpath`, RGB film, ZSobol, EXR) as ground truth; Mitsuba 3 `llvm_ad_rgb`/`scalar_spectral` as scripted cross-check; require pbrt vs Mitsuba agreement (<1% mean relative error at ~16k spp on the bench scenes) before using either as truth for a scene. Cycles is only a visual sanity reference.
2. **Bench material subset** with exact equivalents in all renderers: diffuse (Lambert), rough conductor GGX (named eta/k or F0), smooth/rough dielectric GGX, diffuse emitter, uniform and HDR env. Defer Principled/OpenPBR parity, then compare against Cycles/Mitsuba `principled` separately.
3. **Scene format**: author bench scenes once in a neutral JSON + PLY/OBJ description plus glTF; write exporters to `.pbrt` and Mitsuba XML (small Python script in `research/tools/`); do not depend on Assimp (lossy, but available as fallback for geometry). Use Bitterli resources (cornell box, veach-mis, veach-bidir, bathroom, kitchen, staircase, living-room) as the "known" scene set where pbrt/Mitsuba exports already exist (link only; download is the Windows-machine/CI step, scenes are hundreds of MB; confirm per-scene license).
4. **Coordinate/camera conventions** to pin: pbrt left-handed, Mitsuba right-handed, glTF right-handed Y-up; pbrt `fov` applies to shorter image axis; Mitsuba `fov_axis`. Sensor/film: use box or Gaussian pixel filter identically (pbrt default Gaussian, radius 1.5; Mitsuba default `gaussian`); set `box` filter, radius 0.5, for exact per-pixel comparisons.
5. **Architecture**: start with a megakernel-plus-specialization path tracer and a wavefront variant behind the same scene/material API; GPU-written indirect args everywhere; shader-key counting sort as optional pass; sampler = ZSobol-style default with per-bounce dimension reservation (as pbrt `StartPixelSample(p,index,dim)`).
6. **Deterministic mode**: fixed seed, no atomics-order-dependent accumulation, f32 accumulator with Kahan or two-level (per-batch f32 sum, then f32/f64-emulated total) for long converged references.
7. **Interop with real-time tier**: expose reservoir and hash-cache data structures as separate modules (ReSTIR, world cache) so sibling libs (`reflections`, `webgpu-ddgi`) can reuse; follow Bevy Solari module split (`initial_path`, `restir`, `world_cache_*`).

## 6. Bench / comparison ideas
- Convergence benchmark: noise-vs-time curves (relMSE vs seconds and vs spp) against pbrt-v4 (CPU, 32 threads) and Mitsuba LLVM on the Mac, same scenes, same spp; report Mrays/s and "ms per 1spp at 1080p" for primary, shadow, and incoherent bounce rays, as Cycles/pbrt do not publish such numbers.
- Sampler study: our ZSobol vs independent vs blue-noise at 1, 4, 16, 64 spp; error image and FLIP; compare to pbrt's `zsobol` and `pmj02bn` at the same spp.
- Wavefront vs megakernel and with/without material sorting on materials-heavy scenes (bathroom, kitchen, Sponza with many materials).
- BVH quality: SAH cost and nodes visited per ray vs Embree (BVH4 SAH) and tinybvh CWBVH; separate from path tracer measurements.
- Radiance-cache and ReSTIR convergence vs path-traced truth (bias map).
- Denoiser PSNR/FLIP vs native OIDN on the same noisy buffers with albedo+normal AOVs.
- Per-scene shader compile times on Chrome/D3D12 (DXC/FXC) vs Metal, since LuxCore/Dr.Jit show JIT cost dominates "first frame".

## 7. Needs Windows / native hardware
- RTXPT, Falcor, RTXDI samples, NRD samples, SHaRC samples: DX12 (DXR 1.1/SM6.6) or Vulkan on NVIDIA RTX; RTXPT 1.9 targets Agility SDK 1.619 + DXR 1.2 unless re-configured; CMake 4.1, VS 2022. Needed only to measure absolute real-time PT numbers (baseline for our "faster than any web PT").
- pbrt-v4 GPU (`--gpu`): CUDA 11+/OptiX 7.x/NVIDIA only; a pbrt CPU build runs anywhere (Mac fine, but slow for references).
- Cycles OptiX/CUDA/HIP/oneAPI/Metal: GPU numbers on the Windows box; CPU on Mac.
- HIPRT/Orochi: AMD GPU (or CUDA) on Windows/Linux.
- Mitsuba `cuda_ad_*`: NVIDIA; `llvm_ad_*` fine on Mac.
- OIDN native with CUDA/HIP/SYCL backends (Windows), Metal on Mac.
- Capsaicin (already local): Windows, D3D12, DXR.
- I built nothing and ran nothing (per brief), so no measured numbers; all speeds above are quoted from the cited sources.

## 8. Open questions
- Does Mitsuba 3's `path`/`volpath` and pbrt-v4's `volpath` agree exactly on rough dielectrics and rough conductors (pbrt includes energy-loss multiple scattering? both are single-scatter GGX, believed identical; verify numerically)?
- Which wavefront compaction scheme is fastest with WebGPU subgroup ops across Chrome/D3D12, Metal and Vulkan (subgroup size 4-128 varies)?
- Can we obtain a software `ray_query`-equivalent API shape compatible with Bevy Solari's shaders, to reuse their ReSTIR code with minimal porting?
- 64-bit atomics for hash caches are absent in WGSL: is the two-word CAS pattern robust enough (ABA, key collision on resize)?
- Licenses: confirm NVIDIA RTX SDKs License terms allow copying shader snippets (likely allows redistribution of source in "Licensed Software" with notice, not verified); until confirmed treat as ideas-only.
- Bitterli scene licenses vary per scene (mostly CC BY / CC0); check each before bundling in CI.
- Verify Rayzee, Stucki WebGPU PT and Khronos vk_gltf_renderer claims by reading code (not done here).
