# 11 - AMD Capsaicin: reference path tracer, GI-1.x, and a possible native GPU reference

Date: 2026-11-13. Clone: `research/sources/capsaicin/` (shallow, commit `914b915`, "AMD Capsaicin Framework v1.3", 2025-11-18,
git-ignored, read only, never built). Tags: **[V]** verified from the source or docs read this pass, **[P]** verified from the GI-1.0
paper text, **[K]** from memory, unverified.

## 1. TL;DR

- **Licence: MIT** (Copyright 2023 AMD) [V `LICENSE`]. Code may be ported with attribution (CLAUDE.md); add it to THIRD_PARTY.md when we do.
  Nothing NVIDIA-derived and no NRD in the tree [V]: `grep -i nvidia` hits one driver workaround (`getVendorId() == 0x10DE`) in
  `capsaicin_internal_scene.cpp`; no file or dependency is named NRD/RTXGI/SHaRC.
- **It is a D3D12 DXR 1.1 framework** (Windows 10 20H2+, DXC, DirectX Agility SDK, CMake 3.30) [V README]. It cannot run on the Mac.
  Its renderers are only two: "Reference Path Tracer" and "GI-1.2" [V `renderers/`].
- **The reference path tracer is small and clean (~550 lines of HLSL for the integrator) but it is a game-oriented "reference", not a
  pbrt-class one.** BRDF-only (no transmission, no volumes, no clearcoat), Schlick-Fresnel GGX + Lambert with Hammon diffuse
  compensation (not energy-conserving in the furnace sense), no multiple scattering, cubemap environment, Russian roulette that is not
  clamped. It is a useful third *cross-check on diffuse/metal opaque scenes and on a different API/GPU*, not a replacement for pbrt-v4 or
  Mitsuba (see section 3 and 7).
- **Headless linear HDR is possible but not clean.** `--benchmark-mode --benchmark-frames N` runs N frames, saves the last frame as
  **EXR** (tonemapping is switched off for the save) and a timings CSV, then exits [V]. But the window still opens (swap chain) and the
  saved image is the **R16G16B16A16_FLOAT "Color" shared texture**, so the dump is half-float, not the f32 accumulation buffer [V]. No PFM.
- **Real-time side:** GI-1.0/1.1/1.2 = 8x8 octahedral screen probes + world-space hash-grid radiance cache (distance-LOD cells, tiled, mipped)
  + world-space ReSTIR reservoirs for cache-vertex direct light, with a-trous "ratio estimator" denoising for glossy. Published numbers: GI-1.0
  total 1.93-3.12 ms at 1080p on Radeon RX 6900 XT, 1.91-3.43 ms on GeForce RTX 3080 [P]. No numbers for GI-1.1/1.2 on the GPUOpen page.

## 2. Sources

| name | link | licence | local | why |
|---|---|---|---|---|
| Capsaicin v1.3 | github.com/GPUOpen-LibrariesAndSDKs/Capsaicin | MIT (AMD) | `research/sources/capsaicin` | framework, reference PT, GI-1.2 |
| GI-1.0 report | gpuopen.com/download/publications/GPUOpen2022_GI1_0.pdf | AMD publication | link (text extracted to scratch) | algorithm, Table 1 timings |
| GI-1.1 paper | gpuopen.com/download/publications/SA2023_RealTimeReflection.pdf (SIGGRAPH Asia 2023 Tech Comm, Eto et al.) | ACM | link | glossy reflections with two-level caching |
| Bounded VNDF | Eto and Tokuyoshi, SIGGRAPH Asia 2023, doi 10.1145/3610543.3626163 | ACM | link | sampler used by default in the PT |
| Spherical-cap VNDF | Dupuy and Benyoub 2023, doi 10.1111/cgf.14867 | CGF | link | alternative in the PT |
| Unbiased back-facing VNDF | Tokuyoshi 2021 (GPUOpen) | AMD | link | PDF normalisation used in the PT |
| Stochastic (t,s) sequences | Helmer et al. 2021 | JCGT | link | the PT's sampler |
| CapsaicinTestMedia | github.com/GPUOpen-LibrariesAndSDKs/CapsaicinTestMedia | per-asset, **not read** | not cloned | test scenes and env maps, fetched by CMake [V `CMakeLists.txt:125`] |

### Licence register (item 1 of the brief)

| Component | Licence | Notes |
|---|---|---|
| Capsaicin source (`src/`) | MIT [V] | |
| `third_party/FidelityFX/` (GPU headers: core, spd, parallelsort, blur, lens) | MIT, header says "part of the FidelityFX SDK, Copyright 2024 AMD" [V] | vendored copy of AMD code, same licence |
| `third_party/gfx` (D3D12 abstraction, git submodule `gboisse/gfx`) | **not cloned** (shallow clone has no submodules); [K] MIT | it also holds the glTF/OBJ scene loader, so scene-import behaviour is unverified here |
| FetchContent deps: CLI11, yaml-cpp, nlohmann-json, meshoptimizer, FidelityFX-SDK | BSD / MIT [K], fetched at configure time, not in the tree | |
| gfx's deps: D3D12MA, DirectX Agility SDK, DXC, WinPixEventRuntime, imgui, stb, cgltf, glm, tinyobjloader, tinyexr, ktx, vulkan-headers | listed in `docs/development/getting_started.md` [V]; licences [K] (MIT/BSD/zlib/public domain; Agility SDK, DXC and PIX have Microsoft terms) | only matters if we ship binaries |
| `CapsaicinTestMedia` scenes + HDRIs | per asset, **unchecked** | must be read before we vendor or redistribute any of them. We only use our own scenes |
| NVIDIA / NRD / RTX SDK content | **none found** [V] | NRD files: none exist. Nothing to avoid opening |

## 3. Reference path tracer (`render_techniques/reference_path_tracer`, `ray_tracing/path_tracing.hlsl`)

All [V] unless marked. Options live in `reference_path_tracer.h`.

**Integrator structure.** A single compute (inline `RayQuery`, default) or DXR 1.0 pipeline (`reference_pt_use_dxr10`) kernel,
`[numthreads(4,8,1)]`, one thread per pixel, a megakernel loop `for bounce in 0..maxBounces` (`tracePath`). Per vertex: closest hit, add
emission (MIS-weighted), one NEE shadow ray, sample BRDF for the next direction, Russian roulette. `reference_pt_sample_count` (1-30 in the GUI)
paths per pixel per frame, averaged with a running mean (Welford) in a **f32** accumulation buffer `PT_AccumulationBuffer` (RGBA32F, `.w` =
sample count as uint bits). Accumulation stops and restarts on any camera/animation/mesh/light/size/option change.

**Camera and pixel filter.** Pinhole only (no lens, no DOF), vertical FOV, ray through `pixel + 0.5 + lerp(-0.5, 0.5, rand2)` = **box filter radius 0.5**,
matching our references' box 0.5 [V `path_tracing_shared.h`, `reference_path_tracer.comp`].

**Sampler.** Two generators [V `components/stratified_sampler`, `random_number_generator`]:
- `StratifiedSampler`: **stochastic Sobol (Helmer et al., hashed nested-uniform scrambled (t,s) sequence)**, index = frame*spp+sample, 64
  dimensions per pixel seed then re-seed by hash; used for pixel jitter, light-point sampling and BRDF direction samples. This is an
  Owen-scrambled-Sobol equivalent, progressive, per-pixel seed from a seed buffer.
- `Random`: plain PCG hash, used for light *selection* and Russian roulette. `randInt(max)` is `% max` (slightly biased, noted in the code).
- `BlueNoiseSampler` component exists but the reference PT does not use it.

**BSDF** [V `materials/*.hlsl`]. glTF metallic-roughness only: `F0 = lerp(0.04, albedo, metallic)`, `albedo *= (1 - metallic)`, alpha = roughness²,
Lambert + GGX (Trowbridge-Reitz NDF, height-correlated Smith visibility, **Schlick Fresnel with F90 = 1 evaluated at dot(H,V)**) + Hammon 2017 diffuse
compensation `(1-F)*1.05*(1-(1-HV)^5)`. Single scattering only: **no multiple-scattering compensation**, so rough metals lose energy as
in our pbrt/Mitsuba references (V12 shows tgpt/mutracer differ here, Capsaicin would agree with the references). **No transmission, no
dielectric glass, no clearcoat, sheen, volumes or subsurface**; back-facing shading normals are tolerated with `abs()` tricks. Sampling:
lobe chosen by `p = lum(F)/(lum(F)+lum(albedo*diffuseComp))` with the Fresnel evaluated at the *specular peak direction* (Frostbite
trick), then cosine hemisphere or GGX **VNDF; default is Bounded VNDF (Eto-Tokuyoshi 2023)**, selectable among Heitz full-sphere, Heitz upper,
spherical cap. PDF is the lerp of the two lobe PDFs (a proper mixture PDF, so MIS is consistent).

**Light sampling and MIS.**
- Lights: emissive triangles (each triangle a light, sampled **uniformly by area**, not solid angle; emission two-sided: "we support back face
  triangles"), point/spot/directional (KHR-style delta lights), environment map.
- Light *selection* is a component, switchable at run time: **uniform**, **grid-CDF** (16^3 world grid, per-cell CDF over the top-N lights,
  optional octahedral normal binning, cell lookup at a *jittered* position), **grid-stream** (reservoir-ish). `light_sampler_type` is an index
  into an `unordered_map` of registered names [V `factory.h`], default `1`, so **which sampler is the default is not stable by reading**: pin it
  and check the UI. The grid samplers look up the cell at a random jitter, so the PDF used for the MIS weight of BSDF-sampled hits is itself a
  stochastic estimate (small MIS bias, [V] by reading `sampleLightsPDF`).
- NEE: **one light, one shadow ray per vertex**, power heuristic MIS against BSDF sampling (`heuristicMIS`, power or balance by define),
  delta lights unweighted.
- Environment: a **cubemap** (`TextureCube g_EnvironmentBuffer`, equirect `.hdr`/`.exr` converted on load) sampled uniformly over the hemisphere
  (default `EnvironmentSamplingMode::Uniform`), cosine, or luminance-importance over cube mips. Default is uniform: **noisy on HDRIs; the importance mode
  must be selected** (`environment_sampling_mode=2`). The cubemap resolution is tied to the render size ("scale environment buffer to screen
  resolution", `bit_ceil`) [V `capsaicin_internal_scene.cpp:509-522`], so the environment is resampled, a bias source against our pbrt/Mitsuba env maps.
- No light BVH / light tree, no alias table.

**Russian roulette.** `if (bounce > minBounces) { q = max(throughput); if (q <= rand) terminate; throughput /= q; }`, `minBounces` default 2 [V].
**q is not clamped to 1.** If a throughput component exceeds 1 the path always survives but is divided by q > 1, which loses energy (a bias, small in
practice because VNDF weights are near 1 and albedos <= 1). RR can be turned off by setting `reference_pt_min_rr_bounces >= reference_pt_bounce_count` (the bounce loop ends first).
Default max bounces 30 (GUI range 1-30; `--render-options` has no validation) [V].

**Ray offsets.** The Wächter-Binder integer offset (`offsetPosition`: 256 ulps along the geometric normal, or 2^-16 near the origin where |p| < 1/32)
[V `geometry.hlsl:35`], from Ray Tracing Gems ch. 6 (a book; fine per CLAUDE.md). Shadow rays end at `1 - 2^-14` of the way to the light ("obtained empirically").
Rays below the geometric normal are discarded rather than offset.

**Accumulation and output.** Running mean in f32; **samples that are NaN, Inf or negative are dropped entirely** (the pixel keeps its history and the sample
count is not incremented), which is a (small, dark-side) bias. The displayed value goes to the shared `Color` texture, which is **half float** [V]. Tone mapping
runs after in the same renderer (`ToneMapping`, `AutoExposure`) and is disabled for the save; check `tonemap_enable` and auto-exposure do not touch the dump.
Alpha-tested materials use stochastic alpha (hashed alpha for blend mode).

**How "reference" is it?** Summary of known biases and shortcuts vs pbrt-v4/Mitsuba:
1. No transmission/glass, no volumes, no clearcoat/sheen: scenes with `furnace-dielectric` or glass cannot be reproduced at all.
2. Single-scatter GGX plus Hammon diffuse compensation: the metal furnace should match our single-scatter refs (0.69-class, see V12), but the
   diffuse+specular coupling is a game-style blend, not a layered BSDF; it will not match pbrt `coateddiffuse` and is a different model from
   Mitsuba `principled`. Pure Lambert and pure conductor scenes are the clean comparison.
3. Unclamped RR throughput division; NaN-dropping; stochastic light-selection PDF in MIS; `% max` light selection; environment as a render-size-dependent cubemap with uniform hemisphere sampling by default.
4. Half-float output; triangle-area (not solid-angle) light sampling is fine in expectation but noisy for large close lights.
5. RGB renderer (no spectral), like our own tracer and Mitsuba `llvm_ad_rgb`.
6. Material model is glTF-PBR: `KHR_materials_*` extensions support is not verifiable here [K] (scene loading is in the gfx submodule).

## 4. How it is driven

- **Scenes:** `.gltf`, `.glb`, `.obj` and a `.yaml` scene list [V `capsaicin_internal_scene.cpp:295`]; five bundled test scenes (BreakfastRoom, FlyingWorld,
  GasStation, TropicalBedroom, Sponza) and six HDR environment maps including `White.hdr` (a furnace-style uniform environment) from the separate CapsaicinTestMedia repo [V `main_shared.cpp:69-95`].
  External scene: `--load-external-scene <path>`; environment maps are chosen by index from that list or by dropping a `.hdr/.exr` on the window [V]. **No
  CLI option to load an arbitrary environment map file** [V by reading the option list]: a custom env needs a small patch (add a `--load-external-environment` that calls
  `setEnvironmentMap`) or replacing one of the bundled `.hdr` files on disk.
- **Camera:** from the glTF (`--start-camera-index`) or `--user-camera-position X Y Z --user-camera-lookat X Y Z`; FOV is vertical, aspect from the window [V].
- **Film size:** `--width`, `--height`, `--render-scale` [V]. The window is created at that size; a 512x512 film needs `--width 512 --height 512 --render-scale 1`
  (window-system limits on a small window are a possible trap [K]).
- **Settings:** `--render-options "reference_pt_bounce_count=8 reference_pt_min_rr_bounces=8 reference_pt_sample_count=16 light_sampler_type=N environment_sampling_mode=2 tonemap_enable=false"`; `--start-renderer-index` to pick the PT (`--list-renderers`) [V].
- **spp control:** per-frame `reference_pt_sample_count` x `--benchmark-frames` frames = total spp. Frame index is the sequence index, so results are deterministic given a fixed frame count [V by reading].
- **Output:** `--benchmark-mode --benchmark-frames N --benchmark-first-frame T [I] --benchmark-suffix S --dump-folder D --benchmark-capture-frames true --benchmark-capture-timings true`
  writes `<scene>_<suffix>_<frame>_<avgFrameTime>.exr` (half float by the `Color` texture; PIZ compression, channel order BGR handled in `saveEXR`) and a per-pass CSV with
  `Name,Average,Min,Max,Accumulated,Frame...` [V `main_shared.cpp:1830-1900`]. Run exits itself.
- **Headless?** There is a window and a D3D12 swap chain; no off-screen mode. On the maintainer's laptop that is fine (run it with the window visible). A scripted run is a PowerShell loop.
- **Verdict:** it can produce linear HDR **EXR (f16)** for a camera/scene/bounce/spp combination from a CLI. Needs: an EXR-to-PFM converter in our bench (small), a
  tiny patch or a file swap for custom HDRIs, and, for an unbiased reference, exposing the f32 accumulation buffer (a ~10-line change: register it as a shared
  texture). Without the patch f16 quantisation (~1e-3 relative) is below our reference floor (4e-3 relMSE, P0.27) but not negligible for relMSE targets near 1e-3.

## 5. Real-time techniques worth comparing with our P4 plans

### 5.1 GI-1.0 / 1.1 / 1.2 (`render_techniques/gi1`, ~9000 lines)

Pass order (from the `TimedSection` names in `gi1.cpp`) [V]: `PurgeRadianceCache` -> `ReprojectScreenProbes` -> `LookupScreenProbes` -> `SpawnScreenProbes` -> `PatchScreenProbes`
-> `SampleScreenProbes` -> `PopulateScreenProbesCells` (trace) -> `PopulateMultibounceCells` -> world-space ReSTIR (`GenerateReservoirs`, `GenerateMultibounceReservoirs`,
`CompactReservoirs`, `ResampleReservoirs`) -> `PopulateRadianceCache`/`UpdateRadianceCache`/`UpdateMultibounceCells`/`ResolveCells` -> screen probe `Blend`, `Reorder`, `Filter`,
`Project` (to SH) -> `InterpolateScreenProbes` -> glossy `TraceReflections` (half res) + `MarkFireflies`/`CleanupFireflies` + a-trous `ResolveReflections` -> `ReprojectGI`/`FilterGI`/`ResolveGI1`.

**Level 1, screen probes.** 8x8 octahedral probes placed one per 16x16 pixel spawn tile by default (`kSamplingMode_QuarterSpp`), probe position is a *random pixel in the tile*, reprojected
from the previous frame, holes filled by stochastic ray rebalancing (a fixed ray budget is moved from reprojected tiles to disoccluded ones), "ray guiding" (previous radiance as a PDF over octahedral cells,
jittered inside the cell), adaptive cell size by distance, probe masking, then SH projection and per-pixel interpolation + a spatiotemporal denoiser whose filter radius follows the history length [P, V].

**Level 2, hash-grid radiance cache** (`hash_grid_cache.hlsl`) [V]:
- Key = (log2 LOD of the cell size, which grows with distance from the **camera**, `cell_size_step = max(dist(eye,hit)*cell_size, min_cell_size)` rounded to a power of two) + world tile coordinate
  + **direction bucket (`floor(0.5 + (0.5 d + 0.5) * 4)` per axis)** + a flag "hit distance < tile size". Bucket by PCG hash, checksum by xxHash in a bucket of `num_tiles_per_bucket` (16) tiles; insert with `InterlockedCompareExchange`.
- Each tile is 8x8 cells x 4 mip levels of a **2D image of the surface patch**, the 2D offset is the 3D cell coordinate projected along the dominant axis of the direction (surfaces look planar at tile scale).
- Cell values are updated with **integer atomics on quantised floats (factor 1e3)** into `uint2` (radiance + sample count), temporal accumulation capped at `max_sample_count` (16).
- Lookups **climb mips when the cell's sample count is below the cap**, the sparse-cell fallback.
- Tiles are evicted after **50 frames unused** (`kHashGridCache_TileDecay`) and returned to a pool; `PurgeTiles` compacts.
- **Radiance feedback / multibounce:** a ray's secondary hit is looked up in the cache, so bounces accumulate over frames; a fraction (0.7) of multibounce rays is discarded each frame; the hit "radiance feedback" is the paper's way to get infinite bounces.
- Visibility records (`instance, primitive, barycentrics, front_face`) are stored per cell update (`HashGridCache_VisibilityBuffer`) and then *shaded in a separate pass* (compaction before shading).

**World-space ReSTIR** (`world_space_restir.hlsl`) [V]: reservoirs live in a second **hash grid of cells at distance-LOD (cell size 16)**, RIS with 8 samples (`kReservoir_SampleCount`), temporal reuse via the previous frame's hash grid (the previous
eye is used for the previous cell size), spatial reuse by stochastically iterating the neighbour cell's entries with a squared bilateral weight, spatial jitter sqrt(2). It resamples *direct lighting at cache vertices* (and a "multibounce" variant), not
screen-space GI; one shadow ray per reservoir after resampling and visibility is skipped in the resampling weights, which the paper says adds bias [P].

**Glossy reflections (GI-1.1)** [V, P]: half-res trace by default, a-trous "ratio estimator" (default denoiser mode 1) with 4 passes and roughness-driven radius (low/high roughness thresholds 0.2/0.6), explicit firefly marking and cleanup passes, NaN removal by a tone-compress/expand (`c/(1+c)`, saturate, inverse).
`ratio estimator` = denoise the ratio of (light) / (cheap probe estimate) rather than the radiance itself, so the cache supplies the low-frequency part [K for the exact formulation].

**Denoisers.** Not a standalone library: (a) the screen-probe/GI denoiser `gi_denoiser.hlsl` (temporal accumulation with a per-pixel blur radius mask, dilated disocclusion mask, `kGIDenoiser_MaxBlurMask = 8`), (b) the glossy a-trous ratio filter, (c) `variance_estimate` (a metric helper, not a denoiser). FSR (FidelityFX upscaling) is built in as a technique. There is **no SVGF, ReBLUR or OIDN** in the tree [V].
**Other techniques in tree** [V]: `ssgi` (horizon-based bent normals/AO), `atmosphere`, `skybox`, `bloom`, `lens`, `auto_exposure`, `tone_mapping`, `image_metrics` (compares frames, useful as a reference harness), `visibility_buffer` (mesh-shader V-buffer, with an RT path and a DXR 1.0 debug path), `combine`.

**Published timings.** GI-1.0 total time at 1080p [P Table 1; the figure caption and text give "1080p on Radeon RX 6900 XT"; the table itself does not repeat the resolution]:

| scene (triangles) | RX 6900 XT total | RTX 3080 total | RX 6900 XT trace / cache+sample / interpolate+denoise |
|---|---|---|---|
| Gas station (99k) | 1.932 ms | 1.906 ms | 0.235 / 1.032 / 0.665 |
| Flying world (267k) | 2.463 ms | 2.326 ms | 0.532 / 1.226 / 0.705 |
| Kitchen #1 (1.4M) | 2.734 ms | 2.976 ms | 0.336 / 1.633 / 0.765 |
| Kitchen #2 (9.0M) | 3.124 ms | 3.426 ms | 0.676 / 1.693 / 0.755 |

Figure 1 says 3.5 ms (Kitchen) and 4.2 ms (Sponza) *including direct lighting*. The GPUOpen page gives no GI-1.1/1.2 timings [V web]. The numbers are DXR 1.1 hardware RT on RDNA2/Ampere desktop cards, not comparable to a laptop RTX 3060 or to software traversal; only our paired ratios on one machine count.

## 6. Architecture ideas for WebGPU

| Idea | Verdict | Where it lands |
|---|---|---|
| Distance-LOD hash-grid keys (cell size = f(distance to camera), power-of-two LOD in the hash) | **portable to WGSL** | `@pmndrs/gi` world cache (P4); same family as SHaRC, already in the plan |
| Tile of 2D cells keyed by dominant-axis projection + 4-mip pyramid and mip-climb fallback when sample count is low | **portable** | cache read path; cheap, avoids needing 3D direction bins |
| Integer-atomic accumulation of quantised floats into `uint2` (1e3 scale) | **portable** (WGSL has u32 atomics only, no 64-bit; use two u32 atomic adds, radiance packed, count separate) | cache update; verify overflow bounds |
| `InterlockedCompareExchange` bucket insert with checksum, tile decay of 50 frames, pool free list | **portable** (`atomicCompareExchangeWeak`) | cache; mind WGSL uniformity rules (see CLAUDE.md traps) |
| Visibility record buffer, compact, then shade in a separate dispatch | **portable**, and good for us (wavefront-friendly) | cache update, wavefront path |
| Ray guiding: last-frame probe radiance as a per-cell PDF, jitter inside the cell, redistribute a fixed ray budget from reprojected tiles to holes | **portable** | probe atlas (DDGI/RC side of `@pmndrs/gi`) |
| 8x8 octahedral screen probes, random pixel in the tile, reprojection + SH projection | **portable** | screen-space GI consumer |
| World-space ReSTIR reservoirs in a hash grid, temporal reuse through the previous frame's grid keyed with the previous eye | **portable** (needs 4-u32 reservoirs; nothing RT-specific except one shadow ray) | ReSTIR DI/GI prototype; compare with Bevy Solari's world cache |
| Bounded VNDF sampling (Eto-Tokuyoshi) | **portable**, pure math, small | `ptSampleGGX`; A/B against Heitz 2018 and spherical cap on noise at equal spp |
| Stochastic Sobol (Helmer) sampler, 64 dims then re-seed | **portable** (needs a 32x64-entry xor table in a buffer) | A/B against Owen-Sobol (research note 04) and ZSobol; it is a table-based alternative worth a bench |
| Wächter-Binder integer ray-origin offset | **portable** (`bitcast<i32>`); already common | `rt` core; check it against the bvh watertight oracle |
| Camera-distance-adaptive probe/cell size, a-trous ratio estimator for glossy, firefly mark and cleanup | **portable** | `@pmndrs/reflections`, Denoiser |
| Inline `RayQuery` for the PT; DXR 1.0 shader tables, SBT, hit-group local roots | **needs DXR/inline RT** | not applicable (our traversal is software); the megakernel loop itself is portable |
| Mesh-shader V-buffer (`visibility_buffer.mesh/.task`), FSR through FidelityFX headers | **needs mesh shaders** (not in WebGPU) / **portable** for FSR (ours exists) | not applicable |
| Bindless `Texture2D g_TextureMaps[]` | **needs bindless**; we have the `TextureProvider` abstraction | already planned |

## 7. Comparison

| | pbrt-v4 | Mitsuba 3 | three-gpu-pathtracer | mutracer | **Capsaicin ref PT** |
|---|---|---|---|---|---|
| Runs on | CPU (+OptiX) | CPU/LLVM (+CUDA) | WebGPU | WebGPU | **D3D12 DXR GPU (Windows)** |
| Spectral | yes | optional | RGB | RGB | RGB |
| Materials | full | full | glTF + extensions | glTF + extensions | **metallic-rough only, no transmission** |
| Multiple scattering | no (single) | no | yes (V12) | yes (V12) | **no** (Hammon diffuse comp only) |
| Sampler | ZSobol | multijitter | blue noise / PCG [K] | PCG/blue noise [K] | **stochastic Sobol (Helmer)** + PCG |
| Light selection | power / BVH | uniform | uniform/ MIS | 4 RIS candidates (V13) | uniform / grid-CDF / grid-stream, one light per NEE |
| Environment | equal-area oct. | equirect bilinear | equirect importance | | **cubemap, uniform by default** |
| RR | disableable (patch V4) | disableable | | | disableable by `min_rr = max bounces` |
| Output | EXR f32 | EXR/PFM f32 | in-page | in-page | **EXR f16 from the display texture** |
| Role for us | ground truth | cross-check / default | competitor | competitor | **GPU-API-independent cross-check on opaque scenes; real-time yardstick** |

What it adds that the others do not: (1) a **third independent implementation** of diffuse+rough-metal transport on a GPU, on a different API, with a different sampler and different RR,
useful to localise bias (if pbrt, Mitsuba and Capsaicin agree on Cornell and our tracer deviates, it is ours); (2) a **native GPU timing baseline for a reference-class PT** on the same 3060 as our W6
numbers (inline RT on RT cores: it is the speed we will never reach in software, but it frames the gap); (3) the only **source-available native hash-grid cache + ReSTIR + screen-probe stack** in our set that is permissively licensed (RTXGI/SHaRC/RTXDI are "ideas only"), so ports are allowed with attribution;
(4) the only per-pass **timing CSV** from a CLI-run native real-time GI.
What it cannot do: be adjudicator (pbrt-v4 stays), cover glass/coated/volume scenes, or run on the Mac.

## 8. Suggestions (for DECISIONS; nothing decided here)

**Should Capsaicin become a third reference renderer (D15 names pbrt + Mitsuba)?**
Suggestion: **no as a reference, yes as a "native GPU cross-check"** limited to the opaque Tier 0 scenes (`furnace-diffuse`, `furnace-conductor` single-scatter, `quad-light`, `cornell`, `sponza-white`),
labelled `capsaicin-…` in `bench/references`, never used as the default for error metrics, and not part of the 6-scene agreement table until it agrees with pbrt on Cornell.
Pros: independent bias check on another GPU/API; supplies a native GPU time-to-quality baseline on the 3060; cheap (a task on a machine that already has it built); MIT so findings can seed ports.
Cons: no transmission, f16 dump, env is a render-size cubemap, default light sampler and env sampling need pinning, window-bound CLI, a one-person-maintained Windows-only run. Mixed biases (unclamped RR, NaN drop) mean a disagreement does not prove *our* error.
The A/B that settles it: run W9 on Cornell and `sponza-white` at 1024-4096 spp, compute relMSE and relative bias against pbrt and Mitsuba (`references:compare`). Promote to a listed reference if mean relative bias is below ~0.3% on both scenes and the pbrt-Mitsuba floor (4e-3) is not exceeded;
otherwise keep it as a timing baseline only.

## 9. Draft Windows task W9 (for `docs/WINDOWS-HANDOFF.md`; not merged there)

> | W9 | **Capsaicin as a native GPU cross-check and timing baseline (P0.31)** | see below | `bench/references/<scene>/capsaicin-<spp>spp.pfm` + `.json` for the opaque scenes, a comparison table, and a timing CSV on the 3060 |

**Prerequisites.** Capsaicin v1.3 (commit `914b915`, or the build already on the laptop: record its commit) built RelWithDebInfo/Release, `CapsaicinTestMedia` fetched.
Quiet machine rules as in W6: plugged in, other GPU apps and browsers closed, cool card, `nvidia-smi` clocks and temperature before and after each run recorded in the JSON `host`. The Capsaicin window must stay visible and unobstructed during runs.
Never use Capsaicin's own scenes for matched references; use ours.

**1. Export our scenes.** `pnpm --filter @pathtracer/bench adapters:export` writes `bench/assets/generated/web/<scene>.glb` (and `.json` with `approximations`) for every Tier 0 scene and `sponza-white`.
Use it for `cornell`, `furnace-diffuse`, `furnace-conductor`, `quad-light`, `sponza-white`. Skip `furnace-dielectric` (Capsaicin has no transmission; glTF export turns it into transmission+IOR, DIVERGES V7). `env-sphere` and `env-sky` need the environment patch below, so do them last.
Approximations to write into the JSON sidecar `approximations` (all from V7 and the Capsaicin material model):
- conductors become metallic with F0 from eta/k, F90 = 1 Schlick (pbrt/Mitsuba use full conductor Fresnel): expect a small colour difference on `furnace-conductor`;
- diffuse gets `specular` 0 in our export, but Capsaicin always has a 0.04 dielectric specular layer on non-metals plus Hammon diffuse compensation: **expect Capsaicin diffuse to read a few percent off the pure-Lambert references**, record the `furnace-diffuse` value as the first measurement and decide whether it is "model difference" (DIVERGES) or a bug;
- emission becomes `emissiveFactor` on a mesh and is two-sided in Capsaicin even where our scene is one-sided (V7, V11): for `quad-light` and `cornell` use the same one-sided-twin trick as tgpt (black back face) or accept and record it;
- the sun in `sponza-white` is a glTF `KHR_lights_punctual` directional light (V19): confirm Capsaicin imports it (loader is in the gfx submodule, unverified) and that lux/intensity matches by comparing the image mean with pbrt (<0.05% target, as for tgpt); if not, set the sun by hand and record the factor;
- the environment is only in glTF `extras`: Capsaicin takes an env map from its own list. Either replace the bundled `White.hdr` / one HDR on disk with our `bench/assets/generated/...` equirect (same file, copied under another name) and pick it with `--start-environment-map-index`, or add a one-line `--load-external-environment` that calls `CapsaicinInternal::setEnvironmentMap` (new local patch; record it in DIVERGES as `platform`, and keep the diff in `bench/references/capsaicin-env.patch`). Note the render-size cubemap and importance sampling (V9-like difference: nearest vs bilinear lookups).

**2. Render matched references.** One run per scene, from the Capsaicin folder (command shape; confirm flags with `--help`):
```
SceneViewer.exe --render-options "reference_pt_bounce_count=<maxDepth> reference_pt_min_rr_bounces=<maxDepth> reference_pt_sample_count=16 light_sampler_type=<pinned> environment_sampling_mode=2 tonemap_enable=false"
  --load-external-scene <repo>\bench\assets\generated\web\<scene>.glb --start-renderer-index <Reference Path Tracer idx>
  --width 512 --height 512 --render-scale 1 --user-camera-position ... (only if the glb camera is not imported)
  --benchmark-mode --benchmark-frames 256 --benchmark-capture-frames true --benchmark-capture-timings true --benchmark-suffix cap-<scene> --dump-folder <out>
```
- **Camera/film:** the glb carries the camera (our exporter writes `lookAtCameraToWorld`); Capsaicin's FOV is vertical and its aspect comes from the window, so use the scene's film size (512x512 for Tier 0; the `sponza-white` size from its scene file). If the camera is not imported use `--user-camera-position/--user-camera-lookat` with the scene's values (our scenes are right-handed, +Y up, metres, same as glTF).
- **Bounces:** `bounce_count` = our `maxDepth` (Cornell: 32; Capsaicin's GUI caps at 30, the option itself is not validated; confirm 32 takes effect, else use 30 and record it). Capsaicin emission is added at the last vertex, so check bounce semantics with `quad-light` (maxDepth 1) against the analytic image before trusting any other scene, exactly as for pbrt/Mitsuba (README "Depth"). Record the mapping in DIVERGES like V10.
- **RR off** for references: `reference_pt_min_rr_bounces = reference_pt_bounce_count`. For `sponza-white` use the same `rrStartDepth` semantics as the other references: Capsaicin RR fires when `bounce > minBounces`.
- **spp:** `reference_pt_sample_count x --benchmark-frames` = 16 x 256 = 4096 spp (1024 for Sponza, to match the pbrt budget). Frame index drives the Sobol index, so a rerun is deterministic.
- **Light sampler and environment sampling:** pin `light_sampler_type` explicitly, look at the UI once to learn which sampler the index is, and record it. Use `environment_sampling_mode=2` (importance) for env scenes; uniform is the default and is far noisier.
- **Output:** the run writes an EXR (f16) plus the timings CSV. Convert the EXR to PFM (float32, linear, row 0 = top: **check the row order and channel order**, the dump writes B,G,R channels) with a small script and save as `bench/references/<scene>/capsaicin-<spp>spp.pfm` with the usual JSON sidecar: `renderer: "capsaicin-d3d12-rt"`, `version: "Capsaicin 1.3 914b915"`, `spp`, `seconds` (sum of frame times from the CSV), `command`, `settings` (bounces, `rr: false`, sampler "stochastic-sobol", `lightSampler`, `envSampling`, `film`, `filter: "box 0.5"`, `sensor: "n/a"`, `output: "exr-f16"`), `approximations` (the list above), `host` (GPU name, driver, `nvidia-smi` clocks), `date`.
- If the f16 quantisation matters for a scene (relMSE targets below ~2e-3), apply the 10-line local patch to expose `PT_AccumulationBuffer` as a shared texture and dump that (record as `platform` in DIVERGES).

**3. Compare.** Copy the PFMs next to the Mac references (the `compare.ts` picks the highest-spp file per renderer id with the pattern `^(.+)-(\d+)spp\.pfm$`, so the id is `capsaicin`; if it does not name the file as `capsaicin-<spp>spp.pfm`, adjust):
`pnpm --filter @pathtracer/bench references:compare -- --write`.
It reports furnace/analytic checks per renderer and agreement against pbrt-v4. Add Capsaicin to the Mitsuba column manually if `compare.ts` only hard-codes pbrt-v4 as the reference (see below). Record relMSE, SMAPE, relative bias, and RGB bias for Capsaicin vs pbrt and vs Mitsuba on each scene in `bench/references/agreement.md`. Expected: diffuse/furnace within a few tenths of a percent after the 0.04-specular note above; rough conductor near our single-scatter value (0.69-class at roughness 1, V12) rather than 1.0.

**4. Time it on the 3060.** Reference path tracer: the CSV from step 2 gives per-pass ms per frame; report ms per sample (frame time / `sample_count`), Mpath/s, and time-to-equal-error against the same pbrt reference (1/error crossing, as `bench/adapters/baseline.ts` does for the web renderers). Use `--benchmark-first-frame <T>` so the first frames (shader compile, AS build) are excluded; run 3 times ABBA against our own `matrix.ts` run on the same scene, and quote **paired ratios** only. `--benchmark-capture-timings true` for the pass breakdown.
Real-time mode: `--start-renderer-index <GI-1.2 idx>` on `sponza-white` (our export, with the glTF sun) and on Capsaicin's own Sponza, 1080p and 720p, `--benchmark-frames 600 --benchmark-first-frame 100` with a camera animation or a fixed camera; record the `GI1` pass-by-pass table from the CSV (names in section 5.1) as the native yardstick for P4 (`@pmndrs/gi`). Disable vsync (Release build) and note the build config. Log `nvidia-smi` clocks; rows from a throttled run are `provisional`.

**5. What an adapter or exporter change on our side needs.** Minimal, and only when W9 starts:
- `bench/references/render.ts`: no change if the PFM and JSON are made by hand/script. To make it repeatable add `capsaicin` as a renderer key that shells out to the script from step 2 (Windows only), same sidecar writer.
- `bench/references/compare.ts`: confirm it accepts renderer id `capsaicin`; add it to the agreement columns for Mitsuba as well as pbrt (the agreement is pbrt-referenced today).
- `bench/exporters/gltf.ts`: for Capsaicin-bound exports set `KHR_materials_emissive_strength`-free emission and `doubleSided: true`; add a `target: 'capsaicin'` option that (a) writes the black-back-face emitter twin used for tgpt, (b) skips `furnace-dielectric`, (c) writes the camera FOV and aspect. Record these as new `approximations` entries.
- A small EXR (f16/f32) to PFM converter in `bench/metrics/` if none exists (tinyexr output is PIZ-compressed; a Python `OpenEXR`/`imageio` one-liner on Windows is enough).
- No change to `@pmndrs/raytracer`.
