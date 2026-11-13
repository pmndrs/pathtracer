# 08 - Real-time GI and reflections as consumers of a shared tracing core

Date: 2026-11-13. Author: research agent 08. Scope: what each real-time GI / reflection / AO / shadow / baking technique needs from a tracer, and the layered API + shared caches that follow from that.
Local context read (not edited): `webgpu-ddgi` (SPEC/PLAN/PROGRESS), `ddgi-research` (landscape, gi-1.0 notes), `reflections` (README, docs/TRIAGE.md), `gi-school`, `~/Developer/Capsaicin` (gi1 render technique).
Verification key: **[V]** = read in a saved PDF/page or local source this session. **[M]** = from memory/prior knowledge, not re-verified; treat as a lead.

## 1. TL;DR

- Every real-time GI method is a **cache + a small number of incoherent rays that feed the cache**. The cache (probe grid, hash grid, surfels, screen probes, cascades) is what is shown; rays only refresh it. So our core's most valuable public surface is not "full path" but **closest-hit + minimal surface record**, plus **a one-bounce shade** that reads a cache at the hit.
- Ray budgets are tiny: Lumen High = 1/16 ray per pixel final gather, Epic = 1/4 rpp [V, SIGGRAPH 2022 Lumen]; AMD GI-1.0 runs at about 1/4 spp-equivalent through screen probes [V]; DDGI uses 64-512 rays per *probe*, a few thousand probes (about 0.03-0.3 rays/pixel equivalent) [V/M]; SHaRC updates on ~4% of pixels (1 of every 5x5) [V]. Plan **0.05-0.5 traced rays per pixel per frame for GI, +0.25-1 for reflections** at 1080p/60. At 1080p that is roughly 0.1-1 Mrays/frame GI (6-60 Mrays/s) - achievable in software WebGPU on a mid-range GPU if closest-hit is ~100-300 Mrays/s incoherent; this is the number to benchmark first.
- Almost all of them **tolerate approximate geometry** and **stale/approximate shading at hits** (that is the point of a cache). Only reflections (mirror/low roughness) and shadows need exact-geometry hits. API must expose a **"proxy/LOD geometry + cheap shading" mode**, not just "exact".
- Two caches show up everywhere and should be **first-class shared resources** owned by the core: (a) a **world-space spatial hash radiance cache** (SHaRC / GI-1.0 hash grid / Binder path-space filtering / Boisse world-space ReSTIR / Split RC all use one), (b) a **probe atlas** (DDGI irradiance+visibility octahedral textures; Lumen world radiance cache probes; radiance cascades probes). Both are also useful to the full path tracer (cache-terminated paths for a fast preview; NRC-like early termination).
- **WebGPU constraints that shape the caches:** no hardware ray queries, no 64-bit atomics in WGSL (only 32-bit atomic u32/i32) **[M, verify against current WGSL spec]**, so the hash design must use 32-bit keys/checksums + atomicCompareExchange on 32-bit words (SHaRC's 64-bit path is an optional define; its fallback uses a lock buffer [V]). A WebGPU SHaRC re-implementation already exists as a community project (Usnul, "Shade", three.js forum "Shark!") [V], showing it is feasible; it chose 2^22 cells, SH2 irradiance (view-independent), YCoCg f16, 12 B/cell, and eviction instead of resolve+compact.
- Radiance Cascades is moving into 3D world-space in 2025-26 (Split RC, arXiv 2607.20384: sparse hash of world probes, rays traced *from visible surfaces* and split into intervals; 13.2 ms single-frame at San Miguel on RTX 3080 Laptop of which 6.1 ms is 1 spp raytracing; table: Sponza 8.6 ms total, trace 1.3 ms) [V]. It needs only closest-hit with **ray intervals (tmin, tmax) and distance-weighted deposit** - the tracer must support `tmin>0` and return `t`.
- Recommendation: the core exposes 4 layers (visibility, closest-hit-minimal, one-bounce-shade, full-path) over **one traversal kernel/inline function and one wavefront scheduler**, plus two cache modules (HashCache, ProbeAtlas). Details in section 5.

## 2. Sources

| Name | Link | Year | License / venue | Local path | Why |
|---|---|---|---|---|---|
| Radiance Cascades (Sannikov) | https://github.com/Raikiri/RadianceCascadesPaper | 2023 | paper source (tex), community | not saved (no PDF) | origin; 2D, penumbra-hypothesis interval merging |
| Holographic Radiance Cascades | https://arxiv.org/abs/2505.02041 | 2025 | arXiv (2D), 1.85 ms 512x512 RTX 3080 Laptop [V via search snippet] | `research/papers/gi/holographic-radiance-cascades-2505.02041.pdf` | single-shot scene-agnostic 2D RC; trace structure |
| Split Radiance Cascades | https://arxiv.org/abs/2607.20384 | 2026 | arXiv CC BY-SA 4.0 | `research/papers/gi/split-radiance-cascades-2607.20384.pdf` | 3D RC with sparse hash world probes, tracing from visible surfaces |
| Analyzing the Performance Scaling of RC (Dell'Ova) | https://www.diva-portal.org/smash/get/diva2:2007296/FULLTEXT01.pdf | 2025 | MSc thesis | `research/papers/gi/dellova-rc-perf-thesis-2025.pdf` | 3D RC cost scaling |
| RCPG: product path guiding using RC | https://dl.acm.org/doi/10.1145/3840291 | 2026 | ACM TOG | link only (ACM, not OA-confirmed) | RC as guiding field for a path tracer: a reuse path for our full PT |
| TypeGPU radiance-cascades | https://docs.swmansion.com/TypeGPU/ecosystem/typegpu-radiance-cascades/ | 2025-26 | WebGPU, SDF tracing, holographic runner [V via snippet] | link only | only found WebGPU RC implementation; SDF-traced |
| three-rc (CodyJasonBennett) | https://github.com/CodyJasonBennett/three-rc | 2025 | "All rights reserved" - **reference only, no code reuse** | link only | 3D RC, screen-space probes, three-mesh-bvh; appears WebGL, no perf numbers [V] |
| DDGI | https://jcgt.org/published/0008/02/01/ | 2019 | JCGT open | `papers/gi/majercik-ddgi-jcgt-2019.pdf` | probe grid irradiance + moment visibility |
| DDGI scaling for production | https://jcgt.org/published/0010/02/01/ | 2021 | JCGT open | `papers/gi/majercik-ddgi-scaling-jcgt-2021.pdf` | relocation, classification (Off/Sleeping/Vigilant), 30-50% probe sleeping saving, cascades |
| DDGI Resampling | https://arxiv.org/abs/2108.05263 | 2021 | arXiv | `papers/gi/ddgi-resampling-2108.05263.pdf` | reservoir-guided probe rays |
| Adaptive DDGI | https://arxiv.org/abs/2301.05125 | 2023 | arXiv | `papers/gi/adaptive-ddgi-2301.05125.pdf` | variable ray allocation per probe |
| NVIDIA SHaRC | https://github.com/NVIDIA-RTX/SHARC (docs/Integration.md) | 2024 | NVIDIA SDK license (check; shader-only source) - **ideas only** | link only | hash-grid radiance cache, update on ~4% of pixels |
| NRC (Mueller et al.) | https://arxiv.org/abs/2106.12372 | 2021 | SIGGRAPH | `papers/gi/nrc-mueller-2021-real-time-neural-radiance-caching-2106.12372.pdf` | tiny MLP trained online on 1/64 paths; shorter paths terminated at cache |
| Binder et al. path-space filtering | https://arxiv.org/abs/1902.05942 (massively parallel); original talk SIGGRAPH 2018 | 2018/2019 | OA arXiv | `papers/gi/massively-parallel-path-space-filtering-1902.05942.pdf` | jittered spatial hashing; the algorithmic root of all hash caches |
| World-space spatiotemporal reservoir reuse (Boisse) | https://gpuopen.com/download/SA2021_WorldSpace_ReSTIR.pdf | 2021 | SIGGRAPH Asia TC | `papers/gi/worldspace-restir-boisse-2021.pdf` | reservoirs stored in a GPU-built hash grid (author is Boisse, not Gautron; Gautron's related work is spatial-hash AO) |
| AMD GI-1.0 | https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf | 2022 | AMD, Capsaicin MIT | `papers/gi/gi-1.0-gpuopen-2022.pdf`; code `~/Developer/Capsaicin/src/core/src/render_techniques/gi1/` | screen probes + hash-grid cache + world-space ReSTIR |
| GI-1.1 / GI-1.2 | https://gpuopen.com/learn/gi-1-1-glossy-reflection-rendering/ , .../gi-1-2-multibounce-indirect-rendering/ | 2024/2025 | blog | link only | glossy via GGX rays, roughness switch between screen-probe interpolation and RT + hash cache; multibounce via second-level cells |
| Lumen (SIGGRAPH 2022) | https://advances.realtimerendering.com/s2022/ | 2022 | Epic course notes | `papers/gi/lumen-siggraph2022-wright.pdf` | surface cache, screen probes 8x8, product importance sampling, SW vs HW RT cost tables |
| EA GIBS 2 (Frostbite, 2024) | https://advances.realtimerendering.com/s2024/content/EA-GIBS2/Apers_Advances-s2024_Shipping-Dynamic-GI.pdf | 2024 | course notes | `papers/gi/ea-gibs2-frostbite-dynamic-gi-2024.pdf` | surfels + probes in shipped title |
| ReSTIR GI (Ouyang) | https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing | 2021 | HPG | link only (direct PDF URL did not resolve) | per-pixel indirect with spatiotemporal reservoir reuse of a secondary vertex |
| Voxel cone tracing (Crassin) | https://www.icare3d.org/research-cat/publications/interactive-indirect-illumination-using-voxel-cone-tracing.html | 2011 | I3D | link only | cone tracing through a voxel mip pyramid |
| FidelityFX Brixelizer GI | https://gpuopen.com/fidelityfx-brixelizer/ | 2023-24 | MIT SDK | link only | sparse SDF cascades + GI built on GI-1.0 ideas; software-traversal reference |
| Capsaicin GI-1.x (local) | `~/Developer/Capsaicin` | 2023-25 | MIT | local | read-only reference of cache/ray usage |

## 3. Techniques: what each needs from the tracer

Legend for tracer needs: **VIS** = any-hit visibility only; **CH** = closest hit (t, prim, uv); **CH+ALB** = + albedo/emission/normal; **SHADE1** = hit + direct light at hit (+cache lookup); **PATH** = multi-bounce. "Coherence" = whether rays in a wave share origin/direction.

### 3.1 Radiance Cascades (2D origin; 3D / world-space variants)

- **What:** hierarchy of probe grids; probe spacing grows by 2x per cascade while angular resolution grows 4x (2D) so cost per cascade is constant. Each probe traces **short intervals [t_n, t_{n+1}]**; intervals are merged top-down to form full radiance at cascade 0. Exploits the penumbra condition: far rays need low spatial res, near rays need high.
- **3D status [V]:** Split RC (2026) stores probes in a **sparse world-space hashmap**, spawns rays from **visible surfaces** (not from probe positions), then **splits each ray into intervals deposited into every cascade** so total rays = 1 spp-equivalent instead of rays x cascades. Uses R2 sequence per-probe-segment for even angular coverage; cascade data stored contiguously and merged with wave shuffles; final shading by baking a 6x6 octahedral irradiance map per probe (+1 texel border) then ≤8 filtered samples per pixel instead of 256. Performance on RTX 3080 Laptop: Sponza 8.6 ms (init 1.2, rays 0.4, trace 1.3, deposit 2.2, merge 1.0, shade 2.5), San Miguel 11.5 ms (trace 4.5: foliage = 3x slower). Multibounce via secondary off-screen cache from last frame's hit points. Other 3D variants named in it: Mathis 2024 (probes in 2D texture), Surfel RC (Coppen 2025), "Love and Deepspace" mobile use (Ruan 2026). Holographic RC (2025) is 2D single-shot, 1.85 ms at 512x512. RCPG (2026) uses world-space RC cascades as a path-guiding field for path tracing.
- **Community WebGPU:** TypeGPU `@typegpu/radiance-cascades` (SDF tracing + holographic runner); many 2D WebGPU/WebGL tutorials. No public web 3D RC over a triangle BVH found except three-rc (WebGL, BVH, unlicensed). **Gap we could fill.**
- **Tracer API wanted:** **CH with ray intervals** (`tmin`, `tmax` arbitrary, cascade 0 starts at `t0 = Δs`), return `t` and a flag "reached tmax without hit"; hit shade = albedo + emission + **direct light at hit (or hash-cache lookup)**. Many rays (thousands of directions per probe) but **per-probe rays share an origin** -> highly origin-coherent, direction-incoherent; with ray splitting, origin = visible surface pixel (incoherent). Short rays (≈ few cells) -> cheap, top-of-BVH rarely touched; could even run on an SDF/voxel proxy. Precision: tolerant. Miss -> sample sky only in last cascade.
- **Budget:** ~1 ray/pixel in the Split RC design (1 spp, 6.1 ms at 3080 Laptop for HW RT). Rating: **Prototype** (the 3D variant is the most "tracer-shaped" of the techniques: only needs rays + interval + one cache).

### 3.2 DDGI (Majercik 2019/2021) + relocation / classification

- **What:** regular grid of probes; each traces R rays/frame (RTXGI default 256 rays, up to 512 [M, could not verify without reading SDK docs]; Majercik 2019 [V]: 32-256 rays/probe measured at 1.5-1.65 Grays/s on a 2080 Ti, 64 rays x 32x8x32 probes = 524k rays/frame (0.25 rays/pixel at 1080p) used for the largest scenes, 2.6 ms indirect-light passes (Table 2)), shades hits, then compute passes blend into octahedral irradiance (6x6 + border) and mean-distance/distance² (14x14 + border) textures with hysteresis ~0.97. Lookup: trilinear with Chebyshev visibility weights. 2021 additions: probe **relocation** (move probe out of geometry using backface hits; iterated ~5 times at classification [V]), **classification** (Off / Sleeping / Newly awake / Vigilant; probes whose neighbourhood is static sleep -> **30-50% average perf gain** [V]), cascaded volumes, bias terms, fast convergence via adaptive hysteresis, glossy raycast fallback. Paper claims full GI at 6 ms/frame 1080p RTX 2080 Ti [V].
- **Tracer API wanted:** CH+ALB with **SHADE1** (direct lighting at hit using shadow rays, + previous-frame probe irradiance for multibounce). Rays per probe are **same origin, uniform-sphere directions (spherical Fibonacci, random-rotated per frame)** -> origin-coherent, a good fit for wavefront with one workgroup per probe. Backface hit flag required (relocation/classification) -> `hit.frontFace`, `t`. Miss -> sky/env radiance. Tolerates **LOD geometry and approximate materials**; thin-geometry leaks are controlled by visibility moments, not tracer exactness. Also wants **"first-hit-distance only"** for the distance texture: cheap.
- **Budget:** typically 8k-16k probes x 128-256 rays = 1-4 Mrays/frame (≈0.5-2 rays/pixel at 1080p equivalent) before sleeping; after sleeping/classification roughly half. For our web target aim at **0.5-1 Mrays/frame** (e.g. 4k probes x 128 rays, plus sleeping) -> 30-60 Mrays/s.
- Local state: `webgpu-ddgi` already owns this lane (SPEC pins "ray provider" as separable and has its own platform trace/BVH). Our core should become its ray provider. Rating: **Adopt now** (as consumer #1 and benchmark).

### 3.3 RTXGI SHaRC, NRC

- **SHaRC [V Integration.md]:** world-space hash of voxels with LOD (`sceneScale`, `levelBias`, `logarithmBase`); entries: hash (8 B) + accumulation (16 B, atomic i32 fixed-point scaled by `SHARC_RADIANCE_SCALE`) + resolved (16 B f16 RGB+count..); 2^22 cells -> ~160 MiB (40 B/voxel) or ~256 MiB (64 B, SH directional encoding); steady-state occupancy 10-20%. Passes: (1) **update**: a path-tracer pass over ~1/25 of pixels ("one random pixel per 5x5 block, ~4% of paths", rotated), each bounce a segment: `SharcUpdateHit` accumulates radiance into the cell, `SharcSetThroughput` propagates; default `SHARC_PROPAGATION_DEPTH`=2; (2) **resolve** (per element: merge with history, evict stale, clear) + compact; (3) **render**: normal PT, path terminated when `SharcGetCachedRadiance` hits at a non-primary vertex; terminate only when segment length > voxel size, glossy lobes use cache only if cone spread `2·L·sqrt(0.5a²/(1-a²))` > voxel size.
- **NRC [M/V paper saved]:** MLP (hash-grid encoded) trained online each frame on short "training paths" (~1/64 of pixels) whose tail is supervised by the network's own prediction; render paths terminate after ~3 vertices once spread > threshold. Needs per-vertex position, normal, roughness, albedo/specular reflectance as network inputs -> **hit record with material summary at every vertex**. Needs CUDA/tensor cores; **not viable in WebGPU now**. Rating: **Reference only** (watch WebNN for a later learned cache).
- **Tracer API wanted:** PATH with a **per-vertex callback/hook** (cache update at each vertex; cache query to terminate), i.e. the tracer must let the caller supply `onHit` / `terminate`. In wavefront form this means the full-path kernel is composed of stages, and a `CacheUpdateStage` / `CacheQueryStage` can be swapped in. Hit data: position, normal, albedo, roughness, emission, **voxel size from ray length**; ray cone/spread tracking (`rayLength`, `roughness²`) is part of the path state. Coherence: low (random pixels, full bounces).
- **Budget:** update pass 1/25 of pixels ≈ 0.04 paths/pixel x ~3 segments; render pass shortens each path to ~1-2 bounces. 2^22-cell cache in WebGPU = 160 MiB: **check `maxStorageBufferBindingSize`** (default 128 MiB) -> split into 2-3 buffers or reduce to 2^21. Rating: **Adopt now** for the hash cache module; NRC **Reference only**.

### 3.4 Spatial-hash caches, path-space filtering, world-space ReSTIR

- **Binder et al. [V]:** trace camera paths, quantize hit position (jittered to hide cell edges), atomically accumulate radiance into a hash, query by averaging. Jitter in query/insert avoids grid artifacts. The "massively parallel" variant (arXiv 1902.05942) gives GPU details.
- **Boisse 2021 world-space ReSTIR [V]:** per path vertex, find/insert hash cell, `atom_inc` the cell counter, append `<v, cell, idx>`, parallel prefix-sum over cell counters, scatter reservoirs into contiguous per-cell storage; then WRS merge of neighbouring reservoirs. One-lookup reuse; used for direct light sampling at 30k area lights, "very few shadow rays per pixel".
- **Capsaicin hash cache [V local]:** `hash_grid_cache.hlsl` (495 lines): cell size = `max(distance(eye,pos)*cell_size, min_cell_size)` snapped to a power-of-two level (lines ~98-142), `kHashGridCache_FloatQuantize 1e3`, `kHashGridCache_TileDecay 50` (frames before eviction); plus `world_space_restir.hlsl`, `screen_probes.hlsl`, `glossy_reflections.hlsl` (GI-1.x).
- **Tracer API wanted:** nothing exotic on the tracer; wants **a shared hash module** and **hit position + normal + ray length**. Coherence none. Rating: **Adopt now** (the cache is reusable by DDGI hits, SSR fallbacks, path tracer, and AO).
- **WebGPU risks:** 32-bit atomics only -> keys as (u32 hash, u32 checksum) with cmpxchg on a 32-bit word, or buckets of 32 slots with eviction like the Shade WebGPU SHaRC [V]. `atomicAdd` on i32 fixed-point radiance (SHaRC-style quantization) works with u32/i32. Prefix-sum scatter needs a global scan: WGSL has no decoupled-look-back guarantees across workgroups -> use multi-dispatch scan or avoid with per-cell fixed-size slots.

### 3.5 AMD GI-1.0 / 1.1 / 1.2 (Capsaicin, local)

- **What [V]:** 2-level cache: **screen probes** (8x8 px probe tiles; direction budget per probe) -> **world-space hash grid cache** (first bounce cells from screen probe rays, second-bounce cells from rays out of first cells, direct light only; GI-1.2 adds a second estimator for indirect and multiply by BRDF at resolve) -> world-space ReSTIR for direct lighting at hits, "as low as ¼ sample per pixel". Timings (Table 1, 1080p-class?): total 1.9-3.1 ms RX 6900 XT, 1.9-3.4 ms RTX 3080 across 99k-9.0M tris; ray tracing 0.13-0.68 ms of that (DXR 1.1). The paper says explicitly that **any traversal method works as long as it returns the hit point** [V].
- **GI-1.1 glossy [V]:** GGX-sampled reflection rays; roughness switches between **specular interpolation from screen probes** and **RT + second-level hash cache** at the hit; joint denoise+upscale.
- **Tracer API wanted:** CH + position/normal/albedo + hash-cache read-write at hit; inline ray queries (`Capsaicin` uses inline DXR 1.1 in compute) -> **our traversal must be callable from compute shaders as a function, not only a megakernel.** Rays per frame: ~¼ spp for screen probes + a small set for cache cells. Coherence: screen-probe rays are per-tile direction-coherent in a hemispherical-octahedral layout.
- Rating: **Adopt now as the design reference for "screen probes -> hash cache" two-level**. Windows/D3D12 only to run, but source (MIT) is read-only referenceable. Cross-ref `ddgi-research/research/gi-1.0/native-capsaicin.md` for the timing harness.

### 3.6 Lumen (reference for tracing mix)

- **What [V]:** *Surface cache* (card capture of albedo/normal/emissive per mesh, lit with direct+indirect, partially updated per frame: histogram-select N pages, 8x8 tiles in z-order for trace coherence); *screen probes* (8x8 octahedral rays/probe, 16x16 in some cases; probe adaptive placement) with **Product Importance Sampling** (BRDF x previous-frame lighting; rays from culled directions reassigned) done in a threadgroup per probe; *world radiance cache* (clipmap probes) for far field; *tracing:* screen-space (HZB) -> mesh SDF near-field (**1.65-2.37 ms**, far-field 1.04 ms on console [V lines ~2465]) -> global SDF; or HW RT. Shipping costs (PS5, SW RT, global SDF): 1/16 rpp High = 2.8 ms (surface cache 0.47, gather 2.07, reflections 0.23); 1/4 rpp Epic = 4.6 ms. Lyra: 4.3 ms / 8.7 ms [V]. Reflections: roughness->1 rays are incoherent and slow -> use screen radiance cache (8x8 directions) instead.
- **Tracer API wanted:** **cone/footprint trace against a proxy** (SDF or simplified mesh) with **hit lookup in a surface cache** (lit radiance, no per-hit material eval). Key idea for us: **"hit -> cache lookup" replaces shading**, so hit material depth = zero. Also "trace with distance limit then continue in a coarser structure".
- Rating: **Reference only** for architecture; **Prototype** the "proxy trace + surface-cache lookup" API option (needs a BVH-over-LOD-mesh or SDF mode).

### 3.7 Surfel GI (EA SEED GIBS; kajiya)

- **What [M + 2024 PDF saved]:** surfels (position, normal, radius) placed on visible surfaces, spatially bound via a grid/hash, each accumulates radiance from ~rays/frame with integrated `shading = surfel blend by distance/normal`; coverage-driven spawn/recycle. GIBS 2 keeps probes for off-screen/volumetric and surfels near surfaces; interpolated from lighting ensembles. Kajiya (Embark) uses surfels + ReSTIR-like final gather and a world-space probe fallback.
- **Tracer API wanted:** CH+ALB, SHADE1 with a 2nd surfel lookup (multibounce). Rays: 2-64/surfel/frame budget-driven by "needs more samples". Per-surfel rays are origin-coherent. Rating: **Later** (a hash cache gives most of the benefit with less scene-specific logic).

### 3.8 Voxel cone tracing / voxel GI / SDF GI

- **What:** voxelize scene into mip-mapped radiance 3D texture; trace 5-9 cones for diffuse, 1 cone for specular; leaks with thin walls. Godot VoxelGI/SDFGI, Unity APV-adjacent baking, Brixelizer SDF cascades [M].
- **Tracer API wanted:** **cone query, not a ray**: `traceCone(origin, dir, aperture, tmin, tmax)` over a voxel/SDF structure; not our BVH. The BVH is only needed for (re)voxelization. Cheap and coherent (texture fetches). **Software "proxy tracer" must be selectable at the same API level as BVH tracing** so callers (reflections, AO) can switch per roughness/distance.
- Rating: **Later**; keep the API shape (cone/footprint ray) so a voxel/SDF backend can be slotted in.

### 3.9 SSGI / SSR with world-space fallback

- **What:** HZB march in screen space (downsampler lib provides Hi-Z; AMD FFX SSSR, Stochastic SSR), then on miss or off-screen fall back to **probes -> RT -> sky**. Local `reflections` repo: three-pronged hybrid (SSR + probe + traced) with W/D notes; TRIAGE puts "cone-footprint hit colour", "ratio-estimator ray reuse (1 ray + reuse ≈ 4 rays of noise)", "lobe occlusion for traced hits", "reflection-aware motion for upscaler" as the open work [V].
- **Tracer API wanted:** **closest hit with a cone/footprint** (hit colour filtered through mip pyramid chosen from cone spread at hit), **returns hit position + normal + albedo + a *reprojectable* hit identity** (hit position for reflection-reprojection / reactive masks), and **quad-coherent tracing** (FFX quad copy: 1 ray per 2x2). Also **"ray-length-limited trace"** (so SSR miss region can restart from the exit point: `tmin` = SSR exit distance).
- **Budget:** 0.25-1 ray per pixel for rough-to-mirror, restricted to pixels where `roughness < 0.5`-ish.
- Rating: **Adopt now** (we have the consumer; this is the most concrete second client).

### 3.10 ReSTIR GI (real-time)

- **What [V against the saved PDF 2026-11-13 for numbers; mechanics from memory]:** per pixel: 1 initial sample = path of 1 secondary ray with next-event or cache; spatiotemporal reuse of "sample = (secondary vertex x2, radiance L2, normal)" with a **Jacobian for reconnection**; typically **1 ray (+1 shadow ray) per pixel plus 1-3 extra visibility rays in spatial passes**. Needs a denoiser. **3.2 ms initial sampling (1 bounce; 4.2 ms 2 bounce) + 4.6 ms spatiotemporal reuse on an RTX 3080 at 1920x1050, i.e. ~8 ms, not 2-4 ms [V, Ouyang 2021 Fig.8]; RTX 3090 1080p Fig.1: 8.9 ms total vs 8.0 ms plain PT; reservoir 48 B [V].**
- **Tracer API wanted:** CH + normal + radiance at hit (SHADE1 or cache), then **VIS** for each reuse candidate (re-validation of visibility from pixel to neighbour's x2). So the tracer needs **cheap any-hit** with a *point target* (`visible(a, b)` segment query) in addition to directional rays.
- **Rating:** **Prototype** (high noise reduction per ray; costly in WebGPU due to reservoir memory traffic: reservoirs (~32-48 B) x 2 buffers x 1080p; consider half-res).

### 3.11 Probe-based and ray-traced reflections (roughness-aware)

- **Probe reflections:** parallax-corrected cubemaps/oct maps; filtered per roughness (PMREM). Tracer use: **offline/amortized capture** of probes by tracing (instead of rasterization), 1 tile of an octahedral probe per frame.
- **RT reflections:** GGX VNDF sampled rays for rough, mirror rays for smooth; hit shading = full material (smooth) or cache (rough); **per-roughness LOD**: roughness > ~0.4 -> use cache/probe (GI-1.1, Lumen) [V]; tracing against a **simplified/LOD BVH** or **proxy geometry** for rough lobes (cone spread blurs geometric error); **ray-footprint-based texture LOD** (ray cones/differentials) for hit shading. Denoise: spatiotemporal ReBLUR/ReLAX / FFX reflection denoiser (we ship our own).
- **Tracer API wanted:** CH + **ray cone** (spread angle, width) input, **material hook at hit that is cheap or cached**, hit identity for reprojection. Budget 0.25-1 rpp. Rating: **Adopt now.**

### 3.12 Ray-traced AO and shadows

- **AO:** 1-2 short cosine rays/pixel, max distance r (0.5-2 m), any-hit **VIS** only, can use **a lower-res or simplified BVH** and **terminate early on first hit** (`TERMINATE_ON_FIRST_HIT`). Often traced at half res with an upsample/denoise. Gautron's real-time RT-AO using spatial hashing is an AO-cache variant [M, venue unconfirmed].
- **Shadows:** 1 shadow ray per light-sample per pixel, any-hit VIS; **soft shadows** via 1-2 samples on a disk/sphere (blue noise, spatiotemporal denoise); **inline** in the fragment/compute pass (ray queries in the shading compute, not a separate pipeline) -> our API must be callable from any compute shader as a WGSL function. Optimizations: (a) trace shadow rays with **skip-back-face** and **opaque-only** flags; (b) treat alpha-tested geometry optionally; (c) **ray sorting/compaction** by light for coherence; (d) cache last-occluder.
- **Tracer API wanted:** **VIS(origin, dir, tmax, flags)** -> bool; `visibleSegment(a, b)`; must be ≥2x faster than closest hit (no hit data, no sorting by distance, any-hit early out). Budget: 1-2 rays/pixel each; at 1080p = 2-4 Mrays/frame = 120-250 Mrays/s at 60 fps (that is the *hard* number - shadows are the most ray-hungry consumer and the most latency-tolerant to cull with shadow maps in hybrid).
- Rating: **Adopt now** (VIS is the cheapest, simplest, and highest payoff API; also needed by DDGI/GI-1.0 hit shading and path tracer NEE).

### 3.13 Baking with path tracing

- **What [M]:** Bakery (GPU progressive, RTX or Cycles-like compute), Unity Progressive GPU Lightmapper (Optix/RadeonRays), Godot LightmapGI (compute lightmapper: dilation, denoise with OIDN), Blender baking. Targets: lightmap texels, light probes (L1/L2 SH), reflection probes, AO/bent normals.
- **Tracer API wanted:** **PATH** (full path, multiple bounces, progressive accumulation) with **arbitrary origin** (texel position + normal, probe position uniform sphere) and arbitrary **output sinks** (SH accumulation, lightmap texel) -> i.e. the full path tracer must accept "ray generators" other than a camera: `PathGen(texel/probe)`. Also **adaptive sampling + denoise** (OIDN). Budget: offline-ish, not frame-bound; throughput-limited (Mrays/s).
- Rating: **Adopt now** (almost free given the full PT; this is the right way to produce the static fallback `webgpu-ddgi` exports (LightProbeGrid JSON) and to seed caches).

### 3.14 Quick table: tracer needs by technique

| Technique | Query kind | Rays per px per frame (1080p/60) | Ray coherence | Hit data needed | Shading at hit | Tolerates proxy geometry | Notes |
|---|---|---|---|---|---|---|---|
| Shadows | VIS | 1-2 | high (per light) | none | none | no for contact, yes for far | early-out any-hit |
| AO | VIS (short) | 1-2 | low | none | none | yes | max-distance, half-res |
| DDGI | CH + backface | 0.5-1 (probe rays) | origin-coherent | t, normal, albedo, emissive, frontFace | SHADE1 + probe read | yes | relocation needs backface/dist |
| Radiance cascades 3D | CH with [tmin,tmax] | ~1 (Split RC) | per-pixel incoherent | t, albedo, normal, emissive | SHADE1 or cache | yes | interval + deposit |
| SHaRC/hash cache | PATH w/ hook | 0.04 paths x ~3 segs | low | pos, normal, albedo, rough, L | cache update/query | yes | cache terminates paths |
| GI-1.x screen probes | CH | 0.25 | tile-coherent | pos, normal, albedo | SHADE1 + hash query | yes | two-level |
| Lumen-style | cone vs proxy + cache | 1/16-1/4 | tile-coherent | hit id/uv in surface cache | lookup only | **required** | zero material eval |
| ReSTIR GI | CH + VIS(a,b) | 1 + 1-3 VIS | low | pos, normal, L | SHADE1 | yes | reservoirs memory-bound |
| RT reflections | CH + cone | 0.25-1 | low (rough) / high (mirror) | pos, normal, albedo, rough, hit id, cone footprint | full (mirror) / cache (rough) | roughness-dependent | hit reprojection |
| Voxel cone / SDF GI | cone | ~5-9 cone steps | high | none (texture) | lookup | n/a | non-BVH backend |
| Baking | PATH w/ custom rays | offline | low | everything | full | no | progressive sinks |

## 4. Rays-per-pixel budgets at 1080p / 60 fps on mid-range GPUs

Back-of-envelope - **verify with our own traversal benchmarks first**. 1080p = 2.07 Mpx. Target total frame budget for all ray work ≈ 4-6 ms of a 16.6 ms frame on a mid-range GPU (e.g. RTX 3060 / M-series Pro / RX 6700 class).

| Consumer | rpp | rays/frame | Mrays/s at 60 | Cost driver |
|---|---|---|---|---|
| GI probes/cascades/surfels (all caches) | 0.1-0.5 | 0.2-1.0 M | 12-60 | closest-hit + hit shading |
| Reflections | 0.25-1 | 0.5-2 M | 30-120 | roughness-limited, only on glossy pixels |
| Shadows | 1-2 | 2-4 M | 120-250 | any-hit; use shadow maps hybrid for sun |
| AO | 0.5-1 (half res) | 0.5-1 M | 30-60 | short rays; early out |
| **Sum** | ~2-4 | 3-8 M | 190-480 | |

Reference points (hardware RT, so software is slower): Lumen PS5 **software RT (Global SDF only, not hardware RT)** High 1/16 rpp = 2.77 ms, Epic 1/4 rpp = 4.63 ms at 1080p [V, re-checked in the saved course notes: Final Gather 2.07 / 3.46 ms]; GI-1.0 total 1.9-3.4 ms at "¼ spp" with 0.13-0.68 ms spent in DXR [V]; Split RC trace 1.3-4.5 ms for ~1 spp [V]; DDGI 6 ms full GI RTX 2080 Ti 1080p [V, Fig.1 of Majercik 2019; indirect passes only = 2.6 ms, Table 2]; GI-1.1: not stated [V]. A software WGSL traversal at ~100-300 Mrays/s incoherent on a mid-range GPU (to be measured; bvh lib) implies the shadows+reflection budget above exceeds the frame; **the real-time modes must lean on caches (hash/probe), shadow maps for the sun, and SSR first.** Target: ≤ 60 Mrays/s for GI, ≤ 60 for reflections, shadow rays on demand.

## 5. Proposed layered API for the core

All layers share: one BVH (sibling `bvh`), one traversal function `traverse(ray, flags, stack)`, one ray/hit record, and one scheduling model (wavefront queues, GPU-driven indirect dispatch). Every layer exists as (a) **WGSL function library** callable inline from any compute shader (inline ray query model, like Capsaicin / Lumen inline RT) and (b) a **dispatcher** that takes a GPU ray buffer + count (indirect) for wavefront use.

### L0 - Visibility (VIS)
```
fn rt_visible(origin, dir, tmin, tmax, flags) -> bool          // any-hit, early out
fn rt_visible_segment(a, b, flags) -> bool
fn rt_visible_cone(origin, dir, spread, tmax, flags) -> f32     // soft visibility 0..1 (AO / area light approximation) [Later]
```
flags: opaque-only, cull-back, skip-alpha-test, shadow-LOD (use simplified BVH level), instance mask / layer mask (per-consumer: DDGI sees GI-contributors only).
Consumers: shadows, AO, ReSTIR reuse validation, NEE in PT, DDGI/GI hit shading.

### L1 - Closest hit, minimal surface data (CH)
```
struct Hit { t: f32, prim: u32, inst: u32, bary: vec2<f32>, frontFace: bool, flags: u32 }
fn rt_closest(origin, dir, tmin, tmax, flags) -> Hit
fn rt_closest_cone(origin, dir, spread0, spreadRate, tmin, tmax, flags) -> Hit   // roughness-aware / footprint
// minimal surface: derived lazily from (inst, prim, bary), only what the caller asks:
fn rt_surface(hit, want: SurfaceMask) -> Surface {pos, geomNormal, shadingNormal, uv, albedo, emissive, rough, metal, matId}
```
`SurfaceMask` lets DDGI ask for {pos, geomNormal, albedo, emissive} and skip textures beyond a **mip chosen from cone footprint** (ray cone LOD). Also a **LOD selector**: `GeomLod::Exact | Proxy` (e.g. a decimated BVH per instance or a coarse top-level) for rough reflections/GI.
Interval semantics (`tmin > 0`, `tmax`, miss-at-tmax reporting) are mandatory for Radiance Cascades and for SSR-exit continuation.

### L2 - One-bounce shade (SHADE1)
```
struct ShadeInput { hit/Surface, wo, lightSampler, cache: CacheRef?, mode: Direct | DirectPlusCache }
fn rt_shade1(hit, wo, opts) -> vec3<f32> radiance_leaving_toward_wo
```
Evaluates: emission + direct light (light sampler: sun/env/area lights with 1-N shadow rays via L0, ReSTIR-able) + optional **cache lookup** (hash cache or probe atlas) for indirect at the hit. Material evaluation depth is a parameter: `matDepth: Flat (albedo only) | Lambert | FullBSDF`. This is the single most-reused layer (DDGI, RC, GI-1.x, surfels, SSR fallback, SHaRC update).

### L3 - Full path (PATH)
Wavefront progressive path tracer (the repo's main product) with **stage hooks**: `onVertex(pathState, hit)`, `terminate(pathState, hit)`, `rayGen(i)` so that SHaRC-style cache update/query, NRC-like training (later), baking (custom ray generators: texel, probe) and RCPG-like guiding can be plugged in. Path state record carries `rayLength`, `ray cone`, `roughness` so cache-termination heuristics work.

### Shared caches (core-owned modules; both GI tools and PT consume)
1. **HashCache (world-space spatial hash radiance cache).** 32-bit-atomic design: bucket-of-32 with (hashKey u32, checksum u32, ...) or SHaRC-like entries; LOD by distance (`cell = max(dist·k, min)` snapped to powers of two, as in Capsaicin `hash_grid_cache.hlsl`); accumulate (atomicAdd on fixed-point i32 RGB + count), resolve (temporal merge, stale eviction, decay `kHashGridCache_TileDecay`-style), optional SH2/SH directional encoding. API: `cache_insert(pos, nrm, radiance, w)`, `cache_query(pos, nrm, rayLen, rough) -> radiance|miss`, `cache_resolve()`, `cache_clear()`. Size knob 2^20-2^22 (≤128 MiB per binding on default limits; split buffers or raise `maxStorageBufferBindingSize`). Jittered lookup (Binder) option.
2. **ProbeAtlas.** DDGI-compatible octahedral irradiance + visibility textures (6x6+1 border, 14x14+1 border; rgb10a2 / rg11b10 / rgba16f) plus probe state buffer (Off/Sleeping/Vigilant), relocation offsets; also holds RC cascade levels / Lumen-style world probes through the same allocator (an `OctAtlas` of N x M tiles with per-probe metadata). `webgpu-ddgi` already owns this; core should define the **layout and sampling WGSL** and DDGI plugs in.
3. **ScreenProbes (optional).** 8x8 hemi-oct per 8-16 px tile (Lumen / GI-1.0 style) for diffuse + roughness>0.4 reflections; consumes L1/L2.
4. **Surface/proxy cache (later).** Lumen-style surface cache cards or hash-cache lookup at hit as "zero material eval".

### Scheduling / budgets
- Frame budget API: `rt.budget({ giRays, reflRays, shadowRays, aoRays })` in **rays per pixel** or absolute count, with per-consumer priority and **indirect, GPU-side queue** (no CPU readback). Each consumer enqueues rays into a shared structure-of-arrays ray queue; the traversal stage sorts (cheap bucket sort by direction octant/origin cell for coherence; shadows by light), runs closest-hit/any-hit variants, then shading stage.
- Ray sorting: origin-cell + direction-octant key; consumers with already-coherent rays (probe rays share origin, mirror reflections, shadows per light) opt out via `coherent: true`.
- Precision: f32 ray/box tests; f16 for payloads (radiance, throughput), 16-bit oct normals; stay watertight at tmin>0 (edge-function ordering fix like `giEdgeFunction` in the DDGI repo PROGRESS 2026-11-10: Dawn relaxed-math FMA breaks shared-edge consistency on Metal [V]).

## 6. Per-technique ratings

| Technique | Rating | Tracer layers |
|---|---|---|
| Shadows / AO via VIS | **Adopt now** | L0 |
| DDGI (+relocation/classification) | **Adopt now** | L1, L2, ProbeAtlas |
| HashCache (SHaRC / GI-1.0 style, 32-bit atomics) | **Adopt now** | L2, L3 hooks |
| SSR + traced fallback + cone/footprint (our reflections lib) | **Adopt now** | L1 (cone), L2 |
| Baking via PT | **Adopt now** | L3 |
| Radiance Cascades 3D (Split RC lineage) | **Prototype** | L1 intervals, L2, HashCache |
| ReSTIR GI | **Prototype** | L1, L0 (segment) |
| AMD GI-1.x two-level (screen probes + hash) | **Prototype** (design reference) | L1, L2 |
| Lumen-like proxy trace + surface cache | **Later** | L1 proxy mode |
| Surfel GI (GIBS) | **Later** | L1, L2 |
| Voxel cone / SDF GI | **Later** | cone backend (non-BVH) |
| NRC | **Reference only** (needs tensor cores/CUDA; revisit with WebNN) | - |
| RCPG (RC as path guiding) | **Later / Reference** (idea for full PT) | L3 |

## 7. For the spec (decisions to make)

1. **Inline-callable WGSL traversal** (not only a megakernel/wavefront). Every consumer in this note except baking wants to call `rt_visible` / `rt_closest` from its own compute shader. Offer both entry points over one implementation.
2. **Ray record & hit record layout** shared with `bvh` (`Hit {t, prim, inst, bary, frontFace}` = 16 B + flags), plus `rt_surface()` lazy decode with `SurfaceMask`.
3. **`tmin`/`tmax` intervals + ray cone input are mandatory in L1** (Radiance Cascades, SSR continuation, rough reflections).
4. **Any-hit VIS is a separate, cheaper traversal variant** (early out, no sorting, optional proxy BVH level), not a flag on closest-hit.
5. **Layer masks / instance masks** per consumer (GI contributor flag already in webgpu-ddgi SPEC).
6. **HashCache module in core** with 32-bit atomics only; document the memory sizes (2^22 cells = 160-256 MiB for SHaRC-like; Shade WebGPU variant 12 B/cell f16 YCoCg-SH2 = 48 MiB at 2^22 [V]); default 2^20-2^21 to fit default limits.
7. **Probe atlas layout spec** (shared with DDGI repo) so RC/Lumen-style probes reuse the allocator.
8. **Budget scheduler API in rays/pixel/frame**, with per-consumer caps; GPU-side queues with indirect dispatch.
9. **Two shading depths:** `Flat | Lambert | Full`; GI consumers default to `Flat`/`Lambert` + cache lookup.
10. **Proxy geometry mode:** per-instance optional simplified BVH or LOD level selectable at trace time (`GeomLod`), roughness-driven for reflections.
11. Keep **cone/SDF/voxel backends** pluggable behind `rt_closest_cone` so GI libs can trade BVH for SDF without API change.

## 8. Bench / comparison ideas

- **Incoherent closest-hit Mrays/s** per scene (Sponza, Bistro, San Miguel) on M-series, RTX 3060 laptop (the machine used for Capsaicin timing in ddgi-research), Intel: random uniform-hemisphere rays from random surface points (DDGI-like), vs per-probe-coherent rays, vs shadow any-hit rays, vs GGX rays at roughness {0.1,0.3,0.6}. Compare with the tinybvh / `bvh` numbers.
- **Cost of hit shading** at matDepth Flat / Lambert / Full; cost of hash-cache lookup per hit.
- **DDGI on the core vs native RTXGI-DDGI** (already planned in `ddgi-research`): ms/frame, rays/probe, converged error.
- **Split RC vs DDGI at equal ray budget** on Sponza: image quality (single frame and 10-frame), ms per stage (use Split RC's table as the target shape).
- **Hash cache:** occupancy vs camera motion, collision rate, resolve ms, vs Capsaicin GI-1.2 (`hash_grid_cache.hlsl`) via its timing CSVs.
- **Reflections:** TRIAGE's T1 flicker metrics + traced hits per pixel vs PT reference (already in `reflections` bench).
- Compare **mutracer** (near-competitor) on per-frame traced Mrays/s on shared scenes to place our core.

## 9. Needs Windows / native hardware

- Capsaicin GI-1.x (D3D12 / DXR 1.1 only; Windows) for GI-1.0/1.2 timings and reference images (already built on the RTX 3060 Laptop per ddgi-research).
- NVIDIA RTXGI-DDGI native, SHaRC and NRC (NRC needs CUDA/tensor cores and RTX).
- Unreal Lumen SWRT vs HWRT numbers (UE5 on Windows) for comparison.
- Any HW-RT "gold" comparison of ray throughput (OptiX/DXR) to calibrate software tracer vs hardware.

## 10. Open questions

1. WGSL has no general 64-bit atomics. **[V 2026-11-13]** Only `atomic<vec2u>` min/max (`atomic_vec2u_min_max`, merged in the gpuweb repo, not shipped in browsers yet, Apple9/M3+ only on Metal) is proposed; it cannot do key+value hash inserts (no CAS). Hash entry design must stay bucketed 32-bit with eviction or a lock buffer.
2. Subgroup ops: shipped in Chrome 134 [V]; Safari/Firefox support not confirmed (06 survey: Safari 1.2%). Keep the fallback path.
3. Can ReSTIR-GI reservoir traffic fit 60 fps in WebGPU at half-res? Needs a prototype before committing.
4. Is a **decimated BVH per instance** worth the build cost vs SDF/voxel proxy for rough reflections/GI? Lumen separates Mesh SDF near and Global SDF far; we could mimic with BVH LOD near and a coarse global grid/SDF far.
5. Whether to ship **RCPG-style guiding** (RC cascades as a world-space guiding field) in the full path tracer.
6. Which hash key to use: position+normal-bucket (SHaRC) vs position-only (Binder) vs jittered; how to handle thin walls (light leaks) - needs a test scene (two-room, thin wall).
7. Community WebGPU RC projects (TypeGPU, others): need a licence check and perf measurement before any comparison; three-rc is "all rights reserved".
8. Voxel/SDF cone backend: where does it live (core vs GI libs)?
9. Verify per-pass numbers for Lumen/NRC/GI-1.1 that were stated from memory or blog text without ms (flagged [M] above).
