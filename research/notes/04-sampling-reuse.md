# 04 — Sampling, noise and sample reuse

Date: 2026-11-13. Confidence tags: **[V]** verified from a downloaded PDF/page this session; **[M]** from memory (check before citing a number); **[S]** from search snippet only.

## 1. TL;DR

- **Progressive reference mode: Owen-scrambled Sobol (Burley 2020, hash-based), padded across dimensions with nested-uniform shuffles, per-pixel seed from a hash.** Not blue noise. Blue-noise error distribution only helps when a low-pass denoiser follows; for a converging reference it gives no asymptotic gain and Sobol+Owen gets O(N^-3/2) on smooth integrands **[V]**. ~15 ALU ops per dimension, zero memory. Use ZSobol-style Morton indexing (pbrt-v4 `ZSobolSampler`) only if you want low-spp (<=16) screen-space stratification for free.
- **1-spp real-time + denoiser + temporal upscaler: precomputed blue-noise textures, not Sobol.** Use STBN / FAST-style spatiotemporal textures for the dimensions that matter (first-bounce light/BRDF samples, AO/shadow rays); per-frame toroidal Cranley-Patterson shift (golden-ratio R1 on the frame index) so the texture does not repeat identically and temporal accumulation still converges. Use a hash/PCG for deeper dimensions (blue noise buys nothing past bounce 1-2).
- **Best-known hybrid:** Heitz-style Sobol-with-ranking/scrambling-tile sampler (Heitz & Belcour 2019; rank-1 lattice version in Belcour & Heitz 2021) = progressive-friendly *and* blue-noise error. Dimensions are optimized in **pairs**, so consume dims in pairs. Good single implementation that can serve both modes: switch the tile tables off for reference.
- **FAST noise (Donnelly et al., I3D 2024)** shows blue noise is optimal only for a Gaussian low-pass; if your denoiser/TAA is a box or edge-aware filter, noise optimized for *that* filter beats STBN (lower RMSE, no blocky artifacts) **[V]**. Since our denoiser is OIDN-like (neural) + FSR2-style temporal, a pragmatic path: ship STBN first, make the texture generator pluggable, test FAST textures (EA, open source).
- **Reuse: ReSTIR DI is the only reuse algorithm that is clearly "cheap enough" for WebGPU software tracing now.** ReSTIR GI is plausible (one extra ray/pixel + reservoir 2x~48 B). **ReSTIR PT is a research-grade item**: original 40-80 ms/frame on RTX 3090 **[V]**; ReSTIR PT Enhanced (2026) makes it 2-3x faster (37.1 -> 12.6 ms in Fig. 1 scene, reservoirs 88 -> 64 B) **[V]** but still needs hybrid shift mapping, per-vertex random-replay, and large per-pixel storage. Later.
- **Many lights:** light BVH/light tree (Conty Estevez & Kulla) with importance by orientation-aware cone bound is the production answer; for real-time, RIS over a small candidate set (8-32) drawn from a power/alias table or a light-tree traversal, then ReSTIR DI on top.
- **Path guiding:** OpenPGL-style (SD-tree/VMM) is a CPU-integrated library; GPU neural/online guiding exists (Huang 2024 NASG, wavefront) **[S]** but needs training kernels + atomics-heavy buffers. Not feasible as a core feature in WebGPU v1. Cheap substitute: spatial-hash radiance cache (SHaRC-like) for indirect termination/NEE guiding.
- **Russian roulette:** use throughput-based RR with min-depth (3-4) now; EARS (efficiency-aware RR/splitting) is cheap to implement (needs per-voxel/per-pixel running estimates of variance and cost) and worth prototyping for the reference mode.

## 2. Sources

| Name | Link | Year | License / venue | Local path | Why |
|---|---|---|---|---|---|
| Practical Hash-based Owen Scrambling (Burley) | https://jcgt.org/published/0009/04/01/ | 2020 | JCGT (open access) | research/papers/sampling/burley2020-practical-hash-based-owen.pdf | The Owen-Sobol recipe; constants verified [V] |
| Spatiotemporal reservoir resampling (ReSTIR DI) | https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct | 2020 | SIGGRAPH/TOG, author copy | .../restir-di-bitterli2020.pdf | Reservoir/RIS/temporal+spatial reuse |
| Generalized Resampled Importance Sampling (GRIS) | https://research.nvidia.com/labs/rtr/publication/lin2022generalized/ | 2022 | SIGGRAPH/TOG | .../gris-lin2022.pdf | Theory of reuse with shift maps + MIS; ReSTIR PT |
| ReSTIR GI (Ouyang et al.) | https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing | 2021 | HPG/CGF | link only (direct PDF fetch failed) | Path-sample reuse, one-bounce |
| ReSTIR PT Enhanced (Lin, Kettunen, Wyman) | https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced.pdf | 2026 (PACMCGIT 9(1), art. 13), CC BY 4.0 | link only (131 MB) | 2-3x faster ReSTIR PT [V] |
| ReSTIR BDPT (Hedstrom et al.) | https://research.nvidia.com/labs/rtr/publication/hedstrom2025restir/hedstrom2025restir.pdf | 2025 | TOG | .../restir-bdpt-hedstrom2025.pdf | Caustics / hard-to-reach lights [S] |
| ReSTIR PG (Zeng et al.) | https://research.nvidia.com/labs/rtr/publication/zeng2025restirpg/zeng2025restirpg_paper.pdf | 2025 | SIGGRAPH Asia | link only (197 MB) | Resampled paths fit guiding distributions [S] |
| FAST: Filter-Adapted Spatio-Temporal Sampling (Donnelly, Wolfe, Bütepage, Valdés) | https://www.ea.com/seed/news/spatio-temporal-sampling ; code github.com/electronicarts/fastnoise, .../importance-sampled-FAST-noise | 2024 | I3D 2024 / PACMCGIT | .../fast-noise-donnelly2024.pdf | Filter-matched noise textures |
| Spatiotemporal Blue Noise Masks (Wolfe et al.) | https://research.nvidia.com/labs/rtr/publication/wolfe2022spatiotemporal/ | 2022 | EGSR | link only (fetch failed) | STBN textures (NVIDIA ships 128x128x64 style tiles) [M] |
| Lessons Learned ... Blue-Noise Error Distribution (Belcour & Heitz) | https://arxiv.org/abs/2105.12620 | 2021 | SIGGRAPH Talks | .../belcour-heitz2021-blue-noise-lessons.pdf | Rank-1 lattice screen-space sampler |
| Screen-space blue-noise diffusion ... hierarchical ordering of pixels (Ahmed & Wonka) | https://arxiv.org/abs/2012.00323 | 2020 | TOG 39(6) | .../ahmed-wonka2020-screen-space-blue-noise.pdf | Morton-ordered Sobol with blue-noise error; basis of ZSobol |
| Distributing MC errors as blue noise (Heitz & Belcour) | https://eheitzresearch.wordpress.com/762-2/ | 2019 | HPG | link only | Pixel-permutation approach |
| EARS: Efficiency-Aware Russian Roulette and Splitting (Rath et al.) | ACM TOG 41(4), SIGGRAPH 2022 — link TBD (arXiv 2205.04435 was WRONG: a quantum-annealing paper; file deleted) | 2022 | SIGGRAPH/TOG | — | Efficiency-aware RR & splitting |
| Importance Sampling of Many Lights with Adaptive Tree Splitting (Conty Estevez & Kulla) | https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf | 2018 | HPG | .../conty-kulla2018-many-lights-lighttree.pdf | Light BVH, shipped in Cycles/Arnold |
| Practical Path Guiding (Mueller et al.) | https://tom94.net/data/publications/mueller17guiding/mueller17guiding.pdf | 2017 | EGSR | link only (PDF fetch failed) | SD-tree guiding |
| Online Neural Path Guiding w/ Normalized Anisotropic Spherical Gaussians (Huang et al.) | https://arxiv.org/abs/2303.08064 | 2024 | TOG | link only | GPU wavefront guiding [S] |
| Neural Parametric Mixtures for Path Guiding (Dong et al.) | https://arxiv.org/abs/2504.04315 | 2025 | | link only | [S] |
| Neural Resampling w/ Optimized Candidate Allocation (Rath et al.) | https://diglib.eg.org/items/b52f338b-36d1-4565-a275-c3f11a49bfea/full | 2025 | EGSR | link only | [S] |
| SHaRC (Spatial Hash Radiance Cache) | https://github.com/NVIDIA-RTX/SHARC (docs/Integration.md) | 2024 | MIT-style NVIDIA RTX SDK license [M] | not cloned | World-space hash radiance cache; 4 passes; default 2^22 entries [S] |
| OpenPGL | https://github.com/OpenPathGuidingLibrary/openpgl | | Apache-2.0 [M] | not cloned | Reference guiding library (CPU) |
| pbrt-v4 samplers (ZSobol, PaddedSobol, PMJ02BN) | https://github.com/mmp/pbrt-v4 (src/pbrt/samplers.{h,cpp}) | 2023 | Apache-2.0 [M] | not cloned | Reference code for all QMC samplers |
| Ureña et al. spherical rectangle sampling; Arvo 1995 triangle | EGSR 2013 / SIGGRAPH 1995 | | | link only | Solid-angle area light sampling |

Not obtained: Christensen PMJ02 (Pixar URL failed), Muller 2017, STBN paper PDF, ZSobol author PDF. Fetch manually if wanted.

## 3. Techniques

### 3.1 QMC & sequences

**Owen-scrambled Sobol, hash-based (Burley 2020)** — Adopt now (reference + default).
- Nested uniform scramble of a 32-bit value `x` = `reverse_bits(laine_karras(reverse_bits(x), seed))` where the LK permutation is `x += seed; x ^= x*0x6c50b47c; x ^= x*0xb82f1e52; x ^= x*0xc7afe638; x ^= x*0x8d22f6e6` [V constants]. Because multiplications only propagate entropy upward, reversing bits first makes it a *nested* (Owen) permutation. Note: the plain Laine-Karras hash alone (no bit-reversal) is what Kollig-Keller/LK sampling used; it is not true Owen and shows structured artifacts [V, paper Fig 1].
- Sample i, dim d: scramble the **index** (shuffle, gives decorrelation between pixels/dimensions when each has its own seed), look up Sobol dim d, then Owen-scramble the **value** with seed hashed from (pixel seed, d). Paper: Owen-Sobol gets about O(N^-3/2) on smooth integrands vs O(N^-1) for digit scrambling and O(N^-1/2) independent [V].
- Cost: Sobol dims 0,1 need no table (dim0 = bit-reverse, dim1 = ~32-step loop or 5 mat-vecs); dims >=2 need direction-number tables (Joe-Kuo; 32 u32 per dim). In WGSL, put tables in a `var<storage,read>` or `var<uniform>` array: dims 0..~16 cover almost everything if you *pad* beyond via shuffles. ~10-20 u32 ALU ops per scalar, no divergence. Excellent vs memory-bound texture fetch.
- Pitfalls: power-of-two sample counts give best stratification (progressive accumulation should add samples in blocks of 2^k when possible); sample index overflow past 2^32 not an issue in practice. Use fixed seeds per pixel so the same index sequence is reproducible across tiles/dispatches (essential for deterministic debug).
- Padding (decorrelating dimension groups): reuse the first 2D Sobol pairs with a different index-shuffle seed per bounce/group — this is Burley's recommended trick and what keeps table size small [V].

**ZSobol / Morton-ordered Sobol (Ahmed & Wonka 2020; pbrt-v4 `ZSobolSampler`)** — Prototype.
- Map pixel (x,y) to Morton index m (Z-curve), randomize/permute its base-4 digits per dimension with a hashed, nested permutation, then sample index = (m << log2(spp)) | sampleIndex-digit-permutation. Result: adjacent pixels get *different, well-stratified* subsequences of the same Sobol, giving blue-noise-like (low-frequency-suppressed) error at low spp AND full progressive/Owen behavior. Zero memory, ~30-50 ALU ops/dimension. pbrt-v4 uses it as default for its tests [M].
- WGSL fit: all integer ops, trivial. Pad to power-of-two resolution (log2 of max dimension); fine in WGSL with `countLeadingZeros`/`insertBits`/`extractBits`. Mark: spp must be a known power of two at planning time to get full benefit.

**PMJ02 / PMJ02bn** — Later (reference-only). Tables (precomputed, ~4k-64k points); pbrt-v4 has `PMJ02BNSampler` (blue-noise variant). Needs table in memory and Cranley-Patterson shifts; progressive, 2D only. Owen-Sobol is strictly easier on GPU [M].

**Rank-1 lattices** — Prototype (inside Belcour-Heitz). Good for fixed N and for tiled screen-space samplers because every point set is a shift; progressive extension needs radical-inverse-ordered generators (extensible lattices). 1 mul + frac per dim; GPU-trivial.

**R2 / golden-ratio additive (Roberts)** — Adopt now for **per-frame temporal shifts** only (frame index n -> frac(n*phi)). Cheap (1 mul+frac), good 1D/2D discrepancy, trivially progressive; but not stratified in higher dims and no Owen decorrelation. Use for Cranley-Patterson rotation of blue-noise tiles across frames, and for sub-pixel jitter in TAA.

**Halton** — Reference only. Radical inverse in base 3,5,7... needs loops/divisions; poor high-dim correlation; Sobol dominates on GPU.

**Orthogonal-array samplers (Jarosz et al. 2019, "Orthogonal array sampling")** — Reference only/Later. Gives (t,m,s)-nets-like stratification for multi-dim sets of fixed size; table-driven, little advantage over Owen-Sobol in progressive setting.

**Cost per sample in a shader (estimate [M])**: PCG hash ~5-8 ALU; Owen-Sobol dim0/1 ~15 ALU; ZSobol ~40; blue-noise texture fetch = 1 texture load (L1-cached for tiny 64x64 tile; ~20-40 cycles latency but hidden) + frame shift ~3 ALU. All three are negligible against a BVH traversal; choose by variance, not by ALU. Register pressure: carry only `(pixelSeed, sampleIndex, dimCounter)` as 3 u32 in the path state; never hold arrays of samples.

### 3.2 Blue noise

- **Void-and-cluster (Ulichney 1993)** — generator for static blue-noise masks (64x64..256x256); offline. Reference only (we only need results).
- **STBN (Wolfe 2022)** — Adopt now (real-time). Textures with blue-noise in space *and* blue-noise-like (high-pass) in time per pixel, so a temporal filter (EMA) sees well-distributed values. Typical shipped sizes 128x128x64 for scalar/vec2/unit-vector/cosine-weighted vector (EA SEED and NVIDIA publish ready-made sets) [M]. Cost: 1 texel fetch per sample vector; keep as `texture_3d<f32>` or 2D array (rgba8/rgba16) — 128*128*64 * 4B = 4 MB.
- **FAST (Donnelly 2024)** — Prototype. Analytic loss for post-filtering error on Heaviside integrands; produce textures matched to arbitrary filters (box, edge-aware, EMA alpha). Reported gains over STBN in AO, volumetric ray-march, dithering; Fig.1 shows STBN with box denoiser gives blocky artifacts and higher RMSE than FAST noise tuned for box [V]. Their repos include generator + importance-sampled variant [S]. Implication for us: noise should be **tuned to our denoiser/upscaler filter** (OIDN network receptive field is not Gaussian; FSR2-style temporal uses small-window clamp). Treat "blue noise" as a tunable asset in the texture bank.
- **Screen-space permutation approach (Heitz & Belcour 2019)** — Adopt as the *design reference* for the hybrid sampler: take a good progressive sequence (Sobol), scramble per pixel with seeds taken from a ranking/scrambling tile optimized (simulated annealing) so neighboring pixels' errors anti-correlate; works for **any integrand** because it optimizes the *point sets'* distance, and keeps progressive convergence. Memory: ranking+scrambling tiles = 128x128 (x1 u8 each per 2 dims) and the sobol table [M].
- **Belcour & Heitz 2021** — new rank-1-lattice-based tile sampler, a GPU-friendly optimizer, lower post-denoising error than the 2019 sampler; blue noise only optimized for *pairs* of dimensions, so shade code must consume dims in pairs [S, matches abstract]. Uses L2 error after denoising as quality metric.
- **Ahmed & Wonka 2020** — hierarchical Morton-ordered pixel permutation of Sobol; the origin of ZSobol. Spatial blue noise at *every* spp rather than only fixed spp.
- **Blue noise vs denoisers/TAA**: Blue noise error is concentrated in high frequencies, so Gaussian/bilateral/a-trous/OIDN-class networks remove it more efficiently (that is the entire premise of [Panteleev & Schied 2019, cited in FAST]). Blue-noise error **with temporal reuse** only works if the sequence in time is also designed (STBN) or you use shifted R1. Warnings: (a) any non-point-preserving transform (reprojection with bilinear resampling, DLSS/FSR jitter ≠ 0) destroys the carefully optimized correlation partially [FAST cites Keller 2019]; (b) combining jitter-based TAA subpixel offsets with blue noise sampling in the same dims is fine as long as they are independent dimensions; (c) neural denoisers trained on white-noise inputs (OIDN's training data) may behave differently on blue-noise input — **verify on our network; OIDN blue-noise support is not documented [unverified]**; if the denoiser misbehaves, retrain/fine-tune or fall back to white/Sobol for denoiser-fed buffers.

### 3.3 Recommendation matrix

| Use case | Sampler | Per-dim cost | Notes |
|---|---|---|---|
| (a) Progressive reference | Owen-Sobol, hash seeds per pixel, padded dims; optionally ZSobol for early-frame stratification | ~15 ALU / dim, no memory | Unbiased, deterministic, converges fastest for smooth integrands. No blue noise tiles. Final image on 2^k spp |
| (b) 1-spp realtime + denoiser + temporal upscaler | STBN or FAST texture for bounce-0/1 dims (light sample, BSDF direction, AO), R1-shifted per frame; PCG hash above that | 1 texture load + 3 ALU | Tune noise to the denoiser filter; keep sample-dimension pairs aligned with textures' pair structure |
| (c) Hybrid, one code path | Heitz-style ranking+scrambling tiles on Owen-Sobol with a `mode` flag | tile lookup (2 u8) + Owen-Sobol | Gives blue-noise at 1-4 spp *and* clean progressive; best "single sampler module" for the API |

Decision: ship one `Sampler` interface (`init(pixel, frame, sampleIndex)`, `next1D`, `next2D`) with implementations: `OwenSobol` (reference), `BlueTile` (realtime), `Hash` (fallback/debug).

### 3.4 Reuse

**ReSTIR DI (Bitterli 2020)** — Adopt now (real-time), with caveats. Streaming RIS keeps M candidates in one reservoir (y, W, wsum, M); temporal reuse with motion vectors, spatial reuse with 1-5 neighbors, biased (normalize by count M) or unbiased (re-trace visibility for MIS) variants. Cost on GPU: dominated by shadow rays for candidate visibility (only the final selected sample per pixel in biased mode); reservoir ~16-32 B/pixel ×2 (prev/cur). Memory-bandwidth-bound spatial gather. Bias/cost trade: biased = 1 shadow ray/pixel/pass; unbiased = +1 shadow ray per neighbor reused. Original paper's claim: interactive with millions of emissive triangles [M; original table figures not extracted].
- WGSL map: reservoir = `struct { y: vec4<f32>/lightIdx+uv, wSum, M, W }` (see 5.3); atomics not required (one thread owns its pixel). Needs prior-frame buffers, motion vectors, G-buffer normals/depth (we can get from primary hit). Fits compute; avoid pointer-chasing in spatial gather: cache neighbor G-buffer in workgroup shared memory.

**ReSTIR GI (Ouyang 2021)** — Prototype. Reservoirs store the secondary-vertex sample (position/normal/outgoing radiance) from 1 indirect bounce; spatial reuse with Jacobian; 1 extra ray per pixel per frame. Reservoir ≈ 48 B [M]. The cheapest GI-through-reuse; combines naturally with a hash radiance cache for the rest of the path.

**GRIS / ReSTIR PT (Lin 2022)** — Later. GRIS formalizes resampling with arbitrary shift maps + generalized balance heuristic MIS; ReSTIR PT = path reservoirs + hybrid random-replay/reconnection shift maps, reconnection at a vertex satisfying roughness/distance criteria. 40-80 ms/frame on 3090 at 1080p in prototype [V from 2026 paper intro]. Pre-requisites for WebGPU: deterministic per-path RNG (replay), large reservoir storage (88 B baseline -> 64 B in Enhanced [V]), multiple trace passes for spatial MIS.

**ReSTIR PT Enhanced (Lin, Kettunen, Wyman 2026)** — Prototype *after* basic ReSTIR. Halves spatial-reuse shift cost via **reciprocal neighbor selection**, footprint-based reconnection criteria, **duplication maps** to reduce spatiotemporal correlation (counts duplicates in 17x17 neighborhoods to throttle history), unified direct+indirect in the same reservoirs; **2-3x faster**: Fig.1 37.1 ms -> 12.6 ms; per-pixel state 2×(88+16) B -> 2×64 B; sub-pass timings in the 3.7-6.2 ms range on their test machine [V; GPU model not extracted from the text slice: check Section 6]. Rating: Reference now, Prototype later.

**Others (2023-2026)**: ReSTIR BDPT (Hedstrom 2025: caustics/hard-to-reach lights) [S], ReSTIR PG (Zeng 2025: resampled paths train guiding distributions to improve candidate generation; keeps real-time) [S], Area ReSTIR (Zhang 2024) [M] (area-sampling of lens/pixel for DOF/AA/motion blur), ReSTIR for volumes (Lin 2021 "Fast Volume Rendering with Spatiotemporal Reservoir Resampling") [M], conditional ReSTIR [M, unverified], reservoir compression (pack W/M in f16, store light indices as u32, quantized UV): standard trick, the Enhanced paper's 88->64 B is an example [V]. Also Kettunen 2023 "Conditional resampled importance sampling and ReSTIR" fixes correlation/bias [M].

**Spatial-hash radiance caches (SHaRC)** — Prototype. World-space hash grid keyed by quantized position (+ normal bucket, LOD by distance); `update` pass (path-trace a small fraction of pixels, write radiance into voxel via atomics), `resolve` pass (accumulate/temporal blend), `compaction`; consumers query at the first non-specular hit after a path-length cutoff to terminate. Default **2^22 entries** (hash entries + voxel data buffers) [S]. WGSL: needs `atomic<u32>` add (radiance in fixed-point, e.g. 18-bit mantissa style) and `atomicCompareExchangeWeak` for hash insert; works in WebGPU (no 64-bit atomics -> use two u32 lanes, packing key as 32-bit with collision check on separate checksum). Path-space reuse caveat: caches are low-frequency (rough diffuse indirect), biased, lag under motion.

Cost on GPU / WebGPU specifics: ReSTIR passes are bandwidth-bound; use f16 storage if `shader-f16`, `workgroupBarrier` + shared-memory tiling for the 3x3..9x9 neighbor gather, and subgroup ops (if exposed) for reservoir merging within a wave (not required).

### 3.5 Many lights

- **Light BVH / light tree (Conty Estevez & Kulla 2018)** — Adopt now for reference mode. Bound = AABB + orientation cone + energy; importance of node = energy × cos bound / dist². Traverse stochastically choosing a child by importance (single descent, O(log N)), with adaptive splitting for close-by nodes to reduce noise. Used in Cycles, Arnold [M]. Build: top-down SAH-like with orientation cost, CPU build OK; GPU traversal is a short loop (stack depth ~ log N) and returns (light index, pmf). Cost: ~log2(N) node fetches; 32-48 B/node.
- **Stochastic lightcuts** (Yuksel 2019) — Prototype, if per-pixel adaptive cut size matters; heavier.
- **RIS with alias tables** — Adopt now (real-time). Build alias table over light power on CPU (or GPU), draw 8-32 candidates from it (O(1) each) → RIS → 1 shadow ray. Power-based selection ignores distance/visibility, so noise for large scenes; this is why ReSTIR DI adds reuse. Light BVH ± RIS is the strongest combination.
- **Power-based selection** — Baseline/fallback; trivially cheap.
- **Environment map importance sampling** — Adopt now: (1) piecewise-constant 2D distribution (marginal+conditional CDF; 2 binary searches, no divergence worry) or **alias table over texels** (1 fetch + 1 compare), or (2) **hierarchical warping** (Clarberg 2005 Wavelet Importance Sampling / mip-quad-tree descent, 1 fetch per level, preserves stratification — suits QMC); use equal-area octahedral mapping to avoid pole singularities. Mip-pyramid descent can reuse our `downsampler` library to build the importance pyramid (reduction pyramid in one pass). Combine with MIS vs BSDF sampling; optionally **portal/guided** sampling for windows.
- **Area lights**: **Arvo triangle/Quad solid-angle sampling** (spherical triangle, Arvo 1995) and **Ureña 2013 spherical rectangle** sampling: uniform in solid angle, noise-free close to lights vs area sampling; cost ~40-80 ALU with a few trig/sqrt; use `inverseSqrt`, avoid acos/asin where possible; fall back to area sampling when solid angle small (far lights, <~1e-3 sr) because spherical sampling loses precision. In WGSL f32: numeric precision issues near degenerate triangles — clamp. Rating: Adopt now for emissive quads/triangles; spherical-cap for sphere lights.

### 3.6 Path guiding

- **Practical path guiding (Muller 2017, SD-tree)** — Reference only. Spatial binary tree + directional quadtrees, trained by progressive passes with doubling spp; unbiased, requires radiance-record write via atomics per path vertex. CPU-friendly; GPU port is possible but buffer-heavy.
- **OpenPGL** — Reference only; integrates in Embree-based renderers; VMM-based with product sampling for BSDF. CPU library, not callable from WebGPU. Study its API for our hook design.
- **Real-time GPU (2023-2025)**: Huang 2024 (NASG, GPU wavefront with small MLPs trained online) [S]; Dong 2025 neural parametric mixtures [S]; Rath 2025 Neural Resampling (CPU renders use idle GPU) [S]; ReSTIR PG 2025 (guiding from ReSTIR reservoirs) [S]. All depend on large buffers + per-frame training or heavy MLP evaluation per bounce.
- **WebGPU feasibility**: *Not for v1.* Missing: 64-bit atomics, subgroup matrix (experimental in Chrome), persistent-thread scheduling; per-path MLP inference per bounce needs cooperative-matrix. If WebNN/GPU ML exists, an offline-trained guiding field would have to cross the WebNN/WebGPU boundary each frame. **Cheap alternative that gets 60-80% of the benefit on indirect diffuse (my estimate, unverified): spatial-hash radiance cache + RIS-based NEE + product-of-cosine/BSDF with env map pdf.**
- **Rating**: Later (design the `PathState` & sampler API so a guiding provider (`sampleDirection(state) -> dir, pdf`) can be slotted in).

### 3.7 Russian roulette / splitting

- **Baseline RR**: terminate with probability 1 - min(1, maxComponent(throughput)), start at bounce >=3; throughput weight compensated. Zero cost. Adopt now.
- **EARS (Rath, Grittmann, Herholz, Weier, Slusallek, Bälint 2022)** — Prototype for reference mode. Estimates per-region (voxel / SD-tree leaf) mean radiance (second moment, cost) via a pilot/progressive pass and chooses RR survival probability or splitting factor to maximize **efficiency = 1/(variance × cost)**; claims large gains in equal-time MSE vs. throughput RR and prior RR/splitting [M; PDF not obtained — the earlier arXiv ID was wrong]. Needs per-voxel storage of (E[L], E[L²], cost) - similar to SHaRC's grid — so build the hash grid once, share it with the radiance cache. Splitting complicates a wavefront queue (dynamic ray spawning; requires append to indirect queue with atomic counters, plus a RNG scheme for children).
- **Adjoint-driven RR (Vorba & Křivánek 2016)** — Reference only (needs adjoint/radiance cache; similar infrastructure to EARS).

## 4. For the spec

1. `Sampler` module with **three** implementations: `OwenSobol` (default reference), `BlueTile` (STBN/FAST-textured), `Hash` (PCG, debug). Interface per-dimension pair; dimension index = bounce*K + slot (K fixed budget of 4-8 pairs/bounce), padded with index-shuffles. Seeds: `pixelSeed = hash(pixel.x, pixel.y, imageSalt)`.
2. Include a **dimension map** doc: dims 0-1 pixel jitter, 2-3 lens, 4 time/wavelength, 5-6 light select/point, 7-8 BSDF lobe/dir, 9.. next bounce... keep the first bounces on blue-noise textures in realtime mode.
3. Ship textures in an **asset bank**: STBN (scalar, vec2, unit vector, cosine-hemisphere) 128x128x64; FAST variants; ranking+scrambling tiles. Load as `texture_2d_array`/`3d`; with bindless, make these always-resident.
4. **Light sampling** layers: light BVH (reference), alias-table RIS (realtime), env-map alias/mip-warp. Provide `sampleLight(ctx, u) -> {L, pdf, dist}` API so ReSTIR DI can wrap it.
5. **Reservoir struct** shared across DI/GI (see 5.3), ≤32 B for DI; reserve 64 B layout for PT.
6. Add a **hash-grid** utility (shared by SHaRC, EARS stats, many-lights importance cache); design key packing for 32-bit atomics only.
7. RR: throughput-based default; hook for `efficiency-aware` provider later. Dynamic splitting not in v1 (wavefront complexity).
8. Reference mode must be able to **disable all reuse + blue-noise** and still match pbrt-v4 at equal spp.

## 5. Pseudo-WGSL building blocks

### 5.1 Hash-based Owen-scrambled Sobol (Burley 2020; constants verified against paper)

```wgsl
fn rev32(x: u32) -> u32 { return reverseBits(x); }

fn lk_perm(x_in: u32, seed: u32) -> u32 {
  var x = x_in + seed;
  x ^= x * 0x6c50b47cu;
  x ^= x * 0xb82f1e52u;
  x ^= x * 0xc7afe638u;
  x ^= x * 0x8d22f6e6u;
  return x;
}
fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
  return rev32(lk_perm(rev32(x), seed));
}

// Sobol dims 0,1 without tables.
fn sobol2d(index_in: u32) -> vec2<u32> {
  var index = index_in;
  var s0 = reverseBits(index);          // dim 0 (van der Corput)
  var s1 = 0u; var v = 1u << 31u;       // dim 1
  for (var i = 0u; i < 32u; i++) {
    if ((index & 1u) != 0u) { s1 ^= v; }
    v ^= v >> 1u; index >>= 1u;
    if (index == 0u) { break; }
  }
  return vec2<u32>(s0, s1);
}

fn hash_combine(a: u32, b: u32) -> u32 { // any good 32-bit mix, e.g. murmur3 finalizer
  var h = a ^ (b * 0x9e3779b9u);
  h ^= h >> 16u; h *= 0x85ebca6bu; h ^= h >> 13u; h *= 0xc2b2ae35u; h ^= h >> 16u;
  return h;
}

// dimPair = which pair of dims; seed = per-pixel seed. Returns 2 floats in [0,1).
fn owen_sobol_2d(index: u32, dimPair: u32, seed: u32) -> vec2<f32> {
  let pairSeed = hash_combine(seed, dimPair);               // decorrelation by padding
  let idx = nested_uniform_scramble(index, pairSeed);        // shuffle index
  let s = sobol2d(idx);
  let a = nested_uniform_scramble(s.x, hash_combine(pairSeed, 0u));
  let b = nested_uniform_scramble(s.y, hash_combine(pairSeed, 1u));
  return vec2<f32>(f32(a >> 8u), f32(b >> 8u)) * (1.0 / 16777216.0); // 24-bit mantissa, never 1.0
}
```
Notes: for dims>=2 use per-dimension direction-number tables; the padding-by-pairs above is the cheapest and is what Burley recommends. Tested numerically? **No** (not run). Verify first-16 points against pbrt-v4 `SobolSample` on CPU port.

### 5.2 STBN lookup with temporal rotation (real-time mode)

```wgsl
@group(0) @binding(0) var stbn : texture_3d<f32>;   // 128x128x64, rg = vec2 sample set
const PHI : f32 = 0.6180339887498949;

fn stbn_2d(pixel: vec2<u32>, frame: u32, bounceSalt: vec2<u32>) -> vec2<f32> {
  // spatial tile, temporal slice; per-bounce offset in tile so bounces are decorrelated
  let p = (pixel + bounceSalt) & vec2<u32>(127u);
  let z = frame & 63u;
  let v = textureLoad(stbn, vec3<u32>(p, z), 0).rg;
  // Cranley-Patterson shift using frame>>6 (cycle index) so period 64 is not visible
  let cycle = f32(frame >> 6u);
  return fract(v + vec2<f32>(fract(cycle * PHI), fract(cycle * 0.7548776662)));
}
```
(For FAST noise: same lookup; only the texture differs.) The shift preserves blue-noise in space (toroidal shift) and keeps temporal slices decorrelated across cycles.

### 5.3 Reservoir update (streaming RIS, ReSTIR DI)

```wgsl
struct Reservoir {
  lightIdx : u32,   // sample payload (light index or packed)
  uv       : vec2<f32>, // point on light / direction
  wSum     : f32,
  M        : f32,   // confidence count
  W        : f32,   // unbiased contribution weight 1/p_hat * wSum/M
}

fn reservoir_update(r: ptr<function, Reservoir>, lightIdx: u32, uv: vec2<f32>,
                    w: f32, u: f32) -> bool {
  (*r).wSum += w;
  (*r).M    += 1.0;
  if (u * (*r).wSum < w) {      // replace with probability w / wSum
    (*r).lightIdx = lightIdx; (*r).uv = uv;
    return true;
  }
  return false;
}

// merge reservoir src (from neighbor/temporal) given target pdf p_hat_dst at THIS pixel:
fn reservoir_merge(r: ptr<function, Reservoir>, src: Reservoir, p_hat_dst: f32, u: f32) {
  let w = p_hat_dst * src.W * src.M;        // generalized: p_hat * W * M
  let Msum = (*r).M + src.M;
  let replaced = reservoir_update(r, src.lightIdx, src.uv, w, u);
  (*r).M = Msum;                            // fix M after update added 1
}

fn reservoir_finalize(r: ptr<function, Reservoir>, p_hat_sel: f32) {
  (*r).W = select(0.0, (*r).wSum / ((*r).M * p_hat_sel), p_hat_sel > 0.0);
}
```
Per-pixel candidate loop: draw `x ~ q` (alias/light-tree pmf q), w = p_hat(x)/q(x) with p_hat = unshadowed contribution (BSDF*Le*G), update; after M candidates, trace **one** shadow ray for the winner and set W=0 if occluded. Biased spatial/temporal: reservoir_merge with p_hat_dst re-evaluated unshadowed at the destination; unbiased variant needs MIS with visibility rays. Clamp temporal M to <= 20×current M to bound lag (standard practice [M]).

## 6. Bench / comparison ideas

- **Sampler convergence**: RMSE vs spp (1..4096) for Owen-Sobol vs hash PCG vs ZSobol vs pbrt-v4 `ZSobolSampler` / `PaddedSobolSampler` on 3 scenes (Cornell, direct-light-heavy, glossy). Expected: slopes -1/2 for hash, steeper for Sobol on smooth parts [V paper slopes].
- **Denoised error**: 1 spp/4 spp, STBN vs FAST vs Owen-Sobol vs white, through our OIDN + through a simple a-trous (SVGF-like) + through TAA-upscale; metric FLIP/LPIPS/RMSE vs 8k-spp reference. Key question: does OIDN benefit from blue noise at all.
- **Cost**: sampler ns/sample (WGSL microbench: 64 dims per thread, M1/M-series + whatever NVIDIA/AMD available); texture-fetch vs ALU Owen.
- **Reuse**: ReSTIR DI vs RIS-only vs power sampling, at 1k/100k/1M emissive triangles; ms per pass split (candidates / trace / temporal / spatial); memory B/pixel; compare against Capsaicin's (AMD GI-1.0 / ReSTIR DI) numbers locally in ~/Developer/Capsaicin.
- **Many lights**: light tree vs alias table vs uniform at equal time; SHaRC vs no cache for 4-bounce indoor scenes (energy bias check).
- **EARS vs plain RR**: MSE × time at equal time on 2 scenes.
- Compare against `mutracer` (https://github.com/pissang/mutracer): read its sampler / light sampling code (no copying) to see what it does at 1-spp motion; against `three-gpu-pathtracer`.

## 7. Needs Windows / native hardware

- Native pbrt-v4 and Mitsuba runs for reference images and sampler comparisons (CPU only; any machine).
- NVIDIA hardware needed to run NVIDIA ReSTIR PT / Falcor / RTXDI / RTXGI SHaRC samples for timing baselines.
- AMD: Capsaicin runs (D3D12) on Windows for ReSTIR DI/GI-1.0 baselines.
- OIDN native + CUDA for denoiser behaviour on blue-noise inputs vs our WebNN/WebGPU port.
- 64-bit atomics / subgroup-matrix probes need native (Vulkan/D3D12) comparison runs.

## 8. Open questions

1. How does our OIDN-derived network react to blue-noise / FAST input? (Needs experiment; the OIDN docs do not promise it.)
2. Is STBN's temporal structure still valid after FSR2-style jittered reprojection with bilinear history sampling?
3. Pair-wise optimized blue noise vs our per-bounce dimension budget: what is the max number of *useful* blue-noise dim pairs (likely 2-3)?
4. Reservoir f16 packing: does `W` need f32? (Bias from f16 W underflow in dark regions.)
5. SHaRC-style hash with only 32-bit atomics: collision/checksum scheme and fixed-point radiance range.
6. ReSTIR PT Enhanced GPU model/numbers per pass: open the PDF section 6 and extract (not done in this pass). EARS PDF still needed (the arXiv ID found earlier, 2205.04435, was an unrelated paper).
7. Is there an existing WGSL implementation of ZSobol/Owen-Sobol worth reading (three-gpu-pathtracer uses its own sobol)? Check `three-gpu-pathtracer` local copy and `mutracer` source.
8. Numbers marked [M]/[S] above need verification before going into the spec.
