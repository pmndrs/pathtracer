# Participating-media rendering for a WebGPU path tracer: sourced survey (2026-11-21)

Research agent report for the clouds / volumes track (docs/internal/CLOUDS-HANDOFF.md). About 27 searches; no deep full-text reads.
Marks: **[V]** existence / authors / year confirmed by a search result, link returned by search; **[M]** from memory, not re-verified (existence not in doubt,
details are). Costs and GPU-suitability remarks are engineering judgement unless a source is cited. Licences marked "unverified" must be checked before porting.

## (A) Executive summary: ranked directions for a WebGPU path tracer

1. **Majorant grid + DDA, delta tracking for free flight, ratio / residual-ratio tracking for shadow rays.** The production baseline; fits WGSL. 2024-25 papers report
   a flat DDA grid as fast as or faster than BVH / hierarchical skipping (Hofmann and Evans RTG2; Zellmann et al. 2025). Start flat.
2. **Null-scattering MIS framework (Miller 2019).** Gives free-flight and NEE real pdfs, so equiangular / distance sampling combine with delta tracking by MIS.
3. **Equiangular distance sampling (Kulla and Fajardo 2012) for NEE on point, spot and small area lights,** MIS-combined with transmittance sampling. Cheap, removes
   the 1/r^2 noise near lights; the biggest win for lamp-lit fog.
4. **Unbiased ray-marching transmittance (Kettunen 2021)** (much lower variance than ratio tracking per density lookup, more complex); **jackknife transmittance
   (Peters, TOG 2025)** as a deterministic low-bias alternative. A/B both against ratio tracking.
5. **Approximate Mie (Jendersie and d'Eon 2023)**: closed-form eval / sampling, no tables. Keep HG, two-lobe HG, Draine.
6. **Memory-compact density**: r16float / r8unorm bricks + coarse majorant grid. A WGSL NanoVDB port exists (emcfarlane/webgpu-nanovdb, Apache-2.0 per search) at a
   cost in shader size and bindings. NeuralVDB-style compression is not a WebGPU fit now.
7. **Volumetric ReSTIR (Lin, Wyman, Yuksel 2021)**: the real-time path; reservoirs, temporal reprojection, motion vectors, no float atomics. No 2024-26 ReSTIR PT with
   media paper found.
8. **Multiple-scattering approximations (Wrenninge octaves, Hillaire)** as fast modes only, A/B against the unbiased path.
9. **Volume path guiding (Herholz 2019), neural caches (Deep Scattering, MRPNN 2023)**: high effort, high payoff for dense clouds; storage / training hard in WGSL.
   NRC with media not demonstrated anywhere found.
10. **Denoising** with OIDN / SVGF; volumes need aux substitutes (first-scatter position, transmittance-weighted albedo). A volume-specific neural denoiser exists
    (Hofmann et al. I3D 2021).
11. **Spectral media**: chromatic extinction via spectral / hero-wavelength tracking (Kutz 2017, Miller 2019).
12. **Test assets**: WDAS cloud (licence to verify), OpenVDB samples, Moana Island (custom licence). References: pbrt-v4, Mitsuba 3.
13. **Web competition is thin**: Lesar et al. WebGL2 volumetric path tracer (Web3D 2018); Stucki WebGPU / OpenPBR path tracer (2024, volumes unverified);
    three-gpu-pathtracer volumes unverified. No physically based WebGPU volumetric path tracer with clouds was found.

## (B) Per-method entries

### 1. Free-flight and transmittance estimators

- **Delta / Woodcock, ratio, residual ratio tracking.** Novak, Selle, Jarosz, *Residual ratio tracking for estimating attenuation in participating media*, SIGGRAPH Asia
  2014 [V] https://cs.dartmouth.edu/wjarosz/publications/novak14residual.html . Residual tracking subtracts an analytic control medium and tracks the remainder.
  Needs a majorant (and a control density); variable-length loops (divergence). pbrt-v4 (Apache-2.0), Mitsuba 3 (BSD) [M]. Patent US9767601, listed expired [V].
- **Spectral and decomposition tracking.** Kutz, Habel, Li, Novak, TOG 36(4), SIGGRAPH 2017 [V] https://history.siggraph.org/?p=102427 . Spectral tracking:
  chromatic extinction with per-channel weights; decomposition tracking: control part (analytic / coarse grid) + residual with a smaller majorant, fewer lookups.
- **Null-scattering path integral.** Miller, Georgiev, Jarosz, TOG 38(4), SIGGRAPH 2019 [V] https://cs.dartmouth.edu/wjarosz/publications/miller19null.html .
  Null-collision techniques become path sampling techniques with known pdfs, enabling MIS with equiangular / NEE / bidirectional. Pure path-state arithmetic.
  pbrt-v4 `VolPathIntegrator` r_u / r_l scheme [M].
- **Integral formulations of volumetric transmittance.** Georgiev, Misso, Hachisuka, Nowrouzezahrai, Krivanek, Jarosz, TOG 38(6), SIGGRAPH Asia 2019 [V]
  https://cs.dartmouth.edu/wjarosz/publications/georgiev19integral.html . Power-series view; ratio tracking etc. as Monte Carlo estimators of it.
- **Unbiased ray-marching transmittance.** Kettunen, d'Eon, Pantaleoni, Novak, TOG 40(4), SIGGRAPH 2021 [V] https://research.nvidia.com/publication/2021-06_An-Unbiased-Ray-Marching ,
  https://arxiv.org/pdf/2102.10294 . Zeroth-order ray-marching term most of the time, rare higher orders remove the bias; U-statistics and an equidistant comb.
  Reported up to orders of magnitude lower variance at similar lookup counts. Fixed-step loop: WGSL-friendly. Code / licence unverified.
- **Jackknife transmittance and MIS weights.** Peters, TOG 44(6), 2025 [V] https://publications-cgv01.ewi.tudelft.nl/papers/830 , https://momentsingraphics.de/SiggraphAsia2025.html .
  Deterministic, low-bias transmittance; minimum-variance unbiased combination of two estimates; also MIS weights. Reported fewer fireflies and faster than unbiased
  ray marching. Code unverified.
- **Progressive null-tracking.** Misso, Li, Burley, Teece, Jarosz, SIGGRAPH 2023 [V] https://cs.dartmouth.edu/~wjarosz/publications/misso23progressive.html .
  Procedural media with unknown majorants (progressively adapted), better ratio-tracking importance sampling; bias analysed in MSE terms.
- **Multi-density Woodcock tracking** (EG PGV 2025) [V, abstract]: multi-channel scientific volumes; low priority.

### 2. Majorants and acceleration

- **Majorant grids with multi-level DDA.** Hofmann and Evans, *Efficient Unbiased Volume Path Tracing on the GPU*, Ray Tracing Gems II (2021) [V]
  https://lgdv.tf.fau.de/pub/p2021/efficient-unbiased-volume-path-tracing-on-the-gpu-invisible . Grid of bounds at 1/8-1/16 resolution, multi-level DDA. Code open (licence unverified).
- **Hierarchical skipping may not pay off**: EuroVis 2024 AMR path tracing (DDA grid beat BVH) and Zellmann et al. 2025 (diminishing returns) [V]
  https://diglib.eg.org/bitstream/handle/10.1111/cgf15095/v43i3_20_cgf15095.pdf , https://arxiv.org/pdf/2504.04564 .
- **Interactive path tracing and reconstruction of sparse volumes.** Hofmann, Hasselgren, Clarberg, Munkberg, I3D 2021 [V] https://research.nvidia.com/publication/2021-03_Interactive-Path-Tracing .
  NanoVDB + temporally stable neural denoiser, interactive 1080p.
- **NanoVDB / PNanoVDB.** Museth, SIGGRAPH Talks 2021 [M] https://research.nvidia.com/labs/prl/nanovdb/nanovdb2021.pdf . WGSL port: emcfarlane/webgpu-nanovdb
  https://emcfarlane.github.io/webgpu-nanovdb (Apache-2.0 per search; mentions PicoVDB, a WebGPU-oriented format). NanoVDB itself MPL-2.0 [M].
- **GPU volume rendering with hierarchical compression using VDB.** Zellmann, Jaros, Amstutz, Wald, Eurographics 2025 [V] https://arxiv.org/pdf/2504.04564 (WDAS cloud; notes a ZFP WebGPU port by Usher et al.).
- **NeuralVDB.** Kim, Lee, Museth, arXiv 2208.04448 (2022), TOG 2024 [V] https://arxiv.org/abs/2208.04448 . 10-100x+ compression; inference per lookup: poor fit for a hot WGSL loop.
- **f16 / unorm bricks** [M]: r16float `texture_3d`; the filter footprint must stay consistent with the majorant.

### 3. Variance reduction in media

- **Equiangular sampling.** Kulla and Fajardo, CGF 31(4), EGSR 2012 [V] https://diglib.eg.org/handle/10.1111/v31i4pp1519-1528 . Closed form for homogeneous media,
  MIS with distance sampling for heterogeneous. Pure math.
- **Volume path guiding (zero-variance random walk theory).** Herholz et al., TOG 38(3), 2019 [V] https://cgg.mff.cuni.cz/~jaroslav/papers/2019-volume-path-guiding/index.html .
  Guides distance, direction, RR / splitting with vMF mixtures on a spatial tree; online fitting (reductions in WGSL, no float atomics). Heavy.
- **A practical path guiding method for participating media** (Deng, Wang, Wang, Holzschuch, CVM 2020) [V] https://link.springer.com/article/10.1007/s41095-020-0160-1 .
- **Variance-aware path guiding** (Rath et al., SIGGRAPH 2020) [V] surfaces. 2022-24 volume guiding not found (open search).
- **Once-more NEE** (KIT 2022) [V] https://cg.ivd.kit.edu/publications/2022/once-more-nee/2022_omnee.pdf (unread).
- Emission sampling in heterogeneous media: no dedicated source found.

### 4. Resampling

- **Volumetric ReSTIR.** Lin, Wyman, Yuksel, TOG 40(6), SIGGRAPH Asia 2021 [V] https://research.nvidia.com/publication/2021-11_fast-volume-rendering-spatiotemporal-reservoir-resampling ,
  https://graphics.cs.utah.edu/research/projects/volumetric-restir/ . Multi-dimensional path-space reservoirs; env maps, area lights, volumetric emission; cheap biased
  candidate estimates without shading bias. Code / licence unconfirmed.
- **ReSTIR PT / GRIS** (Lin et al., SIGGRAPH 2022) [V] https://graphics.cs.utah.edu/research/projects/gris/sig22_GRIS.pdf ; no ReSTIR PT with media 2024-26 found.
- **Rendering participating media using path graphs** (Hu, Deng, Luan, Hasan, Marschner, arXiv 2404.11894, 2024) [V]: offline path reuse for high-albedo media.

### 5. Caching and learning

- **Deep Scattering** (Kallweit, Muller, Mehta, Novak, Marrs et al., SIGGRAPH Asia 2017) [V] https://arxiv.org/pdf/1709.05418 : network predicts cloud multiple scattering from hierarchical density features.
- **MRPNN** (Hu, Yu, Liu, Yan, Wu, Jin, SIGGRAPH 2023) [V] https://sites.cs.ucsb.edu/~lingqi/publications/paper_mrpnn.pdf , code https://github.com/unicorn2022/MRPNN (licence unchecked).
- **Neural radiance caching** (Muller et al., SIGGRAPH 2021) [V] https://arxiv.org/pdf/2106.12372 : surfaces; NRC with media not found.
- **GSCache** (arXiv 2507.19718, 2025) [V, abstract]: Gaussian-splat path-space radiance cache for volume path tracing.
- NEPHELE (arXiv 2303.04086) and arXiv 2401.14051 (neural RTE integration for clouds / smoke) [V, abstract].

### 6. Multiple-scattering approximations

- **Octaves**: Wrenninge, Kulla, Lundqvist, *Oz: The Great and Volumetric*, SIGGRAPH 2013 talk [V] https://fpsunflower.github.io/ckulla/data/oz_volumes.pdf ; Wrenninge,
  *Art-Directable Multiple Volumetric Scattering*, SIGGRAPH 2015 Talks [V]. Attenuate extinction / scattering / anisotropy per octave; biased.
- **Hillaire**: sky / atmosphere LUTs (EGSR 2020) and Frostbite clouds (SIGGRAPH 2016 course) [V].

### 7. Phase functions

- **Approximate Mie** (Jendersie and d'Eon, SIGGRAPH 2023 Talks) [V] https://research.nvidia.com/labs/rtr/approximate-mie (built here, P5.v14).
- Two-lobe HG; Draine 2003; SGGX microflake (Heitz et al., HPG 2015) [M]; chromatic (per-wavelength) Mie.

### 8. Denoising and convergence

- I3D 2021 (above) and Volumetric ReSTIR use denoisers. Kernel-prediction volume denoising (2025) [V title] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12028196/ .
- No source on OIDN / OptiX behaviour on volumes; aux candidates: first-scatter position, transmittance-weighted albedo, first-collision depth. Equal-time comparisons matter (cost per sample varies hugely in media).

### 9. Spectral and emissive media

- Spectral tracking (Kutz 2017), null-scattering framework (Miller 2019). Emission: source term at collisions; Volumetric ReSTIR handles volumetric emission.
  Blackbody colour maps are data. "Don't Splat your Gaussians" (arXiv 2405.15425) [V title]: ray-traced Gaussians, a different representation.

### 10. Production and open renderers

| Renderer | Confirmed | Source |
|---|---|---|
| Disney Hyperion | path traced; progressive null-tracking targets it (WDAS cloud) | https://disneyanimation.com/technology/hyperion/ [V] |
| Pixar RenderMan | production volume rendering paper / course exist (unread) | https://graphics.pixar.com/library/ProductionVolumeRendering/paper.pdf [V title] |
| Arnold | equiangular + importance sampling (Kulla and Fajardo) | [V partly] |
| Weta Manuka, Houdini Karma | not researched | gap |
| Blender Cycles | delta tracking, heterogeneous mode; Apache-2.0 [M] | kernel source in research/sources/native/cycles |
| Mitsuba 3 | null-collision volpath; BSD [M] | |
| pbrt-v4 | VolPath, null-scattering MIS (r_u / r_l); Apache-2.0 [M] | |

Overview: Novak, Georgiev, Hanika, Jarosz, *Monte Carlo methods for physically based volume rendering*, EG 2018 STAR [V] https://cs.dartmouth.edu/wjarosz/publications/novak18monte.pdf .

### 11. Web state of the art

- Lesar, Bohak, Marolt, *Real-time interactive platform-agnostic volumetric path tracing in WebGL 2.0*, Web3D 2018 [V] https://lgm.fri.uni-lj.si/research/volumetric-path-tracing-framework/ (WebGL2 fragment shaders, transfer functions).
- Stucki and Ackermann, WebGPU + OpenPBR path tracer [V] https://arxiv.org/html/2407.19977v1 (volumes unconfirmed).
- pocketpt-webgpu [V]: no volumes. webgpu-nanovdb: data access, not a renderer. three.js WebGPU volumetric lighting thread [V]: froxels, not path tracing.
- No physically based WebGPU volumetric path tracer with clouds found.

### 12. Test assets

- WDAS cloud https://disneyanimation.com/resources/clouds (licence: one listing says CC BY-SA 3.0, verify the README); `_sixteenth` low-res version.
- Moana Island (custom licence, not the cloud). OpenVDB samples [M]. pbrt-v4 / Mitsuba 3 references already set up.

## (C) Open-source implementations worth reading

| Repo | Licence | Use |
|---|---|---|
| pbrt-v4 `VolPathIntegrator` | Apache-2.0 [M] | portable with attribution |
| Mitsuba 3 `volpath`, `volpathmis` | BSD-3 [M] | portable |
| Blender Cycles volume kernels | Apache-2.0 [M] | portable (already in research/sources) |
| emcfarlane/webgpu-nanovdb | Apache-2.0 (per search; confirm LICENSE) | portable |
| NanoVDB / OpenVDB | MPL-2.0 [M] | read; file-level copyleft |
| unicorn2022/MRPNN | unchecked | read |
| Lesar framework | unchecked | read |
| NVIDIA RTX SDK, Falcor, NRD | NVIDIA licences | ideas only (CLAUDE.md) |

## (D) Gaps and unverified

No full paper read; the Kettunen 90%, Jendersie 95% and MRPNN speedup figures are summaries. Not searched / found: volume path guiding 2022-26, NRC with media, ReSTIR PT
with media, spectral media 2022-26, emissive sampling, SGGX / Cornette-Shanks details, Manuka / Karma / Arnold internals, OIDN on volumes, volume adaptive sampling,
OpenVDB sample URLs, three-gpu-pathtracer volume support, code releases of Kutz 2017 / Kettunen 2021 / Peters 2025, WDAS licence, renderer licences (from memory).
