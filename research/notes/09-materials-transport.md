# 09 - Materials, light transport, spectral (speed lens)

Scope: BSDF/material stack, sampling, integrator, spectral, media, environment, robustness for the full PBR path tracer.
Note: "SPD" in the sibling libs = single-pass downsampler, NOT spectral power distribution. Everything spectral here is
unrelated to that lib.
Provenance: written from domain knowledge plus the downloaded papers' known contents; I did not re-read each PDF line by line and did
not clone code. Items marked (unverified) are from memory and need a check before they go in a spec. Dates 2024-2026 beyond OpenPBR
1.1 and the 2023 VNDF papers are thin here: a follow-up web sweep is advisable (see Open questions).

## 1. TL;DR

- **v1 material = one fixed "uber" BSDF (OpenPBR-shaped subset), single-wavelength-set RGB, no layering solver.** Base diffuse (Lambert/EON-lite)
  + GGX specular (conductor/dielectric, F82-tint metal) + clearcoat + sheen + transmission (thin & thick) + emission. Stored as a packed
  material table (SoA/structured buffer, ~64-96 B/material), indexed by material id from the hit. Same struct drives utility consumers
  (they read only albedo/roughness/emission).
- **Uber-shader first, wavefront-sorted by material *class* (not per material) later.** WGSL has no function pointers or recursion;
  bindless gives texture-array-like indexing, not callable tables. Specialization by feature bitmask via pipeline-override constants is the
  cheap middle ground. MaterialX graphs: compile on the CPU (MaterialXJS) to either (a) texture bakes into the fixed uber params
  (v1), or (b) generated WGSL "pattern" function per graph + switch (v2).
- **Sampling stack to adopt now:** bounded VNDF (Eto-Tokuyoshi 2023) or spherical-cap VNDF (Dupuy-Benyoub 2023) for GGX; both are
  strictly cheaper/lower variance than Heitz 2018. Spherical caps are ~3 lines changed from Heitz and fine for anisotropic; bounded
  VNDF adds a tighter bound for reflection. Use Turquin multi-scatter compensation with one 2D (or analytic fit) E(mu, alpha) LUT.
- **Integrator v1:** unidirectional PT + NEE (one light sample / bounce) + one-sample MIS (power/balance, BSDF vs light), RR on
  throughput after bounce 3, luminance-capped radiance clamp as a *switchable biased* mode. ReSTIR DI/GI plugs into the same light-sample
  slot; MIS must then use the reservoir's sample pdf (see 3.5).
- **Spectral: defer, but design for it now.** Keep a `Radiance` type (vec3 now, vec4-of-wavelengths later) and keep all material
  eval returning `Radiance`; reserve a hero-wavelength path (4 wavelengths, Wilkie 2014) with Jakob-Hanika sigmoid uplifting for v2. Needed
  only for dispersion, honest iridescence/thin-film, fluorescence, metamerism-accurate references. Cost on GPU: ~1.3-2x ALU on
  shading, but BVH traversal cost is unchanged (traversal is wavelength-independent) -> overhead modest when traversal-bound.
- **Volumes:** homogeneous fast path (analytic exp sampling, chromatic via hero wavelength or single-channel selection with ratio
  weights), heterogeneous via majorant-grid delta/ratio tracking, null-scattering formulation (Miller 2019) as the correct framework
  for MIS with NEE in media. NanoVDB traversal (WebVDB `nanovdb-wgsl`) is a natural fit as majorant + density source.
- **Robustness:** Wachter-Binder ray offset (integer offset of mantissa bits) is ~free and correct; adopt it instead of an epsilon.
  Keep shading in f32; use f16 only for storage (textures, LUTs, packed normals) and never for positions/ray origins/accumulators.
- **Reference-only:** BDPT/VCM, SMS/manifold NEE, full stochastic layered-BSDF random walks. Justified in 3.7/3.8.

## 2. Sources

| name | link | year | venue / license | local | why |
|---|---|---|---|---|---|
| OpenPBR Surface spec (1.0, 1.1) | https://github.com/AcademySoftwareFoundation/OpenPBR + https://academysoftwarefoundation.github.io/OpenPBR/ | 2024-25 | ASWF, Apache-2.0 | link | Target material model; MaterialX node definition `open_pbr_surface` |
| OpenPBR reference impl (MaterialX GLSL/OSL/MDL) | https://github.com/AcademySoftwareFoundation/OpenPBR-viewer ; MaterialX `libraries/bxdf/open_pbr_surface.mtlx` (unverified path) | 2024 | Apache-2.0 | link | Layering structure to copy (albedo-scaling slabs) |
| Autodesk Standard Surface | https://autodesk.github.io/standard-surface/ | 2019 | Apache-2.0 | link | Ancestor of OpenPBR; many assets exist |
| Disney BSDF notes (Burley) | https://blog.selfshadow.com/publications/s2015-shading-course/burley/ | 2015 | SIGGRAPH course | papers/transport/burley2015-disney-bsdf-notes.pdf | Principled BSDF, thin/rough transmission |
| glTF 2.0 + KHR_materials_* | https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos | 2021-25 | Khronos, Apache-2.0 | link | Interchange target: transmission, volume, ior, clearcoat, sheen, iridescence, anisotropy, specular, dispersion, diffuse_transmission (unverified: diffuse_transmission ratification status) |
| Bounded VNDF (Eto, Tokuyoshi) | https://gpuopen.com/download/publications/Bounded_VNDF_Sampling_for_Smith-GGX_Reflections.pdf | 2023 | SIGGRAPH Asia Tech Comm | papers/transport/eto2023-bounded-vndf.pdf | Cheaper, lower-variance GGX reflection sampling |
| Spherical-cap VNDF (Dupuy, Benyoub) | https://arxiv.org/abs/2306.05044 | 2023 | JCGT 2023 | papers/transport/dupuy2023-spherical-cap-vndf.pdf | Simplest VNDF; fewer ops than Heitz |
| Sampling GGX VNDF (Heitz) | https://jcgt.org/published/0007/04/01/ | 2018 | JCGT | papers/transport/heitz2018-vndf-ggx.pdf | Baseline VNDF |
| Turquin MS compensation | https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf | 2019 | tech report | papers/transport/turquin2019-ms-compensation.pdf | Energy compensation without a second LUT |
| Kulla-Conty "Revisiting PBS" | https://blog.selfshadow.com/publications/s2017-shading-course/ | 2017 | SIGGRAPH course | papers/transport/kulla2017-conty-revisiting.pdf | Multi-scatter LUT formulation |
| Position-free layering (Belcour) | https://belcour.github.io/blog/research/2018/05/05/brdf-realtime-layered.html (hal-01785045) | 2018 | SIGGRAPH | not saved (download failed) | Cheap deterministic layering via Gaussian lobe moments |
| Thin-film BRDF (Belcour, Barla) | https://belcour.github.io/blog/research/2017/05/01/brdf-thin-film.html | 2017 | SIGGRAPH | not saved (download failed) | Spectral-free (FFT-of-airy) iridescence, RGB-capable |
| LTC sheen (Zeltner et al.) | https://tizianzeltner.com/projects/Zeltner2022Practical/ | 2022 | SIGGRAPH | not saved (download failed) | Multi-scatter sheen via LTC; OpenPBR fuzz layer in practice uses Charlie/LTC variants |
| Hero wavelength (Wilkie et al.) | https://cgg.mff.cuni.cz/~wilkie/Website/EGSR_14_files/WNDWH14.pdf | 2014 | EGSR | not saved (download failed) | Low-variance spectral MC with 4 wavelengths |
| Spectral uplifting (Jakob, Hanika) | https://rgl.epfl.ch/publications/Jakob2019Spectral | 2019 | EGSR | papers/transport/jakob2019-spectral-uplifting.pdf | 3-coeff sigmoid-polynomial RGB->spectrum, 3D LUT (64^3 x 3 x 3 floats) |
| Spectral primary decomposition (Mallett, Yuksel) | https://graphics.geometrian.com/research/spectral-primary-decomposition.html (unverified URL) | 2019 | EGSR | link | 3 basis spectra, linear blend, cheaper than sigmoid LUT, no LUT lookup per eval |
| Null-scattering path integral (Miller et al.) | https://cs.dartmouth.edu/~wjarosz/publications/miller19null.pdf | 2019 | SIGGRAPH Asia | papers/transport/miller2019-null-scattering.pdf | Unified framework for delta/ratio/spectral tracking + MIS |
| Monte Carlo volumetric light transport STAR (Novak et al.) | https://cs.dartmouth.edu/~wjarosz/publications/novak18monte.pdf | 2018 | EG STAR | papers/transport/novak2018-monte-carlo-volumetric-star.pdf | Survey: delta/ratio/residual tracking, majorants |
| Spectral & decomposition tracking (Kutz) | https://dl.acm.org/doi/10.1145/3072959.3073665 | 2017 | SIGGRAPH | link (paywalled?) | Chromatic media without bias via spectral MIS |
| Specular manifold sampling (Zeltner et al.) | https://rgl.epfl.ch/publications/Zeltner2020Specular | 2020 | SIGGRAPH | papers/transport/zeltner2020-specular-manifold-sampling.pdf | Caustics through specular chains |
| Hosek-Wilkie sky | https://cgg.mff.cuni.cz/projects/SkylightModelling/ | 2012 | SIGGRAPH | papers/transport/hosek2012-sky.pdf | Analytic sky radiance, 9-param fit per turbidity/albedo |
| Prague sky model (Wilkie et al.) | https://cgg.mff.cuni.cz/projects/SkylightModelling/ (2021 page) | 2021 | EGSR | not saved (download failed) | Spectral, extends to ground/dusk/space; data-heavy |
| Hillaire scalable sky | https://sebh.github.io/publications/egsr2020.pdf | 2020 | EGSR | papers/transport/hillaire2020-scalable-sky.pdf | LUT-based atmosphere, real-time, multi-scatter approx |
| Path regularization (Kaplanyan, Dachsbacher) | https://jo.dreggn.org/home/2013_pathreg.pdf | 2013 | HPG | not saved (download failed) | Biased-but-consistent roughening for caustic fireflies |
| Wachter, Binder, ray offset | Ray Tracing Gems ch. 6, https://link.springer.com/chapter/10.1007/978-1-4842-4427-2_6 | 2019 | open access book chapter | not saved (download failed) | Robust self-intersection avoidance |
| pbrt-v4 book | https://pbr-book.org/4ed | 2023 | MIT-ish text, Apache code | link | `DielectricBxDF`, `LayeredBxDF` (stochastic layering), `SampledWavelengths` reference |
| Mitsuba 3 | https://github.com/mitsuba-renderer/mitsuba3 | 2022- | BSD-3 | link, not cloned | Spectral mode, principled BSDF, OpenPBR (unverified) |
| Manuka (Fascione et al.) | https://dl.acm.org/doi/10.1145/3182161 | 2018 | ACM TOG | link (paywalled) | Batch-shading spectral PT: reorders by *texture/material* before shading. Directly relevant to wavefront design |

## 3. Techniques and findings

### 3.1 Material model choice and layout

**OpenPBR Surface (1.0 June 2024; 1.1 ~2025, unverified exact changes).** Slab-based: base (diffuse w/ roughness + metal/dielectric
mix), specular (dielectric GGX with F0 from IOR, anisotropy), transmission (+ subsurface/volume), coat (with darkening and
roughening of base below), fuzz (sheen), thin-film (on specular), emission, geometry (thin-walled, opacity, normal, tangent). The spec
defines the layering semantically as "vertically layered slabs" and the reference MaterialX implementation uses albedo-scaling
(energy-conserving weighted sum with Fresnel/albedo-based transmission attenuation), *not* a stochastic random walk. That is exactly the
cheap approach we want. Standard Surface is the strict subset-ish ancestor; Disney Principled maps onto it with known parameter
translation. Rating: **Adopt now** as the canonical parameter set; glTF + KHR extensions as the import mapping (almost all KHR ext map
1:1 onto OpenPBR params, dispersion maps to OpenPBR `transmission_dispersion_*` / abbe number; iridescence -> thin_film; anisotropy ->
specular_anisotropy; diffuse_transmission -> translucent diffuse approximated by Lambert-transmission lobe).

**Material record.** Packed table in a storage buffer: 4x vec4 (base_color+metal, spec_color/roughness/ior/aniso, coat+sheen params,
transmission/emission/flags) + texture indices (bindless: u32 index into binding_array; fallback: layer index in texture_2d_array
atlas set). Keep to <=128 B so a hit-time fetch is two cache lines max. Texture-less params stay inline. **Adopt now.**

**Compile strategies for MaterialX graphs in WGSL:**
- *Bake to uber params (v1):* evaluate constant sub-graphs on CPU; sample-able procedural/texture inputs become per-material textures baked at
  load into the packed texture set (MaterialXJS does the graph; Baker repo exists). Zero shader divergence. Loses true procedurals.
- *Specialization:* generate WGSL per feature-set (bitmask of active lobes) via override constants / string templates; pipeline cache key = bitmask.
  Cost: pipeline-compile time on web (can be seconds on Windows/D3D12: see webgpu compile skill), and splitting wavefront queues per variant.
  Keep variants <=~16. **Prototype.**
- *Uber with dynamic branches on flags:* lobes skipped by uniform-per-material branches. On a fixed hit batch sorted by material class this
  is cheap; unsorted, divergence ~ number of distinct lobe combos in a wave (usually 2-4). **Adopt now.**
- *Per-material generated pattern function + switch:* only for graphs with real procedurals. `switch(material_graph_id)` into generated
  functions; divergence mitigated by wavefront material sort (Manuka-style). **Later (v2).**
- *Bindless material table:* the table is just a storage buffer; bindless helps only for the texture side (index into binding_array).
  No callable tables in WGSL -> a "bindless BSDF" cannot dispatch to arbitrary code. Be honest in the spec about this.

### 3.2 Layering and energy compensation

- **Albedo-scaling slab stacking (OpenPBR reference, Autodesk/Weidlich-Wilkie lineage):** lobe weight = (1 - E_top(view)) etc. Needs directional
  albedo E(mu, alpha) of each layer: one small LUT per lobe class (dielectric GGX: 3D in mu, alpha, F0 or 2D with analytic F0 fit;
  conductor: 2D). Typical LUT 32x32 (or 16^3) f16/f32 = 2-8 KB each, resident in a texture or even a `const` array. Fetch cost = 1-2 bilinear
  taps/shade. **Adopt now.**
- **Kulla-Conty (2017):** compensation term f_ms = (1-E(mu_o))(1-E(mu_i))/(pi(1-E_avg)) needs E and E_avg LUTs (2 small LUTs) +
  Fresnel averaged term. **Turquin (2019):** scale specular by 1 + F0*(1/E(mu_o, alpha) - 1): one directional-albedo LUT, no
  reciprocity guarantee but simple, works with importance sampling unchanged (multiply the whole lobe, and sampling probability unchanged
  = slightly biased pdf, fine in practice). Pick Turquin for speed. **Adopt now.** For dielectric rough *transmission* compensation
  (Turquin extends to it) needs 3D LUT (mu, alpha, eta): put behind a v2 flag.
- **Stochastic layering (pbrt-v4 `LayeredBxDF`, position-free ~ Belcour 2018):** random walk through layers; unbiased but costs a multi-bounce
  loop inside every eval and sample. Too expensive for the fast path. Belcour's Gaussian-lobe moment propagation: deterministic, GPU-friendly,
  but fitting data lives in LUTs and doesn't map cleanly to our sampler. **Reference only** (stochastic), **Later** (Belcour).
- **Clearcoat:** second GGX lobe (alpha 0.0-0.3), IOR 1.5 Fresnel, attenuate base by (1-F_coat(wi))(1-F_coat(wo)) (albedo-scaled; OpenPBR also
  rougheens/darkens base). Selected by probability ~ F_coat-weighted. Cost: one extra VNDF sample/eval only when coat>0 (uniform branch).
  Rating: **Adopt now.**
- **Sheen (fuzz):** Charlie (Estevez-Kulla 2017) NDF + Neubelt visibility approximation, with analytic directional albedo fit for
  layering (cheap, no LUT). LTC sheen (Zeltner 2022): exact multi-scatter-ish but needs LTC matrices (fetch 2 tex + sample from LTC). For
  sampling, cosine-hemisphere sampling + MIS against light is common (sheen lobe is broad). **Charlie + cosine sampling: Adopt now. LTC sheen: Later.**
- **Thin-film / iridescence fast approximations:** Belcour-Barla uses FFT of Airy summation in the 1D (phase) domain; practical path is
  the Atanasov/Dolby-style or "Practical Real-Time Strategies for Accurate Iridescence" style 3-sample evaluation (Fresnel at R, G, B-ish
  wavelengths, 2-order approximation with an XYZ response fit, e.g. Sony-Belcour). GlTF `KHR_materials_iridescence` expects this level.
  RGB: ~3 complex Fresnel evaluations extra (~60-100 ALU). Proper: evaluate at hero wavelengths directly (spectral mode). **RGB 3-wavelength fit: Adopt now (cheap) ; spectral exact: v2.**
- **Cheap microfacet multiple scattering:** Turquin / Kulla-Conty as above suffices; Heitz 2016 random-walk multi-scatter is unbiased but a loop
  (reference only). **Adopt now (compensation), Reference only (random walk).**

### 3.3 BSDF sampling

- **GGX VNDF baseline cost (Heitz 2018):** vector normalize, orthonormal frame, disc sample (sqrt, sincos), warp, normalize: ~ 40-60 ALU,
  one sincos pair, 2 sqrt. Not the bottleneck vs. memory fetches for texture/material, but sampled once per bounce per lobe.
- **Spherical-cap VNDF (Dupuy-Benyoub 2023):** samples a point in a spherical cap of the unit sphere (z in [-wi.z, 1]), then applies the same
  anisotropic stretch: saves about 1-2 transcendental-ish ops and removes a branch, with exactly the same pdf as Heitz/Heitz-VNDF; ~3 lines
  of diff. **Adopt now** (default sampler; fully equivalent).
- **Bounded VNDF (Eto-Tokuyoshi 2023):** reduces the sampling cap from the full hemisphere visible directions to tightened bound for
  isotropic reflection, reducing wasted samples that point below the horizon after reflection (these are rejected) at low roughness; they report
  noticeably lower variance (~ up to 25%? (unverified numeric; see PDF Table/Fig)) and no extra cost for isotropic; anisotropic requires different bound.
  Mix with spherical-cap: implemented in their reference (HLSL). **Adopt now (isotropic reflect path), Prototype (aniso/transmission).**
- **Refraction sampling:** VNDF-half-vector then refract; for dielectrics choose reflect/transmit by Fresnel (Walter 2007). Rough-transmit
  throughput has the (eta^2 |wi.h||wo.h|)/... Jacobian; keep an f32 `eval_pdf` single function to guarantee consistency; this function is
  where most validation bugs live. Test with a white-furnace and a chi-square harness (see 5).
- **Lobe selection:** keep pdf-proportional-to-estimated-albedo selection (OpenPBR weights); avoid evaluating every lobe pdf for MIS (only the one
  selected + the NEE mixture) except where needed; "one-sample model" (Veach 9.2.4) -> evaluate all lobe pdfs only in MIS calls.
- **Cosine sampling alt for diffuse:** Lambert = cosine hemisphere (Malley), cheapest. Disney diffuse/retro: sample cosine, evaluate weight.
- **Rating all above:** Adopt now except where noted.

### 3.4 Light sampling and MIS

- **NEE + MIS:** two strategies (BSDF, light). Use **power heuristic (beta=2)** by default; balance for low variance difference isn't
  worthwhile. One-sample MIS (choose strategy w.p. 0.5 or by pdf ratio, divide by selection prob) is cheaper (1 shadow ray/bounce) than
  evaluating both, but standard NEE+BSDF-hit-MIS path costs 1 shadow ray + continues the path; the continuation ray's emission hit uses the
  MIS weight with the light pdf (needs `light_pdf(prev hit, hit point)` lookup -> need triangle-light id and area/solid-angle conversion, store in hit).
- **Light selection:** v1 = power-based alias table over emissive triangles (+ env map) when light count <~ 1e4; with many lights, hierarchical
  light BVH (Conty-Kulla / "Importance sampling of many lights with adaptive tree splitting", Estevez-Kulla 2018) is the state of the art,
  cost: tree walk ~log N each shading point. **Alias table: Adopt now. Light BVH / ReSTIR many-lights: Prototype (v2).** Cross-ref ReSTIR note.
- **ReSTIR interplay:** the reservoir's final selected light sample has an unnormalized target pdf p_hat; to MIS the BSDF-sampled hit with it you
  must use the pdf the *reservoir procedure* effectively samples with, which isn't tractable in general -> common practice: apply ReSTIR to light
  (NEE) strategy only, use balance heuristic with the *initial candidate* light pdf, accept slight bias; or "talbot MIS" only inside ReSTIR
  resampling (Pairwise MIS for spatial reuse). Decide in spec which of (a) unbiased reference mode = plain NEE+MIS only, (b) ReSTIR
  mode = documented bias. **Spec decision.**
- **MIS compensation (Karlik et al. 2019, "MIS compensation: optimizing sampling techniques in MIS", unverified venue details):** modifies the
  defensive-sampling pdfs to remove negative-contribution technique weights; gain is modest for typical scenes, adds complexity. **Later.**
- **Russian roulette:** terminate probability q = 1 - min(1, max(throughput luminance*etaScale)); start at depth >= 3 (pbrt-v4 uses
  `maxcomp(beta*etaScale) < 1` and depth > 1); a cheap alternative is the "path-guided" RR via the radiance-cache estimate (ReSTIR-GI style;
  Later). Skipping RR early costs more bounces, early RR makes noise; typical net ~10-20% speedup vs fixed depth. **Adopt now** (pbrt-style).
- **Firefly handling:** (1) per-sample radiance clamp (max luminance) = biased, darkens; use a soft-adaptive threshold and expose "bias mode";
  (2) path regularization (Kaplanyan-Dachsbacher 2013): after a diffuse/glossy hit, raise min roughness of following specular lobes by a
  schedule (alpha' = clamp(alpha + c/n^0.25...)), consistent (bias vanishes as sample count grows); trivially cheap on GPU (one max on roughness
  after the first non-delta bounce); (3) per-pixel/variance-based outlier rejection in the accumulator (e.g. median-of-means over sample groups,
  Jung 2020 "robust"); (4) denoiser. For *fast* mode: regularization + clamp; for *reference*: none (unbiased). **Adopt now: clamp + regularization options.**

### 3.5 Caustics and bidirectional methods

- **Manifold NEE / SMS (Hanika 2015; Zeltner 2020):** Newton solve on specular chain per shading point, multi-iteration, requires analytic
  derivatives of shapes (mesh with smooth shading normals needs special care) and divergent control flow per path.
  On GPU wavefront: it is a loop of 5-20 iterations with intersection queries per iteration ->> 10x+ cost to a normal bounce, extreme divergence,
  and only helps scenes with light->specular->diffuse (pool caustics, glass objects lit by small lights). **Reference only**; for the fast mode rely on regularization + denoiser + photon-less "caustic via path regularization/ clamped BSDF-sampled hit of emissive".
- **BDPT / VCM:** needs light subpath storage and connection visibility rays: O(N^2) connection shadow rays at depth N, vertex storage; GPU
  implementations exist but are much slower per-sample than PT with NEE on *typical* scenes and complicate textures/volumes/spectral. Wins in
  difficult light transport (small apertures, indirect-lit rooms, glass). VCM with photon merging adds a spatial hash and radius schedule.
  **Reference only (Windows native compare path; maybe a later offline mode).** Justification: unbounded complexity vs speed-first rule; a good
  denoiser + ReSTIR GI recovers most of the practical wins; the "difficult" scenes are a small fraction of target use.

### 3.6 Spectral

- **Hero wavelength spectral sampling (Wilkie 2014):** sample lambda_0 from importance pdf (visual response / "XYZ-matching" e.g. pbrt `VisibleWavelengthsPDF`),
  other 3 at equal stratified offsets across the 360-830 nm span (pbrt-v4: `SampledWavelengths::SampleVisible`), single path carries 4 radiance
  values; MIS across the 4 wavelength pdfs only matters when pdf varies with lambda (dispersion/chromatic media): terminate secondary wavelengths when
  dispersion occurs (pbrt: `TerminateSecondary()`). Output: accumulate to XYZ via CMFs/pdf, convert to sRGB at pixel resolve.
  Memory: path state is 4 floats instead of 3 (vec4 aligns naturally, free on GPU).
- **RGB->spectrum:** Jakob-Hanika: spectrum s(lambda)=sigmoid(c0 lambda^2 + c1 lambda + c2); LUT 64^3 x 3 x f32(24 B?)... per texel the 3 coefficients: 3D LUT lookup
  (trilinear, 8 taps) per albedo texel *per evaluation* -> expensive in a shader (8 texture reads + sigmoid per wavelength). Mitigations: (a) uplift
  *per texel at texture-load/bake time* to coefficients (store coefficients in the texture rather than RGB; 3 channel f16 fine) so shading does zero LUT lookups,
  only polynomial+sigmoid per wavelength; (b) Mallett-Yuksel: 3 fixed basis spectra, linear blend with RGB weights, ~ 3 MADs per wavelength after a 3-entry basis
  lookup; error slightly worse on saturated colors, fully reciprocal sRGB<->spectrum round-trip; (c) for emission use the "illuminant" version (`RGBIlluminantSpectrum`).
  The practical GPU recommendation: Jakob-Hanika with coefficients stored in the albedo textures at bake (GPU compute pass), Mallett-Yuksel as fallback if the LUT is too much.
- **Costs vs RGB in a GPU tracer:** BVH traversal, hit fetch, texture fetch: identical. Shading math: 4 wavelengths vs 3 channels (~+33% ALU on spectral
  eval), plus spectral uplift (above), plus film conversion (XYZ accumulate 4 samples via CMF LUT = 4 taps of 1D LUT, trivial). Noise: colored noise at low spp (each pixel sample sees only 4 wavelengths);
  the denoiser must handle it. Expect ~10-30% end-to-end slower than RGB at equal spp in typical scenes (**estimate, unverified; benchmark**).
- **Which effects require spectral:** dispersion (glTF `KHR_materials_dispersion`, OpenPBR abbe), exact thin-film iridescence on arbitrary thickness/IOR (RGB fit is
  approximate), fluorescence, accurate metamerism/illuminant matching (D65 vs A lights on same albedo), spectral sky models, true "Planck blackbody" lights, spectral volume coefficients (e.g. atmosphere, water).
  RGB gives indistinguishable results otherwise to within a few percent on typical materials; this is the justification for deferring.
- **Who does what:** pbrt-v4: always spectral, 4 hero wavelengths, RGB->spectrum via Jakob-Hanika LUT built by `rgb2spec_opt`; Mitsuba 3: variants `scalar_spectral`/`cuda_spectral` and `*_rgb`, hero wavelength + uplift (also JH LUT);
  Manuka (Weta): spectral, 3? hero wavelengths per path (they "carry 3 or 4" (unverified)), batch shading sorted by texture/material for coherent memory access.
- **Design for forward compat:** define `type Radiance = vec4<f32>` with `.w` unused in RGB mode; a `Wavelengths` struct carried in path state only when `SPECTRAL` override is on.

### 3.7 Participating media

- **Homogeneous fast path:** closed-form distance sampling t = -ln(1-u)/sigma_t; transmittance exp(-sigma_t t) analytic (also for shadow rays: no majorant needed). For chromatic sigma use
  hero-wavelength/single-channel sampling with ratio weights (Kutz spectral tracking) to avoid color bias. **Adopt now.**
- **Heterogeneous:** delta tracking (stochastic, unbiased, high variance in shadow rays), ratio tracking (smooth transmittance estimator; use for NEE rays), residual tracking (control variate), spectral tracking.
  Majorant grid (coarse macro-cells 8^3-16^3 + DDA, per-cell sigma_max) makes cost ~ proportional to optical thickness not volume extent.
  Null-scattering formulation (Miller 2019): all of the above as a path-integral with null events; lets you do MIS between NEE and phase-sampling in media with unbiased weights: **adopt the formulation for the spec** (single integral, same code path for homogeneous/hetero).
- **NanoVDB on GPU:** WebVDB (`~/Developer/WebVDB/packages/nanovdb-wgsl`, vendored Apache-2.0 `pnanovdb.wgsl` fork) gives read-access traversal; for path tracing, build a *coarser majorant* grid from the leaf/internal-node
  max values (NanoVDB can store min/max per node: tile max = free majorant; use HDDA (hierarchical DDA, Museth) over node hierarchy for majorant stepping). Expect 2-5x slower per density fetch than a dense grid texture due to tree walks; use `ReadAccessor` caching (pnanovdb accessor in registers) and trilinear 8-tap per step is the dominant cost. Fp8/FpN quantization reduces bandwidth (WebVDB has quantize). **Prototype.**
- **Subsurface scattering:** (a) *random-walk SSS* (Cycles, Arnold; Chiang et al. 2016 Disney "Practical and Controllable Subsurface Scattering for Production Path Tracing"): volumetric walk inside mesh using sigma_t/albedo from `subsurface_color*radius` with albedo inversion; physically plausible, needs closest-hit inside medium (re-using the main BVH) and ~tens of steps; cost grows w/ albedo. (b) *Diffusion/BSSRDF probe-ray (Christensen-Burley):* sample a point on surface via probe ray along random axis + radial profile, 1 extra ray vs walk (v2 fast).
  (c) *Normalized diffusion + screen-space blur* (real-time; not path-tracing-coherent). Pick (b) for fast mode (single extra ray + profile eval, planar assumption, lacks multiple-scatter correctness in thin geometry) and (a) for reference.
  **Prototype (b), Reference/Later (a).** Interplay with thin-walled flags in OpenPBR.

### 3.8 Environment and sky

- **HDRI importance sampling:** build a 2D piecewise-constant CDF (marginal over rows + conditional per row), O(log n) binary search *or* O(1) alias table over texels; MIS with BSDF samples. Alias table in WGSL: 2 storage reads per sample, 1 uniform fetch of prob/alias, +1 fetch of the HDR texel and its pdf (store pdf in alpha). Build is a one-time GPU compute (prefix-sum or Vose on CPU for <= 4k x 2k). Best sampling: with 1k x 512 resolution CDF + hierarchical warp (Clarberg 2005 wavelet importance sampling) is lower variance but higher complexity. **Alias table over (downsampled) luminance map: Adopt now.**
  A sun-disk + env split (peaky highlights removed from map and sampled as an explicit directional light) reduces fireflies greatly. **Adopt now.**
- **Sky models as lights:** Hosek-Wilkie (analytic radiance in sRGB per turbidity, 9 params x 3 channels x 10 turbidities x 2 albedos tables = small const); evaluate per ray at miss, plus a sampled sun disk. Importance sample by *baking* the sky to an env map (e.g. 512x256) and building the alias table (one compute pass whenever sun changes); direct eval is only used for miss rays. **Adopt now.**
  Prague (Wilkie 2021) is spectral-resolved (needs spectral data, ~ 100s of MB of coefficient files in original) -> for the spectral renderer later; **Later**. Hillaire 2020 multi-scatter LUT atmosphere for real-time planet-scale skies: ~4 LUTs (transmittance, multi-scatter, sky-view, aerial perspective) ≈ 0.1-0.3 ms on a mid GPU, gives volumetric (aerial) look; good for atmospheric-perspective in path tracer (shade rays through the LUT or treat atmosphere as volume with delta-tracking = expensive). **Prototype** as sky bake source (the sky-view LUT gives an env map).

### 3.9 Numerical robustness

- **Ray offset (Wachter-Binder, RTG ch. 6):** offset the hit position along the geometric normal by a *scale-adaptive* amount: a fixed object-space-relative epsilon for tiny magnitudes (origin ~ 0) and integer-ulp offset
  (reinterpret float bits as int, add +/-N ulps per component) elsewhere: ~ 15 ALU, no tracing, robust from 1e-3..1e6 magnitudes. WGSL has `bitcast<i32>(f32)` and `bitcast<f32>(i32)` so it ports directly. Parameters (origin=1/32, float_scale=1/65536, int_scale=256 in their reference) should be tuned for f32 BVH precision. Use *object-space* position for instance transforms to avoid error blow-up. **Adopt now.** Also keep tmin=0 (not epsilon) and avoid `t_min` hacks that eat thin geometry.
- **Shading-normal fixes:** the interpolated normal can yield wi/wo on the wrong geometric side (light leaks, black terminator artifacts on low-poly spheres). Remedies: (a) Hanika's "Hacking the shadow terminator" (Chiang/Disney: shadow terminator softening with cheap geometry-based factor; Ray Tracing Gems II); (b) Bump shading point along shading normal (Hanika); (c) Mitsuba/pbrt approach: reject samples that go below the geometric hemisphere but accept shading-hemisphere (one-sided correction) and use Veach's shading-normal adjoint factor |wi.ns||wo.ng|/(|wi.ng||wo.ns|) for non-symmetric correction (only needed for light-tracing/BDPT; unnecessary for PT). **(b) + Chiang terminator softening: Adopt now (user-togglable).** (unverified citation details for the TS-smoothing papers).
- **f16 risks:** f16 range 6.5e4 and 11-bit mantissa: unsuitable for ray origin/dir, t values, BVH node bounds used in traversal tests (our `bvh` owns this; recommend f32 in traversal), radiance accumulators, or pdf values (underflow for peaky GGX / sun disk). Safe: texture storage (albedo, roughness, normal maps in rg16/oct), LUT storage (E(mu,alpha)), material parameter packing (unorm8/f16), probe/radiance-cache payloads, denoiser networks. WGSL: `enable f16;` + `shader-f16` feature; packed ops via `pack2x16float`/`unpack2x16float` work without the feature. Don't compute BSDF eval in f16 even where fast on Windows/AMD: GGX D at alpha<0.05 overflows f16 mantissa (relative error > 1e-3 around the peak), and pdf ratios blow up. **Adopt now: f32 compute, f16 storage.**

## 4. For the spec

### v1 (fast) stack
1. Material: packed OpenPBR-subset uber BSDF (diffuse Lambert, GGX spec w/ F82-tint conductor + dielectric from IOR, clearcoat, Charlie sheen, thin+thick transmission with `volume` absorption via Beer; emission; alpha/opacity; normal & bump maps; anisotropy). Params in one 96-128 B record; textures via bindless index with atlas fallback.
2. Sampling: spherical-cap/bounded VNDF, cosine diffuse, Turquin compensation (1 LUT), pdf-proportional lobe selection, lobe pdfs evaluated only for MIS.
3. Integrator: unidirectional PT, 1 NEE ray (alias-table over emissive tris + env + sun), power-heuristic MIS, pbrt-style RR, path-regularization + luminance clamp toggles, max depth ~8-16.
4. Media: global homogeneous fog + per-object homogeneous media; no heterogeneous in v1 (except optional WebVDB box with ratio tracking for demo).
5. Env: HDRI alias-table sampling + analytic sun; Hosek-Wilkie baked to env.
6. Robustness: Wachter-Binder offset, f32 shading, shadow-terminator softening, no f16 in tracing.
7. Utility consumers: `evalLight`/`sampleDirect` + `materialAlbedoRoughness(hit)` entry points only: they never need lobes.
### v2 (full) stack
1. Spectral hero-wavelength mode with JH coefficients baked in textures, dispersion, exact thin-film, spectral sky (Prague), spectral media.
2. Material graph compile (MaterialXJS -> WGSL pattern functions), wavefront material sort by lobe-set class (Manuka style), LTC sheen, Belcour layering optional, rough-transmission compensation LUT.
3. Light BVH / ReSTIR DI+GI integration with documented bias modes; MIS compensation optional.
4. Heterogeneous media via NanoVDB majorant HDDA + null-scattering MIS; probe-ray BSSRDF (fast) and random-walk SSS (reference).
5. Reference-only on Windows/native: BDPT/VCM, SMS, Heitz random-walk multi-scatter, stochastic layered BSDF as ground truth to validate the fast lobes.

### Reference scenes stress matrix
| feature | scene(s) |
|---|---|
| white furnace (energy conservation of every lobe, MS compensation) | procedural sphere in uniform env, rough 0..1 sweep, metal/dielectric/coated/sheen |
| GGX / VNDF sampling variance | sphere array roughness x metalness (pbrt "material ball"), Mitsuba "veach-mis" |
| MIS / light sampling | Veach MIS scene (4 plates x 4 lights), many-light sponza/Bistro with emissive meshes |
| glass / rough transmission / dispersion | Cornell box with glass sphere, Stanford dragon glass, prism dispersion (Mitsuba dispersion demos) |
| clearcoat/sheen/iridescence/anisotropy | glTF Sample Assets: ClearCoatTest, SheenChair, IridescenceLamp, AnisotropyBarnLamp, TransmissionTest, DispersionTest, AttenuationTest, DiffuseTransmissionTeacup (names from Khronos glTF-Sample-Assets; unverified exact set) |
| volumes | WDAS cloud (WebVDB), Disney/ Mitsuba homogeneous "smoke"/"jade" |
| SSS | skin/head, marble Buddha, Mitsuba jade |
| caustics | pool/caustic ring, "caustic glass" pbrt scenes (reference only) |
| sky/env | Hosek-Wilkie sky sweep over sun elevation; HDRI + sun (Poly Haven) |
| robustness | low-poly smooth sphere (terminator), thin geometry (cloth), large-scale (1e4 units), Moana-like scale |
| LUT validation | LUT vs brute-force MC albedo |

## 5. Bench / comparison ideas
- Per-feature microbenchmarks: ns/sample for VNDF variants (Heitz vs spherical-cap vs bounded) and equal-variance comparison (rel. MSE at fixed 64 spp vs 1M-sample reference) on fixed-roughness sweep.
- Material-eval cost: Mrays/s with uber (all branches) vs flag-specialized pipelines at 1, 2, 8 distinct materials per wave; with/without wavefront sort.
- Spectral overhead: same scene RGB vs 4-wavelength hero; report ms/frame and variance-to-time at equal wall time.
- Furnace + chi-square sampler tests as CI (pdf/eval consistency).
- Compare images against pbrt-v4 and Mitsuba 3 (`scalar_rgb` + `scalar_spectral`) at same camera/materials; error metrics: relMSE, FLIP. Compare against mutracer for the same material scenes.
- RR/clamp bias: converge to 1M spp, measure bias in mean luminance by mode.
- Ray-offset: count self-intersection artifacts on a stress mesh vs epsilon baseline.
- Volume: delta vs ratio tracking time-to-equal-variance on the WDAS cloud; NanoVDB vs dense 3D texture majorant.

## 6. Needs Windows / native hardware
- pbrt-v4 and Mitsuba 3 ground-truth renders (CPU is fine, but large; CUDA/OptiX variant of Mitsuba needs NVIDIA).
- Reference-only integrators (BDPT/VCM/SMS) via Mitsuba/pbrt native.
- f16 throughput differences across vendors (AMD/NVIDIA/Intel), subgroup size effects on material divergence, shader compile times on D3D12/DXC for specialization variants.
- Measuring wavefront-sort vs uber on real discrete GPUs.

## 7. Open questions
- Exact deltas OpenPBR 1.0 -> 1.1 (I believe 1.1 clarified coat/fuzz energy handling and added thin-film refinements; verify before spec freeze). Also whether a reference WGSL/GLSL implementation exists within MaterialX main (check MaterialX `libraries/pbrlib/genglsl`).
- Mallett-Yuksel URL and numbers unverified; confirm and decide between it and JH.
- Which bounded-VNDF variant to ship for anisotropic + transmission; need to read the Eto paper's anisotropic section.
- Can WGSL specialize via `override` on per-pipeline basis cheaply enough on D3D12 for ~16 variants? (profile).
- Does bindless in Chrome allow non-uniform texture index in compute without `nonuniform` hazards on all backends? (cross-ref bindless note.)
- Reservoir pdf for MIS with ReSTIR DI: pick unbiased-reference vs biased-realtime split.
- Failed downloads to retry: Wilkie 2014 hero wavelength, Belcour layering/thin-film, Zeltner LTC sheen, Prague 2021, Wachter-Binder, Kaplanyan path regularization (URLs may have moved).
- 2024-2026 literature sweep still to do (e.g. neural BRDF sampling, new many-light methods, OpenPBR viewer in WebGPU, Khronos dispersion/diffuse_transmission status, three.js MaterialX node support).
