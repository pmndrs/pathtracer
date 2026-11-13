# Divergences

Where we deliberately differ from a source we built on (paper, sibling package, competitor, native renderer), and
why, so each one can be revisited later. One row per divergence. Kinds: **platform** (browser/driver/WGSL limit),
**perf** (measured speed choice), **licence**, **simplification** (do it properly later), **bug-workaround**.

| ID | Date | Source | What we do differently | Kind | Revisit when |
|---|---|---|---|---|---|
| V1 | 2026-11-13 | Spec / research note 06 (bindless) | Texture arrays (gpuTex) are the default texture path; the bindless resource table is opt-in | platform | Chrome ships the resource table on macOS and drops the `chromium-experimental-` prefix |
| V2 | 2026-11-13 | NRD (real-time denoiser reference) | No NRD code or ports anywhere in this repo; native NRD output is used only as images | licence | never (clean room) |
| V3 | 2026-11-13 | NVIDIA STBN (spatiotemporal blue noise) | Own blue-noise generator instead of NVIDIA's textures/generator | licence | never (non-commercial licence) |
| V4 | 2026-11-13 | pbrt-v4 `path` integrator | Our private pbrt build adds `"bool rr" false` (`bench/references/pbrt-no-rr.patch`) so references can disable Russian roulette | simplification | pbrt gains an RR switch upstream |
| V5 | 2026-11-13 | bvh GPU runner (`tests/gpu`) | Pages report a `BenchResult` JSON (not pass/fail); page discovery by file name; macOS verdict from `pmset -g therm` (CPU-side proxy, GPU clocks need sudo); stale locks of dead owners are taken over; ticket names carry a per-process sequence | simplification / bug-workaround | Windows runner validation (P0.10) |
| V6 | 2026-11-13 | NVlabs FLIP (via Denoiser's port) | LDR-FLIP skips the sRGB decode (our tonemap output is already display-linear); separable kernels with clamped borders; AgX is the "minimal AgX" approximation | simplification | P0.14 validation against NVIDIA reference images |
| V7 | 2026-11-13 | glTF export for web renderers | Conductors become metallic with F0 from eta/k; dielectrics become transmission + IOR; diffuse gets specular 0; emission is two-sided; environment only in `extras` | platform (glTF can't express these exactly) | competitor adapters set the exact values through their own APIs where possible |
| V8 | 2026-11-13 | Tier 0 furnace spheres | Flat-shaded triangles, not smooth normals: smooth shading lost 1.5% at the silhouette in Mitsuba | simplification | a dedicated shading-normal test scene |
| V9 | 2026-11-13 | pbrt-v4 environment maps | Exporter resamples equirect → square equal-area octahedral; pbrt lookup is nearest-pixel while Mitsuba's equirect is bilinear | platform | differences visible in env-sphere agreement (relMSE 6.4e-4) |
| V10 | 2026-11-13 | Competitor bounce semantics | Mapped to our `maxDepth` by calibration: tgpt `maxBounces` = maxDepth (+1 when emissive meshes are reached by BSDF sampling only); mutracer `bounces` = maxDepth − 1 (capped at 50 events) | simplification | a competitor changes its bounce counting |
| V11 | 2026-11-13 | Competitor lights | tgpt: quad emitters become rect-area lights (D19); one-sided emitters get a black twin so the back face is dark | platform (glTF/three can't express one-sided emitters) | tgpt gains one-sided emissive meshes |
| V12 | 2026-11-13 | Competitors' rough BSDFs | Both use multiple-scattering energy compensation that can't be turned off; the conductor furnace reads ≈1.0 vs our single-scatter references' 0.69, so that cell is excluded from the default matrix (D20) | platform | we add an MS-compensated reference or toggle |
| V13 | 2026-11-13 | mutracer direct lighting | 4 RIS candidates per vertex, always on; tgpt does a 20-byte sample-count readback every 250 ms during `renderSample` | platform (no switch) | upstream adds a switch |
| V14 | 2026-11-13 | Sponza (Khronos glTF-Sample-Assets) | `file` meshes exported un-welded with flat geometric normals; alpha-cutout foliage renders as solid quads; one degenerate triangle dropped | simplification | a textured/alpha Tier 2 scene |
| V15 | 2026-11-13 | Directional light `angularRadius` | pbrt-v4 and Mitsuba ignore it (delta sun); scenes that need exact agreement use 0 | platform | we need soft-sun references |
| V16 | 2026-11-13 | pbrt-v4 patch (extends V4) | Patch also adds `"integer rrdepth"`; `.cache/native/pbrt-v4` rebuilt, `pbrt-no-rr.patch` regenerated | simplification | upstream RR controls |
