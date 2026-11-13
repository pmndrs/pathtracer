# Loose spec (draft 0)

Status: **draft for discussion**, written 2026-11-13 from research pass 1 ([../research/INDEX.md](../research/INDEX.md)).
Nothing here is decided until we talk through phases. Every claim cites the notes file it comes from (`[05]` =
`research/notes/05-gpu-architecture.md`). Items marked **DECIDE** need your call.

## 0. Goals and rules

- **Speed above all.** Be the fastest software ray/path tracer on the web, and measure it rather than assume it.
- **Quality.** A converged reference mode must match pbrt-v4/Mitsuba within the noise floor. Fast modes are biased
  and documented as such.
- **Good DX.** Engine-agnostic core, three.js adapter first, readable generated WGSL, debug views and timings
  built in.
- **GPU-driven.** No readback inside the frame loop. Work counts live on the GPU and drive `dispatchWorkgroupsIndirect`.
- **Bindless-first in design, fallback-first in support.** Today the resource table covers textures only and is not
  on macOS [06].
- **Built with the siblings:** bvh (geometry and walk), downsampler/SPD (Hi-Z, pyramids, reductions), Denoiser,
  upscaler, reflections, gi, gpuTex.

**Non-goals (v1):** hardware RT (proposal only [06]), WebNN in the frame loop [06][07], BDPT/VCM/manifold caustics
[09], path guiding [04], NRD anything (clean room).

## 1. Package shape

Decided 2026-11-13: one GitHub repo and monorepo named **`pathtracer`**, publishing two packages. The split keeps
siblings that only need rays (gi, reflections) from depending on the path tracer. npm availability of both names
under `@pmndrs` is not yet checked.

```
@pmndrs/bvh          (exists, own repo)  build / refit / layout / bvhTraceClosest|Any   "ends at the hit"

repo: pathtracer
├─ packages/raytracer   @pmndrs/raytracer   WGSL prefix rt*
│    core        L0–L2 inline WGSL library + batch dispatcher, rtSurface decode, detectCaps, queue ABI
│    shade       materials (uber BSDF table), lights, env, textures (TextureProvider)
│    sampling    Sampler (OwenSobol | BlueTile | Hash), STBN/FAST bank, light sampling
│    three       adapter subpath (scene sync via bvh/three, TSL nodes)
├─ packages/pathtracer  @pmndrs/pathtracer  WGSL prefix pt*   depends on @pmndrs/raytracer
│    integrator  progressive PBR path tracer (megakernel + wavefront schedulers), cache hooks
│    restir      reservoirs, ReSTIR DI (GI later)
│    reconstruct AOV contract + glue to Denoiser / upscaler (SVGF-class filter is the only filter of our own)
│    three       adapter subpath
└─ bench/               harness, scenes manifest, exporters (glTF -> pbrt / Mitsuba), results (not published)
```

Reserved prefixes: `bvh*` (bvh), `gi*` (gi), `rt*` (raytracer), `pt*` (pathtracer) [01].

## 2. Layered tracing API (the part siblings consume) [08][05][01]

The same layers are available two ways:
- (a) **inline WGSL functions** callable from any compute shader;
- (b) a **batch dispatcher** over a GPU ray buffer with an indirect count.

| Layer | WGSL entry | Returns | Consumers |
|---|---|---|---|
| L0 VIS | `rtVisible(ray, mask)` | bool | shadows, AO, probe visibility. A separate cheaper traversal (any-hit, far-first order) |
| L1 HIT | `rtClosest(ray, mask, cone?)` | `Hit` 16–40 B. `ray.tMin`/`tMax` intervals are mandatory; optional ray cone | RC intervals, SSR continuation, reflections |
| — | `rtSurface(hit, SurfaceMask)` | lazy: position, UV, shading normal, tangent, material params | everyone who needs more than `t` |
| L2 SHADE1 | `rtShade(hit, depth: Flat\|Lambert\|Full, cache?)` | radiance at hit: emission + direct light (1 NEE ray) + cache lookup | DDGI, GI-1.x, reflections hit shading (fixes its "dark bias" [01]) |
| L3 PATH | integrator with hooks: `rayGen`, `onVertex`, `terminate` | radiance + AOVs | full PT, baking, reference renders |

Other parts of the API:
- **Masks:** 8-bit instance/ray masks (bvh) plus per-consumer layer flags.
- **Proxy geometry:** `GeomLod` / proxy hook reserved for later, plus an `ALLOW_APPROX` flag for a future SDF/voxel
  far-field tier [05][08].
- **Budget scheduler** in rays/pixel/frame with per-consumer caps [08]. Planning numbers at 1080p:

  | Consumer | Rays per pixel per frame |
  |---|---|
  | GI | 0.05–0.5 |
  | Reflections | 0.25–1 |

## 3. Shared data contract (adopt, don't invent) [01]

- **Space and units:** right-handed, +Y up, metres. View looks down −Z. Column-major matrices.
- **Rays and hits:** bvh's `Ray{origin,tMin,dir,tMax}` and `Hit{t<0 miss, instance, triangle, bary(v1,v2), frontFace,
  material, normal}`. Check the layout version at start-up and pin an exact bvh commit.
- **AOV / G-buffer contract:** reflections' `FrameTextures` + `FrameConventions` + `FrameCamera`. Standardize on:
  - `viewZ` negative in front;
  - motion as **NDC delta current−previous, +y up, jitter-free**;
  - world-space normals (the Denoiser adapter rotates them to view space for OIDN aux);
  - linear HDR radiance.
- **AOVs from the tracer:**
  - radiance;
  - demodulated diffuse and specular;
  - first-scatter albedo (diffuse and specular);
  - normal, depth/viewZ, motion (plus optional specular virtual-hit motion);
  - roughness;
  - hit distance (diffuse and specular);
  - object id. [07]
- **Jitter:** one owner per pipeline. When the upscaler owns it, Halton(2,3) with `8·ratio²` phases. When the tracer
  owns it, the AA offset comes from the sampler. [01]
- **Accumulation:** f32. f16 only for AOVs, history and storage. f16 is never used for positions, ray origins or
  accumulators [09][01].
- **Material ids:** the bvh 16-bit id indexes our material table. Revisit if real scenes exceed 65534 materials.
- **Binding budget:** every module publishes its storage-buffer and sampled-texture counts. The fallback path fits
  8 storage buffers and 16 sampled textures per stage [01].

## 4. Capabilities and fallbacks [06][05]

`detectCaps(adapter)` returns individual flags. These drive pipeline selection; `tier` is only a label:

| Label | Adds | What it buys |
|---|---|---|
| T0 baseline | core WebGPU, chunked storage pools, texture-array textures, workgroup-atomic queues | runs on Safari, Firefox, iOS |
| T1 | `shader-f16`, `timestamp-query`, immediates, BC/ASTC, raised limits | f16 storage/AOVs, timings, gpuTex arrays |
| T2 | `subgroups` (+ `subgroup-size-control`) | wave-aggregated queue appends, compaction, waterfall shading |
| T3 | `chromium-experimental-sampling-resource-table` (+ `subgroup-matrix` later) | bindless material textures; coop-matrix denoiser |

- Feature names are tried in both forms (`chromium-experimental-*`, then unprefixed) so the code survives the
  rename.
- Each feature has exactly one fallback. Fallback and fast path are checked against each other by a parity test.
- Ship a `caps.html` diagnostics page.

## 5. Execution architecture [05][03][02]

- **Utility path:** inline megakernel (L0–L2 in one kernel).
- **Path tracer:** two schedulers behind one interface, benchmarked against each other.
  - *Megakernel*: tiles, specialization constants, timestamp-driven frame budget.
  - *Wavefront*:
    - SoA path state ≤96 B;
    - 4–6 shader-class queues;
    - GPU-written indirect args;
    - Cycles-style refill when active paths drop below 25%;
    - compaction of main and shadow queues;
    - octant binning and block-wise bucketing (no global payload sort);
    - subgroup waterfall loops.
  - All bounces are recorded in one command buffer.
- **Primary visibility:** optionally from an engine-rasterized visibility buffer (pre-resolved primary hits).
- **Traversal experiments live in bvh:** BVH4 / CWBVH8 against today's BVH2. Gate: ≥1.3× on both 3060L and M5. Stack
  depth comes from the tree as a shader constant.
- **Shader size:** no generated kernel over ~50 KB of WGSL (Windows DXC compile cliff). Pipelines are created
  asynchronously and cached by (kernel, caps hash).
- **Mrays/s targets** (incoherent closest-hit, Sponza-class; extrapolations have ±2× error until measured) [05]:

  | GPU | Today | Target |
  |---|---|---|
  | RTX 3060 Laptop | 104 | 150–220 |
  | Apple M-Pro class | ~100 | 130–220 |
  | High-end desktop | — | 600–1000 |

## 6. Sampling and light transport [04][09]

- **Sampler interface:** `init(pixel, frame, sampleIndex)`, `next1D/next2D`, dimensions consumed in pairs, plus a
  documented dimension map. Implementations:
  - `OwenSobol` (reference; Burley hash);
  - `BlueTile` (STBN/FAST textures for bounces 0–1, R1-shifted each frame);
  - `Hash` (PCG, debug).
- **Lights:**
  - Alias table over emissive triangles, environment and sun, for real time.
  - Light BVH (Conty-Kulla) for reference and many-light scenes.
  - Spherical solid-angle sampling for area lights.
  - Env map via alias table or mip-warp.
- **Integrator v1:** unidirectional path tracing.
  - One NEE ray per bounce, power-heuristic MIS.
  - Russian roulette after bounce 3; max depth 8–16.
  - Wächter-Binder ray offsets.
  - Switchable biased options: path regularization, luminance clamp.
- **Reuse:**
  - ReSTIR DI in the real-time mode.
  - ReSTIR GI as a prototype.
  - ReSTIR PT later (the 2026 "Enhanced" version is 2–3× faster, still heavy).
  - The reference mode disables all reuse and blue noise and must match pbrt-v4 at equal spp.
- **Materials v1:** packed OpenPBR-subset uber BSDF, 96–128 B per material.
  - Lobes: Lambert/EON diffuse, GGX with F82 tint, dielectric, clearcoat, Charlie sheen, thin and thick transmission
    with Beer absorption, emission, anisotropy, alpha.
  - Sampling: spherical-cap or bounded VNDF; Turquin multiple-scattering compensation via one LUT.
  - MaterialX is baked to these parameters in v1; v2 adds generated WGSL pattern functions plus wavefront sorting
    by material class.
- **Spectral (v2):** carry `Radiance` as vec4 from day one. Hero wavelength (4λ) with Jakob-Hanika uplifting baked
  into textures. Cost estimate +10–30% — **to be measured**.
- **Volumes:** v1 is homogeneous only. v2 adds null-scattering/ratio tracking over WebVDB's NanoVDB majorant grid.

## 7. Reconstruction (Denoiser and upscaler by default) [07][01]

Three modes, with one "temporal strictness" control across them:

| Mode | Pipeline |
|---|---|
| `reference` | accumulate; OIDN on converged frames |
| `preview` | low-res, 1 spp, OIDN-small, upscale. This is the mutracer / three-gpu-pathtracer look; ours adds a variance-guided temporal pre-pass |
| `realtime` | demod → SVGF-class temporal + spatial filter → `@pmndrs/upscaler` (reactive mask from history length) → tonemap |

- Everything runs on the host's `GPUDevice` with zero-copy texture hand-off.
- An OIDN 3 temporal input slot is reserved, not hard-coded (OIDN 3 is not released).
- Neural beyond OIDN is Prototype at most:
  - NRC: needs coop-matrix;
  - NTC: transcode to BCn at load via gpuTex.

**What makes us faster than the competitors** [02]: nobody on the web ships ReSTIR or real motion-vector temporal
reuse. Both already do low-res preview + per-frame denoise in motion. Our advantage is a GPU-driven queue, a wide BVH,
ReSTIR DI and temporal reuse, plus a better denoiser (ours is ~30% faster than mutracer's OIDN).

## 8. Caches live in `@pmndrs/gi`; this repo provides hooks [08]

Decided 2026-11-13. The world-space hash radiance cache and the probe atlas belong to `@pmndrs/gi`, which already has
a world hash cache on its radiance-cache branches (`techniques/radiance-cache/world-cache`, from the Capsaicin GI-1.2
port). gi depends on this repo for tracing, so putting the caches here would make the two depend on each other.

- **This repo provides:**
  - an optional cache-lookup argument on L2 `rtShade(…, cache?)`, typed by an interface, not by gi's types;
  - `onVertex` / `terminate` hooks on the L3 path integrator, so a cache from gi can end paths early (preview mode,
    future NRC-like termination) without this repo depending on gi.
- **gi owns:** `HashCache` and `ProbeAtlas`, their memory layouts and their update/filter policy. Notes for that
  work from research note 08:
  - 32-bit atomics only (64-bit status unverified);
  - eviction, not resolve and compact;
  - 2^20–2^21 cells by default;
  - compact f16 YCoCg-SH2 payload.
- **Later:** a generic GPU hash table could move into the core if something other than GI needs one (for example
  light-importance statistics or EARS-style variance estimates).
- **Open niche for gi:** 3D Split Radiance Cascades over a triangle BVH in WebGPU. No implementation exists.

## 9. DX

- Same lifecycle as the siblings: `configure → await prepare → encode(encoder) → resetHistory → dispose`. The device
  is always supplied by the host.
- `encodePass()` variants, so a caller can record into its own pass.
- Live settings objects, generated settings docs.
- Debug views:
  - AOVs;
  - heatmaps (traversal steps, queue occupancy);
  - furnace mode;
  - per-pass GPU timings.
- Deterministic mode with a fixed seed. `readbackAccumulation()` for tests.

## 10. Bench and test plan [10][03][02]

- **One `Renderer` adapter interface:** `load(scene, camera)`, `renderTo(spp|ms)` returning linear RGBA f32 plus
  timings. Implemented for:
  - ours;
  - mutracer;
  - three-gpu-pathtracer (WebGPU, both schedulers);
  - Rayzee;
  - strahl;
  - pbrt-v4, Mitsuba 3 and Cycles (offline, via exporters).
- **Scenes:**

  | Tier | Content | Role |
  |---|---|---|
  | 0 | furnace variants, Cornell, quad light, env sphere, volume cube | merge gate |
  | 1 | glTF-Sample-Assets `KHR_materials_*` | publishes a feature-coverage matrix |
  | 2 | Bitterli scenes, Sponza, Bistro | quality and speed |
  | 3 | Emerald Square, Zero Day | speed under load |
  | sidecar | Moana | scale |

  Scene bytes stay out of git; the manifest records URL, SHA-256 and license.
- **Reference pipeline:**
  - Neutral scene description → our exporters to `.pbrt` and Mitsuba XML.
  - Bench material subset with exact equivalents in every renderer.
  - Pinned handedness, fov axis and box pixel filter.
  - References are rendered once at high spp and cached. The pbrt-vs-Mitsuba delta is our noise floor.
- **Metrics:**
  - relMSE (ε = 1e-2), FLIP / HDR-FLIP, signed bias map;
  - time-to-equal-quality, efficiency = 1/(variance×time);
  - for real time: boil, lag and still-σ (Denoiser scorecard);
  - Mrays/s as a diagnostic only.
- **Run discipline:**
  - ABBA ordering, median of 3;
  - GPU lock; thermal VALID/INVALID verdict;
  - timestamp calibration (Chrome quantizes to 100 µs without the developer flag);
  - no readback inside timed regions;
  - numbers are never compared across machines.
- **Reuse:** webgpu-ddgi Chrome launcher and CSV schema, bvh `compare/` adapters, Denoiser `scorecard` and `eval`,
  ddgi-research linear compare, gitestbench scenes.

## 11. Candidate phases (to discuss, not decided)

- **P0 — Foundations.**
  - Repo scaffolding.
  - `detectCaps`.
  - Shared contract types.
  - Bench harness skeleton plus the Tier 0 furnace/Cornell suite.
  - pbrt/Mitsuba exporters with the first cached references.
  - Run the competitors behind the adapter interface.
- **P1 — Trace core.**
  - L0–L2 inline library over bvh.
  - `rtSurface` attribute decode.
  - Batch dispatcher and queue ABI.
  - Move one sibling onto it (reflections hit shading, or ddgi's frozen BVH copy) to prove the seam.
  - Run the bvh wide-node experiment in parallel.
- **P2 — Reference path tracer.**
  - Megakernel, Owen-Sobol, uber BSDF, NEE+MIS, env/area lights.
  - Converges to pbrt-v4 on Tier 0–1.
  - OIDN final.
- **P3 — Speed.**
  - Wavefront scheduler vs megakernel.
  - Wide BVH adoption.
  - Coherence tricks.
  - Visibility-buffer primaries.
  - First published web-vs-web and web-vs-native numbers.
- **P4 — Real time.**
  - BlueTile sampler, demodulated AOVs, SVGF-class filter and upscaler chain.
  - ReSTIR DI.
  - Cache hooks on the path integrator, exercised with gi's hash cache.
  - Preview mode.
  - GI tools (gi, reflections) switch to the core.
- **P5 — Breadth.**
  - Bindless Tier 3 path (Windows).
  - ReSTIR GI.
  - Spectral and volumes.
  - MaterialX → WGSL.
  - Support for gi's 3D Split Radiance Cascades (the technique itself lives in gi).
  - Coop-matrix denoiser when available.

## 12. Open decisions for you

1. **Where shading lives.** Should L2 "one-bounce shade" live here and replace reflections' and ddgi's own hit
   shading, so the core owns the shared material/light scene copy? The research says yes; it is a cross-repo
   ownership change.
2. **First consumer.** Reflections (cleanest seam, `TraceProvider`) or ddgi (frozen BVH copy, most duplication)?
3. **Platform priority.** Mac-first development while bindless is unavailable on macOS? Or bring up the Windows
   machine early for bindless, NVIDIA/AMD timing and native references?
4. **Spectral.** v2 as proposed, or earlier? (It matters for dispersion, iridescence and fluorescence.)
5. **mutracer comparisons.** Public, or internal only?
6. **Verification pass.** Run a second, narrower pass on the backlog in INDEX §4 before we lock numbers?

Decided so far (2026-11-13): package split and naming (§1); caches live in `@pmndrs/gi` (§8); wide-BVH work lands in
`@pmndrs/bvh`; native NRD runs only through the Denoiser's Windows harness.
