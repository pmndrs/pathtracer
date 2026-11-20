# 02 - Web ray/path tracers (WebGPU first, notable WebGL)

Date: 2026-11-13. Scope: every web path/ray tracer I could find (GitHub topic/API searches, npm, web search),
with deep reads of mutracer, three-gpu-pathtracer and three-mesh-bvh. All code citations are paths inside
`research/sources/web/<name>` at the commit in `research/sources/MANIFEST.md`. Nothing was built or run;
every performance number below is *claimed by the project* unless marked "derived" (my arithmetic).

---

## 1. TL;DR

1. **The field moved in 2026.** Three serious open-source WebGPU path tracers exist that are not toys:
   **mutracer** (megakernel, MIT, M4 Max numbers), **three-gpu-pathtracer's new `WebGPUPathTracer`** (v0.0.25+, Sep 2026;
   *wavefront by default*, megakernel optional) and **Rayzee** (pure wavefront, 4-wide quantised BVH, OIDN + NRD, a
   real regression bench). The brief's "three-gpu-pathtracer = legacy baggage" is half stale: the WebGPU rewrite is
   active and good; the baggage is the deprecated WebGL tree (10.8k lines) plus the hard three.js/TSL coupling.
2. **mutracer has NO temporal reprojection or history clamping.** Its "blurry, not grainy" motion look is:
   (a) the interactive *preview* Render is traced at 0.125-0.5x resolution, 2 bounces, 1 wavelength, <=2 light
   candidates, **1 spp per camera pose** (history is discarded on every camera change); (b) that 1-spp frame is
   **OIDN-"small" denoised on every sample** (`schedule: 'every-sample'`); (c) then EASU-upscaled (optionally with
   display-resolution depth/albedo detail AOVs). Blur = denoiser + upscaler, not accumulation. Details in section 3.
3. **No web renderer does real temporal reuse of path-traced signal.** Only small demos (webgpu-doom-pathtracer:
   reprojection + YCoCg clamp + a-trous), Hush (SVGF-style + tiny net), Rayzee (OIDN with a motion-history input
   "shimmers half as much"; NRD ReBLUR port), Web-RTRT (2024 ReSTIR DI, abandoned). **ReSTIR/DDGI-class reuse in a
   engine-agnostic WebGPU library is open ground** - nobody in the web has shipped it.
4. **Hardware RT in the browser exists only as out-of-tree forks:** willwaltz/webgpu-ray-tracing (Dawn/Chromium
   patches, Vulkan `VK_KHR_ray_query`, RTX 4090: 1.2-1.4x over software in its own use case) and ocularmagic/web-rayquery
   (Firefox/wgpu, DX12/Vulkan/Metal ray queries). gpuweb#535 is still "Milestone 4+, no design" (last comment 2026-11-11,
   split into ray-query / acceleration-structure issues). Plan for software RT; keep a hardware-RT backend seam.
5. **Wavefront vs megakernel is not settled on the web:** three-gpu-pathtracer and Rayzee chose wavefront;
   mutracer megakernel; a 67.js spike measured wavefront **2x slower** than megakernel on a traversal-bound room.
   Both must be benchmarkable in our core (our BVH scope ends at the hit, so this is a scheduler decision above it).
6. **BVH landscape is unimpressive except Rayzee/mutracer:** three-mesh-bvh = BVH2, plain stack (depth 60, throws on
   overrun), JS SAH build; mutracer = TinyBVH (BVH2/BVH4/GPU-BVH/CWBVH, SBVH) via WASM or JS port + TLAS/BLAS +
   private-memory stacks; Rayzee = Rust-WASM SAH + reinsertion, 4-wide quantised nodes, folded leaves, stack 40-128.
   Our `bvh` package (H-PLOC GPU build, refit, tinybvh WASM) is already ahead of all of them on build side.
7. **Bindless is already used on the web, experimentally:** mutracer feature-detects Chrome's
   `sampling-resource-table` / `chromium-experimental-sampling-resource-table` (`src/renderer/wgpuBindless.ts`) with
   atlas fallback - exactly the "bindless first, fall back" stance we want. Everyone else uses atlases/arrays.
8. **Worth benchmarking against:** mutracer, three-gpu-pathtracer (WebGPU, both backends), Rayzee. Strahl as the
   peer-reviewed baseline (2024 paper, OpenPBR). Others are reference-only.

---

## 2. Sources

| name | link | year | license | local path | one-line why |
|---|---|---|---|---|---|
| mutracer | github.com/pissang/mutracer | 2026 (npm @meimu/mutracer 0.x) | MIT (some MIT/Apache parts adapted, see `LICENSES/`) | sources/web/mutracer | near-competitor; OpenPBR, TinyBVH, OIDN, JNLM, caustics; WebGL2 fallback + native Rust SDK |
| three-gpu-pathtracer | github.com/gkjohnson/three-gpu-pathtracer | 2021-2026 (v0.0.28, 2026-11-02) | MIT | sources/web/three-gpu-pathtracer | WebGPU wavefront+megakernel PT (`src/webgpu/`), OIDN, FSR, SSS, fog volumes |
| three-mesh-bvh | github.com/gkjohnson/three-mesh-bvh | 2018-2026 (v0.9.18 + unreleased) | MIT | sources/web/three-mesh-bvh | `BVHComputeData` WebGPU TLAS+BLAS (`src/webgpu/`) |
| Rayzee | github.com/atul-mourya/rayzee-renderer (npm `rayzee` 7.x) | 2025-2026 | MIT | sources/web/rayzee-renderer | pure wavefront TSL PT, 4-wide BVH, OIDN/NRD, bidirectional, Moana 111M tris |
| strahl | github.com/StuckiSimon/strahl | 2024 (last commit 2025-12) | MIT | sources/web/strahl | ACM Web3D'24 short paper + thesis; OpenPBR; uses three-mesh-bvh builder |
| Stucki & Ackermann paper | arxiv.org/abs/2407.19977 | 2024 | arXiv | papers/web-renderers/stucki2024-...pdf | only peer-reviewed web WebGPU PT measurements (M1 Max, RTX 3080) |
| oidn-webgpu | github.com/pissang/oidn-webgpu | 2024-2026 | MIT | sources/web/oidn-webgpu | OIDN WGSL kernels, browser + native wgpu; mutracer's denoiser (ours claims ~30% faster) |
| Hush | github.com/kanaworksai-hue/hush | 2026 | MIT | sources/web/hush | 1-spp denoiser: SVGF temporal + tiny net predicting filter params + TAA |
| ShadyTracer | github.com/lampmaker/ShadyTracer | 2026 | MIT | sources/web/ShadyTracer | wavefront SDF/analytic tracer, photon-map caustics; `docs/PERFORMANCE.md` has careful ms tables |
| rindr | github.com/schmarante/rindr | 2026 | **PolyForm Noncommercial** | sources/web/rindr | CAD/anything viewer on three-gpu-pathtracer+OIDN. READ ONLY, do not port |
| ThreeRT | github.com/meshy-dev/ThreeRT | 2026 | MIT | sources/web/ThreeRT | Rust+wgpu HW ray-query PT, native Mac/iOS clients (NOT in-browser), a-trous+SVGF-style denoise, UE5 reference harness |
| webgpu-ray-tracing (Chromium RT) | github.com/willwaltz/webgpu-ray-tracing | 2026 | BSD-3 | sources/web/webgpu-ray-tracing | Dawn+Chromium patches: acceleration structures + WGSL ray_query (Vulkan only) |
| web-rayquery | github.com/ocularmagic/web-rayquery | 2026-10 | Apache-2.0 | sources/web/web-rayquery | Firefox/wgpu ray queries behind a pref; DX12 tested; RT pipelines, bindless, mesh shaders (phase 5) |
| Web-RTRT | github.com/C-none/Web-RTRT | 2024-06 (stale) | MIT | sources/web/Web-RTRT | WebGPU ReSTIR DI + spatial reuse + denoise + SR; V-buffer; BVH in worker |
| tinypoly gpu-renderer | github.com/tinypoly/gpu-renderer | 2026 | MIT | sources/web/gpu-renderer | three.js scene PT, MIS, naga-translated custom GLSL materials |
| webgpu-doom-pathtracer | github.com/JamesRandall/webgpu-doom-pathtracer | 2026-03 | MIT | sources/web/webgpu-doom-pathtracer | 2.5k-line RT with reprojection+YCoCg clamp+a-trous; blog post documents the trade-offs |
| solstice | github.com/SamG-Coder/solstice | 2026 | MIT | sources/web/solstice | Rust/WebGPU Nanite+Lumen-style (not a path tracer); reference for GI-in-browser |
| relight | github.com/harri665/relight | 2026 | none found | sources/web/relight | EGSR 2026 neural render proxies, WebGPU inference |
| lumina-wgpu, moonlight, rayfinder, wtrace, gnikoloff/webgpu-raytracer, SSWorld | various | 2023-2026 | MIT/Apache/none | sources/web/* | small/educational; skim only |
| Erich Loftis PT renderers | github.com/erichlof/THREE.js-PathTracing-Renderer, ...Babylon.js-PathTracing-Renderer | 2015-2026 (three: Oct 2026; Babylon: 2023) | CC0 | sources/web/ | WebGL2 real-time analytic/BVH demos, history; no WebGPU |
| dtysky/webgpu-renderer, ferminLR, re-ovo (GPL-3), LGL-Tracer, Nelarius, others | GitHub | 2021-2025 | mixed | (links only) | not cloned: stale, small or GPL |

Links only (not cloned): bevy_solari (jms55.github.io posts 2025-09-20, 2025-12-27, 2026-04-12: ReSTIR DI/GI + world
cache + DLSS-RR, inline ray queries on wgpu; native, not web); gpuweb#535; Use.GPU (has a demo PT, no dedicated work);
Babylon.js 8 (IBL shadows, area lights; no path tracer); `awesome` lists.

**Searched, nothing found:** PlayCanvas, Needle, Orillusion, Galacean (no path tracing features located);
Spline/Vectary/Sketchfab/Polycam/Shapr3D/model-viewer (nothing located; Sketchfab and Polycam are not documented to
path-trace in-browser); Mitsuba/Dr.Jit in the browser (none; Mitsuba 3.9 added Metal, no WebGPU); Dawn/Chrome-team
demos (none for RT); Chrome ray-tracing origin trial (none). Commercial precedents: BRIO Ultra Render (2020), Lighttracer
(2021, native GPU PT behind a browser) - not investigated further. **Mark: these negatives are from search only, not exhaustive.**

---

## 3. mutracer, deeply

Repo: MIT, tip commit 2026-11-07 (06088ce); ~32k lines TS in `src/`, WGSL embedded as TS template strings under
`src/renderer/wgsl/`. Single-shallow clone, so I cannot date the project start. npm `@meimu/mutracer` 0.x.

### 3.1 Architecture
- **API split** (AGENTS.md "Layout"): `Engine` (device + committed scene/camera + shared resources) / `Render` (one
  independent accumulation: samples, denoiser, fallback) / `Display` (canvas, tone map, bloom; changing display
  settings does *not* restart accumulation) / `RenderLoop` (policy: preview vs final). `Render.ts`, `RenderLoop.ts`.
- **Megakernel.** One compute entry `rayTrace.compute.ts:1087` (`@workgroup_size(8,8,1)`, `WORKGROUP_SIZE = 8`,
  `src/renderer/wgsl/rayTrace.compute.ts:32`); a pixel's whole path (camera ray, closest-hit, NEE, BSDF sample, loop) runs
  in one thread. Core pieces: `chunks/rayTraceCore.wgsl.ts`, `chunks/intersect.wgsl.ts` (1551 lines),
  `chunks/openPbrBsdf.wgsl.ts` (1739), `chunks/sampleLight.wgsl.ts`, `chunks/sampleMaterial.wgsl.ts`. No wavefront,
  no sorting, no persistent threads. Image is traced in **tiles** (`renderer/TileRender.ts`) to bound frame time;
  `GPUTraceQueue.ts` tracks estimated GPU ms in flight via `onSubmittedWorkDone` to batch several samples per
  `refine()` within `timeBudgetMs` ("sample batching", docs/progressive-rendering.md).
- **Scene representation:** two-level BVH. Static BLASes built on CPU in Workers (`bvh.worker.ts`,
  `bvhWorkerClient.ts`), TLAS and dynamic BLASes always built/refit by the built-in JS builder (`binnedSahBvh.ts`,
  `bvhAccel.ts`); moving instances only refit the TLAS (docs/updates.md). Builders: `engine.bvhBuilder` `'builtin'`
  (JS port of TinyBVH binned SAH, float64) or `'tinybvh'` (WASM, `src/tinybvh/TinyBvhBuilder.ts`); layouts
  `engine.bvhLayout`: `bvh2` (default), `bvh4`, `bvh-gpu`, **`cwbvh`** (8-wide compressed, `intersectCwbvhInterior`,
  `intersect.wgsl.ts:129`), optional SBVH spatial splits. Workers default `hardwareConcurrency-1` clamped 1-4 (docs/bvh.md).
- **Traversal:** stack-based (not stackless). Private-memory stacks `tlasStack`, `blasStack` (+`cwbvhStack` as
  `vec2u`) `intersect.wgsl.ts:873-877`; sizes `TLAS_STACK_SIZE`/`BLAS_STACK_SIZE` are shader constants recomputed from
  actual tree depth (`WGPURayTracePass.ts:1712, 2319`, BLAS min 32). A comment at ~`intersect.wgsl.ts:351` records a
  D3D12 non-terminating-shader bug with a dynamic stack-top loop (a Windows-only hazard worth remembering). Triangle
  test is Woop-style watertight (shear, `intersectTriangle` :1372). Alpha/displacement (PDM "ray-marched height-field")
  hooks live in the same traversal.
- **Materials:** OpenPBR 1.1 (F82 metals, anisotropic GGX, coat, fuzz, thin film, dispersive transmission, SSS, volume),
  energy LUT (`openPbrEnergyLut.ts`, 1000 lines), packed `MaterialTable`/`materialLayoutSchema`. Spectral: only
  `dispersionWavelengths` (1 or 3 wavelengths per path), not full spectral.
- **Lights/NEE:** rect, sphere, spot (+IES), directional, HDR env with importance-sampled distribution
  (`WGPUEnvMapDistribution.ts`), MIS. `directLightCandidates` = number of resampled NEE candidates per vertex via
  reservoir-style selection with a stratified selector independent of the path sample stream (`rayTrace.compute.ts:205-215`,
  `sampleLight.wgsl.ts:394+`); preview caps at 2. (RIS over lights, not ReSTIR.)
- **Textures:** two modes. Default **atlas/texture-array** (`textureAtlas.ts`, `DEFAULT_MAX_TEXTURE_ARRAY_BYTES`
  384 MB). **Bindless** when Chrome exposes `sampling-resource-table` (or the chromium-experimental name):
  `wgpuBindless.ts` (`createResourceTable`, `setResourceTable`, 65536 slots, needs >=9 storage buffers/stage, probes
  the device and falls back with a reason string). Device features requested: timestamp-query, shader-f16,
  subgroups, bgra8unorm-storage (`deviceRequirements.ts:36`).
- **Sampling** (docs/sampling.md, `chunks/sobol.wgsl.ts`): default `sobol` = Burley 2020 hash-based Owen-scrambled
  Sobol, scrambled per pixel and domain; `zsobol` = Morton-ordered Sobol with blue-noise error (pbrt-v4 ZSobol digit
  walk adapted) for low-spp-without-denoiser; `random` = PCG. ZSobol pattern sized from `maxSamples` (pow2, cap 65536).
  Offline sequences deterministic (`sampleSequenceOffset`). Variance-driven adaptive sampling only in offline
  `converge()` (`OfflineAdaptiveSampling.ts`, `offlineAdaptive.compute.ts`, per-tile).
- **Extras:** light-tracing + photon-mapping caustics (`WGPUCausticLightPass`, `WGPUPhotonMap`), shadow catcher,
  procedural sky baker, bloom, WebGL2 backend with GLSL twin (`renderer/webgl*`, `glsl/`), native Rust/wgpu SDK + egui viewer (`native/`).

### 3.2 Denoise / OIDN path
- Two built-in filters: **JNLM** (joint non-local means, 11x11 search / 3x3 patch / 2 iterations default, WebGPU+WebGL2,
  `jnlm.compute.ts`) and **OIDN** via `oidn-webgpu` (`oidn.ts` entry; models `small` 0.6 MB, `standard` 1.8 MB
  `rt_hdr_alb_nrm`, `cleanAuxLarge` 22 MB with albedo/normal prefilters; tiles <=512; precision/engine `auto`; weights
  Apache-2.0). Inputs: composed *display-linear* (post-tonemap) image + albedo + normal accumulated by the tracer
  (docs/denoising.md: "built-in filters denoise the composed display-linear image", docs/todo.md notes the HDR-stage
  hook issue). Custom denoisers get linear HDR + albedo + normal (`DenoiseInput`).
- **Schedules** (`schedule`): `converged` (final sample only), `progressive` (after sample 1, then each time count
  >= 2x previous and >=4 and >=500 ms elapsed, and the final; at least every 10 s), `every-sample`.
- Our Denoiser being ~30% faster than this path is the brief's claim; I did not measure.

### 3.3 What produces the blurry-not-grainy look (verified in code/docs)
Chain, in viewer defaults (`viewer/src/store.ts:134-168`: `previewEnabled: true, denoise: true,
denoiseModel: 'standard', denoiseWhileConverging: true, displayUpscale: true`):
1. **Two Renders, two accumulation buffers.** `RenderLoop` makes a `preview` Render and a `final` Render with
   `final.fallback = preview` (`RenderLoop.ts:63-69`; `Render.ts:275-282`). Selection default
   `changing ? 'preview' : 'final'` (`RenderLoop.ts:32`). `idleMs` 150 ms of no change before final starts
   (`RenderLoop.ts:61`; docs/progressive-rendering.md).
2. **Preview settings** (`RenderLoop.ts:64-67`): `resolutionScale .5`, `adaptiveResolution {targetFrameMs 80, minScale .125}`,
   `timeBudgetMs 80`, `bounces 2`, `bevel false`, `dispersionWavelengths 1`, `directLightCandidates 2`. So during
   motion you see an 80 ms-budget frame at as low as 1/8 linear resolution, 2 bounces (GI mostly gone => flat, soft).
3. **Reset policy = hard reset, no history carry-over.** "Changes restart each render at its next refine"
   (docs/updates.md:7); camera commit via `engine.update()` is the only edit point; `restart()` discards accumulation
   (docs/api.md:172). The preview has ~1 spp each pose. There is **no reprojection, no motion vectors, no history
   clamp, no temporal filter** anywhere in `src/` (grep for reproject/temporal/motion-vector hits only offline variance
   history in `OfflineAdaptiveSampling.ts` and AOV bookkeeping in `aov.ts`/`WGPUAov.ts`, which keep per-render
   *identity* history for exports, `aov.ts:54`).
4. **Preview is denoised every sample:** viewer builds the preview denoiser as OIDN `model:'small', schedule:'every-sample'`
   (`viewer/src/engine/denoiserSettings.ts:10-17`; applied in `ViewerEngine.ts:795-799`). OIDN on 1 spp gives the
   smooth, blotchy "painted" look - grain is gone, detail is gone.
5. **Upscale hides the low resolution:** `configureUpscale` (`ViewerEngine.ts:777-788`) adds `createEasuPass` (FSR1 EASU
   on display-encoded RGB; ~0.5 ms; "EASU alone cannot recover missing texture detail", docs/post-processing.md:61).
   For a reduced-res *final*, extra display-resolution AOVs (depth, diffuse/specular reflectance) drive a detail pass
   (~2.6 ms @6 MP on M4 Max); preview uses EASU only.
6. **Handoff** (`RenderLoop.ts:139-212`): the last presented image stays visible until the next filtered preview (with
   `coalesceSceneUpdates`) or until final completes a sample; with progressive denoise the final's first image is
   an OIDN-filtered 1-spp estimate, then refines. This produces blur-then-sharpen rather than grain-then-clear.
7. **Frame pacing**: `Render.ts:651` kicks the fallback preview if the full-res first sample can't finish within
   `timeBudgetMs`; `adaptiveResolution` drops scale when slow, recovers gradually.
- **Not verified:** whether per-pixel sample points are identical every preview frame (Owen-scrambled Sobol with a fixed
  seed + sample index 0 would make the 1-spp noise pattern frame-coherent, which helps denoiser stability). Plausible
  from `sobol.wgsl.ts`, not confirmed.
- **Takeaway:** the look is *cheap*: it costs one small-resolution 1-spp trace + one small OIDN + one EASU per frame,
  and buys "no grain". It trades correctness (2 bounces, wrong GI) for calm. We can match it with our Denoiser +
  upscaler, and exceed it by adding a real temporal stage (reprojection + variance-guided history, SVGF/ASVGF-style or
  OIDN-with-motion-history as Rayzee does) so detail and GI survive motion.

### 3.4 Performance claims and how measured
Bugatti Chiron, 4.6M rendered tris (2.5M unique), 35 materials; BVH 1 thread 1.7 s built-in / 1.0 s WASM; 4 workers
0.50 s / 0.28 s; **1920x1080 64 spp = 12.1 s trace, 12.4 s with OIDN**; Apple M4 Max, Chrome WebGPU, viewer defaults
(6 bounces); BVH timings exclude worker startup (README "Performance"). Derived: 2.07M px x 64 = 133M paths / 12.1 s
~= **11 Mpaths/s** at <=6 bounces (not comparable across scenes). Docs also: 100k-instance `engine.update()` 10.6 ms ->
0.0025 ms unchanged; WebGPU target ~110 B/pixel (docs/todo.md); display detail pass 2.6 ms @6MP. No competitor
comparisons published.

### 3.5 Maintenance / license / caveats
MIT; very active (days-old commits); 0.x API "may change". AGENTS.md shows heavy agent-assisted development.
Parts adapted from MIT/Apache code (ray-tracing-renderer chunks in `intersect.wgsl.ts` header, GLSL-PathTracer,
pbrt-v4 ZSobol, ZetaRay, FidelityFX FSR) - attribution in `LICENSES/`. Per brief: **do not copy; cite only**.

---

## 4. three-gpu-pathtracer + three-mesh-bvh (current upstream)

### 4.1 What exists now
- `three-gpu-pathtracer` **v0.0.28 (2026-11-02)**; CHANGELOG: 0.0.25 (2026-09-28) added `WebGPUPathTracer`
  (`three-gpu-pathtracer/webgpu`), requires three r185 then r187 ("binds every storage buffer as an array"), and
  **deprecated `WebGLPathTracer`** ("will be removed"). README warning: WebGPU PT is unstable and needs a locked
  three-mesh-bvh. `three-mesh-bvh` 0.9.18 + unreleased `BVHComputeData` changes, API "unstable".
- **Two backends in one class** (`src/webgpu/WebGPUPathTracer.js:8-9, 585-596, 652`): `MegaKernelPathTracer.js` (286 L)
  + `compute/PathTracerMegaKernel.js` (700 L) and **`WaveFrontPathTracer.js`** (713 L, default). A setter switches
  `value ? Mega : Wave` and rebuilds kernels.
- **Wavefront design** (`compute/wavefront/*.js`, `structs.js`): persistent **path-slot pool** (SoA/struct-of-slots,
  `rayDataStruct` incl. throughput, origin, dir, light sample, medium state, packed 8-bit aux albedo/normal), append-only
  ray / shadow-ray / medium queues, **GPU-side queue counters converted to indirect dispatch** by
  `QueueLengthToDispatchKernel`, kernels at workgroup 64: CameraRay, Logic, Material, TraceRay, TraceShadowRay,
  Subsurface, QueueMediumWalks, CameraMedium, ResetSlots. Pixels wait in a `pixelQueue` for a free slot (continuous
  refill = persistent-thread style without real persistent threads), pool size = `frameBudget`
  (default 250000 paths/frame, auto-tuned to `targetFrameTime` via timestamp queries, `adaptiveFrameBudget`).
  `MAX_BUFFER_SIZE` 128 MB cap per buffer (comment: "TODO increase per platform").
- **BVH** (`three-mesh-bvh/src/webgpu/`): `BVHComputeData` packs objects into **TLAS + per-object (or per-cluster)
  BLAS** storage buffers; `ClusteredBVH` splits skinned/instanced/batched meshes into <=64-primitive clusters (node
  index packing with bit limits, throws when exceeded). **Binary BVH2**, node = AABB + `splitAxisOrTriangleCount` +
  `rightChildOrTriangleOffset` (left child = idx+1, leaf flag in high 16 bits); **stack traversal** with
  `BVH_STACK_DEPTH = 60u` in private memory, nearer-child-first via split-axis sign (`bvh_ray_functions.wgsl.js:95-175`
  deprecated functions; current `shapecastFns/getRaycastFirstHitFn.js`); `BVHComputeData` *throws* if tree depth exceeds
  stack depth (`BVHComputeData.js:378-384`). Build: JS SAH (8 bins/axis, exact sweep for small nodes, 0.9.17), nodes
  written directly into the packed buffer, **worker/parallel builds** (`GenerateMeshBVHWorker`, `ParallelMeshBVHWorker`,
  `setBVHWorker`/`setSceneAsync`). Vertex attributes in one interleaved buffer with packed types (`oct16` normals,
  `2x16float` UVs, `4x16float` colors). No wide BVH, no compression, no hardware-RT path, no refit-only fast path other
  than TLAS refit. `raycastAnyHit` for shadow rays (0.9.18).
- **Shading** (`webgpu/shadingModels/`, `nodes/*.wgsl.js`): glTF/PBR with EON diffuse (`eon.wgsl.js`), GGX, sheen, SSS via
  volumes (KHR_materials_scatter proposal), medium stack (nested volumes), fog volume material, dispersion, alpha test.
  Lights: lights chosen by estimated contribution (0.0.27), spot/point/directional only with MIS, rect/shaped area
  lights, env map importance sampled (`EquirectHdrInfoNode`).
- **Sampling:** Sobol is now default (0.0.28: "instead of Blue Dither"); PCG and blue-dither available
  (`nodes/rand/*.wgsl.js`). `stableNoise` option (same random sequence after reset).
- **Textures:** `AtlasTexture.js` (491 L) - skyline rectangle packing into pages sized to device limit (no longer 4096
  clamp); **no bindless**.
- **Denoise/upscale:** `denoise/OIDNDenoiser.js` uses `oidn-webgpu` 0.6.0, runs *once on completion* (`maxSamples>0`), aux
  albedo/normal accumulated by the tracer; `upscale/FSRUpscaler.js` (FSR, final image only, not in low-res mode).
- **Temporal behavior (`WebGPUPathTracer.js:1495-1700`):** a reset starts `_resetTime`; while `< renderDelay` (16 ms
  default) it is in **low-res mode**: `lowResScale` (0.1 default) or the adaptive scale from
  `getPreviewPixelBudget()`, traced with `dynamicLowRes`; once out of low-res, the low-res frame is copied
  (`copyTextureToTexture`) and **cross-faded** to the full-res target over `fadeDuration` 250 ms once the *average*
  sample count reaches `minSamples`. **No reprojection or history; reset policy = discard everything.** The denoiser and
  FSR are skipped in low-res mode. Sample counts are tallied on GPU and read back a few times per second
  (`TallySampleCountsKernel`; `COMPLETE_CHECK_INTERVAL`).
- **Maintenance:** very active (changelog entries weekly), single maintainer (close friend of the project owner).

### 4.2 What "legacy baggage" concretely means
- ~**10.8k lines of non-WebGPU JS** (`src/core`, `src/shader` GLSL, `src/materials`, `src/uniforms`, `src/detectors`,
  `src/textures`) vs ~15.7k in `src/webgpu`; the WebGL path is data-texture-packed BVH + GLSL string templating +
  `MaterialBase` shader chunk system + precision/compile detectors, now deprecated but still the bulk of the repo and
  all but one example group.
- Hard coupling: `three/webgpu` + TSL node system (`StorageBufferAttribute`, `StructTypeNode`, `wgslFn`, `Fn`),
  three version lock (r185 -> r187 within weeks), `bvhData` exposed through TSL context, `GraphMaterial`,
  `PhysicalCameraShim`/`ArrayCameraShim`. Not engine-agnostic: geometry, materials, lights, cameras come from
  three objects. A core cannot be extracted without untangling TSL-generated WGSL.
- three-mesh-bvh's GPU side is a *by-product* of a CPU raycaster library: BVH2 + stack 60 + throws; `MeshBVH`
  CPU-side shapecast machinery, `ClusteredBVH` limits (`NODE_INDEX_BITS`, `ROOT_INDEX_BITS`), unstable API.
- Take-aways to adopt **as ideas** (not code): persistent slot pool + GPU-counter indirect dispatch; adaptive frame
  budget from timestamp queries; paths-per-frame budget rather than tiles; aux buffers accumulated by the tracer from
  the *first scatter* (so DoF/transparency are right) instead of rasterized G-buffers; shadow rays that attenuate through
  partial transparency and stop at first opaque hit.

---

## 5. Other serious projects (notes)

- **Rayzee** (`rayzee/src/pathtracer`, `bvh`, `docs/internal/design/*`): "pure wavefront": kernels Generate -> Extend ->
  (optional Sort) -> Shade -> Compact (+ FinalWrite), stream compaction, fixed path budget, image streamed in *row bands*
  (`PATH_TRACER_SHADER_ARCHITECTURE.md:25`), SoA path state (`PackedRayBuffer`, `QueueManager`, `KernelManager`), TSL-generated
  WGSL, Node/headless mode on Dawn, deterministic mode. **BVH:** Rust->WASM SAH build (`rayzee/wasm/bvh`), spatial
  split + `ReinsertionOptimizer` (Bittner-style), `SplitBLAS`, `TLASBuilder`, **4-wide `WideBVH` with quantised child boxes
  ("encodeClusterBoxes") and folded leaves** (`BVHLeafFold`), cluster copies for instancing, stack traversal with depth
  computed per tree (40-128, `TraversalStack.js`), refit workers, out-of-core pages (Moana 111.6M tris, 51M instances, 24 GB
  Mac). **Denoise:** OIDN with `OIDNTemporalHistory` + MotionVector stage ("shimmers about half as much"), an NRD port (QUARANTINED under Denoiser clean-room rules: removed from our clone, do not read), Real-ESRGAN
  upscaler (ONNX in a worker). Camera move = *soft reset* so temporal denoisers keep history (PIPELINE_ARCHITECTURE.md:469).
  `interactionModeEnabled` default true, `interactionRenderScale` 0.5 (full bounce budget kept, unlike mutracer).
  Bidirectional + vertex merging opt-in. **Bench harness** (`bench/`: quality vs goldens/ground truth, GPU-timestamp
  per-kernel timing, same-session interleaved A/B with 3% band, memory/leak, load) is the best performance-engineering
  practice in the web field - copy the *method*. Claimed: Country Kitchen 1.4M tris, 2048x1152, 60 spp in 6 s on Apple
  M5 Pro (derived ~24 Mpaths/s at default 3 bounces). MIT, v7.x, one maintainer, active. Gaps: no ReSTIR; sampler details
  not documented in the pages I read (**unverified**).
- **strahl:** megakernel-style single compute pass (`strahl-lib/src/shaders/tracer-shader.ts`, wg 8x8), BVH from
  three-mesh-bvh `MeshBVH` internals (`bvh-util.ts`, asserts private `_roots/_indirectBuffer`), stack traversal
  (`shaders/bvh.ts:151`), OpenPBR, PCG RNG, Russian roulette, Gaussian filter or OIDN denoise, no temporal. Paper numbers:
  Chrome 126, 512^2, 100 spp, depth 5, 1.07M tris: **2319 ms (M1 Max), 1059 ms (R5 5600X+RTX 3080)**; BVH setup 327/383 ms
  (means of 30 runs, timestamp queries). Derived ~11 / ~25 Mpaths/s. Self-described "not optimized". Last commit 2025-12.
- **ShadyTracer:** wavefront (march + shade kernels, GPU counters -> indirect dispatch) but over **SDF/analytic CSG
  scenes**, not triangle meshes; photon-map and light-tracing caustics; rigorous "checked against closed-form" culture;
  `docs/PERFORMANCE.md` has same-GPU A/B tables (e.g. distance-only march -4% to -35% GPU ms/sample). Shader compile
  of scene-inlined kernels takes 10-20 s (a design cost). Interesting: GPU-adaptive minfps, error bars per pixel.
- **webgpu-doom-pathtracer** (blog + repo): temporal accumulation with prev-VP reprojection, depth rejection, YCoCg
  neighborhood clamp, two ping-pong textures; 5-pass a-trous (5x5, steps 1..16, normal+depth weights, colour weight
  dropped for bias); admits moving-object ghosting; ~330M rays/s assumed for fps table (1 spp ~80 fps at 1280x800, 4
  bounces, author's Mac). Useful as the *minimal* temporal design to compare against.
- **Hush:** SVGF-like temporal + per-pixel tiny network (half-res) that predicts filter parameters (not pixels), a-trous,
  TAA; takes 1-spp noisy + G-buffer + motion vectors as GPUTextures; renderer-agnostic. Closest to the "neural + temporal
  denoiser" idea; our Denoiser can borrow the "network steers classical filters" robustness argument.
- **Web-RTRT (2024):** the only WebGPU **ReSTIR (DI, spatial reuse) + V-buffer** web demo found (`reservoir.wgsl`,
  `spatialReuse.wgsl`, `temperalAccum.wgsl`), plus denoise and super-resolution; BVH in a worker. Stale since 2024-06.
  Skim for reservoir layout/ping-pong; not a quality reference.
- **Hardware RT forks:** willwaltz patches (173-file `dawn.patch` + 34-file `chromium.patch`; JS API
  `createAccelerationStructure`, `buildAccelerationStructure`, WGSL `enable chromium_experimental_ray_query;`; Vulkan only;
  per-origin switch `--webgpu-ray-query-origins`; Brave 1.97 / Chromium 155, RTX 4090). Reported "Globes" ray-traced water
  passes at 1528x761: 1.41/2.09/2.05/2.28 ms HW vs 1.76/2.91/2.37/2.49 ms SW (so 1.1-1.4x; their SW tracer is already
  specialized). web-rayquery: wgpu 30.0.0 `EXPERIMENTAL_RAY_QUERY` on DX12 (RT tier 1.1 + SM 6.5), Vulkan, Metal
  (macOS 15); Firefox 161 Nightly build, pref `dom.webgpu.experimental-ray-query.enabled`; RT *pipelines* Vulkan-only.
  **ThreeRT** (Rust/wgpu, Metal/Vulkan ray queries, software-BVH fallback) shows what a HW-RT PT looks like natively
  (60 fps Bistro at 1280x720, 15.5 ms on Apple M3 Max, ΔE vs 4096-spp reference 3.4 vs three.js 10-12) but is *not* in-browser.
- **tinypoly gpu-renderer, rindr, lumina-wgpu, moonlight, rayfinder, wtrace, gnikoloff:** feature-level only; rindr is
  three-gpu-pathtracer + OIDN with a CAD front-end (noncommercial license - do not reuse). Erich Loftis' renderers are
  WebGL2 real-time (analytic + BVH) with an aggressive noise-reduction culture (CC0), no WebGPU port found.
- **Engines:** Babylon.js 8 (Mar 2025) has IBL shadows, area lights, node render graph; Erich's WebGL2 path-tracer port to
  Babylon was a forum project (2020-21); no built-in path tracer found. PlayCanvas/Needle/Orillusion/Galacean: no path
  tracing found. Use.GPU: a demo path tracer only. bevy_solari: native, wgpu ray queries, ReSTIR DI/GI + world cache.

---

## 6. Comparison table

(M = megakernel, W = wavefront. "claim" = project statement, "derived" = my arithmetic, not comparable across scenes.)

| project | backend | arch | BVH (build / layout / traversal) | sampling | materials | light sampling | textures | denoise | temporal in motion | perf claim | license | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mutracer | WebGPU (+WebGL2, native wgpu) | M, 8x8 tiles | TinyBVH JS/WASM in workers; TLAS+BLAS; bvh2/bvh4/GPU/**CWBVH**, SBVH opt; private-mem stack | Owen-Sobol / ZSobol / PCG; adaptive only offline | OpenPBR 1.1 (+SSS, thin film, dispersion, volumes) | NEE+MIS, light RIS (<=N cands), env CDF | atlas array **or bindless (Chrome resource table)** | JNLM or OIDN (small/standard/large) | **none**: low-res 1-spp preview + OIDN-small + EASU | 4.6M tris 1080p 64spp 12.1 s M4 Max (~11 Mp/s derived) | MIT | very active, 0.x |
| three-gpu-pathtracer (WebGPU) | WebGPU via three TSL | **W default**, M optional | three-mesh-bvh: JS SAH + worker; TLAS+BLAS/clusters; **BVH2, stack 60** | Sobol (default), PCG, blue dither | glTF/EON, sheen, SSS, media, fog | contribution-weighted light pick + MIS, env CDF | skyline atlas pages | OIDN on completion; FSR final | none (low-res + 250 ms crossfade) | none published (frameBudget adaptive) | MIT | very active, unstable API |
| Rayzee | WebGPU via three TSL | **W**, row bands, compaction, opt. sort | Rust-WASM SAH+reinsertion; **4-wide quantised**, folded leaves; TLAS; stack 40-128; refit workers | (not documented in what I read) | PBR+clearcoat/sheen/iridescence/aniso/glass/SSS | MIS, env/sky importance, BDPT opt | (not examined) | OIDN **w/ temporal history**, NRD ReBLUR port, ESRGAN | soft reset keeps history; interaction scale .5 | Kitchen 1.4M tris 2048x1152 60spp 6 s M5 Pro; Moana 111M tris | MIT | very active, 1 dev |
| strahl | WebGPU | M | three-mesh-bvh `MeshBVH` (private fields); BVH2 stack | PCG | OpenPBR | NEE + env | (n/a) | Gaussian or OIDN | none | 1.07M tris 512^2 100spp: 2.3 s M1 Max / 1.06 s 3080 | MIT | stalled (Dec 2025) |
| ShadyTracer | WebGPU | **W** | SDF/analytic (no triangle BVH) | (not examined) | physical-unit lights, dispersion, dichroics | NEE+MIS, learned photon targeting | n/a | (none seen) | n/a | per-scene ms tables | MIT | active, niche |
| webgpu-doom PT | WebGPU | M | per-tile SAH BVH2 stack | (not examined) | simple | (n/a) | n/a | a-trous | **reprojection + YCoCg clamp** | ~60 fps default (author Mac) | MIT | demo |
| Web-RTRT | WebGPU | M + V-buffer | JS BVH in worker | RIS/**ReSTIR DI** | PBR (glTF) | RIS + spatial reuse | (n/a) | denoise + SR | temporal accum | none | MIT | stale 2024 |
| Hush | WebGPU (denoiser) | - | - | - | - | - | - | SVGF+net+TAA | yes (motion vectors) | "few passes/frame" | MIT | new |
| willwaltz / web-rayquery | Dawn / Firefox forks | HW ray query | driver AS (TLAS/BLAS) | - | - | - | - | - | - | HW 1.1-1.4x SW (own scene) | BSD-3 / Apache | experimental |
| ThreeRT | native wgpu (Metal/Vulkan) | HW ray query + SW BVH fallback | driver / `bvh.rs` | (not examined) | PBR, physical units | all emissives | texpack | a-trous+variance (SVGF-style) | live, 60 fps | Bistro 15.5 ms @720p M3 Max | MIT | active, native-only |

---

## 7. Techniques / findings (with ratings)

| # | finding | cost/benefit and WebGPU mapping | rating |
|---|---|---|---|
| T1 | Persistent path-slot pool + GPU-side queue counters -> `dispatchWorkgroupsIndirect` (three-gpu-pathtracer; Rayzee; ShadyTracer) | Removes CPU round trips; the only way to be fully GPU-driven in WebGPU today. Needs SoA path state (3GPT `rayDataStruct` ~160+ B/path). Wins when material divergence/register pressure dominate; loses on traversal-bound scenes (67.js: 2x slower). Subgroups can help compaction | **Prototype** (build both M and W in the core, same BVH/sampler API) |
| T2 | Megakernel with tiles + timestamp-budget batching (mutracer) | Simple, low memory (110 B/px incl. accum), good on Apple; register pressure on big OpenPBR bsdf | **Adopt now** as baseline/"fast" path for AO/shadows/reflection utility queries |
| T3 | Low-res 1-spp preview + OIDN-small + EASU (mutracer) | Cheap calm motion; wrong GI. Our Denoiser + upscaler already cover it | **Adopt now** as the *fallback* motion mode |
| T4 | Soft-reset with temporal history + motion vectors into denoiser (Rayzee OIDN-temporal, NRD ReBLUR port, Hush, Doom) | Keeps detail and GI through motion; needs motion vectors/first-hit AOVs (aux accumulated at first scatter, 3GPT) and disocclusion handling | **Prototype** (priority differentiator) |
| T5 | 4-wide quantised BVH + folded leaves + SAH w/ reinsertion (Rayzee) vs CWBVH (mutracer/tinybvh) vs BVH2 (3GPT/strahl) | Wide BVH cuts node fetches/stack traffic. Our `bvh` already has tinybvh layouts + H-PLOC; compare layouts *in the tracer* with identical rays | **Prototype** (bvh repo owns this; core must be layout-agnostic) |
| T6 | Stack depth derived per tree + shader constant (mutracer, Rayzee, 3GPT throws) | Avoids private-array waste/overruns; D3D12 dynamic-stack-loop miscompile hazard noted in mutracer | **Adopt now** (and test on D3D12) |
| T7 | Bindless via `sampling-resource-table` with atlas fallback (mutracer) | Removes atlas packing/filtering seams, mip bleed; needs flag, >=9 storage buffers/stage, probe at init | **Adopt now** (our stance), study `wgpuBindless.ts` API shape (cite only) |
| T8 | Sampler: Owen-Sobol default, ZSobol blue-noise-error for low-spp/no-denoiser (mutracer, 3GPT) | Per-pixel Owen scramble decorrelates error for denoisers; ZSobol gives blue-noise error at 1-4 spp | **Adopt now** (Burley hash Owen + pbrt ZSobol are public) |
| T9 | Adaptive frame budget from GPU timestamps (3GPT: `frameBudget`, `adaptiveFrameBudget`; mutracer `GPUTraceQueue`) | Holds frame time with dynamic res/paths; needs `timestamp-query` (not available on all browsers) | **Adopt now** |
| T10 | Aux albedo/normal accumulated by tracer from first scatter (3GPT 0.0.28, mutracer) | Correct under DoF/transparency vs raster G-buffer; feeds OIDN and temporal filters | **Adopt now** |
| T11 | Light RIS candidates per vertex, stratified selector independent of path stream (mutracer) | Cheap many-light NEE; precursor to ReSTIR DI | **Prototype** |
| T12 | ReSTIR DI/GI, world cache (Web-RTRT demo; bevy_solari natively) | Web gap; needs reservoir ping-pong buffers, G-buffer, motion vectors. | **Later** (after T4) |
| T13 | HW ray queries via forked browsers | 1.1-1.4x in one specialised workload; not shippable | **Reference only**, keep an abstract `trace()` seam |
| T14 | Regression/A-B bench method (Rayzee `bench/`: GPU-timestamp compute-only timing, interleaved same-session A/B with 3% band, machine fingerprint, goldens + ground truth) | Makes "speed above all" enforceable | **Adopt now** (method only) |
| T15 | Neural relighting / proxies (relight), NRD/neural denoise | Research direction; browser inference cost | **Reference only** |

---

## 8. For the spec

1. **Core = engine-agnostic trace + minimal shade; scheduler is pluggable (megakernel / wavefront).** Define one ray
   query contract (closest/any-hit, `maxDist`, flags) over our `bvh` GPU layout; implement both schedulers behind it
   and keep the choice a runtime option (as 3GPT does). Default per-hardware after benchmarking.
2. **Do not couple to TSL or any engine.** The only successful three-based tracers pay for it in version churn and
   unextractable WGSL. Adapters (three, Babylon, glTF-direct) convert to a neutral scene description (mutracer's
   Engine/Scene/Render split is a decent shape to compare).
3. **GPU-driven from day one:** paths-per-frame budget (not tiles) + timestamp-adaptive budget + indirect dispatch +
   GPU-side sample counters (3GPT reads back counts only a few Hz; avoid readbacks in the hot path).
4. **Motion quality is the differentiator:** ship (a) mutracer-style low-res denoised preview as fallback, and (b) a
   temporal path: soft reset on camera move, motion vectors from the first-hit AOV, history validation by depth/normal,
   feed OIDN-with-history or our real-time denoiser. Measure flicker/ghosting numerically (see section 9).
5. **Bindless-first textures with atlas fallback;** mimic mutracer's capability probing and fallback reason strings.
6. **Sampling:** Owen-scrambled Sobol default, ZSobol option, blue-noise-error low-spp mode; deterministic `sampleSequenceOffset`
   for animations/benchmarks; stable per-pixel sequence across preview frames (candidate: improves denoiser stability).
7. **Stack-depth/BVH-layout aware shaders** (compile-time constants), D3D12-safe loop forms, no dynamic stack-top loops.
8. **Benchmark harness is a product feature:** adopt Rayzee-style A/B (interleaved, compute-only GPU time, fingerprinted
   machine) from the first commit.
9. **Licensing hygiene:** MIT sources (mutracer, 3GPT, Rayzee, strahl) are ideas-only per brief; rindr is PolyForm-NC (never read
   to port); web-rayquery Apache-2.0, willwaltz BSD-3 (fine to reference).

---

## 9. Bench / comparison ideas

**Who to benchmark against (priority):** (1) **mutracer** (WebGPU, megakernel, same OpenPBR focus, ships numbers);
(2) **three-gpu-pathtracer WebGPU**, both `useMegaKernel` settings (needs three r187 scene wrapper); (3) **Rayzee** (wavefront,
wide BVH, best temporal story); (4) strahl as the 2024 peer-reviewed baseline (cheap to run, simple). Native references:
pbrt-v4 (quality), Mitsuba 3, our own native Windows runs (OptiX/Capsaicin) later. Skip rindr/educational demos.

**Method (time-to-quality, not just Mrays/s):**
- Same scenes everywhere, exported once to glTF (+ matching `.pbrt` for Mitsuba/pbrt): Cornell box, Sponza, Bistro (Lumberyard),
  Classroom/Kitchen (Bitterli, CC-BY - same ones mutracer/Rayzee use), Bugatti-class 4-5M tri model (CC-BY-NC-SA: bench-only),
  a glass/volume scene, a many-light night scene. Same camera, HDRI, resolution (1280x720 and 1920x1080), bounce cap (6, plus 2
  for preview parity), same tone map off (compare linear HDR).
- Metrics: (a) **throughput** = GPU-timestamp ms per sample and derived Mpaths/s at fixed bounces (Mrays/s only if rays are
  counted identically); (b) **BVH build/refit/TLAS-update times** and memory (VRAM, `bvh` repo); (c) **time-to-quality**: relMSE/FLIP/
  SSIM vs a 4096-spp pbrt-v4 reference at 1, 4, 16, 64, 256 spp and at 100 ms / 1 s / 10 s wall-clock, with and without each
  project's denoiser (denoiser-on is what users see); (d) **motion**: scripted camera orbit at fixed speed, per-frame temporal flicker
  (frame-to-frame luma delta on static regions), ghost lag (error on moved objects), and FPS; compare mutracer preview vs Rayzee
  temporal OIDN vs ours; (e) first-frame latency incl. shader compile (WebGL2 29 s vs WebGPU 0.47 s was seen in one report; ShadyTracer 10-20 s).
- Harness: headless Chrome (`--headless=new`, own profile; mutracer's AGENTS.md uses the same) on M-series and on the Windows box;
  fixed-seed deterministic modes (`sampleSequenceOffset`, Rayzee deterministic mode); interleaved A/B runs, >=30 reps with timestamps,
  fingerprint GPU/driver. Record in a repo-level CSV; plot trend as Rayzee's `bench:trend` does.
- Ground truth: pbrt-v4 outputs for all scenes + OpenPBR-to-pbrt material translation caveats (mutracer/strahl OpenPBR vs pbrt's
  models differ; note bias, compare *within-renderer convergence* too).
- Pitfall list: Apple GPUs overlap render and compute timestamps (Rayzee note) - time compute passes only; M4/M5 vs desktop GPUs
  differ a lot in register pressure behavior (wavefront vs mega); denoiser tile pacing to display frames (3GPT/OIDNDenoiser) distorts
  wall-clock if not isolated.

---

## 10. Needs Windows / native hardware

- **D3D12 behavior** of our WGSL (dynamic stack loops, huge shaders, Tint/DXC compile time, FXC fallback) - mutracer already hit a
  non-terminating shader there; Rayzee/3GPT/Strahl numbers on RTX 3080-class (strahl 2x faster than M1 Max) mean Windows timings are the
  headline for "faster than any web path tracer".
- **Hardware RT experiments:** willwaltz Chromium patches need **Linux + NVIDIA Vulkan** and an 80-minute Brave/Chromium build;
  web-rayquery needs a Firefox build on **Windows 11 + DX12**; native `wgpu` ray queries (DX12 tier 1.1/Vulkan/Metal) for a HW backend seam.
- **Native comparisons:** pbrt-v4, Mitsuba 3 (CUDA/OptiX), OIDN native (CPU/CUDA/HIP), NRD/DLSS-RR, ThreeRT/Capsaicin/RTXPT builds.
- **GPU timestamp queries** on browsers: Chrome desktop only reliably; Safari/Firefox varying.
- **Subgroup / subgroup-matrix OIDN** kernels (Rayzee `bench:oidn` compares default vs subgroup matrix) need NVIDIA/AMD/Apple parts to profile.

---

## 11. Open questions

1. Is the preview frame's sample sequence identical every frame in mutracer (frame-coherent noise)? Not verified.
2. Rayzee's sampler (Sobol/PCG/blue noise), texture strategy, and light-sampling details - not read; a second pass on
   `rayzee/src/shaders` and `docs/internal/design/SHADING.md` is warranted before final comparisons.
3. Real numbers: nobody publishes cross-project comparisons. Need our own harness (section 9). How do mutracer, 3GPT-wavefront,
   Rayzee compare on one M-series and one RTX Windows box, same scenes? Unknown.
4. Wavefront vs megakernel on **our** BVH (H-PLOC + wide layouts) and Apple vs NVIDIA vs AMD: 3GPT/Rayzee say wavefront wins,
   a 67.js spike says 2x loss; scene- and hardware-dependent. Subgroup-based material/ray sorting may flip it.
5. How far can `sampling-resource-table` bindless go (limits, mip/anisotropy support, Safari/Firefox)? Chrome flag status in 2026-11 not
   verified beyond mutracer's probe code.
6. Can temporal OIDN (Rayzee) run inside a frame budget with ours? Need to inspect `OIDNTemporalHistory` (cite only).
7. Does anyone ship WebGPU ReSTIR GI beyond Web-RTRT (2024)? None found; verify via paper/code searches (EGSR/HPG 2025-26).
8. gpuweb ray tracing: acceleration-structure / ray-query proposals (#11053-#11056) are on wgpu, not WebGPU spec; watch for a
   Dawn prototype. Our timeline should assume software RT for years.
9. Maintenance risk: all strong projects are 1-person and agent-assisted; API churn is high (three r185->r187 in weeks).
