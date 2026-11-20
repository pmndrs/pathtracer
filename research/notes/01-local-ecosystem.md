# 01 - Local ecosystem audit (sibling packages in ~/Developer)

Date: 2026-11-13. Read-only audit of README / SPEC / HANDOFF / CLAUDE.md / docs, plus skims of `src`. Nothing in
~/Developer was edited, built or run. Numbers quoted are the siblings' own (mostly Apple M5 / M5 Pro; RTX 3060 Laptop
for ddgi); I did not re-measure anything. Items I could not verify are marked **(unverified)**.

## 1. TL;DR

- **One tracer already exists and is ours: `bvh` (@pmndrs/bvh, layout version 2, unreleased).** Two-level BVH2 (64 B
  Aila-Laine nodes, 48 B triangle records, 64 B instance records), WGSL `bvhTraceClosest/Any`, watertight triangle
  test, any-hit order option, ray masks, merged static instances, GPU H-PLOC/LBVH builders + GPU refit (need
  `subgroups`), CPU port + oracle. It deliberately **ends at the hit**: `BvhHit { t, instance, triangle, barycentrics,
  frontFace, material (u16-ish opaque id), normal (world, geometric) }`. Everything about what a hit looks like
  (UV/shading normal interpolation, textures, BSDF, lights, RNG) is *this repo's* job. The core must build on it, not
  beside it.
- **At least 14 independent ray walks / ray marchers exist across the siblings** (section 4). The big duplicates are
  (a) ddgi's frozen BVH copy (`webgpu-ddgi/packages/bvh`, layout 1, `gi*` names), (b) reflections' port of
  three-mesh-bvh's walk (`MeshBVHTraceProvider`), (c) Denoiser's test-bed tracers (analytic + three-mesh-bvh wgslTagFn),
  (d) ddgi's `ddgi-shared/path-tracer.ts` reference tracer, (e) ssgiDev + reflections screen-space marchers. The core
  should absorb (a)-(d) behind one "trace + minimal shade" module; (e) stays screen-space but should share the G-buffer
  contract.
- **Bindless is used nowhere.** Zero hits for `binding_array`/bindless in any sibling's source. Only mentions: ddgi
  `docs/divergence.md` ("only if binding arrays come to WebGPU"), idTech 8 research notes in ddgi-research and
  ssgiDev. Every sibling is binding-budget-bound instead: default 8 storage buffers/stage, 16 sampled textures
  (reflections), 4 storage textures (downsampler/SwiftShader/compat). A path tracer needs more than 8 storage buffers
  unless it packs aggressively or requests adapter limits (ddgi's `requiredLimits(adapter)` pattern). Bindless-first
  with a classic fallback is therefore a real gain here, and the fallback path must be designed around the 8-buffer
  budget the siblings already live with.
- **Conventions are 90% aligned but 5 things will bite** (section 6): view-Z / linear depth sign, motion vector
  direction (NDC delta vs UV delta vs NRD's opposite), normal space (OIDN aux = *view* space in the Denoiser guide vs
  world everywhere else), hit-struct field sets/names, and `gi*`/`bvh*` WGSL name prefixes + layout-version drift.
- **Measured baseline to beat (RTX 3060 Laptop, ddgi/radiance-cache, our own BVH, before static merge):** surface-origin
  rays 17-35 Mrays/s, air-grid rays about 87 Mrays/s (Sponza). Cause: overlapping instance boxes at TLAS + clutter;
  merged static tree gave Sponza surface rays 1.85x (M5). That is the number path-tracer traversal work should attack
  (wide BVH, ray sorting, TLAS quality), and a Brixelizer-style SDF port measured 10-30x *slower* for short rays.
- **Denoiser/upscaler are ready consumers.** Denoiser: OIDN weights in hand-written WGSL fp16 (2.4-3.1x native Metal
  at 1080p; 33.6/67/167 ms small/base/large at 1080p), WebNN fp16 1.2-1.6x native Metal on base/large (Chrome flag),
  zero-copy `denoiseTextures({color, albedo, normal, hdr})` on a shared `GPUDevice`. Upscaler: FSR1 + FSR3-style
  temporal as WGSL, Halton(2,3) jitter, NDC-delta motion. **Denoiser/upscaler validation is Apple-only so far.**
- **Real-time denoiser branch is parked** (not merged, boil/ripple in real scenes, 306 GB of captures): do not plan on
  `denoiser/realtime`; FFX shadow/reflection denoisers (`denoiser/ffx`, MIT) are the shipped temporal ones.
- **Clean-room hazard:** `~/Developer/nrd-quarantine` and the Denoiser clean-room rules forbid opening NVIDIA NRD
  code. Our path tracer research must not read it either. `rt-denoiser-sources` is papers only (no NRD code) and several
  are CC BY-NC-ND / (c) authors: **do not copy those PDFs into `research/papers/`**; link instead.
- **Texture streaming is not in gputex itself but in HomeFig's fork usage:** three compressed texture *arrays*
  (baseColor sRGB BC7/ASTC, normal BC5 or ASTC with X in RGB / Y in A, ORM linear), 1024x1024 layers (multiple of 4),
  capacity doubles when full, GPU-side `encodeMipChainIntoTexture` writes a layer straight from a compute encode with
  no readback. A path tracer can sample these with `textureSampleLevel` (explicit LOD, no derivatives) from compute.

## 2. Sources (local) and licenses

| name | path | what | license |
|---|---|---|---|
| bvh (@pmndrs/bvh) | `~/Developer/bvh` (`HANDOFF.md` 2026-11-10..13, `docs/internals/design/trace.md`, `src/gpu-layout.ts`, `src/trace/wgsl.ts`) | BVH build/refit/walk, layout v2 | MIT (D. Smolek); tinybvh MIT, HIPRT MIT, three-mesh-bvh MIT (dev only) per `THIRD_PARTY.md` |
| downsampler | `~/Developer/downsampler` (`SPEC.md`, `PROGRESS.md`, `DIVERGES.md`) | SPD-style mip/Hi-Z/reduction + tool generator | MIT (Poimandres) |
| Denoiser | `~/Developer/Denoiser` (`docs/status/ROADMAP.md`, `docs/specs/temporal.md`, `packages/denoiser-core/src/temporal.ts`, `docs/research/rt-denoiser/CAPTURE-FORMAT.md`) | OIDN-in-WGSL/WebNN, FFX denoisers, TSL nodes | MIT; FFX ports MIT (AMD notice kept) |
| upscaler | `~/Developer/upscaler` (`CLAUDE.md`, `docs/inputs-and-contracts.md`, `docs/engine-contract.md`, `docs/temporal-guides.md`) ; `~/Developer/fsr3` is the older copy of the same | FSR1 + FSR2/3-style temporal | MIT (+AMD notice) |
| reflections | `~/Developer/reflections` (`docs/SPEC.md`, `docs/HANDOFF.md` trace-seam section, `packages/core/src/{frame,trace,sssr}`) | probes + SSSR + RT fallback | MIT (+NOTICE) |
| webgpu-ddgi (@pmndrs/gi) | `~/Developer/webgpu-ddgi` (+ worktrees `webgpu-ddgi-radiance-cache`, `-rc-l1`, `-rc5-*`, `-rc6-*`, `-m34`) | GI platform + DDGI; RC branch ports Capsaicin GI-1.x | MIT |
| ddgi-research | `~/Developer/ddgi-research` | earlier experiments, F6/F7 path-tracer reference, G3 ray provider | MIT (**unverified**; own THIRD_PARTY.md) |
| gi-school | `~/Developer/gi-school` | Next.js GI course, 2D GLSL engine (jump-flood SDF marcher) | n/a (docs) |
| gitestbench | `~/Developer/gitestbench` | early DDGI/probe bench (cubemap-raster bake) | n/a |
| homefig/gputex | `~/Developer/homefig/gputex` (`HOMEFIG.md`), renderer use in `homefig/primary/packages/renderer/src/threeJS/Subsystems/MaterialSubSystem/Arrays` | GPU texture compression + array layer writes | MIT (verekia upstream, base `9ec9330` v0.14.0) |
| WebVDB | `~/Developer/WebVDB` (`STATUS.md`, `docs/SPEC.md`) | NanoVDB traversal/HDDA in WGSL, TSL volume material | repo Apache-2.0; `pnanovdb.wgsl` vendored Apache-2.0 (emcfarlane/webgpu-nanovdb) |
| Capsaicin | `~/Developer/Capsaicin` (HEAD `914b915`, 2025-11-17; `src/core/src/render_techniques/gi1`, `.../reference_path_tracer`, `ray_tracing/`) | AMD GI-1.x + reference PT, D3D12 DXR only | MIT (AMD, (c) 2023) |
| rt-denoiser-sources | `~/Developer/rt-denoiser-sources` | papers/slides for denoising (RTG1/2, SVGF, A-SVGF zip BSD-3, ReBLUR chapter, AMD GI-1.0, FAST, STBN, Philippi, ...) | **mixed; several CC BY-NC-ND / (c) authors; keep outside repo** |
| Baker | `~/Developer/Baker` (`HANDOFF.md`) | Python hi->lo mesh/bake pipeline (Blender/Cycles, Houdini, Toolbag) | n/a (**unverified**) |
| ssgiDev | `~/Developer/ssgiDev` (`README.md`, `SESSION-HANDOFF.md`, `bench/src/demos`, `bench/src/harness`) | SSGI/SVGF/FSR3 research bench (three WebGPU) | n/a |
| MaterialXJS | `~/Developer/MaterialXJS` (+ `MaterialX`, `materialx-fidelity`, `io_blender_mtlx`, `blender_materialx_addon`) | MaterialX 1.38.6 WASM (core + GenShader), viewer; no WGSL generator | Apache-2.0 (MaterialX) |
| three-gpu-pathtracer (local) | `~/Developer/three-gpu-pathtracer` | **0.0.23, HEAD `b8b4e295` 2024-07-31, WebGL/GLSL only, three 0.163, three-mesh-bvh 0.7.4** | MIT |

Note: the local three-gpu-pathtracer is useless as a WebGPU comparison. Denoiser pins upstream main (0.0.27 snapshot,
`afc7db6`) and ddgi-research pins `149c7f57b1` (WebGPUPathTracer wavefront); upstream upscaler docs pin `5f0ad515`.
Comparisons must use upstream, not this copy.

## 3. Per-package findings

### 3.1 bvh (@pmndrs/bvh) - the core's foundation

**API surface.** Root: builders (`GpuHplocBuilder`, `GpuLbvhBuilder`, tinybvh WASM via `@pmndrs/bvh/worker`,
`createMeshLayoutBuilder` chooser), refits (`GpuBvhRefit`, `GpuMeshRefit`, `SceneTrees.setPose/setPoseCpu`),
`SceneTrees` (`create(device)`, `addMesh({positions, indices})`, `addInstance({mesh, objectToWorld})`, `update(encoder)`,
`GpuTopLevel`, merged static groups, growable storage), `traceModuleWgsl({traceSlot, triangleFilter, rayMasks,
triangleLeaves, positionLayout, nodeFormat, anyHitOrder})`, `batchKernelWgsl`/`TraceBatchKernel`, `createCpuTracer`,
`CpuMeshBvh` (raycast/shapecast/closest point, three-mesh-bvh-compatible), serialization. `/three`: `bvhSceneFromObject3D`,
`SceneTreesObject3DSync`, `BvhTraceNodes` (TSL/WGSL), `acceleratedRaycast`. `/worker`: tinybvh pool.

**GPU layout (`GPU_LAYOUT_VERSION = 2`, `src/gpu-layout.ts`, `src/trace/layout.ts`).**
- Node 64 B: `leftMin.xyz, leftIndex | leftMax.xyz, rightIndex | rightMin.xyz, leftCount | rightMax.xyz, rightCount`.
  `count>0` = leaf (<=4 tris, `index`=first tri in tree-ordered array), `count==0` = inner. Quantized variant 32 B
  (16-bit planes on per-mesh power-of-two grid, frame slot before root, outward rounding) is opt-in, M5 0.93-0.97x.
- Triangle 48 B in tree order: `v0.xyz+originalIndex, v1.xyz+flags, v2.xyz+pad`. flags: bit0 = "ask
  `bvhTrianglePasses`" (alpha test hook), bit1 = flip front face (merged mirrored), bits 16-31 = triangle material.
  Indexed variant (opt-in): 16 B `vec4u(i0,i1,i2,id)` + library-owned positions buffer (`vec4f` or packed), 8M tris in a
  128 MB binding vs 2.7M for records.
- Instance record 64 B: world-to-object 3x4 (48 B) + rootNodeOffset, triangleOffset, instance index, flags
  (bits 0-3 contributesToGi/visible/mirrored/merged; 8-15 8-bit mask; 16-31 material+1). Records + top-level nodes share
  one buffer after a 64 B header. **Three storage buffers total for tracing** (nodes, triangles, top-level), four with indexed
  positions. Stack `array<u32,64>` private; 47 levels left for a mesh tree under the top level; deeper GPU-built trees refused.
- Hit: `BvhHit{t (<0 = miss), instance, triangle (original index), barycentrics (weights of v1,v2), frontFace, material,
  normal}`; batch record 40 B (`t, instance, triangle, frontFace u32, bary.xy, material, normal.xyz`) or 24 B compact.
  Ray: `BvhRay{origin, tMin, direction (normalized), tMax}` 32 B. All declared names start `bvh`/`Bvh`/`BVH_`.
- Walk: watertight (Woop/Benthin/Wald 2013) with per-ray shear; added `bvhEdgeFunction` Metal fix (rays through shared edges
  4,513 per 2M -> 0). Closest-hit nearer child first; any-hit order `'far'` (default) | `'larger'` | `'near'`
  (any-hit 1.1-2.8x faster than near-first on bench scenes; `'larger'` 1.73x three-mesh-bvh on outside rays, but 0.78x on
  merged surface rays). Ray masks (8-bit, DXR-like) compile out with `rayMasks:false`. Mirrored transforms keep authored front face.

**WebGPU requirements.** `subgroups` for GPU builders/refit (fallback: tinybvh WASM, no GPU refit; `setPoseCpu` exists);
workgroup 256 along x for refit repair pass; storage binding size is the scene-size cap (128 MB default); no f16, no
timestamp requirement in the core (pages use timestamps). Intel UHD (subgroup 8/16) is correctness-only, nothing depends on subgroup size.

**Engine adapters.** three r187 (`>=0.187.0 <0.188.0`): BVH from Object3D (multi-material fold, groups, draw ranges,
GPU skinning/morph via `BvhPosePass`), `BvhTraceNodes`, three-mesh-bvh-compatible CPU API. Babylon: not started (P6).

**Bench/test infra.** `pnpm test:gpu` / `test:gpu:page <name>` (headless Chrome over CDP, `--gpu nvidia|intel`,
`--allow-software`), pages `trace-oracle`, `trace-inline`, `bvh-*`, `scene-trees-*`, `trace-bench-surface-*`;
`compare/` suite (`pnpm compare:node`; adapters for three-mesh-bvh, three, bvh-tree, Babylon, bvh.js, Rapier);
`bench/results/*.csv`; `docs/internals/WINDOWS-RUNBOOK.md` for RTX 3060 + Intel (**nothing from 2026-11-12 has run there**).
Assets from `webgpu-ddgi/assets` + manifests (`BVH_ASSETS_DIR`, `BVH_MANIFESTS_DIR`).

**Own ray walks.** `src/trace/wgsl.ts` (the walk), `src/trace/cpu-traversal.ts` (CPU port), `src/trace/reference-tracer.ts`
(oracle), `src/query/cpu-mesh-bvh.ts` (+ `CpuSceneBvh`) for physics-style queries, `examples/three/pathtracer` (a demo of the
walk that "may move to the new repo" = to us).

**Open (relevant).** Defaults pending RTX 3060 runs: `mergeStaticInstances`, `triangleLeaves:'indexed'`,
`nodeFormat:'quantized'`, 40 B hit record, `anyHitOrder`, 256 MiB build-peak cap. `scene-trees-mid-swap` unexplained
failure once after 405 s. Quantize jobs not batched. tinybvh `maxDepth` not honoured. Material id is opaque: **per-triangle
material fits in 16 bits (max 65534 `PMNDRS_BVH_MAX_MATERIAL`)** - a scene-wide PBR material table may exceed that.
Where others still win: three-mesh-bvh refit 1.6-2x and memory (we hold 9-10x more in the default layout), furnished-room
GPU closest-hit 1.1-1.2x. No wide (BVH4/8) nodes, no ray sorting/compaction, no SER-like coherence in the walk
(**unverified: not seen in docs; grep before assuming**).

### 3.2 downsampler (SPD for WebGPU)

**API.** `Downsampler(device)`, `.generate(tex, opts)`, `.encode(encoder,...)`, `.encodePass(pass,...)` (record into caller pass
for one timestamp pair), `reduction: 'average'|'min'|'max'|'minmax'|{wgsl}`, `source:{texture}` (depth -> Hi-Z, same size or half), `oddSizes:'exact'
(default)|'spd'`, `outputPath()` (buffer path for rgb9e5ufloat/rg11b10ufloat), `Downsampler.requiredLimits(adapter)`.
Tools: `Reducer` (tile kernels with load/reduce/store hooks, payload argmin/argmax folds), `ScalarReduction` (any size -> one
value, 2 dispatches, epilogue; used for exposure/SH sums). three adapter `generateMipmaps(renderer, texture)`.

**Layout/conventions.** 6 levels per dispatch via 64x64 tile in workgroup memory (**no single-dispatch last-workgroup
handoff: WGSL atomics/barriers do not guarantee visibility**; webgpu-spd 3.0 returned stale levels 7+ on M5 Pro). Min/max use conservative
footprint `2x..2x+2` at odd sizes so Hi-Z is correct. Reverse-Z: reflections builds Hi-Z with **max**.
Formats: rgba8unorm, rgba16float, r32float, rg32float, rgba32float, snorm, rgb9e5ufloat/rg11b10ufloat (buffer path), optional 8/16-bit via features.

**WebGPU requirements.** 6 levels/dispatch needs 6 storage textures (default limit 4 -> 4 levels or buffer path); 256 invocations;
optional `texture-formats-tier1`, `float32-filterable`; **compatibility mode** supported (D4100: one view per texture per dispatch ->
scratch levels + copy dispatches); subgroup kernel + fp16 are M3 ideas, not built.

**Perf (M5 Pro, per full chain, vs render passes).** 1920x1080 rgba8 0.037 ms (3.1x), 4096^2 0.34 ms (1.3x), 1024^2 0.016 ms (7.2x).
**Infra.** `npm run test:gpu` (headless Chrome real adapter, `--quick`, `--adapter=swiftshader`, `--compat`, 14 shards, CPU oracles + 12-19 mutation checks per feature), `npm run bench`, device lab page;
Windows RTX 3060 + Intel UHD run + native SPD/D3D11 `GenerateMips`/nvpro matches pending (`docs/internal/WINDOWS-HANDOFF.md`, branch `windows-testing`).
**Own ray walks:** none. **Consumers for us:** Hi-Z for screen-space rays and SSR fallback, mip chains for env/textures, exposure/SH scalar reductions, luminance pyramid for
adaptive sampling/progressive-exposure.

### 3.3 Denoiser

**API.** `Denoiser.create({precision, device, runtime})`, `denoise(ImageData|canvas)`, `denoiseTextures({color, albedo?, normal?, hdr, inputScale?,
output?, transfer, inputFlipY?})` (zero-copy on a shared `GPUDevice`). Runtimes (`NetworkRuntime`): `AutoRuntime` (default: WebNN fp16 for
base/large else WGSL), `denoiser/wgsl` (hand-written, fused U-Net in one encoder, f16 partial sums), `denoiser/webnn` (Chrome flag), `denoiser/kernels`, `denoiser/ort`
(opt-in; **wrong on 6/9-channel aux models**, MS onnxruntime#29651). Temporal: `TemporalDenoiser<Inputs>` (`configure/dispatch/resetHistory/dispose`),
`denoiser/ffx` (FFX SDK v1.1.4 shadows 0.49 ms, reflections 1.66 ms at 1080p M5 Pro), `denoiser/realtime` (parked). three: `denoise()`, `ffxShadows()`, `ffxReflections()` TSL nodes.

**Input formats for the path tracer.** Color: float, **linear HDR** (rgba16float/rgba32float), `hdr:true` applies OIDN PU transfer + autoexposure (or pin `inputScale`).
Aux albedo [0,1] unlit base colour (engine clamps), aux normal [-1,1] **view-space in the three guide** (`normalView`); for env/misses normal must be 0 and
aux must be first-hit (no blended normals, per ldraw-eiffel fixes). Output `rgba16float` unclamped linear (+ optional `transfer:'srgb'|'aces-srgb'` -> `rgba8unorm`);
sRGB formats cannot be storage targets. fp16 end-to-end is 53 dB vs fp32. 70-84 dB parity with native OIDN fp32 on kernels/WGSL/WebNN; one open case
`eiffel/rt_ldr_calb_cnrm` 5 LSB off on all runtimes. Models: RT (not lightmap, not temporal); OIDN 3 temporal waits upstream.
Weights hosted at `pmndrs/denoiser-weights@models-v2` (jsDelivr).

**Speed (M5 Pro, 1080p).** WGSL fp16 33.6/67/167 ms (small/base/large) = 2.4/2.7/3.1x native Metal; WebNN 31.5/40.1/65.4 ms (2.2/1.6/1.2x); ORT 101/210; native Metal 14.3/24.8/54.2; native CPU 189/331/739.
512^2: WGSL small 4.6 ms, WebNN base 7.3, large 10.8. WGSL subgroup-matrix kernel helps fp32 only (39.5/76.3/186 ms).

**Real-time contract (`temporal.ts`, `CAPTURE-FORMAT.md`).** `FrameGuides`: depth (hardware or r32f), world normals [-1,1], **motion = NDC delta current - previous
(three `velocity`, +y up)**, optional roughness, albedo, `viewZ` (**negative in front**, sky = -far), `motionZ = viewZprev - viewZ`, `detailNormal`.
`FrameCamera`: view, **unjittered** projection, prev matrices, `jitter` in pixels (x right, y down), near/far, position, `reversedDepth`. Signal inputs are *demodulated*
(diffuse / albedo with 1e-3 floor; specular / split-sum term E per Karis 2014) + hit distance; `ShadowInputs` visibility + occluder distance + light spec.
Capture buffers are raw f32 row-major top-row-first.

**Bench infra.** `tools/rt-capture` (CaptureWriter, replay with 50 ms GPU warm-up, compare/PNG tools), `tools/gpu-lock.sh`, `tools/oidn-native-compare` (OIDN CPU/Metal native), `examples/bench`
(sizes x precision x quality PSNR parity), `tools/eval`, `tools/scorecard`; Windows RTX 3060 Laptop ran native NRD in `NATIVE-HARNESS.md` (clock-locked). **All production validation is Apple only.**

**Own ray walks (list).** `examples/webgpu-raw/src/pathtracer.wgsl`; `examples/ffx-denoiser/src/scene.ts` and `examples/realtime-denoiser/src/{scene,reflect-oracle,reflect-eval}.ts`
and `packages/denoiser-realtime/dev/scene.ts` (analytic scene tracers: spheres/boxes/plane); `examples/three-temporal/src/realtime/rays.ts`, `reflections-rooms.ts`,
`reflections-hybrid.ts`, `reflect-bench/main.ts`, `golden/*` (three-mesh-bvh `/webgpu` `wgslTagFn`, `raycastFirstHit`); `three-pathtracer-webgpu` (upstream WebGPUPathTracer).

**Known issues.** `denoiseTextures` silently falls back to colour-only when aux textures resolve to undefined; uses three internal `renderer.backend.get(...)`;
upstream WebGPU path tracer lacks analytic lights; real-time branch stocktake (2026-11-06): beat NRD by 2-4 dB on its own ruler but visibly worse than FFX on reflections (boil/ripple),
~200 options, nothing merged; PSNR on analytic scenes was misleading.

### 3.4 upscaler (@pmndrs/upscaler)

**API.** Core `UpscalerCore({device})` (`configure`, `prepare`, `beginFrame()->jitter`, `encode`/`dispatch`, `dispatchGuides`/`dispatchUpscale` split, `resetHistory`, `guides`, `output`),
engine-free (raw WebGPU), three TSL nodes `upscaleScene`/`upscale`/`upscaleSpatial`, `UpscalePass`, `Upscaler`, Babylon + pixi experimental. FSR1 EASU/RCAS ports; FSR2/3-style temporal fused passes (reconstruct+depth clip, reactive, exposure, shading-change multi-scale, accumulate, RCAS in tonemap space).

**Input conventions.** `color` linear/HDR, **filterable** float (`rgba16float`), single-sample, jittered; `depth` real `texture_depth_2d` (linearized from near/far, honours reversedDepth; no log depth);
`velocity` = **three's NDC delta `current - previous`, jitter-free** (converted to UV delta via (0.5,-0.5), `prevUV = uv - motion`); optional `reactive` mask, `reactiveOpaqueColor`, `exposureTexture`/`preExposureTexture` (1x1 texel red).
Jitter: **Halton(2,3), `8 * ratio^2` phases (FSR `ffxFsr2GetJitterPhaseCount`: 18 @1.5x, 32 @2x), pixels in [-0.5,0.5], x right y down, sample at `(i+0.5+x, j+0.5+y)`;** NDC offset `(2x/W, -2y/H)`;
applied via `camera.view` offset; unjittered projection kept separately; `beginFrame` advances before applying. Published `TemporalGuides`: `dilatedMotion` (rgba16f UV delta), `dilatedDepth`/`previousDepth` (r32f linear eye-Z, sign **unverified**), `disocclusion` (rgba8unorm .r), reactive.

**WebGPU requirements.** Baseline only: 13 bindings/group, 9 sampled + 3 storage textures per stage; `timestamp-query` optional; `float32-filterable` if rgba32f color; no subgroups, no f16 required.

**Path tracer relevance.** Already integrated with upstream three-gpu-pathtracer (`FSRUpscaler` adapter, spatial config only because the PT has no motion vectors; `docs/pathtracer-current.md`); exposure pass ~43% faster on RTX 3060 (Windows audit). A path tracer that outputs motion/depth/jitter AOVs gets full temporal path.
**Windows infra.** `scripts/audit-windows.mjs --demos 14-pathtracer-alpha --exercise`, `docs/windows-audit.md`, `windows-cross-device.md`, `windows-evidence/`. Own ray walks: none. FSR 3.1.5 source-style graphs were GPU-A/B'd (+6 to +76% cost, no visible gain; not adopted).

### 3.5 reflections (@pmndrs/reflections)

**Frame contract (`packages/core/src/frame/contract.ts`).** `FrameTextures` (depth, normal xyz after `normalUnpackMul/Add`, roughness channel, metalness channel, albedo, motion, sceneColor (no specular), optional lit, clearcoat),
`FrameConventions` (`depthInverted`, `depthLinear` = r32f **positive** distance along -Z, `roughnessIsPerceptual`, `roughnessUnpack`, `motionVectorScale` maps stored motion to uv offset, `[0.5,-0.5]` for three NDC velocity, `sceneColorSpecular:'none'|'direct'`),
`FrameCamera` = superset of denoiser's (adds `prevViewProjection`, `projectionJittered`). This is the best-defined cross-engine G-buffer contract we have; **adopt it as the PT's AOV contract.**

**Trace seam (`packages/core/src/trace/provider.ts`).** `TraceProvider { code(slot), entries, resources(), version, storageBufferCount, capabilities{hitNormal, hitMaterial, anyHit, shadingNormal}, layoutVersion }`;
WGSL `Hit{t(<0 miss), triangle, barycentrics, material, instance, normal, frontFace}`, `traceFirstHit(origin,dir,tMax)`, `traceAny`, `traceHitMaterial`, `traceHitShadingNormal`, `shadeTracedHit`. Providers: `PmndrsBvhTraceProvider` (wraps `traceModuleWgsl`, checks layout 2, `rayMasks:false`) and a `MeshBVHTraceProvider` (WGSL port of three-mesh-bvh 0.9.18 walk, stopgap).
Pass uses 3 storage buffers + 4 storage textures itself; provider may use 5 (4 with probe fallback). bvh walk beats the three-mesh-bvh port 0.54-0.87x time (Sponza 0.771 vs 1.417 ms, default misses). `rayMasks` costs 1-6%.

**Hybrid.** SSSR (FFX v1.1.4 port, indirect over classify/miss lists, subgroup + workgroup variants), probes (octa vs cube-array, BC6H GPU encoder, per-probe resolution), RT only on SSR-failed pixels capped per tile, FFX reflection denoiser via the denoise seam. 1080p trace 0.004 ms at 0 misses, 0.357 ms at 174k rays, 0.535 at 281k (W8).
**Requirements.** subgroups optional (workgroup fallback required, parity checked bit-exactly), timestamp-query optional, 16 sampled textures cap (probe atlases; asks 32), compat mode supported (D2204), `core-features-and-limits`.
**Engine adapters.** three (TSL + MRT G-buffer `FrameGBuffer`), Babylon WebGPUEngine, pixi-3d (closed beta).
**Own ray walks / marchers.** `packages/core/src/sssr/{shaders,sssr}.ts` (Hi-Z screen march), `packages/three/src/trace/MeshBVHTraceProvider.ts` (three-mesh-bvh port), `packages/core/src/trace/{pass,shaders,hitShading}.ts` (miss-list RT pass + hit shading: probes at hit, env, shadow any-hit), `examples/ssr` oracles.
**Open (RT-relevant).** Traced-hit shading is probe-lit + cheap direct, not full material: dark bias vs path tracer (W28/W41/W45), RT shading normals per provider, hit-material table is 16-bit, mixed probe sizes vs 16 sampled textures (D4900), D1704-class per-material arrays handled via bvh `triangleMaterials`. bvh changes show only after `pnpm build` (reads `dist/`).
Bench: `pnpm check` headless CDP, compare tool vs path-traced reference (three-gpu-pathtracer + OIDN), bench.html tuning (`docs/SETTINGS.md` generated from schema).

### 3.6 webgpu-ddgi (@pmndrs/gi), ddgi-research, gi-school, gitestbench

**Platform split.** `@gi/core` (host contract, scene copy: meshes/materials/instances/lights/env/texture store, change log, budget, timing), `@gi/trace` (BVH + inline/batch trace; **frozen copy** of bvh at `d288adf`), `@gi/shade` (hit shading: materials, direct light + shadow rays, emission, sky), techniques (`ddgi`; radiance-cache branch ports Capsaicin GI-1.x), `@gi/three` adapter. Conventions: **metres, right-handed, Y-up, linear colour**, own documented light units (adapters convert). `isSupported(device)`/`requiredLimits(adapter)`; `configure -> await prepare -> encode(encoder) -> resetHistory -> dispose`.

**Trace WGSL contract (`contract/queries.ts`).** `GiRay{origin, tMin, direction, tMax}` and `GiHit{t(<0 miss), instance, triangle, barycentrics(v1,v2), frontFace}` + 24 B `GiHitRecord`: **bvh's `BvhRay`/`BvhHit` are its direct descendants, `gi*` -> `bvh*`, plus normal/material**. `giTraceClosest/Any`; inline tracing binds 3 storage buffers, batch 5.
DDGI update kernel uses 7 of 8 storage buffers. Scene copy is diffuse-only on purpose (albedo, emissive, cutout) - a PT's material model is much richer.

**DDGI textures.** irradiance 6x6 (+border = 8x8 per probe) octahedral `rgba16float` storing E (not E/pi); distance 14x14 (+border = 16x16) `rg16float` with `texture-formats-tier1` else `rgba16float`; probe data rgba16f; rays/probe default 256 (first 32 fixed), spherical Fibonacci + per-frame random rotation from CPU, hysteresis ~0.97. Requests `subgroups`, `timestamp-query`, `float32-filterable`, `texture-formats-tier1`, `texture-compression-{bc,etc2,astc}`, `maxStorageBufferBindingSize`, `maxStorageBuffersPerShaderStage`, `maxTextureArrayLayers` (adapters give 2048), workgroup 256 x.
**Radiance cache (rc worktrees).** Capsaicin GI-1.2 port: screen probes, world hash cache, world-space ReSTIR, GI denoiser, SSGI. Measured our-BVH tracing for 720p GI-1.2-shaped ray sets on RTX 3060 (pinned 1200 MHz): screen-probe rays 230k in 6.92 ms (33.3 Mrays/s), multibounce 69k in 2.73 ms (25.3), probe-hit shadow rays 104k in 1.85 ms (56.1), bounce shadow 37k in 0.87 (43.0); tracing total 12.36 ms Sponza, 21.9 Bistro interior, 23.8 exterior = 2.6-3.2x native Capsaicin's whole probe group (native hardware-RT, Sponza 4.75 ms, Bistro 7.6-7.8, two-room 1.31). 87 Mrays/s for air-grid rays.
**Own ray walks.** `packages/ddgi-core/src/platform/trace/{wgsl,trace-gpu,cpu-traversal,reference-tracer,oracle-compare}.ts` (frozen BVH walk), `techniques/ddgi/wgsl/trace.ts` (inline probe trace kernel), `tools/gpu/pages/ddgi-shared/path-tracer.ts` (small GPU path tracer over platform tracer + hit shading, for diagnostics), `tools/gpu/pages/trace-bench/{ours,tmbvh}-bench.ts`, `apps/viewer/src/ddgi/albedo-at-hits.ts`, radiance-cache `techniques/radiance-cache/{ssgi.ts,wgsl,world-cache}` (SSGI screen marching + hash cache).
ddgi-research: `experiments/gpu-ray-provider` (G3: three-mesh-bvh traversal in isolated provider, 14 GPU cases, 4096-ray oracle, 1e-4 abs + 1e-5 rel tolerance), `experiments/pathtracer-reference` (three-gpu-pathtracer `149c7f5` calibrated vs DDGI, F6/F7), `research/{dobb-bvh,tetrahedral-cages,transport-providers,gi-1.0/*}` (AMD HPG 2025/2026 ideas relevant to wide BVH/OBB).
gitestbench: no tracing (CubeCamera raster bake). gi-school: 2D GLSL jump-flood SDF marcher, a teaching engine only.
**Bench infra.** `pnpm test:gpu --gpu nvidia|intel --filter`, `bench/results/*.csv`, `tools/reference/bench-lock.ts` (cool start <70 C, 200 ms sampling, clock note), `tools/native/*.ps1` (native RTXGI/Capsaicin launchers on Windows), Cycles (OptiX) references 1920x1080 + `tools/compare` (FLIP, EXR), Sponza/Bistro/two-room/cornell manifests.
**Bindless:** only `docs/divergence.md` row (shared atlas + volume table because no binding arrays).

### 3.7 homefig/gputex

**What it is.** Fork of verekia/gputex (base `9ec9330`, v0.14.0): on-GPU encoders BC1/BC5/BC7/ASTC4x4/ETC2 (WGSL compute, `*_fast_f16.wgsl` variants when `shader-f16`), plus WebGL2 fallback. Fork adds `Encoder.encodeMipChainIntoTexture(srcTex, target, {layer, waitForCompletion})`: encodes then `copyBufferToTexture`s each mip straight into one layer of a caller-owned `GPUTexture` (needs `COPY_DST`, same base size, <=target mips), no `mapAsync`.
`selectFormat(features, hint, {colorSpace})`, `detectCapabilities(adapter)`; features `texture-compression-bc|astc|etc2`; `timestamp-query` for encode timing (Chrome 100 us quantization).
**Array layout (HomeFig renderer, `Arrays/`).** Three arrays `baseColor` (sRGB BC7 / ASTC4x4 / ETC2; also holds emission), `normal` (BC5, or **ASTC with X in RGB and Y in A** -> shader reads `.ra`), `orm` (linear BC7/ASTC); one layer per base-material map; default 1024x1024 layers (multiple of 4); initial capacity 8, **doubles when full**; `MaterialLayerAllocator`, `NO_LAYER` sentinel; encoder cache key `MATERIAL_ENCODER_VERSION = 'gputex-9ec9330.encode-into-texture.r1'`; `supportsMaterialArrays` = BC or ASTC (ETC2 alone cannot hold normals). sRGB transfer carried by the GPU format.
**For the PT.** Sampling a `texture_2d_array` per material with explicit LOD (`textureSampleLevel`) from compute works; a material table must carry (layerIndex per map, UV transform). Hardware-compressed formats cannot be storage targets, so write via buffer copy. `maxTextureArrayLayers` default 256 (adapters give 2048). Capacity growth reallocates (copy) - streaming must handle handle-invalidation. Bindless would replace the per-material layer index with true descriptor tables.
Own ray walks: none.

### 3.8 WebVDB

NanoVDB in WGSL: `pnanovdb.wgsl` (vendored Apache-2.0 fork of emcfarlane/webgpu-nanovdb) with root->upper->lower->leaf descent, HDDA ray marching, Float/Fp8/FpN decoders, bound as one `var<storage,read> array<u32>` (64-bit as `vec2u`); `NanoVDBFile` loader (NONE/ZIP), CPU TS reference (`readValue`, `sampleTrilinear`), `three-nanovdb` TSL volume material, compute utilities, device-first `createVolumeRenderer` (requests raised `maxStorageBufferBindingSize`, `shader-f16`, `float32-filterable`). WDAS cloud 1080p 103.96 FPS (hardware lane; GPU timestamps were invalid on that adapter). HDDA node skipping deferred to v1.1. **Own ray walk:** the HDDA/volume march in `pnanovdb.wgsl` + `three-nanovdb/src/{material,wgsl}.ts`. **For the PT:** volume traversal (heterogeneous media, delta/ratio tracking) is the natural consumer of core's ray API; the NanoVDB buffer is a second big storage binding (budget pressure).

### 3.9 Capsaicin (local AMD clone, MIT, D3D12 DXR)

Reference for GI-1.x: `src/core/src/render_techniques/gi1/{screen_probes,hash_grid_cache,world_space_restir,glossy_reflections,gi_denoiser}.hlsl`, `reference_path_tracer/*` (.rt/.comp variants), `ray_tracing/{trace_ray,path_tracing,intersect_data}.hlsl`, `lights/{reservoir,light_sampling}.hlsl`, `components/{light_sampler_grid_cdf,light_sampler_grid_stream,blue_noise_sampler,stratified_sampler,brdf_lut,prefilter_ibl}`, `materials/*`, `utilities/{gpu_mip,gpu_reduce,gpu_sort}`, `render_techniques/{fsr,ssgi,tone_mapping,auto_exposure,variance_estimate,image_metrics}`. Requires D3D12 Ultimate + Windows 10 SDK: **Windows only**. MIT, so porting is allowed with notice (ddgi RC branch already does, "faithful port first").

### 3.10 rt-denoiser-sources, Baker, ssgiDev, MaterialXJS, three-gpu-pathtracer (local)

- **rt-denoiser-sources:** papers only, no NRD code. Useful for us: SVGF/A-SVGF (+authors' code `asvgf_zip`, BSD-3, usable), ReBLUR chapter (prose/equations only; listings may mirror NRD - skip), RTG1/2, AMD GI-1.0 + glossy reflections SA2023, idTech8 GI, Lumen, MegaLights, Frostbite GIBS, FAST/IS-FAST (EA FAST BSD-3 generator not downloaded), STBN, Philippi 2023 (CC BY), Hanika 2021 reprojection without MV (CC BY-ND), occlusion-point reuse 2026 (arXiv 2607.23122, **unverified**). Missing: ReLAX GTC 2021, Dolp I3D 2024 a-trous schedule.
- **Baker:** Python CLI/`baker/runner.py`; Cycles integrated bake, Houdini/Toolbag experiments, meshoptimizer/Blender Decimate reducers, Asset Lab review app (`python3 review/server.py --port 8777`). Not a ray tracer, but its hi->lo projection bake (ray distance/cage correspondence, normal/PBR transfer) is exactly a "utility consumer" the core's baking query API should serve, and Cycles is already our reference renderer elsewhere.
- **ssgiDev:** three WebGPU SSGI bench, demos 00-21, SVGF (`14-svgf`), FSR3 integration (`20/21`), harness `TemporalGuidesPass`, `reference-marcher.wgsl` (brute-force screen-space reference, accumulating), `03-ssrt3/ssrt-main.wgsl` (GTAO-GI/Hi-Z march). Status: debug repo; demo 19/18 had persistent motion streaking, demo 21 (FSR3 + SVGF) is v1 candidate; BLOCKERS: FSR3 integration jitter. **Own marchers:** `bench/src/harness/reference-marcher.wgsl`, `demos/03-ssrt3/*.wgsl`, `04-gt-vbao`, `05-gt-vbgi`, `06-sshgi-mathis` (all screen-space).
- **MaterialXJS:** MaterialX **1.38.6** WASM (core + GenShader: GLSL/OSL-era generators; **no WGSL generator in 1.38.6**, **unverified for newer**); tests/viewer. Relevance: material graph source for spectral/PBR path tracer (`standard_surface`, `open_pbr_surface` come in later MaterialX); `materialx-fidelity` repo (core/cli/viewer, renderers materialxjs/materialxview/threejs) is the comparison harness.
- **three-gpu-pathtracer (local):** see section 2; WebGL; useful as architecture/GLSL reference only (shader dirs `bsdf/`, `bvh/`, `rand/{pcg,sobol,stratified}`, `sampling/{equirect,light,shape}`). Study upstream main for the WebGPU wavefront renderer.

## 4. Every place a sibling built its own ray tracer / ray walk (what core should absorb)

| # | Path (under ~/Developer) | Kind | Replace with |
|---|---|---|---|
| 1 | `bvh/src/trace/wgsl.ts`, `cpu-traversal.ts`, `reference-tracer.ts` | THE walk (keep, becomes core's closest/any-hit) | is the core |
| 2 | `bvh/examples/three/pathtracer` | demo of the walk | move into this repo |
| 3 | `bvh/src/query/cpu-mesh-bvh.ts` | CPU shapecast/closest-point (physics queries) | stays in bvh |
| 4 | `webgpu-ddgi/packages/bvh/*`, `packages/ddgi-core/src/platform/trace/{wgsl,trace-gpu,cpu-traversal,reference-tracer,oracle-compare}.ts` | frozen copy (layout 1, `gi*`) | switch at ddgi P3 to core |
| 5 | `webgpu-ddgi/packages/ddgi-core/src/techniques/ddgi/wgsl/trace.ts` | inline probe trace kernel | core "inline trace + hit shade" module |
| 6 | `webgpu-ddgi/tools/gpu/pages/ddgi-shared/path-tracer.ts` | small path tracer for diagnostics | core PT (reference mode) |
| 7 | `webgpu-ddgi(-radiance-cache)/.../techniques/radiance-cache/{ssgi.ts,wgsl/,world-cache/}` | screen march + world hash-cache rays | core batch trace + caching utilities |
| 8 | `reflections/packages/three/src/trace/MeshBVHTraceProvider.ts` | port of three-mesh-bvh walk | delete once core is default (already `PmndrsBvhTraceProvider`) |
| 9 | `reflections/packages/core/src/trace/{pass,shaders,hitShading,provider,pmndrsBvh}.ts` | miss-list RT pass, hit shading | core "minimal shade" (probes/env/direct at hit) |
| 10 | `reflections/packages/core/src/sssr/{shaders,sssr}.ts` | Hi-Z screen-space march (FFX SSSR port) | stays; shares G-buffer contract + Hi-Z from downsampler |
| 11 | `Denoiser/examples/{ffx-denoiser,realtime-denoiser,three-temporal}/src/*` (scene.ts, reflect-oracle.ts, rays.ts, golden/*, reflections-*.ts) and `packages/denoiser-realtime/dev/scene.ts`, `examples/webgpu-raw/src/pathtracer.wgsl` | test-bed signal generators (analytic scenes + three-mesh-bvh `/webgpu`) | core's batch/inline trace; analytic oracles stay as tests |
| 12 | `ddgi-research/experiments/{gpu-ray-provider,pathtracer-reference}/src/*`, `.../gpu-ddgi/src` | experiments (three-mesh-bvh provider, Lambert PT) | archive; reuse oracle fixtures + tolerances (1e-4 abs / 1e-5 rel) |
| 13 | `ssgiDev/bench/src/harness/reference-marcher.wgsl`, `demos/03-ssrt3/*.wgsl`, `04..06` | screen-space marchers | stay (screen space); could use core for "reference" arm |
| 14 | `WebVDB/packages/nanovdb-wgsl/vendor/pnanovdb.wgsl`, `three-nanovdb/src/{material,wgsl}.ts` | HDDA volume march | core volume interface (later) |
| 15 | `Capsaicin/src/core/src/{ray_tracing,render_techniques/reference_path_tracer}` | native DXR reference PT | reference for comparison, Windows only |
| 16 | `three-gpu-pathtracer` (local, GLSL) | legacy PT | study only |
| 17 | `gi-school/src/engine/*`, `gitestbench/src/gi/ddgi/ProbeBaker.ts` | 2D SDF marcher / raster cubemap bake | none (education/bench) |

## 5. Bindless status

Not used in any sibling (code or docs); no `binding_array`/`bindless` strings in bvh, downsampler, Denoiser, upscaler, reflections,
WebVDB, gputex, gitestbench, three-gpu-pathtracer. Binding-budget workarounds that bindless would remove: ddgi "one shared atlas for every volume +
layer bases + volume table texture, max 8 volumes, shading module may bind <= 6 textures"; reflections atlas sampling cap of 16 sampled textures (asks for 32,
counts them; four probe sizes fit in 16 via shared atlas nodes); gputex material arrays with per-map layer indices; bvh keeps tracing to 3 storage buffers
"because a default device allows only 8 per stage". **Design implication:** write core WGSL against an abstraction (`materialTexture(i)`, `geometryBuffer(i)`) with two backends:
binding-array/bindless, and packed-buffer + texture-array fallback with <= 8 storage buffers.

## 6. (a) Proposed shared "contract" for the core

Based on what 3+ siblings already agree on, so adopting it breaks nobody:

1. **Space & units.** World: right-handed, +Y up, metres; camera looks down -Z in view space (three/WebGPU); clip NDC x,y in [-1,1] +y up, z in [0,1]. Matrices column-major
   (`Matrix4.elements` / WGSL `mat4x4f`). Local->world objects as `objectToWorld` 4x4 in API, stored world-to-object 3x4 (bvh instance record).
2. **Color/precision.** Linear scene-referred HDR everywhere inside the core; radiance AOV `rgba16float` (f32 accumulation `rgba32float` / `r32float` per channel when
   needed; `float32-filterable` optional); no pre-exposure unless `exposureTexture`/`preExposureTexture` given (upscaler convention). Irradiance stored as E (not E/pi) in probes (ddgi).
   Light units: pick radiometric (W, sr, nits) and document; ddgi has "own documented units" - reconcile before PT lights ship **(open)**.
3. **Rays & hits.** Adopt bvh's: `Ray{origin, tMin, direction(normalized), tMax}` 32 B and `Hit{t (<0 = miss), instance, triangle (original index), barycentrics (v1, v2), frontFace, material (u32 opaque), normal (world geometric, unit)}`; 40 B record (24 B compact). Keep the `bvh*` prefix *reserved for bvh*; core uses a different prefix (`rt*` or `pt*`) for its wrappers so provider/seam code (reflections, ddgi) never collides. One `layoutVersion` check at start-up (bvh's `GPU_LAYOUT_VERSION`).
   Anything past barycentrics (UV, shading normal, tangent, vertex colour) is core-owned: core keeps per-mesh attribute buffers addressed by `(instance -> mesh offset, original triangle index)`.
4. **Geometry buffers & budget.** Plan for <= 8 storage buffers/stage (bvh 3-4: nodes, tris, TLAS, [positions]) in the fallback path; request adapter limits via a `requiredLimits(adapter)` helper (ddgi, downsampler pattern) - maxStorageBufferBindingSize, maxBufferSize, maxStorageBuffersPerShaderStage, maxTextureArrayLayers, 256 x workgroup.
5. **G-buffer / AOV contract = reflections' `FrameTextures` + `FrameConventions` + `FrameCamera` superset** (it already reads as denoiser's `FrameCamera` plus `prevViewProjection`, `projectionJittered`). Required AOVs from the PT for denoise/upscale: radiance (HDR), albedo (unlit base colour [0,1], first hit), normal, depth, motion, optional roughness/metalness, hit distance (diffuse/specular), `viewZ`.
   - **Depth:** hardware-style depth `[0,1]` with explicit `depthInverted` flag; plus optional `viewZ` r32f. Standardise on **`viewZ` negative in front (three/denoiser/NRD-agnostic)** and convert at the edges; do not let a third sign appear.
   - **Normal:** world space unit [-1,1], plus per-consumer view-space conversion in the denoiser adapter (OIDN aux). Env/miss = 0.
   - **Motion:** **NDC delta `current - previous`, +y up, jitter-free** (three `velocity`, denoiser, upscaler input, ddgi capture format v0). Publish `motionVectorScale [0.5,-0.5]` once for UV conversion. Optional `motionZ = viewZprev - viewZ` (denoiser sign).
6. **Jitter.** Pixel units, x right / y down, in [-0.5,0.5], sample position `(i+0.5+jx, j+0.5+jy)`, NDC offset `(2jx/W, -2jy/H)`. Sequence: **Halton(2,3), `8 * ratio^2` phases** when feeding the upscaler (shared with `@pmndrs/upscaler`'s `generateJitterSequence`); the PT's own AA sample for progressive accumulation is a separate (Sobol/blue-noise) pixel offset and must be zero when `upscaler` owns jitter. One jitter owner per pipeline.
7. **Row order / buffers.** Image row 0 = top (capture format; WebGPU framebuffer); `inputFlipY` only at the three bridging layer.
8. **Lifecycle.** `configure() -> await prepare() -> encode(encoder, ...) -> resetHistory() -> dispose()`, `isReady` + typed `NotReady` error, device always supplied by host (never create/destroy), `gpuTiming: true` fills per-pass timings only if `timestamp-query`, settings are live objects, `encodePass()` variant to record into a caller-owned pass (downsampler, for one timestamp pair per chain). Optional features requested once: `subgroups`, `shader-f16`, `timestamp-query`, `float32-filterable`, `texture-formats-tier1`, texture-compression-*.
9. **Compat/fallbacks.** Declare exactly one fallback per feature (bvh rule): subgroups -> tinybvh CPU build + `setPoseCpu`; bindless -> classic bind groups; f16 -> f32; compatibility mode -> downsampler-style scratch/copy dispatches. Verify both paths with a bit-exact parity test (reflections, downsampler practice).
10. **Material IDs/textures.** Material id u32 (< 65535 while bvh packs it in 16 bits - raise or table-redirect); texture access through the gputex array scheme (baseColor sRGB, normal two-channel/ASTC RA, ORM linear) + explicit-LOD sampling.
11. **Testing/bench contract.** Every kernel has a CPU oracle + GPU page runnable under `--gpu nvidia|intel|apple`, same CSV shape (`bench/results/*.csv`), bench-lock/clock-note, paired ratios (not absolute ms), device named on every claim, correctness gates not timing gates. Reuse bvh's `trace-oracle` fixtures.

## 7. (b) Inconsistencies that will bite

| # | What | Where | Risk / suggestion |
|---|---|---|---|
| 1 | **Depth sign:** denoiser `viewZ` negative in front, reflections `depthLinear` positive distance along -Z, upscaler `dilatedDepth` linear eye-Z (sign unverified), NRD either | Denoiser `temporal.ts`; reflections `contract.ts`; upscaler `temporal-guides.md` | Wrong-sign reprojection looks like ghosting. Pick negative `viewZ` + explicit `depthLinear` flag; convert in adapters. |
| 2 | **Motion direction/units:** denoiser/upscaler input NDC delta cur-prev (y up); upscaler *guides* publish UV delta `prevUV = uv - motion` (y flipped); reflections `motionVectorScale`; NRD uses prev-cur | all | Publish only NDC cur-prev; one conversion helper. |
| 3 | **Normal space for aux:** OIDN aux in denoiser guide = **view-space** (`normalView`); realtime FrameGuides, reflections, bvh hit normal, ddgi = world | Denoiser docs vs rest | Core emits world normal; denoiser adapter rotates (or core writes both). Test with OIDN native parity. |
| 4 | **Hit struct drift:** `GiHit` (no normal/material) vs `BvhHit` vs reflections `Hit` (adds normal) vs three-mesh-bvh `RayIntersectionResult`; `GiHitRecord` 24 B vs `BvhHitRecord` 40 B | ddgi frozen, bvh, reflections | ddgi frozen copy is layout 1 / `gi*`; switch at P3 or keep adapters. Layout version check mandatory. |
| 5 | **Name prefixes:** bvh reserves `bvh*/Bvh*/BVH_*`; ddgi `gi*`; reflections must not use them | WGSL globals | Core wrappers must use a distinct prefix and never re-declare bvh's. WGSL has no namespaces; dedupe chunks (reflections `dedupeWgslChunks`). |
| 6 | **Binding budget collisions:** DDGI update uses 7/8 storage buffers; reflections traced pass 3 + provider 3-4; PT needs nodes, tris, TLAS, materials, textures table, lights, rays/queues, accumulation | all | Core must publish a binding budget per module (ddgi `storageBufferCount` pattern) and not assume > 8. |
| 7 | **three version pins diverge:** bvh r187 only; upscaler r186+; reflections r187; ddgi r187; gitestbench r185; Denoiser pins upstream pathtracer SHA | all | Adapter layer must not depend on three internals (Denoiser uses `renderer.backend.get`; bvh uses `threeInternals` checked at run time). |
| 8 | **bvh layout unreleased (v2) and 3 defaults pending Windows data;** `reflections` reads `dist/` (needs `pnpm build`); ddgi on frozen v1 | bvh HANDOFF | Core must pin an exact bvh commit/layout; avoid silent layout drift. |
| 9 | **Material id 16 bits** (bvh packs instance/triangle material in flags 16-31, max 65534) vs PBR scenes with many materials / per-texture variants | bvh layout | Keep material = index into core's table, or widen; do not alias with `PMNDRS_BVH_MAX_MATERIAL`. |
| 10 | **Hi-Z min vs max:** reflections reverse-Z -> max; ssgiDev depth min pyramid; downsampler supports both | usage | Single `depthInverted` flag drives reduction. |
| 11 | **Premultiplied/straight alpha + transparent canvas:** upscaler carries alpha through; OIDN ignores; PT must define | upscaler/Denoiser | Decide PT output alpha semantics. |
| 12 | **Texture arrays vs bind budget:** gputex arrays default 256 layers limit (2048 on adapters); sampled-texture cap 16 (reflections) | gputex, reflections | Fallback needs atlas/array; `maxTextureArrayLayers` request. |
| 13 | **Precision targets differ:** ddgi irradiance rgba16f E; denoiser fp16 end-to-end; downsampler rg11b10/rgb9e5 only via buffer path | various | f16 accumulation inside the PT would visibly band; keep f32 accumulation, f16 only for AOVs/history. |
| 14 | **Validation hardware skew:** Denoiser/upscaler/downsampler/bvh mostly Apple M5; ddgi on RTX 3060; Intel UHD correctness-only | all | Anything tuned on Apple (WGSL conv kernels, subgroup assumptions, any-hit order) must be re-measured on NVIDIA/AMD. |
| 15 | **Jitter ownership:** upscaler claims three view-offset; three `traa()`/`taau()` also do; PT progressive AA jitter | upscaler docs | PT must expose "external jitter" mode and "own AA jitter" mode. |
| 16 | **Real-time denoise vs OIDN:** FFX signals demodulated with split-sum E; OIDN beauty frames not demodulated | denoiser | PT must output both beauty and demod AOVs if both supported. |

## 8. (c) Known bugs / open issues in handoffs relevant to ray tracing

- **bvh:** Moller-Trumbore leaked rays at shared edges (Metal): fixed with watertight `bvhEdgeFunction`; RTX 3060/Intel unrun for all 2026-11-12 work; defaults undecided; any-hit order trade-off (`larger` 0.78x on merged surface rays); one unexplained `scene-trees-mid-swap` failure (405 s, parallel build); Windows D3D12 compile cost of the pose pass unmeasured; `/three` ranges: three r187 returns NaN-distance for triangle past index end (likely a three bug); GPU skinning deform bounds set from CPU (no GPU readback); `CpuSceneBvh` layers 8-31 unseen; tinybvh WASM heap retention (fixed by release-above-bytes); wasm reference is Linux build (2 `__FILE__` separator bytes differ on Windows).
- **ddgi/RC:** probe rays from surfaces trace 17-35 Mrays/s vs ~87 for air rays (partly instance-box overlap, merged tree 1.24-1.86x); Brixelizer SDF port 10-30x slower than bvh for short rays (small room), large scenes untested; hash cache sized 1 GiB (Capsaicin default) vs 47-55k live tiles; back-face hits end probe rays black (DDGI convention) vs Cycles shading both sides; ddgi's DDGI is diffuse-only; Sponza H-PLOC build 46.6 ms on RTX 3060 at ~1000 MHz clocks (power limited, only 2 VALID runs).
- **reflections:** traced-hit dark bias vs PT (probes at hit instead of full shading, W28/W41/W45); SSR striping fixed (D1000); RT per-tile cap with last-frame feedback; pixi/Babylon adapters beta; `rayMasks` kept off (1-6% cost); glass/coat treated approximately.
- **Denoiser:** ORT WebGPU Conv bug (>3 input channels; microsoft/onnxruntime#29651 filed 2026-07-10; WGSL runtime bypasses); `graphCapture` unusable in ORT 1.27; `*_large` wrong on ORT; WebNN requires Chrome flag; WebGPU-only; analytic lights unsupported by upstream WebGPU PT; real-time (NRD-like) branch parked; native NRD harness results on `feat/rt-denoiser-native-results`; Apple-only validation.
- **upscaler:** temporal path "correct enough, not tuned"; spatial config for path-tracer demo (no MV); 3 FSR 3.1.5 candidates rejected; Windows lifecycle audits exist.
- **downsampler:** single-dispatch SPD rejected (WGSL memory model), `'spd'` odd-size mode only opt-in; subgroup/f16 kernel not built; Windows/native parity runs pending.
- **ssgiDev:** motion streaking in demos 18/19 (reprojection/disocclusion), FSR3 jitter integration blocker (demo 16) - resolved by rebuilding on demo 14 + upscaler guides.
- **gputex/HomeFig:** fork-only API `encodeMipChainIntoTexture`; no upstream PR sent; normal on ASTC uses `.ra` remap; no BC6H (HDR env) in gputex (reflections has its own BC6H encoder, Narkowicz MIT).
- **WebVDB:** HDDA node skipping deferred; GPU timestamps invalid on one adapter; strict native parity on native/hardware machine.

## 9. For the spec (concrete recommendations)

1. Core consumes `@pmndrs/bvh` as a pinned dependency (exact layout v2); do not fork the walk. Build "trace + minimal shade" as a module that wraps `bvhTraceClosest/Any` and adds: attribute interpolation (UV, shading normal) by `(instance, triangle)`, material table lookup, alpha test through `bvhTrianglePasses`, env/direct shade, shadow any-hit (reflections' `traceLightVisible` is a good seam).
2. Take reflections' `TraceProvider` + `FrameInputs` as the integration seam and declare it stable; then ddgi/reflections/ssgiDev switch with deletes, not rewrites.
3. Build bindless-first behind `materialTexture(i)/geometryBuffer(i)` abstraction, classic fallback within 8 storage buffers + 16 sampled textures (what siblings already budget for).
4. Request the same feature set siblings already request, once: `subgroups`, `shader-f16`, `timestamp-query`, `float32-filterable`, `texture-formats-tier1`, `texture-compression-*`; keep one declared fallback each.
5. PT AOVs per section 6.5 so denoiser (both OIDN and FFX) and upscaler work with zero adapters.
6. Use downsampler for Hi-Z (screen-space fallbacks), luminance pyramid (adaptive sampling, exposure), env/texture mips, scalar reductions (exposure, SH, variance stats) via `encodePass`.
7. Pull bvh's `trace-oracle`/`trace-bench-surface` fixtures into the PT's CI so traversal regressions are caught against the CPU oracle.
8. Settle the 16-bit material id limit before building the material system.

## 10. Bench / comparison ideas

- **Traversal Mrays/s** per scene (Sponza, Bistro int/ext, furnished room, knot) for: closest-hit coherent primary, incoherent diffuse bounce, surface-origin shadow rays (any-hit), air-grid rays; ours vs three-mesh-bvh/webgpu (bvh `compare/gpu` harness already does paired ratios) vs mutracer. Baseline row: ddgi RC1.5 (17-35 surface / ~87 air Mrays/s RTX 3060).
- **Whole PT:** samples/s at 1080p for Sponza/Cornell with a fixed bounce count; progressive convergence vs Cycles OptiX reference (ddgi `tools/reference/cycles`, 1920x1080, FLIP) and pbrt-v4; compare `ddgi-shared/path-tracer.ts` as a floor.
- **Denoiser pairing:** PT noisy -> OIDN WGSL/WebNN fp16 (`denoiseTextures`), PSNR/FLIP vs converged; temporal stability under motion (NOT PSNR alone: stocktake lesson).
- **Upscaler pairing:** temporal path with PT jitter + MV, NativeAA, ratio 1.5/2.
- **Hybrid:** reflections `ours-rt` vs three SSR vs PT reference (reflections compare tool), compare bias at hit shading.
- Method: paired ratios, device named, bench-lock/clock note, 5 percent noise floor, correctness gates before timing (siblings' rule).

## 11. Needs Windows / native hardware

- RTX 3060 Laptop + Intel UHD: bvh WINDOWS-RUNBOOK (decides merge/indexed/quantized/hit-record/anyHitOrder defaults); downsampler windows-testing + native SPD/D3D11/nvpro parity; upscaler Windows audit continuation; Denoiser WGSL kernels + subgroup/f16 tuning outside Apple; ddgi/RC timing under clock lock; Sponza H-PLOC build re-run with longer page.
- Native: Capsaicin GI-1.2 (D3D12 Ultimate, DXR) for yardstick; NRD native harness on RTX (clean-room - only harness builder may read NRD headers); OIDN native (CPU/CUDA/HIP) via `tools/oidn-native-compare`; Cycles OptiX references; RenderDoc/PIX/Nsight.
- AMD GPU: nothing in siblings has run on AMD (subgroup 32/64 behaviour, any-hit order, workgroup memory); NVIDIA only via RTX 3060 Laptop.
- WebNN NPU/GPU path (Denoiser): Chrome flag, measured on Apple CoreML only; Windows DirectML/NPU not measured.

## 12. Open questions

1. Who owns attribute interpolation: core only (bvh scope ends at hit) - confirm Dennis wants `examples/three/pathtracer` moved here.
2. Radiometric light units for the PT vs ddgi's "own units": reconcile now.
3. Is the 16-bit material id in bvh a hard layout constraint or a flag-word packing choice we can widen with a layout bump (v3)?
4. Wide BVH / ray sorting / SER-like reordering: not in bvh; where to build (bvh vs core)? AMD DOBB-BVH and tetrahedral cages notes exist in `ddgi-research/research/`.
5. Which denoiser inputs for the PT: OIDN aux view-space vs world normals (item 3 in section 7).
6. Does upscaler `dilatedDepth` use positive eye-Z (unverified)?
7. Is `webgpu-ddgi-radiance-cache`'s Capsaicin-port the one to keep (rc worktrees vs `webgpu-ddgi`)? Several `webgpu-ddgi-rc*` copies exist.
8. MaterialX: does 1.39+ WGSL codegen exist for us, or do we author our own node-graph->WGSL (MaterialXJS is 1.38.6)?
9. Whether `three-gpu-pathtracer` upstream main (WebGPU) is the comparison baseline rather than the old local 0.0.23.
