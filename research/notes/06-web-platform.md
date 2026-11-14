# 06 - WebGPU / web platform frontier (as of 2026-11-13)

Method: web search + page fetches (no repos cloned; nothing needed locally). Items I could not verify are marked **[unverified]**.
Support percentages are from web3dsurvey.com (crowd-sourced, skews to Chromium desktop; treat as rough).

## 1. TL;DR

- **Bindless is real but sampling-only today.** Chrome (Dawn) exposes `chromium-experimental-sampling-resource-table` behind `chrome://flags/#enable-unsafe-webgpu` on Windows, Linux, Android (macOS "coming soon" per toji, Oct 2026). Only sampled textures + samplers. **No storage buffers, no heterogeneous tables yet** (`heterogeneous-resource-table` not implemented). Survey adoption 0.14%. So: textures bindless, geometry buffers NOT bindless. This shapes the whole material design.
- Spec proposal is `GPUResourceTable` (gpuweb `proposals/bindless.md`, Draft, Oct 2025), max table size 65536, WGSL `enable resource_table; getResource<T>(i)` / `hasResource<T>(i)`. Dawn's prototype WGSL spelling is `enable chromium_experimental_resource_table;` (same functions). Names will change before shipping.
- **Geometry/BVH/vertex data must stay in a few big storage buffers** (which is what our BVH layout does anyway). Only per-material textures benefit from bindless. The fallback (2D-array texture, gputex-style) is the baseline path and must be first-class, not an afterthought.
- **Subgroups: shipped** (Chrome 134), size-control shipped in 151-152 (`subgroup-size-control`, `@subgroup_size`), `subgroup_uniformity` (145), `subgroup_id` (144), inclusive scans (131). Survey: 68% overall; Chrome 84%, Android 84%, Windows 78%, macOS 62%, **Safari 1.2%, iOS 1.4%, Firefox 0.1%**. Not universal: need a no-subgroup path.
- **shader-f16 94%, timestamp-query 99%** (100us quantised unless `chrome://flags/#enable-webgpu-developer-features` [V]; the command-line switch is `--enable-webgpu-developer-features` [V in Blink web_tests config, not in Chrome docs]; Dawn toggle `timestamp_quantization` (default on); see VERIFY.md §2), f16 on Linux NVIDIA lands 155-156. Immediates (push constants) shipped 149-150.
- **Subgroup matrix (cooperative matrix)**: Chromium experimental `chromium-experimental-subgroup-matrix` (0.04% adoption), gpuweb draft proposal (`subgroup-matrix`, `subgroup_matrix` WGSL; f16/f32/i8/u8/bf16/fp8 configs queried from `adapterInfo.subgroupMatrixConfigs`). Backends: Vulkan `VK_KHR_cooperative_matrix`, Metal simdgroup matrices (Apple7+, 8x8x8) / Metal 4 tensors, D3D12 needs SM6.10 linalg (not generally available). This is the route to a fast in-WebGPU neural denoiser; keep optional.
- **Hardware RT for WebGPU: proposals exist, nothing shippable.** gpuweb PR #11054 (acceleration structure, dj2, opened 2026-10-13, still open), #11053 (creation), #11056 / #11055 (WGSL ray query). Feature name `acceleration-structure`. Zero browser implementation known. Do not plan on it; design the BVH interface so a HW TLAS/BLAS backend could slot in later.
- **WebNN: not shipped.** Chrome origin trial started M146 (Android excluded), timeline slid to M147-149, then a release-blocking pause (OT disabled again in the M147 thread; Edge's own trial page shows closed/expired 2026-08-11); W3C: CR Snapshot 2026-01-22, CRD 2026-03-27; not enabled by default in any stable Chrome found **[V 2026-11-13 from blink-dev + W3C; no newer start date found]**. WebNN<->WebGPU interop (`createExportableTensor`, `exportToGPU`) merged into spec 2026-09-10 (PR #942) but zero-copy device selection (#941) still open. Treat as "later"; ORT-Web WebGPU EP or our own WGSL OIDN port is the shipping path.
- **Texture compression**: BC 87%, ASTC 47%, ETC2 47% (mutually exclusive in practice: desktop = BC, mobile/Apple Silicon = ASTC (+ETC2 on mobile)). 3D BC/ASTC shipped Chrome 139. Texture compression "unaligned" in 155-156.
- Compat mode (GLES 3.1 on Android/ChromeOS) origin trial Chrome 139, GLES 3.1 in 146. It lacks storage-texture/compute guarantees, not a path-tracing target.

## 2. Sources

| name | link | year | status/license | why |
|---|---|---|---|---|
| gpuweb bindless proposal | https://github.com/gpuweb/gpuweb/blob/main/proposals/bindless.md | 2025 draft, active to Nov 2026 | W3C draft | `GPUResourceTable` API, feature names, limits |
| gpuweb issue #380 | https://github.com/gpuweb/gpuweb/issues/380 | 2019-2026 | - | bindless tracking; sub-issues #5471 default types, #5517 implementer/user experience, #11059 min buffer binding size, #11097 metadata builtin, #6335 storage barriers |
| Dawn resource table extension doc | https://dawn.googlesource.com/dawn/+/main/docs/tint/extensions/chromium_experimental_resource_table.md | 2026 | BSD-3 (Dawn) | WGSL syntax; SPIR-V needs `SPV_EXT_descriptor_indexing`+`RuntimeDescriptorArray`; HLSL requires DXC (not FXC); MSL supported; no GLSL |
| Dawn dynamic binding doc (old prototype) | https://dawn.googlesource.com/dawn/+/b38c8a7674a3d8f55be523552b84df504d675447/docs/tint/extensions/chromium_experimental_dynamic_binding.md | 2025 | - | superseded (`resource_binding`, `getBinding`) |
| toji: Experimenting with Bindless Textures in WebGPU | https://toji.dev/2026/10/06/webgpu-bindless.html | 2026-10 | blog | how to enable, usage, caveats (his snippets contain errors) |
| Chrome "What's new in WebGPU" 113-156 | https://developer.chrome.com/blog/new-in-webgpu-149-150 (index), -151-152, -153-154 | 2023-2026 | - | shipped-feature timeline |
| Chrome WebGPU developer features | https://developer.chrome.com/docs/web-platform/webgpu/developer-features | 2025 | - | `enable-webgpu-developer-features`: unquantised timestamps, `adapter.info.memoryHeaps`, driver/backend info, `strictMath`, `isZeroCopy` |
| gpuweb subgroup matrix proposal | https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroup-matrix.md | 2025 draft | - | cooperative matrix API |
| gpuweb acceleration structure PR #11054 | https://github.com/gpuweb/gpuweb/pull/11054 | 2026-10 | open | HW RT proposal; Vecvec(wgpu), JMS55(Bevy), ARM reviewers |
| web3dsurvey | https://web3dsurvey.com/webgpu | live | - | feature / limit adoption |
| W3C WebML minutes 2026-09-10 | https://www.w3.org/2026/09/10-webmachinelearning-minutes.html | 2026 | - | WebNN-WebGPU interop merged |
| blink-dev WebNN intent to experiment | https://groups.google.com/a/chromium.org/g/blink-dev/c/5CWKSChYo98 | 2026 | - | OT schedule, Android excluded |
| Dispatch overhead study | https://arxiv.org/html/2604.02344 | 2026 | open | **WITHDRAWN by author; superseded by arXiv 2608.08730** (v3 2026-10-26). 24-36 us Vulkan, 32-71 us Metal sequential dispatch; Chrome RTX5090/Linux 32.8 us, RTX 2000/D3D12 58.7, Intel/D3D12 66.5, Safari M2 31.7, Firefox ~1040 us (v1 table) [V-v1] |

## 3. Findings

### 3.1 Bindless (resource table) - Prototype now, design for it

What it is: `device.createResourceTable({size})`; `table.insert(resource)` returns index (or `update(slot, r)`, `remove(slot)`); bind with `beginRenderPass({resourceTable})` / `computePass.setResourceTable(table)`; pipeline layout `usesResourceTable` (implicit for default layouts that use the builtins). `setResourceUsage` for storage-texture usage state. New resources start `readonly`. Slot reuse tracked via queue submit indices.
WGSL: `getResource<texture_2d<f32>>(idx)`, `<texture_cube<f32>>`, `<sampler>`; OOB or type mismatch returns a default resource (no trap). `index` i32/u32, **no uniformity requirement** (important: divergent material ids per-lane are legal, which is exactly path-tracing hit shading).
Features: `sampling-resource-table` (textures+samplers only), `heterogeneous-resource-table` (adds storage textures; spec says "any resource type except uniform buffers"; storage and texel buffers "planned later"). In Chrome today: `chromium-experimental-sampling-resource-table`; heterogeneous not implemented.
Enable: `chrome://flags/#enable-unsafe-webgpu` ("Unsafe WebGPU Support"), restart; request device with the feature. Platforms: Windows, Linux, Android; macOS not yet. Backends: Vulkan (`VK_EXT_descriptor_heap` preferred; fallback `VK_EXT_descriptor_indexing`), D3D12 via DXC SM6.6 dynamic resources (FXC unsupported), Metal argument buffers tier 2.
Limits: table max 65536 entries hardcoded (no adapter limit yet; may be added). Table size is count, not bytes.
Gotchas: aliasing relaxation (gpuweb #5611) is a dependency; the index still has to reach the shader via uniform/storage buffer/immediates (store `materialTextureBase` in the material buffer). Texture views inserted must stay alive; `remove` is deferred until in-flight work completes.

Design implications:
1. Materials store u32 slot indices (base color, normal, ORM, emissive...) per material; same struct in both paths, only the *meaning* of the index changes (resource-table slot vs. array layer).
2. Because geometry data cannot go in a resource table, vertex/index/attribute pools remain big storage buffers addressed by offsets (with suballocation). Binding count for the trace pipeline stays small (BVH, tris, materials, lights, accum, output) either way.
3. Bindless wins: mixed texture sizes/formats without padding to one layer size (toji: 2.6 GB array vs <100 MB bindless for his demo), no 256-layer limit pressure, incremental streaming by `update`ing a slot. Array-texture fallback wins: works everywhere WebGPU does, simple, gputex writes straight into layers (HOMEFIG `encodeMipChainIntoTexture`), uniform size per array so bucket by size class (e.g. 4 arrays: 512/1k/2k/4k, material has `(arrayId, layer)` -> needs a small `switch` over a fixed number of array bindings or a 2-level index).
4. Fallback ladder: (a) resource table, (b) N texture_2d_array bindings grouped by size class (<= ~4-8 arrays; layers up to maxTextureArrayLayers, guaranteed 256, commonly 2048), (c) single atlas (virtual-texture style indirection, needed only on mobile/limits).
5. Rating: **Prototype** (resource table path behind feature detect), **Adopt now** (array fallback, shared with gputex).

### 3.2 Subgroups - Adopt now, with no-subgroup fallback
Shipped timeline: experimental 128, expanded 129, inclusive scans 131, cleanup 133, **shipped 134**, `subgroup_id` 144, `subgroup_uniformity` 145, `subgroup-size-control` + `@subgroup_size` 151-152 (enables `subgroups` automatically). Detect: `adapter.features.has('subgroups')`; WGSL `enable subgroups;`; sizes in `adapter.info.subgroupMinSize/MaxSize`. caniuse shows Edge as supported from 144 and Firefox none through 156 (inconsistent with Chrome 134 dates; **[unverified]**). Safari/iOS essentially absent (1.2%/1.4%); Safari standards position is "no position" on `subgroup_uniformity`.
Uses for us: wave-coherent ray compaction/queue append (one atomic per subgroup), BVH traversal stack-sharing/voting, MIS/path-termination ballots, radix-sort histograms, downsampler/SPD reductions (our `downsampler`), denoiser convolutions. Subgroup size varies (Apple 32, NVIDIA 32, AMD 32/64, Intel 8-32) -> don't hardcode; use size-control only where a specific size is needed (it is a requirement flag, may reduce occupancy).
Fallback: workgroup memory + atomics versions of every subgroup op, selected via pipeline-override constant (`override HAS_SUBGROUPS`) or WGSL preprocess variants.

### 3.3 f16, immediates, timestamps, misc shipped
- `shader-f16` (Chrome 120): 94%. Use for denoiser/ML weights, packed normals/colors; keep traversal precision-critical math in f32. f16 on Linux NVIDIA arrives 155-156.
- **Immediates** (Chrome 149-150; WGSL `var<immediate>`, `pass.setImmediates()`; detect `navigator.gpu.wgslLanguageFeatures.has('immediate_address_space')`): use for per-dispatch frame index, bounce, queue offsets, material-slot base. Cuts bind-group updates and buffer writes. Rating: Adopt now (with uniform-buffer fallback).
- Timestamp queries: Chrome 121 in-pass timestamps shipped; `timestamp-query` 99%. Quantised to 100us unless developer-features flag (then ns). Use per-pass GPU timers in the bench harness; run benches with the flag. `chromium-experimental-timestamp-query-inside-passes` 0.16%.
- Dawn "use DXC on Windows" since Chrome 121; D3D12 compile time improvements 136; Metal improvements 130. FXC vs DXC: resource table needs DXC. Large megakernel WGSL still takes seconds to compile on D3D12 (see skill `webgpu-tsl-compile`); use `createComputePipelineAsync`, split kernels, `override` constants, avoid giant unrolled loops. Pipeline cache: browser-level only (Chrome 117 caches automatic-layout pipelines); no explicit pipeline cache API **[no evidence one exists]**.
- Other new: `linear_indexing` WGSL (147-148), `buffer_view` WGSL ext + `swizzle_assignment` (153-154; `buffer_view` may matter for typed views into storage buffers - read the spec before adopting), `texture_and_sampler_let` (146), primitive index (142), texture component swizzle (143), transient attachments (146, render-only), `core-features-and-limits` (139), `texture-formats-tier1` (read-write storage for more formats; JS name mapping fixed in 153-154).
- `BufferMapWriteExtendedUsages` (Dawn experimental) and experimental synchronous buffer mapping in workers (145): niche.

### 3.4 Chromium-experimental features (not for default path)
`chromium-experimental-multi-draw-indirect` (131; 0.12%; render only - irrelevant to compute tracing), `...-read-write-storage-texture` (0%; superseded by shipped RW storage textures in 124 for some formats and by tier1), `...-dp4a` (0%; shipped DP4a builtins 123), `...-timestamp-query-inside-passes` (0.16%), `...-subgroup-matrix` (0.04%), `...-sampling-resource-table` (0.14%). Not found / do not exist in WebGPU: 64-bit atomics **(CORRECTED 2026-11-13: a limited proposal exists and is merged into the spec repo: `atomic_vec2u_min_max`, `atomic<vec2u>` with `atomicStoreMin/Max` only, no return value, storage buffers only; needs Vulkan int64 atomics, Metal Apple9 / M3+ class, D3D12 SM6.6; gpuweb PR #5610, issue #5071; not shipped in a browser as far as found; general 64-bit atomics (add/CAS/exchange) are not proposed)**, texture atomics, buffer device address (no; by design not exposed), unrestricted indirect (indirect dispatch is validated; dispatch dims limited to `maxComputeWorkgroupsPerDimension` = 65535 default - use 2D/3D wrapping for >65535*WG), texel buffers (planned in bindless doc, not available), multiple queues (single queue only), PLS (not found **[unverified]**). Rating for all: Reference only except subgroup matrix (Prototype for denoiser).

### 3.5 Subgroup matrix / neural denoiser
**Update 2026-11-14 [V]:** still `chromium-experimental-subgroup-matrix` (Draft proposal). Since this note: bf16/fp8 limited-use types and D3D12 (SM 6.10, preview) mappings merged; open investigations on element access (#11077), result→left conversion (#11079), workgroup matrices (#11078), `@unroll` (#11073). Shipped around it: `subgroup_id` (Chrome 144), `subgroup_uniformity` (145, verified on 153), `@subgroup_size` (151–152). Details and our asks: `docs/design/subgroup-matrix-feedback.md`.

Gpuweb draft: types `subgroup_matrix_left/right/result<T,..>`, `subgroupMatrixLoad/Store/MultiplyAccumulate`; compute only; subgroup-uniform. Config list via `GPUAdapterInfo.subgroupMatrixConfigs` (componentType, resultComponentType, M,N,K, min/maxSubgroupSize). Metal 3: 8x8x8 f16/f32/bf16; Vulkan: from KHR coop matrix props (NVIDIA/AMD/Intel give 16x16x16 f16); D3D12: needs `D3D12_LINEAR_ALGEBRA_TIER_1_0`/SM6.10, not common yet. Expect 3-10x over plain f16 FMA convs on supported HW (**vendor-typical, not measured by me**). Plan: denoiser conv kernels with 3 variants (subgroup-matrix f16 / plain f16 / f32), pick by feature. Rating: Prototype (our Denoiser repo owns it; coordinate).

### 3.6 Hardware ray tracing in WebGPU
Status: proposals only (see PR #11054). Objects: `GPUAccelerationStructure` (TLAS/BLAS, single create call + type), build descriptors split by level, instance descriptor (4x4 transform, force opaque flags), feature `acceleration-structure`, WGSL ray query separate (#11056). Backends referenced: VK_KHR_acceleration_structure, DXR, Metal. Discussion notes: wgpu TLAS build CPU overhead is severe (Bevy workaround), compaction/batching open. No browser has it behind a flag as far as I found. Rating: Reference only. Keep an `IBvhBackend` seam so a HW path can replace the software traversal later; native comparison belongs on Windows (wgpu/Vulkan RT) per brief.

### 3.7 WebNN
- Status: spec at CR Draft; Chrome/Edge origin trial (M146, 11 Feb 2026 beta) with Android excluded, schedule slipped to M147-149, a later message disabled the trial after release-blocking issues **[current state unverified; check chromestatus]**. Safari/WebKit: no shipping evidence found; Firefox: none found.
- Backends (secondary source): Windows - DirectML / OS ONNX Runtime (Win11 24H2+); macOS - Core ML; LiteRT/XNNPACK (TFLite) on others; OpenVINO via ORT EP. NPU/GPU/CPU selection via `deviceType`/power hints.
- Interop: MLTensor with `exportToGPU` / `createExportableTensor` merged into spec (WebML WG, 2026-09-10); open: zero-copy adapter selection (`createContext({interopDevice})`, #941), custom usage flags (#954, v2). Until that is implemented in Chromium stable, ML output goes through a copy or CPU readback.
- Per-frame U-Net: no numbers found. OIDN-small-class nets at 1080p are tens of GFLOPs per frame; per-frame viability depends on NPU/GPU, and WebNN has no way to fuse our own albedo/normal prefilter kernels (custom ops not expressible - noted in WG minutes), so a hybrid costs extra buffer hops. For real-time path tracing at 1-4 spp, our own WGSL (f16 + subgroup matrix) denoiser is the controllable route.
- ORT-Web: WebGPU EP + WebNN EP + I/O binding (GPU buffers in/out) + graph capture. No published ORT-WebGPU-vs-WebNN U-Net benchmark found. Dispatch overhead 24-58us per dispatch (native study) means a 40-layer U-Net spends ~1-2 ms in dispatch alone; WebGPU in browsers adds more. Rating: WebNN **Later** (feature-detect, benchmark); ORT-Web WebGPU **Prototype** only as comparison baseline.

### 3.8 Texture compression and streaming
- Features: `texture-compression-bc` (87%), `-bc-sliced-3d` (86%; 3D support also shipped for BC and ASTC in 139), `texture-compression-astc` (47%; also `-astc-sliced-3d`), `texture-compression-etc2` (47%). Desktop D3D/Vulkan: BC. Apple Silicon/iOS/Android: ASTC (Apple has no BC on iOS; macOS Apple Silicon supports both). Detect, pick format per adapter; never assume.
- gputex approach (homefig fork, `library/src/Encoder.ts`, `encodeMipChainIntoTexture(src, target, {layer})`): GPU-encodes BC1/BC5/BC7/ASTC4x4/ETC2 at load, then `copyBufferToTexture` blocks straight into one layer of a caller-owned array; no readback. This fits our streaming: material maps in a few arrays, layer per material. Block rows narrower than 256 B need per-row copies (alignment). Writing compressed blocks via compute -> buffer -> `copyBufferToTexture` is the only way (no storage-texture writes to BC formats), so streaming costs one copy per mip.
- KTX2/Basis Universal: transcode (UASTC -> BC7/ASTC) on CPU worker or GPU; deliver pre-transcoded .ktx2 where possible. Keep gputex GPU encode for runtime-generated content.
- Sparse/virtual texturing: **no sparse/tiled resources in WebGPU**. Emulate with an indirection (page table) texture + physical tile cache array; costs an extra dependent fetch per sample and needs feedback; with ray tracing, derivative-free LOD (ray cones) must be computed manually anyway. Rating: Later; start with whole-texture streaming into arrays (with bindless `update` or layer rewrite) and mip-tail residency.
- Memory budgets: no API for VRAM budget; `adapter.info.memoryHeaps` only with developer-features flag. Limits: maxBufferSize >= 2 GB on 77% of reports (>= 1.25 GB 86%); **maxStorageBufferBindingSize >= 2 GiB only 16%**, >= 1.6 GB 78%, >= 1.25 GB 79%, >= 256 MB 98%, 128 MB (spec default) 100%. Defaults requested unless asked: request `adapter.limits.max*` explicitly. Implication: the BVH node/triangle pools must be paged into multiple storage buffer bindings (e.g. chunks of <= 128-256 MB each, address = (chunk,offset)) for portability, or require the larger-limit tier. `maxStorageBuffersPerShaderStage` default 8 (adapters usually higher); count bindings carefully; resource table avoids spending bindings on textures (default `maxSampledTexturesPerShaderStage` 16).

### 3.9 Workers, OffscreenCanvas, compat
WebGPU in dedicated/service/shared workers (service/shared since 124) and OffscreenCanvas: yes. Use a worker for asset decode/BVH WASM build and (optionally) a render worker so the main thread never stalls. Device can be shared per worker only (no cross-thread GPUDevice sharing). Compatibility mode (GLES 3.1 / D3D11): origin trial 139; no guaranteed storage textures/compute limits -> do not target.

### 3.10 Mobile
- Android Chrome (since 121): subgroups 84% of reports, ASTC/ETC2, f16 common, but resource table works on "most" Android (heterogeneous not on some devices), thermal throttling, 2D-array layer limits and storage binding limits are lower (often 128-256 MB storage binding). Mali/Adreno/Immortalis: drivers vary. Realistic tier: 0.5-1 spp/frame at ~540p-720p internal with 1-2 bounces, ReSTIR/probe-assisted or hybrid (raster primary + traced reflections/shadows/AO), upscaled (our `upscaler`) and denoised; full progressive path tracing as "photo mode" only. **[estimate, not measured]**
- iOS/iPadOS Safari 26+: WebGPU on Metal, but subgroups ~1.4%, BC unsupported (ASTC), resource table not available (no Safari bindless evidence). A17/M-class iPads have Apple7+ GPUs with HW-accelerated ray tracing in Metal that WebGPU cannot reach. Same realistic tier as Android, with the baseline Tier 0/1 code path; Safari has smaller dispatch/workgroup limits in places **[unverified]**.
- Safari/WebKit feature list I could not retrieve (WebKit "Features for Safari 26.0" post); verify directly on a device in the bench harness and record `adapter.features` + `limits` dumps.
- Firefox: WebGPU on Windows since 141, macOS Apple Silicon (Tahoe) 145; wgpu-based; subgroups ~0.1% (unshipped), no resource table, no immediates info found **[verify]**.

## 4. Capability tier matrix

Feature strings: those on `adapter.features` unless marked WGSL.

| Tier | Name | Required (exact strings) | Trace-core behaviour |
|---|---|---|---|
| 0 | Baseline WebGPU | core only; `maxStorageBufferBindingSize` >= 128 MB, `maxComputeWorkgroupStorageSize` 16 KB, `maxStorageBuffersPerShaderStage` >= 8; `timestamp-query` optional | Workgroup-atomics queues, f32, pooled geometry in chunked storage buffers, texture_2d_array fallback, uniform buffer for per-pass constants. Works on Safari/iOS/Firefox. |
| 1 | Compute-friendly | Tier 0 + `shader-f16` + `timestamp-query` + `immediates` (WGSL lang feature `immediate_address_space`) + a compressed format family (`texture-compression-bc` or `-astc`) + raised limits (storage binding >= 1 GB, workgroup size 256) | f16 denoise/prefilter, immediates for per-dispatch constants, compressed arrays via gputex, larger buffer chunks. |
| 2 | Subgroups | Tier 1 + `subgroups` (WGSL `enable subgroups;`, `subgroup_uniformity` language feature desirable) optionally `subgroup-size-control` | Wave-aggregated queue appends, ballot-driven compaction, subgroup traversal stack sharing/voting, subgroup reductions in downsampler. |
| 3 | Bindless + ML | Tier 2 + `chromium-experimental-sampling-resource-table` (later `sampling-resource-table`, `heterogeneous-resource-table`) + optionally `chromium-experimental-subgroup-matrix` (later `subgroup-matrix`) + optionally developer features flag for ns timers | Bindless texture sampling in hit shading (any material from any lane), per-slot streaming; coop-matrix denoiser conv. |
| 3+ (watch) | Future | `acceleration-structure` + WGSL ray query; WebNN `MLTensor` export; heterogeneous tables with storage buffers/textures; texel buffers; 64-bit atomics (none proposed) | Not implemented in any browser we can test. |

Orthogonal axes (do not fold into the single tier number): texture format family (BC vs ASTC), maxStorageBufferBindingSize class, `maxTextureArrayLayers`, mobile/thermal class, WebNN availability.

## 5. For the spec (recommendations)

1. **Capability object, not a tier enum.** `detectCaps(adapter)` returns `{features:{subgroups, subgroupSizeControl, f16, timestamp, immediates, resourceTable:'none'|'sampling'|'heterogeneous', subgroupMatrix, accelerationStructure}, formats:{bc, astc, etc2, bcSliced3d}, limits:{storageBindingBytes, maxBufferBytes, storageBuffersPerStage, sampledTexPerStage, maxTextureArrayLayers, computeInvocations, workgroupStorage}, subgroupSize:{min,max}, tier:0..3}`. Tier is a derived convenience label; pipelines select by individual flags.
2. Request the device with *every* feature the adapter offers that we use (`requiredFeatures` only for features that exist: filter first) and `requiredLimits` = adapter's max for the limits we care about, with a cap for sanity. Also try-requests for experimental names (`chromium-experimental-sampling-resource-table`, then `sampling-resource-table`; `chromium-experimental-subgroup-matrix`, then `subgroup-matrix`) so the code survives the rename.
3. **Shader variants by `override` constants and a small WGSL preprocessor** (`#if HAS_SUBGROUPS`, `#if BINDLESS`), compiled lazily with `createComputePipelineAsync`, cached in a map keyed by (kernel, caps hash). Keep generated code readable (DX).
4. **One material struct, two backends**: `TextureProvider` interface with `ResourceTableProvider` and `ArrayProvider` (and `AtlasProvider`); material buffer holds u32 slot/(array,layer) pair; the shader helper `sampleMaterialTexture(slot, uv, lod)` is the only place that differs. gputex writes into arrays; with a resource table it inserts views of the same arrays or individual textures.
5. Geometry never depends on bindless; chunked storage-buffer pool with (chunk, offset) addressing when `maxStorageBufferBindingSize` < pool size.
6. Indirect-dispatch GPU-driven loop (wavefront): queue counters in storage buffer, `dispatchWorkgroupsIndirect`; respect 65535 per-dimension (use 2D dispatch folding). Parameter updates via immediates (T1+) or a tiny uniform with `writeBuffer`.
7. Timers: GPU timestamps per pass behind `timestamp-query`; log whether quantisation is active.
8. Flag bindless as **experimental** in docs: Chrome flag `chrome://flags/#enable-unsafe-webgpu`, document the fallback is the supported default.
9. Ship a `caps.html` diagnostics page that dumps adapter info, features, limits, WGSL language features, and a micro-benchmark per tier path; collect cross-device data (esp. Safari/iOS/Android).
10. Neural denoiser: abstraction `DenoiseBackend = 'wgsl' | 'ort-webgpu' | 'webnn'`; default `wgsl` (our Denoiser). Never depend on WebNN.

## 6. Bench / comparison ideas

- Resource table vs texture-array vs atlas hit-shading cost: scene with 1k+ textures of varied size; measure ms/frame of shading kernel, divergence across lanes (random material per pixel), memory.
- Subgroup vs workgroup-atomic queue append; compaction throughput (Mrays/s) at 1M/4M rays per wave.
- Immediates vs uniform buffer per dispatch at 50-200 dispatches/frame (CPU encode time, GPU idle gaps).
- D3D12 shader compile time with/without `override` specialisation, DXC; cold vs warm cache; `createComputePipelineAsync` stall.
- f16 vs f32 denoise kernel; subgroup-matrix vs scalar f16 conv (3x3, 64ch) on Apple/NVIDIA/AMD/Intel.
- ORT-Web WebGPU EP vs WebNN EP vs our WGSL on an OIDN-sized U-Net at 1080p; record `exportToGPU` copy cost.
- Mobile: Pixel/Adreno, Mali, iPhone/iPad Safari: rays/s at 720p, thermal after 60 s.
- Compare against mutracer and three-gpu-pathtracer on identical scenes using caps dump to ensure same feature level.

## 7. Needs Windows / native hardware

- DXC vs FXC (Chrome flags/Dawn toggles) compile times; resource table on D3D12 (SM6.6 dynamic resources); Intel/AMD/NVIDIA subgroup-size variation (Intel 8-32, AMD 32/64); coop-matrix on NVIDIA/AMD (Vulkan KHR coop matrix) and D3D12 linalg tier; `maxStorageBufferBindingSize` by vendor.
- WebNN with DirectML/OS-ORT backend and NPU; Windows ML.
- Native comparators: OIDN/OptiX, Vulkan RT / DXR path tracer for HW-RT headroom numbers; wgpu native ray query (Vecvec's work) as proxy for what a WebGPU HW RT would give.
- Chrome with developer features for ns timestamps; PIX/Nsight captures.
- Apple devices for Metal behaviour (macOS resource table pending; iOS Safari features).

## 8. Open questions

- Exact current Chrome status/flag of resource table on macOS and in stable vs canary; whether `sampling-resource-table` (non-experimental) name has landed in Chrome 156+. I could not read chromestatus.
- Does Safari 26.x expose subgroups / f16 / which texture compression? (WebKit blog 26.0 features post not retrieved.)
- WebNN OT state post-M147 and interop shipping version; Edge differences.
- `buffer_view` WGSL ext (153-154): can it let us view storage buffers as typed structs with fewer casts? Read spec.
- Whether heterogeneous tables will include storage buffers before ray tracing HW proposal progresses (would allow bindless geometry buffers).
- Pipeline cache control and compile-time numbers on Chrome stable for 5-10k line WGSL kernels.
- Real-world maxStorageBufferBindingSize on Safari/Apple (survey rows per platform not pulled).
- Whether `subgroup_matrix` configs on Metal 3 (8x8x8) are fast enough vs f16 FMA for 3x3 convs.
