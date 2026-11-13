| name | url | commit | license | why |
|---|---|---|---|---|
| native/appleseed | https://github.com/appleseedhq/appleseed | c4ba828 | MIT | production renderer |
| native/bevy-solari | https://github.com/bevyengine/bevy | 493cddd | MIT/Apache-2.0 | Bevy hardware RT GI (sparse crates/bevy_solari) |
| capsaicin (at `research/sources/capsaicin`, not under native/) | https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin | 914b915 (v1.3, 2025-11-18) | MIT (AMD); vendored `third_party/FidelityFX` MIT; `third_party/gfx` submodule NOT cloned; test media separate repo, licence unchecked | D3D12/DXR framework: reference PT, GI-1.2 (hash-grid cache, screen probes, world ReSTIR). No NVIDIA/NRD content found. Read only, never build; see notes/11-capsaicin.md |
| native/cycles | https://github.com/blender/cycles | 6abef8c | Apache-2.0 | GPU integrator kernels, sorting, light tree (standalone repo) |
| native/drjit | https://github.com/mitsuba-renderer/drjit | 847afbf | BSD-3 | JIT megakernel backend |
| native/embree | https://github.com/RenderKit/embree | 3b7e4ee | Apache-2.0 | CPU BVH reference |
| native/falcor | https://github.com/NVIDIAGameWorks/Falcor | 759aad0 | NVIDIA BSD-3-like | NVIDIA research framework |
| native/hiprt | https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT | 0629d08 | MIT | AMD RT lib |
| native/kajiya | https://github.com/EmbarkStudios/kajiya | 9fdec0f | Apache-2.0/MIT | Rust/Vulkan GI renderer |
| native/lightmetrica | https://github.com/lightmetrica/lightmetrica-v3 | 70601db | MIT | research renderer |
| native/luxcore | https://github.com/LuxCoreRender/LuxCore | eadd859 | Apache-2.0 | hybrid renderer |
| native/mitsuba3 | https://github.com/mitsuba-renderer/mitsuba3 | aa01500 | BSD-3 | reference renderer, spectral, Dr.Jit |
| ~~native/nrd~~ REMOVED 2026-11-13 (Denoiser clean-room rules, ~/Developer/Denoiser/docs/research/rt-denoiser/RULES.md) — do not re-clone | https://github.com/NVIDIA-RTX/NRD | 90dea8b | NVIDIA RTX SDKs License (proprietary, redistributable w/ terms) | real-time denoisers |
| native/oidn | https://github.com/RenderKit/oidn | 6602ee2 | Apache-2.0 | denoiser networks |
| native/openpgl | https://github.com/RenderKit/openpgl | 6a759d0 | Apache-2.0 | path guiding |
| native/orochi | https://github.com/GPUOpen-LibrariesAndSDKs/Orochi | 48c8ce5 | MIT | HIP/CUDA loader |
| native/pbrt-v4 | https://github.com/mmp/pbrt-v4 | b0bdf46 | Apache-2.0 | quality reference + GPU wavefront (src/pbrt/wavefront, gpu) |
| native/rtxdi | https://github.com/NVIDIA-RTX/RTXDI | fdbd9b3 | NVIDIA RTX SDKs License | ReSTIR DI/GI |
| native/rtxgi | https://github.com/NVIDIA-RTX/RTXGI | 04372d7 | NVIDIA RTX SDKs License | DDGI/SHaRC |
| native/rtxpt | https://github.com/NVIDIA-RTX/RTXPT | 303b408 | NVIDIA RTX SDKs License | RTX path tracing SDK reference |
| native/sharc | https://github.com/NVIDIA-RTX/SHARC | 4e21b58 | NVIDIA RTX SDKs License | spatial hash radiance cache |
| native/tinybvh | https://github.com/jbikker/tinybvh | c9be310 | MIT | our bvh upstream demos |
| native/tungsten | https://github.com/tunabrain/tungsten | 88ea020 | BSD-like (Bitterli) | Bitterli renderer, scene format |
| web/mutracer | https://github.com/pissang/mutracer | 06088ce | MIT (note: some chunks adapted from MIT ray-tracing-renderer/GLSL-PathTracer; see LICENSES/) | near-competitor WebGPU PT; read/cite/benchmark only, do not copy |
| web/three-gpu-pathtracer | https://github.com/gkjohnson/three-gpu-pathtracer | 4a32415 | MIT | upstream WebGPUPathTracer (wavefront+megakernel), v0.0.28+ |
| web/three-mesh-bvh | https://github.com/gkjohnson/three-mesh-bvh | 65d069f | MIT | upstream BVH + webgpu BVHComputeData (TLAS+BLAS, BVH2) |
| web/rayzee-renderer | https://github.com/atul-mourya/rayzee-renderer | afdfd0d | MIT | strongest wavefront web PT besides ours: 4-wide BVH, OIDN w/ temporal history, NRD port, bench harness |
| web/strahl | https://github.com/StuckiSimon/strahl | 062d25b | MIT | WebGPU OpenPBR PT, ACM Web3D short paper + thesis |
| web/oidn-webgpu | https://github.com/pissang/oidn-webgpu | aee3d60 | MIT | OIDN on WebGPU (mutracer's denoiser); compare with our Denoiser |
| web/hush | https://github.com/kanaworksai-hue/hush | bbb74c0 | MIT | 1spp real-time denoiser: SVGF + learned filter params + TAA |
| web/ShadyTracer | https://github.com/lampmaker/ShadyTracer | 40c046d | MIT | SDF/analytic wavefront PT, photon map caustics, extensive perf notes (docs/PERFORMANCE.md) |
| web/rindr | https://github.com/schmarante/rindr | d2a69ae | PolyForm Noncommercial 1.0.0 - READ ONLY, FLAG | viewer built on three-gpu-pathtracer + oidn |
| web/ThreeRT | https://github.com/meshy-dev/ThreeRT | 53346a0 | MIT | Rust+wgpu HW ray-query PT behind three.js API (native/Metal clients, not browser) |
| web/webgpu-ray-tracing | https://github.com/willwaltz/webgpu-ray-tracing | 8475692 | BSD-3-Clause | Chromium/Dawn patches adding HW ray query + acceleration structures (Vulkan) |
| web/web-rayquery | https://github.com/ocularmagic/web-rayquery | 9580356 | Apache-2.0 | Firefox/wgpu ray-query prototype (DX12/Vulkan), bindless etc. |
| web/Web-RTRT | https://github.com/C-none/Web-RTRT | 549613b | MIT | 2024 WebGPU ReSTIR DI/RIS + denoise + SR (large repo, assets) |
| web/gpu-renderer | https://github.com/tinypoly/gpu-renderer | a18e8ff | MIT | three.js scene WebGPU PT (tinypoly) |
| web/webgpu-doom-pathtracer | https://github.com/JamesRandall/webgpu-doom-pathtracer | 85afcdc | MIT | small real-time PT with reprojection + a-trous (temporal reference) |
| web/solstice | https://github.com/SamG-Coder/solstice | c2a081b | MIT | Rust/WebGPU Nanite+Lumen-style GI in browser (not a path tracer) |
| web/SSWorld | https://github.com/DataArche/SSWorld | 8a631c8 | Apache-2.0 | WebGPU software RT GI engine + MCP (low relevance) |
| web/relight | https://github.com/harri665/relight | 173cdec | none found | neural relighting of PT scenes in browser (EGSR 2026 reimpl) |
| web/lumina-wgpu | https://github.com/opanai404/lumina-wgpu | d19fbe5 | MIT | small MIS + temporal PT with CPU oracle |
| web/moonlight | https://github.com/ivanjermakov/moonlight | cd9ccbb | MIT | small WebGPU PT (large repo) |
| web/rayfinder | https://github.com/Nelarius/rayfinder | 634128c | MIT | Dawn-native WebGPU PT, referencePT-based |
| web/wtrace | https://github.com/alpcihan/wtrace | c5c1b7c | none found | small realtime WebGPU PT lib (2024) |
| web/webgpu-raytracer | https://github.com/gnikoloff/webgpu-raytracer | 42cbf18 | none found | popular tutorial-grade WebGPU PT |
| web/THREE.js-PathTracing-Renderer | https://github.com/erichlof/THREE.js-PathTracing-Renderer | f8ded11 | CC0 | Erich Loftis WebGL2 PT (real-time, BVH-free analytic + BVH demos); history |
| web/Babylon.js-PathTracing-Renderer | https://github.com/erichlof/Babylon.js-PathTracing-Renderer | 840054c | CC0 | Loftis Babylon port (2023, stale) |


**Clean-room note (2026-11-13):** NRD code and third-party NRD ports are off limits (Denoiser RULES.md). Removed `native/nrd` and Rayzee's NRD port files (`rayzee/src/viewer/denoise/NRD.js`, `docs/design/NRD_DENOISER.md`, `tests/gpu/nrdDemodulation.test.js`) from the local clones. NVIDIA RTX SDK repos (RTXPT, RTXDI, RTXGI, SHaRC) remain: ideas only, no code transcription, until their license terms are reviewed.
