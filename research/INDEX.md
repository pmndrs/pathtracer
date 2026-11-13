# Research index and synthesis

Research pass 1, 2026-11-13. Ten parallel passes. Each has a notes file in [notes/](notes/), written to the format in
[BRIEF.md](BRIEF.md). This page shows where the passes agree and where they conflict. It also lists what is still
unverified and the license register. The decisions drawn from it are in [../spec/SPEC-draft.md](../spec/SPEC-draft.md).

| # | Area | File | Main conclusion |
|---|---|---|---|
| 01 | Our sibling packages | [01-local-ecosystem.md](notes/01-local-ecosystem.md) | 17 separate ray walks to absorb; reflections' frame contract + bvh's ray/hit structs become the shared contract |
| 02 | Web ray/path tracers | [02-web-renderers.md](notes/02-web-renderers.md) | Competitors: mutracer, three-gpu-pathtracer (now WebGPU, wavefront), Rayzee. None ships ReSTIR or real temporal reuse |
| 03 | Native renderers | [03-native-renderers.md](notes/03-native-renderers.md) | pbrt-v4 is ground truth, cross-checked by Mitsuba 3; Cycles is the best wavefront engineering source; Bevy Solari is the only WGSL ReSTIR |
| 04 | Sampling and reuse | [04-sampling-reuse.md](notes/04-sampling-reuse.md) | Owen-Sobol for reference, STBN/FAST for real time; ReSTIR DI now, GI as a prototype, PT later |
| 05 | GPU architecture | [05-gpu-architecture.md](notes/05-gpu-architecture.md) | Two schedulers, one BVH; the wide-BVH (CWBVH/BVH4) experiment comes first; per-GPU Mrays/s targets |
| 06 | Web platform | [06-web-platform.md](notes/06-web-platform.md) | Bindless covers sampled textures only and is not on macOS yet; subgroups at 68%; no hardware RT or usable WebNN |
| 07 | Denoise and neural | [07-denoise-neural.md](notes/07-denoise-neural.md) | SVGF-class demodulated filter + our upscaler for real time; OIDN for preview/final; OIDN 3 not released |
| 08 | Real-time GI consumers | [08-realtime-gi.md](notes/08-realtime-gi.md) | Every GI method is a cache fed by few rays → layered API (VIS / HIT / SHADE1 / PATH) + shared hash cache and probe atlas |
| 09 | Materials and transport | [09-materials-transport.md](notes/09-materials-transport.md) | Packed OpenPBR-subset uber BSDF, VNDF sampling, NEE+MIS; carry vec4 radiance now, spectral in v2 |
| 10 | Benchmarks | [10-benchmarks.md](notes/10-benchmarks.md) | Compare in linear HDR; time-to-equal-quality is the headline metric; furnace tests gate merges; siblings supply ~70% of the harness |
| 11 | AMD Capsaicin | [11-capsaicin.md](notes/11-capsaicin.md) | MIT, D3D12 DXR only. Reference PT is opaque glTF-PBR only (no glass, f16 EXR dump), good as a Windows GPU cross-check, not a third reference; GI-1.2 hash-grid cache + world ReSTIR are portable ideas with a permissive licence |

Local material: ~50 shallow clones in `sources/` (5.5 GB, git-ignored, listed in [sources/MANIFEST.md](sources/MANIFEST.md))
and about 50 open-access PDFs in `papers/<area>/` (git-ignored).

---

## 1. Where the passes agree (firm enough to spec)

1. **Two execution models over one BVH** (02, 03, 05, 08).
   - An *inline* `trace()` callable from any compute shader, for utility consumers. GI, AO, shadows and reflections
     use few rays with trivial shading.
   - A *queued/wavefront* scheduler for the full path tracer, where shading is expensive and divergent.
   - This split is forced by Chrome's per-dispatch cost of 33–67 µs: affordable for a 10 ms+ path-tracing iteration,
     too expensive for a 0.2 ms probe update.
2. **The core starts at the hit.** `@pmndrs/bvh` owns build, refit, layout and the walk. This repo owns attribute
   interpolation, materials, lights, sampling, integrators and AOVs (01, bvh's own scope rule). Caches belong to
   `@pmndrs/gi` (see 3).
3. **Caches are first-class shared modules** (04, 08, 03):
   - a world-space spatial hash radiance cache (SHaRC / GI-1.0 / Split RC / path-space filtering / Solari world cache);
   - a probe atlas (DDGI / RC / Lumen-style).
   Both serve the GI tools and speed up the path tracer (cache termination, preview). **Decided 2026-11-13: both live
   in `@pmndrs/gi`**, which already has a world hash cache. This repo provides only hooks (L2 cache-lookup argument,
   L3 `onVertex`/`terminate`), so the dependency runs one way: gi → pathtracer.
4. **Sampler by mode** (04, 07, 02):
   - Owen-scrambled Sobol (Burley 2020) for progressive reference;
   - STBN/FAST blue-noise textures for the first one or two bounces in 1-spp real time, with a PCG hash beyond that;
   - one `Sampler` interface that hands out dimensions in pairs.
5. **The real-time reconstruction chain is ours.** Demodulated diffuse/specular AOVs → SVGF-class temporal filter →
   `@pmndrs/upscaler` (reactive mask from history length) → tonemap. OIDN (our WGSL port) handles preview and final
   frames (07, 01).
6. **The material system is a packed uber-BSDF table** (09, 02). It is an OpenPBR-subset in 96–128 B per material.
   MaterialX is baked to parameters in v1 and gets generated WGSL pattern functions in v2. WGSL has no function
   pointers, so bindless cannot dispatch arbitrary code.
7. **Capability flags, not a tier enum** (06, 05). `detectCaps()` reports individual features. The "tier" is a
   derived label. Pipelines are specialized with `override` constants plus a small WGSL preprocessor.
8. **Ground truth is pbrt-v4, cross-checked by Mitsuba 3** (03, 10). Both run on the Mac's CPU. Comparisons are done
   in linear HDR. Headline metric is time-to-equal-quality; Mrays/s is a per-kernel diagnostic only.
9. **Reuse the siblings' bench infrastructure** (10, 01): Chrome launcher, CSV schema, paired ratios, GPU lock,
   thermal validity, FLIP/scorecard, and CPU oracles for every kernel.

## 2. Where the passes conflict or are unsure (decide with measurements)

| Question | Evidence | Proposed resolution |
|---|---|---|
| Megakernel vs wavefront for the PBR tracer | Laine 2013: wavefront +36% to +221%, growing with shading cost. Dr.Jit: wavefront 2–7.7× slower. Utah 2026: wavefront +16%. 67.js spike: 2× slower when traversal-bound. three-gpu-pathtracer and Rayzee both chose wavefront | Build both behind one scheduler interface in the first milestone; decide per scene class from our own benchmarks |
| Wide BVH | CWBVH is 1.9–2.1× on incoherent rays (native); our quantized BVH2 is 0.93–0.97×; mutracer already uses tinybvh CWBVH; Rayzee uses BVH4 | First traversal experiment. Gate: ≥1.3× on both RTX 3060L and M5 (lives in the bvh repo) |
| Blue-noise samples into OIDN | Not documented anywhere | Bench it in our Denoiser: STBN vs Owen-Sobol at 1–4 spp |
| How to handle motion | mutracer and three-gpu-pathtracer: low-res preview + per-frame OIDN + upscale, no history. Rayzee: temporal history. Nobody does ReSTIR | Ship the preview mode as the fallback. The differentiator is real temporal reuse (MV-driven history + ReSTIR DI) with a single "temporal strictness" control |
| Material id width | bvh packs it in 16 bits (max 65534) | Treat the bvh id as an index into our material table; widen in bvh layout v3 if real scenes need it |
| Bindless | Sampling-only; not on macOS (our main dev machine); 0.14% adoption. mutracer already ships it with an atlas fallback | Build against a `TextureProvider` abstraction. gputex arrays are the supported default; the resource table is an opt-in Tier 3 path, developed on Windows |

## 3. Corrections made during synthesis

- `papers/sampling/arxiv-ears.pdf` was an unrelated quantum-annealing paper. It is deleted and note 04 is fixed; the
  EARS PDF is still needed.
- Two duplicate PDFs were removed (NRC, Lumen).
- **Clean room.** The native pass had cloned NVIDIA NRD, and Rayzee's clone included an NRD port. Both were removed
  under the Denoiser clean-room rules, with a note in the MANIFEST. No note quotes NRD code. The brief now carries the
  rule.

## 4. Verification backlog (before numbers enter a final spec)

- Numbers tagged `[M]` (from memory), `[S]` (search snippet), `[E]` (estimate) or "unverified" in notes 04, 05, 08,
  09 and 10. The most important:
  - per-GPU Mrays/s extrapolations (05 §8);
  - ReSTIR GI cost and DDGI ray counts (08);
  - exact Chrome flag spellings for precise timestamps (10);
  - 64-bit atomics absent from WGSL (08 says [M]; 06 implies none are proposed).
- Note 09 had no web sweep for 2024–26 work, and six of its paper downloads failed: Wilkie hero wavelength, Belcour
  layering and thin film, Zeltner LTC sheen, Prague sky, Wächter-Binder, Kaplanyan regularization.
- Platform status not yet confirmed:
  - WebNN origin trial state;
  - Safari 26 feature list;
  - Chrome macOS bindless timing;
  - whether `sampling-resource-table` has dropped its `chromium-experimental-` prefix.
- OIDN 3: release date, weights and aux inputs. So far this comes only from press coverage and conference recaps.
- License terms of the NVIDIA RTX SDKs License: can we port ideas or snippets from RTXPT, RTXDI, RTXGI or SHaRC?
- Three web repos have no license file: `relight`, `wtrace`, `webgpu-raytracer`. Read only.

## 5. License register (what we may port)

| Status | Repos |
|---|---|
| Port with notice (MIT/BSD/Apache/zlib) | pbrt-v4, Mitsuba 3/Dr.Jit, Cycles, tinybvh, Embree, OIDN, OpenPGL, HIPRT/Orochi, Bevy Solari, kajiya, Tungsten, appleseed, LuxCore, lightmetrica, Capsaicin, mutracer (**cite and compare only, by choice**), three-gpu-pathtracer, three-mesh-bvh, Rayzee (minus its NRD port), strahl, Web-RTRT, ShadyTracer |
| Ideas only | NVIDIA RTX SDKs License: RTXPT, RTXDI, RTXGI, SHaRC. Falcor is "BSD-3-like" from NVIDIA: read the exact text before porting |
| Off limits | NRD and every NRD port; `~/Developer/nrd-quarantine`; rindr (PolyForm Noncommercial) |
| Unknown | relight, wtrace, webgpu-raytracer (no license file) |

## 6. What has to run on the Windows machine

- **Native ground truth on GPU:** pbrt-v4 `--gpu` (OptiX), Mitsuba `cuda_ad_*`, Cycles OptiX/HIP.
- **Real-time native yardsticks:** RTXPT, Falcor, RTXDI, Capsaicin GI-1.2 (D3D12 DXR), HIPRT.
- **Native OIDN on GPU** (CUDA/HIP).
- **Web timing on NVIDIA and AMD under D3D12/Vulkan:**
  - first-frame and pipeline compile cost (DXC);
  - bindless resource-table path (not on macOS);
  - subgroup sizes 32/64.
- **Measurement setup:** clock locking and power measurement; RenderDoc/PIX/Nsight captures.
- **Gaps in the siblings:** nothing has ever run on AMD, and the Denoiser and upscaler are validated only on Apple.
  Every sibling's pending Windows runbook (bvh defaults, downsampler parity, upscaler audit) feeds the same machine.
