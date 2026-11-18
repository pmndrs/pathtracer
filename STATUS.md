# Status

_Last updated 2026-11-15._

**Phase:** P0 passed its gate on the Mac (the RTX 3060 half waits on Windows). P1 (trace core) met its gate: reflections runs
on it (P1.5, merged here; the reflections branch waits for reflections' own review). P2 (reference path tracer) passed the
Tier 1 class A gate (D23, all 7 scenes) and has OIDN on final frames (bench); left are the siblings' switch-over and the
layering default (D24 fix built, P2.32b, default pending the quiet pass). P3 (speed) is mostly built: a wavefront scheduler
beside the megakernel (per pixel equal; refill, samples in flight, early-out), alpha MASK in both, visibility-buffer
primaries, ray-class material LOD, scene-copy packing, cache hooks / counters / AOVs. Every new path is a setting; defaults
wait for a quiet rerun (the 2026-11-15 pass ran loaded after a reboot, ratios held) and the 3060 (W10 compile time, P3 gate).
P4 (real time) has its first piece: mode presets, a blue-noise sampler and per-frame denoiser inputs (demodulated diffuse / specular,
normal, viewZ, camera motion, roughness) on the megakernel, checked against closed forms (`docs/design/realtime.md`); the temporal filter,
upscaler and the wavefront version are next.

**Working now**
- **Bench runner.** Any bench page runs headless in Chrome on the Mac's GPU. Results go to CSV files, images to PFM.
  A/B runs alternate A B B A automatically.
- **A/B viewer.** Opens any two float images with swipe, flip, difference, error heat map and FLIP modes, a pixel
  inspector and an exposure control. There is also a grid view of a whole run.
- **Reference renders.**
  - Six small Tier 0 scenes plus "sponza-white": Sponza with plain grey walls, a sun and a sky.
  - Each is rendered by pbrt-v4 and by Mitsuba 3 on the Mac at 4096 samples per pixel. Sponza takes 24 min in
    Mitsuba and 57 min in pbrt.
  - The furnace tests hit their exact expected values in both renderers.
  - The two renderers agree to within their noise: relMSE 2e-8 to 6.4e-4 on Tier 0, 4.1e-3 on Sponza.
- **Competitors in the bench.** three-gpu-pathtracer (both its schedulers) and mutracer run with their denoisers off.
  - All three pass the diffuse, glass and single-light checks.
  - Speed numbers so far were measured on a heavily loaded Mac and are not trusted yet.
  - mutracer's Cornell "bias" was our adapter dropping its coverage alpha at the box's silhouette; fixed, it now converges
    like the others (relMSE 4.8e-5 at 4096 spp).
  - Both libraries add an energy-compensation trick to rough metals that can't be switched off, so that one test is
    reported separately.
- **Capability detection.** This Mac is "tier 2": subgroups, f16 and timestamps, but no bindless.
- **Literature check.**
  - OIDN 3 on Nov 30 is an alpha; temporal denoising for all GPUs comes in March 2027.
  - NVIDIA's blue-noise textures can't be used (non-commercial licence), so we generate our own.
  - Native wide-BVH traversal is 6.3× ahead of our current BVH, not 3.5×.

- **Trace core (P1).** `@pmndrs/raytracer` traces rays on the GPU: a visibility test, a closest-hit trace and a surface
  lookup, from WGSL or as a batch over a ray buffer. All 590k test rays agree with bvh's CPU version (the remaining
  differences are explained float ties; primary rays must match exactly). It now carries the scene's materials and
  lights on the GPU (the "scene copy"), with power-based light picking. One-bounce shading for the sibling repos
  (`rtShade`) is built and checked. Reflections now runs on it, on a reflections branch (P1.5): same hits, same trace cost
  within the noise, and hit lighting by `rtShade` matches our path tracer's first bounce. Left for the P1 gate: the quiet
  timing run, the maintainer's call on reflections' default hit shading, and merging that branch.
- **Our path tracer (P2).** `@pmndrs/pathtracer` renders the test scenes and converges to the references: the furnace
  tests hit their exact values, the single-light scene matches its closed form, and Cornell sits within the references'
  own disagreement. Sponza agrees too (means within 0.13% of both references, error at the reference noise floor). The Tier 1 class A gate (D23) passes on all seven scenes (P2.32).
  Materials so far: diffuse, metal and glass (smooth and rough), ported from pbrt-v4. Not yet: the HDR sky scene. Speed is not measured properly yet (best effort on the M5: Cornell 512² about 14 ms
  per sample, Sponza 1280×720 about 51 ms).
- **Quiet pass.** Timing that decides defaults or passes a gate is batched into one run the maintainer starts from a CLI
  session with every app closed (docs/QUIET-PASS.md).

- **Wavefront (P3.1, slice 1).** A second scheduler, `PtWavefront`, runs the same estimator as the megakernel over SoA path state with GPU-written indirect dispatch (camera / intersect / shade / shadow / args kernels, each under 50 KB even for the layered + fuzz + film combination that pushes the megakernel past it). It matches the megakernel per pixel on Tier 0, Sponza and 14 feature scenes (alpha MASK not yet). Provisional M5 timing: 1.6x faster on Cornell, 1.28x on Sponza, 1.4-1.7x slower on small films (quiet pass pending).

- **Research track N: neural radiance cache on WebGPU (2026-11-14 to 11-18, milestones M0-M9 done).** NRC runs in the
  browser: on the M5 Pro in real time, on the RTX 3060 in Chrome affordable but not cheap (no subgroup matrices on D3D12;
  on Dawn/Vulkan they reach the tensor cores, 4.4x). A warm cache has 3-4x lower error than plain path tracing per frame and
  keeps a lead through a denoiser + upscaler chain. A plain hash grid matches the network on a diffuse scene for less; the
  network wins on a glossy one once converged. Summary: `bench/results/NRC.md` (accepted by the maintainer). Mac reruns
  and the quiet pass remain: `docs/design/nrc-mac-handoff.md`. Plan and progress: `docs/design/nrc-plan.md`.

**Next**
- P2 Tier 1 wave 2: anisotropy (tangents), textured models, sheen, iridescence; the 50 KB kernel limit with layered materials.
- Reflections onto the core (P1.5, the P1 gate): built on reflections branch `p1.5-raytracer-core`; waits for the quiet timing run, the hit-shading default (P1.5b) and the merge.
- Track N: merge `windows-nrc`, then the Mac reruns (Mitsuba twin N11.4, hash vs network with subgroup matrices N7b, N2r-e) and the quiet pass (`docs/design/nrc-mac-handoff.md`).
- Windows: the `windows-testing` branch and handoff doc are pushed; the maintainer starts it on the laptop.
- Wide BVH in the bvh repo: built and correct on its own branch. Quiet M5 timing: 1.3x or better on merged trees, but BVH4 wins on Sponza and CWBVH8 on Bistro; one default still to choose, then the 3060.

**Speed vs the web competitors (quiet M5, 2026-11-14):** our megakernel is fastest on Sponza (about 1.4x mutracer to equal error) and second on Cornell (mutracer 1.5x faster). Report: bench/results/quiet-pass/20261114-1419/REPORT.md.

**Waiting on the maintainer:** the wide-BVH default; the Tier 1 gate (class A only, or class B too); the default layering (OpenPBR albedo, albedo + internal reflection, or the pbrt walk). Nothing blocks building meanwhile.
