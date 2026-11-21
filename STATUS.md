# Status

_Last updated 2026-11-19._

**Phase:** P0 passed its gate on the Mac (the RTX 3060 half waits on Windows). P1 (trace core) met its gate: reflections runs
on it (P1.5, merged here; the reflections branch waits for reflections' own review). P2 (reference path tracer) passed the
Tier 1 class A gate (D23, all 7 scenes) and has OIDN on final frames (bench); left are the siblings' switch-over and the
layering default (D24 + the P2.32b correction, now the library default, D32). **P3 (speed) is closed on a provisional basis (D32; gate on the M5, 3060 half pending Windows):** a wavefront scheduler
beside the megakernel (per pixel equal; refill, samples in flight, early-out), alpha MASK in both, visibility-buffer
primaries, ray-class material LOD, scene-copy packing, cache hooks / counters / AOVs. Every new path is a setting; since D32 the library
defaults follow the best provisional evidence (packed scene copy, albedo correction texture, `ptwfAutoPolicy` as the scheduler entry point, ReSTIR unbiased when on) and the quiet pass is an optional
confirmation (docs/internal/QUIET-PASS.md). Competitor gate (provisional, `bench/results/P3-GATE.md`): see below. The 3060 (W10 compile time, W13 gate numbers) is the open half.
**P4 (real time) is closed provisionally too (P4.5b, the 3060 30 fps number, pending Windows):** built end to end on both schedulers (megakernel and wavefront agree per pixel): mode presets (D31 cutouts),
blue-noise sampler, per-frame denoiser planes incl. an emission plane, ReSTIR DI (unbiased within the seed spread after the
P4.2b key fix), and the bench chain kernel -> SVGF -> upscaler (P4.5/P4.6) plus an OIDN preview chain (P4.7). M5 numbers
(provisional): ~19 ms per frame at 960x540 -> 1080p, 2 bounces; the 30 fps 3060L gate is borderline by estimate and needs
the real 3060 run (P4.5b). Track N (NRC, hash grid, 3.3x faster 3060 walk) is merged.

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
- **Capability detection.** This Mac is "tier 2" in Chrome stable: subgroups, f16 and timestamps, but no bindless. Chrome 160 Canary
  adds the resource table (tier 3), but Metal cannot use it in compute passes yet, so bindless still can't run here (P1.4c2).
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
  session with every app closed (docs/internal/QUIET-PASS.md).

- **Wavefront (P3.1, slice 1).** A second scheduler, `PtWavefront`, runs the same estimator as the megakernel over SoA path state with GPU-written indirect dispatch (camera / intersect / shade / shadow / args kernels, each under 50 KB even for the layered + fuzz + film combination that pushes the megakernel past it). It matches the megakernel per pixel on Tier 0, Sponza and 14 feature scenes (alpha MASK not yet). Provisional M5 timing: 1.6x faster on Cornell, 1.28x on Sponza, 1.4-1.7x slower on small films (quiet pass pending).

- **Research track N: neural radiance cache on WebGPU (2026-11-14 to 11-18, milestones M0-M9 done).** NRC runs in the
  browser: on the M5 Pro in real time, on the RTX 3060 in Chrome affordable but not cheap (no subgroup matrices on D3D12;
  on Dawn/Vulkan they reach the tensor cores, 4.4x). A warm cache has 3-4x lower error than plain path tracing per frame and
  keeps a lead through a denoiser + upscaler chain. A plain hash grid matches the network on a diffuse scene for less; the
  network wins on a glossy one once converged. Summary: `bench/results/NRC.md` (accepted by the maintainer). Mac reruns
  and the quiet pass remain: `docs/internal/design/nrc-mac-handoff.md`. Plan and progress: `docs/internal/design/nrc-plan.md`.

**Next**
- P2 Tier 1 wave 2: anisotropy (tangents), textured models, sheen, iridescence; the 50 KB kernel limit with layered materials.
- Reflections onto the core (P1.5, the P1 gate): built on reflections branch `p1.5-raytracer-core`; waits for the quiet timing run, the hit-shading default (P1.5b) and the merge.
- Track N (merged into main 2026-11-18): the Mac reruns (Mitsuba twin N11.4, hash vs network with subgroup matrices N7b, N2r-e) and the quiet pass (`docs/internal/design/nrc-mac-handoff.md`).
- Windows: the `windows-testing` branch and handoff doc are pushed; the maintainer starts it on the laptop.
- P5 (breadth): 6 of 7 items built. On main: ReSTIR GI, volumes, spectral + dispersion, MaterialX -> WGSL, Split RC interval tracing for gi (all opt-in, default kernels byte-identical), each on both schedulers (per pixel equal except long dispersive glass paths, P5.s9 open). The subgroup-matrix denoiser is on Denoiser branch `feat/wgsl-sgm-v2` (unmerged, maintainer's call). Bindless is built too (opt-in, megakernel): this Mac's Canary has the feature but Metal can't use it in compute and SwiftShader gets divergent lookups wrong, so whether it works (and is faster) is a 3060 question (W17). Left: the 3060 runs.
- Volumes track (branch `clouds/volume-reference`, 2026-11-21/22, 36 commits; plan V0-V6 in docs/internal/design/volumes-plan.md): in the cloud regime (HG g 0.8-0.9, albedo 0.99-1, optical thickness 10-200, a 256^3 cloud) the estimator agrees with pbrt and Mitsuba at their noise floor, and a regression sweep re-checked every volume closed form with every new option. New, all opt-in (default kernels byte-identical): droplet phases (Mie fit, two-lobe HG), volume outputs + `renderUntil` / `renderAllocated` (order split, transmittance, standard error, unbiased adaptive allocation), r16 / r8 / sparse-atlas grids + `ptVolumeAutoGrid` (the clouds' 437 MB grid in ~67 MB), hardware-filtered density, ratio-tracking roulette, forced in-scatter / equiangular NEE / VSPG for thin and lamp-lit fog, height fog, emissive media, a medium stack, camera inside grids, runtime medium parameters (instant sliders), media with OIDN guides and the real-time frame planes, a preview preset (pending confirmation), glTF absorption / scattering media, cloud and fog examples. Fixed on the way: large-ground self-intersection (per-hit offset bound, V302) and a silent black wavefront image past 4094 iterations. Measured and kept opt-in because they lost: residual and telescoping shadow transmittance, stopping-rule adaptive sampling (biased). Timings wait for the quiet pass.
- P3 / P4 closed on a provisional basis (D32): defaults flipped (packed scene copy, albedo correction texture, ReSTIR unbiased when on, `ptwfAutoPolicy` as the scheduler entry point with a kernel-size rule); open items in TASKS "Open after closing P3 / P4".
- Cutouts in the real-time modes (R2b, D31): `preview` / `realtime` now use exact indirect hits plus `alphaShadow: 'transmit'` (beta 0.45) instead of D27's `opaqueSecondary`; max speed is the opt-in `alphaPolicy: 'opaqueShadow'`. Numbers are provisional (shared Mac) until the quiet pass.
- Wide BVH: bvh 0.1.0 is on npm and we depend on it (D33, X4); `bvh4` and `cwbvh8` are a setting in every kernel (X1.i), trace-core clean, provisional M5 Pro: cwbvh8 -16..-26% megakernel time on Sponza / Cornell. Default still binary until the 3060 run picks one.
- GPU texture mips are the library default (D34): `@pmndrs/raytracer` depends on `@pmndrs/downsampler` (npm, D33); texture loads ~10x faster (24 textures at 2048^2: 1469 -> 150 ms, provisional), images unchanged at the default `textureLod`, `mips: 'cpu'` stays a setting. The bench chain uses the same package for auto-exposure (X5).

**Speed vs the web competitors (M5, provisional P3 gate, 2026-11-18):** with the auto scheduler we are fastest on Sponza (1.64x mutracer, 1.76x tgpt-wavefront, 5.0x tgpt-megakernel to equal error); on Cornell mutracer is about 10% faster to relMSE 1e-2 and we are 1.19x faster to 1e-3. Report: bench/results/P3-GATE.md (the quiet baseline of 2026-11-14 is bench/results/quiet-pass/20261114-1419/REPORT.md). Mrays/s on the 3060L: pending Windows (W13).

**Waiting on the maintainer:** the wide-BVH default (needs the RTX 3060 run, QUIET-PASS). The Tier 1 gate (D23), the default layering (D24 + the correction) and the other P3 / P4 defaults are decided (D32, provisional); the quiet pass is an optional confirmation. Open on the 3060 (docs/internal/WINDOWS-HANDOFF.md W10, W13-W16): DXC compile time, the P3 gate numbers, the P4 30 fps gate, the wide-BVH formats, `downsampler-check`. Nothing blocks building meanwhile.
