# Volumes: the survey against what we have (2026-11-21)

Analysis agent report (Opus), comparing [14-volumes-survey.md](14-volumes-survey.md) with our volume path (docs/internal/design/volumes.md, bench/results/VOLUMES.md,
`packages/pathtracer/src/volume/`) and two local clones (`research/sources/native/cycles`, `research/sources/web/three-gpu-pathtracer`). Plan built from it:
[docs/internal/design/volumes-plan.md](../../docs/internal/design/volumes-plan.md).

## Findings

1. **Two survey "future" items are built**: the flat brick majorant + DDA (`ptDdaInit` / `ptDdaNext` / `ptVolSampleBrick` / `ptVolTrBrick`, 8^3 bricks, binding 17) and the
   approximate Mie phase (P5.v14). Our own docs were stale on the brick walk (volumes.md sections 3 and 6.1, DIVERGES V243).
2. **The survey missed the best portable source.** Blender Cycles (Apache-2.0, `src/kernel/integrator/shade_volume.h`) ships a null-scattering integrator with an octree
   majorant + minorant walked by a hierarchical DDA, shadow transmittance by **Misso et al. 2022 "Unbiased and consistent rendering using biased estimators"** (jittered ray
   marching + debiasing with FIXED counts and one random order shared by all pixels: GPU-coherent; `volume_transmittance`, lines 589-688), **equiangular + distance MIS in
   heterogeneous media**, and **Volume Scattering Probability Guiding** (Xu et al., kehanxuuu.github.io/vspg-website).
3. **Web competition**: three-gpu-pathtracer's WebGPUPathTracer 0.0.28 (2026-11-02) has homogeneous fog with light sampling inside it, a medium stack (nested media, camera
   inside) and subsurface scattering through volumes (`KHR_materials_scatter` proposal); no grids, no equiangular. Our lead: VDB-style grids, the majorant DDA, droplet phases,
   validation against pbrt / Mitsuba, the wavefront scheduler. Our gaps against it: medium stack, glTF volume materials, camera inside any medium, real-time / denoised media.
4. **Ratio tracking has no roulette**: `ptVolTrSegment` / `ptVolTrBrick` stop only at T = 0 and keep reading density deep inside a cloud (pbrt-v4 roulettes at T < 0.05 [M]).
5. **Density lookups are 8 manual `textureLoad` taps** even for r16float; r16float / r8unorm are filterable in core WebGPU (r32float with `float32-filterable`): one
   `textureSampleLevel`. Hardware trilinear weights are convex, so the brick majorant still bounds the value. Also saves kernel bytes.
6. **Volume megakernels are over the ~50 KB DXC class** (grid + brick + outputs 50.5 KB before the offset bound, 53.5 KB after; + lobes ~52-54 KB): new features need a
   size-saving item next to them.

## Method by method (condensed)

| area | have | gap | win | cost | check (minutes) |
|---|---|---|---|---|---|
| free flight | delta tracking, scalar majorant (global / brick), chromatic weights | decomposition tracking (minorant per brick) | fewer lookups in dense interiors | M | TS twin vs line integral; `vol-grid-slab`; lookup counter on `vol-cl-cloud256` at 256 spp |
| shadow transmittance | closed form (homogeneous), ratio tracking (grids, brick-aware) | (a) roulette at low T, (b) residual ratio tracking (minorant), (c) Misso 2022 (Cycles port) | sun NEE dominates cloud cost: biggest noise-per-time lever | S / S-M / M | TS twin mean at 1e5 trials; `vol-cl-sph-g85-a99-t200`, `vol-cl-cloud256` at 256 spp within the second-moment SE |
| majorants | flat 8^3 DDA | brick-size A/B; octree only at 1k^3+ | speed on sparse grids | S | same image; quiet pass |
| storage | dense r32 / r16, manual trilinear, one grid per kernel | hardware filtering; r8unorm + scale; sparse brick atlas (indirection + 9^3 bricks, ~28% of dense at 20% occupancy, several grids per kernel); NanoVDB (L, storage-buffer pressure) | 437 MB cloud, iGPU / Windows | S / S / M / L | same-seed image vs dense; memory printout |
| equiangular | none (no point lights; small emissive triangles yes) | equiangular + distance MIS | lamp-lit fog | S-M homogeneous, L grids | new `vol-lamp-fog` vs 1D quadrature; `vol-cornell-smoke` equal time |
| thin media | delta tracking scatters with p = 1 - T | forced in-scatter NEE; VSPG | god rays in thin fog | S-M / M | thin-fog scene equal-time relMSE |
| null-scattering MIS | MIS on directions only (V240, unbiased) | Miller 2019 / pbrt r_u r_l | chromatic media, equiangular in grids | L | chromatic slab variance |
| phases | HG, two-lobe, HG + Draine, Mie 5-50 um | small / mid Mie fits; Cycles check; wavefront lobe check | | S | `wf-p5-check cases=vol-cl-single-mie10` |
| sun | delta sun (clouds) | finite disc A/B (MIS vs peaked phase) | fewer silver-lining fireflies | S | `vol-cl-sph-mie20` 64 spp fireflies; agree with clouds first |
| depth | a medium scatter counts toward maxDepth | separate `maxVolumeDepth` (Cycles, tgpt have one) | surfaces stay cheap under deep clouds | S | identical when the caps are equal |
| caching / learning | NRC track (surfaces) | NRC at medium vertices, MRPNN, guiding | real-time only (reference must be unbiased) | L | research |
| resampling | volumes refuse ReSTIR | ReSTIR DI behind homogeneous media; volumetric ReSTIR | real time | M / L | |
| real time | volumes refuse frame / aovs / restir; outputs megakernel only | medium primary-vertex semantics (albedo sigma_s / sigma_t, viewZ from first scatter, sky x T in the emission plane) | denoiser / upscaler chain with fog and clouds | M-L | `realtime-check` plane identity on `vol-cornell-smoke` |
| emissive media | none | Le / temperature grid | fire demos | M | closed form Le (1 - exp(-sigma_a d)) |

## Survey claims corrected

- three-gpu-pathtracer HAS WebGPU homogeneous fog / medium stack / SSS (the narrower claim "no grid / cloud web path tracer" holds).
- Kettunen 2021 is not fixed-step: its debiasing draws a random number of higher orders. Misso 2022 as Cycles ships it is the practical port.
- Deep Scattering authors: Kallweit, Muller, McWilliams, Gross, Novak [M].
- NanoVDB / OpenVDB licence: possibly Apache-2.0 now (verify); WebVDB's `nanovdb-wgsl` and emcfarlane/webgpu-nanovdb are different projects, both licences unchecked.
- Residual ratio tracking patent "expired": verify before it becomes a default. Peters 2025 jackknife is biased: preview only.
- Volume guiding 2022-24 exists (VSPG, in Cycles); Cycles is not "delta tracking only".
