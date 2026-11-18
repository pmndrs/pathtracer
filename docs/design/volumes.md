# Participating media (P5)

Status: built and checked on the megakernel (homogeneous and dense-grid media, scattering, NEE through media; seven closed forms and four scenes against pbrt / Mitsuba) and ported to the wavefront scheduler, per pixel equal to the megakernel (section 7, V270). Settings with an A/B: `volumes.shadowTracking` (analytic | ratio, V241), `volumes.majorant` (global | brick, V243).
Code: `packages/pathtracer/src/volume/` (`volume.ts` TS twins and closed forms, `volume-wgsl.ts` WGSL), hooks in `integrator/integrator-wgsl.ts`
(`ptPath`, `ptSampleLd`), `integrator/megakernel.ts` (the 3D density texture), scene side `bench/scenes/volumes.ts`, `bench/scenes/volumes/index.ts`,
exporters `bench/exporters/media.ts`, wavefront stages in `wavefront/wavefront-wgsl.ts` (`ptwfIntersectVolWgsl`, `mediumVertex` in the shade kernel, `ptwfShadowVolWgsl`). DIVERGES V240-V247, V270-V271.

## 1. Scene model

- `SceneDesc.media`: named media. `homogeneous` = `{ sigmaA, sigmaS, g?, scale? }` (RGB per metre, pbrt's parameters); `grid` adds a dense density grid
  (`resolution`, `density` x fastest, `bounds` in the owning mesh's object space; voxel centres at `(i + 0.5) / n`, trilinear, zero outside = pbrt `uniformgrid`).
- `MeshDesc.medium = { interior?, exterior? }`: the media on the two sides of a closed, outward-wound mesh (interior = the side its normals point away from).
  `MaterialDesc { type: 'interface' }` is a pure boundary (pbrt `interface`, Mitsuba `null`); any other material works too (a glass sphere full of milk).
- `SceneDesc.cameraMedium`: the medium the camera starts in (homogeneous only).
- State: one medium id per path (`curMed`, 0 = vacuum). At a surface with a medium pair the next direction decides it (`dot(wi, ng) < 0` = interior); a surface
  without a pair keeps it (pbrt: no `MediumInterface` transition = the ray's own medium). Nesting is "last boundary wins", no stack (pbrt's too).

## 2. Estimators (`volume.ts` header has the algebra)

Free flight is **delta tracking against a scalar majorant** `mu = max density * max_c sigma_t,c`. A tentative collision is a scatter, an absorption or a null event with
per-channel shares `(sigma_s, sigma_a, mu - sigma_t) / mu`. Absorption contributes nothing so it is never sampled: scatter with the scalar probability
`ps = max_c sigma_s,c / mu` (weight `sigma_s / max sigma_s <= 1`), null otherwise (weight `(1 - sigma_t / mu) / (1 - ps)`), reaching the segment end has weight 1. A grey medium
gets weights 1 (scatter) and 0 or 1 (null; 0 kills the path = absorption by roulette), a chromatic one the per-channel ratios, so one path serves RGB. This is unbiased
for any `ps in (0, 1]`; the choice makes the scatter weight at most 1 per channel.

- **NEE at a medium vertex** (`ptSampleLdMedium`): `L = Le * p_HG * Tr / (pmf pdf) * MIS`, MIS (power heuristic, `ptMis`) against the HG sample that then reaches the light.
- **Phase sampling**: HG is sampled exactly (pbrt-v4 `SampleHenyeyGreenstein`), so `f cos / pdf = 1` and the throughput is untouched; `prevPdf = p_HG` feeds the MIS weight of
  an emitter or sky hit afterwards.
- **Shadow transmittance** (`ptVolTr`): a closest-hit loop; a boundary surface passes the ray and switches the medium, any other surface blocks it. Per segment: closed form
  `exp(-sigma_t d)` for a homogeneous medium, **ratio tracking** (product of `1 - sigma_t / mu` over the tentative collisions) for a grid. Shadow rays through solid geometry are
  zero, so a glass boundary shadows its inside like in pbrt and Mitsuba (no caustic path through it).
- Weights of MIS depend on directions only (not on transmittance), which is what keeps the estimator unbiased when the transmittance itself is a random estimate.
- `maxDepth` counts a medium scatter like a surface vertex (`depth >= maxDepth` ends the path before the scatter), the same rule as pbrt `volpath`; Mitsuba's `max_depth` is ours + 1.
  Boundary crossings are free (capped at 256 per path).
- Random numbers: the vertex dimensions (`ptSample2D`) drive NEE and the phase sample like a surface vertex; the variable-length tracking loops draw from a per-path PCG
  stream seeded from `(pixelSeed, sampleIndex)` (`ptVolSeed`), so a render is deterministic.

## 3. Grids

A grid medium's density is a `texture_3d<f32>` (`r32float`, `textureLoad`, 8-tap manual trilinear, zero outside) at group 2 binding 16 (`PT_VOLUME_BINDING`), uploaded by
`PtMegakernel.create({ volumeGrid })`. One grid per kernel (all grid media would share one texture). The medium owns a row-major 3x4 world-to-index matrix (`gridWorldToIndex`,
the owning mesh's transform inverted), so an instance transform moves the grid. The **majorant is global** (`densityMax * max sigma_t`); a brick / coarse-grid majorant with a DDA
walk is the next step (open problem 1) and would cut the null collisions in sparse volumes.
WebVDB: the density arrives as a dense array. `bench/scenes/volumes/vdb-grid.ts` rasterises a `.vdb` float grid parsed by WebVDB's `vdb-web-tools` (`parseVdb`) to such an array
(see TASKS P5.v7); a NanoVDB traversal in WGSL (WebVDB's `nanovdb-wgsl`) would keep it sparse but adds a storage buffer and ~10 KB of WGSL (open problem 2).

## 4. Kernel size and gating

Every kernel without `integrator.volumes` is byte-identical (`node bench/size-snapshot.ts`, 224 variants compared before / after). The media code adds 3.0 KB (homogeneous, 43.7 KB)
and 3.9 KB (grid, 44.7 KB) to the plain megakernel (40.8 KB); with the OpenPBR layered lobes (49-53 KB) a media scene would pass 50 KB (D26 applies: measure on the 3060 first).
`volumes` cannot be combined with `restir`, `frame`, `primary: 'vbuffer'`, alpha MASK, hooks, counters, AOVs or the Beer-law absorber of P2.22 (the wavefront takes the same list, and refuses `materialLod: 'secondary'`).
`ptVolumeWgsl(v, parts)` hands out the pieces (`seedOf`, `bind`, `tables`, `hg`, `density`, `flight`, `tr`, `nee`) so the wavefront kernels carry only their own; the default is all of them in the megakernel's order.

## 5. Checks (closed forms first)

Closed forms (our renderer; pixel mean, 4096 spp unless noted; `node bench/adapters/pt-agreement.ts --report vol-check` writes the table with the per-pixel verdicts):

| scene | closed form | ours | pbrt / Mitsuba (1024 spp) |
|---|---|---|---|
| `vol-absorb-slab` (chromatic absorber, sigma_a 0.3/0.8/1.6, 1 m) | 0.74082 0.44933 0.20190 | 0.74076 0.44926 0.20182 | pbrt 0.718 0.447 0.193 (spectral uplift, V62) / Mitsuba 0.74102 0.44948 0.20198 |
| `vol-slab-single` (tau 0.4, albedo 0.8, g 0.5, max depth 1, normal) | 0.849100 (quadrature) | 0.84908 | 0.84939 / 0.84909 |
| `vol-slab-single-oblique` (cos 0.6) | 0.749184 | 0.74924 | 0.74921 / 0.74897 |
| `vol-furnace-sphere` (albedo 1, g 0.6) | 1 | 1.00004 | 0.99995 / 1.00002 |
| `vol-furnace-sphere-back` (g -0.5) | 1 | 1.00015 | 0.99995 / 0.99986 |
| `vol-camera-medium` (camera inside an absorber) | 0.74082 0.20328 0.05578 | CLOSED_CAM | Mitsuba 0.73995 0.20308 0.05574 |
| `vol-grid-slab` (16 density layers) | 0.49703 | 0.49684 | pbrt `uniformgrid` 0.49702; Mitsuba 0.49222 (clamp outside the grid: -1.0% on a grid WITHOUT a zero border, V246) |

The single-scatter slab is exact only at maximum depth 1 (the free flight and the one scatter, NEE through the rest of the slab, phase-sampled escape): pbrt `maxdepth` 1 = Mitsuba `max_depth` 2.

Scenes against pbrt-v4 and Mitsuba 3 at 4096 spp (128x128; relMSE with the metrics' eps 1e-2; the floor is the pbrt-vs-Mitsuba relMSE; non-finite pixels 0 in every cell):

| scene | pbrt vs Mitsuba (floor) | ours vs pbrt | ours vs Mitsuba |
|---|---|---|---|
| `vol-cornell-smoke` | 6.9e-4, relBias -0.97% | 7.2e-4, +1.01% | 1.0e-4, +0.03% |
| `vol-milk-sphere` | 6.2e-5, -0.01% | 6.6e-5, -0.00% | 5.2e-5, -0.01% |
| `vol-grid-cloud` | 6.6e-5, 0.00% | 5.9e-5, +0.01% | 5.2e-5, +0.01% |
| `vol-vdb-smoke` | 4.8e-5, 0.00% | 3.8e-5, 0.00% | 3.7e-5, -0.00% |

The Cornell scene's 1% is the colour of the walls, not the medium: the plain `cornell` references already differ by pbrt -0.69% (spectral uplift of the chromatic reflectances); we agree with Mitsuba (RGB).

Full tables (agreement with pbrt / Mitsuba, the settings A/B, what did not match and why): [bench/results/VOLUMES.md](../../bench/results/VOLUMES.md). The scenes and what each proves are listed at the top of `bench/scenes/volumes/index.ts`.

## 6. Open problems

1. Brick / coarse-grid majorant (DDA) for sparse grids; today the null-collision count scales with `densityMax * path length`.
2. NanoVDB traversal instead of a dense texture (memory: 48^3 = 0.4 MB is nothing, a 512^3 fog volume is 512 MB dense).
3. ReSTIR / real-time modes with media; `materialLod` with media on the wavefront.
4. Spectral media: RGB sigma with the per-channel weights above is exact in RGB; pbrt's spectral uplift of a saturated sigma differs (V62), so references use grey sigma.
5. Emissive media (`Le`), several grids per scene, nested media with a stack.

## 7. Wavefront (V270, V271)

Where the stages go. The megakernel's loop body is: trace, free flight in the current medium up to the hit, then the miss / interface / surface decision. The wavefront splits it by what each part needs:

| kernel | media work | bindings it adds |
|---|---|---|
| camera / refill start | writes the path's volume stream (`ptVolSeedOf(pixelSeed, sample)`), the camera medium, 0 skips | none (the hash is 5 lines) |
| intersect | the loop: `rtClosest`, `ptVolSample` (delta tracking, scales `beta`), interface surfaces passed (medium switched, origin moved, no depth), up to 256 skips. Exits as hit / miss / scatter (`hit.w` 2, event distance in `hit.t`) / dead (`hit.w` 3). Writes `beta`, medium, skips, stream and origin back to the path state | density grid (16), brick majorant (17); no scene copy |
| shade | `hit.w` 2: a vertex without a BSDF: depth check, NEE at the event (`ptwfSampleLdMedium`), HG sample, roulette, next ray from the event. A surface hit reads `ptVolMat` for `medIn / medOut` (NEE start medium, medium after the bounce) | none (media tables are `var<private>`) |
| shadow | `ptVolTr` of the stored ray from the stored start medium multiplies the stored contribution; interface surfaces pass, solid ones block | grid / majorant textures |

Why the intersect kernel, not a medium kernel between intersect and shade: the free flight needs only the ray, the closest hit distance and the media tables; it is a loop around `rtClosest` exactly like the alpha-MASK cutout loop (P3.7), it adds no dispatch per iteration and no queue, and the surface code in shade stays in one piece. The cost: a path that scatters before its hit has already paid for the closest-hit query (the megakernel too).

Path state: section 4 (the same 16 B as the Beer absorber / ray cone use; they cannot coexist with media: the cone width keeps its lane) holds `(stream state, medium | skips << 16, 0, cone width)`. 80 B per path, as the textured kernel. Shadow-ray records get a 4th vec4 `(start medium, ratio-tracking seed, 0, 0)`: 64 B per path.

Random numbers. The volume stream is a PCG32 sequence per path, seeded from (pixel seed, sample index). The megakernel draws free-flight numbers, then (for a ratio-tracked shadow ray) shadow-tracking numbers, then free flight again, all from one stream; a scheduler that traces the shadow ray later cannot draw them in that order. Since V270 an ESTIMATED shadow (grid medium, or `shadowTracking: 'ratio'`) runs on its own stream: `ptVolFork()` hashes the path stream into the shadow ray's seed (stored in its record) and moves the path stream on by one step, in both schedulers. The analytic homogeneous shadow draws nothing and the megakernel's numbers for those scenes are unchanged (bit-identical images, `bench/results/reports/` check in TASKS P5.v9). The grid scenes' megakernel images changed as noise realisations only.

Per-pixel agreement with the megakernel is the test of the port (it checks the RNG protocol, the state round trip and every branch): numbers in TASKS P5.v9. Refill (inline, batch, ordered), layers, subgroup appends and the sorts work: a slot is started by the one `ptwfStartPath`. Kernel sizes (wavefront, minified, KB): homogeneous camera 3.7 / intersect 16.6 / shade 34.0 / shadow 14.5; grid 17.4 / 34.2 / 16.2; grid + brick 19.9 / 34.2 / 18.7; with textures shade 37.6: all under the 50 KB DXC class (the megakernel with media is 43.7-47.9 KB).
