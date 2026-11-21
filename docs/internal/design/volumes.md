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
the owning mesh's transform inverted), so an instance transform moves the grid. `volumeGrid.format` is `r32float` (default) or `r16float` (half the memory, P5.v18; the majorants are computed from the f16-rounded values).
The majorant has two modes, `volumes.majorant`: `global` (`densityMax * max sigma_t`, the default until the quiet-pass timing) and `brick` (P5.v8, V243): a coarse `r32float` texture
holds the max density of each 8^3-voxel brick (group 2 binding 17, `PT_VOLUME_MAJORANT_BINDING`; TS twin `brickMajorants`), and the free-flight loop walks the ray through it with a 3D DDA
(`ptDdaInit` / `ptDdaNext`, TS twin `ddaSegments`), running delta tracking per segment with that brick's majorant (`ptVolSampleBrick`, shadows `ptVolTrBrick`; `volume-wgsl.ts`). Both schedulers.
Same image as `global` within noise (`bench/results/VOLUMES.md`); it cuts the null collisions in sparse volumes.
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

Full tables (agreement with pbrt / Mitsuba, the settings A/B, what did not match and why): [bench/results/VOLUMES.md](../../../bench/results/VOLUMES.md). The scenes and what each proves are listed at the top of `bench/scenes/volumes/index.ts`.

## 5b. Cloud regime additions (P5.v13-v18, docs/internal/CLOUDS-HANDOFF.md)

- **Phase lobes** (`volume/phase.ts`, `PtMedium.lobes`): `(1 - w) HG(gA) + w Draine(gB, alpha)`; `MediumDesc.phase` is `hg2` (two HG lobes), `hgDraine`, or `mie`
  (Jendersie and d'Eon 2023 large-particle fit, 5-50 um, no clamp of the forward peak, V298). A medium with lobes gets a 7th table row `(gB, w, alpha, 0)`; the phase
  sample picks a lobe with u.x, rescales u.x for the lobe, u.y is the azimuth. HG lobes are sampled as 1 - u (cosine to the forward direction) in a form that does not
  cancel at g 0.995; Draine by the supplemental's closed-form CDF inversion (V299). Both schedulers. Without lobes the text is the plain HG one.
- **Outputs** (`volumes.outputs`, megakernel): 3 vec4 per pixel after the radiance sums: (orders 0-1 rgb, primary transmittance), (radiance^2 rgb, first-scatter
  distance), (first scatter happened, 0, 0, 0); `readVolumeOutputs` / `ptDecodeVolumeOutputs`. The order of a contribution is the number of scatter or bounce vertices
  of its light path: a vertex's NEE is order vertex + 1, an escape or emitter hit after `depth` vertices is order `depth`. Everything added before `depth` goes from 1
  to 2 is orders 0-1, so the plane equals a `maxDepth` 1 render per pixel (checked bit for bit). Interface (boundary) skips do not count. The transmittance is the
  shadow-ray estimator along the camera ray up to the first SOLID surface, on its own stream, so the beauty is bit-identical with the outputs on or off.
- **Depth caps** (`vol-cl-furnace-*`, `vol-cl-sph-*` sweeps, `bench/results/VOLUMES.md`): at albedo 0.99 an optical diameter of 200 needs no more than 512 orders
  (512 vs 8192 differ by 6e-5); at albedo 1 depth 512 is 4-7% dark and 2048 still 0.4-0.6%.
- **Separate depth cap** (`volumes.separateDepth`, `PtLimits.maxVolumeDepth`, V303): the pbrt rule counts a medium scatter as a path vertex, so a cloud that needs 512 scatters
  also grants 512 surface bounces. With the option a second counter counts scatters: a scatter ends the path at `scatters >= maxVolumeDepth`, a surface at
  `depth - scatters >= maxDepth`; emission, MIS and sampler dimensions still use the total vertex count. Both schedulers agree per pixel (1.1e-6).
- **Forced in-scatter NEE** (`volumes.forcedNee`, V305, megakernel, homogeneous media): each free-flight segment [0, L) adds a light sample at t from the truncated exponential (rate mu), weight `beta sigma_s exp(-(sigma_t - mu) t) (1 - exp(-mu L)) / mu`, and the segment's real collision skips its own NEE (same expectation, MIS unchanged because the weights are direction-only). Random numbers: sampler pairs of the forced vertex 2^24 + depth.
- **Residual ratio tracking** (`shadowTracking: 'residual'`, V2.2; US9767601 lapsed 2025-10-27): per DDA cell `tr *= exp(-sigma_t min d)` times ratio tracking of the residual with `mu = (max - min) max sigma_t`; unbiased because min <= density <= max in the cell, exact for a constant cell. On our clouds it is noise-neutral (SE x0.98-1.00) and up to 21% slower per sample (provisional): kept opt-in.
- **r8unorm grids** (`volumeGrid.format: 'r8unorm'`, V306): texel = density / scale (scale = grid max), the scale folded into sigma_t and `info.x` (kernel text unchanged); callers compute `densityMax` and the brick majorants / minima from `ptQuantizeGridR8(..).dequantized`, the upload converts them to texel units (`ptTexelBound`) and refuses data off the r8 grid. A quarter of r32 memory.
- **Telescoping transmittance** (`shadowTracking: 'telescoping'`, V307, port of Cycles / Misso 2022): per brick cell the analytic minorant times a jittered k-step march of the residual plus a debiasing term of random order (shared across lanes: hash of sample index and cell). The estimate is SIGNED (can be < 0 or > 1): with the option on, alive tests use abs(tr) (a zero test would bias the mean). Loses at equal time on our clouds: the shadow transmittance is not their dominant noise.
- **Run-time medium parameters** (`volumes.runtimeParams`, V6.6): the medium table is a uniform at group 2 binding 22; `setMedia` rewrites sigma_t, albedo, g and lobe parameters (then reset the accumulation); the number of media, their kinds, lobe presence and grid geometry stay compiled in.
- **Per-pixel adaptive sampling** (`volumes.adaptive`, V2.5b, megakernel): `ptMain` skips pixels whose done flag (plane 3 lane y) is set; `renderUntil` sets flags between batches after `minSamples`. Saves far more samples than time (workgroups wait on their slowest lane), and stopping on the estimate biases heavy-tailed pixels dark (-0.24% to -1.0% measured): not for reference images. V2.5c: the unbiased adaptive path is `ptRenderAllocated` (a uniform pilot sets per-pixel budgets, then a fresh image on a new seed spends them, so the budgets do not depend on the final samples): +0.03% +- 0.09% where the stopping rules read -0.65% to -0.82%.
- **Camera medium** (V3.3): `cameraMedium` may be a grid medium (the camera only has to be inside its boundary mesh). When a scene sets none, the bench detects it with `ptDetectCameraMedium` (generalised winding number > 0.5 = inside; the smallest containing bounding box wins; explicit values win; boundaries must be closed and outward wound, overlapping ones unsupported).
- **Height fog** (`PtMedium.height = { y0, H }`, V310): sigma_t = sigmaT exp(-(y - y0) / H); unit optical depth G = e^-a L phi(L d.y / H) with a = (o.y - y0) / H, inverted exactly for free flight; scalar rate max_c(albedo_c sigmaT_c) with per-channel weights; shadows exp(-sigmaT G). Table row `info = (0, H, y0, 0)`. Refuses forced / equiangular NEE (their homogeneous test is `info.x == 0`).
- **Wavefront volume outputs** (V5.2, plain scheduler): a per-sample radiance scratch plane after the three output planes; shade and shadow kernels add to it (order <= 1 contributions also to plane 1, flagged in the shadow record's lane w), a `voFinish` kernel per pass moves the scratch into the beauty and its square into plane 2; the primary transmittance runs in the intersect kernel at iteration 0. Refill / layers refuse outputs.
- **Equiangular NEE** (`volumes.equiangular`, V309, megakernel, homogeneous): on top of forced NEE, the light pick and triangle point come first, the distance from a 50/50 mixture of the truncated exponential and equiangular about that point (pdf h / ((thB - thA)(h^2 + (t - t0)^2))), estimate `sigma_s T(t) Ld(t) / (0.5 p_e + 0.5 p_q)`. Sun / environment picks use the exponential only.
- **Sparse brick atlas** (`volumes.gridLayout: 'atlas'`, V308): occupied 8^3 bricks (the majorant's) as 9^3 atlas cells (the trilinear support, +side apron, zero voxels outside the grid stored), `r32uint` indirection at binding 21; the majorant stays dense. Density bit-identical to dense; with hardware filtering the stored zeros replace the boundary factor. The texture at binding 16 is then the ATLAS: `textureDimensions(ptVolGrid)` is not the grid size (use `info.yzw`). A page with its own layout adds binding 21 (`uint`, 3d).
- **Hardware-filtered density** (`volumes.densityFilter: 'hardware'`, V304): one `textureSampleLevel` with a linear clamp-to-edge sampler (group 2 binding 20) times the per-axis factor `clamp(g + 1, 0, 1) clamp(n - g, 0, 1)` (trilinear is separable, so this is exactly the manual zero-outside lookup); r32float needs `float32-filterable`. Saves 246 B on the grid + brick kernel.
- **Ratio-tracking roulette** (`volumes.trRoulette`, V1.1): pbrt-v4's rule (tr < 0.05: end with probability 0.75, else x4) in every ratio-tracked shadow segment.
- **r16float grids** (`volumeGrid.format`, V301): the same kernel text (`textureLoad` returns f32), half the memory; the data must be f16-exact (`ptRoundGridF16`), so
  densityMax and the brick majorants computed from it bound what the GPU reads.
- **Self-intersection on large triangles** (V302, P5.v19): the stopgap `integrator.originOffset` (a scene-global floor that skipped `floor / cos` of medium at every boundary) is gone. `rtSurface` returns a
  per-hit `positionError` (pbrt-v4 bound, largest axis; CLAUDE.md Traps) that `ptOffsetOriginErr` applies to bounce / NEE / mask-skip origins, and free flight after a medium interface skip starts at the
  TRUE crossing: the path carries `volBack` = (normal offset) / max(|cos|, 1/64) and the next segment samples over `[0, segment + volBack]` from `origin - dir * volBack` (megakernel `volSegment`,
  `ptVolTrCore`, `ptVoBegin`, wavefront intersect; the wavefront keeps it in path-state section 4, lane z). Reset to 0 after a scatter or a surface vertex (a glass boundary reached as a surface vertex
  still skips its offset).

## 6. Open problems

1. Brick majorant (section 3) is built; open: the brick-size A/B (8^3 is the first guess), a majorant hierarchy only if 1k^3+ grids appear, and the quiet-pass timing that decides the `global` / `brick` default.
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
