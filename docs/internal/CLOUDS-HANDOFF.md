# Clouds handoff: volumes as a ground-truth reference (branch `clouds/volume-reference`)

_Written 2026-11-21 from the WebGPUClouds session. Read [CLAUDE.md](../../CLAUDE.md), [STATUS.md](../../STATUS.md),
[TASKS.md](../../TASKS.md), [DIVERGES.md](../../DIVERGES.md) and [design/volumes.md](design/volumes.md) first. This page is
the clouds-specific part: what a consumer needs from the volume path, in priority order, each with a gate._

## Why this exists

The sibling repo `~/Developer/WebGPUClouds` (realtime volumetric clouds) now uses this path tracer as its ground truth for
cloud lighting. Its realtime lighting model will be fitted, by script, to path-traced frames of the same density, sun and
sky. That only works if the reference is trusted in the regime clouds live in: strongly forward scattering, albedo near
1, hundreds of scattering orders, optical depths of 50 and more. P5.v5 validated g up to 0.6 on 48^3 to 60x115x60 grids.
Clouds need g of 0.8 to 0.99 on grids of several hundred voxels a side.

Plan and context on the clouds side (read-only for you):
- `docs/AUDIT-2026-11.md`: the plan. Step 0.2 is this handoff. Phase 1 (lighting fit) consumes the results.
- `examples/lab/reference-pt/pt.ts`: how the clouds page drives the megakernel today (one box mesh as the medium
  boundary, one dense grid, brick majorant, delta sun, importance-sampled equirect sky, `maxDepth` 512).
- `examples/lab/reference-pt/bake.ts`, `linear.ts`: the density bake and the linear-radiance readback.
- `scripts/gpu/scorecard.py`: the measures the clouds side computes from a reference and a realtime frame.
- `work/reports/P0-3-linear-scorecard.md`: first linear comparison, including the grain finding below.
- `DIVERGES.md` rows R1 and R3: the two gaps this handoff closes.

## How the clouds side uses the volume path today

| | Value |
| --- | --- |
| API | `PtMegakernel.create` with `integrator.volumes`, `volumeGrid`, `brickMajorants`, `buildTraceScene`, `buildSceneCopy` |
| Medium | one grid, grey `sigmaT` 0.3 to 0.6 per metre at density 1, albedo 0.99, single HG g 0.8 |
| Grids | dense r32float, 376x156x376 (88 MB) up to about 504x352x616 (437 MB); mostly empty box around one cloud |
| Light | delta sun (`angularRadius` 0, `sunVisible` false), equirect sky from the realtime sky LUT, `envSampling: 'importance'` |
| Frame | 1280x720, two seeds at 2048 spp each, `maxDepth` 512, no Russian roulette |
| Output | accumulator read back as linear float, written to PFM |
| Measured | about 23 ms per pass at 1280x720 on the M5 with another agent on the GPU (not a timing result) |

Two facts from that use:
- **Units agree.** Clear-sky pixels of the path tracer and the realtime renderer match within 0.05 % per channel. Please
  keep the meaning of `sun.irradiance`, the environment map's radiance and its direction mapping stable, or flag a change.
- **Grain limits the finest measure.** At 1024 spp the path tracer's own grain is 63 to 87 % of the reference's 4 to 16 px
  detail-band energy on a cloud's lit face. The clouds side works around it with two seeds and a quadrature correction.

## What is needed, in priority order

Suggested task IDs continue the volumes block (P5.v13 onward); renumber to fit TASKS.md. All options default off and
every default kernel stays byte-identical, as P5.v1 required.

### 1. Validation in the cloud regime (P5.v13). Do this first; everything else rests on it

- Scenes against pbrt-v4 `volpath` and Mitsuba 3 at their own noise floor, as P5.v5 did, with:
  - HG g = 0.8, 0.85 and 0.9; albedo 0.99 and 1.0; grey sigma;
  - a homogeneous sphere and slab at optical thickness 10, 50 and 200, sun plus uniform sky, `maxDepth` at least 512;
  - one heterogeneous grid at cloud scale. The clouds side can supply a baked grid (88 MB raw float32) with its sky map,
    sun and camera; ask for it, or use `vol-grid-cloud` scaled up to at least 256^3.
- Closed forms at high g: the furnace (albedo 1 under a uniform sky returns the sky radiance at every depth cap that is
  high enough), and the single-scatter slab at g 0.85.
- Report where a depth cap or Russian roulette changes the answer: for albedo 0.99 at optical thickness 200, how many
  orders are needed before the image stops moving.

Gate: every cell at or below the pbrt-versus-Mitsuba noise floor, tables added to `bench/results/VOLUMES.md`. A miss is a
result: record which g and thickness first disagree.

### 2. Droplet phase functions (P5.v14)

- **HG + Draine blend** (Jendersie and d'Eon 2023, "An Approximate Mie Scattering Function for Fog and Cloud Rendering"):
  `phi(u) = (1 - wD) * HG(u, gHG) + wD * Draine(u, gD, alpha)`, parameters from the mean droplet diameter d in micrometres
  (their large-particle fit, valid 5 to 50):
  `gHG = exp(-0.0990567 / (d - 1.67154))`, `gD = exp(-2.20679 / (d + 3.91029)) - 0.428934`,
  `alpha = exp(3.62489 - 8.29288 / (d + 5.52825))`, `wD = exp(-0.599085 / (d - 0.641583)) - 0.665888`.
  The paper gives a sampling routine for the Draine lobe. Check the constants against the paper before use; they are copied
  from the clouds repo's notes (`docs/research/approximate-mie.md`), not from the paper directly.
- **A general two-lobe HG** (g1, g2, weight), because the clouds runtime's art default is one (0.2 at 60 %, 0.85 at 40 %)
  and Karma's volume shader offers exactly that.
- Per medium, as `g` is today. Phase stays achromatic.

Gate: unit tests for normalisation, mean cosine and sampling-versus-pdf agreement for each lobe; the single-scatter slab
closed form passes with each phase; one scene agrees with Blender Cycles 5.1 (its Volume Scatter node has `DRAINE` and
`MIE`), since pbrt and Mitsuba have neither. The clouds side is building a headless Cycles reference script
(`scripts/reference/` on its branch `worktree-agent-a4724070a7a21e83c`, not merged yet) that reads a JSON manifest plus a
raw float32 grid and a PFM sky; reuse it once it lands.

### 3. Scatter-order split (P5.v15)

The lighting fit treats single scattering (exact, given the shadow optical depth) and multiple scattering separately.
- One render that accumulates **order 1** and **orders 2 and above** in separate buffers, both read back as linear float.
  If that costs too much, a documented `maxDepth` meaning that isolates order 1 is acceptable (volumes.md says a scatter
  plus its NEE at max depth 1 is single scatter; confirm it for a medium inside a boundary mesh, where interface skips
  also consume depth or do not).

Gate: order 1 plus orders 2+ equals the full render within noise; order 1 matches the single-scatter closed form.

### 4. Outputs with volumes on (P5.v16)

`volumes` currently cannot be combined with `aovs`. Needed with a medium present:
- **Transmittance of the primary ray** (alpha = 1 - T), by ratio tracking. The clouds page marches the grid itself for
  this today; the tracer's own estimate would remove one place the two sides can differ.
- **Per-pixel second moment** (sum of luminance squared, or of RGB squared), so one render gives its own standard error.
  This replaces the two-seed run.
- Optional: distance to the first scatter event (mean over samples).

Gate: T matches the absorber-slab closed form; the standard error from the second moment matches the seed-to-seed
difference on one scene.

### 5. A ground under the cloud (P5.v17)

Confirm, with a scene and a closed form, that a diffuse mesh below a grid medium is lit through the medium and bounces
light back into it (sun plus sky), and document the setup. The clouds side has no ground in its references yet, and base
brightness depends on it.

### 6. Grid memory and sparse grids (P5.v18)

- An `r16float` grid option (the Disney quarter cloud is 437 MB as r32float), A/B'd against r32float on one scene.
- Brick majorant correctness on a large, mostly empty grid (cloud in a box). Its timing goes to the quiet pass as P5.v10.

### Not needed now

Spectral media, NanoVDB traversal, several grids per kernel, emissive media, procedural WGSL density, ReSTIR with media,
a denoiser for volume frames. If item 4's second moment lands, a variance-guided stop would be the next step.

## Rules for this branch

- Work on `clouds/volume-reference`; the maintainer merges. Follow CLAUDE.md throughout (A/B by default, correctness
  before timing, no timings from a loaded machine, PFM for interchange).
- WebGPUClouds is read-only from here. If the clouds page should change to use something new, write the request at the
  end of this file under "Requests back to clouds".
- Record departures from the paper in DIVERGES.md (for example clamping the Mie fit's sub-degree forward peak, which the
  clouds runtime does at a phase value of 25 and a reference must not do silently).
- When an item lands, tick it in TASKS.md and add one line here under "Done" with the scene names and the result file.

## Done

- 2026-11-21 **P5.v13 validation**: `vol-cl-sph-*`, `vol-cl-slab-*`, `vol-cl-cloud256` (256^3, 2 m voxels) at or below the pbrt-vs-Mitsuba floor in every cell, relBias within +-0.3%; closed forms `vol-cl-furnace-*`, `vol-cl-single-*`; depth caps: albedo 0.99 needs <= 512 orders at tau 200, albedo 1 at 512 is 4-7% dark. `bench/results/VOLUMES.md` "Cloud regime".
- 2026-11-21 **P5.v14 phases**: `MediumDesc.phase` / `PtMedium.lobes` = `hg2`, `hgDraine`, `mie` (5-50 um), both schedulers; `vol-cl-single-{hg2,mie10,mie20}[-oblique]` against the closed form, `vol-cl-sph-hg2-a99-t50` against Mitsuba `blendphase`. Cycles comparison open.
- 2026-11-21 **P5.v15 / P5.v16 outputs**: `integrator.volumes.outputs` (megakernel) + `PtMegakernel.readVolumeOutputs()`: orders 0-1 (= a `maxDepth` 1 render, bit for bit), primary transmittance (1 - alpha, up to the first solid surface), standard error from the second moment (calibrated against seed spread), first-scatter distance. Bench: `pt_volume_outputs=1`.
- 2026-11-21 **P5.v17 ground**: `vol-cl-ground-furnace`, `vol-cl-ground-absorb` (closed forms), `vol-cl-ground-sph-g85-a99-t50` (at the floor). Use a tessellated ground (see Requests back to clouds).
- 2026-11-21 **P5.v18 r16float**: `volumeGrid.format: 'r16float'` with `ptRoundGridF16` (compute `densityMax` and `brickMajorants` from the rounded array). `vol-cl-cloud256` r16 vs r32: relMSE 5.5e-6, mean -0.003%.

## Requests back to clouds

- **2026-11-21, Mie fit constants (affects the clouds runtime today).** In the paper's large-particle fit (supplemental, Eqs. 8 and 10) the constants
  0.428934 and 0.665888 sit INSIDE the exponent: `gD = exp(-2.20679 / (d + 3.91029) - 0.428934)`, `wD = exp(-0.599085 / (d - 0.641583) - 0.665888)`.
  Checked against the supplemental PDF (pdftotext of research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie-supplemental.pdf, section 3.1)
  and Blender Cycles (`phase_mie_fitted_parameters`, volume_util.h), which agree. `docs/research/approximate-mie.md` lines 23, 25, 90, 92 and
  `packages/runtime/src/runtime.ts:651, 653` subtract them OUTSIDE. At d = 20 um that gives gD 0.483 instead of 0.594 and wD 0.304 instead of 0.498
  (mixture mean cosine 0.896 instead of 0.879). Please fix both, and re-check any fit or tuning done against the HG + Draine option.
- **2026-11-21, a ground under the cloud: tessellate it.** A single large ground quad near the world origin self-intersects with the default (quantized) BVH nodes
  (V302: the ground under an opaque box read 100x pbrt's sky light). Fixed (2026-11-21): `rtSurface` carries a per-hit position
  error bound and the secondary origins use it (DIVERGES V302); `integrator.originOffset` no longer exists. No need to tessellate.
- **2026-11-21, new options the reference page can use:** `volumes.outputs: true` + `readVolumeOutputs()` replaces the second seed (standard error) and the page's own
  alpha march (transmittance); `volumeGrid.format: 'r16float'` (round the bake with `ptRoundGridF16` first) halves the 437 MB grid; `media[i].lobes` (`ptPhaseLobes({ type:
  'mie', diameter })` or `hg2`) for the droplet phase (set `g` to `lobes.gA`).
- **2026-11-22, variance-guided reference renders (V2.5a):** with `integrator.volumes.outputs` on, replace the fixed spp / two-seed run by
  `const r = await kernel.renderUntil({ meanRelSe: 1e-3, maxSamples: 16384 })` after `reset` and `setSeed`; log `r.samples` and `r.achieved.meanRelSe`, use `r.outputs`
  as the final image (no second read). For a per-pixel guarantee add `pixelRelSe: { below: 0.05, quantile: 0.99 }` (both targets must hold). Example: `bench/pages/render-until-check.ts`
  (1e-3 on `vol-cl-sph-g85-a99-t50`: 1024 spp, 11 s). Also new and opt-in: `volumes.trRoulette` (about 1.8x per sample on a 256^3 cloud, unbiased), `volumes.densityFilter:
  'hardware'`, `volumeGrid.format: 'r8unorm'` (`ptQuantizeGridR8`: your 437 MB grid in 109 MB, -0.02% image change), `volumes.separateDepth` + `maxVolumeDepth`,
  `integrator.aovs` with media (OIDN guides). Details: TASKS V1.1-V5.1a, docs/internal/design/volumes.md section 5b.
- **2026-11-22, sun disc for peaked phases (V2.6):** with a Mie (10 / 20 um) or g >= 0.99 phase, a finite sun disc (`sun.angularRadius` 0.00465 rad instead of 0) cut the
  64 spp error about 3.6x and the worst fireflies about 2x on `vol-cl-sph-mie20-a99-t50` at no cost; the mean moves by +0.2% (inside noise); HG g 0.9 is unaffected (3 seeds,
  provisional). It is a slightly different light: use it only if the realtime lighting fit models the same disc (please confirm), and keep the delta sun for comparisons
  against pbrt / Mitsuba.

