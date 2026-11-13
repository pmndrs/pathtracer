# Competitors in the bench

Web renderers we measure against the cached pbrt-v4 / Mitsuba 3 references. All run from **published npm packages**
(pinned exactly in `bench/package.json`, devDependencies); nothing is checked out into `.cache/competitors/` because
both libraries ship their WebGPU renderer in the package. Nothing under `research/sources/` is built or run.

| Renderer | Package | Version | Commit / integrity | Licence | Adapter ids |
|---|---|---|---|---|---|
| three-gpu-pathtracer (WebGPU) | `three-gpu-pathtracer` | 0.0.28 (2026-11-02) | npm gitHead `9f5b910d5c19e9d0dd9bd200097eec9893f208c7`; sha512-Ybfa6qpR…l019A== | MIT (Garrett Johnson) | `tgpt-wavefront`, `tgpt-megakernel` |
| three-mesh-bvh (its BVH, locked as the README asks) | `three-mesh-bvh` | 0.9.18 | npm gitHead `1d92383233482f12512ef0804748de45ad7f2f60` | MIT | (dependency) |
| three.js | `three`, `@types/three` | 0.187.0 | sha512-Qiu7R+Cm…WY8etA== | MIT | (dependency) |
| mutracer | `@meimu/mutracer` (npm name of github.com/pissang/mutracer; the bare name `mutracer` is not on npm) | 0.3.1 (2026-11-05) | sha512-QepLBd4w…4VGg== (no gitHead published) | MIT (Yi Shen) | `mutracer` |

- **Licence rule (CLAUDE.md): we run and measure mutracer; none of its code is copied into our packages.** The adapter
  (`bench/adapters/mutracer/index.ts`) only calls its public API. Same for three-gpu-pathtracer.
- mutracer pulls `oidn-webgpu` (Apache-2.0 weights) as a dependency; we never attach a denoiser, so it is not loaded.
- Re-pin with `pnpm add -D --save-exact --filter @pathtracer/bench <pkg>@<version>`; update `version.ts` in
  `tgpt-common/` and `MUTRACER_VERSION` in `mutracer/index.ts` (they are recorded in every result row via `info.commit`).

## How scenes get in

`node bench/adapters/export-scenes.ts [scene ...]` (default: every scene in `bench/scenes/index.ts`, Tier 0 and Tier 2; `file` meshes need `node bench/assets/fetch.ts` first) writes `bench/assets/generated/web/<scene>.glb` (+ `.json` with a hash of the
SceneDesc and the exporter's approximation list) through `bench/exporters/gltf.ts`; `matrix.ts` does it first. The
exporter runs on Node only (`Buffer`, filesystem for `file` meshes), so the page fetches the files; a changed scene
fails loudly as stale (hash of `JSON.stringify(scene)`); an edited *exporter* is not detected, re-run the export.
The glb goes through three's `GLTFLoader`; the camera is built from the scene's lookAt (exact transform, vertical fov);
the environment is a float equirect `DataTexture` (uniform: 4x2 constant; hdri: rows reversed and columns shifted a quarter
turn to three's mapping, exact for 512 wide).

## Run

```
node bench/adapters/matrix.ts --spp 256 --snapshots 16,64 --name tier0-256    # all adapters x Tier 0 -> bench/results/reports/tier0-256/report.json
node bench/adapters/matrix.ts --scenes sponza-white --spp 256 --snapshots 16,64 --name sponza-256         # Tier 2 (also: --scenes all)
node bench/adapters/matrix.ts --scenes cornell --spp 1024 --snapshots 4,16,64,256 --repeat 3 --name p0-baseline   # --repeat N: runner repeats per cell; --append adds cells to an existing report
node bench/adapters/baseline.ts --report p0-baseline [--csv]                   # error-vs-time, fits, time to relMSE targets (P0.9; see bench/results/BASELINE-P0.md)
node bench/runner/run.ts --page competitor --args renderer=mutracer scene=cornell spp=256 snapshots=16,64
```
Page args: `renderer`, `scene`, `spp`, `snapshots`, `budget` (tgpt path steps per call, default 250000), `emitters=rect|mesh`
(tgpt), `bounces` (override, calibration), `force=1` (render a scene the adapter does not list as exact).
View: `pnpm --filter @pathtracer/bench viewer`, open `?report=/results/reports/<name>/report.json`.

## What is turned off (raw path tracing only)

| | three-gpu-pathtracer | mutracer |
|---|---|---|
| denoiser / upscaler | none attached | none attached |
| tone mapping, exposure | linear accumulation target read back (not the canvas); `renderToCanvas = false` | `readPixels('hdr')`: original linear samples, **times the returned alpha** (it is straight colour; P0.26) |
| preview / low-res / fade | `dynamicLowRes = false`, `renderDelay = 0`, `fadeDuration = 0`, `minSamples = 0`, `adaptiveDynamicRes = false` | fixed-size `Render`, `converge()` (no preview Render, no RenderLoop) |
| adaptive budgets | `adaptiveFrameBudget = false` (frame budget fixed at 250000 path steps) | no `targetBatchTimeMs`, no `adaptiveSampling`, `samplesPerBatch` 8 |
| firefly filters | `filterGlossyFactor = 0`, `clampDirect = clampIndirect = 0` (default clampIndirect is 10) | `clamping = 0` |
| other shading tricks | | `bevel = false` (library default true; no effect on our glTF materials, which have no `bevelSize`: Cornell is bit-identical with it on) |
| temporal | none exists | none exists |
| **cannot be turned off** | alpha-weighted blending (forced opaque with clear alpha 1); periodic internal sample-count tally (a 20-byte readback every 250 ms during `renderSample`); multiple-scattering compensation on rough metals/dielectrics | 4 RIS direct-light candidates per vertex; multiple-scattering compensation; 16-pixel render alignment (our sizes are multiples of 16) |

Sampler: Sobol (library default) in both. MIS on. Bounce mapping (calibrated, see DIVERGES): tgpt `maxBounces` = maxDepth
(+1 when emissive meshes are found by BSDF sampling only), mutracer `bounces` = maxDepth - 1 (min 1, max 50).

## Timing method

tgpt: `renderSample()` called back to back, no readback in the timed region. The call count that reaches the spp is found in
an untimed calibration pass (sample counts polled after every call; snapshot images are read there), then the timed pass
replays that count and waits for `queue.onSubmittedWorkDone()`; the result is read after the clock stops and verified to be at
spp. mutracer: the awaited `converge({samples})`, wall clock, readback afterwards; every snapshot is an independent converge.
Both: wall clock only (no GPU timestamps; `trackTimestamp` would put a readback into the loop). First-call compile is reported
as `compile_ms`, not in `wall_ms`. Frame budget 250000 vs 4e6 on Cornell changed wall time by < 10%: GPU-bound, not JS-bound.

## P0.26: why mutracer stopped converging on Cornell (adapter bug, fixed)

**Cause.** `Render.readPixels('hdr')` returns straight (unassociated) colour: RGB is the mean over the samples that hit
something, alpha is the hit fraction (accumulation `outColor[idx] += liAndAlpha` in
`lib/renderer/wgsl/rayTrace.compute.js` ~l.1100, alpha measured through the adapter; docs/api.md "Pixel outputs"). The Cornell box is open toward the camera,
so the 1-px silhouette against the empty (black) background is partly covered; the adapter dropped alpha, so those pixels
read as fully lit surface. The references (and tgpt) average the black misses in. This is not an adapter-vs-library
bias in shading: it is a pure edge-coverage error.

**Evidence** (Cornell 512x512, Mitsuba RGB 4096 spp reference, `p0-26-mutracer-cornell` report, viewer
`?report=/results/reports/p0-26-mutracer-cornell/report.json`):
- Spatial pattern: 32 px block means of mutracer / Mitsuba are within 0.0 to 0.6% everywhere except the outer ring (+2.5 to
  +6%). Profiles show the silhouette pixels full where Mitsuba and pbrt (which agree with each other and with the analytic
  edge, 8.82 px from the top) have partial coverage: top row 8 reads 0.0370 vs 0.0063 (coverage 17%). Features at
  depth 1.36 m (back-wall corners) are within 0.1 px, only the z = 0 front edges are "bigger", so it is not a camera/fov error.
- Probe: alpha of that pixel is 0.169 (analytic 0.18). Interior pixels have alpha 1.
- 91% of the 4096 spp relMSE sits in the outer 12 px: relMSE 5.7e-4 all pixels, 5.2e-5 with a 12 px crop. The interior
  was already converging (1.5e-4 at 1024 to 5.2e-5 at 4096).
- Fix (`premultiply`, default on; `mt_premul=0` reproduces the old read): relMSE vs Mitsuba 5.2e-4 / 1.4e-4 / 4.8e-5 at
  256 / 1024 / 4096 spp (before: 1.1e-3 / 6.8e-4 / 5.7e-4), about 1/spp again down to the reference's own noise. Mean
  ratio 0.9996 (was 1.0008).
  Quad-light (relMSE 1.4e-6, unchanged) and scenes with a visible environment (alpha 1 everywhere, bit-identical)
  are not affected.

**A/B through page args** (`mt_*`, Cornell, premultiplied, 1024 spp, relMSE vs Mitsuba; default 1.38e-4):

| Hypothesis | Setting | Result |
|---|---|---|
| bevel / rounded edges | `mt_bevel=1` | bit-identical to default (our materials have no `bevelSize`); default config has it off |
| 4 RIS candidates | `mt_candidates=1` (settable: `Render.directLightCandidates`, 1..16; V13 said "no switch", wrong) | 1.46e-4, same mean: not a bias, slightly noisier |
| sampler | `mt_sampler=random` | 2.6e-4 (Owen-scrambled Sobol is the better default), same mean |
| spectral/dispersion | `mt_dispersion=1` | bit-identical (only dispersive materials use it; mutracer is RGB, "not spectral rendering" per docs/technical/continuous-spectrum.md) |
| bounce mapping V10 / 50-event cap | `bounces=16` and `8` (default 31, capped at 50 events) | 16: identical to 5 digits; 8: mean -0.17% (cap not reached at 31) |
| light units/area, two-sided emitter, normals, ray epsilon, MS compensation | no A/B needed | mean within -0.06/-0.03/-0.02% of Mitsuba RGB per channel; Mitsuba RGB itself is +0.7% above pbrt/Mitsuba spectral on Cornell (V12-style RGB-vs-spectral, blue +2%), mutracer is RGB too, so it tracks Mitsuba RGB |

Tier 0 isolation: `quad-light` (1024 spp) relMSE 1.1e-6 vs Mitsuba, relBias -2e-4: direct lighting, light units and area
exact. `furnace-diffuse` reads 0.50063 (+0.13% on albedo 0.5, convex, bounces 1 to 3) and nothing we tried moves it
(RIS 1: 0.50065, random: 0.50068, dispersion 1: 0.50063, env map 64 wide 0.50058, 512 wide 0.50047): left as a small mutracer
property (V22), not related to the Cornell plateau.

The P0.9 baseline table (BASELINE-P0.md) predates the fix: its mutracer Cornell error rows and the "plateau" mark are stale
and need the P0.25 quiet-machine rerun.
