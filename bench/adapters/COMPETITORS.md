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
| tone mapping, exposure | linear accumulation target read back (not the canvas); `renderToCanvas = false` | `readPixels('hdr')`: original linear samples |
| preview / low-res / fade | `dynamicLowRes = false`, `renderDelay = 0`, `fadeDuration = 0`, `minSamples = 0`, `adaptiveDynamicRes = false` | fixed-size `Render`, `converge()` (no preview Render, no RenderLoop) |
| adaptive budgets | `adaptiveFrameBudget = false` (frame budget fixed at 250000 path steps) | no `targetBatchTimeMs`, no `adaptiveSampling`, `samplesPerBatch` 8 |
| firefly filters | `filterGlossyFactor = 0`, `clampDirect = clampIndirect = 0` (default clampIndirect is 10) | `clamping = 0` |
| other shading tricks | | `bevel = false` (default true: rounds facet edges) |
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
