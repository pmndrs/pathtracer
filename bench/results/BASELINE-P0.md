# P0 baseline: time to equal error (web competitors, Apple M5 Pro)

> **PROVISIONAL — do not quote the timings.** These runs were taken while the Mac was under heavy load from other
> jobs (load average 97–779, 20–39 GPU Chrome processes outside the GPU lock; one cell moved 1.75× between passes).
> The maintainer ruled (2026-11-13) that timings from a loaded machine are not trusted. The error-vs-spp data,
> image means and the reference noise floors stand; every ms, s and min below is to be replaced by the quiet-machine
> rerun (TASKS P0.25, with the load check P0.30).
>
> **Also stale: the mutracer Cornell error rows and its "plateau" mark.** They came from an adapter bug (straight
> alpha read back without premultiplying; fixed in P0.26). With the fix mutracer's Cornell relMSE is 1.4e-4 at 1024 spp,
> not 6.8e-4. The rerun regenerates these rows.

First baseline table (P0.9, the P0 exit gate). Question answered: **how long does each web path tracer need to reach a
given error against the Mitsuba reference?** Numbers are paired within this machine only; do not compare with other
machines or with the Windows runs.

## Setup

| | |
|---|---|
| Device | Apple M5 Pro (`metal-3`), 64 GB, macOS 26.6.2 |
| Browser | Chrome 157.0.8103.52, headless, WebGPU, `--enable-webgpu-developer-features` (runner default) |
| Commit | `f3c1421` plus the uncommitted working tree of this task (`bench/adapters/{baseline,matrix,export-scenes}.ts`, adapter flags, `bench/runner/session.ts`, `bench/pages/competitor.ts`) |
| Libraries | three-gpu-pathtracer 0.0.28 (`tgpt-wavefront`, `tgpt-megakernel`), mutracer 0.3.1; three 0.187.0 (see `bench/adapters/COMPETITORS.md` for what is switched off) |
| Scenes | `cornell` (Tier 0, 512x512, maxDepth 32, no Russian roulette) and `sponza-white` (Tier 2, 1280x720, 262,266 triangles, sun + uniform sky, maxDepth 8, RR from vertex 4) |
| References | Mitsuba 3 `llvm_ad_rgb` 4096 spp (headline, D15) and pbrt-v4 4096 spp; relMSE eps 1e-2 (`bench/metrics`) |
| Reference noise floor | pbrt-vs-Mitsuba relMSE: **4.4e-4 on Cornell, 4.1e-3 on Sponza** (computed by `baseline.ts`) |
| Run conditions | GPU lock on (runner default), `--repeat 3` per cell (3 fresh page loads back to back), no readback inside any timed region, tgpt `emitters=rect`, wall clock only |
| Raw data | `bench/results/competitor.csv` (runner rows, run 1..3 + median, plus `run=summary` rows from `baseline.ts --csv`); runner folders in `bench/results/incoming/` (git-ignored): 20261113T084608Z (Sponza wavefront), 085848Z (Sponza megakernel), 091842Z / 092025Z / 092743Z (Cornell wavefront / megakernel / mutracer), 093023Z (Sponza mutracer), 083930Z / 084108Z (Cornell tgpt mesh mode) |
| Viewer | `pnpm --filter @pathtracer/bench viewer`, open `?report=/results/reports/p0-baseline/report.json` (exact cells) and `?report=/results/reports/p0-baseline-tgpt-mesh/report.json` (inexact mesh-mode cells, D19/D20) |
| Reproduce | `node bench/adapters/matrix.ts --scenes cornell --spp 1024 --snapshots 4,16,64,256 --repeat 3 --name p0-baseline` (and `--append` with `--scenes sponza-white ... --spp 1024`, megakernel `--spp 512`), then `node bench/adapters/baseline.ts --report p0-baseline [--csv]` |

Snapshot spp: Cornell 4, 16, 64, 256, 1024 for all three. Sponza 4, 16, 64, 256, (512), 1024 for mutracer and tgpt-wavefront;
**tgpt-megakernel is capped at 512 spp** (3 repeats took about 20 min on the loaded machine; 1024 would have been about 40 min,
over the 30 min budget). Time to a snapshot is the wall clock from the start of that render (tgpt: one continuous run with a GPU
sync at each snapshot; mutracer: an independent converge per snapshot).

## Table (headline: relMSE against Mitsuba 4096 spp)

Time to reach the target is a median over the 3 repeats with min..max in brackets; no mark = interpolated (log-log) between two
snapshots that bracket the target. `~` = extrapolated from the fit past the last snapshot (none left in the exact cells).
"not measurable" = target below 2x the reference noise floor (Sponza 1e-3 is a quarter of the 4.1e-3 floor).

| Scene | Renderer | ms/spp (min..max) | fit k (error ~ t^-k) | Time to relMSE 1e-2 | Time to relMSE 1e-3 | relMSE at the last snapshot (Mitsuba / pbrt) |
|---|---|---|---|---|---|---|
| cornell | mutracer | 8.30 (8.2..8.8) | 0.90 | **133 ms** (130..163 ms) | 2.8 s (2.2..3.2 s) † plateau | 6.8e-4 / 1.1e-3 @1024 |
| cornell | tgpt-wavefront | 13.3 (12.6..13.8) | 1.11 | 242 ms (220..252 ms) | **2.1 s** (1.9..2.1 s) | 1.5e-4 / 5.6e-4 @1024 |
| cornell | tgpt-megakernel | 64.9 (64.6..74.1) | 1.13 | 1.1 s (1.1..1.3 s) | 9.2 s (8.8..10.5 s) | 1.4e-4 / 5.5e-4 @1024 |
| sponza-white | mutracer | 76.6 (74.0..95.8) | 0.95 | 1.2 min (1.2..1.5 min) | not measurable | 9.4e-3 / 9.6e-3 @1024 |
| sponza-white | tgpt-wavefront | 116.5 (116.1..119.3) | 0.97 | **1.1 min** (1.1..1.1 min) | not measurable | 6.3e-3 / 6.4e-3 @1024 |
| sponza-white | tgpt-megakernel | 375.5 (372.0..387.5) | 1.00 | 2.7 min (2.7..2.8 min) | not measurable | 8.7e-3 / 8.9e-3 @512 |

Against pbrt (same snapshots; floor-limited the same way): 1e-2 / 1e-3 on Cornell: mutracer 141 ms / "~5.4 s" (extrapolated and
plateauing, treat as not reached), wavefront 254 ms / 3.8 s, megakernel 1.2 s / 15.5 s. Sponza 1e-2: mutracer 1.3 min, wavefront
1.1 min, megakernel 2.8 min; 1e-3 not measurable. The pbrt column differs from the Mitsuba one by the reference differences only
(Cornell relMSE pbrt-vs-Mitsuba 4.4e-4, the 1e-3 target is only 2.3x that).

† mutracer on Cornell stops following 1/spp: 256 to 1024 spp lowers the error only 1.07e-3 to 6.8e-4 (a 4x drop was expected).
It has a residual of about 6e-4 against Mitsuba (mean ratio 1.0006 to 1.0009, 40-pixel block means up to 3.6% off, vs 0.4% for the
tgpt images), so its 1e-3 time is a bracketed interpolation right at the knee, not a convergence statement.

Error vs time (median wall time, relMSE vs Mitsuba):

| Scene / renderer | 4 spp | 16 | 64 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| cornell mutracer | 31 ms, 5.0e-2 | 114 ms, 1.2e-2 | 446 ms, 2.9e-3 | 2.3 s, 1.1e-3 | - | 8.5 s, 6.8e-4 |
| cornell tgpt-wavefront | 129 ms, 2.1e-2 | 334 ms, 6.9e-3 | 1.2 s, 1.8e-3 | 4.7 s, 4.5e-4 | - | 13.6 s, 1.5e-4 |
| cornell tgpt-megakernel | 260 ms, 5.3e-2 | 1.0 s, 1.1e-2 | 4.1 s, 2.3e-3 | 16.6 s, 5.3e-4 | - | 1.1 min, 1.4e-4 |
| sponza mutracer | 261 ms, 2.2 | 1.1 s, 0.52 | 5.6 s, 0.13 | 21 s, 3.3e-2 | - | 1.3 min, 9.4e-3 |
| sponza tgpt-wavefront | 491 ms, 1.1 | 1.9 s, 0.30 | 7.7 s, 7.5e-2 | 30.5 s, 2.0e-2 | 1.0 min, 1.1e-2 | 2.0 min, 6.3e-3 |
| sponza tgpt-megakernel | 1.5 s, 1.1 | 6.0 s, 0.25 | 23.6 s, 6.0e-2 | 1.6 min, 1.6e-2 | 3.2 min, 8.7e-3 | - |

### tgpt emissive-mesh mode (Cornell only; INEXACT, D19/D20, report `p0-baseline-tgpt-mesh`)

This is what a plain glTF user gets: the light is an emissive mesh found by BSDF sampling only (no next-event estimation). Sponza has
no emissive mesh, so rect and mesh are the same there.

| Renderer | ms/spp | fit k | Time to 1e-2 | Time to 1e-3 | relMSE @1024 |
|---|---|---|---|---|---|
| tgpt-wavefront (mesh) | 13.7 (12.2..14.6) | 1.02 | ~22 s (extrapolated) | ~3.5 min (extrapolated) | 1.5e-2 |
| tgpt-megakernel (mesh) | 46.0 (45.0..48.2) | 1.07 | ~1.1 min (extrapolated) | ~9.1 min (extrapolated) | 1.4e-2 |

Against exact rect lights: about 90x (wavefront) and 60x (megakernel) longer to the same 1e-2 on Cornell, at about the same per-spp price.

## Takeaways (one line per scene)

- **Cornell (512x512):** mutracer reaches relMSE 1e-2 first (133 ms, 1.8x ahead of tgpt-wavefront; per-sample it is cheapest at 8.3 ms
  but noisier), but it has a bias floor near 6e-4 and tgpt-wavefront is first to 1e-3 (2.1 s); tgpt-megakernel is 4.4x to 4.9x slower than
  wavefront at about equal noise per sample.
- **Sponza (1280x720, 262k triangles):** a dead heat between tgpt-wavefront (1.1 min to 1e-2) and mutracer (1.2 min): mutracer is
  1.5x cheaper per sample (76.6 vs 116.5 ms/spp) but its error per sample is about 1.7x higher; tgpt-megakernel needs 2.7 min
  (3.2x the price per sample of wavefront, 2.5x the time to 1e-2 because it is a bit less noisy per sample). Nothing gets to 1e-3 on a 4096 spp reference: the baseline is limited by
  the reference, not the renderers.

## What the P0.19 check found (before timing anything)

All three load the 262k-triangle glb through the shared exporter and the sun goes through `KHR_lights_punctual` (lux 1:1) with no
per-renderer fix needed. Means against the 4096 spp references (RGB ratio): Mitsuba 0.9978..0.9982 (-0.2%), pbrt 1.0000..1.0004
(pbrt itself reads -0.2% vs Mitsuba, the known spectral-vs-RGB bias). 40-pixel block means differ from Mitsuba by 0.5..0.6% rms
(max 2.4..3.4%) against 0.4% rms (max 2.4%) for pbrt vs Mitsuba, i.e. within the references' own spread. Bounce mapping V10
(`maxBounces` = maxDepth for tgpt without emissive meshes, mutracer maxDepth - 1) holds with maxDepth 8 on Sponza; two-sidedness is a
non-issue for a diffuse-only scene with flat normals. Recorded as V19.

## Caveats (read before quoting any number)

1. **The machine was shared and loaded**: during the runs the 1-minute load average was 97 to 779 (median 443) with 20 to 39 Chrome
   GPU processes, mostly other agents' headless Chrome / vitest jobs that do not take our GPU lock. Repeated cells under that load
   moved by up to 1.75x (the first Cornell pass, run before this one, gave 19.2 / 84.5 / 7.7 ms/spp for tgpt-wavefront /
   tgpt-megakernel / mutracer against 13.3 / 64.9 / 8.3 here; the first Sponza passes gave 167 ms/spp wavefront @512 spp, 471
   megakernel @256 spp, 109 mutracer). The table is the quietest complete pass; the spread in brackets is run-to-run only. Treat
   ratios of 1.3x or less as inside the noise, and rerun on a quiet machine (or with `--ab` ABBA across renderers) before
   deciding anything on a ratio that small.
2. **Not interleaved**: repeats of one cell run back to back, cells run one after another; the ABBA ordering the runner offers was not
   used across renderers.
3. **The error includes the reference's noise.** relMSE against a 4096 spp image cannot go below the reference's own noise
   (Mitsuba alone is about 2e-3 on Sponza, 4.1e-3 for the pbrt-vs-Mitsuba pair, 4.4e-4 on Cornell). The "time to 1e-2" on Sponza
   therefore waits for a test noise of about 8e-3, not 1e-2, and is a slight overestimate of the time to true error 1e-2.
   Targets under 2x that floor are not claimed (V18).
4. **Same noise every repeat**: both libraries use a deterministic Sobol sequence (tgpt `stableNoise`), so the 3 repeats share one
   noise realisation and error values have no spread; the bracketed spreads are timing only.
5. Fits use only snapshots above 2x the floor (3 to 5 points; r2 0.98..1.00). k above 1 on Cornell (1.11, 1.13) is fixed per-call
   cost at small spp (4 spp costs about twice the per-spp price in tgpt), not better-than-1/spp convergence.
6. tgpt snapshot times come from one run that syncs the GPU at each snapshot; the 4 spp point includes first-call warm-up. mutracer
   snapshots are independent converge calls (each restarts the sequence).
7. Wall clock only (no GPU timestamps, P0.21); tgpt does a 20-byte sample-count readback every 250 ms inside `renderSample` and
   mutracer takes 4 RIS candidates per vertex always; neither can be switched off (V13). tgpt's timed pass is a replay of an untimed
   calibration pass that finds the call count, so a tgpt cell costs about twice its timed length.
8. maxDepth 8 for Sponza maps to different bounce settings per renderer (V10); firefly clamps are off everywhere.
9. Cornell light is a quad: tgpt uses an exact rect light by default (D19), mutracer finds it as an emissive mesh with its own NEE.

## Follow-ups

See TASKS.md: P0.25 to P0.29 (quiet-machine rerun with ABBA, mutracer Cornell bias, megakernel 1024 spp, yield in the tgpt timed loop,
equal-noise reference).
