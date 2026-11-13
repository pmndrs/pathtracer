# pathtracer: rules for every agent

Repo: github.com/pmndrs/pathtracer. Publishes **@pmndrs/raytracer** (trace core, WGSL prefix `rt`) and
**@pmndrs/pathtracer** (PBR integrator, WGSL prefix `pt`). Rules: speed above all, then quality, then DX.

## Read first, write last

Read these before starting:
- [STATUS.md](STATUS.md) — where we are, in plain words.
- [TASKS.md](TASKS.md) — what's open, who owns it.
- [DECISIONS.md](DECISIONS.md) — what's settled; don't re-litigate.
- [PHASES.md](PHASES.md) — phases and exit gates.
- [DIVERGES.md](DIVERGES.md) — where we split from a source, and why.
- Background: [spec/SPEC-draft.md](spec/SPEC-draft.md), [research/INDEX.md](research/INDEX.md), research/notes/.

When you finish, update TASKS.md (mark done, add follow-ups) and add anything that would have saved you time to
**Traps** below. Update STATUS.md only if the phase picture changed. Record every departure from a source
(paper, sibling, competitor, native renderer) in DIVERGES.md with the reason: platform bug, performance, licence or
simplification.

## Working rules

- **Build both, then decide.** When two approaches compete, implement both behind a setting and measure. Early on, more
  options is better. Decisions go to DECISIONS.md only after an A/B.
- **A/B by default.** Every experiment must produce two outputs the bench viewer can compare side by side (linear HDR
  images + timings), not a prose claim.
- **Questions for the maintainer:** give a suggestion plus pros and cons, and the A/B that would settle it.
- **No timings from a loaded machine.** Check load (`sysctl vm.loadavg`, GPU Chrome processes) before any timed
  run; numbers taken under high load are not reported as results, only as provisional with the load recorded.
- **Don't get trapped on timing.** Build first. Timing that decides a default or passes a gate goes in the
  [quiet pass](docs/QUIET-PASS.md) queue, which the maintainer runs from a CLI session with every app closed; until
  then defaults stay on the current suggestion and best-effort numbers are labelled as such.
- **Correctness before timing.** CPU oracle or analytic expectation first; timing numbers name the device, browser and
  run conditions, and are compared as paired ratios on one machine, never across machines.
- **No readback inside timed regions.** No `mapAsync` in a frame loop.
- **Sibling repos** (`~/Developer/bvh`, `downsampler`, `Denoiser`, `upscaler`, `reflections`, `webgpu-ddgi`): follow
  their own CLAUDE.md/HANDOFF. Read-only unless the task says otherwise; when you do work there, keep their
  TASKS/DECISIONS/DIVERGES docs current the same way.

## Licences and clean room

- MIT/BSD/Apache/zlib code may be ported with attribution (add to THIRD_PARTY.md when it exists).
- **mutracer**: read, cite, compare; never copy code.
- **NVIDIA RTX SDK repos** (RTXPT, RTXDI, RTXGI, SHaRC, NRD): no code. NVIDIA papers, books (Ray Tracing Gems),
  talks and blog posts are fine.
- **NRD**: never open its code or any port of it (Denoiser clean-room rules,
  `~/Developer/Denoiser/docs/research/rt-denoiser/RULES.md`). Native NRD runs only via the Denoiser Windows harness;
  we consume its output images.
- GPL / non-commercial (rindr): ideas only. No licence file: read only.
- Third-party clones in `research/sources/` are read-only. Never build or run them. Native tools we build (pbrt-v4,
  Mitsuba) get their own checkout under `.cache/native/` (git-ignored).

## Conventions (from spec §3; the single source is packages/raytracer/src/contract/)

- Right-handed, +Y up, metres. View space looks down −Z. Column-major matrices.
- Linear scene-referred HDR inside; f32 accumulation; f16 only for storage/AOVs.
- `viewZ` negative in front. Motion = NDC delta **current − previous**, +y up, jitter-free.
- Normals world-space; adapters convert (OIDN aux wants view-space).
- Jitter in pixels, x right / y down, [−0.5, 0.5]; exactly one jitter owner per pipeline.
- Image row 0 = top. Interchange images are **PFM** (float32, linear); PNG only for previews.
- WGSL prefixes: `bvh*` belongs to @pmndrs/bvh, `gi*` to @pmndrs/gi, `rt*` raytracer, `pt*` pathtracer.

## Tooling

pnpm workspace, TypeScript (strict, ESM), Node 22 (runs `.ts` directly via type stripping), Vitest, Vite.
`pnpm test` (unit, no GPU), `pnpm typecheck`. GPU pages run in headless Chrome via `bench/runner` (port of bvh's
`tests/gpu` runner).

## Traps (add to this list)

- Chrome quantizes `timestamp-query` to ~100 µs unless developer features are enabled
  (`--enable-webgpu-developer-features`; the runner sets it). On Chrome 157 / Apple it measured 1 ns steps even
  without the flag. Run `--page timer-calibration` once per session anyway. In-page `performance.now()` is 100 µs.
- Without developer features Chrome hides `adapter.info.description`; on Mac `description`/`device` are empty anyway.
- Node 22 type stripping: `.ts` imports need explicit extensions, and constructor parameter properties
  (`constructor(private x)`) are rejected. `enum` and `namespace` too.
- WGSL uniformity: `subgroupBallot` etc. inside non-uniform control flow fails to compile. Keep subgroup ops in
  uniform flow.
- Prepending `enable` lines shifts WGSL error line numbers; `wgslEnableLines` reports the count.
- pbrt-v4: `path` integrator can't disable RR (we patch it, V4); fov applies to the shorter image axis; only square
  equal-area octahedral env maps; `remaproughness false` + alpha. White L=1 reads exactly 1.0 with default film
  settings: don't set sensor, iso or white balance.
- Mitsuba 3: `max_depth` = our `maxDepth + 1`; BSDFs are one-sided (wrap in `twosided`); spectral variants break
  `rgb` k=1000, export grey eta/k as constant `spectrum`; stdout ends with an ANSI escape.
- Smooth dielectric furnace needs maxDepth ≈1000 (100 loses 0.1%).
- Reference renders are git-ignored (22 MB); re-render with `pnpm --filter @pathtracer/bench references`
  (Cornell 512² takes ~18 min pbrt / ~9 min Mitsuba at 4096 spp on M5 Pro).
- JS FLIP at 4K takes ~10 s (metrics ~2.6 s); the viewer runs them in a worker.
- Non-finite pixels are excluded from every metric and counted; the viewer shows them magenta.
- Metal fma contraction broke watertight triangle tests in bvh (fixed there as `bvhEdgeFunction`). Any new
  triangle/box test must pass the bvh watertight oracle on Apple.
- WGSL has no single-dispatch "last workgroup" handoff guarantee (downsampler learned this). Use separate dispatches.
- Windows DXC compile time blows up with shader size; keep generated kernels under ~50 KB WGSL.
- Bindless (`chromium-experimental-sampling-resource-table`) is textures-only and not on macOS yet.
- bvh's per-triangle material id is 16 bits (max 65534). Treat it as an index into our own table.
- bvh changes are only visible to consumers after `pnpm build` in ~/Developer/bvh (consumers read `dist/`).
- Mitsuba `rr_depth` = pbrt `rrdepth` + 1 (verified by a one-bounce furnace variance test); Mitsuba rejects `rr_depth` 0.
- Pick spp as 1024/4096: Mitsuba multijitter rounds to a square, pbrt zsobol to a power of two.
- Khronos Sponza ships glTF + `.bin` only (no glb). End walls at x = −11.4 / +10.4: a camera at x = −12.5 is outside
  and renders black. Fetch with `node bench/assets/fetch.ts` (pinned commit + sha256 in `bench/assets/MANIFEST.json`).
- Competitors: the glTF exporter is Node-only, so pre-export with `node bench/adapters/export-scenes.ts`
  (→ `bench/assets/generated/web/`); an edited exporter is not detected as stale, re-run it.
- three-gpu-pathtracer: `useMegakernel(true)` resets settings (set them after); rect lights are invisible to camera rays
  unless `visibleToCameraRays`; `clampIndirect` defaults to 10; alpha blending drops escaped samples unless the clear
  alpha is 1.
- This Mac is shared: other agents' headless Chrome and vitest jobs do not take the GPU lock. During P0.9 the load average
  was 97-779 with 20-39 GPU processes and repeats of one cell moved by up to 1.75x. Check `sysctl vm.loadavg` and
  `ps -eo command | grep -c type=gpu-process` before timing, quote min..max, never a single run.
- relMSE against a 4096 spp reference has a floor (Sponza: Mitsuba noise alone about 2e-3, pbrt-vs-Mitsuba 4.1e-3, Cornell
  4.4e-4). Targets under 2x the floor are not measurable; `bench/adapters/baseline.ts` marks them. A curve that stops
  following 1/spp (mutracer on Cornell) is a bias floor: do not extrapolate it.
- A page that runs one long synchronous loop (tgpt's timed replay, minutes on Sponza) cannot answer DevTools: `Runtime.evaluate`
  timed out after 30 s and failed the run. The runner now treats that as busy (V17); a new adapter should still await
  something between chunks.
- tgpt and mutracer use a deterministic Sobol sequence: repeats share one noise realisation, so error has no run-to-run spread,
  only time does. tgpt time to the first snapshot is about 2x per-spp price (fixed per-call cost), so fitted k is slightly above 1.
- `nohup sh -c "a; b" &` chains: killing the `sh` leaves the running child; wait for its pid (`kill -0`) and queue a new chain.
  macOS has no `timeout`; bound loops with a counter.
- The Mac is shared and other agents' Chrome jobs don't take the GPU lock. The runner samples load average and foreign
  Chrome GPU processes before every run (`bench/runner/load-check.ts`), waits `--load-wait` (120 s), then marks rows
  `valid=0` with note `loaded` (`--strict-load` fails). Quote timings only from rows without `loaded`. Windows has no
  `loadavg`, so it uses the CPU busy fraction over 1 s and PowerShell/CIM for process command lines (`tasklist` shows none).
  Count only browser GPU processes (`isBrowserGpuProcess`): Electron/CEF apps (Spotify, Discord, Notion, Creative Cloud,
  the Claude app) each run a `--type=gpu-process` helper too, and counting them marked every run on a quiet Mac `loaded`.
- FLIP (P0.14): HDR-FLIP derives its exposure range from the REFERENCE only (start = log2(xMax/Ymax), stop = log2(xMax/median)).
  A flat or dark reference gives 2 identical exposures and a firefly merely saturates (error ~0.07); a reference with a
  lamp spans the low exposures that expose it. Its tonemap is its own (ACES with 0.6 cancelled), there is no EV, and its mean
  is always higher than LDR-FLIP's (about 1.5-2x on Cornell): quote the two as separate metrics, never mixed. Mean FLIP of
  either kind barely moves for 13-52 fireflies in 262k pixels (0.0236 -> 0.0253); check relMSE, firefly % and max FLIP too.
- FLIP on PNG input needs the sRGB decode; our metric takes tonemapped radiance (display-linear). Feeding encoded values
  without the decode inflates the mean (0.1737 vs 0.1597). `validate-flip.ts` needs `.cache/flip-validation/` (clone of
  NVlabs/flip + a venv with pip `flip-evaluator`; `work/prep.py` writes the PFMs). Run Python with `-I`.
- HDR-FLIP costs one LDR-FLIP per exposure (Cornell: 8, NVlabs pair: 14): ~1.1 s per exposure at 1280x720 single thread, 38 s
  for the 1080p NVlabs pair serial, 8 s on worker_threads. Use `flipHdrParallel` (Node) and never in a frame loop; the viewer computes it on demand (H).
- `pnpm typecheck` can fail on another agent's half-written files (seen: `packages/raytracer/src/trace/index.ts` importing a
  missing `./batch.ts`); check `git status` before chasing it.
- Blue noise (P0.17): a vec2 made of two independent scalar void-and-cluster channels is only weakly blue for a 2D
  integrand (a threshold of a combination keeps ~34% of white's low-frequency error); use the jointly optimised `gf`
  textures. A 2D texture animated by the golden-ratio shift is a strong baseline that our `st` did not beat.
- Never decode the 16-bit blue-noise PNGs through a canvas or `createImageBitmap` (8-bit); use `decodeBlueNoisePng`.
  Their `.f32` twins are git-ignored; `pnpm bluenoise:generate` rebuilds them (~30 s).
- Subagents cannot `Write` report-style `.md` files (harness hook): hand the text to the lead to place.
- bvh's CPU walk (`createCpuTracer`, `CpuSceneBvh.raycast`) does f64 arithmetic on the f32 scene; the GPU walk is f32.
  Never expect bit equality near edges: `bench/pages/trace-core.ts` classifies ties, grazing contacts and t position
  error (V27). Two CPU walks over different trees also disagree on rays that merely touch a convex edge. No CPU walk
  can be bit-exact: GPU `1/x`, division and `sqrt` are up to 2 ulp off and `dot()` is fused (bvh measured on D3D12).
- A ray aimed at a convex edge or corner of a solid from outside only *touches* it: hit and miss are both right. Only a
  miss where the ray *crosses* a shared edge (all triangles around it face the ray the same way) is a watertight slip.
- A new `link:` dependency in `bench/package.json` needs `pnpm install --offline` before typecheck or Vite see it.
- `trace-core` (all four scenes, 590k rays, CPU oracle in the page) takes ~4 s; `scenes=edges,cornell,quad-light
  sceneTrees=0` for a quicker loop. A page that hangs until the runner timeout is usually an un-pumped `SceneTrees`.
- `SceneTrees.whenBuilt()` never resolves on its own: builds start in `update(encoder)`. One update first is enough:
  `frame(); await trees.whenBuilt(); frame();` (bvh, X2 item 4).
- bvh build workers under Vite: with bvh linked, its realpath must be in `server.fs.allow` (done in
  `bench/runner/server.ts`), or Vite answers the worker file with 403 and builds fall back inline; npm installs also
  need `optimizeDeps.exclude: ['@pmndrs/bvh']`. Check `notes.workers.*.inlineReason` in trace-core's result to catch a
  silent inline fallback.
- Sponza has duplicate triangles (same 3 vertices). Rays through them hit both; GPU and CPU (or two trees) legitimately
  name different ones (`primary_tie` in trace-core). They also double-count in area/power sums (P1.10).
- Denoiser's `tools/gpu-lock.sh` (used by bvh sessions) never clears a lock directory whose owner died; our node
  runner (`bench/runner/gpu-lock.ts`) does. After killing a runner, check `/tmp/rt-gpu.lock/owner` and remove it if
  that pid is gone, or every shell-locked job waits forever.
- Short GPU passes (~5 ms) on a Mac in use read up to ~2.4x slow in bursts (other apps' GPU work lands inside the pass,
  correlated across kernels). Time with many interleaved repeats and take a low percentile (trace-timing uses p10 of 30).
- pbrt-v4's VNDF sampler picks its tangent by `wh.z < 0.99999`; GPU `normalize()` rounding flips that basis near the
  threshold, so GPU and CPU `u -> wi` differ for narrow lobes while the distribution is identical. Compare distributions,
  or skip the threshold neighbourhood (`bench/pages/bsdf-check.ts` does).
- A rough dielectric's `f` in radiance transport includes 1/eta'², so its albedo is R + T/eta'² (below 1 from outside,
  above 1 from inside). Check energy conservation with `transport: 'importance'` (R + T ≈ 1).
- ∫pdf of a rough conductor/dielectric lobe is 1 minus the chance the sample fails (below the horizon, TIR), not 1.
  Brute-force integrating a GGX lobe on a uniform (θ, φ) grid fails at roughness 0.05: use polar coordinates around the
  mirror/refracted direction with a radial warp (`bruteForceAlbedo`).
- WGSL reserves identifiers you wouldn't expect (`meta`, `filter`, `target`, `set`, `ref`, `from`, `layout`, `self`,
  `type`, `final`, `partition`, `pass`, `patch`, `module`, ...). Using one only fails on the GPU; run
  `packages/raytracer/src/scene/scene-wgsl.test.ts` (it carries the list) after adding names to generated WGSL.
- bvh's walk WGSL is ~40 KB, mostly comments: a module with it plus the scene copy passes the 50 KB cap unless comments
  are stripped (`rtModuleWgsl({ scene })` strips by default and keeps line breaks, so error line numbers still match).
- WGSL `unpack2x16unorm` isn't guaranteed bit-exact and GPU division isn't correctly rounded: decode unorm16 as
  `f32(q) * fround(1/65535)` (a multiply), which stays bit-identical to the TS twin.
- `rtModuleWgsl` already runs the caps preprocessor and puts the `enable` lines first. Appended WGSL must not go through
  `buildWgsl` again: a second `enable` block mid-file fails to compile.
- One uniform value per dispatch inside one command encoder needs a ring of dynamic-offset slots: repeated
  `queue.writeBuffer` to the same range before one `submit` leaves only the last value.
- Sun cone sampling in f32: `1 - cos(0.00465)` keeps about 3 digits. Use the stored solid angle and test containment
  with `|d - s|² <= 2(1 - cos)`.
- Bench adapters imported from Node tests must lazy-import `runner/page-client.ts` (it reads `location` at load).
- `matrix.ts` appends its CSV rows to the shared `competitor.csv`, also for our own renderer (`pt-megakernel`).
- Conductor furnace: pbrt (0.69285) is the target; Mitsuba reads 0.69040 (its separable Smith G, D15). A Mitsuba plateau
  there is its bias, not ours.
- Own renderer runs: `node bench/adapters/matrix.ts --renderers pt-megakernel --scenes ... --spp 1024 --snapshots
  16,64,256 --name <n> [--extra pt_sampler=pcg]`, then `node bench/adapters/pt-agreement.ts --report <n>`.
