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
  [quiet pass](docs/internal/QUIET-PASS.md) queue, which the maintainer runs from a CLI session with every app closed; until
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
- GPL / non-commercial (rindr): ideas only. No licence file: read only. Exception: VkNRC may be built natively and its
  design followed closely for research track N (D28, D29); its source text still never enters this repo.
- Third-party clones in `research/sources/` are read-only. Never build or run them. Native tools we build (pbrt-v4,
  Mitsuba) get their own checkout under `.cache/native/` (git-ignored).

## Conventions (from spec §3; the single source is packages/raytracer/src/contract/)

- Right-handed, +Y up, metres. View space looks down −Z. Column-major matrices.
- Linear scene-referred HDR inside; f32 accumulation; f16 only for storage/AOVs.
- `viewZ` negative in front. Motion = NDC delta **current − previous**, +y up, jitter-free.
- Normals world-space; adapters convert (OIDN aux wants view-space).
- Jitter in pixels, x right / y down, [−0.5, 0.5]; exactly one jitter owner per pipeline.
- Image row 0 = top. Interchange images are **PFM** (float32, linear); PNG only for previews.
- WGSL prefixes: `bvh*` belongs to @pmndrs/bvh, `gi*` to @pmndrs/gi, `rt*` raytracer, `pt*` pathtracer. `nn*` is the private research MLP
  (`packages/mlp`, track N).

## Tooling

pnpm workspace, TypeScript (strict, ESM), Node 22 (runs `.ts` directly via type stripping), Vitest, Vite.
`pnpm test` (unit, no GPU), `pnpm typecheck`. GPU pages run in headless Chrome via `bench/runner` (port of bvh's
`tests/gpu` runner).

## Developing against local siblings

`@pmndrs/bvh` and `@pmndrs/downsampler` (packages/raytracer, bench) are installed from npm. To work against a checkout, add an
override to `pnpm-workspace.yaml`, run `pnpm install --offline`, and do not commit it:

```
overrides:
  "@pmndrs/bvh": link:../bvh
  "@pmndrs/downsampler": link:../downsampler
```

(paths are relative to the workspace root; `pnpm build` in the sibling after each change, consumers read its `dist/`). `bench/runner/server.ts`
allows the linked realpath in `server.fs.allow` and always sets `optimizeDeps.exclude: ['@pmndrs/bvh']`, so build workers load either way.
To go back: restore `pnpm-workspace.yaml` and `pnpm install --offline` (or online). Verified 2026-11-19: with the override the links point at
`../../../../bvh` and typecheck / pages see the checkout; without it they point into `node_modules/.pnpm/@pmndrs+bvh@0.1.0_...`.

## Traps (add to this list)

- For single-frame NRC texture comparisons, set the same seed and sample index on the plain and cache kernels. The plain bench adapter defaults to seed 0 and the cache defaults to 1. Misaligned primary jitter can look like light leaking into a black texel. Compare the primary albedo planes before scoring. [Checker regression](bench/nrc/factor-check.ts)

- The Chrome/Metal NRC frame kernel can reach a ten-storage-buffer device limit before adding hash queries. Bind the grid checksums and resolved RGB/count as one snapshot. Bind training vertices and per-path sums as one allocation in the hook, with aligned subranges in the later passes. Check both beauty-only and frame-plane kernels. [Phase 1](docs/internal/design/nrc-experiments.md#phase-1-results-on-the-mac)

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
- bvh comes from npm (`^0.1.0`, D33). Only against a LINKED checkout ("Developing against local siblings") are bvh changes visible after `pnpm build` in ~/Developer/bvh (consumers read `dist/`); the npm install is the published build.
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
  runner (`bench/runner/gpu-lock.ts`) does. After killing a runner, check `/tmp/rt-gpu.lock/owner` and, if that pid is
  gone, remove the whole directory (`rmdir`/`rm -r`), never only the `owner` file: an empty lock directory reads as
  "unknown owner" and blocks shell callers forever (our runner clears one after 30 s).
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
- `pt-megakernel` loads file geometry (Sponza) from the web glb export, not the Node glTF loader: run
  `node bench/assets/fetch.ts` and `node bench/adapters/export-scenes.ts` first (a stale export throws on the SceneDesc
  sha256). `readGlbMeshes` (`bench/adapters/shared/glb.ts`) reads only our own export: float POSITION/NORMAL, u32
  indices, no node matrices, no strides.
- Sponza at 1024 spp reads relMSE ~8e-3 against the 4096 spp references; that is noise, not error. Judge agreement by
  mean/relBias, block means against the references' own spread, and the 1/spp slope.
- `SunInput.direction` is the direction light travels (down for an overhead sun); `RtSunLight.direction` points toward
  the sun. The wrong sign gives a plane lit only by the sky, not an error.
- `rtModuleWgsl` with `shade` is ~50 KB unminified: pass `minify: true` (`rtMinifyWgsl`, keeps line numbers).
  `shade: true` already appends `rtLightSampleWgsl`; appending it again declares it twice.
- pt's material adapter thresholds metalness at 0.5 and makes non-metals pure Lambert (no dielectric lobe): compare
  `rtShade` FULL with pt only on diffuse or eta/k conductor materials.
- Mitsuba 3's `envmap` looks rows up at `v * (H - 1)` (not `v * H - 0.5`): a `0.5 - v` texel vertical shift, +0.4% on a
  sun-lit plane. The exporter pre-warps the rows (`equirectForMitsuba`, V50); any new `hdri` Mitsuba reference needs it.
  To separate lookup from shading bias, render a diffuse plane under the map (no occluders): radiance = rho/pi * the
  cosine integral of the bilinear map, a CPU quadrature to 1e-5 (single bounce, so `direct` with emitter-only / BSDF-only
  sampling in Mitsuba tells lookup from sampling). A per-pixel background comparison cannot see a sun outside the view.
- Binding one buffer as read-only and read-write storage in the same compute pass fails validation: give dummy
  bindings their own buffers.
- Timing short GPU passes (P1.6a): time N back-to-back dispatches in one command buffer (>= 0.5 s, timestamps around the
  whole batch) instead of single ~4 ms passes; the old p10-of-30 read ~2x slow and p90/p10 1.2-3 on a quiet M5 Pro (clock
  ramp), the batch reads max/min ~1.001. `trace-timing pageAb=<key>=<A>,<B>` interleaves two variants ABBA in one page.
- Queue at most one ~0.5 s timing sample at a time (`await queue.onSubmittedWorkDone()` between): many queued batches
  ended in Metal "GPU Timeout Error" and a lost device. A runner run also waits on the shared GPU lock behind another
  agent's page, which can stretch a loop of runs by minutes.
- Environment map (P1.4d): an alias table over all pixels of a 512x256 map fails in f32 (a 17-bit index leaves ~7 bits for the
  alias coin, so the sampled probabilities deviate from the table); use row-marginal + conditional CDFs (u.y rows, u.x columns).
  A stratified-grid test of a sin θ-weighted sampler needs many strata in the row dimension (4096) or its quadrature error is 1%.
- The bench adapters run in the page: they fetch assets as `/bench/assets/<path>` (the Vite server serves the repo root); never
  `readFileSync` an environment PFM there. `pt-megakernel` records `envSampling: 'auto'` in its config (importance with a map).
- `matrix.ts` queues on the shared GPU lock: another agent's long `trace-timing` run made a 2.6 s render take 4 minutes of wall
  time. Check `/tmp/rt-gpu.lock/owner` first; the wait is not a hang.
- A Python-style multi-replace of WGSL template text in `light-sample.ts` can hit the same function name twice (the constant and
  the map variant of `rtSampleEnvironmentLi`): anchor replacements on unique text and run `pnpm typecheck` after each.
- Light selection (P1.4f): `lightSelection` only changes the alias-table weights (`lightSelectionWeights`, lights.ts); the real total power
  stays in `copy.totalPower` (the table's own `total` is 1 for a non-power strategy). Cornell, quad-light, env-sphere and Sponza have
  one light kind or no area light, so strategies tie or `cap` equals `power` there: use the `lights-mix` / `lights-mix-lamps` scenes
  (`bench/scenes/ab/`, no pbrt/Mitsuba reference: compare to our own 2 x 32768 spp renders; the 1024 spp cells take 0.1-0.3 s).
- Never `pkill -f` another job's name from an agent shell (auto-mode blocks it, and it could hit other agents' runs). A background
  sweep started with `nohup sh script &` can only be waited for; plan sweeps short (3 seeds x 4 strategies x 4 scenes took ~50 min
  with Sponza at 51 s per 1024 spp cell).
- Megakernel text (P2.10): `ptMegakernelWgsl` prunes unreachable top-level declarations and renames top-level names to 1-2
  letters by default (40.4 KB, 43.2 KB with a map). Compiler messages would name `C5` at shifted lines, so there is a source map
  (P2.10b): `PtMegakernel.create`/`RtBatchTracer.create` already throw translated errors (original names, source line + context);
  by hand use `ptMegakernelBuild(...).map` with `rtMapCompilationInfo(await module.getCompilationInfo(), map)`, and compile
  through `rtCreateComputePipelineAsync({ code, map })`. Lines and names only (no columns); the source is the comment-free readable
  text. A new transform stage must return a line map and compose it (`rtComposeSourceMaps`). Tests that look for a name or constant in the kernel text must pass
  `shortenNames: false`. `rtShortenWgsl` skips any name also used as a member, parameter or local; keep new WGSL names unique
  per top-level scope. `node bench/size-breakdown.ts [--env]` prints where the bytes go.
- Textures (P1.4c): an sRGB texture must be `rgba8unorm-srgb` (hardware decode before filtering); decoding in the shader after a linear
  filter reads dark at edges. Alpha is never decoded. Colour fields (base colour, emission) need `srgb` textures, data fields (metallic-
  roughness, normal) `linear`: `buildSceneCopy` throws on a mismatch. Textures force `surface.positions` on (texture frame, cone LOD).
- Texture uv convention is glTF's: (0, 0) = first texel of row 0 (top-left), v down. pbrt-v4 flips t (the exporter writes v' = 1 - v to
  the PLY); Mitsuba `bitmap` reads v as glTF does (verified with an asymmetric marker texture; `MITSUBA_FLIP_V` in the exporter).
- pbrt `imagemap filter "bilinear"` still picks a mip level from ray differentials scaled by max(1/8, 1/sqrt(spp)); Mitsuba bilinear is
  level 0 only. Keep reference textures at most ~8 texels per pixel (or compare with Mitsuba) so pbrt stays on level 0.
- Normal map frames: glTF green = image up = decreasing v; the bitangent is cross(n, t) x handedness. Verify any change with
  `--page texture-check` (closed-form normal-map planes to 1e-7), not by eye.
- A texture bucket is one binding + one sampler (wrap mode is part of the bucket); 8 buckets by default, merged when more. WGSL has no
  sampler arrays. `rtSceneCopyLayoutEntries(0, copy.textureLayout)` must be used for the layout, or the bind group will not match the kernel.
- `rtTexSample` hardware filtering has 8-bit weights: compare against the f32 twin with 1e-2 absolute, never exact.
- A textured emitter (`emissionTexture`) is left out of the light table on purpose (V52); a kernel built with `textures` and a scene copy
  without (or the reverse) throws in `PtMegakernel.setScene`.
- Adding a member to `MaterialDesc` (bench/scenes/schema.ts) breaks `pnpm typecheck` in every exhaustive `switch (m.type)`:
  `bench/exporters/pbrt.ts`, `mitsuba.ts`, `gltf.ts` (add a `case` that throws) and `bench/src/adapter.ts` (`FeatureSupport.materials`).
  `materialFromDesc` in `to-scene-copy.ts` must map it too.
- glTF material defaults are not OpenPBR's: no `pbrMetallicRoughness` means white, metallic 1, roughness 1 (OpenPBR: 0.8 grey,
  metalness 0, roughness 0.3), so `gltfMaterialToOpenPbr` always writes baseColor/metalness/roughness explicitly. `KHR_materials_transmission`
  without `KHR_materials_volume` (or `thicknessFactor` 0, the default) is a thin surface: `geometryThinWalled`. `attenuationDistance` absent = infinite
  = no medium. `IORTestGrid` carries a volume on transmission-0 materials: read the factor, not the extension's presence.
- glTF anisotropy to OpenPBR (`gltfAnisotropyToOpenPbr`) is not invertible at roughness 1 (alpha_t = alpha_b = 1 whatever the strength).
- Tier 1 models: `node bench/assets/fetch.ts` (24 MB, `downloads/tier1/<Name>/`), then `node bench/scenes/tier1/goldens.ts --check`. The goldens
  carry the sha256 of the model; a glb is read through `parseGlbContainer` (JSON chunk only). A new Tier 1 model: pin it with a throwaway
  script that downloads, hashes and writes the manifest entry (the committed manifest is the only record of the hashes).
- GitHub's contents API (`api.github.com/repos/.../contents/<path>?ref=<commit>`) lists a model directory at the pinned commit without cloning;
  `raw.githubusercontent.com/<repo>/<commit>/...` serves the files. Only `IridescenceDielectricSpheres`/`IridescenceMetallicSpheres` lack a `glTF-Binary/`.
- Layered BSDF (P2.21): pbrt's `LayeredBxDF::f` and `Sample_f` do not agree (the exit-MIS weights mix an inside-direction and an
  outside-direction density, +3.5% at roughness 0.3, +7% at 0.5 for a white base), and its `PDF` integrates to 1.5-2.7, not 1: test `f`
  against the sampler with `exitMis: 'none'` and never assert pdf <= 1 for the stochastic one. OpenPBR-style albedo scaling is ~33%
  brighter than the walk for a base of 0.8 (no internal reflection): a white-furnace test cannot see it, compare a base < 1.
- A white-furnace test of a stochastic layered BSDF needs `thickness: 0` (pbrt's default 0.01 attenuates by exp(-0.01/cos) per crossing),
  a deep walk (maxDepth 200) and a smooth or roughness <= 0.1 interface (a rough interface loses single-scatter energy at every crossing:
  0.94 at 0.3). Eval of a smooth coat is only the walk: add the delta reflection `F(mu)`.
- `thickness` 0 in a layered walk breaks the interface alternation (`z == thickness` always): clamp to a tiny positive value (pbrt does).
- WGSL const-array LUTs cost text: print `Number(v.toFixed(4))` and drop the leading `0` (`String(Float32)` prints 17 digits and tripled
  the table). `var<private>` for a table indexed at run time.
- macOS `sed -i` needs `-i ''`; a failed `sed -i` in a chain is easy to miss. Use python or the Edit tool for source edits.
- pbrt `coateddiffuse`/`coatedconductor` lose energy at their defaults: `thickness` 0.01 is an absorber even with `albedo` 0 (8% of a white
  furnace) and `maxdepth` 10 truncates the walk (7% more). The exporters write thickness 1e-6, maxdepth 64 (V60).
- Mitsuba `rgb` values on a medium parameter are reflectances and must be in [0, 1] (`sigma_t` 2.4 fails to load): write `scale` times a
  normalised rgb. pbrt uplifts RGB `sigma_a` spectrally, so saturated absorbers are off by up to 8% per channel; grey is exact (V62).
- pbrt `volpath` is patched like `path` (`rr`, `rrdepth`, V59). The patch file is the `git diff` of `.cache/native/pbrt-v4`: after editing the
  working tree regenerate it with `git diff > bench/references/pbrt-no-rr.patch`, then `ninja -C .cache/native/pbrt-v4/build`.
- pbrt `mix` picks material[1] with probability `amount` (hash-based per shading point), Mitsuba `blendbsdf` `weight` is the weight of the second
  child. Do not wrap a transmissive child (dielectric) in `twosided` in Mitsuba; wrap the one-sided leaves individually.
- `openpbr` export: class C throws unless `coverage` is passed (`ExportOptions.coverage`, `render.ts --coverage`); coverage output is flagged
  `coverage:` in the sidecar and is not a reference. Both mesh and material emission on one mesh throws.
- glTF scene loader (P2.19, `bench/scenes/gltf-scene.ts`): `loadGltfScene` groups triangles by material; the flat `loadGltfTriangles` (Sponza,
  `file` geometry without `select`) is unchanged and pinned by sha256 in `tier1/gltf-scenes.test.ts`: keep `gltf-loader.ts` output byte-stable.
  A `file` mesh with `select.material` (and `attributes`) resolves through `loadFileGeometry`, which keeps the last loaded scene, so resolve a
  scene's meshes together. Material index -1 = primitives with no material (the glTF default material, white metal roughness 1): sidecars drop it
  unless `includeDefaultMaterial` (MetalRoughSpheresNoTextures: unlabelled text plane at z = 0).
- Tier 1 static scenes (`tier1/gltf-scenes.ts`) build their material table from the committed goldens, so they exist without the download, and a
  test rebuilds them from the model. A golden regenerated by `goldens.ts` changes the scene; keep `scene.json` hash users in mind.
- Texture decoding is Node only: `generator: 'image'` textures need `gltf-images.ts` imported (`file-geometry.ts` does it) or `generateTexture` throws.
  `scenes/index.ts` must stay browser-safe (the competitor page bundles it): static scenes read goldens by JSON import, never by `node:fs`.
- Khronos test models are not metres-sized: `MetalRoughSpheresNoTextures` is 7 mm wide (spheres 0.7 mm), so its scene scales x1000 (`GltfSceneBuild.scale`);
  `IORTestGrid`/`SheenTestGrid` are ~0.5 m, `AttenuationTest` 20 m. Check the bounds before choosing a camera or an epsilon.
- `jpeg-js` (BSD-3-Clause, `bench` dependency) decodes ClearCoatTest's one JPEG: mean 1.7 / 255 (max 22) away from libjpeg (PIL) because it upsamples 4:2:0
  chroma differently. Exporters write the decoded texels as PNG, so every renderer still reads the same bytes. PNG decoding is bit-exact vs PIL.
- Models' "instanced triangles" in tier1.md are per unique mesh: baked, `AnisotropyRotationTest` is 13,876 (a mesh used twice) and the iridescence
  spheres 617,406 file triangles of which 16,212 are zero-area sphere-pole triangles dropped by the loader (601,194 kept).
- OpenPBR mode (P2.21/22/30): `ptBsdfFromMaterial(m)` is the legacy adapter (a non-metal is pure Lambert); `ptBsdfFromMaterial(m, options, u)` is OpenPBR mode, used only by a scene
  with an `openpbr` material (`pt-megakernel` decides; the `pbr` textured scenes stay legacy, so their images and the Tier 0 kernel text do not move: re-check with
  `node bench/size-snapshot.ts <dir>` before and after a change and `cmp` the files, and `cmp` the 64 spp PFMs). Template strings in the WGSL generators: a gated block must emit
  exactly the old text when off, including blank lines (`${on ? `..\n` : ''}` placed so that no extra `\n` appears), or the kernel is not byte-identical.
- `PT_BSDF_LAYERED` reuses `PtBsdfParams` (`reflectance` base, `eta` Fresnel scale, `k` = thickness / internal reflectance): do not read `eta` as an IOR for that kind. A stochastic layered sample's `pdf` is
  only proportional (`PT_FLAG_PROPORTIONAL`): MIS needs `ptLayeredPdf` as the previous pdf, the throughput uses the sample's own `f |cos| / pdf`.
- A layered kernel is 48-53 KB (the Tier 1 scenes have an env map): over the 50 KB DXC class, see integrator.md open choice 11. `ptMaterialFeatures` keeps thin / medium / layered / mix code out of scenes that do not need it.
- The web glb of an `openpbr` scene (`export-scenes.ts`) is geometry only (black placeholder materials; the glTF exporter has no `openpbr`); it is one node per mesh in scene order, already in world space,
  and `pt-megakernel` reads the i-th primitive for the i-th mesh and applies an identity instance transform to file meshes (the old single-file path applied the scene transform on top of the baked one: harmless only because Sponza's is identity).
- Medium / thin tests: `bench/scenes/ab/openpbr-check.ts` has closed forms (thin sheet T' = (1-R)/(1+R), thin furnace 1, Beer slab T^2 x / (1 - R^2 x^2), layered furnace) at a 1 degree field of view; a wider angle changes Fresnel.
  `node bench/runner/run.ts --page openpbr-check` compares the OpenPBR WGSL with its TS twins.
- On a loaded Mac `matrix.ts` waits `--load-wait` (120 s) per cell; correctness-only runs pass `--load-wait 0` (rows are then marked `loaded`, so no timing from them).
- A stochastic layered `Sample_f` overflows f32 at near-smooth roughness (alpha ~0.002: each bounce multiplies f and pdf by ~1e4): inf / inf = NaN pixels, 1 in 1300. Rescaled in the sample (V72). pbrt's own `coateddiffuse` prints
  "Not-a-number radiance value" at roughness 0.06 and blacks those samples. Check `non_finite_pixels` on any new layered scene and run `openpbr-check` (`stochastic_non_finite_mismatch`).
- pbrt is spectral: a saturated base colour under multiple scattering (a layered base, 0.5 / 0.02 / 0.01) reads -6% red, +3% green against our RGB walk at ANY roughness, while a grey base matches to 0.02%. Judge layered code on grey
  bases and against Mitsuba (RGB) for coloured ones.
- Tier 1 wave-1 references (P2.32a): 4096 spp costs 2.5-9 min per render here, all 14 fit. pbrt `volpath` with a saturated absorber (tint 0.1/0.5/0.9
  at 1-2 m) writes negative red and 8-12% low blue (V62): AttenuationTest is not a pbrt reference for thick rows, use Mitsuba. A saturated gold
  (blue base 0.012) differs in blue by 12-25% between pbrt and Mitsuba (spectral uplift); the dielectric-over-diffuse class B rows read 3-4% lower in
  Mitsuba (roughplastic vs coateddiffuse). Agreement between the renderers on class C spheres (specular 0.25) is not evidence: both use the same F0 hack.
  pbrt writes negative channels, count them (`negChannels`).
- Region verdicts on a grid scene are contaminated by neighbours: spheres reflect and refract each other, so a class C (coverage) sphere skews the class A regions next to it (IORTestGrid: whole-image relMSE 1.6e-2, class A spheres off by up to 3%). Gate on a variant with the approximated materials dropped (`IORTestGrid-notex-A`: 2.7e-4, within the 3.1e-4 floor).
- Competitor glb (P2.31): `export-scenes.ts` now writes `openpbr` materials as core + `KHR_materials_*` (`bench/exporters/gltf-openpbr.ts`, the inverse of
  `gltfMaterialToOpenPbr`), no longer black placeholders; re-run it, a stale `.json` hash only covers the SceneDesc, not the exporter. tgpt and mutracer
  skip `openpbr` scenes unless `--force` (`force=1`): the coverage runs need it. `<scene>--no-<tag>` names (`tier1/ablations.ts`) resolve through `getScene`
  and `export-scenes.ts`, so ablation variants run through every adapter without being registered.
- Coverage matrix (P2.31) noise: tgpt and mutracer are deterministic (Sobol), so full and stripped renders share one noise realisation; unchanged regions
  cancel exactly in relMSE(full, stripped). The matrix therefore estimates each renderer's noise as relMSE(full N spp, own N/2 snapshot) (equals the
  variance of the N spp image for independent halves) and needs ablation >= 10x that. Small features in a big image are diluted (whole-image relMSE has
  no material mask): pick scenes where the feature covers much of the image. `bench/adapters/tier1-matrix.ts` imports no page code (tests import it).
- Coat over a metal (P2.23): `coatedconductor` divides the conductor's eta and k by the interface eta, so the base is evaluated INSIDE the coat.
  Layered albedo scaling must therefore take the metal Fresnel at the refracted angle (`mu_c`), not the outside one (the outside angle read +13..28%
  bright). The albedo path overestimates the walk by 7-11% on a rough coated metal; the walk matches pbrt within ~4% (R) / 1% (G, B).
- The conductor-base WGSL (`conductorBase`) is emitted only when the material table has a coated metal (`features.coatedMetal`, key present only
  when true so `ptMaterialFeatures` of other tables keeps its shape). `bench/size-snapshot.ts` now includes openpbr layered variants (plain, internal,
  stochastic, each with and without coated metals): run it before and after touching `layered*.ts` / `adapter.ts` and cmp.
- OpenPBR 1.1.2 layering (checked against the spec text): `f_coat + (1 - E_coat(w_o)) f_sub`; the plastic +33% over pbrt is the spec's behaviour (it adds
  `coat_darkening` to compensate). `specular_weight` is an IOR modulation there (F90 stays 1); we scale F at both ends (V74).
- A glTF-name `replace` on a Tier 1 sidecar (`dropMaterials` / `replace`) is how a metal-base variant of a Khronos scene is made
  (`ClearCoatTest-notex-metal`); `node bench/references/render.ts --scenes <n> --renderers pbrt --spp 4096` renders its pbrt reference in ~8 min on a loaded Mac
  (`--renderers mitsuba --coverage` for the Disney clearcoat stand-in).
- Textured Tier 1 (P2.24): a metallic-roughness texture decides metalness per texel, so the adapter must give `ptMaterialFeatures` a material with `0 < metalness < 1`
  (done in `pt-megakernel/index.ts`); with the factor 1 it compiled a pure-metal kernel and every dielectric texel rendered as plain Lambert (MetalRoughSpheres
  non-metal rows +10..40% bright, flat across roughness). Look at a roughness-by-column strip before trusting any textured scene.
- Native texture exports (`exporters/textures.ts`): pbrt reads a float imagemap from channel 0 (one grey PNG per data channel, `encoding linear`); `Texture "float"
  "scale"` multiplies two textures (roughness alpha = G*G; pbrt's own remap is sqrt). Mitsuba has NO texture arithmetic: alpha, conductor k and mask opacity are baked
  into raw PFM/PNG, `roughplastic` takes only a float `alpha` (blendbsdf chain over <= 8 roughness levels), and a `mask` BSDF needs `volpath` or `path` treats the
  holes as shadow occluders. pbrt's alpha is stochastic (a < 1): gain 1000 makes the transition narrower than an 8-bit step. Colour PNGs are written without alpha.
- Alpha MASK in our kernel (`alphaMask`, set by the adapter from the scene copy): shadow rays become closest-hit loops, so the `bvhTraceAny` walk is pruned and the
  masked kernel is about 1.7 KB SMALLER than the plain textured one (42.4 vs 44.0 KB). No Tier 1 model that is in waves 1-2 uses MASK: `tex-mask` is the test.
- The web glb export now carries TEXCOORD_0; glTF image textures travel as decoded RGBA8 `<scene>.tex<N>.rgba` next to it (`export-scenes.ts`), never through a browser decoder.

- Anisotropy (P2.26): glTF `TANGENT` is dp/du of the uvs (AnisotropyStrengthTest: per smooth vertex max 1.9 deg, per flat triangle 3.7 deg at the centroid, 5.6 deg at its corners), and pbrt and Mitsuba derive their GGX axis from the
  mesh uvs and take no tangent input (pbrt `plymesh` none, `trianglemesh` `S` only). So an anisotropic scene mesh must carry TEXCOORD_0 or each renderer falls back to its own default uv frame and they disagree silently:
  the loader emits uvs for anisotropic materials when `tangents` is on (`gltf-desc.ts` sets both), the exporters add an `approximations` note when a mesh has none. Ours reads the web glb's `TANGENT` (default) or per-triangle dp/du
  (`pt_tangent=uv`, `uvTangents`); on a faceted sphere the `uv` one matches the references per pixel, the smooth one only on means (README).
- WGSL `#if` blocks left blank lines in the output (the preprocessor keeps line numbers), so adding one to `rtSurfaceWgsl` changed every kernel's minified bytes by +46. Gate new decode code with a JS template condition
  (`${o.tangents ? `...\n` : ''}`), then `node bench/size-snapshot.ts` before/after and `cmp`.
- `anisotropyRotation` in the material record is unorm16 TURNS in [0, 1) (`rotationTurns(radians)` in `to-scene-copy.ts`); the field used to take raw radians and clamp at 1. The exporters ignore the rotation.
- macOS `sed -i` needs an empty suffix argument (`sed -i '' ...`); `sed -i 's/..//' file` treats the script as the suffix and the file as a command.
- A second `node bench/references/render.ts` started while another agent's native render runs doubles the load: check `ps -eo command | grep -E "[p]brt|[m]itsuba_render"` first: wait with an `until ! pgrep -f '[b]ench/references/render.ts'` loop first.
- OpenPBR spec text: `curl -sL https://raw.githubusercontent.com/AcademySoftwareFoundation/OpenPBR/main/index.html` (v1.1.2, Markdeep source, grep for
  `Fuzz` / `Thin-film`); the Zeltner LTC tables and code are `tizian/ltc-sheen` (Apache-2.0), the Belcour-Barla RGB film is Khronos
  `glTF-Sample-Renderer/source/Renderer/shaders/iridescence.glsl` (Apache-2.0); CIE 1931 CMFs come from `http://www.cvrl.org/database/data/cmfs/ciexyz31_1.csv`
  (the `cvrl.cvrl.org` and `cvrl.ioo.ucl.ac.uk` hosts fail), D65 stops at 780 nm in the colour-science table (a loop to 830 gives NaN).
- Fuzz and thin film are gated like the other lobes (`features.fuzz`, `features.thinFilm`): the kernel text without them must stay byte-identical, so
  conditional template text may not leave parentheses behind (`(a)` vs `a` is a different byte count) and WGSL refuses `a && b || c` without
  parentheses, so the fuzz variant writes `(a && b) || c` only when on. Check with `node bench/size-snapshot.ts <dir>` before and after and `cmp` every file.
- A kernel with `features.fuzz` binds the LTC table at `@group(2) @binding(2)` (`PtMegakernel` creates it); a page that builds its own pipeline layout
  must add that binding. Thin film needs `layering: 'albedo'` (`ptLayeredWgsl` throws on stochastic; the adapter falls back and notes it) and reads the
  record words 29-31 through `ptMaterialFilm`, not through `RtMaterial` (adding fields to the WGSL struct changes every kernel).
- `node bench/runner/run.ts --page lobe-oracle [--args fuzz=0|film=0 spp=256]`: sphere under a uniform unit sky, one scattering vertex, pixel = the BSDF's
  directional albedo at the view angle, compared per mu_o bin with a brute-force CPU integral of the TS twin (fuzz) or the exact spectral Airy sum
  `bench/metrics/thinfilm-airy.ts` (film; D65, linear Rec.709, constant spectrum c gives (c, c, c)). Film errors are small near normal and grow toward
  grazing (unpolarised Fresnel inside the Airy sum, a metal's phase from k ignored): gate the mean and mu_o > 0.55, report the worst.
- Khronos' `evalIridescence` clamps `R12 R23` to at least 1e-5, which leaves a 0.006 bias for a vanishing film, uses Schlick at the interfaces (0.008 error
  at cos 0.7 against the exact spectral sum) and adds the harmonics through the Rec.709 image of XYZ without a white balance; ours fixes the first two and balances.
- `IridescenceDielectricSpheres` is a 7 x 7 x 7 grid, material index = 49 ix + 7 iy + iz with x = base ior, y = film thickness (0.1-0.7 um), z = film ior (both
  ship only as `.gltf` + `.bin`, pass `'gltf'` to `make`); one z slice (`i % 7 === 2`) is 49 spheres, 88k triangles.
- A fresh pathtracer worktree has no package links: `pnpm install --offline` needs `.claude/worktrees/bvh` -> `~/Developer/bvh`
  (only with the sibling override above; bvh is on npm since D33), as reflections' worktrees have. Reflections links
  `link:../pathtracer/packages/raytracer` (TS source; a reflections worktree needs `.claude/worktrees/pathtracer` too).
- `reflections-ab` (P1.5) loads reflections' trace seam by path: `REFLECTIONS_REPO=<checkout> node bench/runner/run.ts --page
  reflections-ab --args reflections=<checkout>`. The checkout needs the deleted @pmndrs/bvh wrapper for A: use reflections commit
  36e206a (worktree `p1.5-ab-baseline`).
- Our module pasted into someone else's shader: build its caps with no optional features (`buildCaps({ features: new Set(), limits })`)
  or the `enable` lines land mid-file and the shader fails. Reflections' own WGSL also uses `rt`/`RT_` names (`RT_SOURCE`,
  `rtLobeVisibility`); no collision today, check its names when adding ours.
- WGSL refuses `a * b ^ c` (mixing `*` and `^` needs parentheses); it only fails on the GPU.
- Reflections' timing pages show 2-20x bursts on every variant at once on this Mac: compare per-variant floors or medians over
  5+ interleaved rounds, never one run.
- Shader-compile timing (`--page kernel-compile`): a unique comment in the WGSL does NOT defeat the cache. Tint drops it, the generated MSL/HLSL(DXIL) is identical and a backend or driver cache answers (Mac: repeat 1 = 130-475 ms, repeats 2-3 = 9-12 ms). The page puts a unique f32 literal in a live line of `ptMain` (`1 + n/2^23` in the accumulate); if `ptMain`'s accumulate line changes, update `LIVE` in the page (it throws). Check cold/warm and the per-repeat list.
- Tier 1 gate floor (P2.32): relMSE is not symmetric (the reference is the denominator). The floor for reference X is relMSE(other, X): AttenuationTest is 2.4e-2 with pbrt as reference
  and 5.0e-3 with Mitsuba. `compareImages(test, ref)`; use the floor of the reference you judge against. Slope of 1/spp from `relMSE(snapshot s, own final)` reads k 1.1-1.3 (shared samples), not 1.
- D24 default (`layering: 'albedo'`, base `'internal'`, `ptAlbedoBaseInternal`) is the library default only; the `pt-megakernel` adapter still defaults to `stochastic` (reference mode, the gate class B numbers).
  The internal base is +11-15% bright on a white rough base (AttenuationTest backdrop, SheenTestGrid), +2-3% on MetalRoughSpheres: do not take albedo/internal numbers as a Tier 1 result.
- ReSTIR random streams (P4.2b bias hunt): every random number a pixel draws in one frame needs its OWN `ptRsRand2` key (`PT_RESTIR_KEY`, restir-wgsl.ts). The temporal and spatial merges shared the key base 40 (and the temporal sub-pixel shift, key 7,
  collided with candidate 3's u), so the spatial "replace the kept sample" decision was correlated with the temporal one: the kept sample was no longer chosen with probability proportional to its weight. That is invisible when the domains' weights are alike
  (legacy scenes: +1e-3 over NEE, now within the seed spread) and reads -15% on a face when they differ a lot (a stochastic material pick that changes every frame: `restir-pbr-lamps` box, whole image -0.36%). The OpenPBR adapter, `ptEval` and the G-buffer pick were
  innocent (the pick written and the pick shaded agree: same `sampleIndex` / `P.dims.z`). A new random draw in the passes gets a new key constant, never a literal.
- Measuring ReSTIR bias (`restir-check check=biasscan seeds=24+`): ONE accumulated static-camera run has a relBias noise of ~1e-3 on a 256^2 film (reuse correlates the frames), so the old single-seed "-0.13% Lambert" was noise; only the mean over 24-32 seeds with its standard
  error says anything (SE ~2-3e-4 at 256-512 frames), and the NEE reference's own offset (~1e-4..4e-4) is the zero line. A fixed material (or one fixed per pixel) hides a stochastic-pick bug: bisect with one stochastic material on an otherwise Lambert scene
  (`restir-pbr-lamps-mixonly`), and read the per-block map (`notes.biasscan__*__blocks`, seed-averaged blocks off by > 0.4%) to see WHERE (the -15% blocks sat on the box top).
- Wavefront (P3.1, `packages/pathtracer/src/wavefront/`): Chrome on Apple GPUs allows 10 storage buffers per stage whatever the device requests (the
  caps ask for 16), and a declared binding counts even if the kernel never reads it. Walk 3 + scene 2 + the 7 wavefront buffers did not fit one
  pipeline layout, so each kernel has its own layout with only the buffers it declares (empty group where it has no walk or scene), and the shade
  module drops the walk's `@group(0)` lines (`stripWalkBindings`). The error text comes from a validation error scope around layout creation.
  Switching pipelines in one pass invalidates groups with a different layout: set all of them again before each dispatch.
- Wavefront state is `array<vec4u>` with floats as bit patterns: a small u32 (instance, triangle index) read back through an f32 array is a denormal
  and a load may flush it. The indirect args buffer lives in its own group 3, bound only for the args kernel, so no dispatch uses it as storage and
  as INDIRECT at once. The counter protocol (ping-pong of the extension queue and of the shadow counter, zeroing in the args kernel) is simulated in
  `layout.test.ts`: change both together. `maxDepth` 1000 (furnace-dielectric) is 4000 dispatches per sample: 95 s per 1024 spp, not a speed test.
- Wavefront vs megakernel agreement is per pixel (max relative difference <= 7e-5, mostly 1e-6): `node bench/adapters/pt-wavefront-agreement.ts --a <mk report> --b <wf report>`.
  A difference above that is a bug in the wavefront port, not noise: both use the same sampler dimensions per pixel and vertex. The `pt-megakernel`
  adapter config now carries `scheduler`, so its config hash differs from rows written before 2026-11-15.
- Scene copy packing (P3.11, `sceneCopy.packing` = `uploadSceneCopy({ packing })` + `scenePacking` on `PtMegakernel` / `PtWavefront` / `RtModuleOptions.scene.packing`): `packed` binds
  the table and the data as ONE `rtScene: array<vec4u>` = [header vec4 | table | data], one storage buffer less in every kernel (hooks get one more: megakernel 5, wavefront shade 3; Apple's 10-buffer wavefront shade
  kernel goes 8 -> 7, intersect with MASK 9 -> 8). Generators still WRITE `rtSceneTable[..]` / `rtSceneData[..]`; `rtPackSceneAccess(source)` rewrites a finished source to `rtScene[i + 1u]` / `rtSceneWord(i)`, so
  `split` text is byte-identical to before packing (check: `node bench/size-snapshot.ts <dir> split` before / after, `diff -r`; `... packed` is the other set). Any WGSL you append to a packed kernel must go through
  `rtPackSceneAccess` too (a stray `rtSceneTable` is an undeclared identifier; builders do it for their own text, a bench page's extra kernel does not). Textures / samplers move down one binding when packed
  (`rtSceneCopyLayoutEntries(first, textures, packing)`, `rtSceneBufferBindings`); a kernel, its layout and the uploaded buffers must agree on the packing (`setScene` throws). The packed buffer is
  header + table + data (16 B aligned, data zero-padded to a whole vec4), so the 128 MiB-class binding cap applies to the SUM (`uploadSceneCopy` throws a RangeError that names both sizes and suggests `split`);
  Sponza is 21.0 MiB (84 B per unwelded triangle with normals + positions), so the cap is ~1.5M such triangles, and the table is small unless there are many lights (96 B each) or instances (128 B each). Packed images
  equal split per pixel on 6 of 8 scenes and differ by <= 3.2e-7 relative (1-3 ulp, relMSE 3e-17) on IORTestGrid (both schedulers) and env-sphere (wavefront): not addressing (every accessor is bit-exact
  in `--page scene-copy packing=both`), a different compiled kernel contracts some float ops differently. `DEFAULT_SCENE_PACKING` is still `split`; `writeSceneCopy(device, buffers, copy)` rewrites a rebuilt copy
  in place for either packing (the packed one rewrites the whole buffer: the data region moves when the table length changes).
- Optional integrator outputs (R10 hooks, R4 counters, AOVs; `integrator/outputs-wgsl.ts`): their template text must emit nothing (no blank line, no `\n`) when off. Two traps hit while adding them: a gated block ending in a
  newline placed before the original newline adds a blank line (the megakernel stayed identical but the wavefront shade kernel grew by one `\n`), so put the leading `\n` inside the gate for end-of-line additions
  (`x;${on ? `\n    y;` : ''}`). `node bench/size-snapshot.ts` now writes the `wf-*` wavefront kernels too (`bench/kernel-variants.ts` `wavefrontKernelVariants`): `diff -r` two snapshot directories, ignore `Only in`.
- AOVs live in the TAIL of the accumulation buffer (`width * height` radiance vec4s, then 2 vec4 per pixel), not in their own binding: the wavefront shade kernel is at 8 of the 10 storage buffers per stage on Apple
  (scene 2 + 6 io), so hooks can add 2 buffers there (1 with counters), 4 in the megakernel (`--page pt-outputs-check --args check=budget`); with the scene copy packed (P3.11) 3 (2) and 5 (4): the page prints both, `_packed` keys. `readImage` copies only the first `width * height * 16` bytes.
- The Cornell red wall's four corners are not coplanar and its mesh has smooth vertex normals: test its AOV normal against the wall within ~0.02, never exactly. Minified kernel text has no spaces around `,` `)` `(` (`@group(2)@binding(3)var<storage,read_write> x`): regex tests need `\s*`.
- OIDN on our frames (`bench/adapters/shared/oidn.ts`): the Denoiser wants view-space normals ([-1,1], miss = 0), albedo [0,1], rgba32float color with `hdr: true` and `device` = our device (texture uploads need the same device; the
  weights come from `/@fs/<Denoiser>/packages/denoiser/tzas`, served because `server.ts` adds the linked checkout to `fs.allow` and defines `__BENCH_DENOISER_TZAS__`). A `link:` dev dependency needs `pnpm install --offline`.
  Buffer usage bits are not texture usage bits (`MAP_READ` is 1 on a buffer, `COPY_SRC` is 1 on a texture): a wrong constant fails device-side with a validation error, not at the call.
- A counted or AOV render's wall time is not a timing result: `pt_counters=1` rows are invalid for timing (`stats.instrumented`); `pt_aovs` / `pt_denoise` readbacks run after the timed region but the AOV writes are in the kernel.
- Wavefront refill (P3.2): the primary rays of a refilled sample must be restarted in PIXEL order. `inline` refill appends a restarted path in atomic order (random), which broke
  primary-ray coherence (+25% on Sponza); `batch` scans the slots in pixel order (the camera kernel over the film) and is as fast as slice 1 there. Any refill or sort that scrambles
  the queue order shows up first on a big-film scene with short paths, not on Cornell. Per-path SoA sections are indexed by `n` = SLOTS (K x pixels), the pixel is `path % (width*height)`.
- Depth word of the wavefront state: bit 31 specular, bit 30 AOV `aovDone` (R4), bit 29 live (refill), depth 29 bits. Zero word = dormant slot (a `clearBuffer` of the state resets a film).
- A refill `encode` does not finish its paths: call `await pt.finish()` before `readImage` (it is part of the timed region; the adapter does it). `reset()` zeroes state + counters in refill mode.
- Only the PARITY of the iteration is read by the wavefront kernels (`iter & 1`), so refill needs 4 uniform records per encode and has no params-ring limit; the start records are `iter` 2 | parity.
- `git worktree` agents: a fresh worktree has no `node_modules`/`bench/assets/{generated,downloads}`: `pnpm install --offline`, copy the two asset dirs from the main checkout; `link:` paths like
  `../../Denoiser/...` resolve relative to the worktree (deeper), so re-point `bench/node_modules/denoiser` (and others) to the absolute path or typecheck says "Cannot find module".
- `git checkout --merge <file>` during a rebase labels the sides `ours`/`theirs` (the normal markers say `HEAD`/the commit): a regex written for one silently matches nothing.
- A small WebGPU `mapAsync` of 20 B per chunk, issued while the next chunk is already submitted, costs nothing measurable (finish() of furnace-dielectric: 3 ms for 96 empty iterations); the cost to
  avoid is the dispatch overhead of empty iterations (~22 us each, 4 per iteration), not the readback.
- Alpha MASK on the wavefront (P3.7): the cutout loop lives in the INTERSECT kernel (it binds the scene copy then: walk 3 + scene 2 + 4 buffers = 9 of the 10 Apple storage buffers; one more wavefront buffer in intersect or shadow would not fit), shade only sees solid hits (`hit.w` = 1 marks a path over the skip limit). Intersect writes the path state back (origin, cone width, medium `beta`) after a skip, so shade's `st.origin` is the post-skip origin and `hit.t` is relative to it. `integrator.counters` with MASK throws on the wavefront. `alphaPolicy` is ignored without `alphaMask`; `exact` and every non-MASK kernel stay byte-identical (check with `node bench/size-snapshot.ts`, the policy variants `tex-mask-opaque*` differ on purpose). The exact masked megakernel is SMALLER than `opaqueShadow` (42.4 vs 44.3 KB): the masked shadow loop replaces the any-hit walk, which `opaqueShadow` brings back. `ptVisibleMasked` text lives in `PT_VISIBLE_MASKED_WGSL` (integrator-wgsl.ts), shared by both schedulers.
- Alpha policy A/B (R2): the policies are biased (-3..-17% on card scenes) and `opaqueIndirect` is no faster; the wavefront ratios are paired against the same scheduler's exact policy (`alpha-policy-ab.ts` reads the last `repeat` interleaved CSV rows of each cell: the CSV is append-only, other runs of the same cell would mix in otherwise). Renderer ids `pt-mk-*` / `pt-wf-*` are policy presets; `pt-wf-secondary` etc. are the wavefront (plain) with the policy.
- V-buffer primaries (P3.6, `integrator.primary: 'vbuffer'`, `integrator/vbuffer.ts`): `rgba32uint` texel = (instance + 1, triangle, material, unorm16 b1 | b2 << 16); a cleared target (0) is a miss. The ENGINE writes the
  material: bvh keeps per-triangle materials inside its tree and the scene copy does not store them, so (instance, triangle) alone cannot name the material. The engine owns the sub-pixel jitter and passes the same numbers with
  `setVbufferJitter(x, y)` (pixels, [-0.5, 0.5], 0 = centre; the params record carries `0.5 + offset` in the otherwise unused `origin.w` / `right.w` lanes, so `PtParams` and every default kernel stay byte-identical). The kernel still
  needs the jitter: a miss pixel's view ray (sky) and `wo` come from the jittered camera ray, hits do not use it. A raster jitter that disagrees with the one passed is a sub-pixel shift, not an error: test it with the
  edge classification of `--page vbuffer-check`. `encode(encoder, 1)` per frame (one V-buffer = one jitter; wavefront `refill`/`layers` throw); needs `layout.positions`; texture binding is group 2 binding 9, sample type `uint`
  (not filterable). The raster side: clip.x -= 2 jx z / W, clip.y += 2 jy z / H (y down), `@interpolate(flat)` u32 ids, bary as a perspective-interpolated vec2; no primitive_index needed (draw non-indexed, corner id per vertex).
- A same-seed V-buffer vs traced comparison is NOT independent: the dimension map hashes the pair index, so every bounce / light / RR number of a pixel is identical in both and only the camera jitter differs (relMSE far BELOW
  the noise floor: cornell 0.07x). The distribution test is vbuffer(seed 2) vs traced(seed 0) against traced(seed 1) vs traced(seed 0): ratio 1.00.
- A V-buffer hit on an alpha MASK cutout texel needs no alpha-tested raster: the kernel's own cutout step passes through and continues by TRACING (`primary` goes false after the first iteration), so images are exact; an engine
  that alpha-tests only saves those re-traces.
- The runner's `loaded` gate is load1 > core count, so a 4-6 load (post-reboot Spotlight + fileproviderd at ~100% CPU each) is NOT marked: rows read `valid=1`
  with `best effort (background apps: ...)`. After a reboot wait until `mds_stores`/`fileproviderd` settle (can take > 30 min) before a quiet pass; quote the
  per-round load1 next to any number. `alpha-policy-ab.ts` reads the last N CSV rows of a cell, so later runs of the same cell corrupt its timing columns; use
  `bench/adapters/paired-ratios.ts --report <name>` (per-round `run.json`) for paired ratios.
- Albedo correction (P2.32b, `albedoCorrection: 'none' | 'lut' | 'texture'`, `layered-correction.ts`): the internal base is multiplied by a fitted `c(r, mu_o, base Ri)`; refit with
  `node bench/layered-correction-fit.ts gen --ior X --out f.json` (3 iors, ~4 min each, 3 in parallel) then `fit f1.json f2.json f3.json` (writes `layered-correction-data.ts`). Two traps in the fit:
  below r ~0.03 the lobe is smooth (`ptIsSmooth`), the sampler returns a delta and a quadrature of the "specular" part is meaningless (the table rows 0-1 are the identity); the cosine-sampled
  `evalAlbedo` of a narrow lobe is far too noisy below r 0.33 (use the sampler there). `texture` declares `@group(2) @binding(10)` inside the layered WGSL, so a page that builds its own
  pipeline layout for a kernel with that option must add `ptLayerCorrLayoutEntry()`; `PtMegakernel` / `PtWavefront` (shade kernel) do it. With `none` every kernel stays byte-identical
  (`node bench/size-snapshot.ts`, cmp all incl. `wf-*`). The megakernel with all features + `lut` is 51.0 KB (texture 50.0 KB; plain `albedoInternal` 49.2 KB): over the ~50 KB DXC class.
  Render-level numbers sit between the walk's sampler albedo and its pbrt-exit-MIS `f` (they differ by up to 13% at r 1, V58): fit and judge against both, and against the references.

- Ray-class material LOD (R1, `integrator.materialLod: 'secondary'`, `materialLodDepth` N, page args `pt_lod`, `pt_lod_depth`, renderer ids `pt-mk-lod` / `pt-wf-lod`): `ptBsdfReduced` (`bsdf/adapter.ts`, TS twin + WGSL, tests `bsdf/lod.test.ts`) picks ONE lobe
  per vertex at random (the expectation is the full material's albedo), so its images are noisier (relMSE vs the full 4096 spp image 1.07-3.4x that of the full material's own 1024 spp image on Tier 1) but not biased on average. 'full' is byte-identical (`node bench/size-snapshot.ts`).
  The wavefront builds `shadeSecondary` (a second shade kernel from `ptSecondaryIntegratorOptions`) and the plain scheduler switches kernel by depth iteration; refill modes throw. `PtWavefront.kernelBytes.shadeSecondary` / `node bench/lod-sizes.ts` give the bytes.
  `node bench/runner/run.ts --page lod-check --args spp=512` is the furnace check (depth 0 = every vertex reduced); `node bench/adapters/lod-agreement.ts --full <report> --lod <report>` the scene A/B.
- A reduced adapter must not use `smooth`, `target` ... as WGSL identifiers (reserved): the unit test `wavefront/lod-wgsl.test.ts` checks the declared names of the secondary kernel. A feature-stripped kernel that still has to recognise the full table's materials gets them through
  `PtMaterialOptions.reducedFrom` (the full `features`), while its own `features` decide which code is compiled.
- A stochastic lobe pick that mixes a rough-GGX lobe with a Fresnel-1 conductor must use the lobe's OWN albedo (the layered LUT at f0 = 1, `A + B`), not 1: the smooth Fresnel `F(mu)` as the pick probability read 25-70% too bright
  at rough 0.3-0.6 and grazing angles. A coat over a metal needs the conductor refitted AT the view angle (`ptRedFitK`, bisection): a conductor with the stack's NORMAL reflectance rises too fast toward grazing (+13% on a low-reflectance channel).
- Dropping the fuzz of a reduced vertex reads 10-20% dark on a sheen material; the fuzz albedo has to come back as a Lambert lobe plus a base weight of `1 - w E_f` (a 9 x 9 copy of the LTC table's R is enough; `bsdf/reduced-fuzz.ts`).
- ReSTIR DI (P4, `packages/pathtracer/src/restir/`, docs/internal/design/restir-di.md): a reservoir sample is `(light index, u quantised to unorm16)`, re-evaluated at any pixel through `rtSampleLightLi`; target and source density are both per unit of `u`
  (no area-measure Jacobian). The passes are ONE module with four entry points (`ptRestirGbuf/Init/Temporal/Spatial`, groups 0/1 shared with the megakernel, group 2 = params ring + G-buffer + reservoirs); the shade side is the megakernel
  (`integrator.restir`, binding 11, final region base in `ptParams.limits.w`). One `encode` pass = one frame; `readImage` is the mean of frames. Not on `PtWavefront` (throws).
- Judging ReSTIR by the accumulated mean of a static camera hides the point of it: temporal reuse correlates the frames, so 1024 accumulated frames are worth far fewer independent samples. Look at the PER-FRAME image (`restir-check noise`: frame t =
  t x mean_t - (t-1) x mean_(t-1)) and the moving-camera rows; the accumulated rows are for bias (unbiased mode must read 0 within the seed spread).
- NEE's Owen-Sobol numbers are stratified over frames, ReSTIR's candidates are PCG: against NEE with Sobol, `ris1` (M = 1, no reuse) reads up to 9x the relMSE on small-light scenes although it is the same estimator. Compare against `sampler: 'pcg'` NEE (`nee_pcg_*` rows) and
  remember M = 1 has no MIS weight (V134), so it is also worse than NEE + MIS on big lights (quad-light 1.97x, Cornell 1.06x).
- ReSTIR biased mode darkens: no support test on reused samples; with `initialVisibility` also zeroing occluded samples. Quote `restir-check converge` relBias next to any biased image; `unbiased` + `initialVisibility: false` is the reference-grade setting.
- `bench/runner/run.ts` kills a page after 60 s: a long check (`restir-check` with 4096-frame references takes minutes) needs `--timeout <ms>`.
- A kernel with `integrator.restir` needs `nee`, no textures, legacy materials, no alpha mask, no V-buffer, no hooks (`ptRestirUnsupported` says which); restir's own bytes: 39.6 KB passes module, 41.4 KB shade kernel (limit 50 KB), `node bench/size-snapshot.ts` writes `restir-*`.
- ReSTIR unify (P4.2b/c/e): the passes' G-buffer is one `array<u32>` of `PT_RS_WORDS` per pixel (20 B: position f32 x 3, octahedral normal, material; +uv/footprint with textures, +mix pick with a stochastic material; the view depth is derived from the position and THAT frame's camera, so
  the previous G-buffer is read with the previous camera). A miss is `material == PT_RESTIR_GBUF_MISS` (0xfffffffe). The target must run the shade's own decode (`ptRsBsdf`) and `ptEval` for layered materials; anything stochastic in `f` (stochastic layering, fuzz) cannot be a target (`ptRestirUnsupported`).
  The previous camera lives in `PtMegakernel` (`setPrevCamera` / `nextFrame`, else the last encoded camera, `reset` forgets it): never keep another copy. The passes need the same textures the shade binds (blue noise 12, V-buffer 9, layer correction 10: `setInputs`, keyed on view identity, so cache the views).
  `restir.sharedHit`: with `sampler: 'bluenoise'` (Owen shift) the shade's own jittered ray did NOT reproduce the G-buffer pass's hit (15% of Cornell read black, kronecker shift and the other samplers were fine; root cause not found), so the shade continues along the ray through the shared hit point: never assume two kernels' sampler rays agree, test shared-vs-traced images with `restir-check shareddiff` (same seed: a few hundred edge pixels differ, not thousands). It needs `layout.positions` and binding 15 in the shade io group (3 in the passes). `ptRestirUnsupported` runs in `PtMegakernel.create`. Check pages: `adapter.createKernel({ integrator })` builds a kernel with the adapter's own materials / textures / env and the override merged; use `layering: 'albedo'` for any ReSTIR check on OpenPBR
  (the adapter default is the stochastic walk). OPEN: OpenPBR scenes read -0.1..-0.8% dark in `unbiased` mode with reuse (legacy scenes < 4e-4), TASKS P4.2b.
- ReSTIR + layered / OpenPBR kernels exceed the 50 KB class (`restir-shade-pbr` 50.2 KB, textured + mask 51.8 KB, with frame planes and blue noise 57 KB, see `node bench/size-snapshot.ts`); the passes module stays under (49.4 KB textured + OpenPBR + mask).
- Real-time mode (P4.1, `docs/internal/design/realtime.md`): `integrator.frame` and `sampler: 'bluenoise'` are MEGAKERNEL only (the wavefront throws `ptwfUnsupported`; `mode: 'realtime'` on the wavefront
  silently drops them). A frame is ONE `encode(encoder, 1)`: `setCamera` for frame 0, then `nextFrame(camera)` (the old camera becomes the previous one, which motion reprojects through);
  the sampler's sample index is the FRAME index (`setFrameIndex`; `reset` zeroes it). The kernel needs bindings 12 (blue-noise texture, `rg32float` `unfilterable-float`), 13 (previous camera, dynamic
  offset in step with binding 0) and 14 (planes) in group 2 (11 is ReSTIR's); a page that builds its own layout must add them. `readImage` does NOT divide by the sample count in frame mode.
- Blue noise in the sampler: a per-frame shift shared by all pixels keeps the spatial structure; ONE shift step for every pair (a plain golden-ratio frame offset) puts pairs on a diagonal and the time
  average converges to the wrong integral, so each pair has its own irrational step. A same-seed comparison of two samplers is not independent: compare each against a reference at a different seed.
- `PtFrame` planes: the diffuse / specular split is by the PRIMARY vertex's BSDF kind (V122), the stored demodulation factors are floored at `PT_DEMOD_EPS` (V126); `diffuse * albedo + specular * specularFactor` is the
  radiance exactly (1e-7). Motion is camera-only (V124), jitter-free by construction: closed-form check = project the hit (rebuilt from viewZ and the sampler's exact jitter, `ptSampleBlueNoise` twin) through both cameras.
- Blue-noise shift default is `owen` (P4.1 A/B on Cornell: blur-sigma-2 error 1.47x below Owen-Sobol for both shifts, but `kronecker` accumulates 1.75x worse over 128 frames, `owen` equals Owen-Sobol). A camera path that orbits INSIDE Sponza hits a wall within a few degrees (viewZ ~7e-5 m, f32 noise of 5-10 px in a closed-form check): pan in place.
- A file scene (Sponza) cannot be resized in a page: its web export is keyed on the whole SceneDesc including `film` ("exported glTF is stale"). `realtime-check` therefore runs it at its own film on a pixel grid.
- Bench helper scripts: the scratchpad directory is SHARED between agents (`edit1.py`, `snap-before` were other agents' files): use your own subdirectory.
- Subgroup appends (P3.3, `appends: 'subgroup'`): Tint treats the RESULT of `subgroupAdd` / `subgroupBallot` as non-uniform, so `if (total > 0u && subgroupElect())` fails ("subgroupElect must only be called from subgroup uniform control flow"):
  write `if (subgroupElect()) { if (total > 0u) { atomicAdd } }` and broadcast with `subgroupBroadcastFirst(base)` AFTER the ifs. Every subgroup op needs uniform flow up to the entry point, so the
  shade body (`ptwfShadePath`, many early returns) only sets private `ptwfWantShadow` / `ptwfWantNext` flags and the kernel appends once after `if (go) { ... }`; no `return` before the ballots, and the
  `slot < count` test is a flag (`go`), not an early return. A `workgroupBarrier` after a `return` that depends on an `atomicLoad` is rejected the same way (`ptwfSort` has no early return).
- `rtModuleWgsl` puts `enable subgroups;` in the shade module only: the camera / args / sort kernels do not use subgroup ops (the start pass appends once per encode, not worth it).
- Block sorts (P3.4): a workgroup sorts ITS 64 queue entries in place (read before the barrier, write after), so only the order inside a block changes and the per-pixel image is unchanged (relMSE 1e-15 vs the
  megakernel). A global sort would need a histogram + scan + scatter; not built (V110). 'hit' sorts `Q[iter & 1]` after intersect, 'ray' sorts `Q[(iter + 1) & 1]` AFTER the args kernel (the main args then hold that queue's count).
- Ordered restart (P3.10, `refill: 'ordered'`): the restart marks the slot with `state[slot].w = 0xffffffff` (prevPdf word; a NaN pattern a real pdf never has) and the `primary` kernel (direct dispatch over all
  slots, between shade and args) clears it and appends the flagged slots per 64-slot block in slot order. The mark must be cleared before the next intersect / shade reads the path.
- R1b (`materialLod: 'secondary'` + refill): a queue mixes depths, so both shade kernels run over the SAME queue (first, then second) and gate on the path's depth. The first kernel can push a path past
  the lod depth (or restart a slot at depth 0), so it overwrites each entry it consumed with `PTWF_CONSUMED` (0xffffffff); a gate that re-read the changed state double-shaded those paths (images 20% dark, 93% of pixels differ).
- WGSL refuses `a & b >= c` ("mixing '&' and '>=' requires parenthesis"): parenthesise a masked word before comparing it.
- The scratchpad directory is SHARED by the agents of all worktrees: a helper named `edit1.py` was overwritten by another agent between my Write and my run, and ran their edit script (with its own `os.chdir`) in their worktree.
  Put scripts in a private subdirectory (`scratchpad/<task>/`), never in the scratchpad root; and `snap-before` style output directories are shared too.
- Worktree agents: the harness refuses a Bash command that mixes a heredoc, `cd` elsewhere or several git invocations ("too complex to verify"); write the file with the Write tool, then run it in a plain command.
- Windows (track N): a cloud session that starts the Windows handoff is Linux with no GPU and no route to the laptop (Desktop Commander runs in the same VM): only code tasks
  (WN6) and SwiftShader correctness run there. SwiftShader in the runner: `CHROME_PATH=/opt/pw-browsers/chromium-*/chrome-linux64/chrome PT_ALLOW_SOFTWARE_ADAPTER=1
  PT_CHROME_FLAGS="--no-sandbox --enable-unsafe-webgpu --use-webgpu-adapter=swiftshader --use-angle=swiftshader --enable-unsafe-swiftshader"`; never timings from it.
  bvh and downsampler come from npm (no sibling needed); the Denoiser sibling does not, so `pnpm typecheck` keeps one error in `oidn.ts`.
- Windows (VkNRC, WN7): its OBJ loader refuses a file without a material (every face needs `usemtl`), reads no normals (flat geometric, two-sided) and rescales the model to [-1, 1];
  its sky (10), path length (8 hits) and BRDF (Cook-Torrance, `Ks 0` = Lambert) are compiled in. Use `node bench/exporters/export-obj.ts` (not a hand-made OBJ) and take the camera pose from
  its `vknrc.json` (normalised position, yaw, pitch); our matching scene is `sponza-white-vknrc` (maxDepth 7, no RR).
- Cloud sessions reach GitHub (raw.githubusercontent.com included) but not arXiv, research.nvidia.com or author pages
  (proxy 403). Paper numbers have to be checked against `research/papers/` on the Mac.
- `subgroup-matrix` (gpuweb proposal) has no element-wise ops besides scalar add/sub/mul: activations such as ReLU need a
  store to workgroup memory and a reload. Operands and control flow must be subgroup-uniform, and workgroup x size must
  be a multiple of the max subgroup size. WGSL has no float atomics, so batched gradient sums need partials plus a
  reduction dispatch (docs/internal/design/nrc.md §2).
- tiny-cuda-nn is BSD-3 (portable with attribution); the NRC SDK and RTXGI are RTX SDK licence (no code); instant-ngp is
  non-commercial (ideas only). VkNRC (the NRC reference we follow) has no licence: follow its design, never paste its code (D29).
- VkNRC's MLP leans on three things WGSL lacks: per-element access to matrix tiles (ReLU), NaN as a mask (ReLU
  derivative) and f32 `atomicAdd` (weight gradients). Each needs a workgroup-memory pass or a reduction dispatch here.
- VkNRC builds on Linux (Ubuntu 24.04, GCC 13, CMake 3.28, glslc 2023.8) after `apt-get install libvulkan-dev glslc
  libgl-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`; the build in a fresh clone under
  `.cache/native/vknrc` takes ~50 s. It runs only on NVIDIA (requires `VK_NV_cooperative_matrix`, float atomics, ray
  query), loads one OBJ (normalised to [-1, 1]) and shows only FPS: timing it needs a local timestamp patch.
- WebGPU works in a GPU-less cloud container: the runner with `PT_CHROME_FLAGS="--no-sandbox --use-webgpu-adapter=swiftshader
  --use-angle=swiftshader --enable-unsafe-swiftshader --enable-unsafe-webgpu"`, `--allow-software` and `--chrome
  /opt/pw-browsers/chromium-*/chrome-linux*/chrome`. SwiftShader gives subgroups, 32 KiB workgroup memory and timestamps,
  but no `shader-f16` and no subgroup-matrix: f32 correctness only, never timings. `navigator.gpu` is undefined on
  `about:blank` (not a secure context); pages must be served from localhost.
- `layout: 'auto'` drops bindings a kernel never reads (e.g. a uniform only another entry point uses); a bind group that
  still lists them fails validation. Bind exactly what each entry point uses.
- `pnpm install --offline` fails in a fresh cloud container (empty pnpm metadata cache for `xatlas-web`); run a plain
  `pnpm install` after adding a workspace dependency.
- Since Chrome 145 (`subgroup_uniformity` + `@builtin(subgroup_id)` from 144), control flow on `subgroup_id` is
  subgroup-uniform; `local_invocation_index / 32` is not (verified Chrome 153). Derive per-subgroup offsets from
  `subgroup_id`. Subgroup-matrix load/store accept such offsets without any diagnostic override (Chrome 157, Metal,
  verified 2026-11-15); `local_invocation_index / 32` offsets need `diagnostic(off, chromium.subgroup_matrix_uniformity)`
  (the unprefixed `subgroup_matrix_uniformity` is not the rule's name).
- Apple (M5 Pro, Chrome 157) offers subgroup matrices only as f32→f32 and f16→f16, 8×8×8, subgroup size fixed at 32.
- Never run an "every lane computes the whole tile" emulation on a real GPU: the first emulated sgm form (each of 32 lanes
  doing the full 8×8 or 16×16 tile in a by-value `array<f32, 64|256>`) compiled in 65 ms but one dispatch did not finish
  in 30-70 s on the M5 Pro, and the Mac rebooted during the 16³ run. The emulation now splits each tile across the lanes
  (`sgmEmuElementsPerLane`). Try a new kernel shape on SwiftShader first (`PT_CHROME_FLAGS="--use-webgpu-adapter=swiftshader
  --enable-unsafe-swiftshader"`, `--allow-software --no-lock`, works on the Mac), then one tiny dispatch on the GPU.
- zsh does not word-split `$var`: `node bench/runner/run.ts --args $a` with `a="infer=sgm subgroups=4"` passes ONE argument
  (the page then reads `infer` as `sgm subgroups=4`). Loop in `bash -c '…'` or write `${=a}`.
- The desktop app's browser pane runs WebGPU on the real GPU (canvas works) but exposes no subgroup matrices, and a hidden
  pane throttles `requestAnimationFrame` to ~1 fps. For the sgm backends drive the runner's Chrome flags over CDP
  (`--remote-debugging-port=0`, `DevToolsActivePort` in the profile; `Runtime.evaluate`).
- Branches cut before main's P1.4 load-check change count Electron GPU helpers (Claude, Spotify, Notion, Creative Cloud)
  as Chrome and mark every run `loaded`: take `bench/runner/load-check.ts` (+ test) from main.
- gpuweb issue bodies (not comments) are readable through `gh api -X GET search/issues -f q='repo:gpuweb/gpuweb <n>
  in:number'`; proposal files through raw.githubusercontent.com. developer.chrome.com and googlesource are not
  reachable from cloud sessions; use WebSearch summaries for those.
- WebGPU **canvas presentation** kills the GPU instance on headless SwiftShader in cloud containers ("A valid external
  Instance reference no longer exists"), with every flag combination tried; compute-only pages are fine. Render to a
  texture instead (`mlp-live.html?offscreen=1`).
- Headless `requestAnimationFrame` does not wait for the GPU: a page that encodes work per frame must cap frames in
  flight (`onSubmittedWorkDone`), or the queue grows without bound and later waits never return.
- SwiftShader reports subgroup size 4. A kernel laid out for 32-wide subgroups still runs there only in the emulated
  sgm form (`infer=sgm-emu`), which assumes nothing about subgroup size.
- Vite's dependency scan at the repo root (`mlp:live`) logs an error for `@pmndrs/bvh` when the sibling is missing;
  pages that do not import it still load.
- Worktrees under `.claude/worktrees/<name>` resolve the `link:../../Denoiser/...` dependency (and `link:../../bvh` with the sibling override) to
  `.claude/worktrees/Denoiser` (and `bvh`): those are symlinks to `~/Developer/*` (re-create them if
  typecheck says `Cannot find module 'denoiser'`).
- A WGSL loop that contains `workgroupBarrier()` needs a workgroup-uniform bound: a count read from a storage buffer is
  not. Have one lane copy it into a `var<workgroup>` and loop on `workgroupUniformLoad(&v)` (`wgsl-gradient.ts`).
- `mlp-check`'s `gradient_rel_error` is normalised by the largest gradient: a wrong small-magnitude layer can hide in it.
  The per-weight check is the `mlp-fit --args oracleSteps=N batch=B` replay (use B = the real batch to cover chunking).
- The runner's Vite server (`startBenchServer`, also `bench/live.ts`) has no file watching and no HMR: after editing a page or
  module, restart it, or the browser keeps getting the first version.
- `set` is a reserved word in WGSL (so are many short names); a binding called `set` fails to parse.
- A worktree has no `bench/assets/generated/web/`: run `node bench/adapters/export-scenes.ts <scene>` in it. The main
  checkout's export can be stale against the current scene definition (Sponza was, 2026-11-15).
- A Metal "Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)" device loss in the app's browser pane went away
  on reload (shared GPU, many earlier reloads); retry once before debugging.
- Generated WGSL that repeats a block per layer must wrap each block in `{}`: `var acc0` declared twice in one function scope
  fails. Unrolled tiled kernels pass the ~50 KB guideline quickly (tiled forward with 128 records: 50.4 KB); keep R ≤ 64.
- `mlp-check`'s `update_rel_error` is the max over weights and is dominated by near-zero gradients where Adam's sign decides
  a full step; compare `updateP99` and the `mlp-fit oracleSteps=` replay instead.
- A hook that marks scattered pixels for long work (training paths, update paths) makes every 8×8 megakernel workgroup wait
  on its longest path: NRC per-pixel training cost 2 ms more per 640×360 frame than whole-tile training. Select whole tiles.
- `bench/live.ts` picks the first free port from 5291 (the runner may hold 5291): read the URL it prints. A hidden or
  minimised Chrome window pauses `requestAnimationFrame`; drive headless measurements through `__nrcLive.step(n)`.
- BSD `grep -E` rejects `{0,300}` ("maximum repetition exceeds 255") and the broken pipe can leave the runner hanging; keep
  repetition counts under 256 when filtering runner output.
- Clearing a megakernel's image (`reset`) restarts its sample index: anything keyed on `sampleIndex` (training-tile rotation,
  `updateBlock` update pixels) then repeats while the camera moves or accumulation is off. NRC rotates on its own GPU frame
  counter, and every clear gets a new seed (else each restarted image repeats the same noise).
- NRC wall-clock frame times from the runner's headless Chrome were 2-4x the windowed flagged Chrome's (640×360 cache frame
  37.9 vs 9.0 ms) and not in the same ratio to plain path tracing: do not mix the two; the cause is open.
- Subgroup-matrix training on Apple (N3b): f32 needs R ≤ 32 records per workgroup (R×S×8 B + 16 B per record of targets in
  32 KiB); one subgroup per workgroup is 2-3× slower than four; Apple's only f16 config is f16→f16, so f16 sums are noisier than
  the tiled kernel's (f32 row sums). Short MLP kernels time bimodally on this Mac (0.3-1.0 ms medians on a quiet machine):
  quote min and p25.
- `infer=portable` on the MLP pages is the naive per-record kernel (kept so older CSV rows keep their meaning); the engine's
  default portable inference is the tiled kernel (`infer: 'tiled'`, 8.6× faster in f16). A one-off 73 ms/M reading for tiled f16
  was a load spike (loadavg 5.2): rerun before believing a single bad A/B.
- Windows: the siblings live in `pmndrs/denoiser/` (`bvh`, `denoiser`, `downsampler`), not next to this repo; a worktree needs junctions
  `.claude/worktrees/bvh` -> `pmndrs/denoiser/bvh` and `.claude/worktrees/Denoiser` -> `pmndrs/denoiser/denoiser`
  (`New-Item -ItemType Junction`), then `pnpm install`. That bvh is ahead of the Mac's.
- Windows: Chrome (157 stable, 160 Canary) runs WebGPU on Dawn D3D12 only, and Dawn D3D12 has no subgroup matrices: no flag exposes
  `chromium-experimental-subgroup-matrix` on Windows, and no switch gives Dawn a hardware Vulkan adapter (`--use-webgpu-adapter=vulkan`,
  `--force-webgpu-adapter`, `--enable-features=Vulkan`, `--use-vulkan=native` all leave chrome://gpu at D3D12/D3D11). Matrix arms run in
  dawn.node on Vulkan instead: `node bench/runner/run.ts --page <p> --dawn vulkan --dawn-toggles allow_unsafe_apis,vulkan_enable_f16_on_nvidia`
  (same load checks, ABBA, CSV; `browser` column `dawn.node/<ver> vulkan +toggles`). It is not what a Windows Chrome user gets.
- Windows: dawn.node (`.cache/dawn-node`, `npm install webgpu` there; or `DAWN_NODE=<index.js>`) fails to load Vulkan ("vulkan-1.dll Windows
  Error: 87") until `C:\Windows\System32\vulkan-1.dll` is copied next to `dist/win32-x64/dawn.node`. Without `vulkan_enable_f16_on_nvidia`
  Dawn hides `shader-f16` on NVIDIA Vulkan (driver `storageInputOutput16 = false`) and with it every f16 matrix config (only int8 remain).
  dawn.node does not keep Node's event loop alive while GPU promises are pending (the host keeps a timer), and prints Tint compile logs to
  stdout without a final newline (the host starts each record on a new line). Dawn quantizes timestamps to 65 536 ns by default (a 0.6 ms
  training step read 0.655 or 0.721): the host disables `timestamp_quantization` (the twin of Chrome's developer-features flag; NVIDIA then
  steps at 1024 ns). dawn.node is not pinned by LUID: a page asking for the default adapter got the Intel iGPU, so the host forces the
  power preference from `--gpu`. Stopping a background `sh` chain (TaskStop) leaves its `sh` and runner children running: kill the tree.
- Windows: Dawn/Vulkan subgroup matrices reach the RTX 3060's tensor cores (42 TFLOP/s f16 sums, 21 f32 in `--page sgm-roofline mode=peak`); an MLP-shaped kernel lost 4x of
  that to layout, not to Dawn: f32 sums need a result buffer + convert pass and more workgroup memory (4.4x), 4 subgroups per workgroup vs 1 (2.4-2.7x), an unpadded
  64-half row stride (1.3x, bank conflicts). Measure a kernel shape there (`mode=chain`) before blaming the API. The MLP kernel in that layout
  (`layout=blocked`, N2r-b) went 14.0 -> 5.7 ms per M queries, then 3.85 with constant-trip lane loops (N2r-c): `for (var i = lid; i < N; i += wg)` has a
  runtime START, so Tint adds its loop counter there too; write `for (var j = 0u; j < N / wg; j++) { let i = lid + j * wg; }` (`laneLoop` in wgsl-sgm.ts). f16->f16 configs exist only with `vulkan_enable_f16_on_nvidia`.
- Windows: on NVIDIA the per-record MLP kernel (`infer=portable`) beats the tiled one in Chrome/D3D12 (17 vs 24 ms per M queries, f16), the
  reverse of Apple, and the same per-record kernel is 3.5x slower on Dawn/Vulkan (60 ms): compare kernels within one backend.
- Windows: Canary 160 ignores `--use-adapter-luid` and hands out the Intel UHD; `--gpu nvidia` then fails the run. Use Chrome stable.
- Windows: the user environment sets `CMAKE_TOOLCHAIN_FILE` to a vcpkg checkout that does not exist; CMake then fails at `project()`. Build native tools
  with `env -u CMAKE_TOOLCHAIN_FILE cmake ...` (VkNRC: `-A x64 -G "Visual Studio 17 2022"`, then `cmake --build build --target VkNRC --config Release -- -m`).
- Windows: the RTX 3060 Laptop heats to 80-87 °C within minutes of back-to-back full-load runs (VkNRC, a 1024 spp megakernel cell) and the runner then marks
  rows `thermal throttling` although clocks stay at 1.5-2.0 GHz (reason 0x20, software thermal slowdown). Pause between long runs; quote the start temperature.
- Windows: Tint (all backends, part of robustness) puts a 64-bit iteration counter into every loop it cannot prove finite; on D3D12 with DXC SM 6.6+ such
  counters on loops nested inside a big kernel can cost 3-4x (P-W1: the BVH walks inside the megakernel; standalone kernels did not pay). Only
  `for (var i = <int>; i <op> <literal constant>; i++)` with the bound test first escapes it (a runtime bound, `while`, or `loop {}` does not). For a new
  unbounded loop in a hot kernel use `for (var s = 0u; s < 0xffffffffu; s++)` (WGSL then wants a return after it). To bisect a D3D12-only slowdown,
  dawn.node `--dawn d3d12 --dawn-toggles allow_unsafe_apis,disable_robustness` (or `d3d12_dont_use_shader_model_66_or_higher`, `-use_dxc`; `-x` disables
  a toggle) and `dump_shaders` (HLSL + DXIL in the console with `--verbose`).
- Windows: the `caps` page used to set only `window.__benchResult`, so the runner always timed out on it; it now calls `reportResult`.
  Its subgroup-matrix probe compiles a result type the adapter lists (NVIDIA has no f32 or 8×8 config).
- Windows: a shell gate that counts processes by command line (`Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'bvh-gpu-chrome' }`) run from
  Git Bash matches its own `powershell` and `bash -c` wrappers and never reads 0: filter on `$_.Name -eq 'chrome.exe'` too. It stalled the N2r-d sweep for hours.
- Windows: the Claude app's browser pane runs WebGPU on the Intel iGPU even with `powerPreference: 'high-performance'` (nrc-live 724 ms/frame): open demos in
  Chrome (`Start-Process chrome <url>`). A headless Chrome from another session (profile `bvh-gpu-chrome`) shares the RTX 3060: check before timing.
- Windows: in the blocked sgm training kernel a ReLU mask kept as bits in workgroup atomics (`atomicOr`/`atomicLoad`) was ~10% slower than re-reading a_l
  from global memory (N2r-d); the sgm gradient sums whole K-record steps, so deltas up to `capacity` must be zero (the blocked forward-backward writes them).
- Windows: `nrc-live`'s own requestAnimationFrame loop keeps the RTX 3060 at full load (80-88 °C within seconds); pause it (`__nrcLive.pause(true)`) and cool the card
  before timing. `node bench/live-bench.ts` does that and presses Benchmark over CDP; the page's idle status is the fps line, not empty. A 404 in a page's console fails a runner
  run even when the page handles it: `nrc-study ref=ours` does not ask for the (git-ignored, absent on Windows) Mitsuba PFMs.
- NRC pipeline (M9, docs/internal/design/nrc-pipeline.md): `active` is a reserved WGSL word too (a uniform field named `active` fails only on the GPU). In frame mode the cache's
  resolve must also write the demodulated planes (`resolveWgsl(frame)` in `bench/nrc/cache.ts`), or `diffuse x albedo + specular x specFactor` falls short of the beauty by
  exactly the cache's light; `nrc-pipeline-study` checks the identity (`plane_identity_*`). rg16float / r16float are not storage formats in core WebGPU: the pipeline packs
  into one buffer and `copyBufferToTexture`s (rows 256-byte aligned) instead of writing 12 storage textures (also past the 4-8 storage-texture limit).
- pmndrs/upscaler is three-shaped (`new Upscaler({ renderer })`) but only uses `backend.device`, `backend.get(tex).texture` and `initTexture`: `UpscalerHost` in
  `bench/nrc/pipeline.ts` stands in, no three renderer. Its dist imports its OWN `three` (0.186, from the upscaler checkout), not bench's 0.187: harmless because only
  `.elements` of our camera stand-in's matrices cross over. Link: `@pmndrs/upscaler` `link:../../upscaler` (Windows worktree: junction `.claude/worktrees/upscaler` ->
  `GitHub/upscaler/upscaler`); `bench/runner/server.ts` adds its realpath to `fs.allow`. Its `beginFrame` must still run with `jitter: false`: it snapshots the projection
  `_writeReproject` reads.
- Windows laptop (Intel UHD + RTX 3060): the desktop app's browser pane and a default Chrome window get the Intel GPU (`powerPreference` is ignored on Windows, crbug
  369219127): `nrc-pipeline` ran at ~1 fps there. Use the runner (`--gpu nvidia`, pins the LUID) or set Chrome to "High performance" in Windows graphics settings.
- The Mitsuba / pbrt reference PFMs are not on the Windows machine (only their JSON sidecars; references are git-ignored and were rendered on the Mac). `nrc-pipeline-study`
  falls back to our own 4096 spp render (uploaded as `reference-ours-4096spp`; reuse it with `refFile=/bench/results/incoming/<run>/nrc-pipeline-study/reference-ours-4096spp.pfm`).
- Git Bash (Windows) rewrites a page arg that starts with `/` into a Windows path (`refFile=/bench/...` became `C:/Program Files/Git/bench/...`): prefix the command with
  `MSYS_NO_PATHCONV=1`. A page's `fetch` of a missing file logs a console 404 and the runner then FAILS the run even when the page reported its numbers: probe optional files
  only when an arg asks for them (`nrc-pipeline-study ref=`).
- OIDN in the bench on Windows: `__BENCH_DENOISER_TZAS__` was `/@fs` + a backslash realpath (`/@fsC:\...\tzas`), so every `.tza` 404'd (the error then names
  `rt_ldr_small.tza`, because `Denoiser.create` first loads the default LDR model); `server.ts` builds it with `fsUrl()` now. One Denoiser instance loads ONE network:
  `quality` (fast = `_small`, balanced = base, high = `_large`) and the guides passed (albedo -> `_alb`, albedo + normal -> `_calb_cnrm`) pick it, and a change reloads
  the weights on the next call, so `bench/nrc/pipeline.ts` keeps one instance per (quality, guides, precision). `denoiseTextures` awaits its GPU work
  (`onSubmittedWorkDone`) but never maps a buffer; the pipeline's `post` / `session.frame` are async because of it.
- Judge a young cache through the denoiser, not on the raw frame: after 3 training frames its raw relMSE and bias already look trained, but
  its remaining error is low-frequency and survives SVGF / OIDN (OIDN needs ~150 training frames before the cache arm beats plain, N11.6).
  `nrc-pipeline-study scenarios=coldstart warmups=...` measures it; `NrcSettings.warmupFrames` is the gate.
- The NRC warm-up gate (N11.7, D30) is a budget of training STEPS (120 SVGF, 600 OIDN), not frames: `warmupFrames` counts trained frames, so
  with `warmupStepsPerFrame` set divide the budget by it (`autoGateFrames` in `nrc-pipeline.ts`). `cache.warming` is read before the frame;
  the gated frame is plain path tracing at 1 spp. The demo's headless `step()` must apply the controls too (`applyCacheSettings`), or it
  silently runs with the settings of the last rAF frame (a hidden pane runs none).
- A reference's own noise is a relMSE floor of about (raw 1-spp relMSE) / spp: the dim upper corridor at 2048 spp sits near 7e-3, the size of
  a 4-frame OIDN mean's error there, so ratios against it are pulled toward 1. Estimate the floor before quoting a small-error ratio.
- `NrcCache.readLoss()` is the last step's loss on ONE frame's noisy targets (relative L2: 30-190 on sponza-white, frame to frame): check it
  for drift or NaN (training stability), never read it as a convergence curve; convergence is the image error through the chain.
- `node bench/runner/run.ts --args` takes each `k=v` as its own token: `--args "a=1 b=2"` (one quoted string) sets `a` to `1 b=2` and the page silently runs
  its defaults (an `nrc-check model=hash` run tested the network). Check the echoed args JSON on the PASS line.
- Hash-grid cache (M7, `bench/nrc/hash-grid.ts`): a cell with one or two samples shows its noise as a blotch the size of the cell, and young cells appear after every
  cut, cold start or LOD change while the camera moves (single-frame relMSE spiked to 10.6, above plain path tracing). A sample threshold fixes it only together with
  the pyramid update (coarser cells always populated); a threshold alone turns into dark misses (−4% bias). Jitter makes it worse.
- Cutout shadows (R2b, `integrator.alphaShadow`): the kernels compile bvh instance masks OUT for speed (`rayMasks: false`, so `rtVisible(ray, mask)` ignores `mask`);
  only `'classified'` turns them on (`ptNeedsRayMasks`), and a mask bit then costs a test per instance. Anything new that needs a ray mask must go through `ptRaytracerModuleOptions`, or the
  mask is silently all-ones (the first `classified` run read exactly the solid-walk image). The `rt` walk of a WGSL module is built once per kernel: a mask-using kernel and a plain one differ in text.
- Scratchpad files shared between agents: two agents writing `edit2.py` to the same scratchpad directory ran each other's script (one edited the other's worktree). Put scratch files in your own
  subdirectory (`scratchpad/<task>/`).
- `classifyAlphaCards` (bench/scenes/alpha-classify.ts) must read the STORED level-0 texels (`buildSceneTextures` resamples a 96 px texture to 128), not the input texels: against the input data
  a checker edge is off by a texel and half the triangles of an aligned grid read mixed. Use power-of-two textures for cutout A/B scenes; the contour of a bilinear alpha of a hard edge sits ON the texel
  boundary, so a triangle bounded by it is pure (zero-measure contact, `TOUCH`); the GPU's 8-bit filter weights move the contour by ~1/256 texel (`classified` reads -0.01% of exact on grid16).
- A cutout `roulette` shadow (q) is unbiased only because solid blockers are a superset of exact ones; it needs the solid walk to run FIRST for every shadow ray, so at q = 0.25 on a scene where
  most shadow rays are blocked it costs MORE than the exact loop (the extra any-hit plus q of the loop). Judge it by time to equal noise (`eqTime` of `alpha-shadow-ab.ts`), never by ms/spp alone.
- Merging a long-lived branch after another track took numbers (R2b after track N): DIVERGES V / DECISIONS D numbers collide silently (both sides added "V140"). Check `git show main:DIVERGES.md | grep -oE '^\| V[0-9]+' | tail` first and renumber the branch's rows (and every reference in TASKS, code comments, reports) BEFORE cherry-picking: a throwaway branch from the old commit plus `perl -pi -e 's/\bV14([0-4])\b/"V14".($1+4)/ge'` on the files that name them. `OWN_ADAPTER_IDS` (`bench/adapters/registry.ts`) is one long line: a conflict there joins two lists as `'a''b'`; typecheck says "',' expected".
- Before/after kernel check across a merge: `git worktree add --detach .claude/worktrees/main-snap <main sha>` inside your own worktree, `pnpm install --offline`, copy `bench/assets/{generated,downloads}`, and `node bench/size-snapshot.ts <dir>` in each; `diff -rq` must show only `Only in` new files. `link:` deps resolve from `.claude/worktrees/`: symlink `bvh`, `Denoiser`, `upscaler` there (to `~/Developer/...`). The nested worktree is not committed (`git worktree remove` it when done).
- Real-time cutouts (D31): `mode: 'preview' | 'realtime'` sets `alphaShadow: 'transmit'`, which only works if the cutout cards are in instances carrying `alphaMixedMask`; without that the kernel compiles and the shadows silently equal the solid walk's (-12% on foliage). The bench adapter does the tagging; a library caller must set `instance.mask`. `ptApplyMode` skips the preset's `alphaShadow` / `alphaTransmit` when the caller sets a non-`exact` `alphaPolicy` (the max-speed opt-in), because an explicit `alphaShadow` overrides the policy's shadow side.
- Do not run `pnpm test` / vitest while a timing page is running (a 10 s vitest run during an `alpha-shadow-ab` page shows up as a burst in that page's rounds); queue them after, and record the load next to every number.
- Real-time chain without a cache (P4.5 / P4.6, `bench/realtime/`, page `realtime-pipeline`): the chain (`chain.ts`, was `bench/nrc/pipeline.ts`) and its session (`session.ts`) do not care what fills the frame planes;
  `nrc/pipeline-session.ts` and the nrc pages import `../realtime/chain.ts`. A worktree also needs `.claude/worktrees/upscaler` -> `~/Developer/upscaler` (the `@pmndrs/upscaler` link resolves there) next to the bvh / Denoiser links.
  References for it are OUR OWN plain tracer at the chain's path length (maxDepth 2, no RR) rendered at the DISPLAY size; the pbrt / Mitsuba references are 8 vertices, a different expectation (sponza-white at maxDepth 2 has mean luma 0.173 against 0.241 at the scene's 8 vertices: 28% darker).
- The upscaler's darkening (N11.5) depends on input noise: fed the noise-free reference it reads +0.2%; behind SVGF (residual noise) -7%; fed the raw 1-spp frame -90% (it clips the noisy highlights). Never feed it a raw 1-spp frame, and judge a chain
  bias only at the stage that has the input noise of the real chain. The `control` scenario of `realtime-pipeline` is the noise-free test.
- An emitter seen directly by the primary ray breaks the demodulated chain: the albedo factor of an emissive black material is the floor (1e-3), so the diffuse plane holds emission / 1e-3 (4.8e5 for L = 480, f16 clamp 65000 in `pipePack`) and SVGF spreads it.
  `sponza-lamps` keeps its lamps at 1 m facing down (below eye level, only their black backs are visible) for that reason; a scene with visible lamps needs emission outside the demodulated planes (open, realtime.md section 5).
- ReSTIR keeps the Owen-Sobol sampler in `mode: 'realtime'` (modes.ts, the passes do not bind the blue-noise texture yet), so with ReSTIR the frames are not blue-noise: the on/off comparison in `realtime-pipeline` also changes the sampler.
  Biased ReSTIR (`bias: 'biased'`, the default) darkens more every frame under temporal reuse (maxDepth 1 on sponza-lamps: -0.1% at frame 1, -12% at frame 64): judge it by relBias next to relMSE and FLIP, not by relMSE alone.
- Timing a chain stage (P4.5): record N frames into ONE command buffer (`session.timeEncoded`, >= 250 ms per round) for what is one encoder, and queue N calls then await once (`session.timeCalls`) for what submits its own command buffers (`arm.post`: the denoiser and the
  upscaler). Rounds agree to 0.3-5% on a Mac at load 3-10 (min..max are in the metrics). `runner` commands over 120 s are moved to the background by the agent shell: start them with `run_in_background`, never poll with `sleep`.
- The reactive mask reads the denoiser's `accum` debug texture (`DiffuseSpecularDenoiser` has a private `pipes`; `debugTexture('accum')` alpha = history length): an internal API of the sibling, it may move. `PipelineArm.reactiveEnabled` switches it at run time.
- `sponza-lamps` (tier 2) needs `node bench/adapters/export-scenes.ts sponza-lamps` before a page loads it (the web glb is keyed on the whole SceneDesc, lamps included).
- Emission plane (P4.5a, `integrator.frameEmission`, plane 6 of `PT_FRAME_PLANES` = 7): light the primary BSDF does not modulate (an emitter seen by the primary ray, the sky behind a miss, and with the default `'mirror'` what perfect-mirror
  bounces reach) is written to plane 6 and added back AFTER the filter (`bench/realtime/chain.ts` composite); everything else is demodulated as before, so `diffuse * albedo + specular * specularFactor + emission = radiance`
  (check `realtime-check check=emission`, which also has the closed forms for a lamp, the sky and a mirror). Without it a lamp's emission / albedo floor (1e-3) hit the f16 clamp and SVGF spread it (denoised relMSE 228 on
  `sponza-lamps-visible`, 1e4 on Cornell). Traps: (1) the frame kernels grew 1.1 KB and the plane buffer 1/6; every non-frame kernel is byte-identical (`node bench/size-snapshot.ts`, only `rt-frame*` / `rt-pbr-albedo-frame*` /
  `restir-shade-realtime` change); (2) the pipeline page's `realtime-pipeline` plane identity must include `emission`; (3) `PipelineArm`'s pack has 8 rgba16float sections (emission is section 7) and `sums` 6 vec4: a bench page that
  builds its own pack must follow; (4) a sub-pixel emitter still flickers at its silhouette (binary 1-spp coverage, plane not filtered): judge lamp regions with `emitThreshold=8` (`_emit` / `_rest` metrics, `_clamp_hits`),
  and note that the whole-image relBias of a lamp scene is mostly the lamps' energy (the upscaler darkens them).
- Wavefront real-time (P4.1b / P4.2a, V210-V212): `integrator.frame`, `sampler: 'bluenoise'` and `integrator.restir` run on the PLAIN wavefront only (refill / layers throw). The frame planes live in the accumulator TAIL at `ptFrameBase(n)`
  (256 B aligned; `frameBuffers` returns `{ accum, planes: accum, planesOffset }`, the bench chain binds `planes` at that offset), so no kernel gets a storage binding. Planes 0 / 1 accumulate PRE-DIVIDED contributions (`ptFrameAdd`: shade and shadow kernels read the class
  and factors back from planes 4 / 5), which agrees with the megakernel to <= 5e-7 per pixel (`--page wf-frame-check`, geometry planes bit-identical): a difference above 1e-5 is a bug. The previous camera is a second dynamic-offset uniform
  (binding 13, shade kernel only, written at the iteration-0 record, bind with `[offset, offset]`); ReSTIR's final-reservoir region rides in `ptParams.up.w` (`limits.w` is the wavefront's iteration). `frame` + `aovs` both want the tail: refused.
  `ptApplyMode` no longer drops `frame` / `bluenoise` for the wavefront. A scene resized after load (`setFilm`) works for file scenes; the SceneDesc's own film does not (export is keyed on it).
- `wf-frame-check timing=1` is the paired ABBA timing of both schedulers in ONE page (same device, same scene copy): prefer it to two runner runs, which interleave with other agents' lock holders for minutes each.
- Alpha MASK x R1 x cone LOD on the wavefront (P4.1c): the cutout test runs in the INTERSECT kernel, which is built with `materialLod: 'full'`, so anything the megakernel's per-hit `mat` read does with `lod` (the widened footprint
  `PT_LOD_TEX_BIAS` at depth >= `materialLodDepth`) must be handed to intersect explicitly (`PtwfConfig.lodTexDepth`); without it the two schedulers skip different cutout texels at depth >= N (up to 32% per pixel in `mode: 'realtime'`,
  +1..3% mean on foliage-mask). Needs all of MASK + cone + secondary to show, so test the combination: `wf-frame-check cases=foliage,foliage-lod2,tex-mask` (radiance plane; the diffuse plane's alpha is the first bounce distance and differs by design on MASK scenes: `rgbOnly`).
  Any new texture read the megakernel does under `lod` before the hit is accepted has the same problem in intersect.
- D32 default flips (2026-11-18): scene copy `packed` and `albedoCorrection: 'texture'` are the library defaults, so a test or page that wants the old text passes `packing: 'split'` / `albedoCorrection: 'none'` explicitly
  (the size-snapshot `split` set is byte-identical to before except the layered-albedo kernels, which carry the correction texture: `node bench/size-snapshot.ts <dir> split`). The full-feature albedo megakernel is 50.2 KB: `ptwfAutoPolicy({ kernelBytes })` sends it to the wavefront.
  The bench `pt-megakernel` adapter still defaults to `layering: 'stochastic'`; the `pt-mk-lay-int` / `pt-wf-lay-int` presets now say `albedoCorrection: 'none'` to stay the uncorrected baseline. zsh does not split an unquoted `$VAR` of options: write the `matrix.ts` flags out.
- Media (P5, `docs/internal/design/volumes.md`): `integrator.volumes` runs on both schedulers (the wavefront since V270) and refuses restir / frame / vbuffer / alpha MASK / hooks / counters / AOVs / the P2.22 Beer absorber; the default kernels stay byte-identical (`node bench/size-snapshot.ts`, 224 variants).
  A medium boundary is a CLOSED, OUTWARD-wound mesh with `MaterialDesc { type: 'interface' }` (or any material for a glass boundary): `MeshDesc.medium.interior` is the side the normals point away from. A scene with media gets one scene-copy material per
  (material, medium pair); `DescSceneCopyInput.materialVolumes` is indexed like `materials`. Boundary surfaces are skipped by camera, bounce and shadow rays (`ptVolTr` is a closest-hit loop); any other surface blocks a shadow ray, so glass shadows its inside.
- Media references: use GREY `sigma_a` / `sigma_s` (pbrt uplifts RGB spectrally: a chromatic absorber reads 0.718 vs the closed form 0.7408 on `vol-absorb-slab`, Mitsuba is exact; V62). pbrt keeps `MediumInterface` set before `Camera` as the default of every later
  shape (the exporter resets it with `MediumInterface "" ""` after `WorldBegin`). Mitsuba `heterogeneous` returns a rare NaN sample (about 1 pixel per 3000 at 512 spp): `mitsuba_render.py` drops that pixel's chunk and reports `nan_pixel_chunks`.
  Grid media keep a zero border of at least one cell (pbrt reads zero outside the grid, Mitsuba `gridvolume` clamps); Mitsuba's grid medium takes a grey sigma_t. `max_depth` = ours + 1 holds with media (the single-scatter slab at max depth 1 reads the quadrature to 1e-4 in Mitsuba).
- Media estimator (V240): a path vertex inside a medium is a vertex like a surface one (`depth >= maxDepth` ends the path BEFORE the scatter, as pbrt `volpath`); MIS weights use directions only. A tracking loop draws from the per-path PCG stream (`ptVolSeed`), the NEE / phase directions from the
  sampler pairs of the vertex. `ptSampleLd` gains `medIn` / `medOut` when volumes are on (the medium on each side of the surface, equal for a surface without a boundary). The grid is a `texture_3d<f32>` (`r32float`, binding 16; brick majorants binding 17).
- OpenVDB leaves (WebVDB `parseVdb`): value index = `x * 64 + y * 8 + z` (x slowest) within the 8^3 leaf, `valueMask` word `i >> 5` bit `i & 31`; checked against `readValue` on 39,936 voxels of `smoke.vdb`. `bench/scenes/volumes/vdb-grid.ts` is Node only; `vol-vdb-smoke` needs its output (git-ignored).
- A matrix cell of `pt-megakernel` waits on `/tmp/rt-gpu.lock` behind every other agent's page (a 1.3 s render took 4-30 minutes of wall time on a day with 6 agents): queue the cells in one background `matrix.ts` call, wait with an `until` loop on its log, and keep working on CPU-side code meanwhile.
- Spectral mode (P5, `integrator.spectral`, docs/internal/design/spectral.md): a path carries THREE wavelengths in the vec3 lanes; the uplift happens in `ptSpecMaterial` (a copy of the decoded `RtMaterial`), `ptSpecIllum` (emission, NEE `li.radiance`,
  sky, sun) and `ptSpecToRgb` (return). A new place that reads an RGB colour or radiance in the integrator must go through one of them, or it is added in RGB to wavelength lanes (silently wrong colour, grey scenes still pass). The table
  texture is group 2 binding 18 (`PT_SPEC_BINDING`; see the P5 group-2 binding table below); a page that builds its own layout needs it. `defaultRgb2SpecTable(64)` fits in ~4 s: `node bench/spectral-table.ts` writes `bench/assets/generated/rgb2spec-64.f32` (git-ignored; copy it into a
  worktree with the other generated assets). `RtMaterial.dispersion` is in the high half of record word 28 and NOT in the WGSL struct (`ptMaterialDispersion` reads it). Dispersive scenes are skipped by the adapter unless `pt_spectral=1`.
  Greys must stay exact (furnaces 0.5000 / 1.0000); a saturated colour differs from the RGB render by up to ~2% (that is the point: pbrt's spectral uplift), so judge it against pbrt, never against RGB mode.
- `bench/references/render.ts` skips a scene whose PFM exists, even if you changed the SceneDesc (`(cached)`): pass `--force` after editing a scene. Worktree: `ln -s <main>/.cache .cache` gives it the native pbrt / Mitsuba builds (do not commit the link).
- Several agents queue on the one GPU lock: a `matrix.ts` cell took 2-3 minutes of wall time for 1-20 s of render, so a 20-cell batch is an hour. Put every cell of a study in ONE queued script and do not poll.
- ReSTIR GI (P5, `integrator.restir.gi`, docs/internal/design/restir-gi.md): a sample is a SURFACE point x2 (+ n2, the radiance Lo it reflects toward x1), so its contribution weight is in the AREA measure and the reconnection Jacobian is just the ratio of
  `G = cos(phi2) / r^2` between two pixels (no separate factor: do not multiply it in twice). Its target is SHADOWED (a reused sample must be visible from its new pixel; there is no shade-time shadow ray, unlike DI's unshadowed target); `Lo` is
  fixed, so only a Lambert x2 can be shifted (the `diffuse` flag), every other sample has `phat = 0` away from its own pixel. GI + DI is a depth-2 estimator: compare against OUR plain path tracer with `setLimits({ maxDepth: 2 })`, never against the scene's
  own depth (Cornell +11.8%, Sponza-white +39%), and against `sampler: 'pcg'` for noise (the DI trap again). `lights-mix-lamps` is a flat plane under lamps: GI is a no-op there; use Cornell, Sponza-white, env-sphere.
- The GI passes are a second pruned module (`ptRestirBuild(.., 'gi')`; the DI entry points are left out of its text, `ptRestirWgsl(.., part)`), because DI + GI in one module was 53 KB. A new GI-only helper that DI already has by name (`ptRsLum`, `ptRsBsdf`) is
  reachable only if a GI entry calls it; the `restir-gipasses-*` rows of `size-snapshot.ts` are the sizes. The G-buffer word of the geometric normal (`gl.ng`) exists only with GI (the other layouts and kernels are unchanged: `diff -r` the snapshots).
- A GI sample's `Lo` is stored as f16 / max(Lo) plus an f32 scale (2 words); W stays f32. Do not widen the reservoir past 2 vec4u per pixel per region without checking the 128 MiB binding limit (1280 x 720 x 4 regions x 32 B = 118 MB).
- `restir-check check=gi` args: `modes=gi_off,gi_ts_unb,...` (one mode per run is much faster), `diag=1` (bias by octave of the reference luminance: it is noisy at the dark end, binning on a noisy reference is biased), `seeds`, `k`.
- MaterialX (P5, `packages/pathtracer/src/materialx/`, `docs/internal/design/materialx.md`): `integrator.materialx` (generated `ptMtlxApply` / `ptMtlxNormal`, approach b) needs `textures` (the kernel only decodes uv, position and tangent
  with them): a graph that reads no image still gets a 1x1 placeholder texture from `buildMaterialX`. The text goes into `ptShadingModulesWgsl`'s parts BEFORE the sampler (the wavefront takes `parts.slice(-1)` as the sampler). Every kernel without it is
  byte-identical (`node bench/size-snapshot.ts`; the `mtlx-*` / `wf-mtlx-*` files are new). A generated helper that is called but not declared (a missing `ctx.need`) only fails on the GPU: `bench/scenes/materialx.test.ts` checks it.
- MaterialX values are four lanes everywhere (TS `MtlxValue`, WGSL `vec4f`): a one-number nodedef default fills all lanes (`clamp.high` = 1 must be (1, 1, 1) for a colour), a scalar feeding a vector input is splatted, and only the lanes the type has are meaningful. Compare `dim(type)` lanes, never all four.
- A baked MaterialX material (approach a): a constant `base_weight` stays a record factor, so only a VARYING weight is folded into the colour texture (folding both applied it twice). Bake over the uv square [0, 1]^2 only: mesh uv must stay in that range. `bakeMtlxMaterial(m, { texels: false })` is the cheap plan (parameters, notes, texture sizes) the scene description is built from; the texels are lazy (`generator: 'lazy'`, `registerLazyTexture`).
- `standard_surface` defaults are not OpenPBR's (base 0.8, base colour 1, specular roughness 0.2, coat roughness 0.1, sheen roughness 0.3, thin-film IOR 1.5): a constant standard_surface scene equals its native `openpbr` twin only when the twin carries them (`mtlx-const-std-ref`). MaterialX 1.39 documents write multi-output names `x`/`y` (1.38: `outx`/`outy`); both are read. `copper` of the 1.38 examples writes a colour with spaces instead of commas: the number reader splits on both.
- MaterialX `position` / `normal` / `tangent` are the WORLD hit values in the kernel (V260): a procedural pattern on a translated instance is offset against MaterialX's object-space result. The TypeScript twin and the check page use the same convention, so they agree with each other.
- Group 2 binding table of the megakernel / passes (P5 merge; every constant lives next to its feature, grep `_BINDING`): 0-3 params / accumulation / counters etc. (the scene copy and its textures are group 1, not 2), 9 V-buffer, 10 layered albedo correction (`PT_LAYER_CORR_BINDING`), 11 ReSTIR final DI region (`PT_RESTIR_SHADE_BINDING`), 12 blue noise, 13 / 14 frame params / planes, 15 ReSTIR sharedHit (`PT_RESTIR_HIT_BINDING`),
  16 volume density texture, 17 volume brick majorant (`PT_VOLUME_BINDING` / `PT_VOLUME_MAJORANT_BINDING`), 18 spectral table (`PT_SPEC_BINDING`, was 15 on its branch), 19 ReSTIR GI final region (`PT_RESTIR_GI_SHADE_BINDING`, was 16 on its branch).
  MaterialX uses no binding of its own. ReSTIR passes module (separate group 2): 0 params, 1 G-buffer, 2 reservoirs, 3 hit, 4 GI reservoirs. Volumes, spectral and ReSTIR (DI or GI) refuse each other (`resolveIntegratorOptions`, test
  `integrator/p5-combinations.test.ts`) and volumes + spectral also run on the wavefront (V270-V272, last Traps; 16 / 17 bound in its intersect and shadow kernels, 18 in its shade kernel); ReSTIR GI runs on the wavefront too (V280); the numbers are unique anyway so a later lift of a refusal cannot collide. A new group-2 binding takes the next free number (20+).
- MaterialX x spectral works (the shade does `ptSpecMaterial(ptMtlxApply(...))`: the graph in RGB, then the uplift; `mtlx-uv-codegen` means agree to 0.3% with and without spectral, the constants stay per-pixel equal); MaterialX x volumes is untested.
- The P5 branches each added `volumes` / `spectral` / `restirGi` / `mtlx` to `ResolvedIntegratorOptions`: the default-options `toEqual` in `integrator-wgsl.test.ts` lists them all; a new option goes there and in `resolveIntegratorOptions`.
- ReSTIR GI on the wavefront (P5.6, V280): per-pixel comparison of a wavefront frame with the megakernel's needs `sharedHit: false` on BOTH (the megakernel's real-time preset adds `sharedHit`: V-buffer unorm16 barycentrics shift viewZ by 7e-5 and reservoir picks flip, so DI alone read 1e-1 on a diffuse plane); `wf-frame-check restir=unbiased|gi|giu` does it. The realtime preset's `materialLod: 'secondary'` is refused with GI (case `pbr-full` keeps the full material: adapter setting `materialLod: 'full'`). The GI shade binding is 19 (DI 11); a split scene copy then has 10 of 10 Apple storage buffers in the wavefront shade kernel, so counters / AOVs with GI need `packed`. The same region index serves DI and GI (`up.w` on the wavefront, `limits.w` on the megakernel): `ptRestirGiShadeWgsl(evalFn, base)`.
- zsh does not word-split an unquoted `$a` (`--args $a` arrives as ONE argument and the page silently ignores the second key): use `${=a}` or write the args out.

- Interval queries (P5.rc1, `interval.ts`, `interval-batch.ts`, docs/internal/design/split-rc-tracing.md): `rtClosestInterval` is `(t0, t1]` (the Split RC convention; `rtClosestIntervalLo` is `[t0, t1)`), exact on the f32 `t` the walk computes.
  bvh's slab test culls a hit at exactly `tMax` (12.9% of rays whose tMax is their own hit t on the plain walk, 0.4% at tMin), so the closest forms walk a widened range and cut back (V290); the any-hit forms
  (`rtVisibleInterval*`, output `visible`) are NOT boundary exact: use `distance` where a boundary hit matters. A bit-equality test between an interval's answer and the unbounded walk's needs the tie rule (two triangles at one
  `t` swap with the range): `sameOrTie` in `rc-interval-check`. The edge classification of V27 now lives in `bench/pages/trace-compare.ts` (trace-core imports it).
- `RtIntervalTracer`: the uniform is a dynamic-offset ring of 256 slots per `beginFrame()` (a repeat of the previous job takes `reuseParams: true`, no slot), results are probe-major and cell-minor, `probes[i].w` is the interval scale
  (0 skips the probe in place), a shaded kernel with the scene copy is 46.9 KB only because its module is minified (`minify` defaults true there). GPU `sin` / `cos` are 3.3e-5 off on Apple: `rtOctEqualAreaDecode` normalises, and a
  probe bundle's directions differ from the TypeScript twin's by that (0.1% of rays hit another triangle).
- A Python `s.replace(old, new)` where `old` came from `s[s.index(a):s.index(b)]` is the empty string when `b` occurs before `a`: it inserted `new` between every character and made a 19 MB file. `assert old and s.count(old) == 1` in every
  edit script (and keep the scripts in your own scratchpad subdirectory).
- Media and spectral on the wavefront (V270-V272, `node bench/runner/run.ts --page wf-p5-check [--args cases=... variants=plain,inline,batch,ordered,layers2,subgroup,sorted,refillsg spp=16 timing=1]`: megakernel vs wavefront per pixel in ONE page, one hold of the GPU lock; `matrix.ts` with many cells re-queues for the lock per cell behind other agents' multi-minute holds, so use the page for agreement). Media: free flight and interface skips live in the INTERSECT kernel (a loop around `rtClosest`, like the alpha-MASK loop), `hit.w` = 2 scatter event (distance in `hit.t`, `beta` already scaled), 3 dead path; the scatter vertex is a branch at the top of shade; the path state's section 4 holds `(stream, medium | skips << 16, 0, cone width)`; the shadow-ray record has a 4th vec4 `(start medium, ratio seed)`. The volume random stream protocol is shared by both schedulers and must change in both: a free flight draws from the path stream, an ESTIMATED shadow (grid or `shadowTracking: 'ratio'`) draws from its own stream forked by `ptVolFork()` when the ray is built (the path stream moves one step); the analytic shadow draws nothing. Never trace a shadow ray on the path's stream in either scheduler. `ptVolumeWgsl(v, parts)` hands out pieces: a kernel that includes `tables` must include `seedOf` too (`ptVolSeed` calls it); `ptVolTr` is the wrapper (fork + `ptVolTrCore`) only for estimated shadows, else it is the body, so a kernel names it with `ptVolumeRatioShadows`.
- Spectral on the wavefront: the shade kernel calls `ptSpecBegin(ptSample2D(pixelSeed, sampleIndex, 1u).x)` per shaded path (wavelengths are not state) and restores `ptSpecHero` from bit 30 of the depth word (the AOV bit; spectral refuses AOVs); shadow rays carry RGB (`ptSpecToRgb(beta * ld.contrib)`, linear, so per-add conversion equals the megakernel's one conversion); the shadow kernel has no spectral code. `materialLod: 'secondary'` with media or spectral throws on the wavefront (the reduced shade kernel has no scatter vertex / wavelengths).
- Checking a lock-step RNG protocol: the first run of media on the wavefront agreed with the megakernel to 7e-7 (grid and absorbing scenes bit-identical) in all refill modes. Judge a per-pixel check by the pixel's largest |difference| over its largest component (`relpx`), not per channel: the spectral XYZ -> RGB matrix cancels in a channel near zero (per channel 1.2e-4 where `relpx` is 6e-7). A scene whose paths amplify tiny differences (long dispersive glass: `DispersionTest-notex`, TASKS P5.s9) flips a few hundred pixels while the means agree to 2e-6: sweep the path limit (`dispersion-test-d2..d16`) and add controls (the megakernel built for the other scene packing against itself, `control=1`) before calling it a bug or noise; here the hero IOR differs between the two compiled kernels by more than an ulp (rounding it to 2^-16, 2^-12, 2^-9 or the hero wavelength to 1/16 nm did not absorb it), cause open.
- Wide BVH (X1.i, bvh 0.1.0 experimental): a scene is assembled with `buildTraceScene({ nodeFormat: 'bvh4' | 'cwbvh8' })` (`SceneTrees.create({ nodeFormat })`) and the kernel must be compiled for the
  same format (`sceneFormatOf(trace)` / `format` of `PtMegakernel` / `PtWavefront` / `RtBatchTracer`; `setScene` throws on a mismatch). Wide formats default to `wideDecode: 'folded'` here (bvh's own default is `exact`).
  The wide walk is ~4-5 KB more WGSL than the binary one (plain megakernel 40.7 -> 44.9-46.1 KB) and has six `loop {}`s: `rtBoundWalkLoops` bounds them like the binary walk's four, but only the binary one was ever
  timed on D3D12. Page args: `pt_bvh`, `pt_bvh_decode`, `pt_bvh_walk` (competitor), `nodeFormat` / `wideDecode` / `wideWalk` (trace-timing), `formats=` / `treesFormat=` (trace-core). A hit on a CWBVH scene may name another
  one of two duplicate triangles (it reorders triangles), so Sponza images differ in ~0.15% of values from the binary scene's: judge it with trace-core's classification, not `cmp`.
- bvh 0.1.0's closest-hit walk takes the right child first when two boxes tie unless the ray starts inside both (+24 bytes per walk kernel; 4-12 pixel values change on cornell / sponza-white): a before / after image `cmp` across the
  bvh bump is not byte-identical, and the kernel texts differ by that one line. `node bench/size-snapshot.ts` writes the `bvh4*` / `cwbvh8*` wide kernels too.
- `bench/pages/*.html` is needed for every new page (a copy of scene-copy.html with the name changed), or the runner says "No such page".
- GPU mips (X5, D34): GPU mips are the DEFAULT: `TextureOptions.mips` is `'gpu'` unless set, so `buildSceneTextures` / `buildSceneCopy` keep level 0 only (`sampleSceneTexture` throws above it; the scene hash covers level 0 only) and `uploadSceneCopy`
  creates (once per device, `rtMipGenerator(device)`) a `Downsampler` from `@pmndrs/downsampler/webgpu` (now a dependency of `packages/raytracer`) and adds RENDER_ATTACHMENT (+ STORAGE_BINDING for unorm) to the texture arrays. Anything that
  samples the CPU twin above level 0 (oracle pages / tests) must build with `mips: 'cpu'` (texture-check builds a second, CPU copy as its twin); the adapter setting `pt_mips=cpu` is the old path. Images only differ where a real texture is read at level > 0
  (`textureLod: 'cone'`): the default `textureLod` 0 never samples a mip, so most A/Bs of the two are bit-identical (tex-checker's 0/255 checker boxes are exact in both chains too): use MetalRoughSpheres to see a difference. The Downsampler's own device wants
  `requiredFeatures(adapter)` / `requiredLimits(adapter)` for six levels per dispatch; with the default limits it does fewer per dispatch, same result. sRGB chains take its render or buffer path (same bits).
- Bindless (P1.4c2, `TextureOptions.binding: 'table'`): Chrome 160 Canary on the M5 lists `chromium-experimental-sampling-resource-table` (caps tier 3; `--chrome "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary"`),
  but Metal fails any COMPUTE pass that sets a table (internal error, device lost). RENDER passes work on Metal: there the table goes in the pass DESCRIPTOR (`beginRenderPass({ resourceTable })`; a render pass
  encoder has no `setResourceTable`, by the proposal's design). `bindless-check check=fragment` uses that to test divergent indices on the real M5 GPU: correct, bit-identical to arrays. SwiftShader
  (`PT_CHROME_FLAGS="--use-webgpu-adapter=swiftshader --enable-unsafe-swiftshader"`, `--allow-software`) runs the compute path but takes the table index
  from lane 0 of each 4-wide subgroup even when that lane has exited: only coherent lookups are right there (a SwiftShader / Dawn-Vulkan bug: the proposal puts no uniformity rule on the index, the implementation must emit `NonUniform`), a textured megakernel reads 25-72% dark, and an out-of-range index from a dead lane sometimes kills the instance ("A valid external
  Instance reference no longer exists"; two table megakernels in one page always did: render each binding in its own run with `upload=1` and `cmp` the PFMs). API facts: `device.createResourceTable({ size })`, `table.insert(view | sampler)` returns
  the index, `pass.setResourceTable(table)`, WGSL `enable chromium_experimental_resource_table;` + `getResource<T>(i)` / `hasResource<T>(i)`; an explicit pipeline layout needs `usesResourceTable: true` (validation error otherwise); a 1-layer texture's default view is
  `2d`, and `getResource<texture_2d_array<f32>>` of it silently returns a default (black) texture: insert `createView({ dimension: '2d-array' })`. A wrong type never errors, so test every new `getResource` against a CPU twin.
- `ScalarReduction` (downsampler) with `load` / `finish` hooks: group 1 is the tool's (`reduction.bindGroupLayouts[1]`), `encode(encoder, { width, height, bindGroup })` records both dispatches in their own passes; the
  `finish` output buffer may be the tonemap uniform when it has STORAGE | UNIFORM usage (separate passes). The texel count (`weight`) is exact at odd sizes.
- Docs and examples site (S, 2026-11-20): `docs/*.mdx` is the public pmndrs/docs site; internal notes live in `docs/internal/`. The docs CLI builds plain
  `.md` too (as pages and `/md/` exports; `nrc-diverges.md` even fails to compile as MDX), so `scripts/build-site.mjs --docs` builds a copy of `docs/` without
  `internal/` (`dist/.docs-src`): never point the CLI or pmndrs' reusable workflow at `docs/` itself. `examples/catalog.ts` is the one list the gallery, the injected nav, the Vite inputs and `docs/getting-started/examples.mdx` read: after
  editing it run `node scripts/build-site.mjs` (CI runs `--check` and fails on a stale page). Example pages never import the nav: `examples/vite.config.ts`
  injects it (and `bench/hero/build-pages.mjs` for the Classroom, `HERO_OUT`). `?offscreen=1` draws to a texture: use it for headless checks
  (`globalThis.__progressive.readImage()` reads the image). SwiftShader needs minutes for 256 spp at 640x360; check images at 320x200.
- The glTF reader, OpenPBR mapping and scene grouping live in `packages/raytracer/src/gltf/` (public `loadGltf`); `bench/scenes/gltf-*.ts` and `openpbr.ts`
  wrap them (file I/O, reference class, `SceneDesc` shapes). A mapping change goes in the package, and the Tier 1 goldens test it (download the models with
  `node bench/assets/fetch.ts` or 18 golden tests skip). Never pass a texture field as `undefined` to `createMaterial` (`roughnessMetalTexture: undefined`
  throws): spread it in conditionally. `bench/scenes/tangents.test.ts` fails without `bench/assets/generated/env-sky-512x256.pfm` (`node bench/scenes/tier0/generate-assets.ts`).
- pmndrs/docs sorts every page by its frontmatter `nav` across the whole site and groups the sidebar in that order (ties: folder order, alphabetical): each section keeps its own range
  (getting-started 0-9, concepts 10-19, raytracer 20-29, pathtracer 30-39). The sidebar label is the folder name.
- Tweakpane 4 needs `@tweakpane/core` installed next to it for its types (`Pane` has no `addFolder` otherwise).
- Examples that call `scenePicker` must not read the returned `picker` inside the callbacks they pass to it: it calls `onScene` and `onCamera` before it returns
  (`Cannot access 'v' before initialization` in the built bundle). Headless checks of an example: serve `examples/dist`, open `<page>/?offscreen=1&scene=Cornell%20box`
  (the Cornell box needs no network), wait for `globalThis.__progressive.samples`, then `readImage()`; the Ray tracer examples expose the same through `exposeForChecks`.
  The `rtOffsetOrigin` / `rtSideNormal` helpers exist only in a module built with `shade`: a kernel without it declares its own.
- WGSL `select(a, b, cond)` evaluates BOTH arms: `select(0.0, rtCascadeEnd(n - 1u, ..), n > 0u)` called the loop with `n - 1u` = 2^32 - 1 at n = 0 and hung the GPU (found by the
  intervals example, 2026-11-20; `interval.test.ts` now forbids a looping helper inside `select`). Guard any call that loops, divides or indexes with an `if`, never with `select`.
- Example smoke / thumbnails (S7): `node scripts/examples-smoke.mjs [--only a,b] [--thumbs] [--gpu] [--url <server>]` opens every catalog example with `?offscreen=1` and the entry's
  `check.query` on its DEFAULT scene (AO once passed on the Cornell box and failed on every textured model: `materials: false` with a texture layout throws), waits for `__progressive.samples` / `__realtime.frames` (`check.samples`), and reads the
  canvas back through `__exampleTarget` (the offscreen texture the template exposes). A new example passes only if it publishes one of those globals (`ProgressiveRenderer` and `exposeForChecks` do;
  a page that draws nothing sets `check: { image: false }`). The gallery finds `examples/<path>/thumb.webp` through `import.meta.glob` (a plain `src` path would not be copied by Vite). The script's
  stdout is per example: do not pipe it through `tail` in a background run (nothing shows until the end).
- Built-in blue noise (S10): `loadBlueNoise()` reads `packages/raytracer/src/noise/builtin/<name>.ts` (base64 of the committed PNG, `png: string` so the .d.ts stays 200 B, not 105 KB). After regenerating the PNGs run `node scripts/embed-bluenoise.mjs`; `bluenoise-builtin.test.ts` fails on drift. `mode: 'realtime'` now always selects the `bluenoise` sampler and `create` loads the default texture, so a test that wants Owen-Sobol in real-time mode sets `integrator.sampler` explicitly.
- Publishing (`@pmndrs/raytracer`, `@pmndrs/pathtracer`): in the workspace `exports` point at `src/*.ts`; `publishConfig.exports` swaps them to `dist/` (built by each package's
  `tsconfig.build.json`, `rewriteRelativeImportExtensions`) only in the packed tarball. Always publish with `pnpm` (`pnpm release`, or `pnpm -r publish`): `npm publish` ignores the
  swap and ships `workspace:^` literally. `pnpm -r publish` skips versions already on npm, so CI (`.github/workflows/publish.yml`, trusted publishing) releases on a version bump.
  CI installs only the two packages (`--filter`): bench's `link:` siblings do not exist there. Check a release with `pnpm pack` + a clean consumer install before bumping.
- Cloud regime (P5.v13-v18, `bench/scenes/volumes/clouds.ts`, docs/internal/CLOUDS-HANDOFF.md): the uniform-cosine single-scatter quadrature `slabSingleScatter` is
  1% off at HG g 0.85 and 28% at 0.99 (its 0.017-wide cosine bins miss the forward peak; even `vol-slab-single`'s expected 0.849100 is 5e-5 low, 0.849144): use
  `slabSingleScatterPhase` (analytic depth integral, angles warped around the forward peak) for any g above ~0.5 or any droplet phase.
- Approximate Mie (Jendersie and d'Eon 2023): the constants 0.428934 (gD) and 0.665888 (wD) are INSIDE the exponent (supplemental Eqs. 8, 10; Cycles agrees). The clouds
  repo's notes and runtime had them outside (reported back in the handoff). The PDFs at research.nvidia.com/labs/rtr/approximate-mie/ fetch fine; read them with
  `pdftotext -layout` (WebFetch returns compressed bytes). The large-particle fit is valid for 5-50 um only (`ptMieFit` throws outside).
- A large ground quad near the world origin self-intersects (V302, FIXED by the per-hit position error bound): a ray leaving a point near the origin of a 2 km triangle starts ~1.5e-5 m above it, below
  that triangle's own test precision (~1e-4 m); the default quantized BVH nodes (boxes rounded out by a grid step) expose it (f32 nodes hide it for an axis-aligned plane): the ground under an opaque box read
  100x pbrt's sky light, with no media in the scene. `rtSurface` now returns `positionError` (pbrt-v4 bound, collapsed to its LARGEST AXIS: a flat plane's per-axis error along its normal is 0 yet the next
  ray's test errs by the triangle's extent, so a per-axis pbrt offset `|n| . err` changed nothing: 0.0129 stayed 0.0129 even at x1000) and `ptOffsetOriginErr` offsets by it before the ulp step. Only when positions are
  decoded from vertices (`positions: 'barycentric'`, the default); `positions: 'ray'`, the G-buffer positions of ReSTIR's init / temporal / spatial passes (`g.p`) and light sample end points keep the ulp offset alone.
  An interface skip (medium boundary) does not use the bound, and the free flight after it starts at the true crossing (`volBack`), so no offset skips medium. Check any new big-plane scene against pbrt / Mitsuba.
- `volumes.outputs` (orders 0-1, primary transmittance, second moment, first-scatter distance) live in the accumulator tail; z = seed difference / standard error has a
  spread of ~0.7, not 1, because of heavy tails (a firefly inflates both): judge the standard error by mean(diff^2) / mean(se_a^2 + se_b^2) (1.005 PCG, 0.93-1.006 Owen-Sobol).
  The order-0-1 plane equals a `maxDepth` 1 render of the same seed per pixel, bit for bit. Grid + brick + outputs is 50.5 KB, + two-lobe phase 51.5 KB.
- `vol-vdb-smoke` needs WebVDB's `packages/vdb-web-tools/dist/` (sibling, not built on this Mac 2026-11-21): `vdb-grid.ts` fails with ERR_MODULE_NOT_FOUND until it is built there.
- Matrix runs under one `--name` with different `--extra` args can return bit-identical numbers when the extra changes nothing the image depends on (here: f32 BVH
  nodes made every order >= 2 contribution exactly zero, so the image equalled the maxDepth 1 one): that is a result, not a cached cell. Check `run.json` args.
- Volumes plan V1-V6 (docs/internal/design/volumes-plan.md): most new media options are opt-in and keep every default kernel byte-identical; check with `node bench/size-snapshot.ts <dir>` and `diff -rq` against a snapshot taken at the round's start. Several agents edit `volume-wgsl.ts` / `volume.ts` / the bench adapter at once: commit their work together when hunks interleave (git add -p is not available). `PtMegakernel.renderUntil` continues the accumulator (reset + setSeed first); its `meanRelSe` is a statistical error (independent pixels), not a bias bound. Telescoping transmittance estimates are SIGNED (alive tests use abs()). Residual and telescoping shadow tracking lost at equal time on our clouds: the shadow transmittance is not their dominant noise.
- The walk-only wavefront intersect module has no `rtHash32` (it lives in the Owen-Sobol sampler text): a kernel that needs `ptPixelSeed` there carries its own copy (`ptwfVoHash`, V5.2). `vol-lamp-fog` has a camera-jitter singularity at the lamp's silhouette (one pixel can hold most of the error): measure noise outside a mask around the lamp. The `vol-cornell-smoke` Mitsuba / pbrt PFMs are not in this checkout (sidecars only).
- `bench/runner/run.ts` queues for the shared GPU lock behind other agents' runs (one held it over an hour on 2026-11-22): small correctness-only pages may pass `--no-lock`; never quote timings from such a run. Brief agents with a per-run GPU time cap (e.g. 5 minutes) so one A/B cannot block the rest.
