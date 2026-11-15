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
- Shader-compile timing (`--page kernel-compile`): a unique comment in the WGSL does NOT defeat the cache. Tint drops it, the generated MSL/HLSL(DXIL) is identical and a backend or driver cache answers (Mac: repeat 1 = 130-475 ms, repeats 2-3 = 9-12 ms). The page puts a unique f32 literal in a live line of `ptMain` (`1 + n/2^23` in the accumulate); if `ptMain`'s accumulate line changes, update `LIVE` in the page (it throws). Check cold/warm and the per-repeat list.
