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
