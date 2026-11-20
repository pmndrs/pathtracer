# Windows handoff (branch `windows-testing`)

_Written 2026-11-13 on the Mac, at the end of P0. Read [CLAUDE.md](../../CLAUDE.md), [STATUS.md](../../STATUS.md),
[TASKS.md](../../TASKS.md) and [DIVERGES.md](../../DIVERGES.md) first. This page is the Windows-specific part._

For the November 19 compact NRC and WebNN experiments, use the [phase 15 agent queue](design/nrc-windows-handoff.md#phase-15-agent-queue) for the published NRC baseline and instructions for an existing work branch. The remaining sections cover the general renderer work on `windows-testing`.

## What this machine is for

The Mac (Apple M5 Pro, Metal) is where the code is written. The Windows laptop (RTX 3060 Laptop, plus an Intel UHD
for correctness only) is the second target, with Chrome running on D3D12 and compiling shaders with DXC.

Nothing on the Mac has been checked on Windows yet. P0's job here is to show that the bench runs on Windows and to
produce the first numbers there you can trust.

**Update 2026-11-19:** `origin/windows-testing` was fast-forwarded to main (P3 and P4 closed provisionally on the M5, P5
6 of 7 built, bvh and downsampler from npm, GPU mips the default). A local checkout: `git pull --ff-only` (or, with local
commits, `git merge origin/main`), then `pnpm install` (new npm dependencies) and re-run `adapters:export` (scene exports
changed). The open 3060 work, in priority order: **W13** (P3 gate), **P4.5b / W14** (30 fps), **W15** (wide BVH default),
**W10** (DXC compile), **W16** (GPU mips), **W17** (bindless textures: first real-driver verdict). Each decides a default or a gate that is provisional on the M5 until then.

## Rules for this branch

- **Where work goes:** work on `windows-testing`. Commit and push this branch only. Never push `main`; the Mac side
  merges.
- **Docs:** keep the same docs current as on the Mac.
  - Mark results in TASKS.md.
  - Add Windows traps to the CLAUDE.md Traps list, prefixed "Windows:".
  - Record every workaround in DIVERGES.md, with kind `platform` or `bug-workaround`.
- **Timings need a quiet machine.**
  - Before any timed run, close other GPU apps and browsers and plug the laptop in.
  - Record what else was running, plus `nvidia-smi` clocks and temperature (the runner's clock note does this).
  - Timings from a loaded or throttling machine are reported only as "provisional" (CLAUDE.md working rules).
- **Comparisons:** compare timings only as paired ratios within this machine, never against Mac numbers.
- **NRD:** never open NRD code or any port of it. Native NRD output comes only from the Denoiser repo's Windows
  harness, as images.

## Setup

1. Install the tools:
   - Node 22.x, with type stripping (`node -v` ≥ 22.18).
   - pnpm 10.23 via `corepack enable`.
   - Git.
   - Chrome stable, and Chrome Canary for step 6.
2. Clone the repo and install:
   ```
   git clone https://github.com/pmndrs/pathtracer.git
   cd pathtracer
   git switch windows-testing
   ```
   `@pmndrs/bvh` (0.1.0) and `@pmndrs/downsampler` come from npm since D33, so no sibling checkout is needed:
   ```
   pnpm install
   ```
   (To work against a bvh checkout, see CLAUDE.md "Developing against local siblings".) The wide node formats are in bvh 0.1.0 and wired into
   our kernels (X1.i): W-row in the quiet queue: `trace-timing` on Sponza with `nodeFormat=quantized|f32|bvh4|cwbvh8` and `wideDecode=exact|folded`.
3. Run the non-GPU checks: `pnpm test`, `pnpm typecheck` and `pnpm lint`. All must pass. Any failure is a Windows
   portability bug to fix here, and most likely a path separator or a line ending.
4. Generate the assets:
   - `pnpm --filter @pathtracer/bench assets:generate` makes the procedural sky.
   - `pnpm --filter @pathtracer/bench assets:fetch` downloads Sponza, pinned by sha256.
   - `pnpm --filter @pathtracer/bench adapters:export` writes the glb files for the web competitors.
5. Get the reference images. They are git-ignored (about 43 MB) and every error metric needs them. Use one of:
   - **Copy them from the Mac:** every `bench/references/**/*.pfm` plus the `.json` next to it. This is the fastest
     way and the one to prefer.
   - **Re-render Mitsuba only:**
     ```
     python -m venv .cache/native/mitsuba-venv
     .cache/native/mitsuba-venv/Scripts/pip install mitsuba numpy
     pnpm --filter @pathtracer/bench references -- --renderers mitsuba
     ```
     Building pbrt-v4 on Windows is not needed. pbrt adjudicates on the Mac (D15).

## Tasks, in order

| # | Task | How | Done when |
|---|---|---|---|
| W1 | Runner works on Windows (DIVERGES V5) | `node bench/runner/run.ts --page smoke --gpu nvidia`, then `--gpu intel` | Both pass; the CSV `device` column names the right GPU. If the wrong GPU is picked, check the LUID pinning in `bench/runner/chrome.ts`. |
| W2 | Timer calibration | `node bench/runner/run.ts --page timer-calibration --gpu nvidia` | Records whether timestamps are quantized on D3D12 with and without `PT_NO_DEV_FEATURES=1` (CLAUDE.md trap). |
| W3 | Capability detection | `node bench/runner/run.ts --page caps --gpu nvidia` and `--gpu intel` | Tier, caps hash, subgroup sizes (NVIDIA should be 32; Intel varies) and whether `chromium-experimental-sampling-resource-table` is present. Add each machine to STATUS.md. |
| W4 | GPU lock on Windows | The lock defaults to `/tmp/rt-gpu.lock`, which becomes `C:\tmp\...` on Windows. Set `RT_GPU_LOCK` / `RT_GPU_QUEUE` if the Denoiser harness uses other paths. | The runner and the Denoiser harness wait for each other. |
| W5 | Competitor correctness on Tier 0 | `node bench/adapters/matrix.ts --spp 256 --snapshots 16,64 --name win-tier0-256` | The same pass/fail picture as on the Mac (diffuse, glass and quad light pass; the conductor furnace sits in its own report, D20). Look at the report in the viewer. |
| W6 | **P0.25 quiet baseline on the RTX 3060** | `node bench/adapters/matrix.ts --scenes cornell --spp 1024 --snapshots 4,16,64,256 --repeat 3 --name p0-baseline-3060`, then the same with `--scenes sponza-white --append`, then `node bench/adapters/baseline.ts --report p0-baseline-3060 --csv` | A trusted baseline table (`bench/results/BASELINE-P0-3060.md`, same format as BASELINE-P0.md), taken on a quiet machine with clocks recorded. The Mac table is provisional (taken under load); this one can count. |
| W7 | P0.18 probes | Chrome Canary: on the caps page, try `chromium-experimental-sampling-resource-table` and the subgroup-matrix feature or language feature | The real feature names and whether each one compiles, recorded in TASKS P0.18 and research note 06. |
| W8 | DXC compile cost | Note `compile_ms` from the competitor runs (first call) | A first data point for the CLAUDE.md trap about keeping kernels under 50 KB of WGSL. |
| W9 | **Capsaicin as a native GPU cross-check and timing baseline** (TASKS P0.31). The laptop already has Capsaicin built. | Follow [research/notes/11-capsaicin.md §9](../../research/notes/11-capsaicin.md): export our glbs, render the opaque scenes (no glass) with its reference path tracer in `--benchmark-mode`, RR off, bounces matched and checked on `quad-light` first; convert the f16 EXR to PFM (watch the BGR channel order and row order); compare; then time it and GI-1.2 on the 3060, quiet-machine rules apply | `bench/references/<scene>/capsaicin-<spp>spp.pfm` + JSON for cornell, the furnaces, quad-light and sponza-white; an agreement table vs pbrt and Mitsuba; a timing CSV. Record the Capsaicin commit, and every patch (environment map, f32 accumulation dump) in DIVERGES. |
| W10 | **Megakernel compile time on DXC** (DECISIONS D26, TASKS P2.33). Quiet machine, every other app closed. | `node bench/runner/run.ts --page kernel-compile --gpu nvidia --args repeat=5`, then once more with `--args variants=plain,pbr-albedo-fuzz-thinFilm repeat=3 nonce=comment` (comment-only nonce: shows whether the driver caches DXIL). The page compiles each of the 18 kernel variants (`bench/kernel-variants.ts`, same table as `size-snapshot.ts`, built with the 3060's real caps) with `createShaderModule` + `createComputePipelineAsync`, one at a time, each repeat on text made unique by a live f32 literal in `ptMain` (a comment alone is stripped by the compiler and a backend cache then answers; on the Mac it made repeat 2+ read 20-40x too fast). `warm` = the identical text again, so cold/warm is the cache check. Record from `run.json` `notes.table`: per variant `bytes`, cold min / median / max ms, warm ms, and `cold_ms_per_repeat` (the first repeat must not be much slower than the rest; if it is, a DXIL/driver cache is still in the way, then use a fresh browser profile per repeat). **The 50 KB question:** plot cold median against `bytes` over the variants. Is it linear (Mac: 190 ms at 40 KB, 285-475 ms at 49-55 KB, driven by features, not size: `readable` at 58 KB unminified costs the same as `plain` at 40 KB) or does it jump past about 50 KB on DXC? Also note which feature adds the most (fuzz, thin film, coated metal, layered). Add the `--gpu intel` run if time allows. | A table of cold/warm ms per variant on the 3060 (device, driver, Chrome version, `loaded` flag or not) pasted into TASKS P2.33 with a one-line answer to the 50 KB question, and D26 updated: a cold compile under a few seconds for the heaviest variant keeps the single megakernel; tens of seconds means split rare lobes into specialised kernels or pull the P3 wavefront split forward. |
| W11 | **VkNRC native NRC baseline** (TASKS N6.1, D28) | Clone AdamYuan/VkNRC at `a667859` into `.cache/native/vknrc`, build x64 Release as in its `windows-msvc.yml` (Vulkan SDK + VS 2022), apply the three local patches in [docs/internal/design/nrc.md §9](design/nrc.md) (per-pass timestamps to CSV, fixed camera + frame count, PFM dump), run on the sponza-white OBJ (N6.2) | Per-pass ms on the 3060 and a PFM after a fixed frame count. Patches stay local (never commit VkNRC code, D29). Quiet-machine rules apply. |
| W12 | **Track N (neural radiance cache) on the 3060** (milestone M6) | Its own handoff with its own branch (`windows-nrc`): [docs/internal/design/nrc-windows-handoff.md](design/nrc-windows-handoff.md), tasks WN1-WN7 (W11 is WN7 there) | As listed there |
| W13 | **P3 gate on the RTX 3060** (PHASES P3 is closed provisionally on the M5, D32; this is the 3060 half) | (a) time to equal quality against the web competitors: `node bench/adapters/matrix.ts --renderers tgpt-wavefront,tgpt-megakernel,mutracer,pt-auto,pt-megakernel --scenes cornell --spp 1024 --snapshots 4,16,64,256 --repeat 3 --interleave --name p3-gate-3060`, then `--scenes sponza-white --spp 512 --append`, then `node bench/adapters/baseline.ts --report p3-gate-3060 --csv` (the M5 table is `bench/results/P3-GATE.md`); (b) the Mrays/s gate: `node bench/runner/run.ts --page trace-timing --gpu nvidia --args scene=sponza-white` (incoherent bounce rays, batch method; gate >= 150 Mrays/s; last 3060 read: 102 bounce / 229 primary, P-W1, before any wide BVH) and the same with the wide-BVH branch once it is merged; (c) `ptwfAutoPolicy`'s choice on the 3060 (`pt-auto` vs the best fixed scheduler, `--page wf-sched-ab`) | The gate numbers on the 3060 in the same format; if `pt-auto` is not the fastest of ours, the policy constants get a per-device note. |

| W14 | **P4 gate on the RTX 3060: 30 fps** (TASKS P4.5b) | `node bench/adapters/export-scenes.ts sponza-lamps`, then `node bench/runner/run.ts --page realtime-pipeline --gpu nvidia --timeout 3600000 --args scenes=sponza-white,sponza-lamps restir=off,on` (a worktree needs the `.claude/worktrees/upscaler` junction, CLAUDE.md Windows traps) | Frame ms and the stage split (kernel, SVGF, upscaler) at 960x540 -> 1080p, maxDepth 2, with and without ReSTIR, in TASKS P4.5b. The M5 estimate for the 3060 is 23-32 fps without ReSTIR, 19-29 with. |
| W15 | **Wide BVH default** (TASKS X1.i, D33) | `node bench/runner/run.ts --page trace-timing --gpu nvidia --ab nodeFormat=quantized,cwbvh8 --repeat 3 --args scene=sponza-white wideDecode=folded`, then the same with `--ab nodeFormat=quantized,bvh4` (the page's own `pageAb` does not take `nodeFormat`); then `node bench/adapters/matrix.ts --renderers pt-megakernel --scenes cornell,sponza-white --spp 256 --repeat 3 --interleave --extra pt_bvh=cwbvh8 --name wide-3060` against the same without `--extra` | Mrays/s per ray class and ms/spp per format on the 3060 (the M5: cwbvh8 -26% Cornell, -16% Sponza megakernel time). Also check the wide walks' loops are not hit by the D3D12 loop-counter cost (P-W1): wide megakernel vs wavefront on Sponza should stay in the M5's ratio. Picks the library default. |
| W16 | **GPU mips on D3D12** (TASKS X5, D34) | `node bench/runner/run.ts --page downsampler-check --gpu nvidia` | Level 0 identical, levels 1+ within 2 codes of the CPU chain on both the render and buffer paths for sRGB; GPU vs CPU load time. D34 made GPU mips the default on M5 evidence only. |
| W17 | **Bindless textures on real drivers** (TASKS P1.4c2a, DIVERGES V296-V297) | Chrome stable (the runner passes `--enable-unsafe-webgpu`; check `node bench/runner/run.ts --page caps --gpu nvidia` lists `chromium-experimental-sampling-resource-table`): `node bench/runner/run.ts --page bindless-check --gpu nvidia --args check=lookup,render scenes=tex-checker,tex-mask spp=16`, then the same with `--dawn vulkan --dawn-toggles allow_unsafe_apis,dump_shaders --verbose` and `--dawn d3d12 --dawn-toggles allow_unsafe_apis,dump_shaders --verbose` (grep the dump for `NonUniformResourceIndex` / `NonUniform`), then `--args check=lookup timing=1 rounds=8` on a quiet, cool card | `issues` 0 and `findings` 0 per backend (divergent and holed lookups equal to arrays; tex-checker / tex-mask images per-pixel identical); if findings appear, which backend and whether the shader dump lacks the non-uniform marker. Timing: `*_table_over_arrays` for coh / div on `fit` and `many` (provisional unless quiet). On the Mac (Canary 160) Metal fails compute passes with a table (render passes work and are correct, `check=fragment`) and SwiftShader takes lane 0's index, so this is the first compute verdict on real drivers. |

Later, once the Mac side asks for it: the wide-BVH experiment (X1) in `~/Developer/bvh` needs its RTX 3060 runs.
Its branch and exact commands will be in that repo's HANDOFF.md. The 1.3× decision gate needs this machine too.

## Things that will probably bite

- **Paths:** the runner and exporters were written on macOS. Look for `/` vs `\` in paths, `process.platform`
  branches, and CRLF line endings in generated files. The bvh repo had both kinds of bug.
- **Picking the GPU:** Chrome may pick the Intel GPU by default. `--gpu nvidia` pins the adapter by LUID, and a run
  fails loudly if Chrome hands out another one.
- **Long synchronous loops:** three-gpu-pathtracer's timed loop on Sponza blocks the page for minutes. The runner now
  treats a DevTools evaluate timeout as a busy page (V17). On Windows, watch for TDR (GPU timeout) resets too.
- **Thermals:** a laptop 3060 drops to about 500 MHz when it overheats. The runner's `nvidia-smi` clock note marks
  such runs invalid. Start from a cool card.
