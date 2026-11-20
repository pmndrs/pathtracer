# NRC on Windows

## Phase 15 agent queue

The NRC renderer improvements are available independently of the WebNN experiment. From the existing Windows work branch, integrate the published main revision using the team's normal merge workflow. The following commands keep that branch checked out. Start the merge from a clean working tree so local changes are preserved.

```powershell
git fetch origin
git merge origin/main
git rev-parse HEAD
pnpm install --frozen-lockfile
```

For a separate benchmark checkout, `git switch codex/nrc-windows-handoff` selects the published handoff branch. WN8 is the first task for the renderer agent. WN9–WN11 can run afterward on the same laptop. [Phases 0–15](nrc-experiments.md#phase-table) · [Task tracker](../../../TASKS.md) · [Mac compact WebNN results](../../../bench/results/NRC-WEBNN-COMPACT.md)

Use Node 22.18 or later in the Node 22 line and pnpm 10.23.0. The bench workspace links the Denoiser and upscaler checkouts at `../Denoiser` and `../upscaler` relative to the pathtracer root. The blue-card fixture builds its geometry procedurally and renders its reference locally. The committed Mac reports and figures are in `bench/results/`. The eight-pair viewer in the Mac review uses captures stored locally on that machine.

| Task | Agent | Deliverable |
|---|---|---|
| WN8 · Portable baseline | Windows WebGPU | Run the compact tiled and per-record fp16 controls. Record the faster verified kernel, frame time, reflection contrast, adapter, driver and browser version. |
| WN9 · WebNN GPU | Windows WebNN | On the RTX laptop, request a GPU WebNN context. Check four successive weight snapshots against WGSL and the CPU oracle. Report p50/p95 for GPU-visible predictions at 4,096 and 8,192 rows per slab. |
| WN10 · WebNN NPU | Windows WebNN | Repeat the GPU protocol with the NPU context if available. Record the exact capability error if context creation, tensor export or a shape fails. |
| WN11 · Backend decision | Windows results | Add the run IDs, query/padding counts, parity errors, timing and thermal status to the compact WebNN report and update N12.12. Compare with the fastest same-browser WGSL control. If replay beats that control, measure the integrated frame before recommending a backend switch. |

Run the timed jobs sequentially on the laptop so agents share the same GPU budget. WN8 and WN9/WN10 use the same geometry, seed, camera, compact MLP, one traced bounce, four position bands and raw RGB. Still refinement is off. Use the same browser and display mode across the comparison.

```powershell
$study = @('phase10=mlp', 'scene=nrc-materials-card', 'res=320x180', 'scenario=static', 'frames=16,64', 'warm=0', 'flip=0', 'refSpp=512', 'upload=0')
node bench/runner/run.ts --page nrc-study --gpu nvidia --headed --repeat 2 --no-csv --ab infer=tiled,portable --args @study

$env:PT_CHROME_FLAGS = '--enable-features=WebMachineLearningNeuralNetwork'
# Change to portable if WN8 selects the per-record kernel.
$infer = 'tiled'
foreach ($device in @('gpu', 'npu')) {
  node bench/runner/run.ts --page nrc-study --gpu nvidia --headed --repeat 2 --no-csv --ab rows=4096,8192 --args @study "infer=$infer" webnnReplay=1 "device=$device" repeats=20 weights=packed publish=frame
}
```

Each run writes `bench/results/incoming/<run-id>/run.json`. The WebNN replay includes feature packing, weight publication, graph dispatch, synchronization and returning predictions to WebGPU. Its paired WGSL timer uses the same records and weights. Record requested WebNN device type separately from the renderer adapter. Use the earlier Dawn/Vulkan matrix results as a separate comparison. [Replay implementation](../../../bench/nrc/webnn-replay.ts) · [Windows matrix measurements](../../../bench/results/NRC.md)

## Historical M6 handoff for WN1–WN7

The instructions below document the November 16 experiment. Its completed Windows results are in [the NRC report](../../../bench/results/NRC.md).

_Written on the Mac at the end of M5. This is milestone **M6** of [nrc-plan.md](nrc-plan.md): the neural radiance cache
(NRC) on NVIDIA. Read [CLAUDE.md](../../../CLAUDE.md) first (rules, licences, Traps), then
[docs/internal/WINDOWS-HANDOFF.md](../WINDOWS-HANDOFF.md) (machine setup and rules; they all apply here). Background for this
track: [nrc-plan.md](nrc-plan.md) (questions, milestones, budget, log), [nrc.md](nrc.md) (spec; §9 is the VkNRC
baseline), [nrc-diverges.md](nrc-diverges.md), TASKS.md section N and [bench/results/NRC-STUDY.md](../../../bench/results/NRC-STUDY.md)._

_Matrix kernels on Windows run outside Chrome, in dawn.node on Vulkan: setup, switches and pitfalls in
[dawn-node-vulkan.md](dawn-node-vulkan.md)._

## Where things stand

Everything below was built and verified on the Mac (Apple M5 Pro, Metal, Chrome 157). Nothing in track N has run on
Windows or NVIDIA yet.

| Piece | Where | Mac result |
|---|---|---|
| MLP engine, 64 → 5×64 → 3 network, Adam, f64 CPU oracle | `packages/mlp` (`nn*` WGSL) | GPU matches the oracle (f32 ~1e-7, f16 ~1e-3) |
| Portable kernels, no flag needed | `wgsl-forward.ts` (tiled inference and training), `wgsl-gradient.ts` | Inference 11.0 ms per million queries (f16); training step 0.51 ms (16k records, f16) |
| Subgroup-matrix kernels (`chromium-experimental-subgroup-matrix`) | `wgsl-sgm.ts` (inference), `wgsl-sgm-train.ts` (forward-backward) | Inference 8.1 ms/M (4 subgroups, f16); training 1.55× tiled in f32, 10-25% slower in f16 |
| The cache on main's path tracer | `bench/nrc/cache.ts`, opt-in `ptOnRadiance` hook (megakernel) | Closed-room gate passes on every backend |
| Demo | `pnpm --filter @pathtracer/bench nrc:live` | 960×540 cache frame: 20.7 ms tiled, 19.3 ms subgroup matrix |
| Equal-time study | `bench/pages/nrc-study.ts` | Warm cache 3.2-4.3× lower relMSE than plain PT at equal frames, 5-7× per unit time |

**The question M6 answers (Q3):** what do subgroup matrices buy on NVIDIA tensor cores, compared with the tiled plain
kernel? On Apple the answer was "almost nothing" because Apple offers only 8×8×8 f32/f16 tiles. A second question
is the ratio of web to native: our WebGPU cache against VkNRC (Vulkan, `VK_NV_cooperative_matrix`) on the same
laptop.

## Rules (in addition to WINDOWS-HANDOFF.md)

- **Branch:**
  - Create `windows-nrc` from `origin/claude/eager-brown-8uj9yy`.
  - Commit and push only that branch. Never push `main` or the Mac's branch; the Mac side merges.
- **VkNRC has no licence** (D28, D29):
  - Build and run it only under `.cache/native/vknrc` (git-ignored).
  - Follow its design, but never paste or translate its code into this repo.
  - Your local timing patches stay in `.cache` and are never committed. Describe them in words in TASKS, not as a diff.
- **NVIDIA RTX SDK repos** (NRC SDK, RTXGI, SHaRC): no code.
- **Nothing outward:** do not post to gpuweb, Chromium or Dawn bug trackers (N9). Write findings into
  `docs/internal/design/subgroup-matrix-feedback.md`; the maintainer decides.
- **Timing:**
  - Use a quiet machine: plugged in, card cool, other GPU apps closed.
  - Record `nvidia-smi` clocks and temperature.
  - Compare only paired ratios on this laptop, never against the Mac's numbers.
  - The maintainer prefers windowed Chrome for interactive timing (TASKS N6e). The runner's headless Chrome read NRC
    frames 2-4× slower on the Mac. Microbenchmarks (`mlp-timing`) can use the runner.
- **New kernel shapes:** before running a kernel shape that has never run on this GPU, run it on SwiftShader first,
  then one small dispatch on the GPU (Trap: an early emulated kernel hung the Mac).
- **Docs:**
  - Mark results in TASKS (N0.3, N2, N3, N6.1, N6.2, and a new N-row for anything else).
  - Add Windows traps to CLAUDE.md prefixed "Windows:".
  - Add each departure from the paper or VkNRC to `nrc-diverges.md`.
  - Add a dated "Windows" entry to the log in `nrc-plan.md`.

## Setup

1. **Machine:** follow WINDOWS-HANDOFF.md "Setup" steps 1-4. You need the bvh sibling built, `pnpm install`, and
   `pnpm test` / `pnpm typecheck` green.
2. **Branch:**
   ```
   git fetch origin
   git switch -c windows-nrc origin/claude/eager-brown-8uj9yy
   pnpm install
   ```
3. **Scene assets:**
   ```
   node bench/assets/fetch.ts
   node bench/adapters/export-scenes.ts sponza-white
   ```
   The demo loads sponza-white from the web glb.
4. **References:** they are git-ignored. Copy `bench/references/sponza-white/` from the Mac:
   - `mitsuba3-llvm_ad_rgb-4096spp.{pfm,json}`, the study's reference;
   - `pbrt-v4-cpu-4096spp.{pfm,json}`;
   - `ours-unseen-gallery-640x360-4096spp.pfm`, the reference for the unseen corridor.

   Without the corridor file, `nrc-study` renders its own reference: leave out `refC=1`, and it takes minutes.
5. **Chrome:**
   - Plain Chrome stable runs the tiled kernels.
   - For subgroup matrices, start Chrome with `--enable-unsafe-webgpu`. On Windows it may also need
     `--enable-features=...` or Canary; find out and record the exact switches (task WN1).
   - For driving pages over CDP, add `--remote-debugging-port=9333 --user-data-dir=<a fresh dir>`. Open
     `http://localhost:<port>/bench/pages/<page>.html?<args>` and read `window.__benchReport` with `Runtime.evaluate`
     once it is set.
   - The URL comes from `node bench/live.ts <page> --no-open`, which also serves the PFM upload endpoint.

## Tasks, in order

| # | Task | How | Done when |
|---|---|---|---|
| WN1 | **Which backend reaches the tensor cores** (TASKS N0.3). | `node bench/runner/run.ts --page caps --gpu nvidia` and `--page mlp-sgm-probe --gpu nvidia`, once on Dawn's default D3D12 backend and once on its Vulkan backend. Find the Chrome switch that selects Vulkan on Windows and record it as a Trap. | For each backend: whether `chromium-experimental-subgroup-matrix` is offered, the `subgroupMatrixConfigs` list (component and result types, M×N×K, subgroup size), and whether the probe kernel compiles. Written into TASKS N0.3 and `subgroup-matrix-feedback.md`. |
| WN2 | **Correctness on NVIDIA** | Run `mlp-check` for each arm: `--args infer=tiled precision=f16`, then `f32`, `infer=portable`, `infer=sgm subgroups=4`, `infer=sgm-emu tile=16`, plus `forward=sgm fwdSubgroups=4` for training. Then `mlp-fit --args oracleSteps=12 batch=16384` (tiled and sgm forward). `wgsl-sgm.ts` picks the config from the adapter. NVIDIA will probably offer 16×16 tiles, which on the Mac were verified only in the emulated twin: run the 16³ `sgm-emu` first. | Same tolerances as the Mac (f32 about 1e-7; f16 about 1e-3, p99 replay under 0.05). A mismatch is a bug to fix on this branch, with a test. |
| WN3 | **N2 timing table on the 3060** (Q3) | `node bench/runner/run.ts --page mlp-timing --gpu nvidia --ab infer=tiled,sgm --args precision=f16 subgroups=4 --repeat 3`, the same in f32, `--ab infer=portable,tiled`, then training: `--ab forward=tiled,sgm --args fwdSubgroups=4 precision=f16` (and f32). Quiet machine. | A table like TASKS N2 (ms per million queries, ms per 16k training step, TFLOP/s, min..max over ABBA rounds) for the 3060, with a one-line answer: how much faster than tiled is the matrix path on tensor cores? |
| WN4 | **Room gate on NVIDIA** | `node bench/runner/run.ts --page nrc-check --gpu nvidia --args infer=tiled precision=f16`, then `infer=sgm subgroups=4`. | Passes, as on the Mac (mean error under 1%). The page throws on a missed query. |
| WN5 | **Demo and equal-time study, windowed** (M4/M5 on NVIDIA) | Run `pnpm --filter @pathtracer/bench nrc:live` and open the URL in plain and in flagged Chrome. Check that it renders, that the split view looks right and that the camera moves. Press **Benchmark** at 960×540 and 1280×720 for each network backend. Then run `nrc-study.html?scenario=all&res=1280x720&steps=4&refC=1` in the windowed flagged Chrome over CDP (and `res=640x360 steps=2`). | Per-stage frame times (render, inference, training) per backend, and a 3060 column for the NRC-STUDY.md tables (relMSE, FLIP, bias and flicker vs plain PT at equal frames and equal time). Also note the first-load compile time of the cache's megakernel on DXC (Traps: about 50 KB WGSL). |
| WN6 | **OBJ export for VkNRC** (TASKS N6.2) | VkNRC loads one OBJ (positions + normals). Write `bench/exporters/obj.ts` (our own code) for sponza-white, and a "VkNRC lighting" option on `nrc-live` / `nrc-study` (constant white sky of radiance 10, no sun) so the two renderers see the same light. | The OBJ loads in VkNRC; our page renders the same lighting. |
| WN7 | **VkNRC native baseline** (TASKS N6.1, WINDOWS-HANDOFF W11) | Clone AdamYuan/VkNRC at `a667859` into `.cache/native/vknrc`. Build x64 Release as its `.github/workflows/windows-msvc.yml` does (Vulkan SDK, VS 2022). Add local patches (nrc.md §9): GPU timestamps per pass written to CSV; fixed camera and frame count from the command line, with exit; a PFM dump of the image. | Per-pass ms (path tracing, inference, each training step, optimiser) at the same resolution, network and batch as ours, and a PFM after N frames. The **web/native ratio per pass** goes into TASKS N6.1 and the plan log. It is the headline number for the write-up (M8). |

If time is short, do WN1-WN4 first: they answer Q3. WN5 shows the demo on NVIDIA. WN6-WN7 give the native comparison.

**Status 2026-11-15 (cloud session, no GPU):** WN6 is done (TASKS N6.2). `node bench/exporters/export-obj.ts` writes
`bench/assets/generated/vknrc/sponza-white-vknrc/{scene.obj, scene.mtl, vknrc.json}`; the JSON has the camera pose in VkNRC's
frame for the fixed-camera patch. Our side of the comparison is the scene `sponza-white-vknrc` (export its glb with
`node bench/adapters/export-scenes.ts sponza-white-vknrc`; `nrc-live` scene menu, `nrc-study --args lighting=vknrc`).
WN1-WN5 and WN7 still need the laptop.

## Things that will probably bite

- **Tile size:** the sgm kernels assume 8×8 tiles in places only the emulated twin has tested at 16. If the native
  16×16 form fails, check the padding of the 3-wide output layer and the weight view first. Use `tile=16` on
  `sgm-emu` to tell a kernel bug from a driver bug.
- **f16 on NVIDIA:** it may offer f16 inputs with f32 accumulation. Apple has only f16→f16, so the f16 training
  numbers were noisier there. If an f16→f32 config is listed, try it (it should be both faster and more accurate).
- **Uniformity:** subgroup-matrix offsets come from `@builtin(subgroup_id)`, which needs Chrome 145+. If Windows
  Chrome rejects that, the fallback is `diagnostic(off, chromium.subgroup_matrix_uniformity)` (Traps).
- **Kernel size:** the cache's megakernel adds the hook and query code to the path tracer. If the DXC compile takes
  seconds, record it (W10 / D26 ask the same question for the plain kernel).
- **Vite:** `bench/live.ts` has no file watching. Restart it after editing a page.
- **Shared sessions:** the Mac's NRC agent may still push to `claude/eager-brown-8uj9yy`. Rebase `windows-nrc` onto
  it if you need its changes; never force-push someone else's branch.
