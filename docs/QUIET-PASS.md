# Quiet pass

_All timing that needs a quiet machine, run in one batch. The main sessions keep building and take best-effort
numbers only; anything that decides a default or passes a gate waits for this pass._

## How to run it (maintainer)

1. Close every other app (browsers included), plug the machine in, let it sit a minute.
2. Start Claude Code from a terminal in this repo: `claude`.
3. Tell it: **"Run the quiet pass in docs/QUIET-PASS.md."** Then leave the machine alone until it says it is done
   (about 30–45 min on the M5 Pro; most of it is the Sponza baseline).

## What the session does (agent)

1. Read CLAUDE.md, this file, and the "Queue" table below.
2. Pre-flight: `sysctl vm.loadavg` (1-minute value should be under ~4) and
   `ps -axo command | grep -E '(Google Chrome|Chromium|Microsoft Edge).*--type=gpu-process'` (should print nothing).
   If not quiet, tell the maintainer what is running and wait for a go.
3. Do not run anything else meanwhile: no builds, no tests, no subagents, no other GPU work.
4. Run `bash bench/quiet-pass.sh` in the background and wait for it (it logs to
   `bench/results/quiet-pass/<stamp>/quiet-pass.log`). `--only trace` or `--only baseline` runs one part.
5. Then the sibling items in the Queue table, if any are listed, each with the exact command given there.
6. Report, without changing any defaults or DECISIONS (that is the maintainer's call):
   - for each trace-timing A/B: paired B/A ratio of `*_batch_mrays_per_s` per round (min..max over the 5 rounds) per
     ray set (the headline: long back-to-back batches), plus `*_batch_spread` (max/min of the samples inside a page run;
     ~1.00 = clean, above ~1.05 = something interfered) and, as the second column, the old p10-of-30 ratio and
     `*_p90_over_p10`;
   - the baseline table from `bench/results/quiet-pass/<stamp>/baseline.md`;
   - load at start and end, and any row whose note says `loaded` or `best effort`.
   Write the report to `bench/results/quiet-pass/<stamp>/REPORT.md`, and add one line per item to TASKS.md pointing at it.
7. Commit only the report and the TASKS line if the maintainer asks; never push.

## Queue

What the next pass should measure. Add a row when something needs a quiet-machine number; remove it once reported.

| Item | What | Command | Decides |
|---|---|---|---|
| P1.5 | Reflections' trace pass on our core vs three-mesh-bvh (the @pmndrs/bvh wrapper it replaced: best-effort numbers in reflections `docs/research/w81/`) | in the reflections worktree pinned before the deletion (`~/Developer/reflections/.claude/worktrees/p1.5-ab-baseline`, commit 36e206a: both providers, four variants): `REFLECTIONS_PORT=5453 pnpm check -- --only ssr-rt-bvh --rt-bvh-timing-only --rt-bvh-rounds 5` (writes `out/rt-bvh-timing.json`; compare `@pmndrs/bvh` with `@pmndrs/raytracer`) | P1 gate "no rays/s regression" |
| P2.3 | VNDF standard vs spherical cap (and maybe a branchless basis): cost per sample, noise at equal spp | not runnable yet: needs the integrator (P2.4); add the command here when it exists | `vndf` default, docs/design/bsdf.md choice 1 |
| P3.1 (rerun on a quiet machine: 20261115-1714 was under load 3.4-6.4) | Wavefront (slice 1) vs megakernel: ms per sample at equal spp, both schedulers interleaved ABBA, 5 repeats, plus the per-stage split (camera / intersect / shade / shadow / args timestamps if the runner can wrap them) on Cornell 512x512 (maxDepth 32), sponza-white 1280x720 (maxDepth 8, RR 4), env-sphere, quad-light. Report paired wf/mk ratios min..max, never absolute across runs | `node bench/adapters/matrix.ts --renderers pt-megakernel,pt-wavefront --scenes cornell --spp 256 --snapshots 64 --repeat 5 --interleave --name p31-cornell` then the same with `--scenes sponza-white --spp 64 --append --name p31-cornell` (appended cells share the report); add `--scenes env-sphere,quad-light --spp 1024 --name p31-small` (a separate report). Skip furnace-dielectric (maxDepth 1000: the wavefront runs 4000 dispatches per sample, a launch-overhead test, not a speed test) | the P3 scheduler default (D11): megakernel or wavefront, and whether P3.2 (refill, compaction, binning) is worth building on this wavefront |
| P3.2 (rerun on a quiet machine: 20261115-1714 was under load 3.4-6.4) | Wavefront refill (`inline`, `batch`, `layers`) vs slice 1 vs megakernel, ms per sample, all interleaved, 5 repeats: Cornell 512x512 maxDepth 32, sponza-white 1280x720, env-sphere 384x256, quad-light 256x256, furnace-dielectric (maxDepth 1000, `--spp 256`), plus a small-film size sweep for `layers` (env-sphere at L = 1, 2, 4, 8, 16, 32, 64) and `refillThreshold` 0.1 / 0.25 / 0.5 on Sponza | `R=pt-megakernel,pt-wavefront,pt-wf-refill,pt-wf-batch,pt-wf-refill-l2,pt-wf-refill-l4; node bench/adapters/matrix.ts --renderers $R --scenes cornell,furnace-dielectric --spp 256 --repeat 5 --interleave --name p32-big` and `--scenes sponza-white --spp 128`, `--scenes env-sphere,quad-light --spp 1024`; `--renderers pt-megakernel,pt-wf-refill --scenes env-sphere --extra pt_layers=N` for the sweep; the 3-round load 2.7-7 numbers are in TASKS P3.2 | the P3.9 auto policy and whether any refill mode becomes the library default (D11 row) |
| R2 / P3.7 (rerun on a quiet machine: 20261115-1714 was under load 3.4-6.4) | Alpha policy and wavefront alpha MASK timing: `pt-megakernel` vs `pt-mk-shadow` / `pt-mk-indirect` / `pt-mk-secondary` and `pt-wavefront` vs `pt-wf-*`, ms per sample at 1024 spp, interleaved, 5 repeats, on `tex-mask` and `foliage-mask`; wavefront-with-MASK vs megakernel ratio (provisional: wf 1.23x slower on `tex-mask`, 0.87x on `foliage-mask`) | `node bench/adapters/matrix.ts --renderers pt-megakernel,pt-mk-shadow,pt-mk-indirect,pt-mk-secondary,pt-wavefront,pt-wf-shadow,pt-wf-indirect,pt-wf-secondary --scenes tex-mask,foliage-mask --spp 1024 --snapshots 1024 --repeat 5 --interleave --name p37-ab` then `node bench/adapters/alpha-policy-ab.ts --report p37-ab --ref foliage-mask=<our 16384 spp pt-megakernel PFM>,tex-mask=bench/references/tex-mask/mitsuba3-llvm_ad_rgb-4096spp.pfm` | whether any alpha policy is worth keeping (provisionally: no, 3-17% too dark for at best 8-23%), and the scheduler default for cutout scenes |
| P3.6 (rerun on a quiet machine: 20261115-1714 was under load 3.4-6.4) | V-buffer primaries (`primary: 'vbuffer'`) vs traced primaries: ms per sample at equal spp, the bench rasteriser's pass INCLUDED in the vbuffer rows (a real engine's G-buffer pass is its own and usually already paid; the bench one is non-instanced, vertex-pulled, one draw per frame), both schedulers, interleaved, 5 repeats: cornell 512x512, sponza-white 1280x720, tex-mask, MetalRoughSpheres, and each again at `pt_maxdepth=1` and `2` (the real-time regime: the primary ray is a larger share of the traced rays there) | `R=pt-megakernel,pt-mk-vbuffer,pt-wavefront,pt-wf-vbuffer; node bench/adapters/matrix.ts --renderers $R --scenes cornell,tex-mask --spp 256 --repeat 5 --interleave --name p36-time`, `--scenes sponza-white --spp 64 --append`, then the same with `--extra pt_maxdepth=2`; correctness first: `node bench/runner/run.ts --page vbuffer-check --timeout 3000000` | whether `primary: 'vbuffer'` is worth wiring into the P4 real-time mode, and what raster cost an engine must hide for it to win (provisional numbers: TASKS P3.6) |
| P4 ReSTIR DI | ms per frame, interleaved, 5 repeats: NEE vs RIS (M = 8) vs RIS + temporal vs + spatial (k = 2) biased vs unbiased, megakernel, at 1280x720 on `sponza-white` (sun + sky only: one light pick) and `many-lights-256` (film overridden to 1280x720); the per-frame noise side is already measured (relMSE, not timing), this decides whether the equal-time ratio holds | `node bench/runner/run.ts --page restir-check --timeout 3000000 --args check=timing scenes=sponza-white,many-lights-256 w=1280 h=720 reps=5 timingframes=16` (best-effort numbers from 20261115 are in TASKS P4.2, loaded) | the real-time default for `restir` settings (candidates, k, bias, initialVisibility), D-row after the A/B |
| P3.11 | Scene copy packing (`sceneCopy.packing: 'packed'`, the table + data as ONE storage buffer) vs split: ms per sample at equal spp, both schedulers, interleaved, 5 repeats: cornell 512x512, sponza-white 1280x720, tex-mask, MetalRoughSpheres (the textured / MASK kernels are the ones that gain a binding), and each again at `pt_maxdepth=2` (the primary-hit / surface-decode share is larger there) | `R=pt-megakernel,pt-mk-packed,pt-wavefront,pt-wf-packed; node bench/adapters/matrix.ts --renderers $R --scenes cornell,tex-mask,MetalRoughSpheres --spp 256 --repeat 5 --interleave --name p311-time`, `--scenes sponza-white --spp 64 --append`, then `--extra pt_maxdepth=2`; `node bench/adapters/paired-ratios.ts --report p311-time --base-map pt-mk-packed=pt-megakernel,pt-wf-packed=pt-wavefront --md`; correctness first: `node bench/runner/run.ts --page scene-copy` (packing=both), `--page trace-core --args packing=packed` | whether `DEFAULT_SCENE_PACKING` flips to `packed` (suggestion: yes if every paired ratio stays within ~2%, since a freed binding is worth more than 2% only where hooks / counters need it; if Cornell stays +2% megakernel, hoist `dataBase` once per `rtSurface` and re-measure); provisional numbers (loaded 3.6-6.2): TASKS P3.11 |
| P2.32b | Layered albedo correction (`albedoCorrection: none \| lut \| texture`) vs the stochastic walk, ms per sample, megakernel and wavefront, interleaved, 5 repeats (provisional: megakernel `lut` 1.03-1.15x, `texture` 1.01-1.03x, stochastic 0.83-1.03x of plain `internal`; wavefront within noise) | `node bench/adapters/matrix.ts --renderers pt-mk-lay-stoch,pt-mk-lay-int,pt-mk-lay-lut,pt-mk-lay-tex,pt-wf-lay-int,pt-wf-lay-lut,pt-wf-lay-tex --scenes AttenuationTest,MetalRoughSpheresNoTextures,SheenTestGrid-notex --spp 512 --repeat 5 --interleave --name p232b-time` | the D24 layering default: albedo+internal+texture correction, or the stochastic walk if it is no slower |
| R1 | Ray-class material LOD (`materialLod: 'secondary'`) vs full: ms per sample, interleaved, 5 repeats, megakernel and wavefront, with the albedo kernel AND the stochastic walk compiled, on `lod-cornell`, ClearCoatTest-notex, SheenTestGrid-notex, IORTestGrid-notex-A, MetalRoughSpheres; plus DXC / compile time of `shadeSecondary` (kernel-compile page with a `materialLod` variant is not built yet: time `PtWavefront.create` via `compile_ms`) | `node bench/adapters/matrix.ts --renderers pt-megakernel,pt-mk-lod --scenes lod-cornell,ClearCoatTest-notex,SheenTestGrid-notex,IORTestGrid-notex-A,MetalRoughSpheres --spp 512 --repeat 5 --interleave --name r1-time-mk --extra pt_layering=albedo,pt_tint=0,pt_coverage=1` then `--renderers pt-wavefront,pt-wf-lod` (`--name r1-time-wf`) and the same with `pt_layering=stochastic` | whether the reduced secondary material is worth enabling in the real-time preset, and at which depth (`pt_lod_depth`) |
| P4.1 | Real-time frame costs: plain 1 spp kernel vs `frame` outputs vs `bluenoise` vs both vs `mode: 'realtime'`, ms per dispatch, 5 ABBA rounds (provisional, load 5.5: Cornell 512 x 512 frame 1.05-1.06x, both 1.09-1.10x; sponza-white 1280 x 720 frame 1.02-1.03x, both 1.03-1.04x) | `node bench/runner/run.ts --page realtime-check --args check=timing rounds=5 batch=64` | whether the 6 planes (96 B/pixel) need to become `rgba16float` storage textures; the blue-noise default |
| P3.3 / P3.4 / P3.10 / R1b / P3.9 (provisional numbers: TASKS rows of the same names, load 4-20) | Wavefront scheduling settings on a quiet machine: `appends: subgroup` (`pt-wf-sg`, `pt-wf-refill-sg`), block sorts (`pt-wf-sort-mat` / `-oct` / `-both`), `refill: ordered` (`pt-wf-ordered`, `-sg`) against inline / batch / plain / megakernel, the auto policy (`pt-auto`) against the best fixed setting, and `materialLod: secondary` with refill (`pt-wf-refill-lod`, `pt-wf-ordered-lod`): cornell 512x512, sponza-white 1280x720, quad-light, foliage-mask, env-sphere at layers 1 / 4 / 16, lod-cornell, ClearCoatTest-notex. 5 ABBA rounds INSIDE one page per scene (one GPU lock per scene), each variant's image checked against the first id in the same run | `bash bench/quiet-pass.sh --only p3` (steps 7-10), or one scene: `node bench/runner/run.ts --page wf-sched-ab --suite wf-sched-ab --args scene=cornell renderers=pt-megakernel,pt-wavefront,pt-wf-sg,pt-wf-sort-both,pt-wf-refill,pt-wf-ordered spp=256 rounds=5` (ratios are `<id>_ratio_min/max` against the first id; `<id>_max_rel_pixel` must stay <= 1e-5); correctness alone: `node bench/adapters/matrix.ts --renderers pt-megakernel,pt-wf-sg,pt-wf-sort-both,pt-wf-ordered-sg,... --spp 32 --snapshots 32 --load-wait 0 --name X` then `node bench/adapters/pt-wf-sched-agreement.ts --report X` | `appends` default (auto?), whether any sort is worth keeping (provisionally no), whether `ordered` stays, and the constants of `ptwfAutoPolicy` (P3.9 D-row) |
| N11 (M9 NRC pipeline) | The chain on sponza-white at 960 × 540 → 1920 × 1080: one plain sample, one cache frame (4 steps, tiled f16; sgm where available), pack + depth, denoise + composite, upscale; ms each, wall clock in a WINDOWED Chrome (NRC headless times differ, N6e), 3 runs, both the M5 Pro and the RTX 3060 (provisional, 3060 headless under load 16: plain 55, cache 50, denoise 2.5, upscale 4.8 ms) | `node bench/runner/run.ts --page nrc-pipeline-study --headed [--gpu nvidia] --timeout 3600000 --args scenarios=static frames=8 checkpoints=8 refFile=<ours or Mitsuba> flip=0 upload=0` (the `time_*` metrics) | plain N at equal time (nrc-pipeline.md §5), whether the chain fits 16.7 ms |
| N11.7 (warm-up gate) | A gated frame (plain paths + training) at 4 vs 16 training steps, and an open cache frame, ms each, interleaved, 5 rounds, windowed Chrome, sponza-white 960 × 540, OIDN small | demo `nrc-pipeline?denoiser=oidn&gate=auto&gateSteps=16` + Benchmark (or a twin `timeStages` with the gate set) | whether the 16-step default (D30, OIDN gate 38 frames instead of 150) is worth its per-frame cost or goes back to 4 |

## Reported

- 2026-11-14 14:19: P1.6 (no call), P0.25 (M5 baseline done), P2.15 (extra pt-megakernel row) and X1 (M5):
  `bench/results/quiet-pass/20261114-1419/REPORT.md`.
- 2026-11-14 15:43: P1.6 rerun with batch timing (defaults best or tied; compact record slower):
  `bench/results/quiet-pass/20261114-1543/REPORT.md`.
- 2026-11-15 17:14: `--only p3` (P3.1, P3.2, R2 / P3.7, P3.6): `bench/results/quiet-pass/20261115-1714/REPORT.md`. **Not a clean pass**: the
  machine had just rebooted (Spotlight and fileproviderd at ~100% CPU each, 1 browser GPU process, background apps Claude, Creative
  Cloud, Fabric, Notion, Spotify); load1 3.4-6.4 per round. The runner's loaded gate is load1 > core count, so no row is marked
  `loaded` (all valid, noted `best effort (background apps ...)`). Paired interleaved ratios agree with the provisional TASKS numbers,
  so they are informative; absolute ms are not quiet numbers. The four items stay in the Queue (marked "rerun on a quiet machine") and no
  gate or default (P3.9, `alphaPolicy`, `primary: 'vbuffer'`) is decided from this pass. Paired ratios: `node bench/adapters/paired-ratios.ts --report <name> [--base-map a=b,...]`.
  R2, median ms saved per sample vs the same scheduler's exact policy (320x240; 1080p is x27 pixels, a linear ESTIMATE):

  | scene | sched | policy | exact ms | policy ms | saved ms (min..max) | saved % | darker | 1080p ms (est.) |
  |---|---|---|---|---|---|---|---|---|
  | tex-mask | mk | shadow | 0.324 | 0.279 | 0.046 (0.042..0.050) | 14.1% | 5.3% | 1.2 |
  | tex-mask | mk | indirect | 0.324 | 0.336 | -0.012 (-0.014..-0.008) | -3.7% | 2.9% | -0.3 |
  | tex-mask | mk | secondary | 0.324 | 0.284 | 0.039 (0.039..0.041) | 12.2% | 8.4% | 1.1 |
  | foliage-mask | mk | shadow | 1.265 | 1.004 | 0.260 (0.249..0.262) | 20.6% | 12.7% | 7.0 |
  | foliage-mask | mk | indirect | 1.265 | 1.281 | -0.019 (-0.026..-0.010) | -1.5% | 3.8% | -0.5 |
  | foliage-mask | mk | secondary | 1.265 | 0.980 | 0.284 (0.278..0.292) | 22.5% | 17.3% | 7.7 |
  | tex-mask | wf | shadow | 0.406 | 0.383 | 0.021 (0.018..0.025) | 5.3% | 5.3% | 0.6 |
  | tex-mask | wf | indirect | 0.406 | 0.401 | 0.005 (-0.001..0.009) | 1.3% | 2.9% | 0.1 |
  | tex-mask | wf | secondary | 0.406 | 0.380 | 0.023 (0.022..0.027) | 5.6% | 8.4% | 0.6 |
  | foliage-mask | wf | shadow | 1.104 | 0.975 | 0.131 (0.126..0.133) | 11.9% | 12.7% | 3.5 |
  | foliage-mask | wf | indirect | 1.104 | 1.056 | 0.048 (0.046..0.052) | 4.4% | 3.8% | 1.3 |
  | foliage-mask | wf | secondary | 1.104 | 0.906 | 0.198 (0.196..0.201) | 17.9% | 17.3% | 5.3 |

## Lessons from the first attempts (2026-11-14)

- Other apps' GPU work (a Chrome window being browsed, the Claude app's UI) lands inside short passes in bursts: the
  same ~4 ms pass read ~9 ms about two thirds of the time. trace-timing therefore reports the 10th-percentile pass of
  30 interleaved repeats after a 2 s warm-up, and records the median and p90/p10 next to it.
- Pages that time 0.5 s blocks of back-to-back dispatches (bvh's) average the interference instead: tight numbers,
  but they include it.
- The runner's load check ignores desktop apps' GPU helpers (they can't all be closed) and lists them in the note as
  `best effort (background apps: ...)`.
- Even with every app closed, trace-timing's ~4 ms passes read ~4 ms or ~10 ms (p90/p10 about 3): the GPU clock
  drops between short, separated passes. Closing apps does not fix it; long back-to-back batches do (P1.6a, best
  effort on a 2 load-average Mac with only the Claude app running): one command buffer of N passes (N calibrated to
  ~0.5 s), timestamps around the whole batch, per-pass = batch / N. Samples then agree to 0.03-0.3% (max/min ~1.001)
  against p90/p10 1.2-2.5 for the old method, and the same pass reads ~3.7 ms instead of ~7.2 ms (old p10). The old
  numbers were mostly clock ramp, not interference: do not compare old and new absolute values.
- One sample in flight at a time: queueing all samples without `onSubmittedWorkDone` between them hit
  "GPU Timeout Error" (Metal kills a command buffer after ~5 s and Dawn merged the queued submits). Awaiting
  completion between samples is not a readback and keeps the timed region clean.
- Our renderer is not in `ADAPTER_IDS`, so the baseline step leaves it out; the 2026-11-14 pass ran it as an extra step
  with the same settings (`--renderers pt-megakernel`).
