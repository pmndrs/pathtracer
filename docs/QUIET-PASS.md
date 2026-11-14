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
| P2.3 | VNDF standard vs spherical cap (and maybe a branchless basis): cost per sample, noise at equal spp | not runnable yet: needs the integrator (P2.4); add the command here when it exists | `vndf` default, docs/design/bsdf.md choice 1 |

## Reported

- 2026-11-14 14:19: P1.6 (no call), P0.25 (M5 baseline done), P2.15 (extra pt-megakernel row) and X1 (M5):
  `bench/results/quiet-pass/20261114-1419/REPORT.md`.
- 2026-11-14 15:43: P1.6 rerun with batch timing (defaults best or tied; compact record slower):
  `bench/results/quiet-pass/20261114-1543/REPORT.md`.

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
