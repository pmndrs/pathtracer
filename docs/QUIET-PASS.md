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
   - for each trace-timing A/B: paired B/A ratio of `*_mrays_per_s` per round (min..max over the 5 rounds) per ray
     set, plus the median of `*_p90_over_p10` (near 1 = clean passes; above ~1.3 = something interfered);
   - the baseline table from `bench/results/quiet-pass/<stamp>/baseline.md`;
   - load at start and end, and any row whose note says `loaded` or `best effort`.
   Write the report to `bench/results/quiet-pass/<stamp>/REPORT.md`, and add one line per item to TASKS.md pointing at it.
7. Commit only the report and the TASKS line if the maintainer asks; never push.

## Queue

What the next pass should measure. Add a row when something needs a quiet-machine number; remove it once reported.

| Item | What | Command | Decides |
|---|---|---|---|
| P1.6 | Trace core settings: workgroup 64/128, AoS/SoA rays, full/compact hit record | `bench/quiet-pass.sh --only trace` | defaults in docs/design/trace-core.md, open choices 1–3 |
| P0.25 | Competitor baseline, renderers interleaved, Cornell + Sponza | `bench/quiet-pass.sh --only baseline` | P0 exit gate; replaces the provisional BASELINE-P0.md |
| P2.3 | VNDF standard vs spherical cap (and maybe a branchless basis): cost per sample, noise at equal spp | not runnable yet: needs the integrator (P2.4); add the command here when it exists | `vndf` default, docs/design/bsdf.md choice 1 |
| X1 | Wide BVH (BVH4 / CWBVH8) vs BVH2, incl. the single-loop walk (L7) | in the bvh worktree `~/Developer/bvh/.claude/worktrees/wide-bvh-684631` (branch `claude/wide-bvh-684631`): its HANDOFF.md "Quiet pass" section, commands as written | the 1.3× gate |

## Lessons from the first attempts (2026-11-14)

- Other apps' GPU work (a Chrome window being browsed, the Claude app's UI) lands inside short passes in bursts: the
  same ~4 ms pass read ~9 ms about two thirds of the time. trace-timing therefore reports the 10th-percentile pass of
  30 interleaved repeats after a 2 s warm-up, and records the median and p90/p10 next to it.
- Pages that time 0.5 s blocks of back-to-back dispatches (bvh's) average the interference instead: tight numbers,
  but they include it.
- The runner's load check ignores desktop apps' GPU helpers (they can't all be closed) and lists them in the note as
  `best effort (background apps: ...)`.
