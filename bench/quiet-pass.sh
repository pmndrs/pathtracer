#!/bin/bash
# Quiet pass: every timing that needs a quiet machine, in one run (docs/QUIET-PASS.md).
#
#   bash bench/quiet-pass.sh [--only trace|baseline] [--gpu apple|nvidia]
#
# Run it with every other app closed and the machine plugged in. Results go to bench/results/quiet-pass/<stamp>/.
# Each step goes through the shared GPU lock and the runner's load check, so a busy machine is still recorded as
# such (valid=0 `loaded`, or `best effort (background apps: ...)` in the note).
set -u
cd "$(dirname "$0")/.."
ONLY=""
GPU="apple"
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
STAMP=$(date +%Y%m%d-%H%M)
OUT="bench/results/quiet-pass/$STAMP"
mkdir -p "$OUT"
LOG="$OUT/quiet-pass.log"
note() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
load() { if [ "$(uname)" = "Darwin" ]; then sysctl -n vm.loadavg; else echo "n/a"; fi; }

note "quiet pass $STAMP on $(uname -n), gpu=$GPU, commit $(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty')"
note "load $(load); browser GPU processes: $(pgrep -f '(Google Chrome|Chromium|Microsoft Edge).*--type=gpu-process' | wc -l | tr -d ' ')"

note "1/3 timer calibration"
node bench/runner/run.ts --page timer-calibration --gpu "$GPU" --suite quiet-timer-calibration --json "$OUT/timer-calibration.json" >> "$LOG" 2>&1 || note "FAILED timer calibration"

if [ -z "$ONLY" ] || [ "$ONLY" = "trace" ]; then
  # Trace core settings (TASKS P1.6, design doc open choices 1-3). Each page run times two ways (trace-timing `timing=both`):
  # the headline is the batch method (>= 0.5 s of back-to-back passes per sample, P1.6a; spread ~1.00x on a quiet M5 Pro),
  # next to the old 10th-percentile pass of 30 (`*_mrays_per_s`, `*_p90_over_p10`). 5 ABBA page rounds per setting.
  for ab in workgroupSize=64,128 rayLayout=aos,soa hitRecord=full,compact; do
    key=${ab%%=*}
    note "2/3 trace-timing --ab $ab"
    node bench/runner/run.ts --page trace-timing --gpu "$GPU" --ab "$ab" --repeat 5 --suite quiet-trace-timing \
      --json "$OUT/trace-timing-$key.json" --timeout 300000 --args scene=sponza-white >> "$LOG" 2>&1 || note "FAILED trace-timing $ab"
  done
fi

if [ -z "$ONLY" ] || [ "$ONLY" = "baseline" ]; then
  # P0 exit gate (TASKS P0.25): competitors, renderers interleaved per round (order reverses every round).
  # Sponza at 512 spp for all three so they interleave (1e-2 is the measurable target; the floor is ~4e-3).
  NAME="p0-baseline-quiet-$STAMP"
  note "3/3 P0 baseline: cornell"
  node bench/adapters/matrix.ts --scenes cornell --spp 1024 --snapshots 4,16,64,256 --repeat 3 --interleave --name "$NAME" >> "$LOG" 2>&1 || note "FAILED baseline cornell"
  note "3/3 P0 baseline: sponza-white"
  node bench/adapters/matrix.ts --scenes sponza-white --spp 512 --snapshots 4,16,64,256 --repeat 3 --interleave --append --name "$NAME" >> "$LOG" 2>&1 || note "FAILED baseline sponza"
  node bench/adapters/baseline.ts --report "$NAME" --csv > "$OUT/baseline.md" 2>> "$LOG" || note "FAILED baseline table"
  note "baseline table: $OUT/baseline.md (report bench/results/reports/$NAME)"
fi

note "done; load $(load)"
