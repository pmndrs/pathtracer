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

if [ "$ONLY" = "p3" ]; then
  # P3 queue items (docs/QUIET-PASS.md rows P3.1/P3.2, R2/P3.7, P3.6). Not in the default run: --only p3.
  # P3.2's renderer set includes the megakernel and the slice-1 wavefront, so it also answers P3.1.
  R=pt-megakernel,pt-wavefront,pt-wf-refill,pt-wf-batch,pt-wf-refill-l2,pt-wf-refill-l4
  note "p3 1/6 schedulers: cornell + furnace-dielectric"
  node bench/adapters/matrix.ts --renderers $R --scenes cornell,furnace-dielectric --spp 256 --repeat 5 --interleave --name "p32-big-$STAMP" >> "$LOG" 2>&1 || note "FAILED p32 big"
  note "p3 2/6 schedulers: sponza-white"
  node bench/adapters/matrix.ts --renderers $R --scenes sponza-white --spp 128 --repeat 5 --interleave --name "p32-sponza-$STAMP" >> "$LOG" 2>&1 || note "FAILED p32 sponza"
  note "p3 3/6 schedulers: env-sphere + quad-light"
  node bench/adapters/matrix.ts --renderers $R --scenes env-sphere,quad-light --spp 1024 --repeat 5 --interleave --name "p32-small-$STAMP" >> "$LOG" 2>&1 || note "FAILED p32 small"
  note "p3 4/6 layers sweep (env-sphere) and refill threshold (sponza-white)"
  for L in 1 2 4 8 16 32 64; do
    node bench/adapters/matrix.ts --renderers pt-megakernel,pt-wf-refill --scenes env-sphere --spp 1024 --repeat 3 --interleave --extra pt_layers=$L --name "p32-layers$L-$STAMP" >> "$LOG" 2>&1 || note "FAILED layers $L"
  done
  for T in 0.1 0.25 0.5; do
    node bench/adapters/matrix.ts --renderers pt-megakernel,pt-wf-batch --scenes sponza-white --spp 64 --repeat 3 --interleave --extra pt_refill_threshold=$T --name "p32-thr$T-$STAMP" >> "$LOG" 2>&1 || note "FAILED threshold $T"
  done
  note "p3 5/6 alpha policy + wavefront MASK"
  node bench/adapters/matrix.ts --renderers pt-megakernel,pt-mk-shadow,pt-mk-indirect,pt-mk-secondary,pt-wavefront,pt-wf-shadow,pt-wf-indirect,pt-wf-secondary --scenes tex-mask,foliage-mask --spp 1024 --snapshots 1024 --repeat 5 --interleave --name "p37-ab-$STAMP" >> "$LOG" 2>&1 || note "FAILED p37"
  note "p3 6/6 V-buffer primaries"
  RV=pt-megakernel,pt-mk-vbuffer,pt-wavefront,pt-wf-vbuffer
  node bench/adapters/matrix.ts --renderers $RV --scenes cornell,tex-mask --spp 256 --repeat 5 --interleave --name "p36-time-$STAMP" >> "$LOG" 2>&1 || note "FAILED p36 small"
  node bench/adapters/matrix.ts --renderers $RV --scenes sponza-white --spp 64 --repeat 5 --interleave --append --name "p36-time-$STAMP" >> "$LOG" 2>&1 || note "FAILED p36 sponza"
  node bench/adapters/matrix.ts --renderers $RV --scenes cornell,tex-mask --spp 256 --repeat 5 --interleave --extra pt_maxdepth=2 --name "p36-d2-$STAMP" >> "$LOG" 2>&1 || note "FAILED p36 d2"
  node bench/adapters/matrix.ts --renderers $RV --scenes sponza-white --spp 64 --repeat 5 --interleave --extra pt_maxdepth=2 --append --name "p36-d2-$STAMP" >> "$LOG" 2>&1 || note "FAILED p36 d2 sponza"

  # P3.3 / P3.4 / P3.10 / R1b / P3.9 (docs/QUIET-PASS.md rows of the same names): subgroup appends, block sorts, ordered restart, LOD with refill.
  # Paired ratios: node bench/adapters/paired-ratios.ts --report <name> [--base pt-megakernel | pt-wavefront]
  note "p3 7/10 P3.3 + P3.4 + P3.10: appends, sorts, refill flavours (one page per scene, all variants interleaved ABBA inside it; each also checks its image against the first id)"
  RA=pt-megakernel,pt-wavefront,pt-wf-sg,pt-wf-sort-mat,pt-wf-sort-oct,pt-wf-sort-both
  RB=pt-megakernel,pt-wavefront,pt-wf-refill,pt-wf-ordered,pt-wf-refill-sg,pt-wf-ordered-sg,pt-wf-batch
  ab() { # name scene spp renderers extra-args...
    local name=$1 scene=$2 spp=$3 r=$4; shift 4
    node bench/runner/run.ts --page wf-sched-ab --suite wf-sched-ab --timeout 3000000 --json "$OUT/wf-sched-ab-$name.json" --args scene=$scene renderers=$r spp=$spp rounds=5 "$@" >> "$LOG" 2>&1 || note "FAILED wf-sched-ab $name"
  }
  for set in A B; do
    R=$RA; [ $set = B ] && R=$RB
    ab $set-cornell cornell 256 $R
    ab $set-sponza sponza-white 64 $R
    ab $set-quad quad-light 1024 $R
    ab $set-foliage foliage-mask 1024 $R
    ab $set-env-l1 env-sphere 1024 $R
  done
  for L in 4 16; do ab B-env-l$L env-sphere 1024 pt-megakernel,pt-wf-refill,pt-wf-ordered,pt-wf-ordered-sg pt_layers=$L; done
  ab B-cornell-l2 cornell 256 pt-megakernel,pt-wf-refill,pt-wf-ordered,pt-wf-ordered-sg pt_layers=2
  note "p3 8/10 P3.9 auto policy vs the best fixed setting per scene"
  for sc in cornell:256 sponza-white:64 env-sphere:1024 quad-light:1024 foliage-mask:1024; do
    ab auto-${sc%%:*} ${sc%%:*} ${sc##*:} pt-megakernel,pt-wavefront,pt-wf-refill,pt-wf-batch,pt-auto
  done
  note "p3 9/10 R1b: materialLod secondary with refill"
  ab r1b-lod-cornell lod-cornell 512 pt-mk-lod,pt-wf-lod,pt-wf-refill-lod,pt-wf-ordered-lod
  ab r1b-clearcoat ClearCoatTest-notex 512 pt-mk-lod,pt-wf-lod,pt-wf-refill-lod,pt-wf-ordered-lod
  note "p3 10/10 correctness of every scheduling variant (per pixel vs the megakernel; load-wait 0, no timing from this)"
  node bench/adapters/matrix.ts --renderers pt-megakernel,pt-wavefront,pt-wf-sg,pt-wf-sort-both,pt-wf-refill-sg,pt-wf-ordered-sg,pt-wf-ordered-l2,pt-wf-batch --scenes cornell,quad-light,env-sphere,sponza-white,tex-mask,foliage-mask --spp 32 --snapshots 32 --load-wait 0 --name "wfs-agree-$STAMP" >> "$LOG" 2>&1 || note "FAILED agreement run"
  node bench/adapters/pt-wf-sched-agreement.ts --report "wfs-agree-$STAMP" --out "$OUT/wfs-agreement.md" >> "$LOG" 2>&1 || note "FAILED: a scheduling variant differs from the megakernel"
fi

note "done; load $(load)"
