#!/bin/bash
# Rerun of the V7 / V4.3 rows whose 256 spp renders were ~30 ms (timing noise dominated): 8192 spp, same ABBA driver.
cd /Users/dex/Developer/pathtracer || exit 1
QP="bench/results/quiet-pass/20261124-vol"
LOG="$QP/rerun-short.log"
SP=/private/tmp/claude-502/-Users-dex-Developer-pathtracer/ecb4a7f4-10e5-420e-b42b-56dd55c7fb50/scratchpad/quiet-pass
AB="$PWD/$QP/abba-8192.json"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
abba() { note "abba $*"; node "$SP/abba.ts" "$AB" "$@" >> "$LOG" 2>&1 || note "FAILED abba $1"; }
note "start load $(sysctl -n vm.loadavg)"
for s in vol-lamp-fog vol-thin-fog-sun; do
  abba "v7-$s-8192" "$s" 8192 5 "default=" "forced-equi=pt_forced_nee=1;pt_equiangular=1" "forced-equi-march4=pt_forced_nee=1;pt_equiangular=1;pt_march=4"
  abba "v43-$s-8192" "$s" 8192 5 "plain=" "vspg=pt_vspg=1"
done
note "done, load $(sysctl -n vm.loadavg)"
echo RERUN_DONE >> "$LOG"
