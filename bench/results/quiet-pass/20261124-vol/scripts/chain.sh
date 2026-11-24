#!/bin/bash
# Quiet pass 2026-11-24: phase 1 = bench/quiet-pass.sh (timer, trace, baseline); phase 2 = volume rows; phase 3 = single-page rows.
cd /Users/dex/Developer/pathtracer || exit 1
QP="bench/results/quiet-pass/20261124-vol"
mkdir -p "$QP"
LOG="$QP/chain.log"
SP=/private/tmp/claude-502/-Users-dex-Developer-pathtracer/ecb4a7f4-10e5-420e-b42b-56dd55c7fb50/scratchpad/quiet-pass
AB="$PWD/$QP/abba.json"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
page() { local tag="$1"; shift; note "page $tag: $*"; node bench/runner/run.ts --no-csv --timeout 3600000 --json "$PWD/$QP/$tag.json" "$@" >> "$LOG" 2>&1 || note "FAILED $tag"; }
abba() { note "abba $*"; node "$SP/abba.ts" "$AB" "$@" >> "$LOG" 2>&1 || note "FAILED abba $1"; }
note "start load $(sysctl -n vm.loadavg)"

if [ "${SKIP1:-0}" != 1 ]; then
  note "phase 1: bench/quiet-pass.sh"
  bash bench/quiet-pass.sh >> "$LOG" 2>&1 || note "FAILED quiet-pass.sh"
fi

note "phase 2: volumes, load $(sysctl -n vm.loadavg)"
B=pt_volume_majorant=brick
# V1 media (cloud256 + sphere t200)
abba v1-cloud256 vol-cl-cloud256 256 5 "global=pt_volume_majorant=global" "brick=$B" "brick-roulette=$B;pt_tr_roulette=1" "brick-r16=$B;pt_volume_grid=r16float" "brick-r16-hw=$B;pt_volume_grid=r16float;pt_density_filter=hardware" "wf-refill-brick=renderer=pt-wf-refill;$B"
abba v1-sph-t200 vol-cl-sph-g85-a99-t200 256 5 "ratio=pt_volume_shadow=ratio" "ratio-roulette=pt_volume_shadow=ratio;pt_tr_roulette=1" "wf-refill-ratio=renderer=pt-wf-refill;pt_volume_shadow=ratio"
# P5 media: shadow tracking, majorant
abba p5-shadow-smoke vol-cornell-smoke 1024 5 "analytic=pt_volume_shadow=analytic" "ratio=pt_volume_shadow=ratio"
abba p5-shadow-milk vol-milk-sphere 1024 5 "analytic=pt_volume_shadow=analytic" "ratio=pt_volume_shadow=ratio"
abba p5-maj-gridcloud vol-grid-cloud 1024 5 "global=pt_volume_majorant=global" "brick=$B"
abba p5-maj-vdbsmoke vol-vdb-smoke 256 5 "global=pt_volume_majorant=global" "brick=$B"
# V2.4 decomposition tracking
abba v24-cloud256 vol-cl-cloud256 256 5 "delta=$B;pt_tr_roulette=1" "decomposition=$B;pt_tr_roulette=1;pt_volume_flight=decomposition"
abba v24-bunny vol-vdb-bunny-cloud 256 5 "delta=$B;pt_tr_roulette=1" "decomposition=$B;pt_tr_roulette=1;pt_volume_flight=decomposition"
# V7 media defaults
for s in vol-cornell-fog vol-cornell-smoke vol-lamp-fog vol-thin-fog-sun vol-milk-sphere; do
  abba "v7-$s" "$s" 256 5 "default=" "forced-equi=pt_forced_nee=1;pt_equiangular=1" "forced-equi-march4=pt_forced_nee=1;pt_equiangular=1;pt_march=4"
done
# V4.3 VSPG
for s in vol-thin-fog-sun vol-lamp-fog vol-grid-cloud; do
  abba "v43-$s" "$s" 256 5 "plain=" "vspg=pt_vspg=1"
done
# P5 wavefront media / spectral, V5.1d wavefront frame + media, V4.4, V2.5
page wfp5-media --page wf-p5-check --args cases=vol-cornell-smoke,vol-milk-sphere,vol-vdb-smoke-brick variants=plain,inline,ordered,batch spp=128 timing=1 rounds=7
page wfp5-spectral --page wf-p5-check --args cases=cornell,swatches-refl,dispersion-prism variants=plain,inline,ordered,batch spp=16 timing=1 rounds=7
page v51d --page wf-frame-check --args cases=vol-emit,vol-smoke-lift,vol-cloud timing=1 rounds=7
page v44 --page vol-forced-grid-check --args cases=vol-lamp-fog-grid:plain,vol-lamp-fog-grid:forced-g,vol-lamp-fog-grid:equi-g,vol-cornell-smoke-grid:plain,vol-cornell-smoke-grid:equi-g seeds=8 spp=256 pairs=0-1,0-2,3-4
page v25-sph --page render-until-check --args scene=vol-cl-sph-g85-a99-t50 match=1 pixel=0.05 seeds=6 w=64 h=64 max=4096
page v25-cloud --page render-until-check --args scene=vol-cl-cloud256 match=1 pixel=0.15 seeds=6 w=64 h=64 max=4096

note "phase 3: single-page rows, load $(sysctl -n vm.loadavg)"
page p41 --page realtime-check --args check=timing rounds=5 batch=64
page p41b-sponza --page wf-frame-check --args cases=sponza-white timing=1 res=960x540 rounds=7
page p41b-sponza-restir --page wf-frame-check --args cases=sponza-white timing=1 res=960x540 rounds=7 restir=unbiased
page p41b-cornell --page wf-frame-check --args cases=cornell timing=1 res=512x512 rounds=7
page p41b-manylights --page wf-frame-check --args cases=many-lights timing=1 res=256x256 rounds=7 restir=unbiased
page p4-restir --page restir-check --args check=timing scenes=sponza-white,many-lights-256 w=1280 h=720 reps=5 timingframes=16
page p45-mk --page realtime-pipeline --args scenes=sponza-white scenarios=timing restir=off,on,gi,giu scheduler=megakernel
page p45-wf --page realtime-pipeline --args scenes=sponza-white scenarios=timing restir=off,on,gi,giu scheduler=wavefront
page p47-oidn --page realtime-pipeline --args scenes=sponza-white scenarios=timing restir=off timingVariant=oidn-fast res=640x360 maxDepth=4
page rc4 --page rc-cascades --suite rc-cascades
page rc4-dirs8 --page rc-cascades --suite rc-cascades --args dirs0=8
page x5 --page downsampler-check
note "done, load $(sysctl -n vm.loadavg)"
echo QUIET_CHAIN_DONE >> "$LOG"
