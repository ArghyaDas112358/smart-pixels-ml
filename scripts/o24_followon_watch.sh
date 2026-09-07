#!/bin/bash
# Session-independent follow-on for O24 (2026-09-07). Waits for warm seed 41042
# to finish on the A100, then (1) runs the -40.4K recovery job -- warm start
# from the O22 checkpoint, 12 epochs, best.weights picked on PLAIN NLL -- and
# (2) migrates the last cold seed 40642 from its Gautschi chain to the A100.
# Launch: setsid nohup bash scripts/o24_followon_watch.sh > runs/o24_followon.log 2>&1 &
set -uo pipefail
R=/work/users/das214/SmartPixels/smart-pixels-ml; cd $R
PY=/work/users/das214/envs/smartpix-2bit/bin/python
D=/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs/o24_scratch_bias
log(){ echo "[$(date '+%F %T')] $*"; }
BASE="SMARTPIX_MODEL_NAME=ViT_MaxDeep_PairLattice SMARTPIX_LOSS_V2=1 SMARTPIX_ZEROBIAS=1 SMARTPIX_BINBAL=byterm SMARTPIX_NBINS=15 SMARTPIX_BIAS_TOL=0.17 SMARTPIX_BIAS_MAXLAM=5.0 SMARTPIX_BIAS_INFCAP=2.0 SMARTPIX_ABORT_THR=1e12"

log "waiting for warm seed 41042 to exit"
# NOT ps|grep -q: under pipefail grep -q ends the pipe early, the upstream grep
# dies of SIGPIPE, the pipeline "fails" and the loop exits on its first check
# (that put a 5th trainer on the A100 on 2026-09-07). Match the python
# executable by exact path so no shell whose cmdline quotes this file matches.
PYX=/work/users/das214/envs/smartpix-2bit/bin/python
alive(){ ps -eo pid,args --no-headers | awk -v py="$PYX" '$2==py && index($0,"--seeds 41042 ")>0' | grep -c . ; }
while [ "$(alive)" -gt 0 ]; do sleep 120; done
log "41042 exited"

# (1) recovery: reproduce the first constrained epochs from the O22 checkpoint, keep the plain-NLL best
log "launching -40.4K recovery run (12 epochs, ckpt monitor = val_plain_nll)"
env $BASE SMARTPIX_WARM_START=$R/runs/o22_ep100/seed_22042/last.weights.hdf5 SMARTPIX_WARM_SRC_MODEL=ViT_MaxDeep_PairLattice \
    SMARTPIX_PIN_MODE=inherit_free SMARTPIX_FIX_SLICES=10,21 SMARTPIX_CKPT_MONITOR=val_plain_nll \
    $PY scripts/distillation/run_simplerouter_mdmm_discovery.py --epochs 12 --target 1 --seeds 41142 \
    --out runs/o24b_recover/s41142 > runs/o24b_recover_41142.log 2>&1
log "recovery run finished: $(grep -m1 -oE 'best val plain NLL[^)]*|CONVERGED.*' runs/o24b_recover_41142.log | head -1)"
[ -f runs/o24b_recover/s41142/seed_41142/history.csv ] && log "recovery best val_plain_nll: $($PY -c "
import csv,numpy as np; r=list(csv.DictReader(open('runs/o24b_recover/s41142/seed_41142/history.csv'))); v=[float(x['val_plain_nll']) for x in r]; print(f'{min(v):,.0f} at ep {int(np.argmin(v))+1}')")"

# (2) migrate cold seed 40642 (its chain is the only o24bias job left on Gautschi)
log "migrating cold seed 40642: cancelling its Gautschi chain"
timeout 90 ssh -o BatchMode=yes gautschi 'scancel -u $USER -n o24bias; sleep 8; scancel -u $USER -n o24bias; sleep 5; squeue -u $USER -h -n o24bias | wc -l' 2>/dev/null | tail -1 | sed 's/^/  gautschi o24bias jobs left: /'
sleep 30
mkdir -p runs/o24_scratch_bias/s40642
if [ ! -d runs/o24_scratch_bias/s40642/seed_40642 ]; then cp -a $D/s40642/seed_40642 runs/o24_scratch_bias/s40642/ && cp -a $D/s40642/discovery.log runs/o24_scratch_bias/s40642/ 2>/dev/null; fi
log "40642 copied, resumes from epoch $(tail -1 runs/o24_scratch_bias/s40642/seed_40642/history.csv | cut -d, -f1)"
env $BASE SMARTPIX_SMOOTH_SIGMA0=4 SMARTPIX_SMOOTH_EPOCHS=3000 \
    setsid nohup $PY scripts/distillation/run_simplerouter_mdmm_discovery.py --epochs 10000 --target 1 --seeds 40642 --extend \
    --out runs/o24_scratch_bias/s40642 > /tmp/o24cold_40642.log 2>&1 < /dev/null &
sleep 240
log "40642 on A100: $(grep 'RESUMING from epoch' runs/o24_scratch_bias/s40642/discovery.log | tail -1 | cut -c1-90)"
log "trainers on A100 now: $(ps -eo cmd | grep run_simplerouter_mdmm_discovery | grep -v grep | wc -l)"
log "done"
