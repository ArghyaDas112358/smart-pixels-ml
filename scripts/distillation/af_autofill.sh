#!/bin/bash
# Keep the Purdue-AF A100 saturated with MDMM x SimpleRouter seeds.
#
# Policy (user, 2026-07-31): the AF A100 is the PREFERRED resource -- whenever a
# worker slot frees, immediately start the next seed on it; everything that does
# not fit goes to Gautschi. This script implements the AF half of that.
#
# MAX_WORKERS=4 is a HARD limit measured on this GPU: a 5th and 6th worker both
# died with OOM in MultiHeadAttention's einsum (the 4.87 GiB sanity figure is a
# solo steady-state peak; transient attention tensors at batch 5000 push real
# usage well past it). 4 workers each run at ~68 s/epoch, i.e. essentially solo
# speed, because the job is input-bound -- a 5th would buy nothing even if it fit.
#
# Safety: never launches a seed that is already running (checked from ps), already
# finished (result.json escaped + at target epochs), or already abandoned (*_STUCK).
# Resume is exact, so a seed picked up mid-flight continues rather than restarts.
#
#   nohup setsid bash scripts/distillation/af_autofill.sh > runs/simplerouter_mdmm_discovery/autofill.log 2>&1 &
set -uo pipefail

REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
DRIVER=$REPO/scripts/distillation/run_simplerouter_mdmm_discovery.py
OUT=$REPO/runs/simplerouter_mdmm_o4beta
MAX_WORKERS=4
TARGET_EPOCHS=5000
POLL=300                       # seconds between checks
# 8042 is deliberately absent: that seed is the O4/beta run on Gautschi.
# O4/beta seeds now: the A100 is the priority resource and runs the beta campaign;
# the plain O11 seeds moved to Gautschi (submit_o11.sbatch) on 2026-08-01.
QUEUE=(8442 8542 8642 8742)

stamp() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

running_seeds() {
    # Seeds with a LIVE driver process. Match on the interpreter path so this
    # script's own argv (which contains the pattern) cannot self-match -- that
    # trap has bitten every pgrep in this project.
    for p in $(pgrep -f "run_simplerouter_mdmm_discovery.py" 2>/dev/null); do
        args=$(ps -o args= -p "$p" 2>/dev/null) || continue
        case "$args" in
            */envs/smartpix-2bit/bin/py*) echo "$args" | grep -oP '(?<=--seeds )[0-9]+' ;;
        esac
    done | sort -u
}

seed_done() {
    local s=$1 d=$OUT/seed_$s
    [[ -d ${d}_STUCK ]] && return 0                       # abandoned
    [[ -f $d/result.json ]] || return 1
    local last
    last=$(tail -1 "$d/history.csv" 2>/dev/null | cut -d, -f1 | tr -d '\r')
    [[ "$last" =~ ^[0-9]+$ ]] && (( last >= TARGET_EPOCHS - 1 ))
}

stamp "autofill started (max $MAX_WORKERS workers, target $TARGET_EPOCHS ep, queue: ${QUEUE[*]})"
while true; do
    mapfile -t live < <(running_seeds)
    n=${#live[@]}
    if (( n < MAX_WORKERS )); then
        for s in "${QUEUE[@]}"; do
            (( n >= MAX_WORKERS )) && break
            # skip if live
            printf '%s\n' "${live[@]}" | grep -qx "$s" && continue
            seed_done "$s" && continue
            stamp "slot free ($n/$MAX_WORKERS busy) -> launching seed $s"
            setsid nohup "$PY" "$DRIVER" --epochs $TARGET_EPOCHS --seeds "$s" \
                   --extend --target 1 --beta-final 30 --beta-start 1000 --beta-epochs 2000 \
                   --out "$OUT" >> "$OUT/launch_$s.log" 2>&1 < /dev/null &
            sleep 30                                       # let it register in ps
            mapfile -t live < <(running_seeds)
            n=${#live[@]}
        done
    fi
    sleep "$POLL"
done
