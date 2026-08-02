#!/bin/bash
# SIMPLE (Option D) discovery orchestrator — survives SSH/VS Code/Claude death.
#   ① 3-epoch GPU sanity (measured s/epoch + peak memory) — gate before launch
#   ② launch discovery: 1000 epochs x 6 converged seeds (detached, resumable)
#   ③ when discovery ends: auto-generate the study report
# The all-101 discovery TFRecords already exist and were verified by
# orchestrate_discovery.sh (row-exact, 80+20 files, pinned labels_scale), so no
# build-wait/verify steps here. Every step logs to
# runs/simplerouter_discovery/orchestrator.log; any failure aborts BEFORE the
# GPU launch and says why.
set -u
R=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python

# true iff a REAL driver process (the env python) is running --epochs. Watchers/editors
# whose command line merely CONTAINS the pattern must not count (bug seen live 2026-07-11:
# a monitoring shell matched the bare pgrep and the launch was skipped).
driver_running() {
  for pid in $(pgrep -f 'run_simplerouter_discovery.py --epochs' 2>/dev/null); do
    ps -o args= -p "$pid" 2>/dev/null | grep -q "bin/python" && return 0
  done
  return 1
}

D=$R/runs/simplerouter_discovery
S=$D/orchestrator.log
mkdir -p "$D"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$S"; }

log "=== simplerouter orchestrator started (pid $$) ==="

# ---- ① measured GPU sanity ----------------------------------------------------
rm -rf "$D/sanity"
$PY $R/scripts/distillation/run_simplerouter_discovery.py --sanity >> "$S" 2>&1
grep -q "SANITY done" "$D/sanity/discovery.log" 2>/dev/null \
  || { log "ABORT: GPU sanity FAILED -- discovery NOT launched"; exit 1; }
log "sanity: $(grep 'SANITY' "$D/sanity/discovery.log" | tail -2 | tr '\n' ' | ')"

# ---- ② launch discovery (idempotent) ------------------------------------------
if driver_running; then
  log "discovery already running -- not launching a second copy"
else
  setsid nohup $PY $R/scripts/distillation/run_simplerouter_discovery.py --epochs 1000 --target 6 \
      > "$D/discovery_stdout.log" 2>&1 < /dev/null &
  sleep 10
  log "DISCOVERY LAUNCHED (1000 ep x 6 converged seeds) pid=$(pgrep -f 'run_simplerouter_discovery.py --epochs' | head -1)"
fi

# ---- ③ report when it all ends -------------------------------------------------
sleep 30
while driver_running; do sleep 300; done
log "discovery finished -- generating study report"
CUDA_VISIBLE_DEVICES='' $PY $R/scripts/distillation/make_simplerouter_study.py >> "$S" 2>&1
log "=== ALL DONE -- report: $D/REPORT.md ==="
