#!/bin/bash
# Emits ONE line per notable change. O21 v1 is COMPLETE; what remains live is
# the O21.v2 loss-v2 retrain + the last fxD chain, both queued/running on
# Gautschi (counts.sh emits v2a/v2b/v2c=ep@(pair) lines once dirs exist).
# AF workers should stay 0 -- a nonzero count means something unexpected.
SCDIR=/tmp/claude-978920/-work-users-das214-SmartPixels/7a0a041e-b87d-47e5-ae22-8b5c100f4193/scratchpad
COUNTS=/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/tools/counts.sh
prev=""
while true; do
  cur=""
  n=$(ps -eo comm,args | awk '$1 ~ /^python/ && /run_simplerouter_mdmm_discovery/' | wc -l)
  cur+="af_workers=$n"$'\n'
  g=$(ssh -o ConnectTimeout=25 -o BatchMode=yes gautschi "bash $COUNTS" 2>/dev/null | tr -s " ")
  if [ -n "$g" ]; then cur+="gautschi: $g"$'\n'; else cur+="gautschi: UNREACHABLE"$'\n'; fi
  if [ "$cur" != "$prev" ]; then
    diff <(printf '%s' "$prev") <(printf '%s' "$cur") 2>/dev/null | grep '^>' | sed 's/^> //'
  fi
  prev="$cur"
  sleep 600
done
