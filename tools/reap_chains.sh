#!/bin/bash
# Revive silently-dead PAIR-LATTICE chains on Gautschi.
#
# Why: each chunk ends by calling sbatch, which the smallgpu QOS refuses once
# the per-user submit limit is hit (QOSMaxSubmitJobPerUserLimit). The chain then
# stops -- silently before the RESUBMIT GUARD, loudly after it. Either way the
# seed stalls, so this reaper resubmits it.
#
# CRITICAL: staleness alone is NOT enough. A job that is QUEUED but not yet
# started writes nothing, so history.csv looks stale while a perfectly good job
# waits. Resubmitting then would (a) spam the submit limit and (b) risk TWO
# jobs running one seed and corrupting each other's checkpoint. So every
# submission records its job id in .reaper_job, and we skip while that job is
# still in the queue.
#
# Scope: lattice arms only -- the bandit arm was dropped by the user and is
# deliberately left dead. Seeds at the 9999 target are finished, not stalled.
B=/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml
STALE_MIN=${1:-15}
now=$(date +%s)
cd $B || exit 1
QUEUED=$(squeue -u "$USER" -h -o "%i" | tr '\n' ' ')
NJOBS=$(squeue -u "$USER" -h | wc -l)
# smallgpu QOS caps MaxSubmitJobsPerUser at 8 (and 4 GPUs/user). At the ceiling
# every sbatch is refused, and staleness then just means "queued behind other
# users", not "chain died" -- so submitting is pointless noise. Bail early.
CAP=8
if [ "$NJOBS" -ge "$CAP" ]; then
  echo "  queue at cap ($NJOBS/$CAP) -- seeds are waiting on the scheduler, not dead; no resubmits"
  exit 0
fi

revive() {   # $1=run subdir  $2=seed  $3=sbatch script
  d=$B/runs/$1/seed_$2
  f=$d/history.csv
  [ -f "$f" ] || { echo "  $2: no history -- skip"; return; }
  ep=$(tail -1 "$f" | cut -d, -f1)
  age=$(( (now - $(stat -c %Y "$f")) / 60 ))
  if [ "$ep" -ge 9999 ] 2>/dev/null; then echo "  $2: ep $ep DONE"; return; fi
  if [ "$age" -le "$STALE_MIN" ]; then echo "  $2: ep $ep alive (${age}m)"; return; fi
  # already have a pending/running job from a previous reap? then just wait.
  if [ -f "$d/.reaper_job" ]; then
    j=$(cat "$d/.reaper_job")
    case " $QUEUED " in *" $j "*) echo "  $2: ep $ep stale ${age}m but job $j still queued -- waiting"; return;; esac
  fi
  if [ "$NJOBS" -ge "$CAP" ]; then
    echo "  $2: ep $ep stale ${age}m but queue now at cap ($NJOBS/$CAP) -- deferring"
    return
  fi
  out=$(ARM=a SEED=$2 sbatch scripts/gautschi/$3 2>&1)
  NJOBS=$((NJOBS + 1))
  jid=$(echo "$out" | grep -oE '[0-9]+$')
  [ -n "$jid" ] && echo "$jid" > "$d/.reaper_job"
  echo "  $2: ep $ep STALE ${age}m -> $out"
}
echo "=== lattice chain reaper (stale > ${STALE_MIN}m) ==="
for s in 22042 22142 22242; do revive o21v2a2_pairlattice   $s submit_o21v2_cold.sbatch; done
for s in 24042 24142 24242; do revive o21v3a_pairlattice_phi $s submit_o21v3_phi.sbatch;  done
