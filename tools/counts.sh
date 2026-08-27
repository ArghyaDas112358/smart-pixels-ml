#!/bin/bash
# Status one-liner for the monitor. Lives on Depot so the quoting survives the
# ssh round-trip -- inlining greps with nested quotes silently produced escaped=0
# when the true count was 9.
B=/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs
squeue -u das214 -h -o "%j" 2>/dev/null | sort | tr '\n' ';'
printf ' escaped=%s' "$(grep -l '"escaped": true' $B/fixedslice*_p*/seed_*/result.json 2>/dev/null | wc -l)"
printf ' aborted=%s' "$(ls -d $B/fixedslice*_p*/seed_*_STUCK 2>/dev/null | wc -l)"
printf ' running=%s\n' "$(ls -d $B/fixedslice*_p*/seed_* 2>/dev/null | grep -v _STUCK | while read d; do [ -f "$d/result.json" ] || echo x; done | wc -l)"
# O20 warm-start probe: per-seed epoch + current router pair from the run CSVs.
O20=$B/simplerouter_o20warm
for d in $O20/seed_*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed s/seed_//)
  ep=$(tail -1 "$d/history.csv" 2>/dev/null | cut -d, -f1)
  pr=$(tail -1 "$d/router_epochs.csv" 2>/dev/null | cut -d, -f2,3)
  printf "o20_%s=ep%s@(%s) " "$s" "${ep:-?}" "${pr:-?}"
done
[ -d "$O20" ] && ls -d $O20/seed_* >/dev/null 2>&1 && echo ""
# O21.v2: loss-v2 retrain of the three mechanism arms (paired seeds vs O21).
for d in $B/o21v2*/seed_*; do
  [ -d "$d" ] || continue
  tag=$(basename "$(dirname "$d")" | sed s/o21v2//;)
  ep=$(tail -1 "$d/history.csv" 2>/dev/null | cut -d, -f1)
  pr=$(tail -1 "$d/router_epochs.csv" 2>/dev/null | cut -d, -f2,3)
  printf "v2%s=ep%s@(%s) " "${tag%%_*}" "${ep:-?}" "${pr:-?}"
done
ls -d $B/o21v2* >/dev/null 2>&1 && echo ""
# O21.v2a.2 / v2c.2 -- COLD zero-prior proof runs (3 seeds each).
for d in $B/o21v2a2_pairlattice/seed_* $B/o21v2c2_ucb/seed_*; do
  [ -d "$d" ] || continue
  arm=a; case "$d" in *ucb*) arm=c ;; esac
  s=$(basename "$d" | sed s/seed_//)
  ep=$(tail -1 "$d/history.csv" 2>/dev/null | cut -d, -f1)
  pr=$(tail -1 "$d/router_epochs.csv" 2>/dev/null | cut -d, -f2,3)
  printf "cold%s_%s=ep%s@(%s) " "$arm" "$s" "${ep:-0}" "${pr:-?}"
done
ls -d $B/o21v2a2_pairlattice/seed_* $B/o21v2c2_ucb/seed_* >/dev/null 2>&1 && echo ""
# O21.v3: cold + loss-v2 + FULL phi history from epoch 0 (fresh seeds).
for d in $B/o21v3*/seed_*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed s/seed_//)
  ep=$(tail -1 "$d/history.csv" 2>/dev/null | cut -d, -f1)
  pr=$(tail -1 "$d/router_epochs.csv" 2>/dev/null | cut -d, -f2,3)
  nf=$(ls -la "$d/phi_history.npz" 2>/dev/null | awk "{print \$5}")
  printf "v3_%s=ep%s@(%s)phi%s " "$s" "${ep:-?}" "${pr:-?}" "${nf:-0}"
done
ls -d $B/o21v3* >/dev/null 2>&1 && echo ""
