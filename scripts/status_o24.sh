#!/bin/bash
# One-shot status of the O24 campaigns on both machines. Any agent (or human)
# can run this; it needs no session state. Usage: bash scripts/status_o24.sh
R=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
summ='
import csv, sys, numpy as np, os, json
f=sys.argv[1]; tag=sys.argv[2]
r=list(csv.DictReader(open(f)))
c="val_plain_nll" if "val_plain_nll" in r[0] else "val_loss"
v=np.array([float(x[c]) for x in r if x.get(c) not in (None,"")]); v=v[np.isfinite(v)&(np.abs(v)<5e6)]
n=len(v); b=int(np.argmin(v)); done=os.path.exists(os.path.join(os.path.dirname(f),"result.json"))
flag="DONE" if done else ""
print(f"  {tag:<14} ep {n:>5}/10000  best {v.min():>10,.0f} (ep {b+1:>5}, {n-1-b:>4} stale)  med100 {np.median(v[-100:]):>10,.0f}  {flag}")
'
echo "=== AF A100 trainers ==="
ps -eo pid,etime,cmd --no-headers | grep run_simplerouter_mdmm_discovery | grep -v grep | sed -E 's/.*--seeds ([0-9]+).*/  running seed \1/' | sort
echo "=== O24b warm (runs/o24b_from_o22) ==="
for s in 41042 41142 41242; do f=$R/runs/o24b_from_o22/s$s/seed_$s/history.csv; [ -f $f ] && python3 -c "$summ" $f "warm $s"; done
echo "=== O24 cold on the AF (runs/o24_scratch_bias) ==="
for s in 40342 40442 40542 40642; do f=$R/runs/o24_scratch_bias/s$s/seed_$s/history.csv; [ -f $f ] && python3 -c "$summ" $f "cold $s"; done
echo "=== O24 cold still on Gautschi (Depot copy) ==="
D=/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs/o24_scratch_bias
for s in 40342 40442 40542 40642; do f=$D/s$s/seed_$s/history.csv; [ -f $f ] && python3 -c "$summ" $f "depot $s"; done
timeout 60 ssh -o BatchMode=yes gautschi 'echo "  gautschi queue: $(squeue -u $USER -h -o "%j:%T" 2>/dev/null | sort | uniq -c | tr -s " " | tr "\n" " ")"' 2>/dev/null || echo "  (gautschi ssh unavailable)"
echo "=== reference ===  parent -40,162 | O22 ep100 -39,775 | O24b best ckpts -39.2/-39.4K"
