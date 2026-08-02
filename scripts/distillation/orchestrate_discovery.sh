#!/bin/bash
# Self-contained pipeline orchestrator — survives SSH/VS Code/Claude death.
#   ① wait for the all-101 (80+20) TFR build to finish
#   ② VERIFY it (row-exact vs manifest, pinned labels_scale, 80 train files)
#   ③ 3-epoch GPU sanity (measured s/epoch + peak memory)
#   ④ launch discovery: 1000 epochs x 4 converged seeds (detached, resumable)
#   ⑤ when discovery ends: auto-generate the study report
# Every step logs to runs/softrouter_discovery/orchestrator.log; any failure
# aborts BEFORE the GPU launch and says why.
set -u
R=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
L=$R/runs/tfr_all101_discovery
D=$R/runs/softrouter_discovery
S=$D/orchestrator.log
O=/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/TFR_files_all101_noise_contained_discovery
mkdir -p "$D"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$S"; }

log "=== orchestrator started (pid $$) ==="

# ---- ① wait for the TFR build process to exit --------------------------------
while pgrep -f make_tfrs_all101.py >/dev/null; do sleep 30; done
sleep 5
log "TFR build process exited -- verifying output"

# ---- ② decisive verification (this, not the SUCCESS marker, is the gate) -----
CUDA_VISIBLE_DEVICES='' $PY - "$O" >> "$S" 2>&1 <<'EOF'
import sys, json, glob
import tensorflow as tf
O = sys.argv[1]
man = json.load(open(f"{O}/MANIFEST.json"))
assert len(man["subset"]["train_files"]) == 80, "manifest is not the FULL 80-file build"
assert len(man["subset"]["test_files"]) == 20, "manifest is not the FULL 20-file build"
exp = {"TFR_train": man["subset"]["train_rows"], "TFR_test": man["subset"]["test_rows"]}
def parse(ex):
    f = tf.io.parse_single_example(ex, {'X': tf.io.FixedLenFeature([], tf.string),
                                        'y': tf.io.FixedLenFeature([], tf.string)})
    return tf.io.parse_tensor(f['X'], tf.float32), tf.io.parse_tensor(f['y'], tf.float32)
for split, expected in exp.items():
    tot = 0
    for r in sorted(glob.glob(f"{O}/{split}/*.tfrecord")):
        X, _ = next(iter(tf.data.TFRecordDataset(r).map(parse))); tot += int(X.shape[0])
    m = json.load(open(f"{O}/{split}/metadata.json"))
    assert tot == expected, f"{split}: rows {tot} != manifest {expected}"
    assert [round(v, 4) for v in m['labels_scale']] == [123.7301, 30.9423, 6.5879, 1.8481], "labels_scale not pinned"
    print(f"VERIFY {split}: rows={tot} exact, labels_scale pinned")
print("VERIFY_OK")
EOF
tail -5 "$S" | grep -q VERIFY_OK || { log "ABORT: verification FAILED -- discovery NOT launched"; exit 1; }
log "verification passed (row-exact, 80+20, pinned scale)"

# ---- ③ measured GPU sanity ----------------------------------------------------
rm -rf "$D/sanity"
$PY $R/scripts/distillation/run_softrouter_discovery.py --sanity >> "$S" 2>&1
grep -q "SANITY done" "$D/sanity/discovery.log" 2>/dev/null \
  || { log "ABORT: GPU sanity FAILED -- discovery NOT launched"; exit 1; }
log "sanity: $(grep 'SANITY' "$D/sanity/discovery.log" | tail -2 | tr '\n' ' | ')"

# ---- ④ launch discovery (idempotent) ------------------------------------------
if pgrep -f 'run_softrouter_discovery.py --epochs' >/dev/null; then
  log "discovery already running -- not launching a second copy"
else
  setsid nohup $PY $R/scripts/distillation/run_softrouter_discovery.py --epochs 1000 --target 4 \
      > "$D/discovery_stdout.log" 2>&1 < /dev/null &
  sleep 10
  log "DISCOVERY LAUNCHED (1000 ep x 4 converged seeds) pid=$(pgrep -f 'run_softrouter_discovery.py --epochs' | head -1)"
fi

# ---- ⑤ report when it all ends -------------------------------------------------
sleep 30
while pgrep -f 'run_softrouter_discovery.py --epochs' >/dev/null; do sleep 300; done
log "discovery finished -- generating study report"
CUDA_VISIBLE_DEVICES='' $PY $R/scripts/distillation/make_softrouter_study.py >> "$S" 2>&1
log "=== ALL DONE -- report: $D/REPORT.md ==="
