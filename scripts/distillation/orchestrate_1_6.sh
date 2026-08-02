#!/bin/bash
# Self-contained 1_6 threshold-study orchestrator — survives SSH/VS Code/Claude death.
#   ① wait for the 1_6 all-data TFR build to finish
#   ② VERIFY it (rows>0, shape (2,16,16), recon cols = [6,31] block)
#   ③ launch threshold training: 5000 ep x 5 converged seeds (detached, resumable)
#   ④ report auto-regenerates per converged seed (run script calls make_1_6_study.py)
set -u
R=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
D=$R/runs/part1_long_1_6_iid_contained
S=$D/orchestrator.log
O=/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/TFR_files_1_6_iid_contained
mkdir -p "$D"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$S"; }

log "=== 1_6 orchestrator started (pid $$) ==="

# ① wait for the TFR build to exit
while pgrep -f make_tfrs_contained_1_6_noise.py >/dev/null; do sleep 30; done
sleep 5
log "TFR build exited -- verifying"

# ② verify (gate before GPU)
CUDA_VISIBLE_DEVICES='' $PY - "$O" >> "$S" 2>&1 <<'EOF'
import sys, json, glob
import tensorflow as tf
O = sys.argv[1]
def parse(ex):
    f = tf.io.parse_single_example(ex, {'X': tf.io.FixedLenFeature([], tf.string),
                                        'y': tf.io.FixedLenFeature([], tf.string)})
    return tf.io.parse_tensor(f['X'], tf.float32), tf.io.parse_tensor(f['y'], tf.float32)
for split in ("TFR_train", "TFR_test"):
    m = json.load(open(f"{O}/{split}/metadata.json"))
    assert list(m['input_shape']) == [2, 16, 16], f"{split} shape {m['input_shape']}"
    assert m['recon_cols'][0] == '1536' and m['recon_cols'][-1] == '8191', "recon cols not [6,31]"
    tot = 0
    for r in glob.glob(f"{O}/{split}/*.tfrecord"):
        X, _ = next(iter(tf.data.TFRecordDataset(r).map(parse))); tot += int(X.shape[0])
    assert tot > 0, f"{split} empty"
    print(f"VERIFY {split}: rows={tot}, shape=(2,16,16), cols=[6,31]")
print("VERIFY_OK")
EOF
tail -4 "$S" | grep -q VERIFY_OK || { log "ABORT: verification FAILED -- training NOT launched"; exit 1; }
log "verification passed"

# ③ launch training (idempotent)
if pgrep -f 'run_part1_long_1_6_iid_contained.py --epochs' >/dev/null; then
  log "training already running -- not double-launching"
else
  setsid nohup $PY $R/scripts/distillation/run_part1_long_1_6_iid_contained.py --epochs 5000 --target 5 \
      > "$D/train_stdout.log" 2>&1 < /dev/null &
  sleep 10
  log "TRAINING LAUNCHED (5000 ep x 5 seeds) pid=$(pgrep -f 'run_part1_long_1_6_iid_contained.py --epochs' | head -1)"
fi

# ④ wait for completion, then a final report pass
sleep 30
while pgrep -f 'run_part1_long_1_6_iid_contained.py --epochs' >/dev/null; do sleep 300; done
log "training finished -- final report pass"
CUDA_VISIBLE_DEVICES='' $PY $R/scripts/distillation/make_1_6_study.py >> "$S" 2>&1
log "=== ALL DONE -- report: $D/REPORT.md ==="
