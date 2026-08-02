#!/bin/bash
# Clean-contained control orchestrator: TFR build wait -> verify -> launch -> report.
# Answers: is the -28K plateau a NOISE ceiling? (same contained data, sigma=0)
set -u
R=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
D=$R/runs/part1_long_2_5_clean_contained
S=$D/orchestrator.log
O=/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/TFR_files_2_5_clean_contained
mkdir -p "$D"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$S"; }

# only a REAL env-python driver counts (watcher shells that merely contain the
# pattern in their argv must not — bug seen live 2026-07-11 on the simplerouter chain)
driver_running() {
  for pid in $(pgrep -f 'run_part1_long_2_5_clean_contained.py --epochs' 2>/dev/null); do
    ps -o args= -p "$pid" 2>/dev/null | grep -q "bin/python" && return 0
  done
  return 1
}
builder_running() {
  for pid in $(pgrep -f 'make_tfrs_contained_2_5_clean.py' 2>/dev/null); do
    ps -o args= -p "$pid" 2>/dev/null | grep -q "bin/python" && return 0
  done
  return 1
}

log "=== 2_5 clean-contained orchestrator started (pid $$) ==="
while builder_running; do sleep 60; done
sleep 5
log "TFR build exited -- verifying"

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
    assert m['recon_cols'][0] == '2816' and m['recon_cols'][-1] == '6911', "recon cols not [11,26]"
    tot = 0
    for r in glob.glob(f"{O}/{split}/*.tfrecord"):
        X, _ = next(iter(tf.data.TFRecordDataset(r).map(parse))); tot += int(X.shape[0])
    assert tot > 0, f"{split} empty"
    print(f"VERIFY {split}: rows={tot}, shape=(2,16,16), cols=[11,26]")
print("VERIFY_OK")
EOF
tail -4 "$S" | grep -q VERIFY_OK || { log "ABORT: verification FAILED -- training NOT launched"; exit 1; }
log "verification passed"

if driver_running; then
  log "training already running -- not double-launching"
else
  setsid nohup $PY $R/scripts/distillation/run_part1_long_2_5_clean_contained.py --epochs 5000 --target 3 \
      > "$D/train_stdout.log" 2>&1 < /dev/null &
  sleep 10
  log "TRAINING LAUNCHED (5000 ep x 3 converged seeds) pid=$(pgrep -f 'run_part1_long_2_5_clean_contained.py --epochs' | head -1)"
fi

sleep 30
while driver_running; do sleep 300; done
log "training finished -- final report pass"
CUDA_VISIBLE_DEVICES='' $PY $R/scripts/distillation/make_2_5_clean_study.py >> "$S" 2>&1
log "=== ALL DONE -- report: $D/REPORT.md ==="
