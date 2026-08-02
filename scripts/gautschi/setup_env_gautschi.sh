#!/bin/bash
# One-time environment build for the SmartPixels router runs on Purdue Gautschi.
# Run this ON A GAUTSCHI LOGIN NODE (it needs no GPU):
#     bash /depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/scripts/gautschi/setup_env_gautschi.sh
#
# Versions are pinned to EXACTLY what the Purdue-AF env runs, so results are
# comparable run-for-run. tensorflow[and-cuda] pulls the CUDA 12.2 wheels, which
# is what makes TF 2.15 work on H100 (sm90) without a system CUDA module.
set -euo pipefail

DEPOT=/depot/cms/users/das214/SmartPixels_gautschi
ENV_PREFIX=$DEPOT/envs/smartpix-2bit

echo "=== module setup ==="
module purge
module load conda || module load anaconda    # RCAC names it one of these

echo "=== creating env at $ENV_PREFIX ==="
conda create -y -p "$ENV_PREFIX" python=3.10
# shellcheck disable=SC1091
source activate "$ENV_PREFIX" 2>/dev/null || conda activate "$ENV_PREFIX"

echo "=== pinned deps (mirror of the AF env) ==="
pip install --no-cache-dir \
  "tensorflow[and-cuda]==2.15.1" \
  "tensorflow-probability==0.23.0" \
  "numpy==1.23.5" \
  "pandas==1.5.3" \
  "scipy==1.15.3" \
  "h5py==3.16.0" \
  "matplotlib==3.9.2" \
  "seaborn==0.13.2" \
  "QKeras==0.9.0"

echo "=== verify: TF sees the H100 and the SIMPLE router builds ==="
python - <<'PY'
import tensorflow as tf, numpy as np, sys, os
print("TF", tf.__version__, "| keras", tf.keras.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if not gpus:
    print("WARNING: no GPU visible -- on a login node this is expected; "
          "re-check inside an sbatch/salloc job before trusting it.")
sys.path.insert(0, os.path.join(os.environ["DEPOT"], "smart-pixels-ml",
                                "two_bit_optimization_helpers")) if "DEPOT" in os.environ else None
PY

echo
echo "DONE. Next:"
echo "  1) confirm your allocation + partition:   slist ; sinfo -s"
echo "  2) edit the -A/-p lines in scripts/gautschi/submit_o4.sbatch"
echo "  3) sbatch scripts/gautschi/submit_o4.sbatch"
