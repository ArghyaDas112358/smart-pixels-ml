#!/bin/bash
# One-time environment build for the SmartPixels router runs on Purdue Gautschi.
#   bash /depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/scripts/gautschi/setup_env_gautschi.sh
#
# Uses a plain venv on the module python, NOT conda. Two conda attempts died with
# `OSError: [Errno 5] Input/output error` during "Verifying transaction" on the
# Lustre scratch filesystem (2026-07-30) -- conda's hardlink-farm install pattern
# does not agree with Lustre, and building it on the Depot NFS mount instead
# saturated that mount so badly every /depot stat on the login node hung. A venv
# writes far fewer files and installs cleanly.
#
# Python is 3.11 here vs 3.10 on Purdue AF (3.11 is the available module); TF
# 2.15.1 supports both and every other version is pinned to the AF env exactly.
set -euo pipefail

SCRATCH=/scratch/gautschi/das214
ENV_PREFIX=$SCRATCH/envs/smartpix-2bit

echo "=== modules ==="
# python/3.11.9 lives under modtree/cpu (it is NOT in modtree/gpu). That is fine:
# tensorflow[and-cuda] ships its own CUDA 12 libraries as pip wheels, so no CUDA
# module is needed and the same env runs on the L40 and H100 partitions.
module --ignore_cache load modtree/cpu
module --ignore_cache load python/3.11.9
python3 --version

echo "=== creating venv at $ENV_PREFIX ==="
rm -rf "$ENV_PREFIX"
python3 -m venv "$ENV_PREFIX"
# shellcheck disable=SC1091
source "$ENV_PREFIX/bin/activate"
# Do NOT `pip install --upgrade pip` here: on Lustre that produced a partially
# written pip (`ModuleNotFoundError: pip._vendor.rich.align`) that ensurepip then
# refused to repair ("already satisfied"). Bootstrapping a clean pip up front with
# get-pip.py avoids the whole failure mode.
curl -sS https://bootstrap.pypa.io/pip/get-pip.py -o /tmp/get-pip-$$.py
python /tmp/get-pip-$$.py --no-cache-dir
rm -f /tmp/get-pip-$$.py
python -m pip install --no-cache-dir setuptools wheel

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
  "QKeras==0.9.0" \
  "natsort==8.4.0" \
  "tqdm==4.67.3" \
  "pyarrow==17.0.0"

echo "=== verify (CPU-side; the GPU is checked inside the job) ==="
python - <<'PY'
import tensorflow as tf, numpy as np, sys
print("TF", tf.__version__, "| keras", tf.keras.__version__, "| numpy", np.__version__)
sys.path.insert(0, "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/two_bit_optimization_helpers")
from train import create_model
m = create_model('ViT_Max_SimpleRouterBeta', timeslices=101, soft_quantize_layer=True,
                 initial_thresholds=[30., 60., 120.], threshold_offset=0.0,
                 initial_levels=np.array([0., 1., 2., 3.], np.float32))
r = m.get_layer('simple_router_output')
print("router beta weight present:", any('log_k' in w.name for w in r.weights))
print("params:", m.count_params())
print("ENV_OK")
PY

echo
echo "DONE -> $ENV_PREFIX"
