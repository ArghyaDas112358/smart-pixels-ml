---
name: smartpixels-fresh-env
description: "The fresh conda env for running the smart-pixels-ml 2bit_optimization notebook, and how to launch it"
metadata: 
  node_type: memory
  type: project
  originSessionId: bab40003-6e1e-4266-ad76-90edb4e81a2c
---

Fresh env for the new notebook lives at `/work/users/das214/envs/smartpix-2bit` (conda/mamba prefix, Python 3.10). Built from `/work/users/das214/SmartPixels/smart-pixels-ml/requirements.txt` (which was created since the new repo shipped none).

Versions (must stay pinned — TF 2.15 uses Keras 2, which QKeras 0.9 needs; TF 2.16+ Keras 3 breaks QKeras): tensorflow[and-cuda]==2.15.1, tensorflow-probability==0.23.0, qkeras==0.9.0, numpy 1.23.5, pandas 1.5.3, pyarrow 17, matplotlib 3.9.2, seaborn 0.13.2 + natsort/tqdm/pyyaml/jupyter/ipykernel.

Run the notebook in this env, e.g. register a kernel:
`/work/users/das214/envs/smartpix-2bit/bin/python -m ipykernel install --user --name smartpix-2bit`

Keep all outputs on `/work` (22T free); `/home/das214` only has ~3.7G free. The notebook writes weights/parquets under `/work/users/das214/SmartPixels/smart-pixels-ml/runs/` (inside the repo, gitignored). Plain-python conversions of the notebooks live in `smart-pixels-ml/scripts/` (two_bit_optimization.py = main pipeline; training_tracker.py + comparison_plots.py = analysis; codesign left as notebook only). Smoke test is `smart-pixels-ml/smoke_test.py`.

Node has an A100-40GB GPU but `nvidia-smi` mem query returns "Insufficient Permissions" in this sandbox — smoke test was run on CPU (`CUDA_VISIBLE_DEVICES=""`). Full 1000+1000-epoch training should run on a GPU node.

Smoke test: `/work/users/das214/SmartPixels/smoke_test.py` (+ `runs/smoke_dataset/` symlinks) ran Part 1 + Part 2 for 2 epochs each and PASSED (Part 1 val_loss 1840->1280, thresholds optimized). See [[smartpixels-repo-layout]], [[smartpixels-data-locations]].
