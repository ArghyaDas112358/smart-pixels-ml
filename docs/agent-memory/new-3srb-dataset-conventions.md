---
name: new-3srb-dataset-conventions
description: "User-specified conventions for the new 3srb 10ps dataset (batch 5000, time samples [11,26], 80/20 split)"
metadata: 
  node_type: memory
  type: project
  originSessionId: bab40003-6e1e-4266-ad76-90edb4e81a2c
---

New dataset (2026-06-04): `/depot/cms/users/das214/datasets/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d`. 100 pre-shuffled `part.N.parquet` files, 4000 rows each (~400K events), 16x16 pixels x **101 time samples** at 10ps (25,856 pixel cols), convolved to a 200ps response. Labels include x/y-midplane, cotAlpha, cotBeta (+ a new `pt` column).

**User-specified conventions (do NOT guess differently):**
- TFR batch size: **5000** (user explicitly rejected my 2500 OOM workaround).
- 2-timeslice selection: **time samples [11, 26]** (i.e. 110ps and 260ps), NOT the old [0,19] convention and NOT first/last [0,100].
- Split: parts 0-79 -> `train/`, 80-99 -> `test/` (done 2026-06-04).

Pipeline support: `generate_tfrecords(..., time_stamps_override=[11,26])` (added param in `prepare_tfrecords.py`). Gen script: `scripts/distillation/make_tfrs_new_dataset.py`; Part-1 thresholds: `scripts/distillation/run_part1_new_dataset.py`. Known issue: ViT Part-1 TRAINING at batch 5000 OOMed once on the shared GPU (other users' memory) -- if it recurs, stop and ask the user instead of rebatching. See [[ask-before-launching-runs]], [[smartpixels-data-locations]].
