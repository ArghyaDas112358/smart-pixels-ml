---
name: smartpixels-data-locations
description: Where the smart-pixels training datasets live on disk and which one the new 2bit notebook uses
metadata: 
  node_type: memory
  type: project
  originSessionId: bab40003-6e1e-4266-ad76-90edb4e81a2c
---

Pixel-detector parquet datasets live under `/depot/cms/users/das214/datasets/`.

The new `two_bit_optimization.ipynb` is wired to:
`/depot/cms/users/das214/datasets/largerWindowPreliminary/dataset_3sr_16x16_50x12P5_centeredIncidence_parquets`
— 16x16 array, centeredIncidence, with `train/` (80 parquet files) + `test/` (20) + `TFR_files/`. This is the only 16x16 centeredIncidence set present. Note the new repo's default config spelled it `dataset_3src` (with a c); the user's actual dir is `dataset_3sr` (no c).

There is **no** matching `dataset_2sc` independent test set on disk. In the user's legacy 16x16 runs the "test set" was simply the `test/` split of the same 3sr dataset (legacy `train_loop.py` loaded train from `train/`, val/test from `test/`). So Part 3 of the new notebook is pointed at the same 3sr dataset, reusing its `test/` split.

Important: the data has `train/` + `test/`, NOT `train_contained/` + `test_contained/`. The new `prepare_tfrecords.generate_tfrecords` builds the subdir name from `select_contained` (True -> `*_contained/`), so the notebook config was set to `select_contained=False`.

Other datasets present (not used by the new notebook): `dataset_3sr_16x16_50x12P5_parquets` (non-centered, 18G), `dataset_2s_50x12P5_parquets` / `_50x10_parquets` (not 16x16), `dataset8/unflipped` (13x21, 85G), large `dataset2s/` and `dataset3s/` trees. See [[smartpixels-repo-layout]].
