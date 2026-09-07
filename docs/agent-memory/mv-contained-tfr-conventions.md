---
name: mv-contained-tfr-conventions
description: "Conventions for the new mV with_contained_var 3srb dataset and its baked-noise contained TFRs (cut, sigma, time indices, naming)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
---

The new mV dataset is `preamp_response_with_contained_var` (pixel data now in **mV**, not electrons). On Purdue it lands at `/depot/cms/private/users/kuang14/Smart_Pixel/dataset_s_series/preamp_response/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps_parquets_contained/shuffled_3d`. The working copy lives at `/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/contained/{train,test}` (parts 0-79 / 80-99, 80/20).

**Containment cut (creator's authoritative definition):** contained = `original_atEdge == False`. The bool `original_atEdge` (True = at edge = NOT contained) is **bit-for-bit identical to `rawElectronChargeOriginal_atEdge == 0`** (~47.1% of rows). Do NOT use `rawElectronChargeOriginal_atEdge < 50` — that's the loose legacy port of the old stock `chargeOriginal_atEdge < 50` and admits ~1.8% extra near-edge clusters. The stock `OptimizedDataGenerator_v3` `select_contained` hardcodes the missing column `chargeOriginal_atEdge`, so it CRASHES on this dataset — containment must be applied manually.

**Noise:** Gaussian N(0, **4.64 mV**) — 80e (old charge-units noise) translates to 4.64 mV in the mV dataset. **Baked into the TFRs** (not load-time). Stock generator only adds noise at load-time (`__getitem__`, `noise=(mu,sigma)`); baking requires subclassing and adding noise in `prepare_batch_data` before `serialize_example` (creator write loop is sequential, so subclassing is safe).

**Time indices [11, 26]** (2-timeslice, `time_stamps_override=[11,26]`), batch 5000 — same standing conventions as [[new-3srb-dataset-conventions]].

**Gen script:** `scripts/distillation/make_tfrs_contained_2_5_noise.py` (pre-filters to contained + slim [11,26] columns into scratch, then noise-baking subclass via monkeypatched `prepare_tfrecords.OptimizedDataGenerator`). Output mirrors the colleague's (uid 2076095) naming: `<shuffled_3d>/TFR_files_2_5_noise_corr_contained/{TFR_train,TFR_test}`. The user works the 2&5ns slices; colleague works 1&6ns (`TFR_files_1_6_noise_corr_contained`, unreadable to das214). Verified 2026-06-29: train 31 tfr/151300 evt, test 8 tfr/37825 evt, baked-noise std 4.637. See [[ask-before-launching-runs]].
