---
name: threshold-opt-2_5-noise-contained
description: Part-1 optimal 2-bit thresholds for the new mV baked-noise CONTAINED 2_5 dataset (5000-epoch convergence run)
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a0a041e-b87d-47e5-ae22-8b5c100f4193
---

Part-1 (ViT_Max + SoftQuantize) 2-bit threshold optimization for the **new mV
baked-noise contained 2_5 dataset** is DONE (2026-06-30, 5000-epoch convergence run).

**Dataset:** `/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d/TFR_files_2_5_noise_corr_contained/{TFR_train,TFR_test}`
- time samples [11,26], contained = `original_atEdge==False` (== rawElectronChargeOriginal_atEdge==0, ~47%), N(0, 4.64 mV) noise **baked into the TFRs** (load with noise=-1). Built by `scripts/distillation/make_tfrs_contained_2_5_noise.py`. Data scale max ~644 (matches prior new-3srb set). See [[new-3srb-dataset-conventions]].

**Optimal thresholds (levels [0,1,2,3], offset=0): [12.09, 22.72, 52.73]**
- best val_loss -28004 @ epoch 3914; converged stable to +-0.1..0.2 over the last 1500 epochs.
- vs the archived CLEAN-data thresholds **[4.15, 13.01, 47.24]** ([[threshold-optimization-3srb-archive]]), all three shifted UP (T0 4->12, T1 13->23, T2 47->53) — the expected effect of the 4.64 mV baked noise raising thresholds off the noise floor for robustness.

**Archive (full report, like the clean-data one):** `/work/users/das214/SmartPixels/smart-pixels-ml/threshold_optimization_2_5_noise_contained/` (kept in the repo, NOT /work/projects — see [[reports-stay-in-repo]]) — README.md, SUMMARY.json, data/threshold_loss_epochs.csv, best_models/best_seed42.weights.hdf5 (+config), slides/slides.md + figs/{loss_vs_epoch,thresholds_vs_epoch,threshold_stability_zoom}.png, scripts/.

**Run:** `runs/part1_long_2_5_noise_contained/seed_42/` — `best.weights.hdf5` (ep 3914), `threshold_loss_epochs.csv` (per-epoch loss/val_loss/T0/T1/T2), `WINNER.json`. Single long run, seed 42, Nadam(1e-3), cosine k-anneal 1->67 over 5000 ep.

**Gotcha fixed this session:** stock `AbortOnStuck` used an absolute thr=1e5, but THIS dataset's stuck zero-likelihood ceiling is ~98,886 (just *under* 1e5), so a stuck seed froze for 379 epochs undetected. Fix in `run_part1_long_2_5_noise_contained.py`: thr=**1e4** + a flat-line (no-improvement-for-15-epochs) detector, wrapped in a reseed loop (abort stuck seed in ~15 ep, run first escaping seed to 5000). Seed 42 escaped first try. See [[ask-before-launching-runs]].
