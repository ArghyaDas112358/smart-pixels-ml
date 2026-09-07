---
name: threshold-optimization-3srb-archive
description: "Where the new-3srb Part-1 threshold-optimization results are archived, and the recommended thresholds"
metadata: 
  node_type: memory
  type: project
  originSessionId: bab40003-6e1e-4266-ad76-90edb4e81a2c
---

Part-1 (ViT_Max + SoftQuantize) 2-bit threshold optimization for the new 3srb dataset (`dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d`, 2 time slices = samples [11,26]) is DONE and **persisted** so we never re-train:

**Archive (stable, world-readable, shareable):** `/work/projects/SmartPixML/threshold_optimization_3srb/`
- `SUMMARY.json` — all numbers; `README.md`; `data/offset{0,10}_threshold_runs.jsonl` (raw per-run thresholds); `best_models/` (best checkpoints + config, so inference needs no re-train); `slides/slides.md` + `slides/figs/`; `scripts/`.

**Key config (the units changed — NOT mV/e-, ~24x smaller than old set, max ~630):** random-threshold window **[25,160]**, **NO noise** (noise=-1; old [0,80] swamped the signal), offset floor as a knob, 300 epochs/run, cosine k-anneal 1->67. Old [80,2000] window + [0,80] noise caused the stuck-at-+103,616-ceiling init. Method: many random-init runs, **median** across runs (robust to bad seeds).

**Results (median thresholds, levels [0,1,2,3]):**
- offset=10: [10.53, 21.96, 56.60] (10 runs) -- T0 pinned at the floor; best NLL -5.65/evt.
- **offset=0 (300-ep archive): [4.15, 13.01, 47.24]** (20 runs) -- T0 freed to ~4; best NLL -6.33/evt (seed 4042).

**CONVERGENCE CORRECTION (2026-07-08):** the [4.15,13.01,47.24] archive is **300 ep = UNDER-CONVERGED**. The same no-noise [11,26] config run to **5000 ep** (`runs/part1_long_5k{,_fixed20,_fixed40}/`, summarized in `runs/part1_long_5k_study/REPORT_v2.md`) converges MUCH lower: **median [0.40, 7.00, 30.45]**, best NLL/batch **-52,433** (fixed20 seed 3042). T0 collapses to ~0.4 (no noise floor). Use the 5000-ep numbers for any like-for-like comparison with the 5000-ep noise studies [[threshold-opt-2_5-noise-contained]] [[threshold-opt-1-6-iid]]. Note this no-noise study is `select_contained=False` (FULL set); the noise studies are contained. Training-history + epoch-vs-threshold figs: `runs/part1_long_5k_study/figs/{01_nll_vs_epoch,04_epoch_vs_threshold_by_index}.png`.

**PERFORMANCE FINDING (best-seed from_weights plots, `runs/perf_plots_talk/`, 2026-07-08):** position resolves in ALL cases (σx~6.4-7.0 µm, σy~1.9-2.1 µm, pulls~1.0), but **angle only resolves WITHOUT noise** (no-noise σα~2.8°, σβ~1.5°, pulls~1.0). Both noise cases (2_5 seed4042, 1_6 seed1042) show α/β collapsing to the mean (summary diagonal; cotα/β pulls ~1.37-1.39) — position survives the 2-bit+noise readout, angle info largely does not. Best-seed by NLL: 2_5+noise=4042, 1_6+noise=1042. Plot driver: `scripts/distillation/make_perf_plots_talk.py`.

**Performance (test set, best offset=0 vs best offset=10):** offset=0 better resolution on all 4 params -- x 10.78 vs 11.28um, y 2.99 vs 3.23um, cotA 3.80 vs 4.90, cotB 0.82 vs 1.14; both well-calibrated (pulls ~0.94-1.05). Both best-model checkpoints are archived (`best_models/offset{0,10}_best_*.weights.hdf5`), so the comparison never needs re-training.

The retry/reseed + AbortOnStuck machinery (from `legacy/smart_pixels_ml/train_loop.py`, with AbortOnStuck actually wired into the callbacks) lives in `scripts/distillation/run_part1_new_dataset.py`; copy for sharing at `/work/projects/SmartPixML/train_loop_part1_3srb.py`. NOTE: don't `rm -rf` a run's `weights/` dir before archiving its best checkpoint (that mistake forced the offset=10 re-run). See [[new-3srb-dataset-conventions]], [[ask-before-launching-runs]].
