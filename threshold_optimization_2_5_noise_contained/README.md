# Threshold Optimization — 2_5 mV baked-noise CONTAINED dataset

Persistent archive so we never re-train this result. Part-1 = ViT_Max +
SoftQuantizeLayer learning the optimal 2-bit ADC thresholds. Unlike the earlier
3srb study (many short runs + median), this is **one long 5000-epoch convergence
run** on the new **mV dataset with σ=4.64 mV noise baked into the TFRs**, contained
clusters only (`original_atEdge == False`), time samples [11,26] ("2 and 5").

## Recommended thresholds (levels [0,1,2,3], offset=0)

**[12.09, 22.72, 52.73]**  — best checkpoint (epoch 3914, best val_loss −28,004/batch).
Converged stable to ±0.1–0.2 over the final 1500 epochs.

## Contents
- `SUMMARY.json` — all key numbers (config, result, convergence ranges, comparison).
- `data/threshold_loss_epochs.csv` — per-epoch loss / val_loss / T0 / T1 / T2 (all 5000 epochs).
- `best_models/best_seed42.weights.hdf5` (+ config json) — best checkpoint, so inference never needs re-training.
- `slides/slides.md` (+ `slides/figs/`) — the study slide deck + convergence figures.
- `scripts/` — the exact scripts used: TFR generation (`make_tfrs_contained_2_5_noise.py`),
  the Part-1 long run + reseed wrapper (`run_part1_long_2_5_noise_contained.py`), plotting (`make_plots_2_5_noise_contained.py`).

## Key differences vs the clean 3srb archive (`../threshold_optimization_3srb/`)
- **Data is mV with σ=4.64 mV noise BAKED IN** (loaded with `noise=-1`); the clean study had no noise.
- **Contained = `original_atEdge==False`** (creator's definition; == zero edge charge).
- Thresholds came out **higher** than the clean [4.15, 13.01, 47.24] — the noise floor pushes
  T0 up from ~4 to ~12. Expected and intended (robustness to the baked noise).
- One long convergence run (seed 42) rather than a 20-run median.

## Reproduce
```
# 1) TFRs (already built): scripts/make_tfrs_contained_2_5_noise.py
# 2) Part-1 long run (reseed wrapper picks first escaping seed -> 5000 ep):
python scripts/run_part1_long_2_5_noise_contained.py --epochs 5000
# 3) figures:
python scripts/make_plots_2_5_noise_contained.py
```
