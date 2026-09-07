---
name: smartpixels-repo-layout
description: Layout of the two smart-pixels-ml clones under /work/users/das214/SmartPixels and how they differ
metadata: 
  node_type: memory
  type: project
  originSessionId: bab40003-6e1e-4266-ad76-90edb4e81a2c
---

Under `/work/users/das214/SmartPixels/`:
- `smart-pixels-ml/` — davidgjiang/smart-pixels-ml, branch `2bit_optimization`. The NEWER version. **Notebook-centric** (no package): entry `two_bit_optimization.ipynb`; all logic in `two_bit_optimization_helpers/`. New features: 2-bit input quantization (`SoftQuantizeLayer`, `AnnealingScheduler`), SLIM/FULL/MAX multi-output regression heads, hls4ml/Catapult codesign (`codesign_catapult.ipynb`).
- `smart-pixels-ml/legacy/` — ArghyaRanjanDas/smart-pixels-ml (the user's own fork), branch `CleanMain`. The version the user ran before. **Package-based**: `smart_pixels_ml/` with `train_model.py`/`train_loop.py` (CLI + Submitit/SLURM), `src/`, `tests/`, `docs/`. Pins in `requirements.txt`: TF 2.15.1, QKeras 0.9.0, tf-probability 0.23.

Pipeline (new): Part 1 trains a soft-quantize layer to find optimal charge thresholds; Part 2 trains the 2-bit-digitized model; Part 3 evaluates on an independent test set. Loss: NLL of multivariate Gaussian (`custom_loss` Max / `custom_diag_loss` Full / `custom_sse_loss` Slim).

Decided to migrate by running the new notebook as-is (not porting into legacy). See [[smartpixels-data-locations]] and [[smartpixels-fresh-env]].

The legacy Vision Transformer (defined inline in legacy `train_loop.py` as `create_vit_model` + `PatchExtractor`/`PatchEncoder`/`transformer_encoder`) was ported into the new repo at `two_bit_optimization_helpers/models/transformer_model_nonquantized.py` and registered in `train.py`'s `model_list` as `ViT_Max`/`ViT_Full`/`ViT_Slim` (each with a `_SoftQuantizer` variant). Defaults baked for 16x16 input: patch_size (3,4), embed_dim 64, num_heads 4, ff_dim 128, num_layers 4. Verified building + Part-1 training. No QKeras-quantized ViT exists (legacy never had one).
