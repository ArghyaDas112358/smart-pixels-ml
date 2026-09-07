---
name: smartpixels-distillation-findings
description: "Why ViT->tiny-MoE soft-KL distillation keeps losing to standalone, and the TBR/DeiT direction that replaced it"
metadata: 
  node_type: memory
  type: project
  originSessionId: bab40003-6e1e-4266-ad76-90edb4e81a2c
---

Distilling the ViT_Max teacher (418,702 params, val -8.92 NLL/event = -43,762/batch) into the tiny QMlp_MoE_Max student (4,780 params, 8-bit QKeras) for the 2-bit smart-pixels regression.

**The verified result:** every SOFT distribution-matching recipe loses to just training the student standalone (best standalone QMlp_MoE_Max = -30,526/batch = -6.105/event, run `runs/qmlp_moe_max_run_full`). Losers, all reproduced and measured apples-to-apples: forward-KL (MDMM and fixed-beta 0.3/1/3), reverse-KL, distribution-space TAID (-26,666 best distilled), always-on means-only MSE (-25,605), confidence-weighting. Two real bugs were found+fixed first (KL diag floor 0.02 blurred teacher 43%; MDMM lambda runaway to 191); the failure SURVIVED the fixes, so it is paradigm-level, not a bug.

**Why (multi-agent research + numeric audit, taskId wpezm2g6m):** forward Gaussian-KL precision-weights the mean error by the teacher's INVERSE covariance; a sharp teacher makes that gradient pathological and it fights the data-NLL. Matching the teacher's 4x4 covariance also wastes a tiny student's capacity on 10 numbers it cannot represent. Refuted the pessimistic framing: the teacher scores -8.92/event AND is well-calibrated on the EXACT hard {0,1,2,3} 2-bit input, so the input is NOT the ceiling and there is ~2.8 NLL/event of genuine transferable signal. The field (smart-pixels papers arXiv:2602.15946, 2312.11676) trains these tiny models DIRECTLY with QAT and does not distill the deployed model; it uses the big transformer only to learn the 2-bit ADC thresholds.

**New direction (means-only, fixed weights, NO MDMM, NO covariance matching):**
- E1 = Teacher-Bounded Regression: `distill14_tbr.py` / `scripts/distillation/train_distill_tbr.py`. Transfer teacher means only, masked to events where teacher beats student (stop-grad hinge). Structurally cannot lose to standalone.
- E2 = DeiT separate head: model `QMlp_MoE_Max_Distill` (18 outputs, +68 params) in `models/mlp_encoder_model_quantized.py`; `distill14_deit.py` / `train_distill_deit.py`. Main head fits GT via NLL, separate 4-dim head fits teacher means via MSE, deploy = average the two mean heads. Best shot at clearly beating standalone.
- Deferred: E3 float-teacher QAT self-distill (low value here because student is 8-bit, near-lossless), E4 beta-NLL (may raise the standalone bar itself).

Enabling change: `loss.py` now has `custom_loss_perevent` (returns the (B,) NLL vector); `custom_loss` just sums it (unchanged behaviour). Teacher ckpt: `runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints`. Thresholds: `runs/vit_max_run_1000ep/optimized_thresholds.json` = [226.04, 601.01, 1456.94]. See [[smartpixels-repo-layout]], [[smartpixels-fresh-env]].

**THE PIPELINE CONFOUND (found while running the fair retests):** every prior "distillation loses to standalone" result was confounded. The -30,526 standalone used Nadam, no clipnorm, AbortOnStuck+retry-on-stuck-init, and ran to epoch 420 (best). The distill driver used Adam+clipnorm=1.0 + EarlyStopping(patience 50) which stopped ~epoch 128 (where the standalone was only -26,917) and converged to a worse optimum. So distillation was judged at ~1/3 of the baseline's training. Fix: `two_bit_optimization_helpers/fair_fit.py` (`fit_with_retry`: Nadam, no clipnorm, AbortOnStuckData, auto-reseed). Verified: seed=42 gets stuck at the +103,616/batch ceiling with Nadam; the harness aborts at ep5 and reseeds to 1042 which escapes. A w=0 fair control tracks the standalone curve exactly (ep3: -14,752 vs -14,953), validating the fair pipeline.

**GOAL PIVOT (user, 2026-06-02): "we have to reach the teacher" (-8.75/event), not just beat standalone.** That needs capacity, not just a better recipe (4,780 params provably cannot represent the teacher; KL floors ~3.7). User chose a CAPACITY-SCALING SWEEP, moderate ceiling (tens of thousands of params OK). New scalable model `QMlp_MoE_Max_Scaled(n_experts, expert_hidden, enc_hidden)` + fixed sizes QMlp_MoE_M(11K)/L(31K)/XL(85K) in `models/mlp_encoder_model_quantized.py`. Recipe: forward-KL (full distribution match) becomes the RIGHT tool once the student has capacity (research: it fails only for capacity-limited students). Driver `scripts/distillation/train_distill_kl_fair.py` (Distiller14 fixed_beta + fair_fit). Sweep queue `run_capacity_sweep.sh`: M->L->XL, beta=1.0, finds smallest student that reaches -8.75.
