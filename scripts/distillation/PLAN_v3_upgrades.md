# V3 plan: physics-model capacity + distillation method upgrades

## Context

After three rounds of distillation against the extracted ViT_Max teacher
(per-event NLL = -8.75), the best students sit here:

| model | params | per-event NLL | per-batch (-43K scale) |
|---|---:|---:|---:|
| ViT_Max teacher (frozen) | 418,702 | -8.75 | -43,762 |
| Conv2D_Max standalone (float, no distill) | 1,898 | -3.01 | -15,055 |
| QConv2D_Max distilled (running, current best) | 1,869 | -1.87 (@ ep 31) | -9,364 |
| pysr_aug student (symbolic+tiny-NN) | 382 | +0.65 | +3,240 |

Two observed pathologies from the pysr_aug run:

1. **Architectural saturation.** Only 24 trainable physics-side params (theta_L
   + 4 aff_scale + 4 aff_bias + 4 pysr_alpha + 11 PySR equation constants)
   plus 354 in the error MLP. All 4 mean outputs come from a closed-form
   formula; the error MLP only produces covariance. Best val_loss_data was
   reached at epoch 55 of 106 (then plateau).
2. **Fast lock-on to teacher then plateau.** KL hits its target (~0.44) within
   ~5-10 epochs, so afterward only `loss_data` drives optimization, and
   the student's expressive ceiling is hit fast.

This plan addresses both, plus brings in distillation techniques that are
state-of-the-art for our regime (small regression student, big probabilistic
teacher, hardware constraints).

The QConv2D_Max distillation run finishes first; if it dominates pysr_aug
(very likely given the ~5x parameter advantage and conv expressivity), the
symbolic upgrades below are still worth pursuing as a separate "minimum
deploy footprint" track. The distillation upgrades benefit both tracks.

---

## Part A. Symbolic model upgrades

Goal: lift the symbolic student's per-event NLL from +0.65 toward 0.0 or
below, without exploding the parameter budget past ~1k.

### A1. Add a mean-correction head to the error MLP (Recommended, primary)

Current `error_mlp.py`:
```
input (B, 16, 16, 2)
  -> time_mean -> AvgPool x + AvgPool y -> concat (B, 32)
  -> Dense(8, tanh) -> Dense(10, linear)        # 354 params, 10 Chol entries
```

New version: same projection, wider hidden, output 14 instead of 10. Split
the 14 outputs into `[4 mean corrections, 10 Cholesky entries]`.

```
input (B, 16, 16, 2)
  -> time_mean -> AvgPool x + AvgPool y -> concat (B, 32)
  -> Dense(16, tanh) -> Dense(14, linear)       # 32*16+16 + 16*14+14 = 766 params
```

Then in `PhysicsAnsatz(variant='pysr_aug_v3')`:
```
mean_corrections = mlp_out[..., :4]
chol_entries     = mlp_out[..., 4:14]
symbolic_means   = (existing symbolic + PySR correction)
final_means      = symbolic_means + mean_corrections
```

- **Param count: ~800** (was 382).
- The physics ansatz still provides a strong inductive bias; the MLP adds
  learned residual corrections to the means.
- One-file change: `models/error_mlp.py` (widen + change output dim) and
  `models/student_max.py` (split MLP output, add to means).

### A2. Optional: per-output small MLPs on the symbolic prediction

Concat each symbolic output with a fixed small feature set (e.g. the 8
features that matter most for that output per PySR) and pass through
`Dense(8, tanh) -> Dense(1, linear)`. Adds ~400 params total (4 outputs x
100). Stack on top of A1 for ~1,200 params.

Lower priority. A1 alone should close most of the symbolic-student gap.

### A3. Optional: trainable per-pixel weights for barycenter

The current barycenter uses raw charge weights. Replace with
`q_eff = q * tf.nn.softplus(W_pixel)` where `W_pixel` is a trainable
(16, 16) weight map (256 params). Soft-init at 1.0. Adds 256 params; lets
the centroid de-emphasize edge or noisy pixels per the training data.

Defer to v4 unless A1+A2 still underperforms.

---

## Part B. Distillation method upgrades

Goal: extract more signal from the teacher beyond just matching its output
distribution. Most are layered on top of the current `Distiller` / `Distiller14`
infrastructure; none requires a redesign.

### B1. Hint / FitNet-style feature distillation (highest leverage for QConv2D_Max)

Match intermediate feature maps between teacher and student. For the
QConv2D_Max student (5 conv filters, post-pool feature) vs ViT_Max teacher
(post-attention pre-head dense, 64-dim):

```
# In Distiller14.train_step:
h_T = teacher_hidden(x)        # (B, D_T)  -- e.g. ViT's penultimate-layer features
h_S = student_hidden(x)        # (B, D_S)  -- e.g. QConv2D's post-pool flat features
h_S_proj = ProjLayer(h_S)      # learnable Dense(D_T) projection
hint_loss = || h_T - h_S_proj || ^2   # element-wise MSE

total = data_loss + lam * kl + beta * hint_loss
```

- Requires exposing the teacher and student intermediate tensors as Keras
  outputs (refactor `create_model` to return a dict with `{out, hidden}`,
  or use a Keras Functional submodel).
- Projection layer (e.g. Dense(D_T) on student's hidden) is trainable;
  adds ~64 * 5 + 64 ~ 400 params just for the projection (not deployed).
- `beta` is a new hyperparameter, init 0.1, no scheduling needed initially.
- Literature: FitNets (Romero 2014), Patient KD (Sun 2019), AT (Zagoruyko
  2016). Documented 1-3 nat improvement on hard regression in many works.

### B2. Teacher Assistant chain (TAKD) for the symbolic student

The 1,100x gap from ViT_Max -> symbolic is huge. Distill in two hops:

```
ViT_Max  -->  Conv2D_Max (assistant, 1,898 params, distilled in B1 above)
              -->  symbolic student (382 params)
```

The Conv2D_Max assistant is closer in spirit to the symbolic student
(same probabilistic Gaussian output, much smaller capacity gap). Easier
distillation target. Standard TAKD: Mirzadeh et al. 2020.

- No new code on the symbolic side; just point `--teacher-checkpoints` at
  the Conv2D_Max checkpoint instead of ViT_Max.
- Free intermediate: we *already* have a distilled QConv2D_Max from the
  current run; can use that directly as the assistant.

### B3. Uncertainty-weighted output distillation

Per-event teacher confidence `sigma_T` should drive how hard the student
tries to match: where teacher is confident, residuals matter; where teacher
is uncertain, don't waste capacity.

```
# weight = 1 / sigma_T^2, clipped to 1st-99th percentile
kl_per_event = tfp KL between Gaussians   # currently mean over batch
kl_weighted = tf.reduce_mean(weight * kl_per_event)
```

- We already infrastructure this in `pysr_residuals_physics.py` for PySR
  training; just port to the live `Distiller`/`Distiller14` train_step.
- One-line addition; no new params.

### B4. DKD-style decoupled KL (separate mean vs covariance terms)

Decompose the Gaussian KL into a "mean-matching" term and a "covariance-
matching" term, weight them independently:

```
KL[T||S] = 0.5 * (mu_T - mu_S)^T Sigma_S^-1 (mu_T - mu_S)             # mean part
         + 0.5 * (tr(Sigma_S^-1 Sigma_T) - k + log_det(Sigma_S/Sigma_T))  # cov part
```

```
total = data_loss + lam_mean * kl_mean + lam_cov * kl_cov
```

- Two MDMM dual variables, one per term. Lets us push the student to match
  means tightly (lam_mean -> infinity) while keeping covariance constraint
  soft, or vice versa.
- Inspired by Decoupled KD (Zhao 2022), adapted to Gaussian regression.
- Implementation: rewrite `gaussian_kl` to return `(kl_mean, kl_cov)`, dual
  ascent on each. ~20 lines.

### B5. Teacher temperature

Scale teacher's predicted covariance by `T > 1` before computing KL:
```
Sigma_T_eff = T^2 * Sigma_T
```

- Softer teacher = easier KL = student has more slack to fit data NLL.
- One hyperparameter (T, typical range 1.5 to 3).
- One-line change in `gaussian_kl`.

### B6. Two-phase schedule

Replace the constant-MDMM training with:
- **Phase 1** (first 30% of epochs): MDMM aggressively binding (high eta),
  student locks onto teacher distribution.
- **Phase 2** (rest): decay lam toward 0, let data NLL drive.

Documented to help with the "student plateaus once teacher is matched"
pathology we observed. Curriculum-style.

---

## Locked v3 sequence (user-approved)

Picks: **A1 only** (~800 param budget, no per-output MLPs in v1), **B2 TAKD via
current QConv2D_Max** as the assistant, **B3 uncertainty-weighted KL** baked
in, hint training (**B1**) **deferred unless QConv2D_Max plateaus below -4**.
Acceptance bar: symbolic student should reach **per-event NLL ~ -3.0**,
matching the Conv2D_Max non-quantized standalone baseline.

| step | what | expected gain | wall-clock |
|---|---|---|---|
| 1 | Wait for the running QConv2D_Max distillation to finish. Use its best checkpoint as the TAKD assistant for steps 2-4. | (assistant becomes available) | 0 (already in flight) |
| 2 | Implement **A1**: wider error MLP `Dense(16,tanh)->Dense(14)` with 4-mean-correction + 10-Cholesky split. New variant `pysr_aug_v3` in `PhysicsAnsatz`. ~800 params total student. | (code only) | ~20 min |
| 3 | Implement **B3**: per-event uncertainty weighting in `gaussian_kl` (weight = 1/sigma_T^2, percentile-clipped). Apply in both `Distiller` and `Distiller14`. | (code only) | ~15 min |
| 4 | Run symbolic v3 distillation against QConv2D_Max TAKD assistant. EarlyStopping patience=50. | symbolic: +0.65 -> ~-3.0 (target) | ~1 h |
| 5 | Escalate only if step 4 misses target. Options: **B4** decoupled KL, **B5** teacher temperature, **B6** two-phase schedule. Try one at a time, isolated. | symbolic: extra 0.5-1.5 nat | ~1 h each |
| 6 | **B1** (hint distillation) for QConv2D_Max v2 ONLY if the current QConv2D_Max run plateaus below -4. Refactor `create_model` to expose intermediate features; add `hint_loss` to `Distiller14`. | QConv2D_Max: -1.87 -> -3 to -4 | ~1-2 h |
| 7 | Optional cleanup: **A2** (per-output MLPs) or **A3** (per-pixel barycenter weights) only if step 4 still misses target after steps 5/6. | symbolic: extra 0.2-0.5 nat | ~1 h |

Total wall-clock for the locked path (steps 1-4): **~1.5 hours of work on top of the in-flight QConv2D_Max run**.

---

## Acceptance criteria

- Symbolic student: per-event NLL should drop from current +0.65 to **at least 0 or lower** after A1+B2 (TAKD via QConv2D_Max). Stretch: reach -1 to -2 per event, comparable to a QConv2D_Max trained standalone.
- QConv2D_Max v2 (with B1 + B3): should drop from current best -1.87 to **at least -3.5 to -4.5**, closing more than half the teacher gap.
- All v3 students should keep their distributions well-calibrated: unclipped per-event NLL (no clip floor) within 2x of clipped values. Currently unclipped is +1e9 for the symbolic, which means the predicted covariance is catastrophically wrong on outlier events.

---

## Files to be created/modified

- new: `two_bit_optimization_helpers/models/error_mlp_v2.py` (widened + 14-output)
- modified: `two_bit_optimization_helpers/symbolic/ansatz.py` (add `variant='pysr_aug_v3'` branch that uses error_mlp_v2 and adds mean corrections)
- modified: `two_bit_optimization_helpers/distill.py` and `distill14.py` (add B3 uncertainty weights; optional B4 decoupled KL; B5 temperature; B6 two-phase scheduler)
- new: `scripts/distillation/train_distill_hint.py` (B1 hint distillation for QConv2D_Max; refactor teacher and student to expose intermediate features)
- modified: `scripts/distillation/distill_train.py` (accept `--teacher-from-checkpoint-dir` so we can chain assistants; B2 TAKD)

## Files to be reused (no reinvention)

- `Distiller14` already handles direct-14-output students; subclass for v3 features.
- `PhysicsAnsatz` already has the variant-switching infrastructure.
- `prepare_tfrecords.generate_tfrecords` + `load_tfrecords` unchanged.
- All run output paths under `runs/distill_*/` follow the existing pattern.

## Out of scope (defer to v4 if needed)

- PDE/physics-informed losses (we don't have a strong PDE; just empirical residuals).
- Adversarial / GAN-style distillation.
- LoRA / low-rank adapters as the student.
- Self-distillation (student of self via born-again networks).
- Online / mutual distillation (multiple students concurrent).
- Cross-modal hint training (no good cross-modal analogue for our setup).
