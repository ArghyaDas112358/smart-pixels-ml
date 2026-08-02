"""
CPU smoke suite for the MDMM x SimpleRouter stack (mdmm.py port +
run_simplerouter_mdmm_discovery.py wiring). Mirrors smoke_simplerouter.py:
each check prints PASS/FAIL, exits nonzero on any FAIL.

  1. constraint math: perfect corr -> zero penalty; constant pred -> hinge penalty
  2. wrapper exposes inner vars + exactly 4 lambdas
  3. fit() runs (custom train_step traces); losses finite
  4. lambda ascent: violated constraint -> lambdas grow, never negative-effective
  5. visits invariant: constraint pass must NOT double-count router visits
  6. theta receives gradient (router learns under the wrapper)
  7. val loss is the plain NLL (wrapper.evaluate == inner-model NLL)
  8. get_layer delegation (AnnealingScheduler drives k through the wrapper)
  9. save_weights delegation -> weights load into a PLAIN model, same output
"""
import os, sys, tempfile
os.environ['CUDA_VISIBLE_DEVICES'] = ''

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf

from train import create_model
from AnnealingScheduler import AnnealingScheduler
from loss import custom_loss
from mdmm import MDMM, MinCorrConstraint

FAILED = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILED.append(name)

rng = np.random.default_rng(0)
B, T = 32, 101
X = rng.uniform(0, 300, size=(B, 16, 16, T)).astype(np.float32)
# The full-cov NLL (custom_loss) is NOT usable for a synthetic-data wiring test:
# at random init the predicted Cholesky diagonal can be ~1e-9, the 4D Gaussian is
# a near-delta, ANY label underflows to likelihood 0, clip_by_value saturates and
# the WHOLE network gets exactly-zero gradient (verified: dense kernels get 0 too).
# The smoke therefore compiles with a smooth surrogate (MSE) -- it tests the MDMM
# WIRING (gradient routing, ascent, delegation); the real custom_loss stack is
# exercised by the 3-epoch GPU --sanity run on real data, where init density is
# finite (the unconstrained discovery converged 6/6 from these exact inits).
Y = rng.normal(0, 0.5, size=(B, 14)).astype(np.float32)   # (B,14) for MSE; cols 0-3 feed the corr constraints

OUT_COLS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
LAB_COLS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}

# ---- 1. constraint math ------------------------------------------------------
c = MinCorrConstraint(column=0, label_column=0, min_value=0.5, scale=1e4, damping=1.0,
                      name="corr_test")
fake_y = np.zeros((B, 4), np.float32); fake_y[:, 0] = np.arange(B)
perfect = np.zeros((B, 14), np.float32); perfect[:, 0] = 2.0 * np.arange(B) + 1.0
pen_perfect = float(c(tf.constant(perfect), y_true=tf.constant(fake_y)))
const = np.zeros((B, 14), np.float32); const[:, 0] = 7.0
pen_const = float(c(tf.constant(const), y_true=tf.constant(fake_y)))
# corr(const)=0 via the eps-guarded denom -> inf=0.5 -> at lambda=0: 1e4*0.5*0.25
check("MinCorr: perfect corr -> zero penalty", abs(pen_perfect) < 1e-3, f"pen={pen_perfect:.2g}")
check("MinCorr: constant pred -> hinge penalty ~1250", abs(pen_const - 1250.0) < 1.0,
      f"pen={pen_const:.6g}")

# ---- build the wrapped model -------------------------------------------------
thr0 = sorted(np.random.default_rng(42).uniform(25., 160., 3).tolist())
vit = create_model('ViT_Max_SimpleRouter', timeslices=T, soft_quantize_layer=True,
                   initial_thresholds=thr0, threshold_offset=0.0,
                   initial_levels=np.array([0., 1., 2., 3.], np.float32))
constraints = [MinCorrConstraint(column=OUT_COLS[p], label_column=LAB_COLS[p],
                                 min_value=0.5, scale=1e4, damping=1.0, name=f"corr_{p}")
               for p in OUT_COLS]
# constraint_pass='primary' is the shipped mode: the deterministic second pass
# dead-ends at SoftQuantizeLayer's eval-branch stop_gradient (theta starved) and
# its dropout-free train-batch corr is satisfiable by memorization.
model = MDMM(vit, constraints, constraint_pass='primary', name='mdmm_smoke')
model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss='mse')

# ---- 2. variable bookkeeping -------------------------------------------------
n_inner = len(vit.trainable_variables)
n_wrap = len(model.trainable_variables)
check("wrapper tracks inner vars + 4 lambdas", n_wrap == n_inner + 4,
      f"inner={n_inner} wrapped={n_wrap}")

# ---- 3-4. fit + lambda ascent ------------------------------------------------
router = model.get_layer('simple_router_output')
visits0 = float(router.visits.numpy().sum())
theta0 = router.theta.numpy().copy()
lam0 = [float(c.lmbda.numpy()) for c in constraints]

EPOCHS, BS = 3, 8
steps_per_epoch = B // BS
h = model.fit(X, Y, epochs=EPOCHS, batch_size=BS, verbose=0, shuffle=False,
              callbacks=[AnnealingScheduler('cosine', target_layer_name='soft_quantizer_output',
                                            initial_k=1.0, final_k=67.0, verbose=0)])
losses = h.history['loss']
check("fit runs, losses finite", all(np.isfinite(v) for v in losses),
      f"loss={losses[-1]:.4g}")

lam1 = [float(c.lmbda.numpy()) for c in constraints]
# random-init model on random labels: corr ~ 0 -> all 4 constraints violated -> ascent
check("lambdas ascend under violated constraints",
      all(l1 > l0 for l0, l1 in zip(lam0, lam1)),
      f"{[round(l, 5) for l in lam1]}")

# ---- 5. visits invariant (no double-count from the constraint pass) ----------
n_steps = EPOCHS * steps_per_epoch
visits1 = float(router.visits.numpy().sum())
check("visits == 2 * train steps (constraint pass adds none)",
      abs((visits1 - visits0) - 2 * n_steps) < 1e-6,
      f"visits+={visits1 - visits0:g}, steps={n_steps}")

# ---- 6. theta gets gradient --------------------------------------------------
dtheta = float(np.abs(router.theta.numpy() - theta0).max())
check("theta moved (router learns under wrapper)", dtheta > 0, f"max|dtheta|={dtheta:.3g}")

# ---- 7. val loss is the plain compiled loss (no penalties) -------------------
wrapped_eval = float(model.evaluate(X, Y, batch_size=B, verbose=0))
plain_mse = float(tf.reduce_mean(tf.square(tf.constant(Y) - vit(X, training=False))))
check("evaluate == plain compiled loss (no penalties in val)",
      abs(wrapped_eval - plain_mse) < max(1e-3 * abs(plain_mse), 1e-2),
      f"eval={wrapped_eval:.6g} mse={plain_mse:.6g}")

# ---- 8. annealer drove k through the wrapper ---------------------------------
quant = model.get_layer('soft_quantizer_output')
k_now = float(np.exp(quant.log_k.numpy()))
check("AnnealingScheduler drove quantizer k via delegation", k_now > 1.0 + 1e-6,
      f"k={k_now:.3f}")

# ---- 9. save_weights delegation -> plain model round-trip --------------------
with tempfile.TemporaryDirectory() as td:
    wpath = os.path.join(td, 'w.hdf5')
    model.save_weights(wpath)
    vit2 = create_model('ViT_Max_SimpleRouter', timeslices=T, soft_quantize_layer=True,
                        initial_thresholds=thr0, threshold_offset=0.0,
                        initial_levels=np.array([0., 1., 2., 3.], np.float32))
    vit2.load_weights(wpath)
    p1 = vit(X[:4], training=False).numpy()
    p2 = vit2(X[:4], training=False).numpy()
    check("checkpoint loads into PLAIN model, identical output",
          np.allclose(p1, p2, atol=1e-6), f"max|dp|={np.abs(p1 - p2).max():.3g}")

# ---- 10. legacy deterministic mode still runs (Harshul-compat regression) ----
vit_d = create_model('ViT_Max_SimpleRouter', timeslices=T, soft_quantize_layer=True,
                     initial_thresholds=thr0, threshold_offset=0.0,
                     initial_levels=np.array([0., 1., 2., 3.], np.float32))
cons_d = [MinCorrConstraint(column=0, label_column=0, min_value=0.5, scale=1e4,
                            damping=1.0, name='corr_det')]
model_d = MDMM(vit_d, cons_d, constraint_pass='deterministic', name='mdmm_det_smoke')
model_d.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss='mse')
h_d = model_d.fit(X, Y, epochs=1, batch_size=8, verbose=0, shuffle=False)
check("deterministic mode still trains (legacy compat)",
      all(np.isfinite(v) for v in h_d.history['loss']),
      f"loss={h_d.history['loss'][-1]:.4g}")

print()
if FAILED:
    print(f"SMOKE: {len(FAILED)} FAILED -> {FAILED}")
    sys.exit(1)
print(f"SMOKE: 10/10 PASS" if False else f"SMOKE: all {10 - len(FAILED)} checks PASS")
