"""O22 smoke test: does the per-bin zero-bias constraint measure and remove a
KNOWN, planted bias, and does quasi-binning actually flatten the weighting?

Deliberately synthetic. If the constraint cannot see a bias we put there on
purpose, it will not see the real one either.

  CUDA_VISIBLE_DEVICES='' python smoke_o22_bias.py
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np, tensorflow as tf
from conditional_nll import soft_bin_weights, soft_bin_membership, cot_to_deg
from mdmm import ZeroBiasConstraint

rng = np.random.default_rng(22)
N, NB = 4000, 15
ok = True

# ---- 1. quasi-binning flattens a deliberately lopsided label distribution ---
# two thirds of the events piled into the middle, like a peaked training set
v = np.concatenate([rng.uniform(-1, 1, N//3), rng.normal(0, 0.15, 2*N//3)]).astype(np.float32)
y = np.stack([v, v, v, v], 1)
centers = np.linspace(-1, 1, NB).astype(np.float32)
sig = float(centers[1] - centers[0])
w = soft_bin_weights(tf.constant(y), [(0, centers, sig)]).numpy()
K = soft_bin_membership(tf.constant(v), centers, sig).numpy()
occ_raw = K.sum(0)
occ_wt = (K * w[:, None]).sum(0)
print("1) quasi-binning")
print(f"   raw occupancy      min {occ_raw.min():8.1f}  max {occ_raw.max():8.1f}  ratio {occ_raw.max()/occ_raw.min():6.1f}x")
print(f"   weighted occupancy min {occ_wt.min():8.1f}  max {occ_wt.max():8.1f}  ratio {occ_wt.max()/occ_wt.min():6.1f}x")
print(f"   mean weight {w.mean():.3f} (must be 1.000), range [{w.min():.2f}, {w.max():.2f}] (clip 0.25-4)")
flat = (occ_wt.max()/occ_wt.min()) < (occ_raw.max()/occ_raw.min()) / 3
print(f"   -> flattened by >3x: {'PASS' if flat else 'FAIL'}"); ok &= flat

# ---- 2. the constraint recovers a planted, position-dependent bias ----------
# plant exactly the shape the real plot shows: pred too high at low v, too low
# at high v, i.e. residual = true - pred = +k*(v - mean)
k = 0.30
true = rng.uniform(-1, 1, N).astype(np.float32)
pred = true - k * true                      # residual = k*true, a clean tilt
outputs = np.zeros((N, 8), np.float32); outputs[:, 0] = pred
labels  = np.zeros((N, 4), np.float32); labels[:, 0] = true
c = ZeroBiasConstraint(column=0, label_column=0, centers=centers, sigma=sig,
                       transform="identity", scale=1.0, damping=1.0, name="zb")
meas = c.fn(tf.constant(outputs), y_true=tf.constant(labels)).numpy()
want = k * centers
print("\n2) constraint measures the planted bias")
print(f"   planted  slope {k:.3f} -> bias at bin edges {want[0]:+.3f} .. {want[-1]:+.3f}")
print(f"   measured                                    {meas[0]:+.3f} .. {meas[-1]:+.3f}")
err = np.abs(meas - want).max()
print(f"   max |measured - planted| = {err:.4f}  -> {'PASS' if err < 0.05 else 'FAIL'}"); ok &= err < 0.05

# ---- 3. one multiplier per bin, and the penalty is L1 in the bias -----------
print("\n3) multipliers and penalty shape")
print(f"   lambda shape {tuple(c.lmbda.shape)} for {NB} bins -> {'PASS' if tuple(c.lmbda.shape)==(NB,) else 'FAIL'}")
ok &= tuple(c.lmbda.shape) == (NB,)
c.lmbda.assign(np.ones(NB, np.float32))
pen = float(c(tf.constant(outputs), y_true=tf.constant(labels)))
# with lambda = 1 and damping = 1: sum_b |bias_b| + |bias_b|^2/2
exp = float(np.abs(meas).sum() + (meas**2).sum()/2)
print(f"   penalty {pen:.4f}  vs  sum|bias| + sum(bias^2)/2 = {exp:.4f} -> {'PASS' if abs(pen-exp)<1e-3 else 'FAIL'}")
ok &= abs(pen - exp) < 1e-3
# unbiased predictions must cost ~0
outputs0 = outputs.copy(); outputs0[:, 0] = true
pen0 = float(c(tf.constant(outputs0), y_true=tf.constant(labels)))
print(f"   penalty on unbiased predictions: {pen0:.6f} -> {'PASS' if pen0 < 1e-3 else 'FAIL'}"); ok &= pen0 < 1e-3

# ---- 4. gradients flow to the model AND up to the multipliers ---------------
print("\n4) gradients")
xv = tf.Variable(outputs)
with tf.GradientTape() as t:
    pen_t = c(xv, y_true=tf.constant(labels))
g_out, g_lam = t.gradient(pen_t, [xv, c.lmbda])
print(f"   d(penalty)/d(prediction) finite and non-zero: {'PASS' if g_out is not None and np.isfinite(g_out.numpy()).all() and np.abs(g_out.numpy()).sum()>0 else 'FAIL'}")
ok &= g_out is not None and np.isfinite(g_out.numpy()).all() and np.abs(g_out.numpy()).sum() > 0
print(f"   d(penalty)/d(lambda) = |bias| (ascent raises lambda where biased): {'PASS' if np.allclose(g_lam.numpy(), np.abs(meas), atol=1e-4) else 'FAIL'}")
ok &= np.allclose(g_lam.numpy(), np.abs(meas), atol=1e-4)

# ---- 5. degree-space transform is finite through cot = 0 -------------------
print("\n5) cot -> degrees through the branch point")
cs = tf.Variable(np.array([-0.5, -1e-4, 0.0, 1e-4, 0.5, 3.0], np.float32))
with tf.GradientTape() as t:
    d = cot_to_deg(cs, 1.0)
gd = t.gradient(d, cs).numpy()
print(f"   degrees {np.round(d.numpy(),2)}")
print(f"   finite values and gradients: {'PASS' if np.isfinite(d.numpy()).all() and np.isfinite(gd).all() else 'FAIL'}")
ok &= np.isfinite(d.numpy()).all() and np.isfinite(gd).all()

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
