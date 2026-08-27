"""
Prove conditional_nll is an exact re-itemisation of loss.custom_loss, not a new loss.

Checks, in order:
  1. sum of the four conditional terms == the tfp joint NLL              (random p)
  2. custom_loss_split == custom_loss                                    (unclipped regime)
  3. the angle block is invariant to swapping cotA and cotB in the order
  4. gradients wrt the model output agree with the current loss
  5. beta = 0 reproduces plain NLL; beta > 0 only rescales, never reorders
  6. the same equality on a REAL trained checkpoint's outputs             (opt-in)

  CUDA_VISIBLE_DEVICES='' python test_loss_split.py            # 1-5
  CUDA_VISIBLE_DEVICES='' python test_loss_split.py --real 2042
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
from loss import custom_loss, custom_loss_perevent
from conditional_nll import nll_terms, custom_loss_split, angle_block, beta_nll_terms, scale_tril

TOL = 1e-9
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def make_batch(B=64, seed=0, scale=1.0):
    """A batch of plausible 14-vectors: positive L diagonal, moderate residuals."""
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(B, 14)).astype(np.float64) * scale
    p[:, 1:8:2] = np.abs(p[:, 1:8:2]) + 0.5
    y = (p[:, 0:8:2] + rng.normal(size=(B, 4)) * 0.5).astype(np.float64)
    return tf.constant(y), tf.constant(p)


print("1) terms sum to the tfp joint NLL")
y, p = make_batch()
L, _ = scale_tril(p)
dist = tfp.distributions.MultivariateNormalTriL(loc=p[:, 0:8:2], scale_tril=L)
joint = -dist.log_prob(y).numpy()
terms = nll_terms(y, p).numpy()
d = np.abs(joint - terms.sum(1)).max()
check("max |joint - sum(terms)|", d < TOL, f"= {d:.2e}")

print("2) custom_loss_split == custom_loss (away from the 1e-9 clip)")
a = float(custom_loss(y, p).numpy())
b = float(custom_loss_split(y, p).numpy())
check("scalar losses agree", abs(a - b) / max(abs(a), 1.0) < 1e-9,
      f"{a:.6f} vs {b:.6f}")

print("3) angle block invariant to cotA/cotB ordering")
Sig = (L @ tf.linalg.matrix_transpose(L)).numpy()
P = np.array([0, 1, 3, 2])
L2 = np.linalg.cholesky(Sig[:, P][:, :, P])
mu2 = p.numpy()[:, 0:8:2][:, P]
z2 = np.linalg.solve(L2, (y.numpy()[:, P] - mu2)[..., None])[..., 0]
t2 = 0.5 * z2**2 + np.log(np.diagonal(L2, axis1=1, axis2=2)) + 0.5 * np.log(2 * np.pi)
d = np.abs((t2[:, 2] + t2[:, 3]) - angle_block(y, p).numpy()).max()
check("max |block(A,B) - block(B,A)|", d < 1e-8, f"= {d:.2e}")
d2 = np.abs(t2.sum(1) - terms.sum(1)).max()
check("joint unchanged by reordering", d2 < 1e-8, f"= {d2:.2e}")

print("4) gradients wrt the model output agree")
pv = tf.Variable(p)
with tf.GradientTape() as tape:
    g_old = custom_loss(y, pv)
go = tape.gradient(g_old, pv).numpy()
with tf.GradientTape() as tape:
    g_new = custom_loss_split(y, pv)
gn = tape.gradient(g_new, pv).numpy()
rel = np.abs(go - gn).max() / max(np.abs(go).max(), 1e-12)
check("max rel grad difference", rel < 1e-8, f"= {rel:.2e}")

print("5) beta-NLL: beta=0 is plain NLL; beta>0 rescales per target")
b0 = beta_nll_terms(y, p, [0, 0, 0, 0]).numpy()
check("beta=0 == nll_terms", np.abs(b0 - terms).max() < 1e-9)
b1 = beta_nll_terms(y, p, [0, 0, 0.5, 0.5]).numpy()
check("position untouched at beta=0", np.abs(b1[:, :2] - terms[:, :2]).max() < 1e-9)
w = (b1[:, 2] / np.maximum(terms[:, 2], 1e-12))
_, diag = scale_tril(p)
expect = (diag.numpy()[:, 2] ** 1.0)
check("angle weight == L_kk^(2*beta)", np.abs(w - expect).max() < 1e-6,
      f"max dev {np.abs(w - expect).max():.2e}")

if "--real" in sys.argv:
    seed = int(sys.argv[sys.argv.index("--real") + 1])
    print(f"6) real checkpoint, seed {seed}")
    from prepare_tfrecords import load_tfrecords
    from train import create_model
    BASE = ("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"
            "_10ps_300k_convolved_to_200ps/shuffled_3d")
    TFR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")
    _, vg = load_tfrecords(os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test"),
                           noise=-1, seed=42, shuffle=False)
    m = create_model(os.environ.get("SMARTPIX_MODEL", "ViT_Max_SimpleRouter"),
                     timeslices=101, soft_quantize_layer=True,
                     initial_thresholds=[1., 2., 3.], threshold_offset=0.0,
                     initial_levels=np.array([0., 1., 2., 3.], dtype=np.float32))
    m.load_weights(os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR",
                   "simplerouter_mdmm_discovery"), f"seed_{seed}", "last.weights.hdf5"))
    X, yt = vg[0]
    pr = tf.cast(m(X, training=False), tf.float64)
    yt = tf.cast(yt, tf.float64)
    a = float(custom_loss(yt, pr).numpy())
    b = float(custom_loss_split(yt, pr).numpy())               # clip=True default
    check("real batch: clip=True reproduces custom_loss", abs(a - b) / max(abs(a), 1.0) < 1e-9,
          f"{a:.4f} vs {b:.4f}")
    # and per event, everything away from the clip must match to machine precision
    cur = custom_loss_perevent(yt, pr).numpy()
    raw = nll_terms(yt, pr).numpy().sum(1)
    hi = -np.log(1e-9)
    pinned = np.abs(cur - hi) < 1e-6
    d = np.abs(cur[~pinned] - raw[~pinned]).max()
    check(f"{(~pinned).sum()} unclipped events agree", d < 1e-9, f"max |diff| = {d:.2e}")
    print(f"     {pinned.sum()} event(s) truncated by the clip; true NLL "
          f"{np.round(raw[pinned], 1) if pinned.any() else '-'} vs cap {hi:.2f}")
    t = nll_terms(yt, pr).numpy().mean(0)
    print(f"     mean per-event cost:  x {t[0]:+.4f} | y|x {t[1]:+.4f} | "
          f"cotA|x,y {t[2]:+.4f} | cotB|.. {t[3]:+.4f}")
    print(f"     position block {t[0]+t[1]:+.4f}   angle block {t[2]+t[3]:+.4f}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
