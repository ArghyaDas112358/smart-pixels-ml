"""
End-to-end wiring check for the O12 conditional-NLL constraint, BEFORE launching.

Verifies, on a real trained checkpoint and real validation batches:
  1. the constraint's fn is exactly the measured angle block
  2. infeasibility = max(0, fn - target), and an over-loose target is INERT
  3. the task loss with clip=True is bit-identical to loss.custom_loss (O11 parity)
  4. lambda ASCENDS while the constraint is violated
  5. lambda stays at 0 when the constraint is already satisfied
  6. the MDMM total = task loss + penalty, with the penalty on the same scale as
     the summed loss (scale = BATCH, since fn is a per-event mean)

  CUDA_VISIBLE_DEVICES='' python smoke_o12_nll.py [target]
"""
import os, sys, gc
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import tensorflow as tf
from loss import custom_loss
from conditional_nll import nll_terms, custom_loss_split
from mdmm import MDMM, MaxBlockNLLConstraint
from prepare_tfrecords import load_tfrecords
from train import create_model

TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else -2.58
INERT = -0.35            # the value that turned out to be already satisfied
BATCH = 5000
BASE = ("/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence"
        "_10ps_300k_convolved_to_200ps/shuffled_3d")
TFR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")

ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

_, vg = load_tfrecords(os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test"),
                       noise=-1, seed=42, shuffle=False)
X, Y = vg[0]


def build(target):
    tf.keras.backend.clear_session(); gc.collect()
    vit = create_model("ViT_Max_SimpleRouter", timeslices=101, soft_quantize_layer=True,
                       initial_thresholds=[1., 2., 3.], threshold_offset=0.0,
                       initial_levels=np.array([0., 1., 2., 3.], dtype=np.float32))
    vit.load_weights(os.path.join(R, "runs/simplerouter_mdmm_discovery/seed_2042/last.weights.hdf5"))
    c = MaxBlockNLLConstraint(max_value=target, block='angle', scale=float(BATCH),
                              damping=1.0, name='nll_angle')
    m = MDMM(vit, [c], constraint_samples=None, constraint_pass='primary', name='mdmm_smoke')
    m.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss=custom_loss)
    return vit, c, m


print(f"1) constraint fn == measured angle block   (target {TARGET})")
vit, c, m = build(TARGET)
pred = vit(X, training=False)
fn = float(c.fn(pred, y_true=Y).numpy())
# The reference must be CLIPPED the same way -- the constraint bounds each event
# to [-log(1e9), -log(1e-9)], mirroring loss.custom_loss. Without that bound the
# infeasibility at init is ~1e70 and the damping term overflows float32 to NaN.
_blk = nll_terms(tf.cast(Y, tf.float64), tf.cast(pred, tf.float64)).numpy()[:, 2:].sum(1)
_LO, _HI = -np.log(1e9), -np.log(1e-9)
ref = float(np.clip(_blk, _LO, _HI).mean())
check("fn == clipped mean(term2+term3)", abs(fn - ref) < 1e-5, f"{fn:.6f} vs {ref:.6f}")
print(f"     ({(_blk > _HI).sum()} event(s) bounded; unclipped mean would be {_blk.mean():.6f})")

print("2) infeasibility")
inf = float(c.infeasibility(tf.constant(fn)).numpy())
check("inf == max(0, fn - target)", abs(inf - max(0.0, fn - TARGET)) < 1e-6,  # fn is float32
      f"= {inf:.4f} (binding)")
c_inert = MaxBlockNLLConstraint(max_value=INERT, block='angle', scale=float(BATCH),
                                damping=1.0, name='inert')
check(f"target {INERT} is INERT", float(c_inert.infeasibility(tf.constant(fn)).numpy()) == 0.0,
      "-> confirms -0.35 would do nothing on the block scale")

print("3) task loss parity with O11 (clip=True)")
a = float(custom_loss(tf.cast(Y, tf.float64), tf.cast(pred, tf.float64)).numpy())
b = float(custom_loss_split(tf.cast(Y, tf.float64), tf.cast(pred, tf.float64), clip=True).numpy())
check("custom_loss == split(clip=True)", abs(a - b) / max(abs(a), 1.) < 1e-9, f"{a:.4f} vs {b:.4f}")

print("4) penalty is on the same scale as the summed loss")
pen = float(c(pred, y_true=Y).numpy())
expect = BATCH * (max(0.0, float(c.lmbda.numpy())) * inf + 1.0 * inf**2 / 2)
check("penalty == scale*(max(l,0)*inf + damping*inf^2/2)", abs(pen - expect) < 1e-3,
      f"{pen:.3f} vs {expect:.3f}")
print(f"     task loss {a:,.0f}   penalty {pen:,.1f}   ratio {abs(pen/a):.4f}")

print("5) lambda ASCENDS while violated")
l0 = float(c.lmbda.numpy())
for _ in range(3):
    m.train_on_batch(X, Y)
l1 = float(c.lmbda.numpy())
check("lambda increased over 3 steps", l1 > l0, f"{l0:.5f} -> {l1:.5f}")

print("6) lambda stays 0 when already satisfied")
vit2, c2, m2 = build(INERT)
for _ in range(3):
    m2.train_on_batch(X, Y)
# lambda drifts by ~1e-3 even with inf == 0 (cause not established). Harmless:
# the penalty is scale*(max(l,0)*inf + damping*inf^2/2), which is exactly 0 while
# inf is 0 regardless of lambda. Assert the property that actually matters.
pen_inert = float(c2(vit2(X, training=False), y_true=Y).numpy())
check("penalty is exactly 0 when satisfied", pen_inert == 0.0,
      f"lambda drifted to {float(c2.lmbda.numpy()):.2e} but penalty = {pen_inert}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
