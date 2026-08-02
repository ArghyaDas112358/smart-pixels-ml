"""SMOKE TEST for SoftRouterLayer + the joint discovery model (plan checklist item 1).

Runs on CPU with synthetic mV-like data. Checks:
  UNIT (layer alone):
    U1 output shapes, training + inference
    U2 STE identity: training output == the hard 2-slice pick, numerically
    U3 distinct hard indices even with IDENTICAL slot logits (worst case)
    U4 gradients reach slot_logits through the soft path
    U5 sharpening: at k=67 the top slot weight ~ 1 (one-hot)
  INTEGRATION (joint ViT_Max_SoftRouter):
    I1 model builds via create_model('ViT_Max_SoftRouter', timeslices=101, ...)
    I2 BOTH AnnealingSchedulers drive their layer's log_k per the cosine schedule
    I3 slot_logits AND thresholds move during fit; losses finite
    I4 selected_indices() -> num_slots distinct ints in range

Usage:  CUDA_VISIBLE_DEVICES="" python scripts/distillation/smoke_softrouter.py
"""
import os, sys, math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import numpy as np
import tensorflow as tf

from SoftRouterLayer import SoftRouterLayer
from AnnealingScheduler import AnnealingScheduler
from train import create_model
from loss import custom_loss

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

rng = np.random.default_rng(42)
T, B = 101, 8

# mV-like synthetic waveforms: pulse shape * amplitude + baked-noise-like jitter
t = np.arange(T, dtype=np.float32)
pulse = (t / 12.0) * np.exp(-t / 22.0); pulse /= pulse.max()
amp = rng.exponential(150.0, size=(B, 16, 16, 1)).astype(np.float32)
x_np = amp * pulse[None, None, None, :] + rng.normal(0, 4.64, (B, 16, 16, T)).astype(np.float32)
x = tf.constant(x_np)

print("== UNIT: SoftRouterLayer ==")
layer = SoftRouterLayer(num_slots=2, initial_k=1.0, seed=7)
y_tr = layer(x, training=True)
y_inf = layer(x, training=False)

check("U1 shapes", tuple(y_tr.shape) == (B, 16, 16, 2) and tuple(y_inf.shape) == (B, 16, 16, 2),
      f"train={tuple(y_tr.shape)} infer={tuple(y_inf.shape)}")

idx = layer.selected_indices()
expected_hard = np.stack([x_np[..., idx[0]], x_np[..., idx[1]]], axis=-1)
ste_err = float(np.max(np.abs(y_tr.numpy() - expected_hard)))
check("U2 STE: forward == hard pick", ste_err < 1e-4, f"max|diff|={ste_err:.2e}, idx={idx}")

lay_dup = SoftRouterLayer(num_slots=2, initial_k=1.0, seed=7)
_ = lay_dup(x, training=False)  # build
same_row = rng.normal(0, 0.5, T).astype(np.float32)
lay_dup.slot_logits.assign(np.stack([same_row, same_row]))
d_idx = lay_dup.selected_indices()
check("U3 distinct picks w/ identical logits", d_idx[0] != d_idx[1], f"idx={d_idx}")

with tf.GradientTape() as tape:
    out = layer(x, training=True)
    loss = tf.reduce_mean(tf.square(out))
g = tape.gradient(loss, layer.slot_logits)
gnorm = float(tf.norm(g)) if g is not None else -1.0
check("U4 gradient reaches slot_logits", g is not None and gnorm > 0, f"|grad|={gnorm:.3e}")

layer.log_k.assign([math.log(67.0)])
wmax = float(np.max(layer.slot_weights_numpy()[0]))
check("U5 one-hot at k=67", wmax > 0.95, f"max weight={wmax:.4f}")

print("== INTEGRATION: joint ViT_Max_SoftRouter ==")
thr0 = sorted(np.random.default_rng(42).uniform(25.0, 160.0, 3).tolist())  # random init, NOT the optimum
model = create_model('ViT_Max_SoftRouter', timeslices=T, soft_quantize_layer=True,
                     initial_thresholds=thr0, threshold_offset=0.0,
                     initial_levels=np.array([0., 1., 2., 3.], dtype=np.float32))
router = model.get_layer('soft_router_output')
quant = model.get_layer('soft_quantizer_output')
check("I1 joint model builds", router is not None and quant is not None,
      f"params={model.count_params():,}, thr0={[round(v,1) for v in thr0]}")

N = 256
amp_n = rng.exponential(150.0, size=(N, 16, 16, 1)).astype(np.float32)
X = amp_n * pulse[None, None, None, :] + rng.normal(0, 4.64, (N, 16, 16, T)).astype(np.float32)
Y = rng.normal(0, 0.5, (N, 4)).astype(np.float32)
ds = tf.data.Dataset.from_tensor_slices((X, Y)).batch(64)
vs = tf.data.Dataset.from_tensor_slices((X[:64], Y[:64])).batch(64)

model.compile(optimizer=tf.keras.optimizers.Nadam(1e-3), loss=custom_loss)
w_before = router.slot_logits.numpy().copy()
t_before = quant.thresholds.numpy().copy()

EPOCHS = 3
cbs = [AnnealingScheduler('cosine', target_layer_name='soft_router_output',
                          initial_k=1.0, final_k=67.0, verbose=0),
       AnnealingScheduler('cosine', target_layer_name='soft_quantizer_output',
                          initial_k=1.0, final_k=67.0, verbose=0)]
hist = model.fit(ds, validation_data=vs, epochs=EPOCHS, callbacks=cbs, verbose=2)

# expected k at the LAST on_epoch_begin (epoch index EPOCHS-1 of total EPOCHS)
k_expect = 1 + 66 * 0.5 * (1 - math.cos(math.pi * (EPOCHS - 1) / EPOCHS))
k_router = float(tf.exp(router.log_k)[0]); k_quant = float(tf.exp(quant.log_k)[0])
# trainable_k=True (as in _vit_softquantizer): the optimizer may nudge log_k a
# little WITHIN the epoch after the scheduler assigns it -> allow 2% drift.
tol = 0.02 * k_expect
check("I2 both schedulers drove log_k",
      abs(k_router - k_expect) < tol and abs(k_quant - k_expect) < tol,
      f"k_router={k_router:.2f} k_quant={k_quant:.2f} expected={k_expect:.2f}±{tol:.2f}")

dW = float(np.abs(router.slot_logits.numpy() - w_before).max())
dT = float(np.abs(quant.thresholds.numpy() - t_before).max())
losses = hist.history['loss'] + hist.history['val_loss']
check("I3 weights move, losses finite",
      dW > 0 and dT > 0 and all(np.isfinite(losses)),
      f"max|dW|={dW:.3e} max|dT|={dT:.3e} loss[-1]={hist.history['loss'][-1]:.1f}")

fi = router.selected_indices()
check("I4 selected_indices valid",
      len(set(fi)) == 2 and all(0 <= i < T for i in fi), f"indices={fi}")

print(f"\n{'='*50}\nSMOKE: {len(PASS)} passed, {len(FAIL)} failed"
      + (f"  FAILED: {FAIL}" if FAIL else "  — ALL GREEN"))
sys.exit(1 if FAIL else 0)
