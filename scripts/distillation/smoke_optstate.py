"""
CPU smoke test for optimizer-state checkpointing (the thing that makes a
12h-capped, self-chaining Slurm run behave like one continuous run).

It does NOT just check that files appear. It checks the property that matters:
after a save -> rebuild -> restore cycle, the NEXT optimizer update must match
what an uninterrupted run would have produced. A cold optimizer fails this
because Nadam's moment estimates start at zero.

  1. train N steps, snapshot optimizer state, then take one more step -> W_cont
  2. rebuild from weights only (cold optimizer), one step            -> W_cold
  3. rebuild from weights + restored optimizer state, one step       -> W_warm
  PASS iff W_warm == W_cont (to float tolerance) and W_cold != W_cont
"""
import os, sys, tempfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

#
# NB: this runs on a small DETERMINISTIC model, not the real ViT+router. The
# router samples a fresh slice pair every step and the ViT has dropout, so a
# "continuous" and a "restarted" run diverge from RNG drift alone -- an earlier
# version of this test on the real model showed warm and cold restarts equally
# far from continuous (2.6e-3 vs 2.9e-3) purely from that, telling us nothing.
# Optimizer-state checkpointing is model-agnostic, so isolating it here gives an
# exact, unambiguous answer.
import numpy as np
import tensorflow as tf
from mdmm import MDMM, MinCorrConstraint

FAILED = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILED.append(name)

rng = np.random.default_rng(3)
X = rng.uniform(-1, 1, size=(64, 32)).astype(np.float32)
Y = rng.normal(0, 0.5, size=(64, 14)).astype(np.float32)
OUT_COLS = {"x": 0, "y": 2, "cotA": 4, "cotB": 6}
LAB_COLS = {"x": 0, "y": 1, "cotA": 2, "cotB": 3}


def build():
    """Deterministic MLP (no dropout, no sampling) with 14 outputs, wrapped in
    the real MDMM so the optimizer state includes the Lagrange multipliers."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(0); np.random.seed(0)
    init = tf.keras.initializers.GlorotUniform(seed=0)
    net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)),
        tf.keras.layers.Dense(24, activation='relu', kernel_initializer=init, name='d1'),
        tf.keras.layers.Dense(14, kernel_initializer=init, name='d2'),
    ])
    cons = [MinCorrConstraint(column=OUT_COLS[p], label_column=LAB_COLS[p], min_value=0.5,
                              scale=1e4, damping=1.0, name=f"corr_{p}") for p in OUT_COLS]
    m = MDMM(net, cons, constraint_pass='primary', name='mdmm_opt')
    m.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=1e-3), loss='mse')
    return m, net


def probe(model):
    """A weight that Nadam actually updates, as a fingerprint."""
    v = [w for w in model.model.trainable_variables if 'd1' in w.name][0]
    return v.numpy().copy()


with tempfile.TemporaryDirectory() as td:
    wpath = os.path.join(td, 'w.hdf5')
    opath = os.path.join(td, 'opt_state')

    # ---- 1. continuous run: 3 epochs, snapshot, then one more epoch ----------
    m, vit = build()
    m.fit(X, Y, epochs=3, batch_size=8, verbose=0, shuffle=False)
    m.save_weights(wpath)
    ck = tf.train.Checkpoint(optimizer=m.optimizer, constraints=list(m.constraints_list))
    ck.write(opath)
    n_slots = len(m.optimizer.variables())
    m.fit(X, Y, epochs=1, batch_size=8, verbose=0, shuffle=False)
    W_cont = probe(m)

    check("optimizer had state to save", n_slots > 2, f"{n_slots} optimizer variables")
    check("checkpoint files written", len([f for f in os.listdir(td) if f.startswith('opt_state')]) > 0)

    # ---- 2. cold restart: weights only ---------------------------------------
    m_cold, _ = build()
    m_cold.load_weights(wpath)
    m_cold.fit(X, Y, epochs=1, batch_size=8, verbose=0, shuffle=False)
    W_cold = probe(m_cold)

    # ---- 3. warm restart: weights + optimizer state --------------------------
    m_warm, _ = build()
    m_warm.load_weights(wpath)
    tf.train.Checkpoint(optimizer=m_warm.optimizer,
                        constraints=list(m_warm.constraints_list)).read(opath).expect_partial()
    m_warm.fit(X, Y, epochs=1, batch_size=8, verbose=0, shuffle=False)
    W_warm = probe(m_warm)

    d_warm = float(np.abs(W_warm - W_cont).max())
    d_cold = float(np.abs(W_cold - W_cont).max())
    check("WARM restart reproduces the uninterrupted run", d_warm < 1e-6,
          f"max|W_warm - W_cont| = {d_warm:.3e}")
    check("COLD restart does NOT (i.e. the test is sensitive)", d_cold > 1e-6,
          f"max|W_cold - W_cont| = {d_cold:.3e}")
    check("warm is >100x closer than cold", d_cold > 100 * max(d_warm, 1e-12),
          f"ratio {d_cold / max(d_warm, 1e-12):.1f}x")

print()
if FAILED:
    print(f"SMOKE: {len(FAILED)} FAILED -> {FAILED}"); sys.exit(1)
print("SMOKE: all checks PASS")
