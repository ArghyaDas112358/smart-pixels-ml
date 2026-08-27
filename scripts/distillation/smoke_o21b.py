# -*- coding: utf-8 -*-
# smoke_o21b.py — CPU smoke test for TwoRouterLayer (option O21b).
#
# Exercises the full public interface (the SimpleRouterLogger attribute
# contract), the per-router covariance gradient including the slot mapping
# under the ascending output sort, collision-free sampling, the anti-overlap
# penalty, the shared smoothing kernel, tf.function safety, and the
# save/load_weights roundtrip the Gautschi chunk-resume chains depend on.
#
# Run:
#   CUDA_VISIBLE_DEVICES='' /work/users/das214/envs/smartpix-2bit/bin/python \
#       scripts/distillation/smoke_o21b.py

import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'two_bit_optimization_helpers'))

import numpy as np
import tensorflow as tf

from TwoRouterLayer import TwoRouterLayer

T = 101
SHAPE = (2, 4, 4, T)          # (B, H, W, T) — small pixels, full slice axis
np.random.seed(0)
tf.random.set_seed(0)

n_pass = 0


def ok(cond, msg):
    global n_pass
    assert cond, f"FAIL: {msg}"
    n_pass += 1
    print(f"  ok: {msg}")


def fresh_layer(**kw):
    layer = TwoRouterLayer(name='simple_router_output', **kw)
    layer.build(SHAPE)
    return layer


def slice_index_input(batch=2):
    """x[b,h,w,t] = t, so the output channels REVEAL which slices were picked."""
    x = np.broadcast_to(np.arange(T, dtype=np.float32),
                        (batch, 4, 4, T)).copy()
    return tf.constant(x)


def softmax64(v):
    v = np.asarray(v, dtype=np.float64)
    w = np.exp(v - v.max())
    return w / w.sum()


# ---------------------------------------------------------------- interface --
print("[1] interface + logger contract")
r = fresh_layer()
ok(r.theta.numpy().shape == (T,), "theta property -> (101,) with .numpy()")
ok(r.mu_numpy().shape == (T,), "mu_numpy -> (101,)")
ok(isinstance(float(r.sigma()), float) and float(r.sigma()) == 0.0,
   "sigma() float-converts, 0.0 before any scheduler touches it")
ok(r.visits.numpy().shape == (T,) and r.visits.numpy().sum() == 0.0,
   "visits weight present, zeroed")
wnames = {w.name.split('/')[-1].split(':')[0] for w in r.weights}
ok({'thetaA', 'thetaB', 'visits', 'smooth_sigma'} <= wnames,
   f"all persistent state is layer weights ({sorted(wnames)})")
tv = {w.name.split('/')[-1].split(':')[0] for w in r.trainable_weights}
ok(tv == {'thetaA', 'thetaB'}, "exactly thetaA/thetaB trainable")
# logger line, verbatim shape of SimpleRouterLogger.on_epoch_end
th = r.theta.numpy().astype(np.float64)
mu = r.mu_numpy().astype(np.float64)
i1, i2 = r.selected_indices()
ok(isinstance(i1, int) and isinstance(i2, int) and i1 < i2,
   f"selected_indices ascending ints ({i1},{i2})")
q = mu / mu.sum()
ent = float(-(q * np.log(q + 1e-12)).sum())
ok(abs(ent - np.log(T)) < 1e-6, "uniform init -> mu entropy ln(101)")

# --------------------------------------------------------------- sum(mu)==2 --
print("[2] sum(mu) == 2 (MUST-PASS 6)")
# mu is exact in float64 internally but RETURNED in float32 (the reference
# layer's dtype contract), so summing 101 float32 terms carries ~1e-7 of
# representation rounding — 1e-5 is the tightest honest tolerance here.
ok(abs(mu.sum() - 2.0) < 1e-5, "sum(mu)==2 at uniform init")
r.thetaA.assign(np.random.default_rng(1).normal(0, 5, T).astype(np.float32))
r.thetaB.assign(np.random.default_rng(2).normal(0, 5, T).astype(np.float32))
ok(abs(r.mu_numpy().astype(np.float64).sum() - 2.0) < 1e-5,
   "sum(mu)==2 for random committed logits")
# commitment stress: one logit dominating must not NaN (float64 discipline)
va = np.zeros(T, np.float32); va[7] = 60.0
r.thetaA.assign(va); r.thetaB.assign(va)
m = r.mu_numpy()
ok(np.all(np.isfinite(m)) and abs(float(m.sum()) - 2.0) < 1e-5,
   "no NaN and sum(mu)==2 at full commitment (theta=+60)")

# ------------------------------------------------------ planted bumps 11/26 --
print("[3] planted bumps (MUST-PASS 1)")
r = fresh_layer()
va = np.zeros(T, np.float32); va[11] = 8.0
vb = np.zeros(T, np.float32); vb[26] = 8.0
r.thetaA.assign(va); r.thetaB.assign(vb)
mu = r.mu_numpy()
top2 = set(int(i) for i in np.argsort(mu)[-2:])
ok(top2 == {11, 26}, f"mu bumps at 11 and 26 (top-2 of mu = {sorted(top2)})")
ok(mu[11] > 0.9 and mu[26] > 0.9, "both bumps carry ~unit marginal mass")
ok(r.selected_indices() == [11, 26], "selected_indices() == [11, 26]")
y = r(slice_index_input(), training=False)
ok(np.allclose(y.numpy()[..., 0], 11.0) and np.allclose(y.numpy()[..., 1], 26.0),
   "eval forward reads out exactly slices [11, 26], ascending")

# ------------------------------------------------------------- collisions ----
print("[4] collision handling (MUST-PASS 2)")
r = fresh_layer(seed=None)
va = np.zeros(T, np.float32); va[50] = 10.0
r.thetaA.assign(va); r.thetaB.assign(va)   # both routers want slice 50
x = slice_index_input()
r.reset_visits()
n_coll = 0
for _ in range(1000):
    y = r(x, training=True).numpy()
    lo, hi = float(y[0, 0, 0, 0]), float(y[0, 0, 0, 1])
    if lo == hi:
        n_coll += 1
    assert lo < hi, "output not strictly ascending"
ok(n_coll == 0, "1000 training samples, a == b never happens")
vis = r.visits.numpy()
ok(vis.sum() == 2000.0 and vis[50] >= 1000.0,
   "visits counted 2 slices/step; contested slice taken every step")
# eval collision: argmax(phiB) == argmax(phiA) must fall back to B's runner-up
vb = va.copy(); vb[60] = 9.0
r.thetaB.assign(vb)
ok(r.selected_indices() == [50, 60],
   "eval collision resolved to B's runner-up, ascending")
y = r(x, training=False).numpy()
ok(y[0, 0, 0, 0] == 50.0 and y[0, 0, 0, 1] == 60.0,
   "eval forward agrees with selected_indices()")

# ------------------------------------------------------------ anti-overlap ---
print("[5] anti-overlap penalty separates the routers (MUST-PASS 3)")
r = fresh_layer(overlap_scale=100.0)
# EXACTLY identical inits are a symmetric fixed point of any deterministic
# descent (both routers receive identical gradients forever), so break the
# symmetry at float-noise level — the point of the test is that the penalty
# AMPLIFIES an infinitesimal asymmetry into distinct argmaxes.
rng = np.random.default_rng(3)
peak = np.zeros(T, np.float32); peak[40] = 5.0
r.thetaA.assign(peak + rng.normal(0, 1e-4, T).astype(np.float32))
r.thetaB.assign(peak + rng.normal(0, 1e-4, T).astype(np.float32))
assert int(np.argmax(r.thetaA.numpy())) == int(np.argmax(r.thetaB.numpy())) == 40
x1 = tf.zeros((1, 4, 4, T))
# lr chosen for the TEST's timescale: 0.2 leaves the shared peak the argmax
# of both routers after 200 steps (the descent parks in a symmetric near-
# equilibrium), 0.5 pushes through it and the noise decides two DIFFERENT
# argmaxes. The training runs don't need this — there the task gradient, not
# the penalty, does the moving.
opt = tf.keras.optimizers.SGD(learning_rate=0.5)
ov0 = None
for step in range(200):
    with tf.GradientTape() as tape:
        _ = r(x1, training=False)          # add_loss fires on every call
        pen = tf.add_n(r.losses)           # the penalty ALONE — no task loss
    if ov0 is None:
        ov0 = float(pen)
    g = tape.gradient(pen, [r.thetaA, r.thetaB])
    opt.apply_gradients(zip(g, [r.thetaA, r.thetaB]))
_ = r(x1, training=False)
ov1 = float(tf.add_n(r.losses))
aA = int(np.argmax(r.thetaA.numpy())); aB = int(np.argmax(r.thetaB.numpy()))
ok(aA != aB, f"argmaxes separated after 200 penalty-only steps ({aA} vs {aB})")
ok(ov1 < 0.2 * ov0, f"overlap penalty fell {ov0:.3f} -> {ov1:.4f}")

# --------------------------------------------------------- slot-mapping grad --
print("[6] slot-mapping gradient check (MUST-PASS 4)")


def check_slot_case(peak_a, peak_b):
    """Peaked routers force the sampled pair; slot-0/slot-1 downstream
    gradients differ by 100x, so routing dz to the wrong slot is a ~100x
    error the tolerance cannot miss."""
    r = fresh_layer()
    va = np.zeros(T, np.float32); va[peak_a] = 20.0
    vb = np.zeros(T, np.float32); vb[peak_b] = 20.0
    r.thetaA.assign(va); r.thetaB.assign(vb)
    x = tf.constant(np.random.default_rng(peak_a).normal(
        0, 1, SHAPE).astype(np.float32))
    with tf.GradientTape() as tape:
        y = r(x, training=True)
        # dy[...,0] = 1, dy[...,1] = 100 — strongly slot-asymmetric
        loss = tf.reduce_sum(y[..., 0]) + 100.0 * tf.reduce_sum(y[..., 1])
    gA, gB = tape.gradient(loss, [r.thetaA, r.thetaB])
    # +20 logits -> the peak pair is drawn with prob ~1-3e-7; verify anyway
    vis = r.visits.numpy()
    assert vis[peak_a] == 1.0 and vis[peak_b] == 1.0, "unexpected sampled pair"

    # manual reference, float64: g0_t = sum_bhw x[...,t], g1_t = 100*g0_t
    S = x.numpy().astype(np.float64).sum(axis=(0, 1, 2))
    g0, g1 = S, 100.0 * S
    mean = 0.5 * (g0 + g1)
    a_is_lo = peak_a < peak_b
    dzA = mean.copy(); dzA[peak_a] = (g0 if a_is_lo else g1)[peak_a]
    dzB = mean.copy(); dzB[peak_b] = (g1 if a_is_lo else g0)[peak_b]
    pA, pB = softmax64(va), softmax64(vb)
    refA = pA * (dzA - (pA * dzA).sum())
    refB = pB * (dzB - (pB * dzB).sum())
    np.testing.assert_allclose(gA.numpy(), refA, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(gB.numpy(), refB, rtol=1e-5, atol=1e-8)
    # the wrong-slot assignment must NOT also pass (test has power)
    dzA_wrong = mean.copy(); dzA_wrong[peak_a] = (g1 if a_is_lo else g0)[peak_a]
    refA_wrong = pA * (dzA_wrong - (pA * dzA_wrong).sum())
    assert not np.allclose(gA.numpy(), refA_wrong, rtol=1e-3), \
        "slot-mapping test cannot distinguish the slots"


check_slot_case(20, 80)   # a < b: router A -> slot 0 (g0)
ok(True, "a<b case: A's dz took the slot-0 gradient, B the slot-1 (exact match)")
check_slot_case(80, 20)   # a > b: ascending sort puts router A in slot 1 (g1)
ok(True, "a>b case: A's dz took the slot-1 gradient, B the slot-0 (exact match)")

# ------------------------------------------------------------- smoothing -----
print("[7] shared smoothing kernel")
r = fresh_layer()
va = np.zeros(T, np.float32); va[7] = 6.0
r.thetaA.assign(va)
r.smooth_sigma.assign([np.float32(0.0)])
ok(np.allclose(r.smooth(tf.constant(va)).numpy(), va),
   "sigma=0 -> smooth is the identity (bit-path parity with no kernel)")
r.smooth_sigma.assign([np.float32(2.0)])     # the SmoothSigmaScheduler hook
ok(abs(float(r.sigma()) - 2.0) < 1e-7, "scheduler hook: assign -> sigma()==2")
sm = r.smooth(tf.constant(va)).numpy()
ok(sm[7] < 6.0 and sm[5] > 0.0 and sm[9] > 0.0,
   "sigma=2 spreads the slice-7 spike to its neighbours")
const = r.smooth(tf.constant(np.full(T, 3.25, np.float32))).numpy()
ok(np.allclose(const, 3.25, atol=1e-6),
   "constant vector is a fixed point -> edge normalisation exact at slice 0/100")
# ONE sigma drives BOTH routers: mu must reflect the smoothing on each track
vb = np.zeros(T, np.float32); vb[93] = 6.0
r.thetaB.assign(vb)
mu_s = r.mu_numpy()
r.smooth_sigma.assign([np.float32(0.0)])
mu_0 = r.mu_numpy()
ok(mu_s[7] < mu_0[7] and mu_s[93] < mu_0[93],
   "shared sigma softens BOTH routers' marginals")
ok(r.selected_indices() == [7, 93], "readout after smoothing round-trip intact")

# --------------------------------------------------------- tf.function loop --
print("[8] tf.function 20-step training loop (MUST-PASS 5)")
r = fresh_layer()
opt = tf.keras.optimizers.Nadam(learning_rate=1e-2)
xg = tf.constant(np.random.default_rng(9).normal(0, 1, SHAPE).astype(np.float32))


@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        y = r(x, training=True)
        loss = tf.reduce_sum(tf.square(y)) + tf.add_n(r.losses)
    grads = tape.gradient(loss, r.trainable_variables)
    opt.apply_gradients(zip(grads, r.trainable_variables))
    return loss


losses = [float(train_step(xg)) for _ in range(20)]
ok(all(np.isfinite(losses)), "20 graph-mode steps, finite losses")
ok(np.all(np.isfinite(r.thetaA.numpy())) and np.all(np.isfinite(r.thetaB.numpy())),
   "thetas finite after graph-mode updates")
ok(float(r.visits.numpy().sum()) == 40.0,
   "visits advanced inside tf.function (2 per step x 20)")
ok(abs(float(r.mu_numpy().sum()) - 2.0) < 1e-6, "sum(mu)==2 after training")

# --------------------------------------------------------- checkpoint cycle --
print("[9] save/load_weights roundtrip (chunked-resume contract)")
scratch = os.environ.get('SMOKE_SCRATCH',
                         '/tmp/claude-978920/-work-users-das214-SmartPixels/'
                         '7a0a041e-b87d-47e5-ae22-8b5c100f4193/scratchpad')
os.makedirs(scratch, exist_ok=True)
ckpt = os.path.join(scratch, 'smoke_o21b.weights.h5')
x_in = tf.keras.Input(shape=SHAPE[1:], name='raw_input')
router = TwoRouterLayer(name='simple_router_output')
m = tf.keras.Model(x_in, router(x_in))
rng = np.random.default_rng(4)
router.thetaA.assign(rng.normal(0, 2, T).astype(np.float32))
router.thetaB.assign(rng.normal(0, 2, T).astype(np.float32))
router.visits.assign(rng.integers(0, 50, T).astype(np.float32))
router.smooth_sigma.assign([np.float32(1.75)])
saved = [w.numpy().copy() for w in router.weights]
sel_saved = router.selected_indices()
m.save_weights(ckpt)
router.thetaA.assign(np.zeros(T, np.float32))
router.thetaB.assign(np.zeros(T, np.float32))
router.visits.assign(np.zeros(T, np.float32))
router.smooth_sigma.assign([np.float32(0.0)])
m.load_weights(ckpt)
ok(all(np.array_equal(a, w.numpy()) for a, w in zip(saved, router.weights)),
   "thetaA/thetaB/visits/smooth_sigma all restored EXACTLY")
ok(router.selected_indices() == sel_saved, "readout identical after resume")

print(f"\nALL {n_pass} CHECKS PASSED (smoke_o21b)")
