# -*- coding: utf-8 -*-
# smoke_o21c.py -- CPU smoke test for UCBRouterLayer (option O21c).
#
# Run:
#   CUDA_VISIBLE_DEVICES='' python scripts/distillation/smoke_o21c.py
#
# Covers the full public interface plus the arm's MUST-PASS list:
#   1. untried-first: no arm repeats over the first 200 chooses
#   2. synthetic bandit (no NN): planted best pair (11,26) is found
#   3. nonstationarity: planted best flipped mid-run, bandit follows
#   4. checkpoint round-trip: save/load restores q, n, t (and the rest) exactly
#   5. tf.function integration loop (call + bandit_update) without retracing
#   6. sum(mu) == 2 and every logger-contract method works

import os
import sys
import tempfile

import numpy as np
import tensorflow as tf

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'two_bit_optimization_helpers'))

from UCBRouterLayer import UCBRouterLayer  # noqa: E402

T = 101
np.random.seed(0)
tf.random.set_seed(0)

# tiny spatial extent: the bandit never looks at the pixels, only the loss
X_SMALL = tf.constant(np.random.rand(2, 4, 4, T).astype(np.float32))
X_REAL = tf.constant(np.random.rand(2, 16, 16, T).astype(np.float32))


def arm_index(layer, i, j):
    """Arm id of pair (i, j), i < j, in the layer's triu ordering."""
    ia, ib = layer.ia.numpy(), layer.ib.numpy()
    (s,) = np.where((ia == i) & (ib == j))[0]
    return int(s)


def make_layer():
    lyr = UCBRouterLayer(name='simple_router_output')
    lyr.build((None, 4, 4, T))
    return lyr


def ok(msg):
    print(f'  PASS  {msg}', flush=True)


# --------------------------------------------------------------- 6a. interface
print('[interface]', flush=True)
layer = make_layer()

y = layer(X_REAL, training=False)
assert y.shape == (2, 16, 16, 2), y.shape
th = layer.theta
assert th.shape == (T,)
_ = th.numpy()                                   # logger does r.theta.numpy()
mu = layer.mu()
assert mu.shape == (T,)
assert abs(float(tf.reduce_sum(mu)) - 2.0) < 1e-6, float(tf.reduce_sum(mu))
mu_np = layer.mu_numpy()
assert isinstance(mu_np, np.ndarray) and mu_np.shape == (T,)
sig = layer.sigma()
assert isinstance(sig, float) and sig == 0.0
assert float(f'{float(layer.sigma()):.4g}' == '0') is not None  # logger format
i1, i2 = layer.selected_indices()
assert isinstance(i1, int) and isinstance(i2, int) and i1 < i2
# eval readout must be exactly the selected_indices() slices, ascending
np.testing.assert_array_equal(
    y.numpy(), X_REAL.numpy()[..., [i1, i2]])
assert layer.visits.shape == (T,)
assert not layer.visits.trainable
assert len(layer.trainable_weights) == 0        # NOTHING trains by gradient
ok('shapes, theta/mu/sigma/selected_indices/visits, eval readout agrees')

# gradient flows to the input through the hard gather, nowhere else
with tf.GradientTape() as tape:
    tape.watch(X_SMALL)
    out = layer(X_SMALL, training=True)
    loss = tf.reduce_sum(tf.square(out))
gx = tape.gradient(loss, X_SMALL)
a0, b0 = int(layer.ia.numpy()[int(layer.last_arm.numpy())]), \
         int(layer.ib.numpy()[int(layer.last_arm.numpy())])
gnp = gx.numpy()
sel = np.zeros(T, bool); sel[[a0, b0]] = True
assert np.abs(gnp[..., sel]).sum() > 0          # chosen slices carry gradient
assert np.abs(gnp[..., ~sel]).sum() == 0        # unchosen slices carry none
ok('plain gather gradient: nonzero only on the chosen pair')

# visits counted the chosen pair; reset_visits zeroes
v = layer.visits.numpy()
assert v[a0] == 1 and v[b0] == 1 and v.sum() == 2
layer.reset_visits()
assert layer.visits.numpy().sum() == 0
ok('visits / reset_visits')

# ---------------------------------------------------------- 1. untried-first
print('[untried-first]', flush=True)
layer = make_layer()
arms = []
for _ in range(200):
    layer(X_SMALL, training=True)
    arms.append(int(layer.last_arm.numpy()))
    layer.bandit_update(tf.constant(np.random.randn() * 100.0, tf.float32))
assert len(set(arms)) == 200, f'repeats in first 200 chooses: {len(set(arms))}'
ok('200 chooses, 200 distinct arms')

# ----------------------------------------------- 5. tf.function, no retracing
print('[tf.function integration]', flush=True)
layer = make_layer()


@tf.function
def train_like_step(x):
    yy = layer(x, training=True)
    ls = tf.reduce_sum(yy)                       # stands in for the task loss
    layer.bandit_update(ls)
    return ls


for _ in range(30):
    train_like_step(X_SMALL)
count = train_like_step.experimental_get_tracing_count()
assert count == 1, f'retraced: {count} traces'
assert float(layer.t.numpy()) > 0
ok('30 steps, 1 trace, state advanced')

# ------------------------------------------------------- 2. synthetic bandit
print('[synthetic bandit]', flush=True)
layer = make_layer()
best_arm = tf.Variable(arm_index(layer, 11, 26), dtype=tf.int32, trainable=False)


@tf.function
def bandit_steps(steps):
    for _ in tf.range(steps):
        layer(X_SMALL, training=True)
        hit = tf.equal(layer.last_arm, best_arm)
        # reward 1.0 on the planted pair, 0.0 + noise elsewhere; fed as a LOSS
        reward = tf.where(hit, 1.0, 0.0) + tf.random.normal((), stddev=0.05)
        layer.bandit_update(-reward)


bandit_steps(tf.constant(5050 + 200))            # full sweep + a little settle
assert layer.selected_indices() == [11, 26], layer.selected_indices()
# stability under the survey wave: post-sweep the bonus recycles every arm
# faster than any q gap can hold it off (5050 arms > the 2000-step discount
# horizon -- see the layer's design notes), so chooses keep cycling and mu
# stays survey-flat. The GUARANTEE is that q keeps the planted arm on top
# through the wave -- selected_indices() must not wobble.
bandit_steps(tf.constant(3000))
assert layer.selected_indices() == [11, 26], layer.selected_indices()
qv = layer.q.numpy()
s_best = arm_index(layer, 11, 26)
margin = qv[s_best] - np.delete(qv, s_best).max()
assert margin > 1.0, f'q margin too thin: {margin}'
mu = layer.mu_numpy()
assert abs(mu.sum() - 2.0) < 1e-5
ok(f'found planted pair [11, 26], stable; q margin={margin:.2f}, '
   f'sum(mu)={mu.sum():.6f}')

# -------------------------------------------------------- 3. nonstationarity
print('[nonstationarity]', flush=True)
best_arm.assign(arm_index(layer, 40, 80))        # flip the world mid-run
found_at = None
for chunk in range(12):                          # up to 12k steps post-flip;
    bandit_steps(tf.constant(1000))              # one survey cycle is ~5050
    if layer.selected_indices() == [40, 80]:
        found_at = (chunk + 1) * 1000
        break
assert found_at is not None, \
    f'did not follow the flip; stuck on {layer.selected_indices()}'
# and it must STAY there once re-found
bandit_steps(tf.constant(2000))
assert layer.selected_indices() == [40, 80], layer.selected_indices()
ok(f'followed the flip to [40, 80] within {found_at} steps post-flip')

# ------------------------------------------------- 4. checkpoint round-trip
print('[checkpoint round-trip]', flush=True)


def wrap(lyr):
    x_in = tf.keras.layers.Input(shape=(4, 4, T), name='raw_input')
    return tf.keras.Model(x_in, lyr(x_in))


m1 = wrap(layer)                                 # carries the trained state
m2 = wrap(make_layer())
r2 = m2.get_layer('simple_router_output')
for _ in range(7):                               # make m2's state differ
    m2(X_SMALL, training=True)
    r2.bandit_update(tf.constant(3.0))

with tempfile.TemporaryDirectory() as tmp:
    ck = os.path.join(tmp, 'router.weights.hdf5')
    m1.save_weights(ck)
    m2.load_weights(ck)

for name in ('q', 'n', 't', 'last_arm', 'tried',
             'loss_mean', 'loss_var', 'visits'):
    w1 = getattr(layer, name).numpy()
    w2 = getattr(r2, name).numpy()
    assert np.array_equal(w1, w2), f'{name} not restored exactly'
assert r2.selected_indices() == layer.selected_indices() == [40, 80]
ok('q/n/t (+ last_arm, tried, EMA stats, visits) restored bit-exactly')

# ------------------------------------------------ 6b. logger-contract replay
print('[logger contract]', flush=True)
r = m1.get_layer('simple_router_output')
th = r.theta.numpy().astype(np.float64)          # verbatim SimpleRouterLogger
mu = r.mu_numpy().astype(np.float64)
vis = r.visits.numpy().astype(np.float64)
i1, i2 = r.selected_indices()
q = mu / mu.sum()
ent = float(-(q * np.log(q + 1e-12)).sum())
_ = f'{float(r.sigma()):.4g}'
assert np.isfinite(ent) and 0.0 <= ent <= np.log(T) + 1e-9
assert th.shape == mu.shape == vis.shape == (T,)
ok(f'logger lines execute; readout=({i1},{i2}), mu_entropy={ent:.3f}')

print('\nALL SMOKE TESTS PASSED', flush=True)
