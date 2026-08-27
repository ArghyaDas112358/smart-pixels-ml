# -*- coding: utf-8 -*-
# smoke_o21a.py -- CPU smoke test for PairLatticeRouterLayer (option O21a).
#
# Exercises every public interface method plus the arm's MUST-PASS checks:
#   1. planted psi on wide pair (11,26): mu bumps at 11 AND 26, readout == [11,26]
#   2. sigma=2.0: the two-bump structure SURVIVES 2D pair-space smoothing
#   3. gradient: finite, sums to ~0, and pulls TOWARD a loss-decreasing pair
#   4. sum(mu) == 2 within 1e-6 (planted, random, smoothed)
#   5. tf.function training loop: 20 steps, no retracing errors, psi moves
# plus weight-only checkpoint round-trip (Gautschi chunk-resume contract).
#
# Run: CUDA_VISIBLE_DEVICES='' python smoke_o21a.py

import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

HELPERS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'two_bit_optimization_helpers')
sys.path.insert(0, os.path.abspath(HELPERS))

import numpy as np
import tensorflow as tf

from PairLatticeRouterLayer import PairLatticeRouterLayer

T = 101
np.random.seed(0)
tf.random.set_seed(0)

ia, ib = np.triu_indices(T, k=1)
NUM_PAIRS = len(ia)                     # 5050


def pair_index(a, b):
    s = np.where((ia == a) & (ib == b))[0]
    assert len(s) == 1
    return int(s[0])


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f'  ({detail})' if detail else ''))
    assert cond, name


layer = PairLatticeRouterLayer(smooth_logits=True, name='simple_router_output')
x = tf.random.normal((4, 16, 16, T))
_ = layer(x, training=False)            # build

# ---- interface surface -------------------------------------------------------
check('psi weight shape', tuple(layer.psi.shape) == (NUM_PAIRS,))
check('theta property shape + numpy()', layer.theta.numpy().shape == (T,))
check('sigma() is python float 0.0',
      isinstance(layer.sigma(), float) and layer.sigma() == 0.0)
check('visits is a (101,) weight',
      tuple(layer.visits.shape) == (T,) and not layer.visits.trainable)
check('mu()/mu_numpy() shape', layer.mu_numpy().shape == (T,))
check('uniform-psi mu is uniform 2/101',
      np.allclose(layer.mu_numpy(), 2.0 / T, atol=1e-6))
names = [w.name for w in layer.weights]
check('persistent state is all layer weights (psi, visits, smooth_sigma)',
      len(layer.weights) == 3 and
      any('psi' in n for n in names) and
      any('visits' in n for n in names) and
      any('smooth_sigma' in n for n in names), ';'.join(names))

# ---- MUST-PASS 1: planted wide pair (11,26), sigma = 0 -----------------------
s_star = pair_index(11, 26)
psi0 = np.zeros(NUM_PAIRS, np.float32)
psi0[s_star] = 40.0
layer.psi.assign(psi0)

mu = layer.mu_numpy().astype(np.float64)
check('planted: sum(mu) == 2 within 1e-6', abs(mu.sum() - 2.0) < 1e-6,
      f'sum={mu.sum():.9f}')
check('planted: mu bump at 11', mu[11] > 0.9, f'mu[11]={mu[11]:.4f}')
check('planted: mu bump at 26', mu[26] > 0.9, f'mu[26]={mu[26]:.4f}')
check('planted: selected_indices() == [11, 26]',
      layer.selected_indices() == [11, 26], str(layer.selected_indices()))

y = layer(x, training=False).numpy()
want = np.stack([x.numpy()[..., 11], x.numpy()[..., 26]], axis=-1)
check('eval forward: hard gather of [11, 26], ascending, shape (B,H,W,2)',
      y.shape == (4, 16, 16, 2) and np.array_equal(y, want))
th = layer.theta.numpy()
check('theta: planted slices dominate the log-marginal score',
      sorted(np.argsort(th)[-2:].tolist()) == [11, 26])

# ---- MUST-PASS 2: sigma = 2.0, two-bump structure survives -------------------
layer.smooth_sigma.assign([2.0])
check('sigma() reads back 2.0', abs(layer.sigma() - 2.0) < 1e-6)

mu2 = layer.mu_numpy().astype(np.float64)
check('smoothed: sum(mu) == 2 within 1e-6', abs(mu2.sum() - 2.0) < 1e-6,
      f'sum={mu2.sum():.9f}')
nb1 = np.arange(9, 14)                  # 11 +- 2
nb2 = np.arange(24, 29)                 # 26 +- 2
outside = np.setdiff1d(np.arange(T), np.concatenate([nb1, nb2]))
check('smoothed: global mu argmax inside 11+-2 or 26+-2',
      int(np.argmax(mu2)) in set(nb1) | set(nb2), f'argmax={int(np.argmax(mu2))}')
check('smoothed: bump at 11+-2 beats everything outside both neighbourhoods',
      mu2[nb1].max() > mu2[outside].max(),
      f'{mu2[nb1].max():.4f} vs {mu2[outside].max():.4f}')
check('smoothed: bump at 26+-2 beats everything outside both neighbourhoods',
      mu2[nb2].max() > mu2[outside].max(),
      f'{mu2[nb2].max():.4f} vs {mu2[outside].max():.4f}')
sel = layer.selected_indices()
check('smoothed: readout stays in the (11,26) neighbourhood',
      sel[0] in set(nb1) and sel[1] in set(nb2), str(sel))
# pair-space geometry: (10,27) and (12,25) are diagonal neighbours of (11,26),
# so the smoothed logit there must stay close to the smoothed peak -- the
# design's reason to exist vs slice-space kernels.
phi2 = layer.smooth(tf.convert_to_tensor(layer.psi)).numpy()
peak = phi2[pair_index(*sel)]
check('smoothed: wide neighbours (10,27)/(12,25) sit near the peak',
      phi2[pair_index(10, 27)] > 0.5 * peak and
      phi2[pair_index(12, 25)] > 0.5 * peak,
      f'peak={peak:.3f} n1={phi2[pair_index(10, 27)]:.3f} n2={phi2[pair_index(12, 25)]:.3f}')

# ---- MUST-PASS 4 (random psi leg): sum(mu) == 2 ------------------------------
layer.psi.assign(np.random.default_rng(1).normal(0, 3, NUM_PAIRS).astype(np.float32))
check('random psi, sigma=2: sum(mu) == 2 within 1e-6',
      abs(layer.mu_numpy().astype(np.float64).sum() - 2.0) < 1e-6)
layer.smooth_sigma.assign([0.0])
check('random psi, sigma=0: sum(mu) == 2 within 1e-6',
      abs(layer.mu_numpy().astype(np.float64).sum() - 2.0) < 1e-6)

# ---- MUST-PASS 3: gradient check --------------------------------------------
# Loss = sum(y): dy is all ones, so dz[t] = sum_bhw x[..., t] for every t (the
# channel-mean fill equals the true slot gradient here, so the check is
# deterministic regardless of which pair got sampled). Slices 11 and 26 carry
# large NEGATIVE values -> selecting (11,26) DECREASES the loss -> the
# covariance gradient at s* must be NEGATIVE (descent then RAISES psi[s*],
# pulling the sampler toward the good pair -- the reference's convention).
psi3 = np.zeros(NUM_PAIRS, np.float32)
psi3[s_star] = 3.0                      # moderate: p(s*) ~ 0.004, non-degenerate
layer.psi.assign(psi3)

x3 = np.ones((4, 16, 16, T), np.float32)
x3[..., 11] = -50.0
x3[..., 26] = -50.0
x3 = tf.constant(x3)
with tf.GradientTape() as tape:
    y3 = layer(x3, training=True)
    loss3 = tf.reduce_sum(y3)
dpsi = tape.gradient(loss3, layer.psi).numpy().astype(np.float64)
check('grad: finite everywhere', np.all(np.isfinite(dpsi)))
check('grad: negative at the loss-decreasing pair (11,26)', dpsi[s_star] < 0.0,
      f'dpsi[s*]={dpsi[s_star]:.4g}')
check('grad: covariance estimator sums to ~0',
      abs(dpsi.sum()) < 1e-3 * np.abs(dpsi).max(),
      f'sum={dpsi.sum():.4g} max={np.abs(dpsi).max():.4g}')
# same tape through the smoothing path: adjoint must stay finite and keep
# pulling toward the planted neighbourhood
layer.smooth_sigma.assign([2.0])
with tf.GradientTape() as tape:
    loss3s = tf.reduce_sum(layer(x3, training=True))
dpsi_s = tape.gradient(loss3s, layer.psi).numpy()
check('grad through smoothing: finite everywhere', np.all(np.isfinite(dpsi_s)))
check('grad through smoothing: still negative at (11,26)',
      float(dpsi_s[s_star]) < 0.0, f'{float(dpsi_s[s_star]):.4g}')
layer.smooth_sigma.assign([0.0])

# ---- MUST-PASS 5: tf.function training loop ----------------------------------
layer.psi.assign(np.zeros(NUM_PAIRS, np.float32))
layer.reset_visits()
check('reset_visits()', float(layer.visits.numpy().sum()) == 0.0)
opt = tf.keras.optimizers.SGD(learning_rate=0.5)


@tf.function
def train_step(xb):
    with tf.GradientTape() as tp:
        yb = layer(xb, training=True)
        loss = tf.reduce_sum(tf.square(yb))
    g = tp.gradient(loss, [layer.psi])
    opt.apply_gradients(zip(g, [layer.psi]))
    return loss


psi_before = layer.psi.numpy().copy()
losses = []
for i in range(20):
    losses.append(float(train_step(tf.random.normal((8, 16, 16, T)))))
check('tf.function loop: 20 steps, losses finite', np.all(np.isfinite(losses)))
check('tf.function loop: psi moved',
      float(np.abs(layer.psi.numpy() - psi_before).max()) > 0.0)
check('tf.function loop: visits counted 2 per step',
      float(layer.visits.numpy().sum()) == 40.0,
      f'sum={float(layer.visits.numpy().sum())}')

# ---- checkpoint round-trip (weights-only, the chunk-resume contract) ---------
layer.psi.assign(psi0)
layer.smooth_sigma.assign([1.5])
saved = layer.get_weights()
fresh = PairLatticeRouterLayer(smooth_logits=True, name='simple_router_output')
_ = fresh(x, training=False)
fresh.set_weights(saved)
check('weight round-trip: psi/visits/smooth_sigma exact',
      all(np.array_equal(a, b) for a, b in zip(saved, fresh.get_weights())))
check('weight round-trip: readout preserved', fresh.selected_indices() == [11, 26])
check('weight round-trip: sigma preserved', abs(fresh.sigma() - 1.5) < 1e-6)

print('\nALL CHECKS PASSED (smoke_o21a)')
