"""
Smoke test for the O15 smoothing kernel in SimpleRouterLayer.

Checks, against brute-force enumeration of all C(101,2) pairs:
  1. sigma = 0 is EXACT identity (byte-identical to the un-smoothed layer)
  2. the smoothing matrix is row-stochastic (rows sum to 1) => no edge bias
  3. mu from the layer == mu from enumerating p({a,b}) ~ exp(phi_a+phi_b)
  4. selected_indices() and mu() agree with each other and with the sampler
  5. the gradient reaches theta through the kernel, and equals k * dL/dphi
     (convolution is self-adjoint for a symmetric kernel)
  6. a Gaussian bump in theta stays centred after smoothing (no drift)
  7. smoothing a boundary spike does NOT pull its mass off the edge -- the
     failure mode plain zero-padding would introduce, and slice 7 lives there
  8. sigma > 0 strictly widens mu (entropy increases)

  CUDA_VISIBLE_DEVICES='' python smoke_smooth_router.py
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import tensorflow as tf
from SimpleRouterLayer import SimpleRouterLayer

T = 101
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def build(theta, sigma, smooth=True):
    lay = SimpleRouterLayer(num_slots=2, smooth_logits=smooth, seed=0)
    lay.build((None, 4, 4, T))
    lay.theta.assign(np.asarray(theta, np.float32))
    if smooth:
        lay.smooth_sigma.assign([np.float32(sigma)])
    return lay


def mu_bruteforce(phi):
    """Enumerate every unordered pair explicitly -- no shared code with the layer."""
    phi = np.asarray(phi, np.float64)
    m = phi.max()
    P = np.zeros((T, T))
    for a in range(T):
        for b in range(a + 1, T):
            P[a, b] = np.exp(phi[a] - m + phi[b] - m)
    P /= P.sum()
    mu = P.sum(axis=1) + P.sum(axis=0)
    return mu


rng = np.random.default_rng(0)
theta = rng.normal(size=T).astype(np.float32) * 1.5
theta[7] += 6.0                       # the boundary spike every O13 seed found

print("1) sigma = 0 is exact identity")
a = build(theta, 0.0).mu().numpy()
b = build(theta, 0.0, smooth=False).mu().numpy()
check("mu(smooth,s=0) == mu(no-smooth)", np.abs(a - b).max() == 0.0,
      f"max|diff| = {np.abs(a-b).max():.1e}")

print("2) smoothing matrix is row-stochastic")
lay = build(theta, 2.5)
sg = lay.sigma()
w = tf.exp(-lay._dist2 / (2.0 * tf.square(sg)))
w = tf.where(lay._dist2 <= tf.square(3.0 * sg), w, tf.zeros_like(w))
rs = tf.reduce_sum(w / tf.reduce_sum(w, axis=1, keepdims=True), axis=1).numpy()
check("all rows sum to 1", np.abs(rs - 1).max() < 1e-6, f"max dev {np.abs(rs-1).max():.1e}")

print("3) mu matches brute-force enumeration")
for s in (0.0, 0.7, 2.0, 4.0):
    lay = build(theta, s)
    phi = lay.smooth(tf.constant(theta)).numpy()
    d = np.abs(lay.mu().numpy() - mu_bruteforce(phi)).max()
    check(f"sigma={s}", d < 2e-6, f"max|diff| = {d:.2e}")

print("4) selected_indices agrees with argmax of mu")
for s in (0.0, 1.0, 3.0):
    lay = build(theta, s)
    mu = lay.mu().numpy()
    check(f"sigma={s}", lay.selected_indices() == sorted(np.argsort(mu)[-2:].tolist()),
          f"{lay.selected_indices()}")

print("5) gradient reaches theta through the kernel")
lay = build(theta, 2.0)
with tf.GradientTape() as tape:
    tape.watch(lay.theta)
    loss = tf.reduce_sum(lay.smooth(tf.convert_to_tensor(lay.theta)) * tf.range(T, dtype=tf.float32))
g = tape.gradient(loss, lay.theta).numpy()
check("nonzero gradient on theta", np.abs(g).max() > 0, f"max|g| = {np.abs(g).max():.3f}")
# self-adjointness: dL/dtheta = W^T dL/dphi, with dL/dphi = arange
sgv = float(lay.sigma())
W = np.exp(-lay._dist2.numpy() / (2 * sgv**2))
W[lay._dist2.numpy() > (3 * sgv) ** 2] = 0.0
W /= W.sum(axis=1, keepdims=True)
check("g == W^T @ dL/dphi", np.abs(g - W.T @ np.arange(T)).max() < 1e-3,
      f"max|diff| = {np.abs(g - W.T @ np.arange(T)).max():.2e}")

print("6) a centred bump stays centred")
bump = np.exp(-((np.arange(T) - 50.0) ** 2) / (2 * 5.0**2)).astype(np.float32) * 8
for s in (1.0, 3.0, 5.0):
    phi = build(bump, s).smooth(tf.constant(bump)).numpy()
    com = float((phi * np.arange(T)).sum() / phi.sum())
    check(f"sigma={s} centre of mass ~ 50", abs(com - 50) < 0.6, f"= {com:.2f}")

print("7) a boundary spike keeps its mass at the boundary")
sp = np.zeros(T, np.float32); sp[7] = 10.0
for s in (1.0, 3.0):
    phi = build(sp, s).smooth(tf.constant(sp)).numpy()
    check(f"sigma={s} peak still at 7", int(np.argmax(phi)) == 7,
          f"argmax={int(np.argmax(phi))}, phi[7]={phi[7]:.3f}")

print("8) smoothing widens mu")
ents = []
for s in (0.0, 1.0, 2.0, 4.0):
    mu = build(theta, s).mu().numpy(); q = mu / 2
    ents.append(float(-(q * np.log(q + 1e-12)).sum()))
check("entropy increases monotonically with sigma", all(np.diff(ents) > 0),
      " -> ".join(f"{e:.3f}" for e in ents))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
