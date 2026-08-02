"""
Adversarial smoke-test suite for SimpleRouterLayer (k=2 pair router).

Standalone, CPU-only. Written to the CONTRACT + THE MATH ONLY -- it does not
import or inspect any private internals of the implementation; every expected
value is recomputed here independently (brute-force pair enumeration, numpy
marginals, finite-difference Jacobians).

Distribution under test:
    over all C(T,2) pairs {a,b},  p({a,b}) ~ exp(theta_a + theta_b).
    marginals mu_i = P(i selected),  sum(mu) == 2.
    exponential family in natural params theta, sufficient stat z (indicator):
        grad_theta log Z = mu ,  Hess = Cov(z)  ->  d mu_i / d theta_j = Cov_ij.

Run:
    python scripts/distillation/smoke_simplerouter.py
Exit code 1 if any numbered test fails. Prints 'SMOKE N/N PASS' at the end.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only, before importing TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
import itertools
import numpy as np

HELPERS = "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers"
sys.path.insert(0, HELPERS)

import tensorflow as tf
from SimpleRouterLayer import SimpleRouterLayer


# --------------------------------------------------------------------------- #
# Independent reference math (numpy, float64) -- derived from the contract only #
# --------------------------------------------------------------------------- #
def ref_mu(theta):
    """Exact marginals mu_i = P(slice i in sampled pair)."""
    theta = np.asarray(theta, dtype=np.float64)
    m = theta.max()
    w = np.exp(theta - m)
    S1 = w.sum()
    S2 = (w * w).sum()
    Z = 0.5 * (S1 * S1 - S2)
    return w * (S1 - w) / Z


def ref_cov(theta):
    """Cov(z): diag mu_i(1-mu_i); off-diag p_ij - mu_i mu_j, p_ij = w_i w_j / Z."""
    theta = np.asarray(theta, dtype=np.float64)
    m = theta.max()
    w = np.exp(theta - m)
    S1 = w.sum()
    S2 = (w * w).sum()
    Z = 0.5 * (S1 * S1 - S2)
    mu = w * (S1 - w) / Z
    P = np.outer(w, w) / Z                 # p_ij for i != j
    cov = P - np.outer(mu, mu)
    np.fill_diagonal(cov, mu * (1.0 - mu))
    return cov


def brute_force_pair_probs(theta):
    """Enumerate all C(T,2) pairs -> dict {(a,b): p({a,b})} and marginals."""
    theta = np.asarray(theta, dtype=np.float64)
    T = theta.shape[0]
    pairs = list(itertools.combinations(range(T), 2))
    unn = np.array([np.exp(theta[a] + theta[b]) for a, b in pairs])
    p = unn / unn.sum()
    probs = {pr: float(pv) for pr, pv in zip(pairs, p)}
    marg = np.zeros(T)
    for (a, b), pv in probs.items():
        marg[a] += pv
        marg[b] += pv
    return probs, marg


def channel_index_input(B, H, W, T):
    """x[...,t] == t so that a training/eval forward pass reveals the picked
    slice indices directly through the output channel values."""
    x = np.zeros((B, H, W, T), dtype=np.float32)
    for t in range(T):
        x[..., t] = float(t)
    return tf.constant(x)


def recover_pair(y):
    """Given a forward output built from channel_index_input, return (a, b)."""
    yn = y.numpy()
    a = int(round(float(yn[..., 0].flat[0])))
    b = int(round(float(yn[..., 1].flat[0])))
    return a, b


def build_layer(T, theta=None, **kw):
    l = SimpleRouterLayer(num_slots=2, **kw)
    l.build((None, 1, 1, T))
    if theta is not None:
        l.theta.assign(np.asarray(theta, dtype=np.float32))
    return l


# --------------------------------------------------------------------------- #
# Test harness                                                                 #
# --------------------------------------------------------------------------- #
_results = []


def record(num, name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] test {num}: {name}" + (f"  ({detail})" if detail else ""))
    _results.append(bool(ok))


def run(num, name, fn):
    try:
        fn(num, name)
    except Exception as e:  # noqa: BLE001 -- adversarial: any throw is a FAIL
        import traceback
        record(num, name, False, f"exception {type(e).__name__}: {e}")
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# 1. mu() vs brute-force enumeration + sum(mu) == 2, incl. real 101-slice shape #
# --------------------------------------------------------------------------- #
def test_1(num, name):
    ok_all = True
    details = []
    for T, seed in [(8, 0), (101, 1)]:
        rng = np.random.default_rng(seed)
        theta = rng.normal(0.0, 1.3, size=T)
        l = build_layer(T, theta)
        mu_layer = np.asarray(l.mu_numpy(), dtype=np.float64)
        _, marg = brute_force_pair_probs(theta)
        close = np.allclose(mu_layer, marg, rtol=1e-5, atol=1e-7)
        sum2 = abs(mu_layer.sum() - 2.0) < 1e-5
        ok_all = ok_all and close and sum2 and mu_layer.shape == (T,)
        details.append(f"T={T} maxerr={np.max(np.abs(mu_layer-marg)):.2e} sum={mu_layer.sum():.6f}")
    record(num, name, ok_all, "; ".join(details))


# --------------------------------------------------------------------------- #
# 2. Empirical sampling frequencies vs enumerated p(S)                          #
# --------------------------------------------------------------------------- #
def test_2(num, name):
    T = 8
    rng = np.random.default_rng(7)
    theta = rng.normal(0.0, 0.8, size=T)
    l = build_layer(T, theta, seed=1234)
    probs, _ = brute_force_pair_probs(theta)

    tf.random.set_seed(1234)
    np.random.seed(1234)
    x = channel_index_input(1, 1, 1, T)
    N = 20000
    counts = {pr: 0 for pr in probs}
    for _ in range(N):
        y = l(x, training=True)
        counts[recover_pair(y)] += 1

    max_diff = 0.0
    for pr, pv in probs.items():
        emp = counts[pr] / N
        max_diff = max(max_diff, abs(emp - pv))
    record(num, name, max_diff < 0.015, f"max|emp-p|={max_diff:.4f} over {len(probs)} pairs, N={N}")


# --------------------------------------------------------------------------- #
# 3. Sampled pairs are always distinct and ascending (a < b)                    #
# --------------------------------------------------------------------------- #
def test_3(num, name):
    T = 8
    rng = np.random.default_rng(3)
    l = build_layer(T, rng.normal(0.0, 1.0, size=T), seed=99)
    tf.random.set_seed(99)
    x = channel_index_input(1, 1, 1, T)
    bad = 0
    for _ in range(3000):
        a, b = recover_pair(l(x, training=True))
        if not (0 <= a < b < T):
            bad += 1
    record(num, name, bad == 0, f"{bad} non-ascending/duplicate out of 3000")


# --------------------------------------------------------------------------- #
# 4. Covariance identity: FD d(mu)/d(theta) vs analytic Cov                     #
# --------------------------------------------------------------------------- #
def test_4(num, name):
    T = 8
    rng = np.random.default_rng(11)
    theta = rng.normal(0.0, 1.0, size=T).astype(np.float64)
    eps = 1e-4
    J = np.zeros((T, T))
    for j in range(T):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        J[:, j] = (ref_mu(tp) - ref_mu(tm)) / (2.0 * eps)   # d mu_i / d theta_j
    cov = ref_cov(theta)
    max_err = float(np.max(np.abs(J - cov)))
    # cross-check the layer's own mu against ref_mu too (same object under test)
    l = build_layer(T, theta)
    mu_layer = np.asarray(l.mu_numpy(), dtype=np.float64)
    mu_ok = np.allclose(mu_layer, ref_mu(theta), rtol=1e-5, atol=1e-7)
    record(num, name, (max_err < 1e-4) and mu_ok, f"max|FD-Cov|={max_err:.2e}")


# --------------------------------------------------------------------------- #
# 5. Gradient plumbing: grads wrt theta non-None, finite, sum ~ 0              #
# --------------------------------------------------------------------------- #
def test_5(num, name):
    T = 8
    rng = np.random.default_rng(5)
    l = build_layer(T, rng.normal(0.0, 1.0, size=T), seed=42)
    tf.random.set_seed(42)
    x = tf.constant(rng.normal(0.0, 1.0, size=(16, 4, 4, T)).astype(np.float32))
    with tf.GradientTape() as tape:
        y = l(x, training=True)
        loss = tf.reduce_mean(y * y)
    grads = tape.gradient(loss, l.trainable_variables)
    g = grads[0] if grads else None
    non_none = g is not None
    gn = g.numpy() if non_none else None
    finite = non_none and np.all(np.isfinite(gn))
    shape_ok = non_none and gn.shape == (T,)
    sum_zero = non_none and abs(float(gn.sum())) < 1e-4
    record(num, name, non_none and finite and shape_ok and sum_zero,
           f"non_none={non_none} finite={finite} sum={0.0 if gn is None else gn.sum():.2e}")


# --------------------------------------------------------------------------- #
# 6. Learning direction: reward content only slices 2 and 5 carry              #
# --------------------------------------------------------------------------- #
def test_6(num, name):
    T = 8
    tf.random.set_seed(2026)
    np.random.seed(2026)
    l = build_layer(T, np.zeros(T), seed=2026)
    x = np.zeros((16, 4, 4, T), dtype=np.float32)
    x[..., 2] = 1.0
    x[..., 5] = 1.0
    x = tf.constant(x)
    opt = tf.keras.optimizers.legacy.SGD(learning_rate=0.5)
    for _ in range(300):
        with tf.GradientTape() as tape:
            y = l(x, training=True)
            loss = -tf.reduce_mean(y)          # reward selecting high-content slices
        grads = tape.gradient(loss, l.trainable_variables)
        opt.apply_gradients(zip(grads, l.trainable_variables))
    theta = l.theta.numpy()
    top2 = set(int(i) for i in np.argsort(theta)[-2:])
    record(num, name, top2 == {2, 5}, f"top2={sorted(top2)} theta={np.round(theta,2).tolist()}")


# --------------------------------------------------------------------------- #
# 7. training=False determinism: gather ascending top-2; shape; channel order  #
#    + no visits increment; real-shape (2,16,16,101) forward                    #
# --------------------------------------------------------------------------- #
def test_7(num, name):
    T = 8
    theta = np.array([0.1, 5.0, 0.2, 0.3, 4.0, 0.05, 0.4, 0.2])  # top2 -> {1,4}
    l = build_layer(T, theta)
    # expected ascending top-2
    exp_i = sorted(int(i) for i in np.argsort(theta)[-2:])       # [1, 4]
    sel = [int(i) for i in np.asarray(l.selected_indices())]
    sel_ok = (sel == exp_i) and (sel[0] < sel[1])

    x = channel_index_input(4, 3, 3, T)
    l.reset_visits()
    v0 = l.visits.numpy().copy()
    y1 = l(x, training=False)
    y2 = l(x, training=False)
    v1 = l.visits.numpy()
    shape_ok = tuple(y1.shape) == (4, 3, 3, 2)
    # deterministic + equals gather of ascending top-2 (channel order ascending)
    expected = np.stack([x.numpy()[..., exp_i[0]], x.numpy()[..., exp_i[1]]], axis=-1)
    det_ok = np.array_equal(y1.numpy(), y2.numpy()) and np.allclose(y1.numpy(), expected)
    chan_order_ok = (float(y1.numpy()[..., 0].flat[0]) == exp_i[0] and
                     float(y1.numpy()[..., 1].flat[0]) == exp_i[1])
    no_visit = np.array_equal(v0, v1) and float(v1.sum()) == 0.0

    # compute_output_shape contract
    cos = tuple(l.compute_output_shape((None, 16, 16, T)))
    cos_ok = cos[-1] == 2 and cos[1:3] == (16, 16)

    # real-shape 101-slice eval forward
    lr = build_layer(101, np.random.default_rng(0).normal(0, 1, 101))
    yr = lr(tf.constant(np.random.default_rng(1).normal(0, 1, (2, 16, 16, 101)).astype(np.float32)),
            training=False)
    real_ok = tuple(yr.shape) == (2, 16, 16, 2)

    ok = sel_ok and shape_ok and det_ok and chan_order_ok and no_visit and cos_ok and real_ok
    record(num, name, ok,
           f"sel={sel} shape_ok={shape_ok} det={det_ok} chan_asc={chan_order_ok} "
           f"no_visit={no_visit} real={real_ok}")


# --------------------------------------------------------------------------- #
# 8. visits counter: +2 per training call at (a,b); reset_visits() zeroes it    #
# --------------------------------------------------------------------------- #
def test_8(num, name):
    T = 8
    rng = np.random.default_rng(8)
    l = build_layer(T, rng.normal(0, 1, T), seed=321)
    tf.random.set_seed(321)
    x = channel_index_input(1, 1, 1, T)

    l.reset_visits()
    ok = float(l.visits.numpy().sum()) == 0.0
    per_call_ok = True
    K = 50
    for _ in range(K):
        v_before = l.visits.numpy().copy()
        y = l(x, training=True)
        a, b = recover_pair(y)
        v_after = l.visits.numpy()
        diff = v_after - v_before
        # exactly two entries incremented by 1, at the sampled a and b
        if not (diff.sum() == 2.0 and diff[a] == 1.0 and diff[b] == 1.0 and
                np.count_nonzero(diff) == 2):
            per_call_ok = False
    total_ok = float(l.visits.numpy().sum()) == 2.0 * K
    l.reset_visits()
    reset_ok = float(l.visits.numpy().sum()) == 0.0
    record(num, name, ok and per_call_ok and total_ok and reset_ok,
           f"per_call={per_call_ok} total={total_ok} reset={reset_ok}")


# --------------------------------------------------------------------------- #
# 9. get_config round-trip                                                      #
# --------------------------------------------------------------------------- #
def test_9(num, name):
    l = SimpleRouterLayer(num_slots=2, logits_init_stddev=0.05, seed=777)
    cfg = l.get_config()
    l2 = SimpleRouterLayer.from_config(cfg)
    ok = (cfg.get("num_slots") == 2 and
          abs(float(cfg.get("logits_init_stddev")) - 0.05) < 1e-12 and
          cfg.get("seed") == 777 and
          l2.num_slots == 2 and
          abs(float(l2.logits_init_stddev) - 0.05) < 1e-12 and
          l2.seed == 777)
    # ensure the rebuilt layer actually functions
    l2.build((None, 1, 1, 8))
    _ = l2.mu_numpy()
    record(num, name, ok, f"cfg={{num_slots,logits_init_stddev,seed}} round-trips")


# --------------------------------------------------------------------------- #
def main():
    tests = [
        (1, "mu() == brute-force pair marginals; sum(mu)==2 (T=8 and T=101)", test_1),
        (2, "empirical pair frequencies match enumerated p(S) (N=20000)", test_2),
        (3, "sampled pairs distinct and ascending (a<b)", test_3),
        (4, "Cov identity: FD d(mu)/d(theta) == analytic Cov", test_4),
        (5, "grad plumbing: theta grad non-None, finite, sums to ~0", test_5),
        (6, "learning direction: rewards slices 2 & 5 -> theta top-2 == {2,5}", test_6),
        (7, "training=False deterministic gather of asc top-2; shape/order; no visits; real 101", test_7),
        (8, "visits += 2 per training call at (a,b); reset_visits() zeroes", test_8),
        (9, "get_config / from_config round-trip", test_9),
    ]
    for num, name, fn in tests:
        run(num, name, fn)

    passed = sum(_results)
    total = len(_results)
    print(f"SMOKE {passed}/{total} PASS")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
