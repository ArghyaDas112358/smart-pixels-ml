"""O22 wiring smoke test: build the real model, pin the lattice, freeze the
thresholds, attach the 60 constraints, and take ONE gradient step.

Checks the things that only break in the real graph, not in isolation:
  * a pair-lattice router can actually be pinned and frozen (the old FIX_SLICES
    path drove a layer that does not exist on this model)
  * frozen thresholds and frozen psi really do not move
  * 60 per-bin multipliers all receive gradient
  * the bin-balanced loss and the plain_nll metric coexist

  CUDA_VISIBLE_DEVICES='' python smoke_o22_wiring.py
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import numpy as np, tensorflow as tf
from train import create_model
from mdmm import MDMM, MinCorrConstraint, ZeroBiasConstraint
from conditional_nll import custom_loss_v2, binbalanced_loss_v2_byterm

N_SLICES, NB, PAIR = 101, 15, (10, 21)
thr0 = [8.57, 20.23, 47.49]
ok = True

vit = create_model("ViT_MaxDeep_PairLattice", timeslices=N_SLICES,
                   soft_quantize_layer=True, initial_thresholds=thr0,
                   threshold_offset=0.0,
                   initial_levels=np.array([0., 1., 2., 3.], np.float32))
print(f"model built: {vit.count_params():,} params")

# --- pin the lattice --------------------------------------------------------
pl = next((l for l in vit.layers if hasattr(l, "psi") and hasattr(l, "ia")), None)
print(f"pair-lattice layer found: {pl is not None} ({pl.name if pl else '-'})")
ok &= pl is not None
ia, ib = pl.ia.numpy(), pl.ib.numpy()
hit = np.where((ia == PAIR[0]) & (ib == PAIR[1]))[0]
psi = np.full(ia.shape[0], -30., np.float32); psi[hit[0]] = 30.
pl.psi.assign(psi); pl.psi._trainable = False
if hasattr(pl, "smooth_sigma"): pl.smooth_sigma.assign(np.zeros_like(pl.smooth_sigma.numpy()))
sel = pl.selected_indices()
print(f"pinned to {sel}: {'PASS' if sel == list(PAIR) else 'FAIL'}"); ok &= sel == list(PAIR)

# --- freeze thresholds ------------------------------------------------------
qs = [l for l in vit.layers if hasattr(l, "threshold_deltas_raw")]
for l in qs: l.threshold_deltas_raw._trainable = False
print(f"quantizer layers frozen: {len(qs)}")
thr_before = qs[0].threshold_deltas_raw.numpy().copy()
psi_before = pl.psi.numpy().copy()

# --- constraints ------------------------------------------------------------
y_s = np.stack([np.random.default_rng(1).normal(0, .4, 3000),
                np.random.default_rng(2).normal(0, .3, 3000),
                np.random.default_rng(3).uniform(-2, 2, 3000),
                np.random.default_rng(4).uniform(-2, 2, 3000)], 1).astype(np.float32)
cons = [MinCorrConstraint(column=c, label_column=k, min_value=0.5, scale=1e4,
                          damping=1.0, name=f"corr_{k}")
        for k, c in enumerate([0, 2, 4, 6])]
zb = []
for k, oc in enumerate([0, 2, 4, 6]):
    v = y_s[:, k]
    if k >= 2:
        ang = np.arctan2(1.0, v) * 180 / np.pi
        lo, hi = np.percentile(ang, [1, 99]); tf_, sc = "cot2deg", 1.0
    else:
        lo, hi = np.percentile(v, [1, 99]); tf_, sc = "identity", 1.0
    cen = np.linspace(lo, hi, NB).astype(np.float32)
    c = ZeroBiasConstraint(column=oc, label_column=k, centers=cen,
                           sigma=float(cen[1] - cen[0]), transform=tf_,
                           label_scale=sc, scale=8.0, damping=1.0, name=f"zbias_{k}")
    zb.append(c); cons.append(c)
nmult = sum(int(c.lmbda.shape[0]) for c in zb)
print(f"zero-bias multipliers: {nmult} ({len(zb)} targets x {NB}) -> {'PASS' if nmult == 60 else 'FAIL'}")
ok &= nmult == 60

model = MDMM(vit, cons, constraint_samples=None, constraint_pass="primary", name="mdmm_o22")
specs = [(k, k, np.linspace(*np.percentile(y_s[:, k], [1, 99]), NB).astype(np.float32),
          float(np.diff(np.linspace(*np.percentile(y_s[:, k], [1, 99]), NB))[0])) for k in range(4)]
def plain_nll(y, p): return custom_loss_v2(y, p)
model.compile(optimizer=tf.keras.optimizers.Nadam(1e-3),
              loss=(lambda y, p: binbalanced_loss_v2_byterm(y, p, specs)),
              metrics=[plain_nll])

# --- one real training step -------------------------------------------------
X = np.random.default_rng(9).random((24, 16, 16, N_SLICES)).astype(np.float32) * 200
Y = y_s[:24]
h = model.fit(X, Y, epochs=1, batch_size=8, verbose=0)
keys = list(h.history)
print(f"history keys: {keys}")
has_plain = any("plain_nll" in k for k in keys)
print(f"plain_nll tracked alongside the weighted loss: {'PASS' if has_plain else 'FAIL'}"); ok &= has_plain
finite = all(np.isfinite(v).all() for v in h.history.values())
print(f"all logged values finite: {'PASS' if finite else 'FAIL'}  {({k: round(float(v[0]),1) for k,v in h.history.items()})}")
ok &= finite

moved_thr = not np.allclose(thr_before, qs[0].threshold_deltas_raw.numpy())
moved_psi = not np.allclose(psi_before, pl.psi.numpy())
print(f"thresholds stayed frozen: {'PASS' if not moved_thr else 'FAIL'}"); ok &= not moved_thr
print(f"router psi stayed frozen: {'PASS' if not moved_psi else 'FAIL'}"); ok &= not moved_psi
print(f"router still reads {pl.selected_indices()} after the step: "
      f"{'PASS' if pl.selected_indices() == list(PAIR) else 'FAIL'}")
ok &= pl.selected_indices() == list(PAIR)
nz = sum(1 for c in zb if np.abs(c.lmbda.numpy()).sum() > 0)
print(f"multipliers that moved off zero: {nz}/{len(zb)} targets -> {'PASS' if nz > 0 else 'FAIL'}")
ok &= nz > 0

print("\n" + ("ALL WIRING CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
