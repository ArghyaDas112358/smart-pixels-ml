"""
O21 training-history overlay: best val NLL vs epoch for the three mechanism
arms (pair-lattice / two-router / UCB bandit), one curve each, with the O17
frozen-[11,26] gold standard and O20's adjacent-pair best as reference lines.

The three arms live in SEPARATE run dirs (one seed each), which is why this is
not a make_history_plot.py env invocation.

  CUDA_VISIBLE_DEVICES='' python make_history_o21.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
ARMS = [
    ("o21a_pairlattice", 21042, "a: pair-lattice (2D kernel)", "#7c3aed"),
    ("o21b_tworouter",   21142, "b: two routers (anti-overlap)", "#0891b2"),
    ("o21c_ucb",         21242, "c: UCB bandit",                 "#16a34a"),
]
O17_BEST = -38994.0   # frozen [11,26] deep-head best (seed 41942)
O20_BEST = -36004.0   # warm free router, adjacent (12,14) (seed 20142)
OUT = os.path.join(R, "runs", "perf_plots_o21", "history_o21.png")

fig, ax = plt.subplots(figsize=(9, 5.2))
xmax = 0
for run, seed, label, colr in ARMS:
    p = os.path.join(R, "runs", run, f"seed_{seed}", "history.csv")
    ep, vl = [], []
    for r in csv.DictReader(open(p)):
        v = r.get("val_loss", "")
        try:
            f = float(v)
        except ValueError:
            continue
        if np.isfinite(f):
            ep.append(int(r["epoch"])); vl.append(f)
    ep, vl = np.array(ep), np.array(vl)
    best = np.minimum.accumulate(vl)
    ax.plot(ep, best, lw=2.0, color=colr, label=f"{label}  best {best.min():,.0f}", zorder=3)
    xmax = max(xmax, ep.max())

ax.axhline(O17_BEST, color="#b45309", ls="--", lw=1.4, zorder=2)
ax.text(xmax, O17_BEST + 150, "O17 imposed [11,26] best", color="#b45309",
        fontsize=9, ha="right", va="bottom")
ax.axhline(O20_BEST, color="#6b7280", ls=":", lw=1.4, zorder=2)
ax.text(xmax, O20_BEST + 150, "O20 adjacent-pair best", color="#6b7280",
        fontsize=9, ha="right", va="bottom")

ax.set_xlabel("epoch"); ax.set_ylabel("best val NLL so far")
ax.set_xlim(0, max(xmax, 1500)); ax.set_ylim(-41500, -28000)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_title("O21: three search mechanisms vs the adjacency trap (warm start, one seed each)",
             fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
