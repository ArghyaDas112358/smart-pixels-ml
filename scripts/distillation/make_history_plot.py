"""
Training-history plot: val NLL vs epoch for every completed O11 seed, with the
unconstrained reference band drawn behind them.

The point of the figure is that the constrained runs sit BELOW the band the
unconstrained (angle-collapsed) runs reached -- i.e. we did not pay NLL for
keeping the angles alive -- and that seed 3042 sat flat for ~4000 epochs before
escaping, which is why "it looks dead" is not a reason to kill a run.

  CUDA_VISIBLE_DEVICES='' python make_history_plot.py
"""
import os, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "simplerouter_mdmm_discovery")
OUT = os.path.join(R, "runs", "perf_plots_mdmm", "history_o11.png")
SEEDS = [4042, 1042, 2042, 3042]
# best/worst best_val over the six unconstrained SIMPLE seeds (angles collapsed)
UNCON = (-29124.0, -28004.0)


def series(seed):
    p = os.path.join(RUN, f"seed_{seed}", "history.csv")
    ep, vl = [], []
    for r in csv.DictReader(open(p)):
        v = r.get("val_loss", "")
        if v in ("", "nan", "val_loss"):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if np.isfinite(f):
            ep.append(int(r["epoch"])); vl.append(f)
    return np.array(ep), np.array(vl)


fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axhspan(UNCON[0], UNCON[1], color="0.75", alpha=0.45, zorder=0)
ax.text(4900, (UNCON[0] + UNCON[1]) / 2, " unconstrained\n (angles collapsed)",
        va="center", ha="right", fontsize=9, color="0.35")

colors = {4042: "#1d4ed8", 1042: "#0f766e", 2042: "#15803d", 3042: "#b45309"}
for s in SEEDS:
    ep, vl = series(s)
    if len(ep) == 0:
        continue
    # running best -- the quantity actually quoted everywhere else
    best = np.minimum.accumulate(vl)
    rj = os.path.join(RUN, f"seed_{s}", "result.json")
    lbl = f"seed {s}"
    if os.path.exists(rj):
        d = json.load(open(rj))
        lbl = f"seed {s}  {d['final_indices']}  {round(d['best_val_loss']):,}"
    ax.plot(ep, best, lw=1.9, color=colors.get(s, "k"), label=lbl, zorder=3)

ax.set_xlabel("epoch"); ax.set_ylabel("best val NLL so far")
ax.set_xlim(0, 5000); ax.set_ylim(-31000, -18000)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_title("O11: 5000-epoch runs vs the unconstrained band", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
