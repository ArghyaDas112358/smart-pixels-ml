"""O23: are the ceiling arms underperforming, or just young?

Plots best-so-far validation NLL against epoch for the parent and every ceiling
arm on one axis. The whole point is the overlay: if the arms sat BELOW the
parent's curve at matched epochs, something would be wrong with them (input
scaling, loss dialect, data path). They don't -- they sit on it.

Caveat annotated on the figure: the parent's logged column is val_loss from an
MDMM run, so it carries small constraint penalties on top of the NLL. At
convergence it reads -41,217 against the -40,162 measured by direct evaluation,
so the penalties are ~2% and the trajectory comparison holds.

  python make_o23_trajectory_plot.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
OUT = os.path.join(R, "runs", "perf_plots_o22"); os.makedirs(OUT, exist_ok=True)
PLAUSIBLE = 5e6


def best_so_far(path, col):
    if not os.path.exists(path): return None, None
    v = []
    with open(path) as fh:
        rd = csv.DictReader(fh); n = len(rd.fieldnames)
        c = col if col in rd.fieldnames else "val_loss"
        for row in rd:
            if None in row or len(row) != n or row.get(c) in (None, ""): continue
            try: x = float(row[c])
            except (TypeError, ValueError): continue
            if np.isfinite(x) and abs(x) < PLAUSIBLE: v.append(x)
    if not v: return None, None
    v = np.asarray(v)
    return np.arange(1, len(v) + 1), np.minimum.accumulate(v)


SERIES = [
    ("parent — 2 slices, 2-bit (O21v2a2)",
     f"{R}/runs/o21v2a2_pairlattice/seed_22042/history.csv", "val_loss", "#7c3aed", 2.4, "-"),
    ("O23 A — 20 slices float + slice-dropout 0.15",
     f"{R}/runs/o23_armA_n20_sd15/seed_30042/history.csv", "val_plain_nll", "#15803d", 1.9, "-"),
    ("O23 A — 20 slices float, no regularization",
     f"{R}/runs/o23_armA_n20/seed_30042/history.csv", "val_plain_nll", "#b45309", 1.7, "--"),
    ("O23 A — 101 slices float, no regularization",
     f"{R}/runs/o23_ceiling_armA/seed_30042/history.csv", "val_plain_nll", "#b91c1c", 1.7, ":"),
]

fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.0), dpi=115,
                       gridspec_kw={"width_ratios": [1.35, 1]})

for a, (xmax, title) in zip(ax, [(10000, "Full 10,000-epoch view — the parent's phase transition"),
                                 (800,   "Zoom: matched epochs — the arms LEAD early, then plateau")]):
    for lab, path, col, c, lw, ls in SERIES:
        e, v = best_so_far(path, col)
        if e is None: continue
        k = e <= xmax
        if k.sum() < 2: continue
        a.plot(e[k], v[k], color=c, lw=lw, ls=ls, label=lab if xmax == 10000 else None)
    a.set_xlabel("epoch", fontsize=11.5)
    a.set_ylabel("best-so-far validation NLL", fontsize=11.5)
    a.set_title(title, fontsize=12.5)
    a.grid(alpha=.22, lw=.6)
    a.set_xlim(0, xmax)
    # frame on the meaningful range: the first few epochs sit near +10,000 and
    # would otherwise compress the entire 10,000-epoch story into a thin band
    a.set_ylim(-43500, -15000) if xmax == 10000 else a.set_ylim(-29500, -17000)

ax[0].axvspan(2000, 3000, color="#7c3aed", alpha=.09, zorder=0)
ax[0].annotate("phase transition\n-31,348 → -39,162", xy=(2750, -35500),
               xytext=(4300, -30500), fontsize=11, color="#5b2566", weight="bold",
               arrowprops=dict(arrowstyle="->", color="#5b2566", lw=1.4))
ax[0].annotate("every O23 arm stops here", xy=(650, -26500), xytext=(1500, -21500),
               fontsize=10, color="#334155",
               arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2))
ax[0].legend(fontsize=9.2, loc="lower right", framealpha=.95)
ax[0].text(.985, .965,
           "parent column is val_loss from an MDMM run (~2% constraint penalty);\n"
           "arms are unweighted val_plain_nll",
           transform=ax[0].transAxes, ha="right", va="top", fontsize=8.4, color="#64748b")

ax[1].axvline(500, color="#334155", ls=":", lw=1.1, zorder=1)
ax[1].annotate("", xy=(500, -27300), xytext=(690, -24700),
               arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2))
ax[1].text(.985, .045,
           "ep 100   parent -19,730   arms -22,046 / -23,080   arms AHEAD\n"
           "ep 300   parent -25,635   arms -25,348 / -25,288   level\n"
           "ep 500   parent -27,590   arms -26,837 / -25,842   parent ahead",
           transform=ax[1].transAxes, ha="right", va="bottom", fontsize=9.2,
           family="monospace", color="#334155",
           bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cbd5e1", alpha=.95))

fig.suptitle("O23 — the ceiling arms are not underperforming, they are young",
             fontsize=14.5, y=.98)
fig.tight_layout(rect=[0, 0, 1, .95])
dst = os.path.join(OUT, "o23_trajectory.png")
fig.savefig(dst, facecolor="white"); plt.close(fig)
print("wrote", dst)

print(f"\n{'epoch':>7}" + "".join(f"{n.split('—')[0].strip()[:14]:>16}" for n, *_ in SERIES))
for ep in [100, 200, 300, 500, 700, 1000, 2000, 3000, 5000, 10000]:
    row = f"{ep:>7}"
    for lab, path, col, *_ in SERIES:
        e, v = best_so_far(path, col)
        row += f"{v[ep-1]:>16,.0f}" if e is not None and len(v) >= ep else f"{'—':>16}"
    print(row)
