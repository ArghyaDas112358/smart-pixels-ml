"""
Time evolution of the router occupancy for the O4 (beta-annealing) seeds.

IMPORTANT — this is NOT the same data as the O11 evolution figure. The full
101-slice mu history lives in `theta_mu.npz`, which the trainer only writes in
on_train_end; the O4 runs are still in flight, so no npz exists yet. What IS
logged, every single epoch, is the TOP-5 mu (index + value) in
router_epochs.csv. This plot draws those five points per epoch.

So: finer time resolution than the O11 figure (every epoch vs every 25), but
only the five heaviest slices per epoch instead of all 101. Where the O11 panels
show a diffuse band, these show only its crest. The subtitle on the slide says
so; do not present the two as equivalent.

No final pair is marked. Mid-flight the argmax hops almost every epoch (86-117
distinct pairs per 200 epochs), so a star would imply a decision that has not
been made.

  CUDA_VISIBLE_DEVICES='' python make_mu_evolution_o4.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from seed_palette import ACCENT_O4, SEEDS_O4

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "simplerouter_mdmm_o4beta")
OUT = os.path.join(R, "runs", "perf_plots_o4beta", "mu_evolution_o4.png")
NSLICE = 101


def cmap_for(hexc):
    r, g, b = to_rgb(hexc)
    mid = (r + (1 - r) * 0.45, g + (1 - g) * 0.45, b + (1 - b) * 0.45)
    dark = (r * 0.55, g * 0.55, b * 0.55)
    return LinearSegmentedColormap.from_list("m", ["#ffffff", mid, hexc, dark], N=256)


def load(seed):
    """epochs, top-5 slice indices, top-5 mu values, and the mu entropy."""
    ep, idx, val, ent = [], [], [], []
    with open(os.path.join(RUN, f"seed_{seed}", "router_epochs.csv")) as f:
        for r in csv.DictReader(f):
            try:
                i = [int(x) for x in r["mu_top5_idx"].split(";")]
                v = [float(x) for x in r["mu_top5_val"].split(";")]
            except (ValueError, KeyError):
                continue
            ep.append(int(r["epoch"])); idx.append(i); val.append(v)
            ent.append(float(r["mu_entropy"]))
    return np.array(ep), np.array(idx), np.array(val), np.array(ent)


DATA = {s: load(s) for s in SEEDS_O4}
vmax = max(float(v.max()) for _, _, v, _ in DATA.values())
xmax = max(int(e.max()) for e, _, _, _ in DATA.values())

fig, axes = plt.subplots(2, 2, figsize=(18, 7.6), sharex=True, sharey=True)

for ax, s in zip(axes.ravel(), SEEDS_O4):
    ep, idx, val, ent = DATA[s]
    c = ACCENT_O4[s]
    # One point per (epoch, top-5 rank); colour = that slice's mu.
    xs = np.repeat(ep, idx.shape[1])
    sc = ax.scatter(xs, idx.ravel(), c=val.ravel(), cmap=cmap_for(c),
                    vmin=0.0, vmax=vmax, s=2.2, marker="s", linewidths=0)

    ax.set_title(f"seed {s}   ep {ep[-1]:,}/5000   H(μ) {ent[-1]:.2f}   "
                 f"top-5 μ {val[-1].sum():.2f} / 2.0", fontsize=13, color=c)
    ax.set_ylim(0, NSLICE - 1)
    ax.set_xlim(0, xmax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # beta ramp window: 1 -> 30 starting at epoch 1000 over 2000 epochs
    ax.axvspan(1000, 3000, color="#111318", alpha=0.05, lw=0, zorder=0)

    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.012)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)

axes[0][0].text(2000, 96, "β ramp 1→30", fontsize=10, color="#5b6470",
                ha="center", va="top")
for ax in axes[1]:
    ax.set_xlabel("epoch", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("time slice index (of 101)", fontsize=12)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}   vmax={vmax:.4f}  epochs to {xmax}")
for s in SEEDS_O4:
    ep, idx, val, ent = DATA[s]
    print(f"  seed {s}: ep {ep[-1]}, H(mu) {ent[-1]:.3f}, top-5 mu {val[-1].sum():.3f}")
