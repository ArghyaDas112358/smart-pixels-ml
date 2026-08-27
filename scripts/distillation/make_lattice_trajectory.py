"""
Pair TRAJECTORY through the 2-D lattice: the sampled pair joined epoch-to-epoch
as a continuous line, so the router's walk reads as a path rather than a cloud.

Two panels per seed:
  left  -- the raw per-epoch pair, smoothed (rolling mean, k=51) into a single
           trajectory line, colour-graded by epoch with arrowheads every ~600
           epochs; the raw samples sit underneath as faint dots so the smoothing
           never hides the true scatter.
  right -- the same trajectory in (mid-point, separation) coordinates, which is
           the physically meaningful pair of axes: WHERE in the waveform the two
           taps sit, and HOW FAR APART they are (the lever arm).

Start (o) and latest (star) are marked in both panels.

  CUDA_VISIBLE_DEVICES='' python make_lattice_trajectory.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from seed_palette import color as seed_color

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "o21v2a2_pairlattice")
OUT = os.path.join(R, "runs", "perf_plots_o21v2", "pair_trajectory.png")
SEEDS = [22042, 22142, 22242]
K = 51                                     # rolling window (epochs)


def pairs(seed):
    p = os.path.join(RUN, f"seed_{seed}", "router_epochs.csv")
    ep, i1, i2 = [], [], []
    for r in csv.DictReader(open(p)):
        try:
            ep.append(int(r["epoch"])); i1.append(int(r["i1"])); i2.append(int(r["i2"]))
        except (ValueError, KeyError):
            continue
    ep, i1, i2 = np.array(ep), np.array(i1), np.array(i2)
    return ep, np.minimum(i1, i2), np.maximum(i1, i2)


def roll(v, k=K):
    if len(v) < k:
        return v.astype(float)
    return np.convolve(v, np.ones(k) / k, mode="valid")


def graded_line(ax, x, y, ep, cmap="viridis", lw=2.6):
    """Trajectory as a LineCollection so colour can encode epoch along the path."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(ep.min(), ep.max()),
                        linewidths=lw, alpha=0.95)
    lc.set_array(ep[:-1])
    ax.add_collection(lc)
    return lc


fig, axes = plt.subplots(2, 3, figsize=(19.2, 11.4))
for col, s in enumerate(SEEDS):
    ep, lo, hi = pairs(s)
    c = seed_color(s)
    xs, ys = roll(lo), roll(hi)
    eps = ep[K // 2: K // 2 + len(xs)] if len(ep) >= K else ep

    # --- top: the lattice plane -------------------------------------------
    ax = axes[0][col]
    ax.plot([0, 100], [0, 100], color="#9aa4ad", lw=0.9, ls=":")
    ax.scatter(lo, hi, s=6, alpha=0.12, color=c, linewidths=0)      # raw samples
    lc = graded_line(ax, xs, ys, eps)
    step = max(1, len(xs) // 6)
    for k in range(step, len(xs) - 1, step):                         # direction arrows
        ax.annotate("", xy=(xs[k + 1], ys[k + 1]), xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="-|>", color="#111318", lw=1.1, alpha=0.75))
    # start marker on the RAW first sample so it agrees with the panel title
    ax.plot(lo[0], hi[0], marker="o", ms=11, color="white", mec="#111318", mew=1.6, zorder=5)
    ax.plot(lo[-1], hi[-1], marker="*", ms=18, color="#e2001a", mec="white", mew=0.8, zorder=6)
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_xlabel(r"earlier slice $t_i$"); ax.grid(alpha=0.15, lw=0.4)
    ax.set_title(f"seed {s}:  ({lo[0]}, {hi[0]})  →  ({lo[-1]}, {hi[-1]})",
                 color=c, fontsize=13, fontweight="bold")
    if col == 0:
        ax.set_ylabel(r"later slice $t_j$")

    # --- bottom: physics coordinates --------------------------------------
    ax2 = axes[1][col]
    mid_raw, sep_raw = 0.5 * (lo + hi), hi - lo
    mid, sep = roll(mid_raw), roll(sep_raw)
    ax2.scatter(mid_raw, sep_raw, s=6, alpha=0.12, color=c, linewidths=0)
    graded_line(ax2, mid, sep, eps)
    ax2.plot(mid_raw[0], sep_raw[0], marker="o", ms=11, color="white", mec="#111318", mew=1.6, zorder=5)
    ax2.plot(mid_raw[-1], sep_raw[-1], marker="*", ms=18, color="#e2001a", mec="white", mew=0.8, zorder=6)
    ax2.axhline(0, color="#9aa4ad", lw=0.9, ls=":")
    ax2.set_xlim(-2, 102); ax2.set_ylim(-2, 102)
    ax2.set_xlabel(r"pair mid-point  $(t_i + t_j)/2$"); ax2.grid(alpha=0.15, lw=0.4)
    if col == 0:
        ax2.set_ylabel(r"separation  $|t_j - t_i|$   (lever arm)")

cb = fig.colorbar(lc, ax=axes, fraction=0.014, pad=0.012)
cb.set_label("epoch")
fig.suptitle("O21a.v2-cold: the router's trajectory through the pair lattice "
             "(line = rolling mean, faint dots = per-epoch samples, ○ start, ★ latest)",
             fontsize=14)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
