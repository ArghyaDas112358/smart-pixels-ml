"""
Animated GIFs of the pair lattice filling in over training — one per seed.

WHAT IS ANIMATED (and what is not): psi is only persisted in the final
checkpoint, so there is no per-epoch history of the LEARNED distribution to
replay. What we do have every epoch is the pair the router actually SAMPLED,
so each frame shows a rolling window of those samples as a 2-D occupancy
histogram -- the empirical shadow of the distribution at that moment. The final
frame is held next to the true softmax(psi) surface on the slide, so the two
are never confused.

Frame f covers epochs [e - WIN, e]; the current pair is starred, the running
trail is drawn faintly, and the epoch is printed in the corner.

  CUDA_VISIBLE_DEVICES='' python make_lattice_gifs.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from seed_palette import color as seed_color

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "o21v2a2_pairlattice")
OUT = os.path.join(R, "runs", "perf_plots_o21v2")
SEEDS = [22042, 22142, 22242]
WIN = 250          # epochs per frame window
FRAMES = 60        # frames per gif
BIN = 2            # lattice bin width (slices)
os.makedirs(OUT, exist_ok=True)


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


edges = np.arange(0, 102, BIN)
centers = edges[:-1]

for s in SEEDS:
    ep, lo, hi = pairs(s)
    c = seed_color(s)
    cmap = LinearSegmentedColormap.from_list("m", ["#ffffff", c])
    # from epoch 0 -- early frames pool whatever samples exist so far
    marks = np.linspace(ep.min(), ep.max(), FRAMES).astype(int)

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.plot([0, 100], [0, 100], color="#9aa4ad", lw=0.9, ls=":", zorder=1)
    mesh = ax.pcolormesh(centers, centers, np.zeros((len(centers), len(centers))),
                         cmap=cmap, vmin=0, vmax=1, shading="auto", zorder=2)
    trail, = ax.plot([], [], lw=0.9, color=c, alpha=0.35, zorder=3)
    star, = ax.plot([], [], marker="*", ms=17, color="#e2001a", mec="white", mew=0.8,
                    ls="none", zorder=5)
    label = ax.text(0.03, 0.965, "", transform=ax.transAxes, fontsize=12,
                    fontweight="bold", va="top", color=c)
    sub = ax.text(0.03, 0.915, "", transform=ax.transAxes, fontsize=10,
                  va="top", color="#5b6470")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel(r"earlier slice $t_i$"); ax.set_ylabel(r"later slice $t_j$")
    ax.set_title(f"seed {s} — sampled pairs, {WIN}-epoch window",
                 color=c, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.15, lw=0.4)
    fig.tight_layout()

    def update(k):
        e = marks[k]
        m = (ep > e - WIN) & (ep <= e)
        if not m.any():
            m = ep <= max(e, ep.min())
        h, _, _ = np.histogram2d(lo[m], hi[m], bins=[edges, edges])
        mesh.set_array(h.T.ravel())
        mesh.set_clim(0, max(1.0, h.max()))
        upto = ep <= e
        trail.set_data(lo[upto], hi[upto])
        if m.any():
            star.set_data([lo[m][-1]], [hi[m][-1]])
        label.set_text(f"epoch {e}")
        sep = (hi[m][-1] - lo[m][-1]) if m.any() else 0
        sub.set_text(f"current pair ({lo[m][-1]}, {hi[m][-1]})   |Δt| = {sep}" if m.any() else "")
        return mesh, trail, star, label, sub

    anim = FuncAnimation(fig, update, frames=len(marks), blit=False)
    path = os.path.join(OUT, f"lattice_evolution_{s}.gif")
    anim.save(path, writer=PillowWriter(fps=6))
    plt.close(fig)
    print("wrote", path, f"({os.path.getsize(path)/1e6:.1f} MB)")
