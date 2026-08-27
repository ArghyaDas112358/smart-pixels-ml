"""
The learned pair distribution on a LINEAR colour scale (companion to the log
version in make_lattice_2d_plots.py).

Why both: the log scale shows the structure of the whole surface -- ridges,
edges, the excluded diagonal -- but visually flatters commitment, because two
decades of near-zero still look like colour. Linear shows what the router will
ACTUALLY draw: mass in proportion. If a panel is almost entirely dark with one
bright knot, that is the honest picture of a distribution that has (or has not)
concentrated.

Each panel is normalised to its OWN maximum, and the max probability is printed
in the title so the three seeds stay comparable despite that.

  CUDA_VISIBLE_DEVICES='' python make_lattice_final_linear.py
"""
import os, csv
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seed_palette import color as seed_color

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "o21v2a2_pairlattice")
OUT = os.path.join(R, "runs", "perf_plots_o21v2", "lattice_final_linear.png")
SEEDS = [22042, 22142, 22242]
T = 101
IA, IB = np.triu_indices(T, k=1)
UNIFORM = 1.0 / len(IA)


def psi(seed):
    found = {}
    def visit(name, obj):
        if hasattr(obj, "shape") and tuple(obj.shape) == (len(IA),):
            found["psi"] = np.array(obj)
    with h5py.File(os.path.join(RUN, f"seed_{seed}", "last.weights.hdf5"), "r") as f:
        f.visititems(visit)
    return found.get("psi")


fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.4), sharex=True, sharey=True)
for ax, s in zip(axes, SEEDS):
    p_ = psi(s)
    w = np.exp(p_ - p_.max()); p = w / w.sum()
    grid = np.zeros((T, T)); grid[IA, IB] = p; grid[IB, IA] = p
    a, b = IA[np.argmax(p)], IB[np.argmax(p)]
    im = ax.pcolormesh(np.arange(T), np.arange(T), grid.T, cmap="magma",
                       vmin=0.0, vmax=p.max(), shading="nearest")
    ax.plot([0, 100], [0, 100], color="#9aa4ad", lw=0.8, ls=":")
    ax.plot(a, b, marker="*", color="#00e5ff", ms=14, mec="black", mew=0.5)
    ax.set_title(f"seed {s}: peak ({a}, {b})   p$_{{max}}$ = {p.max()*100:.2f}%   "
                 f"({p.max()/UNIFORM:.0f}× uniform)",
                 color=seed_color(s), fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$t_i$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="pair probability (linear)")
axes[0].set_ylabel(r"$t_j$")
fig.suptitle("O21a.v2-cold: the learned pair distribution on a LINEAR scale — "
             "mass in proportion, each panel scaled to its own peak", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
for s in SEEDS:
    p_ = psi(s); w = np.exp(p_ - p_.max()); p = w / w.sum()
    print(f"  seed {s}: p_max {p.max()*100:.2f}%  top-10 {np.sort(p)[::-1][:10].sum()*100:.1f}%")
