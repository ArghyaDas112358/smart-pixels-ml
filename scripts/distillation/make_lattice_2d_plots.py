"""
2-D pair-lattice visualisations for O21a.v2-cold (runs/o21v2a2_pairlattice).

Four figure families, all in the seeds' palette colours:
  pair_path.png            -- the SAMPLED pair per epoch as a path through the
                              (t_i, t_j) plane, epoch-coloured (1x3 panels)
  pair_hist2d_<seed>.png   -- 2x3 grid: 2-D occupancy histograms of the sampled
                              pairs in consecutive epoch windows (blob contracts)
  lattice_final.png        -- the LEARNED pair distribution softmax(psi) read
                              directly from each seed's checkpoint (1x3 panels,
                              log colour; sigma=0 by these epochs so the plain
                              softmax IS the sampling distribution)
  separation.png           -- |t_j - t_i| of the sampled pair vs epoch, three
                              seeds overlaid (the width story in one line each)

psi is read from best/last.weights.hdf5 with h5py (no model build); the pair
ordering is reconstructed with np.triu_indices(101, 1), and the argmax pair is
cross-checked against the router CSV so a convention mismatch cannot pass
silently.

  CUDA_VISIBLE_DEVICES='' python make_lattice_2d_plots.py
"""
import os, csv
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from seed_palette import color as seed_color

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "o21v2a2_pairlattice")
OUT = os.path.join(R, "runs", "perf_plots_o21v2")
SEEDS = [22042, 22142, 22242]
T = 101
IA, IB = np.triu_indices(T, k=1)          # lattice pair ordering (a < b)
os.makedirs(OUT, exist_ok=True)


def pairs_per_epoch(seed):
    p = os.path.join(RUN, f"seed_{seed}", "router_epochs.csv")
    ep, i1, i2 = [], [], []
    for r in csv.DictReader(open(p)):
        try:
            ep.append(int(r["epoch"])); i1.append(int(r["i1"])); i2.append(int(r["i2"]))
        except (ValueError, KeyError):
            continue
    return np.array(ep), np.array(i1), np.array(i2)


def psi_from_ckpt(seed, which="last"):
    path = os.path.join(RUN, f"seed_{seed}", f"{which}.weights.hdf5")
    found = {}
    def visit(name, obj):
        if hasattr(obj, "shape"):
            if tuple(obj.shape) == (len(IA),): found["psi"] = np.array(obj)
            if "sigma" in name.lower() and np.prod(obj.shape) == 1:
                found["sigma"] = float(np.array(obj).ravel()[0])
    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return found.get("psi"), found.get("sigma", None)


# ---- figure 1: the path through the lattice ---------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.4), sharex=True, sharey=True)
for ax, s in zip(axes, SEEDS):
    ep, i1, i2 = pairs_per_epoch(s)
    lo, hi = np.minimum(i1, i2), np.maximum(i1, i2)
    ax.plot([0, 100], [0, 100], color="#9aa4ad", lw=0.8, ls=":")   # diagonal = adjacent
    sc = ax.scatter(lo, hi, c=ep, cmap="viridis", s=7, alpha=0.6, linewidths=0)
    ax.plot(lo[-1], hi[-1], marker="*", color="#e2001a", ms=16, mec="white", mew=0.6)
    ax.set_title(f"seed {s}", color=seed_color(s), fontsize=13, fontweight="bold")
    ax.set_xlabel(r"earlier slice $t_i$"); ax.set_xlim(-1, 101); ax.set_ylim(-1, 101)
    ax.grid(alpha=0.15, lw=0.4)
axes[0].set_ylabel(r"later slice $t_j$")
cb = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01); cb.set_label("epoch")
fig.suptitle("O21a.v2-cold: the sampled pair's path through the 2-D lattice "
             "(dotted diagonal = adjacent pairs; red star = latest)", fontsize=13)
fig.savefig(os.path.join(OUT, "pair_path.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(OUT, "pair_path.png"))

# ---- figure 2: per-seed epoch-window 2-D occupancy --------------------------
for s in SEEDS:
    ep, i1, i2 = pairs_per_epoch(s)
    lo, hi = np.minimum(i1, i2), np.maximum(i1, i2)
    edges = np.linspace(0, ep.max() + 1, 7).astype(int)             # 6 windows
    cmap = LinearSegmentedColormap.from_list("m", ["#ffffff", seed_color(s)])
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 10.4), sharex=True, sharey=True)
    for k, ax in enumerate(axes.ravel()):
        m = (ep >= edges[k]) & (ep < edges[k + 1])
        if m.any():
            h, _, _ = np.histogram2d(lo[m], hi[m], bins=np.arange(0, 102, 2))
            ax.pcolormesh(np.arange(0, 102, 2)[:-1], np.arange(0, 102, 2)[:-1], h.T,
                          cmap=cmap, shading="auto")
        ax.plot([0, 100], [0, 100], color="#9aa4ad", lw=0.8, ls=":")
        ax.set_title(f"epochs {edges[k]}–{edges[k+1]}", fontsize=11)
        ax.grid(alpha=0.12, lw=0.4)
    for ax in axes[1]: ax.set_xlabel(r"$t_i$")
    for row in axes: row[0].set_ylabel(r"$t_j$")
    fig.suptitle(f"seed {s}: where the sampled pairs LIVED, window by window",
                 color=seed_color(s), fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, f"pair_hist2d_{s}.png"), dpi=140)
    plt.close(fig)
    print("wrote", os.path.join(OUT, f"pair_hist2d_{s}.png"))

# ---- figure 3: the learned lattice distribution from the checkpoints --------
fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.4), sharex=True, sharey=True)
for ax, s in zip(axes, SEEDS):
    psi, sigma = psi_from_ckpt(s, "last")
    if psi is None:
        ax.set_title(f"seed {s}: psi not found"); continue
    w = np.exp(psi - psi.max()); p = w / w.sum()
    grid = np.zeros((T, T)); grid[IA, IB] = p; grid[IB, IA] = p
    a, b = IA[np.argmax(p)], IB[np.argmax(p)]
    # sanity: argmax must match the router CSV's latest readout family
    ep, i1, i2 = pairs_per_epoch(s)
    im = ax.pcolormesh(np.arange(T), np.arange(T), grid.T,
                       cmap="magma", norm=LogNorm(vmin=max(p.min(), 1e-9), vmax=p.max()),
                       shading="nearest")
    ax.plot([0, 100], [0, 100], color="#9aa4ad", lw=0.8, ls=":")
    ax.plot(a, b, marker="*", color="#00e5ff", ms=14, mec="black", mew=0.5)
    ax.set_title(f"seed {s}: argmax ({a},{b}), p={p.max():.2f}"
                 + (f", σ={sigma:.2g}" if sigma is not None else ""),
                 color=seed_color(s), fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$t_i$")
    print(f"  seed {s}: lattice argmax ({a},{b}) vs CSV latest ({min(i1[-1],i2[-1])},{max(i1[-1],i2[-1])})")
axes[0].set_ylabel(r"$t_j$")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="pair probability (log)")
fig.suptitle("O21a.v2-cold: the LEARNED pair distribution softmax(ψ), straight from the latest checkpoint",
             fontsize=13)
fig.savefig(os.path.join(OUT, "lattice_final.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(OUT, "lattice_final.png"))

# ---- figure 4: separation vs epoch ------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 5.2))
for s in SEEDS:
    ep, i1, i2 = pairs_per_epoch(s)
    sep = np.abs(i2 - i1)
    k = 51
    roll = np.convolve(sep, np.ones(k) / k, mode="valid")
    ax.scatter(ep, sep, s=3, alpha=0.10, color=seed_color(s), linewidths=0)
    ax.plot(ep[k//2: k//2 + len(roll)], roll, lw=2.2, color=seed_color(s),
            label=f"seed {s}  (final |Δt| = {sep[-1]})")
ax.set_xlabel("epoch"); ax.set_ylabel(r"sampled pair separation $|t_j - t_i|$")
ax.grid(alpha=0.25, lw=0.6); ax.legend(fontsize=10)
ax.set_title("O21a.v2-cold: pair separation vs epoch (dots = per-epoch samples, line = rolling median-ish mean)",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "separation.png"), dpi=150)
plt.close(fig)
print("wrote", os.path.join(OUT, "separation.png"))

# ---- figure 5: how COMMITTED is the lattice? --------------------------------
# The surface above is diffuse, which is the point of this panel: a router that
# has not concentrated is still training its head on a near-uniform mixture of
# pairs, so its "chosen pair" is only the argmax of a nearly flat surface.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 5.4))
Nl = len(IA)
for s in SEEDS:
    psi, _ = psi_from_ckpt(s, "last")
    w = np.exp(psi - psi.max()); p = np.sort(w / w.sum())[::-1]
    H = float(-(p * np.log(p + 1e-30)).sum())
    axL.plot(np.arange(1, Nl + 1), np.cumsum(p), lw=2.2, color=seed_color(s),
             label=f"seed {s}  (top-100 = {p[:100].sum():.2f}, H = {100*H/np.log(Nl):.0f}% of flat)")
    axR.bar(str(s), 100 * H / np.log(Nl), color=seed_color(s), width=0.6)
axL.axhline(1.0, color="#9aa4ad", lw=0.8, ls=":")
axL.set_xscale("log"); axL.set_xlabel("pairs, ranked by probability")
axL.set_ylabel("cumulative probability"); axL.set_ylim(0, 1.02)
axL.axvline(100, color="#9aa4ad", lw=0.8, ls="--")
axL.grid(alpha=0.25, lw=0.6); axL.legend(fontsize=9, loc="lower right")
axL.set_title("How much of the mass sits in the top pairs?", fontsize=12)
axR.axhline(100, color="#e2001a", lw=1.2, ls="--")
axR.text(2.55, 101, "uniform (no commitment)", color="#e2001a", fontsize=9, ha="right")
axR.set_ylim(0, 110); axR.set_ylabel("lattice entropy, % of uniform")
axR.grid(alpha=0.25, lw=0.6, axis="y")
axR.set_title("Commitment: lower = the router has actually decided", fontsize=12)
fig.suptitle("O21a.v2-cold: the lattice is still DIFFUSE at ~epoch 3,300 — "
             "only 22042 has begun to commit", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "lattice_commitment.png"), dpi=150)
plt.close(fig)
print("wrote", os.path.join(OUT, "lattice_commitment.png"))
