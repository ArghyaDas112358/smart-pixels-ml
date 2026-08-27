"""
Slice-choice evolution from router_epochs.csv alone: paint each epoch's TOP-5
mu slices into an (epoch x 101) map, one panel per seed.

Exists because theta_mu.npz is written only by on_train_end, which never fires
on Gautschi chains (chunks die of the GPU leak and self-resubmit), so the full
per-snapshot mu history is unavailable there. Top-5 per epoch is DENSER in
time (every epoch vs every 25) and covers all the mass once the router
commits; the flat early phase reads as faint speckle, which is honest.

  SMARTPIX_RUN_DIR / SMARTPIX_EVO_SEEDS / SMARTPIX_EVO_OUT / SMARTPIX_EVO_TITLE
  CUDA_VISIBLE_DEVICES='' python make_mu_evolution_top5.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from seed_palette import color as seed_color

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR", "o21v2a2_pairlattice"))
SEEDS = [int(x) for x in os.environ.get("SMARTPIX_EVO_SEEDS", "22042,22142,22242").split(",")]
OUT = os.path.join(R, "runs", os.environ.get("SMARTPIX_EVO_OUT", "perf_plots_o21v2/mu_evolution_top5.png"))
TITLE = os.environ.get("SMARTPIX_EVO_TITLE",
                       "O21a.v2-cold: slice choice vs epoch (top-5 μ per epoch)")

def cmap_for(hexc):
    return LinearSegmentedColormap.from_list("m", ["#ffffff", hexc])

def top5_map(seed):
    p = os.path.join(RUN, f"seed_{seed}", "router_epochs.csv")
    eps, grids = [], []
    for r in csv.DictReader(open(p)):
        try:
            ep = int(r["epoch"])
            idx = [int(i) for i in r["mu_top5_idx"].split(";")]
            val = [float(v) for v in r["mu_top5_val"].split(";")]
        except (ValueError, KeyError):
            continue
        g = np.zeros(101, dtype=np.float32)
        g[idx] = val
        eps.append(ep); grids.append(g)
    return np.array(eps), np.stack(grids)

fig, axes = plt.subplots(1, len(SEEDS), figsize=(6.4 * len(SEEDS), 4.6),
                         sharey=True, squeeze=False)
for ax, s in zip(axes[0], SEEDS):
    ep, mu = top5_map(s)
    c = seed_color(s)
    vmax = max(0.35, float(np.percentile(mu[mu > 0], 99)) if (mu > 0).any() else 0.35)
    ax.pcolormesh(ep, np.arange(101), mu.T, cmap=cmap_for(c), vmin=0.0, vmax=vmax,
                  shading="nearest", rasterized=True)
    # final pair, marked the way the other evolution slides do it
    last = mu[-1]
    for i in np.argsort(last)[-2:]:
        ax.axhline(int(i), color="#e2001a", ls="--", lw=1.1)
        ax.plot([ep[-1]], [int(i)], marker="*", color="#e2001a", ms=13, clip_on=False)
    ax.set_title(f"seed {s}", color=c, fontsize=13, fontweight="bold")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.15, lw=0.4)
axes[0][0].set_ylabel("time slice index")
fig.suptitle(TITLE, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
