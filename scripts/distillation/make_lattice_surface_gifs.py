"""
Animated LATTICE SURFACE per seed, in the same visual language as the static
"learned pair distribution" figure (magma, log colour, cyan star on the peak).

WHAT EACH FRAME IS. The learned psi is persisted only in the final checkpoint,
so the surface cannot be replayed exactly for epochs already trained. Each
frame is therefore a kernel-density ESTIMATE of the sampling distribution at
that moment: the pairs sampled in a rolling epoch window, blurred with the same
2-D Gaussian the router itself uses at that epoch's sigma, then normalised.
Where the estimate is reliable (the router keeps sampling from its own
distribution) this converges to softmax(psi); the final frame is compared with
the true surface on the slide.

Going forward the driver now writes phi_history.npz every SNAP epochs
(SimpleRouterLogger), so future runs animate the REAL surface — if that file is
present this script uses it instead and says so in the title.

  CUDA_VISIBLE_DEVICES='' python make_lattice_surface_gifs.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from seed_palette import color as seed_color

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", "o21v2a2_pairlattice")
OUT = os.path.join(R, "runs", "perf_plots_o21v2")
SEEDS = [22042, 22142, 22242]
T = 101
WIN = 400          # epochs pooled per frame (KDE needs samples)
FRAMES = 55
IA, IB = np.triu_indices(T, k=1)
os.makedirs(OUT, exist_ok=True)


def router_rows(seed):
    p = os.path.join(RUN, f"seed_{seed}", "router_epochs.csv")
    ep, i1, i2, sg = [], [], [], []
    for r in csv.DictReader(open(p)):
        try:
            ep.append(int(r["epoch"])); i1.append(int(r["i1"])); i2.append(int(r["i2"]))
            sg.append(float(r.get("sigma", 0) or 0))
        except (ValueError, KeyError):
            continue
    ep, i1, i2 = np.array(ep), np.array(i1), np.array(i2)
    return ep, np.minimum(i1, i2), np.maximum(i1, i2), np.array(sg)


def phi_history(seed):
    p = os.path.join(RUN, f"seed_{seed}", "phi_history.npz")
    if not os.path.exists(p):
        return None
    try:
        z = np.load(p)
        return np.asarray(z["epochs"]), np.asarray(z["phi"]).astype(np.float32)
    except Exception:
        return None


for s in SEEDS:
    ep, lo, hi, sg = router_rows(s)
    c = seed_color(s)
    real = phi_history(s)
    # start at epoch 0: early frames simply pool whatever exists so far, so the
    # animation opens on the flat/uncommitted lattice instead of mid-training
    marks = (np.linspace(ep.min(), ep.max(), FRAMES).astype(int) if real is None
             else real[0][np.linspace(0, len(real[0]) - 1, min(FRAMES, len(real[0]))).astype(int)])

    def surface(e):
        """(101,101) probability surface for epoch e -- real psi if logged, else KDE."""
        if real is not None:
            k = int(np.argmin(np.abs(real[0] - e)))
            phi = real[1][k].astype(np.float64)
            w = np.exp(phi - phi.max()); p = w / w.sum()
        else:
            m = (ep > e - WIN) & (ep <= e)
            if not m.any():                      # first frames: take what we have
                m = ep <= max(e, ep.min())
            g = np.zeros((T, T))
            if m.any():
                np.add.at(g, (lo[m], hi[m]), 1.0)
            # blur with the router's own kernel width at this epoch (floor so a
            # committed sigma=0 frame is still visible), then symmetrise
            sig = max(1.2, float(sg[m][-1]) if m.any() and len(sg) else 1.2)
            g = gaussian_filter(g, sigma=sig, mode="nearest")
            p = np.zeros((T, T))
            p[IA, IB] = g[IA, IB]
            tot = p.sum()
            p = p / tot if tot > 0 else p
        grid = np.zeros((T, T))
        grid[IA, IB] = p[IA, IB] if p.ndim == 2 else p
        grid[IB, IA] = grid[IA, IB]
        return grid

    g0 = surface(marks[0])
    fig, ax = plt.subplots(figsize=(6.6, 6.3))
    floor = 1e-7
    mesh = ax.pcolormesh(np.arange(T), np.arange(T), np.maximum(g0, floor).T,
                         cmap="magma", norm=LogNorm(vmin=floor, vmax=max(g0.max(), 10 * floor)),
                         shading="nearest")
    ax.plot([0, 100], [0, 100], color="#9aa4ad", lw=0.9, ls=":")
    star, = ax.plot([], [], marker="*", ms=15, color="#00e5ff", mec="black", mew=0.5, ls="none")
    lbl = ax.text(0.03, 0.965, "", transform=ax.transAxes, fontsize=12, fontweight="bold",
                  va="top", color="white")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel(r"$t_i$"); ax.set_ylabel(r"$t_j$")
    kind = "learned surface softmax(φ)" if real is not None else "sampling-distribution estimate"
    ax.set_title(f"seed {s} — {kind}", color=c, fontsize=12, fontweight="bold")
    fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.02, label="pair probability (log)")
    fig.tight_layout()

    def update(k):
        e = int(marks[k])
        g = surface(e)
        mesh.set_array(np.maximum(g, floor).T.ravel())
        mesh.set_clim(floor, max(g.max(), 10 * floor))
        a, b = np.unravel_index(np.argmax(np.triu(g, 1)), g.shape)
        star.set_data([a], [b])
        lbl.set_text(f"epoch {e}   peak ({a}, {b})")
        return mesh, star, lbl

    anim = FuncAnimation(fig, update, frames=len(marks), blit=False)
    path = os.path.join(OUT, f"lattice_surface_{s}.gif")
    anim.save(path, writer=PillowWriter(fps=5))
    plt.close(fig)
    print("wrote", path, f"({os.path.getsize(path)/1e6:.1f} MB, "
          f"{'REAL phi' if real is not None else 'KDE estimate'})")
