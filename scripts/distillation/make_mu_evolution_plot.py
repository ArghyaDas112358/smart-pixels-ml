"""
Time evolution of the router occupancy mu, one panel per O11 seed.

Each panel is epoch (x) vs time-slice index (y) with mu as colour -- the slide-3
histogram, but every 25 epochs instead of only at the end. It answers what the
single final histogram cannot: WHEN a seed picks its window, and whether it ever
moves again.

Each panel uses its own seed colour (white -> seed hue) and therefore carries
its OWN colorbar: one shared colorbar can only render a single colormap, so with
four different ramps it would mislabel three of the panels. The numeric scale is
shared across panels, so intensities stay comparable between seeds.

A ridgeline/joyplot rendering of the same data is in make_mu_ridgeline_plot.py.

Source is `theta_mu.npz` (full 101-slice mu snapshotted every 25 epochs), not
router_epochs.csv, which keeps only the top-5. sum(mu) == 2 is asserted on every
frame.

  CUDA_VISIBLE_DEVICES='' python make_mu_evolution_plot.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from seed_palette import ACCENT, SEEDS, color as seed_color
# Parameterised for any campaign: SMARTPIX_RUN_DIR / SMARTPIX_EVO_SEEDS /
# SMARTPIX_EVO_OUT. Panels are laid out in a grid that fits the seed count, so a
# 2-seed campaign gets 1x2 rather than a 2x2 with two blanks.
_env = os.environ.get("SMARTPIX_EVO_SEEDS", "")
if _env:
    SEEDS = [int(x) for x in _env.split(",")]
    ACCENT = {s: seed_color(s) for s in SEEDS}

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR", "simplerouter_mdmm_discovery"))
OUT = os.path.join(R, "runs", os.environ.get("SMARTPIX_EVO_OUT", "perf_plots_mdmm/mu_evolution_o11.png"))


def cmap_for(hexc):
    """White -> seed colour, so 'no mass' reads as empty page, not as a colour."""
    r, g, b = to_rgb(hexc)
    mid = (r + (1 - r) * 0.45, g + (1 - g) * 0.45, b + (1 - b) * 0.45)
    dark = (r * 0.55, g * 0.55, b * 0.55)
    return LinearSegmentedColormap.from_list("m", ["#ffffff", mid, hexc, dark], N=256)


MU = {s: np.load(os.path.join(RUN, f"seed_{s}", "theta_mu.npz")) for s in SEEDS}

# Shared numeric scale, clipped at a high percentile rather than the global max:
# seed 3042 parks a narrow spike at slice ~7 that is ~2.5x anything else, and
# scaling to it flattens all four panels to near-white. Values above vmax
# saturate; every colorbar is drawn with extend="max" so that is visible.
allmu = np.concatenate([MU[s]["mu"].ravel() for s in SEEDS])
vmax = float(np.percentile(allmu, 99.5))

_nr, _nc = (1, len(SEEDS)) if len(SEEDS) <= 2 else (2, (len(SEEDS)+1)//2)
fig, axes = plt.subplots(_nr, _nc, figsize=(9*_nc, 3.9*_nr + 0.2), sharex=True, sharey=True, squeeze=False)

for ax, s in zip(axes.ravel(), SEEDS):
    d = MU[s]
    ep, mu = d["epochs"], d["mu"]
    assert np.allclose(mu.sum(axis=1), 2.0, atol=1e-3), f"seed {s}: sum(mu) != 2"

    c = ACCENT[s]
    im = ax.pcolormesh(ep, np.arange(mu.shape[1]), mu.T,
                       cmap=cmap_for(c), vmin=0.0, vmax=vmax, shading="nearest")

    # The two slices the router actually picked. Red star on a red dashed rule,
    # in a fixed colour rather than the seed hue: the marker sits on top of four
    # different colormaps, and the seed hue would vanish into its own dark end
    # exactly where the mass (and therefore the pick) is. The star carries a thin
    # white edge so it stays readable where it lands on its own rule.
    pair = json.load(open(os.path.join(RUN, f"seed_{s}", "result.json")))["final_indices"]
    for i in pair:
        ax.axhline(i, color="#dc2626", lw=1.2, ls=(0, (5, 3)), alpha=0.95, zorder=4)
        ax.plot(ep[-1], i, marker="*", ms=20, color="#dc2626",
                markeredgecolor="#ffffff", markeredgewidth=1.0,
                clip_on=False, zorder=5)

    top2 = float(np.sort(mu[-1])[-2:].sum())
    ax.set_title(f"seed {s}   final pair {pair}   top-2 $\\mu$ = {top2:.2f} / 2.0",
                 fontsize=13, color=c)
    ax.set_ylim(0, mu.shape[1] - 1)
    ax.set_xlim(0, 5000)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.012, extend="max")
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_visible(False)

for ax in axes[-1]:
    ax.set_xlabel("epoch", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("time slice index (of 101)", fontsize=12)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}   shared vmax={vmax:.4f} (99.5th pct), global max={allmu.max():.4f}")
