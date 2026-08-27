"""
Time evolution of the router occupancy mu, one ridgeline (joyplot) per O11 seed.

Each ridge is the full mu histogram over the 101 time slices at one epoch;
ridges march down the panel as training proceeds, drawn back-to-front so later
epochs occlude earlier ones, shaded dark (early) to light (late). It answers
what the single final histogram cannot: WHEN a seed picks its window, and
whether it ever moves again.

Source is `theta_mu.npz` (full 101-slice mu snapshotted every 25 epochs), not
router_epochs.csv, which keeps only the top-5. sum(mu) == 2 is asserted on every
frame drawn.

  CUDA_VISIBLE_DEVICES='' python make_mu_evolution_plot.py
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
RUN = os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR", "simplerouter_mdmm_discovery"))
OUT = os.path.join(R, "runs", os.environ.get("SMARTPIX_EVO_OUT", "perf_plots_mdmm/mu_ridgeline_o11.png"))
# Palette + seed list from seed_palette (this file predated it and had its own
# hardcoded copy, which silently ignored SMARTPIX_EVO_SEEDS).
from seed_palette import ACCENT, SEEDS, color as seed_color
_env = os.environ.get("SMARTPIX_EVO_SEEDS", "")
if _env:
    SEEDS = [int(x) for x in _env.split(",")]
    ACCENT = {s: seed_color(s) for s in SEEDS}

NRIDGE = 26          # ridges per panel; ~200-epoch spacing over the 5000
PEAK = 9.0           # a typical peak, in units of the inter-ridge step
STEP = 1.0
SMOOTH = 3           # moving average over slices, purely for legibility


def shades(hexc, n):
    """Dark -> light ramp of one seed colour (early epochs dark, like the ref)."""
    r, g, b = to_rgb(hexc)
    dark = (r * 0.55, g * 0.55, b * 0.55)
    light = (r + (1 - r) * 0.55, g + (1 - g) * 0.55, b + (1 - b) * 0.55)
    return LinearSegmentedColormap.from_list("s", [dark, light], N=n)


MU = {s: np.load(os.path.join(RUN, f"seed_{s}", "theta_mu.npz")) for s in SEEDS}

def smooth(v, w=SMOOTH):
    if w <= 1:
        return v
    k = np.ones(w) / w
    return np.convolve(np.pad(v, w // 2, mode="edge"), k, mode="valid")[:len(v)]


# One height scale for all four panels so peak heights are comparable between
# seeds. Set by a high percentile, not the global max: seed 3042 parks a narrow
# spike at slice ~7 that is ~2.5x anything else, and scaling to it would flatten
# every other ridge to a bump. That spike is therefore drawn taller than PEAK --
# deliberate overshoot, not a clip, so nothing is silently hidden.
allmu = np.concatenate([MU[s]["mu"].ravel() for s in SEEDS])
scale = float(np.percentile(allmu, 99.5))

_nr, _nc = (1, len(SEEDS)) if len(SEEDS) <= 2 else (2, (len(SEEDS)+1)//2)
fig, axes = plt.subplots(_nr, _nc, figsize=(10*_nc, 4.3*_nr + 0.2), sharex=True, squeeze=False)

for ax, s in zip(axes.ravel(), SEEDS):
    d = MU[s]
    ep_all, mu_all = d["epochs"], d["mu"]
    pick = np.linspace(0, len(ep_all) - 1, NRIDGE).round().astype(int)
    cmap = shades(ACCENT[s], NRIDGE)
    x = np.arange(mu_all.shape[1])
    top = -np.inf

    for k, j in enumerate(pick):
        mu = mu_all[j]
        assert abs(mu.sum() - 2.0) < 1e-3, f"seed {s} ep {ep_all[j]}: sum(mu)={mu.sum()}"
        base = -k * STEP
        y = base + (smooth(mu) / scale) * PEAK * STEP
        top = max(top, float(y.max()))
        col = cmap(k)
        # Later ridges sit lower AND in front, so they occlude earlier ones.
        ax.fill_between(x, base, y, color=col, lw=0, zorder=k + 2)
        ax.plot(x, y, color="white", lw=0.7, zorder=k + 2)

    # Epoch axis: label a handful of ridges on the left.
    want = np.linspace(ep_all[pick[0]], ep_all[pick[-1]], 5)
    ticks = [int(np.abs(ep_all[pick] - w).argmin()) for w in want]
    ax.set_yticks([-k * STEP for k in ticks])
    ax.set_yticklabels([f"{ep_all[pick[k]]:,}" for k in ticks], fontsize=10)
    # Top margin follows the tallest ridge actually drawn, so seed 3042's
    # narrow spike (~2.5x anything else) overshoots PEAK without being cut off.
    ax.set_ylim(-(NRIDGE - 1) * STEP - 0.6, top + 0.8)
    ax.set_xlim(0, mu_all.shape[1] - 1)

    pair = json.load(open(os.path.join(RUN, f"seed_{s}", "result.json")))["final_indices"]
    top2 = float(np.sort(mu_all[-1])[-2:].sum())
    ax.set_title(f"seed {s}   final pair {pair}   top-2 $\\mu$ = {top2:.2f} / 2.0",
                 fontsize=13, color=ACCENT[s])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.18, lw=0.6, zorder=0)

for ax in axes[-1]:
    ax.set_xlabel("time slice index (of 101)", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel("epoch  (dark → light)", fontsize=12)

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}   ridges={NRIDGE}  height scale={scale:.4f} (99.5th pct), "
      f"global max={allmu.max():.4f}")
