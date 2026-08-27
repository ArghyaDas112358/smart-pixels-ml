"""
Training-history plot: val NLL vs epoch for every completed O11 seed.

The point of the figure is that seed 3042 sat flat for ~4000 epochs before
escaping, which is why "it looks dead" is not a reason to kill a run. The
unconstrained reference range (UNCON below) is quoted in the slide text rather
than drawn as a band.

  CUDA_VISIBLE_DEVICES='' python make_history_plot.py
"""
import os, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seed_palette import SEEDS as _PAL_SEEDS
from seed_palette import color as _seed_color

SEEDS = list(_PAL_SEEDS)
_env = os.environ.get("SMARTPIX_HIST_SEEDS", "")
if _env:
    SEEDS = [int(x) for x in _env.split(",")]

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
# Parameterised for any campaign (same env convention as the mu/evolution plots):
#   SMARTPIX_RUN_DIR / SMARTPIX_HIST_SEEDS / SMARTPIX_HIST_OUT / SMARTPIX_HIST_TITLE
#   SMARTPIX_HIST_XMAX (epochs; default 5000)
RUN = os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR", "simplerouter_mdmm_discovery"))
OUT = os.path.join(R, "runs", os.environ.get("SMARTPIX_HIST_OUT", "perf_plots_mdmm/history_o11.png"))
# best/worst best_val over the six unconstrained SIMPLE seeds (angles collapsed)
UNCON = (-29124.0, -28004.0)


def series(seed):
    p = os.path.join(RUN, f"seed_{seed}", "history.csv")
    ep, vl = [], []
    for r in csv.DictReader(open(p)):
        v = r.get("val_loss", "")
        if v in ("", "nan", "val_loss"):
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if np.isfinite(f):
            ep.append(int(r["epoch"])); vl.append(f)
    return np.array(ep), np.array(vl)


fig, ax = plt.subplots(figsize=(9, 5.2))

# The unconstrained reference band was removed on request; the slide states the
# same range in text, so the curves get the full plotting area.

for s in SEEDS:
    ep, vl = series(s)
    if len(ep) == 0:
        continue
    # running best -- the quantity actually quoted everywhere else
    best = np.minimum.accumulate(vl)
    rj = os.path.join(RUN, f"seed_{s}", "result.json")
    lbl = f"seed {s}"
    if os.path.exists(rj):
        d = json.load(open(rj))
        lbl = f"seed {s}  {d['final_indices']}  {round(d['best_val_loss']):,}"
    ax.plot(ep, best, lw=1.9, color=_seed_color(s), label=lbl, zorder=3)

ax.set_xlabel("epoch"); ax.set_ylabel("best val NLL so far")
ax.set_xlim(0, int(os.environ.get("SMARTPIX_HIST_XMAX", "5000"))); ax.set_ylim(*[float(v) for v in os.environ.get("SMARTPIX_HIST_YLIM", "-31000,-18000").split(",")])
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.set_title(os.environ.get("SMARTPIX_HIST_TITLE", "O11: best validation NLL across the 5000-epoch runs"), fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
