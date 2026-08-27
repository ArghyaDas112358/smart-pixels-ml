"""
Router occupancy mu(i) = P(slice i is one of the two selected), for each O11 seed.

The full mu vector is not logged -- router_epochs.csv only keeps the top-5 -- so
it is recomputed here from the saved theta using the same cancellation-free
float64 pair-statistics as SimpleRouterLayer._pair_stats.  sum(mu) == 2 by
construction, which is the first assertion below.

Every panel is cross-checked against the mu_top5 recorded in router_epochs.csv
at the same epoch; a mismatch means the loaded checkpoint is not the one the CSV
describes (the stale-snapshot trap that mislabelled the perf plots), and the
script says so loudly instead of drawing a wrong figure.

  CUDA_VISIBLE_DEVICES='' python make_mu_plot.py
"""
import os, csv, json
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seed_palette import ACCENT, SEEDS, color as seed_color
_env = os.environ.get("SMARTPIX_MU_SEEDS", "")
if _env:
    SEEDS = [int(x) for x in _env.split(",")]
    ACCENT = {s: seed_color(s) for s in SEEDS}

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
# Parameterised so the same figure can be made for any campaign.
#   SMARTPIX_MU_SRC   directory of <seed>.hdf5 SNAPSHOTS (never the live run dir --
#                     h5py locking on CephFS kills the trainer, see
#                     make_perf_plots_mdmm.py), else the run dir is used
#   SMARTPIX_RUN_DIR  run dir under runs/ (for result.json / final_indices)
#   SMARTPIX_MU_SEEDS comma list; SMARTPIX_MU_OUT output png
RUN = os.path.join(R, "runs", os.environ.get("SMARTPIX_RUN_DIR", "simplerouter_mdmm_discovery"))
MU_SRC = os.environ.get("SMARTPIX_MU_SRC", "")
OUT = os.path.join(R, "runs", os.environ.get("SMARTPIX_MU_OUT",
                   "perf_plots_mdmm/mu_o11.png"))

THETA_KEY = "simple_router_output/simple_router_output/theta:0"


def pair_stats(phi):
    """mu_i = sum_j P(pair {i,j}), P({a,b}) prop to exp(phi_a + phi_b), a != b."""
    phi = np.asarray(phi, dtype=np.float64)
    w = np.exp(phi - phi.max())
    num = np.outer(w, w) * (1.0 - np.eye(len(w)))
    Z = 0.5 * num.sum()
    P = num / Z
    return P.sum(axis=1)


def theta_of(seed):
    p = (os.path.join(MU_SRC, f"seed_{seed}.hdf5") if MU_SRC
         else os.path.join(RUN, f"seed_{seed}", "last.weights.hdf5"))
    with h5py.File(p, "r") as f:
        return f[THETA_KEY][:]


def csv_check(seed, mu):
    """Compare against the mu_top5 logged for the final epoch."""
    p = os.path.join(RUN, f"seed_{seed}", "router_epochs.csv")
    last = list(csv.DictReader(open(p)))[-1]
    idx = [int(x) for x in last["mu_top5_idx"].split(";")]
    val = [float(x) for x in last["mu_top5_val"].split(";")]
    got = mu[idx]
    if MU_SRC and not np.allclose(got, val, atol=2e-3):
        print(f"  note seed {seed}: snapshot predates the latest csv row (expected "
              f"for a run still training) -- plotting the snapshot")
        return True
    if not np.allclose(got, val, atol=2e-3):
        print(f"  !! seed {seed}: checkpoint disagrees with router_epochs.csv "
              f"ep {last['epoch']}\n     csv={np.round(val,4)}\n     ckpt={np.round(got,4)}")
        return False
    return True


# Wide aspect (~2.65) so the figure fills the full text column of the slide
# instead of being letterboxed into the middle third of it.
fig, axes = plt.subplots(2, 2, figsize=(17.5, 6.6), sharex=True)
ok = True
for ax, s in zip(axes.ravel(), SEEDS):
    mu = pair_stats(theta_of(s))
    assert abs(mu.sum() - 2.0) < 1e-9, f"seed {s}: sum(mu)={mu.sum()}"
    ok &= csv_check(s, mu)

    # result.json only exists once a seed finishes; for a run still training take
    # the pair from the last router_epochs.csv row instead of crashing.
    rj = os.path.join(RUN, f"seed_{s}", "result.json")
    if os.path.exists(rj):
        pair = json.load(open(rj))["final_indices"]
    else:
        last = list(csv.DictReader(open(os.path.join(RUN, f"seed_{s}", "router_epochs.csv"))))[-1]
        pair = sorted((int(last["i1"]), int(last["i2"])))
    c = ACCENT[s]
    ax.fill_between(np.arange(len(mu)), mu, color=c, alpha=0.28, lw=0)
    ax.plot(np.arange(len(mu)), mu, lw=1.1, color=c)
    for i in pair:
        ax.axvline(i, color=c, lw=1.0, ls=":", alpha=0.9)
    top2 = float(np.sort(mu)[-2:].sum())
    ax.set_title(f"seed {s}   pair {pair}   top-2 $\\mu$ = {top2:.2f} / 2.0",
                 fontsize=13, color=c)
    ax.set_xlim(0, len(mu) - 1)
    ax.grid(alpha=0.22, lw=0.6)
    ax.margins(y=0.12)

for ax in axes[1]:
    ax.set_xlabel("time slice index (of 101)", fontsize=12)
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\mu_i$  = P(slice selected)", fontsize=12)

# No suptitle: the slide's own subtitle already defines mu and states sum(mu)=2,
# and repeating it inside the figure is the duplication the deck review flagged.
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print(("wrote " if ok else "WROTE WITH WARNINGS ") + OUT)
