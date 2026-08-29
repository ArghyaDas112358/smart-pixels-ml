"""Training history for the O22 campaign -- validation curves for every arm.

Four panels: the epoch scan, the tolerance scan, and the two failures that the
campaign is built around. Everything is plotted as `val_plain_nll`, the
UNWEIGHTED loss-v2 value logged as a compiled metric. `val_loss` under bin
balancing is a weighted composite whose scale depends on the bin weights, so
curves from different arms cannot be laid on the same axis; plain_nll can.

CSV lines torn by a concurrent writer parse into plausible-looking numbers --
this has silently poisoned a plot before. The filter here is an ABSOLUTE bound,
not a local-median cut: a torn line reads in the millions, while a real early
spike under a freshly-armed constraint reaches -23,600 and must be kept. A
6-MAD rolling cut discarded 16 of 100 real points on the first pass.

  python make_history_plots_o22.py
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
OUT = os.path.join(R, "runs", "perf_plots_o22"); os.makedirs(OUT, exist_ok=True)
PARENT_NLL = -40161.79          # seed 22042 last.weights, evaluated on the test set
SEED_C = {"22042": "#1f77b4", "22142": "#d62728", "22242": "#2ca02c"}
BUD_LS = {"100": "-", "200": "--", "400": ":"}


PLAUSIBLE = 5e6   # a torn line reads ~-3.7e7; nothing real here leaves +-5e6


def series(path, col):
    """Read one column, dropping rows with a bad field count and values outside
    the physically plausible range for this loss."""
    if not os.path.exists(path):
        return None, None
    ep, v = [], []
    with open(path) as fh:
        rd = csv.DictReader(fh)
        n = len(rd.fieldnames)
        for row in rd:
            if None in row or len(row) != n or row.get(col) in (None, ""):
                continue
            try:
                e, x = float(row["epoch"]), float(row[col])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(x) or abs(x) > PLAUSIBLE:
                continue
            ep.append(e); v.append(x)
    if not v:
        return None, None
    return np.asarray(ep), np.asarray(v)


def rollmed(v, w=11):
    """Validation NLL spikes hard whenever the constraint re-arms; the trend
    lives in the envelope, so the readable curve is a rolling median. The raw
    series is still drawn faintly behind it -- the spikes are real."""
    if len(v) < w:
        return v
    k = w // 2
    pad = np.pad(v, k, mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(v))])


def yclip(axis, lo=1.0, hi=99.5, pad=.06):
    """Frame on the bulk of the data so one early spike does not flatten the
    curve everyone actually wants to read."""
    d = np.concatenate([ln.get_ydata() for ln in axis.get_lines()
                        if len(ln.get_ydata()) > 2] or [np.array([0., 1.])])
    a, b = np.percentile(d, lo), np.percentile(d, hi)
    m = (b - a) * pad or 1.0
    axis.set_ylim(a - m, b + m)


fig, ax = plt.subplots(2, 2, figsize=(14.5, 9.2), dpi=115)

# ---- (0,0) epoch scan: 3 budgets x 3 seeds --------------------------------
a = ax[0][0]; trend_a = []
for bud in ("100", "200", "400"):
    for sd in ("22042", "22142", "22242"):
        e, v = series(f"{R}/runs/o22_ep{bud}/seed_{sd}/history.csv", "val_plain_nll")
        if e is None: continue
        a.plot(e, v, ls="-", lw=.7, color=SEED_C[sd], alpha=.16, zorder=1)
        m = rollmed(v); trend_a.append(m)
        a.plot(e, m, ls=BUD_LS[bud], lw=1.7, color=SEED_C[sd], alpha=.95, zorder=3,
               label=f"{bud} ep · seed {sd}")
a.axhline(PARENT_NLL, ls="-.", lw=1.6, c="#7c3aed", zorder=1,
          label=f"parent  {PARENT_NLL:,.0f}")
a.set_title("Epoch scan at tol 0.17 — validation (rolling median over faint raw)", fontsize=12.5)
a.legend(fontsize=7.6, ncol=2, framealpha=.94, loc="lower right")

# ---- (0,1) tolerance scan -------------------------------------------------
b = ax[0][1]; trend_b = []
for lab, run, col in [("control", "o22_control_final", "#64748b"),
                      ("tol 0.50", "o22_tol050", "#93c5fd"),
                      ("tol 0.30", "o22_tol030", "#3b82f6"),
                      ("tol 0.17", "o22_tol017", "#15803d"),
                      ("tol 0.10", "o22_tol010", "#b45309")]:
    e, v = series(f"{R}/runs/{run}/seed_22042/history.csv", "val_plain_nll")
    if e is None: continue
    k = e <= 400; e, v = e[k], v[k]
    b.plot(e, v, lw=.7, color=col, alpha=.16, zorder=1)
    m = rollmed(v); trend_b.append(m)
    b.plot(e, m, lw=1.8, color=col, zorder=3, label=lab)
b.axhline(PARENT_NLL, ls="-.", lw=1.6, c="#7c3aed", label=f"parent  {PARENT_NLL:,.0f}")
b.set_title("Tolerance scan — validation, seed 22042 (rolling median)", fontsize=12.5)
b.legend(fontsize=8.6, framealpha=.94, loc="lower right")

# ---- (1,0) attempt 1: the one-hot overfit --------------------------------
c = ax[1][0]
for lab, run, col in [("constraint ON", "_o22_onehot_overfit_on", "#b91c1c"),
                      ("control (OFF)", "_o22_onehot_overfit_off", "#64748b")]:
    for what, ls, alpha in (("plain_nll", "-", .45), ("val_plain_nll", "-", 1.0)):
        e, v = series(f"{R}/runs/{run}/seed_{'22042'}/history.csv", what)
        if e is None: continue
        c.plot(e, v, ls=ls, lw=1.7, color=col, alpha=alpha,
               label=f"{lab} — {'train' if what=='plain_nll' else 'val'}")
c.axhline(PARENT_NLL, ls="-.", lw=1.4, c="#7c3aed")
c.set_title("Attempt 1 — one-hot pin: train (pale) leaves val (solid)", fontsize=12.5)
c.legend(fontsize=8.4, framealpha=.94, loc="center left")

# ---- (1,1) attempt 2: the runaway ----------------------------------------
d = ax[1][1]
e, v = series(f"{R}/runs/_o22_runaway_notol_on/seed_22042/history.csv", "val_plain_nll")
if e is not None:
    d.plot(e, v, lw=1.6, color="#b91c1c", label="no tolerance band — val")
e, v = series(f"{R}/runs/o22_control_final/seed_22042/history.csv", "val_plain_nll")
if e is not None:
    d.plot(e, v, lw=1.6, color="#64748b", label="control (no constraint) — val")
d.axhline(PARENT_NLL, ls="-.", lw=1.6, c="#7c3aed", label=f"parent  {PARENT_NLL:,.0f}")
d.set_title("Attempt 2 — equality constraint, no band", fontsize=12.5)
d.legend(fontsize=8.6, framealpha=.94, loc="lower left")

def frame(axis, curves, pad=.10):
    d = np.concatenate(curves)
    lo, hi = d.min(), d.max()
    m = (hi - lo) * pad or 1.0
    axis.set_ylim(lo - m, hi + m)

if trend_a: frame(a, trend_a)
if trend_b: frame(b, trend_b)
for p in (ax[1][0], ax[1][1]):
    yclip(p)
for p in ax.ravel():
    p.set_xlabel("epoch", fontsize=11)
    p.grid(alpha=.22, lw=.6)
    p.ticklabel_format(axis="y", style="plain")

for p, lab in zip(ax.ravel(), ["validation plain NLL", "validation plain NLL",
                              "plain NLL — train and val", "validation plain NLL"]):
    p.set_ylabel(lab, fontsize=11)

fig.suptitle("O22 — training history, all arms (validation, unweighted NLL)",
             fontsize=14.5, y=.985)
fig.tight_layout(rect=[0, 0, 1, .97])
dst = os.path.join(OUT, "history_o22.png")
fig.savefig(dst, facecolor="white"); plt.close(fig)
print("wrote", dst)

# ---- console ledger -------------------------------------------------------
print(f"\n{'run':<28}{'epochs':>8}{'val start':>13}{'val end':>13}{'val best':>13}")
rows = [("o22_ep100/seed_22042", "runs/o22_ep100/seed_22042"),
        ("o22_ep200/seed_22042", "runs/o22_ep200/seed_22042"),
        ("o22_ep400/seed_22042", "runs/o22_ep400/seed_22042"),
        ("o22_tol050", "runs/o22_tol050/seed_22042"),
        ("o22_tol030", "runs/o22_tol030/seed_22042"),
        ("o22_tol017", "runs/o22_tol017/seed_22042"),
        ("o22_tol010", "runs/o22_tol010/seed_22042"),
        ("control_final", "runs/o22_control_final/seed_22042"),
        ("FAIL onehot ON", "runs/_o22_onehot_overfit_on/seed_22042"),
        ("FAIL onehot OFF", "runs/_o22_onehot_overfit_off/seed_22042"),
        ("FAIL runaway", "runs/_o22_runaway_notol_on/seed_22042")]
for lab, p in rows:
    e, v = series(os.path.join(R, p, "history.csv"), "val_plain_nll")
    if e is None:
        print(f"{lab:<28}{'--':>8}"); continue
    print(f"{lab:<28}{len(e):>8}{v[0]:>13,.0f}{v[-1]:>13,.0f}{v.min():>13,.0f}")
