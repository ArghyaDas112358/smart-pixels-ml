"""
Talk figures for the two NOISE cases (2_5, 1_6), replotted from the per-seed
threshold_loss_epochs.csv:
  epoch_vs_threshold_combined_<case>.png  -- T0/T1/T2 on ONE axes, epoch downward,
                                             all converged seeds (3 colours, one per threshold)
  nll_vs_epoch_red_<case>.png             -- val NLL vs epoch, all seeds, in RED
                                             (shows the plateau into a local minimum)
Outputs to runs/talk_figs/.  Run: python make_talk_figs.py  (no GPU needed)
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
OUT = os.path.join(R, "runs", "talk_figs"); os.makedirs(OUT, exist_ok=True)
TCOL = ["#2563eb", "#f59e0b", "#16a34a"]   # T0 blue, T1 amber, T2 green
TNAME = ["T0", "T1", "T2"]
RED = "#dc2626"

CASES = [
    dict(key="2_5_noise", dirs=[f"{R}/runs/part1_long_2_5_noise_contained"],
         title="2_5  (+noise)", med=[12.18, 22.94, 52.75]),
    dict(key="1_6_noise", dirs=[f"{R}/runs/part1_long_1_6_iid_contained"],
         title="1_6  (+noise)", med=[12.64, 28.17, 69.89]),
    # no-noise 5000-ep study: 3 batches (random / fixed40 / fixed20 inits), converged runs only
    dict(key="2_5_no_noise", dirs=[f"{R}/runs/part1_long_5k", f"{R}/runs/part1_long_5k_fixed40",
                                   f"{R}/runs/part1_long_5k_fixed20"],
         title="2_5  (no noise)", med=[0.40, 7.00, 30.45], no_red=True),
]


def converged_seeds(d):
    """Clean (non-stuck) seed dirs with a genuinely converged result."""
    out = []
    for sd in sorted(glob.glob(os.path.join(d, "seed_*"))):
        base = os.path.basename(sd)
        if "STUCK" in base or "DEAD" in base:
            continue
        csv = os.path.join(sd, "threshold_loss_epochs.csv")
        rj = os.path.join(sd, "result.json")
        if not os.path.exists(csv):
            continue
        try:
            if os.path.exists(rj) and json.load(open(rj)).get("best_val_loss", 1e9) > -2e4:
                continue
        except Exception:
            pass
        out.append((base.replace("seed_", ""), csv))
    return out


def smooth(v, w=25):
    # centered rolling mean with shrinking window at the edges (no zero-pad artifact)
    return pd.Series(v).rolling(w, center=True, min_periods=1).mean().values


for c in CASES:
    seeds = [sc for d in c["dirs"] for sc in converged_seeds(d)]
    print(f"{c['key']}: {len(seeds)} converged seeds -> {[s for s,_ in seeds]}")

    # ---- combined epoch-vs-threshold (single axes) ----
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for sid, csv in seeds:
        df = pd.read_csv(csv)
        ep = df["epoch"].values
        for i in range(3):
            ax.plot(df[f"T{i}"].values, ep, color=TCOL[i], lw=1.0, alpha=0.55)
            ax.scatter([df[f"T{i}"].values[-1]], [ep[-1]], color=TCOL[i], marker="*", s=70, zorder=5)
    for i in range(3):
        ax.axvline(c["med"][i], color=TCOL[i], ls="--", lw=1.3, alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("threshold (mV)"); ax.set_ylabel("epoch  (training progresses downward)")
    ax.set_title(f"{c['title']}: epoch vs threshold — all seeds, T0/T1/T2 on one axes")
    ax.grid(alpha=0.3)
    handles = [Line2D([0],[0], color=TCOL[i], lw=2,
                      label=f"{TNAME[i]}  (median {c['med'][i]:.2f} mV)") for i in range(3)]
    handles.append(Line2D([0],[0], color="k", marker="*", ls="none", label="final (best epoch)"))
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    p1 = os.path.join(OUT, f"epoch_vs_threshold_combined_{c['key']}.png")
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)

    if c.get("no_red"):
        print(f"  (skipping red NLL fig for {c['key']})")
        continue

    # ---- training history (val NLL) in RED ----
    fig, ax = plt.subplots(figsize=(10, 3.7))
    best = []
    for sid, csv in seeds:
        df = pd.read_csv(csv)
        v = pd.to_numeric(df["val_loss"], errors="coerce").values
        ax.plot(df["epoch"].values, smooth(v), color=RED, lw=1.2, alpha=0.6)
        best.append(np.nanmin(v))
    medbest = float(np.median(best))
    ax.axhline(medbest, color="k", ls=":", lw=1.2, alpha=0.8,
               label=f"median best NLL {medbest:,.0f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation NLL / batch")
    ax.set_title(f"{c['title']}: training history — all seeds converge to the same plateau (local minimum)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")
    p2 = os.path.join(OUT, f"nll_vs_epoch_red_{c['key']}.png")
    fig.tight_layout(); fig.savefig(p2, dpi=140); plt.close(fig)

    print(f"  wrote {p1}\n  wrote {p2}")

print("DONE ->", OUT)
