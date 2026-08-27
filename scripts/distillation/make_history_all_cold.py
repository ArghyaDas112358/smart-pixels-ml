"""
Every cold seed we have, on one axis: best val NLL so far vs epoch.

Three campaigns, ten seeds:
  * O21a.v2-cold  (Gautschi, 10k ep)  22042 22142 22242
  * O21a.v3-phi   (Gautschi, 10k ep)  24042 24142 24242
  * O21a.v2-phi0  (AF A100,   5k ep)  23042 23142 23242 23342

Colour = outcome, not campaign: the seeds that found the early-anchor basin are
warm (violet/blue family), the ones that parked mid-window are grey-red. Line
style = campaign, so both readings are available at once.

Torn CSV lines (chunk restarts on a chained run) can parse into PLAUSIBLE
numbers, so a lone point far below its neighbours is rejected here rather than
being taken as a record.

  CUDA_VISIBLE_DEVICES='' python make_history_all_cold.py
"""
import os, csv, io
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
D = "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml"
OUT = os.path.join(R, "runs", "perf_plots_o21v2", "history_all_cold.png")

SEEDS = [
    # seed,  run dir,                            campaign,     pair,      good?
    (22242, f"{D}/runs/o21v2a2_pairlattice", "v2-cold", "(10,23)", True),
    (22042, f"{D}/runs/o21v2a2_pairlattice", "v2-cold", "(10,21)", True),
    (22142, f"{D}/runs/o21v2a2_pairlattice", "v2-cold", "(11,18)", True),
    (24042, f"{R}/runs/o21v3a_phi_eval",     "v3-phi",  "(10,18)", True),
    (24242, f"{R}/runs/o21v3a_phi_eval",     "v3-phi",  "(10,19)", True),
    (24142, f"{R}/runs/o21v3a_phi_eval",     "v3-phi",  "(11,18)", True),
    (23042, f"{R}/runs/o21v2a_phi0",         "phi0-AF", "(11,21)", True),
    (23142, f"{R}/runs/o21v2a_phi0",         "phi0-AF", "(30,31)", False),
    (23242, f"{R}/runs/o21v2a_phi0",         "phi0-AF", "(53,56)", False),
    (23342, f"{R}/runs/o21v2a_phi0",         "phi0-AF", "(65,73)", False),
]
STYLE = {"v2-cold": "-", "v3-phi": "--", "phi0-AF": "-."}
GOOD  = ["#5b21b6", "#7c3aed", "#a78bfa", "#1d4ed8", "#3b82f6", "#60a5fa", "#0e7490"]
BAD   = ["#b45309", "#be123c", "#7f1d1d"]

O17_BEST = -38994.0


def curve(path):
    """(epoch, running-best) with torn rows and implausible outliers removed."""
    raw = open(path, "rb").read().replace(b"\x00", b"").decode("utf8", "replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    hdr = lines[0].split(","); n = len(hdr)
    ie, iv = hdr.index("epoch"), hdr.index("val_loss")
    ep, vl = [], []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split(",")
        if len(parts) != n:
            continue
        try:
            e, v = int(float(parts[ie])), float(parts[iv])
        except ValueError:
            continue
        if not np.isfinite(v) or not (-1e5 < v < 1e5):
            continue
        if not (0 <= e <= 20000):     # torn lines corrupt the epoch field too
            continue
        ep.append(e); vl.append(v)
    ep, vl = np.array(ep), np.array(vl)
    o = np.argsort(ep, kind="stable"); ep, vl = ep[o], vl[o]
    # a torn line can parse into a plausible-looking record: drop any point far
    # below the median of its local window before taking the running minimum
    med = np.array([np.median(vl[max(0, i - 10):i + 11]) for i in range(len(vl))])
    keep = vl > med - 2000
    ep, vl = ep[keep], vl[keep]
    return ep, np.minimum.accumulate(vl)


fig, ax = plt.subplots(figsize=(8.8, 7.8), dpi=118)
gi = bi = 0
rows = []
for seed, run, camp, pair, good in SEEDS:
    p = os.path.join(run, f"seed_{seed}", "history.csv")
    if not os.path.exists(p):
        print("missing", p); continue
    ep, best = curve(p)
    if good:
        c = GOOD[gi % len(GOOD)]; gi += 1
    else:
        c = BAD[bi % len(BAD)]; bi += 1
    ax.plot(ep, best, lw=1.9, color=c, ls=STYLE[camp], zorder=3 if good else 2,
            alpha=1.0 if good else 0.85,
            label=f"{seed} {pair}  {best.min():,.0f}")
    rows.append((seed, pair, best.min(), ep.max()))

ax.axhline(O17_BEST, color="#334155", ls=":", lw=1.5, zorder=1)
ax.text(10050, O17_BEST + 200, "O17 hand-imposed [11,26] best", color="#334155",
        fontsize=10.5, ha="right", va="bottom")

ax.set_xlabel("epoch", fontsize=13)
ax.set_ylabel("best val NLL so far", fontsize=13)
ax.set_xlim(0, 10200); ax.set_ylim(-42000, -25000)
ax.grid(alpha=0.25, lw=0.6)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.105), fontsize=11.5,
                framealpha=0.0, ncol=2, title="seed · pair · best NLL",
                title_fontsize=12, handlelength=2.8, borderpad=0.2,
                columnspacing=2.4, labelspacing=0.55)
leg._legend_box.align = "left"
ax.set_title("Seven reached the basin, three never left mid-window",
             fontsize=14.5, pad=10)
ax.text(0.985, 0.965, "solid  v2-cold     dashed  v3-φ     dash-dot  φ0-AF (5k ep)",
        transform=ax.transAxes, ha="right", va="top", fontsize=11, color="#64748b")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor="white"); plt.close(fig)
print("wrote", OUT)
for s, pr, b, e in sorted(rows, key=lambda r: r[2]):
    print(f"  {s} {pr:<9} best {b:>9,.0f}  ep {e}")
