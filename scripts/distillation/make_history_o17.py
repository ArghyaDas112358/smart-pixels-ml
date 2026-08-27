"""
O17 training curves: SHALLOW vs DEEP head on frozen [11,26], paired by seed.

One colour per seed; dashed = shallow head, solid = deep head. The paired story
is visible directly: 41142 dives to the deep basin under both heads, 41542 only
under the deep one. Gautschi seed histories are STAGED COPIES (static, runs
finished there); AF seeds are read from their live history.csv, which is
append-only and safe to read (unlike the hdf5 checkpoints).

  CUDA_VISIBLE_DEVICES='' python make_history_o17.py
"""
import os, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
SH = os.path.join(R, "runs", "fixedslice_p11_26")          # staged shallow (csv only)
DP = os.path.join(R, "runs", "fixedslice_deep_p11_26")     # deep (local + staged)
OUT = os.path.join(R, "runs", "perf_plots_o17", "history_o17.png")
COLORS = {41042:"#1d4ed8", 41142:"#15803d", 41242:"#0e7490", 41542:"#e2001a",
          41642:"#b45309", 41942:"#6d28d9", 42542:"#be123c", 42642:"#4d7c0f"}


def series(run, seed):
    p = os.path.join(run, f"seed_{seed}", "history.csv")
    if not os.path.exists(p): return None, None
    ep, vl = [], []
    for r in csv.DictReader(open(p)):
        v = r.get("val_loss", "")
        try: f = float(v)
        except (TypeError, ValueError): continue
        if np.isfinite(f): ep.append(int(r["epoch"])); vl.append(f)
    if not ep: return None, None
    return np.array(ep), np.minimum.accumulate(np.array(vl))


fig, ax = plt.subplots(figsize=(10.5, 5.4))
for s, c in COLORS.items():
    e1, v1 = series(SH, s)
    e2, v2 = series(DP, s)
    if v1 is not None and len(v1) > 50:
        ax.plot(e1, v1, lw=1.4, ls=(0, (4, 2)), color=c, alpha=0.75)
    if v2 is not None and len(v2) > 50:
        lbl = f"seed {s}" + ("  (extra)" if s >= 41900 else "")
        ax.plot(e2, v2, lw=2.0, color=c, label=lbl)
ax.plot([], [], color="0.3", ls=(0, (4, 2)), lw=1.4, label="shallow head (dashed)")
ax.plot([], [], color="0.3", lw=2.0, label="deep head (solid)")
ax.set_xlabel("epoch"); ax.set_ylabel("best val NLL so far")
ax.set_xlim(0, 2000); ax.set_ylim(-40000, 5000)
ax.axhline(-30000, color="0.6", lw=0.8, ls=":")
ax.text(30, -30500, "deep-basin threshold", fontsize=8, color="0.4", va="top")
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
ax.set_title("O17: frozen [11,26] — shallow vs deep regression head, paired seeds", fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
