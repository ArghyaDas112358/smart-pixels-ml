"""Convergence figures for the 2_5 noise-contained Part-1 long run (seed 42)."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/part1_long_2_5_noise_contained/seed_42/threshold_loss_epochs.csv"
FIGS = "/work/users/das214/SmartPixels/smart-pixels-ml/threshold_optimization_2_5_noise_contained/slides/figs"

ep, loss, val, T0, T1, T2 = [], [], [], [], [], []
with open(CSV) as f:
    r = csv.reader(f); next(r)
    for row in r:
        if not row or row[2] == "": continue
        ep.append(int(row[0])); loss.append(float(row[1])); val.append(float(row[2]))
        T0.append(float(row[3])); T1.append(float(row[4])); T2.append(float(row[5]))
ep = np.array(ep)
best_i = int(np.argmin(val)); best_ep = ep[best_i]; best_val = val[best_i]
fin = [T0[-1], T1[-1], T2[-1]]

# Fig 1: loss + val_loss vs epoch
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, loss, lw=1, alpha=.7, label="train loss")
ax.plot(ep, val, lw=1.4, label="val loss")
ax.axvline(best_ep, ls="--", c="gray", lw=.8)
ax.scatter([best_ep], [best_val], c="red", zorder=5,
           label=f"best val {best_val:.0f} @ ep {best_ep}")
ax.set_xlabel("epoch"); ax.set_ylabel("NLL loss (per 5000-event batch)")
ax.set_title("Part-1 convergence — 2_5 mV baked-noise contained (seed 42, 5000 ep)")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); p1 = f"{FIGS}/loss_vs_epoch.png"; fig.savefig(p1, dpi=140); plt.close(fig)

# Fig 2: thresholds vs epoch
fig, ax = plt.subplots(figsize=(8, 5))
for arr, name, c in [(T0, "T0", "C0"), (T1, "T1", "C1"), (T2, "T2", "C2")]:
    ax.plot(ep, arr, lw=1.3, color=c, label=f"{name} -> {arr[-1]:.2f}")
for arr, c in [(T0, "C0"), (T1, "C1"), (T2, "C2")]:
    ax.axhline(arr[-1], ls=":", c=c, lw=.7)
ax.set_xlabel("epoch"); ax.set_ylabel("2-bit threshold (mV)")
ax.set_title("Threshold convergence — final [%.2f, %.2f, %.2f]" % tuple(fin))
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); p2 = f"{FIGS}/thresholds_vs_epoch.png"; fig.savefig(p2, dpi=140); plt.close(fig)

# Fig 3: zoom on threshold stability (last 1500 ep)
m = ep >= 3500
fig, ax = plt.subplots(figsize=(8, 5))
for arr, name, c in [(T0, "T0", "C0"), (T1, "T1", "C1"), (T2, "T2", "C2")]:
    a = np.array(arr)[m]
    ax.plot(ep[m], a, lw=1.1, color=c,
            label=f"{name}: {a.min():.2f}-{a.max():.2f} (range {a.max()-a.min():.2f})")
ax.set_xlabel("epoch"); ax.set_ylabel("threshold (mV)")
ax.set_title("Threshold stability, epochs 3500-5000 (converged)")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); p3 = f"{FIGS}/threshold_stability_zoom.png"; fig.savefig(p3, dpi=140); plt.close(fig)

print("FINAL thresholds:", [round(x, 3) for x in fin])
print("best_val_loss:", round(best_val, 1), "@ ep", int(best_ep))
for p in (p1, p2, p3): print("FIG:", p)
