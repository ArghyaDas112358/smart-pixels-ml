"""Supporting plots for docs/soft_router_plan.md — all ILLUSTRATIVE (no fitted data).
Design per the dataviz method: validated categorical order (blue #2a78d6, aqua #1baf7a,
yellow #eda100), single-hue sequential blue ramp for the heatmap, NO dual axes
(stacked small multiples instead), thin marks, recessive grid, ink-colored text.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG = "/work/users/das214/SmartPixels/smart-pixels-ml/docs/figs"
os.makedirs(FIG, exist_ok=True)

# ---- palette (reference instance, light mode) ----
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seqblue", [
    "#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#dddcd6", "grid.linewidth": 0.6,
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

T = np.arange(101)

# ============ P1 — the 101-sample waveform, 2 slices read out ============
wave = (T / 12.0) * np.exp(-T / 22.0); wave /= wave.max()
fig, ax = plt.subplots(figsize=(8.2, 3.6))
ax.plot(T, wave, color=BLUE, lw=2, solid_capstyle="round")
ax.fill_between(T, wave, color=BLUE, alpha=0.10, lw=0)
for i in (11, 26):
    ax.axvline(i, color=RED, lw=1.4, ls=(0, (4, 3)))
    ax.plot([i], [np.interp(i, T, wave)], "o", ms=8, mfc=RED, mec=SURFACE, mew=1.5)
    ax.annotate(f"slice {i}", (i, np.interp(i, T, wave)),
                xytext=(8, 10), textcoords="offset points", color=RED, fontweight="bold")
ax.set_xlabel("time sample index (10 ps spacing)")
ax.set_ylabel("preamp response (a.u.)")
ax.set_title("Each pixel records 101 time samples — the chip reads out only 2 (today: a human guess)")
ax.set_xlim(0, 100); ax.set_ylim(0, 1.12)
fig.tight_layout(); fig.savefig(f"{FIG}/plot_waveform_101.png", dpi=150); plt.close(fig)

# ============ P2 — softmax sharpening: soft blend -> one-hot ============
w = -0.5 * ((T - 12) / 6.0) ** 2
def smax(logits, k):
    e = np.exp(k * (logits - logits.max())); return e / e.sum()
fig, ax = plt.subplots(figsize=(8.2, 3.9))
for k, c in [(1, BLUE), (8, AQUA), (67, YELLOW)]:
    a = smax(w, k)
    ax.plot(T, a, color=c, lw=2, solid_capstyle="round", label=f"k = {k}")
    j = int(np.argmax(a))
    ax.annotate(f"k = {k}", (j + 2 + (6 if k == 1 else 0), a[j] * (0.92 if k == 67 else 1.0)),
                color=c, fontweight="bold")
ax.set_xlabel("time-slice index")
ax.set_ylabel(r"selection weight  $a_1[t]$")
ax.set_title("One selector slot: softmax(k·w) sharpens from a soft blend to a one-hot as k anneals")
ax.set_xlim(0, 60)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{FIG}/plot_softmax_sharpening.png", dpi=150); plt.close(fig)

# ============ P3 — the cosine anneal (two stacked panels, ONE axis each) ============
ep = np.linspace(0, 1, 400)
k = 1 + (67 - 1) * 0.5 * (1 - np.cos(np.pi * ep))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True, layout="constrained")
a1.plot(ep, k, color=BLUE, lw=2)
a1.set_ylabel("sharpness k")
a1.set_title("Cosine anneal drives both SoftQuantize and SoftRouter — k: 1 → 67 over training")
a1.annotate("k = 67 (hard)", (0.97, 65), ha="right", color=INK2)
a2.plot(ep, 1 / k, color=AQUA, lw=2)
a2.set_ylabel(r"temperature  $\beta = 1/k$")
a2.set_xlabel("training progress (epoch / total)")
a2.annotate("soft early → commits late", (0.03, 0.55), color=INK2)
fig.savefig(f"{FIG}/plot_anneal_schedule.png", dpi=150); plt.close(fig)

# ============ P4 — ILLUSTRATIVE: slot attention converging over epochs ============
EPOCHS = 150  # rows (each = a training checkpoint 0..5000)
rng = np.random.default_rng(7)
w1_0 = rng.normal(0, .35, 101); w2_0 = rng.normal(0, .35, 101)
w1_f = -0.5 * ((T - 12) / 3.0) ** 2
w2_f = -0.5 * ((T - 27) / 3.5) ** 2
A = np.zeros((EPOCHS, 101))
for r in range(EPOCHS):
    p = r / (EPOCHS - 1)
    kk = 1 + 66 * 0.5 * (1 - np.cos(np.pi * p))
    blend = min(1.0, (p ** 0.55) * 1.15)
    a1_ = smax((1 - blend) * w1_0 + blend * w1_f, kk)
    a2_ = smax((1 - blend) * w2_0 + blend * w2_f, kk)
    A[r] = a1_ + a2_
fig, ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
im = ax.imshow(A, aspect="auto", cmap=SEQ, origin="upper",
               extent=[0, 100, 5000, 0], vmin=0, vmax=A.max())
ax.grid(False)
for i, lab in ((12, "slice 12"), (27, "slice 27")):
    ax.annotate(lab, (i, 4550), ha="center", color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
ax.set_xlabel("time-slice index")
ax.set_ylabel("epoch  (training ↓)")
ax.set_title("ILLUSTRATIVE — discovery: slot attention narrows to 2 columns", loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.012)
cb.set_label(r"total selection weight  $a_1[t]+a_2[t]$", color=INK2)
cb.outline.set_visible(False)
fig.savefig(f"{FIG}/plot_slot_convergence.png", dpi=150); plt.close(fig)

print("wrote:")
for f in ["plot_waveform_101", "plot_softmax_sharpening", "plot_anneal_schedule", "plot_slot_convergence"]:
    print(f"  {FIG}/{f}.png")
