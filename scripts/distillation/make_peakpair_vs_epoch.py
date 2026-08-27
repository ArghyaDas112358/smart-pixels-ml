"""
Peak pair slice vs epoch, overlaid for every seed that converged to ~-40k NLL.

Both slices of each seed's readout pair are drawn in that seed's colour:
  SOLID  = the earlier time t_i
  DASHED = the later time  t_j

Source is router_epochs.csv, whose i1/i2 columns are the layer's
selected_indices() -- the argmax of the SMOOTHED phi, i.e. the pair the ASIC
would actually read out at that epoch. That is logged from epoch 0 for every
seed, so unlike phi_history.npz there is no partial-coverage problem here.

Colours are the same per-seed hexes as history_all_cold.png, so a seed is the
same colour on both slides.

  CUDA_VISIBLE_DEVICES='' python make_peakpair_vs_epoch.py
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
D = "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml"
OUT = os.path.join(R, "runs", "perf_plots_o21v2", "peakpair_vs_epoch.png")

# seed, run dir, colour (identical to history_all_cold.png), best NLL
SEEDS = [
    (22242, f"{D}/runs/o21v2a2_pairlattice", "#5b21b6", "-40,912"),
    (22042, f"{D}/runs/o21v2a2_pairlattice", "#7c3aed", "-41,217"),
    (22142, f"{D}/runs/o21v2a2_pairlattice", "#a78bfa", "-40,195"),
    (24042, f"{R}/runs/o21v3a_phi_eval",     "#1d4ed8", "-40,916"),
    (24242, f"{R}/runs/o21v3a_phi_eval",     "#3b82f6", "-40,172"),
    (24142, f"{R}/runs/o21v3a_phi_eval",     "#60a5fa", "-40,317"),
    (23042, f"{R}/runs/o21v2a_phi0",         "#0e7490", "-40,463"),
]


def peak_pair(path):
    """epoch, earlier slice, later slice -- torn rows dropped, resumes deduped."""
    raw = open(path, "rb").read().replace(b"\x00", b"").decode("utf8", "replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    hdr = lines[0].split(",")
    ie, i1, i2 = hdr.index("epoch"), hdr.index("i1"), hdr.index("i2")
    n = len(hdr)
    d = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        p = ln.split(",")
        if len(p) != n:
            continue
        try:
            e, a, b = int(float(p[ie])), int(float(p[i1])), int(float(p[i2]))
        except ValueError:
            continue
        if not (0 <= e <= 20000 and 0 <= a <= 100 and 0 <= b <= 100):
            continue
        d[e] = (min(a, b), max(a, b))     # a resume rewrites its overlap
    ep = np.array(sorted(d))
    lo = np.array([d[e][0] for e in ep])
    hi = np.array([d[e][1] for e in ep])
    return ep, lo, hi


fig = plt.figure(figsize=(13.6, 6.9), dpi=115)
gs = fig.add_gridspec(1, 2, width_ratios=[4.35, 1.0], wspace=0.045)
ax  = fig.add_subplot(gs[0, 0])
axh = fig.add_subplot(gs[0, 1], sharey=ax)

# where the seven seeds ended up, so the convergence reads at a glance
ax.axhspan(10, 11, color="#16a34a", alpha=0.10, zorder=0)
ax.axhspan(18, 23, color="#ea580c", alpha=0.09, zorder=0)
ax.text(9900, 6.4, "earlier-time band 10–11", fontsize=10.5, color="#166534",
        va="center", ha="right", fontweight="bold")
ax.text(9900, 26.0, "later-time band 18–23", fontsize=10.5, color="#9a3412",
        va="center", ha="right", fontweight="bold")

handles, xmax = [], 0
fin_lo, fin_hi = [], []
for seed, run, colr, nll in SEEDS:
    p = os.path.join(run, f"seed_{seed}", "router_epochs.csv")
    if not os.path.exists(p):
        print("missing", p); continue
    ep, lo, hi = peak_pair(p)
    ax.plot(ep, lo, lw=1.7, color=colr, ls="-",  alpha=0.95, zorder=3)
    ax.plot(ep, hi, lw=1.7, color=colr, ls="--", alpha=0.95, zorder=3)
    handles.append(Line2D([], [], color=colr, lw=2.4,
                          label=f"{seed}   ({lo[-1]}, {hi[-1]})   {nll}" + ("   5k" if ep.max() < 6000 else "")))
    fin_lo.append(int(lo[-1])); fin_hi.append(int(hi[-1]))
    xmax = max(xmax, ep.max())
    print(f"{seed}: {len(ep)} epochs 0..{ep.max()}  final ({lo[-1]}, {hi[-1]})")

ax.set_xlim(0, 10100)
ax.set_ylim(0, 100)
ax.set_xticks(range(0, 10001, 2000))
ax.set_xlabel("epoch", fontsize=13)
ax.set_ylabel("peak pair slice index", fontsize=13)
ax.grid(alpha=0.22, lw=0.6)
ax.set_title("Peak pair time slices vs epoch — 7 seeds, final NLL ≈ −40k",
             fontsize=14, pad=11)

l1 = ax.legend(handles=handles, loc="upper right", fontsize=10.5, framealpha=0.93,
               title="seed · final pair · best NLL", title_fontsize=11, ncol=2)
l1._legend_box.align = "left"
ax.add_artist(l1)
# measure the seed legend so the style box sits flush beneath it
fig.canvas.draw()
bb = l1.get_window_extent().transformed(ax.transAxes.inverted())
style = [Line2D([], [], color="#334155", lw=2.2, ls="-",  label="solid — earlier time  $t_i$"),
         Line2D([], [], color="#334155", lw=2.2, ls="--", label="dashed — later time  $t_j$")]
ax.legend(handles=style, loc="upper right", bbox_to_anchor=(bb.x1, bb.y0 - 0.022),
          bbox_transform=ax.transAxes, fontsize=11, framealpha=0.95,
          borderpad=0.55, handlelength=3.0, edgecolor="#cbd5e1")

GREEN, ORANGE = "#16a34a", "#ea580c"
bins = np.arange(-0.5, 101.5, 1.0)
axh.hist(fin_lo, bins=bins, orientation="horizontal", color=GREEN,  alpha=0.85,
         label=f"earlier time $t_i$  (n={len(fin_lo)})")
axh.hist(fin_hi, bins=bins, orientation="horizontal", color=ORANGE, alpha=0.85,
         label=f"later time $t_j$  (n={len(fin_hi)})")
axh.axhspan(10, 11, color=GREEN,  alpha=0.10, zorder=0)
axh.axhspan(18, 23, color=ORANGE, alpha=0.09, zorder=0)
axh.set_xlabel("seeds", fontsize=12)
axh.set_title("final pair", fontsize=12, pad=11)
axh.set_xlim(0, max(4, max(np.bincount(fin_lo).max(), np.bincount(fin_hi).max()) + 1))
axh.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axh.grid(alpha=0.22, lw=0.6, axis="x")
axh.tick_params(labelleft=False, labelsize=10)
axh.legend(fontsize=9.5, loc="upper right", framealpha=0.95, edgecolor="#cbd5e1")
for v, c in ((fin_lo, GREEN), (fin_hi, ORANGE)):
    lo_, hi_ = min(v), max(v)
    axh.text(axh.get_xlim()[1] * 0.97, (lo_ + hi_) / 2 + 3.4,
             f"{lo_}–{hi_}" if lo_ != hi_ else f"{lo_}",
             fontsize=10, color=c, ha="right", va="center", fontweight="bold")

fig.tight_layout()
fig.savefig(OUT, facecolor="white"); plt.close(fig)
print("wrote", OUT)
print("  final earlier times:", sorted(fin_lo))
print("  final later times  :", sorted(fin_hi))
