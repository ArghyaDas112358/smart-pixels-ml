"""
ONE seed, the REAL learned pair distribution, animated over its whole life.

Replaces the three-canvas widget on "Scrub the true pair probability (log)".
The three seeds that widget showed (22042/22142/22242) only have phi snapshots
from epoch ~3400 onward -- the logger was added mid-run -- so their animation
starts after the interesting part is over. Seed 23042 (AF A100, cold-from-0,
loss v2) has the COMPLETE record: 200 frames, epoch 0 -> 4975, every one of the
5050 pair logits.

Each frame is the TRUE distribution softmax(phi) read straight from
phi_history.npz -- not a kernel-density estimate of what was sampled.

Colour scale is FIXED across frames (log, 1e-6 .. 3e-2) so the collapse from
near-uniform to a committed peak is visible as real brightening, not rescaling.

  CUDA_VISIBLE_DEVICES='' python make_prob_gif_single.py [seed]
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from seed_palette import color as seed_color

R    = "/work/users/das214/SmartPixels/smart-pixels-ml"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 23042
OUT  = os.path.join(R, "runs", "perf_plots_o21v2")
T    = 101
VMIN, VMAX = 1e-6, 3e-2
IA, IB = np.triu_indices(T, k=1)

CAND = [os.path.join(R, "runs", "o21v2a_phi0",   f"seed_{SEED}", "phi_history.npz"),
        os.path.join(R, "runs", "o21v3a_phi_eval", f"seed_{SEED}", "phi_history.npz"),
        os.path.join(R, "runs", "o21v2a2_pairlattice", f"seed_{SEED}", "phi_history.npz")]
src = next((p for p in CAND if os.path.exists(p)), None)
if src is None:
    raise SystemExit(f"no phi_history.npz for seed {SEED}")

z   = np.load(src)
ep  = np.asarray(z["epochs"])
phi = np.asarray(z["phi"]).astype(np.float32)
# a 10k-epoch run snapshots 400 frames; stride so the GIF stays slide-sized
# while still spanning the whole run (the final frame is always kept)
MAXF = int(os.environ.get("SMARTPIX_GIF_MAXF", "240"))
if len(ep) > MAXF:
    k = int(np.ceil(len(ep) / MAXF))
    sel = np.unique(np.r_[np.arange(0, len(ep), k), len(ep) - 1])
    ep, phi = ep[sel], phi[sel]
    print(f"  strided every {k}th frame -> {len(ep)} frames")
# softmax per frame, in log-space for stability
p = np.exp(phi - phi.max(1, keepdims=True))
p /= p.sum(1, keepdims=True)
peak = p.argmax(1)
PI, PJ, PP = IA[peak], IB[peak], p.max(1)
print(f"seed {SEED}: {len(ep)} frames, epochs {ep[0]}..{ep[-1]}, "
      f"final peak ({PI[-1]},{PJ[-1]}) p={PP[-1]:.4f}  <- {src}")

acc = seed_color(SEED)
fig = plt.figure(figsize=(12.5, 7.0), dpi=105)
fig.patch.set_facecolor("white")
ax  = fig.add_axes([0.055, 0.120, 0.490, 0.780])
axc = fig.add_axes([0.552, 0.120, 0.013, 0.780])
axt = fig.add_axes([0.700, 0.120, 0.275, 0.330])   # peak-pair trajectory
axp = fig.add_axes([0.700, 0.600, 0.275, 0.300])   # peak probability

img = np.full((T, T), np.nan)
cmap = matplotlib.colormaps["magma"].copy(); cmap.set_bad("#ffffff")
im  = ax.imshow(img, origin="lower", cmap=cmap, norm=LogNorm(VMIN, VMAX),
                interpolation="nearest", extent=[0, T, 0, T])
star, = ax.plot([], [], "*", ms=17, mfc="none", mec="#22d3ee", mew=2.0, zorder=5)
ax.plot([0, T], [0, T], ls=":", lw=0.9, c="#94a3b8", zorder=4)
ax.set_xlabel("later time  $t_j$", fontsize=12)
ax.set_ylabel("earlier time  $t_i$", fontsize=12)
ttl = ax.set_title("", fontsize=13, pad=9)
ax.tick_params(labelsize=10)
cb = fig.colorbar(im, cax=axc); cb.set_label("softmax($\\phi$)", fontsize=10, labelpad=2)
cb.ax.tick_params(labelsize=9)

# --- right column: how the peak moved, and how committed it got --------------
axt.plot(ep, PI, lw=1.4, c="#2563eb", label="earlier time  $t_i$")
axt.plot(ep, PJ, lw=1.4, c="#ea580c", label="later time  $t_j$")
mi, = axt.plot([], [], "o", ms=6, c="#2563eb"); mj, = axt.plot([], [], "o", ms=6, c="#ea580c")
axt.set_xlim(ep[0], ep[-1]); axt.set_ylim(0, T)
axt.set_xlabel("epoch", fontsize=10); axt.set_ylabel("peak pair time slice", fontsize=10)
axt.legend(fontsize=9, loc="upper right", framealpha=.9)
axt.grid(alpha=.25); axt.tick_params(labelsize=9)

axp.semilogy(ep, PP, lw=1.6, c=acc)
mp, = axp.plot([], [], "o", ms=6, c=acc)
axp.axhline(1.0 / 5050, ls="--", lw=1.0, c="#94a3b8")
axp.text(ep[-1], 1.0 / 5050, "  uniform (1/5050)", fontsize=8.5, c="#64748b",
         va="bottom", ha="right")
axp.set_xlim(ep[0], ep[-1]); axp.set_ylabel("peak probability", fontsize=10)
axp.set_xlabel("epoch", fontsize=10)
axp.grid(alpha=.25, which="both"); axp.tick_params(labelsize=9)

cap = fig.text(0.065, 0.028, "", fontsize=11.5, color="#334155")
fig.text(0.977, 0.028, f"seed {SEED} · true softmax($\\phi$) from phi_history.npz",
         fontsize=10, color="#94a3b8", ha="right")

HOLD = 18   # extra frames on the final state so the loop lands on the answer
NF   = len(ep) + HOLD

def draw(k):
    f = min(k, len(ep) - 1)
    img = np.full((T, T), np.nan)
    img[IA, IB] = p[f]
    im.set_data(img)
    star.set_data([PJ[f] + .5], [PI[f] + .5])
    mi.set_data([ep[f]], [PI[f]]); mj.set_data([ep[f]], [PJ[f]])
    mp.set_data([ep[f]], [PP[f]])
    ttl.set_text(f"seed {SEED} — learned pair distribution,  epoch {ep[f]}")
    cap.set_text(f"peak pair  ({PI[f]}, {PJ[f]})    separation {PJ[f]-PI[f]}    "
                 f"p = {PP[f]:.4f}    ({PP[f]*5050:.0f}× uniform)")
    return im, star, mi, mj, mp, ttl, cap

os.makedirs(OUT, exist_ok=True)
dst = os.path.join(OUT, f"prob_evolution_seed{SEED}.gif")
FuncAnimation(fig, draw, frames=NF, blit=False).save(
    dst, writer=PillowWriter(fps=14))
plt.close(fig)
print(f"wrote {dst}  ({os.path.getsize(dst)/1e6:.2f} MB, {NF} frames)")

# sidecar so the slide caption can never drift from the figure it describes
import json
meta = dict(seed=SEED, frames=int(len(ep)), ep0=int(ep[0]), ep1=int(ep[-1]),
            anchor=int(PI[-1]), partner=int(PJ[-1]), sep=int(PJ[-1] - PI[-1]),
            p=float(PP[-1]), xuniform=float(PP[-1] * 5050), gif=os.path.basename(dst))
json.dump(meta, open(dst.replace(".gif", ".json"), "w"), indent=1)
print("  sidecar:", json.dumps(meta))
