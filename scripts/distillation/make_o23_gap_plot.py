"""Why the 10K run was killed at epoch 2,001.

The under-training hypothesis said: our arms stop at ~600 epochs, the parent
transitioned at ~2,500, so run longer. We ran longer. It did not transition --
it diverged. This plots the reason: the train/val gap.

The parent's gap PLATEAUS around +3,600 and then COLLAPSES at its transition
(val improves more than train -- a generalization event). Ours grows without
bound. A model whose generalization is degrading every epoch is moving away
from the kind of solution the transition represents, not toward it.
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
OUT = os.path.join(R, "runs", "perf_plots_o22")


def cols(path, tr, va):
    r = list(csv.DictReader(open(path)))
    def g(c):
        v = []
        for x in r:
            try: v.append(float(x[c]))
            except Exception: v.append(np.nan)
        return np.array(v)
    return g(tr), g(va)

ptr, pva = cols(f"{R}/runs/o21v2a2_pairlattice/seed_22042/history.csv", "loss_obj", "val_loss")
otr, ova = cols(f"{R}/runs/o23_armA_n20_10k/seed_30042/history.csv", "plain_nll", "val_plain_nll")

def roll(a, w=50):
    return np.array([np.nanmedian(a[max(0, i - w):i + 1]) for i in range(len(a))])

fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=115)

a = ax[0]
a.plot(np.arange(1, len(pva) + 1), roll(pva - ptr), color="#7c3aed", lw=2.2,
       label="parent — 2 slices, 2-bit (router sampling)")
a.plot(np.arange(1, len(ova) + 1), roll(ova - otr), color="#b91c1c", lw=2.2,
       label="O23 arm A — 20 slices float, slice-dropout 0.15")
a.axvspan(2000, 3000, color="#7c3aed", alpha=.09, zorder=0)
a.annotate("parent's transition:\ngap COLLAPSES +4,680 → +1,679", xy=(3000, 1679),
           xytext=(3900, 9000), fontsize=10.5, color="#5b2566",
           arrowprops=dict(arrowstyle="->", color="#5b2566", lw=1.3))
a.annotate("killed here — ep 2,001\ngap +14,279 and climbing", xy=(2050, 14279),
           xytext=(3500, 14000), fontsize=10.5, color="#b91c1c",
           arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.3))
a.set_xlim(0, 10000); a.set_ylim(-1500, 18500)
a.set_xlabel("epoch", fontsize=11.5)
a.set_ylabel("train → val gap  (rolling median, 50 ep)", fontsize=11.5)
a.set_title("The gap is why: ours diverges, the parent's does not", fontsize=12.5)
a.axhline(0, color="#334155", lw=1, ls="--")
a.grid(alpha=.22, lw=.6); a.legend(fontsize=9.4, loc="lower right", framealpha=.95)

b = ax[1]
pb = np.fmin.accumulate(pva); ob = np.fmin.accumulate(ova)
b.plot(np.arange(1, len(pb) + 1), pb, color="#7c3aed", lw=2.2, label="parent")
b.plot(np.arange(1, len(ob) + 1), ob, color="#b91c1c", lw=2.2, label="O23 arm A (10K attempt)")
b.axvline(871, color="#b91c1c", ls=":", lw=1.3)
b.annotate("best -27,519 at ep 871,\nnever improved again\n(1,130 epochs stale)",
           xy=(1400, -27519), xytext=(1150, -22800), fontsize=10, color="#b91c1c",
           arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.2))
b.set_xlim(0, 3200); b.set_ylim(-42000, -20000)
b.set_xlabel("epoch", fontsize=11.5)
b.set_ylabel("best-so-far validation NLL", fontsize=11.5)
b.set_title("Best-so-far: frozen from epoch 871 onward", fontsize=12.5)
b.grid(alpha=.22, lw=.6); b.legend(fontsize=9.6, loc="lower left", framealpha=.95)

fig.suptitle("O23 — the 10,000-epoch run was killed at epoch 2,001, and why",
             fontsize=14.5, y=.98)
fig.tight_layout(rect=[0, 0, 1, .94])
dst = os.path.join(OUT, "o23_gap.png")
fig.savefig(dst, facecolor="white"); plt.close(fig)
print("wrote", dst)
for e in [500, 1000, 1500, 2000]:
    print(f"  ep {e:>5}  ours gap {np.nanmedian(ova[e-50:e]-otr[e-50:e]):>+9,.0f}"
          f"   parent gap {np.nanmedian(pva[e-50:e]-ptr[e-50:e]):>+9,.0f}")
