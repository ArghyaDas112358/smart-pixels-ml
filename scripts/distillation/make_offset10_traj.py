"""Extract the offset=10 best-run threshold trajectory and make a combined
epoch-vs-threshold plot (offset=0 solid vs offset=10 dashed)."""
import os, sys, json, glob
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import create_model

FIG = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1/slides/figs"
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
COL = ['#1f77b4', '#ff7f0e', '#2ca44c']
epoch_of = lambda f: int(os.path.basename(f).split('.')[1].split('-')[0])


def extract(ckpt_dir, init_thr, offset):
    fs = sorted(glob.glob(os.path.join(ckpt_dir, "*.hdf5")), key=epoch_of)
    m = create_model('ViT_Max', timeslices=2, soft_quantize_layer=True,
                     initial_thresholds=init_thr, threshold_offset=offset, initial_levels=LEVELS)
    L = m.get_layer('soft_quantizer_output')
    eps, thr = [], []
    for f in fs:
        m.load_weights(f); eps.append(epoch_of(f)); thr.append([float(t) for t in np.array(L.thresholds).ravel()])
    return np.array(eps), np.array(thr)


res = json.load(open("/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1_off10best/result.json"))
e10, t10 = extract(res['ckpt_dir'], res['init_thresholds'], res['offset'])
np.savez(os.path.join(FIG, "traj_offset10.npz"), eps=e10, thr=t10)

d0 = np.load(os.path.join(FIG, "traj_offset0.npz")); e0, t0 = d0['eps'], d0['thr']
fig, ax = plt.subplots(figsize=(6.5, 7))
for i in range(3):
    ax.plot(t0[:, i], e0, color=COL[i], lw=2.2, label=f"T{i} offset=0 (→{t0[-1,i]:.1f})")
    ax.plot(t10[:, i], e10, color=COL[i], lw=1.8, ls='--', label=f"T{i} offset=10 (→{t10[-1,i]:.1f})")
ax.invert_yaxis()
ax.set_xlabel("threshold value"); ax.set_ylabel("epoch  (training progresses downward)")
ax.set_title("Threshold convergence: offset=0 (solid) vs offset=10 (dashed)")
ax.legend(loc='lower right', fontsize=8, ncol=1); ax.grid(alpha=0.3)
fig.tight_layout(); out = os.path.join(FIG, "epoch_vs_threshold_both.png")
fig.savefig(out, dpi=130); print("wrote", out)
print("offset=10 final thresholds:", [round(float(x), 2) for x in t10[-1]])
