"""offset=0 threshold convergence across ALL seeds: extract each run's per-epoch
threshold trajectory (matched to its checkpoint dir by best_val_loss) and plot
all 20 seeds (faint) + the median trajectory (bold), flipped-y like before."""
import os, sys, json, glob
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from train import create_model

RUN = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1"
FIG = RUN + "/slides/figs"
WDIR = RUN + "/weights"
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
COL = ['#1f77b4', '#ff7f0e', '#2ca44c']
STEP = 3                                   # subsample epochs for speed (trajectory is smooth)
epoch_of = lambda f: int(os.path.basename(f).split('.')[1].split('-')[0])
vloss_of = lambda f: float(f.split('-v')[1].split('.hdf5')[0])


def dir_minval(d):
    fs = glob.glob(d + "/*.hdf5")
    return min(vloss_of(f) for f in fs) if fs else 1e9, len(fs)


def main():
    runs = [json.loads(l) for l in open(RUN + '/threshold_runs.jsonl') if l.strip()]
    dirs = [d for d in glob.glob(WDIR + "/*/") if dir_minval(d)[1] >= 280]   # converged only
    dmin = {d: dir_minval(d)[0] for d in dirs}
    model = create_model('ViT_Max', timeslices=2, soft_quantize_layer=True,
                         initial_thresholds=[5, 15, 45], threshold_offset=0.0, initial_levels=LEVELS)
    layer = model.get_layer('soft_quantizer_output')

    all_eps, all_thr, used = None, [], []
    for r in runs:
        # match this run to its checkpoint dir by best_val_loss (filename v is .2f)
        d = min(dirs, key=lambda d: abs(dmin[d] - r['best_val_loss']))
        if abs(dmin[d] - r['best_val_loss']) > 5:        # no good match
            continue
        fs = sorted(glob.glob(d + "/*.hdf5"), key=epoch_of)[::STEP]
        eps, thr = [], []
        for f in fs:
            model.load_weights(f); eps.append(epoch_of(f))
            thr.append([float(t) for t in np.array(layer.thresholds).ravel()])
        all_eps = np.array(eps); all_thr.append(np.array(thr)); used.append(r['seed'])
        print(f"  seed {r['seed']}: {len(eps)} pts, final {[round(x,1) for x in thr[-1]]}")
    all_thr = np.array(all_thr)              # (n_runs, n_epochs, 3)
    np.savez(FIG + "/traj_offset0_allseeds.npz", eps=all_eps, thr=all_thr, seeds=used)

    med = np.median(all_thr, axis=0)         # (n_epochs, 3)
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    for i in range(3):
        for k in range(all_thr.shape[0]):
            ax.plot(all_thr[k, :, i], all_eps, color=COL[i], lw=0.8, alpha=0.25)
        ax.plot(med[:, i], all_eps, color=COL[i], lw=2.8, label=f"T{i} median (→{med[-1, i]:.1f})")
    ax.invert_yaxis()
    ax.set_xlabel("threshold value"); ax.set_ylabel("epoch  (training progresses downward)")
    ax.set_title(f"offset=0: threshold convergence — all {all_thr.shape[0]} seeds (faint) + median (bold)")
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); out = FIG + "/epoch_vs_threshold_offset0_allseeds.png"
    fig.savefig(out, dpi=130); print("wrote", out)


main()
