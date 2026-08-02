"""
Threshold-optimization study plots for the new 3srb dataset (2t, samples [11,26]).
Builds:
  1) epoch-vs-threshold trajectory (x=threshold, y=epoch, y inverted so epochs
     increase top->bottom) by reading the SoftQuantize thresholds out of every
     per-epoch checkpoint of a run.
  2) thresholds-vs-NLL scatter across the collected runs (offset=0 and offset=10).
Run with CUDA_VISIBLE_DEVICES="" (CPU; just loads small weight files).
"""
import os, sys, json, glob
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from train import create_model

RUN = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1"
FIG = os.path.join(RUN, "slides", "figs")
os.makedirs(FIG, exist_ok=True)
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
COL = ['#1f77b4', '#ff7f0e', '#2ca44c']   # T0, T1, T2


def epoch_of(fp):
    return int(os.path.basename(fp).split('.')[1].split('-')[0])


def extract_trajectory(ckpt_dir, init_thr, offset):
    """Read the 3 SoftQuantize thresholds from every epoch checkpoint."""
    fs = sorted(glob.glob(os.path.join(ckpt_dir, "*.hdf5")), key=epoch_of)
    model = create_model('ViT_Max', timeslices=2, soft_quantize_layer=True,
                         initial_thresholds=init_thr, threshold_offset=offset, initial_levels=LEVELS)
    layer = model.get_layer('soft_quantizer_output')
    eps, thr = [], []
    for f in fs:
        model.load_weights(f)
        eps.append(epoch_of(f))
        thr.append([float(t) for t in np.array(layer.thresholds).ravel()])
    return np.array(eps), np.array(thr)


def plot_epoch_vs_threshold(eps, thr, title, outfile):
    fig, ax = plt.subplots(figsize=(6, 7))
    for i in range(3):
        ax.plot(thr[:, i], eps, color=COL[i], lw=2, label=f"T{i}  (final {thr[-1, i]:.1f})")
        ax.scatter([thr[0, i]], [eps[0]], color=COL[i], marker='o', s=40, zorder=5)   # init (top)
        ax.scatter([thr[-1, i]], [eps[-1]], color=COL[i], marker='*', s=120, zorder=5)  # final (bottom)
    ax.invert_yaxis()                      # epochs increase top -> bottom
    ax.set_xlabel("threshold value")
    ax.set_ylabel("epoch  (training progresses downward)")
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=130); plt.close(fig)
    print("wrote", outfile)


def plot_threshold_vs_nll():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    data = {'offset=0': (os.path.join(RUN, 'threshold_runs.jsonl'), 'tab:green'),
            'offset=10': (os.path.join(RUN, 'threshold_runs_offset10.jsonl'), 'tab:red')}
    for label, (p, c) in data.items():
        if not (os.path.exists(p) and os.path.getsize(p)):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        thr = np.array([r['thresholds'] for r in rows])
        nll = np.array([r['best_val_loss'] for r in rows])   # per-BATCH NLL
        for i in range(3):
            axes[i].scatter(thr[:, i], nll, c=c, label=label, alpha=0.8, edgecolor='k', lw=0.4)
    for i in range(3):
        axes[i].set_xlabel(f"threshold T{i}")
        axes[i].grid(alpha=0.3)
    axes[0].set_ylabel("best val NLL / batch")
    axes[0].legend(fontsize=9)
    fig.suptitle("Optimized thresholds vs achieved NLL per batch (each point = one training run)")
    fig.tight_layout(); out = os.path.join(FIG, "thresholds_vs_nll.png")
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    # --- best offset=0 run: seed 4042 -> ckpt dir 49b6ef00 ---
    o0 = [json.loads(l) for l in open(os.path.join(RUN, 'threshold_runs.jsonl')) if l.strip()]
    best0 = min(o0, key=lambda r: r['best_val_loss'])
    ck0 = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1/weights/weights-2t-ViT_Max-soft_quantize_layer-49b6ef00-checkpoints"
    print(f"best offset=0: seed {best0['seed']} thr={[round(t,2) for t in best0['thresholds']]} init={best0['init_thresholds']}")
    eps, thr = extract_trajectory(ck0, best0['init_thresholds'], 0.0)
    np.savez(os.path.join(FIG, "traj_offset0.npz"), eps=eps, thr=thr)
    plot_epoch_vs_threshold(eps, thr, f"offset=0 best (seed {best0['seed']}): threshold convergence", os.path.join(FIG, "epoch_vs_threshold_offset0.png"))
    plot_threshold_vs_nll()
    print("DONE offset=0 plots")
