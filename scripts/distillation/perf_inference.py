"""
Performance (residual + pull) plots comparing the best offset=10 vs best offset=0
Part-1 ViT models on the 3srb test set. Each model has the SoftQuantize layer
built in, so it takes the RAW input and digitizes internally with its learned
thresholds. Residuals shown in physical units (x,y in um; cotA,cotB unitless);
pulls = residual/predicted-sigma (unitless).
"""
import os, sys, json, glob
sys.path.insert(0, "/work/users/das214/SmartPixels/smart-pixels-ml/two_bit_optimization_helpers")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from train import create_model
from prepare_tfrecords import generate_tfrecords, load_tfrecords

NEW = '/depot/cms/users/das214/datasets/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d'
RUN = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1"
FIG = os.path.join(RUN, "slides", "figs")
os.makedirs(FIG, exist_ok=True)
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)
LSCALE = np.array([123.47780266, 30.86897376, 8.66687422, 1.98544467])   # x,y,cotA,cotB
PNAME = ['x (um)', 'y (um)', 'cotAlpha', 'cotBeta']
MINVAL = 1e-9


def best_ckpt(d):
    fs = glob.glob(os.path.join(d, "*.hdf5"))
    return min(fs, key=lambda f: float(f.split('-v')[1].split('.hdf5')[0]))


def sigmas_from_14(p):
    """diagonal predictive sigma for each of the 4 params from the 14-vector."""
    diag = MINVAL + np.maximum(p[:, 1:8:2], 0.0)          # (N,4) Cholesky diag
    off = p[:, 8:14]                                       # (N,6)
    N = p.shape[0]
    L = np.zeros((N, 4, 4), np.float32)
    L[:, 0, 0] = diag[:, 0]
    L[:, 1, 0] = off[:, 0]; L[:, 1, 1] = diag[:, 1]
    L[:, 2, 0] = off[:, 1]; L[:, 2, 1] = off[:, 2]; L[:, 2, 2] = diag[:, 2]
    L[:, 3, 0] = off[:, 3]; L[:, 3, 1] = off[:, 4]; L[:, 3, 2] = off[:, 5]; L[:, 3, 3] = diag[:, 3]
    Sig = np.matmul(L, np.transpose(L, (0, 2, 1)))
    return np.sqrt(np.maximum(np.diagonal(Sig, axis1=1, axis2=2), 1e-12))   # (N,4)


def evaluate(ckpt_dir, init_thr, offset, vg):
    model = create_model('ViT_Max', timeslices=2, soft_quantize_layer=True,
                         initial_thresholds=init_thr, threshold_offset=offset, initial_levels=LEVELS)
    model.load_weights(best_ckpt(ckpt_dir))
    res, pull = [], []
    for x, y in vg:
        p = model(x, training=False).numpy()
        mu = p[:, 0:8:2]; sig = sigmas_from_14(p)
        r = (mu - y.numpy())                  # normalized residual
        res.append(r * LSCALE)                # physical residual
        pull.append(r / sig)                  # unitless pull
    return np.concatenate(res), np.concatenate(pull)


def plot_compare(d0, d10):
    # d0/d10 = dict(res=..., pull=...) ; offset=0 green, offset=10 red
    for kind, title, fname in [('res', 'Residuals (physical units)', 'perf_residuals.png'),
                               ('pull', 'Pulls (residual / predicted sigma)', 'perf_pulls.png')]:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
        for i in range(4):
            a = axes[i]
            for d, c, lab in [(d0, 'tab:green', 'offset=0'), (d10, 'tab:red', 'offset=10')]:
                if d is None: continue
                v = d[kind][:, i]
                rng = (-4, 4) if kind == 'pull' else np.percentile(np.abs(v), 99) * np.array([-1, 1])
                a.hist(v, bins=80, range=tuple(rng), histtype='step', color=c, lw=1.8,
                       label=f"{lab}  (μ={v.mean():.2f}, σ={v.std():.2f})", density=True)
            a.set_title(PNAME[i]); a.grid(alpha=0.3); a.legend(fontsize=7)
            if kind == 'pull':
                xs = np.linspace(-4, 4, 200); a.plot(xs, np.exp(-xs**2/2)/np.sqrt(2*np.pi), 'k--', lw=1, alpha=0.6)
        fig.suptitle(title + "  —  best offset=0 vs best offset=10, on the 3srb test set")
        fig.tight_layout(); out = os.path.join(FIG, fname); fig.savefig(out, dpi=130); plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    _, _, tr, va = generate_tfrecords(dataset_dir=NEW, model_type='ViT_Max', train_batch_size=5000,
        val_batch_size=5000, select_contained=False, timeslices=2, tfrecords_exist=True, seed=42,
        time_stamps_override=[11, 26])
    _, vg = load_tfrecords(tr, va, noise=-1, seed=42)

    o0 = [json.loads(l) for l in open(os.path.join(RUN, 'threshold_runs.jsonl')) if l.strip()]
    best0 = min(o0, key=lambda r: r['best_val_loss'])
    ck0 = os.path.join(RUN, "weights/weights-2t-ViT_Max-soft_quantize_layer-49b6ef00-checkpoints")
    print(f"eval offset=0 best (seed {best0['seed']})...")
    r0, p0 = evaluate(ck0, best0['init_thresholds'], 0.0, vg)
    d0 = {'res': r0, 'pull': p0}

    OFF10 = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/new_dataset_part1_off10best"
    d10 = None
    if os.path.exists(os.path.join(OFF10, 'result.json')):
        res = json.load(open(os.path.join(OFF10, 'result.json')))
        print(f"eval offset=10 best (seed {res['seed']})...")
        r10, p10 = evaluate(res['ckpt_dir'], res['init_thresholds'], res['offset'], vg)
        d10 = {'res': r10, 'pull': p10}
    else:
        print("offset=10 re-run not done yet -> plotting offset=0 only for now")

    plot_compare(d0, d10)
    summ = {'offset0_resid_um_std': [float(r0[:, i].std()) for i in range(4)],
            'offset0_pull_std': [float(p0[:, i].std()) for i in range(4)]}
    if d10: summ['offset10_resid_um_std'] = [float(r10[:, i].std()) for i in range(4)]; summ['offset10_pull_std'] = [float(p10[:, i].std()) for i in range(4)]
    json.dump(summ, open(os.path.join(FIG, 'perf_summary.json'), 'w'), indent=1)
    print("PERF SUMMARY:", json.dumps(summ, indent=1))
