"""
Summary + pull plots for the MDMM x SimpleRouter seeds (mid-flight snapshots).

Same from_weights-style plots as make_perf_plots_simplerouter.py, pointed at
runs/simplerouter_mdmm_discovery. Two differences that matter:
  * these seeds are STILL TRAINING, so there is no result.json -- run metadata
    (epoch, best val, slice pair, angle corr) is read from the live CSV logs,
    and the weights are read from a SNAPSHOT COPY of best.weights.hdf5 (never
    the live file, which the trainer rewrites whenever val improves);
  * runs on CPU (CUDA_VISIBLE_DEVICES='') -- the GPU is full with 4 training
    workers and a 5th process OOMs them.

Run: CUDA_VISIBLE_DEVICES='' python make_perf_plots_mdmm.py [seed ...]
"""
import os, sys, json, gc, csv
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import tensorflow as tf
from prepare_tfrecords import load_tfrecords
from train import create_model

pi = np.pi
OUT = os.path.join(R, "runs", "perf_plots_mdmm"); os.makedirs(OUT, exist_ok=True)
RUN = os.path.join(R, "runs", "simplerouter_mdmm_discovery")
CKPT_SNAP = ("/tmp/claude-978920/-work-users-das214-SmartPixels/"
             "7a0a041e-b87d-47e5-ae22-8b5c100f4193/scratchpad/ckpt")
BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
TFR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)

SEEDS = [int(s) for s in sys.argv[1:]] or [3042, 2042]

_, vg = load_tfrecords(os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test"),
                       noise=-1, seed=42, shuffle=False)
meta = json.load(open(os.path.join(TFR, "TFR_test", "metadata.json")))
SCALE = np.array(meta["labels_scale"], dtype=np.float64)
print("labels_scale:", [round(float(x), 3) for x in SCALE], flush=True)


def live_state(seed):
    """Epoch / best val / last angle corr straight from the running logs."""
    d = os.path.join(RUN, f"seed_{seed}")
    rows = list(csv.DictReader(open(os.path.join(d, "history.csv"))))
    vals = [(float(r["val_loss"]), int(r["epoch"])) for r in rows
            if r.get("val_loss") not in (None, "", "nan", "val_loss")]
    best, best_ep = min(vals) if vals else (float("nan"), -1)
    m = list(csv.DictReader(open(os.path.join(d, "mdmm_epochs.csv"))))[-1]
    rt = list(csv.DictReader(open(os.path.join(d, "router_epochs.csv"))))[-1]
    return dict(epoch=int(rows[-1]["epoch"]) + 1, best=best, best_ep=best_ep,
                corrA=float(m["pred_corr_cotA"]), corrB=float(m["pred_corr_cotB"]),
                pair=[int(rt["i1"]), int(rt["i2"])])


def predict_df(weights):
    tf.keras.backend.clear_session(); gc.collect()
    model = create_model('ViT_Max_SimpleRouter', timeslices=101, soft_quantize_layer=True,
                         initial_thresholds=[1., 2., 3.], threshold_offset=0.0, initial_levels=LEVELS)
    model.load_weights(weights)
    router = model.get_layer('simple_router_output')
    quant = model.get_layer('soft_quantizer_output')
    idx = sorted(int(i) for i in np.array(router.selected_indices()).ravel())
    thr = [float(x) for x in np.array(quant.thresholds).ravel()]
    preds, truth = [], []
    for i in range(len(vg)):
        X, y = vg[i]
        preds.append(np.asarray(model.predict_on_batch(X)))
        truth.append(np.asarray(y))
    P = np.concatenate(preds, 0); Y = np.concatenate(truth, 0)
    cols = ['x','M11','y','M22','cotA','M33','cotB','M44','M21','M31','M32','M41','M42','M43']
    df = pd.DataFrame(P, columns=cols)
    df['xtrue'], df['ytrue'], df['cotAtrue'], df['cotBtrue'] = Y[:,0], Y[:,1], Y[:,2], Y[:,3]
    mn = 1e-9
    for m in ('M11','M22','M33','M44'):
        df[m] = mn + np.maximum(df[m].values, 0.0)
    df['sigmax']    = np.abs(df['M11'])
    df['sigmay']    = np.sqrt(df['M21']**2 + df['M22']**2)
    df['sigmacotA'] = np.sqrt(df['M31']**2 + df['M32']**2 + df['M33']**2)
    df['sigmacotB'] = np.sqrt(df['M41']**2 + df['M42']**2 + df['M43']**2 + df['M44']**2)
    df['pullx']    = (df['xtrue'] - df['x'])       / df['sigmax']
    df['pully']    = (df['ytrue'] - df['y'])       / df['sigmay']
    df['pullcotA'] = (df['cotAtrue'] - df['cotA']) / df['sigmacotA']
    df['pullcotB'] = (df['cotBtrue'] - df['cotB']) / df['sigmacotB']
    del model; gc.collect()
    return df, idx, thr


def gauss(x, A, mu, sigma): return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def pull_ax(ax, df, var, name):
    h = ax.hist(df[var], bins=np.linspace(-5, 5, 50), histtype='step')
    ax.set_xlabel(name); ax.set_yscale('log')
    xdata = h[1][:-1] + 3/50.; ydata = h[0]
    try:
        pars, _ = curve_fit(gauss, xdata, ydata, p0=[max(ydata.max(), 1), 0, 1], maxfev=10000)
        mu, sg = pars[1], abs(pars[2])
        xb = np.linspace(-5, 5, 100); ax.plot(xb, gauss(xb, *pars), color='black')
    except Exception:
        mu, sg = float(np.mean(df[var])), float(np.std(df[var]))
    ax.set_ylim(0.5, max(ydata.max()*2, 10))
    ax.text(-4.6, ax.get_ylim()[1]*0.15, r"$\mu$=%.2f" % mu)
    ax.text(-4.6, ax.get_ylim()[1]*0.03, r"$\sigma$=%.2f" % sg)
    return mu, sg

def make_pull(df, title, path):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 6))
    fig.suptitle("Pull — " + title, fontsize=10)
    r = {}
    r['x']    = pull_ax(ax[0][0], df, 'pullx',    r'$x$ pull')
    r['y']    = pull_ax(ax[0][1], df, 'pully',    r'$y$ pull')
    r['cotA'] = pull_ax(ax[1][0], df, 'pullcotA', r'$\cot\alpha$ pull')
    r['cotB'] = pull_ax(ax[1][1], df, 'pullcotB', r'$\cot\beta$ pull')
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(path, dpi=140); plt.close(fig)
    return r


def inverse_cot(c):
    a = np.arctan(1.0/c); return np.where(a < 0, a + pi, a)

def resid_pos(ax, df, v1, v2, name, scaling):
    a = df[v1].values*scaling; b = df[v2].values*scaling; res = a - b
    nb = 15; xmin, xmax = np.min(a), np.max(a); step = (xmax-xmin)/nb
    sns.regplot(x=a, y=res, x_bins=np.linspace(xmin, xmax, nb), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True '+name); ax.set_ylabel('True − pred '+name)
    sig = df['sigma'+v2].values*scaling
    up, dn = [], []
    for i in range(nb):
        m = (a > xmin+i*step) & (a < xmin+(i+1)*step)
        mu = np.mean(res[m]) if m.any() else 0.0; s = np.mean(sig[m]) if m.any() else 0.0
        up.append(mu+s); dn.append(mu-s)
    ax.fill_between(np.linspace(xmin, xmax, nb), up, dn, alpha=0.2)
    return float(np.std(res))

def resid_ang(ax, df, v1, v2, name, scaling):
    ang  = inverse_cot(df[v2].values*scaling)*180/pi
    angT = inverse_cot(df[v1].values*scaling)*180/pi
    aup = np.abs(inverse_cot((df[v2].values + df['sigma'+v2].values)*scaling)*180/pi - ang)
    adn = np.abs(inverse_cot((df[v2].values - df['sigma'+v2].values)*scaling)*180/pi - ang)
    nb = 15; xmin, xmax = np.min(angT), np.max(angT); step = (xmax-xmin)/nb
    res = angT - ang
    sns.regplot(x=angT, y=res, x_bins=np.linspace(xmin, xmax, nb), fit_reg=None, marker='.', ax=ax)
    ax.set_xlabel('True '+name); ax.set_ylabel('True − pred '+name)
    up, dn = [], []
    for i in range(nb):
        m = (angT > xmin+i*step) & (angT < xmin+(i+1)*step)
        mu = np.mean(res[m]) if m.any() else 0.0
        up.append(mu+(np.mean(aup[m]) if m.any() else 0.0)); dn.append(mu-(np.mean(adn[m]) if m.any() else 0.0))
    ax.fill_between(np.linspace(xmin, xmax, nb), up, dn, alpha=0.2)
    return float(np.std(res))

def make_summary(df, title, path):
    fig, ax = plt.subplots(2, 2, figsize=(9, 7)); fig.tight_layout(pad=4.0)
    fig.suptitle("Summary — " + title, fontsize=10)
    r = {}
    r['x [um]']     = resid_pos(ax[0][0], df, 'xtrue', 'x', r'$x$ [$\mu$m]', SCALE[0])
    r['y [um]']     = resid_pos(ax[0][1], df, 'ytrue', 'y', r'$y$ [$\mu$m]', SCALE[1])
    r['alpha[deg]'] = resid_ang(ax[1][0], df, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', SCALE[2])
    r['beta[deg]']  = resid_ang(ax[1][1], df, 'cotBtrue', 'cotB', r'$\beta$ [deg]', SCALE[3])
    fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
    return r


summary = {}
for seed in SEEDS:
    st = live_state(seed)
    snap = os.path.join(CKPT_SNAP, f"seed_{seed}.hdf5")
    if not os.path.exists(snap):                       # fall back to the live file
        snap = os.path.join(RUN, f"seed_{seed}", "best.weights.hdf5")
    df, idx, thr = predict_df(snap)
    tag = (f"seed {seed} @ ep {st['epoch']}/5000 — slices {idx}, "
           f"T=[{thr[0]:.1f}, {thr[1]:.1f}, {thr[2]:.1f}] mV, best NLL {st['best']:.0f}")
    print(f"\n===== {tag} =====", flush=True)
    sp = os.path.join(OUT, f"summary_mdmm_seed{seed}.png")
    pp = os.path.join(OUT, f"pull_mdmm_seed{seed}.png")
    res = make_summary(df, tag, sp)
    pl = make_pull(df, tag, pp)
    summary[str(seed)] = dict(epoch=st['epoch'], slices=idx,
                              thresholds=[round(x, 2) for x in thr],
                              best_nll=st['best'], best_epoch=st['best_ep'],
                              corr_cotA=st['corrA'], corr_cotB=st['corrB'],
                              events=int(len(df)), resolution_std=res,
                              pull_mu_sigma={k: [round(v[0], 3), round(v[1], 3)] for k, v in pl.items()})
    print(f"  events={len(df)}  res std: " + ", ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    print(f"  pull sigma: " + ", ".join(f"{k}={v[1]:.2f}" for k, v in pl.items()), flush=True)
    print(f"  wrote {sp}\n  wrote {pp}", flush=True)

json.dump(summary, open(os.path.join(OUT, "perf_summary_mdmm.json"), "w"), indent=2)
print("\nALL DONE ->", OUT, flush=True)
