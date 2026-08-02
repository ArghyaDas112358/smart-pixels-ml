"""
Per-seed performance plots for the 4 SoftRouter discovery seeds.
For each seed: load best.weights.hdf5 into ViT_Max_SoftRouter (101 slices),
read the committed slice pair + thresholds FROM THE MODEL (cross-checked vs
result.json), predict on the all-101 discovery test set, and emit the
from_weights-style summary (true-pred vs true, sigma band) and pull plots,
titled with that seed's indices + thresholds.

Outputs 8 PNGs + perf_summary.json to runs/perf_plots_router/.
Run: /work/users/das214/envs/smartpix-2bit/bin/python make_perf_plots_router.py
"""
import os, sys, json, gc, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
sys.path.insert(0, os.path.join(R, "two_bit_optimization_helpers"))
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
from prepare_tfrecords import load_tfrecords
from train import create_model

pi = np.pi
OUT = os.path.join(R, "runs", "perf_plots_router"); os.makedirs(OUT, exist_ok=True)
RUN = os.path.join(R, "runs", "softrouter_discovery")
BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
TFR = os.path.join(BASE, "TFR_files_all101_noise_contained_discovery")
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)

SEEDS = [42, 1042, 2042, 3042]

# one shared test generator + labels scale
_, vg = load_tfrecords(os.path.join(TFR, "TFR_train"), os.path.join(TFR, "TFR_test"), noise=-1, seed=42, shuffle=False)
meta = json.load(open(os.path.join(TFR, "TFR_test", "metadata.json")))
SCALE = np.array(meta["labels_scale"], dtype=np.float64)
print("labels_scale:", [round(float(x), 3) for x in SCALE], flush=True)


def predict_df(weights):
    tf.keras.backend.clear_session(); gc.collect()
    model = create_model('ViT_Max_SoftRouter', timeslices=101, soft_quantize_layer=True,
                         initial_thresholds=[1., 2., 3.], threshold_offset=0.0, initial_levels=LEVELS)
    model.load_weights(weights)
    router = model.get_layer('soft_router_output')
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
    fig.suptitle("Pull — " + title, fontsize=11)
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
    fig.suptitle("Summary — " + title, fontsize=11)
    r = {}
    r['x [um]']     = resid_pos(ax[0][0], df, 'xtrue', 'x', r'$x$ [$\mu$m]', SCALE[0])
    r['y [um]']     = resid_pos(ax[0][1], df, 'ytrue', 'y', r'$y$ [$\mu$m]', SCALE[1])
    r['alpha[deg]'] = resid_ang(ax[1][0], df, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', SCALE[2])
    r['beta[deg]']  = resid_ang(ax[1][1], df, 'cotBtrue', 'cotB', r'$\beta$ [deg]', SCALE[3])
    fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
    return r


summary = {}
for seed in SEEDS:
    sd = os.path.join(RUN, f"seed_{seed}")
    rj = json.load(open(os.path.join(sd, "result.json")))
    df, idx, thr = predict_df(os.path.join(sd, "best.weights.hdf5"))
    rj_idx = sorted(int(i) for i in (rj.get('final_indices') or rj.get('best_indices') or []))
    tag = f"seed {seed} — slices {idx}, T=[{thr[0]:.2f}, {thr[1]:.2f}, {thr[2]:.2f}]"
    match = "OK" if rj_idx == idx else f"MISMATCH vs result.json {rj_idx}"
    print(f"\n===== {tag}  (result.json check: {match}, best NLL {rj.get('best_val_loss'):.0f}) =====", flush=True)
    sp = os.path.join(OUT, f"summary_seed{seed}.png")
    pp = os.path.join(OUT, f"pull_seed{seed}.png")
    res = make_summary(df, tag, sp)
    pl = make_pull(df, tag, pp)
    summary[str(seed)] = dict(slices=idx, thresholds=[round(x, 2) for x in thr],
                              best_nll=rj.get('best_val_loss'), events=int(len(df)),
                              resolution_std=res,
                              pull_mu_sigma={k: [round(v[0], 3), round(v[1], 3)] for k, v in pl.items()})
    print(f"  events={len(df)}  res std: " + ", ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    print(f"  pull sigma: " + ", ".join(f"{k}={v[1]:.2f}" for k, v in pl.items()), flush=True)
    print(f"  wrote {sp}\n  wrote {pp}", flush=True)

json.dump(summary, open(os.path.join(OUT, "perf_summary.json"), "w"), indent=2)
print("\nALL DONE ->", OUT, flush=True)
