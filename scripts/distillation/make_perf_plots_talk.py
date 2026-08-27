"""
Best-seed performance plots for the 3 threshold cases (for the talk).
Ports the summary-plot + pull-plot recipe from
  legacy/notebooks/physics/from_weights.ipynb
to the ViT_Max + SoftQuantize models, loading each case's BEST-seed
best.weights.hdf5 and evaluating on that case's own test set.

Cases (best seed = lowest val NLL):
  1) 2_5 no noise   : part1_long_5k_fixed20/seed_3042   (full set, noise=-1, TS[11,26])
  2) 2_5 +noise     : part1_long_2_5_noise_contained/seed_4042 (contained, baked noise, TS[11,26])
  3) 1_6 +noise     : part1_long_1_6_iid_contained/seed_1042   (contained, baked noise, TS[6,31])

Outputs 6 PNGs to runs/perf_plots_talk/:
  summary_<case>.png  (2x2 true-minus-pred vs true, with sigma band; x,y[um], alpha,beta[deg])
  pull_<case>.png     (2x2 pull hists with gaussian mu/sigma fit)
Run in the smartpix-2bit env:
  /work/users/das214/envs/smartpix-2bit/bin/python make_perf_plots_talk.py
"""
import os, sys, json, gc
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

R = "/work/users/das214/SmartPixels/smart-pixels-ml"
HELPERS = os.path.join(R, "two_bit_optimization_helpers")
sys.path.insert(0, HELPERS)
import tensorflow as tf
for g in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
from prepare_tfrecords import generate_tfrecords, load_tfrecords
from train import create_model

pi = np.pi
OUT = os.path.join(R, "runs", "perf_plots_talk"); os.makedirs(OUT, exist_ok=True)
BASE = "/work/projects/SmartPixML/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
NEW  = "/depot/cms/users/das214/datasets/dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d"
LEVELS = np.array([0., 1., 2., 3.], dtype=np.float32)

CASES = [
    dict(key="2_5_no_noise", title="2_5  (no noise)  — seed 3042",
         weights=f"{R}/runs/part1_long_5k_fixed20/seed_3042/best.weights.hdf5", kind="nonoise", ts=[11, 26]),
    dict(key="2_5_noise", title="2_5  (+noise σ=4.64 mV)  — seed 4042",
         weights=f"{R}/runs/part1_long_2_5_noise_contained/seed_4042/best.weights.hdf5",
         kind="baked", tfr=f"{BASE}/TFR_files_2_5_noise_corr_contained"),
    dict(key="1_6_noise", title="1_6  (+noise σ=4.64 mV)  — seed 1042",
         weights=f"{R}/runs/part1_long_1_6_iid_contained/seed_1042/best.weights.hdf5",
         kind="baked", tfr=f"{BASE}/TFR_files_1_6_iid_contained"),
]


def get_test_gen(case):
    """Return (val/test generator, labels_scale) reproducing the run's exact pipeline."""
    if case["kind"] == "nonoise":
        _, _, tr, va = generate_tfrecords(dataset_dir=NEW, model_type="ViT_Max", train_batch_size=5000,
            val_batch_size=5000, select_contained=False, timeslices=2, tfrecords_exist=True, seed=42,
            time_stamps_override=case["ts"])
        _, vg = load_tfrecords(tr, va, noise=-1, seed=42, shuffle=False)
        meta = json.load(open(os.path.join(va, "metadata.json")))
    else:
        tr = os.path.join(case["tfr"], "TFR_train"); va = os.path.join(case["tfr"], "TFR_test")
        _, vg = load_tfrecords(tr, va, noise=-1, seed=42, shuffle=False)
        meta = json.load(open(os.path.join(va, "metadata.json")))
    return vg, np.array(meta["labels_scale"], dtype=np.float64), meta.get("labels_list")


def predict_df(case):
    tf.keras.backend.clear_session(); gc.collect()
    model = create_model("ViT_Max", timeslices=2, soft_quantize_layer=True,
                         initial_thresholds=[1., 2., 3.], threshold_offset=0.0, initial_levels=LEVELS)
    model.load_weights(case["weights"])
    gen, scale, llist = get_test_gen(case)
    preds, truth = [], []
    for i in range(len(gen)):
        X, y = gen[i]
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
    return df, scale, len(df)


# ---- pull plot (ported from from_weights cell 38/39) ----
def gauss(x, A, mu, sigma): return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def pull_ax(ax, df, var, name):
    h = ax.hist(df[var], bins=np.linspace(-5, 5, 50), histtype='step')
    ax.set_xlabel(name); ax.set_yscale('log')
    xdata = h[1][:-1] + 3/50.; ydata = h[0]
    try:
        pars, _ = curve_fit(gauss, xdata, ydata, p0=[max(ydata.max(),1), 0, 1], maxfev=10000)
        xb = np.linspace(-5, 5, 100); ax.plot(xb, gauss(xb, *pars), color='black')
        mu, sg = pars[1], abs(pars[2])
    except Exception:
        mu, sg = float(np.mean(df[var])), float(np.std(df[var]))
    ax.set_ylim(0.5, max(ydata.max()*2, 10))
    ax.text(-4.6, ax.get_ylim()[1]*0.15, r"$\mu$=%.2f" % mu)
    ax.text(-4.6, ax.get_ylim()[1]*0.03, r"$\sigma$=%.2f" % sg)
    return mu, sg

def make_pull(df, title, path):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 6))
    fig.suptitle("Pull — " + title, fontsize=12)
    r = {}
    r['x']    = pull_ax(ax[0][0], df, 'pullx',    r'$x$ pull')
    r['y']    = pull_ax(ax[0][1], df, 'pully',    r'$y$ pull')
    r['cotA'] = pull_ax(ax[1][0], df, 'pullcotA', r'$\cot\alpha$ pull')
    r['cotB'] = pull_ax(ax[1][1], df, 'pullcotB', r'$\cot\beta$ pull')
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(path, dpi=140); plt.close(fig)
    return r


# ---- summary plot (ported from from_weights cell 41/42) ----
def inverse_cot(cota):
    a = np.arctan(1.0/cota); a = np.where(a < 0, a + pi, a); return a

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
    ang   = inverse_cot(df[v2].values*scaling)*180/pi
    angT  = inverse_cot(df[v1].values*scaling)*180/pi
    aup   = np.abs(inverse_cot((df[v2].values + df['sigma'+v2].values)*scaling)*180/pi - ang)
    adn   = np.abs(inverse_cot((df[v2].values - df['sigma'+v2].values)*scaling)*180/pi - ang)
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

def make_summary(df, scale, title, path):
    fig, ax = plt.subplots(2, 2, figsize=(9, 7)); fig.tight_layout(pad=4.0)
    fig.suptitle("Summary — " + title, fontsize=12)
    r = {}
    r['x [um]']    = resid_pos(ax[0][0], df, 'xtrue', 'x', r'$x$ [$\mu$m]', scale[0])
    r['y [um]']    = resid_pos(ax[0][1], df, 'ytrue', 'y', r'$y$ [$\mu$m]', scale[1])
    r['alpha[deg]']= resid_ang(ax[1][0], df, 'cotAtrue', 'cotA', r'$\alpha$ [deg]', scale[2])
    r['beta[deg]'] = resid_ang(ax[1][1], df, 'cotBtrue', 'cotB', r'$\beta$ [deg]', scale[3])
    fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
    return r


summary = {}
for c in CASES:
    print(f"\n===== {c['key']} =====", flush=True)
    df, scale, n = predict_df(c)
    print(f"  events={n}  labels_scale={[round(float(x),3) for x in scale]}", flush=True)
    sp = os.path.join(OUT, f"summary_{c['key']}.png")
    pp = os.path.join(OUT, f"pull_{c['key']}.png")
    res_std = make_summary(df, scale, c['title'], sp)
    pulls   = make_pull(df, c['title'], pp)
    summary[c['key']] = dict(events=n, resolution_std=res_std,
                             pull_mu_sigma={k: [round(v[0],3), round(v[1],3)] for k, v in pulls.items()})
    print(f"  resolution std: {{ " + ", ".join(f'{k}={v:.3f}' for k,v in res_std.items()) + " }", flush=True)
    print(f"  pull sigma: " + ", ".join(f"{k}={v[1]:.2f}" for k,v in pulls.items()), flush=True)
    print(f"  wrote {sp}\n  wrote {pp}", flush=True)

json.dump(summary, open(os.path.join(OUT, "perf_summary.json"), "w"), indent=2)
print("\nALL DONE ->", OUT, flush=True)
