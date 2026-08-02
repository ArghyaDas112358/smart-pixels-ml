"""
Full plot set + report data for the Part-1 LONG study (1000 epochs, offset=0,
anneal 1->67 stretched over 1000). Three batches:
  A random init, B fixed [40,40.1,40.2], C fixed [20,20.1,20.2]; 5 seeds each.
Reads per-epoch CSVs (epoch,loss,val_loss,T0,T1,T2) + result.json.
Writes all figures into runs/part1_long_study/figs/ (CPU only).
"""
import os, json, glob, shutil
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml/runs"
OUT = f"{R}/part1_long_study"; FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
BATCHES = [("A random", f"{R}/part1_long", "random"),
           ("B fixed[40,40.1,40.2]", f"{R}/part1_long_fixed40", "[40,40.1,40.2]"),
           ("C fixed[20,20.1,20.2]", f"{R}/part1_long_fixed20", "[20,20.1,20.2]")]
SEEDS = [42, 1042, 2042, 3042, 4042]
TCOL = ['#1f77b4', '#ff7f0e', '#2ca02c']     # T0,T1,T2
GCOL = {'better': 'tab:green', 'worse': 'tab:red'}
THR = -40000.0                                # NLL group cut
sm = lambda v, k=25: np.convolve(v, np.ones(k)/k, mode='valid')


def load():
    runs = []
    for bn, bd, binit in BATCHES:
        for s in SEEDS:
            r = json.load(open(f"{bd}/seed_{s}/result.json"))
            a = np.genfromtxt(f"{bd}/seed_{s}/threshold_loss_epochs.csv", delimiter=",", names=True)
            grp = 'better' if r['best_val_loss'] < THR else 'worse'
            runs.append(dict(batch=bn, binit=binit, seed=s, csv=a,
                             ep=a['epoch'], val=a['val_loss'], loss=a['loss'],
                             T=np.vstack([a['T0'], a['T1'], a['T2']]).T,
                             ft=r['final_thresholds'], best=r['best_val_loss'],
                             final=r['final_val_loss'], grp=grp))
    return runs


def fig_nll_by_group(runs):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for r in runs:
        ax.plot(r['ep'][24:], sm(r['val']), color=GCOL[r['grp']], lw=1.4, alpha=0.8)
    for g, c in GCOL.items():
        ax.plot([], [], color=c, lw=2, label=f"{g} ({sum(r['grp']==g for r in runs)} runs)")
    ax.set_xlabel("epoch"); ax.set_ylabel("val NLL / batch (25-ep avg)")
    ax.set_title("val NLL vs epoch — all 15 runs, colored by outcome group\n(clean bimodal split: better ~-48k vs worse ~-32k)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    p = f"{FIG}/01_nll_vs_epoch_by_group.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_threshold_by_index(runs):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for i, nm in enumerate(['T0', 'T1', 'T2']):
        for r in runs:
            ax[i].plot(r['ep'], r['T'][:, i], color=GCOL[r['grp']], lw=1.0, alpha=0.6)
        ax[i].set_title(nm); ax[i].set_xlabel("epoch"); ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("threshold value")
    for g, c in GCOL.items(): ax[0].plot([], [], color=c, lw=2, label=g)
    ax[0].legend()
    fig.suptitle("Each threshold vs epoch — all 15 runs by group (T1 is the discriminator: better keeps T1~11, worse drops to ~7.5)")
    fig.tight_layout(); p = f"{FIG}/02_threshold_vs_epoch_by_index.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_epoch_vs_threshold_perbatch(runs):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for j, (bn, bd, binit) in enumerate(BATCHES):
        br = [r for r in runs if r['batch'] == bn]
        for r in br:
            for i in range(3):
                ax[j].plot(r['T'][:, i], r['ep'], color=TCOL[i], lw=1.0, alpha=0.5)
        med = np.median(np.array([r['T'] for r in br]), axis=0)
        for i in range(3):
            ax[j].plot(med[:, i], br[0]['ep'], color=TCOL[i], lw=2.6, label=f"T{i} median ->{med[-1,i]:.1f}")
        ax[j].invert_yaxis(); ax[j].set_xlabel("threshold value"); ax[j].set_title(f"{bn}\ninit {binit}")
        ax[j].legend(fontsize=8, loc='lower right'); ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("epoch  (training progresses downward)")
    fig.suptitle("Threshold convergence per batch — 5 seeds (faint) + median (bold), epochs increase downward")
    fig.tight_layout(); p = f"{FIG}/03_epoch_vs_threshold_perbatch.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_final_threshold_hist(runs):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ft = {g: np.array([r['ft'] for r in runs if r['grp'] == g]) for g in GCOL}
    for i, nm in enumerate(['T0', 'T1', 'T2']):
        allv = np.array([r['ft'][i] for r in runs])
        # SHARED bin edges across both groups, padded so the clustered group isn't a sliver
        lo, hi = allv.min(), allv.max(); pad = 0.06 * (hi - lo + 1e-6)
        edges = np.linspace(lo - pad, hi + pad, 13)            # 12 common bins
        for g, c in GCOL.items():
            ax[i].hist(ft[g][:, i], bins=edges, color=c, alpha=0.55, label=g, edgecolor='k', lw=0.5)
        ax[i].axvline(np.median(allv), color='k', ls='--', lw=1.2, label=f"all-median {np.median(allv):.2f}")
        ax[i].set_title(nm); ax[i].set_xlabel("final threshold"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7)
        ax[i].yaxis.get_major_locator().set_params(integer=True)
    ax[0].set_ylabel("# runs")
    fig.suptitle("Final-threshold distributions across all 15 runs, split by outcome group (shared bins)")
    fig.tight_layout(); p = f"{FIG}/04_final_threshold_hist.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_best_nll_strip(runs):
    fig, ax = plt.subplots(figsize=(8, 5))
    bn_list = [b[0] for b in BATCHES]
    rng = np.random.default_rng(0)
    for r in runs:
        x = bn_list.index(r['batch']) + (rng.uniform(-0.12, 0.12))
        ax.scatter(x, r['best'], color=GCOL[r['grp']], s=70, edgecolor='k', lw=0.5, zorder=3)
    ax.axhline(THR, color='gray', ls=':', label=f"group cut {THR:,.0f}")
    ax.set_xticks(range(3)); ax.set_xticklabels(['A\nrandom', 'B\nfixed40', 'C\nfixed20'])
    ax.set_ylabel("best val NLL / batch"); ax.set_title("Best NLL per run by batch — both clusters appear in every batch\n(green=better, red=worse)")
    ax.legend(); ax.grid(alpha=0.3, axis='y'); fig.tight_layout()
    p = f"{FIG}/05_best_nll_strip.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


def fig_final_scatter(runs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for r in runs:
        ax[0].scatter(r['ft'][0], r['ft'][1], color=GCOL[r['grp']], s=60, edgecolor='k', lw=0.4)
        ax[1].scatter(r['ft'][1], r['ft'][2], color=GCOL[r['grp']], s=60, edgecolor='k', lw=0.4)
    ax[0].set_xlabel("T0"); ax[0].set_ylabel("T1"); ax[0].set_title("final T0 vs T1")
    ax[1].set_xlabel("T1"); ax[1].set_ylabel("T2"); ax[1].set_title("final T1 vs T2")
    for g, c in GCOL.items(): ax[0].scatter([], [], color=c, label=g)
    ax[0].legend()
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle("Final thresholds colored by outcome — better cluster separates mainly in T1")
    fig.tight_layout(); p = f"{FIG}/06_final_thresholds_scatter.png"; fig.savefig(p, dpi=130); plt.close(fig); return p


if __name__ == "__main__":
    runs = load()
    made = [fig_nll_by_group(runs), fig_threshold_by_index(runs),
            fig_epoch_vs_threshold_perbatch(runs), fig_final_threshold_hist(runs),
            fig_best_nll_strip(runs), fig_final_scatter(runs)]
    # copy in the 3 diagnostic figs already produced
    for src, dst in [(f"{R}/part1_long/k_vs_convergence.png", "07_k_vs_convergence.png"),
                     (f"{R}/part1_long/good_vs_worse_same_thresholds.png", "08_good_vs_worse_same_thresholds.png"),
                     (f"{R}/part1_long_fixed40/good_vs_worse_B_seed42_vs_1042.png", "09_good_vs_worse_B_42_vs_1042.png")]:
        if os.path.exists(src): shutil.copy(src, f"{FIG}/{dst}"); made.append(f"{FIG}/{dst}")
    print("FIGURES:")
    for p in made: print(" ", p)
    # dump table json for the report
    tbl = [dict(batch=r['batch'], seed=r['seed'], final_T=[round(x,2) for x in r['ft']],
                best=round(r['best']), final=round(r['final']), group=r['grp']) for r in runs]
    json.dump(tbl, open(f"{OUT}/runs_table.json", "w"), indent=1)
    print("wrote", f"{OUT}/runs_table.json")
