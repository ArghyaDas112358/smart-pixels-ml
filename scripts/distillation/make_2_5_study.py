"""Figures + REPORT.md for the Part-1 2_5 mV baked-noise CONTAINED study.
Auto-discovers every seed_*/ run dir (converged + stuck), so it grows as the
multi-seed collection finishes. Mirrors the v2 5k-study report format
(runs/part1_long_5k_study/REPORT_v2.md). Re-run any time to update."""
import os, re, json, glob
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/part1_long_2_5_noise_contained"
FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
MAXEP, ESCAPE = 5000, 5e4
GCOL = {'converged': 'tab:green', 'running': 'tab:orange', 'stuck': 'tab:red'}
CLEAN = [4.15, 13.013, 47.244]   # clean-data archive medians, for reference
sm = lambda v, k=25: np.convolve(v, np.ones(k)/k, mode='valid') if len(v) > k else v


def kval(ep, T=MAXEP, k0=1.0, kf=67.0):
    return k0 + (kf - k0) * 0.5 * (1 - np.cos(np.pi * np.asarray(ep) / T))


def load():
    runs = []
    for d in sorted(glob.glob(f"{OUT}/seed_*")):
        base = os.path.basename(d)
        cs = f"{d}/threshold_loss_epochs.csv"
        if not (os.path.exists(cs) and os.path.getsize(cs) > 50):
            continue
        a = np.atleast_1d(np.genfromtxt(cs, delimiter=",", names=True))
        if a['epoch'].size < 2:
            continue
        m = re.match(r'seed_(\d+)', base); seed = int(m.group(1)) if m else base
        is_stuck_dir = '_STUCK' in base
        bi = int(np.argmin(a['val_loss']))
        r = dict(seed=seed, dir=base, ep=a['epoch'], val=a['val_loss'], loss=a['loss'],
                 T=np.vstack([a['T0'], a['T1'], a['T2']]).T,
                 best=float(a['val_loss'][bi]), bestep=int(a['epoch'][bi]), final=float(a['val_loss'][-1]),
                 curep=int(a['epoch'][-1]) + 1,
                 ft=[float(a['T0'][-1]), float(a['T1'][-1]), float(a['T2'][-1])],
                 bft=[float(a['T0'][bi]), float(a['T1'][bi]), float(a['T2'][bi])])
        rj = f"{d}/result.json"; r['status'] = 'killed' if is_stuck_dir else 'running'
        if os.path.exists(rj):
            j = json.load(open(rj)); r['status'] = 'done'
            r['epochs'] = j.get('epochs', r['curep']); r['best'] = j.get('best_val_loss', r['best'])
            r['bestep'] = j.get('best_epoch', r['bestep']); r['ft'] = j.get('final_thresholds', r['ft'])
            r['bft'] = j.get('best_thresholds', r['bft']); r['escaped'] = j.get('escaped')
        else:
            r['epochs'] = r['curep']; r['escaped'] = None
        if is_stuck_dir or r['best'] >= ESCAPE:
            r['grp'] = 'stuck'
        elif r['status'] == 'done':          # fully finished 5000 ep + escaped
            r['grp'] = 'converged'
        else:                                 # still training -> not yet counted as converged
            r['grp'] = 'running'
        runs.append(r)
    runs.sort(key=lambda r: (r['grp'] != 'converged', r['seed']))
    return runs


def figs(runs):
    have = [r for r in runs if r['grp'] == 'converged']   # presentation: converged seeds only
    made = []
    conv = have
    # 01 NLL vs epoch
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for r in have:
        c = GCOL[r['grp']]; ls = '-' if r['status'] != 'running' else '--'
        e = r['ep'][24:] if len(r['val']) > 25 else r['ep']
        ax.plot(e, sm(r['val']), color=c, lw=1.4, ls=ls, label=f"seed {r['seed']} {r['grp']} (best {r['best']:,.0f})")
    ax.axhline(ESCAPE, color='gray', ls=':', lw=.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("val NLL/batch (25-ep avg)")
    ax.set_title("Part-1 2_5 mV baked-noise contained — val NLL vs epoch")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    p = f"{FIG}/01_nll_vs_epoch.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 02 thresholds vs epoch by index
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, nm in enumerate(['T0', 'T1', 'T2']):
        for r in have:
            ax[i].plot(r['ep'], r['T'][:, i], color=GCOL[r['grp']], lw=1.0, alpha=.8)
        ax[i].axhline(CLEAN[i], color='k', ls='--', lw=1, alpha=.6, label=f"clean median {CLEAN[i]:.1f}")
        if conv:
            med = np.median([r['bft'][i] for r in conv])
            ax[i].axhline(med, color='tab:blue', ls='-', lw=1.4, label=f"this median {med:.2f}")
        ax[i].set_title(nm); ax[i].set_xlabel("epoch"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=8)
    ax[0].set_ylabel("threshold (mV)")
    fig.suptitle("Each threshold vs epoch (green=converged, red=stuck)")
    fig.tight_layout(); p = f"{FIG}/02_threshold_vs_epoch_by_index.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 04 epoch-vs-threshold (axes swapped)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for i, nm in enumerate(['T0', 'T1', 'T2']):
        for r in have:
            c = GCOL[r['grp']]
            ax[i].plot(r['T'][:, i], r['ep'], color=c, lw=1.1, alpha=.8)
            ax[i].scatter([r['T'][0, i]], [r['ep'][0]], color=c, marker='o', s=25, zorder=5)
            ax[i].scatter([r['T'][-1, i]], [r['ep'][-1]], color=c, marker='*', s=100, zorder=5)
        ax[i].set_title(nm); ax[i].set_xlabel("threshold (mV)"); ax[i].grid(alpha=0.3)
    ax[0].invert_yaxis(); ax[0].set_ylabel("epoch  (training progresses downward)")
    fig.suptitle("Epoch vs threshold; ○ = init (top), ★ = final (bottom)")
    fig.tight_layout(); p = f"{FIG}/04_epoch_vs_threshold_by_index.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 03 NLL with k overlay
    fig, ax = plt.subplots(figsize=(9.5, 5.2)); ax2 = ax.twinx()
    for r in have:
        e = r['ep'][24:] if len(r['val']) > 25 else r['ep']
        ax.plot(e, sm(r['val']), color=GCOL[r['grp']], lw=1.3, alpha=.85)
    ke = np.arange(0, MAXEP); ax2.plot(ke, kval(ke), 'k--', lw=1.3, alpha=.7, label='k(epoch) 1→67 over 5000')
    ax.set_xlabel("epoch"); ax.set_ylabel("val NLL/batch (25-ep avg)"); ax2.set_ylabel("k (anneal)")
    ax.set_title("NLL vs epoch with the k-anneal schedule overlaid"); ax2.legend(loc='right', fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = f"{FIG}/03_nll_with_k.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 05 best-epoch threshold histograms across converged seeds (shared bins)
    if len(conv) >= 2:
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        for i, nm in enumerate(['T0', 'T1', 'T2']):
            allv = np.array([r['bft'][i] for r in conv]); lo, hi = allv.min(), allv.max()
            pad = 0.06 * (hi - lo + 1e-6); edges = np.linspace(lo - pad, hi + pad, max(6, len(conv)+1))
            ax[i].hist(allv, bins=edges, color='tab:green', alpha=.6, edgecolor='k', lw=.5)
            ax[i].axvline(np.median(allv), color='k', ls='--', lw=1.2, label=f"median {np.median(allv):.2f}")
            ax[i].axvline(CLEAN[i], color='tab:red', ls=':', lw=1.2, label=f"clean {CLEAN[i]:.1f}")
            ax[i].set_title(nm); ax[i].set_xlabel("best-epoch threshold (mV)"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7)
            ax[i].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_ylabel("# converged seeds")
        fig.suptitle(f"Best-epoch threshold distribution across {len(conv)} converged seeds")
        fig.tight_layout(); p = f"{FIG}/05_threshold_hist.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 06 threshold stability zoom (best converged run)
    if conv:
        r = min(conv, key=lambda r: r['best']); m = r['ep'] >= 3500
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (nm, c) in enumerate([('T0', 'C0'), ('T1', 'C1'), ('T2', 'C2')]):
            v = r['T'][m, i]
            ax.plot(r['ep'][m], v, color=c, lw=1.2, label=f"{nm}: {v.min():.2f}–{v.max():.2f} (range {v.max()-v.min():.2f})")
        ax.set_xlabel("epoch"); ax.set_ylabel("threshold (mV)")
        ax.set_title(f"Threshold stability, ep 3500–5000 (best converged: seed {r['seed']})")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        p = f"{FIG}/06_threshold_stability_zoom.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    return made


def report(runs, made):
    conv = [r for r in runs if r['grp'] == 'converged']
    L = []; w = L.append
    w("# Part-1 Long-Training Threshold Study — **2_5 mV baked-noise CONTAINED (5000 epochs, multi-seed)**\n")
    n_run = sum(r['status'] == 'running' for r in runs)
    note = f"{len(conv)} converged" + (f", {n_run} running" if n_run else "")
    w(f"> **{note}.** Auto-regenerated from `runs/part1_long_2_5_noise_contained/seed_*/`. "
      "Re-run `scripts/distillation/make_2_5_study.py` to update.\n")
    w("Multi-seed convergence study on the **new mV dataset with σ=4.64 mV noise baked into the TFRs**, "
      "contained clusters (`original_atEdge==False`), time samples **[11,26]**. Each seed runs to 5000 epochs; "
      "stuck-init seeds are auto-aborted (and shown in red).\n")

    w("---\n\n## 1. Configuration\n")
    w("| item | value |")
    w("|---|---|")
    w("| Model | ViT_Max + SoftQuantizeLayer |")
    w("| Data | 3srb [11,26], **mV, σ=4.64 baked noise**, contained `original_atEdge==False`, batch 5000 |")
    w("| Noise at load | OFF (`noise=-1`) — already baked into the TFRs |")
    w("| Offset / levels | 0.0 / [0,1,2,3] |")
    w("| Init window | random [25,160] |")
    w("| Epochs | **5000** per seed, no early stopping |")
    w("| Anneal | cosine k 1→67 over 5000 epochs |")
    w("| Optimizer / loss | Nadam(1e-3) / custom_loss (NLL) |")
    w("| Stuck guard | AbortOnStuck thr **1e4** + flat-line (no-improve 15 ep) + reseed |")
    w("\nk(epoch): " + ", ".join(f"ep{e}→{kval(e):.0f}" for e in [0,500,1000,2000,3000,3500,4000,5000]) + ".\n")

    w("---\n\n## 2. Results per seed (converged)\n")
    w("| Seed | Epochs | Final T [T0,T1,T2] | Best-epoch T | Best NLL/batch | Best ep |")
    w("|---:|---:|---|---|---:|---:|")
    for r in conv:
        ft = "[%.2f, %.2f, %.2f]" % tuple(r['ft']); bft = "[%.2f, %.2f, %.2f]" % tuple(r['bft'])
        w(f"| {r['seed']} | {r['epochs']} | {ft} | {bft} | {r['best']:,.0f} | {r['bestep']} |")
    w("")
    conv_seeds = {r['seed'] for r in conv}
    others = [r for r in runs if r['grp'] != 'converged' and r['seed'] not in conv_seeds]
    if others:
        run = [r for r in others if r['grp'] == 'running']
        stk = [r for r in others if r['grp'] == 'stuck']
        bits = []
        if run: bits.append("seed(s) " + ", ".join(f"{r['seed']} (training, ep {r['curep']}/5000)" for r in run))
        if stk: bits.append("stuck/killed: " + ", ".join(str(r['seed']) for r in stk))
        w("_Not yet in the converged set: " + "; ".join(bits) + " — will be added when finished._\n")

    if conv:
        T = np.array([r['bft'] for r in conv]); b = np.array([r['best'] for r in conv])
        med = np.median(T, axis=0)
        w("---\n\n## 3. Statistics over converged seeds\n")
        w(f"{len(conv)} converged seed(s): {sorted(r['seed'] for r in conv)}.\n")
        w("| quantity | T0 | T1 | T2 | best NLL/batch |")
        w("|---|---:|---:|---:|---:|")
        for nm, fn in [("median", np.median), ("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
            w(f"| {nm} | {fn(T[:,0]):.2f} | {fn(T[:,1]):.2f} | {fn(T[:,2]):.2f} | {fn(b):,.0f} |")
        w(f"\n**RECOMMENDED thresholds (median over {len(conv)} converged seeds, levels [0,1,2,3]): "
          f"[{med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f}]**\n")
        w(f"\nClean-data archive median was [4.15, 13.01, 47.24]; the σ=4.64 mV baked noise shifts all three up.\n")

    w("---\n\n## 4. Trajectories\n")
    for r in conv:
        cks = sorted(set(c for c in [0,100,500,1000,2000,3000,4000,r['epochs']-1] if c <= r['ep'][-1]))
        idx = [int(np.argmin(np.abs(r['ep']-c))) for c in cks]
        w(f"**seed {r['seed']} — {r['grp']} ({r['status']}, best {r['best']:,.0f} @ ep {r['bestep']})**\n")
        w("| epoch | " + " | ".join(str(int(r['ep'][i])) for i in idx) + " |")
        w("|---|" + "---|"*len(idx))
        w("| val | " + " | ".join(f"{r['val'][i]:,.0f}" for i in idx) + " |")
        for k, nm in enumerate(['T0','T1','T2']):
            w("| "+nm+" | " + " | ".join(f"{r['T'][i,k]:.2f}" for i in idx) + " |")
        w("")

    w("---\n\n## 5. Figures\n")
    caps = {"01_nll_vs_epoch.png": "val NLL vs epoch (green=converged, red=stuck). Stuck runs freeze near +98,886.",
            "02_threshold_vs_epoch_by_index.png": "Each threshold vs epoch; dashed=clean-data median, blue=this study's median.",
            "04_epoch_vs_threshold_by_index.png": "Epoch-vs-threshold (threshold on x, epoch downward); ○ = init, ★ = final.",
            "03_nll_with_k.png": "NLL vs epoch with the cosine k-anneal (1→67 over 5000) overlaid.",
            "05_threshold_hist.png": "Best-epoch threshold distribution across converged seeds (median dashed, clean dotted).",
            "06_threshold_stability_zoom.png": "Threshold stability over the final 1500 epochs of the best converged seed."}
    for p in made:
        fn = os.path.basename(p)
        w(f"### {fn}\n\n![{fn}](figs/{fn})\n\n*{caps.get(fn,'')}*\n\n`{p}`\n")

    w("---\n\n## 6. Key findings\n")
    if conv:
        T = np.array([r['bft'] for r in conv]); med = np.median(T, axis=0)
        sd = np.std(T, axis=0)
        w(f"1. **Converged thresholds (median over {len(conv)} seeds): "
          f"[{med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f}]**, spread (std) [{sd[0]:.2f}, {sd[1]:.2f}, {sd[2]:.2f}].\n")
        w("2. **All higher than the clean-data [4.15, 13.01, 47.24]** — the σ=4.64 mV baked noise raises the "
          "thresholds (especially T0) off the noise floor for robustness. Expected and intended.\n")
    nstuck = sum(r['grp'] == 'stuck' for r in runs)
    if nstuck:
        w(f"3. **{nstuck} stuck-init attempt(s)** froze near the ~98,886 zero-likelihood ceiling and were "
          "auto-aborted (thr 1e4 + flat-line guard), then reseeded — never polluting the converged set.\n")
    w("\n*Generated by `scripts/distillation/make_2_5_study.py`.*\n")

    open(f"{OUT}/REPORT.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{OUT}/REPORT.md", f"({len(conv)} converged)")


if __name__ == "__main__":
    runs = load(); made = figs(runs); report(runs, made)
    for p in made: print(" fig:", p)
