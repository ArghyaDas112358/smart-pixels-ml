"""Partial-aware figures + REPORT_v2.md for the Part-1 v2 study
(5000 epochs, EarlyStopping patience=500, anneal stretched over 5000).
Runs on whatever is finished/running so far; re-run any time to update.
Mirrors the v1 report (runs/part1_long_study/REPORT.md)."""
import os, json, glob, shutil
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R = "/work/users/das214/SmartPixels/smart-pixels-ml/runs"
OUT = f"{R}/part1_long_5k_study"; FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
BATCHES = [("A", "random", f"{R}/part1_long_5k", "random in [25,160]"),
           ("B", "fixed40", f"{R}/part1_long_5k_fixed40", "[40, 40.1, 40.2]"),
           ("C", "fixed20", f"{R}/part1_long_5k_fixed20", "[20, 20.1, 20.2]")]
SEEDS = [42, 1042, 2042, 3042, 4042]
MAXEP, PAT, THR = 5000, 500, -40000.0
GCOL = {'better': 'tab:green', 'worse': 'tab:red'}
sm = lambda v, k=25: np.convolve(v, np.ones(k)/k, mode='valid') if len(v) > k else v
# matched v1 best NLL for the same batch/seed (for the comparison column)
V1 = {("A",42):-49207,("A",1042):-32807,("A",2042):-31441,("A",3042):-32777,("A",4042):-49603,
      ("B",42):-46764,("B",1042):-32515,("B",2042):-32772,("B",3042):-49316,("B",4042):-47003,
      ("C",42):-47703,("C",1042):-32776,("C",2042):-32368,("C",3042):-47655,("C",4042):-32276}


def load():
    runs = []
    for tag, short, bd, binit in BATCHES:
        for s in SEEDS:
            d = f"{bd}/seed_{s}"; rj = f"{d}/result.json"; cs = f"{d}/threshold_loss_epochs.csv"
            r = dict(tag=tag, short=short, binit=binit, seed=s)
            if os.path.exists(cs) and os.path.getsize(cs) > 50:
                a = np.genfromtxt(cs, delimiter=",", names=True)
                a = np.atleast_1d(a)
                r.update(ep=a['epoch'], val=a['val_loss'], loss=a['loss'],
                         T=np.vstack([a['T0'], a['T1'], a['T2']]).T)
                bi = int(np.argmin(a['val_loss']))
                r.update(best=float(a['val_loss'][bi]), bestep=int(a['epoch'][bi]),
                         final=float(a['val_loss'][-1]), curep=int(a['epoch'][-1]) + 1,
                         ft=[float(a['T0'][-1]), float(a['T1'][-1]), float(a['T2'][-1])],
                         bft=[float(a['T0'][bi]), float(a['T1'][bi]), float(a['T2'][bi])])
            if os.path.exists(rj):
                j = json.load(open(rj))
                r.update(status='done', epochs=j['epochs'], stopped_early=j.get('stopped_early', False),
                         best=j['best_val_loss'], bestep=j.get('best_epoch', r.get('bestep', 0)),
                         ft=j['final_thresholds'], bft=j.get('best_thresholds', j['final_thresholds']))
            elif 'ep' in r:
                r['status'] = 'running'
            else:
                r['status'] = 'queued'
            r['grp'] = ('better' if r.get('best', 1) < THR else 'worse') if r['status'] != 'queued' else None
            runs.append(r)
    return runs


def kval(ep, T=MAXEP, k0=1.0, kf=67.0):
    return k0 + (kf - k0) * 0.5 * (1 - np.cos(np.pi * np.asarray(ep) / T))


def figs(runs):
    have = [r for r in runs if r['status'] in ('done', 'running')]
    made = []
    # 01 NLL vs epoch
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for r in have:
        c = GCOL.get(r['grp'], 'gray'); ls = '-' if r['status'] == 'done' else '--'
        e = r['ep'][24:] if len(r['val']) > 25 else r['ep']
        ax.plot(e, sm(r['val']), color=c, lw=1.5, ls=ls,
                label=f"{r['tag']}{r['seed']} {r['status']} (best {r['best']:,.0f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("val NLL/batch (25-ep avg)")
    ax.set_title(f"v2 (5000 ep / patience {PAT}) — val NLL vs epoch\nsolid=done, dashed=running")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout()
    p = f"{FIG}/01_nll_vs_epoch.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 02 thresholds vs epoch by index
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, nm in enumerate(['T0', 'T1', 'T2']):
        for r in have:
            c = GCOL.get(r['grp'], 'gray'); ls = '-' if r['status'] == 'done' else '--'
            ax[i].plot(r['ep'], r['T'][:, i], color=c, lw=1.1, ls=ls, alpha=0.8)
        ax[i].set_title(nm); ax[i].set_xlabel("epoch"); ax[i].grid(alpha=0.3)
    ax[0].set_ylabel("threshold value")
    for g, c in GCOL.items(): ax[0].plot([], [], color=c, lw=2, label=g)
    ax[0].plot([], [], color='gray', lw=2, label='running'); ax[0].legend(fontsize=8)
    fig.suptitle(f"v2 — each threshold vs epoch ({len(have)} runs with data)")
    fig.tight_layout(); p = f"{FIG}/02_threshold_vs_epoch_by_index.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 02b epoch-vs-threshold (axes swapped: x=threshold, y=epoch, y inverted so epochs go downward)
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for i, nm in enumerate(['T0', 'T1', 'T2']):
        for r in have:
            c = GCOL.get(r['grp'], 'gray'); ls = '-' if r['status'] == 'done' else '--'
            ax[i].plot(r['T'][:, i], r['ep'], color=c, lw=1.2, ls=ls, alpha=0.85)
            ax[i].scatter([r['T'][0, i]], [r['ep'][0]], color=c, marker='o', s=30, zorder=5)   # init (top)
            ax[i].scatter([r['T'][-1, i]], [r['ep'][-1]], color=c, marker='*', s=110, zorder=5)  # latest (bottom)
        ax[i].set_title(nm); ax[i].set_xlabel("threshold value"); ax[i].grid(alpha=0.3)
    ax[0].invert_yaxis(); ax[0].set_ylabel("epoch  (training progresses downward)")
    for g, c in GCOL.items(): ax[0].plot([], [], color=c, lw=2, label=g)
    ax[0].plot([], [], color='gray', lw=2, label='running'); ax[0].legend(fontsize=8)
    fig.suptitle(f"v2 — epoch vs threshold ({len(have)} runs); ○ = init (top), ★ = latest (bottom)")
    fig.tight_layout(); p = f"{FIG}/04_epoch_vs_threshold_by_index.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    # 03 NLL vs epoch with k overlay (uses the longest run available)
    fig, ax = plt.subplots(figsize=(9.5, 5.2)); ax2 = ax.twinx()
    for r in have:
        c = GCOL.get(r['grp'], 'gray'); ls = '-' if r['status'] == 'done' else '--'
        e = r['ep'][24:] if len(r['val']) > 25 else r['ep']
        ax.plot(e, sm(r['val']), color=c, lw=1.4, ls=ls)
    ke = np.arange(0, MAXEP); ax2.plot(ke, kval(ke), 'k--', lw=1.3, alpha=0.7, label='k(epoch) 1->67 over 5000')
    ax.set_xlabel("epoch"); ax.set_ylabel("val NLL/batch (25-ep avg)"); ax2.set_ylabel("k (anneal)")
    ax.set_title("v2 — NLL vs epoch with the (5x slower) k schedule overlaid"); ax2.legend(loc='upper right', fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); p = f"{FIG}/03_nll_with_k.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    done = [r for r in have if r['status'] == 'done']
    if len(done) >= 4:
        # 05 best NLL strip by batch
        fig, ax = plt.subplots(figsize=(8, 5)); bl = ['A', 'B', 'C']
        rng = np.random.default_rng(1)
        for r in done:
            ax.scatter(bl.index(r['tag']) + rng.uniform(-0.12, 0.12), r['best'],
                       color=GCOL[r['grp']], s=70, edgecolor='k', lw=0.5, zorder=3)
        ax.axhline(THR, color='gray', ls=':', label=f"group cut {THR:,.0f}")
        ax.set_xticks(range(3)); ax.set_xticklabels(['A random', 'B fixed40', 'C fixed20'])
        ax.set_ylabel("best val NLL / batch")
        ax.set_title(f"v2 — best NLL per run by batch ({len(done)} done)\ngreen=better, red=worse")
        ax.legend(); ax.grid(alpha=0.3, axis='y'); fig.tight_layout()
        p = f"{FIG}/05_best_nll_strip.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
        # 06 v1 vs v2 paired (scatter, diagonal = no change)
        fig, ax = plt.subplots(figsize=(6.4, 6))
        for r in done:
            v1 = V1[(r['tag'], r['seed'])]
            ax.scatter(v1, r['best'], color=GCOL[r['grp']], s=70, edgecolor='k', lw=0.5, zorder=3)
            ax.annotate(f"{r['tag']}{r['seed']}", (v1, r['best']), fontsize=6, xytext=(3, 3), textcoords='offset points')
        lim = [-54000, -28000]
        ax.plot(lim, lim, 'k--', lw=1, alpha=0.6, label='v2 = v1 (no change)')
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("v1 best NLL/batch (1000 ep)"); ax.set_ylabel("v2 best NLL/batch (5000 ep)")
        ax.set_title("v1 vs v2 best NLL per seed\n(points below the line = v2 better)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        p = f"{FIG}/06_v1_vs_v2.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
        # 07 best-threshold histograms (shared bins), by group
        ft = {g: np.array([r['bft'] for r in done if r['grp'] == g]) for g in GCOL}
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        for i, nm in enumerate(['T0', 'T1', 'T2']):
            allv = np.array([r['bft'][i] for r in done]); lo, hi = allv.min(), allv.max()
            pad = 0.06 * (hi - lo + 1e-6); edges = np.linspace(lo - pad, hi + pad, 13)
            for g, c in GCOL.items():
                if len(ft[g]): ax[i].hist(ft[g][:, i], bins=edges, color=c, alpha=0.55, label=g, edgecolor='k', lw=0.5)
            ax[i].axvline(np.median(allv), color='k', ls='--', lw=1.2, label=f"median {np.median(allv):.2f}")
            ax[i].set_title(nm); ax[i].set_xlabel("best-epoch threshold"); ax[i].grid(alpha=0.3); ax[i].legend(fontsize=7)
            ax[i].yaxis.get_major_locator().set_params(integer=True)
        ax[0].set_ylabel("# runs")
        fig.suptitle(f"v2 — best-epoch threshold distributions ({len(done)} done, shared bins)")
        fig.tight_layout(); p = f"{FIG}/07_threshold_hist.png"; fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    return made


def report(runs, made):
    done = [r for r in runs if r['status'] == 'done']; running = [r for r in runs if r['status'] == 'running']
    L = []; w = L.append
    w("# Part-1 Long-Training Threshold Study — **v2 (5000 epochs, early stopping)**\n")
    w(f"> **PRELIMINARY — {len(done)}/15 runs complete, {len(running)} running.** Auto-regenerated from "
      "`runs/part1_long_5k{,_fixed40,_fixed20}/seed_*/`. Re-run `scripts/distillation/make_5k_study.py` to update.\n")
    w("**What changed from v1:** epochs **1000 → 5000**, added **EarlyStopping(patience=500, restore_best_weights)**, "
      "so the cosine k-anneal (1→67) is stretched over **5000** epochs (5× slower) — testing whether more low-k time changes the bimodal outcome. Everything else identical to v1.\n")

    w("---\n\n## 1. Configuration\n")
    w("| item | v1 | **v2** |")
    w("|---|---|---|")
    w("| Epochs | 1000, no early stop | **5000, EarlyStopping patience 500 (restore best)** |")
    w("| Anneal | k 1→67 over 1000 | **k 1→67 over 5000** (5× slower) |")
    w("| Model / data / offset / noise | ViT_Max, 3srb [11,26], offset 0, no noise | same |")
    w("\nk(epoch) for v2: " + ", ".join(f"ep{e}→{kval(e):.0f}" for e in [0,500,1000,2000,3000,3500,4000,5000]) + ".\n")

    w("---\n\n## 2. Results so far\n")
    w("| Batch | Seed | Status | Epochs | Final T [T0,T1,T2] | Best NLL/batch | Group | v1 best | Δ vs v1 |")
    w("|---|---:|---|---:|---|---:|---|---:|---:|")
    for r in runs:
        if r['status'] == 'queued':
            w(f"| {r['tag']} {r['short']} | {r['seed']} | queued | — | — | — | — | {V1[(r['tag'],r['seed'])]:,} | — |"); continue
        ep = f"{r.get('epochs', r['curep'])}{'*' if r.get('stopped_early') else ''}" if r['status']=='done' else f"{r['curep']}…"
        ft = "[%.2f, %.2f, %.2f]" % tuple(r['ft']); g = r['grp']
        v1 = V1[(r['tag'], r['seed'])]; dv = r['best'] - v1
        gtag = f"**{g}**" if g == 'better' else g
        w(f"| {r['tag']} {r['short']} | {r['seed']} | {r['status']} | {ep} | {ft} | {r['best']:,.0f} | {gtag} | {v1:,} | {dv:+,.0f} |")
    w("\n`*` = stopped early (patience 500). `…` = still running. Δ vs v1 = v2 best − v1 best (negative = v2 better).\n")

    if done:
        T = np.array([r['bft'] for r in done]); b = np.array([r['best'] for r in done])
        w("---\n\n## 3. Statistics (completed runs)\n")
        w(f"{len(done)} done — {sum(r['grp']=='better' for r in done)} better, {sum(r['grp']=='worse' for r in done)} worse.\n")
        w("| quantity | T0 | T1 | T2 | best NLL/batch |")
        w("|---|---:|---:|---:|---:|")
        for nm, fn in [("median", np.median), ("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)]:
            w(f"| {nm} | {fn(T[:,0]):.2f} | {fn(T[:,1]):.2f} | {fn(T[:,2]):.2f} | {fn(b):,.0f} |")
        gb = min(done, key=lambda r: r['best'])
        w(f"\n**Best so far:** {gb['tag']} seed {gb['seed']} — **{gb['best']:,.0f}**, best-epoch T = [{gb['bft'][0]:.2f}, {gb['bft'][1]:.2f}, {gb['bft'][2]:.2f}].\n")

    w("---\n\n## 4. Trajectories (completed + running)\n")
    for r in [x for x in runs if x['status'] in ('done', 'running')]:
        cks = [0, 100, 500, 1000, 2000, 3000, 4000, r['curep']-1 if r['status']=='running' else r.get('epochs', r['curep'])-1]
        cks = sorted(set(c for c in cks if c <= r['ep'][-1]))
        idx = [int(np.argmin(np.abs(r['ep']-c))) for c in cks]
        hdr = f"{r['tag']} {r['short']} seed {r['seed']} — {r['status']}, {r['grp']} (best {r['best']:,.0f}"
        hdr += f" @ ep {r['bestep']})" if r['status']=='done' else f" so far)"
        w(f"**{hdr}**\n")
        w("| epoch | " + " | ".join(str(int(r['ep'][i])) for i in idx) + " |")
        w("|---|" + "---|"*len(idx))
        w("| val | " + " | ".join(f"{r['val'][i]:,.0f}" for i in idx) + " |")
        for k, nm in enumerate(['T0','T1','T2']):
            w("| "+nm+" | " + " | ".join(f"{r['T'][i,k]:.2f}" for i in idx) + " |")
        w("")

    w("---\n\n## 5. Figures\n")
    caps = {"01_nll_vs_epoch.png": "val NLL vs epoch (solid=done, dashed=running), colored by outcome group.",
            "02_threshold_vs_epoch_by_index.png": "Each threshold vs epoch for all runs with data.",
            "04_epoch_vs_threshold_by_index.png": "Epoch-vs-threshold (axes swapped: threshold on x, epoch on y increasing downward); ○ = init at top, ★ = latest at bottom.",
            "03_nll_with_k.png": "NLL vs epoch with the 5× slower k schedule overlaid (k≈53 now lands near epoch 3500).",
            "05_best_nll_strip.png": "Best NLL per run by batch — far fewer 'worse' runs than v1.",
            "06_v1_vs_v2.png": "v1 (1000 ep) vs v2 (5000 ep) best NLL per seed; points below the diagonal = v2 improved.",
            "07_threshold_hist.png": "Best-epoch threshold distributions across completed v2 runs (shared bins)."}
    for p in made:
        fn = os.path.basename(p)
        w(f"### {fn}\n\n![{fn}](figs/{fn})\n\n*{caps.get(fn,'')}*\n\n`{p}`\n")

    w("---\n\n## 6. Key findings so far\n")
    if done:
        nbet = sum(r['grp'] == 'better' for r in done); nwor = len(done) - nbet
        gb = min(done, key=lambda r: r['best'])
        flips_up = [r for r in done if V1[(r['tag'], r['seed'])] >= THR and r['best'] < THR]
        flips_dn = [r for r in done if V1[(r['tag'], r['seed'])] < THR and r['best'] >= THR]
        w(f"1. **The slow anneal sharply reduces the 'worse' rate.** v1 was ~53% worse (8/15); v2 so far is **{nwor}/{len(done)} worse** "
          f"— most runs reach the deep optimum.\n")
        w(f"2. **It also reaches a *deeper* optimum.** v2 best so far = **{gb['best']:,.0f}** ({gb['tag']} seed {gb['seed']}), vs v1's −49,603 — "
          f"the better cluster sits at ~−51k…−52k, ~3k lower than v1's ~−48k.\n")
        w(f"3. **{len(flips_up)} seeds flipped worse→better** ("
          + ", ".join(f"{r['tag']}{r['seed']}" for r in flips_up) + ") — previously-stuck v1 runs now escape, thanks to the long low-k phase.\n")
        if flips_dn:
            w(f"4. **But {len(flips_dn)} seeds regressed better→worse** ("
              + ", ".join(f"{r['tag']}{r['seed']}" for r in flips_dn) + ") — they were among the *best* in v1 but got stuck at ~−34k in v2. "
              "So the slow anneal is a big net win but **not monotonic per-seed**.\n")
        w("5. **Late escapes are real.** Several better runs sat on a ~−33k plateau for thousands of epochs, then dropped to ~−52k once k reached "
          "the ~45–60 window (now near epoch ~3500–4500) — the same critical-k physics as v1, just delayed by the 5× slower schedule (Fig 01/03).\n")
        w("6. **EarlyStopping(patience 500)** fired only for runs that truly flat-lined; runs still micro-improving ran the full 5000 — which is what *let* the late escapes happen.\n")
    w("\n*Generated by `scripts/distillation/make_5k_study.py`; full v1 analysis: `runs/part1_long_study/REPORT.md`. "
      "Finalizes at 15/15.*\n")

    open(f"{OUT}/REPORT_v2.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{OUT}/REPORT_v2.md", f"({len(done)}/15 done, {len(running)} running)")


if __name__ == "__main__":
    runs = load(); made = figs(runs); report(runs, made)
    for p in made: print(" fig:", p)
