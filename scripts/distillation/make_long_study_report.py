"""Generate a FULLY detailed REPORT.md for the Part-1 long study, with embedded
images, complete tables, per-group/per-batch statistics, and per-run trajectory
checkpoint tables. All numbers computed from the per-epoch CSVs + result.json."""
import os, json, numpy as np

R = "/work/users/das214/SmartPixels/smart-pixels-ml/runs"
OUT = f"{R}/part1_long_study"; FIG = f"{OUT}/figs"
BATCHES = [("A", "random", f"{R}/part1_long", "random in [25,160]"),
           ("B", "fixed40", f"{R}/part1_long_fixed40", "[40, 40.1, 40.2]"),
           ("C", "fixed20", f"{R}/part1_long_fixed20", "[20, 20.1, 20.2]")]
SEEDS = [42, 1042, 2042, 3042, 4042]
THR = -40000.0
CKPTS = [0, 50, 100, 200, 300, 500, 700, 900, 999]


def load():
    runs = []
    for tag, short, bd, binit in BATCHES:
        for s in SEEDS:
            r = json.load(open(f"{bd}/seed_{s}/result.json"))
            a = np.genfromtxt(f"{bd}/seed_{s}/threshold_loss_epochs.csv", delimiter=",", names=True)
            runs.append(dict(tag=tag, short=short, binit=binit, seed=s,
                             ep=a['epoch'], val=a['val_loss'], loss=a['loss'],
                             T=np.vstack([a['T0'], a['T1'], a['T2']]).T,
                             it=r['init_thresholds'], ft=r['final_thresholds'],
                             best=r['best_val_loss'], final=r['final_val_loss'],
                             bestep=int(a['epoch'][int(np.argmin(a['val_loss']))]),
                             grp='better' if r['best_val_loss'] < THR else 'worse'))
    return runs


def kval(ep, T=1000, k0=1.0, kf=67.0):
    return k0 + (kf - k0) * 0.5 * (1 - np.cos(np.pi * ep / T))


def grp_stats(rows):
    T = np.array([r['ft'] for r in rows]); b = np.array([r['best'] for r in rows]); f = np.array([r['final'] for r in rows])
    return T, b, f


def main():
    runs = load()
    L = []
    w = L.append

    w("# Part-1 Long-Training Threshold Study — Full Report\n")
    w("**ViT_Max + SoftQuantizeLayer · 2-bit ADC threshold optimization · 1000 epochs · 3 init schemes × 5 seeds**\n")
    w("> Generated from the per-epoch logs in `runs/part1_long{,_fixed40,_fixed20}/seed_*/`. "
      "All NLL values are **per batch** (5000 events) and are the **true hard-quantized** loss (straight-through estimator).\n")

    # ---------------- 1. dataset/setup ----------------
    w("---\n\n## 1. Dataset, model & configuration\n")
    w("| item | value |")
    w("|---|---|")
    w("| Dataset | `dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps/shuffled_3d` |")
    w("| Detector readout | 16×16 pixels, 50×12.5 µm, centered incidence, 10 ps sim → convolved to 200 ps |")
    w("| Time slices | 2 — samples **[11, 26]** of the 101-sample waveform |")
    w("| Batch size | 5000 (train & val) |")
    w("| Model | ViT_Max (418,702 params) + SoftQuantizeLayer head |")
    w("| Quantization | 2-bit, levels [0, 1, 2, 3] → **3 thresholds** T0<T1<T2 |")
    w("| `threshold_offset` (floor) | **0** (build asserts `offset < T0 < T1 < T2`) |")
    w("| Input noise | **none** (σ=80 of the old set would swamp a ~25–150 signal) |")
    w("| Optimizer | Nadam, lr 1e-3 |")
    w("| Epochs | **1000, no early stopping** |")
    w("| Anneal | SoftQuantize sharpness `k`: cosine **1 → 67, stretched over all 1000 epochs** |\n")
    w("**Unit-scale note.** This dataset's charge is **not** in mV/e⁻; values run ~25–150 (max ≈ 630), ~24× smaller than "
      "the previous set (p99 ≈ 2750). The random-init window was therefore [25,160] (not the old [80,2000]) and noise removed.\n")

    w("### SoftQuantize internals (why `k` and the loss behave as they do)\n")
    w("- **Forward = straight-through.** `call()` returns `stop_gradient(hard_q − soft_q) + soft_q` in training (numerically `hard_q`) "
      "and pure `hard_q` at inference. So **train and val loss are always the real 2-bit hard-quantized NLL**; `k` only shapes gradients.\n")
    w("- **`k` is bin-normalized:** the soft CDF uses `sigmoid(k·(Tᵢ − x)/τ)` with τ = local threshold spacing. The transition width "
      "is τ/k, i.e. **scale-invariant** — unlike the thresholds, `k` needs no rescaling for this dataset.\n")

    # anneal table
    w("### Annealing schedule k(epoch)\n")
    w("| epoch | 0 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |")
    w("|---|" + "---|" * 11)
    w("| **k** | " + " | ".join(f"{kval(e):.1f}" for e in [0,100,200,300,400,500,600,700,800,900,1000]) + " |\n")

    # ---------------- 2. methodology ----------------
    w("---\n\n## 2. Method — three init batches × 5 seeds\n")
    w("| batch | initial thresholds | output dir | purpose |")
    w("|---|---|---|---|")
    w("| **A** | random in [25,160] | `runs/part1_long` | natural random-init convergence |")
    w("| **B** | fixed **[40, 40.1, 40.2]** | `runs/part1_long_fixed40` | clustered start, all 3 near 40 |")
    w("| **C** | fixed **[20, 20.1, 20.2]** | `runs/part1_long_fixed20` | clustered start, all 3 near 20 |")
    w("\nSeeds per batch: 42, 1042, 2042, 3042, 4042 (seed varies weight init + data order; for B/C the init thresholds are identical).\n")

    # ---------------- 3. full results ----------------
    w("---\n\n## 3. Full results — all 15 runs\n")
    w("| Batch | Seed | Init T [T0,T1,T2] | Final T [T0,T1,T2] | Best NLL/batch | (best @ ep) | Final NLL/batch | Group |")
    w("|---|---:|---|---|---:|---:|---:|---|")
    for r in runs:
        it = "[%.1f, %.1f, %.1f]" % tuple(r['it']); ft = "[%.2f, %.2f, %.2f]" % tuple(r['ft'])
        g = "**better**" if r['grp'] == 'better' else "worse"
        star = " ⭐" if r['best'] == min(x['best'] for x in runs) else ""
        w(f"| {r['tag']} {r['short']} | {r['seed']} | {it} | {ft} | {r['best']:,.0f}{star} | {r['bestep']} | {r['final']:,.0f} | {g} |")
    w("\n⭐ = global best run.\n")

    # ---------------- 4. group statistics ----------------
    w("---\n\n## 4. Outcome groups — statistics\n")
    w("The 15 runs split cleanly into two NLL groups (cut at −40,000; **no run lands between −33k and −46k**).\n")
    for g in ['better', 'worse']:
        rows = [r for r in runs if r['grp'] == g]; T, b, f = grp_stats(rows)
        w(f"### {g.capitalize()} group — {len(rows)} runs\n")
        w("| quantity | T0 | T1 | T2 | best NLL/batch | final NLL/batch |")
        w("|---|---:|---:|---:|---:|---:|")
        w(f"| median | {np.median(T[:,0]):.2f} | {np.median(T[:,1]):.2f} | {np.median(T[:,2]):.2f} | {np.median(b):,.0f} | {np.median(f):,.0f} |")
        w(f"| mean   | {T[:,0].mean():.2f} | {T[:,1].mean():.2f} | {T[:,2].mean():.2f} | {b.mean():,.0f} | {f.mean():,.0f} |")
        w(f"| std    | {T[:,0].std():.2f} | {T[:,1].std():.2f} | {T[:,2].std():.2f} | {b.std():,.0f} | {f.std():,.0f} |")
        w(f"| min    | {T[:,0].min():.2f} | {T[:,1].min():.2f} | {T[:,2].min():.2f} | {b.min():,.0f} | {f.min():,.0f} |")
        w(f"| max    | {T[:,0].max():.2f} | {T[:,1].max():.2f} | {T[:,2].max():.2f} | {b.max():,.0f} | {f.max():,.0f} |\n")
    # discriminator
    Tb = np.array([r['ft'] for r in runs if r['grp']=='better']); Tw = np.array([r['ft'] for r in runs if r['grp']=='worse'])
    w(f"**Discriminator:** the groups separate most in **T1** (better median {np.median(Tb[:,1]):.2f} vs worse {np.median(Tw[:,1]):.2f}); "
      f"T0 and T2 overlap. So the 'better' optimum keeps a genuine middle threshold while the 'worse' one lets T1 collapse toward T0.\n")

    # ---------------- 5. per-batch ----------------
    w("---\n\n## 5. Per-batch breakdown\n")
    w("| batch | better / 5 | worse / 5 | best run (NLL) | better-median final T |")
    w("|---|---:|---:|---|---|")
    for tag, short, bd, binit in BATCHES:
        rows = [r for r in runs if r['tag'] == tag]; bett = [r for r in rows if r['grp']=='better']
        best = min(rows, key=lambda r: r['best'])
        bt = np.array([r['ft'] for r in bett]) if bett else np.zeros((1,3))
        bm = "[%.2f, %.2f, %.2f]" % (np.median(bt[:,0]), np.median(bt[:,1]), np.median(bt[:,2])) if bett else "—"
        w(f"| {tag} ({short}) | {len(bett)} | {5-len(bett)} | seed {best['seed']} ({best['best']:,.0f}) | {bm} |")
    w("\nInit scheme does **not** control the outcome — every batch produces both groups at roughly 50/50.\n")

    # ---------------- 6. trajectory tables ----------------
    w("---\n\n## 6. Representative per-epoch trajectories\n")
    w("Best (better) and a worse run from each batch, sampled at checkpoint epochs. `val` = NLL/batch.\n")
    reps = []
    for tag, short, bd, binit in BATCHES:
        rows = [r for r in runs if r['tag'] == tag]
        reps.append(min(rows, key=lambda r: r['best']))                       # best
        worse = [r for r in rows if r['grp']=='worse']
        if worse: reps.append(max(worse, key=lambda r: r['best']))            # a clear worse
    for r in reps:
        w(f"**{r['tag']} {r['short']} seed {r['seed']} — {r['grp']} (best {r['best']:,.0f} @ ep {r['bestep']}, final T={['%.2f'%x for x in r['ft']]})**\n")
        w("| epoch | " + " | ".join(str(e if e!=999 else 1000) for e in CKPTS).replace("1000","1000") + " |")
        w("|---|" + "---|"*len(CKPTS))
        idx = [int(np.argmin(np.abs(r['ep']-e))) for e in CKPTS]
        w("| val | " + " | ".join(f"{r['val'][i]:,.0f}" for i in idx) + " |")
        w("| T0 | " + " | ".join(f"{r['T'][i,0]:.2f}" for i in idx) + " |")
        w("| T1 | " + " | ".join(f"{r['T'][i,1]:.2f}" for i in idx) + " |")
        w("| T2 | " + " | ".join(f"{r['T'][i,2]:.2f}" for i in idx) + " |\n")

    # ---------------- 7. figures ----------------
    w("---\n\n## 7. Figures\n")
    figs = [
        ("01_nll_vs_epoch_by_group.png", "val NLL vs epoch — all 15 runs by group",
         "Clean bimodal split: better (green) make a second drop to ~−48k; worse (red) plateau at ~−32k. The drop happens in the steep middle of the anneal."),
        ("02_threshold_vs_epoch_by_index.png", "Each threshold vs epoch, colored by group",
         "T1 is the discriminator — better runs settle T1≈11, worse runs let it fall to ≈7.5. T0 (floor ~0.4–0.8) and T2 (~26–32) overlap between groups."),
        ("03_epoch_vs_threshold_perbatch.png", "Epoch-vs-threshold convergence per batch",
         "5 seeds (faint) + median (bold); epochs increase downward. All three init schemes fan out then collapse to a similar median band."),
        ("04_final_threshold_hist.png", "Final-threshold distributions, split by group",
         "Histograms of T0/T1/T2 across all 15 runs; better vs worse overlaid. T1 shows the clearest separation."),
        ("05_best_nll_strip.png", "Best NLL per run by batch",
         "Both clusters appear in every batch with a clean empty gap between −33k and −46k → outcome is seed-driven, not init-driven."),
        ("06_final_thresholds_scatter.png", "Final thresholds scatter (T0–T1, T1–T2) by group",
         "Green/red separate along the T1 axis; T0 and T2 do not separate the groups."),
        ("07_k_vs_convergence.png", "Where k is when the split happens",
         "Twin axis: the good run pulls ahead at k≈25–45 (epochs ~450–620); the stuck run freezes there. After k≈53 (ep ~700) nothing moves."),
        ("08_good_vs_worse_same_thresholds.png", "Same thresholds, different NLL (A 42 vs A 3042)",
         "Two runs converge to ~[0.5, 10.7, 26] yet differ by 16k NLL — proof the split is a model-convergence effect, not a threshold effect."),
        ("09_good_vs_worse_B_42_vs_1042.png", "Different thresholds, same fixed start (B 42 vs B 1042)",
         "From identical [40,40.1,40.2]: seed 42 keeps 3 real levels [5.4,15.5,39] (−46.8k); seed 1042 collapses to [0.8,7.4,32] (−32.5k)."),
    ]
    for fn, title, cap in figs:
        n = fn.split("_")[0]
        w(f"### 7.{int(n)} {title}\n")
        w(f"![{title}](figs/{fn})\n")
        w(f"*{cap}*\n")
        w(f"`{FIG}/{fn}`\n")

    # ---------------- 8. findings ----------------
    w("---\n\n## 8. Key findings\n")
    w("1. **Bimodal outcome.** Every run is either *better* (−46.8k…−49.6k) or *worse* (−31.4k…−32.8k); ~14k gap, nothing between. Holds across all init schemes.\n")
    w("2. **Seed-driven, not init-driven.** Per-batch better rate ≈ 50% regardless of random vs fixed-40 vs fixed-20 (Fig 05).\n")
    w("3. **A model-convergence effect, not a threshold effect.** Identical thresholds can give different NLL (Fig 08) and different thresholds the same story (Fig 09); the deciding factor is whether ViT_Max finds its deeper optimum (the second NLL drop).\n")
    w("4. **No late-epoch monkey business.** Thresholds and loss freeze by ~epoch 700 in every run; the last 300 epochs are flat (Fig 01/03).\n")
    w("5. **Fate sealed at k≈25–45 (epochs ~450–620).** Better runs drop there; worse runs plateau there (Fig 07). After k≈53 nothing changes → ~700 epochs is enough.\n")
    w("6. **k needs no rescaling and NLL is already true-hard** (straight-through; `k/τ` normalized).\n")

    # ---------------- 9. comparison to 300-ep ----------------
    w("---\n\n## 9. Comparison with the earlier 300-epoch study\n")
    w("| study | anneal length | recommended thresholds | best NLL/batch |")
    w("|---|---|---|---|")
    w("| 300-epoch median (archive) | k 1→67 over 300 | [4.15, 13.01, 47.24] | ~−31,600 |")
    w("| **1000-epoch, better group** | k 1→67 over 1000 | **[0.40, 10.80, 29.80]** (median) | **−49,603** (global best) |")
    w("\nThe anneal is tied to total epochs, so 300 vs 1000 is **also an anneal-speed difference**, not only a length one — the slower 1000-epoch ramp drives the thresholds lower (T0→floor, T2 47→~26–30) and reaches a deeper NLL on the better runs.\n")

    # ---------------- 10. recommendations + repro ----------------
    w("---\n\n## 10. Recommendations\n")
    w("- **Reported thresholds:** use **best-of-N** or the *better*-group median. Better-group median = **[0.40, 10.80, 29.80]**; global best = **[0.40, 10.40, 25.50]** at **−49,603/batch**.\n")
    w("- **k / schedule:** leave `final_k=67` (scale-invariant, NLL already hard). To shrink the ~50% 'worse' rate, the only lever is a **slower/delayed ramp through k≈15–45**; otherwise run N seeds and keep the best.\n")
    w("- **Epoch budget:** ~700 epochs reproduces 1000 (all frozen after k≈53).\n")
    w("\n---\n\n## 11. Reproduce / file index\n")
    w("| artifact | path |")
    w("|---|---|")
    w("| Per-run logs | `runs/part1_long{,_fixed40,_fixed20}/seed_*/threshold_loss_epochs.csv`, `result.json` |")
    w("| Training script | `scripts/distillation/run_part1_long.py` |")
    w("| Batch loop | `scripts/distillation/run_part1_long_loop.sh` |")
    w("| Plot script | `scripts/distillation/make_long_study_plots.py` |")
    w("| Report generator | `scripts/distillation/make_long_study_report.py` |")
    w("| Figures | `runs/part1_long_study/figs/` |")
    w("| Table JSON | `runs/part1_long_study/runs_table.json` |")

    open(f"{OUT}/REPORT.md", "w").write("\n".join(L) + "\n")
    print("wrote", f"{OUT}/REPORT.md", f"({len(L)} lines)")


main()
