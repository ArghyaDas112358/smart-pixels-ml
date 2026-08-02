"""Figures + REPORT.md for the SimpleRouter (Option D) mu-marginal slice-discovery run.
Auto-discovers every runs/simplerouter_discovery/seed_*/ dir (converged, stuck,
running) and grows as seeds land. Mirrors make_softrouter_study.py; dataviz-method
styling (validated palette, single-hue sequential ramp, no dual axes).
Re-run any time:  python scripts/distillation/make_simplerouter_study.py
"""
import os, re, json, glob, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/simplerouter_discovery"
FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
OUT_A = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/softrouter_discovery"   # Option A, read live
ESCAPE = 5e4                                 # must match driver's ESCAPE_BELOW
BASELINE = [11, 26]                          # the hand-picked pair being challenged

# dataviz reference palette
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seqblue", [
    "#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#c9c8c2", "axes.grid": True, "grid.color": "#dddcd6",
    "grid.linewidth": 0.6, "font.size": 10.5, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
})


def _semilist(v):
    """Parse a ';'-joined list-of-floats cell; '' -> []."""
    return [float(x) for x in v.split(";")] if v else []


def load():
    runs = []
    for d in sorted(glob.glob(f"{OUT}/seed_*")):
        base = os.path.basename(d)
        if base.startswith("sanity"):
            continue
        cs = f"{d}/router_epochs.csv"
        if not (os.path.exists(cs) and os.path.getsize(cs) > 60):
            continue
        rows = list(csv.DictReader(open(cs)))
        if len(rows) < 2:
            continue
        fnum = lambda key: np.array([float(r[key]) for r in rows if r.get(key, "") != ""])
        flist = lambda key: [_semilist(r.get(key, "")) for r in rows]
        m = re.match(r"seed_(\d+)", base)
        ep = fnum("epoch")
        i1 = fnum("i1").astype(int); i2 = fnum("i2").astype(int)
        theta_top5_val = flist("theta_top5_val"); theta_top5_idx = flist("theta_top5_idx")
        mu_top5_val = flist("mu_top5_val"); mu_top5_idx = flist("mu_top5_idx")
        mu_entropy = fnum("mu_entropy") if rows and "mu_entropy" in rows[0] else np.array([])
        # theta_(2) - theta_(3): rank-2 minus rank-3 of the top-5 (descending) theta values
        gap = np.array([v[1] - v[2] if len(v) >= 3 else np.nan for v in theta_top5_val])
        r = dict(seed=int(m.group(1)), dir=base, ep=ep, i1=i1, i2=i2,
                 theta_top5_val=theta_top5_val, theta_top5_idx=theta_top5_idx,
                 mu_top5_val=mu_top5_val, mu_top5_idx=mu_top5_idx,
                 mu_entropy=mu_entropy, gap=gap)
        hc = f"{d}/history.csv"
        if os.path.exists(hc) and os.path.getsize(hc) > 0:
            hrows = list(csv.DictReader(open(hc)))
            r["hep"] = np.array([float(x["epoch"]) for x in hrows if x.get("epoch", "") != ""])
            r["loss"] = np.array([float(x["loss"]) for x in hrows if x.get("loss", "") != ""])
            r["val"] = np.array([float(x["val_loss"]) for x in hrows if x.get("val_loss", "") != ""])
        else:
            r["hep"] = ep; r["loss"] = np.array([]); r["val"] = np.array([])
        if len(r["val"]):
            bi = int(np.argmin(r["val"]))
            r.update(best=float(r["val"][bi]), bestep=int(r["hep"][bi]))
        else:
            r.update(best=float("inf"), bestep=-1)
        r["curep"] = int(ep[-1]) + 1 if len(ep) else 0
        rj = f"{d}/result.json"
        r["status"] = "killed" if "_STUCK" in base else "running"
        if os.path.exists(rj):
            j = json.load(open(rj)); r["status"] = "done"
            r["indices"] = j.get("final_indices", [int(i1[-1]), int(i2[-1])] if len(i1) else [None, None])
            r["fthr"] = j.get("final_thresholds")
            r["epochs"] = j.get("epochs", r["curep"]); r["escaped"] = j.get("escaped")
            r["final_mu_top5"] = j.get("final_mu_top5")
            r["best_val_loss"] = j.get("best_val_loss", r["best"])
            r["best_epoch"] = j.get("best_epoch", r["bestep"])
        else:
            r["indices"] = [int(i1[-1]), int(i2[-1])] if len(i1) else [None, None]
            r["fthr"] = None; r["epochs"] = r["curep"]; r["escaped"] = None
            r["final_mu_top5"] = None
            r["best_val_loss"] = r["best"]; r["best_epoch"] = r["bestep"]
        npz = f"{d}/theta_mu.npz"
        if os.path.exists(npz):
            try:
                z = np.load(npz)
                r["snap_ep"] = z["epochs"]; r["snap_theta"] = z["theta"]
                r["snap_mu"] = z["mu"]; r["snap_visits"] = z["visits"]
            except Exception:
                pass
        r["grp"] = ("stuck" if ("_STUCK" in base or r["best"] >= ESCAPE)
                    else ("converged" if r["status"] == "done" else "running"))
        runs.append(r)
    runs.sort(key=lambda r: (r["grp"] != "converged", r["seed"]))
    return runs


def load_option_a():
    """Read Option A (SoftRouter) converged results live, for the A-vs-D table."""
    out = []
    for d in sorted(glob.glob(f"{OUT_A}/seed_*")):
        base = os.path.basename(d)
        if base.startswith("sanity"):
            continue
        rj = f"{d}/result.json"
        if not os.path.exists(rj):
            continue
        m = re.match(r"seed_(\d+)", base)
        if not m:
            continue
        try:
            j = json.load(open(rj))
        except Exception:
            continue
        out.append(dict(seed=int(m.group(1)), escaped=j.get("escaped"),
                         indices=j.get("final_indices"), fthr=j.get("final_thresholds"),
                         best=j.get("best_val_loss"), bestep=j.get("best_epoch")))
    out.sort(key=lambda r: (not r["escaped"], r["seed"]))
    return out


def figs(runs):
    made = []
    conv = [r for r in runs if r["grp"] == "converged"] or [r for r in runs if r["grp"] == "running"]
    # 01 -- val NLL vs epoch
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for n, r in enumerate(runs):
        if len(r["val"]) == 0:
            continue
        sm = np.convolve(r["val"], np.ones(15) / 15, mode="valid") if len(r["val"]) > 15 else r["val"]
        e = r["hep"][len(r["val"]) - len(sm):]
        ax.plot(e, sm, color=CAT[n % len(CAT)], lw=1.6,
                label=f"seed {r['seed']} (best {r['best']:,.0f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("val NLL/batch (15-ep avg)")
    ax.set_title("SimpleRouter discovery -- val NLL vs epoch")
    ax.legend(frameon=False, fontsize=8.5)
    p = f"{FIG}/01_nll_vs_epoch.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    # 02 -- top-2 (i1,i2) vs epoch, all seeds
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for n, r in enumerate(runs):
        c = CAT[n % len(CAT)]
        ax.plot(r["ep"], r["i1"], color=c, lw=1.6, label=f"seed {r['seed']}")
        ax.plot(r["ep"], r["i2"], color=c, lw=1.6, ls="--")
    for b in BASELINE:
        ax.axhline(b, color=INK2, lw=1, ls=":")
        ax.annotate(f"[{b}] baseline", (0.99, b), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", color=INK2, fontsize=8.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("selected slice index")
    ax.set_title("Committed top-2 indices vs epoch (solid = i1, dashed = i2)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    p = f"{FIG}/02_top2_vs_epoch.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    # 03 -- final mu profile vs slice, seeds overlaid (THE importance map)
    withmu = [r for r in conv if "snap_mu" in r and len(r["snap_mu"])]
    if withmu:
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        for n, r in enumerate(withmu):
            mu = r["snap_mu"][-1]
            ax.plot(np.arange(len(mu)), mu, color=CAT[n % len(CAT)], lw=1.4,
                    label=f"seed {r['seed']}")
        for b in BASELINE:
            ax.axvline(b, color=INK2, lw=1, ls=":")
        ax.set_xlabel("time-slice index"); ax.set_ylabel(r"final marginal $\mu_t$")
        ax.set_title("Final mu profile -- exact marginal inclusion probability per slice")
        ax.legend(frameon=False, fontsize=8.5)
        p = f"{FIG}/03_mu_profile_final.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        made.append(p)
    # 04 -- visits heatmap vs epoch, best seed
    withv = [r for r in conv if "snap_visits" in r and len(r["snap_visits"])]
    if withv:
        r = min(withv, key=lambda r: r["best"])
        V = r["snap_visits"]                                # (n_snap, T)
        fig, ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
        im = ax.imshow(V, aspect="auto", cmap=SEQ, origin="upper",
                        extent=[0, V.shape[1] - 1, r["snap_ep"][-1], r["snap_ep"][0]], vmin=0)
        ax.grid(False)
        for i in r["indices"]:
            if i is None:
                continue
            ax.annotate(f"slice {i}", (i, r["snap_ep"][-1] * 0.9), ha="center", color=INK,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
        ax.set_xlabel("time-slice index"); ax.set_ylabel("epoch  (training ↓)")
        ax.set_title(f"Pair-sampling visit counts vs epoch (seed {r['seed']})", loc="left")
        cb = fig.colorbar(im, ax=ax, pad=0.012); cb.outline.set_visible(False)
        cb.set_label("visits", color=INK2)
        p = f"{FIG}/04_visits_heatmap.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    # 05 -- theta gap (theta_(2) - theta_(3)) vs epoch
    haveg = [r for r in runs if np.any(np.isfinite(r["gap"]))]
    if haveg:
        fig, ax = plt.subplots(figsize=(8.2, 4.0))
        for n, r in enumerate(haveg):
            ax.plot(r["ep"], r["gap"], color=CAT[n % len(CAT)], lw=1.5, label=f"seed {r['seed']}")
        ax.axhline(0, color=INK2, lw=1, ls=":")
        ax.set_xlabel("epoch"); ax.set_ylabel(r"$\theta_{(2)} - \theta_{(3)}$")
        ax.set_title("Selection margin: gap between rank-2 and rank-3 theta")
        ax.legend(frameon=False, fontsize=8.5)
        p = f"{FIG}/05_theta_gap.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        made.append(p)
    return made


def report(runs, made):
    conv = [r for r in runs if r["grp"] == "converged"]
    L = []; w = L.append
    w("# SimpleRouter (Option D) Discovery -- mu-marginal slice search\n")
    n_run = sum(r["status"] == "running" for r in runs)
    w(f"> **{len(conv)} converged{', ' + str(n_run) + ' running' if n_run else ''}.** "
      "Auto-regenerated from `runs/simplerouter_discovery/seed_*/`. Re-run "
      "`scripts/distillation/make_simplerouter_study.py` to update.\n")
    w("SimpleRouter (Option D) picks exactly 2 of 101 time slices via an exact-marginal "
      "categorical-over-pairs distribution (no annealer on the router itself -- SimpleRouterLayer "
      "samples one pair per training step and back-props the exact pairwise-inclusion covariance). "
      "SoftQuantizeLayer still learns the matching thresholds on its own cosine anneal, downstream "
      "of the router, on the FULL all-101 dataset (noise baked, discovery split). Baseline under "
      f"challenge: slices **{BASELINE}**.\n")
    w("---\n\n## Results per seed\n")
    w("| Seed | Status | Epochs | Indices [i1,i2] | Thresholds [T0,T1,T2] | Best NLL | @ep |")
    w("|---:|---|---:|---|---|---:|---:|")
    for r in runs:
        idx = sorted([i for i in r["indices"] if i is not None]) or r["indices"]
        thr = ("[%.2f, %.2f, %.2f]" % tuple(r["fthr"])) if r["fthr"] else "n/a"
        best = r["best_val_loss"] if np.isfinite(r["best_val_loss"]) else float("nan")
        w(f"| {r['seed']} | {r['status']} | {r['epochs']} | **{idx}** | {thr} "
          f"| {best:,.0f} | {r['best_epoch']} |")
    w("")
    if conv:
        pairs = [tuple(sorted(i for i in r["indices"] if i is not None)) for r in conv
                 if len([i for i in r["indices"] if i is not None]) == 2]
        if pairs:
            vals, counts = np.unique(pairs, axis=0, return_counts=True)
            modal = vals[np.argmax(counts)]; agree = int(counts.max())
            w("## Consensus\n")
            w(f"- **Modal pair: {list(modal)}** -- {agree}/{len(pairs)} converged seeds agree"
              + (" (unanimous)" if agree == len(pairs) else "") + ".")
            w(f"- Per-slot median indices: [{int(np.median([p[0] for p in pairs]))}, "
              f"{int(np.median([p[1] for p in pairs]))}].")
            thrs = [r["fthr"] for r in conv if r["fthr"]]
            if thrs:
                T = np.array(thrs); med = T.mean(axis=0) if len(thrs) == 1 else np.median(T, axis=0)
                std = T.std(axis=0)
                w(f"- Median jointly-learned thresholds: [{med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f}] "
                  f"(std [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]).")
            w(f"- Baseline comparison: discovered {list(modal)} vs hand-picked {BASELINE} -- settled by "
              "the validation retrain, not by this run.\n")
    # A vs D comparison, sourced live from Option A's result.json files
    a_runs = load_option_a()
    w("---\n\n## Option A (SoftRouter) vs Option D (SimpleRouter)\n")
    if not a_runs and not conv:
        w("*Neither Option A nor Option D has a converged seed yet -- table will populate as seeds land.*\n")
    else:
        w("| Option | Seed | Indices | Best NLL | @ep |")
        w("|---|---:|---|---:|---:|")
        for r in a_runs:
            if not r["escaped"]:
                continue
            idx = sorted(r["indices"]) if r["indices"] else r["indices"]
            best = r["best"] if r["best"] is not None else float("nan")
            w(f"| A (soft) | {r['seed']} | {idx} | {best:,.0f} | {r['bestep']} |")
        for r in conv:
            idx = sorted([i for i in r["indices"] if i is not None]) or r["indices"]
            w(f"| D (simple) | {r['seed']} | {idx} | {r['best_val_loss']:,.0f} | {r['best_epoch']} |")
        w("")
        a_conv = [r for r in a_runs if r["escaped"]]
        if a_conv and conv:
            a_best = min(r["best"] for r in a_conv if r["best"] is not None)
            d_best = min(r["best_val_loss"] for r in conv if np.isfinite(r["best_val_loss"]))
            w(f"- Best-of-run NLL: Option A {a_best:,.0f} vs Option D {d_best:,.0f} "
              f"({'D' if d_best < a_best else 'A'} lower so far -- context only, not yet a "
              "controlled comparison since seeds/epochs may differ).\n")
        elif a_conv and not conv:
            w("- Option A has converged seeds; Option D does not yet -- comparison pending.\n")
        elif conv and not a_conv:
            w("- Option D has converged seeds; Option A does not yet -- comparison pending.\n")
    w("---\n\n## Figures\n")
    caps = {
        "01_nll_vs_epoch.png": "val NLL per seed (15-epoch moving average).",
        "02_top2_vs_epoch.png": "Committed [i1,i2] per epoch, all seeds; dotted = the [11,26] baseline.",
        "03_mu_profile_final.png": "THE importance map: final exact marginal inclusion probability "
                                    "per slice, seeds overlaid.",
        "04_visits_heatmap.png": "Pair-sampling visit counts per slice vs epoch, best-NLL seed.",
        "05_theta_gap.png": "Selection margin: theta_(2) - theta_(3) (rank-2 minus rank-3 of the "
                             "top-5 logits) vs epoch -- widening gap means the top-2 is settling.",
    }
    for p in made:
        fn = os.path.basename(p)
        w(f"### {fn}\n\n![{fn}](figs/{fn})\n\n*{caps.get(fn, '')}*\n\n`{p}`\n")
    w("---\n\n*Next step: validation retrain of the modal Option D pair on the standard full "
      "2-slice pipeline, compared against both the [11,26] baseline and the Option A modal pair.*\n")
    open(f"{OUT}/REPORT.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT}/REPORT.md  ({len(conv)} converged, {len(runs)} total)")


if __name__ == "__main__":
    runs = load()
    if not runs:
        print("no seed dirs with data yet under", OUT)
    else:
        made = figs(runs)
        report(runs, made)
        for p in made:
            print(" fig:", p)
