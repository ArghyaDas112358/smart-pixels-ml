"""Figures + REPORT.md for the SoftRouter JOINT slice+threshold discovery.
Auto-discovers every runs/softrouter_discovery/seed_*/ dir (converged, stuck,
running) and grows as seeds land. Mirrors make_2_5_study.py; dataviz-method
styling (validated palette, single-hue sequential ramp, no dual axes).
Re-run any time:  python scripts/distillation/make_softrouter_study.py
"""
import os, re, json, glob, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = "/work/users/das214/SmartPixels/smart-pixels-ml/runs/softrouter_discovery"
FIG = f"{OUT}/figs"; os.makedirs(FIG, exist_ok=True)
ESCAPE = 5e4
BASELINE = [11, 26]                         # the hand-picked pair being challenged
THR_25 = [12.14, 22.83, 52.74]              # 2_5 study medians (different slices — context only)

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
        f = lambda key: np.array([float(r[key]) for r in rows if r[key] != ""])
        m = re.match(r"seed_(\d+)", base)
        r = dict(seed=int(m.group(1)), dir=base,
                 ep=f("epoch"), loss=f("loss"), val=f("val_loss"),
                 i1=f("i1").astype(int), i2=f("i2").astype(int),
                 w1=f("w1_max"), w2=f("w2_max"),
                 T=np.stack([f("T0"), f("T1"), f("T2")], axis=1),
                 kk=f("k_router"))
        bi = int(np.argmin(r["val"]))
        r.update(best=float(r["val"][bi]), bestep=int(r["ep"][bi]), curep=int(r["ep"][-1]) + 1)
        rj = f"{d}/result.json"
        r["status"] = "killed" if "_STUCK" in base else "running"
        if os.path.exists(rj):
            j = json.load(open(rj)); r["status"] = "done"
            r["indices"] = j.get("final_indices", [int(r["i1"][-1]), int(r["i2"][-1])])
            r["fthr"] = j.get("final_thresholds", list(r["T"][-1]))
            r["epochs"] = j.get("epochs", r["curep"]); r["escaped"] = j.get("escaped")
        else:
            r["indices"] = [int(r["i1"][-1]), int(r["i2"][-1])]
            r["fthr"] = list(r["T"][-1]); r["epochs"] = r["curep"]; r["escaped"] = None
        npz = f"{d}/slot_weights.npz"
        if os.path.exists(npz):
            z = np.load(npz); r["snap_ep"] = z["epochs"]; r["snap_w"] = z["weights"]
        r["grp"] = ("stuck" if ("_STUCK" in base or r["best"] >= ESCAPE)
                    else ("converged" if r["status"] == "done" else "running"))
        runs.append(r)
    runs.sort(key=lambda r: (r["grp"] != "converged", r["seed"]))
    return runs


def figs(runs):
    made = []
    conv = [r for r in runs if r["grp"] == "converged"] or [r for r in runs if r["grp"] == "running"]
    # 01 — REAL slot-convergence heatmap (best available seed)
    withsnap = [r for r in conv if "snap_w" in r]
    if withsnap:
        r = min(withsnap, key=lambda r: r["best"])
        A = r["snap_w"].sum(axis=1)                       # (n_snap, 101): a1+a2
        fig, ax = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
        im = ax.imshow(A, aspect="auto", cmap=SEQ, origin="upper",
                       extent=[0, 100, r["snap_ep"][-1], r["snap_ep"][0]], vmin=0)
        ax.grid(False)
        for i in r["indices"]:
            ax.annotate(f"slice {i}", (i, r["snap_ep"][-1] * 0.9), ha="center", color=INK,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85))
        ax.set_xlabel("time-slice index"); ax.set_ylabel("epoch  (training ↓)")
        ax.set_title(f"REAL discovery: slot attention vs epoch (seed {r['seed']})", loc="left")
        cb = fig.colorbar(im, ax=ax, pad=0.012); cb.outline.set_visible(False)
        cb.set_label(r"$a_1[t]+a_2[t]$", color=INK2)
        p = f"{FIG}/01_slot_convergence_real.png"; fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    # 02 — selected indices vs epoch, all seeds
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
    ax.set_title("Committed indices vs epoch (solid = slot 1, dashed = slot 2)")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    p = f"{FIG}/02_indices_vs_epoch.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    # 03 — val NLL vs epoch
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for n, r in enumerate(runs):
        sm = np.convolve(r["val"], np.ones(15) / 15, mode="valid") if len(r["val"]) > 15 else r["val"]
        e = r["ep"][len(r["val"]) - len(sm):]
        ax.plot(e, sm, color=CAT[n % len(CAT)], lw=1.6,
                label=f"seed {r['seed']} (best {r['best']:,.0f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("val NLL/batch (15-ep avg)")
    ax.set_title("Joint discovery — val NLL vs epoch"); ax.legend(frameon=False, fontsize=8.5)
    p = f"{FIG}/03_nll_vs_epoch.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    # 04 — thresholds vs epoch
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True)
    for j, nm in enumerate(["T0", "T1", "T2"]):
        for n, r in enumerate(runs):
            axs[j].plot(r["ep"], r["T"][:, j], color=CAT[n % len(CAT)], lw=1.4)
        axs[j].axhline(THR_25[j], color=INK2, ls=":", lw=1)
        axs[j].set_title(nm); axs[j].set_xlabel("epoch")
    axs[0].set_ylabel("threshold (mV)")
    fig.suptitle("Jointly-learned thresholds (dotted = 2_5 [11,26] medians, context only)",
                 fontweight="bold")
    p = f"{FIG}/04_thresholds_vs_epoch.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    # 05 — commitment (max slot weight) vs epoch
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for n, r in enumerate(runs):
        c = CAT[n % len(CAT)]
        ax.plot(r["ep"], r["w1"], color=c, lw=1.4, label=f"seed {r['seed']}")
        ax.plot(r["ep"], r["w2"], color=c, lw=1.4, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("max slot weight")
    ax.set_ylim(0, 1.05)
    ax.set_title("Commitment: slot softmax peak vs epoch (1.0 = one-hot; solid=slot1, dashed=slot2)")
    ax.legend(frameon=False, fontsize=8.5)
    p = f"{FIG}/05_commitment_vs_epoch.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig); made.append(p)
    return made


def report(runs, made):
    conv = [r for r in runs if r["grp"] == "converged"]
    L = []; w = L.append
    w("# SoftRouter JOINT Discovery — slices + thresholds\n")
    n_run = sum(r["status"] == "running" for r in runs)
    w(f"> **{len(conv)} converged{', ' + str(n_run) + ' running' if n_run else ''}.** "
      "Auto-regenerated from `runs/softrouter_discovery/seed_*/`. Re-run "
      "`scripts/distillation/make_softrouter_study.py` to update.\n")
    w("Joint optimization on the FULL all-101 dataset (80+20 files): SoftRouterLayer picks 2 of "
      "101 slices while SoftQuantizeLayer learns the matching thresholds, both on the same cosine "
      "anneal, thresholds from random [25,160] init. Baseline under challenge: slices "
      f"**{BASELINE}** (thresholds [12.14, 22.83, 52.74]).\n")
    w("---\n\n## Results per seed\n")
    w("| Seed | Status | Epochs | Indices [i1,i2] | Thresholds [T0,T1,T2] | Commitment | Best NLL | @ep |")
    w("|---:|---|---:|---|---|---|---:|---:|")
    for r in runs:
        com = f"{r['w1'][-1]:.2f}/{r['w2'][-1]:.2f}"
        thr = "[%.2f, %.2f, %.2f]" % tuple(r["fthr"])
        w(f"| {r['seed']} | {r['status']} | {r['epochs']} | **{sorted(r['indices'])}** | {thr} "
          f"| {com} | {r['best']:,.0f} | {r['bestep']} |")
    w("")
    if conv:
        pairs = [tuple(sorted(r["indices"])) for r in conv]
        vals, counts = np.unique(pairs, axis=0, return_counts=True)
        modal = vals[np.argmax(counts)]; agree = int(counts.max())
        T = np.array([r["fthr"] for r in conv]); med = np.median(T, axis=0)
        w("## Consensus\n")
        w(f"- **Modal pair: {list(modal)}** — {agree}/{len(conv)} seeds agree"
          + (" (unanimous)" if agree == len(conv) else "") + ".")
        w(f"- Per-slot median indices: [{int(np.median([p[0] for p in pairs]))}, "
          f"{int(np.median([p[1] for p in pairs]))}].")
        w(f"- Median jointly-learned thresholds: [{med[0]:.2f}, {med[1]:.2f}, {med[2]:.2f}] "
          f"(std [{T.std(0)[0]:.2f}, {T.std(0)[1]:.2f}, {T.std(0)[2]:.2f}]).")
        w(f"- Baseline comparison: discovered {list(modal)} vs hand-picked {BASELINE} — settled by "
          "the validation retrain, not by this run.\n")
    w("---\n\n## Figures\n")
    caps = {"01_slot_convergence_real.png": "The REAL version of the plan's illustrative heatmap: total slot attention per slice vs epoch.",
            "02_indices_vs_epoch.png": "Committed [i1,i2] per epoch, all seeds; dotted = the [11,26] baseline.",
            "03_nll_vs_epoch.png": "val NLL per seed (joint model: honest hard-2-slice loss at every epoch).",
            "04_thresholds_vs_epoch.png": "Thresholds co-adapting with slice choice; dotted = 2_5 medians (different slices — context only).",
            "05_commitment_vs_epoch.png": "Slot softmax peak — the anneal's soft→hard commitment curve."}
    for p in made:
        fn = os.path.basename(p)
        w(f"### {fn}\n\n![{fn}](figs/{fn})\n\n*{caps.get(fn, '')}*\n\n`{p}`\n")
    w("---\n\n*Next step per docs/soft_router_plan.md §4: validation retrain of the modal pair on "
      "the standard full 2-slice pipeline vs the [11,26] baseline.*\n")
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
