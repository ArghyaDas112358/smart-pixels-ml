import json, glob, os, csv, statistics as st
B = "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs"
rows = {}
for rj in sorted(glob.glob(os.path.join(B, "fixedslice_p*/seed_*/result.json"))):
    pair = os.path.basename(os.path.dirname(os.path.dirname(rj))).replace("fixedslice_p", "").replace("_", ",")
    seed = os.path.basename(os.path.dirname(rj)).replace("seed_", "")
    d = json.load(open(rj))
    m = list(csv.DictReader(open(os.path.join(os.path.dirname(rj), "mdmm_epochs.csv"))))[-1]
    hc = "nll_cotA_c" in m
    g = lambda k: float(m[k + "_c"]) if hc else float(m[k])
    rows.setdefault(pair, []).append((seed, d["epochs"], d["best_val_loss"], g("nll_cotA"), g("nll_cotB")))
hdr = ("pair", "seed", "ep", "best val", "cotA", "cotB")
print("%8s %7s %6s %11s %9s %9s" % hdr)
for p in sorted(rows):
    for s, ep, bv, a, b in sorted(rows[p]):
        print("%8s %7s %6d %11s %+9.3f %+9.3f" % (p, s, ep, format(bv, ",.0f"), a, b))
    A = [r[3] for r in rows[p]]; Bb = [r[4] for r in rows[p]]
    sd = "  (sd %.3f)" % st.stdev(A) if len(A) > 1 else ""
    print("%8s %7s %6s %11s %+9.3f %+9.3f%s" % (p, "MEAN", "", "", st.mean(A), st.mean(Bb), sd))
    print()
