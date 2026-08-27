import json, glob, os, csv
out = {}
for tag, B in (("shallow", "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs/fixedslice_p11_26"),
               ("deep",    "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs/fixedslice_deep_p11_26")):
    rows = []
    for d in sorted(glob.glob(os.path.join(B, "seed_*"))):
        s = int(os.path.basename(d).replace("seed_", "").replace("_STUCK", ""))
        if d.endswith("_STUCK"):
            rows.append(dict(seed=s, state="aborted")); continue
        rj = os.path.join(d, "result.json")
        m = list(csv.DictReader(open(os.path.join(d, "mdmm_epochs.csv"))))[-1]
        hc = "nll_cotA_c" in m
        g = lambda k: float(m[k + "_c"]) if hc else float(m[k])
        if os.path.exists(rj):
            j = json.load(open(rj))
            rows.append(dict(seed=s, state="done", nll=round(j["best_val_loss"]), cotA=round(g("nll_cotA"), 3)))
        else:
            h = [r for r in csv.DictReader(open(os.path.join(d, "history.csv"))) if r.get("val_loss") not in ("", "nan", None)]
            rows.append(dict(seed=s, state="ep %d" % len(h), nll=round(min(float(x["val_loss"]) for x in h)), cotA=round(g("nll_cotA"), 3)))
    out[tag] = rows
print(json.dumps(out))
