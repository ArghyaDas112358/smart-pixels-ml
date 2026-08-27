import json, glob, os, csv
BASES = [("SHALLOW", "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs/fixedslice_p11_26"),
         ("DEEP",    "/depot/cms/users/das214/SmartPixels_gautschi/smart-pixels-ml/runs/fixedslice_deep_p11_26")]
for tag, B in BASES:
    if not os.path.isdir(B): continue
    print("%s head  (%s)" % (tag, os.path.basename(B)))
    for d in sorted(glob.glob(os.path.join(B, "seed_*"))):
        s = os.path.basename(d).replace("seed_", "")
        rj = os.path.join(d, "result.json")
        if os.path.exists(rj):
            j = json.load(open(rj))
            m = list(csv.DictReader(open(os.path.join(d, "mdmm_epochs.csv"))))[-1]
            hc = "nll_cotA_c" in m
            g = lambda k: float(m[k + "_c"]) if hc else float(m[k])
            print("  %s: DONE ep %d  bestNLL %s  cotA %+.3f  escaped=%s aborted=%s"
                  % (s, j["epochs"], format(j["best_val_loss"], ",.0f"), g("nll_cotA"),
                     j.get("escaped"), j.get("aborted_stuck")))
        else:
            h = [r for r in csv.DictReader(open(os.path.join(d, "history.csv")))
                 if r.get("val_loss") not in ("", "nan", None)]
            print("  %s: ep %d/2000 running" % (s, len(h)))
    print()
